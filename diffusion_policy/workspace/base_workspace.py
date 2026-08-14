from typing import Optional
import os
import pathlib
import shutil
import tempfile
import uuid
import hydra
import copy
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import dill
import torch
import threading

from diffusion_policy.common.checkpoint_security import require_trusted_pickle_artifact


class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str]=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        self._saving_error = None

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    
    def run(self):
        """
        Create any resource shouldn't be serialized as local variables
        """
        pass

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=True,
            retain_last_n=0,
            retained_tag=None):
        self.wait_for_pending_checkpoint()
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        save_kwargs = {
            'retain_last_n': retain_last_n,
            'retained_tag': retained_tag,
        }
        if use_thread:
            def save_in_thread():
                try:
                    self._save_checkpoint_atomically(payload, path, **save_kwargs)
                except Exception as exc:
                    self._saving_error = exc

            self._saving_thread = threading.Thread(
                target=save_in_thread,
                name="checkpoint-writer",
            )
            self._saving_thread.start()
        else:
            self._save_checkpoint_atomically(payload, path, **save_kwargs)
        return str(path.absolute())

    def wait_for_pending_checkpoint(self):
        """Finish the preceding checkpoint writer and surface any write failure."""
        if self._saving_thread is not None:
            self._saving_thread.join()
            self._saving_thread = None
        if self._saving_error is not None:
            error = self._saving_error
            self._saving_error = None
            raise RuntimeError("Asynchronous checkpoint save failed.") from error

    @staticmethod
    def _save_checkpoint_atomically(
            payload,
            path,
            retain_last_n=0,
            retained_tag=None):
        """Write a checkpoint beside its target, then atomically publish it."""
        path = pathlib.Path(path)
        temp_fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(temp_fd)
        temp_path = pathlib.Path(temp_name)
        try:
            with temp_path.open('wb') as file_obj:
                torch.save(payload, file_obj, pickle_module=dill)
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(temp_path, path)
            if retained_tag is not None:
                BaseWorkspace._retain_checkpoint(
                    path=path,
                    retained_tag=retained_tag,
                    retain_last_n=retain_last_n,
                )
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def _retain_checkpoint(path, retained_tag, retain_last_n):
        path = pathlib.Path(path)
        retain_last_n = max(0, int(retain_last_n))
        if retain_last_n == 0:
            return
        if not retained_tag.startswith('rolling-'):
            raise ValueError("retained_tag must start with 'rolling-'.")

        retained_path = path.parent.joinpath(f'{retained_tag}.ckpt')
        temp_path = path.parent.joinpath(
            f".{retained_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            try:
                os.link(path, temp_path)
            except OSError:
                shutil.copy2(path, temp_path)
            os.replace(temp_path, retained_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

        rolling_paths = sorted(
            path.parent.glob('rolling-*.ckpt'),
            key=lambda candidate: (candidate.stat().st_mtime_ns, candidate.name),
            reverse=True,
        )
        for stale_path in rolling_paths[retain_last_n:]:
            stale_path.unlink()
    
    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            trust_checkpoint=False,
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        path = require_trusted_pickle_artifact(
            path,
            trusted=trust_checkpoint,
            artifact_name="workspace checkpoint",
        )
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            trust_checkpoint=False,
            **kwargs):
        path = require_trusted_pickle_artifact(
            path,
            trusted=trust_checkpoint,
            artifact_name="workspace checkpoint",
        )
        payload = torch.load(path.open('rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path, trust_checkpoint=False):
        path = require_trusted_pickle_artifact(
            path,
            trusted=trust_checkpoint,
            artifact_name="workspace snapshot",
        )
        return torch.load(path.open('rb'), pickle_module=dill)


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)
