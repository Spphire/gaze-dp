import collections
import pathlib
from typing import Dict, Optional, Sequence, Tuple

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.checkpoint_security import require_trusted_pickle_artifact
from diffusion_policy.common.gaze_wam_image import image_to_chw_float
from diffusion_policy.common.omegaconf_resolvers import register_safe_omegaconf_resolvers
from diffusion_policy.policy.gaze_wam_policy import GazeWamPolicy
from diffusion_policy.real_world.gaze_wam_action_base import action_base_abs_to_10d
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pose_util import mat_to_pose10d, pose_to_mat

register_safe_omegaconf_resolvers()


def _require_finite_array(name: str, value: np.ndarray) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values.")


def _require_finite_scalar(name: str, value: Optional[float]) -> None:
    if value is None:
        return
    if not np.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}.")


def _validate_positive_int(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if int_value != value and not (isinstance(value, np.integer) and int_value == int(value)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if int_value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int_value


def _validate_image_size(name: str, value: Sequence[int]) -> Tuple[int, int]:
    try:
        values = list(value)
    except TypeError:
        raise ValueError(f"{name} must contain exactly two positive integers, got {value!r}.")
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two positive integers, got {value!r}.")
    return (
        _validate_positive_int(f"{name}[0]", values[0]),
        _validate_positive_int(f"{name}[1]", values[1]),
    )


def _validate_camera_shape(camera_key: str, obs_meta: dict) -> Tuple[int, int]:
    if "shape" not in obs_meta:
        raise ValueError(f"shape_meta.obs[{camera_key!r}] must define a shape.")
    shape = list(obs_meta["shape"])
    if len(shape) < 3:
        raise ValueError(
            f"shape_meta.obs[{camera_key!r}].shape must be at least [C,H,W], got {shape!r}."
        )
    channel_dim = _validate_positive_int(f"shape_meta.obs[{camera_key!r}].shape[-3]", shape[-3])
    if channel_dim not in (1, 3, 4):
        raise ValueError(
            f"shape_meta.obs[{camera_key!r}].shape channel dim must be 1, 3, or 4, got "
            f"{channel_dim}."
        )
    return _validate_image_size(f"shape_meta.obs[{camera_key!r}].shape[-2:]", shape[-2:])


def _image_to_chw_float(image: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    return image_to_chw_float(image, image_size=image_size, name="image")


def tcp_pose_to_action_base_abs(
    tcp_pose,
    gripper_width: Optional[float] = None,
) -> np.ndarray:
    """Convert deployment TCP pose input into the 10D Gaze-WAM action base convention."""
    tcp_pose = np.asarray(tcp_pose, dtype=np.float32)
    _require_finite_array("tcp_pose", tcp_pose)
    _require_finite_scalar("gripper_width", gripper_width)
    if tcp_pose.shape[-1] == 10:
        base = tcp_pose.copy()
    elif tcp_pose.shape[-1] == 9:
        base = np.concatenate([tcp_pose, np.zeros(tcp_pose.shape[:-1] + (1,), dtype=np.float32)], axis=-1)
    elif tcp_pose.shape[-1] == 6:
        base9 = mat_to_pose10d(pose_to_mat(tcp_pose))[..., :9].astype(np.float32)
        base = np.concatenate([base9, np.zeros(base9.shape[:-1] + (1,), dtype=np.float32)], axis=-1)
    else:
        raise ValueError(f"tcp_pose must be 6D pose, 9D pose10d, or 10D pose+gripper, got {tcp_pose.shape}.")
    if gripper_width is not None:
        base[..., 9] = float(gripper_width)
    _require_finite_array("action_base_abs", base)
    return base.astype(np.float32)


def load_gaze_wam_policy_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda:0",
    use_ema: bool = True,
    num_inference_steps: Optional[int] = None,
    trust_checkpoint: bool = False,
) -> Tuple[GazeWamPolicy, object]:
    ckpt_path = pathlib.Path(checkpoint_path)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "checkpoints" / "latest.ckpt"
    ckpt_path = require_trusted_pickle_artifact(
        ckpt_path,
        trusted=trust_checkpoint,
        artifact_name="Gaze-WAM checkpoint",
    )
    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location=device)
    cfg = payload["cfg"]
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = workspace_cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if use_ema and cfg.training.use_ema else workspace.model
    if num_inference_steps is not None:
        policy.num_inference_steps = policy._validate_positive_int(
            "num_inference_steps",
            num_inference_steps,
        )
    policy.eval().to(torch.device(device))
    return policy, cfg


class GazeWamInferenceAdapter:
    """Deployment-facing adapter for action-only Gaze-WAM inference.

    This class deliberately stops at model IO. Robot-specific scheduling, safety checks,
    controller command timing, and emergency-stop handling belong in the real runner.
    """

    def __init__(
        self,
        policy: GazeWamPolicy,
        shape_meta: dict,
        camera_key: str = "camera0_rgb",
        image_size: Optional[Sequence[int]] = None,
        n_obs_steps: Optional[int] = None,
        obs_downsample_steps: int = 1,
        device: Optional[str] = None,
        cfg_scale: float = 1.0,
    ) -> None:
        self.policy = policy
        self.shape_meta = shape_meta
        self.camera_key = camera_key
        try:
            obs_meta = shape_meta["obs"][camera_key]
        except KeyError:
            available = sorted(shape_meta.get("obs", {}).keys()) if isinstance(shape_meta, dict) else []
            raise ValueError(
                f"camera_key {camera_key!r} is missing from shape_meta['obs']; "
                f"available keys: {available}."
            )
        shape_image_size = _validate_camera_shape(camera_key, obs_meta)
        if image_size is None:
            image_size = shape_image_size
        self.image_size = _validate_image_size("image_size", image_size)
        if n_obs_steps is None:
            n_obs_steps = int(obs_meta.get("horizon", 1))
        self.n_obs_steps = _validate_positive_int("n_obs_steps", n_obs_steps)
        self.obs_downsample_steps = _validate_positive_int(
            "obs_downsample_steps",
            obs_downsample_steps,
        )
        self.cfg_scale = self.policy._validate_nonnegative_float("cfg_scale", cfg_scale)
        self.device = torch.device(device) if device is not None else policy.device
        self.image_history = collections.deque(maxlen=self.n_obs_steps)
        self.policy.eval().to(self.device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda:0",
        use_ema: bool = True,
        num_inference_steps: Optional[int] = None,
        camera_key: str = "camera0_rgb",
        cfg_scale: float = 1.0,
        trust_checkpoint: bool = False,
    ) -> "GazeWamInferenceAdapter":
        policy, cfg = load_gaze_wam_policy_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            use_ema=use_ema,
            num_inference_steps=num_inference_steps,
            trust_checkpoint=trust_checkpoint,
        )
        return cls(
            policy=policy,
            shape_meta=cfg.task.shape_meta,
            camera_key=camera_key,
            obs_downsample_steps=OmegaConf.select(
                cfg,
                "task.obs_downsample_steps",
                default=1,
            ),
            device=device,
            cfg_scale=cfg_scale,
        )

    def reset(self) -> None:
        self.image_history.clear()
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def push_image(self, image: np.ndarray) -> None:
        self.image_history.append(_image_to_chw_float(image, image_size=self.image_size))

    def set_image_history(self, images: Sequence[np.ndarray]) -> None:
        """Replace history with timestamp-selected frames from the runtime."""
        images = list(images)
        if not images:
            raise ValueError("image_history must contain at least one image.")
        processed = [
            _image_to_chw_float(image, image_size=self.image_size)
            for image in images[-self.n_obs_steps :]
        ]
        self.image_history.clear()
        self.image_history.extend(processed)

    def _stack_history(self) -> np.ndarray:
        if not self.image_history:
            raise RuntimeError("No image has been pushed into the Gaze-WAM inference adapter.")
        images = list(self.image_history)
        while len(images) < self.n_obs_steps:
            images.insert(0, images[0])
        return np.stack(images[-self.n_obs_steps :], axis=0)

    def build_obs(
        self,
        gaze_xy: Optional[Sequence[float]] = None,
        action_base_abs: Optional[Sequence[float]] = None,
        gripper_width: Optional[float] = None,
        use_gaze_condition: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        obs_np = {
            self.camera_key: self._stack_history()[None],
        }
        if gaze_xy is None:
            gaze = np.zeros((1, 2), dtype=np.float32)
            has_gaze_label = np.asarray([False], dtype=bool)
            if use_gaze_condition is None:
                use_gaze_condition = False
        else:
            gaze = np.asarray(gaze_xy, dtype=np.float32).reshape(1, 2)
            _require_finite_array("gaze_xy", gaze)
            gaze = np.clip(gaze, 0.0, 1.0)
            has_gaze_label = np.asarray([True], dtype=bool)
            if use_gaze_condition is None:
                use_gaze_condition = True
        obs_np["gaze_xy"] = gaze
        obs_np["has_gaze_label"] = has_gaze_label
        obs_np["use_gaze_condition"] = np.asarray([bool(use_gaze_condition)], dtype=bool)
        if action_base_abs is not None:
            obs_np["action_base_abs"] = action_base_abs_to_10d(
                action_base_abs,
                gripper_width=gripper_width,
            ).reshape(1, 10)
        return dict_apply(obs_np, lambda x: torch.from_numpy(x).to(self.device))

    @torch.no_grad()
    def predict_action(
        self,
        image: Optional[np.ndarray] = None,
        image_history: Optional[Sequence[np.ndarray]] = None,
        gaze_xy: Optional[Sequence[float]] = None,
        tcp_pose: Optional[Sequence[float]] = None,
        gripper_width: Optional[float] = None,
        action_base_abs: Optional[Sequence[float]] = None,
        cfg_scale: Optional[float] = None,
        use_gaze_condition: Optional[bool] = None,
    ) -> Dict[str, np.ndarray]:
        validated_cfg_scale = (
            self.cfg_scale
            if cfg_scale is None
            else self.policy._validate_nonnegative_float("cfg_scale", cfg_scale)
        )
        if image is not None and image_history is not None:
            raise ValueError("Provide either image or image_history, not both.")
        if image_history is not None:
            self.set_image_history(image_history)
        elif image is not None:
            self.push_image(image)
        if action_base_abs is None and tcp_pose is not None:
            action_base_abs = tcp_pose_to_action_base_abs(tcp_pose, gripper_width=gripper_width)
        obs = self.build_obs(
            gaze_xy=gaze_xy,
            action_base_abs=action_base_abs,
            gripper_width=gripper_width,
            use_gaze_condition=use_gaze_condition,
        )
        result = self.policy.predict_action(
            obs,
            cfg_scale=validated_cfg_scale,
        )
        return {
            key: value.detach().cpu().numpy()
            for key, value in result.items()
            if isinstance(value, torch.Tensor)
        }

    @torch.no_grad()
    def warmup(
        self,
        image: Optional[np.ndarray] = None,
        gaze_xy: Optional[Sequence[float]] = None,
        action_base_abs: Optional[Sequence[float]] = None,
    ) -> Dict[str, np.ndarray]:
        if image is None:
            image = np.zeros(self.image_size + (3,), dtype=np.uint8)
        return self.predict_action(
            image=image,
            gaze_xy=gaze_xy,
            action_base_abs=action_base_abs,
        )
