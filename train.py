"""
Usage:
Training:
python train.py --config-name=train_gaze_wam_workspace task=gaze_wam

Server 8-GPU AMP training should use:
bash train_scripts/start_open_only_8gpu_tmux.sh
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.common.omegaconf_resolvers import register_safe_omegaconf_resolvers
from diffusion_policy.workspace.base_workspace import BaseWorkspace

register_safe_omegaconf_resolvers()

ALLOWED_WORKSPACE_TARGETS = frozenset(
    {
        "diffusion_policy.workspace.train_gaze_wam_workspace.TrainGazeWamWorkspace",
    }
)


def _validate_workspace_target(cfg: OmegaConf) -> str:
    target = OmegaConf.select(cfg, "_target_")
    if target not in ALLOWED_WORKSPACE_TARGETS:
        allowed = ", ".join(sorted(ALLOWED_WORKSPACE_TARGETS))
        raise ValueError(
            "This repository entrypoint is restricted to Gaze-WAM training. "
            f"Got _target_={target!r}; allowed targets: {allowed}."
        )
    return str(target)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    target = _validate_workspace_target(cfg)
    cls = hydra.utils.get_class(target)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
