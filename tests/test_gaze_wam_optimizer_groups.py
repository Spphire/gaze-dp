from types import SimpleNamespace

import torch

from diffusion_policy.policy.gaze_wam_policy import GazeWamPolicy
from diffusion_policy.workspace.train_gaze_wam_workspace import (
    _deepspeed_state_checkpoint_path,
    _retain_deepspeed_state_checkpoints,
    _workspace_checkpoint_exclude_keys,
)


def test_gaze_wam_optimizer_omits_empty_parameter_groups_for_deepspeed():
    trainable = torch.nn.Parameter(torch.ones(2))
    policy = object.__new__(GazeWamPolicy)
    torch.nn.Module.__init__(policy)
    policy.model = SimpleNamespace(
        get_optim_groups=lambda weight_decay: [
            {"params": [trainable], "weight_decay": weight_decay},
            {"params": [], "weight_decay": 0.0},
        ]
    )
    policy.gaze_encoder = torch.nn.Linear(2, 2)
    policy.gaze_encoder.requires_grad_(False)
    policy.heatmap_image_decoder = None
    policy.obs_encoder = torch.nn.Linear(2, 2)

    optimizer = GazeWamPolicy.get_optimizer(
        policy,
        lr=1e-3,
        weight_decay=1e-3,
        obs_encoder_lr=1e-3,
        obs_encoder_weight_decay=1e-3,
        betas=(0.9, 0.95),
    )

    assert optimizer.param_groups
    assert all(group["params"] for group in optimizer.param_groups)
    assert all(
        param.requires_grad
        for group in optimizer.param_groups
        for param in group["params"]
    )


def test_deepspeed_workspace_checkpoint_excludes_partitioned_optimizer_state():
    workspace = SimpleNamespace(exclude_keys=("transient",))

    assert _workspace_checkpoint_exclude_keys(workspace, is_deepspeed=True) == (
        "transient",
        "optimizer",
    )
    assert _workspace_checkpoint_exclude_keys(workspace, is_deepspeed=False) == (
        "transient",
    )


def test_deepspeed_state_checkpoint_path_and_retention(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    paths = [
        _deepspeed_state_checkpoint_path(tmp_path, step)
        for step in (2, 4, 6)
    ]
    for path in paths:
        path.mkdir(parents=True)

    _retain_deepspeed_state_checkpoints(checkpoint_dir, keep_last_n=2)

    assert not paths[0].exists()
    assert paths[1].is_dir()
    assert paths[2].is_dir()
