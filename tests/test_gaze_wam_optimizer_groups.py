from types import SimpleNamespace

import torch

from diffusion_policy.policy.gaze_wam_policy import GazeWamPolicy


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
