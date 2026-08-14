from __future__ import annotations

import math

import pytest
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.checkpoint_security import require_trusted_pickle_artifact
from diffusion_policy.common.omegaconf_resolvers import (
    register_safe_omegaconf_resolvers,
    safe_arithmetic_eval,
)
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.workspace.base_workspace import BaseWorkspace


class _CheckpointWorkspace(BaseWorkspace):
    include_keys = ("value",)

    def __init__(self, cfg, output_dir=None, value=0):
        super().__init__(cfg=cfg, output_dir=output_dir)
        self.value = value


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 * 256", 512),
        ("(12 + 6) // 5", 3),
        ("-3 + 8 / 2", 1.0),
        ("17 % 5", 2),
    ],
)
def test_safe_arithmetic_eval_accepts_numeric_arithmetic(expression, expected):
    result = safe_arithmetic_eval(expression)
    assert result == expected
    assert math.isfinite(float(result))


@pytest.mark.parametrize(
    "expression",
    [
        '__import__("os").system("echo unsafe")',
        "os.environ",
        "value",
        "[1, 2, 3]",
        "True",
        "2 ** 8",
        "1e309",
        "1000000000000 * 2",
    ],
)
def test_safe_arithmetic_eval_rejects_executable_or_unbounded_syntax(expression):
    with pytest.raises((TypeError, ValueError)):
        safe_arithmetic_eval(expression)


def test_safe_eval_resolver_preserves_numeric_config_compatibility():
    register_safe_omegaconf_resolvers()
    cfg = OmegaConf.create(
        {
            "n_obs_steps": 2,
            "max_image_tokens": "${eval:'${n_obs_steps} * 256'}",
        }
    )
    OmegaConf.resolve(cfg)
    assert cfg.max_image_tokens == 512


def test_pickle_artifact_requires_explicit_trust(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"not loaded")

    with pytest.raises(PermissionError, match="can execute arbitrary code"):
        require_trusted_pickle_artifact(
            checkpoint,
            trusted=False,
            artifact_name="test checkpoint",
        )

    assert require_trusted_pickle_artifact(
        checkpoint,
        trusted=True,
        artifact_name="test checkpoint",
    ) == checkpoint


def test_workspace_rejects_untrusted_checkpoint_before_torch_load(tmp_path, monkeypatch):
    checkpoint = tmp_path / "workspace.ckpt"
    checkpoint.write_bytes(b"not a pickle")

    def fail_if_loaded(*args, **kwargs):
        pytest.fail("torch.load must not run before the trust boundary")

    monkeypatch.setattr(torch, "load", fail_if_loaded)
    with pytest.raises(PermissionError, match="trust_checkpoint=True"):
        BaseWorkspace.create_from_checkpoint(checkpoint)


def test_eval_loader_rejects_untrusted_checkpoint_before_torch_load(tmp_path, monkeypatch):
    from diffusion_policy.scripts.eval_gaze_wam_metrics import load_policy_for_eval

    checkpoint = tmp_path / "eval.ckpt"
    checkpoint.write_bytes(b"not a pickle")

    def fail_if_loaded(*args, **kwargs):
        pytest.fail("torch.load must not run before the trust boundary")

    monkeypatch.setattr(torch, "load", fail_if_loaded)
    with pytest.raises(PermissionError, match="--trust-checkpoint"):
        load_policy_for_eval(checkpoint=str(checkpoint), device="cpu")


def test_normalizer_state_roundtrip_uses_weights_only_loading(tmp_path):
    normalizer = LinearNormalizer()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    state_path = tmp_path / "normalizer_state.pt"
    torch.save(normalizer.state_dict(), state_path)

    restored = LinearNormalizer()
    restored.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))

    value = torch.tensor([[0.5]], dtype=torch.float32)
    assert torch.equal(restored.normalize({"action": value})["action"], value)


def test_workspace_keeps_only_the_requested_rolling_checkpoints(tmp_path):
    workspace = _CheckpointWorkspace(
        cfg=OmegaConf.create({"test": True}), output_dir=str(tmp_path)
    )

    for index in range(1, 8):
        workspace.value = index
        workspace.save_checkpoint(
            use_thread=False,
            retain_last_n=5,
            retained_tag=f"rolling-epoch={index:04d}-step={index:06d}",
        )

    checkpoint_dir = tmp_path / "checkpoints"
    rolling_names = sorted(path.name for path in checkpoint_dir.glob("rolling-*.ckpt"))
    assert rolling_names == [
        f"rolling-epoch={index:04d}-step={index:06d}.ckpt"
        for index in range(3, 8)
    ]
    restored = _CheckpointWorkspace.create_from_checkpoint(
        checkpoint_dir / "latest.ckpt", trust_checkpoint=True
    )
    assert restored.value == 7


def test_workspace_failed_atomic_save_preserves_previous_latest(tmp_path, monkeypatch):
    workspace = _CheckpointWorkspace(
        cfg=OmegaConf.create({"test": True}), output_dir=str(tmp_path), value=1
    )
    workspace.save_checkpoint(use_thread=False)
    latest_path = tmp_path / "checkpoints" / "latest.ckpt"
    original_bytes = latest_path.read_bytes()

    def fail_after_partial_write(_payload, file_obj, **_kwargs):
        file_obj.write(b"incomplete")
        raise OSError("simulated checkpoint write failure")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    workspace.value = 2
    with pytest.raises(OSError, match="simulated checkpoint write failure"):
        workspace.save_checkpoint(use_thread=False)

    assert latest_path.read_bytes() == original_bytes
    restored = _CheckpointWorkspace.create_from_checkpoint(
        latest_path, trust_checkpoint=True
    )
    assert restored.value == 1
    assert not [path for path in latest_path.parent.iterdir() if path.suffix == ".tmp"]
