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
