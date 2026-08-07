from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import torch


GAZE_WAM_TRANSFER_FORMAT = "gaze_wam_transfer_v1"
GAZE_WAM_TRANSFER_SCOPES = {
    "obs_encoder": ("obs_encoder",),
    "obs_and_gaze": ("obs_encoder", "gaze_encoder"),
}


def normalize_gaze_wam_transfer_scope(name: str, value) -> str:
    scope = str(value or "").strip().lower()
    if scope not in GAZE_WAM_TRANSFER_SCOPES:
        choices = ", ".join(sorted(GAZE_WAM_TRANSFER_SCOPES))
        raise ValueError(f"{name} must be one of: {choices}; got {value!r}.")
    return scope


def _scope_components(policy, scope: str) -> Sequence[Tuple[str, torch.nn.Module]]:
    scope = normalize_gaze_wam_transfer_scope("transfer scope", scope)
    components = []
    for component_name in GAZE_WAM_TRANSFER_SCOPES[scope]:
        component = getattr(policy, component_name, None)
        if not isinstance(component, torch.nn.Module):
            raise TypeError(
                f"Policy transfer component {component_name!r} must be a torch.nn.Module, "
                f"got {type(component).__name__}."
            )
        components.append((component_name, component))
    return tuple(components)


def _transfer_state_dict(policy, scope: str) -> Dict[str, torch.Tensor]:
    state = {}
    for component_name, component in _scope_components(policy, scope):
        for key, value in component.state_dict().items():
            if not torch.is_tensor(value):
                raise TypeError(
                    f"Transfer state {component_name}.{key} must be a tensor, "
                    f"got {type(value).__name__}."
                )
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(
                    f"Transfer state {component_name}.{key} contains non-finite values."
                )
            state[f"{component_name}.{key}"] = value.detach().to("cpu").clone()
    if not state:
        raise ValueError(f"Transfer scope {scope!r} produced an empty state dict.")
    return state


def _state_summary(state: Mapping[str, torch.Tensor]) -> Dict[str, int]:
    return {
        "tensor_count": len(state),
        "parameter_count": int(sum(int(value.numel()) for value in state.values())),
    }


def export_gaze_wam_transfer_artifact(
    policy,
    path,
    *,
    scope: str = "obs_encoder",
    metadata: Mapping[str, object] = None,
    overwrite: bool = False,
) -> Dict[str, object]:
    scope = normalize_gaze_wam_transfer_scope("transfer export scope", scope)
    artifact_path = Path(path).expanduser()
    if artifact_path.exists() and not bool(overwrite):
        raise FileExistsError(
            f"Transfer artifact already exists: {artifact_path}. "
            "Set training.transfer.export_overwrite=true only when replacement is intended."
        )
    if artifact_path.exists() and not artifact_path.is_file():
        raise ValueError(f"Transfer artifact path must be a regular file: {artifact_path}")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    state = _transfer_state_dict(policy, scope)
    payload = {
        "format": GAZE_WAM_TRANSFER_FORMAT,
        "scope": scope,
        "state_dict": state,
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, artifact_path)
    return {
        "path": str(artifact_path),
        "scope": scope,
        "format": GAZE_WAM_TRANSFER_FORMAT,
        **_state_summary(state),
    }


def load_gaze_wam_transfer_artifact(
    policy,
    path,
    *,
    scope: str = "obs_encoder",
) -> Dict[str, object]:
    scope = normalize_gaze_wam_transfer_scope("transfer load scope", scope)
    artifact_path = Path(path).expanduser()
    if not artifact_path.exists():
        raise FileNotFoundError(f"Transfer artifact does not exist: {artifact_path}")
    if not artifact_path.is_file():
        raise ValueError(f"Transfer artifact must be a regular file: {artifact_path}")
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Transfer artifact root must be a mapping.")
    if payload.get("format") != GAZE_WAM_TRANSFER_FORMAT:
        raise ValueError(
            "Unsupported transfer artifact format: "
            f"{payload.get('format')!r}; expected {GAZE_WAM_TRANSFER_FORMAT!r}."
        )
    artifact_scope = normalize_gaze_wam_transfer_scope(
        "transfer artifact scope",
        payload.get("scope"),
    )
    if artifact_scope != scope:
        raise ValueError(
            f"Transfer artifact scope {artifact_scope!r} does not match requested "
            f"scope {scope!r}."
        )
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Transfer artifact state_dict must be a non-empty mapping.")
    expected_state = _transfer_state_dict(policy, scope)
    artifact_keys = set(state)
    expected_keys = set(expected_state)
    missing = sorted(expected_keys - artifact_keys)
    unexpected = sorted(artifact_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "Transfer artifact keys do not match the target policy; "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}."
        )
    for key, value in state.items():
        if not torch.is_tensor(value):
            raise TypeError(
                f"Transfer artifact value {key!r} must be a tensor, "
                f"got {type(value).__name__}."
            )
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Transfer artifact value {key!r} contains non-finite values.")
        expected = expected_state[key]
        if tuple(value.shape) != tuple(expected.shape):
            raise ValueError(
                f"Transfer artifact value {key!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(expected.shape)}."
            )
    for component_name, component in _scope_components(policy, scope):
        prefix = f"{component_name}."
        component_state = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        component.load_state_dict(component_state, strict=True)
    metadata = payload.get("metadata")
    return {
        "path": str(artifact_path),
        "scope": scope,
        "format": GAZE_WAM_TRANSFER_FORMAT,
        "metadata": dict(metadata) if isinstance(metadata, dict) else {},
        **_state_summary(state),
    }
