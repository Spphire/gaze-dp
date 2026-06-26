import torch


def gaze_dependency_ratio(
    conditioned: torch.Tensor,
    masked: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute per-sample gaze dependency ratio between conditioned and masked outputs."""
    if conditioned.shape != masked.shape:
        raise ValueError(
            f"conditioned and masked tensors must have the same shape, got "
            f"{conditioned.shape} and {masked.shape}."
        )
    if conditioned.ndim < 2:
        raise ValueError("Expected at least [B, ...] tensors.")
    diff = conditioned - masked
    diff_norm = torch.linalg.vector_norm(diff.flatten(start_dim=1), dim=1)
    base_norm = torch.linalg.vector_norm(conditioned.flatten(start_dim=1), dim=1)
    return diff_norm / base_norm.clamp_min(eps)
