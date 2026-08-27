import torch
import torch.distributed as dist
import torch.nn.functional as F


def distributed_mask_count(mask: torch.Tensor) -> torch.Tensor:
    """Count mask entries with cross-rank reduction when DDP is active."""
    if not torch.is_tensor(mask):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask).__name__}.")
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be a BoolTensor, got dtype {mask.dtype}.")
    if mask.ndim != 1:
        raise ValueError(f"mask must have shape [B], got {tuple(mask.shape)}.")
    count = mask.to(dtype=torch.float32).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return count.detach()


def distributed_masked_mean(
    per_sample_loss: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Differentiable masked mean for the local rank.

    DDP already averages gradients across ranks. Reducing the loss tensor itself
    with ``dist.all_reduce`` would put a collective op in the autograd path.
    Use ``distributed_mask_count`` and explicit metric gathering for logging.
    """
    if not torch.is_tensor(mask):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask).__name__}.")
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be a BoolTensor, got dtype {mask.dtype}.")
    if per_sample_loss.shape[0] != mask.shape[0]:
        raise ValueError(
            f"Batch size mismatch: loss {per_sample_loss.shape}, mask {mask.shape}."
        )
    if mask.ndim != 1:
        raise ValueError(f"mask must have shape [B], got {tuple(mask.shape)}.")
    mask = mask.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)
    while mask.ndim < per_sample_loss.ndim:
        mask = mask.unsqueeze(-1)

    numerator = (per_sample_loss * mask).sum()
    denominator = mask.expand_as(per_sample_loss).sum()

    if denominator.detach().item() <= 0:
        return per_sample_loss.sum() * 0.0
    return numerator / denominator.clamp_min(eps)


def _as_spatial_image(image: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(image):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(image).__name__}.")
    if not torch.is_floating_point(image):
        raise ValueError(f"{name} must be a floating point tensor, got {image.dtype}.")
    if image.ndim == 4:
        if image.shape[1] != 1:
            raise ValueError(f"{name} must have one channel, got {tuple(image.shape)}.")
        image = image[:, 0]
    if image.ndim != 3:
        raise ValueError(f"{name} must have shape [B, H, W] or [B, 1, H, W], got {tuple(image.shape)}.")
    if not torch.all(torch.isfinite(image)):
        raise ValueError(f"{name} must contain only finite values.")
    return image


def spatial_softmax_2d(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Normalize per-pixel logits into a spatial probability distribution."""
    logits = _as_spatial_image(logits, "logits")
    temperature = float(temperature)
    if not temperature > 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    flat = logits.float().flatten(start_dim=1) / temperature
    prob = torch.softmax(flat, dim=-1)
    return prob.reshape_as(logits.float())


def spatial_distribution_2d(
    image: torch.Tensor,
    mode: str = "logits_softmax",
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert a dense heatmap prediction to a spatial probability distribution.

    ``logits_softmax`` is the standard DSNT interpretation for a direct heatmap
    head. ``intensity_*`` modes treat the input as a decoded heatmap image,
    which matches frozen image-codec outputs such as Cosmos tokenizer decoded
    latents. ``intensity_softplus`` is zero-calibrated so a decoded background
    value of 0 contributes no probability mass.
    """
    mode = str(mode)
    if mode == "logits_softmax":
        return spatial_softmax_2d(image, temperature=temperature)
    image = _as_spatial_image(image, "image").float()
    if mode == "intensity_clamp":
        return normalize_spatial_distribution(image, eps=eps)
    if mode == "intensity_softplus":
        temperature = float(temperature)
        if not temperature > 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        zero = torch.zeros((), device=image.device, dtype=image.dtype)
        intensity = F.softplus(image / temperature) - F.softplus(zero)
        return normalize_spatial_distribution(
            intensity,
            eps=eps,
        )
    raise ValueError(
        "mode must be one of: logits_softmax, intensity_clamp, intensity_softplus; "
        f"got {mode!r}."
    )


def normalize_spatial_distribution(
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Clamp and L1-normalize a dense heatmap label per sample."""
    target = _as_spatial_image(target, "target").float().clamp_min(0.0)
    flat = target.flatten(start_dim=1)
    flat = flat / flat.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    return flat.reshape_as(target)
