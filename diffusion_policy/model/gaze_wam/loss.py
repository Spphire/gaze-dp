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


def dsnt_expectation(prob: torch.Tensor) -> torch.Tensor:
    """Return DSNT expected ``(x, y)`` coordinates in normalized image space."""
    prob = _as_spatial_image(prob, "prob")
    batch_size, height, width = prob.shape
    y = (torch.arange(height, device=prob.device, dtype=prob.dtype) + 0.5) / height
    x = (torch.arange(width, device=prob.device, dtype=prob.dtype) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    expected_x = (prob * xx.reshape(1, height, width)).sum(dim=(-2, -1))
    expected_y = (prob * yy.reshape(1, height, width)).sum(dim=(-2, -1))
    return torch.stack([expected_x, expected_y], dim=-1).reshape(batch_size, 2)


def per_sample_dsnt_xy_loss(
    pred_heatmap_logits: torch.Tensor,
    target_xy: torch.Tensor,
    temperature: float = 1.0,
    distribution_mode: str = "logits_softmax",
) -> torch.Tensor:
    """Per-sample DSNT Euclidean coordinate loss from heatmap logits to gaze ``xy``."""
    if not torch.is_tensor(target_xy):
        raise TypeError(f"target_xy must be a torch.Tensor, got {type(target_xy).__name__}.")
    if not torch.is_floating_point(target_xy):
        raise ValueError(f"target_xy must be a floating point tensor, got {target_xy.dtype}.")
    if target_xy.ndim != 2 or target_xy.shape[-1] != 2:
        raise ValueError(f"target_xy must have shape [B, 2], got {tuple(target_xy.shape)}.")
    if not torch.all(torch.isfinite(target_xy)):
        raise ValueError("target_xy must contain only finite values.")
    if torch.any((target_xy < 0.0) | (target_xy > 1.0)):
        raise ValueError("target_xy must be normalized to [0, 1].")

    prob = spatial_distribution_2d(
        pred_heatmap_logits,
        mode=distribution_mode,
        temperature=temperature,
    )
    pred_xy = dsnt_expectation(prob).to(dtype=target_xy.dtype)
    return torch.linalg.vector_norm(pred_xy - target_xy, ord=2, dim=-1)


def per_sample_spatial_point_nll_loss(
    pred_heatmap_logits: torch.Tensor,
    target_xy: torch.Tensor,
    temperature: float = 1.0,
    distribution_mode: str = "logits_softmax",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-sample point negative log likelihood at normalized gaze ``xy``."""
    if not torch.is_tensor(target_xy):
        raise TypeError(f"target_xy must be a torch.Tensor, got {type(target_xy).__name__}.")
    if not torch.is_floating_point(target_xy):
        raise ValueError(f"target_xy must be a floating point tensor, got {target_xy.dtype}.")
    if target_xy.ndim != 2 or target_xy.shape[-1] != 2:
        raise ValueError(f"target_xy must have shape [B, 2], got {tuple(target_xy.shape)}.")
    if not torch.all(torch.isfinite(target_xy)):
        raise ValueError("target_xy must contain only finite values.")
    if torch.any((target_xy < 0.0) | (target_xy > 1.0)):
        raise ValueError("target_xy must be normalized to [0, 1].")

    prob = spatial_distribution_2d(
        pred_heatmap_logits,
        mode=distribution_mode,
        temperature=temperature,
        eps=eps,
    )
    batch_size, height, width = prob.shape
    x_idx = torch.floor(target_xy[:, 0].float() * float(width)).long().clamp(0, width - 1)
    y_idx = torch.floor(target_xy[:, 1].float() * float(height)).long().clamp(0, height - 1)
    batch_idx = torch.arange(batch_size, device=prob.device)
    point_prob = prob[batch_idx, y_idx.to(prob.device), x_idx.to(prob.device)]
    return -point_prob.clamp_min(float(eps)).log().to(dtype=target_xy.dtype)


def per_sample_spatial_js_loss(
    pred_heatmap_logits: torch.Tensor,
    target_heatmap: torch.Tensor,
    temperature: float = 1.0,
    distribution_mode: str = "logits_softmax",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-sample Jensen-Shannon loss between predicted and target heatmap distributions."""
    pred = spatial_distribution_2d(
        pred_heatmap_logits,
        mode=distribution_mode,
        temperature=temperature,
        eps=eps,
    )
    target = normalize_spatial_distribution(target_heatmap, eps=eps)
    if pred.shape != target.shape:
        raise ValueError(
            "pred_heatmap_logits and target_heatmap spatial shapes must match, got "
            f"{tuple(pred.shape)} vs {tuple(target.shape)}."
        )
    pred = pred.flatten(start_dim=1).clamp_min(float(eps))
    target = target.flatten(start_dim=1).clamp_min(float(eps))
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    mixture = 0.5 * (pred + target)
    pred_kl = pred * (pred.log() - mixture.clamp_min(float(eps)).log())
    target_kl = target * (target.log() - mixture.clamp_min(float(eps)).log())
    return 0.5 * (pred_kl.sum(dim=-1) + target_kl.sum(dim=-1))
