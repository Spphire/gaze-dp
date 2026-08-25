from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from diffusion_policy.common.gaze_wam_training_config import (
    normalize_gaze_wam_positive_float_field,
    normalize_gaze_wam_positive_int_sequence,
)


def _validate_positive_int_pair(name: str, value: Sequence[int]) -> Tuple[int, int]:
    try:
        parsed = normalize_gaze_wam_positive_int_sequence(name, value, length=2)
    except ValueError as exc:
        if "must contain 2 positive integers" in str(exc):
            raise ValueError(f"{name} must be a pair of (height, width).") from exc
        raise ValueError(f"{name} dimensions must be positive, got {value}.") from exc
    if len(parsed) != 2:
        raise ValueError(f"{name} must be a pair of (height, width).")
    return tuple(parsed)


def _validate_positive_float(name: str, value: float) -> float:
    try:
        return normalize_gaze_wam_positive_float_field(name, value)
    except ValueError as exc:
        raise ValueError(f"{name} must be positive.") from exc


def _require_bool_mask(name: str, value: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if value.dtype != torch.bool:
        raise ValueError(f"{name} must be a BoolTensor, got dtype {value.dtype}.")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(value.shape)}.")
    return value


class GaussianSpatialEncoder(nn.Module):
    """Encode normalized gaze coordinates with a fixed Gaussian spatial basis.

    Finite coordinates may lie outside the image. They are clamped only inside
    the encoder so very distant projections cannot underflow the entire basis.
    """

    EXTENDED_COORDINATE_RANGE = (-0.5, 1.5)

    def __init__(
        self,
        grid_size: Sequence[int] = (8, 8),
        sigma: float = 0.15,
        normalize: str = "max",
    ) -> None:
        super().__init__()
        grid_size = _validate_positive_int_pair("grid_size", grid_size)
        sigma = _validate_positive_float("sigma", sigma)
        if normalize not in ("none", "max", "sum"):
            raise ValueError("normalize must be one of: none, max, sum.")

        self.grid_size = grid_size
        self.sigma = sigma
        self.normalize = normalize

        height, width = self.grid_size
        y = (torch.arange(height, dtype=torch.float32) + 0.5) / height
        x = (torch.arange(width, dtype=torch.float32) + 0.5) / width
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        centers = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        self.register_buffer("centers", centers, persistent=False)

    @property
    def output_dim(self) -> int:
        return self.grid_size[0] * self.grid_size[1]

    def forward(
        self,
        gaze_xy: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(gaze_xy):
            raise TypeError(f"gaze_xy must be a torch.Tensor, got {type(gaze_xy).__name__}.")
        if not torch.is_floating_point(gaze_xy):
            raise ValueError(f"gaze_xy must be a floating point tensor, got {gaze_xy.dtype}.")
        if gaze_xy.shape[-1] != 2:
            raise ValueError(f"Expected gaze_xy last dim 2, got {gaze_xy.shape}.")
        if not torch.all(torch.isfinite(gaze_xy)):
            raise ValueError("gaze_xy must contain only finite values.")
        xy = gaze_xy.to(dtype=self.centers.dtype).clamp(
            min=self.EXTENDED_COORDINATE_RANGE[0],
            max=self.EXTENDED_COORDINATE_RANGE[1],
        )
        centers = self.centers.to(device=xy.device, dtype=xy.dtype)
        diff = xy.unsqueeze(-2) - centers
        out = torch.exp(-(diff.square().sum(dim=-1)) / (2 * self.sigma ** 2))

        if self.normalize == "max":
            out = out / out.amax(dim=-1, keepdim=True).clamp_min(1e-12)
        elif self.normalize == "sum":
            out = out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        if valid_mask is not None:
            valid_mask = _require_bool_mask(
                "valid_mask",
                valid_mask,
                gaze_xy.shape[:-1],
            ).to(device=out.device)
            out = out * valid_mask.to(dtype=out.dtype).unsqueeze(-1)
        return out


class GazeConditionEncoder(nn.Module):
    """Project gaze coordinates to one transformer token, with a trainable mask token."""

    def __init__(
        self,
        embed_dim: int,
        grid_size: Sequence[int] = (8, 8),
        sigma: float = 0.15,
        normalize: str = "max",
    ) -> None:
        super().__init__()
        self.spatial_encoder = GaussianSpatialEncoder(
            grid_size=grid_size,
            sigma=sigma,
            normalize=normalize,
        )
        self.proj = nn.Linear(self.spatial_encoder.output_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(
        self,
        gaze_xy: torch.Tensor,
        use_gaze_condition: Optional[torch.Tensor] = None,
        has_gaze_condition: Optional[torch.Tensor] = None,
        has_gaze_label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if has_gaze_label is not None:
            has_gaze_label = _require_bool_mask(
                "has_gaze_label",
                has_gaze_label,
                gaze_xy.shape[:-1],
            )
        if has_gaze_condition is None:
            has_gaze_condition = has_gaze_label
        else:
            has_gaze_condition = _require_bool_mask(
                "has_gaze_condition",
                has_gaze_condition,
                gaze_xy.shape[:-1],
            )
        if has_gaze_label is not None and has_gaze_condition is not None:
            invalid_label = has_gaze_label.to(device=has_gaze_condition.device) & ~has_gaze_condition
            if torch.any(invalid_label):
                raise ValueError(
                    "has_gaze_label cannot be True where has_gaze_condition is False."
                )

        if use_gaze_condition is None:
            if has_gaze_condition is None:
                use_gaze_condition = torch.ones(
                    gaze_xy.shape[:-1],
                    device=gaze_xy.device,
                    dtype=torch.bool,
                )
            else:
                use_gaze_condition = has_gaze_condition

        mask = _require_bool_mask(
            "use_gaze_condition",
            use_gaze_condition,
            gaze_xy.shape[:-1],
        )
        if has_gaze_condition is not None:
            route_condition_mask = has_gaze_condition.to(device=mask.device)
            invalid_route = mask & ~route_condition_mask
        else:
            invalid_route = None
        if invalid_route is not None and torch.any(invalid_route):
            raise ValueError(
                "use_gaze_condition cannot be True where has_gaze_condition is False; "
                "missing gaze rows must use the trainable mask token."
            )

        basis = self.spatial_encoder(gaze_xy, valid_mask=has_gaze_condition)
        gaze_token = self.proj(basis).unsqueeze(1)
        mask = mask.to(device=gaze_token.device)
        mask_token = self.mask_token.to(dtype=gaze_token.dtype).expand(
            gaze_token.shape[0], -1, -1
        )
        return torch.where(mask.reshape(-1, 1, 1), gaze_token, mask_token)
