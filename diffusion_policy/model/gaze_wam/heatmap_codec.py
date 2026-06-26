from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

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


def _validate_binary_mask(name: str, value: torch.Tensor, expected_shape: torch.Size) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if value.shape != expected_shape:
        raise ValueError(f"{name} must have shape {tuple(expected_shape)}, got {tuple(value.shape)}.")
    if value.dtype == torch.bool:
        return value
    if torch.is_complex(value):
        raise ValueError(f"{name} must be a boolean or numeric 0/1 tensor, got complex dtype.")
    if torch.is_floating_point(value):
        if not torch.all(torch.isfinite(value)):
            raise ValueError(f"{name} must contain only finite boolean or numeric 0/1 values.")
    if not torch.all((value == 0) | (value == 1)):
        raise ValueError(f"{name} must contain only boolean or numeric 0/1 values.")
    return value.to(dtype=torch.bool)


def _validate_positive_float(name: str, value: float) -> float:
    try:
        return normalize_gaze_wam_positive_float_field(name, value)
    except ValueError as exc:
        raise ValueError(f"{name} must be positive.") from exc


class HeatmapTokenCodec:
    """Fixed codec between gaze points, latent heatmap tokens, and image heatmaps."""

    def __init__(
        self,
        token_grid: Sequence[int] = (16, 16),
        image_size: Sequence[int] = (256, 256),
        sigma_tokens: float = 1.25,
        normalize: str = "max",
    ) -> None:
        if normalize not in ("none", "max", "sum"):
            raise ValueError("normalize must be one of: none, max, sum.")

        self.token_grid = _validate_positive_int_pair("token_grid", token_grid)
        self.image_size = _validate_positive_int_pair("image_size", image_size)
        self.sigma_tokens = _validate_positive_float("sigma_tokens", sigma_tokens)
        self.normalize = normalize

    @property
    def num_tokens(self) -> int:
        return self.token_grid[0] * self.token_grid[1]

    @property
    def patch_size(self) -> Tuple[int, int]:
        image_h, image_w = self.image_size
        token_h, token_w = self.token_grid
        if image_h % token_h != 0 or image_w % token_w != 0:
            raise ValueError(
                "image_size must be divisible by token_grid for patchified heatmap "
                f"tokens, got image_size={self.image_size}, token_grid={self.token_grid}."
            )
        return image_h // token_h, image_w // token_w

    @property
    def patch_area(self) -> int:
        patch_h, patch_w = self.patch_size
        return patch_h * patch_w

    def _centers(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        height, width = self.token_grid
        y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(-1, 2)

    def _normalize_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.normalize == "max":
            return tokens / tokens.amax(dim=-2, keepdim=True).clamp_min(1e-12)
        if self.normalize == "sum":
            return tokens / tokens.sum(dim=-2, keepdim=True).clamp_min(1e-12)
        return tokens

    def encode_points(
        self,
        gaze_xy: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Create token heatmap labels from normalized gaze points.

        Args:
            gaze_xy: Tensor with shape ``[..., 2]`` in normalized ``[0, 1]`` image coords.
            valid_mask: Optional boolean or numeric 0/1 tensor with shape ``[...]``.

        Returns:
            Tensor with shape ``[..., num_tokens, 1]``.
        """
        if not torch.is_tensor(gaze_xy):
            raise TypeError(f"gaze_xy must be a torch.Tensor, got {type(gaze_xy).__name__}.")
        if not torch.is_floating_point(gaze_xy):
            raise ValueError(f"gaze_xy must be a floating point tensor, got {gaze_xy.dtype}.")
        if gaze_xy.shape[-1] != 2:
            raise ValueError(f"Expected gaze_xy last dim 2, got {gaze_xy.shape}.")
        if not torch.all(torch.isfinite(gaze_xy)):
            raise ValueError("gaze_xy must contain only finite values.")
        if not torch.all((gaze_xy >= 0.0) & (gaze_xy <= 1.0)):
            raise ValueError("gaze_xy must be normalized to [0, 1].")

        xy = gaze_xy
        centers = self._centers(xy.device, xy.dtype)
        sigma_x = self.sigma_tokens / self.token_grid[1]
        sigma_y = self.sigma_tokens / self.token_grid[0]
        diff = xy.unsqueeze(-2) - centers
        dist = (diff[..., 0] / sigma_x).square() + (diff[..., 1] / sigma_y).square()
        tokens = torch.exp(-0.5 * dist).unsqueeze(-1)
        tokens = self._normalize_tokens(tokens)

        if valid_mask is not None:
            valid_mask = _validate_binary_mask(
                "valid_mask",
                valid_mask,
                tokens.shape[:-2],
            ).to(device=tokens.device)
            tokens = tokens * valid_mask.to(dtype=tokens.dtype).reshape(
                tokens.shape[:-2] + (1, 1)
            )
        return tokens

    def encode_image(self, heatmap_image: torch.Tensor) -> torch.Tensor:
        """Area-pool 1-channel heatmap images into ``[..., num_tokens, 1]`` labels."""
        squeeze_channel = False
        if not torch.is_tensor(heatmap_image):
            raise TypeError(
                f"heatmap_image must be a torch.Tensor, got {type(heatmap_image).__name__}."
            )
        if not torch.is_floating_point(heatmap_image):
            raise ValueError(
                f"heatmap_image must be a floating point tensor, got {heatmap_image.dtype}."
            )
        if not torch.all(torch.isfinite(heatmap_image)):
            raise ValueError("heatmap_image must contain only finite values.")
        if torch.any(heatmap_image < 0.0):
            raise ValueError("heatmap_image must be non-negative.")
        if heatmap_image.ndim < 2:
            raise ValueError("heatmap_image must have at least two spatial dims.")
        if tuple(heatmap_image.shape[-2:]) != self.image_size:
            raise ValueError(
                f"heatmap_image spatial shape must match image_size={self.image_size}, "
                f"got {tuple(heatmap_image.shape[-2:])}."
            )
        if heatmap_image.ndim >= 4 and heatmap_image.shape[-3:-2] == (1,):
            image = heatmap_image
        else:
            if heatmap_image.ndim >= 4:
                raise ValueError(
                    "heatmap_image must be a single-channel tensor with shape [..., H, W] "
                    "or [..., 1, H, W]."
                )
            image = heatmap_image.unsqueeze(-3)
            squeeze_channel = True

        leading = image.shape[:-3]
        image = image.reshape((-1,) + image.shape[-3:])
        pooled = F.adaptive_avg_pool2d(image, self.token_grid)
        pooled = pooled.reshape(leading + (1,) + self.token_grid)
        pooled = pooled.flatten(-2).transpose(-1, -2)
        if squeeze_channel:
            pooled = pooled.reshape(leading + (self.num_tokens, 1))
        return self._normalize_tokens(pooled)

    def patchify_image(self, heatmap_image: torch.Tensor) -> torch.Tensor:
        """Patchify RGB-aligned heatmaps into ``[..., num_tokens, patch_area]`` tokens."""
        dense = self._prepare_single_channel_image(heatmap_image)
        leading = dense.shape[:-3]
        patch_h, patch_w = self.patch_size
        token_h, token_w = self.token_grid
        flat = dense.reshape((-1, 1) + self.image_size)
        patches = flat.reshape(
            flat.shape[0],
            1,
            token_h,
            patch_h,
            token_w,
            patch_w,
        )
        patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(
            flat.shape[0],
            self.num_tokens,
            self.patch_area,
        )
        return patches.reshape(leading + (self.num_tokens, self.patch_area))

    def encode_latent_image(
        self,
        heatmap_image: torch.Tensor,
        latent_channels: int,
    ) -> torch.Tensor:
        """Encode a full-resolution heatmap into compact spatial latent tokens.

        This deterministic fallback preserves the new channel-latent contract:
        ``[B, token_h * token_w, C]`` represents ``[B, C, token_h, token_w]``.
        It is intentionally simple and exists for smoke tests/backward
        compatibility. The canonical full-resolution heatmap path uses the
        frozen NVIDIA Cosmos tokenizer to produce clean latent targets.
        """
        latent_channels = normalize_gaze_wam_positive_int_sequence(
            "latent_channels",
            [latent_channels],
            length=1,
        )[0]
        dense = self._prepare_single_channel_image(heatmap_image)
        leading = dense.shape[:-3]
        token_h, token_w = self.token_grid
        flat = dense.reshape((-1, 1) + self.image_size)
        pooled = F.adaptive_avg_pool2d(flat, output_size=(token_h, token_w))
        latents = pooled.repeat(1, int(latent_channels), 1, 1)
        latents = latents.permute(0, 2, 3, 1).reshape(
            flat.shape[0],
            self.num_tokens,
            int(latent_channels),
        )
        return latents.reshape(leading + (self.num_tokens, int(latent_channels)))

    def unpatchify_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode ``[..., num_tokens, patch_area]`` tokens to ``[..., H, W]`` heatmaps."""
        if not torch.is_tensor(tokens):
            raise TypeError(f"tokens must be a torch.Tensor, got {type(tokens).__name__}.")
        if not torch.is_floating_point(tokens):
            raise ValueError(f"tokens must be a floating point tensor, got {tokens.dtype}.")
        if not torch.all(torch.isfinite(tokens)):
            raise ValueError("tokens must contain only finite values.")
        if tokens.shape[-2:] != (self.num_tokens, self.patch_area):
            raise ValueError(
                "Patchified heatmap tokens must have shape "
                f"[..., {self.num_tokens}, {self.patch_area}], got {tokens.shape}."
            )

        leading = tokens.shape[:-2]
        patch_h, patch_w = self.patch_size
        token_h, token_w = self.token_grid
        flat = tokens.reshape((-1, self.num_tokens, self.patch_area))
        image = flat.reshape(flat.shape[0], token_h, token_w, patch_h, patch_w)
        image = image.permute(0, 1, 3, 2, 4).reshape(flat.shape[0], *self.image_size)
        return image.reshape(leading + self.image_size)

    def decode_latent_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode compact latent tokens to full-resolution heatmap logits/images."""
        if not torch.is_tensor(tokens):
            raise TypeError(f"tokens must be a torch.Tensor, got {type(tokens).__name__}.")
        if not torch.is_floating_point(tokens):
            raise ValueError(f"tokens must be a floating point tensor, got {tokens.dtype}.")
        if not torch.all(torch.isfinite(tokens)):
            raise ValueError("tokens must contain only finite values.")
        if tokens.ndim < 3 or tokens.shape[-2] != self.num_tokens:
            raise ValueError(
                "Latent heatmap tokens must have shape "
                f"[..., {self.num_tokens}, C], got {tokens.shape}."
            )
        leading = tokens.shape[:-2]
        token_h, token_w = self.token_grid
        flat = tokens.reshape((-1, self.num_tokens, int(tokens.shape[-1])))
        latent_map = flat.reshape(flat.shape[0], token_h, token_w, int(tokens.shape[-1]))
        latent_map = latent_map.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)
        decoded = F.interpolate(
            latent_map,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        return decoded[:, 0].reshape(leading + self.image_size)

    def _prepare_single_channel_image(self, heatmap_image: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(heatmap_image):
            raise TypeError(
                f"heatmap_image must be a torch.Tensor, got {type(heatmap_image).__name__}."
            )
        if not torch.is_floating_point(heatmap_image):
            raise ValueError(
                f"heatmap_image must be a floating point tensor, got {heatmap_image.dtype}."
            )
        if not torch.all(torch.isfinite(heatmap_image)):
            raise ValueError("heatmap_image must contain only finite values.")
        if torch.any(heatmap_image < 0.0):
            raise ValueError("heatmap_image must be non-negative.")
        if heatmap_image.ndim < 2:
            raise ValueError("heatmap_image must have at least two spatial dims.")
        if tuple(heatmap_image.shape[-2:]) != self.image_size:
            raise ValueError(
                f"heatmap_image spatial shape must match image_size={self.image_size}, "
                f"got {tuple(heatmap_image.shape[-2:])}."
            )
        if heatmap_image.ndim >= 4 and heatmap_image.shape[-3:-2] == (1,):
            return heatmap_image
        if heatmap_image.ndim >= 4:
            raise ValueError(
                "heatmap_image must be a single-channel tensor with shape [..., H, W] "
                "or [..., 1, H, W]."
            )
        return heatmap_image.unsqueeze(-3)

    def decode_tokens(
        self,
        tokens: torch.Tensor,
        image_size: Optional[Sequence[int]] = None,
        method: str = "gaussian_splat",
    ) -> torch.Tensor:
        """Decode token heatmaps to image heatmaps for visualization."""
        if not torch.is_tensor(tokens):
            raise TypeError(f"tokens must be a torch.Tensor, got {type(tokens).__name__}.")
        if not torch.is_floating_point(tokens):
            raise ValueError(f"tokens must be a floating point tensor, got {tokens.dtype}.")
        if not torch.all(torch.isfinite(tokens)):
            raise ValueError("tokens must contain only finite values.")
        if torch.any(tokens < 0.0):
            raise ValueError("tokens must be non-negative for heatmap visualization decode.")
        if tokens.shape[-2:] == (self.num_tokens, self.patch_area):
            if image_size is not None and tuple(image_size) != self.image_size:
                raise ValueError(
                    "Patchified heatmap-token decode requires image_size to match "
                    f"codec image_size={self.image_size}, got {tuple(image_size)}."
                )
            return self.unpatchify_tokens(tokens)

        if tokens.shape[-2:] != (self.num_tokens, 1):
            if tokens.ndim >= 3 and tokens.shape[-1] == 1:
                raise ValueError(
                    "Expected scalar tokens with shape "
                    f"[..., {self.num_tokens}, 1], got {tokens.shape}."
                )
            if image_size is not None and tuple(image_size) != self.image_size:
                raise ValueError(
                    "Latent heatmap-token decode requires image_size to match "
                    f"codec image_size={self.image_size}, got {tuple(image_size)}."
                )
            return self.decode_latent_tokens(tokens)

        out_size = _validate_positive_int_pair("image_size", image_size or self.image_size)

        leading = tokens.shape[:-2]
        token_image = tokens.reshape(leading + self.token_grid)

        if method == "bilinear":
            flat = token_image.reshape((-1, 1) + self.token_grid)
            decoded = F.interpolate(
                flat,
                size=out_size,
                mode="bilinear",
                align_corners=False,
            )
            return decoded.reshape(leading + out_size)

        if method != "gaussian_splat":
            raise ValueError("method must be 'bilinear' or 'gaussian_splat'.")

        image_h, image_w = out_size
        y = (torch.arange(image_h, device=tokens.device, dtype=tokens.dtype) + 0.5) / image_h
        x = (torch.arange(image_w, device=tokens.device, dtype=tokens.dtype) + 0.5) / image_w
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        pixels = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        centers = self._centers(tokens.device, tokens.dtype)
        sigma_x = self.sigma_tokens / self.token_grid[1]
        sigma_y = self.sigma_tokens / self.token_grid[0]
        diff = pixels.unsqueeze(-2) - centers
        dist = (diff[..., 0] / sigma_x).square() + (diff[..., 1] / sigma_y).square()
        basis = torch.exp(-0.5 * dist)
        flat_tokens = tokens.squeeze(-1).reshape((-1, self.num_tokens))
        decoded = flat_tokens @ basis.transpose(0, 1)
        decoded = decoded.reshape(leading + out_size)
        max_value = decoded.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        return decoded / max_value
