import pathlib
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.common.gaze_wam_training_config import (
    normalize_gaze_wam_positive_int_sequence,
)


def _positive_int_pair(name: str, value: Sequence[int]) -> Tuple[int, int]:
    parsed = normalize_gaze_wam_positive_int_sequence(name, value, length=2)
    return int(parsed[0]), int(parsed[1])


class CosmosHeatmapCodec(nn.Module):
    """Frozen NVIDIA Cosmos continuous image tokenizer for heatmap latents.

    The policy denoises compact 16x16 latent tokens, then decodes them to a
    full-resolution 256x256 heatmap for DSNT/JS supervision and visualization.
    This wrapper adapts a pretrained RGB Cosmos image tokenizer to the
    1-channel heatmap contract by repeating heatmaps to RGB before encoding and
    averaging decoded RGB channels back to heatmap logits.
    """

    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        token_grid: Sequence[int] = (16, 16),
        image_size: Sequence[int] = (256, 256),
        latent_channels: int = 16,
        input_range: str = "minus_one_one",
        output_range: str = "minus_one_one",
        input_normalization: str = "max",
    ) -> None:
        super().__init__()
        self.token_grid = _positive_int_pair("token_grid", token_grid)
        self.image_size = _positive_int_pair("image_size", image_size)
        self.latent_channels = normalize_gaze_wam_positive_int_sequence(
            "latent_channels",
            [latent_channels],
            length=1,
        )[0]
        if input_range not in ("zero_one", "minus_one_one"):
            raise ValueError("input_range must be one of: zero_one, minus_one_one.")
        if output_range not in ("zero_one", "minus_one_one"):
            raise ValueError("output_range must be one of: zero_one, minus_one_one.")
        if input_normalization not in ("none", "max", "mass"):
            raise ValueError("input_normalization must be one of: none, max, mass.")
        self.input_range = str(input_range)
        self.output_range = str(output_range)
        self.input_normalization = str(input_normalization)

        encoder_file = pathlib.Path(encoder_path).expanduser()
        decoder_file = pathlib.Path(decoder_path).expanduser()
        if not encoder_file.is_file():
            raise FileNotFoundError(f"Cosmos encoder JIT file does not exist: {encoder_file}")
        if not decoder_file.is_file():
            raise FileNotFoundError(f"Cosmos decoder JIT file does not exist: {decoder_file}")
        self.encoder_path = str(encoder_file)
        self.decoder_path = str(decoder_file)
        self.encoder = torch.jit.load(str(encoder_file), map_location="cpu")
        self.decoder = torch.jit.load(str(decoder_file), map_location="cpu")
        self.encoder.eval()
        self.decoder.eval()
        self.requires_grad_(False)

    @property
    def num_tokens(self) -> int:
        return self.token_grid[0] * self.token_grid[1]

    @property
    def latent_image_size(self) -> Tuple[int, int]:
        return self.token_grid

    @staticmethod
    def _module_dtype(module: nn.Module) -> torch.dtype:
        for value in module.parameters():
            if torch.is_floating_point(value):
                return value.dtype
        for value in module.buffers():
            if torch.is_floating_point(value):
                return value.dtype
        return torch.float32

    @staticmethod
    def _call_encoder(module: nn.Module, image: torch.Tensor):
        if hasattr(module, "encode"):
            return module.encode(image)
        return module(image)

    @staticmethod
    def _call_decoder(module: nn.Module, latent: torch.Tensor):
        if hasattr(module, "decode"):
            return module.decode(latent)
        return module(latent)

    def _prepare_heatmap(self, heatmap_image: torch.Tensor) -> torch.Tensor:
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
        if heatmap_image.ndim == 3:
            image = heatmap_image.unsqueeze(1)
        elif heatmap_image.ndim == 4 and heatmap_image.shape[1] == 1:
            image = heatmap_image
        else:
            raise ValueError(
                "heatmap_image must have shape [B, H, W] or [B, 1, H, W], "
                f"got {tuple(heatmap_image.shape)}."
            )
        if tuple(image.shape[-2:]) != self.image_size:
            raise ValueError(
                f"heatmap_image spatial shape must match image_size={self.image_size}, "
                f"got {tuple(image.shape[-2:])}."
            )
        if torch.any(image < 0.0):
            raise ValueError("heatmap_image must be non-negative.")
        if self.input_normalization == "max":
            image = image / image.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        elif self.input_normalization == "mass":
            image = image * float(self.image_size[0] * self.image_size[1])
        image = image.clamp(0.0, 1.0).repeat(1, 3, 1, 1)
        if self.input_range == "minus_one_one":
            image = image * 2.0 - 1.0
        return image

    def _normalize_encoder_output(self, encoded) -> torch.Tensor:
        if isinstance(encoded, (tuple, list)):
            encoded = encoded[0]
        if not torch.is_tensor(encoded):
            raise TypeError(
                "Cosmos encoder output must be a tensor or tuple/list whose first item "
                f"is a tensor, got {type(encoded).__name__}."
            )
        if not torch.is_floating_point(encoded):
            raise ValueError(f"Cosmos encoder output must be floating point, got {encoded.dtype}.")
        if encoded.ndim == 5 and encoded.shape[2] == 1:
            encoded = encoded[:, :, 0]
        if encoded.ndim != 4:
            raise ValueError(
                "Cosmos encoder output must have shape [B,C,H,W] or [B,C,1,H,W], "
                f"got {tuple(encoded.shape)}."
            )
        expected = (self.latent_channels, self.token_grid[0], self.token_grid[1])
        if tuple(encoded.shape[1:]) != expected:
            raise ValueError(
                "Cosmos encoder output shape does not match policy heatmap latent contract: "
                f"expected [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {tuple(encoded.shape)}."
            )
        if not torch.all(torch.isfinite(encoded)):
            raise ValueError("Cosmos encoder output must contain only finite values.")
        return encoded

    def latent_image_to_tokens(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4 or latent.shape[1:] != (
            self.latent_channels,
            self.token_grid[0],
            self.token_grid[1],
        ):
            raise ValueError(
                "latent must have shape "
                f"[B,{self.latent_channels},{self.token_grid[0]},{self.token_grid[1]}], "
                f"got {tuple(latent.shape)}."
            )
        return latent.permute(0, 2, 3, 1).reshape(
            latent.shape[0],
            self.num_tokens,
            self.latent_channels,
        )

    def tokens_to_latent_image(self, tokens: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(tokens):
            raise TypeError(f"tokens must be a torch.Tensor, got {type(tokens).__name__}.")
        if not torch.is_floating_point(tokens):
            raise ValueError(f"tokens must be floating point, got {tokens.dtype}.")
        if tokens.ndim != 3 or tokens.shape[1:] != (self.num_tokens, self.latent_channels):
            raise ValueError(
                "tokens must have shape "
                f"[B,{self.num_tokens},{self.latent_channels}], got {tuple(tokens.shape)}."
            )
        if not torch.all(torch.isfinite(tokens)):
            raise ValueError("tokens must contain only finite values.")
        token_h, token_w = self.token_grid
        return tokens.reshape(tokens.shape[0], token_h, token_w, self.latent_channels).permute(
            0,
            3,
            1,
            2,
        )

    @torch.no_grad()
    def encode_image(self, heatmap_image: torch.Tensor) -> torch.Tensor:
        image = self._prepare_heatmap(heatmap_image)
        dtype = image.dtype
        module_dtype = self._module_dtype(self.encoder)
        encoded = self._normalize_encoder_output(
            self._call_encoder(self.encoder, image.to(dtype=module_dtype))
        )
        return self.latent_image_to_tokens(encoded).to(dtype=dtype)

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        module_dtype = self._module_dtype(self.decoder)
        latent = self.tokens_to_latent_image(tokens).to(dtype=module_dtype)
        decoded = self._call_decoder(self.decoder, latent)
        if isinstance(decoded, (tuple, list)):
            decoded = decoded[0]
        if not torch.is_tensor(decoded):
            raise TypeError(
                "Cosmos decoder output must be a tensor or tuple/list whose first item "
                f"is a tensor, got {type(decoded).__name__}."
            )
        if decoded.ndim == 5 and decoded.shape[2] == 1:
            decoded = decoded[:, :, 0]
        if decoded.ndim != 4 or decoded.shape[1] not in (1, 3):
            raise ValueError(
                "Cosmos decoder output must have shape [B,1,H,W], [B,3,H,W], "
                f"[B,1,1,H,W], or [B,3,1,H,W], got {tuple(decoded.shape)}."
            )
        if tuple(decoded.shape[-2:]) != self.image_size:
            decoded = F.interpolate(
                decoded,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )
        if decoded.shape[1] == 3:
            decoded = decoded.mean(dim=1, keepdim=True)
        if self.output_range == "minus_one_one":
            decoded = (decoded + 1.0) * 0.5
        return decoded[:, 0].to(dtype=tokens.dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if (
            torch.is_tensor(value)
            and value.ndim == 3
            and value.shape[1:] == (self.num_tokens, self.latent_channels)
        ):
            return self.decode_tokens(value)
        return self.decode_tokens(self.encode_image(value))
