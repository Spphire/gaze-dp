from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.common.gaze_wam_training_config import (
    normalize_gaze_wam_bool_field,
    normalize_gaze_wam_positive_int_field,
)
from diffusion_policy.model.gaze_wam.joint_transformer import (
    GazeWamTransformerOutput,
    SinusoidalPosEmb,
)


def _normalize_positive_int(name: str, value) -> int:
    return normalize_gaze_wam_positive_int_field(name, value, default=None)


def _normalize_optional_positive_int(name: str, value) -> Optional[int]:
    if value is None:
        return None
    return normalize_gaze_wam_positive_int_field(name, value)


class Fp32LayerNorm(nn.LayerNorm):
    """Compute normalization in FP32 while preserving the surrounding dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps,
        )
        return output.to(dtype=self.weight.dtype)


@dataclass(frozen=True)
class GazeWamWorldCache:
    """Per-layer image and gaze K/V caches consumed by target decoders."""

    image_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    gaze_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    num_image_tokens: int
    batch_size: int


GazeWamConditionCache = GazeWamWorldCache


class FeedForwardBlock(nn.Module):
    def __init__(self, n_emb: int, p_drop: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(p_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContextSelfBlock(nn.Module):
    """Fast-WAM-style world block over stable image/gaze tokens.

    The block exposes the per-layer world K/V produced from the same normalized
    tokens used by world self-attention. Action denoising can then reuse those
    K/V tensors instead of cross-attending to a separate condition encoder.
    """

    def __init__(self, n_emb: int, n_head: int, p_drop_attn: float) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_emb // n_head
        self.scale = self.head_dim ** -0.5
        self.ln_self = Fp32LayerNorm(n_emb)
        self.query = nn.Linear(n_emb, n_emb)
        self.key = nn.Linear(n_emb, n_emb)
        self.value = nn.Linear(n_emb, n_emb)
        self.out = nn.Linear(n_emb, n_emb)
        self.attn_drop = nn.Dropout(p_drop_attn)
        self.resid_drop = nn.Dropout(p_drop_attn)
        self.ln_mlp = Fp32LayerNorm(n_emb)
        self.mlp = FeedForwardBlock(n_emb=n_emb, p_drop=p_drop_attn)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        return x.reshape(batch, tokens, self.n_head, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, tokens, channels = x.shape
        return x.transpose(1, 2).reshape(batch, tokens, heads * channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        norm = self.ln_self(x)
        query = self._reshape_heads(self.query(norm))
        key = self._reshape_heads(self.key(norm))
        value = self._reshape_heads(self.value(norm))
        attn = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        attended = torch.matmul(attn, value)
        attended = self.resid_drop(self.out(self._merge_heads(attended)))
        x = x + attended
        return x + self.mlp(self.ln_mlp(x)), key, value


class CachedMixedAttention(nn.Module):
    """Target attention with separate target, image, and gaze softmaxes.

    The previous implementation concatenated image, gaze, and target K/V before
    one softmax.  That made the single gaze token compete with hundreds of
    visual/target tokens.  Each source now gets its own normalized attention
    distribution, and the resulting value contexts are fused before the output
    projection.  The parameter names remain compatible with existing weights.
    """

    def __init__(self, n_emb: int, n_head: int, p_drop_attn: float) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_emb // n_head
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(n_emb, n_emb)
        self.key = nn.Linear(n_emb, n_emb)
        self.value = nn.Linear(n_emb, n_emb)
        self.out = nn.Linear(n_emb, n_emb)
        self.attn_drop = nn.Dropout(p_drop_attn)
        # A single gaze source token must not be dropped as a whole branch.
        self.gaze_attn_drop = nn.Identity()
        self.resid_drop = nn.Dropout(p_drop_attn)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        return x.reshape(batch, tokens, self.n_head, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, tokens, channels = x.shape
        return x.transpose(1, 2).reshape(batch, tokens, heads * channels)

    def forward(
        self,
        target_tokens: torch.Tensor,
        image_cache_key: torch.Tensor,
        image_cache_value: torch.Tensor,
        gaze_cache_key: torch.Tensor,
        gaze_cache_value: torch.Tensor,
    ) -> torch.Tensor:
        query = self._reshape_heads(self.query(target_tokens))
        target_key = self._reshape_heads(self.key(target_tokens))
        target_value = self._reshape_heads(self.value(target_tokens))

        def attend(
            key: torch.Tensor,
            value: torch.Tensor,
            dropout: nn.Module,
        ) -> torch.Tensor:
            attn = torch.matmul(query, key.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = dropout(attn)
            return torch.matmul(attn, value)

        target_context = attend(target_key, target_value, self.attn_drop)
        image_context = attend(image_cache_key, image_cache_value, self.attn_drop)
        gaze_context = attend(
            gaze_cache_key,
            gaze_cache_value,
            self.gaze_attn_drop,
        )
        fused = target_context + image_context + gaze_context
        return self.resid_drop(self.out(self._merge_heads(fused)))


class TargetDecoderBlock(nn.Module):
    """Action or heatmap target block with Fast-WAM-style cached mixed attention."""

    def __init__(self, n_emb: int, n_head: int, p_drop_attn: float) -> None:
        super().__init__()
        self.ln_attn = Fp32LayerNorm(n_emb)
        self.mixed_attn = CachedMixedAttention(
            n_emb=n_emb,
            n_head=n_head,
            p_drop_attn=p_drop_attn,
        )
        self.ln_mlp = Fp32LayerNorm(n_emb)
        self.mlp = FeedForwardBlock(n_emb=n_emb, p_drop=p_drop_attn)

    def forward(
        self,
        target: torch.Tensor,
        image_cache_key: torch.Tensor,
        image_cache_value: torch.Tensor,
        gaze_cache_key: torch.Tensor,
        gaze_cache_value: torch.Tensor,
    ) -> torch.Tensor:
        target = target + self.mixed_attn(
            target_tokens=self.ln_attn(target),
            image_cache_key=image_cache_key,
            image_cache_value=image_cache_value,
            gaze_cache_key=gaze_cache_key,
            gaze_cache_value=gaze_cache_value,
        )
        return target + self.mlp(self.ln_mlp(target))


class CachedDualStreamGazeWamTransformer(nn.Module):
    """Fast-WAM-style cached world expert with separate action and heatmap decoders."""

    MODALITY_IMAGE = 0
    MODALITY_GAZE = 1
    MODALITY_ACTION = 2
    MODALITY_HEATMAP = 3

    def __init__(
        self,
        action_dim: int = 10,
        heatmap_dim: int = 16,
        action_horizon: int = 48,
        heatmap_num_tokens: int = 256,
        max_image_tokens: int = 512,
        n_layer: int = 7,
        n_head: int = 8,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        use_block_attention_mask: bool = True,
        use_frame_embedding: bool = False,
        image_tokens_per_frame: Optional[int] = None,
        max_obs_frames: Optional[int] = None,
    ) -> None:
        super().__init__()

        action_dim = _normalize_positive_int("action_dim", action_dim)
        heatmap_dim = _normalize_positive_int("heatmap_dim", heatmap_dim)
        action_horizon = _normalize_positive_int("action_horizon", action_horizon)
        heatmap_num_tokens = _normalize_positive_int("heatmap_num_tokens", heatmap_num_tokens)
        max_image_tokens = _normalize_positive_int("max_image_tokens", max_image_tokens)
        n_layer = _normalize_positive_int("n_layer", n_layer)
        n_head = _normalize_positive_int("n_head", n_head)
        n_emb = _normalize_positive_int("n_emb", n_emb)
        if n_emb % n_head != 0:
            raise ValueError(
                f"n_emb must be divisible by n_head, got n_emb={n_emb}, n_head={n_head}."
            )

        self.action_dim = action_dim
        self.heatmap_dim = heatmap_dim
        self.action_horizon = action_horizon
        self.heatmap_num_tokens = heatmap_num_tokens
        self.max_image_tokens = max_image_tokens
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_emb = n_emb
        self.supports_skip_action_decoder = True
        self.use_block_attention_mask = normalize_gaze_wam_bool_field(
            "policy.use_block_attention_mask",
            use_block_attention_mask,
            default=True,
        )
        if not self.use_block_attention_mask:
            raise ValueError(
                "CachedDualStreamGazeWamTransformer always enforces condition/target "
                "modality isolation; set use_block_attention_mask=true for the active "
                "policy-training path."
            )
        self.use_frame_embedding = normalize_gaze_wam_bool_field(
            "policy.use_frame_embedding",
            use_frame_embedding,
            default=False,
        )
        if self.use_frame_embedding:
            image_tokens_per_frame = _normalize_positive_int(
                "image_tokens_per_frame",
                image_tokens_per_frame,
            )
            if max_obs_frames is None:
                max_obs_frames = (max_image_tokens + image_tokens_per_frame - 1) // image_tokens_per_frame
            max_obs_frames = _normalize_positive_int("max_obs_frames", max_obs_frames)
            self.image_tokens_per_frame = image_tokens_per_frame
            self.max_obs_frames = max_obs_frames
            self.frame_emb = nn.Embedding(self.max_obs_frames, n_emb)
        else:
            self.image_tokens_per_frame = _normalize_optional_positive_int(
                "image_tokens_per_frame",
                image_tokens_per_frame,
            )
            self.max_obs_frames = _normalize_optional_positive_int(
                "max_obs_frames",
                max_obs_frames,
            )
            self.frame_emb = None

        self.image_proj = nn.Identity()
        self.gaze_proj = nn.Identity()
        self.action_proj = nn.Linear(action_dim, n_emb)
        self.heatmap_proj = nn.Linear(heatmap_dim, n_emb)

        self.context_pos_emb = nn.Parameter(torch.zeros(1, max_image_tokens + 1, n_emb))
        self.action_pos_emb = nn.Parameter(torch.zeros(1, action_horizon, n_emb))
        self.heatmap_pos_emb = nn.Parameter(torch.zeros(1, heatmap_num_tokens, n_emb))
        self.modality_emb = nn.Embedding(4, n_emb)
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.drop = nn.Dropout(p_drop_emb)

        self.context_blocks = nn.ModuleList(
            [
                ContextSelfBlock(n_emb=n_emb, n_head=n_head, p_drop_attn=p_drop_attn)
                for _ in range(n_layer)
            ]
        )
        self.action_blocks = nn.ModuleList(
            [
                TargetDecoderBlock(n_emb=n_emb, n_head=n_head, p_drop_attn=p_drop_attn)
                for _ in range(n_layer)
            ]
        )
        self.heatmap_blocks = nn.ModuleList(
            [
                TargetDecoderBlock(n_emb=n_emb, n_head=n_head, p_drop_attn=p_drop_attn)
                for _ in range(n_layer)
            ]
        )

        self.action_ln_f = Fp32LayerNorm(n_emb)
        self.action_head = nn.Linear(n_emb, action_dim)
        self.heatmap_ln_f = Fp32LayerNorm(n_emb)
        self.heatmap_head = nn.Linear(n_emb, heatmap_dim)

        self.apply(self._init_weights)
        nn.init.normal_(self.context_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.action_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.heatmap_pos_emb, mean=0.0, std=0.02)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                "in_proj_weight",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
            ]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    nn.init.normal_(weight, mean=0.0, std=0.02)
            for name in ["in_proj_bias", "bias_k", "bias_v"]:
                bias = getattr(module, name)
                if bias is not None:
                    nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _validate_num_image_tokens(self, num_image_tokens) -> int:
        num_image_tokens = _normalize_positive_int("num_image_tokens", num_image_tokens)
        if num_image_tokens > self.max_image_tokens:
            raise ValueError(
                f"num_image_tokens={num_image_tokens} exceeds max_image_tokens="
                f"{self.max_image_tokens}."
            )
        return num_image_tokens

    def build_image_frame_ids(
        self,
        num_image_tokens: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if not self.use_frame_embedding:
            raise RuntimeError("Frame ids are only defined when use_frame_embedding=True.")
        if num_image_tokens % self.image_tokens_per_frame != 0:
            raise ValueError(
                f"num_image_tokens={num_image_tokens} is not divisible by "
                f"image_tokens_per_frame={self.image_tokens_per_frame}."
            )
        num_frames = num_image_tokens // self.image_tokens_per_frame
        if num_frames > self.max_obs_frames:
            raise ValueError(
                f"num_frames={num_frames} exceeds max_obs_frames={self.max_obs_frames}."
            )
        return torch.arange(num_frames, dtype=torch.long, device=device).repeat_interleave(
            self.image_tokens_per_frame
        )

    def _add_frame_embedding(self, image_tokens: torch.Tensor) -> torch.Tensor:
        if not self.use_frame_embedding:
            return image_tokens
        frame_ids = self.build_image_frame_ids(
            num_image_tokens=image_tokens.shape[1],
            device=image_tokens.device,
        )
        return image_tokens + self.frame_emb(frame_ids).unsqueeze(0)

    def _prepare_timesteps(
        self,
        timestep: Union[torch.Tensor, float, int],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.as_tensor(timestep, device=device)
        else:
            timestep = timestep.to(device=device)
        if timestep.ndim == 0:
            timestep = timestep[None]
        elif timestep.ndim != 1:
            raise ValueError(
                "timestep must be a scalar or 1D tensor/list, got shape "
                f"{tuple(timestep.shape)}."
            )
        if timestep.dtype == torch.bool or torch.is_complex(timestep):
            raise ValueError("timestep must contain integer diffusion timesteps.")
        if torch.is_floating_point(timestep):
            if not torch.isfinite(timestep).all():
                raise ValueError("timestep must contain only finite diffusion timesteps.")
            if not torch.all(timestep == timestep.round()):
                raise ValueError("timestep must contain integer diffusion timesteps.")
        timestep = timestep.reshape(-1)
        if timestep.numel() not in (1, batch_size):
            raise ValueError(
                f"timestep must contain either 1 or B={batch_size} diffusion timesteps, "
                f"got {timestep.numel()}."
            )
        return timestep.expand(batch_size).long()

    def _validate_condition_inputs(
        self,
        image_tokens: torch.Tensor,
        gaze_token: torch.Tensor,
    ) -> int:
        if image_tokens.ndim != 3 or image_tokens.shape[-1] != self.n_emb:
            raise ValueError(
                f"image_tokens must be [B, N_v, {self.n_emb}], got {image_tokens.shape}."
            )
        batch_size, num_image_tokens = image_tokens.shape[:2]
        num_image_tokens = self._validate_num_image_tokens(num_image_tokens)
        if gaze_token.shape != (batch_size, 1, self.n_emb):
            raise ValueError(
                "gaze_token must be [B, 1, D] and match image batch/hidden dims, "
                f"got {gaze_token.shape}."
            )
        return num_image_tokens

    def prefill_world_cache(
        self,
        image_tokens: torch.Tensor,
        gaze_token: torch.Tensor,
    ) -> GazeWamWorldCache:
        """Encode image/gaze once and return per-layer world K/V cache.

        This is the Gaze-WAM analogue of FastWAM's action-only first-frame
        prefill path: stable visual/gaze condition tokens are cached once, then
        action denoising consumes the cache at every step. No noisy heatmap
        target tokens are cached for action inference.
        """
        num_image_tokens = self._validate_condition_inputs(image_tokens, gaze_token)
        image_part = self._add_frame_embedding(self.image_proj(image_tokens))
        image_ids = torch.full(
            (num_image_tokens,),
            self.MODALITY_IMAGE,
            dtype=torch.long,
            device=image_tokens.device,
        )
        gaze_ids = torch.full(
            (1,),
            self.MODALITY_GAZE,
            dtype=torch.long,
            device=image_tokens.device,
        )
        context = torch.cat([image_part, self.gaze_proj(gaze_token)], dim=1)
        modality_ids = torch.cat([image_ids, gaze_ids], dim=0)
        context = context + self.modality_emb(modality_ids).unsqueeze(0)
        context = context + self.context_pos_emb[:, : context.shape[1], :]
        context = self.drop(context)

        image_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = []
        gaze_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.context_blocks:
            context, key, value = block(context)
            image_key_values.append(
                (
                    key[:, :, :num_image_tokens, :],
                    value[:, :, :num_image_tokens, :],
                )
            )
            gaze_key_values.append(
                (
                    key[:, :, num_image_tokens:, :],
                    value[:, :, num_image_tokens:, :],
                )
            )
        return GazeWamWorldCache(
            image_key_values=tuple(image_key_values),
            gaze_key_values=tuple(gaze_key_values),
            num_image_tokens=int(num_image_tokens),
            batch_size=int(image_tokens.shape[0]),
        )

    def prefill_condition_cache(
        self,
        image_tokens: torch.Tensor,
        gaze_token: torch.Tensor,
    ) -> GazeWamWorldCache:
        """Backward-compatible alias for older cached-dual-stream call sites."""
        return self.prefill_world_cache(image_tokens=image_tokens, gaze_token=gaze_token)

    def _validate_world_cache(
        self,
        world_cache: GazeWamWorldCache,
        batch_size: int,
        num_image_tokens: int,
    ) -> None:
        if not isinstance(world_cache, GazeWamWorldCache):
            raise TypeError(
                "world_cache must be a GazeWamWorldCache produced by "
                "prefill_world_cache()."
            )
        if world_cache.batch_size != batch_size:
            raise ValueError(
                "world_cache batch size must match target batch size, got "
                f"{world_cache.batch_size} vs {batch_size}."
            )
        if world_cache.num_image_tokens != num_image_tokens:
            raise ValueError(
                "world_cache num_image_tokens must match image_tokens, got "
                f"{world_cache.num_image_tokens} vs {num_image_tokens}."
            )
        if len(world_cache.image_key_values) != self.n_layer:
            raise ValueError(
                f"world_cache image cache must contain {self.n_layer} layer K/V pairs, "
                f"got {len(world_cache.image_key_values)}."
            )
        if len(world_cache.gaze_key_values) != self.n_layer:
            raise ValueError(
                f"world_cache gaze cache must contain {self.n_layer} layer K/V pairs, "
                f"got {len(world_cache.gaze_key_values)}."
            )
        for layer, ((image_key, image_value), (gaze_key, gaze_value)) in enumerate(
            zip(world_cache.image_key_values, world_cache.gaze_key_values)
        ):
            if image_key.shape[2] != num_image_tokens or gaze_key.shape[2] != 1:
                raise ValueError(
                    "world_cache layer "
                    f"{layer} must contain {num_image_tokens} image and 1 gaze token, "
                    f"got {image_key.shape[2]} image and {gaze_key.shape[2]} gaze tokens."
                )

    def _validate_condition_cache(
        self,
        condition_cache: GazeWamWorldCache,
        batch_size: int,
        num_image_tokens: int,
    ) -> None:
        """Backward-compatible validation alias."""
        self._validate_world_cache(
            world_cache=condition_cache,
            batch_size=batch_size,
            num_image_tokens=num_image_tokens,
        )

    def _prepare_action_tokens(
        self,
        noisy_action: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        action_ids = torch.full(
            (self.action_horizon,),
            self.MODALITY_ACTION,
            dtype=torch.long,
            device=noisy_action.device,
        )
        tokens = self.action_proj(noisy_action)
        tokens = tokens + time_emb + self.action_pos_emb
        tokens = tokens + self.modality_emb(action_ids).unsqueeze(0)
        return self.drop(tokens)

    def _prepare_heatmap_tokens(
        self,
        noisy_heatmap: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        heatmap_ids = torch.full(
            (self.heatmap_num_tokens,),
            self.MODALITY_HEATMAP,
            dtype=torch.long,
            device=noisy_heatmap.device,
        )
        tokens = self.heatmap_proj(noisy_heatmap)
        tokens = tokens + time_emb + self.heatmap_pos_emb
        tokens = tokens + self.modality_emb(heatmap_ids).unsqueeze(0)
        return self.drop(tokens)

    def forward(
        self,
        image_tokens: torch.Tensor,
        gaze_token: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        noisy_heatmap: Optional[torch.Tensor] = None,
        is_inference: bool = False,
        world_cache: Optional[GazeWamWorldCache] = None,
        condition_cache: Optional[GazeWamWorldCache] = None,
        skip_action: bool = False,
    ) -> GazeWamTransformerOutput:
        num_image_tokens = self._validate_condition_inputs(image_tokens, gaze_token)
        batch_size = int(image_tokens.shape[0])
        skip_action = normalize_gaze_wam_bool_field(
            "skip_action",
            skip_action,
            default=False,
        )
        if noisy_action.shape != (batch_size, self.action_horizon, self.action_dim):
            raise ValueError(
                f"noisy_action must be [B, {self.action_horizon}, {self.action_dim}], "
                f"match image batch size {batch_size}, got {noisy_action.shape}."
            )
        if skip_action and is_inference:
            raise ValueError("skip_action=True is only valid for training/preview heatmap paths.")
        if is_inference and noisy_heatmap is not None:
            raise ValueError(
                "Inference must omit noisy_heatmap; heatmap decoder is skipped for "
                "the action-only cached path."
            )
        if (not is_inference) and noisy_heatmap is None:
            raise ValueError(
                "Training forward must include noisy_heatmap; only the action-only "
                "inference fast path may omit heatmap tokens."
            )
        include_heatmap = noisy_heatmap is not None
        if include_heatmap and noisy_heatmap.shape != (
            batch_size,
            self.heatmap_num_tokens,
            self.heatmap_dim,
        ):
            raise ValueError(
                "noisy_heatmap must be "
                f"[B, {self.heatmap_num_tokens}, {self.heatmap_dim}], "
                f"match image batch size {batch_size}, got {noisy_heatmap.shape}."
            )

        if world_cache is not None and condition_cache is not None and world_cache is not condition_cache:
            raise ValueError("Pass only one of world_cache or condition_cache.")
        if world_cache is None:
            world_cache = condition_cache

        if world_cache is None:
            world_cache = self.prefill_world_cache(image_tokens, gaze_token)
        else:
            self._validate_world_cache(
                world_cache=world_cache,
                batch_size=batch_size,
                num_image_tokens=num_image_tokens,
            )

        timesteps = self._prepare_timesteps(timestep, batch_size, image_tokens.device)
        time_emb = self.time_emb(timesteps).unsqueeze(1)

        if skip_action:
            action_features = noisy_action.new_zeros(
                batch_size,
                self.action_horizon,
                self.n_emb,
            )
            pred_action = noisy_action.new_zeros(
                batch_size,
                self.action_horizon,
                self.action_dim,
            )
        else:
            action_features = self._prepare_action_tokens(noisy_action, time_emb)
            for block, (image_kv, gaze_kv) in zip(
                self.action_blocks,
                zip(world_cache.image_key_values, world_cache.gaze_key_values),
            ):
                action_features = block(
                    action_features,
                    image_cache_key=image_kv[0],
                    image_cache_value=image_kv[1],
                    gaze_cache_key=gaze_kv[0],
                    gaze_cache_value=gaze_kv[1],
                )
            pred_action = self.action_head(self.action_ln_f(action_features))

        pred_heatmap = None
        heatmap_features = None
        if include_heatmap:
            heatmap_features = self._prepare_heatmap_tokens(noisy_heatmap, time_emb)
            for block, (image_kv, gaze_kv) in zip(
                self.heatmap_blocks,
                zip(world_cache.image_key_values, world_cache.gaze_key_values),
            ):
                heatmap_features = block(
                    heatmap_features,
                    image_cache_key=image_kv[0],
                    image_cache_value=image_kv[1],
                    gaze_cache_key=gaze_kv[0],
                    gaze_cache_value=gaze_kv[1],
                )
            pred_heatmap = self.heatmap_head(self.heatmap_ln_f(heatmap_features))

        return GazeWamTransformerOutput(
            action=pred_action,
            heatmap=pred_heatmap,
            action_features=action_features,
            heatmap_features=heatmap_features,
        )

    def attention_contract_summary(
        self,
        num_image_tokens: Optional[int] = None,
    ) -> dict:
        num_image_tokens = (
            self.max_image_tokens
            if num_image_tokens is None
            else self._validate_num_image_tokens(num_image_tokens)
        )
        train_sequence_tokens = int(
            num_image_tokens + 1 + self.action_horizon + self.heatmap_num_tokens
        )
        inference_sequence_tokens = int(num_image_tokens + 1 + self.action_horizon)
        return {
            "architecture": "cached_dual_stream",
            "use_block_attention_mask": True,
            "num_image_tokens": int(num_image_tokens),
            "gaze_token_count": 1,
            "action_horizon": int(self.action_horizon),
            "heatmap_num_tokens": int(self.heatmap_num_tokens),
            "train_sequence_tokens": train_sequence_tokens,
            "inference_sequence_tokens": inference_sequence_tokens,
            "inference_action_tokens": int(self.action_horizon),
            "shared_condition_kv_cache": False,
            "shared_world_kv_cache": True,
            "world_cache_source": "separate_image_gaze_prefill_fastwam_first_frame_analogue",
            "target_attention_mode": "independent_image_gaze_target_softmax",
            "target_attention_concatenates_image_gaze": False,
            "gaze_cross_attention_source_tokens": 1,
            "world_cache_consumed_by_action": True,
            "action_reads_heatmap_world_cache": True,
            "condition_reads_targets": False,
            "action_reads_heatmap": False,
            "action_reads_noisy_heatmap": False,
            "action_reads_heatmap_target": False,
            "heatmap_reads_action": False,
            "action_inference_drops_heatmap": True,
        }

    def get_optim_groups(self, weight_decay: float = 1e-3):
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if (
                param.ndim <= 1
                or name.endswith("bias")
                or name.endswith("pos_emb")
                or "pos_emb" in name
                or "modality_emb" in name
                or "frame_emb" in name
            ):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
    ):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
