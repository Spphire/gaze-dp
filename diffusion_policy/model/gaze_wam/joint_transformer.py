from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusion_policy.common.gaze_wam_training_config import (
    normalize_gaze_wam_bool_field,
    normalize_gaze_wam_positive_int_field,
)
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb


@dataclass
class GazeWamTransformerOutput:
    action: torch.Tensor
    heatmap: Optional[torch.Tensor]
    action_features: torch.Tensor
    heatmap_features: Optional[torch.Tensor]


def _normalize_positive_int(name: str, value) -> int:
    return normalize_gaze_wam_positive_int_field(name, value)


def _normalize_optional_positive_int(name: str, value) -> Optional[int]:
    if value is None:
        return None
    return _normalize_positive_int(name, value)


class JointGazeWamTransformer(ModuleAttrMixin):
    """Joint-sequence DiT trunk for Gaze-WAM action and heatmap denoising."""

    MODALITY_IMAGE = 0
    MODALITY_GAZE = 1
    MODALITY_ACTION = 2
    MODALITY_HEATMAP = 3

    def __init__(
        self,
        action_dim: int = 10,
        heatmap_dim: int = 1,
        action_horizon: int = 16,
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
        heatmap_num_tokens = _normalize_positive_int(
            "heatmap_num_tokens",
            heatmap_num_tokens,
        )
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
        self.n_emb = n_emb
        self.use_block_attention_mask = normalize_gaze_wam_bool_field(
            "policy.use_block_attention_mask",
            use_block_attention_mask,
            default=True,
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

        max_tokens = max_image_tokens + 1 + action_horizon + heatmap_num_tokens
        self.pos_emb = nn.Parameter(torch.zeros(1, max_tokens, n_emb))
        self.modality_emb = nn.Embedding(4, n_emb)
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.drop = nn.Dropout(p_drop_emb)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layer,
        )

        self.action_ln_f = nn.LayerNorm(n_emb)
        self.action_head = nn.Linear(n_emb, action_dim)
        self.heatmap_ln_f = nn.LayerNorm(n_emb)
        self.heatmap_head = nn.Linear(n_emb, heatmap_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerEncoder,
            nn.ModuleList,
            nn.Identity,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
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
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, JointGazeWamTransformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError(f"Unaccounted module {module}")

    def get_optim_groups(self, weight_decay: float = 1e-3):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        no_decay.add("pos_emb")
        if "_dummy_variable" in param_dict:
            no_decay.add("_dummy_variable")

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, (
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )

        return [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
    ):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def build_block_attention_mask(
        self,
        num_image_tokens: int,
        include_heatmap: bool = True,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Build an additive modality-level mask for nn.TransformerEncoder."""
        num_image_tokens = self._validate_num_image_tokens(num_image_tokens)
        include_heatmap = self._validate_include_heatmap(include_heatmap)
        modality_ids = self._build_modality_ids(
            num_image_tokens=num_image_tokens,
            include_heatmap=include_heatmap,
            device=device,
        )
        query = modality_ids[:, None]
        key = modality_ids[None, :]

        condition_q = (query == self.MODALITY_IMAGE) | (query == self.MODALITY_GAZE)
        condition_k = (key == self.MODALITY_IMAGE) | (key == self.MODALITY_GAZE)
        action_q = query == self.MODALITY_ACTION
        action_k = key == self.MODALITY_ACTION
        heatmap_q = query == self.MODALITY_HEATMAP
        heatmap_k = key == self.MODALITY_HEATMAP

        allowed = (
            (condition_q & condition_k)
            | (action_q & (condition_k | action_k))
            | (heatmap_q & (condition_k | heatmap_k))
        )
        mask = torch.zeros(
            allowed.shape,
            device=device or modality_ids.device,
            dtype=torch.float32,
        )
        return mask.masked_fill(~allowed, float("-inf"))

    def attention_contract_summary(
        self,
        num_image_tokens: Optional[int] = None,
    ) -> dict:
        """Return review-friendly block-mask invariants for training artifacts."""
        num_image_tokens = (
            self.max_image_tokens
            if num_image_tokens is None
            else self._validate_num_image_tokens(num_image_tokens)
        )
        train_tokens = num_image_tokens + 1 + self.action_horizon + self.heatmap_num_tokens
        inference_tokens = num_image_tokens + 1 + self.action_horizon
        cross_target_reads = not bool(self.use_block_attention_mask)
        return {
            "use_block_attention_mask": bool(self.use_block_attention_mask),
            "num_image_tokens": num_image_tokens,
            "gaze_token_count": 1,
            "action_horizon": int(self.action_horizon),
            "heatmap_num_tokens": int(self.heatmap_num_tokens),
            "train_sequence_tokens": int(train_tokens),
            "inference_sequence_tokens": int(inference_tokens),
            "condition_reads_targets": cross_target_reads,
            "action_reads_heatmap": cross_target_reads,
            "heatmap_reads_action": cross_target_reads,
            "action_inference_drops_heatmap": True,
        }

    def _validate_num_image_tokens(self, num_image_tokens) -> int:
        num_image_tokens = _normalize_positive_int("num_image_tokens", num_image_tokens)
        if num_image_tokens > self.max_image_tokens:
            raise ValueError(
                f"num_image_tokens={num_image_tokens} exceeds max_image_tokens="
                f"{self.max_image_tokens}."
            )
        return num_image_tokens

    @staticmethod
    def _validate_include_heatmap(include_heatmap) -> bool:
        return normalize_gaze_wam_bool_field(
            "include_heatmap",
            include_heatmap,
            default=True,
        )

    def _build_modality_ids(
        self,
        num_image_tokens: int,
        include_heatmap: bool,
        device: Optional[torch.device],
    ) -> torch.Tensor:
        include_heatmap = self._validate_include_heatmap(include_heatmap)
        ids = [
            torch.full(
                (num_image_tokens,),
                self.MODALITY_IMAGE,
                dtype=torch.long,
                device=device,
            ),
            torch.full((1,), self.MODALITY_GAZE, dtype=torch.long, device=device),
            torch.full(
                (self.action_horizon,),
                self.MODALITY_ACTION,
                dtype=torch.long,
                device=device,
            ),
        ]
        if include_heatmap:
            ids.append(
                torch.full(
                    (self.heatmap_num_tokens,),
                    self.MODALITY_HEATMAP,
                    dtype=torch.long,
                    device=device,
                )
            )
        return torch.cat(ids, dim=0)

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

    def _add_frame_embedding(
        self,
        image_tokens: torch.Tensor,
    ) -> torch.Tensor:
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

    def forward(
        self,
        image_tokens: torch.Tensor,
        gaze_token: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        noisy_heatmap: Optional[torch.Tensor] = None,
        is_inference: bool = False,
    ) -> GazeWamTransformerOutput:
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
        if noisy_action.shape != (batch_size, self.action_horizon, self.action_dim):
            raise ValueError(
                f"noisy_action must be [B, {self.action_horizon}, {self.action_dim}], "
                f"match image batch size {batch_size}, got {noisy_action.shape}."
            )
        if is_inference and noisy_heatmap is not None:
            raise ValueError(
                "Inference must omit noisy_heatmap; heatmap tokens are dropped for "
                "the action-only fast path."
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

        timesteps = self._prepare_timesteps(timestep, batch_size, image_tokens.device)
        time_emb = self.time_emb(timesteps).unsqueeze(1)

        action_tokens = self.action_proj(noisy_action) + time_emb
        image_token_part = self._add_frame_embedding(self.image_proj(image_tokens))
        token_parts = [
            image_token_part,
            self.gaze_proj(gaze_token),
            action_tokens,
        ]
        if include_heatmap:
            token_parts.append(self.heatmap_proj(noisy_heatmap) + time_emb)
        tokens = torch.cat(token_parts, dim=1)

        modality_ids = self._build_modality_ids(
            num_image_tokens=num_image_tokens,
            include_heatmap=include_heatmap,
            device=tokens.device,
        )
        tokens = tokens + self.modality_emb(modality_ids).unsqueeze(0)
        tokens = tokens + self.pos_emb[:, : tokens.shape[1], :]
        tokens = self.drop(tokens)

        attention_mask = None
        if self.use_block_attention_mask:
            attention_mask = self.build_block_attention_mask(
                num_image_tokens=num_image_tokens,
                include_heatmap=include_heatmap,
                device=tokens.device,
            ).to(dtype=tokens.dtype)

        encoded = self.encoder(tokens, mask=attention_mask)

        action_start = num_image_tokens + 1
        action_end = action_start + self.action_horizon
        action_features = encoded[:, action_start:action_end, :]
        if action_features.shape[1] != self.action_horizon:
            raise RuntimeError(
                "Action feature slice length mismatch: expected "
                f"{self.action_horizon}, got {action_features.shape[1]}."
            )
        pred_action = self.action_head(self.action_ln_f(action_features))

        pred_heatmap = None
        heatmap_features = None
        if include_heatmap:
            heatmap_features = encoded[:, action_end:, :]
            if heatmap_features.shape[1] != self.heatmap_num_tokens:
                raise RuntimeError(
                    "Heatmap feature slice length mismatch: expected "
                    f"{self.heatmap_num_tokens}, got {heatmap_features.shape[1]}."
                )
            pred_heatmap = self.heatmap_head(self.heatmap_ln_f(heatmap_features))

        return GazeWamTransformerOutput(
            action=pred_action,
            heatmap=pred_heatmap,
            action_features=action_features,
            heatmap_features=heatmap_features,
        )
