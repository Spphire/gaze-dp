import inspect
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from diffusion_policy.common.action_utils import relative_actions_to_absolute_actions
from diffusion_policy.common.gaze_wam_training_config import (
    gaze_wam_required_loss_routing_validation_flags,
    normalize_gaze_wam_bool_field,
    normalize_gaze_wam_nonnegative_float_field,
    normalize_gaze_wam_positive_float_field,
    normalize_gaze_wam_positive_int_field,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.gaze_wam.cached_dual_stream_transformer import (
    CachedDualStreamGazeWamTransformer,
)
from diffusion_policy.model.gaze_wam.gaze_encoder import GazeConditionEncoder
from diffusion_policy.model.gaze_wam.heatmap_codec import HeatmapTokenCodec
from diffusion_policy.model.gaze_wam.heatmap_decoder import CosmosHeatmapCodec
from diffusion_policy.model.gaze_wam.loss import (
    distributed_mask_count,
    distributed_masked_mean,
    per_sample_dsnt_xy_loss,
    per_sample_spatial_point_nll_loss,
    per_sample_spatial_js_loss,
    spatial_distribution_2d,
)
from diffusion_policy.model.gaze_wam.metrics import gaze_dependency_ratio
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class GazeWamPolicy(BaseImagePolicy):
    """Gaze-conditioned joint action/heatmap diffusion policy."""

    NON_MODEL_OBS_KEYS = frozenset(
        {
            "action",
            "action_abs",
            "action_base_abs",
            "gaze_xy",
            "has_action",
            "has_action_abs",
            "has_action_base_abs",
            "has_gaze_label",
            "has_heatmap",
            "has_heatmap_image",
            "heatmap",
            "heatmap_image",
            "is_gaze_condition_dropped",
            "is_open",
            "use_gaze_condition",
            "valid_mask",
        }
    )

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder,
        model: Optional[torch.nn.Module] = None,
        gaze_encoder: Optional[GazeConditionEncoder] = None,
        num_inference_steps: Optional[int] = None,
        input_pertub: float = 0.1,
        action_loss_weight: float = 1.0,
        heatmap_loss_weight: float = 1.0,
        heatmap_token_kl_loss_weight: float = 0.0,
        heatmap_objective: str = "diffusion",
        heatmap_xy_loss_weight: float = 1.0,
        heatmap_point_nll_loss_weight: float = 0.0,
        heatmap_js_loss_weight: float = 1.0,
        heatmap_diffusion_final_loss_enabled: bool = False,
        heatmap_final_loss_timestep_weighting: str = "none",
        heatmap_dsnt_temperature: float = 0.1,
        heatmap_distribution_mode: str = "intensity_softplus",
        heatmap_dsnt_target_sigma_px: float = 6.0,
        cfg_scale: float = 1.0,
        # arch defaults used when model is not injected
        model_architecture: str = "cached_dual_stream",
        n_layer: int = 7,
        n_head: int = 8,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        max_image_tokens: int = 512,
        heatmap_num_tokens: int = 256,
        heatmap_dim: int = 1,
        heatmap_token_grid: Optional[Sequence[int]] = None,
        heatmap_image_size: Sequence[int] = (256, 256),
        heatmap_sigma_tokens: float = 1.25,
        heatmap_decode_method: str = "gaussian_splat",
        heatmap_spatial_decoder: str = "cosmos_tokenizer",
        heatmap_cosmos_encoder_path: str = "",
        heatmap_cosmos_decoder_path: str = "",
        heatmap_cosmos_input_range: str = "minus_one_one",
        heatmap_cosmos_output_range: str = "minus_one_one",
        heatmap_cosmos_input_normalization: str = "max",
        heatmap_latent_scale: float = 1.0,
        heatmap_latent_offset: float = 0.0,
        heatmap_latent_stats_path: str = "",
        heatmap_scheduler_clip_sample: Optional[bool] = None,
        use_block_attention_mask: bool = True,
        use_frame_embedding: bool = False,
        image_tokens_per_frame: Optional[int] = None,
        max_obs_frames: Optional[int] = None,
        # parameters passed to scheduler.step
        **kwargs,
    ) -> None:
        super().__init__()

        action_shape = shape_meta["action"]["shape"]
        if len(action_shape) != 1:
            raise ValueError(f"Expected 1D action shape, got {action_shape}.")
        action_dim = self._validate_positive_int(
            "shape_meta['action']['shape'][0]",
            action_shape[0],
        )
        action_horizon = self._validate_positive_int(
            "shape_meta['action']['horizon']",
            shape_meta["action"]["horizon"],
        )
        n_layer = self._validate_positive_int("n_layer", n_layer)
        n_head = self._validate_positive_int("n_head", n_head)
        n_emb = self._validate_positive_int("n_emb", n_emb)
        max_image_tokens = self._validate_positive_int(
            "max_image_tokens",
            max_image_tokens,
        )
        heatmap_num_tokens = self._validate_positive_int(
            "heatmap_num_tokens",
            heatmap_num_tokens,
        )
        heatmap_dim = self._validate_positive_int("heatmap_dim", heatmap_dim)

        obs_shape = tuple(
            self._validate_positive_int(f"obs_encoder.output_shape()[{idx}]", value)
            for idx, value in enumerate(obs_encoder.output_shape())
        )
        if len(obs_shape) != 3:
            raise ValueError(
                "obs_encoder.output_shape() must be [B, N_v, D], "
                f"got {obs_shape}."
            )
        obs_num_tokens = int(obs_shape[-2])
        obs_embed_dim = int(obs_shape[-1])
        if obs_embed_dim != n_emb:
            raise ValueError(
                "obs_encoder output dim must match policy n_emb, got "
                f"{obs_embed_dim} vs {n_emb}."
            )
        use_block_attention_mask = normalize_gaze_wam_bool_field(
            "policy.use_block_attention_mask",
            use_block_attention_mask,
            default=True,
        )
        use_frame_embedding = normalize_gaze_wam_bool_field(
            "policy.use_frame_embedding",
            use_frame_embedding,
            default=False,
        )

        model_architecture = str(model_architecture)
        if model_architecture != "cached_dual_stream":
            raise ValueError(
                "The active Gaze-WAM policy-training path only supports "
                "model_architecture='cached_dual_stream'."
            )

        if model is None:
            model = CachedDualStreamGazeWamTransformer(
                action_dim=action_dim,
                heatmap_dim=heatmap_dim,
                action_horizon=action_horizon,
                heatmap_num_tokens=heatmap_num_tokens,
                max_image_tokens=max(max_image_tokens, obs_shape[-2]),
                n_layer=n_layer,
                n_head=n_head,
                n_emb=n_emb,
                p_drop_emb=p_drop_emb,
                p_drop_attn=p_drop_attn,
                use_block_attention_mask=use_block_attention_mask,
                use_frame_embedding=use_frame_embedding,
                image_tokens_per_frame=image_tokens_per_frame,
                max_obs_frames=max_obs_frames,
            )
        if gaze_encoder is None:
            gaze_encoder = GazeConditionEncoder(embed_dim=n_emb)
        self._validate_component_contract(
            model=model,
            gaze_encoder=gaze_encoder,
            action_dim=action_dim,
            action_horizon=action_horizon,
            heatmap_num_tokens=heatmap_num_tokens,
            heatmap_dim=heatmap_dim,
            n_emb=n_emb,
            obs_num_tokens=obs_num_tokens,
        )

        self.obs_encoder = obs_encoder
        self.gaze_encoder = gaze_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.heatmap_num_tokens = heatmap_num_tokens
        self.heatmap_dim = heatmap_dim
        if heatmap_token_grid is None:
            side = int(np.sqrt(heatmap_num_tokens))
            heatmap_token_grid = (side, side) if side * side == heatmap_num_tokens else (1, heatmap_num_tokens)
        self.heatmap_codec = HeatmapTokenCodec(
            token_grid=heatmap_token_grid,
            image_size=heatmap_image_size,
            sigma_tokens=heatmap_sigma_tokens,
        )
        heatmap_height, heatmap_width = self.heatmap_codec.image_size
        heatmap_y = (torch.arange(heatmap_height, dtype=torch.float32) + 0.5) / float(
            heatmap_height
        )
        heatmap_x = (torch.arange(heatmap_width, dtype=torch.float32) + 0.5) / float(
            heatmap_width
        )
        heatmap_yy, heatmap_xx = torch.meshgrid(heatmap_y, heatmap_x, indexing="ij")
        self.register_buffer(
            "_heatmap_target_x_grid",
            heatmap_xx.reshape(1, heatmap_height, heatmap_width),
            persistent=False,
        )
        self.register_buffer(
            "_heatmap_target_y_grid",
            heatmap_yy.reshape(1, heatmap_height, heatmap_width),
            persistent=False,
        )
        if self.heatmap_codec.num_tokens != heatmap_num_tokens:
            raise ValueError(
                "heatmap_token_grid product must match heatmap_num_tokens, got "
                f"{self.heatmap_codec.token_grid} vs {heatmap_num_tokens}."
            )
        self.heatmap_decode_method = heatmap_decode_method
        self.heatmap_spatial_decoder = str(heatmap_spatial_decoder)
        if self.heatmap_spatial_decoder != "cosmos_tokenizer":
            raise ValueError(
                "The active heatmap ED path only supports "
                "policy.heatmap_spatial_decoder='cosmos_tokenizer'. "
                "The project-local learned/autoencoder ED paths have been removed "
                "from the training contract."
            )
        self.heatmap_cosmos_encoder_path = str(heatmap_cosmos_encoder_path or "")
        self.heatmap_cosmos_decoder_path = str(heatmap_cosmos_decoder_path or "")
        self.heatmap_image_decoder = CosmosHeatmapCodec(
            encoder_path=self.heatmap_cosmos_encoder_path,
            decoder_path=self.heatmap_cosmos_decoder_path,
            token_grid=self.heatmap_codec.token_grid,
            image_size=self.heatmap_codec.image_size,
            latent_channels=heatmap_dim,
            input_range=heatmap_cosmos_input_range,
            output_range=heatmap_cosmos_output_range,
            input_normalization=heatmap_cosmos_input_normalization,
        )
        self.heatmap_image_decoder.eval()
        self.heatmap_image_decoder.requires_grad_(False)
        self.heatmap_latent_scale = self._validate_positive_float(
            "heatmap_latent_scale",
            heatmap_latent_scale,
        )
        self.heatmap_latent_offset = self._validate_finite_float(
            "heatmap_latent_offset",
            heatmap_latent_offset,
        )
        self.heatmap_latent_stats_path = str(heatmap_latent_stats_path or "")
        self.heatmap_scheduler_clip_sample = None
        if heatmap_scheduler_clip_sample is not None:
            self.heatmap_scheduler_clip_sample = normalize_gaze_wam_bool_field(
                "heatmap_scheduler_clip_sample",
                heatmap_scheduler_clip_sample,
                default=False,
            )
        self.input_pertub = self._validate_nonnegative_float("input_pertub", input_pertub)
        self.action_loss_weight = self._validate_nonnegative_float(
            "action_loss_weight",
            action_loss_weight,
        )
        self.heatmap_loss_weight = self._validate_nonnegative_float(
            "heatmap_loss_weight",
            heatmap_loss_weight,
        )
        self.heatmap_token_kl_loss_weight = self._validate_nonnegative_float(
            "heatmap_token_kl_loss_weight",
            heatmap_token_kl_loss_weight,
        )
        self.heatmap_xy_loss_weight = self._validate_nonnegative_float(
            "heatmap_xy_loss_weight",
            heatmap_xy_loss_weight,
        )
        self.heatmap_point_nll_loss_weight = self._validate_nonnegative_float(
            "heatmap_point_nll_loss_weight",
            heatmap_point_nll_loss_weight,
        )
        self.heatmap_js_loss_weight = self._validate_nonnegative_float(
            "heatmap_js_loss_weight",
            heatmap_js_loss_weight,
        )
        self.heatmap_diffusion_final_loss_enabled = normalize_gaze_wam_bool_field(
            "policy.heatmap_diffusion_final_loss_enabled",
            heatmap_diffusion_final_loss_enabled,
            default=False,
        )
        self.heatmap_final_loss_timestep_weighting = str(
            heatmap_final_loss_timestep_weighting
        )
        if self.heatmap_final_loss_timestep_weighting not in ("none", "alpha_cumprod"):
            raise ValueError(
                "heatmap_final_loss_timestep_weighting must be one of: "
                "none, alpha_cumprod."
            )
        self.heatmap_dsnt_temperature = self._validate_positive_float(
            "heatmap_dsnt_temperature",
            heatmap_dsnt_temperature,
        )
        self.heatmap_distribution_mode = str(heatmap_distribution_mode)
        if self.heatmap_distribution_mode not in (
            "logits_softmax",
            "intensity_clamp",
            "intensity_softplus",
        ):
            raise ValueError(
                "heatmap_distribution_mode must be one of: logits_softmax, "
                f"intensity_clamp, intensity_softplus; got {self.heatmap_distribution_mode!r}."
            )
        self.heatmap_dsnt_target_sigma_px = self._validate_positive_float(
            "heatmap_dsnt_target_sigma_px",
            heatmap_dsnt_target_sigma_px,
        )
        if heatmap_objective == "sample":
            heatmap_objective = "clean_token"
        if heatmap_objective not in ("diffusion", "clean_token", "dsnt_js"):
            raise ValueError(
                "heatmap_objective must be one of: diffusion, clean_token, dsnt_js."
            )
        self.heatmap_objective = heatmap_objective
        if (
            self.heatmap_diffusion_final_loss_enabled
            and self.heatmap_objective != "diffusion"
        ):
            raise ValueError(
                "policy.heatmap_diffusion_final_loss_enabled requires "
                "policy.heatmap_objective='diffusion'."
            )
        self.cfg_scale = self._validate_nonnegative_float("cfg_scale", cfg_scale)
        self.kwargs = self._sanitize_scheduler_step_kwargs(noise_scheduler, kwargs)

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = self._validate_positive_int(
            "num_inference_steps",
            num_inference_steps,
        )
        if self.num_inference_steps < 2:
            raise ValueError(
                "num_inference_steps must be at least 2; single-step denoising "
                "is not part of the Gaze-WAM inference contract."
            )

    @staticmethod
    def _sanitize_scheduler_step_kwargs(
        scheduler: DDPMScheduler,
        kwargs: Dict[str, object],
    ) -> Dict[str, object]:
        """Keep only kwargs accepted by the configured scheduler.step.

        Older checkpoint configs may contain now-removed policy knobs. Hydra
        passes unknown constructor keys through ``**kwargs``; without filtering
        they would be forwarded to DDIM/DDPM ``step`` and break checkpoint
        reload previews.
        """
        if not kwargs:
            return {}
        try:
            parameters = inspect.signature(scheduler.step).parameters
        except (TypeError, ValueError):
            return dict(kwargs)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
            return dict(kwargs)
        reserved = {"model_output", "timestep", "sample", "generator"}
        return {
            key: value
            for key, value in kwargs.items()
            if key in parameters and key not in reserved
        }

    def _heatmap_sampling_scheduler(self):
        """Return the scheduler used only for heatmap latent sampling.

        Action inference keeps ``self.noise_scheduler`` untouched. Heatmap
        latents may use a modality-specific clip setting after the Cosmos
        affine latent scale maps raw tokenizer latents into scheduler space.
        """
        if self.heatmap_scheduler_clip_sample is None:
            return self.noise_scheduler
        scheduler_cls = type(self.noise_scheduler)
        if not hasattr(scheduler_cls, "from_config"):
            raise RuntimeError(
                "Configured noise scheduler does not support from_config, so "
                "policy.heatmap_scheduler_clip_sample cannot be applied only "
                "to heatmap sampling."
            )
        config = dict(self.noise_scheduler.config)
        config["clip_sample"] = bool(self.heatmap_scheduler_clip_sample)
        return scheduler_cls.from_config(config)

    @staticmethod
    def _validate_nonnegative_float(name: str, value) -> float:
        return normalize_gaze_wam_nonnegative_float_field(name, value)

    @staticmethod
    def _validate_positive_float(name: str, value) -> float:
        return normalize_gaze_wam_positive_float_field(name, value)

    @staticmethod
    def _validate_finite_float(name: str, value) -> float:
        parsed = float(value)
        if not np.isfinite(parsed):
            raise ValueError(f"{name} must be finite, got {value!r}.")
        return parsed

    @staticmethod
    def _validate_positive_int(name: str, value) -> int:
        return normalize_gaze_wam_positive_int_field(name, value)

    @staticmethod
    def _validate_equal(name: str, actual, expected) -> None:
        if actual is None:
            return
        actual = GazeWamPolicy._validate_positive_int(name, actual)
        expected = GazeWamPolicy._validate_positive_int(f"{name} expected", expected)
        if actual != expected:
            raise ValueError(f"{name} must be {expected}, got {actual}.")

    @classmethod
    def _validate_component_contract(
        cls,
        model: torch.nn.Module,
        gaze_encoder: GazeConditionEncoder,
        action_dim: int,
        action_horizon: int,
        heatmap_num_tokens: int,
        heatmap_dim: int,
        n_emb: int,
        obs_num_tokens: int,
    ) -> None:
        """Fail early when Hydra-injected components drift from the Gaze-WAM IO contract."""
        model_checks = {
            "model.action_dim": action_dim,
            "model.action_horizon": action_horizon,
            "model.heatmap_num_tokens": heatmap_num_tokens,
            "model.heatmap_dim": heatmap_dim,
            "model.n_emb": n_emb,
        }
        for attr_name, expected in model_checks.items():
            attr = attr_name.split(".", 1)[1]
            cls._validate_equal(attr_name, getattr(model, attr, None), expected)

        max_image_tokens = getattr(model, "max_image_tokens", None)
        if max_image_tokens is not None and int(max_image_tokens) < int(obs_num_tokens):
            raise ValueError(
                "model.max_image_tokens must cover obs_encoder image tokens, got "
                f"{max_image_tokens} < {obs_num_tokens}."
            )

        gaze_dim = None
        mask_token = getattr(gaze_encoder, "mask_token", None)
        if mask_token is not None:
            gaze_dim = int(mask_token.shape[-1])
        elif hasattr(getattr(gaze_encoder, "proj", None), "out_features"):
            gaze_dim = int(gaze_encoder.proj.out_features)
        if gaze_dim is not None and gaze_dim != int(n_emb):
            raise ValueError(
                "gaze_encoder output dim must match policy n_emb, got "
                f"{gaze_dim} vs {n_emb}."
            )

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.normalizer.to(device=self.device)

    def get_optimizer(
        self,
        lr: float,
        weight_decay: float,
        obs_encoder_lr: float,
        obs_encoder_weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        # PyTorch optimizers tolerate empty parameter groups, but DeepSpeed
        # ZeRO does not: it attempts to flatten every group and torch.cat([])
        # fails. Keep the optimizer contract identical while omitting empty
        # groups before Accelerate prepares the DeepSpeed engine.
        optim_groups = [
            group
            for group in self.model.get_optim_groups(weight_decay=weight_decay)
            if len(group.get("params", ())) > 0
        ]

        gaze_params = [
            param for param in self.gaze_encoder.parameters() if param.requires_grad
        ]
        if gaze_params:
            optim_groups.append(
                {
                    "params": gaze_params,
                    "weight_decay": weight_decay,
                }
            )

        if self.heatmap_image_decoder is not None:
            heatmap_params = [
                param
                for param in self.heatmap_image_decoder.parameters()
                if param.requires_grad
            ]
            if heatmap_params:
                optim_groups.append(
                    {
                        "params": heatmap_params,
                        "weight_decay": weight_decay,
                    }
                )

        backbone_params = list()
        other_obs_params = list()
        for key, value in self.obs_encoder.named_parameters():
            if not value.requires_grad:
                continue
            if key.startswith("key_model_map"):
                backbone_params.append(value)
            else:
                other_obs_params.append(value)
        if len(backbone_params) == 0 and (
            obs_encoder_lr != lr or obs_encoder_weight_decay != weight_decay
        ):
            raise ValueError(
                "obs_encoder_lr/obs_encoder_weight_decay were configured separately, "
                "but obs_encoder.named_parameters() did not expose any parameters "
                "under the 'key_model_map' prefix. Check the visual encoder module "
                "or set obs_encoder optimizer hyperparameters equal to the policy "
                "defaults."
            )
        if backbone_params:
            optim_groups.append(
                {
                    "params": backbone_params,
                    "weight_decay": obs_encoder_weight_decay,
                    "lr": obs_encoder_lr,
                }
            )
        if other_obs_params:
            optim_groups.append(
                {
                    "params": other_obs_params,
                    "weight_decay": obs_encoder_weight_decay,
                }
            )
        if not optim_groups:
            raise ValueError("Gaze-WAM optimizer has no trainable parameters.")
        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas)

    def _encode_conditions(
        self,
        obs_dict: Dict[str, torch.Tensor],
        gaze_xy: torch.Tensor,
        use_gaze_condition: Optional[torch.Tensor] = None,
        has_gaze_label: Optional[torch.Tensor] = None,
    ):
        obs_tokens = self.obs_encoder(obs_dict)
        gaze_token = self.gaze_encoder(
            gaze_xy=gaze_xy,
            use_gaze_condition=use_gaze_condition,
            has_gaze_label=has_gaze_label,
        )
        return obs_tokens, gaze_token

    @classmethod
    def _model_obs_from_obs_dict(
        cls,
        obs_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        model_obs = {
            key: value
            for key, value in obs_dict.items()
            if key not in cls.NON_MODEL_OBS_KEYS
        }
        if len(model_obs) == 0:
            raise ValueError(
                "obs_dict must contain at least one model observation tensor after "
                "filtering Gaze-WAM labels, masks, and metadata."
            )
        return model_obs

    def _scheduler_target(self, clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            return noise
        if pred_type == "sample":
            return clean
        raise ValueError(f"Unsupported prediction type {pred_type}")

    def _heatmap_target(self, clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if self.heatmap_objective in ("diffusion", "dsnt_js"):
            return self._scheduler_target(clean, noise)
        if self.heatmap_objective == "clean_token":
            return clean
        raise ValueError(f"Unsupported heatmap_objective {self.heatmap_objective}")

    def _heatmap_clean_prediction(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if self.heatmap_objective in ("diffusion", "dsnt_js"):
            return self._estimate_clean_sample(
                sample=sample,
                model_output=model_output,
                timestep=timestep,
            )
        if self.heatmap_objective == "clean_token":
            return model_output
        raise ValueError(f"Unsupported heatmap_objective {self.heatmap_objective}")

    @staticmethod
    def _per_sample_heatmap_token_kl(
        pred_clean: torch.Tensor,
        target_clean: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        pred = pred_clean.float().flatten(start_dim=1)
        target = target_clean.float().clamp_min(0.0)
        target = target.flatten(start_dim=1)
        pred_log_dist = F.log_softmax(pred, dim=-1)
        target_dist = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
        kl = target_dist * (
            (target_dist + eps).log()
            - pred_log_dist
        )
        return kl.sum(dim=-1)

    def _heatmap_tokens_to_spatial_image(
        self,
        tokens: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if not torch.is_tensor(tokens):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tokens).__name__}.")
        if not torch.is_floating_point(tokens):
            raise ValueError(f"{name} must be a floating point tensor, got {tokens.dtype}.")
        if tokens.ndim != 3:
            raise ValueError(
                f"{name} must have shape [B, N_h, D_h], got {tuple(tokens.shape)}."
            )
        if tokens.shape[1] != self.heatmap_num_tokens:
            raise ValueError(
                f"{name} token count must be {self.heatmap_num_tokens}, "
                f"got {tokens.shape[1]}."
            )
        if not torch.all(torch.isfinite(tokens)):
            raise ValueError(f"{name} must contain only finite values.")
        cosmos_tokens = self._denormalize_heatmap_latent_tokens(tokens)
        return self.heatmap_image_decoder.decode_tokens(cosmos_tokens)

    def _normalize_heatmap_latent_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens - float(self.heatmap_latent_offset)) * float(
            self.heatmap_latent_scale
        )

    def _denormalize_heatmap_latent_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens / float(self.heatmap_latent_scale) + float(
            self.heatmap_latent_offset
        )

    def _target_heatmap_image_from_xy(
        self,
        gaze_xy: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build the DSNT regularizer target distribution from normalized gaze xy."""
        if gaze_xy.ndim != 2 or gaze_xy.shape[-1] != 2:
            raise ValueError(f"gaze_xy must have shape [B, 2], got {tuple(gaze_xy.shape)}.")
        self._require_bool_vector("valid_mask", valid_mask, int(gaze_xy.shape[0]))
        height, width = self.heatmap_codec.image_size
        xx = self._heatmap_target_x_grid.to(device=gaze_xy.device, dtype=gaze_xy.dtype)
        yy = self._heatmap_target_y_grid.to(device=gaze_xy.device, dtype=gaze_xy.dtype)
        sigma_x = float(self.heatmap_dsnt_target_sigma_px) / float(width)
        sigma_y = float(self.heatmap_dsnt_target_sigma_px) / float(height)
        gx = gaze_xy[:, 0].reshape(-1, 1, 1)
        gy = gaze_xy[:, 1].reshape(-1, 1, 1)
        dist = ((xx - gx) / sigma_x) ** 2
        dist = dist + ((yy - gy) / sigma_y) ** 2
        target = torch.exp(-0.5 * dist).to(dtype=gaze_xy.dtype)
        target = target * valid_mask.to(device=gaze_xy.device, dtype=gaze_xy.dtype).reshape(-1, 1, 1)
        target = target / target.flatten(start_dim=1).sum(dim=-1, keepdim=True).reshape(-1, 1, 1).clamp_min(1e-12)
        return target

    def _target_heatmap_image_from_batch_or_xy(
        self,
        batch: Dict[str, torch.Tensor],
        gaze_xy: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Use dense heatmap labels when present, otherwise fall back to gaze xy."""
        target = self._target_heatmap_image_from_xy(
            gaze_xy=gaze_xy,
            valid_mask=valid_mask,
        )
        if "heatmap_image" not in batch:
            return target

        heatmap_image = batch["heatmap_image"]
        if not torch.is_tensor(heatmap_image):
            raise TypeError(
                "batch['heatmap_image'] must be a torch.Tensor, "
                f"got {type(heatmap_image).__name__}."
            )
        if not torch.is_floating_point(heatmap_image):
            raise ValueError(
                "batch['heatmap_image'] must be floating point, "
                f"got {heatmap_image.dtype}."
            )
        if heatmap_image.ndim == 4:
            if heatmap_image.shape[1] != 1:
                raise ValueError(
                    "batch['heatmap_image'] must have one channel when rank-4, "
                    f"got {tuple(heatmap_image.shape)}."
                )
            heatmap_image = heatmap_image[:, 0]
        elif heatmap_image.ndim != 3:
            raise ValueError(
                "batch['heatmap_image'] must have shape [B,H,W] or [B,1,H,W], "
                f"got {tuple(heatmap_image.shape)}."
            )
        if heatmap_image.shape[0] != gaze_xy.shape[0]:
            raise ValueError(
                "batch['heatmap_image'] batch dimension must match gaze_xy, "
                f"got {heatmap_image.shape[0]} and {gaze_xy.shape[0]}."
            )
        if tuple(heatmap_image.shape[-2:]) != tuple(self.heatmap_codec.image_size):
            raise ValueError(
                "batch['heatmap_image'] spatial shape must match "
                f"{tuple(self.heatmap_codec.image_size)}, got "
                f"{tuple(heatmap_image.shape[-2:])}."
            )
        heatmap_image = heatmap_image.to(device=gaze_xy.device, dtype=gaze_xy.dtype)
        if not torch.all(torch.isfinite(heatmap_image)):
            raise ValueError("batch['heatmap_image'] must contain only finite values.")
        if torch.any(heatmap_image < 0.0):
            raise ValueError("batch['heatmap_image'] must be non-negative.")

        if "has_heatmap_image" in batch:
            has_heatmap_image = batch["has_heatmap_image"]
            if not torch.is_tensor(has_heatmap_image):
                raise TypeError(
                    "batch['has_heatmap_image'] must be a torch.Tensor, "
                    f"got {type(has_heatmap_image).__name__}."
                )
            self._require_bool_vector(
                "batch['has_heatmap_image']",
                has_heatmap_image,
                int(gaze_xy.shape[0]),
            )
            has_heatmap_image = has_heatmap_image.to(device=gaze_xy.device)
        else:
            has_heatmap_image = torch.ones(
                int(gaze_xy.shape[0]),
                device=gaze_xy.device,
                dtype=torch.bool,
            )
        use_dense_target = valid_mask.to(device=gaze_xy.device) & has_heatmap_image
        if torch.any(use_dense_target):
            target = target.clone()
            target[use_dense_target] = heatmap_image[use_dense_target]
        return target

    def _heatmap_image_to_training_tokens(
        self,
        heatmap_image: torch.Tensor,
    ) -> torch.Tensor:
        if heatmap_image.ndim != 3:
            raise ValueError(
                "heatmap_image must have shape [B, H, W], "
                f"got {tuple(heatmap_image.shape)}."
            )
        with torch.no_grad():
            tokens = self.heatmap_image_decoder.encode_image(heatmap_image)
        return self._normalize_heatmap_latent_tokens(tokens)

    def _should_generate_cosmos_latent_target(self) -> bool:
        return self.heatmap_spatial_decoder == "cosmos_tokenizer"

    def _prepare_timesteps(
        self,
        timestep,
        batch_size: int,
        device: torch.device,
        name: str = "timestep",
    ) -> torch.Tensor:
        if timestep is None:
            timestep = torch.zeros(batch_size, device=device, dtype=torch.long)
        elif not torch.is_tensor(timestep):
            timestep = torch.as_tensor(timestep, device=device)
        else:
            timestep = timestep.to(device=device)
        if timestep.ndim == 0:
            timestep = timestep[None]
        elif timestep.ndim != 1:
            raise ValueError(
                f"{name} must be a scalar or 1D tensor/list, got shape "
                f"{tuple(timestep.shape)}."
            )
        if timestep.dtype == torch.bool or torch.is_complex(timestep):
            raise ValueError(f"{name} must contain integer diffusion timesteps.")
        if torch.is_floating_point(timestep):
            if not torch.isfinite(timestep).all():
                raise ValueError(f"{name} must contain only finite diffusion timesteps.")
            if not torch.all(timestep == timestep.round()):
                raise ValueError(f"{name} must contain integer diffusion timesteps.")
        timestep = timestep.reshape(-1)
        if timestep.numel() not in (1, batch_size):
            raise ValueError(
                f"{name} must contain either 1 or B={batch_size} diffusion timesteps, "
                f"got {timestep.numel()}."
            )
        timestep = timestep.expand(batch_size).long()
        num_train_timesteps = self._validate_positive_int(
            "noise_scheduler.config.num_train_timesteps",
            self.noise_scheduler.config.num_train_timesteps,
        )
        invalid_range = (timestep < 0) | (timestep >= num_train_timesteps)
        if torch.any(invalid_range):
            min_value = int(timestep.detach().amin().item())
            max_value = int(timestep.detach().amax().item())
            raise ValueError(
                f"{name} must be in [0, {num_train_timesteps - 1}], "
                f"got min={min_value}, max={max_value}."
            )
        return timestep

    def _estimate_clean_sample(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "sample":
            return model_output
        if pred_type != "epsilon":
            raise ValueError(f"Unsupported prediction type {pred_type}")
        if not hasattr(self.noise_scheduler, "alphas_cumprod"):
            raise RuntimeError(
                "Heatmap clean-sample reconstruction for epsilon prediction requires "
                "a scheduler with alphas_cumprod."
            )

        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=sample.device,
            dtype=sample.dtype,
        )
        timestep = timestep.to(device=sample.device, dtype=torch.long)
        alpha_prod_t = alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        broadcast_shape = (sample.shape[0],) + (1,) * (sample.ndim - 1)
        alpha_prod_t = alpha_prod_t.reshape(broadcast_shape)
        beta_prod_t = beta_prod_t.reshape(broadcast_shape)
        return (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt().clamp_min(1e-12)

    def _per_sample_final_loss_weight(
        self,
        timestep: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.heatmap_final_loss_timestep_weighting == "none":
            return torch.ones(batch_size, device=device, dtype=dtype)
        if self.heatmap_final_loss_timestep_weighting != "alpha_cumprod":
            raise ValueError(
                "Unsupported heatmap_final_loss_timestep_weighting "
                f"{self.heatmap_final_loss_timestep_weighting!r}."
            )
        if not hasattr(self.noise_scheduler, "alphas_cumprod"):
            raise RuntimeError(
                "heatmap_final_loss_timestep_weighting='alpha_cumprod' requires "
                "a scheduler with alphas_cumprod."
            )
        timestep = timestep.to(device=device, dtype=torch.long).reshape(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(batch_size)
        if timestep.numel() != batch_size:
            raise ValueError(
                "timestep must contain either 1 or batch_size entries for final loss "
                f"weighting, got {timestep.numel()} vs {batch_size}."
            )
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=device,
            dtype=dtype,
        )
        return alphas_cumprod[timestep].clamp_min(0.0)

    def conditional_sample(
        self,
        image_tokens: torch.Tensor,
        gaze_token: torch.Tensor,
        mask_gaze_token: Optional[torch.Tensor] = None,
        cfg_scale: Optional[float] = None,
        generator=None,
        **kwargs,
    ) -> torch.Tensor:
        scheduler = self.noise_scheduler
        action = torch.randn(
            size=(image_tokens.shape[0], self.action_horizon, self.action_dim),
            dtype=image_tokens.dtype,
            device=image_tokens.device,
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)
        if cfg_scale is None:
            cfg_scale = self.cfg_scale
        else:
            cfg_scale = self._validate_nonnegative_float("cfg_scale", cfg_scale)
        cache_kwarg = None
        world_cache = None
        mask_world_cache = None
        if hasattr(self.model, "prefill_world_cache"):
            cache_kwarg = "world_cache"
            world_cache = self.model.prefill_world_cache(
                image_tokens=image_tokens,
                gaze_token=gaze_token,
            )
            if mask_gaze_token is not None and cfg_scale != 1.0:
                mask_world_cache = self.model.prefill_world_cache(
                    image_tokens=image_tokens,
                    gaze_token=mask_gaze_token,
                )
        elif hasattr(self.model, "prefill_condition_cache"):
            cache_kwarg = "condition_cache"
            world_cache = self.model.prefill_condition_cache(
                image_tokens=image_tokens,
                gaze_token=gaze_token,
            )
            if mask_gaze_token is not None and cfg_scale != 1.0:
                mask_world_cache = self.model.prefill_condition_cache(
                    image_tokens=image_tokens,
                    gaze_token=mask_gaze_token,
                )
        for t in scheduler.timesteps:
            model_kwargs = {}
            if world_cache is not None:
                model_kwargs[cache_kwarg] = world_cache
            model_output = self.model(
                image_tokens=image_tokens,
                gaze_token=gaze_token,
                noisy_action=action,
                noisy_heatmap=None,
                timestep=t,
                is_inference=True,
                **model_kwargs,
            ).action
            if mask_gaze_token is not None and cfg_scale != 1.0:
                masked_model_kwargs = {}
                if mask_world_cache is not None:
                    masked_model_kwargs[cache_kwarg] = mask_world_cache
                masked_output = self.model(
                    image_tokens=image_tokens,
                    gaze_token=mask_gaze_token,
                    noisy_action=action,
                    noisy_heatmap=None,
                    timestep=t,
                    is_inference=True,
                    **masked_model_kwargs,
                ).action
                model_output = masked_output + cfg_scale * (model_output - masked_output)
            action = scheduler.step(
                model_output,
                t,
                action,
                generator=generator,
                **kwargs,
            ).prev_sample
        return action

    def conditional_heatmap_sample(
        self,
        image_tokens: torch.Tensor,
        gaze_token: torch.Tensor,
        noisy_action: Optional[torch.Tensor] = None,
        generator=None,
    ) -> torch.Tensor:
        """Iteratively denoise heatmap latent tokens for preview/paper figures.

        Main robot inference deliberately omits heatmap target tokens. This path is
        only for evaluating the heatmap branch: start from latent noise, reuse the
        same cached image/gaze world K/V, skip the action decoder, and decode the
        final clean heatmap latent to a full-resolution heatmap image.
        """
        batch_size = int(image_tokens.shape[0])
        if noisy_action is None:
            noisy_action = torch.zeros(
                batch_size,
                self.action_horizon,
                self.action_dim,
                dtype=image_tokens.dtype,
                device=image_tokens.device,
            )
        else:
            noisy_action = noisy_action.to(device=image_tokens.device, dtype=image_tokens.dtype)
        self._validate_noisy_action(
            noisy_action,
            batch_size=batch_size,
            name="conditional_heatmap_sample noisy_action",
        )
        heatmap = torch.randn(
            size=(batch_size, self.heatmap_num_tokens, self.heatmap_dim),
            dtype=image_tokens.dtype,
            device=image_tokens.device,
            generator=generator,
        )

        scheduler = self._heatmap_sampling_scheduler()
        scheduler.set_timesteps(self.num_inference_steps)
        cache_kwarg = None
        world_cache = None
        if hasattr(self.model, "prefill_world_cache"):
            cache_kwarg = "world_cache"
            world_cache = self.model.prefill_world_cache(
                image_tokens=image_tokens,
                gaze_token=gaze_token,
            )
        elif hasattr(self.model, "prefill_condition_cache"):
            cache_kwarg = "condition_cache"
            world_cache = self.model.prefill_condition_cache(
                image_tokens=image_tokens,
                gaze_token=gaze_token,
            )

        for t in scheduler.timesteps:
            model_kwargs = {}
            if world_cache is not None:
                model_kwargs[cache_kwarg] = world_cache
            if getattr(self.model, "supports_skip_action_decoder", False):
                model_kwargs["skip_action"] = True
            model_output = self.model(
                image_tokens=image_tokens,
                gaze_token=gaze_token,
                noisy_action=noisy_action,
                noisy_heatmap=heatmap,
                timestep=t,
                is_inference=False,
                **model_kwargs,
            ).heatmap
            heatmap = scheduler.step(
                model_output,
                t,
                heatmap,
                generator=generator,
                **self.kwargs,
            ).prev_sample
        return heatmap

    def _to_model_bool(self, value: Optional[torch.Tensor], batch_size: int, default: bool):
        if value is None:
            return torch.full(
                (batch_size,),
                default,
                device=self.device,
                dtype=torch.bool,
            )
        if not torch.is_tensor(value):
            raise TypeError(
                "model boolean route mask must be a torch.Tensor, "
                f"got {type(value).__name__}."
            )
        value = value.to(device=self.device)
        self._require_bool_vector("model boolean route mask", value, batch_size)
        return value

    def _validate_gaze_condition_inputs(
        self,
        gaze_xy: torch.Tensor,
        has_gaze_label: torch.Tensor,
        name: str,
    ) -> None:
        batch_size = int(has_gaze_label.shape[0])
        if gaze_xy.shape != (batch_size, 2):
            raise ValueError(
                f"{name} gaze_xy must have shape [B, 2] with B={batch_size}, "
                f"got {tuple(gaze_xy.shape)}."
            )
        self._require_floating_finite(f"{name} gaze_xy", gaze_xy)
        self._require_unit_interval(
            f"{name} gaze_xy rows with has_gaze_label=True",
            gaze_xy,
            has_gaze_label,
        )
        inactive_gaze_xy = gaze_xy[~has_gaze_label]
        if inactive_gaze_xy.numel() > 0 and torch.any(inactive_gaze_xy != 0):
            raise ValueError(
                f"{name} gaze_xy rows with has_gaze_label=False must be zero placeholders."
            )

    def _validate_action_base_abs(
        self,
        action_base_abs: torch.Tensor,
        batch_size: int,
        name: str = "action_base_abs",
    ) -> None:
        if action_base_abs.shape != (batch_size, self.action_dim):
            raise ValueError(
                f"{name} must have shape [B, {self.action_dim}] with B={batch_size}, "
                f"got {tuple(action_base_abs.shape)}."
            )
        self._require_floating_finite(name, action_base_abs)

    def _validate_noisy_action(
        self,
        noisy_action: torch.Tensor,
        batch_size: int,
        name: str = "noisy_action",
    ) -> None:
        if noisy_action.shape != (batch_size, self.action_horizon, self.action_dim):
            raise ValueError(
                f"{name} must have shape [B, {self.action_horizon}, {self.action_dim}] "
                f"with B={batch_size}, got {tuple(noisy_action.shape)}."
            )
        self._require_floating_finite(name, noisy_action)

    def _validate_noisy_heatmap(
        self,
        noisy_heatmap: torch.Tensor,
        batch_size: int,
        name: str = "noisy_heatmap",
    ) -> None:
        if noisy_heatmap.shape != (batch_size, self.heatmap_num_tokens, self.heatmap_dim):
            raise ValueError(
                f"{name} must have shape [B, {self.heatmap_num_tokens}, {self.heatmap_dim}] "
                f"with B={batch_size}, got {tuple(noisy_heatmap.shape)}."
            )
        self._require_floating_finite(name, noisy_heatmap)

    def _validate_loss_action_metadata(
        self,
        batch: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> None:
        if "action_abs" in batch:
            action_abs = self._require_batch_tensor(batch, "action_abs")
            if action_abs.shape != (batch_size, self.action_horizon, self.action_dim):
                raise ValueError(
                    "batch['action_abs'] must have shape "
                    f"[B, {self.action_horizon}, {self.action_dim}], "
                    f"got {tuple(action_abs.shape)}."
                )
            self._require_floating_finite("batch['action_abs']", action_abs)
        if "action_base_abs" in batch:
            action_base_abs = self._require_batch_tensor(batch, "action_base_abs")
            if action_base_abs.shape != (batch_size, self.action_dim):
                raise ValueError(
                    "batch['action_base_abs'] must have shape "
                    f"[B, {self.action_dim}], got {tuple(action_base_abs.shape)}."
                )
            self._require_floating_finite("batch['action_base_abs']", action_base_abs)

        for optional_key in ("action_abs", "action_base_abs"):
            mask_key = f"has_{optional_key}"
            if mask_key not in batch:
                continue
            if optional_key not in batch:
                raise KeyError(f"batch[{mask_key!r}] requires batch[{optional_key!r}].")
            optional_value = batch[optional_key]
            optional_mask = self._require_batch_tensor(batch, mask_key)
            self._require_bool_vector(
                f"batch[{mask_key!r}]",
                optional_mask,
                batch_size,
            )
            self._require_zero_rows_where_mask_false(
                f"batch[{optional_key!r}]",
                optional_value,
                optional_mask,
                f"batch[{mask_key!r}]",
            )

        is_open = batch.get("is_open")
        if is_open is None:
            return
        if "has_action_abs" in batch and torch.any(batch["has_action_abs"] & is_open):
            raise ValueError(
                "Open-source rows must not mark batch['has_action_abs']=True."
            )
        if "has_action_base_abs" in batch and torch.any(batch["has_action_base_abs"] & is_open):
            raise ValueError(
                "Open-source rows must not mark batch['has_action_base_abs']=True."
            )

    def _validate_action_base_abs_presence_mask(
        self,
        has_action_base_abs: Optional[torch.Tensor],
        batch_size: int,
    ) -> None:
        if has_action_base_abs is None:
            return
        if not torch.is_tensor(has_action_base_abs):
            raise TypeError(
                "obs_dict['has_action_base_abs'] must be a torch.Tensor, "
                f"got {type(has_action_base_abs).__name__}."
            )
        has_action_base_abs = has_action_base_abs.to(device=self.device)
        self._require_bool_vector(
            "obs_dict['has_action_base_abs']",
            has_action_base_abs,
            batch_size,
        )
        if not torch.all(has_action_base_abs):
            raise ValueError(
                "obs_dict['action_base_abs'] was provided, but "
                "obs_dict['has_action_base_abs'] is False for some rows. "
                "Do not convert placeholder action bases to absolute commands."
            )

    def _relative_to_absolute_action(
        self,
        relative_action: torch.Tensor,
        action_base_abs: torch.Tensor,
    ) -> torch.Tensor:
        rel_np = relative_action.detach().cpu().numpy()
        base_np = action_base_abs.detach().cpu().numpy()
        abs_np = relative_actions_to_absolute_actions(rel_np, base_np).astype(np.float32)
        return torch.from_numpy(abs_np).to(
            device=relative_action.device,
            dtype=relative_action.dtype,
        )

    @staticmethod
    def _require_batch_tensor(batch: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
        if key not in batch:
            raise KeyError(f"batch must contain {key!r}.")
        value = batch[key]
        if not torch.is_tensor(value):
            raise TypeError(f"batch[{key!r}] must be a torch.Tensor, got {type(value).__name__}.")
        return value

    @staticmethod
    def _require_floating_finite(name: str, value: torch.Tensor) -> None:
        if not torch.is_floating_point(value):
            raise ValueError(f"{name} must be a floating tensor, got dtype {value.dtype}.")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values.")

    @staticmethod
    def _require_unit_interval(
        name: str,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        if mask is not None:
            value = value[mask]
        if value.numel() == 0:
            return
        if torch.any((value < 0.0) | (value > 1.0)):
            min_value = value.detach().amin().item()
            max_value = value.detach().amax().item()
            raise ValueError(
                f"{name} must be in [0, 1], got min={min_value:.6g}, "
                f"max={max_value:.6g}."
            )

    @staticmethod
    def _require_zero_rows_where_mask_false(
        name: str,
        value: torch.Tensor,
        mask: torch.Tensor,
        mask_name: str,
    ) -> None:
        inactive_value = value.reshape(int(mask.shape[0]), -1)[~mask]
        if inactive_value.numel() > 0 and torch.any(inactive_value != 0):
            raise ValueError(
                f"{name} rows with {mask_name}=False must be zero placeholders."
            )

    @staticmethod
    def _require_bool_vector(name: str, value: torch.Tensor, batch_size: int) -> None:
        if value.dtype != torch.bool:
            raise ValueError(f"{name} must be a BoolTensor, got dtype {value.dtype}.")
        if value.shape != (batch_size,):
            raise ValueError(f"{name} must have shape [B]={batch_size}, got {tuple(value.shape)}.")

    def loss_routing_contract_summary(self) -> Dict[str, object]:
        uses_latent_mse = self.heatmap_objective != "dsnt_js"
        uses_diffusion_final_loss = (
            self.heatmap_objective != "dsnt_js"
            and self.heatmap_diffusion_final_loss_enabled
        )
        if self.heatmap_objective == "dsnt_js":
            heatmap_supervision = "full_resolution_dsnt_plus_js_after_frozen_decoder"
            heatmap_loss_mask = "has_heatmap & has_gaze_label"
            heatmap_target = (
                "gaze_xy -> online 256x256 Gaussian target -> frozen Cosmos encoder "
                "-> affine-normalized clean latent target for scheduler space; "
                "predicted clean latent is denormalized and decoded by frozen Cosmos "
                "before full-resolution final decoded supervision"
            )
            open_heatmap_training = "xy DSNT plus generated Gaussian JS target"
        elif uses_diffusion_final_loss:
            heatmap_supervision = (
                "latent_diffusion_mse_plus_decoded_final_heatmap_loss"
            )
            heatmap_loss_mask = "has_heatmap & has_gaze_label"
            heatmap_target = (
                "gaze_xy/dense temporal label -> online 256x256 target -> frozen "
                "Cosmos encoder -> affine-normalized clean latent target for "
                "scheduler space; DiT predicts the configured diffusion target in "
                "latent space and its reconstructed clean latent is decoded by the "
                "frozen Cosmos decoder for auxiliary final XY/NLL/JS supervision"
            )
            open_heatmap_training = (
                "latent diffusion MSE plus decoded final heatmap XY/NLL/JS loss"
            )
        else:
            heatmap_supervision = "latent_diffusion_mse_against_frozen_cosmos_target"
            heatmap_loss_mask = "has_heatmap & has_gaze_label"
            heatmap_target = (
                "gaze_xy -> online 256x256 Gaussian target -> frozen Cosmos encoder "
                "-> affine-normalized clean latent target for scheduler space; "
                "DiT predicts the configured diffusion target in latent space"
            )
            open_heatmap_training = "latent diffusion MSE against generated Cosmos target"
        return {
            "source": "policy",
            "dynamic_head_freezing": False,
            "action_loss_mask": "(~is_open) & has_action",
            "heatmap_loss_mask": heatmap_loss_mask,
            "heatmap_target": heatmap_target,
            "heatmap_supervision": heatmap_supervision,
            "heatmap_distribution_mode": self.heatmap_distribution_mode,
            "latent_mse_loss": uses_latent_mse,
            "diffusion_final_heatmap_loss": uses_diffusion_final_loss,
            "heatmap_diffusion_final_loss_enabled": (
                self.heatmap_diffusion_final_loss_enabled
            ),
            "heatmap_final_loss_timestep_weighting": (
                self.heatmap_final_loss_timestep_weighting
            ),
            "heatmap_xy_loss_weight": self.heatmap_xy_loss_weight,
            "heatmap_point_nll_loss_weight": self.heatmap_point_nll_loss_weight,
            "heatmap_js_loss_weight": self.heatmap_js_loss_weight,
            "heatmap_spatial_decoder": self.heatmap_spatial_decoder,
            "heatmap_latent_codec": "frozen_cosmos_tokenizer",
            "heatmap_objective": self.heatmap_objective,
            "heatmap_cosmos_encoder_path": self.heatmap_cosmos_encoder_path,
            "heatmap_cosmos_decoder_path": self.heatmap_cosmos_decoder_path,
            "heatmap_latent_scale": self.heatmap_latent_scale,
            "heatmap_latent_offset": self.heatmap_latent_offset,
            "heatmap_latent_stats_path": self.heatmap_latent_stats_path,
            "heatmap_scheduler_clip_sample": self.heatmap_scheduler_clip_sample,
            "open_rows": {
                "has_action": False,
                "has_heatmap": True,
                "use_gaze_condition": False,
                "gaze_token": "learned_mask",
                "trains_action": False,
                "trains_heatmap": open_heatmap_training,
            },
            "robot_real_gaze_rows": {
                "is_open": False,
                "has_action": True,
                "use_gaze_condition": True,
                "has_heatmap": False,
                "trains_action": True,
                "trains_heatmap": False,
            },
            "robot_masked_gaze_rows": {
                "is_open": False,
                "has_action": True,
                "use_gaze_condition": False,
                "gaze_token": "learned_mask",
                "trains_action": True,
                "trains_heatmap": "has_heatmap & has_gaze_label",
            },
            "validation": {
                key: True
                for key in gaze_wam_required_loss_routing_validation_flags()
            },
        }

    def _validate_loss_batch_contract(self, batch: Dict[str, torch.Tensor]) -> None:
        action = self._require_batch_tensor(batch, "action")
        if action.ndim != 3 or action.shape[1:] != (self.action_horizon, self.action_dim):
            raise ValueError(
                "batch['action'] must have shape "
                f"[B, {self.action_horizon}, {self.action_dim}], got {tuple(action.shape)}."
            )
        if action.shape[0] <= 0:
            raise ValueError("batch['action'] must contain at least one sample.")
        self._require_floating_finite("batch['action']", action)
        batch_size = int(action.shape[0])

        if "obs" not in batch or not isinstance(batch["obs"], dict) or len(batch["obs"]) == 0:
            raise KeyError("batch must contain a non-empty 'obs' dict.")
        for obs_key, obs_value in batch["obs"].items():
            if not torch.is_tensor(obs_value):
                raise TypeError(
                    f"batch['obs'][{obs_key!r}] must be a torch.Tensor, "
                    f"got {type(obs_value).__name__}."
                )
            if obs_value.shape[0] != batch_size:
                raise ValueError(
                    f"batch['obs'][{obs_key!r}] batch dim must be {batch_size}, "
                    f"got {tuple(obs_value.shape)}."
                )
            self._require_floating_finite(f"batch['obs'][{obs_key!r}]", obs_value)

        gaze_xy = self._require_batch_tensor(batch, "gaze_xy")
        if gaze_xy.shape != (batch_size, 2):
            raise ValueError(
                f"batch['gaze_xy'] must have shape [B, 2] with B={batch_size}, "
                f"got {tuple(gaze_xy.shape)}."
            )
        self._require_floating_finite("batch['gaze_xy']", gaze_xy)

        heatmap = self._require_batch_tensor(batch, "heatmap")
        expected_heatmap_3d = (batch_size, self.heatmap_num_tokens, self.heatmap_dim)
        expected_heatmap_4d = (batch_size, 1, self.heatmap_num_tokens, self.heatmap_dim)
        if tuple(heatmap.shape) not in (expected_heatmap_3d, expected_heatmap_4d):
            raise ValueError(
                "batch['heatmap'] must have shape "
                f"{expected_heatmap_4d} or {expected_heatmap_3d}, got {tuple(heatmap.shape)}."
            )
        self._require_floating_finite("batch['heatmap']", heatmap)
        if self.heatmap_objective != "dsnt_js":
            self._require_unit_interval("batch['heatmap']", heatmap)

        if "heatmap_image" in batch:
            heatmap_image = self._require_batch_tensor(batch, "heatmap_image")
            expected_image_size = tuple(self.heatmap_codec.image_size)
            if heatmap_image.shape[0] != batch_size:
                raise ValueError(
                    "batch['heatmap_image'] batch dim must be "
                    f"{batch_size}, got {tuple(heatmap_image.shape)}."
                )
            if heatmap_image.ndim == 3:
                if tuple(heatmap_image.shape[-2:]) != expected_image_size:
                    raise ValueError(
                        "batch['heatmap_image'] spatial shape must be "
                        f"{expected_image_size}, got {tuple(heatmap_image.shape[-2:])}."
                    )
            elif heatmap_image.ndim == 4:
                if heatmap_image.shape[1] != 1:
                    raise ValueError(
                        "batch['heatmap_image'] must have a single channel when "
                        f"rank-4, got {tuple(heatmap_image.shape)}."
                    )
                if tuple(heatmap_image.shape[-2:]) != expected_image_size:
                    raise ValueError(
                        "batch['heatmap_image'] spatial shape must be "
                        f"{expected_image_size}, got {tuple(heatmap_image.shape[-2:])}."
                    )
            else:
                raise ValueError(
                    "batch['heatmap_image'] must have shape [B, H, W] or "
                    f"[B, 1, H, W], got {tuple(heatmap_image.shape)}."
                )
            self._require_floating_finite("batch['heatmap_image']", heatmap_image)
            if torch.any(heatmap_image < 0.0):
                raise ValueError("batch['heatmap_image'] must be non-negative.")

        if "has_heatmap_image" in batch:
            if "heatmap_image" not in batch:
                raise KeyError(
                    "batch['has_heatmap_image'] requires batch['heatmap_image']."
                )
            self._require_bool_vector(
                "batch['has_heatmap_image']",
                self._require_batch_tensor(batch, "has_heatmap_image"),
                batch_size,
            )
            self._require_zero_rows_where_mask_false(
                "batch['heatmap_image']",
                heatmap_image,
                batch["has_heatmap_image"],
                "batch['has_heatmap_image']",
            )

        for key in (
            "is_open",
            "has_action",
            "has_heatmap",
            "has_gaze_label",
            "use_gaze_condition",
        ):
            self._require_bool_vector(
                f"batch[{key!r}]",
                self._require_batch_tensor(batch, key),
                batch_size,
            )

        is_open = batch["is_open"]
        has_action = batch["has_action"]
        has_heatmap = batch["has_heatmap"]
        has_gaze_label = batch["has_gaze_label"]
        use_gaze_condition = batch["use_gaze_condition"]
        self._validate_loss_action_metadata(batch, batch_size)
        self._require_unit_interval(
            "batch['gaze_xy'] rows with has_gaze_label=True",
            gaze_xy,
            has_gaze_label,
        )
        inactive_gaze_xy = gaze_xy[~has_gaze_label]
        if inactive_gaze_xy.numel() > 0 and torch.any(inactive_gaze_xy != 0):
            raise ValueError(
                "batch['gaze_xy'] rows with has_gaze_label=False must be zero placeholders."
            )
        if torch.any(use_gaze_condition & ~has_gaze_label):
            raise ValueError(
                "batch['use_gaze_condition'] cannot be True where "
                "batch['has_gaze_label'] is False."
            )
        if torch.any(is_open & has_action):
            raise ValueError("Open-source rows must have batch['has_action']=False.")
        if torch.any(is_open & use_gaze_condition):
            raise ValueError("Open-source rows must use the learned gaze mask token.")
        if torch.any((~is_open) & ~has_action):
            raise ValueError("Robot rows must have batch['has_action']=True.")
        if torch.any(is_open & ~has_heatmap):
            raise ValueError("Open-source rows must have batch['has_heatmap']=True.")
        if torch.any((~is_open) & use_gaze_condition & has_heatmap):
            raise ValueError(
                "Robot rows with real gaze condition must not train heatmap loss; "
                "set batch['has_heatmap']=False or drop the gaze condition."
            )
        inactive_action = action.reshape(batch_size, -1)[~has_action]
        if inactive_action.numel() > 0 and torch.any(inactive_action != 0):
            raise ValueError(
                "batch['action'] rows with has_action=False must be zero placeholders."
            )
        inactive_heatmap = heatmap.reshape(batch_size, -1)[~has_heatmap]
        if inactive_heatmap.numel() > 0 and torch.any(inactive_heatmap != 0):
            raise ValueError(
                "batch['heatmap'] rows with has_heatmap=False must be zero placeholders."
            )
        dropped = self._require_batch_tensor(batch, "is_gaze_condition_dropped")
        self._require_bool_vector(
            "batch['is_gaze_condition_dropped']",
            dropped,
            batch_size,
        )
        if not torch.equal(dropped, ~use_gaze_condition):
            raise ValueError(
                "batch['is_gaze_condition_dropped'] must equal "
                "~batch['use_gaze_condition']."
            )

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        cfg_scale: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        model_obs_raw = self._model_obs_from_obs_dict(obs_dict)
        nobs = self.normalizer.normalize(model_obs_raw)
        batch_size = next(iter(nobs.values())).shape[0]
        gaze_xy = obs_dict.get("gaze_xy")
        if gaze_xy is None:
            gaze_xy = torch.zeros(
                batch_size,
                2,
                device=self.device,
                dtype=self.dtype,
            )
            use_gaze_condition = torch.zeros(
                batch_size,
                device=self.device,
                dtype=torch.bool,
            )
            has_gaze_label = use_gaze_condition
        else:
            gaze_xy = gaze_xy.to(device=self.device, dtype=self.dtype)
            use_gaze_condition = self._to_model_bool(
                obs_dict.get("use_gaze_condition"),
                batch_size=batch_size,
                default=True,
            )
            has_gaze_label = self._to_model_bool(
                obs_dict.get("has_gaze_label"),
                batch_size=batch_size,
                default=True,
            )
            self._validate_gaze_condition_inputs(
                gaze_xy,
                has_gaze_label,
                "predict_action",
            )

        image_tokens, gaze_token = self._encode_conditions(
            obs_dict=nobs,
            gaze_xy=gaze_xy,
            use_gaze_condition=use_gaze_condition,
            has_gaze_label=has_gaze_label,
        )
        mask_gaze_token = None
        effective_cfg_scale = (
            self.cfg_scale
            if cfg_scale is None
            else self._validate_nonnegative_float("cfg_scale", cfg_scale)
        )
        if effective_cfg_scale != 1.0:
            mask_gaze_token = self.gaze_encoder(
                gaze_xy=gaze_xy,
                use_gaze_condition=torch.zeros(
                    batch_size,
                    device=self.device,
                    dtype=torch.bool,
                ),
                has_gaze_label=has_gaze_label,
            )
        nsample = self.conditional_sample(
            image_tokens=image_tokens,
            gaze_token=gaze_token,
            mask_gaze_token=mask_gaze_token,
            cfg_scale=effective_cfg_scale,
            **self.kwargs,
        )
        action_pred = self.normalizer["action"].unnormalize(nsample)
        result = {
            "action": action_pred,
            "action_pred": action_pred,
            "action_pred_relative": action_pred,
            "cfg_scale": torch.as_tensor(
                effective_cfg_scale,
                device=action_pred.device,
                dtype=action_pred.dtype,
            ),
            "cfg_enabled": torch.as_tensor(
                effective_cfg_scale != 1.0,
                device=action_pred.device,
                dtype=torch.bool,
            ),
        }
        action_base_abs = obs_dict.get("action_base_abs")
        if action_base_abs is not None:
            action_base_abs = action_base_abs.to(device=action_pred.device, dtype=action_pred.dtype)
            self._validate_action_base_abs(
                action_base_abs,
                batch_size=batch_size,
                name="obs_dict['action_base_abs']",
            )
            self._validate_action_base_abs_presence_mask(
                obs_dict.get("has_action_base_abs"),
                batch_size=batch_size,
            )
            action_abs = self._relative_to_absolute_action(action_pred, action_base_abs)
            result["action_abs"] = action_abs
            result["action_pred_abs"] = action_abs
            result["action"] = action_abs
        return result

    @torch.no_grad()
    def predict_heatmap(
        self,
        obs_dict: Dict[str, torch.Tensor],
        use_gaze_condition: Optional[torch.Tensor] = None,
        timestep=None,
        noisy_action: Optional[torch.Tensor] = None,
        noisy_heatmap: Optional[torch.Tensor] = None,
        decode: bool = True,
        decode_method: Optional[str] = None,
        clamp_tokens: bool = True,
        generator=None,
    ) -> Dict[str, torch.Tensor]:
        """Predict heatmap tokens and optionally decode them for visualization.

        This is a diagnostic/paper-figure path. Main action inference still drops heatmap tokens.
        If no gaze routing is supplied, the method uses the trainable mask token by default to
        avoid copying a provided gaze label into the heatmap prediction.
        """
        model_obs_raw = self._model_obs_from_obs_dict(obs_dict)
        nobs = self.normalizer.normalize(model_obs_raw)
        batch_size = next(iter(nobs.values())).shape[0]
        gaze_xy = obs_dict.get("gaze_xy")
        if gaze_xy is None:
            gaze_xy = torch.zeros(
                batch_size,
                2,
                device=self.device,
                dtype=self.dtype,
            )
            has_gaze_label = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        else:
            gaze_xy = gaze_xy.to(device=self.device, dtype=self.dtype)
            has_gaze_label = self._to_model_bool(
                obs_dict.get("has_gaze_label"),
                batch_size=batch_size,
                default=True,
            )
            self._validate_gaze_condition_inputs(
                gaze_xy,
                has_gaze_label,
                "predict_heatmap",
            )

        if use_gaze_condition is None:
            use_gaze_condition = obs_dict.get("use_gaze_condition")
        use_gaze_condition = self._to_model_bool(
            use_gaze_condition,
            batch_size=batch_size,
            default=False,
        )

        image_tokens, gaze_token = self._encode_conditions(
            obs_dict=nobs,
            gaze_xy=gaze_xy,
            use_gaze_condition=use_gaze_condition,
            has_gaze_label=has_gaze_label,
        )
        use_iterative_denoise = timestep is None and noisy_heatmap is None
        if noisy_action is None:
            noisy_action = torch.zeros(
                batch_size,
                self.action_horizon,
                self.action_dim,
                device=self.device,
                dtype=image_tokens.dtype,
            )
        else:
            noisy_action = noisy_action.to(device=self.device, dtype=image_tokens.dtype)
        self._validate_noisy_action(
            noisy_action,
            batch_size=batch_size,
            name="predict_heatmap noisy_action",
        )
        if use_iterative_denoise:
            heatmap_tokens_raw = self.conditional_heatmap_sample(
                image_tokens=image_tokens,
                gaze_token=gaze_token,
                noisy_action=noisy_action,
                generator=generator,
            )
            heatmap_model_output = None
            heatmap_features = None
            timesteps = self.noise_scheduler.timesteps.to(device=image_tokens.device)
            noisy_heatmap_for_result = None
            prediction_mode = "iterative_denoise"
        else:
            prediction_mode = "single_step_diagnostic"
        if noisy_heatmap is None:
            noisy_heatmap = torch.zeros(
                batch_size,
                self.heatmap_num_tokens,
                self.heatmap_dim,
                device=self.device,
                dtype=image_tokens.dtype,
            )
        else:
            noisy_heatmap = noisy_heatmap.to(device=self.device, dtype=image_tokens.dtype)
        self._validate_noisy_heatmap(
            noisy_heatmap,
            batch_size=batch_size,
            name="predict_heatmap noisy_heatmap",
        )
        if not use_iterative_denoise:
            timesteps = self._prepare_timesteps(
                timestep,
                batch_size,
                image_tokens.device,
                name="predict_heatmap timestep",
            )

            pred = self.model(
                image_tokens=image_tokens,
                gaze_token=gaze_token,
                noisy_action=noisy_action,
                noisy_heatmap=noisy_heatmap,
                timestep=timesteps,
                is_inference=False,
            )
            heatmap_tokens_raw = self._heatmap_clean_prediction(
                sample=noisy_heatmap,
                model_output=pred.heatmap,
                timestep=timesteps,
            )
            heatmap_model_output = pred.heatmap
            heatmap_features = pred.heatmap_features
            noisy_heatmap_for_result = noisy_heatmap
        if self.heatmap_objective == "dsnt_js":
            heatmap_tokens = heatmap_tokens_raw
        else:
            heatmap_tokens = heatmap_tokens_raw
        result = {
            "heatmap_tokens": heatmap_tokens,
            "heatmap_tokens_raw": heatmap_tokens_raw,
            "heatmap_model_output": heatmap_model_output,
            "heatmap_features": heatmap_features,
            "noisy_heatmap": noisy_heatmap_for_result,
            "noisy_action": noisy_action,
            "timestep": timesteps,
            "heatmap_prediction_mode": prediction_mode,
        }
        if decode:
            if self.heatmap_spatial_decoder == "cosmos_tokenizer":
                heatmap_image_logits = self._heatmap_tokens_to_spatial_image(
                    heatmap_tokens_raw,
                    "predict_heatmap heatmap_tokens_raw",
                )
                result["heatmap_image_logits"] = heatmap_image_logits
                if self.heatmap_objective == "dsnt_js":
                    result["heatmap_image"] = spatial_distribution_2d(
                        heatmap_image_logits,
                        mode=self.heatmap_distribution_mode,
                        temperature=self.heatmap_dsnt_temperature,
                    )
                else:
                    result["heatmap_image"] = spatial_distribution_2d(
                        heatmap_image_logits,
                        mode=self.heatmap_distribution_mode,
                        temperature=self.heatmap_dsnt_temperature,
                    )
            else:
                method = decode_method or self.heatmap_decode_method
                result["heatmap_image"] = self.heatmap_codec.decode_tokens(
                    heatmap_tokens,
                    method=method,
                )
        return result

    def compute_gaze_dependency_ratio(
        self,
        obs_dict: Dict[str, torch.Tensor],
        noisy_action: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compare action predictions/features under real gaze vs the trainable mask token."""
        model_obs_raw = self._model_obs_from_obs_dict(obs_dict)
        nobs = self.normalizer.normalize(model_obs_raw)
        batch_size = next(iter(nobs.values())).shape[0]
        gaze_xy = obs_dict.get("gaze_xy")
        if gaze_xy is None:
            raise ValueError("compute_gaze_dependency_ratio requires obs_dict['gaze_xy'].")
        gaze_xy = gaze_xy.to(device=self.device, dtype=self.dtype)
        has_gaze_label = self._to_model_bool(
            obs_dict.get("has_gaze_label"),
            batch_size=batch_size,
            default=True,
        )
        self._validate_gaze_condition_inputs(
            gaze_xy,
            has_gaze_label,
            "compute_gaze_dependency_ratio",
        )
        if not torch.all(has_gaze_label):
            raise ValueError(
                "compute_gaze_dependency_ratio requires every row to have "
                "obs_dict['has_gaze_label']=True; filter eligible point-gaze rows "
                "before calling GDR."
            )

        image_tokens = self.obs_encoder(nobs)
        gaze_token = self.gaze_encoder(
            gaze_xy=gaze_xy,
            use_gaze_condition=torch.ones(batch_size, device=self.device, dtype=torch.bool),
            has_gaze_label=has_gaze_label,
        )
        mask_gaze_token = self.gaze_encoder(
            gaze_xy=gaze_xy,
            use_gaze_condition=torch.zeros(batch_size, device=self.device, dtype=torch.bool),
            has_gaze_label=has_gaze_label,
        )

        if noisy_action is None:
            noisy_action = torch.zeros(
                batch_size,
                self.action_horizon,
                self.action_dim,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            noisy_action = noisy_action.to(device=self.device, dtype=self.dtype)
        self._validate_noisy_action(
            noisy_action,
            batch_size=batch_size,
            name="compute_gaze_dependency_ratio noisy_action",
        )
        timestep = self._prepare_timesteps(
            timestep,
            batch_size,
            self.device,
            name="compute_gaze_dependency_ratio timestep",
        )

        conditioned = self.model(
            image_tokens=image_tokens,
            gaze_token=gaze_token,
            noisy_action=noisy_action,
            noisy_heatmap=None,
            timestep=timestep,
            is_inference=True,
        )
        masked = self.model(
            image_tokens=image_tokens,
            gaze_token=mask_gaze_token,
            noisy_action=noisy_action,
            noisy_heatmap=None,
            timestep=timestep,
            is_inference=True,
        )
        feature_gdr = gaze_dependency_ratio(conditioned.action_features, masked.action_features)
        output_gdr = gaze_dependency_ratio(conditioned.action, masked.action)
        return {
            "feature_gdr": feature_gdr,
            "output_gdr": output_gdr,
            "feature_gdr_mean": feature_gdr.mean(),
            "output_gdr_mean": output_gdr.mean(),
        }

    def compute_loss_components(self, batch, return_per_sample: bool = False):
        if "valid_mask" in batch:
            raise ValueError(
                "Gaze-WAM mixed batches must not contain batch['valid_mask']; "
                "use has_action/has_heatmap/source masks for loss routing."
            )
        self._validate_loss_batch_contract(batch)

        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        heatmap = batch["heatmap"].to(device=nactions.device, dtype=nactions.dtype)
        if heatmap.ndim == 4:
            if heatmap.shape[1] != 1:
                raise ValueError(
                    "Only heatmap_horizon=1 is supported by the first policy slice, "
                    f"got heatmap shape {heatmap.shape}."
                )
            heatmap = heatmap[:, 0]
        gaze_xy = batch["gaze_xy"].to(device=nactions.device, dtype=nactions.dtype)
        has_heatmap = batch["has_heatmap"].to(device=nactions.device)
        has_gaze_label = batch["has_gaze_label"].to(device=nactions.device)
        target_gaze_mask = has_heatmap & has_gaze_label
        target_heatmap_image = None
        if self._should_generate_cosmos_latent_target():
            target_heatmap_image = self._target_heatmap_image_from_batch_or_xy(
                batch=batch,
                gaze_xy=gaze_xy,
                valid_mask=target_gaze_mask,
            )
            target_heatmap = self._heatmap_image_to_training_tokens(target_heatmap_image).to(
                device=heatmap.device,
                dtype=heatmap.dtype,
            )
            heatmap = torch.where(
                target_gaze_mask.reshape(-1, 1, 1),
                target_heatmap,
                heatmap,
            )

        action_noise = torch.randn(
            nactions.shape,
            device=nactions.device,
            dtype=nactions.dtype,
        )
        heatmap_noise = torch.randn(
            heatmap.shape,
            device=heatmap.device,
            dtype=heatmap.dtype,
        )
        noisy_action_input = action_noise + self.input_pertub * torch.randn(
            nactions.shape,
            device=nactions.device,
            dtype=nactions.dtype,
        )

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (nactions.shape[0],),
            device=nactions.device,
        ).long()

        noisy_action = self.noise_scheduler.add_noise(
            nactions,
            noisy_action_input,
            timesteps,
        )
        noisy_heatmap = self.noise_scheduler.add_noise(
            heatmap,
            heatmap_noise,
            timesteps,
        )

        image_tokens, gaze_token = self._encode_conditions(
            obs_dict=nobs,
            gaze_xy=gaze_xy,
            use_gaze_condition=batch.get("use_gaze_condition"),
            has_gaze_label=batch.get("has_gaze_label"),
        )
        target_action = self._scheduler_target(nactions, action_noise)
        action_loss_mask = (~batch["is_open"].to(device=nactions.device)) & batch[
            "has_action"
        ].to(device=nactions.device)
        model_kwargs = {}
        if getattr(self.model, "supports_skip_action_decoder", False):
            model_kwargs["skip_action"] = not bool(action_loss_mask.any().detach().item())
        pred = self.model(
            image_tokens=image_tokens,
            gaze_token=gaze_token,
            noisy_action=noisy_action,
            noisy_heatmap=noisy_heatmap,
            timestep=timesteps,
            **model_kwargs,
        )

        if self.heatmap_objective == "dsnt_js":
            heatmap_loss_mask = target_gaze_mask
        else:
            heatmap_loss_mask = (
                target_gaze_mask
                if self._should_generate_cosmos_latent_target()
                else has_heatmap
            )
        heatmap_xy_loss_mask = target_gaze_mask

        per_sample_action_loss = F.mse_loss(
            pred.action,
            target_action,
            reduction="none",
        )
        per_sample_action_loss = reduce(per_sample_action_loss, "b ... -> b", "mean")

        pred_clean_heatmap = self._heatmap_clean_prediction(
            sample=noisy_heatmap,
            model_output=pred.heatmap,
            timestep=timesteps,
        )

        action_loss = distributed_masked_mean(
            per_sample_action_loss,
            action_loss_mask,
        )
        if self.heatmap_objective == "dsnt_js":
            pred_heatmap_image = self._heatmap_tokens_to_spatial_image(
                pred_clean_heatmap,
                "pred_clean_heatmap",
            )
            if target_heatmap_image is None:
                raise RuntimeError("DSNT/JS heatmap target was not generated.")
            per_sample_heatmap_xy_loss = per_sample_dsnt_xy_loss(
                pred_heatmap_image,
                gaze_xy,
                temperature=self.heatmap_dsnt_temperature,
                distribution_mode=self.heatmap_distribution_mode,
            )
            per_sample_heatmap_js_loss = per_sample_spatial_js_loss(
                pred_heatmap_image,
                target_heatmap_image,
                temperature=self.heatmap_dsnt_temperature,
                distribution_mode=self.heatmap_distribution_mode,
            )
            per_sample_heatmap_token_kl_loss = torch.zeros_like(
                per_sample_heatmap_js_loss
            )
            per_sample_heatmap_point_nll_loss = torch.zeros_like(
                per_sample_heatmap_js_loss
            )
            heatmap_xy_loss = distributed_masked_mean(
                per_sample_heatmap_xy_loss,
                heatmap_xy_loss_mask,
            )
            heatmap_point_nll_loss = heatmap_xy_loss * 0.0
            heatmap_js_loss = distributed_masked_mean(
                per_sample_heatmap_js_loss,
                heatmap_loss_mask,
            )
            heatmap_token_kl_loss = distributed_masked_mean(
                per_sample_heatmap_token_kl_loss,
                heatmap_loss_mask,
            )
            per_sample_heatmap_loss = (
                self.heatmap_xy_loss_weight * per_sample_heatmap_xy_loss
                + self.heatmap_js_loss_weight * per_sample_heatmap_js_loss
            )
            heatmap_loss = (
                self.heatmap_xy_loss_weight * heatmap_xy_loss
                + self.heatmap_js_loss_weight * heatmap_js_loss
            )
        else:
            target_heatmap = self._heatmap_target(heatmap, heatmap_noise)
            per_sample_heatmap_loss = F.mse_loss(
                pred.heatmap,
                target_heatmap,
                reduction="none",
            )
            per_sample_heatmap_loss = reduce(
                per_sample_heatmap_loss,
                "b ... -> b",
                "mean",
            )
            if self._should_generate_cosmos_latent_target():
                per_sample_heatmap_token_kl_loss = torch.zeros_like(per_sample_heatmap_loss)
            else:
                per_sample_heatmap_token_kl_loss = self._per_sample_heatmap_token_kl(
                    pred_clean=pred_clean_heatmap,
                    target_clean=heatmap,
                )
            heatmap_loss = distributed_masked_mean(
                per_sample_heatmap_loss,
                heatmap_loss_mask,
            )
            heatmap_token_kl_loss = distributed_masked_mean(
                per_sample_heatmap_token_kl_loss,
                heatmap_loss_mask,
            )
            if self.heatmap_diffusion_final_loss_enabled:
                pred_heatmap_image = self._heatmap_tokens_to_spatial_image(
                    pred_clean_heatmap,
                    "pred_clean_heatmap",
                )
                if target_heatmap_image is None:
                    target_heatmap_image = self._target_heatmap_image_from_batch_or_xy(
                        batch=batch,
                        gaze_xy=gaze_xy,
                        valid_mask=target_gaze_mask,
                    )
                per_sample_final_loss_weight = self._per_sample_final_loss_weight(
                    timestep=timesteps,
                    batch_size=int(per_sample_heatmap_loss.shape[0]),
                    device=per_sample_heatmap_loss.device,
                    dtype=per_sample_heatmap_loss.dtype,
                )
                per_sample_heatmap_xy_loss = (
                    per_sample_dsnt_xy_loss(
                        pred_heatmap_image,
                        gaze_xy,
                        temperature=self.heatmap_dsnt_temperature,
                        distribution_mode=self.heatmap_distribution_mode,
                    )
                    * per_sample_final_loss_weight
                )
                per_sample_heatmap_point_nll_loss = (
                    per_sample_spatial_point_nll_loss(
                        pred_heatmap_image,
                        gaze_xy,
                        temperature=self.heatmap_dsnt_temperature,
                        distribution_mode=self.heatmap_distribution_mode,
                    )
                    * per_sample_final_loss_weight
                )
                per_sample_heatmap_js_loss = (
                    per_sample_spatial_js_loss(
                        pred_heatmap_image,
                        target_heatmap_image,
                        temperature=self.heatmap_dsnt_temperature,
                        distribution_mode=self.heatmap_distribution_mode,
                    )
                    * per_sample_final_loss_weight
                )
                heatmap_xy_loss = distributed_masked_mean(
                    per_sample_heatmap_xy_loss,
                    heatmap_xy_loss_mask,
                )
                heatmap_point_nll_loss = distributed_masked_mean(
                    per_sample_heatmap_point_nll_loss,
                    heatmap_xy_loss_mask,
                )
                heatmap_js_loss = distributed_masked_mean(
                    per_sample_heatmap_js_loss,
                    heatmap_loss_mask,
                )
            else:
                per_sample_heatmap_xy_loss = torch.zeros_like(per_sample_heatmap_loss)
                per_sample_heatmap_point_nll_loss = torch.zeros_like(
                    per_sample_heatmap_loss
                )
                per_sample_heatmap_js_loss = torch.zeros_like(per_sample_heatmap_loss)
                heatmap_xy_loss = heatmap_loss * 0.0
                heatmap_point_nll_loss = heatmap_loss * 0.0
                heatmap_js_loss = heatmap_loss * 0.0

        final_heatmap_loss = heatmap_loss * 0.0
        if self.heatmap_diffusion_final_loss_enabled:
            final_heatmap_loss = (
                self.heatmap_xy_loss_weight * heatmap_xy_loss
                + self.heatmap_point_nll_loss_weight * heatmap_point_nll_loss
                + self.heatmap_js_loss_weight * heatmap_js_loss
            )

        loss = (
            self.action_loss_weight * action_loss
            + self.heatmap_loss_weight * heatmap_loss
            + self.heatmap_token_kl_loss_weight * heatmap_token_kl_loss
            + final_heatmap_loss
        )

        result = {
            "loss": loss,
            "action_loss": action_loss.detach(),
            "heatmap_loss": heatmap_loss.detach(),
            "heatmap_xy_loss": heatmap_xy_loss.detach(),
            "heatmap_point_nll_loss": heatmap_point_nll_loss.detach(),
            "heatmap_js_loss": heatmap_js_loss.detach(),
            "heatmap_token_kl_loss": heatmap_token_kl_loss.detach(),
            "heatmap_token_kl_loss_weight": self.heatmap_token_kl_loss_weight,
            "heatmap_xy_loss_weight": self.heatmap_xy_loss_weight,
            "heatmap_point_nll_loss_weight": self.heatmap_point_nll_loss_weight,
            "heatmap_js_loss_weight": self.heatmap_js_loss_weight,
            "heatmap_diffusion_final_loss_enabled": (
                self.heatmap_diffusion_final_loss_enabled
            ),
            "heatmap_final_loss_timestep_weighting": (
                self.heatmap_final_loss_timestep_weighting
            ),
            "heatmap_dsnt_temperature": self.heatmap_dsnt_temperature,
            "heatmap_distribution_mode": self.heatmap_distribution_mode,
            "heatmap_dsnt_target_sigma_px": self.heatmap_dsnt_target_sigma_px,
            "heatmap_latent_scale": self.heatmap_latent_scale,
            "heatmap_latent_offset": self.heatmap_latent_offset,
            "heatmap_scheduler_clip_sample": self.heatmap_scheduler_clip_sample,
            "action_loss_mask_count": distributed_mask_count(action_loss_mask),
            "heatmap_loss_mask_count": distributed_mask_count(heatmap_loss_mask),
            "heatmap_xy_loss_mask_count": distributed_mask_count(heatmap_xy_loss_mask),
        }
        if return_per_sample:
            result.update(
                {
                    "per_sample_action_loss": per_sample_action_loss.detach(),
                    "per_sample_heatmap_loss": per_sample_heatmap_loss.detach(),
                    "per_sample_heatmap_xy_loss": per_sample_heatmap_xy_loss.detach(),
                    "per_sample_heatmap_point_nll_loss": (
                        per_sample_heatmap_point_nll_loss.detach()
                    ),
                    "per_sample_heatmap_js_loss": per_sample_heatmap_js_loss.detach(),
                    "per_sample_heatmap_token_kl_loss": (
                        per_sample_heatmap_token_kl_loss.detach()
                    ),
                    "action_loss_mask": action_loss_mask.detach(),
                    "heatmap_loss_mask": heatmap_loss_mask.detach(),
                    "heatmap_xy_loss_mask": heatmap_xy_loss_mask.detach(),
                }
            )
        return result

    def compute_loss(self, batch):
        return self.compute_loss_components(batch)["loss"]

    def forward(self, batch, return_per_sample: bool = False):
        if return_per_sample:
            return self.compute_loss_components(batch, return_per_sample=True)
        return self.compute_loss(batch)
