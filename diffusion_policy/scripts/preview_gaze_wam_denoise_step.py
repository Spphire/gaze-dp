import argparse
import json
import pathlib
from typing import Dict, Optional, Sequence

import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.gaze_wam.loss import spatial_distribution_2d
from diffusion_policy.scripts.eval_gaze_wam_metrics import load_policy_for_eval
from diffusion_policy.scripts.preview_gaze_wam_checkpoint import (
    _checkpoint_ema_summary,
    _draw_gaze,
    _heatmap_panel,
    _heatmap_to_uint8,
    _imwrite_unicode,
    _instantiate_dataset,
    _latest_obs_rgb,
    _overlay_heatmap,
    _parse_sample_indices,
    _select_sample_indices,
    _title_panel,
)


def _decode_tokens_to_heatmap(policy, tokens: torch.Tensor) -> torch.Tensor:
    if policy.heatmap_spatial_decoder == "cosmos_tokenizer":
        image_logits = policy._heatmap_tokens_to_spatial_image(
            tokens,
            "denoise_step heatmap tokens",
        )
        return spatial_distribution_2d(
            image_logits,
            mode=policy.heatmap_distribution_mode,
            temperature=policy.heatmap_dsnt_temperature,
        )
    return policy.heatmap_codec.decode_tokens(
        tokens,
        method=policy.heatmap_decode_method,
    )


def _denoise_trace(policy, obs: Dict[str, torch.Tensor], generator=None) -> Dict[str, object]:
    model_obs_raw = policy._model_obs_from_obs_dict(obs)
    nobs = policy.normalizer.normalize(model_obs_raw)
    batch_size = next(iter(nobs.values())).shape[0]
    gaze_xy = obs.get("gaze_xy")
    if gaze_xy is None:
        gaze_xy = torch.zeros(
            batch_size,
            2,
            device=policy.device,
            dtype=policy.dtype,
        )
        has_gaze_condition = torch.zeros(
            batch_size, device=policy.device, dtype=torch.bool
        )
        has_gaze_label = has_gaze_condition
    else:
        gaze_xy = gaze_xy.to(device=policy.device, dtype=policy.dtype)
        has_gaze_label = policy._to_model_bool(
            obs.get("has_gaze_label"),
            batch_size=batch_size,
            default=True,
        )
        has_gaze_condition = policy._to_model_bool(
            obs.get("has_gaze_condition", has_gaze_label),
            batch_size=batch_size,
            default=True,
        )
        policy._validate_gaze_condition_inputs(
            gaze_xy,
            has_gaze_condition,
            has_gaze_label,
            "preview_gaze_wam_denoise_step",
        )

    use_gaze_condition = policy._to_model_bool(
        obs.get("use_gaze_condition"),
        batch_size=batch_size,
        default=False,
    )
    image_tokens, gaze_token = policy._encode_conditions(
        obs_dict=nobs,
        gaze_xy=gaze_xy,
        use_gaze_condition=use_gaze_condition,
        has_gaze_condition=has_gaze_condition,
        has_gaze_label=has_gaze_label,
    )
    noisy_action = torch.zeros(
        batch_size,
        policy.action_horizon,
        policy.action_dim,
        dtype=image_tokens.dtype,
        device=image_tokens.device,
    )
    policy._validate_noisy_action(
        noisy_action,
        batch_size=batch_size,
        name="preview_gaze_wam_denoise_step noisy_action",
    )

    heatmap = torch.randn(
        size=(batch_size, policy.heatmap_num_tokens, policy.heatmap_dim),
        dtype=image_tokens.dtype,
        device=image_tokens.device,
        generator=generator,
    )
    init_tokens = heatmap.detach().clone()

    scheduler = policy._heatmap_sampling_scheduler()
    scheduler.set_timesteps(policy.num_inference_steps)
    timesteps = [int(t.item()) if torch.is_tensor(t) else int(t) for t in scheduler.timesteps]

    cache_kwarg = None
    world_cache = None
    if hasattr(policy.model, "prefill_world_cache"):
        cache_kwarg = "world_cache"
        world_cache = policy.model.prefill_world_cache(
            image_tokens=image_tokens,
            gaze_token=gaze_token,
        )
    elif hasattr(policy.model, "prefill_condition_cache"):
        cache_kwarg = "condition_cache"
        world_cache = policy.model.prefill_condition_cache(
            image_tokens=image_tokens,
            gaze_token=gaze_token,
        )

    first_step_tokens = None
    for step_idx, t in enumerate(scheduler.timesteps, start=1):
        model_kwargs = {}
        if world_cache is not None:
            model_kwargs[cache_kwarg] = world_cache
        if getattr(policy.model, "supports_skip_action_decoder", False):
            model_kwargs["skip_action"] = True
        model_output = policy.model(
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
            **policy.kwargs,
        ).prev_sample
        if step_idx == 1:
            first_step_tokens = heatmap.detach().clone()

    if first_step_tokens is None:
        raise RuntimeError("No denoise step was executed.")

    final_tokens = heatmap.detach().clone()
    return {
        "init_tokens": init_tokens,
        "first_step_tokens": first_step_tokens,
        "final_tokens": final_tokens,
        "timesteps": timesteps,
        "noisy_action": noisy_action,
    }


def _comparison_strip(
    image_rgb: np.ndarray,
    init_heatmap: np.ndarray,
    first_heatmap: np.ndarray,
    final_heatmap: np.ndarray,
    target_heatmap: np.ndarray,
    gaze_xy,
    valid: bool,
) -> np.ndarray:
    panels = [
        _title_panel(_draw_gaze(image_rgb, gaze_xy, valid), "RGB + gaze_xy"),
        _title_panel(_heatmap_panel(init_heatmap, gaze_xy, valid), "Init noise decode"),
        _title_panel(_heatmap_panel(first_heatmap, gaze_xy, valid), "After step 1"),
        _title_panel(_heatmap_panel(final_heatmap, gaze_xy, valid), "Final step 8"),
        _title_panel(_heatmap_panel(target_heatmap, gaze_xy, valid), "Target"),
    ]
    separator = np.full(
        (panels[0].shape[0], max(4, panels[0].shape[1] // 64), 3),
        255,
        dtype=panels[0].dtype,
    )
    strip_panels = []
    for idx, panel in enumerate(panels):
        if idx > 0:
            strip_panels.append(separator)
        strip_panels.append(panel)
    return np.concatenate(strip_panels, axis=1)


def _overlay_comparison_strip(
    image_rgb: np.ndarray,
    init_heatmap: np.ndarray,
    first_heatmap: np.ndarray,
    final_heatmap: np.ndarray,
    target_heatmap: np.ndarray,
    gaze_xy,
    valid: bool,
) -> np.ndarray:
    panels = [
        _title_panel(_draw_gaze(image_rgb, gaze_xy, valid), "RGB + gaze_xy"),
        _title_panel(_overlay_heatmap(image_rgb, init_heatmap, gaze_xy, valid), "Init overlay"),
        _title_panel(_overlay_heatmap(image_rgb, first_heatmap, gaze_xy, valid), "Step 1 overlay"),
        _title_panel(_overlay_heatmap(image_rgb, final_heatmap, gaze_xy, valid), "Step 8 overlay"),
        _title_panel(_overlay_heatmap(image_rgb, target_heatmap, gaze_xy, valid), "Target overlay"),
    ]
    separator = np.full(
        (panels[0].shape[0], max(4, panels[0].shape[1] // 64), 3),
        255,
        dtype=panels[0].dtype,
    )
    strip_panels = []
    for idx, panel in enumerate(panels):
        if idx > 0:
            strip_panels.append(separator)
        strip_panels.append(panel)
    return np.concatenate(strip_panels, axis=1)


def preview_gaze_wam_denoise_step(
    checkpoint: str,
    output_dir: str,
    source: str = "open",
    split: str = "val",
    max_samples: int = 4,
    device: str = "cuda:0",
    use_ema: bool = True,
    use_gaze_condition: bool = False,
    sample_indices: Optional[Sequence[int]] = None,
    sample_seed: Optional[int] = None,
    overrides: Optional[Sequence[str]] = None,
    trust_checkpoint: bool = False,
) -> Dict[str, object]:
    max_samples = max(1, int(max_samples))
    policy, cfg = load_policy_for_eval(
        checkpoint=checkpoint,
        device=device,
        use_ema=use_ema,
        overrides=overrides,
        trust_checkpoint=trust_checkpoint,
    )
    ema_summary = _checkpoint_ema_summary(cfg, use_ema_requested=use_ema)
    dataset = _instantiate_dataset(cfg, source=source, split=split)
    selected_indices = _select_sample_indices(
        dataset_len=len(dataset),
        max_samples=max_samples,
        sample_indices=sample_indices,
        sample_seed=sample_seed,
    )
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=len(selected_indices),
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    device_obj = torch.device(device)
    batch = dict_apply(batch, lambda x: x.to(device_obj, non_blocking=True))

    obs = dict(batch["obs"])
    obs["gaze_xy"] = batch["gaze_xy"]
    obs["has_gaze_label"] = batch.get(
        "has_gaze_label",
        torch.ones(batch["gaze_xy"].shape[0], device=device_obj, dtype=torch.bool),
    )
    obs["has_gaze_condition"] = batch.get(
        "has_gaze_condition", obs["has_gaze_label"]
    )
    obs["use_gaze_condition"] = torch.full(
        (batch["gaze_xy"].shape[0],),
        bool(use_gaze_condition),
        device=device_obj,
        dtype=torch.bool,
    )
    generator = None
    if sample_seed is not None:
        generator = torch.Generator(device=device_obj)
        generator.manual_seed(int(sample_seed))

    with torch.no_grad():
        trace = _denoise_trace(policy, obs, generator=generator)
        init_image = _decode_tokens_to_heatmap(policy, trace["init_tokens"])
        first_image = _decode_tokens_to_heatmap(policy, trace["first_step_tokens"])
        final_image = _decode_tokens_to_heatmap(policy, trace["final_tokens"])
        target_mask = obs["has_gaze_label"].to(device=policy.device)
        target_image = policy._target_heatmap_image_from_batch_or_xy(
            batch=batch,
            gaze_xy=batch["gaze_xy"].to(device=policy.device, dtype=policy.dtype),
            valid_mask=target_mask,
        )

    init_images = init_image.detach().float().cpu().numpy()
    first_images = first_image.detach().float().cpu().numpy()
    final_images = final_image.detach().float().cpu().numpy()
    target_images = target_image.detach().float().cpu().numpy()
    gaze_xy_all = batch["gaze_xy"].detach().float().cpu().numpy()
    has_gaze_all = obs["has_gaze_label"].detach().cpu().numpy().astype(bool)

    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    camera_key = str(cfg.task.camera_key)
    sample_summaries = []
    for sample_idx in range(min(max_samples, final_images.shape[0])):
        image_rgb = _latest_obs_rgb(batch, camera_key=camera_key, sample_idx=sample_idx)
        gaze_xy = gaze_xy_all[sample_idx]
        has_gaze = bool(has_gaze_all[sample_idx])
        init = init_images[sample_idx]
        first = first_images[sample_idx]
        final = final_images[sample_idx]
        target = target_images[sample_idx]
        sample_dir = out_dir / f"sample_{sample_idx:03d}"
        paths = {
            "rgb": sample_dir / "rgb.png",
            "init_noise_heatmap": sample_dir / "init_noise_heatmap.png",
            "first_step_heatmap": sample_dir / "first_step_heatmap.png",
            "final_heatmap": sample_dir / "final_heatmap.png",
            "target_heatmap": sample_dir / "target_heatmap.png",
            "init_noise_overlay": sample_dir / "init_noise_overlay.png",
            "first_step_overlay": sample_dir / "first_step_overlay.png",
            "final_overlay": sample_dir / "final_overlay.png",
            "target_overlay": sample_dir / "target_overlay.png",
            "comparison": sample_dir / "comparison.png",
            "overlay_comparison": sample_dir / "overlay_comparison.png",
        }
        _imwrite_unicode(paths["rgb"], cv2.cvtColor(_draw_gaze(image_rgb, gaze_xy, has_gaze), cv2.COLOR_RGB2BGR))
        _imwrite_unicode(paths["init_noise_heatmap"], cv2.applyColorMap(_heatmap_to_uint8(init), cv2.COLORMAP_JET))
        _imwrite_unicode(paths["first_step_heatmap"], cv2.applyColorMap(_heatmap_to_uint8(first), cv2.COLORMAP_JET))
        _imwrite_unicode(paths["final_heatmap"], cv2.applyColorMap(_heatmap_to_uint8(final), cv2.COLORMAP_JET))
        _imwrite_unicode(paths["target_heatmap"], cv2.applyColorMap(_heatmap_to_uint8(target), cv2.COLORMAP_JET))
        _imwrite_unicode(paths["init_noise_overlay"], cv2.cvtColor(_overlay_heatmap(image_rgb, init, gaze_xy, has_gaze), cv2.COLOR_RGB2BGR))
        _imwrite_unicode(paths["first_step_overlay"], cv2.cvtColor(_overlay_heatmap(image_rgb, first, gaze_xy, has_gaze), cv2.COLOR_RGB2BGR))
        _imwrite_unicode(paths["final_overlay"], cv2.cvtColor(_overlay_heatmap(image_rgb, final, gaze_xy, has_gaze), cv2.COLOR_RGB2BGR))
        _imwrite_unicode(paths["target_overlay"], cv2.cvtColor(_overlay_heatmap(image_rgb, target, gaze_xy, has_gaze), cv2.COLOR_RGB2BGR))
        _imwrite_unicode(
            paths["comparison"],
            cv2.cvtColor(
                _comparison_strip(image_rgb, init, first, final, target, gaze_xy, has_gaze),
                cv2.COLOR_RGB2BGR,
            ),
        )
        _imwrite_unicode(
            paths["overlay_comparison"],
            cv2.cvtColor(
                _overlay_comparison_strip(image_rgb, init, first, final, target, gaze_xy, has_gaze),
                cv2.COLOR_RGB2BGR,
            ),
        )
        if sample_idx == 0:
            for name, path in paths.items():
                (out_dir / path.name).write_bytes(path.read_bytes())

        sample_summaries.append(
            {
                "index": int(sample_idx),
                "dataset_index": int(selected_indices[sample_idx]),
                "gaze_xy": [float(v) for v in gaze_xy.tolist()],
                "has_gaze_label": has_gaze,
                "use_gaze_condition": bool(use_gaze_condition),
                "paths": {key: str(value) for key, value in paths.items()},
            }
        )

    timesteps = trace["timesteps"]
    temporal_mode = str(cfg.task.get("temporal_heatmap_mode", "off"))
    heatmap_label_source = (
        "temporal_window_dense_heatmap"
        if temporal_mode != "off"
        else "single_point_gaussian_from_gaze_xy"
    )
    summary = {
        "checkpoint": str(checkpoint),
        "source": source,
        "split": split,
        **ema_summary,
        "use_gaze_condition": bool(use_gaze_condition),
        "num_samples": len(sample_summaries),
        "camera_key": camera_key,
        "heatmap_objective": str(policy.heatmap_objective),
        "heatmap_label_source": heatmap_label_source,
        "temporal_heatmap": {
            "mode": temporal_mode,
            "window_radius": int(cfg.task.get("temporal_heatmap_window_radius", 0)),
            "beta": float(cfg.task.get("temporal_heatmap_beta", 0.0)),
            "sigma_px": float(cfg.task.get("temporal_heatmap_sigma_px", 0.0)),
            "current_weight": float(
                cfg.task.get("temporal_heatmap_current_weight", 1.0)
            ),
        },
        "heatmap_num_tokens": int(policy.heatmap_num_tokens),
        "heatmap_dim": int(policy.heatmap_dim),
        "heatmap_image_size": [int(v) for v in policy.heatmap_codec.image_size],
        "heatmap_prediction_mode": "iterative_denoise_trace",
        "denoise_state_columns": [
            "init_noise_decode",
            "after_scheduler_step_1",
            f"after_scheduler_step_{int(policy.num_inference_steps)}",
            "target_from_batch_heatmap_or_gaze_xy",
        ],
        "num_inference_steps": int(policy.num_inference_steps),
        "first_step_index": 1,
        "first_step_timestep": None if not timesteps else int(timesteps[0]),
        "scheduler_timesteps": timesteps,
        "heatmap_distribution_mode": str(policy.heatmap_distribution_mode),
        "heatmap_dsnt_temperature": float(policy.heatmap_dsnt_temperature),
        "heatmap_latent_scale": float(getattr(policy, "heatmap_latent_scale", 1.0)),
        "heatmap_latent_offset": float(getattr(policy, "heatmap_latent_offset", 0.0)),
        "heatmap_scheduler_clip_sample": getattr(policy, "heatmap_scheduler_clip_sample", None),
        "selected_indices": [int(idx) for idx in selected_indices],
        "sample_seed": None if sample_seed is None else int(sample_seed),
        "samples": sample_summaries,
        "paths": {
            "summary": str(out_dir / "summary.json"),
            "samples": [sample["paths"] for sample in sample_summaries],
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Save decoded intermediate heatmaps from the first step of Gaze-WAM iterative denoising."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Acknowledge that the dill checkpoint is trusted and may execute code.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", choices=["open", "robot"], default="open")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-gaze-condition", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sample-indices", default=None, help="Comma-separated dataset indices to preview.")
    parser.add_argument("--sample-seed", type=int, default=None, help="Random seed for heatmap latent noise.")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    args = parse_args(argv)
    summary = preview_gaze_wam_denoise_step(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        source=args.source,
        split=args.split,
        max_samples=args.max_samples,
        device=args.device,
        use_ema=args.use_ema,
        use_gaze_condition=args.use_gaze_condition,
        sample_indices=_parse_sample_indices(args.sample_indices),
        sample_seed=args.sample_seed,
        overrides=args.override,
        trust_checkpoint=args.trust_checkpoint,
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
