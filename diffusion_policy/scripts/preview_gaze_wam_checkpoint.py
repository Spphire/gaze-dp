from __future__ import annotations

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
from diffusion_policy.scripts.eval_gaze_wam_metrics import load_policy_for_eval
from diffusion_policy.model.gaze_wam.loss import normalize_spatial_distribution


def _imwrite_unicode(path: pathlib.Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Could not encode image for '{path}'.")
    encoded.tofile(str(path))


def _heatmap_to_uint8(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = heatmap - heatmap.min(initial=0.0)
    denom = heatmap.max(initial=0.0)
    if denom > 1e-12:
        heatmap = heatmap / denom
    return (heatmap * 255.0).round().astype(np.uint8)


def _latest_obs_rgb(batch, camera_key: str, sample_idx: int) -> np.ndarray:
    image = batch["obs"][camera_key][sample_idx, -1].detach().float().cpu().numpy()
    if image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image[:3], 0, -1)
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).round().astype(np.uint8)


def _draw_gaze(image_rgb: np.ndarray, gaze_xy, valid: bool) -> np.ndarray:
    out = image_rgb.copy()
    if not valid:
        return out
    h, w = out.shape[:2]
    x = int(round(float(gaze_xy[0]) * (w - 1)))
    y = int(round(float(gaze_xy[1]) * (h - 1)))
    cv2.drawMarker(
        out,
        (x, y),
        color=(255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=max(8, min(h, w) // 10),
        thickness=2,
    )
    cv2.circle(out, (x, y), radius=max(2, min(h, w) // 40), color=(255, 0, 0), thickness=-1)
    return out


def _overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, gaze_xy, valid: bool) -> np.ndarray:
    heat_u8 = _heatmap_to_uint8(heatmap)
    heat_color_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.55, heat_color_rgb, 0.45, 0.0)
    return _draw_gaze(overlay, gaze_xy=gaze_xy, valid=valid)


def _heatmap_panel(heatmap: np.ndarray, gaze_xy, valid: bool) -> np.ndarray:
    heat_u8 = _heatmap_to_uint8(heatmap)
    heat_color_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)
    return _draw_gaze(heat_color_rgb, gaze_xy=gaze_xy, valid=valid)


def _title_panel(image_rgb: np.ndarray, title: str) -> np.ndarray:
    out = image_rgb.copy()
    h, w = out.shape[:2]
    bar_h = max(22, min(32, h // 9))
    cv2.rectangle(out, (0, 0), (w, bar_h), color=(0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        str(title),
        (8, max(16, bar_h - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.45, min(0.65, h / 420.0)),
        (255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return out


def _comparison_strip(
    image_rgb: np.ndarray,
    pred_heatmap: np.ndarray,
    target_heatmap: np.ndarray,
    gaze_xy,
    valid: bool,
) -> np.ndarray:
    panels = [
        _title_panel(_draw_gaze(image_rgb, gaze_xy, valid), "RGB + gaze_xy"),
        _title_panel(_heatmap_panel(pred_heatmap, gaze_xy, valid), "Pred heatmap"),
        _title_panel(_overlay_heatmap(image_rgb, pred_heatmap, gaze_xy, valid), "Pred overlay"),
        _title_panel(_heatmap_panel(target_heatmap, gaze_xy, valid), "Target heatmap"),
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


def _instantiate_dataset(cfg, source: str, split: str):
    if source == "open":
        dataset = hydra.utils.instantiate(cfg.task.open_dataset)
    elif source == "robot":
        dataset = hydra.utils.instantiate(cfg.task.robot_dataset)
    else:
        raise ValueError("source must be 'open' or 'robot'.")
    if split == "val":
        dataset = dataset.get_validation_dataset()
    elif split != "train":
        raise ValueError("split must be 'train' or 'val'.")
    if len(dataset) <= 0:
        raise ValueError(f"{source} {split} dataset has zero samples.")
    return dataset


def _parse_sample_indices(value: Optional[str]) -> Optional[Sequence[int]]:
    if value is None:
        return None
    tokens = [token.strip() for token in str(value).split(",") if token.strip()]
    if not tokens:
        raise ValueError("--sample-indices must contain at least one integer index.")
    return [int(token) for token in tokens]


def _select_sample_indices(
    dataset_len: int,
    max_samples: int,
    sample_indices: Optional[Sequence[int]] = None,
    sample_seed: Optional[int] = None,
) -> Sequence[int]:
    if dataset_len <= 0:
        raise ValueError("dataset must contain at least one sample.")
    max_samples = max(1, int(max_samples))
    if sample_indices is not None:
        selected = [int(idx) for idx in sample_indices]
        if not selected:
            raise ValueError("sample_indices must contain at least one index.")
        selected = selected[:max_samples]
    elif sample_seed is not None:
        rng = np.random.default_rng(int(sample_seed))
        count = min(max_samples, dataset_len)
        selected = [int(idx) for idx in rng.choice(dataset_len, size=count, replace=False).tolist()]
    else:
        selected = list(range(min(max_samples, dataset_len)))
    for idx in selected:
        if idx < 0 or idx >= dataset_len:
            raise ValueError(f"sample index {idx} is out of range for dataset length {dataset_len}.")
    return selected


def _checkpoint_ema_summary(cfg, use_ema_requested: bool) -> Dict[str, bool]:
    checkpoint_has_ema = bool(OmegaConf.select(cfg, "training.use_ema", default=False))
    use_ema_effective = bool(use_ema_requested and checkpoint_has_ema)
    return {
        "use_ema": use_ema_effective,
        "use_ema_requested": bool(use_ema_requested),
        "checkpoint_has_ema": checkpoint_has_ema,
    }


def preview_gaze_wam_checkpoint(
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
        pred = policy.predict_heatmap(obs, decode=True, generator=generator)
        target_mask = obs["has_gaze_label"].to(device=policy.device)
        target_image = policy._target_heatmap_image_from_batch_or_xy(
            batch=batch,
            gaze_xy=batch["gaze_xy"].to(device=policy.device, dtype=policy.dtype),
            valid_mask=target_mask,
        )
        heatmap_is_cached = batch.get(
            "heatmap_is_cached",
            torch.zeros_like(target_mask),
        ).to(device=policy.device, dtype=torch.bool)
        cached_target_mask = target_mask & heatmap_is_cached
        if torch.any(cached_target_mask):
            cached_heatmap = batch["heatmap"].to(
                device=policy.device,
                dtype=policy.dtype,
            )
            if cached_heatmap.ndim == 4:
                if cached_heatmap.shape[1] != 1:
                    raise ValueError(
                        "Cached preview heatmap must have horizon 1, got "
                        f"{tuple(cached_heatmap.shape)}."
                    )
                cached_heatmap = cached_heatmap[:, 0]
            decoded_cached_target = policy._heatmap_tokens_to_spatial_image(
                cached_heatmap[cached_target_mask],
                "cached preview heatmap tokens",
            )
            decoded_cached_target = normalize_spatial_distribution(
                decoded_cached_target
            )
            target_image = target_image.clone()
            target_image[cached_target_mask] = decoded_cached_target.to(
                device=target_image.device,
                dtype=target_image.dtype,
            )

    pred_images = pred["heatmap_image"].detach().float().cpu().numpy()
    target_images = target_image.detach().float().cpu().numpy()
    gaze_xy_all = batch["gaze_xy"].detach().float().cpu().numpy()
    has_gaze_all = obs["has_gaze_label"].detach().cpu().numpy().astype(bool)
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_summaries = []
    legacy_paths = None
    camera_key = str(cfg.task.camera_key)
    for sample_idx in range(min(max_samples, pred_images.shape[0])):
        image_rgb = _latest_obs_rgb(batch, camera_key=camera_key, sample_idx=sample_idx)
        gaze_xy = gaze_xy_all[sample_idx]
        has_gaze = bool(has_gaze_all[sample_idx])
        pred_image = pred_images[sample_idx]
        target = target_images[sample_idx]
        sample_dir = out_dir / f"sample_{sample_idx:03d}"
        paths = {
            "rgb": sample_dir / "rgb.png",
            "pred_heatmap": sample_dir / "pred_heatmap.png",
            "target_heatmap": sample_dir / "target_heatmap.png",
            "pred_overlay": sample_dir / "pred_overlay.png",
            "target_overlay": sample_dir / "target_overlay.png",
            "comparison": sample_dir / "comparison.png",
        }
        _imwrite_unicode(paths["rgb"], cv2.cvtColor(_draw_gaze(image_rgb, gaze_xy, has_gaze), cv2.COLOR_RGB2BGR))
        _imwrite_unicode(paths["pred_heatmap"], cv2.applyColorMap(_heatmap_to_uint8(pred_image), cv2.COLORMAP_JET))
        _imwrite_unicode(paths["target_heatmap"], cv2.applyColorMap(_heatmap_to_uint8(target), cv2.COLORMAP_JET))
        _imwrite_unicode(paths["pred_overlay"], cv2.cvtColor(_overlay_heatmap(image_rgb, pred_image, gaze_xy, has_gaze), cv2.COLOR_RGB2BGR))
        _imwrite_unicode(paths["target_overlay"], cv2.cvtColor(_overlay_heatmap(image_rgb, target, gaze_xy, has_gaze), cv2.COLOR_RGB2BGR))
        _imwrite_unicode(paths["comparison"], cv2.cvtColor(_comparison_strip(image_rgb, pred_image, target, gaze_xy, has_gaze), cv2.COLOR_RGB2BGR))

        if sample_idx == 0:
            legacy_paths = {name: out_dir / path.name for name, path in paths.items()}
            for name, path in paths.items():
                legacy_paths[name].write_bytes(path.read_bytes())

        sample_summaries.append(
            {
                "index": int(sample_idx),
                "dataset_index": int(selected_indices[sample_idx]),
                "gaze_xy": [float(v) for v in gaze_xy.tolist()],
                "has_gaze_label": has_gaze,
                "use_gaze_condition": bool(use_gaze_condition),
                "source": source,
                "split": split,
                "pred_heatmap_shape": [int(v) for v in pred_image.shape],
                "target_heatmap_shape": [int(v) for v in target.shape],
                "pred_heatmap_sum": float(pred_image.sum()),
                "target_heatmap_sum": float(target.sum()),
                "target_source": (
                    "cached_latent_decode"
                    if bool(cached_target_mask[sample_idx].item())
                    else "online_temporal_or_xy"
                ),
                "paths": {key: str(value) for key, value in paths.items()},
            }
        )

    heatmap_objective = str(policy.heatmap_objective)
    latent_mse_loss = heatmap_objective != "dsnt_js"
    diffusion_final_heatmap_loss = bool(
        heatmap_objective == "diffusion"
        and getattr(policy, "heatmap_diffusion_final_loss_enabled", False)
    )
    heatmap_supervision = (
        "full_resolution_dsnt_plus_js_after_frozen_decoder"
        if heatmap_objective == "dsnt_js"
        else "latent_diffusion_mse_plus_decoded_final_heatmap_loss"
        if diffusion_final_heatmap_loss
        else "latent_diffusion_mse_against_frozen_cosmos_target"
    )
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
        "heatmap_objective": heatmap_objective,
        "heatmap_num_tokens": int(policy.heatmap_num_tokens),
        "heatmap_dim": int(policy.heatmap_dim),
        "heatmap_image_size": [int(v) for v in policy.heatmap_codec.image_size],
        "heatmap_prediction_mode": str(pred.get("heatmap_prediction_mode", "unknown")),
        "latent_mse_loss": bool(latent_mse_loss),
        "diffusion_final_heatmap_loss": diffusion_final_heatmap_loss,
        "heatmap_diffusion_final_loss_enabled": bool(
            getattr(policy, "heatmap_diffusion_final_loss_enabled", False)
        ),
        "heatmap_final_loss_timestep_weighting": str(
            getattr(policy, "heatmap_final_loss_timestep_weighting", "none")
        ),
        "heatmap_xy_loss_weight": float(
            getattr(policy, "heatmap_xy_loss_weight", 0.0)
        ),
        "heatmap_point_nll_loss_weight": float(
            getattr(policy, "heatmap_point_nll_loss_weight", 0.0)
        ),
        "heatmap_js_loss_weight": float(
            getattr(policy, "heatmap_js_loss_weight", 0.0)
        ),
        "heatmap_distribution_mode": str(
            getattr(policy, "heatmap_distribution_mode", "unknown")
        ),
        "heatmap_dsnt_temperature": float(
            getattr(policy, "heatmap_dsnt_temperature", 1.0)
        ),
        "heatmap_latent_scale": float(getattr(policy, "heatmap_latent_scale", 1.0)),
        "heatmap_latent_offset": float(getattr(policy, "heatmap_latent_offset", 0.0)),
        "heatmap_latent_stats_path": str(
            getattr(policy, "heatmap_latent_stats_path", "")
        ),
        "heatmap_scheduler_clip_sample": getattr(
            policy,
            "heatmap_scheduler_clip_sample",
            None,
        ),
        "heatmap_supervision": heatmap_supervision,
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
        "heatmap_decoder_output_interpretation": "decoded_intensity_distribution",
        "latent_mse_loss": latent_mse_loss,
        "num_inference_steps": int(policy.num_inference_steps),
        "selected_indices": [int(idx) for idx in selected_indices],
        "sample_seed": None if sample_seed is None else int(sample_seed),
        "samples": sample_summaries,
    }
    if sample_summaries:
        first = sample_summaries[0]
        summary.update(
            {
                "gaze_xy": first["gaze_xy"],
                "pred_heatmap_shape": first["pred_heatmap_shape"],
                "target_heatmap_shape": first["target_heatmap_shape"],
                "pred_heatmap_sum": first["pred_heatmap_sum"],
                "target_heatmap_sum": first["target_heatmap_sum"],
                "target_source": first["target_source"],
                "paths": {
                    **{key: str(value) for key, value in (legacy_paths or {}).items()},
                    "summary": str(out_dir / "summary.json"),
                    "samples": [sample["paths"] for sample in sample_summaries],
                },
            }
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Reload a Gaze-WAM checkpoint and save full-res heatmap previews.")
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
    parser.add_argument("--sample-seed", type=int, default=None, help="Random seed for diverse preview samples.")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    args = parse_args(argv)
    summary = preview_gaze_wam_checkpoint(
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
