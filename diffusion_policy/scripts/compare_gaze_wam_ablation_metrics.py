import argparse
import csv
import json
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple

from diffusion_policy.common.gaze_wam_training_config import (
    normalize_gaze_wam_bool_field,
    normalize_gaze_wam_nonnegative_float_field,
)
from diffusion_policy.scripts.gaze_wam_provenance import add_provenance_contract


DEFAULT_VARIANTS = (
    "main=train_gaze_wam_workspace",
    "cached_dual_stream=train_gaze_wam_cached_dual_stream_workspace",
    "open_only=train_gaze_wam_open_only_workspace",
)


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available())


def _parse_variant(spec: str) -> Dict[str, Optional[str]]:
    parts = spec.split(":", maxsplit=1)
    left = parts[0]
    checkpoint = parts[1] if len(parts) == 2 and parts[1] else None
    if "=" in left:
        name, config_name = left.split("=", maxsplit=1)
    else:
        config_name = left
        name = left
    name = name.strip()
    config_name = config_name.strip()
    if not name or not config_name:
        raise ValueError(
            "Variant specs must be 'name=config' or 'name=config:checkpoint', "
            f"got {spec!r}."
        )
    return {
        "variant": name,
        "config_name": config_name,
        "checkpoint": checkpoint,
    }


def _variant_override_map(
    variant_overrides: Optional[Sequence[Tuple[str, str]]] = None,
) -> Dict[str, List[str]]:
    override_map: Dict[str, List[str]] = {}
    for pair in variant_overrides or []:
        if len(pair) != 2:
            raise ValueError(
                "variant_overrides must contain (variant, override) pairs."
            )
        variant, override = str(pair[0]).strip(), str(pair[1]).strip()
        if not variant or not override:
            raise ValueError(
                f"Invalid variant override pair {pair!r}; variant and override are required."
            )
        override_map.setdefault(variant, []).append(override)
    return override_map


def _flat_fieldnames(rows: Sequence[Dict[str, object]]) -> List[str]:
    preferred = [
        "variant",
        "config_name",
        "checkpoint",
        "checkpoint_provided",
        "global_overrides",
        "variant_overrides",
        "provenance_contract_version",
        "provenance_contract_id",
        "eval_sources",
        "eval_batch_size",
        "eval_max_batches",
        "eval_cfg_scale",
        "policy_cfg_scale",
        "effective_cfg_scale",
        "cfg_scale",
        "robot_batch_size",
        "open_batch_size",
        "robot_ratio",
        "open_ratio",
        "gradient_accumulate_every",
        "num_processes",
        "mixed_precision",
        "distributed_type",
        "effective_robot_batch_size_per_optimizer_step",
        "effective_open_batch_size_per_optimizer_step",
        "effective_train_batch_size_per_optimizer_step",
        "robot_gaze_dropout_prob",
        "robot_heatmap_on_gaze_dropout",
        "n_latency_steps",
        "robot_obs_downsample_steps",
        "robot_action_downsample_steps",
        "robot_action_padding",
        "open_obs_downsample_steps",
        "open_action_downsample_steps",
        "open_action_padding",
        "use_block_attention_mask",
        "heatmap_objective",
        "n_obs_steps",
        "action_horizon",
        "action_dim",
        "heatmap_num_tokens",
        "heatmap_token_grid",
        "image_shape",
        "image_resize_mode",
        "robot_image_resize_mode",
        "open_image_resize_mode",
        "obs_encoder_model_name",
        "obs_encoder_pretrained",
        "obs_encoder_checkpoint_path",
        "obs_encoder_checkpoint_path_exists",
        "obs_encoder_checkpoint_path_is_file",
        "obs_encoder_cache_dir",
        "obs_encoder_cache_dir_exists",
        "obs_encoder_cache_dir_is_dir",
        "obs_encoder_local_weight_source_configured",
        "obs_encoder_local_weight_source_valid",
    ]
    keys = set()
    for row in rows:
        keys.update(row.keys())
    return preferred + sorted(key for key in keys if key not in preferred)


def write_metrics_csv(rows: Sequence[Dict[str, object]], output_path: str) -> None:
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _flat_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _config_provenance(
    cfg,
    checkpoint: Optional[str],
    sources: Sequence[str],
    batch_size: int,
    max_batches: Optional[int],
    cfg_scale: Optional[float],
) -> Dict[str, object]:
    stamped_robot_batch = cfg.training.get("robot_batch_size_per_process", None)
    stamped_open_batch = cfg.training.get("open_batch_size_per_process", None)
    robot_batch = (
        int(stamped_robot_batch)
        if stamped_robot_batch is not None
        else int(cfg.robot_dataloader.batch_size)
    )
    open_batch = (
        int(stamped_open_batch)
        if stamped_open_batch is not None
        else int(cfg.open_dataloader.get("batch_size", 0))
    )
    total_batch = robot_batch + open_batch
    gradient_accumulate_every = int(cfg.training.get("gradient_accumulate_every", 1))
    num_processes = int(cfg.training.get("num_processes", 1))
    mixed_precision = str(cfg.training.get("mixed_precision", "no"))
    distributed_type = str(cfg.training.get("distributed_type", "NO"))
    stamped_effective_robot = cfg.training.get(
        "effective_robot_batch_size_per_optimizer_step",
        None,
    )
    stamped_effective_open = cfg.training.get(
        "effective_open_batch_size_per_optimizer_step",
        None,
    )
    stamped_effective_total = cfg.training.get(
        "effective_train_batch_size_per_optimizer_step",
        None,
    )
    effective_robot_batch = (
        int(stamped_effective_robot)
        if stamped_effective_robot is not None
        else robot_batch * num_processes * gradient_accumulate_every
    )
    effective_open_batch = (
        int(stamped_effective_open)
        if stamped_effective_open is not None
        else open_batch * num_processes * gradient_accumulate_every
    )
    effective_train_batch = (
        int(stamped_effective_total)
        if stamped_effective_total is not None
        else total_batch * num_processes * gradient_accumulate_every
    )
    policy_cfg_scale = normalize_gaze_wam_nonnegative_float_field(
        "policy.cfg_scale",
        cfg.policy.get("cfg_scale", 1.0),
        default=1.0,
    )
    eval_cfg_scale = (
        None
        if cfg_scale is None
        else normalize_gaze_wam_nonnegative_float_field(
            "eval cfg_scale",
            cfg_scale,
            default=policy_cfg_scale,
        )
    )
    effective_cfg_scale = eval_cfg_scale if eval_cfg_scale is not None else policy_cfg_scale
    robot_dataset_cfg = cfg.task.robot_dataset
    open_dataset_cfg = cfg.task.open_dataset
    image_resize_mode = str(cfg.task.get("image_resize_mode", "stretch"))
    obs_encoder_checkpoint_path = str(cfg.policy.obs_encoder.get("checkpoint_path", "") or "").strip()
    obs_encoder_cache_dir = str(cfg.policy.obs_encoder.get("cache_dir", "") or "").strip()
    obs_encoder_checkpoint_path_exists = bool(
        obs_encoder_checkpoint_path
    ) and pathlib.Path(obs_encoder_checkpoint_path).exists()
    obs_encoder_cache_dir_exists = bool(obs_encoder_cache_dir) and pathlib.Path(
        obs_encoder_cache_dir
    ).exists()
    obs_encoder_checkpoint_path_is_file = bool(
        obs_encoder_checkpoint_path
    ) and pathlib.Path(obs_encoder_checkpoint_path).is_file()
    obs_encoder_cache_dir_is_dir = bool(obs_encoder_cache_dir) and pathlib.Path(
        obs_encoder_cache_dir
    ).is_dir()
    obs_encoder_local_weight_source_configured = bool(
        obs_encoder_checkpoint_path or obs_encoder_cache_dir
    )
    obs_encoder_local_weight_source_valid = (
        (not obs_encoder_checkpoint_path or obs_encoder_checkpoint_path_is_file)
        and (not obs_encoder_cache_dir or obs_encoder_cache_dir_is_dir)
    )
    robot_heatmap_on_gaze_dropout = normalize_gaze_wam_bool_field(
        "task.robot_heatmap_on_gaze_dropout",
        cfg.task.get("robot_heatmap_on_gaze_dropout", True),
        default=True,
    )
    robot_action_padding = normalize_gaze_wam_bool_field(
        "task.robot_dataset.action_padding",
        robot_dataset_cfg.get("action_padding", True),
        default=True,
    )
    open_action_padding = normalize_gaze_wam_bool_field(
        "task.open_dataset.action_padding",
        open_dataset_cfg.get("action_padding", True),
        default=True,
    )
    use_block_attention_mask = normalize_gaze_wam_bool_field(
        "policy.use_block_attention_mask",
        cfg.policy.get("use_block_attention_mask", True),
        default=True,
    )
    obs_encoder_pretrained = normalize_gaze_wam_bool_field(
        "policy.obs_encoder.pretrained",
        cfg.policy.obs_encoder.get("pretrained", False),
        default=False,
    )
    return add_provenance_contract({
        "checkpoint_provided": bool(checkpoint),
        "eval_sources": ",".join(str(source) for source in sources),
        "eval_batch_size": int(batch_size),
        "eval_max_batches": "" if max_batches is None else int(max_batches),
        "eval_cfg_scale": "" if eval_cfg_scale is None else eval_cfg_scale,
        "policy_cfg_scale": policy_cfg_scale,
        "effective_cfg_scale": effective_cfg_scale,
        "cfg_scale": policy_cfg_scale,
        "robot_batch_size": robot_batch,
        "open_batch_size": open_batch,
        "robot_ratio": float(robot_batch / total_batch) if total_batch > 0 else 0.0,
        "open_ratio": float(open_batch / total_batch) if total_batch > 0 else 0.0,
        "gradient_accumulate_every": gradient_accumulate_every,
        "num_processes": num_processes,
        "mixed_precision": mixed_precision,
        "distributed_type": distributed_type,
        "effective_robot_batch_size_per_optimizer_step": effective_robot_batch,
        "effective_open_batch_size_per_optimizer_step": effective_open_batch,
        "effective_train_batch_size_per_optimizer_step": effective_train_batch,
        "robot_gaze_dropout_prob": float(cfg.task.get("robot_gaze_dropout_prob", 0.0)),
        "robot_heatmap_on_gaze_dropout": robot_heatmap_on_gaze_dropout,
        "n_latency_steps": int(cfg.task.get("n_latency_steps", 0)),
        "robot_obs_downsample_steps": int(robot_dataset_cfg.get("obs_downsample_steps", 1)),
        "robot_action_downsample_steps": int(robot_dataset_cfg.get("action_downsample_steps", 1)),
        "robot_action_padding": robot_action_padding,
        "open_obs_downsample_steps": int(open_dataset_cfg.get("obs_downsample_steps", 1)),
        "open_action_downsample_steps": int(open_dataset_cfg.get("action_downsample_steps", 1)),
        "open_action_padding": open_action_padding,
        "use_block_attention_mask": use_block_attention_mask,
        "heatmap_objective": str(cfg.policy.get("heatmap_objective", "diffusion")),
        "n_obs_steps": int(cfg.task.n_obs_steps),
        "action_horizon": int(cfg.task.action_horizon),
        "action_dim": int(cfg.task.action_dim),
        "heatmap_num_tokens": int(cfg.task.heatmap_num_tokens),
        "heatmap_token_grid": "x".join(str(int(v)) for v in cfg.task.heatmap_token_grid),
        "image_shape": "x".join(str(int(v)) for v in cfg.task.image_shape),
        "image_resize_mode": image_resize_mode,
        "robot_image_resize_mode": str(
            robot_dataset_cfg.get("image_resize_mode", image_resize_mode)
        ),
        "open_image_resize_mode": str(
            open_dataset_cfg.get("image_resize_mode", image_resize_mode)
        ),
        "obs_encoder_model_name": str(cfg.policy.obs_encoder.model_name),
        "obs_encoder_pretrained": obs_encoder_pretrained,
        "obs_encoder_checkpoint_path": obs_encoder_checkpoint_path,
        "obs_encoder_checkpoint_path_exists": obs_encoder_checkpoint_path_exists,
        "obs_encoder_checkpoint_path_is_file": obs_encoder_checkpoint_path_is_file,
        "obs_encoder_cache_dir": obs_encoder_cache_dir,
        "obs_encoder_cache_dir_exists": obs_encoder_cache_dir_exists,
        "obs_encoder_cache_dir_is_dir": obs_encoder_cache_dir_is_dir,
        "obs_encoder_local_weight_source_configured": obs_encoder_local_weight_source_configured,
        "obs_encoder_local_weight_source_valid": obs_encoder_local_weight_source_valid,
    })


def compare_gaze_wam_ablation_metrics(
    variants: Sequence[str] = DEFAULT_VARIANTS,
    overrides: Optional[Sequence[str]] = None,
    variant_overrides: Optional[Sequence[Tuple[str, str]]] = None,
    device: str = "cpu",
    batch_size: int = 16,
    num_workers: int = 0,
    max_batches: Optional[int] = None,
    sources: Sequence[str] = ("robot", "open"),
    cfg_scale: Optional[float] = None,
    seed: int = 42,
    use_ema: bool = True,
    compute_denoising_loss: bool = True,
    compute_sampling: bool = True,
    compute_heatmap: bool = True,
    compute_gdr: bool = True,
    validate_zarr: bool = True,
    timestamp_key: Optional[str] = None,
    require_timestamps: bool = False,
    timestamp_max_delta: Optional[float] = None,
    timestamp_max_step: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Evaluate multiple Gaze-WAM configs/checkpoints and return table-like rows."""
    import torch
    from diffusion_policy.scripts.eval_gaze_wam_metrics import (
        evaluate_gaze_wam_sources,
        load_cfg,
        load_policy_for_eval,
    )

    torch.manual_seed(seed)
    rows: List[Dict[str, object]] = []
    global_overrides = list(overrides or [])
    per_variant_overrides = _variant_override_map(variant_overrides)
    parsed_variants = [_parse_variant(spec) for spec in variants]
    variant_names = {str(parsed["variant"]) for parsed in parsed_variants}
    unknown_override_variants = sorted(set(per_variant_overrides) - variant_names)
    if unknown_override_variants:
        raise ValueError(
            "Variant-specific overrides refer to unknown variant(s): "
            + ", ".join(unknown_override_variants)
        )
    for parsed in parsed_variants:
        variant_specific_overrides = per_variant_overrides.get(parsed["variant"], [])
        merged_overrides = list(global_overrides) + list(variant_specific_overrides)
        cfg = (
            None
            if parsed["checkpoint"] is not None
            else load_cfg(parsed["config_name"], overrides=merged_overrides)
        )
        policy, cfg = load_policy_for_eval(
            cfg=cfg,
            checkpoint=parsed["checkpoint"],
            device=device,
            use_ema=use_ema,
            overrides=merged_overrides,
        )
        metrics = evaluate_gaze_wam_sources(
            policy=policy,
            cfg=cfg,
            sources=sources,
            batch_size=batch_size,
            num_workers=num_workers,
            max_batches=max_batches,
            device=device,
            cfg_scale=cfg_scale,
            compute_denoising_loss=compute_denoising_loss,
            compute_sampling=compute_sampling,
            compute_heatmap=compute_heatmap,
            compute_gdr=compute_gdr,
            validate_zarr=validate_zarr,
            timestamp_key=timestamp_key,
            require_timestamps=require_timestamps,
            timestamp_max_delta=timestamp_max_delta,
            timestamp_max_step=timestamp_max_step,
            robot_gaze_dropout_seed=seed,
        )
        provenance = _config_provenance(
            cfg=cfg,
            checkpoint=parsed["checkpoint"],
            sources=sources,
            batch_size=batch_size,
            max_batches=max_batches,
            cfg_scale=cfg_scale,
        )
        rows.append(
            {
                "variant": parsed["variant"],
                "config_name": parsed["config_name"],
                "checkpoint": parsed["checkpoint"] or "",
                "global_overrides": " ".join(global_overrides),
                "variant_overrides": " ".join(variant_specific_overrides),
                **provenance,
                **metrics,
            }
        )
    return rows


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Compare offline metrics across Gaze-WAM ablation configs/checkpoints."
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help=(
            "Variant spec. Use 'name=config' or 'name=config:checkpoint'. "
            "Repeat for multiple variants. Defaults to "
            "main/cached_dual_stream/open_only."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override applied to every variant. Repeat for multiple overrides.",
    )
    parser.add_argument(
        "--variant-override",
        action="append",
        nargs=2,
        metavar=("VARIANT", "OVERRIDE"),
        default=[],
        help=(
            "Hydra override applied only to one variant name. Repeat for multiple overrides. "
            "Useful for checkpoint-free sweep eval plans."
        ),
    )
    parser.add_argument("--device", default="cuda:0" if _torch_cuda_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--sources", default="robot,open")
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="Optional global CFG scale override. Defaults to each policy config's cfg_scale.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-denoising-loss", action="store_true")
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-heatmap", action="store_true")
    parser.add_argument("--skip-gdr", action="store_true")
    parser.add_argument("--validate-zarr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timestamp-key", default=None)
    parser.add_argument("--require-timestamps", action="store_true")
    parser.add_argument("--timestamp-max-delta", type=float, default=None)
    parser.add_argument("--timestamp-max-step", type=float, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    args = parse_args(argv)
    variants = args.variant if args.variant else list(DEFAULT_VARIANTS)
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    rows = compare_gaze_wam_ablation_metrics(
        variants=variants,
        overrides=args.override,
        variant_overrides=args.variant_override,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        sources=sources,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        use_ema=args.use_ema,
        compute_denoising_loss=not args.skip_denoising_loss,
        compute_sampling=not args.skip_sampling,
        compute_heatmap=not args.skip_heatmap,
        compute_gdr=not args.skip_gdr,
        validate_zarr=args.validate_zarr,
        timestamp_key=args.timestamp_key,
        require_timestamps=args.require_timestamps,
        timestamp_max_delta=args.timestamp_max_delta,
        timestamp_max_step=args.timestamp_max_step,
    )
    print(json.dumps(rows, indent=2, sort_keys=True))
    if args.output_json is not None:
        path = pathlib.Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_csv is not None:
        write_metrics_csv(rows, args.output_csv)
    return rows


if __name__ == "__main__":
    main()
