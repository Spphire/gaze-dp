"""Offline probe for Gaze-WAM gaze attention and output sensitivity.

This intentionally runs outside the training validation loop.  It keeps the
image and diffusion seeds fixed while comparing image-only, recorded-gaze, and
arbitrary-gaze conditions.  The active independent-attention model has one gaze
source token, so its gaze softmax is one by construction; the gaze K/V ablation
is the causal-use check.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from diffusion_policy.common.omegaconf_resolvers import register_safe_omegaconf_resolvers
from diffusion_policy.model.gaze_wam.cached_dual_stream_transformer import (
    CachedMixedAttention,
)
from diffusion_policy.scripts.eval_gaze_wam_metrics import load_policy_for_eval

register_safe_omegaconf_resolvers()


def _move(value, device):
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if torch.is_tensor(value):
        return value.to(device=device)
    return value


def _clone_obs(obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in obs.items()}


def _obs_from_batch(batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
    obs = _clone_obs(batch["obs"])
    obs["gaze_xy"] = batch["gaze_xy"]
    obs["has_gaze_condition"] = batch.get("has_gaze_condition", batch["has_gaze_label"])
    obs["has_gaze_label"] = batch["has_gaze_label"]
    obs["use_gaze_condition"] = batch["use_gaze_condition"]
    return obs


def _condition_obs(
    base: Dict[str, torch.Tensor],
    mode: str,
    gaze_xy: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    obs = _clone_obs(base)
    batch_size = int(gaze_xy.shape[0])
    if mode == "image_only":
        obs["gaze_xy"] = torch.zeros_like(gaze_xy)
        obs["has_gaze_condition"] = torch.zeros(batch_size, device=gaze_xy.device, dtype=torch.bool)
        obs["has_gaze_label"] = torch.zeros(batch_size, device=gaze_xy.device, dtype=torch.bool)
        obs["use_gaze_condition"] = torch.zeros(batch_size, device=gaze_xy.device, dtype=torch.bool)
    elif mode in ("gt_gaze", "arbitrary_gaze"):
        obs["gaze_xy"] = gaze_xy
        obs["has_gaze_condition"] = torch.ones(batch_size, device=gaze_xy.device, dtype=torch.bool)
        obs["has_gaze_label"] = torch.ones(batch_size, device=gaze_xy.device, dtype=torch.bool)
        obs["use_gaze_condition"] = torch.ones(batch_size, device=gaze_xy.device, dtype=torch.bool)
    else:
        raise ValueError(f"Unsupported conditioning mode: {mode}")
    return obs


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_commit(path: Path) -> Optional[str]:
    """Return the checked-out code revision without making Git a runtime dependency."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _heatmap_to_hw(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().float()
    if value.ndim == 4:
        if value.shape[1] != 1:
            raise ValueError(f"Expected one heatmap channel, got {tuple(value.shape)}")
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError(f"Expected [B,H,W] heatmap, got {tuple(value.shape)}")
    return value


def _peak_xy(heatmap: torch.Tensor) -> torch.Tensor:
    heatmap = _heatmap_to_hw(heatmap)
    batch_size, height, width = heatmap.shape
    flat_idx = heatmap.reshape(batch_size, -1).argmax(dim=-1)
    row = torch.div(flat_idx, width, rounding_mode="floor")
    col = flat_idx.remainder(width)
    return torch.stack(
        [
            (col.float() + 0.5) / float(width),
            (row.float() + 0.5) / float(height),
        ],
        dim=-1,
    )


def _l2_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(left.detach().float() - right.detach().float(), dim=-1)


def _capture_forward_factory(branch_map, records, state):
    def probe_forward(
        self,
        target_tokens,
        image_cache_key,
        image_cache_value,
        gaze_cache_key,
        gaze_cache_value,
    ):
        query = self._reshape_heads(self.query(target_tokens))
        target_key = self._reshape_heads(self.key(target_tokens))
        target_value = self._reshape_heads(self.value(target_tokens))

        def attend(key, value, dropout):
            attn = F.softmax(torch.matmul(query, key.transpose(-2, -1)) * self.scale, dim=-1)
            context = torch.matmul(dropout(attn), value)
            return attn, context

        target_attn, target_context = attend(target_key, target_value, self.attn_drop)
        image_attn, image_context = attend(
            image_cache_key,
            image_cache_value,
            self.attn_drop,
        )
        gaze_attn, gaze_context = attend(
            gaze_cache_key,
            gaze_cache_value,
            self.gaze_attn_drop,
        )
        if state["disable_gaze"]:
            gaze_context = torch.zeros_like(gaze_context)

        context_norms = {
            "target": float(target_context.detach().float().norm(dim=-1).mean().item()),
            "image": float(image_context.detach().float().norm(dim=-1).mean().item()),
            "gaze": float(gaze_context.detach().float().norm(dim=-1).mean().item()),
        }
        context_norm_total = sum(context_norms.values())
        context_ratios = {
            f"{name}_context_ratio": (
                value / context_norm_total if context_norm_total > 0.0 else 0.0
            )
            for name, value in context_norms.items()
        }

        # The final projection is shared, so measure each branch through its
        # weight without repeatedly adding the projection bias.
        def projected_norm(context):
            projected = F.linear(
                self._merge_heads(context),
                self.out.weight,
                bias=None,
            )
            return float(projected.detach().float().norm(dim=-1).mean().item())

        projected_norms = {
            "target": projected_norm(target_context),
            "image": projected_norm(image_context),
            "gaze": projected_norm(gaze_context),
        }
        projected_norm_total = sum(projected_norms.values())
        projected_ratios = {
            f"{name}_projected_ratio": (
                value / projected_norm_total if projected_norm_total > 0.0 else 0.0
            )
            for name, value in projected_norms.items()
        }

        branch, layer = branch_map.get(id(self), ("unknown", -1))
        records.append(
            {
                "condition": state["condition"],
                "ablation": bool(state["disable_gaze"]),
                "branch": branch,
                "layer": int(layer),
                "gaze_weight_mean": float(gaze_attn.detach().float().mean().item()),
                "gaze_weight_min": float(gaze_attn.detach().float().amin().item()),
                "gaze_weight_max": float(gaze_attn.detach().float().amax().item()),
                "gaze_weight_by_head": gaze_attn.detach()
                .float()
                .mean(dim=(0, 2, 3))
                .cpu()
                .tolist(),
                "gaze_context_norm": float(gaze_context.detach().float().norm(dim=-1).mean().item()),
                "target_context_norm": context_norms["target"],
                "image_context_norm": context_norms["image"],
                "context_norm_total": context_norm_total,
                **context_ratios,
                "target_projected_norm": projected_norms["target"],
                "image_projected_norm": projected_norms["image"],
                "gaze_projected_norm": projected_norms["gaze"],
                "projected_norm_total": projected_norm_total,
                **projected_ratios,
                "image_weight_mean": float(image_attn.detach().float().mean().item()),
                "image_weight_by_head": image_attn.detach()
                .float()
                .mean(dim=(0, 2, 3))
                .cpu()
                .tolist(),
                "target_weight_mean": float(target_attn.detach().float().mean().item()),
                "target_weight_by_head": target_attn.detach()
                .float()
                .mean(dim=(0, 2, 3))
                .cpu()
                .tolist(),
                "target_key_count": int(target_key.shape[2]),
                "image_key_count": int(image_cache_key.shape[2]),
                "gaze_key_count": int(gaze_cache_key.shape[2]),
            }
        )
        fused = target_context + image_context + gaze_context
        return self.resid_drop(self.out(self._merge_heads(fused)))

    return probe_forward


def _aggregate_attention(records: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    for record in records:
        key = (record["condition"], record["ablation"], record["branch"], record["layer"])
        grouped.setdefault(key, []).append(record)
    result = []
    for (condition, ablation, branch, layer), rows in sorted(grouped.items(), key=str):
        row = {
            "condition": condition,
            "ablation": bool(ablation),
            "branch": branch,
            "layer": int(layer),
            "calls": len(rows),
        }
        for name in (
            "gaze_weight_mean",
            "gaze_weight_min",
            "gaze_weight_max",
            "gaze_context_norm",
            "target_context_norm",
            "image_context_norm",
            "context_norm_total",
            "target_context_ratio",
            "image_context_ratio",
            "gaze_context_ratio",
            "target_projected_norm",
            "image_projected_norm",
            "gaze_projected_norm",
            "projected_norm_total",
            "target_projected_ratio",
            "image_projected_ratio",
            "gaze_projected_ratio",
            "image_weight_mean",
            "target_weight_mean",
        ):
            row[name] = float(np.mean([float(item[name]) for item in rows]))
        for name in (
            "gaze_weight_by_head",
            "image_weight_by_head",
            "target_weight_by_head",
        ):
            head_values = np.asarray([item[name] for item in rows], dtype=np.float64)
            row[name] = head_values.mean(axis=0).tolist()
        row["image_key_count"] = int(rows[0]["image_key_count"])
        row["gaze_key_count"] = int(rows[0]["gaze_key_count"])
        row["target_key_count"] = int(rows[0]["target_key_count"])
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", choices=("robot", "open"), default="robot")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trust-checkpoint", action="store_true")
    parser.add_argument(
        "--heatmap-mode",
        choices=("iterative", "single_step"),
        default="iterative",
        help="Heatmap evaluation mode. Iterative is the final-output diagnostic.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.count <= 0:
        raise ValueError("--count must be positive")

    torch.manual_seed(42)
    device = torch.device(args.device)
    policy, cfg = load_policy_for_eval(
        checkpoint=str(checkpoint_path),
        device=args.device,
        use_ema=True,
        trust_checkpoint=args.trust_checkpoint,
    )
    policy.eval()

    dataset_cfg = OmegaConf.create(
        OmegaConf.to_container(
            cfg.task.robot_dataset if args.source == "robot" else cfg.task.open_dataset,
            resolve=True,
        )
    )
    dataset = hydra.utils.instantiate(dataset_cfg)
    indices = list(range(args.start_index, args.start_index + args.count))
    batch = next(
        iter(
            DataLoader(
                Subset(dataset, indices),
                batch_size=args.count,
                shuffle=False,
                num_workers=0,
            )
        )
    )
    batch = _move(batch, device)
    base_obs = _obs_from_batch(batch)
    real_gaze = batch["gaze_xy"].to(device=device, dtype=policy.dtype)
    arbitrary_gaze = (1.0 - real_gaze).clamp(0.05, 0.95)

    branch_map = {}
    for layer, block in enumerate(policy.model.action_blocks):
        branch_map[id(block.mixed_attn)] = ("action", layer)
    for layer, block in enumerate(policy.model.heatmap_blocks):
        branch_map[id(block.mixed_attn)] = ("heatmap", layer)

    records = []
    state = {"condition": "", "disable_gaze": False}
    original_forward = CachedMixedAttention.forward
    CachedMixedAttention.forward = _capture_forward_factory(branch_map, records, state)
    predictions = {}
    heatmap_tensors = {}
    try:
        for mode, gaze in (
            ("image_only", real_gaze),
            ("gt_gaze", real_gaze),
            ("arbitrary_gaze", arbitrary_gaze),
        ):
            state["condition"] = mode
            state["disable_gaze"] = False
            obs = _condition_obs(base_obs, mode, gaze)
            with torch.no_grad():
                torch.manual_seed(1234)
                action = policy.predict_action(
                    obs,
                    cfg_scale=1.0,
                )["action_pred_relative"]
                heatmap_generator = torch.Generator(device=device).manual_seed(1234)
                heatmap_result = policy.predict_heatmap(
                    obs,
                    timestep=(
                        None
                        if args.heatmap_mode == "iterative"
                        else torch.zeros(args.count, device=device, dtype=torch.long)
                    ),
                    decode=True,
                    generator=heatmap_generator,
                )
            peak = _peak_xy(heatmap_result["heatmap_image"])
            heatmap_tensors[mode] = _heatmap_to_hw(
                heatmap_result["heatmap_image"]
            ).detach().float().cpu()
            predictions[mode] = {
                "gaze_xy": gaze.detach().float().cpu().tolist(),
                "peak_xy": peak.detach().float().cpu().tolist(),
                "action": action.detach().float().cpu().tolist(),
            }

        state["condition"] = "gt_gaze"
        state["disable_gaze"] = True
        with torch.no_grad():
            torch.manual_seed(1234)
            ablated_action = policy.predict_action(
                _condition_obs(base_obs, "gt_gaze", real_gaze),
                cfg_scale=1.0,
            )["action_pred_relative"]
            heatmap_generator = torch.Generator(device=device).manual_seed(1234)
            ablated_heatmap = policy.predict_heatmap(
                _condition_obs(base_obs, "gt_gaze", real_gaze),
                timestep=(
                    None
                    if args.heatmap_mode == "iterative"
                    else torch.zeros(args.count, device=device, dtype=torch.long)
                ),
                decode=True,
                generator=heatmap_generator,
            )
        heatmap_tensors["gt_gaze_gaze_kv_zero"] = _heatmap_to_hw(
            ablated_heatmap["heatmap_image"]
        ).detach().float().cpu()
        predictions["gt_gaze_gaze_kv_zero"] = {
            "peak_xy": _peak_xy(ablated_heatmap["heatmap_image"]).detach().float().cpu().tolist(),
            "action": ablated_action.detach().float().cpu().tolist(),
        }
    finally:
        CachedMixedAttention.forward = original_forward

    action_tensors = {
        key: torch.tensor(value["action"], dtype=torch.float32)
        for key, value in predictions.items()
        if "action" in value
    }
    peak_tensors = {
        key: torch.tensor(value["peak_xy"], dtype=torch.float32)
        for key, value in predictions.items()
        if "peak_xy" in value
    }
    heatmap_l1 = {
        "heatmap_l1_gt_vs_image_only_mean": float(
            (heatmap_tensors["gt_gaze"] - heatmap_tensors["image_only"])
            .abs()
            .mean(dim=(1, 2))
            .mean()
            .item()
        ),
        "heatmap_l1_arbitrary_vs_gt_mean": float(
            (heatmap_tensors["arbitrary_gaze"] - heatmap_tensors["gt_gaze"])
            .abs()
            .mean(dim=(1, 2))
            .mean()
            .item()
        ),
        "heatmap_l1_gaze_kv_zero_mean": float(
            (heatmap_tensors["gt_gaze"] - heatmap_tensors["gt_gaze_gaze_kv_zero"])
            .abs()
            .mean(dim=(1, 2))
            .mean()
            .item()
        ),
    }
    metrics = {
        "heatmap_peak_to_gt_gaze_mean": float(_l2_rows(peak_tensors["gt_gaze"], torch.tensor(predictions["gt_gaze"]["gaze_xy"])).mean().item()),
        "heatmap_peak_to_arbitrary_gaze_mean": float(_l2_rows(peak_tensors["arbitrary_gaze"], torch.tensor(predictions["arbitrary_gaze"]["gaze_xy"])).mean().item()),
        "heatmap_peak_shift_gt_vs_image_only_mean": float(_l2_rows(peak_tensors["gt_gaze"], peak_tensors["image_only"]).mean().item()),
        "heatmap_peak_shift_arbitrary_vs_gt_mean": float(_l2_rows(peak_tensors["arbitrary_gaze"], peak_tensors["gt_gaze"]).mean().item()),
        "action_shift_gt_vs_image_only_mean": float(_l2_rows(action_tensors["gt_gaze"].reshape(args.count, -1), action_tensors["image_only"].reshape(args.count, -1)).mean().item()),
        "action_shift_arbitrary_vs_gt_mean": float(_l2_rows(action_tensors["arbitrary_gaze"].reshape(args.count, -1), action_tensors["gt_gaze"].reshape(args.count, -1)).mean().item()),
        "action_shift_gaze_kv_zero_mean": float(_l2_rows(action_tensors["gt_gaze"].reshape(args.count, -1), action_tensors["gt_gaze_gaze_kv_zero"].reshape(args.count, -1)).mean().item()),
        "heatmap_shift_gaze_kv_zero_mean": float(_l2_rows(peak_tensors["gt_gaze"], peak_tensors["gt_gaze_gaze_kv_zero"]).mean().item()),
    }
    recorded_gaze = torch.tensor(
        predictions["gt_gaze"]["gaze_xy"],
        dtype=torch.float32,
    )
    for mode in ("image_only", "gt_gaze", "arbitrary_gaze"):
        metrics[f"heatmap_peak_to_recorded_gaze_{mode}_mean"] = float(
            _l2_rows(peak_tensors[mode], recorded_gaze).mean().item()
        )
        metrics[f"heatmap_peak_to_supplied_gaze_{mode}_mean"] = float(
            _l2_rows(
                peak_tensors[mode],
                torch.tensor(predictions[mode]["gaze_xy"], dtype=torch.float32),
            )
            .mean()
            .item()
        )
    metrics.update(heatmap_l1)

    repository_commit = _repository_commit(Path(__file__).resolve().parents[1])
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _hash_file(checkpoint_path),
        "source": args.source,
        "indices": indices,
        "seed": 42,
        "action_seed": 1234,
        "heatmap_mode": args.heatmap_mode,
        # The checkpoint embeds the resolved Hydra config; this revision identifies
        # the repository config/model contract used to interpret that payload.
        "code_commit": repository_commit,
        "config_commit": repository_commit,
        "architecture": "cached_dual_stream_independent_image_gaze_target_attention",
        "metrics": metrics,
        "predictions": predictions,
        "attention_by_layer": _aggregate_attention(records),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
