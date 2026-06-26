from __future__ import annotations

import argparse
import json
from typing import Sequence, Tuple

import torch

from diffusion_policy.model.gaze_wam.cached_dual_stream_transformer import (
    CachedDualStreamGazeWamTransformer,
)
from diffusion_policy.model.gaze_wam.heatmap_codec import HeatmapTokenCodec


def _positive_int_pair(name: str, value: Sequence[int]) -> Tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain two integers, got {value}.")
    parsed = tuple(int(v) for v in value)
    if parsed[0] <= 0 or parsed[1] <= 0:
        raise ValueError(f"{name} values must be positive, got {value}.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the cached dual-stream Gaze-WAM DiT and latent-to-full-res "
            "heatmap codec."
        )
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-tokens", type=int, default=8)
    parser.add_argument("--image-size", type=int, nargs=2, default=[8, 8])
    parser.add_argument("--heatmap-token-grid", type=int, nargs=2, default=[2, 2])
    parser.add_argument("--heatmap-dim", type=int, default=16)
    parser.add_argument("--action-horizon", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=10)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-emb", type=int, default=32)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for the smoke run, for example cpu, cuda, or cuda:0.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _resolve_device(device: str) -> torch.device:
    if device == "cuda":
        device = "cuda:0"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False.")
    return resolved


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)

    image_size = _positive_int_pair("image_size", args.image_size)
    heatmap_token_grid = _positive_int_pair("heatmap_token_grid", args.heatmap_token_grid)
    codec = HeatmapTokenCodec(
        token_grid=heatmap_token_grid,
        image_size=image_size,
        sigma_tokens=1.25,
    )
    heatmap_dim = int(args.heatmap_dim)
    heatmap_num_tokens = codec.num_tokens

    heatmap_tokens = torch.randn(
        args.batch_size,
        heatmap_num_tokens,
        heatmap_dim,
        device=device,
    )

    model = CachedDualStreamGazeWamTransformer(
        action_dim=args.action_dim,
        heatmap_dim=heatmap_dim,
        action_horizon=args.action_horizon,
        heatmap_num_tokens=heatmap_num_tokens,
        max_image_tokens=args.image_tokens,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_emb=args.n_emb,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
    ).to(device)
    image_tokens = torch.randn(args.batch_size, args.image_tokens, args.n_emb, device=device)
    gaze_token = torch.randn(args.batch_size, 1, args.n_emb, device=device)
    noisy_action = torch.randn(
        args.batch_size,
        args.action_horizon,
        args.action_dim,
        device=device,
    )
    noisy_heatmap = torch.randn(args.batch_size, heatmap_num_tokens, heatmap_dim, device=device)
    timestep = torch.arange(args.batch_size, dtype=torch.long, device=device) % 10

    world_cache = model.prefill_world_cache(
        image_tokens=image_tokens,
        gaze_token=gaze_token,
    )
    train_out = model(
        image_tokens=image_tokens,
        gaze_token=gaze_token,
        noisy_action=noisy_action,
        noisy_heatmap=noisy_heatmap,
        timestep=timestep,
        world_cache=world_cache,
    )
    infer_out = model(
        image_tokens=image_tokens,
        gaze_token=gaze_token,
        noisy_action=noisy_action,
        noisy_heatmap=None,
        timestep=timestep,
        is_inference=True,
        world_cache=world_cache,
    )

    summary = {
        "ok": True,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "batch_size": int(args.batch_size),
        "image_tokens": int(args.image_tokens),
        "world_cache_layers": len(world_cache.key_values),
        "world_cache_key_shape": [int(v) for v in world_cache.key_values[0][0].shape],
        "action_shape": [int(v) for v in train_out.action.shape],
        "heatmap_token_shape": [int(v) for v in train_out.heatmap.shape],
        "inference_action_shape": [int(v) for v in infer_out.action.shape],
        "inference_heatmap_is_none": infer_out.heatmap is None,
        "attention_contract": model.attention_contract_summary(
            num_image_tokens=args.image_tokens,
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
