#!/usr/bin/env python3
"""Measure multi-node NCCL all-reduce throughput under torchrun."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _summary(step_seconds: list[float], tensor_bytes: int, world_size: int) -> dict:
    mean_seconds = float(np.mean(step_seconds))
    algorithm_gbps = tensor_bytes / mean_seconds / 1e9
    bus_gbps = algorithm_gbps * 2.0 * (world_size - 1) / world_size
    return {
        "steps": len(step_seconds),
        "step_seconds": {
            "mean": mean_seconds,
            "p50": _percentile(step_seconds, 50),
            "p95": _percentile(step_seconds, 95),
            "max": max(step_seconds),
        },
        "algorithm_bandwidth_GBps": algorithm_gbps,
        "bus_bandwidth_GBps": bus_gbps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-mib", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.tensor_mib <= 0 or args.steps <= 0 or args.warmup_steps < 0:
        parser.error("tensor size and steps must be positive; warm-up must be nonnegative")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    started = time.perf_counter()
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=args.timeout_seconds),
    )
    initialization_seconds = time.perf_counter() - started

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tensor_bytes = args.tensor_mib * 1024 * 1024
    element_count = tensor_bytes // torch.tensor([], dtype=torch.float32).element_size()
    tensor = torch.empty(element_count, dtype=torch.float32, device=local_rank)
    expected = world_size * (world_size + 1) / 2

    def run_collective() -> float:
        tensor.fill_(rank + 1)
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        return time.perf_counter() - step_started

    for _ in range(args.warmup_steps):
        run_collective()

    local_step_seconds = [run_collective() for _ in range(args.steps)]
    if not torch.isclose(tensor[0], torch.tensor(expected, device=local_rank)):
        raise RuntimeError(f"all-reduce validation failed on rank {rank}")

    step_tensor = torch.tensor(local_step_seconds, dtype=torch.float64, device=local_rank)
    dist.all_reduce(step_tensor, op=dist.ReduceOp.MAX)
    if rank == 0:
        result = {
            "host": socket.gethostname(),
            "world_size": world_size,
            "tensor_mib": args.tensor_mib,
            "tensor_bytes": tensor_bytes,
            "warmup_steps": args.warmup_steps,
            "initialization_seconds_rank0": initialization_seconds,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": ".".join(str(part) for part in torch.cuda.nccl.version()),
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "NCCL_IB_DISABLE",
                    "NCCL_SOCKET_IFNAME",
                    "NCCL_IB_HCA",
                    "NCCL_IB_GID_INDEX",
                    "NCCL_NET_GDR_LEVEL",
                )
            },
            **_summary(step_tensor.cpu().tolist(), tensor_bytes, world_size),
        }
        payload = json.dumps(result, indent=2, sort_keys=True)
        print(payload, flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
