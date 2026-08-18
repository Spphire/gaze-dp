#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


def _percentile(values, percentile):
    values = sorted(float(value) for value in values)
    if not values:
        raise ValueError("Cannot compute a percentile from an empty sequence.")
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * float(percentile) / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _load_json_lines(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
    return rows


def summarize_benchmark(log_path):
    log_path = Path(log_path).resolve()
    rows = [
        row
        for row in _load_json_lines(log_path)
        if "perf_step_seconds_max" in row
    ]
    if not rows:
        raise ValueError(f"No performance rows found in {log_path}.")

    steady_rows = [row for row in rows if not bool(row["perf_is_warmup"])]
    if not steady_rows:
        raise ValueError(
            "No steady-state rows remain after excluding performance warm-up steps."
        )

    batch_sizes = {int(row["perf_effective_batch_size"]) for row in rows}
    if len(batch_sizes) != 1:
        raise ValueError(
            f"Performance rows use inconsistent effective batch sizes: {batch_sizes}."
        )
    effective_batch_size = batch_sizes.pop()
    step_seconds = [float(row["perf_step_seconds_max"]) for row in steady_rows]
    throughputs = [
        float(row["perf_effective_samples_per_second"])
        for row in steady_rows
    ]
    total_seconds = sum(step_seconds)

    summary = {
        "log_path": str(log_path),
        "measured_steps": len(rows),
        "warmup_steps": len(rows) - len(steady_rows),
        "steady_steps": len(steady_rows),
        "effective_batch_size": effective_batch_size,
        "steady_total_seconds": total_seconds,
        "steady_total_samples": effective_batch_size * len(steady_rows),
        "steady_samples_per_second": (
            effective_batch_size * len(steady_rows) / total_seconds
        ),
        "step_seconds": {
            "mean": total_seconds / len(step_seconds),
            "p50": _percentile(step_seconds, 50),
            "p95": _percentile(step_seconds, 95),
            "min": min(step_seconds),
            "max": max(step_seconds),
        },
        "samples_per_second": {
            "mean": sum(throughputs) / len(throughputs),
            "p50": _percentile(throughputs, 50),
            "p05": _percentile(throughputs, 5),
            "min": min(throughputs),
            "max": max(throughputs),
        },
        "global_steps": [int(row["global_step"]) for row in steady_rows],
    }

    allocated = [
        float(row["perf_peak_cuda_memory_allocated_gib"])
        for row in rows
        if "perf_peak_cuda_memory_allocated_gib" in row
    ]
    reserved = [
        float(row["perf_peak_cuda_memory_reserved_gib"])
        for row in rows
        if "perf_peak_cuda_memory_reserved_gib" in row
    ]
    if allocated:
        summary["peak_cuda_memory_allocated_gib"] = max(allocated)
    if reserved:
        summary["peak_cuda_memory_reserved_gib"] = max(reserved)

    contract_path = log_path.parent / "training_contract.json"
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        batching = contract.get("batching", {})
        summary["distributed_type"] = batching.get("distributed_type")
        summary["num_processes"] = batching.get("num_processes")
        summary["batch_size_per_process"] = batching.get(
            "train_batch_size_per_process"
        )
        summary["mixed_precision"] = batching.get("mixed_precision")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Summarize instrumented Gaze-WAM distributed benchmark logs."
    )
    parser.add_argument("--logs", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = summarize_benchmark(args.logs)
    output_path = args.output or args.logs.parent / "benchmark_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
