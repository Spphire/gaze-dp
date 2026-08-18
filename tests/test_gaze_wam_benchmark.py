import json
from pathlib import Path

import pytest
import torch

from diffusion_policy.scripts.summarize_gaze_wam_benchmark import (
    summarize_benchmark,
)
from diffusion_policy.workspace.train_gaze_wam_workspace import (
    _gaze_wam_step_performance_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


def test_step_performance_metrics_compute_effective_throughput_on_cpu():
    metrics = _gaze_wam_step_performance_metrics(
        elapsed_seconds=2.0,
        effective_batch_size=512,
        global_step=1,
        warmup_steps=3,
        device=torch.device("cpu"),
    )

    assert metrics["perf_step_seconds_max"] == pytest.approx(2.0)
    assert metrics["perf_effective_samples_per_second"] == pytest.approx(256.0)
    assert metrics["perf_effective_batch_size"] == 512
    assert metrics["perf_is_warmup"] is True
    assert "perf_peak_cuda_memory_allocated_gib" not in metrics


def test_benchmark_summary_excludes_warmup_and_weights_total_throughput(tmp_path):
    rows = [
        {
            "global_step": 0,
            "perf_step_seconds_max": 8.0,
            "perf_effective_samples_per_second": 64.0,
            "perf_effective_batch_size": 512,
            "perf_is_warmup": True,
        },
        {
            "global_step": 1,
            "perf_step_seconds_max": 2.0,
            "perf_effective_samples_per_second": 256.0,
            "perf_effective_batch_size": 512,
            "perf_is_warmup": False,
            "perf_peak_cuda_memory_allocated_gib": 10.0,
        },
        {
            "global_step": 2,
            "perf_step_seconds_max": 4.0,
            "perf_effective_samples_per_second": 128.0,
            "perf_effective_batch_size": 512,
            "perf_is_warmup": False,
            "perf_peak_cuda_memory_allocated_gib": 12.0,
        },
    ]
    log_path = tmp_path / "logs.json.txt"
    log_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    summary = summarize_benchmark(log_path)

    assert summary["measured_steps"] == 3
    assert summary["warmup_steps"] == 1
    assert summary["steady_steps"] == 2
    assert summary["steady_total_seconds"] == pytest.approx(6.0)
    assert summary["steady_samples_per_second"] == pytest.approx(1024.0 / 6.0)
    assert summary["step_seconds"]["p50"] == pytest.approx(3.0)
    assert summary["peak_cuda_memory_allocated_gib"] == pytest.approx(12.0)


def test_benchmark_summary_requires_steady_state_rows(tmp_path):
    log_path = tmp_path / "logs.json.txt"
    log_path.write_text(
        json.dumps(
            {
                "global_step": 0,
                "perf_step_seconds_max": 1.0,
                "perf_effective_samples_per_second": 1.0,
                "perf_effective_batch_size": 1,
                "perf_is_warmup": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No steady-state rows"):
        summarize_benchmark(log_path)


def test_distributed_benchmark_launcher_uses_equal_effective_batch_and_no_checkpoints():
    text = (
        ROOT / "train_scripts" / "benchmark_gaze_wam_distributed.sh"
    ).read_text()

    assert 'EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-512}"' in text
    assert "NUM_PROCESSES=8" in text
    assert "NUM_PROCESSES=16" in text
    assert "PER_PROCESS_BATCH_SIZE=$((EFFECTIVE_BATCH_SIZE / NUM_PROCESSES))" in text
    assert "training.measure_step_performance=true" in text
    assert "training.performance_warmup_steps=${WARMUP_STEPS}" in text
    assert "training.val_every=0" in text
    assert "training.max_val_steps=1" in text
    assert "checkpoint.save_deepspeed_state=false" in text
    assert "checkpoint.save_last_ckpt=false" in text
