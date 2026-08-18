from diffusion_policy.scripts.benchmark_nccl_allreduce import _summary


def test_summary_reports_ring_bus_bandwidth():
    result = _summary([0.5, 1.0], tensor_bytes=1_000_000_000, world_size=4)

    assert result["steps"] == 2
    assert result["step_seconds"]["mean"] == 0.75
    assert result["algorithm_bandwidth_GBps"] == 4 / 3
    assert result["bus_bandwidth_GBps"] == 2.0
