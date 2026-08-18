import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_deepspeed_config_uses_bf16_zero2_and_auto_batch_values():
    config = json.loads(
        (ROOT / "accelerate" / "deepspeed_zero2_bf16.json").read_text()
    )

    assert config["bf16"]["enabled"] == "auto"
    assert config["train_batch_size"] == "auto"
    assert config["train_micro_batch_size_per_gpu"] == "auto"
    assert config["gradient_accumulation_steps"] == "auto"
    assert config["zero_optimization"]["stage"] == 2


def test_multinode_accelerate_config_has_fixed_world_size_and_rendezvous():
    text = (ROOT / "accelerate" / "2node-16gpu-deepspeed-bf16.yaml").read_text()

    assert "distributed_type: DEEPSPEED" in text
    assert "main_process_ip: 10.0.8.64" in text
    assert "main_process_port: 29500" in text
    assert "num_machines: 2" in text
    assert "num_processes: 16" in text
    assert "deepspeed_config_file: accelerate/deepspeed_zero2_bf16.json" in text


def test_multinode_launcher_requires_explicit_rank_and_uses_research_config():
    text = (
        ROOT / "train_scripts" / "train_gaze_wam_deepspeed_multinode.sh"
    ).read_text()

    assert 'MACHINE_RANK="${MACHINE_RANK:?Set MACHINE_RANK' in text
    assert "--machine_rank" in text
    assert "--main_process_ip" in text
    assert "accelerate/2node-16gpu-deepspeed-bf16.yaml" in text
    assert "training.max_train_steps=${MAX_TRAIN_STEPS}" in text
