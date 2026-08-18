from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_multinode_accelerate_config_has_fixed_world_size_and_rendezvous():
    text = (ROOT / "accelerate" / "2node-16gpu-deepspeed-bf16.yaml").read_text()

    assert "distributed_type: DEEPSPEED" in text
    assert "main_process_ip: 10.0.8.64" in text
    assert "main_process_port: 29500" in text
    assert "num_machines: 2" in text
    assert "num_processes: 16" in text
    assert "deepspeed_multinode_launcher: standard" in text
    assert "zero_stage: 2" in text
    assert "gradient_accumulation_steps: 1" in text
    assert "gradient_clipping: 0.0" in text


def test_multinode_launcher_requires_explicit_rank_and_uses_research_config():
    text = (
        ROOT / "train_scripts" / "train_gaze_wam_deepspeed_multinode.sh"
    ).read_text()

    assert 'MACHINE_RANK="${MACHINE_RANK:?Set MACHINE_RANK' in text
    assert "--machine_rank" in text
    assert "--main_process_ip" in text
    assert "accelerate/2node-16gpu-deepspeed-bf16.yaml" in text
    assert '"$PYTHON_BIN" -m accelerate.commands.accelerate_cli' in text
    assert "training.max_train_steps=${MAX_TRAIN_STEPS}" in text
