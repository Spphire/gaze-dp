from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_multinode_accelerate_configs_have_expected_world_sizes():
    two_node_text = (
        ROOT / "accelerate" / "2node-16gpu-deepspeed-bf16.yaml"
    ).read_text()
    four_node_text = (
        ROOT / "accelerate" / "4node-32gpu-deepspeed-bf16.yaml"
    ).read_text()

    for text in (two_node_text, four_node_text):
        assert "distributed_type: DEEPSPEED" in text
        assert "main_process_ip:" in text
        assert "main_process_port: 29500" in text
        assert "deepspeed_multinode_launcher: standard" in text
        assert "zero_stage: 2" in text
        assert "gradient_clipping: 0.0" in text

    assert "num_machines: 2" in two_node_text
    assert "num_processes: 16" in two_node_text
    assert "gradient_accumulation_steps: 2" in two_node_text
    assert "num_machines: 4" in four_node_text
    assert "num_processes: 32" in four_node_text
    assert "gradient_accumulation_steps: 1" in four_node_text


def test_multinode_launcher_supports_configurable_world_size_and_step_budget():
    text = (
        ROOT / "train_scripts" / "train_gaze_wam_deepspeed_multinode.sh"
    ).read_text()

    assert 'MACHINE_RANK="${MACHINE_RANK:?Set the zero-based host rank.}"' in text
    assert 'MAIN_PROCESS_IP="${MAIN_PROCESS_IP:?Set MAIN_PROCESS_IP' in text
    assert "--machine_rank" in text
    assert "--main_process_ip" in text
    assert 'ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-' in text
    assert 'NUM_MACHINES="${NUM_MACHINES:-4}"' in text
    assert 'GPUS_PER_NODE="${GPUS_PER_NODE:-8}"' in text
    assert "MACHINE_RANK >= NUM_MACHINES" in text
    assert '--config_file "$ACCELERATE_CONFIG"' in text
    assert '"$PYTHON_BIN" -m accelerate.commands.accelerate_cli' in text
    assert 'if [[ -n "$MAX_TRAIN_STEPS" ]]' in text
    assert 'ARGS+=("training.max_train_steps=${MAX_TRAIN_STEPS}")' in text
    assert "training.resume=${RESUME}" in text


def test_multinode_launcher_validates_resume_flag():
    text = (
        ROOT / "train_scripts" / "train_gaze_wam_deepspeed_multinode.sh"
    ).read_text()

    assert 'RESUME="${RESUME:-false}"' in text
    assert 'case "$RESUME" in' in text
    assert "true|false" in text
    assert "RESUME must be true or false" in text


def test_multinode_launcher_uses_validated_nccl_transport_profile():
    launcher_text = (
        ROOT / "train_scripts" / "train_gaze_wam_deepspeed_multinode.sh"
    ).read_text()
    profile_text = (
        ROOT / "train_scripts" / "configure_nccl_transport.sh"
    ).read_text()

    assert 'CONFIGURE_NCCL_TRANSPORT:-false' in launcher_text
    assert "source \"$ROOT/train_scripts/configure_nccl_transport.sh\"" in launcher_text
    assert "configure_gaze_wam_nccl_transport" in launcher_text
    assert 'NCCL_TRANSPORT:-roce' in profile_text
    assert 'NCCL_ROCE_GID_INDEX:-3' in profile_text
    assert "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7" in profile_text
    assert 'gid_type" != "RoCE v2"' in profile_text
    assert "NCCL_IB_DISABLE=1" in profile_text


def test_deepspeed_state_save_remains_enabled_by_default_and_can_be_disabled():
    config_text = (
        ROOT / "diffusion_policy" / "config" / "train_gaze_wam_workspace.yaml"
    ).read_text()
    workspace_text = (
        ROOT / "diffusion_policy" / "workspace" / "train_gaze_wam_workspace.py"
    ).read_text()

    assert "save_deepspeed_state: true" in config_text
    assert "checkpoint.save_deepspeed_state" in workspace_text
    assert "def save_recovery_checkpoint" in workspace_text
    assert "if is_deepspeed and save_deepspeed_state" in workspace_text
    assert "accelerator.save_state" in workspace_text


def test_deepspeed_world_size_change_can_skip_native_state_restore():
    config_text = (
        ROOT / "diffusion_policy" / "config" /
        "train_gaze_wam_open_pretrain_workspace.yaml"
    ).read_text()
    workspace_text = (
        ROOT / "diffusion_policy" / "workspace" /
        "train_gaze_wam_workspace.py"
    ).read_text()

    assert "resume_deepspeed_state: true" in config_text
    assert '"training.resume_deepspeed_state"' in workspace_text
    assert "and resume_deepspeed_state" in workspace_text
    assert "Skipped DeepSpeed-native state restore" in workspace_text


def test_timed_launcher_records_rank_zero_wall_clock_and_exit_status():
    text = (
        ROOT / "train_scripts" / "launch_gaze_wam_deepspeed_timed.sh"
    ).read_text()

    assert 'TIMING_FILE="$OUTPUT_DIR/run_timing.json"' in text
    assert 'if [[ "$MACHINE_RANK" == "0" ]]' in text
    assert '"status": "running"' in text
    assert '"elapsed_seconds"' in text
    assert '"launcher_exit_code"' in text
