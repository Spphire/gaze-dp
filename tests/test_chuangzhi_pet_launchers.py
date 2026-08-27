from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "diffusion_policy" / "config"


def load_cfg(name):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=name)


def assert_temporal_latent_only(cfg):
    assert cfg.task.heatmap_key is None
    assert cfg.task.temporal_heatmap_mode == "bidirectional"
    assert cfg.task.temporal_heatmap_window_radius == 30
    assert cfg.task.temporal_heatmap_beta == 10.0
    assert cfg.task.temporal_heatmap_sigma_px == 6.0
    assert cfg.task.temporal_heatmap_current_weight == 2.0
    assert cfg.policy.heatmap_objective == "diffusion"
    assert cfg.policy.heatmap_loss_weight == 1.0
    assert cfg.policy.heatmap_token_kl_loss_weight == 0.0
    assert cfg.policy.heatmap_diffusion_final_loss_enabled is False
    assert cfg.policy.heatmap_final_loss_timestep_weighting == "none"
    assert cfg.policy.heatmap_xy_loss_weight == 0.0
    assert cfg.policy.heatmap_point_nll_loss_weight == 0.0
    assert cfg.policy.heatmap_js_loss_weight == 0.0
    assert all(len(str(tag)) <= 64 for tag in cfg.logging.tags)


def test_chuangzhi_open_pretrain_is_4n8g_gbs512_with_temporal_latent_only():
    cfg = load_cfg("train_gaze_wam_chuangzhi_open_pretrain_4n8g_gbs512")

    assert_temporal_latent_only(cfg)
    assert cfg.task.n_obs_steps == 1
    assert cfg.task.gaze_key == "gaze_xy"
    assert cfg.data_mixing.total_batch_size_per_process == 16
    assert cfg.data_mixing.robot_ratio == 0.0
    assert cfg.data_mixing.open_ratio == 1.0
    assert cfg.training.gradient_accumulate_every == 1
    assert 16 * 4 * 8 * cfg.training.gradient_accumulate_every == 512
    assert cfg.training.num_epochs == 3000
    assert cfg.training.checkpoint_every_steps == 30000
    assert cfg.training.latest_checkpoint_every_steps == 3000
    assert cfg.training.measure_step_performance is True
    assert cfg.open_dataloader.num_workers == 4
    assert cfg.training.save_val_heatmap_preview is True
    assert cfg.training.save_checkpoint_heatmap_preview is True


def test_chuangzhi_dual_view_mix_is_single_frame_shared_vit_gbs64():
    cfg = load_cfg("train_gaze_wam_chuangzhi_mix_dual_single_frame_1n8g_gbs64")

    assert_temporal_latent_only(cfg)
    assert cfg.task.n_obs_steps == 1
    assert list(cfg.task.camera_keys) == ["camera0_rgb", "camera1_rgb"]
    assert cfg.policy.obs_encoder.share_rgb_model is True
    assert cfg.robot_dataloader.batch_size == 6
    assert cfg.open_dataloader.batch_size == 2
    assert cfg.data_mixing.total_batch_size_per_process == 8
    assert cfg.training.gradient_accumulate_every == 1
    assert cfg.training.measure_step_performance is True
    assert 8 * 1 * 8 * cfg.training.gradient_accumulate_every == 64


def test_chuangzhi_wrist_mix_is_single_frame_single_view_gbs64():
    cfg = load_cfg("train_gaze_wam_chuangzhi_mix_wrist_single_frame_1n8g_gbs64")

    assert_temporal_latent_only(cfg)
    assert cfg.task.n_obs_steps == 1
    assert cfg.task.camera_key == "camera0_rgb"
    assert list(cfg.task.camera_keys) == ["camera0_rgb"]
    assert cfg.robot_dataloader.batch_size == 6
    assert cfg.open_dataloader.batch_size == 2
    assert cfg.data_mixing.total_batch_size_per_process == 8
    assert cfg.training.gradient_accumulate_every == 1
    assert cfg.training.measure_step_performance is True
    assert 8 * 1 * 8 * cfg.training.gradient_accumulate_every == 64


def test_chuangzhi_independent_cross_attention_wrist_matches_wrist_run():
    base = load_cfg("train_gaze_wam_chuangzhi_mix_wrist_single_frame_1n8g_gbs64")
    cfg = load_cfg(
        "train_gaze_wam_chuangzhi_independent_cross_attention_mix_wrist_single_frame_1n8g_gbs64"
    )

    assert cfg.name == (
        "train_gaze_wam_chuangzhi_independent_cross_attention_mix_wrist_single_frame_1n8g_gbs64"
    )
    assert cfg.exp_name == (
        "chuangzhi_independent_cross_attention_mix_wrist_single_frame_temporal_mixed_nll_1n8g_gbs64"
    )
    assert cfg.task == base.task
    assert cfg.policy == base.policy
    assert cfg.data_mixing == base.data_mixing
    assert cfg.robot_dataloader == base.robot_dataloader
    assert cfg.open_dataloader == base.open_dataloader
    assert cfg.val_robot_dataloader == base.val_robot_dataloader
    assert cfg.val_open_dataloader == base.val_open_dataloader
    assert cfg.training == base.training
    assert cfg.checkpoint == base.checkpoint
    assert cfg.logging.project == base.logging.project
    assert cfg.logging.resume == base.logging.resume is False
    assert cfg.logging.mode == base.logging.mode == "disabled"
    assert cfg.logging.id == base.logging.id is None
    assert cfg.logging.group == base.logging.group is None


def test_chuangzhi_independent_cross_attention_wrist_wrapper_is_new_8gpu_run():
    wrapper = (
        ROOT
        / "train_scripts"
        / "launch_chuangzhi_independent_cross_attention_mix_wrist_single_frame_1n8g_gbs64.sh"
    )
    text = wrapper.read_text()

    assert (
        'export TASK_NAME="independent_cross_attention_mix_wrist_single_frame_temporal_mixed_nll_1n8g_gbs64"'
        in text
    )
    assert (
        'export CONFIG_NAME="train_gaze_wam_chuangzhi_independent_cross_attention_mix_wrist_single_frame_1n8g_gbs64"'
        in text
    )
    assert 'export ACCELERATE_CONFIG="accelerate/8gpu-amp.yaml"' in text
    assert "export EXPECTED_NNODES=1" in text
    assert "export EXPECTED_NPROC_PER_NODE=8" in text
    assert "launch_gaze_wam_chuangzhi_pet.sh" in text
    assert "RESUME" not in text


def test_chuangzhi_task_names_separate_temporal_mixed_nll_runs():
    wrappers = (
        "launch_chuangzhi_open_pretrain_4n8g_gbs512.sh",
        "launch_chuangzhi_mix_dual_single_frame_1n8g_gbs64.sh",
        "launch_chuangzhi_mix_wrist_single_frame_1n8g_gbs64.sh",
    )
    for wrapper in wrappers:
        text = (ROOT / "train_scripts" / wrapper).read_text()
        task_name_line = next(
            line for line in text.splitlines() if line.startswith("export TASK_NAME=")
        )
        assert "temporal_mixed_nll" in task_name_line


def test_chuangzhi_resume_wrappers_force_resume_and_validate_latest_checkpoint():
    wrappers = {
        "resume_chuangzhi_open_pretrain_4n8g_gbs512.sh":
            "open_pretrain_temporal_mixed_nll_4n8g_gbs512",
        "resume_chuangzhi_mix_wrist_single_frame_1n8g_gbs64.sh":
            "mix_wrist_single_frame_temporal_mixed_nll_1n8g_gbs64",
        "resume_chuangzhi_mix_dual_single_frame_1n8g_gbs64.sh":
            "mix_dual_shared_vit_temporal_mixed_nll_1n8g_gbs64",
    }
    for wrapper, task_name in wrappers.items():
        text = (ROOT / "train_scripts" / wrapper).read_text()
        assert f'export TASK_NAME="{task_name}"' in text
        assert 'export RESUME=true' in text
        assert 'export OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_RESUME_OUTPUT_DIR}"' in text
        assert '"$OUTPUT_DIR/checkpoints/latest.ckpt"' in text
        if "open_pretrain" in wrapper:
            assert "accelerate_state_step_*" in text
        assert 'launch_gaze_wam_chuangzhi_pet.sh' in text


def test_dual_and_open_wrappers_require_offline_heatmap_latent_cache():
    wrappers = {
        "launch_chuangzhi_open_pretrain_4n8g_gbs512.sh": (
            "hot3d_open_train",
        ),
        "resume_chuangzhi_open_pretrain_4n8g_gbs512.sh": (
            "hot3d_open_train",
        ),
        "launch_chuangzhi_mix_dual_single_frame_1n8g_gbs64.sh": (
            "hot3d_open_train",
            "gaze_wam_robot_20260814_from_162120_dual_view",
        ),
        "resume_chuangzhi_mix_dual_single_frame_1n8g_gbs64.sh": (
            "hot3d_open_train",
            "gaze_wam_robot_20260814_from_162120_dual_view",
        ),
    }
    for wrapper, datasets in wrappers.items():
        text = (ROOT / "train_scripts" / wrapper).read_text()
        assert "export GAZE_WAM_HEATMAP_CACHE_ROOT=" in text
        assert "job-1b9f80a1-bb74-4e62-99a7-76bfeb242898-worker-0_23456" in text
        assert "Required heatmap latent cache manifest is missing or empty" in text
        for dataset in datasets:
            assert dataset in text


def test_chuangzhi_pet_launcher_maps_platform_topology_and_writes_node_logs():
    text = (ROOT / "train_scripts" / "launch_gaze_wam_chuangzhi_pet.sh").read_text()

    for variable in (
        "PET_MASTER_PORT",
        "PET_MASTER_ADDR",
        "PET_NPROC_PER_NODE",
        "PET_NNODES",
        "PET_NODE_RANK",
    ):
        assert f'${{{variable}:' in text
    for option in (
        "--machine_rank",
        "--main_process_ip",
        "--main_process_port",
        "--num_machines",
        "--num_processes",
    ):
        assert option in text
    for filename in (
        "launcher_node${PET_NODE_RANK}.log",
        "environment_node${PET_NODE_RANK}.txt",
        "nvidia_smi_node${PET_NODE_RANK}.txt",
        "gpu_monitor_node${PET_NODE_RANK}.csv",
        "gpu_monitor_node${PET_NODE_RANK}.pid",
        "pip_freeze_node${PET_NODE_RANK}.txt",
        "environment_setup_node${PET_NODE_RANK}.log",
        "exit_code_node${PET_NODE_RANK}.txt",
        "run_timing.json",
        "resolved_config.yaml",
        "launch_manifest.json",
    ):
        assert filename in text
    assert "NCCL_DEBUG_FILE" in text
    assert "--cfg=job" in text
    assert 'CHUANGZHI_NCCL_DEBUG:-WARN' in text
    assert 'CHUANGZHI_NCCL_DEBUG_SUBSYS:-INIT,NET' in text
    assert "GPU_MONITOR_INTERVAL_SEC" in text
    assert "stop_gpu_monitor" in text
    for runtime_variable in (
        "RUNTIME_ROOT",
        "SHORT_TMP_ROOT",
        'SHORT_TMP_ROOT="$ROOT/../.tmp"',
        'export HOME="$RUNTIME_ROOT/home"',
        'export TMPDIR="$SHORT_TMP_ROOT"',
        'export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg-cache"',
        'export TORCH_EXTENSIONS_DIR="$RUNTIME_ROOT/torch-extensions"',
        'export TORCHINDUCTOR_CACHE_DIR="$RUNTIME_ROOT/torch"',
        'export TRITON_HOME="$RUNTIME_ROOT/triton"',
        'export TRITON_CACHE_DIR="$RUNTIME_ROOT/triton"',
        'export HF_HOME="$RUNTIME_ROOT/huggingface"',
        'export WANDB_DIR="$RUNTIME_ROOT/wandb"',
        'df -h "$ROOT" "$RUNTIME_ROOT" "$HOME" "$TMPDIR" /root',
    ):
        assert runtime_variable in text
    assert not any(
        line.startswith("export NCCL_ASYNC_ERROR_HANDLING=")
        for line in text.splitlines()
    )
    assert 'pip freeze --python "$PYTHON_BIN"' in text
    assert "PIPESTATUS[0]" in text
    assert 'launch_args+=("$@")' in text


def test_chuangzhi_pet_launcher_uses_rank_zero_uv_environment_bootstrap():
    launcher = (ROOT / "train_scripts" / "launch_gaze_wam_chuangzhi_pet.sh").read_text()
    setup = (ROOT / "train_scripts" / "setup_chuangzhi_uv_env.sh").read_text()
    requirements = (ROOT / "requirements-chuangzhi.txt").read_text()

    assert 'CHUANGZHI_AUTO_SETUP_VENV:-true' in launcher
    assert 'setup_chuangzhi_uv_env.sh' in launcher
    assert 'PYTHON_BIN="$ROOT/.venv/bin/python"' in launcher
    assert 'if [[ "$PET_NODE_RANK" == "0" ]]' in setup
    assert '"$UV_BIN" venv' in setup
    assert '--managed-python' in setup
    assert '"$UV_BIN" pip install' in setup
    assert '--torch-backend cu128' in setup
    assert 'CHUANGZHI_PYTHON_VERSION:-3.12.12' in setup
    assert 'CHUANGZHI_TORCH_VERSION:-2.7.1' in setup
    assert 'CHUANGZHI_TORCHVISION_VERSION:-0.22.1' in setup
    assert 'CHUANGZHI_ENV_PREPARE_ONLY:-false' in setup
    assert 'torch.cuda.is_available()' in setup
    assert 'CHUANGZHI_EXPECTED_CUDA:-12.8' in setup
    assert 'torch.version.cuda' in setup
    assert '.chuangzhi_uv_setup.lock' in setup
    assert 'Another job is preparing the shared uv environment' in setup
    assert 'torch==' not in requirements
    assert 'torchvision==' not in requirements
    assert 'zarr==2.18.7' in requirements
    assert 'tokenizers==0.21.4' in requirements
    assert 'transformers==4.55.2' in requirements
    assert 'importlib.metadata.version(distribution)' in setup
    assert '"tokenizers"' in setup
    assert '"transformers"' in setup
