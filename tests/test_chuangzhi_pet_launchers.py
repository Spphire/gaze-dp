from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "diffusion_policy" / "config"


def load_cfg(name):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=name)


def test_chuangzhi_open_pretrain_is_4n8g_gbs512_with_point_heatmap_target():
    cfg = load_cfg("train_gaze_wam_chuangzhi_open_pretrain_4n8g_gbs512")

    assert cfg.task.n_obs_steps == 1
    assert cfg.task.gaze_key == "gaze_xy"
    assert cfg.task.heatmap_key is None
    assert cfg.data_mixing.total_batch_size_per_process == 16
    assert cfg.data_mixing.robot_ratio == 0.0
    assert cfg.data_mixing.open_ratio == 1.0
    assert cfg.training.gradient_accumulate_every == 1
    assert 16 * 4 * 8 * cfg.training.gradient_accumulate_every == 512
    assert cfg.training.num_epochs == 3000
    assert cfg.training.checkpoint_every_steps == 30000
    assert cfg.training.save_val_heatmap_preview is True
    assert cfg.training.save_checkpoint_heatmap_preview is True


def test_chuangzhi_dual_view_mix_is_single_frame_shared_vit_gbs64():
    cfg = load_cfg("train_gaze_wam_chuangzhi_mix_dual_single_frame_1n8g_gbs64")

    assert cfg.task.n_obs_steps == 1
    assert list(cfg.task.camera_keys) == ["camera0_rgb", "camera1_rgb"]
    assert cfg.policy.obs_encoder.share_rgb_model is True
    assert cfg.robot_dataloader.batch_size == 6
    assert cfg.open_dataloader.batch_size == 2
    assert cfg.data_mixing.total_batch_size_per_process == 8
    assert cfg.training.gradient_accumulate_every == 1
    assert 8 * 1 * 8 * cfg.training.gradient_accumulate_every == 64


def test_chuangzhi_wrist_mix_is_single_frame_single_view_gbs64():
    cfg = load_cfg("train_gaze_wam_chuangzhi_mix_wrist_single_frame_1n8g_gbs64")

    assert cfg.task.n_obs_steps == 1
    assert cfg.task.camera_key == "camera0_rgb"
    assert list(cfg.task.camera_keys) == ["camera0_rgb"]
    assert cfg.robot_dataloader.batch_size == 6
    assert cfg.open_dataloader.batch_size == 2
    assert cfg.data_mixing.total_batch_size_per_process == 8
    assert cfg.training.gradient_accumulate_every == 1
    assert 8 * 1 * 8 * cfg.training.gradient_accumulate_every == 64


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
        "pip_freeze_node${PET_NODE_RANK}.txt",
        "environment_setup_node${PET_NODE_RANK}.log",
        "exit_code_node${PET_NODE_RANK}.txt",
        "run_timing.json",
        "resolved_config.yaml",
        "launch_manifest.json",
    ):
        assert filename in text
    assert "NCCL_DEBUG_FILE" in text
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
