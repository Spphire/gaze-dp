from pathlib import Path

from hydra import compose, initialize_config_dir
from diffusion_policy.model.gaze_wam.cached_dual_stream_transformer import (
    CachedDualStreamGazeWamTransformer,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "diffusion_policy" / "config"

CONFIGS = {
    "train_gaze_wam_chuangzhi_open_pretrain_1n8g_gbs64": {
        "script": "launch_chuangzhi_open_pretrain_1n8g_gbs64.sh",
        "robot_batch": 0,
        "open_batch": 8,
        "robot_ratio": 0.0,
        "open_ratio": 1.0,
        "gaze_dropout": 0.2,
        "share_rgb": False,
        "epochs": 3000,
    },
    "train_gaze_wam_chuangzhi_mix_wrist_single_frame_all_gaze_1n8g_gbs64": {
        "script": "launch_chuangzhi_mix_wrist_single_frame_all_gaze_1n8g_gbs64.sh",
        "robot_batch": 6,
        "open_batch": 2,
        "robot_ratio": 0.75,
        "open_ratio": 0.25,
        "gaze_dropout": 0.2,
        "share_rgb": False,
        "epochs": 1000,
    },
    "train_gaze_wam_chuangzhi_mix_dual_single_frame_all_gaze_1n8g_gbs64": {
        "script": "launch_chuangzhi_mix_dual_single_frame_all_gaze_1n8g_gbs64.sh",
        "robot_batch": 6,
        "open_batch": 2,
        "robot_ratio": 0.75,
        "open_ratio": 0.25,
        "gaze_dropout": 0.2,
        "share_rgb": True,
        "epochs": 1000,
    },
    "train_gaze_wam_chuangzhi_robot_wrist_single_frame_always_gaze_1n8g_gbs64": {
        "script": "launch_chuangzhi_robot_wrist_single_frame_always_gaze_1n8g_gbs64.sh",
        "robot_batch": 8,
        "open_batch": 0,
        "robot_ratio": 1.0,
        "open_ratio": 0.0,
        "gaze_dropout": 0.0,
        "stage": "robot_only",
        "share_rgb": False,
        "epochs": 1000,
    },
}


def load_cfg(name):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=name)


def test_only_formal_chuangzhi_configs_remain():
    names = sorted(p.stem for p in CONFIG_DIR.glob("train_gaze_wam_chuangzhi_*.yaml"))
    assert names == sorted(CONFIGS)


def test_configs_are_single_node_gbs64_latent_only_all_gaze():
    for name, expected in CONFIGS.items():
        cfg = load_cfg(name)
        assert cfg.policy.model_architecture == "cached_dual_stream"
        assert "heatmap_objective" not in cfg.policy
        assert "heatmap_xy_loss_weight" not in cfg.policy
        assert "heatmap_point_nll_loss_weight" not in cfg.policy
        assert "heatmap_js_loss_weight" not in cfg.policy
        assert cfg.policy.heatmap_loss_weight == 1.0
        assert cfg.task.robot_heatmap_supervision == "all_valid"
        assert cfg.data_mixing.total_batch_size_per_process == 8
        assert cfg.robot_dataloader.batch_size == expected["robot_batch"]
        assert cfg.open_dataloader.batch_size == expected["open_batch"]
        assert cfg.data_mixing.robot_ratio == expected["robot_ratio"]
        assert cfg.data_mixing.open_ratio == expected["open_ratio"]
        assert cfg.task.robot_gaze_dropout_prob == expected["gaze_dropout"]
        if "stage" in expected:
            assert cfg.training.stage == expected["stage"]
        assert cfg.training.gradient_accumulate_every == 1
        assert cfg.training.num_epochs == expected["epochs"]
        assert cfg.policy.obs_encoder.share_rgb_model is expected["share_rgb"]
        assert 8 * 8 * cfg.training.gradient_accumulate_every == 64


def test_formal_launchers_pin_pet_single_node_eight_processes():
    for expected in CONFIGS.values():
        text = (ROOT / "train_scripts" / expected["script"]).read_text()
        assert 'export ACCELERATE_CONFIG="accelerate/8gpu-amp.yaml"' in text
        assert "export EXPECTED_NNODES=1" in text
        assert "export EXPECTED_NPROC_PER_NODE=8" in text
        assert "export RESUME=false" in text
        assert "launch_gaze_wam_chuangzhi_pet.sh" in text


def test_cached_dual_stream_uses_independent_gaze_cross_attention():
    model = CachedDualStreamGazeWamTransformer(
        action_dim=10,
        heatmap_dim=16,
        action_horizon=48,
        heatmap_num_tokens=256,
        max_image_tokens=256,
        n_layer=1,
        n_head=8,
        n_emb=768,
    )
    contract = model.attention_contract_summary(num_image_tokens=256)
    assert contract["target_attention_mode"] == "independent_image_gaze_target_softmax"
    assert contract["target_attention_concatenates_image_gaze"] is False
    assert contract["shared_condition_kv_cache"] is False
    assert contract["gaze_cross_attention_source_tokens"] == 1


def test_pet_launcher_maps_platform_topology_and_keeps_runtime_on_project_disk():
    text = (ROOT / "train_scripts" / "launch_gaze_wam_chuangzhi_pet.sh").read_text()
    for variable in (
        "PET_MASTER_PORT",
        "PET_MASTER_ADDR",
        "PET_NPROC_PER_NODE",
        "PET_NNODES",
        "PET_NODE_RANK",
    ):
        assert f"${{{variable}:" in text
    for option in (
        "--machine_rank",
        "--main_process_ip",
        "--main_process_port",
        "--num_machines",
        "--num_processes",
    ):
        assert option in text
    assert 'SHORT_TMP_ROOT="$ROOT/../.tmp"' in text
    assert 'export HOME="$RUNTIME_ROOT/home"' in text
    assert 'export TMPDIR="$SHORT_TMP_ROOT"' in text
