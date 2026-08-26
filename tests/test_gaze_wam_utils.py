import copy
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from hydra import compose, initialize_config_dir
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.common.action_utils import (  # noqa: E402
    absolute_actions_to_relative_actions,
    relative_actions_to_absolute_actions,
)
from diffusion_policy.common.gaze_utils import (  # noqa: E402
    as_optional_gaze_wam_key,
    gaussian_heatmaps_from_points,
)
from diffusion_policy.common.gaze_wam_training_config import (  # noqa: E402
    _normalize_gaze_wam_early_bool_config,
    gaze_wam_action_normalizer_contract,
    gaze_wam_data_stream_contract,
    gaze_wam_loss_routing_validation_guardrails_ok,
    gaze_wam_required_loss_routing_validation_flags,
    normalize_gaze_wam_nonnegative_float_field,
    normalize_gaze_wam_nonnegative_int_field,
    normalize_gaze_wam_positive_float_field,
    normalize_gaze_wam_positive_int_sequence,
    normalize_gaze_wam_positive_int_field,
    normalize_gaze_wam_unit_interval_float_field,
)
from diffusion_policy.model.common.normalizer import (  # noqa: E402
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.dataset.gaze_wam_mixing import (  # noqa: E402
    build_gaze_wam_mixed_batch,
)
from diffusion_policy.dataset.gaze_wam_dataset import (  # noqa: E402
    GazeWamOpenDataset,
    GazeWamRobotDataset,
    _image_to_chw_float as _dataset_image_to_chw_float,
    _validate_nonnegative_int as _validate_gaze_wam_dataset_nonnegative_int,
    _validate_positive_int as _validate_gaze_wam_dataset_positive_int,
    _validate_positive_int_pair as _validate_gaze_wam_dataset_positive_int_pair,
)
from diffusion_policy.model.gaze_wam.gaze_encoder import (  # noqa: E402
    GaussianSpatialEncoder,
    GazeConditionEncoder,
)
from diffusion_policy.model.gaze_wam.cached_dual_stream_transformer import (  # noqa: E402
    CachedDualStreamGazeWamTransformer,
)
from diffusion_policy.model.gaze_wam.heatmap_codec import HeatmapTokenCodec  # noqa: E402
from diffusion_policy.model.gaze_wam.heatmap_decoder import CosmosHeatmapCodec  # noqa: E402
from diffusion_policy.model.gaze_wam.joint_transformer import (  # noqa: E402
    JointGazeWamTransformer,
)
from diffusion_policy.model.gaze_wam.loss import (  # noqa: E402
    dsnt_expectation,
    distributed_mask_count,
    distributed_masked_mean,
    per_sample_dsnt_xy_loss,
    per_sample_spatial_point_nll_loss,
    per_sample_spatial_js_loss,
    spatial_distribution_2d,
    spatial_softmax_2d,
)
from diffusion_policy.model.gaze_wam.metrics import gaze_dependency_ratio  # noqa: E402
from diffusion_policy.model.gaze_wam.routing import loss_routing_summary  # noqa: E402
from diffusion_policy.policy.gaze_wam_policy import GazeWamPolicy  # noqa: E402
from diffusion_policy.scripts.eval_gaze_wam_metrics import (  # noqa: E402
    _action_abs_mask,
    _optional_presence_mask,
    evaluate_gaze_wam_dataset,
)
import diffusion_policy.scripts.eval_gaze_wam_metrics as eval_gaze_wam_metrics_module  # noqa: E402
from diffusion_policy.scripts.compare_gaze_wam_ablation_metrics import (  # noqa: E402
    DEFAULT_VARIANTS as COMPARE_DEFAULT_VARIANTS,
    _config_provenance,
    compare_gaze_wam_ablation_metrics,
    write_metrics_csv,
)
import diffusion_policy.scripts.compare_gaze_wam_ablation_metrics as compare_gaze_wam_ablation_metrics_module  # noqa: E402
from diffusion_policy.scripts.gaze_wam_provenance import (  # noqa: E402
    PROVENANCE_CONTRACT_VERSION,
    provenance_contract_id,
)
from diffusion_policy.scripts.plan_gaze_wam_experiments import (  # noqa: E402
    build_gaze_wam_experiment_plan,
    write_plan_csv,
    write_plan_script,
)
from diffusion_policy.scripts.preview_gaze_wam_dataset import preview_gaze_wam_dataset  # noqa: E402
from diffusion_policy.scripts.preview_gaze_wam_checkpoint import (  # noqa: E402
    _checkpoint_ema_summary as _checkpoint_preview_ema_summary,
    _select_sample_indices as _select_checkpoint_preview_sample_indices,
)
from diffusion_policy.scripts.preflight_gaze_wam import (  # noqa: E402
    _check_policy_contract,
    preflight_gaze_wam,
)
import diffusion_policy.scripts.preflight_gaze_wam as preflight_gaze_wam_module  # noqa: E402
from diffusion_policy.scripts.launch_gaze_wam_training import (  # noqa: E402
    build_gaze_wam_train_command,
    check_real_data_readiness,
    launch_gaze_wam_training,
    _looks_like_debug_data_path,
    _preflight_loss_routing_validation_guardrails_ok,
)
import diffusion_policy.scripts.launch_gaze_wam_training as launch_gaze_wam_training_module  # noqa: E402
from diffusion_policy.scripts.eval_gaze_wam_metrics import load_cfg  # noqa: E402
from diffusion_policy.scripts.prepare_robot_gaze_wam_zarr import (  # noqa: E402
    prepare_robot_gaze_wam_zarr,
)
from diffusion_policy.scripts.prepare_open_gaze_wam_zarr import (  # noqa: E402
    prepare_open_gaze_wam_zarr,
)
from diffusion_policy.scripts.review_gaze_wam_data_onboarding import (  # noqa: E402
    review_gaze_wam_data_onboarding,
)
import diffusion_policy.scripts.review_gaze_wam_data_onboarding as review_gaze_wam_data_onboarding_module  # noqa: E402
from diffusion_policy.scripts.review_gaze_wam_training_readiness import (  # noqa: E402
    review_gaze_wam_training_readiness,
)
import diffusion_policy.scripts.review_gaze_wam_training_readiness as review_gaze_wam_training_readiness_module  # noqa: E402
from diffusion_policy.scripts.gaze_wam_smoke_pipeline import (  # noqa: E402
    run_gaze_wam_smoke_pipeline,
)
import diffusion_policy.scripts.gaze_wam_smoke_pipeline as gaze_wam_smoke_pipeline_module  # noqa: E402
from diffusion_policy.scripts.rehearse_gaze_wam_split_deployment import (  # noqa: E402
    run_gaze_wam_config_split_deployment_rehearsal,
    run_gaze_wam_split_deployment_rehearsal,
)
from diffusion_policy.scripts.make_gaze_wam_split_deployment_config import (  # noqa: E402
    build_gaze_wam_split_deployment_config,
    main as make_gaze_wam_split_deployment_config_main,
)
from diffusion_policy.scripts.generate_gaze_wam_debug_data import (  # noqa: E402
    generate_gaze_wam_debug_data,
)
from diffusion_policy.scripts.convert_open_gaze_manifest import (  # noqa: E402
    convert_open_gaze_manifest,
)
from diffusion_policy.scripts.convert_hot3d_processed_to_open_zarr import (  # noqa: E402
    convert_hot3d_processed_to_open_zarr,
)
from diffusion_policy.scripts.export_video_gaze_manifest import (  # noqa: E402
    export_video_gaze_manifest,
)
from diffusion_policy.scripts.adapt_open_video_gaze_metadata import (  # noqa: E402
    adapt_open_video_gaze_metadata,
)
from diffusion_policy.scripts.inspect_open_video_gaze_metadata import (  # noqa: E402
    inspect_open_video_gaze_metadata,
)
from diffusion_policy.scripts.canonicalize_robot_gaze_wam_zarr import (  # noqa: E402
    canonicalize_robot_gaze_wam_zarr,
)
from diffusion_policy.scripts.inspect_gaze_wam_zarr import inspect_gaze_wam_zarr  # noqa: E402
from diffusion_policy.scripts.validate_gaze_wam_zarr import validate_gaze_wam_zarr  # noqa: E402
from diffusion_policy.scripts.verify_gaze_wam_dino_source import (  # noqa: E402
    verify_gaze_wam_dino_source,
)
from diffusion_policy.real_world.gaze_wam_inference import (  # noqa: E402
    GazeWamInferenceAdapter,
    _image_to_chw_float as _inference_image_to_chw_float,
    tcp_pose_to_action_base_abs,
)
from diffusion_policy.real_world.gaze_wam_action_base import action_base_abs_to_10d  # noqa: E402
from diffusion_policy.real_world.gaze_wam_runner import (  # noqa: E402
    GazeWamDeploymentRunner,
    GazeWamRobotState,
    GazeWamSafetyConfig,
    GazeWamScheduledCommand,
)
from diffusion_policy.real_world.gaze_wam_deployment_bindings import (  # noqa: E402
    GazeWamJsonlCommandSink,
    GazeWamJsonlGazeProvider,
    GazeWamJsonlStateProvider,
    GazeWamOpenCVCameraProvider,
    GazeWamRecordingCommandSink,
    GazeWamStaticStateProvider,
    build_gaze_wam_deployment_runner_from_config,
    build_gaze_wam_gaze_provider,
    build_gaze_wam_image_provider,
)
from diffusion_policy.real_world.gaze_wam_zarr_replay import (  # noqa: E402
    GazeWamZarrReplaySource,
    _validate_rehearsal_robot_zarr,
    run_gaze_wam_config_zarr_deployment_rehearsal,
    run_gaze_wam_zarr_deployment_rehearsal,
    safety_config_from_json,
)
from diffusion_policy.workspace.train_gaze_wam_workspace import (  # noqa: E402
    TrainGazeWamWorkspace,
    _RestartingDataLoaderIterator,
    _check_training_dataloader_lengths,
    _check_training_dataset_lengths,
    _gaze_wam_checkpoint_due,
    _gaze_wam_step_checkpoint_due,
    _make_cpu_generator,
    _normalize_gaze_wam_training_config,
    _normalize_gaze_wam_task_routing_config,
    _runtime_gradient_accumulation_steps,
    _select_heatmap_preview_indices,
    _write_checkpoint_heatmap_log,
    _validate_prepared_epoch_driver_length,
    validate_gaze_wam_training_config,
    validate_gaze_wam_task_routing_config,
)
import diffusion_policy.workspace.train_gaze_wam_workspace as train_gaze_wam_workspace_module  # noqa: E402
from diffusion_policy.model.vision.transformer_obs_encoder import (  # noqa: E402
    TransformerObsEncoder,
)
from diffusion_policy.common.pose_util import mat_to_rot6d  # noqa: E402


def _random_pose10(batch_shape):
    from scipy.spatial.transform import Rotation

    pos = np.random.normal(size=batch_shape + (3,)).astype(np.float64)
    rot = Rotation.random(np.prod(batch_shape)).as_matrix().reshape(batch_shape + (3, 3))
    d6 = mat_to_rot6d(rot)
    gripper = np.random.uniform(0, 0.08, size=batch_shape + (1,))
    return np.concatenate([pos, d6, gripper], axis=-1)


def test_runtime_gradient_accumulation_uses_deepspeed_plugin_value():
    class DistributedTypeStub:
        name = "DEEPSPEED"

    class DeepSpeedPluginStub:
        deepspeed_config = {"gradient_accumulation_steps": 1}

    class StateStub:
        deepspeed_plugin = DeepSpeedPluginStub()

    class AcceleratorStub:
        distributed_type = DistributedTypeStub()
        state = StateStub()

    steps, source = _runtime_gradient_accumulation_steps(AcceleratorStub(), 2)

    assert steps == 1
    assert source == "deepspeed_config"


def test_runtime_gradient_accumulation_keeps_training_value_without_deepspeed():
    class AcceleratorStub:
        distributed_type = "NO"

    steps, source = _runtime_gradient_accumulation_steps(AcceleratorStub(), 2)

    assert steps == 2
    assert source == "training_config"


def _replace_zarr_array(group, name, values):
    if name in group:
        del group[name]
    group.array(name, values, shape=values.shape, dtype=values.dtype)


def _write_gaze_wam_zarr(path: Path, include_action=True, image_hw=(16, 16), gaze=None):
    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    episode_ends = np.asarray([6], dtype=np.int64)
    meta.array("episode_ends", episode_ends, shape=episode_ends.shape, dtype=episode_ends.dtype)

    image_h, image_w = image_hw
    image = np.zeros((6, image_h, image_w, 3), dtype=np.uint8)
    for i in range(6):
        image[i] = i * 10
    if gaze is None:
        gaze = np.asarray(
            [
                [0.1, 0.2],
                [0.2, 0.3],
                [0.3, 0.4],
                [0.4, 0.5],
                [0.5, 0.6],
                [0.6, 0.7],
            ],
            dtype=np.float32,
        )
    else:
        gaze = np.asarray(gaze, dtype=np.float32)
    heatmap = gaussian_heatmaps_from_points(gaze, image_size=image_hw, sigma_px=4.0)
    has_gaze_label = np.ones((6,), dtype=np.bool_)
    has_heatmap_image = np.ones((6,), dtype=np.bool_)
    data.array("camera0_rgb", image, shape=image.shape, dtype=image.dtype)
    data.array("gaze_xy", gaze, shape=gaze.shape, dtype=gaze.dtype)
    data.array("gaze_heatmap", heatmap, shape=heatmap.shape, dtype=heatmap.dtype)
    data.array("has_gaze_label", has_gaze_label, shape=has_gaze_label.shape, dtype=has_gaze_label.dtype)
    data.array(
        "has_heatmap_image",
        has_heatmap_image,
        shape=has_heatmap_image.shape,
        dtype=has_heatmap_image.dtype,
    )

    if include_action:
        action_abs = _random_pose10((6,)).astype(np.float32)
        tcp_pose_abs = action_abs[:, :9].copy()
        gripper_width = action_abs[:, 9:10].copy()
        data.array("action_abs_tcp", action_abs, shape=action_abs.shape, dtype=action_abs.dtype)
        data.array("tcp_pose_abs", tcp_pose_abs, shape=tcp_pose_abs.shape, dtype=tcp_pose_abs.dtype)
        data.array(
            "gripper_width",
            gripper_width,
            shape=gripper_width.shape,
            dtype=gripper_width.dtype,
        )
    return path


def _write_real_data_readiness_zarr_metadata(
    path: Path,
    dataset_type: str,
    image_size=(256, 256),
    image_resize_mode="stretch",
):
    root = zarr.open(str(path), mode="w")
    meta = root.create_group("meta")
    episode_ends = np.asarray([1], dtype=np.int64)
    meta.array("episode_ends", episode_ends, shape=episode_ends.shape, dtype=episode_ends.dtype)
    meta.attrs["dataset_type"] = dataset_type
    meta.attrs["image_size"] = [int(image_size[0]), int(image_size[1])]
    meta.attrs["image_resize_mode"] = image_resize_mode
    return path


def _add_timestamp_arrays(path: Path, offsets=None, nonmonotonic: bool = False):
    if offsets is None:
        offsets = {
            "image_timestamp": 0.0,
            "robot_state_timestamp": 0.002,
            "action_timestamp": 0.004,
            "gaze_timestamp": 0.003,
        }
    root = zarr.open(str(path), mode="a")
    data = root["data"]
    length = int(data["camera0_rgb"].shape[0])
    base = np.arange(length, dtype=np.float64) * 0.05
    if nonmonotonic and length > 2:
        base[2] = base[1] - 0.01
    if "timestamp" in data:
        del data["timestamp"]
    data.array("timestamp", base, shape=base.shape, dtype=base.dtype)
    for key, offset in offsets.items():
        if key in data:
            del data[key]
        values = base + float(offset)
        data.array(key, values, shape=values.shape, dtype=values.dtype)
    return path


def _write_multi_episode_gaze_wam_zarr(path: Path, include_action=True, episode_lengths=(4, 5, 6, 7)):
    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    episode_lengths = np.asarray(episode_lengths, dtype=np.int64)
    episode_ends = np.cumsum(episode_lengths)
    meta.array("episode_ends", episode_ends, shape=episode_ends.shape, dtype=episode_ends.dtype)

    length = int(episode_ends[-1])
    image = np.zeros((length, 16, 16, 3), dtype=np.uint8)
    for i in range(length):
        image[i] = i
    timestamp = np.arange(length, dtype=np.float64) * 0.05
    gaze = np.stack(
        [
            np.linspace(0.1, 0.9, length, dtype=np.float32),
            np.linspace(0.9, 0.1, length, dtype=np.float32),
        ],
        axis=-1,
    )
    data.array("camera0_rgb", image, shape=image.shape, dtype=image.dtype)
    data.array("gaze_xy", gaze, shape=gaze.shape, dtype=gaze.dtype)
    heatmap = gaussian_heatmaps_from_points(
        gaze,
        image_size=(16, 16),
        sigma_px=4.0,
        episode_ends=episode_ends,
    )
    has_gaze_label = np.ones((length,), dtype=np.bool_)
    has_heatmap_image = np.ones((length,), dtype=np.bool_)
    data.array("gaze_heatmap", heatmap, shape=heatmap.shape, dtype=heatmap.dtype)
    data.array("has_gaze_label", has_gaze_label, shape=has_gaze_label.shape, dtype=has_gaze_label.dtype)
    data.array(
        "has_heatmap_image",
        has_heatmap_image,
        shape=has_heatmap_image.shape,
        dtype=has_heatmap_image.dtype,
    )

    if include_action:
        action_abs = _random_pose10((length,)).astype(np.float32)
        tcp_pose_abs = action_abs[:, :9].copy()
        gripper_width = action_abs[:, 9:10].copy()
        data.array("action_abs_tcp", action_abs, shape=action_abs.shape, dtype=action_abs.dtype)
        data.array("tcp_pose_abs", tcp_pose_abs, shape=tcp_pose_abs.shape, dtype=tcp_pose_abs.dtype)
        data.array(
            "gripper_width",
            gripper_width,
            shape=gripper_width.shape,
            dtype=gripper_width.dtype,
        )
    return path


def _write_linear_action_zarr(path: Path, length=8):
    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    episode_ends = np.asarray([length], dtype=np.int64)
    meta.array("episode_ends", episode_ends, shape=episode_ends.shape, dtype=episode_ends.dtype)

    image = np.zeros((length, 16, 16, 3), dtype=np.uint8)
    gaze = np.full((length, 2), 0.5, dtype=np.float32)
    action_abs = np.zeros((length, 10), dtype=np.float32)
    action_abs[:, 0] = np.arange(length, dtype=np.float32)
    action_abs[:, 3] = 1.0
    action_abs[:, 7] = 1.0
    action_abs[:, 9] = np.arange(length, dtype=np.float32) * 0.01
    tcp_pose_abs = action_abs[:, :9].copy()
    gripper_width = action_abs[:, 9:10].copy()

    data.array("camera0_rgb", image, shape=image.shape, dtype=image.dtype)
    data.array("gaze_xy", gaze, shape=gaze.shape, dtype=gaze.dtype)
    heatmap = gaussian_heatmaps_from_points(gaze, image_size=(16, 16), sigma_px=4.0)
    has_gaze_label = np.ones((length,), dtype=np.bool_)
    has_heatmap_image = np.ones((length,), dtype=np.bool_)
    data.array("gaze_heatmap", heatmap, shape=heatmap.shape, dtype=heatmap.dtype)
    data.array("has_gaze_label", has_gaze_label, shape=has_gaze_label.shape, dtype=has_gaze_label.dtype)
    data.array(
        "has_heatmap_image",
        has_heatmap_image,
        shape=has_heatmap_image.shape,
        dtype=has_heatmap_image.dtype,
    )
    data.array("action_abs_tcp", action_abs, shape=action_abs.shape, dtype=action_abs.dtype)
    data.array("tcp_pose_abs", tcp_pose_abs, shape=tcp_pose_abs.shape, dtype=tcp_pose_abs.dtype)
    data.array(
        "gripper_width",
        gripper_width,
        shape=gripper_width.shape,
        dtype=gripper_width.dtype,
    )
    return path


def _write_linear_action_pose_only_zarr(path: Path, length=8):
    _write_linear_action_zarr(path, length=length)
    root = zarr.open(str(path), mode="a")
    data = root["data"]
    action_abs = np.asarray(data["action_abs_tcp"][:], dtype=np.float32)
    del data["action_abs_tcp"]
    data.array(
        "action_abs_tcp",
        action_abs[:, :9],
        shape=action_abs[:, :9].shape,
        dtype=action_abs.dtype,
    )
    return path


def _write_open_heatmap_only_zarr(path: Path):
    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    episode_ends = np.asarray([6], dtype=np.int64)
    meta.array("episode_ends", episode_ends, shape=episode_ends.shape, dtype=episode_ends.dtype)

    image = np.zeros((6, 32, 32, 3), dtype=np.uint8)
    heatmap = np.zeros((6, 32, 32), dtype=np.float32)
    heatmap[:, 8:16, 16:24] = 1.0

    data.array("camera0_rgb", image, shape=image.shape, dtype=image.dtype)
    data.array("gaze_heatmap", heatmap, shape=heatmap.shape, dtype=heatmap.dtype)
    return path


def _write_hot3d_processed_sequence(root: Path, sequence_id="P0001_test", frames=8, image_hw=(32, 32)):
    sequence_dir = root / sequence_id
    sequence_dir.mkdir(parents=True, exist_ok=True)
    height, width = image_hw
    video_path = sequence_dir / "raw_rgb.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create synthetic HOT3D mp4 for regression.")
    try:
        for idx in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = idx * 10
            frame[:, :, 1] = 40
            frame[:, :, 2] = 200 - idx * 5
            writer.write(frame)
    finally:
        writer.release()

    fieldnames = [
        "sequence",
        "frame_index",
        "timecode_timestamp_ns",
        "raw_width",
        "raw_height",
        "gaze_source",
        "gaze_available",
        "raw_x_px",
        "raw_y_px",
        "raw_x_norm",
        "raw_y_norm",
        "in_raw_bounds",
        "upright_x_px",
        "upright_y_px",
        "upright_x_norm",
        "upright_y_norm",
        "upright_width",
        "upright_height",
        "tracking_timestamp_us",
        "yaw_rad",
        "pitch_rad",
        "depth_m",
        "yaw_low_rad",
        "yaw_high_rad",
        "pitch_low_rad",
        "pitch_high_rad",
        "session_uid",
    ]
    with (sequence_dir / "gaze_projected_raw_rgb_normalized.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(frames):
            x = 0.25 + 0.5 * idx / max(frames - 1, 1)
            y = 0.75 - 0.5 * idx / max(frames - 1, 1)
            writer.writerow(
                {
                    "sequence": sequence_id,
                    "frame_index": idx,
                    "timecode_timestamp_ns": 1_000_000_000 + idx * 33_333_333,
                    "raw_width": width,
                    "raw_height": height,
                    "gaze_source": "synthetic",
                    "gaze_available": "True",
                    "raw_x_px": x * width,
                    "raw_y_px": y * height,
                    "raw_x_norm": x,
                    "raw_y_norm": y,
                    "in_raw_bounds": "True",
                    "upright_x_px": x * width,
                    "upright_y_px": y * height,
                    "upright_x_norm": x,
                    "upright_y_norm": y,
                    "upright_width": width,
                    "upright_height": height,
                    "tracking_timestamp_us": idx * 33_333,
                    "yaw_rad": "",
                    "pitch_rad": "",
                    "depth_m": "",
                    "yaw_low_rad": "",
                    "yaw_high_rad": "",
                    "pitch_low_rad": "",
                    "pitch_high_rad": "",
                    "session_uid": "synthetic",
                }
            )
    (sequence_dir / "processing_summary.json").write_text(
        json.dumps(
            {
                "sequence": sequence_id,
                "frames": frames,
                "valid_gaze_frames": frames,
                "visible_gaze_frames": frames,
                "upright_width": width,
                "upright_height": height,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return sequence_dir


def _write_robot_heatmap_only_zarr(path: Path):
    _write_gaze_wam_zarr(path, include_action=True)
    root = zarr.open(str(path), mode="a")
    data = root["data"]
    if "gaze_xy" in data:
        del data["gaze_xy"]
    heatmap = np.zeros((6, 32, 32), dtype=np.float32)
    heatmap[:, 8:16, 16:24] = 1.0
    _replace_zarr_array(data, "gaze_heatmap", heatmap)
    return path


def _write_mixed_point_dense_gaze_zarr(path: Path):
    _write_gaze_wam_zarr(path, include_action=True)
    root = zarr.open(str(path), mode="a")
    data = root["data"]
    heatmap = np.zeros((6, 16, 16), dtype=np.float32)
    heatmap[:, 4:12, 2:6] = 1.0
    has_gaze_label = np.asarray([1, 0, 1, 1, 1, 1], dtype=np.uint8)
    has_gaze_condition = np.ones((6,), dtype=np.uint8)
    gaze_xy = np.asarray(data["gaze_xy"][:], dtype=np.float32)
    gaze_xy[1] = np.asarray([1.25, 0.4], dtype=np.float32)
    _replace_zarr_array(data, "gaze_heatmap", heatmap)
    _replace_zarr_array(data, "gaze_xy", gaze_xy)
    _replace_zarr_array(data, "has_gaze_condition", has_gaze_condition)
    _replace_zarr_array(data, "has_gaze_label", has_gaze_label)
    return path


def _write_noncanonical_robot_zarr(path: Path, image_hw=(20, 40)):
    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    episode_ends = np.asarray([6], dtype=np.int64)
    meta.array("episode_ends", episode_ends, shape=episode_ends.shape, dtype=episode_ends.dtype)

    image_h, image_w = image_hw
    image = np.zeros((6, image_h, image_w, 3), dtype=np.uint8)
    for i in range(6):
        image[i, :, :, 0] = i * 10
    timestamp = np.arange(6, dtype=np.float64) * 0.05
    action_pose = _random_pose10((6,))[:, :9].astype(np.float32)
    tcp_pose = _random_pose10((6,))[:, :9].astype(np.float32)
    gripper = np.linspace(0.01, 0.06, 6, dtype=np.float32)
    gaze_pixel = np.stack(
        [
            np.linspace(0, image_w - 1, 6, dtype=np.float32),
            np.linspace(0, image_h - 1, 6, dtype=np.float32),
        ],
        axis=-1,
    )

    data.array("front_rgb", image, shape=image.shape, dtype=image.dtype)
    data.array("sensor_time", timestamp, shape=timestamp.shape, dtype=timestamp.dtype)
    data.array("image_timestamp", timestamp + 0.001, shape=timestamp.shape, dtype=timestamp.dtype)
    data.array(
        "robot_state_timestamp",
        timestamp + 0.002,
        shape=timestamp.shape,
        dtype=timestamp.dtype,
    )
    data.array("action_timestamp", timestamp + 0.004, shape=timestamp.shape, dtype=timestamp.dtype)
    data.array("gaze_timestamp", timestamp + 0.003, shape=timestamp.shape, dtype=timestamp.dtype)
    data.array("future_tcp_pose", action_pose, shape=action_pose.shape, dtype=action_pose.dtype)
    data.array("current_tcp_pose", tcp_pose, shape=tcp_pose.shape, dtype=tcp_pose.dtype)
    data.array("jaw_width", gripper, shape=gripper.shape, dtype=gripper.dtype)
    data.array("eye_pixel_xy", gaze_pixel, shape=gaze_pixel.shape, dtype=gaze_pixel.dtype)
    return path


def _write_noncanonical_robot_heatmap_only_zarr(path: Path, image_hw=(20, 40)):
    _write_noncanonical_robot_zarr(path, image_hw=image_hw)
    root = zarr.open(str(path), mode="a")
    data = root["data"]
    if "eye_pixel_xy" in data:
        del data["eye_pixel_xy"]
    image_h, image_w = image_hw
    heatmap = np.zeros((6, image_h, image_w), dtype=np.float32)
    heatmap[:, image_h // 4 : image_h // 2, image_w // 3 : image_w // 2] = 1.0
    data.array("eye_gaze_heatmap", heatmap, shape=heatmap.shape, dtype=heatmap.dtype)
    return path


class FakeTokenObsEncoder(nn.Module):
    def __init__(self, num_tokens=8, embed_dim=32):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.proj = nn.Linear(3, embed_dim)

    def output_shape(self):
        return (1, self.num_tokens, self.embed_dim)

    def forward(self, obs_dict):
        self.last_obs_keys = tuple(sorted(obs_dict.keys()))
        image = obs_dict["camera0_rgb"]
        pooled = image.mean(dim=(-1, -2))
        token_seed = self.proj(pooled).mean(dim=1)
        return token_seed[:, None, :].expand(-1, self.num_tokens, -1)


def _write_fake_cosmos_jit_pair(
    tmp_path: Path,
    image_size=(32, 32),
    token_grid=(4, 4),
    latent_channels=4,
):
    """Create a tiny frozen-Cosmos stand-in for policy/unit tests."""
    image_h, image_w = image_size
    token_h, token_w = token_grid
    pool_h = image_h // token_h
    pool_w = image_w // token_w

    class FakeCosmosEncoder(nn.Module):
        def forward(self, image):
            return torch.nn.functional.avg_pool2d(
                image[:, :1],
                kernel_size=(pool_h, pool_w),
                stride=(pool_h, pool_w),
            ).repeat(1, latent_channels, 1, 1)

    class FakeCosmosDecoder(nn.Module):
        def forward(self, latent):
            image = latent[:, :1]
            image = torch.nn.functional.interpolate(
                image,
                size=(image_h, image_w),
                mode="nearest",
            )
            return image.repeat(1, 3, 1, 1) * 2.0 - 1.0

    encoder_path = tmp_path / "encoder.jit"
    decoder_path = tmp_path / "decoder.jit"
    torch.jit.trace(
        FakeCosmosEncoder(),
        torch.zeros(1, 3, image_h, image_w),
    ).save(str(encoder_path))
    torch.jit.trace(
        FakeCosmosDecoder(),
        torch.zeros(1, latent_channels, token_h, token_w),
    ).save(str(decoder_path))
    return str(encoder_path), str(decoder_path)


class SplitTokenObsEncoder(nn.Module):
    def __init__(self, num_tokens=8, embed_dim=32):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.key_model_map = nn.ModuleDict({"camera0_rgb": nn.Linear(3, embed_dim)})
        self.adapter = nn.Linear(embed_dim, embed_dim)

    def output_shape(self):
        return (1, self.num_tokens, self.embed_dim)

    def forward(self, obs_dict):
        image = obs_dict["camera0_rgb"]
        pooled = image.mean(dim=(-1, -2))
        token_seed = self.adapter(self.key_model_map["camera0_rgb"](pooled)).mean(dim=1)
        return token_seed[:, None, :].expand(-1, self.num_tokens, -1)


def test_heatmap_codec_point_tokens_and_decode():
    codec = HeatmapTokenCodec(token_grid=(16, 16), image_size=(256, 256), sigma_tokens=1.25)
    row, col = 7, 8
    point = torch.tensor([[(col + 0.5) / 16, (row + 0.5) / 16]], dtype=torch.float32)

    tokens = codec.encode_points(point)

    assert tokens.shape == (1, 256, 1)
    assert torch.isclose(tokens.max(), torch.tensor(1.0))
    assert tokens.argmax().item() == row * 16 + col

    decoded = codec.decode_tokens(tokens, method="gaussian_splat")
    assert decoded.shape == (1, 256, 256)
    assert decoded.min() >= 0
    assert decoded.max() <= 1.00001

    bilinear = codec.decode_tokens(tokens, method="bilinear")
    assert bilinear.shape == (1, 256, 256)


def test_heatmap_codec_image_pooling_and_valid_mask():
    codec = HeatmapTokenCodec(token_grid=(16, 16), image_size=(256, 256), sigma_tokens=1.25)
    image = torch.zeros(2, 256, 256)
    image[:, 112:128, 128:144] = 1.0

    tokens = codec.encode_image(image)
    channel_tokens = codec.encode_image(image.unsqueeze(1))
    masked = codec.encode_points(torch.tensor([[0.5, 0.5], [0.25, 0.25]]), valid_mask=torch.tensor([1, 0]))

    assert tokens.shape == (2, 256, 1)
    assert channel_tokens.shape == (2, 256, 1)
    assert torch.allclose(channel_tokens, tokens)
    assert torch.isclose(tokens.max(), torch.tensor(1.0))
    assert masked.shape == (2, 256, 1)
    assert torch.allclose(masked[1], torch.zeros_like(masked[1]))

    invalid_images = [
        ([[0.0]], "heatmap_image"),
        (torch.zeros(2, 256, 256, dtype=torch.int64), "floating point"),
        (torch.full((2, 256, 256), float("nan")), "finite values"),
        (torch.full((2, 256, 256), -0.1), "non-negative"),
        (torch.zeros(2, 128, 256), "spatial shape"),
        (torch.zeros(2, 3, 256, 256), "single-channel"),
        (torch.zeros(2, 256, 256, 1), "spatial shape"),
    ]
    for bad_image, expected in invalid_images:
        try:
            codec.encode_image(bad_image)
        except (TypeError, ValueError) as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"Expected invalid heatmap image to fail: {expected}.")


def test_heatmap_codec_patchify_unpatchify_full_resolution_heatmap():
    codec = HeatmapTokenCodec(token_grid=(16, 16), image_size=(256, 256), sigma_tokens=1.25)
    image = torch.rand(2, 1, 256, 256)

    tokens = codec.patchify_image(image)
    decoded = codec.unpatchify_tokens(tokens)
    decoded_via_decode = codec.decode_tokens(tokens)

    assert codec.patch_size == (16, 16)
    assert codec.patch_area == 256
    assert tokens.shape == (2, 256, 256)
    assert decoded.shape == (2, 256, 256)
    assert torch.allclose(decoded, image[:, 0], atol=1e-6)
    assert torch.allclose(decoded_via_decode, image[:, 0], atol=1e-6)


def test_cosmos_heatmap_codec_wraps_jit_modules(tmp_path):
    class FakeCosmosEncoder(nn.Module):
        def forward(self, image):
            return torch.nn.functional.avg_pool2d(
                image[:, :1],
                kernel_size=8,
                stride=8,
            ).repeat(1, 4, 1, 1)

    class FakeCosmosDecoder(nn.Module):
        def forward(self, latent):
            image = latent[:, :1]
            image = torch.nn.functional.interpolate(image, size=(32, 32), mode="nearest")
            return image.repeat(1, 3, 1, 1) * 2.0 - 1.0

    encoder_path = tmp_path / "encoder.jit"
    decoder_path = tmp_path / "decoder.jit"
    torch.jit.trace(FakeCosmosEncoder(), torch.zeros(1, 3, 32, 32)).save(str(encoder_path))
    torch.jit.trace(FakeCosmosDecoder(), torch.zeros(1, 4, 4, 4)).save(str(decoder_path))

    codec = CosmosHeatmapCodec(
        encoder_path=str(encoder_path),
        decoder_path=str(decoder_path),
        token_grid=(4, 4),
        image_size=(32, 32),
        latent_channels=4,
        input_range="minus_one_one",
        output_range="minus_one_one",
    )
    image = torch.rand(2, 32, 32)

    tokens = codec.encode_image(image)
    train_tokens = tokens.detach().requires_grad_(True)
    decoded = codec.decode_tokens(train_tokens)
    decoded.square().mean().backward()

    assert tokens.shape == (2, 16, 4)
    assert decoded.shape == (2, 32, 32)
    assert codec.latent_image_size == (4, 4)
    assert train_tokens.grad is not None
    assert torch.isfinite(train_tokens.grad).all()
    assert not any(param.requires_grad for param in codec.parameters())


def test_heatmap_codec_rejects_invalid_valid_mask_and_decode_size():
    codec = HeatmapTokenCodec(token_grid=(16, 16), image_size=(256, 256), sigma_tokens=1.25)
    points = torch.tensor([[0.5, 0.5], [0.25, 0.25]], dtype=torch.float32)
    tokens = codec.encode_points(points)

    bad_masks = [
        torch.tensor([1.0, 0.5]),
        torch.tensor([True, False]).reshape(2, 1),
    ]
    for mask in bad_masks:
        try:
            codec.encode_points(points, valid_mask=mask)
        except ValueError as exc:
            assert "valid_mask" in str(exc)
        else:
            raise AssertionError(f"Expected invalid heatmap valid_mask to fail: {mask!r}")

    for bad_size in ((True, 256), (128.5, 256), (0, 256)):
        try:
            codec.decode_tokens(tokens, image_size=bad_size)
        except ValueError as exc:
            assert "image_size dimensions must be positive" in str(exc)
        else:
            raise AssertionError(f"Expected invalid decode image_size to fail: {bad_size!r}")

    invalid_token_cases = [
        ([[0.0]], "tokens must be a torch.Tensor"),
        (torch.zeros(2, 256, 1, dtype=torch.int64), "floating point"),
        (torch.full((2, 256, 1), float("nan")), "finite values"),
        (torch.full((2, 256, 1), -0.1), "non-negative"),
        (torch.zeros(2, 255, 1), "Expected scalar tokens"),
    ]
    for bad_tokens, expected in invalid_token_cases:
        try:
            codec.decode_tokens(bad_tokens)
        except (TypeError, ValueError) as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"Expected invalid decode tokens to fail: {expected}.")


def test_heatmap_codec_rejects_invalid_gaze_points():
    codec = HeatmapTokenCodec(token_grid=(16, 16), image_size=(256, 256), sigma_tokens=1.25)
    bad_points = [
        [[0.5, 0.5]],
        torch.tensor([[0, 1]], dtype=torch.int64),
        torch.tensor([[float("nan"), 0.5]], dtype=torch.float32),
        torch.tensor([[1.1, 0.5]], dtype=torch.float32),
    ]

    for point in bad_points:
        try:
            codec.encode_points(point)
        except (TypeError, ValueError) as exc:
            assert "gaze_xy" in str(exc)
        else:
            raise AssertionError(f"Expected invalid gaze point to fail: {point!r}")


def test_heatmap_codec_rejects_nonpositive_geometry():
    bad_cases = [
        ("token_grid", {"token_grid": (0, 16)}, "token_grid dimensions must be positive"),
        ("token_grid", {"token_grid": (True, 16)}, "token_grid dimensions must be positive"),
        ("token_grid", {"token_grid": ("4.5", 16)}, "token_grid dimensions must be positive"),
        ("image_size", {"image_size": (0, 256)}, "image_size dimensions must be positive"),
        ("image_size", {"image_size": (True, 256)}, "image_size dimensions must be positive"),
        ("sigma_tokens", {"sigma_tokens": 0.0}, "sigma_tokens must be positive"),
        ("sigma_tokens", {"sigma_tokens": True}, "sigma_tokens must be positive"),
        ("sigma_tokens", {"sigma_tokens": float("nan")}, "sigma_tokens must be positive"),
    ]
    for name, overrides, expected in bad_cases:
        kwargs = {
            "token_grid": (16, 16),
            "image_size": (256, 256),
            "sigma_tokens": 1.25,
        }
        kwargs.update(overrides)
        try:
            HeatmapTokenCodec(**kwargs)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"Expected invalid {name} to fail.")


def test_gaussian_spatial_encoder_and_mask_token():
    spatial = GaussianSpatialEncoder(grid_size=(8, 8), sigma=0.15)
    xy = torch.tensor([[0.5, 0.5], [0.1, 0.9]], dtype=torch.float32)
    basis = spatial(xy)

    assert basis.shape == (2, 64)
    assert torch.allclose(basis.max(dim=-1).values, torch.ones(2))
    string_geometry = GaussianSpatialEncoder(grid_size=("8", "8"), sigma="0.15")
    assert string_geometry.grid_size == (8, 8)
    assert string_geometry.sigma == 0.15

    invalid_geometry_cases = [
        ({"grid_size": (0, 8)}, "grid_size dimensions must be positive"),
        ({"grid_size": (True, 8)}, "grid_size dimensions must be positive"),
        ({"grid_size": (8.5, 8)}, "grid_size dimensions must be positive"),
        ({"grid_size": (8,)}, "grid_size must be a pair"),
        ({"sigma": 0.0}, "sigma must be positive"),
        ({"sigma": True}, "sigma must be positive"),
        ({"sigma": float("nan")}, "sigma must be positive"),
    ]
    for kwargs, expected in invalid_geometry_cases:
        options = {"grid_size": (8, 8), "sigma": 0.15}
        options.update(kwargs)
        try:
            GaussianSpatialEncoder(**options)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"Expected invalid GaussianSpatialEncoder config: {kwargs!r}.")

    invalid_points = [
        ([[0.5, 0.5]], "gaze_xy"),
        (torch.tensor([[0, 1]], dtype=torch.int64), "floating point"),
        (torch.tensor([[float("nan"), 0.5]], dtype=torch.float32), "finite values"),
    ]
    for bad_point, expected in invalid_points:
        try:
            spatial(bad_point)
        except (TypeError, ValueError) as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"Expected invalid spatial gaze input to fail: {expected}.")

    encoder = GazeConditionEncoder(embed_dim=32, grid_size=(8, 8), sigma=0.15)
    use_gaze = torch.tensor([True, False])
    token = encoder(xy, use_gaze_condition=use_gaze)

    assert token.shape == (2, 1, 32)
    assert torch.allclose(token[1], encoder.mask_token[0])
    assert not torch.allclose(token[0], encoder.mask_token[0])

    try:
        spatial(xy, valid_mask=torch.ones(2))
    except ValueError as exc:
        assert "valid_mask must be a BoolTensor" in str(exc)
    else:
        raise AssertionError("Expected non-bool Gaussian valid_mask to fail.")

    try:
        encoder(xy, has_gaze_label=torch.ones(2))
    except ValueError as exc:
        assert "has_gaze_label must be a BoolTensor" in str(exc)
    else:
        raise AssertionError("Expected non-bool encoder has_gaze_label to fail.")

    outside_basis = spatial(torch.tensor([[1.2, -0.1]], dtype=torch.float32))
    extreme_basis = spatial(torch.tensor([[100.0, -100.0]], dtype=torch.float32))
    clamped_basis = spatial(torch.tensor([[1.5, -0.5]], dtype=torch.float32))
    assert torch.isfinite(outside_basis).all()
    assert torch.allclose(extreme_basis, clamped_basis)

    try:
        encoder(xy, use_gaze_condition=torch.ones(2, 1, dtype=torch.bool))
    except ValueError as exc:
        assert "use_gaze_condition must have shape" in str(exc)
    else:
        raise AssertionError("Expected misshaped encoder route mask to fail.")

    try:
        encoder(
            xy,
            use_gaze_condition=torch.tensor([True, False]),
            has_gaze_condition=torch.tensor([False, True]),
            has_gaze_label=torch.tensor([False, True]),
        )
    except ValueError as exc:
        assert "use_gaze_condition cannot be True" in str(exc)
        assert "trainable mask token" in str(exc)
    else:
        raise AssertionError("Expected missing gaze label route to force the mask token.")


def test_relative_action_roundtrip_single_and_batch():
    base = _random_pose10(())
    actions = _random_pose10((5,))
    rel = absolute_actions_to_relative_actions(actions, base)
    reconstructed = relative_actions_to_absolute_actions(rel, base)

    assert rel.shape == actions.shape
    assert np.allclose(reconstructed, actions, atol=1e-6)
    assert np.allclose(rel[:, 9], actions[:, 9])

    batch_base = _random_pose10((3,))
    batch_actions = _random_pose10((3, 4))
    batch_rel = absolute_actions_to_relative_actions(batch_actions, batch_base)
    batch_reconstructed = relative_actions_to_absolute_actions(batch_rel, batch_base)

    assert batch_rel.shape == batch_actions.shape
    assert np.allclose(batch_reconstructed, batch_actions, atol=1e-6)


def test_relative_action_conversion_uses_finite_float_arrays():
    base9 = np.asarray([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.int64)
    actions9 = np.tile(base9, (2, 1))
    actions9[:, 0] = [1, 2]

    rel = absolute_actions_to_relative_actions(actions9, base9)
    reconstructed = relative_actions_to_absolute_actions(rel, base9)

    assert np.issubdtype(rel.dtype, np.floating)
    assert np.allclose(rel[:, 0], [1.0, 2.0], atol=1e-6)
    assert np.allclose(reconstructed, actions9.astype(np.float64), atol=1e-6)

    bad_actions = actions9.astype(np.float32)
    bad_actions[0, 0] = np.nan
    try:
        absolute_actions_to_relative_actions(bad_actions, base9)
    except ValueError as exc:
        assert "actions must contain only finite values" in str(exc)
    else:
        raise AssertionError("Expected non-finite action conversion input to fail.")

    bad_base = base9.astype(np.float32)
    bad_base[0] = np.inf
    try:
        absolute_actions_to_relative_actions(actions9, bad_base)
    except ValueError as exc:
        assert "base_absolute_action must contain only finite values" in str(exc)
    else:
        raise AssertionError("Expected non-finite base action conversion input to fail.")


def test_distributed_masked_mean_local_behavior():
    loss = torch.tensor([1.0, 2.0, 10.0])
    mask = torch.tensor([True, True, False])
    out = distributed_masked_mean(loss, mask)
    zero = distributed_masked_mean(loss, torch.zeros(3, dtype=torch.bool))
    count = distributed_mask_count(mask)

    assert torch.allclose(out, torch.tensor(1.5))
    assert torch.allclose(zero, torch.tensor(0.0))
    assert torch.allclose(count, torch.tensor(2.0))

    invalid_masks = [
        (torch.tensor([1.0, 0.0, 1.0]), "BoolTensor"),
        (torch.tensor([[True], [False], [True]]), "shape [B]"),
        ([True, False, True], "torch.Tensor"),
    ]
    for bad_mask, expected in invalid_masks:
        try:
            distributed_masked_mean(loss, bad_mask)
        except (TypeError, ValueError) as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"Expected invalid distributed mask to fail: {expected}.")

    try:
        distributed_mask_count(torch.tensor([1, 0, 1]))
    except ValueError as exc:
        assert "BoolTensor" in str(exc)
    else:
        raise AssertionError("Expected non-bool distributed mask count to fail.")

    try:
        distributed_mask_count(torch.tensor([[True], [False], [True]]))
    except ValueError as exc:
        assert "shape [B]" in str(exc)
    else:
        raise AssertionError("Expected broadcast-shaped distributed mask count to fail.")


def test_dsnt_js_heatmap_losses_use_dense_spatial_distribution():
    logits = torch.full((2, 8, 8), -8.0)
    logits[0, 1, 6] = 8.0
    logits[1, 6, 1] = 8.0
    prob = spatial_softmax_2d(logits, temperature=1.0)
    routed_prob = spatial_distribution_2d(
        logits,
        mode="logits_softmax",
        temperature=1.0,
    )
    assert torch.allclose(routed_prob, prob)
    xy = dsnt_expectation(prob)

    expected_xy = torch.tensor(
        [
            [(6.5 / 8.0), (1.5 / 8.0)],
            [(1.5 / 8.0), (6.5 / 8.0)],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(xy, expected_xy, atol=1e-3)

    xy_loss = per_sample_dsnt_xy_loss(logits, expected_xy, temperature=1.0)
    shifted_xy_loss = per_sample_dsnt_xy_loss(
        torch.flip(logits, dims=(-1,)),
        expected_xy,
        temperature=1.0,
    )
    assert torch.all(xy_loss < 1e-4)
    assert torch.all(shifted_xy_loss > xy_loss)

    point_nll = per_sample_spatial_point_nll_loss(
        logits,
        expected_xy,
        temperature=1.0,
    )
    shifted_point_nll = per_sample_spatial_point_nll_loss(
        torch.flip(logits, dims=(-1,)),
        expected_xy,
        temperature=1.0,
    )
    assert torch.all(point_nll < 1e-4)
    assert torch.all(shifted_point_nll > point_nll)

    target = prob.detach()
    same_js = per_sample_spatial_js_loss(logits, target, temperature=1.0)
    shifted_js = per_sample_spatial_js_loss(
        torch.flip(logits, dims=(-1,)),
        target,
        temperature=1.0,
    )
    assert torch.all(same_js < 1e-6)
    assert torch.all(shifted_js > same_js)

    decoded_intensity = torch.tensor(
        [[[-1.0, 0.0], [2.0, 6.0]]],
        dtype=torch.float32,
    )
    clamp_prob = spatial_distribution_2d(decoded_intensity, mode="intensity_clamp")
    expected_clamp = torch.tensor([[[0.0, 0.0], [0.25, 0.75]]])
    assert torch.allclose(clamp_prob, expected_clamp)

    softplus_prob = spatial_distribution_2d(
        decoded_intensity,
        mode="intensity_softplus",
        temperature=0.1,
    )
    assert softplus_prob[0, 0, 0] == 0.0
    assert softplus_prob[0, 0, 1] == 0.0
    assert torch.isclose(softplus_prob.sum(), torch.tensor(1.0))
    assert softplus_prob[0, 1, 1] > softplus_prob[0, 1, 0]

    zero_intensity = torch.zeros((1, 2, 2), dtype=torch.float32)
    zero_prob = spatial_distribution_2d(
        zero_intensity,
        mode="intensity_softplus",
        temperature=0.1,
    )
    assert torch.count_nonzero(zero_prob) == 0


def test_gaze_dependency_ratio_metric():
    conditioned = torch.tensor(
        [
            [[3.0, 4.0]],
            [[1.0, 0.0]],
        ]
    )
    masked = torch.tensor(
        [
            [[0.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    ratio = gaze_dependency_ratio(conditioned, masked)

    assert torch.allclose(ratio, torch.tensor([1.0, 0.0]))


def test_joint_transformer_train_and_inference_shapes():
    torch.manual_seed(0)
    model = JointGazeWamTransformer(
        action_dim=10,
        heatmap_dim=1,
        action_horizon=16,
        heatmap_num_tokens=256,
        max_image_tokens=512,
        n_layer=1,
        n_head=4,
        n_emb=32,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
    )
    image_tokens = torch.randn(2, 512, 32)
    gaze_token = torch.randn(2, 1, 32)
    noisy_action = torch.randn(2, 16, 10)
    noisy_heatmap = torch.randn(2, 256, 1)
    timesteps = torch.tensor([3, 7])

    train_out = model(
        image_tokens=image_tokens,
        gaze_token=gaze_token,
        noisy_action=noisy_action,
        noisy_heatmap=noisy_heatmap,
        timestep=timesteps,
    )
    infer_out = model(
        image_tokens=image_tokens,
        gaze_token=gaze_token,
        noisy_action=noisy_action,
        noisy_heatmap=None,
        timestep=timesteps,
        is_inference=True,
    )

    assert train_out.action.shape == (2, 16, 10)
    assert train_out.heatmap.shape == (2, 256, 1)
    assert train_out.action_features.shape == (2, 16, 32)
    assert train_out.heatmap_features.shape == (2, 256, 32)
    assert infer_out.action.shape == (2, 16, 10)
    assert infer_out.heatmap is None
    assert infer_out.heatmap_features is None
    try:
        model(
            image_tokens=image_tokens,
            gaze_token=gaze_token,
            noisy_action=noisy_action,
            noisy_heatmap=None,
            timestep=timesteps,
            is_inference=False,
        )
    except ValueError as exc:
        assert "Training forward must include noisy_heatmap" in str(exc)
    else:
        raise AssertionError("Expected training forward without noisy_heatmap to fail.")


def test_cached_dual_stream_transformer_shapes_and_world_cache():
    torch.manual_seed(0)
    model = CachedDualStreamGazeWamTransformer(
        action_dim=10,
        heatmap_dim=16,
        action_horizon=4,
        heatmap_num_tokens=4,
        max_image_tokens=8,
        n_layer=2,
        n_head=4,
        n_emb=32,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
    )
    image_tokens = torch.randn(2, 8, 32)
    gaze_token = torch.randn(2, 1, 32)
    noisy_action = torch.randn(2, 4, 10)
    noisy_heatmap = torch.randn(2, 4, 16)
    timesteps = torch.tensor([2, 5])

    cache = model.prefill_world_cache(image_tokens=image_tokens, gaze_token=gaze_token)
    legacy_cache = model.prefill_condition_cache(image_tokens=image_tokens, gaze_token=gaze_token)
    assert len(cache.key_values) == 2
    assert len(legacy_cache.key_values) == 2
    first_key, first_value = cache.key_values[0]
    assert first_key.shape == (2, 4, 9, 8)
    assert first_value.shape == (2, 4, 9, 8)

    train_out = model(
        image_tokens=image_tokens,
        gaze_token=gaze_token,
        noisy_action=noisy_action,
        noisy_heatmap=noisy_heatmap,
        timestep=timesteps,
        world_cache=cache,
    )
    infer_out = model(
        image_tokens=image_tokens,
        gaze_token=gaze_token,
        noisy_action=noisy_action,
        noisy_heatmap=None,
        timestep=timesteps,
        is_inference=True,
        world_cache=cache,
    )

    assert train_out.action.shape == (2, 4, 10)
    assert train_out.heatmap.shape == (2, 4, 16)
    assert train_out.action_features.shape == (2, 4, 32)
    assert train_out.heatmap_features.shape == (2, 4, 32)
    assert infer_out.action.shape == (2, 4, 10)
    assert infer_out.heatmap is None
    summary = model.attention_contract_summary(num_image_tokens=8)
    assert summary["architecture"] == "cached_dual_stream"
    assert summary["shared_condition_kv_cache"] is False
    assert summary["shared_world_kv_cache"] is True
    assert summary["world_cache_consumed_by_action"] is True
    assert summary["action_reads_heatmap_world_cache"] is True
    assert summary["action_reads_noisy_heatmap"] is False
    assert summary["inference_action_tokens"] == 4

    bad_cache = model.prefill_world_cache(
        image_tokens=torch.randn(1, 8, 32),
        gaze_token=torch.randn(1, 1, 32),
    )
    try:
        model(
            image_tokens=image_tokens,
            gaze_token=gaze_token,
            noisy_action=noisy_action,
            noisy_heatmap=noisy_heatmap,
            timestep=timesteps,
            world_cache=bad_cache,
        )
    except ValueError as exc:
        assert "world_cache batch size" in str(exc)
    else:
        raise AssertionError("Expected mismatched world cache batch to fail.")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA BF16 autocast")
def test_cached_dual_stream_transformer_bf16_autocast():
    model = CachedDualStreamGazeWamTransformer(
        action_dim=10,
        heatmap_dim=16,
        action_horizon=4,
        heatmap_num_tokens=4,
        max_image_tokens=8,
        n_layer=2,
        n_head=4,
        n_emb=32,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
    ).cuda()
    image_tokens = torch.randn(2, 8, 32, device="cuda")
    gaze_token = torch.randn(2, 1, 32, device="cuda")
    noisy_action = torch.randn(2, 4, 10, device="cuda")
    noisy_heatmap = torch.randn(2, 4, 16, device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(
            image_tokens=image_tokens,
            gaze_token=gaze_token,
            noisy_action=noisy_action,
            noisy_heatmap=noisy_heatmap,
            timestep=torch.tensor([2, 5], device="cuda"),
        )

    assert output.action.dtype == torch.bfloat16
    assert output.heatmap.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA BF16")
def test_cached_dual_stream_transformer_native_bf16_weights():
    model = CachedDualStreamGazeWamTransformer(
        action_dim=10,
        heatmap_dim=16,
        action_horizon=4,
        heatmap_num_tokens=4,
        max_image_tokens=8,
        n_layer=2,
        n_head=4,
        n_emb=32,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
    ).cuda().bfloat16()
    output = model(
        image_tokens=torch.randn(2, 8, 32, device="cuda", dtype=torch.bfloat16),
        gaze_token=torch.randn(2, 1, 32, device="cuda", dtype=torch.bfloat16),
        noisy_action=torch.randn(2, 4, 10, device="cuda", dtype=torch.bfloat16),
        noisy_heatmap=torch.randn(2, 4, 16, device="cuda", dtype=torch.bfloat16),
        timestep=torch.tensor([2, 5], device="cuda"),
    )

    assert output.action.dtype == torch.bfloat16
    assert output.heatmap.dtype == torch.bfloat16


def test_cached_dual_stream_policy_patchified_heatmap_prediction_decode(tmp_path):
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    encoder_path, decoder_path = _write_fake_cosmos_jit_pair(
        tmp_path,
        image_size=(8, 8),
        token_grid=(2, 2),
        latent_channels=4,
    )
    policy = GazeWamPolicy(
        shape_meta={"action": {"shape": [10], "horizon": 4}},
        noise_scheduler=scheduler,
        obs_encoder=FakeTokenObsEncoder(num_tokens=4, embed_dim=32),
        model_architecture="cached_dual_stream",
        gaze_encoder=GazeConditionEncoder(embed_dim=32),
        num_inference_steps=2,
        input_pertub=0.0,
        heatmap_num_tokens=4,
        heatmap_dim=16,
        heatmap_token_grid=(2, 2),
        heatmap_image_size=(8, 8),
        heatmap_cosmos_encoder_path=encoder_path,
        heatmap_cosmos_decoder_path=decoder_path,
        n_layer=1,
        n_head=4,
        n_emb=32,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
    )
    assert isinstance(policy.model, CachedDualStreamGazeWamTransformer)

    normalizer = LinearNormalizer()
    normalizer["camera0_rgb"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)

    obs = {
        "camera0_rgb": torch.rand(2, 2, 3, 8, 8),
        "gaze_xy": torch.tensor([[0.5, 0.5], [0.25, 0.75]], dtype=torch.float32),
        "use_gaze_condition": torch.tensor([False, False]),
        "has_gaze_label": torch.tensor([True, True]),
    }
    noisy_action = torch.zeros(2, 4, 10)
    noisy_heatmap = torch.zeros(2, 4, 16)

    pred = policy.predict_heatmap(
        obs,
        noisy_action=noisy_action,
        noisy_heatmap=noisy_heatmap,
        timestep=torch.zeros(2, dtype=torch.long),
        decode=True,
    )
    assert pred["heatmap_tokens"].shape == (2, 4, 16)
    assert pred["heatmap_image"].shape == (2, 8, 8)

    sampled_action = policy.predict_action(obs)
    assert sampled_action["action"].shape == (2, 4, 10)


def test_checkpoint_preview_sample_index_selection_supports_seed_and_explicit_indices():
    seeded_a = _select_checkpoint_preview_sample_indices(
        dataset_len=100,
        max_samples=4,
        sample_seed=7,
    )
    seeded_b = _select_checkpoint_preview_sample_indices(
        dataset_len=100,
        max_samples=4,
        sample_seed=7,
    )
    assert seeded_a == seeded_b
    assert len(seeded_a) == 4
    assert len(set(seeded_a)) == 4
    assert seeded_a != [0, 1, 2, 3]

    explicit = _select_checkpoint_preview_sample_indices(
        dataset_len=10,
        max_samples=2,
        sample_indices=[3, 5, 7],
    )
    assert explicit == [3, 5]

    with pytest.raises(ValueError, match="out of range"):
        _select_checkpoint_preview_sample_indices(
            dataset_len=3,
            max_samples=2,
            sample_indices=[0, 3],
        )


def test_val_heatmap_preview_indices_support_seeded_nonprefix_selection():
    candidates = torch.arange(12)
    seeded_a = _select_heatmap_preview_indices(
        preview_indices=candidates,
        max_samples=4,
        sample_seed=42,
    )
    seeded_b = _select_heatmap_preview_indices(
        preview_indices=candidates,
        max_samples=4,
        sample_seed=42,
    )
    prefix = _select_heatmap_preview_indices(
        preview_indices=candidates,
        max_samples=4,
        sample_seed=None,
    )

    assert torch.equal(seeded_a, seeded_b)
    assert seeded_a.shape == (4,)
    assert not torch.equal(seeded_a, prefix)
    assert torch.equal(prefix, torch.tensor([0, 1, 2, 3]))


def test_checkpoint_heatmap_log_materializes_epoch_preview_bundle(tmp_path):
    source_dir = tmp_path / "media" / "val_heatmap" / "epoch_0003"
    sample_dir = source_dir / "sample_000"
    sample_dir.mkdir(parents=True)
    (source_dir / "summary.json").write_text(
        json.dumps(
            {
                "epoch": 3,
                "paths": {"summary": str(source_dir / "summary.json")},
            }
        ),
        encoding="utf-8",
    )
    (source_dir / "comparison.png").write_bytes(b"comparison")
    (sample_dir / "comparison.png").write_bytes(b"sample comparison")

    manifest_path = _write_checkpoint_heatmap_log(
        output_dir=str(tmp_path),
        global_step=30000,
        checkpoint_epoch=29,
        preview_summary={
            "epoch": 3,
            "paths": {"summary": str(source_dir / "summary.json")},
        },
    )

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest["global_step"] == 30000
    assert manifest["checkpoint_epoch"] == 29
    assert manifest["validation_epoch"] == 3
    assert Path(manifest["comparison"]).exists()
    assert len(manifest["samples"]) == 1
    assert (tmp_path / "media" / "checkpoint_heatmap" / "latest.json").exists()


def test_checkpoint_preview_ema_summary_reports_effective_use():
    no_ema_cfg = OmegaConf.create({"training": {"use_ema": False}})
    assert _checkpoint_preview_ema_summary(no_ema_cfg, use_ema_requested=True) == {
        "use_ema": False,
        "use_ema_requested": True,
        "checkpoint_has_ema": False,
    }

    ema_cfg = OmegaConf.create({"training": {"use_ema": True}})
    assert _checkpoint_preview_ema_summary(ema_cfg, use_ema_requested=True) == {
        "use_ema": True,
        "use_ema_requested": True,
        "checkpoint_has_ema": True,
    }
    assert _checkpoint_preview_ema_summary(ema_cfg, use_ema_requested=False) == {
        "use_ema": False,
        "use_ema_requested": False,
        "checkpoint_has_ema": True,
    }


def test_gaze_wam_policy_rejects_joint_architecture_string():
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    with pytest.raises(ValueError, match="cached_dual_stream"):
        GazeWamPolicy(
            shape_meta={"action": {"shape": [10], "horizon": 4}},
            noise_scheduler=scheduler,
            obs_encoder=FakeTokenObsEncoder(num_tokens=4, embed_dim=32),
            model_architecture="joint",
            gaze_encoder=GazeConditionEncoder(embed_dim=32),
            num_inference_steps=2,
            input_pertub=0.0,
            heatmap_num_tokens=4,
            heatmap_dim=16,
            heatmap_token_grid=(2, 2),
            heatmap_image_size=(8, 8),
            n_layer=1,
            n_head=4,
            n_emb=32,
            p_drop_emb=0.0,
            p_drop_attn=0.0,
        )


def test_cached_dual_stream_can_skip_action_decoder_for_open_only_heatmap_path():
    model = CachedDualStreamGazeWamTransformer(
        action_dim=10,
        heatmap_dim=16,
        action_horizon=4,
        heatmap_num_tokens=4,
        max_image_tokens=8,
        n_layer=1,
        n_head=4,
        n_emb=32,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
    )
    image_tokens = torch.randn(2, 8, 32)
    gaze_token = torch.randn(2, 1, 32)
    noisy_action = torch.randn(2, 4, 10)
    noisy_heatmap = torch.randn(2, 4, 16)
    timesteps = torch.zeros(2, dtype=torch.long)

    out = model(
        image_tokens=image_tokens,
        gaze_token=gaze_token,
        noisy_action=noisy_action,
        noisy_heatmap=noisy_heatmap,
        timestep=timesteps,
        skip_action=True,
    )
    assert out.heatmap.shape == (2, 4, 16)
    assert out.action.shape == (2, 4, 10)
    assert out.action_features.shape == (2, 4, 32)
    assert torch.count_nonzero(out.action) == 0
    assert torch.count_nonzero(out.action_features) == 0

    try:
        model(
            image_tokens=image_tokens,
            gaze_token=gaze_token,
            noisy_action=noisy_action,
            noisy_heatmap=None,
            timestep=timesteps,
            is_inference=True,
            skip_action=True,
        )
    except ValueError as exc:
        assert "skip_action=True is only valid" in str(exc)
    else:
        raise AssertionError("Expected inference skip_action=True to fail.")


def test_joint_transformer_optimizer_groups_cover_params_without_dummy_variable():
    model = JointGazeWamTransformer(
        action_dim=10,
        heatmap_dim=1,
        action_horizon=4,
        heatmap_num_tokens=8,
        max_image_tokens=16,
        n_layer=1,
        n_head=2,
        n_emb=16,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
    )

    groups = model.get_optim_groups(weight_decay=0.01)
    grouped_param_ids = [
        id(param)
        for group in groups
        for param in group["params"]
    ]
    named_params = dict(model.named_parameters())

    assert "_dummy_variable" not in named_params
    assert len(groups) == 2
    assert len(grouped_param_ids) == len(set(grouped_param_ids))
    assert set(grouped_param_ids) == {id(param) for param in named_params.values()}
    assert groups[0]["weight_decay"] == 0.01
    assert groups[1]["weight_decay"] == 0.0


def test_gaze_wam_policy_optimizer_requires_backbone_group_for_separate_obs_lr(tmp_path):
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    shape_meta = {"action": {"shape": [10], "horizon": 4}}
    encoder_path, decoder_path = _write_fake_cosmos_jit_pair(
        tmp_path,
        image_size=(256, 256),
        token_grid=(2, 4),
        latent_channels=1,
    )
    common_policy_kwargs = {
        "shape_meta": shape_meta,
        "noise_scheduler": scheduler,
        "heatmap_num_tokens": 8,
        "heatmap_token_grid": (2, 4),
        "heatmap_image_size": (256, 256),
        "heatmap_dim": 1,
        "heatmap_cosmos_encoder_path": encoder_path,
        "heatmap_cosmos_decoder_path": decoder_path,
        "n_emb": 32,
    }

    split_policy = GazeWamPolicy(
        obs_encoder=SplitTokenObsEncoder(num_tokens=8, embed_dim=32),
        **common_policy_kwargs,
    )
    optimizer = split_policy.get_optimizer(
        lr=1e-4,
        weight_decay=1e-3,
        obs_encoder_lr=1e-5,
        obs_encoder_weight_decay=1e-4,
        betas=(0.9, 0.95),
    )
    assert any(group.get("lr") == 1e-5 for group in optimizer.param_groups)

    plain_policy = GazeWamPolicy(
        obs_encoder=FakeTokenObsEncoder(num_tokens=8, embed_dim=32),
        **common_policy_kwargs,
    )
    try:
        plain_policy.get_optimizer(
            lr=1e-4,
            weight_decay=1e-3,
            obs_encoder_lr=1e-5,
            obs_encoder_weight_decay=1e-4,
            betas=(0.9, 0.95),
        )
    except ValueError as exc:
        assert "key_model_map" in str(exc)
        assert "obs_encoder_lr" in str(exc)
    else:
        raise AssertionError("Expected separate obs encoder LR without backbone group to fail.")


def test_joint_transformer_rejects_mismatched_batches_and_inference_heatmap():
    model = JointGazeWamTransformer(
        action_dim=10,
        heatmap_dim=1,
        action_horizon=4,
        heatmap_num_tokens=8,
        max_image_tokens=16,
        n_layer=1,
        n_head=2,
        n_emb=16,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
    )
    image_tokens = torch.randn(2, 16, 16)
    gaze_token = torch.randn(2, 1, 16)
    noisy_action = torch.randn(2, 4, 10)
    noisy_heatmap = torch.randn(2, 8, 1)

    try:
        model(
            image_tokens=image_tokens,
            gaze_token=gaze_token,
            noisy_action=torch.randn(3, 4, 10),
            noisy_heatmap=noisy_heatmap,
            timestep=0,
        )
    except ValueError as exc:
        assert "match image batch size" in str(exc)
    else:
        raise AssertionError("Expected noisy_action batch mismatch to fail.")

    try:
        model(
            image_tokens=image_tokens,
            gaze_token=gaze_token,
            noisy_action=noisy_action,
            noisy_heatmap=torch.randn(3, 8, 1),
            timestep=0,
        )
    except ValueError as exc:
        assert "match image batch size" in str(exc)
    else:
        raise AssertionError("Expected noisy_heatmap batch mismatch to fail.")

    try:
        model(
            image_tokens=image_tokens,
            gaze_token=gaze_token,
            noisy_action=noisy_action,
            noisy_heatmap=noisy_heatmap,
            timestep=0,
            is_inference=True,
        )
    except ValueError as exc:
        assert "Inference must omit noisy_heatmap" in str(exc)
    else:
        raise AssertionError("Expected inference with noisy_heatmap to fail.")

    try:
        model(
            image_tokens=torch.randn(2, 0, 16),
            gaze_token=gaze_token,
            noisy_action=noisy_action,
            noisy_heatmap=noisy_heatmap,
            timestep=0,
        )
    except ValueError as exc:
        assert "num_image_tokens" in str(exc)
        assert "positive integer" in str(exc)
    else:
        raise AssertionError("Expected zero image-token forward input to fail.")

    try:
        model(
            image_tokens=torch.randn(2, 17, 16),
            gaze_token=gaze_token,
            noisy_action=noisy_action,
            noisy_heatmap=noisy_heatmap,
            timestep=0,
        )
    except ValueError as exc:
        assert "exceeds max_image_tokens" in str(exc)
    else:
        raise AssertionError("Expected overlong image-token forward input to fail.")

    invalid_timestep_cases = [
        (torch.zeros(3, dtype=torch.long), "B=2"),
        (torch.zeros(1, 1, dtype=torch.long), "scalar or 1D"),
        (torch.tensor([0.5]), "integer diffusion timesteps"),
        (torch.tensor([float("nan")]), "finite diffusion timesteps"),
    ]
    for bad_timestep, expected in invalid_timestep_cases:
        try:
            model(
                image_tokens=image_tokens,
                gaze_token=gaze_token,
                noisy_action=noisy_action,
                noisy_heatmap=noisy_heatmap,
                timestep=bad_timestep,
            )
        except ValueError as exc:
            assert "timestep" in str(exc)
            assert expected in str(exc)
        else:
            raise AssertionError("Expected invalid transformer timestep to fail.")


def test_joint_transformer_block_attention_mask_direction():
    model = JointGazeWamTransformer(
        action_horizon=2,
        heatmap_num_tokens=3,
        max_image_tokens=4,
        n_layer=1,
        n_head=2,
        n_emb=16,
    )
    mask = model.build_block_attention_mask(num_image_tokens=4, include_heatmap=True)
    action_idx = torch.arange(5, 7)
    heatmap_idx = torch.arange(7, 10)
    condition_idx = torch.arange(0, 5)

    assert mask.shape == (10, 10)
    assert torch.isneginf(mask[heatmap_idx[:, None], action_idx]).all()
    assert torch.isneginf(mask[action_idx[:, None], heatmap_idx]).all()
    assert torch.isneginf(mask[condition_idx[:, None], action_idx]).all()
    assert torch.isneginf(mask[condition_idx[:, None], heatmap_idx]).all()
    assert torch.isfinite(mask[action_idx[:, None], condition_idx]).all()
    assert torch.isfinite(mask[heatmap_idx[:, None], condition_idx]).all()

    inference_mask = model.build_block_attention_mask(num_image_tokens=4, include_heatmap=False)
    assert inference_mask.shape == (7, 7)
    assert torch.isneginf(inference_mask[condition_idx[:, None], action_idx]).all()
    assert torch.isfinite(inference_mask[action_idx[:, None], condition_idx]).all()
    assert torch.isfinite(inference_mask[action_idx[:, None], action_idx]).all()
    string_false_mask = model.build_block_attention_mask(
        num_image_tokens=4,
        include_heatmap="false",
    )
    assert string_false_mask.shape == inference_mask.shape

    invalid_mask_values = [
        (False, "positive integer"),
        (0, "positive integer"),
        (4.5, "positive integer"),
        (5, "exceeds max_image_tokens"),
    ]
    for bad_value, expected in invalid_mask_values:
        try:
            model.build_block_attention_mask(num_image_tokens=bad_value)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(
                f"Expected invalid attention-mask num_image_tokens={bad_value!r} to fail."
            )

    try:
        model.build_block_attention_mask(num_image_tokens=4, include_heatmap="maybe")
    except ValueError as exc:
        assert "include_heatmap" in str(exc)
        assert "boolean" in str(exc)
    else:
        raise AssertionError("Expected invalid include_heatmap value to fail.")

    summary = model.attention_contract_summary(num_image_tokens=4)
    assert summary["train_sequence_tokens"] == 10
    assert summary["inference_sequence_tokens"] == 7
    assert summary["use_block_attention_mask"] is True
    assert summary["condition_reads_targets"] is False
    assert summary["action_reads_heatmap"] is False
    assert summary["heatmap_reads_action"] is False
    assert summary["action_inference_drops_heatmap"] is True

    assert model.attention_contract_summary(num_image_tokens="4")["num_image_tokens"] == 4
    invalid_summary_values = [
        (True, "positive integer"),
        (0, "positive integer"),
        (4.5, "positive integer"),
        (5, "exceeds max_image_tokens"),
    ]
    for bad_value, expected in invalid_summary_values:
        try:
            model.attention_contract_summary(num_image_tokens=bad_value)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(
                f"Expected invalid attention num_image_tokens={bad_value!r} to fail."
            )


def test_joint_transformer_no_block_mask_forward_skips_mask_builder():
    model = JointGazeWamTransformer(
        action_dim=10,
        heatmap_dim=1,
        action_horizon=2,
        heatmap_num_tokens=4,
        max_image_tokens=8,
        n_layer=1,
        n_head=2,
        n_emb=16,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
        use_block_attention_mask=False,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("build_block_attention_mask should not be called.")

    model.build_block_attention_mask = fail_if_called
    out = model(
        image_tokens=torch.randn(1, 8, 16),
        gaze_token=torch.randn(1, 1, 16),
        noisy_action=torch.randn(1, 2, 10),
        noisy_heatmap=torch.randn(1, 4, 1),
        timestep=0,
    )

    assert out.action.shape == (1, 2, 10)
    assert out.heatmap.shape == (1, 4, 1)


def test_joint_transformer_normalizes_string_bool_flags():
    model = JointGazeWamTransformer(
        action_dim=10,
        heatmap_dim=1,
        action_horizon=2,
        heatmap_num_tokens=4,
        max_image_tokens=8,
        n_layer=1,
        n_head=2,
        n_emb=16,
        use_block_attention_mask="false",
        use_frame_embedding="on",
        image_tokens_per_frame=4,
        max_obs_frames=2,
    )

    assert model.use_block_attention_mask is False
    assert model.use_frame_embedding is True


def test_joint_transformer_action_slice_with_variable_image_tokens():
    torch.manual_seed(1)
    model = JointGazeWamTransformer(
        action_dim=10,
        heatmap_dim=1,
        action_horizon=4,
        heatmap_num_tokens=8,
        max_image_tokens=512,
        n_layer=1,
        n_head=4,
        n_emb=32,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
    )

    for num_image_tokens in (256, 512):
        image_tokens = torch.randn(2, num_image_tokens, 32)
        gaze_token = torch.randn(2, 1, 32)
        noisy_action = torch.randn(2, 4, 10)
        noisy_heatmap = torch.randn(2, 8, 1)
        out = model(
            image_tokens=image_tokens,
            gaze_token=gaze_token,
            noisy_action=noisy_action,
            noisy_heatmap=noisy_heatmap,
            timestep=0,
        )

        assert out.action.shape == (2, 4, 10)
        assert out.heatmap.shape == (2, 8, 1)


def test_joint_transformer_optional_frame_embedding_ids_and_validation():
    model = JointGazeWamTransformer(
        action_dim=10,
        heatmap_dim=1,
        action_horizon=2,
        heatmap_num_tokens=4,
        max_image_tokens=8,
        n_layer=1,
        n_head=2,
        n_emb=16,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
        use_block_attention_mask=False,
        use_frame_embedding=True,
        image_tokens_per_frame=4,
        max_obs_frames=2,
    )

    frame_ids = model.build_image_frame_ids(num_image_tokens=8)
    assert frame_ids.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]

    small_model_kwargs = {
        "action_dim": 10,
        "heatmap_dim": 1,
        "action_horizon": 2,
        "heatmap_num_tokens": 4,
        "max_image_tokens": 8,
        "n_layer": 1,
        "n_head": 2,
        "n_emb": 16,
    }
    string_geometry_model = JointGazeWamTransformer(
        **small_model_kwargs,
        use_frame_embedding=True,
        image_tokens_per_frame="4",
        max_obs_frames="2",
    )
    assert string_geometry_model.image_tokens_per_frame == 4
    assert string_geometry_model.max_obs_frames == 2

    image_tokens = torch.zeros(1, 8, 16)
    gaze_token = torch.zeros(1, 1, 16)
    noisy_action = torch.zeros(1, 2, 10)
    noisy_heatmap = torch.zeros(1, 4, 1)
    out = model(
        image_tokens=image_tokens,
        gaze_token=gaze_token,
        noisy_action=noisy_action,
        noisy_heatmap=noisy_heatmap,
        timestep=0,
    )
    assert out.action.shape == (1, 2, 10)
    assert out.heatmap.shape == (1, 4, 1)
    summary = model.attention_contract_summary(num_image_tokens=4)
    assert summary["use_block_attention_mask"] is False
    assert summary["condition_reads_targets"] is True
    assert summary["action_reads_heatmap"] is True
    assert summary["heatmap_reads_action"] is True
    assert summary["action_inference_drops_heatmap"] is True

    try:
        model.build_image_frame_ids(num_image_tokens=6)
    except ValueError as exc:
        assert "not divisible" in str(exc)
    else:
        raise AssertionError("Expected non-divisible image-token count to fail.")

    try:
        JointGazeWamTransformer(use_frame_embedding=True, image_tokens_per_frame=None)
    except ValueError as exc:
        assert "image_tokens_per_frame" in str(exc)
    else:
        raise AssertionError("Expected missing image_tokens_per_frame to fail.")

    bad_frame_geometry = [
        {"image_tokens_per_frame": True, "max_obs_frames": 2},
        {"image_tokens_per_frame": 4.5, "max_obs_frames": 2},
        {"image_tokens_per_frame": 4, "max_obs_frames": True},
        {"image_tokens_per_frame": 4, "max_obs_frames": 0},
    ]
    for kwargs in bad_frame_geometry:
        try:
            JointGazeWamTransformer(
                **small_model_kwargs,
                use_frame_embedding=True,
                **kwargs,
            )
        except ValueError as exc:
            assert "positive integer" in str(exc)
        else:
            raise AssertionError(f"Expected bad frame geometry to fail: {kwargs!r}")

    try:
        JointGazeWamTransformer(
            **small_model_kwargs,
            use_frame_embedding=False,
            max_obs_frames=True,
        )
    except ValueError as exc:
        assert "max_obs_frames" in str(exc)
    else:
        raise AssertionError("Expected inactive bool max_obs_frames to fail.")


def test_joint_transformer_rejects_invalid_core_integer_dimensions():
    string_core_model = JointGazeWamTransformer(
        action_dim="10",
        heatmap_dim="1",
        action_horizon="2",
        heatmap_num_tokens="4",
        max_image_tokens="8",
        n_layer="1",
        n_head="2",
        n_emb="16",
    )

    assert string_core_model.action_dim == 10
    assert string_core_model.heatmap_dim == 1
    assert string_core_model.action_horizon == 2
    assert string_core_model.heatmap_num_tokens == 4
    assert string_core_model.max_image_tokens == 8
    assert string_core_model.n_emb == 16

    valid_kwargs = {
        "action_dim": 10,
        "heatmap_dim": 1,
        "action_horizon": 2,
        "heatmap_num_tokens": 4,
        "max_image_tokens": 8,
        "n_layer": 1,
        "n_head": 2,
        "n_emb": 16,
    }
    invalid_cases = [
        ({"action_dim": True}, "positive integer"),
        ({"heatmap_dim": 0}, "positive integer"),
        ({"action_horizon": 2.5}, "positive integer"),
        ({"heatmap_num_tokens": "4.0"}, "positive integer"),
        ({"max_image_tokens": None}, "positive integer"),
        ({"n_layer": -1}, "positive integer"),
        ({"n_head": 0}, "positive integer"),
        ({"n_emb": float("inf")}, "positive integer"),
        ({"n_head": 2, "n_emb": 15}, "divisible"),
    ]
    for overrides, expected in invalid_cases:
        kwargs = dict(valid_kwargs)
        kwargs.update(overrides)
        try:
            JointGazeWamTransformer(**kwargs)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"Expected invalid core dimensions to fail: {overrides!r}")


def test_cached_dual_stream_config_rejects_unmasked_policy():
    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train_gaze_wam_debug_workspace",
            overrides=["policy.use_block_attention_mask='false'"],
        )

    with pytest.raises(InstantiationException, match="always enforces condition/target"):
        instantiate(cfg.policy)


def test_gaze_wam_policy_and_obs_encoder_normalize_string_bool_overrides():
    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train_gaze_wam_debug_workspace",
            overrides=[
                "policy.use_block_attention_mask='true'",
                "policy.obs_encoder.pretrained='false'",
                "policy.obs_encoder.use_group_norm='false'",
                "policy.obs_encoder.share_rgb_model='off'",
            ],
        )

    policy = instantiate(cfg.policy)

    assert policy.model.use_block_attention_mask is True
    assert policy.obs_encoder.pretrained is False
    assert policy.obs_encoder.share_rgb_model is False


def test_gaze_wam_debug_config_instantiates_policy_contract():
    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train_gaze_wam_debug_workspace")

    policy = instantiate(cfg.policy)

    assert policy.obs_encoder.model_name == "vit_base_patch16_dinov3"
    assert policy.obs_encoder.feature_aggregation == "patch"
    assert tuple(policy.obs_encoder.output_shape()[-2:]) == (512, 768)
    assert policy.gaze_encoder.spatial_encoder.grid_size == (8, 8)
    assert policy.gaze_encoder.mask_token.shape == (1, 1, 768)
    assert policy.model.n_emb == 768
    assert policy.model.action_dim == 10
    assert policy.model.action_horizon == 48
    assert policy.model.heatmap_num_tokens == 256
    assert policy.model.heatmap_dim == 16
    assert policy.model.max_image_tokens >= 512
    assert policy.model.use_block_attention_mask is True
    assert policy.model.action_head.out_features == 10
    assert policy.model.heatmap_head.out_features == 16
    assert policy.heatmap_codec.token_grid == (16, 16)
    assert policy.heatmap_codec.image_size == (256, 256)
    assert policy.num_inference_steps == 2
    routing = policy.loss_routing_contract_summary()
    assert routing["source"] == "policy"
    assert routing["dynamic_head_freezing"] is False
    assert routing["action_loss_mask"] == "(~is_open) & has_action"
    assert routing["heatmap_loss_mask"] == "has_heatmap & has_gaze_label"
    assert (
        routing["heatmap_supervision"]
        == "full_resolution_dsnt_plus_js_after_frozen_decoder"
    )
    assert routing["open_rows"] == {
        "has_action": False,
        "has_heatmap": True,
        "use_gaze_condition": False,
        "gaze_token": "learned_mask",
        "trains_action": False,
        "trains_heatmap": "xy DSNT plus generated Gaussian JS target",
    }
    assert routing["robot_real_gaze_rows"] == {
        "is_open": False,
        "has_action": True,
        "use_gaze_condition": True,
        "has_heatmap": True,
        "trains_action": True,
        "trains_heatmap": "has_heatmap & has_gaze_label",
    }
    assert routing["robot_masked_gaze_rows"] == {
        "is_open": False,
        "has_action": True,
        "use_gaze_condition": False,
        "gaze_token": "learned_mask",
        "trains_action": True,
        "trains_heatmap": "has_heatmap & has_gaze_label",
    }
    assert set(routing["validation"]) == set(
        gaze_wam_required_loss_routing_validation_flags()
    )
    assert all(routing["validation"].values())


def test_gaze_wam_ablation_workspace_config_switches():
    assert COMPARE_DEFAULT_VARIANTS == (
        "robot_only_baseline=train_gaze_wam_robot_only_workspace",
        "mixed_main=train_gaze_wam_workspace",
    )

    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        main_cfg = compose(config_name="train_gaze_wam_workspace")
        cached_dual_stream_cfg = compose(
            config_name="train_gaze_wam_cached_dual_stream_workspace"
        )
        open_only_cfg = compose(config_name="train_gaze_wam_open_only_workspace")
        open_pretrain_cfg = compose(
            config_name="train_gaze_wam_open_pretrain_workspace"
        )
        robot_finetune_cfg = compose(
            config_name="train_gaze_wam_robot_finetune_workspace"
        )

    assert main_cfg.open_dataloader.batch_size > 0
    assert main_cfg.val_robot_dataloader.shuffle is False
    assert main_cfg.val_open_dataloader.shuffle is False
    assert main_cfg.training.val_every == 1
    assert main_cfg.training.val_mixing_seed == 100003
    assert main_cfg.checkpoint.topk.monitor_key == "val_robot_loss"
    assert main_cfg.task.robot_gaze_dropout_prob == 0.2
    assert main_cfg.task.robot_heatmap_on_gaze_dropout is True
    assert main_cfg.task.val_ratio == 0.0
    assert main_cfg.task.robot_dataset.image_resize_mode == "letterbox"
    assert main_cfg.task.open_dataset.image_resize_mode == "stretch"
    assert main_cfg.policy.use_block_attention_mask is True
    assert main_cfg.policy.heatmap_objective == "dsnt_js"
    assert main_cfg.policy.heatmap_token_kl_loss_weight == 0.0
    assert main_cfg.policy.heatmap_xy_loss_weight == 1.0
    assert main_cfg.policy.heatmap_point_nll_loss_weight == 0.0
    assert main_cfg.policy.heatmap_js_loss_weight == 1.0
    assert main_cfg.policy.heatmap_dsnt_temperature == 0.1
    assert main_cfg.policy.heatmap_distribution_mode == "intensity_softplus"

    assert cached_dual_stream_cfg.policy.model_architecture == "cached_dual_stream"
    assert cached_dual_stream_cfg.policy.heatmap_objective == "dsnt_js"
    assert cached_dual_stream_cfg.policy.heatmap_distribution_mode == "intensity_softplus"

    assert open_only_cfg.robot_dataloader.batch_size == 0
    assert open_only_cfg.open_dataloader.batch_size > 0
    assert open_only_cfg.val_robot_dataloader.batch_size == 0
    assert open_only_cfg.policy.action_loss_weight == 0.0
    assert open_only_cfg.policy.heatmap_loss_weight == 1.0
    assert open_only_cfg.policy.heatmap_objective == "dsnt_js"
    assert open_only_cfg.policy.heatmap_token_kl_loss_weight == 0.0
    assert open_only_cfg.policy.heatmap_xy_loss_weight == 1.0
    assert open_only_cfg.policy.heatmap_point_nll_loss_weight == 0.0
    assert open_only_cfg.policy.heatmap_js_loss_weight == 1.0
    assert open_only_cfg.policy.heatmap_dsnt_temperature == 0.1
    assert open_only_cfg.policy.heatmap_distribution_mode == "intensity_softplus"
    assert open_only_cfg.training.gdr_every == 0
    assert open_pretrain_cfg.task.n_obs_steps == 1
    assert open_pretrain_cfg.task.val_ratio == pytest.approx(0.05)
    assert open_pretrain_cfg.task.open_dataset.n_obs_steps == 1
    assert open_pretrain_cfg.task.open_dataset.val_ratio == pytest.approx(0.05)
    assert open_pretrain_cfg.task.shape_meta.obs.camera0_rgb.horizon == 1
    assert open_pretrain_cfg.policy.obs_encoder.pretrained is False
    assert open_pretrain_cfg.policy.obs_encoder.checkpoint_path.endswith(
        "vit_base_patch16_dinov3.lvd1689m.safetensors"
    )
    open_pretrain_training = validate_gaze_wam_training_config(open_pretrain_cfg)
    assert open_pretrain_training["valid"] is True
    assert open_pretrain_training["open_batch_size"] == 16
    assert open_pretrain_training["batching"]["train_batch_size_per_process"] == 16
    assert open_pretrain_training["gradient_accumulate_every"] == 2
    assert open_pretrain_cfg.training.lr_scheduler == "constant_with_warmup"
    assert open_pretrain_cfg.training.lr_warmup_steps == 2000
    assert open_pretrain_cfg.training.checkpoint_every == 1
    assert open_pretrain_cfg.training.checkpoint_every_steps == 30000
    assert open_pretrain_cfg.training.max_train_steps is None
    assert open_pretrain_cfg.training.num_epochs == 3000
    assert robot_finetune_cfg.training.stage == "robot_finetune"
    assert robot_finetune_cfg.robot_dataloader.batch_size > 0
    assert robot_finetune_cfg.open_dataloader.batch_size == 0
    assert robot_finetune_cfg.training.transfer.load_scope == "obs_encoder"
    assert robot_finetune_cfg.training.transfer.load_path.endswith(
        "gaze_wam_open_pretrain_obs_encoder.pt"
    )


def test_compare_provenance_resolves_ratio_quota_from_data_mixing():
    cfg = load_cfg(
        "train_gaze_wam_workspace",
        overrides=[
            "robot_dataloader.batch_size=6",
            "open_dataloader.batch_size=2",
        ],
    )
    row = _config_provenance(
        cfg=cfg,
        checkpoint=None,
        sources=("robot", "open"),
        batch_size=2,
        max_batches=1,
        cfg_scale=None,
    )

    assert row["training_stage"] == "mixed_train"
    assert row["batch_size_source"] == "ratio"
    assert row["requested_batch_size_source"] == "ratio"
    assert row["total_batch_size_per_process"] == 64
    assert row["requested_total_batch_size_per_process"] == 64
    assert row["requested_robot_ratio"] == pytest.approx(0.75)
    assert row["requested_open_ratio"] == pytest.approx(0.25)
    assert (row["robot_batch_size"], row["open_batch_size"]) == (48, 16)


def test_gaze_wam_prepared_length_and_terminal_checkpoint_boundaries():
    assert _validate_prepared_epoch_driver_length(5, 5) == 5
    with pytest.raises(RuntimeError, match="planned=5, actual=4"):
        _validate_prepared_epoch_driver_length(5, 4)

    assert _gaze_wam_checkpoint_due(3, 2, False) is False
    assert _gaze_wam_checkpoint_due(4, 2, False) is True
    assert _gaze_wam_checkpoint_due(3, 100, True) is True
    assert _gaze_wam_step_checkpoint_due(29999, 30000) is False
    assert _gaze_wam_step_checkpoint_due(30000, 30000) is True
    assert _gaze_wam_step_checkpoint_due(60000, 30000) is True
    assert _gaze_wam_checkpoint_due(4, 1, False, 29999, 30000) is False
    assert _gaze_wam_checkpoint_due(4, 1, False, 30000, 30000) is True
    assert _gaze_wam_checkpoint_due(4, 1, True, 30001, 30000) is True


def test_gaze_wam_policy_training_root_wrappers_prefer_repo_imports():
    wrappers = [
        "adapt_open_video_gaze_metadata.py",
        "canonicalize_robot_gaze_wam_zarr.py",
        "compare_gaze_wam_ablation_metrics.py",
        "convert_hot3d_processed_to_open_zarr.py",
        "convert_open_gaze_manifest.py",
        "eval_gaze_wam_metrics.py",
        "export_video_gaze_manifest.py",
        "gaze_wam_smoke_pipeline.py",
        "generate_gaze_wam_debug_data.py",
        "inspect_gaze_wam_zarr.py",
        "inspect_open_video_gaze_metadata.py",
        "launch_gaze_wam_training.py",
        "plan_gaze_wam_experiments.py",
        "preflight_gaze_wam.py",
        "prepare_open_gaze_wam_zarr.py",
        "prepare_robot_gaze_wam_zarr.py",
        "preview_gaze_wam_dataset.py",
        "review_gaze_wam_data_onboarding.py",
        "review_gaze_wam_training_readiness.py",
        "validate_gaze_wam_zarr.py",
        "verify_gaze_wam_dino_source.py",
    ]
    for name in wrappers:
        text = Path("scripts", name).read_text(encoding="utf-8")
        assert "sys.path.insert(0, str(ROOT_DIR))" in text
        assert "sys.path.append(str(ROOT_DIR))" not in text


def test_gaze_wam_workspace_direct_entry_prefers_repo_imports():
    text = Path("diffusion_policy/workspace/train_gaze_wam_workspace.py").read_text(
        encoding="utf-8"
    )
    assert "sys.path.insert(0, ROOT_DIR)" in text
    assert "sys.path.append(ROOT_DIR)" not in text


def test_gaze_wam_policy_loss_noise_uses_target_dtype():
    text = Path("diffusion_policy/policy/gaze_wam_policy.py").read_text(
        encoding="utf-8"
    )
    action_noise_call = (
        "torch.randn(\n"
        "            nactions.shape,\n"
        "            device=nactions.device,\n"
        "            dtype=nactions.dtype,\n"
        "        )"
    )
    heatmap_noise_call = (
        "torch.randn(\n"
        "            heatmap.shape,\n"
        "            device=heatmap.device,\n"
        "            dtype=heatmap.dtype,\n"
        "        )"
    )

    assert text.count(action_noise_call) == 2
    assert heatmap_noise_call in text


def test_gaze_wam_training_config_validation_reports_loop_errors():
    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train_gaze_wam_debug_workspace",
            overrides=[
                "training.gradient_accumulate_every=0",
                "training.checkpoint_every=0",
                "training.max_train_steps=0",
                "training.tqdm_interval_sec=-0.1",
            ],
        )

    summary = validate_gaze_wam_training_config(cfg)

    assert summary["valid"] is False
    assert summary["gradient_accumulate_every"] == 0
    assert any("training.gradient_accumulate_every" in error for error in summary["errors"])
    assert any("training.checkpoint_every" in error for error in summary["errors"])
    assert any("training.max_train_steps" in error for error in summary["errors"])
    assert any("training.tqdm_interval_sec" in error for error in summary["errors"])


def test_gaze_wam_training_config_validation_reports_dataloader_errors():
    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train_gaze_wam_debug_workspace",
            overrides=[
                "robot_dataloader.num_workers=-1",
                "open_dataloader.num_workers=0",
                "open_dataloader.persistent_workers=true",
                "val_robot_dataloader.pin_memory=maybe",
                "val_open_dataloader.drop_last=maybe",
            ],
        )

    summary = validate_gaze_wam_training_config(cfg)

    assert summary["valid"] is False
    assert summary["dataloaders"]["robot_dataloader"]["num_workers"] == -1
    assert summary["dataloaders"]["open_dataloader"]["persistent_workers"] is True
    assert any("robot_dataloader.num_workers" in error for error in summary["errors"])
    assert any("open_dataloader.persistent_workers=true requires" in error for error in summary["errors"])
    assert any("val_robot_dataloader.pin_memory must be a boolean" in error for error in summary["errors"])
    assert any("val_open_dataloader.drop_last must be a boolean" in error for error in summary["errors"])


def test_gaze_wam_training_config_validation_reports_parse_errors():
    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train_gaze_wam_debug_workspace",
            overrides=[
                "training.gradient_accumulate_every=oops",
                "training.max_train_steps=oops",
                "training.tqdm_interval_sec=oops",
                "open_dataloader.batch_size=oops",
            ],
        )

    summary = validate_gaze_wam_training_config(cfg)

    assert summary["valid"] is False
    assert summary["gradient_accumulate_every"] == 0
    assert summary["max_train_steps"] == "oops"
    assert summary["open_batch_size"] == 0
    assert any("training.gradient_accumulate_every must be an integer" in error for error in summary["errors"])
    assert any("training.max_train_steps must be an integer or null" in error for error in summary["errors"])
    assert any("training.tqdm_interval_sec must be a number" in error for error in summary["errors"])
    assert any("open_dataloader.batch_size must be an integer" in error for error in summary["errors"])


def test_gaze_wam_training_config_validation_rejects_bool_numeric_fields():
    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train_gaze_wam_debug_workspace")

    cfg.robot_dataloader.batch_size = True
    cfg.robot_dataloader.num_workers = False
    cfg.training.gradient_accumulate_every = True
    cfg.training.max_train_steps = False
    cfg.training.tqdm_interval_sec = True

    summary = validate_gaze_wam_training_config(cfg)

    assert summary["valid"] is False
    assert summary["robot_batch_size"] == 0
    assert summary["gradient_accumulate_every"] == 0
    assert summary["max_train_steps"] is False
    assert summary["tqdm_interval_sec"] == 0.0
    assert any("robot_dataloader.batch_size must be an integer" in error for error in summary["errors"])
    assert any("robot_dataloader.num_workers must be an integer" in error for error in summary["errors"])
    assert any("training.gradient_accumulate_every must be an integer" in error for error in summary["errors"])
    assert any("training.max_train_steps must be an integer or null" in error for error in summary["errors"])
    assert any("training.tqdm_interval_sec must be a number" in error for error in summary["errors"])


def test_gaze_wam_training_config_validation_rejects_nonfinite_float_fields():
    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train_gaze_wam_debug_workspace")

    cfg.training.tqdm_interval_sec = float("nan")

    summary = validate_gaze_wam_training_config(cfg)

    assert summary["valid"] is False
    assert summary["tqdm_interval_sec"] == 0.0
    assert any("training.tqdm_interval_sec must be finite" in error for error in summary["errors"])

    cfg.training.tqdm_interval_sec = float("inf")
    summary = validate_gaze_wam_training_config(cfg)

    assert summary["valid"] is False
    assert summary["tqdm_interval_sec"] == 0.0
    assert any("training.tqdm_interval_sec must be finite" in error for error in summary["errors"])


def test_gaze_wam_training_config_validation_rejects_fractional_integer_fields():
    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train_gaze_wam_debug_workspace")

    cfg.robot_dataloader.batch_size = 3.5
    cfg.robot_dataloader.num_workers = 1.5
    cfg.training.gradient_accumulate_every = 2.5
    cfg.training.max_train_steps = "7.0"
    cfg.training.lr_warmup_steps = float("inf")

    summary = validate_gaze_wam_training_config(cfg)

    assert summary["valid"] is False
    assert summary["robot_batch_size"] == 0
    assert summary["gradient_accumulate_every"] == 0
    assert summary["max_train_steps"] == "7.0"
    assert summary["lr_warmup_steps"] == 0
    assert any("robot_dataloader.batch_size must be an integer" in error for error in summary["errors"])
    assert any("robot_dataloader.num_workers must be an integer" in error for error in summary["errors"])
    assert any("training.gradient_accumulate_every must be an integer" in error for error in summary["errors"])
    assert any("training.max_train_steps must be an integer or null" in error for error in summary["errors"])
    assert any("training.lr_warmup_steps must be an integer" in error for error in summary["errors"])

    cfg.robot_dataloader.batch_size = 3.0
    cfg.robot_dataloader.num_workers = 1.0
    cfg.training.gradient_accumulate_every = 2.0
    cfg.training.max_train_steps = 7.0
    cfg.training.lr_warmup_steps = 0.0

    summary = validate_gaze_wam_training_config(cfg)

    assert summary["valid"] is True
    assert summary["robot_batch_size"] == 3
    assert summary["dataloaders"]["robot_dataloader"]["num_workers"] == 1
    assert summary["gradient_accumulate_every"] == 2
    assert summary["max_train_steps"] == 7
    assert summary["lr_warmup_steps"] == 0


def test_gaze_wam_training_config_normalization_applies_parsed_values():
    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train_gaze_wam_debug_workspace")

    cfg.robot_dataloader.batch_size = "6"
    cfg.open_dataloader.batch_size = "2"
    cfg.val_robot_dataloader.batch_size = "4"
    cfg.val_open_dataloader.batch_size = "1"
    cfg.training.gradient_accumulate_every = "2"
    cfg.training.num_epochs = "3"
    cfg.training.checkpoint_every = "1"
    cfg.training.checkpoint_every_steps = "30000"
    cfg.training.val_every = "1"
    cfg.training.sample_every = "0"
    cfg.training.gdr_every = "5"
    cfg.training.max_train_steps = "7"
    cfg.training.max_val_steps = "2"
    cfg.training.lr_warmup_steps = "9"
    cfg.training.tqdm_interval_sec = "0.25"
    cfg.robot_dataloader.num_workers = "6"
    cfg.robot_dataloader.pin_memory = "true"
    cfg.robot_dataloader.persistent_workers = "true"
    cfg.robot_dataloader.drop_last = "false"
    cfg.open_dataloader.num_workers = "0"
    cfg.open_dataloader.pin_memory = "false"
    cfg.open_dataloader.persistent_workers = "false"
    cfg.open_dataloader.drop_last = "true"

    summary = validate_gaze_wam_training_config(cfg)
    assert summary["valid"] is True

    normalized_cfg = _normalize_gaze_wam_training_config(cfg, summary)

    assert normalized_cfg.robot_dataloader.batch_size == 6
    assert normalized_cfg.open_dataloader.batch_size == 2
    assert normalized_cfg.val_robot_dataloader.batch_size == 4
    assert normalized_cfg.val_open_dataloader.batch_size == 1
    assert normalized_cfg.training.gradient_accumulate_every == 2
    assert normalized_cfg.training.num_epochs == 3
    assert normalized_cfg.training.checkpoint_every == 1
    assert normalized_cfg.training.checkpoint_every_steps == 30000
    assert normalized_cfg.training.val_every == 1
    assert normalized_cfg.training.sample_every == 0
    assert normalized_cfg.training.gdr_every == 5
    assert normalized_cfg.training.max_train_steps == 7
    assert normalized_cfg.training.max_val_steps == 2
    assert normalized_cfg.training.lr_warmup_steps == 9
    assert normalized_cfg.training.tqdm_interval_sec == 0.25
    assert normalized_cfg.robot_dataloader.num_workers == 6
    assert normalized_cfg.robot_dataloader.pin_memory is True
    assert normalized_cfg.robot_dataloader.persistent_workers is True
    assert normalized_cfg.robot_dataloader.drop_last is False
    assert normalized_cfg.open_dataloader.num_workers == 0
    assert normalized_cfg.open_dataloader.pin_memory is False
    assert normalized_cfg.open_dataloader.persistent_workers is False
    assert normalized_cfg.open_dataloader.drop_last is True


def test_gaze_wam_training_config_allows_open_only_batch_ratio():
    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train_gaze_wam_open_only_debug_workspace")

    summary = validate_gaze_wam_training_config(cfg)

    assert summary["valid"] is True
    assert summary["robot_batch_size"] == 0
    assert summary["open_batch_size"] == 4
    assert summary["val_robot_batch_size"] == 0
    assert summary["val_open_batch_size"] == 4
    assert summary["train_batch_size_per_process"] == 4

    cfg.val_robot_dataloader.batch_size = 1
    bad_summary = validate_gaze_wam_training_config(cfg)

    assert bad_summary["valid"] is False
    assert any(
        "val_robot_dataloader.batch_size must be 0" in error
        for error in bad_summary["errors"]
    )


def test_gaze_wam_workspace_open_batch_gates_use_normalized_config_values():
    text = Path("diffusion_policy/workspace/train_gaze_wam_workspace.py").read_text(
        encoding="utf-8"
    )

    assert "open_batch_size = cfg.open_dataloader.batch_size" in text
    assert "open_val_batch_size = cfg.val_open_dataloader.batch_size" in text
    assert 'open_batch_size = int(cfg.open_dataloader.get("batch_size", 0))' not in text
    assert (
        'open_val_batch_size = int(cfg.val_open_dataloader.get("batch_size", open_batch_size))'
        not in text
    )


def test_gaze_wam_workspace_val_heatmap_preview_gate_uses_normalized_bool():
    text = Path("diffusion_policy/workspace/train_gaze_wam_workspace.py").read_text(
        encoding="utf-8"
    )

    assert "cfg.training.save_val_heatmap_preview" in text
    assert 'bool(cfg.training.get("save_val_heatmap_preview", False))' not in text


def test_gaze_wam_launcher_real_data_readiness_uses_shared_routing_parse():
    text = Path("diffusion_policy/scripts/launch_gaze_wam_training.py").read_text(
        encoding="utf-8"
    )

    assert "task_routing_config = validate_gaze_wam_task_routing_config(cfg)" in text
    assert 'task_routing_config.get("robot_gaze_dropout_prob", 0.0)' in text
    assert 'task_routing_config.get(\n        "robot_heatmap_on_gaze_dropout",' in text
    assert "robot_gaze_dropout_prob = float(" not in text
    assert "robot_heatmap_on_gaze_dropout = bool(" not in text


def test_gaze_wam_launcher_real_data_readiness_uses_shared_geometry_parse():
    text = Path("diffusion_policy/scripts/launch_gaze_wam_training.py").read_text(
        encoding="utf-8"
    )
    fn_text = text.split("def check_real_data_readiness", 1)[1].split(
        "def launch_gaze_wam_training",
        1,
    )[0]

    assert "normalize_gaze_wam_positive_int_field" in fn_text
    assert "normalize_gaze_wam_nonnegative_int_field" in fn_text
    assert "normalize_gaze_wam_positive_int_sequence" in fn_text
    assert "core_config_parse_valid" in fn_text
    assert 'task_sampling = {\n        "n_obs_steps": parse_positive_int(' in fn_text
    assert 'robot_sampling = {\n        "n_obs_steps": parse_positive_int(' in fn_text
    assert 'open_sampling = {\n        "n_obs_steps": parse_positive_int(' in fn_text
    assert "obs_encoder_downsample_ratio = parse_positive_int(" in fn_text
    assert "heatmap_num_tokens=heatmap_num_tokens" in fn_text
    assert "int(cfg.task.n_obs_steps)" not in fn_text
    assert "int(cfg.task.action_horizon)" not in fn_text
    assert 'int(cfg.task.get("n_latency_steps", 0))' not in fn_text
    assert "int(cfg.task.action_dim)" not in fn_text
    assert "int(cfg.task.heatmap_num_tokens)" not in fn_text
    assert 'int(cfg.policy.obs_encoder.get("downsample_ratio", 16))' not in fn_text


def test_gaze_wam_early_bool_config_normalizes_string_overrides():
    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train_gaze_wam_debug_workspace")

    cfg.training.use_ema = "false"
    cfg.training.resume = "off"
    cfg.training.debug = "false"
    cfg.training.freeze_encoder = "on"
    cfg.training.save_val_heatmap_preview = "false"
    cfg.checkpoint.save_last_ckpt = "false"
    cfg.checkpoint.save_last_snapshot = "off"
    cfg.policy.use_block_attention_mask = "false"
    cfg.policy.use_frame_embedding = "off"
    cfg.policy.obs_encoder.pretrained = "false"
    cfg.policy.obs_encoder.frozen = "off"
    cfg.policy.obs_encoder.use_group_norm = "false"
    cfg.policy.obs_encoder.share_rgb_model = "off"

    normalized_cfg = _normalize_gaze_wam_early_bool_config(cfg)

    assert normalized_cfg.training.use_ema is False
    assert normalized_cfg.training.resume is False
    assert normalized_cfg.training.debug is False
    assert normalized_cfg.training.freeze_encoder is True
    assert normalized_cfg.training.save_val_heatmap_preview is False
    assert normalized_cfg.checkpoint.save_last_ckpt is False
    assert normalized_cfg.checkpoint.save_last_snapshot is False
    assert normalized_cfg.policy.use_block_attention_mask is False
    assert normalized_cfg.policy.use_frame_embedding is False
    assert normalized_cfg.policy.obs_encoder.pretrained is False
    assert normalized_cfg.policy.obs_encoder.frozen is False
    assert normalized_cfg.policy.obs_encoder.use_group_norm is False
    assert normalized_cfg.policy.obs_encoder.share_rgb_model is False


def test_gaze_wam_eval_load_cfg_normalizes_early_string_bool_overrides():
    cfg = load_cfg(
        "train_gaze_wam_debug_workspace",
        overrides=[
            "training.use_ema='false'",
            "training.resume='off'",
            "policy.use_block_attention_mask='false'",
            "policy.obs_encoder.pretrained='false'",
            "checkpoint.save_last_ckpt='false'",
        ],
    )

    assert cfg.training.use_ema is False
    assert cfg.training.resume is False
    assert cfg.policy.use_block_attention_mask is False
    assert cfg.policy.obs_encoder.pretrained is False
    assert cfg.checkpoint.save_last_ckpt is False


def test_gaze_wam_task_routing_config_normalizes_string_overrides():
    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train_gaze_wam_debug_workspace")

    cfg.task.robot_gaze_dropout_prob = "0.35"
    cfg.task.robot_heatmap_on_gaze_dropout = "false"

    summary = validate_gaze_wam_task_routing_config(cfg)
    assert summary == {
        "robot_gaze_dropout_prob": 0.35,
        "robot_heatmap_on_gaze_dropout": False,
        "errors": [],
        "valid": True,
    }

    normalized_cfg = _normalize_gaze_wam_task_routing_config(cfg, summary)
    assert normalized_cfg.task.robot_gaze_dropout_prob == 0.35
    assert normalized_cfg.task.robot_heatmap_on_gaze_dropout is False


def test_gaze_wam_task_routing_config_rejects_invalid_values():
    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train_gaze_wam_debug_workspace")

    cfg.task.robot_gaze_dropout_prob = "1.5"
    cfg.task.robot_heatmap_on_gaze_dropout = "maybe"

    summary = validate_gaze_wam_task_routing_config(cfg)

    assert summary["valid"] is False
    assert summary["robot_gaze_dropout_prob"] == 1.5
    assert summary["robot_heatmap_on_gaze_dropout"] is True
    assert any("task.robot_gaze_dropout_prob must be in [0, 1]" in error for error in summary["errors"])
    assert any("task.robot_heatmap_on_gaze_dropout must be a boolean" in error for error in summary["errors"])


def test_gaze_wam_task_routing_config_uses_shared_normalizers():
    text = Path("diffusion_policy/common/gaze_wam_training_config.py").read_text(
        encoding="utf-8"
    )
    fn_text = text.split("def validate_gaze_wam_task_routing_config", 1)[1].split(
        "def _normalize_gaze_wam_task_routing_config",
        1,
    )[0]

    assert "normalize_gaze_wam_unit_interval_float_field" in fn_text
    assert "include_one=True" in fn_text
    assert "normalize_gaze_wam_bool_field" in fn_text
    assert "_parse_bool_field(" not in fn_text
    assert "robot_gaze_dropout_prob < 0.0" not in fn_text
    assert "robot_gaze_dropout_prob > 1.0" not in fn_text


def test_gaze_wam_validation_mixing_generator_is_deterministic():
    robot_batch = {
        "obs": {"camera0_rgb": torch.zeros(6, 2, 3, 16, 16)},
        "action": torch.zeros(6, 3, 10),
        "heatmap": torch.zeros(6, 1, 16, 1),
        "gaze_xy": torch.zeros(6, 2),
        "has_gaze_label": torch.ones(6, dtype=torch.bool),
    }
    open_batch = {
        "obs": {"camera0_rgb": torch.zeros(2, 2, 3, 16, 16)},
        "action": torch.zeros(2, 3, 10),
        "heatmap": torch.zeros(2, 1, 16, 1),
        "gaze_xy": torch.zeros(2, 2),
        "has_gaze_label": torch.ones(2, dtype=torch.bool),
    }

    batch_a = build_gaze_wam_mixed_batch(
        robot_batch=robot_batch,
        open_batch=open_batch,
        robot_gaze_dropout_prob=0.5,
        generator=_make_cpu_generator(1234),
        shuffle=False,
    )
    batch_b = build_gaze_wam_mixed_batch(
        robot_batch=robot_batch,
        open_batch=open_batch,
        robot_gaze_dropout_prob=0.5,
        generator=_make_cpu_generator(1234),
        shuffle=False,
    )
    batch_c = build_gaze_wam_mixed_batch(
        robot_batch=robot_batch,
        open_batch=open_batch,
        robot_gaze_dropout_prob=0.5,
        generator=_make_cpu_generator(1235),
        shuffle=False,
    )

    assert torch.equal(batch_a["use_gaze_condition"], batch_b["use_gaze_condition"])
    assert torch.equal(batch_a["has_heatmap"], batch_b["has_heatmap"])
    assert batch_a["use_gaze_condition"].shape == batch_c["use_gaze_condition"].shape
    assert batch_a["is_open"][-2:].all()
    assert batch_a["has_heatmap"][-2:].all()


def test_restartable_open_dataloader_iterator_rebuilds_on_exhaustion():
    class EpochIterable:
        def __init__(self):
            self.iter_calls = 0

        def __iter__(self):
            self.iter_calls += 1
            return iter([f"{self.iter_calls}:0", f"{self.iter_calls}:1"])

    source = EpochIterable()
    iterator = _RestartingDataLoaderIterator(source, "open train")

    assert iterator.next() == "1:0"
    assert iterator.next() == "1:1"
    assert iterator.restart_count == 0
    assert iterator.next() == "2:0"
    assert iterator.restart_count == 1
    assert source.iter_calls == 2


def test_restartable_open_dataloader_iterator_rejects_empty_stream():
    iterator = _RestartingDataLoaderIterator([], "open train")

    try:
        iterator.next()
    except ValueError as exc:
        assert "open train dataloader produced no batches" in str(exc)
    else:
        raise AssertionError("Expected empty open dataloader to raise ValueError.")


def test_gaze_wam_debug_data_records_geometry_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = generate_gaze_wam_debug_data(
            output_dir=str(Path(tmpdir) / "debug_data"),
            num_episodes=1,
            episode_length=4,
            image_size=16,
            image_resize_mode="stretch",
            seed=123,
        )

        robot_root = zarr.open(result["robot_path"], mode="a")
        open_root = zarr.open(result["open_path"], mode="r")

        assert robot_root["meta"].attrs["dataset_type"] == "robot"
        assert open_root["meta"].attrs["dataset_type"] == "open"
        assert robot_root["meta"].attrs["image_resize_mode"] == "stretch"
        assert open_root["meta"].attrs["image_resize_mode"] == "stretch"
        assert robot_root["meta"].attrs["image_size"] == [16, 16]
        assert open_root["meta"].attrs["gaze_is_normalized"] is True

        robot_summary = validate_gaze_wam_zarr(
            dataset_path=result["robot_path"],
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_dataset_sample=False,
        )
        open_summary = validate_gaze_wam_zarr(
            dataset_path=result["open_path"],
            dataset_type="open",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_dataset_sample=False,
        )

        assert robot_summary["valid"] is True
        assert open_summary["valid"] is True
        assert robot_summary["metadata_attrs"]["dataset_type"] == "robot"
        assert open_summary["metadata_attrs"]["image_resize_mode"] == "stretch"

        robot_root["meta"].attrs["dataset_type"] = "open"
        bad_type_summary = validate_gaze_wam_zarr(
            dataset_path=result["robot_path"],
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_dataset_sample=False,
        )
        assert bad_type_summary["valid"] is False
        assert any("metadata dataset_type" in error for error in bad_type_summary["errors"])
        robot_root["meta"].attrs["dataset_type"] = "robot"

        robot_root["meta"].attrs["image_size"] = [32, 32]
        bad_size_summary = validate_gaze_wam_zarr(
            dataset_path=result["robot_path"],
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_dataset_sample=False,
        )
        assert bad_size_summary["valid"] is False
        assert any("metadata image_size" in error for error in bad_size_summary["errors"])
        robot_root["meta"].attrs["image_size"] = [16, 16]

        robot_root["meta"].attrs["image_resize_mode"] = "letterbox"
        bad_summary = validate_gaze_wam_zarr(
            dataset_path=result["robot_path"],
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_dataset_sample=False,
        )
        assert bad_summary["valid"] is False
        assert any("metadata image_resize_mode" in error for error in bad_summary["errors"])


def test_gaze_wam_debug_workspace_logs_validation_metrics():
    config_dir = str(Path("diffusion_policy/config").resolve())
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data_dir = root / "debug_data"
        output_dir = root / "outputs"
        generate_gaze_wam_debug_data(
            output_dir=str(data_dir),
            num_episodes=2,
            episode_length=18,
            image_size=256,
            seed=321,
        )
        overrides = [
            f"task.robot_dataset_path={str(data_dir / 'robot.zarr')}",
            f"task.open_dataset_path={str(data_dir / 'open.zarr')}",
            "task.val_ratio=0.5",
            "task.robot_gaze_dropout_prob=1.0",
            "training.gradient_accumulate_every=2",
            "checkpoint.topk.k=0",
            "checkpoint.save_last_ckpt=false",
            "checkpoint.save_last_snapshot=false",
            f"hydra.run.dir={str(output_dir)}",
        ]
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="train_gaze_wam_debug_workspace", overrides=overrides)

        workspace = TrainGazeWamWorkspace(cfg, output_dir=str(output_dir))
        workspace.run()

        log_path = output_dir / "logs.json.txt"
        rows = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        contract_path = output_dir / "training_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        assert rows
        first_row = rows[0]
        final_row = rows[-1]
        assert contract["canonical_main_config_ok"] is False
        assert contract["checks"]["robot_train_samples_positive"] is True
        assert contract["checks"]["open_train_samples_positive_when_enabled"] is True
        assert contract["data"]["robot_train_samples"] > 0
        assert contract["data"]["open_train_samples"] > 0
        assert contract["data"]["requires_robot_train_samples"] is True
        assert contract["data"]["requires_open_train_samples"] is True
        assert contract["data"]["allows_empty_validation_sets"] is True
        assert contract["data"]["action_target_start_offset_steps"] == 1
        assert contract["data"]["action_chunk_semantics"] == (
            "state@(t+1...t+H) relative to the latest observed state@t"
        )
        assert contract["checks"]["image_resize_modes_supported"] is True
        assert contract["checks"]["robot_image_size_matches_task"] is True
        assert contract["checks"]["open_image_size_matches_task"] is True
        assert contract["checks"]["robot_sampling_matches_task"] is True
        assert contract["checks"]["open_sampling_matches_task"] is True
        assert contract["checks"]["robot_metadata_dataset_type_when_enabled"] is True
        assert contract["checks"]["robot_metadata_image_resize_mode_when_enabled"] is True
        assert contract["checks"]["robot_metadata_image_size_when_enabled"] is True
        assert contract["checks"]["open_metadata_dataset_type_when_enabled"] is True
        assert contract["checks"]["open_metadata_image_resize_mode_when_enabled"] is True
        assert contract["checks"]["open_metadata_image_size_when_enabled"] is True
        assert contract["data_sources"]["image_resize_modes"] == {
            "task": "stretch",
            "robot_dataset": "stretch",
            "open_dataset": "stretch",
        }
        assert contract["data_sources"]["image_sizes"] == {
            "task": [256, 256],
            "robot_dataset": [256, 256],
            "open_dataset": [256, 256],
        }
        assert contract["data_sources"]["robot"]["expected_image_size"] == [256, 256]
        assert contract["data_sources"]["open"]["expected_image_size"] == [256, 256]
        assert contract["data_sources"]["robot"]["metadata_attrs"]["dataset_type"] == "robot"
        assert contract["data_sources"]["robot"]["metadata_attrs"]["image_resize_mode"] == "stretch"
        assert contract["data_sources"]["robot"]["metadata_attrs"]["image_size"] == [256, 256]
        assert contract["data_sources"]["robot"]["dataset_type_matches_expected"] is True
        assert contract["data_sources"]["robot"]["image_resize_mode_matches_expected"] is True
        assert contract["data_sources"]["robot"]["image_size_matches_expected"] is True
        assert contract["data_sources"]["open"]["metadata_attrs"]["dataset_type"] == "open"
        assert contract["data_sources"]["open"]["metadata_attrs"]["image_resize_mode"] == "stretch"
        assert contract["data_sources"]["open"]["metadata_attrs"]["image_size"] == [256, 256]
        assert contract["data_sources"]["open"]["dataset_type_matches_expected"] is True
        assert contract["data_sources"]["open"]["image_resize_mode_matches_expected"] is True
        assert contract["data_sources"]["open"]["image_size_matches_expected"] is True
        assert contract["checks"]["data_stream_separate_zarr_sources"] is True
        assert contract["checks"]["data_stream_robot_dataset_class"] is True
        assert contract["checks"]["data_stream_open_dataset_class"] is True
        assert contract["checks"]["data_stream_online_mixed_batch_builder"] is True
        assert contract["data_stream"]["source"] == "two_zarr_two_dataset_online_mixed_batch"
        assert contract["data_stream"]["separate_zarr_sources"] is True
        assert contract["data_stream"]["offline_merged_zarr"] is False
        assert contract["data_stream"]["robot"]["dataset_path"] == str(data_dir / "robot.zarr")
        assert contract["data_stream"]["open"]["dataset_path"] == str(data_dir / "open.zarr")
        assert contract["data_stream"]["robot"]["dataset_class_matches_expected"] is True
        assert contract["data_stream"]["open"]["dataset_class_matches_expected"] is True
        assert contract["data_stream"]["robot"]["dataloader"] == "robot_dataloader"
        assert contract["data_stream"]["open"]["dataloader"] == "open_dataloader"
        assert contract["data_stream"]["robot"]["batch_size_per_process"] == 3
        assert contract["data_stream"]["open"]["batch_size_per_process"] == 1
        assert (
            contract["data_stream"]["mixing"]["builder"]
            == "diffusion_policy.dataset.gaze_wam_mixing.build_gaze_wam_mixed_batch"
        )
        assert contract["data_stream"]["mixing"]["mode"] == "online_per_step_concat_after_fetch"
        assert (
            contract["data_stream"]["mixing"]["ratio_source"]
            == (
                "data_mixing.total_batch_size_per_process+"
                "data_mixing.robot_ratio+data_mixing.open_ratio"
            )
        )
        assert contract["data_stream"]["mixing"]["robot_ratio_per_process"] == 0.75
        assert contract["data_stream"]["mixing"]["open_ratio_per_process"] == 0.25
        assert contract["optional_metadata"]["optional_keys"] == [
            "action_abs",
            "action_base_abs",
            "heatmap_image",
        ]
        assert contract["optional_metadata"]["presence_mask_keys"] == [
            "has_action_abs",
            "has_action_base_abs",
            "has_heatmap_image",
        ]
        assert "sampled target timestep" in contract["optional_metadata"][
            "presence_mask_semantics"
        ]["has_action_abs"]
        assert contract["optional_metadata"]["samples"]["robot_train"]["available"] is True
        assert contract["optional_metadata"]["samples"]["robot_train"]["optional_shapes"][
            "action_abs"
        ] == [48, 10]
        assert contract["optional_metadata"]["samples"]["robot_train"]["optional_shapes"][
            "action_base_abs"
        ] == [10]
        assert contract["optional_metadata"]["samples"]["open_train"]["available"] is True
        assert contract["checks"]["obs_encoder_pretrained"] is False
        assert contract["checks"]["obs_encoder_local_weight_source_optional"] is True
        assert contract["checks"]["obs_encoder_local_weight_source_exists"] is True
        assert contract["checks"]["obs_encoder_local_weight_source_valid"] is True
        assert contract["checks"]["obs_encoder_checkpoint_path_file_when_configured"] is True
        assert contract["checks"]["obs_encoder_cache_dir_directory_when_configured"] is True
        assert contract["checks"]["robot_gaze_dropout_prob_0p2"] is False
        assert contract["batching"]["robot_batch_size_per_process"] == 3
        assert contract["batching"]["open_batch_size_per_process"] == 1
        assert contract["batching"]["train_batch_size_per_process"] == 4
        assert contract["batching"]["num_processes"] == 1
        assert contract["batching"]["gradient_accumulate_every"] == 2
        assert contract["batching"]["effective_robot_batch_size_per_optimizer_step"] == 6
        assert contract["batching"]["effective_open_batch_size_per_optimizer_step"] == 2
        assert contract["batching"]["effective_train_batch_size_per_optimizer_step"] == 8
        assert contract["batching"]["mixed_precision"] == "no"
        assert contract["batching"]["distributed_type"] == "NO"
        assert contract["batching"]["requested_batch_size_source"] == "auto"
        assert contract["batching"]["resolved_batch_size_source"] == "ratio"
        assert contract["batching"]["compatibility_fallback_to_dataloader"] is False
        assert contract["batching"]["ratio_fields_present"] is True
        assert contract["batching"]["requested_total_batch_size_per_process"] == 4
        assert contract["batching"]["requested_robot_ratio"] == 0.75
        assert contract["batching"]["requested_open_ratio"] == 0.25
        assert contract["batching"]["configured_robot_dataloader_batch_size"] == 3
        assert contract["batching"]["configured_open_dataloader_batch_size"] == 1
        expected_batch_streaming = {
            "robot_train_enabled": True,
            "open_train_enabled": True,
            "open_val_configured": True,
            "open_val_enabled": True,
            "open_iterator_policy": "restart_on_exhaustion",
            "open_iterator_caches_epoch_batches": False,
            "open_iterator_preserves_dataloader_shuffle_on_restart": True,
            "robot_iterator_policy": "single_pass_epoch_driver",
            "primary_epoch_driver": "robot_dataloader",
        }
        for key, expected in expected_batch_streaming.items():
            assert contract["batch_streaming"][key] == expected
        assert contract["dataloader_batches"]["robot_train_batches_per_epoch"] > 0
        assert contract["dataloader_batches"]["open_train_batches_per_epoch"] > 0
        assert contract["dataloader_batches"]["robot_val_batches_per_epoch"] >= 0
        assert contract["dataloader_batches"]["open_val_batches_per_epoch"] > 0
        assert contract["sampling"] == {
            "task": {
                "n_obs_steps": 2,
                "action_horizon": 48,
                "n_latency_steps": 0,
            },
            "robot_dataset": {
                "n_obs_steps": 2,
                "obs_downsample_steps": 1,
                "action_horizon": 48,
                "n_latency_steps": 0,
                "action_downsample_steps": 1,
                "action_padding": True,
            },
            "open_dataset": {
                "n_obs_steps": 2,
                "obs_downsample_steps": 1,
                "action_horizon": 48,
                "n_latency_steps": 0,
                "action_downsample_steps": 1,
                "action_padding": True,
            },
            "compare_keys": ["n_obs_steps", "action_horizon", "n_latency_steps"],
            "robot_matches_task": True,
            "open_matches_task": True,
        }
        assert contract["training_config"]["valid"] is True
        assert contract["training_config"]["gradient_accumulate_every"] == 2
        assert contract["training_config"]["train_batch_size_per_process"] == 4
        assert contract["batching"]["robot_ratio"] == 0.75
        assert contract["batching"]["open_ratio"] == 0.25
        assert contract["routing"]["robot_gaze_dropout_prob"] == 1.0
        assert contract["routing"]["robot_heatmap_on_gaze_dropout"] is True
        assert contract["routing"]["source"] == "policy"
        assert contract["routing"]["dynamic_head_freezing"] is False
        assert contract["routing"]["action_loss_mask"] == "(~is_open) & has_action"
        assert contract["routing"]["heatmap_loss_mask"] == "has_heatmap & has_gaze_label"
        assert contract["routing"]["open_rows"]["trains_action"] is False
        assert (
            contract["routing"]["open_rows"]["trains_heatmap"]
            == "xy DSNT plus generated Gaussian JS target"
        )
        assert contract["routing"]["open_rows"]["use_gaze_condition"] is False
        assert contract["routing"]["robot_real_gaze_rows"]["trains_action"] is True
        assert contract["routing"]["robot_real_gaze_rows"]["trains_heatmap"] == "has_heatmap & has_gaze_label"
        assert contract["routing"]["robot_masked_gaze_rows"]["trains_action"] is True
        assert contract["routing"]["validation"]["open_rows_must_use_mask_token"] is True
        assert contract["routing"]["validation"][
            "inactive_action_rows_must_be_zero_placeholders"
        ] is True
        assert contract["routing"]["validation"][
            "inactive_heatmap_rows_must_be_zero_placeholders"
        ] is True
        assert contract["routing"]["validation"][
            "inactive_gaze_rows_must_be_zero_placeholders"
        ] is True
        assert contract["routing"]["validation"][
            "inactive_optional_metadata_rows_must_be_zero_placeholders"
        ] is True
        assert contract["checks"]["routing_validation_guardrails"] is True
        assert contract["checks"]["heatmap_objective_dsnt_js"] is True
        assert contract["checks"]["heatmap_token_kl_loss_weight_0"] is True
        assert contract["checks"]["heatmap_point_nll_loss_weight_0"] is True
        assert contract["checks"]["heatmap_xy_loss_weight_1"] is True
        assert contract["checks"]["heatmap_js_loss_weight_1"] is True
        assert contract["checks"]["heatmap_distribution_mode_intensity_softplus"] is True
        assert contract["checks"]["heatmap_dsnt_temperature_0p1"] is True
        assert contract["loss"]["action_loss_weight"] == 1.0
        assert contract["loss"]["heatmap_loss_weight"] == 1.0
        assert contract["loss"]["heatmap_token_kl_loss_weight"] == 0.0
        assert contract["loss"]["heatmap_xy_loss_weight"] == 1.0
        assert contract["loss"]["heatmap_point_nll_loss_weight"] == 0.0
        assert contract["loss"]["heatmap_js_loss_weight"] == 1.0
        assert contract["loss"]["heatmap_dsnt_temperature"] == 0.1
        assert contract["loss"]["heatmap_distribution_mode"] == "intensity_softplus"
        assert (
            contract["loss"]["heatmap_decoder_output_interpretation"]
            == "decoded_intensity_distribution"
        )
        assert contract["loss"]["heatmap_objective"] == "dsnt_js"
        assert (
            contract["checks"]["normalizer_source_robot_relative_when_robot_enabled"]
            is True
        )
        assert contract["checks"]["normalizer_source_matches_active_sources"] is True
        assert contract["checks"]["normalizer_keys_match_robot_camera_and_action"] is True
        assert contract["checks"]["normalizer_action_dim_10"] is True
        assert contract["checks"]["normalizer_excludes_open_dummy_actions"] is True
        assert contract["normalizer"]["source"] == "robot_dataset_relative_actions_only"
        assert (
            contract["normalizer"]["action_normalizer_source"]
            == "GazeWamRobotDataset.get_all_actions"
        )
        assert contract["normalizer"]["normalizer_keys"] == ["camera0_rgb", "action"]
        assert contract["normalizer"]["action_dim"] == 10
        assert contract["normalizer"]["excludes_open_source_dummy_actions"] is True
        assert contract["normalizer"]["open_source_get_normalizer_allowed"] is False
        assert contract["tokens"]["visual_token_count"] == 512
        assert contract["tokens"]["train_sequence_tokens"] == 785
        assert contract["tokens"]["inference_sequence_tokens"] == 529
        assert contract["attention"]["source"] == "model"
        assert contract["attention"]["num_image_tokens"] == 512
        assert contract["attention"]["condition_reads_targets"] is False
        assert contract["attention"]["action_reads_heatmap"] is False
        assert contract["attention"]["heatmap_reads_action"] is False
        assert contract["attention"]["action_inference_drops_heatmap"] is True
        assert contract["model"]["obs_encoder_checkpoint_path"] == ""
        assert contract["model"]["obs_encoder_checkpoint_path_exists"] is False
        assert contract["model"]["obs_encoder_checkpoint_path_is_file"] is False
        assert contract["model"]["obs_encoder_cache_dir"] == ""
        assert contract["model"]["obs_encoder_cache_dir_exists"] is False
        assert contract["model"]["obs_encoder_cache_dir_is_dir"] is False
        assert contract["model"]["obs_encoder_local_weight_source_configured"] is False
        assert contract["model"]["obs_encoder_local_weight_source_exists"] is True
        assert contract["model"]["obs_encoder_local_weight_source_valid"] is True
        contract_row = next(
            row for row in rows if "training_contract_canonical_main_config_ok" in row
        )
        assert contract_row["training_contract_canonical_main_config_ok"] == 0
        assert contract_row["training_contract_robot_ratio"] == 0.75
        assert contract_row["training_contract_open_ratio"] == 0.25
        assert contract_row["training_contract_num_processes"] == 1
        assert contract_row["training_contract_effective_train_batch_size"] == 8
        assert contract_row["training_contract_robot_gaze_dropout_prob"] == 1.0
        ckpt_path = output_dir / "checkpoints" / "latest.ckpt"
        if ckpt_path.exists():
            checkpoint_payload = torch.load(
                open(ckpt_path, "rb"),
                pickle_module=__import__("dill"),
                map_location="cpu",
            )
            checkpoint_cfg = checkpoint_payload["cfg"]
        else:
            assert cfg.checkpoint.save_last_ckpt is False
            checkpoint_cfg = workspace.cfg
        assert (
            int(checkpoint_cfg.training.robot_batch_size_per_process)
            == contract["batching"]["robot_batch_size_per_process"]
        )
        assert (
            int(checkpoint_cfg.training.open_batch_size_per_process)
            == contract["batching"]["open_batch_size_per_process"]
        )
        assert (
            int(checkpoint_cfg.training.train_batch_size_per_process)
            == contract["batching"]["train_batch_size_per_process"]
        )
        assert checkpoint_cfg.training.robot_ratio == contract["batching"]["robot_ratio"]
        assert checkpoint_cfg.training.open_ratio == contract["batching"]["open_ratio"]
        assert int(checkpoint_cfg.training.num_processes) == contract["batching"]["num_processes"]
        assert checkpoint_cfg.training.mixed_precision == contract["batching"]["mixed_precision"]
        assert checkpoint_cfg.training.distributed_type == contract["batching"]["distributed_type"]
        assert (
            checkpoint_cfg.data_mixing.requested_batch_size_source
            == contract["batching"]["requested_batch_size_source"]
        )
        assert (
            checkpoint_cfg.data_mixing.resolved_batch_size_source
            == contract["batching"]["resolved_batch_size_source"]
        )
        assert (
            checkpoint_cfg.data_mixing.requested_total_batch_size_per_process
            == contract["batching"]["requested_total_batch_size_per_process"]
        )
        assert (
            checkpoint_cfg.data_mixing.requested_robot_ratio
            == contract["batching"]["requested_robot_ratio"]
        )
        assert (
            checkpoint_cfg.data_mixing.requested_open_ratio
            == contract["batching"]["requested_open_ratio"]
        )
        assert (
            int(checkpoint_cfg.training.effective_robot_batch_size_per_optimizer_step)
            == contract["batching"]["effective_robot_batch_size_per_optimizer_step"]
        )
        assert (
            int(checkpoint_cfg.training.effective_open_batch_size_per_optimizer_step)
            == contract["batching"]["effective_open_batch_size_per_optimizer_step"]
        )
        assert (
            int(checkpoint_cfg.training.effective_train_batch_size_per_optimizer_step)
            == contract["batching"]["effective_train_batch_size_per_optimizer_step"]
        )
        assert [row["global_step"] for row in rows] == list(range(len(rows)))
        assert first_row["train_accumulated_microbatches"] == 2
        assert "train_heatmap_xy_loss" in first_row
        assert "train_heatmap_js_loss" in first_row
        assert "train_heatmap_token_kl_loss" in first_row
        assert first_row["train_routing_robot_rows"] == 6
        assert first_row["train_routing_open_rows"] == 2
        assert first_row["train_routing_robot_action_loss_count"] == 6
        assert first_row["train_routing_open_action_loss_count"] == 0
        assert first_row["train_routing_open_heatmap_loss_count"] == 2
        assert first_row["train_routing_robot_real_gaze_heatmap_loss_count"] == 0
        assert first_row["train_routing_robot_masked_gaze_heatmap_loss_count"] == 6
        assert (
            first_row["train_routing_robot_action_loss_count"]
            + first_row["train_routing_open_action_loss_count"]
            == first_row["train_action_mask_count"]
        )
        assert (
            first_row["train_routing_robot_heatmap_loss_count"]
            + first_row["train_routing_open_heatmap_loss_count"]
            == first_row["train_heatmap_mask_count"]
        )
        assert "val_loss" in final_row
        assert "val_action_loss" in final_row
        assert "val_heatmap_loss" in final_row
        assert "val_heatmap_xy_loss" in final_row
        assert "val_heatmap_js_loss" in final_row
        assert "val_heatmap_token_kl_loss" in final_row
        assert "val_robot_action_loss" in final_row
        assert "val_open_heatmap_loss" in final_row
        assert final_row["val_action_mask_count"] > 0
        assert final_row["val_heatmap_mask_count"] > 0
        assert final_row["val_robot_action_mask_count"] == final_row["val_action_mask_count"]
        assert final_row["val_robot_heatmap_mask_count"] >= 0
        assert final_row["val_open_heatmap_mask_count"] > 0
        assert (
            final_row["val_robot_heatmap_mask_count"]
            + final_row["val_open_heatmap_mask_count"]
            == final_row["val_heatmap_mask_count"]
        )
        if final_row["val_robot_heatmap_mask_count"] > 0:
            assert "val_robot_heatmap_loss" in final_row
        assert final_row["val_heatmap_preview_saved"] == 1
        assert final_row["val_heatmap_preview_num_samples"] == 2
        assert final_row["val_loss"] > 0

        preview_dir = output_dir / "media" / "val_heatmap" / "epoch_0000"
        for name in (
            "rgb.png",
            "pred_heatmap.png",
            "target_heatmap.png",
            "pred_overlay.png",
            "target_overlay.png",
            "comparison.png",
            "summary.json",
        ):
            assert (preview_dir / name).exists()
        summary = json.loads((preview_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["num_samples"] == 2
        assert summary["max_samples"] == 2
        assert len(summary["samples"]) == 2
        assert summary["pred_heatmap_shape"] == [256, 256]
        assert summary["target_heatmap_shape"] == [256, 256]
        assert "pred_token_argmax" in summary
        for sample_idx in range(2):
            sample_dir = preview_dir / f"sample_{sample_idx:03d}"
            for name in (
                "rgb.png",
                "pred_heatmap.png",
                "target_heatmap.png",
                "pred_overlay.png",
                "target_overlay.png",
                "comparison.png",
            ):
                assert (sample_dir / name).exists()
        pred_overlay = cv2.imread(str(preview_dir / "pred_overlay.png"), cv2.IMREAD_UNCHANGED)
        assert pred_overlay.shape[:2] == (256, 256)


def test_gaze_wam_validation_preview_runs_under_amp_context():
    workspace_source = (
        Path(__file__).resolve().parents[1]
        / "diffusion_policy"
        / "workspace"
        / "train_gaze_wam_workspace.py"
    )
    text = workspace_source.read_text(encoding="utf-8")
    preview_call = text.index("val_preview_summary = _write_heatmap_preview(")
    amp_context = text.rfind(
        "with _gaze_wam_autocast(accelerator):", 0, preview_call
    )
    assert amp_context >= 0
    assert text[amp_context:preview_call].count(
        "with _gaze_wam_autocast(accelerator):"
    ) == 1


def test_gaze_wam_dense_heatmap_target_matches_generated_target_dtype():
    class FakeHeatmapCodec:
        image_size = (4, 4)

    class FakePolicy:
        heatmap_codec = FakeHeatmapCodec()

        @staticmethod
        def _target_heatmap_image_from_xy(gaze_xy, valid_mask):
            return torch.zeros(
                gaze_xy.shape[0],
                4,
                4,
                device=gaze_xy.device,
                dtype=torch.float32,
            )

        @staticmethod
        def _require_bool_vector(name, value, expected_length):
            assert value.dtype == torch.bool
            assert value.shape == (expected_length,)

    batch = {
        "heatmap_image": torch.ones(2, 1, 4, 4, dtype=torch.bfloat16),
        "has_heatmap_image": torch.tensor([True, False]),
    }
    target = GazeWamPolicy._target_heatmap_image_from_batch_or_xy(
        FakePolicy(),
        batch=batch,
        gaze_xy=torch.zeros(2, 2, dtype=torch.bfloat16),
        valid_mask=torch.tensor([True, True]),
    )

    assert target.dtype == torch.float32
    assert torch.allclose(target[0], torch.ones(4, 4))
    assert torch.allclose(target[1], torch.zeros(4, 4))

    batch["heatmap_image"][0] = 0.0
    try:
        GazeWamPolicy._target_heatmap_image_from_batch_or_xy(
            FakePolicy(),
            batch=batch,
            gaze_xy=torch.zeros(2, 2, dtype=torch.bfloat16),
            valid_mask=torch.tensor([True, True]),
        )
    except ValueError as exc:
        assert "dense targets must have positive spatial mass" in str(exc)
    else:
        raise AssertionError("Expected an all-zero active dense target to be rejected.")


def test_gaze_wam_policy_mixed_batch_loss_and_inference(tmp_path):
    torch.manual_seed(2)
    shape_meta = {
        "action": {
            "shape": [10],
            "horizon": 4,
        }
    }
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    encoder_path, decoder_path = _write_fake_cosmos_jit_pair(
        tmp_path,
        image_size=(256, 256),
        token_grid=(1, 8),
        latent_channels=1,
    )
    obs_encoder = FakeTokenObsEncoder(num_tokens=8, embed_dim=32)
    model = JointGazeWamTransformer(
        action_dim=10,
        heatmap_dim=1,
        action_horizon=4,
        heatmap_num_tokens=8,
        max_image_tokens=8,
        n_layer=1,
        n_head=4,
        n_emb=32,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
    )
    gaze_encoder = GazeConditionEncoder(embed_dim=32)
    policy = GazeWamPolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        obs_encoder=obs_encoder,
        model=model,
        gaze_encoder=gaze_encoder,
        num_inference_steps=2,
        input_pertub=0.0,
        heatmap_num_tokens=8,
        heatmap_token_grid=(1, 8),
        heatmap_cosmos_encoder_path=encoder_path,
        heatmap_cosmos_decoder_path=decoder_path,
        n_emb=32,
    )
    normalizer = LinearNormalizer()
    normalizer["camera0_rgb"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)

    batch = {
        "obs": {
            "camera0_rgb": torch.randn(4, 2, 3, 16, 16),
        },
        "action": torch.randn(4, 4, 10),
        "action_abs": torch.randn(4, 4, 10),
        "action_base_abs": torch.randn(4, 10),
        "has_action_abs": torch.tensor([True, True, False, False]),
        "has_action_base_abs": torch.tensor([True, True, False, False]),
        "heatmap": torch.rand(4, 1, 8, 1),
        "heatmap_image": torch.rand(4, 1, 256, 256),
        "has_heatmap_image": torch.tensor([False, True, True, True]),
        "gaze_xy": torch.tensor(
            [
                [0.5, 0.5],
                [0.25, 0.25],
                [0.75, 0.75],
                [0.1, 0.8],
            ],
            dtype=torch.float32,
        ),
        "is_open": torch.tensor([False, False, True, True]),
        "has_action": torch.tensor([True, True, False, False]),
        "has_heatmap": torch.tensor([False, True, True, True]),
        "has_gaze_label": torch.tensor([True, True, True, True]),
        "use_gaze_condition": torch.tensor([True, False, False, False]),
        "is_gaze_condition_dropped": torch.tensor([False, True, True, True]),
    }
    batch["action"][2:] = 0.0
    batch["action_abs"][2:] = 0.0
    batch["action_base_abs"][2:] = 0.0
    batch["heatmap"][0] = 0.0
    batch["heatmap_image"][0] = 0.0
    components = policy.compute_loss_components(batch, return_per_sample=True)

    assert components["loss"].ndim == 0
    assert components["action_loss_mask_count"].item() == 2
    assert components["heatmap_loss_mask_count"].item() == 3
    assert components["per_sample_action_loss"].shape == (4,)
    assert components["per_sample_heatmap_loss"].shape == (4,)
    assert components["per_sample_heatmap_token_kl_loss"].shape == (4,)
    assert components["action_loss_mask"].dtype == torch.bool
    assert components["heatmap_loss_mask"].dtype == torch.bool
    assert components["heatmap_token_kl_loss"].ndim == 0
    assert components["heatmap_token_kl_loss_weight"] == 0.0
    assert torch.allclose(
        components["loss"].detach(),
        components["action_loss"] + components["heatmap_loss"],
        atol=1e-6,
    )

    policy.heatmap_token_kl_loss_weight = 0.5
    torch.manual_seed(20)
    weighted_components = policy.compute_loss_components(batch, return_per_sample=True)
    assert weighted_components["heatmap_token_kl_loss_weight"] == 0.5
    assert torch.allclose(
        weighted_components["loss"].detach(),
        (
            weighted_components["action_loss"]
            + weighted_components["heatmap_loss"]
            + 0.5 * weighted_components["heatmap_token_kl_loss"]
        ),
        atol=1e-6,
    )
    policy.heatmap_token_kl_loss_weight = 0.0

    components["loss"].backward()
    assert policy.model.action_head.weight.grad is not None
    assert policy.model.heatmap_head.weight.grad is not None
    assert policy.gaze_encoder.mask_token.grad is not None

    pred = policy.predict_action(
        {
            "camera0_rgb": batch["obs"]["camera0_rgb"],
            "gaze_xy": batch["gaze_xy"],
            "use_gaze_condition": batch["use_gaze_condition"],
            "has_gaze_label": batch["has_gaze_label"],
        }
    )
    assert pred["action"].shape == (4, 4, 10)
    assert pred["action_pred"].shape == (4, 4, 10)
    assert pred["action_pred_relative"].shape == (4, 4, 10)


def test_gaze_wam_policy_open_only_dsnt_js_heatmap_loss_ignores_action(tmp_path):
    torch.manual_seed(12)
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    codec = HeatmapTokenCodec(token_grid=(2, 2), image_size=(4, 4), sigma_tokens=1.0)
    encoder_path, decoder_path = _write_fake_cosmos_jit_pair(
        tmp_path,
        image_size=codec.image_size,
        token_grid=codec.token_grid,
        latent_channels=codec.patch_area,
    )
    pixels_yx = [(1, 2), (2, 1), (3, 3)]
    gaze_xy = []
    for pixel_y, pixel_x in pixels_yx:
        gaze_xy.append([(pixel_x + 0.5) / 4.0, (pixel_y + 0.5) / 4.0])

    model = JointGazeWamTransformer(
        action_dim=10,
        heatmap_dim=codec.patch_area,
        action_horizon=2,
        heatmap_num_tokens=codec.num_tokens,
        max_image_tokens=4,
        n_layer=1,
        n_head=2,
        n_emb=16,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
    )
    policy = GazeWamPolicy(
        shape_meta={"action": {"shape": [10], "horizon": 2}},
        noise_scheduler=scheduler,
        obs_encoder=FakeTokenObsEncoder(num_tokens=4, embed_dim=16),
        model=model,
        gaze_encoder=GazeConditionEncoder(embed_dim=16),
        num_inference_steps=2,
        input_pertub=0.0,
        action_loss_weight=0.0,
        heatmap_loss_weight=1.0,
        heatmap_token_kl_loss_weight=0.0,
        heatmap_objective="dsnt_js",
        heatmap_num_tokens=codec.num_tokens,
        heatmap_dim=codec.patch_area,
        heatmap_token_grid=codec.token_grid,
        heatmap_image_size=codec.image_size,
        heatmap_cosmos_encoder_path=encoder_path,
        heatmap_cosmos_decoder_path=decoder_path,
        n_emb=16,
    )
    normalizer = LinearNormalizer()
    normalizer["camera0_rgb"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)

    batch = {
        "obs": {
            "camera0_rgb": torch.randn(3, 2, 3, 8, 8),
        },
        "action": torch.zeros(3, 2, 10),
        "heatmap": torch.zeros(3, 1, codec.num_tokens, codec.patch_area),
        "gaze_xy": torch.tensor(gaze_xy, dtype=torch.float32),
        "is_open": torch.ones(3, dtype=torch.bool),
        "has_action": torch.zeros(3, dtype=torch.bool),
        "has_heatmap": torch.ones(3, dtype=torch.bool),
        "has_gaze_label": torch.ones(3, dtype=torch.bool),
        "use_gaze_condition": torch.zeros(3, dtype=torch.bool),
        "is_gaze_condition_dropped": torch.ones(3, dtype=torch.bool),
    }

    components = policy.compute_loss_components(batch, return_per_sample=True)

    assert components["action_loss_mask_count"].item() == 0
    assert components["heatmap_loss_mask_count"].item() == 3
    assert components["heatmap_xy_loss_mask_count"].item() == 3
    assert components["per_sample_heatmap_xy_loss"].shape == (3,)
    assert components["per_sample_heatmap_js_loss"].shape == (3,)
    assert torch.isfinite(components["heatmap_xy_loss"])
    assert torch.isfinite(components["heatmap_js_loss"])
    assert torch.allclose(components["heatmap_token_kl_loss"], torch.tensor(0.0))
    assert torch.allclose(
        components["loss"].detach(),
        components["heatmap_loss"],
        atol=1e-6,
    )
    components["loss"].backward()
    assert policy.model.heatmap_head.weight.grad is not None
    assert torch.isfinite(policy.model.heatmap_head.weight.grad).all()
    assert policy.model.heatmap_head.weight.grad.abs().sum() > 0
    if policy.model.action_head.weight.grad is not None:
        assert torch.allclose(
            policy.model.action_head.weight.grad,
            torch.zeros_like(policy.model.action_head.weight.grad),
        )

    pred = policy.predict_heatmap(
        {
            "camera0_rgb": batch["obs"]["camera0_rgb"],
            "gaze_xy": batch["gaze_xy"],
            "has_gaze_label": batch["has_gaze_label"],
            "use_gaze_condition": batch["use_gaze_condition"],
        },
        decode=True,
    )
    assert pred["heatmap_image"].shape == (3, 4, 4)
    assert torch.allclose(
        pred["heatmap_image"].flatten(start_dim=1).sum(dim=-1),
        torch.ones(3),
        atol=1e-5,
    )


def test_gaze_wam_policy_diffusion_mixed_final_heatmap_loss_adds_decoded_terms(tmp_path):
    torch.manual_seed(13)
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    codec = HeatmapTokenCodec(token_grid=(2, 2), image_size=(4, 4), sigma_tokens=1.0)
    encoder_path, decoder_path = _write_fake_cosmos_jit_pair(
        tmp_path,
        image_size=codec.image_size,
        token_grid=codec.token_grid,
        latent_channels=codec.patch_area,
    )

    policy = GazeWamPolicy(
        shape_meta={"action": {"shape": [10], "horizon": 2}},
        noise_scheduler=scheduler,
        obs_encoder=FakeTokenObsEncoder(num_tokens=4, embed_dim=16),
        model=JointGazeWamTransformer(
            action_dim=10,
            heatmap_dim=codec.patch_area,
            action_horizon=2,
            heatmap_num_tokens=codec.num_tokens,
            max_image_tokens=4,
            n_layer=1,
            n_head=2,
            n_emb=16,
            p_drop_emb=0.0,
            p_drop_attn=0.0,
        ),
        gaze_encoder=GazeConditionEncoder(embed_dim=16),
        num_inference_steps=2,
        input_pertub=0.0,
        action_loss_weight=0.0,
        heatmap_loss_weight=1.0,
        heatmap_token_kl_loss_weight=0.0,
        heatmap_objective="diffusion",
        heatmap_diffusion_final_loss_enabled=True,
        heatmap_final_loss_timestep_weighting="none",
        heatmap_xy_loss_weight=0.05,
        heatmap_point_nll_loss_weight=0.001,
        heatmap_js_loss_weight=0.10,
        heatmap_num_tokens=codec.num_tokens,
        heatmap_dim=codec.patch_area,
        heatmap_token_grid=codec.token_grid,
        heatmap_image_size=codec.image_size,
        heatmap_cosmos_encoder_path=encoder_path,
        heatmap_cosmos_decoder_path=decoder_path,
        n_emb=16,
    )
    normalizer = LinearNormalizer()
    normalizer["camera0_rgb"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)

    target_heatmap = torch.zeros(3, 1, 4, 4)
    target_heatmap[0, 0, 1, 1] = 1.0
    target_heatmap[1, 0, 2, 1] = 1.0
    target_heatmap[2, 0, 1, 3] = 1.0
    batch = {
        "obs": {
            "camera0_rgb": torch.randn(3, 2, 3, 8, 8),
        },
        "action": torch.zeros(3, 2, 10),
        "heatmap": torch.zeros(3, 1, codec.num_tokens, codec.patch_area),
        "heatmap_image": target_heatmap,
        "has_heatmap_image": torch.ones(3, dtype=torch.bool),
        "gaze_xy": torch.tensor(
            [
                [0.375, 0.375],
                [0.375, 0.625],
                [0.875, 0.375],
            ],
            dtype=torch.float32,
        ),
        "is_open": torch.ones(3, dtype=torch.bool),
        "has_action": torch.zeros(3, dtype=torch.bool),
        "has_heatmap": torch.ones(3, dtype=torch.bool),
        "has_gaze_label": torch.ones(3, dtype=torch.bool),
        "use_gaze_condition": torch.zeros(3, dtype=torch.bool),
        "is_gaze_condition_dropped": torch.ones(3, dtype=torch.bool),
    }

    components = policy.compute_loss_components(batch, return_per_sample=True)

    assert components["heatmap_diffusion_final_loss_enabled"] is True
    assert components["heatmap_final_loss_timestep_weighting"] == "none"
    assert components["per_sample_heatmap_xy_loss"].shape == (3,)
    assert components["per_sample_heatmap_point_nll_loss"].shape == (3,)
    assert components["per_sample_heatmap_js_loss"].shape == (3,)
    assert components["heatmap_xy_loss"] > 0
    assert components["heatmap_point_nll_loss"] > 0
    assert components["heatmap_js_loss"] > 0
    assert torch.allclose(
        components["loss"].detach(),
        (
            components["heatmap_loss"]
            + 0.05 * components["heatmap_xy_loss"]
            + 0.001 * components["heatmap_point_nll_loss"]
            + 0.10 * components["heatmap_js_loss"]
        ),
        atol=1e-6,
    )
    components["loss"].backward()
    assert policy.model.heatmap_head.weight.grad is not None
    assert torch.isfinite(policy.model.heatmap_head.weight.grad).all()


def test_gaze_wam_policy_rejects_invalid_loss_batch_contract(tmp_path):
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    encoder_path, decoder_path = _write_fake_cosmos_jit_pair(
        tmp_path,
        image_size=(256, 256),
        token_grid=(1, 8),
        latent_channels=1,
    )
    policy = GazeWamPolicy(
        shape_meta={"action": {"shape": [10], "horizon": 4}},
        noise_scheduler=scheduler,
        obs_encoder=FakeTokenObsEncoder(num_tokens=8, embed_dim=32),
        model=JointGazeWamTransformer(
            action_dim=10,
            heatmap_dim=1,
            action_horizon=4,
            heatmap_num_tokens=8,
            max_image_tokens=8,
            n_layer=1,
            n_head=4,
            n_emb=32,
            p_drop_emb=0.0,
            p_drop_attn=0.0,
        ),
        gaze_encoder=GazeConditionEncoder(embed_dim=32),
        num_inference_steps=2,
        input_pertub=0.0,
        heatmap_num_tokens=8,
        heatmap_token_grid=(1, 8),
        heatmap_cosmos_encoder_path=encoder_path,
        heatmap_cosmos_decoder_path=decoder_path,
        n_emb=32,
    )
    normalizer = LinearNormalizer()
    normalizer["camera0_rgb"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)

    batch = {
        "obs": {"camera0_rgb": torch.randn(4, 2, 3, 16, 16)},
        "action": torch.randn(4, 4, 10),
        "action_abs": torch.randn(4, 4, 10),
        "action_base_abs": torch.randn(4, 10),
        "has_action_abs": torch.tensor([True, True, False, False]),
        "has_action_base_abs": torch.tensor([True, True, False, False]),
        "heatmap": torch.rand(4, 1, 8, 1),
        "heatmap_image": torch.rand(4, 1, 256, 256),
        "has_heatmap_image": torch.tensor([False, True, True, True]),
        "gaze_xy": torch.zeros(4, 2),
        "is_open": torch.tensor([False, False, True, True]),
        "has_action": torch.tensor([True, True, False, False]),
        "has_heatmap": torch.tensor([False, True, True, True]),
        "has_gaze_label": torch.tensor([True, True, True, True]),
        "use_gaze_condition": torch.tensor([True, False, False, False]),
        "is_gaze_condition_dropped": torch.tensor([False, True, True, True]),
    }
    batch["action"][2:] = 0.0
    batch["action_abs"][2:] = 0.0
    batch["action_base_abs"][2:] = 0.0
    batch["heatmap"][0] = 0.0
    batch["heatmap_image"][0] = 0.0

    def clone_batch():
        return {
            key: (
                {obs_key: obs_value.clone() for obs_key, obs_value in value.items()}
                if isinstance(value, dict)
                else value.clone()
            )
            for key, value in batch.items()
        }

    invalid_cases = [
        ("bad_action_shape", lambda b: b.__setitem__("action", torch.randn(4, 3, 10)), "batch['action'] must have shape"),
        (
            "bad_action_abs_shape",
            lambda b: b.__setitem__("action_abs", torch.randn(4, 3, 10)),
            "batch['action_abs'] must have shape",
        ),
        (
            "bad_action_abs_dtype",
            lambda b: b.__setitem__("action_abs", torch.zeros(4, 4, 10, dtype=torch.int64)),
            "batch['action_abs'] must be a floating tensor",
        ),
        (
            "nonfinite_action_abs",
            lambda b: b["action_abs"].__setitem__((0, 0, 0), float("nan")),
            "batch['action_abs'] must contain only finite values",
        ),
        (
            "bad_action_base_abs_shape",
            lambda b: b.__setitem__("action_base_abs", torch.randn(4, 9)),
            "batch['action_base_abs'] must have shape",
        ),
        (
            "bad_action_base_abs_dtype",
            lambda b: b.__setitem__("action_base_abs", torch.zeros(4, 10, dtype=torch.int64)),
            "batch['action_base_abs'] must be a floating tensor",
        ),
        (
            "nonfinite_action_base_abs",
            lambda b: b["action_base_abs"].__setitem__((0, 0), float("inf")),
            "batch['action_base_abs'] must contain only finite values",
        ),
        (
            "bad_action_abs_mask_dtype",
            lambda b: b.__setitem__("has_action_abs", torch.ones(4)),
            "batch['has_action_abs'] must be a BoolTensor",
        ),
        (
            "bad_action_abs_mask_shape",
            lambda b: b.__setitem__("has_action_abs", torch.zeros(4, 1, dtype=torch.bool)),
            "batch['has_action_abs'] must have shape",
        ),
        (
            "bad_action_base_abs_mask_dtype",
            lambda b: b.__setitem__("has_action_base_abs", torch.ones(4)),
            "batch['has_action_base_abs'] must be a BoolTensor",
        ),
        (
            "bad_action_base_abs_mask_shape",
            lambda b: b.__setitem__(
                "has_action_base_abs",
                torch.zeros(4, 1, dtype=torch.bool),
            ),
            "batch['has_action_base_abs'] must have shape",
        ),
        (
            "orphan_action_abs_mask",
            lambda b: b.pop("action_abs"),
            "batch['has_action_abs'] requires batch['action_abs']",
        ),
        (
            "orphan_action_base_abs_mask",
            lambda b: b.pop("action_base_abs"),
            "batch['has_action_base_abs'] requires batch['action_base_abs']",
        ),
        (
            "open_action_abs_marked_valid",
            lambda b: b["has_action_abs"].__setitem__(2, True),
            "Open-source rows must not mark batch['has_action_abs']=True",
        ),
        (
            "open_action_base_abs_marked_valid",
            lambda b: b["has_action_base_abs"].__setitem__(2, True),
            "Open-source rows must not mark batch['has_action_base_abs']=True",
        ),
        (
            "nonzero_inactive_action_abs",
            lambda b: b["action_abs"].__setitem__((2, 0, 0), 0.5),
            "batch['action_abs'] rows with batch['has_action_abs']=False must be zero placeholders",
        ),
        (
            "nonzero_inactive_action_base_abs",
            lambda b: b["action_base_abs"].__setitem__((2, 0), 0.5),
            (
                "batch['action_base_abs'] rows with batch['has_action_base_abs']=False "
                "must be zero placeholders"
            ),
        ),
        ("bad_heatmap_shape", lambda b: b.__setitem__("heatmap", torch.randn(4, 8)), "batch['heatmap'] must have shape"),
        ("bad_gaze_shape", lambda b: b.__setitem__("gaze_xy", torch.zeros(4, 3)), "batch['gaze_xy'] must have shape"),
        ("legacy_valid_mask", lambda b: b.__setitem__("valid_mask", torch.ones(4, dtype=torch.bool)), "valid_mask"),
        ("bad_mask_dtype", lambda b: b.__setitem__("has_action", torch.ones(4)), "batch['has_action'] must be a BoolTensor"),
        ("bad_mask_shape", lambda b: b.__setitem__("is_open", torch.zeros(4, 1, dtype=torch.bool)), "batch['is_open'] must have shape"),
        ("bad_obs_batch", lambda b: b["obs"].__setitem__("camera0_rgb", torch.randn(3, 2, 3, 16, 16)), "batch['obs']['camera0_rgb'] batch dim"),
        ("bad_obs_dtype", lambda b: b["obs"].__setitem__("camera0_rgb", torch.zeros(4, 2, 3, 16, 16, dtype=torch.uint8)), "batch['obs']['camera0_rgb'] must be a floating tensor"),
        ("nonfinite_obs", lambda b: b["obs"]["camera0_rgb"].__setitem__((0, 0, 0, 0, 0), float("inf")), "batch['obs']['camera0_rgb'] must contain only finite values"),
        (
            "nonzero_inactive_action",
            lambda b: b["action"].__setitem__((2, 0, 0), 0.5),
            "batch['action'] rows with has_action=False must be zero placeholders",
        ),
        ("nonfinite_heatmap", lambda b: b["heatmap"].__setitem__((0, 0, 0, 0), float("nan")), "batch['heatmap'] must contain only finite values"),
        ("negative_heatmap", lambda b: b["heatmap"].__setitem__((0, 0, 0, 0), -0.1), "batch['heatmap'] must be in [0, 1]"),
        ("large_heatmap", lambda b: b["heatmap"].__setitem__((0, 0, 0, 0), 1.1), "batch['heatmap'] must be in [0, 1]"),
        (
            "nonzero_inactive_heatmap",
            lambda b: b["heatmap"].__setitem__((0, 0, 0, 0), 0.5),
            "batch['heatmap'] rows with has_heatmap=False must be zero placeholders",
        ),
        (
            "bad_heatmap_image_batch",
            lambda b: b.__setitem__("heatmap_image", torch.rand(3, 1, 256, 256)),
            "batch['heatmap_image'] batch dim",
        ),
        (
            "bad_heatmap_image_rank",
            lambda b: b.__setitem__("heatmap_image", torch.rand(4, 1, 1, 256, 256)),
            "batch['heatmap_image'] must have shape",
        ),
        (
            "bad_heatmap_image_channel",
            lambda b: b.__setitem__("heatmap_image", torch.rand(4, 2, 256, 256)),
            "batch['heatmap_image'] must have a single channel",
        ),
        (
            "bad_heatmap_image_spatial",
            lambda b: b.__setitem__("heatmap_image", torch.rand(4, 1, 128, 256)),
            "batch['heatmap_image'] spatial shape",
        ),
        (
            "bad_heatmap_image_dtype",
            lambda b: b.__setitem__("heatmap_image", torch.ones(4, 1, 256, 256, dtype=torch.uint8)),
            "batch['heatmap_image'] must be a floating tensor",
        ),
        (
            "nonfinite_heatmap_image",
            lambda b: b["heatmap_image"].__setitem__((0, 0, 0, 0), float("inf")),
            "batch['heatmap_image'] must contain only finite values",
        ),
        (
            "negative_heatmap_image",
            lambda b: b["heatmap_image"].__setitem__((0, 0, 0, 0), -0.1),
            "batch['heatmap_image'] must be non-negative",
        ),
        (
            "bad_heatmap_image_mask_dtype",
            lambda b: b.__setitem__("has_heatmap_image", torch.ones(4)),
            "batch['has_heatmap_image'] must be a BoolTensor",
        ),
        (
            "bad_heatmap_image_mask_shape",
            lambda b: b.__setitem__(
                "has_heatmap_image",
                torch.zeros(4, 1, dtype=torch.bool),
            ),
            "batch['has_heatmap_image'] must have shape",
        ),
        (
            "orphan_heatmap_image_mask",
            lambda b: b.pop("heatmap_image"),
            "batch['has_heatmap_image'] requires batch['heatmap_image']",
        ),
        (
            "nonzero_inactive_heatmap_image",
            lambda b: b["heatmap_image"].__setitem__((0, 0, 0, 0), 0.5),
            "batch['heatmap_image'] rows with batch['has_heatmap_image']=False must be zero placeholders",
        ),
        ("out_of_bounds_gaze", lambda b: b["gaze_xy"].__setitem__((0, 0), 1.1), "batch['gaze_xy'] rows with has_gaze_label=True"),
        (
            "nonzero_inactive_gaze_xy",
            lambda b: (
                b["has_gaze_label"].__setitem__(3, False),
                b["gaze_xy"].__setitem__((3, 0), 0.5),
            ),
            "batch['gaze_xy'] rows with has_gaze_condition=False must be zero placeholders",
        ),
        ("gaze_without_label", lambda b: b["has_gaze_label"].__setitem__(0, False), "use_gaze_condition"),
        ("open_has_action", lambda b: b["has_action"].__setitem__(2, True), "Open-source rows must have"),
        ("open_uses_gaze", lambda b: b["use_gaze_condition"].__setitem__(2, True), "Open-source rows must use"),
        ("robot_missing_action", lambda b: b["has_action"].__setitem__(0, False), "Robot rows must have"),
        ("open_missing_heatmap", lambda b: b["has_heatmap"].__setitem__(2, False), "Open-source rows must have"),
        (
            "missing_dropout_mask",
            lambda b: b.pop("is_gaze_condition_dropped"),
            "batch must contain 'is_gaze_condition_dropped'",
        ),
        (
            "bad_dropout_mask",
            lambda b: b.__setitem__(
                "is_gaze_condition_dropped",
                torch.tensor([False, False, True, True]),
            ),
            "is_gaze_condition_dropped",
        ),
    ]
    for _, mutate, expected in invalid_cases:
        bad_batch = clone_batch()
        mutate(bad_batch)
        try:
            policy.compute_loss_components(bad_batch)
        except (KeyError, TypeError, ValueError) as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"Expected invalid loss batch case {expected!r} to fail.")


def test_gaze_wam_policy_heatmap_clean_token_objective_target(tmp_path):
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    encoder_path, decoder_path = _write_fake_cosmos_jit_pair(
        tmp_path,
        image_size=(4, 4),
        token_grid=(2, 2),
        latent_channels=1,
    )
    cosmos_kwargs = {
        "heatmap_image_size": (4, 4),
        "heatmap_cosmos_encoder_path": encoder_path,
        "heatmap_cosmos_decoder_path": decoder_path,
    }
    policy = GazeWamPolicy(
        shape_meta={"action": {"shape": [10], "horizon": 2}},
        noise_scheduler=scheduler,
        obs_encoder=FakeTokenObsEncoder(num_tokens=4, embed_dim=16),
        model=JointGazeWamTransformer(
            action_dim=10,
            heatmap_dim=1,
            action_horizon=2,
            heatmap_num_tokens=4,
            max_image_tokens=4,
            n_layer=1,
            n_head=2,
            n_emb=16,
            p_drop_emb=0.0,
            p_drop_attn=0.0,
        ),
        gaze_encoder=GazeConditionEncoder(embed_dim=16),
        heatmap_num_tokens=4,
        heatmap_token_grid=(2, 2),
        n_emb=16,
        heatmap_objective="clean_token",
        **cosmos_kwargs,
    )
    clean = torch.randn(2, 4, 1)
    noise = torch.randn(2, 4, 1)
    assert torch.allclose(policy._heatmap_target(clean, noise), clean)
    assert torch.allclose(policy._scheduler_target(clean, noise), noise)

    legacy_alias_policy = GazeWamPolicy(
        shape_meta={"action": {"shape": [10], "horizon": 2}},
        noise_scheduler=scheduler,
        obs_encoder=FakeTokenObsEncoder(num_tokens=4, embed_dim=16),
        model=JointGazeWamTransformer(
            action_dim=10,
            heatmap_dim=1,
            action_horizon=2,
            heatmap_num_tokens=4,
            max_image_tokens=4,
            n_layer=1,
            n_head=2,
            n_emb=16,
            p_drop_emb=0.0,
            p_drop_attn=0.0,
        ),
        gaze_encoder=GazeConditionEncoder(embed_dim=16),
        heatmap_num_tokens=4,
        heatmap_token_grid=(2, 2),
        n_emb=16,
        heatmap_objective="sample",
        **cosmos_kwargs,
    )
    assert legacy_alias_policy.heatmap_objective == "clean_token"
    assert torch.allclose(legacy_alias_policy._heatmap_target(clean, noise), clean)

    diffusion_policy_obj = GazeWamPolicy(
        shape_meta={"action": {"shape": [10], "horizon": 2}},
        noise_scheduler=scheduler,
        obs_encoder=FakeTokenObsEncoder(num_tokens=4, embed_dim=16),
        model=JointGazeWamTransformer(
            action_dim=10,
            heatmap_dim=1,
            action_horizon=2,
            heatmap_num_tokens=4,
            max_image_tokens=4,
            n_layer=1,
            n_head=2,
            n_emb=16,
            p_drop_emb=0.0,
            p_drop_attn=0.0,
        ),
        gaze_encoder=GazeConditionEncoder(embed_dim=16),
        heatmap_num_tokens=4,
        heatmap_token_grid=(2, 2),
        n_emb=16,
        heatmap_objective="diffusion",
        **cosmos_kwargs,
    )
    assert torch.allclose(diffusion_policy_obj._heatmap_target(clean, noise), noise)

    dsnt_policy_obj = GazeWamPolicy(
        shape_meta={"action": {"shape": [10], "horizon": 2}},
        noise_scheduler=scheduler,
        obs_encoder=FakeTokenObsEncoder(num_tokens=4, embed_dim=16),
        model=JointGazeWamTransformer(
            action_dim=10,
            heatmap_dim=1,
            action_horizon=2,
            heatmap_num_tokens=4,
            max_image_tokens=4,
            n_layer=1,
            n_head=2,
            n_emb=16,
            p_drop_emb=0.0,
            p_drop_attn=0.0,
        ),
        gaze_encoder=GazeConditionEncoder(embed_dim=16),
        heatmap_num_tokens=4,
        heatmap_token_grid=(2, 2),
        n_emb=16,
        heatmap_objective="dsnt_js",
        **cosmos_kwargs,
    )
    assert torch.allclose(dsnt_policy_obj._heatmap_target(clean, noise), noise)

    timestep = torch.tensor([3, 3])
    clean_like_prediction = torch.rand(2, 4, 1)
    noisy_sample = scheduler.add_noise(clean, noise, timestep)
    assert torch.allclose(
        policy._heatmap_clean_prediction(
            sample=noisy_sample,
            model_output=clean_like_prediction,
            timestep=timestep,
        ),
        clean_like_prediction,
    )
    diffusion_clean = diffusion_policy_obj._heatmap_clean_prediction(
        sample=noisy_sample,
        model_output=noise,
        timestep=timestep,
    )
    assert torch.allclose(diffusion_clean, clean, atol=1e-5)
    dsnt_clean = dsnt_policy_obj._heatmap_clean_prediction(
        sample=noisy_sample,
        model_output=noise,
        timestep=timestep,
    )
    assert torch.allclose(dsnt_clean, clean, atol=1e-5)

    try:
        GazeWamPolicy(
            shape_meta={"action": {"shape": [10], "horizon": 2}},
            noise_scheduler=scheduler,
            obs_encoder=FakeTokenObsEncoder(num_tokens=4, embed_dim=16),
            heatmap_num_tokens=4,
            n_emb=16,
            heatmap_objective="bad",
            **cosmos_kwargs,
        )
    except ValueError as exc:
        assert "heatmap_objective" in str(exc)
    else:
        raise AssertionError("Expected invalid heatmap_objective to fail.")


def test_gaze_wam_policy_rejects_component_contract_mismatch():
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    shape_meta = {"action": {"shape": [10], "horizon": 2}}

    try:
        GazeWamPolicy(
            shape_meta=shape_meta,
            noise_scheduler=scheduler,
            obs_encoder=FakeTokenObsEncoder(num_tokens=4, embed_dim=16),
            model=JointGazeWamTransformer(
                action_dim=9,
                heatmap_dim=1,
                action_horizon=2,
                heatmap_num_tokens=4,
                max_image_tokens=4,
                n_layer=1,
                n_head=2,
                n_emb=16,
                p_drop_emb=0.0,
                p_drop_attn=0.0,
            ),
            gaze_encoder=GazeConditionEncoder(embed_dim=16),
            heatmap_num_tokens=4,
            heatmap_token_grid=(2, 2),
            n_emb=16,
        )
    except ValueError as exc:
        assert "model.action_dim" in str(exc)
    else:
        raise AssertionError("Expected mismatched action dimension to fail.")

    try:
        GazeWamPolicy(
            shape_meta=shape_meta,
            noise_scheduler=scheduler,
            obs_encoder=FakeTokenObsEncoder(num_tokens=4, embed_dim=16),
            model=JointGazeWamTransformer(
                action_dim=10,
                heatmap_dim=1,
                action_horizon=2,
                heatmap_num_tokens=4,
                max_image_tokens=4,
                n_layer=1,
                n_head=2,
                n_emb=16,
                p_drop_emb=0.0,
                p_drop_attn=0.0,
            ),
            gaze_encoder=GazeConditionEncoder(embed_dim=32),
            heatmap_num_tokens=4,
            heatmap_token_grid=(2, 2),
            n_emb=16,
        )
    except ValueError as exc:
        assert "gaze_encoder output dim" in str(exc)
    else:
        raise AssertionError("Expected mismatched gaze encoder dimension to fail.")


def test_gaze_wam_policy_rejects_invalid_scalar_hyperparameters(tmp_path):
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    shape_meta = {"action": {"shape": [10], "horizon": 2}}
    encoder_path, decoder_path = _write_fake_cosmos_jit_pair(
        tmp_path,
        image_size=(4, 4),
        token_grid=(2, 2),
        latent_channels=1,
    )

    def make_policy(**kwargs):
        policy_kwargs = {
            "heatmap_num_tokens": 4,
            "heatmap_dim": 1,
            "heatmap_token_grid": (2, 2),
            "heatmap_image_size": (4, 4),
            "heatmap_cosmos_encoder_path": encoder_path,
            "heatmap_cosmos_decoder_path": decoder_path,
            "n_emb": 16,
        }
        policy_kwargs.update(kwargs)
        return GazeWamPolicy(
            shape_meta=shape_meta,
            noise_scheduler=scheduler,
            obs_encoder=FakeTokenObsEncoder(num_tokens=4, embed_dim=16),
            **policy_kwargs,
        )

    valid_policy = make_policy(
        input_pertub=0,
        action_loss_weight=0,
        heatmap_loss_weight=0,
        heatmap_token_kl_loss_weight=0,
        heatmap_xy_loss_weight=0,
        heatmap_point_nll_loss_weight=0,
        heatmap_js_loss_weight=0,
        heatmap_dsnt_temperature=1,
        cfg_scale=0,
        num_inference_steps=2,
    )
    assert valid_policy.input_pertub == 0.0
    assert valid_policy.action_loss_weight == 0.0
    assert valid_policy.heatmap_loss_weight == 0.0
    assert valid_policy.heatmap_token_kl_loss_weight == 0.0
    assert valid_policy.heatmap_xy_loss_weight == 0.0
    assert valid_policy.heatmap_point_nll_loss_weight == 0.0
    assert valid_policy.heatmap_js_loss_weight == 0.0
    assert valid_policy.heatmap_dsnt_temperature == 1.0
    assert valid_policy.cfg_scale == 0.0
    assert valid_policy.num_inference_steps == 2
    string_steps_policy = make_policy(num_inference_steps="2")
    assert string_steps_policy.num_inference_steps == 2
    string_grid_policy = make_policy(heatmap_token_grid=("2", "2"))
    assert string_grid_policy.heatmap_codec.token_grid == (2, 2)
    string_core_policy = make_policy(
        heatmap_num_tokens="4",
        heatmap_dim="1",
        max_image_tokens="4",
        n_layer="1",
        n_head="2",
        n_emb="16",
    )
    assert string_core_policy.heatmap_num_tokens == 4
    assert string_core_policy.heatmap_dim == 1
    assert string_core_policy.model.n_emb == 16

    invalid_cases = [
        ({"input_pertub": -0.1}, "input_pertub"),
        ({"input_pertub": float("nan")}, "input_pertub"),
        ({"input_pertub": True}, "input_pertub"),
        ({"action_loss_weight": -1.0}, "action_loss_weight"),
        ({"action_loss_weight": False}, "action_loss_weight"),
        ({"heatmap_loss_weight": -1.0}, "heatmap_loss_weight"),
        ({"heatmap_loss_weight": True}, "heatmap_loss_weight"),
        ({"heatmap_token_kl_loss_weight": -1.0}, "heatmap_token_kl_loss_weight"),
        ({"heatmap_token_kl_loss_weight": False}, "heatmap_token_kl_loss_weight"),
        ({"heatmap_xy_loss_weight": -1.0}, "heatmap_xy_loss_weight"),
        ({"heatmap_xy_loss_weight": False}, "heatmap_xy_loss_weight"),
        ({"heatmap_point_nll_loss_weight": -1.0}, "heatmap_point_nll_loss_weight"),
        ({"heatmap_point_nll_loss_weight": True}, "heatmap_point_nll_loss_weight"),
        ({"heatmap_js_loss_weight": -1.0}, "heatmap_js_loss_weight"),
        ({"heatmap_js_loss_weight": True}, "heatmap_js_loss_weight"),
        ({"heatmap_dsnt_temperature": 0.0}, "heatmap_dsnt_temperature"),
        ({"heatmap_dsnt_temperature": "oops"}, "heatmap_dsnt_temperature"),
        ({"heatmap_distribution_mode": "bad"}, "heatmap_distribution_mode"),
        ({"cfg_scale": -0.5}, "cfg_scale"),
        ({"cfg_scale": True}, "cfg_scale"),
        ({"num_inference_steps": 0}, "num_inference_steps"),
        ({"num_inference_steps": 1}, "at least 2"),
        ({"num_inference_steps": 1.5}, "num_inference_steps"),
        ({"num_inference_steps": True}, "num_inference_steps"),
        ({"num_inference_steps": "0"}, "num_inference_steps"),
        ({"num_inference_steps": "1"}, "at least 2"),
        ({"num_inference_steps": "1.0"}, "num_inference_steps"),
        ({"num_inference_steps": "oops"}, "num_inference_steps"),
        ({"num_inference_steps": float("inf")}, "num_inference_steps"),
        ({"heatmap_num_tokens": True}, "heatmap_num_tokens"),
        ({"heatmap_dim": 1.5}, "heatmap_dim"),
        ({"max_image_tokens": 0}, "max_image_tokens"),
        ({"n_layer": "1.0"}, "n_layer"),
        ({"n_head": True}, "n_head"),
        ({"n_emb": float("inf")}, "n_emb"),
        ({"heatmap_token_grid": (True, 4)}, "token_grid dimensions"),
        ({"heatmap_token_grid": (2.5, 2)}, "token_grid dimensions"),
        ({"heatmap_token_grid": (0, 2)}, "token_grid dimensions"),
        ({"heatmap_token_grid": (2, 3)}, "heatmap_token_grid product"),
        ({"n_head": 3}, "divisible"),
    ]
    for kwargs, expected in invalid_cases:
        try:
            make_policy(**kwargs)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"Expected invalid {expected}={kwargs[expected]!r} to fail.")


def test_gaze_wam_policy_predicts_absolute_action_and_cfg(tmp_path):
    torch.manual_seed(3)
    shape_meta = {
        "action": {
            "shape": [10],
            "horizon": 4,
        }
    }
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    encoder_path, decoder_path = _write_fake_cosmos_jit_pair(
        tmp_path,
        image_size=(16, 16),
        token_grid=(2, 4),
        latent_channels=1,
    )
    obs_encoder = FakeTokenObsEncoder(num_tokens=8, embed_dim=32)
    model = JointGazeWamTransformer(
        action_dim=10,
        heatmap_dim=1,
        action_horizon=4,
        heatmap_num_tokens=8,
        max_image_tokens=8,
        n_layer=1,
        n_head=4,
        n_emb=32,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
    )
    policy = GazeWamPolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        obs_encoder=obs_encoder,
        model=model,
        gaze_encoder=GazeConditionEncoder(embed_dim=32),
        num_inference_steps=2,
        input_pertub=0.0,
        heatmap_num_tokens=8,
        heatmap_token_grid=(2, 4),
        heatmap_image_size=(16, 16),
        heatmap_cosmos_encoder_path=encoder_path,
        heatmap_cosmos_decoder_path=decoder_path,
        n_emb=32,
    )
    normalizer = LinearNormalizer()
    normalizer["camera0_rgb"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)

    base_abs_np = _random_pose10((2,)).astype(np.float32)
    obs = {
        "camera0_rgb": torch.randn(2, 2, 3, 16, 16),
        "gaze_xy": torch.tensor([[0.5, 0.5], [0.2, 0.8]], dtype=torch.float32),
        "use_gaze_condition": torch.tensor([True, True]),
        "has_gaze_label": torch.tensor([True, True]),
        "action_base_abs": torch.from_numpy(base_abs_np),
        "has_action_base_abs": torch.tensor([True, True]),
    }

    torch.manual_seed(10)
    pred_plain = policy.predict_action(obs, cfg_scale=1.0)
    torch.manual_seed(10)
    pred_cfg = policy.predict_action(obs, cfg_scale=1.5)

    assert pred_plain["action"].shape == (2, 4, 10)
    assert pred_plain["action_abs"].shape == (2, 4, 10)
    assert pred_plain["action_pred_relative"].shape == (2, 4, 10)
    assert torch.allclose(pred_plain["action"], pred_plain["action_abs"])
    assert torch.allclose(pred_plain["action_abs"][..., 9], pred_plain["action_pred_relative"][..., 9])
    assert pred_plain["cfg_enabled"].dtype == torch.bool
    assert pred_plain["cfg_enabled"].item() is False
    assert torch.isclose(pred_plain["cfg_scale"], torch.tensor(1.0))
    assert pred_cfg["action"].shape == (2, 4, 10)
    assert pred_cfg["cfg_enabled"].item() is True
    assert torch.isclose(pred_cfg["cfg_scale"], torch.tensor(1.5))
    assert not torch.allclose(pred_plain["action_pred_relative"], pred_cfg["action_pred_relative"])

    try:
        policy.predict_action(obs, cfg_scale=-0.1)
    except ValueError as exc:
        assert "cfg_scale" in str(exc)
    else:
        raise AssertionError("Expected negative predict_action cfg_scale override to fail.")

    bad_gaze_obs = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in obs.items()
    }
    bad_gaze_obs["gaze_xy"][0, 0] = 1.1
    try:
        policy.predict_action(bad_gaze_obs, cfg_scale=1.0)
    except ValueError as exc:
        assert "predict_action gaze_xy rows with has_gaze_label=True" in str(exc)
    else:
        raise AssertionError("Expected out-of-range predict_action gaze to fail.")

    bad_mask_obs = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in obs.items()
    }
    bad_mask_obs["use_gaze_condition"] = torch.ones(2, 1, dtype=torch.bool)
    try:
        policy.predict_action(bad_mask_obs, cfg_scale=1.0)
    except ValueError as exc:
        assert "model boolean route mask" in str(exc)
    else:
        raise AssertionError("Expected bad predict_action route-mask shape to fail.")

    bad_mask_dtype_obs = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in obs.items()
    }
    bad_mask_dtype_obs["use_gaze_condition"] = torch.ones(2)
    try:
        policy.predict_action(bad_mask_dtype_obs, cfg_scale=1.0)
    except ValueError as exc:
        assert "model boolean route mask must be a BoolTensor" in str(exc)
    else:
        raise AssertionError("Expected non-bool predict_action route mask to fail.")

    bad_base_shape_obs = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in obs.items()
    }
    bad_base_shape_obs["action_base_abs"] = torch.zeros(2, 9)
    try:
        policy.predict_action(bad_base_shape_obs, cfg_scale=1.0)
    except ValueError as exc:
        assert "obs_dict['action_base_abs'] must have shape" in str(exc)
    else:
        raise AssertionError("Expected bad action_base_abs shape to fail.")

    bad_base_value_obs = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in obs.items()
    }
    bad_base_value_obs["action_base_abs"][0, 0] = float("nan")
    try:
        policy.predict_action(bad_base_value_obs, cfg_scale=1.0)
    except ValueError as exc:
        assert "obs_dict['action_base_abs'] must contain only finite values" in str(exc)
    else:
        raise AssertionError("Expected non-finite action_base_abs to fail.")

    bad_base_mask_obs = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in obs.items()
    }
    bad_base_mask_obs["has_action_base_abs"] = torch.tensor([True, False])
    try:
        policy.predict_action(bad_base_mask_obs, cfg_scale=1.0)
    except ValueError as exc:
        assert "Do not convert placeholder action bases" in str(exc)
    else:
        raise AssertionError("Expected missing action_base_abs mask row to fail.")

    bad_base_mask_shape_obs = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in obs.items()
    }
    bad_base_mask_shape_obs["has_action_base_abs"] = torch.ones(2, 1, dtype=torch.bool)
    try:
        policy.predict_action(bad_base_mask_shape_obs, cfg_scale=1.0)
    except ValueError as exc:
        assert "obs_dict['has_action_base_abs']" in str(exc)
    else:
        raise AssertionError("Expected malformed action_base_abs mask to fail.")


def test_gaze_wam_policy_filters_non_model_obs_metadata(tmp_path):
    torch.manual_seed(30)
    shape_meta = {
        "action": {
            "shape": [10],
            "horizon": 4,
        }
    }
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    encoder_path, decoder_path = _write_fake_cosmos_jit_pair(
        tmp_path,
        image_size=(16, 16),
        token_grid=(2, 4),
        latent_channels=1,
    )
    obs_encoder = FakeTokenObsEncoder(num_tokens=8, embed_dim=32)
    policy = GazeWamPolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        obs_encoder=obs_encoder,
        model=JointGazeWamTransformer(
            action_dim=10,
            heatmap_dim=1,
            action_horizon=4,
            heatmap_num_tokens=8,
            max_image_tokens=8,
            n_layer=1,
            n_head=4,
            n_emb=32,
            p_drop_emb=0.0,
            p_drop_attn=0.0,
        ),
        gaze_encoder=GazeConditionEncoder(embed_dim=32),
        num_inference_steps=2,
        input_pertub=0.0,
        heatmap_num_tokens=8,
        heatmap_token_grid=(2, 4),
        heatmap_image_size=(16, 16),
        heatmap_cosmos_encoder_path=encoder_path,
        heatmap_cosmos_decoder_path=decoder_path,
        n_emb=32,
    )
    normalizer = LinearNormalizer()
    normalizer["camera0_rgb"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)

    obs = {
        "camera0_rgb": torch.randn(2, 2, 3, 16, 16),
        "action": torch.randn(2, 4, 10),
        "action_abs": torch.randn(2, 4, 10),
        "action_base_abs": torch.randn(2, 10),
        "gaze_xy": torch.tensor([[0.5, 0.5], [0.2, 0.8]], dtype=torch.float32),
        "has_action": torch.tensor([True, True]),
        "has_action_abs": torch.tensor([True, True]),
        "has_action_base_abs": torch.tensor([True, True]),
        "has_gaze_label": torch.tensor([True, True]),
        "has_heatmap": torch.tensor([False, True]),
        "has_heatmap_image": torch.tensor([False, True]),
        "heatmap": torch.rand(2, 1, 8, 1),
        "heatmap_image": torch.rand(2, 1, 256, 256),
        "is_gaze_condition_dropped": torch.tensor([False, True]),
        "is_open": torch.tensor([False, False]),
        "use_gaze_condition": torch.tensor([True, True]),
    }

    torch.manual_seed(31)
    policy.predict_action(obs, cfg_scale=1.0)
    assert obs_encoder.last_obs_keys == ("camera0_rgb",)

    policy.predict_heatmap(
        obs,
        timestep=torch.zeros(2, dtype=torch.long),
        decode=False,
    )
    assert obs_encoder.last_obs_keys == ("camera0_rgb",)

    policy.compute_gaze_dependency_ratio(
        obs,
        noisy_action=torch.zeros(2, 4, 10),
        timestep=torch.zeros(2, dtype=torch.long),
    )
    assert obs_encoder.last_obs_keys == ("camera0_rgb",)


def test_gaze_wam_policy_rejects_metadata_only_obs_dict(tmp_path):
    torch.manual_seed(32)
    shape_meta = {
        "action": {
            "shape": [10],
            "horizon": 4,
        }
    }
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    encoder_path, decoder_path = _write_fake_cosmos_jit_pair(
        tmp_path,
        image_size=(16, 16),
        token_grid=(2, 4),
        latent_channels=1,
    )
    policy = GazeWamPolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        obs_encoder=FakeTokenObsEncoder(num_tokens=8, embed_dim=32),
        model=JointGazeWamTransformer(
            action_dim=10,
            heatmap_dim=1,
            action_horizon=4,
            heatmap_num_tokens=8,
            max_image_tokens=8,
            n_layer=1,
            n_head=4,
            n_emb=32,
            p_drop_emb=0.0,
            p_drop_attn=0.0,
        ),
        gaze_encoder=GazeConditionEncoder(embed_dim=32),
        num_inference_steps=2,
        input_pertub=0.0,
        heatmap_num_tokens=8,
        heatmap_token_grid=(2, 4),
        heatmap_image_size=(16, 16),
        heatmap_cosmos_encoder_path=encoder_path,
        heatmap_cosmos_decoder_path=decoder_path,
        n_emb=32,
    )
    normalizer = LinearNormalizer()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)

    metadata_only_obs = {
        "action": torch.randn(1, 4, 10),
        "gaze_xy": torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        "has_action": torch.tensor([True]),
        "has_gaze_label": torch.tensor([True]),
        "has_heatmap": torch.tensor([False]),
        "heatmap": torch.rand(1, 1, 8, 1),
        "is_open": torch.tensor([False]),
        "use_gaze_condition": torch.tensor([True]),
    }
    try:
        policy.predict_action(metadata_only_obs, cfg_scale=1.0)
    except ValueError as exc:
        assert "at least one model observation tensor" in str(exc)
    else:
        raise AssertionError("Expected metadata-only obs_dict to fail.")


def test_gaze_wam_inference_adapter_history_mask_and_absolute_output(tmp_path):
    torch.manual_seed(31)
    shape_meta = {
        "obs": {
            "camera0_rgb": {
                "shape": [3, 16, 16],
                "type": "rgb",
                "horizon": 2,
            }
        },
        "action": {
            "shape": [10],
            "horizon": 4,
        },
    }
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    encoder_path, decoder_path = _write_fake_cosmos_jit_pair(
        tmp_path,
        image_size=(16, 16),
        token_grid=(2, 4),
        latent_channels=1,
    )
    policy = GazeWamPolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        obs_encoder=FakeTokenObsEncoder(num_tokens=8, embed_dim=32),
        model=JointGazeWamTransformer(
            action_dim=10,
            heatmap_dim=1,
            action_horizon=4,
            heatmap_num_tokens=8,
            max_image_tokens=8,
            n_layer=1,
            n_head=4,
            n_emb=32,
            p_drop_emb=0.0,
            p_drop_attn=0.0,
        ),
        gaze_encoder=GazeConditionEncoder(embed_dim=32),
        num_inference_steps=2,
        input_pertub=0.0,
        heatmap_num_tokens=8,
        heatmap_token_grid=(2, 4),
        heatmap_image_size=(16, 16),
        heatmap_cosmos_encoder_path=encoder_path,
        heatmap_cosmos_decoder_path=decoder_path,
        n_emb=32,
    )
    normalizer = LinearNormalizer()
    normalizer["camera0_rgb"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)

    adapter = GazeWamInferenceAdapter(
        policy=policy,
        shape_meta=shape_meta,
        camera_key="camera0_rgb",
        device="cpu",
    )
    try:
        GazeWamInferenceAdapter(
            policy=policy,
            shape_meta=shape_meta,
            camera_key="camera0_rgb",
            device="cpu",
            cfg_scale=-0.1,
        )
    except ValueError as exc:
        assert "cfg_scale" in str(exc)
    else:
        raise AssertionError("Expected negative adapter cfg_scale to fail.")

    image = np.zeros((20, 24, 3), dtype=np.uint8)
    image[..., 0] = 255
    base_abs = tcp_pose_to_action_base_abs(
        np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], dtype=np.float32),
        gripper_width=0.04,
    )

    try:
        adapter.build_obs(gaze_xy=None, action_base_abs=base_abs)
        assert False, "build_obs should require at least one image before inference."
    except RuntimeError:
        pass

    torch.manual_seed(32)
    result = adapter.predict_action(
        image=image,
        gaze_xy=None,
        action_base_abs=base_abs,
    )
    try:
        adapter.predict_action(
            image=image,
            gaze_xy=None,
            action_base_abs=base_abs,
            cfg_scale=-0.1,
        )
    except ValueError as exc:
        assert "cfg_scale" in str(exc)
    else:
        raise AssertionError("Expected negative adapter predict_action cfg_scale override to fail.")

    assert len(adapter.image_history) == 1
    stacked = adapter._stack_history()
    assert stacked.shape == (2, 3, 16, 16)
    assert np.allclose(stacked[0], stacked[1])
    assert base_abs.shape == (10,)
    assert np.isclose(base_abs[9], 0.04)
    assert result["action_pred_relative"].shape == (1, 4, 10)
    assert result["action_abs"].shape == (1, 4, 10)
    assert result["action"].shape == (1, 4, 10)
    assert result["cfg_enabled"].shape == ()
    assert result["cfg_enabled"].item() is False
    assert np.isclose(result["cfg_scale"].item(), 1.0)

    history = [
        np.full((20, 24, 3), 32, dtype=np.uint8),
        np.full((20, 24, 3), 224, dtype=np.uint8),
    ]
    adapter.set_image_history(history)
    stacked_history = adapter._stack_history()
    assert stacked_history.shape == (2, 3, 16, 16)
    assert float(stacked_history[0].mean()) < float(stacked_history[1].mean())
    try:
        adapter.predict_action(
            image=image,
            image_history=history,
            action_base_abs=base_abs,
        )
    except ValueError as exc:
        assert "either image or image_history" in str(exc)
    else:
        raise AssertionError("Expected ambiguous image input to fail.")

    obs = adapter.build_obs(gaze_xy=None, action_base_abs=base_abs)
    assert obs["has_gaze_condition"].item() is False
    assert obs["has_gaze_label"].item() is False
    assert obs["use_gaze_condition"].item() is False
    assert obs["action_base_abs"].shape == (1, 10)

    pose_only_base = base_abs[:9]
    obs_pose_only = adapter.build_obs(
        gaze_xy=None,
        action_base_abs=pose_only_base,
        gripper_width=0.07,
    )
    assert obs_pose_only["action_base_abs"].shape == (1, 10)
    assert torch.allclose(obs_pose_only["action_base_abs"][0, :9], torch.from_numpy(pose_only_base))
    assert torch.isclose(obs_pose_only["action_base_abs"][0, 9], torch.tensor(0.07))

    outside_obs = adapter.build_obs(gaze_xy=[1.25, -0.1], action_base_abs=base_abs)
    assert outside_obs["has_gaze_condition"].item() is True
    assert outside_obs["has_gaze_label"].item() is False
    assert outside_obs["use_gaze_condition"].item() is True
    assert torch.allclose(
        outside_obs["gaze_xy"][0],
        torch.tensor([1.25, -0.1], dtype=torch.float32),
    )


def test_gaze_wam_training_and_inference_image_preprocessing_are_identical():
    rng = np.random.default_rng(20260812)
    image = rng.integers(0, 256, size=(37, 53, 3), dtype=np.uint8)

    training = _dataset_image_to_chw_float(image[None], image_size=(16, 16))[0]
    inference = _inference_image_to_chw_float(image, image_size=(16, 16))

    assert training.shape == (3, 16, 16)
    np.testing.assert_array_equal(inference, training)


def test_gaze_wam_action_base_abs_to_10d_requires_gripper_for_pose_only_base():
    pose9 = np.zeros(9, dtype=np.float32)
    pose9[3] = 1.0
    pose9[7] = 1.0

    base = action_base_abs_to_10d(pose9, gripper_width=0.05)
    assert base.shape == (10,)
    assert np.allclose(base[:9], pose9)
    assert np.isclose(base[9], 0.05)

    base10 = action_base_abs_to_10d(np.arange(10, dtype=np.float32), gripper_width=0.02)
    assert base10.shape == (10,)
    assert np.isclose(base10[9], 0.02)

    try:
        action_base_abs_to_10d(pose9)
    except ValueError as exc:
        assert "9D action_base_abs requires gripper_width" in str(exc)
    else:
        raise AssertionError("Expected 9D action_base_abs without gripper_width to fail.")

    try:
        action_base_abs_to_10d(np.zeros(8, dtype=np.float32))
    except ValueError as exc:
        assert "action_base_abs must be 9D" in str(exc)
    else:
        raise AssertionError("Expected invalid action_base_abs dim to fail.")


def test_gaze_wam_inference_adapter_rejects_nonfinite_inputs(tmp_path):
    torch.manual_seed(33)
    shape_meta = {
        "obs": {
            "camera0_rgb": {
                "shape": [3, 16, 16],
                "type": "rgb",
                "horizon": 2,
            }
        },
        "action": {
            "shape": [10],
            "horizon": 4,
        },
    }
    encoder_path, decoder_path = _write_fake_cosmos_jit_pair(
        tmp_path,
        image_size=(16, 16),
        token_grid=(2, 4),
        latent_channels=1,
    )
    policy = GazeWamPolicy(
        shape_meta=shape_meta,
        noise_scheduler=DDPMScheduler(
            num_train_timesteps=10,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        ),
        obs_encoder=FakeTokenObsEncoder(num_tokens=8, embed_dim=32),
        model=JointGazeWamTransformer(
            action_dim=10,
            heatmap_dim=1,
            action_horizon=4,
            heatmap_num_tokens=8,
            max_image_tokens=8,
            n_layer=1,
            n_head=4,
            n_emb=32,
            p_drop_emb=0.0,
            p_drop_attn=0.0,
        ),
        gaze_encoder=GazeConditionEncoder(embed_dim=32),
        num_inference_steps=2,
        input_pertub=0.0,
        heatmap_num_tokens=8,
        heatmap_token_grid=(2, 4),
        heatmap_image_size=(16, 16),
        heatmap_cosmos_encoder_path=encoder_path,
        heatmap_cosmos_decoder_path=decoder_path,
        n_emb=32,
    )
    normalizer = LinearNormalizer()
    normalizer["camera0_rgb"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)

    adapter = GazeWamInferenceAdapter(
        policy=policy,
        shape_meta=shape_meta,
        camera_key="camera0_rgb",
        device="cpu",
    )
    image = np.zeros((16, 16, 3), dtype=np.float32)
    adapter.push_image(image)

    bad_image = image.copy()
    bad_image[0, 0, 0] = np.nan
    try:
        adapter.push_image(bad_image)
    except ValueError as exc:
        assert "image must contain only finite values" in str(exc)
    else:
        raise AssertionError("Expected non-finite image to fail.")

    try:
        adapter.build_obs(gaze_xy=[np.nan, 0.5])
    except ValueError as exc:
        assert "gaze_xy must contain only finite values" in str(exc)
    else:
        raise AssertionError("Expected non-finite gaze_xy to fail.")

    bad_action_base = np.zeros(10, dtype=np.float32)
    bad_action_base[0] = np.inf
    try:
        action_base_abs_to_10d(bad_action_base)
    except ValueError as exc:
        assert "action_base_abs must contain only finite values" in str(exc)
    else:
        raise AssertionError("Expected non-finite action_base_abs to fail.")

    bad_tcp_pose = np.zeros(6, dtype=np.float32)
    bad_tcp_pose[1] = np.nan
    try:
        tcp_pose_to_action_base_abs(bad_tcp_pose)
    except ValueError as exc:
        assert "tcp_pose must contain only finite values" in str(exc)
    else:
        raise AssertionError("Expected non-finite tcp_pose to fail.")

    try:
        action_base_abs_to_10d(np.zeros(9, dtype=np.float32), gripper_width=np.nan)
    except ValueError as exc:
        assert "gripper_width must be finite" in str(exc)
    else:
        raise AssertionError("Expected non-finite gripper_width to fail.")


def test_gaze_wam_inference_adapter_validates_geometry_contract(tmp_path):
    torch.manual_seed(34)
    shape_meta = {
        "obs": {
            "camera0_rgb": {
                "shape": [3, 16, 16],
                "type": "rgb",
                "horizon": 2,
            }
        },
        "action": {
            "shape": [10],
            "horizon": 4,
        },
    }
    encoder_path, decoder_path = _write_fake_cosmos_jit_pair(
        tmp_path,
        image_size=(16, 16),
        token_grid=(2, 4),
        latent_channels=1,
    )
    policy = GazeWamPolicy(
        shape_meta=shape_meta,
        noise_scheduler=DDPMScheduler(
            num_train_timesteps=10,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        ),
        obs_encoder=FakeTokenObsEncoder(num_tokens=8, embed_dim=32),
        model=JointGazeWamTransformer(
            action_dim=10,
            heatmap_dim=1,
            action_horizon=4,
            heatmap_num_tokens=8,
            max_image_tokens=8,
            n_layer=1,
            n_head=4,
            n_emb=32,
            p_drop_emb=0.0,
            p_drop_attn=0.0,
        ),
        gaze_encoder=GazeConditionEncoder(embed_dim=32),
        num_inference_steps=2,
        input_pertub=0.0,
        heatmap_num_tokens=8,
        heatmap_token_grid=(2, 4),
        heatmap_image_size=(16, 16),
        heatmap_cosmos_encoder_path=encoder_path,
        heatmap_cosmos_decoder_path=decoder_path,
        n_emb=32,
    )
    normalizer = LinearNormalizer()
    normalizer["camera0_rgb"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)

    adapter = GazeWamInferenceAdapter(
        policy=policy,
        shape_meta=shape_meta,
        camera_key="camera0_rgb",
        image_size=np.asarray([16, 16], dtype=np.int64),
        n_obs_steps=np.int64(2),
        device="cpu",
    )
    assert adapter.image_size == (16, 16)
    assert adapter.n_obs_steps == 2
    assert adapter.obs_downsample_steps == 1

    invalid_cases = [
        (
            {"camera_key": "missing_camera"},
            "camera_key 'missing_camera' is missing from shape_meta['obs']",
        ),
        (
            {"shape_meta": {"obs": {"camera0_rgb": {"shape": [16, 16], "horizon": 2}}}},
            "shape_meta.obs['camera0_rgb'].shape must be at least [C,H,W]",
        ),
        (
            {"shape_meta": {"obs": {"camera0_rgb": {"shape": [2, 16, 16], "horizon": 2}}}},
            "shape_meta.obs['camera0_rgb'].shape channel dim must be 1, 3, or 4",
        ),
        (
            {"image_size": [0, 16]},
            "image_size[0] must be a positive integer",
        ),
        (
            {"image_size": [16]},
            "image_size must contain exactly two positive integers",
        ),
        (
            {"n_obs_steps": 0},
            "n_obs_steps must be a positive integer",
        ),
        (
            {"n_obs_steps": True},
            "n_obs_steps must be a positive integer",
        ),
        (
            {"obs_downsample_steps": 0},
            "obs_downsample_steps must be a positive integer",
        ),
    ]
    for overrides, expected_message in invalid_cases:
        kwargs = {
            "policy": policy,
            "shape_meta": shape_meta,
            "camera_key": "camera0_rgb",
            "device": "cpu",
        }
        kwargs.update(overrides)
        try:
            GazeWamInferenceAdapter(**kwargs)
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError(f"Expected invalid adapter geometry override {overrides} to fail.")


def test_gaze_wam_deployment_runner_dry_run_safety_and_schedule():
    class FakeAdapter:
        def __init__(self):
            self.calls = []

        def reset(self):
            self.calls.clear()

        def predict_action(self, **kwargs):
            self.calls.append(kwargs)
            action_abs = np.zeros((1, 4, 10), dtype=np.float32)
            action_abs[0, :, 0] = [0.0, 0.7, 1.4, 2.1]
            action_abs[0, :, 1] = [0.0, 0.0, 0.0, 0.0]
            action_abs[0, :, 2] = [0.0, 0.0, 0.0, 0.0]
            action_abs[0, :, 3] = 1.0
            action_abs[0, :, 7] = 1.0
            action_abs[0, :, 9] = [0.0, 0.03, 0.09, 0.2]
            return {
                "action_abs": action_abs,
                "action_pred_relative": np.ones((1, 4, 10), dtype=np.float32),
            }

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    state = GazeWamRobotState(
        action_base_abs=np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.04], dtype=np.float32)
    )
    safety = GazeWamSafetyConfig(
        position_min=[-0.2, -0.25, -0.1],
        position_max=[0.75, 0.25, 0.1],
        gripper_min=0.01,
        gripper_max=0.08,
        max_position_step=0.3,
    )
    sink_calls = []
    runner = GazeWamDeploymentRunner(
        adapter=FakeAdapter(),
        image_provider=lambda: image,
        state_provider=lambda: state,
        gaze_provider=lambda: [0.25, 0.75],
        command_sink=lambda commands: sink_calls.append(commands),
        safety=safety,
        command_dt=0.2,
        command_start_delay=0.1,
        max_commands_per_step=3,
        dry_run=True,
        cfg_scale=1.25,
    )

    output = runner.step(now=10.0)

    assert output["dry_run"] is True
    assert sink_calls == []
    assert len(output["commands"]) == 3
    assert np.allclose([cmd.target_time for cmd in output["commands"]], [10.1, 10.3, 10.5])
    assert output["commands"][0].step_index == 0
    assert output["commands"][0].was_clipped is True
    assert np.isclose(output["commands"][0].action_abs[9], 0.01)
    assert np.isclose(output["commands"][1].action_abs[0], 0.3)
    assert np.isclose(output["commands"][2].action_abs[0], 0.6)
    assert output["commands"][2].action_abs[1] >= -0.25
    assert np.isclose(output["commands"][2].action_abs[9], 0.08)
    assert runner.adapter.calls[0]["cfg_scale"] == 1.25
    assert np.allclose(runner.adapter.calls[0]["gaze_xy"], [0.25, 0.75])
    assert "action_base_abs" in runner.adapter.calls[0]

    try:
        GazeWamDeploymentRunner(
            adapter=FakeAdapter(),
            image_provider=lambda: image,
            state_provider=lambda: state,
            cfg_scale=-0.1,
        )
    except ValueError as exc:
        assert "cfg_scale" in str(exc)
    else:
        raise AssertionError("Expected negative runner cfg_scale to fail.")


def test_gaze_wam_safety_config_validates_bounds_at_construction():
    valid = GazeWamSafetyConfig(
        position_min=np.asarray([-0.2, -0.1, 0.0], dtype=np.float64),
        position_max=[0.4, 0.5, 0.6],
        gripper_min=np.float32(0.01),
        gripper_max=0.08,
        max_position_step=np.float64(0.03),
    )
    assert valid.position_min == [-0.2, -0.1, 0.0]
    assert valid.position_max == [0.4, 0.5, 0.6]
    assert np.isclose(valid.gripper_min, 0.01)
    assert np.isclose(valid.gripper_max, 0.08)
    assert np.isclose(valid.max_position_step, 0.03)

    invalid_cases = [
        (
            {"position_min": [0.0, 0.1]},
            "position_min must contain exactly 3 finite floats",
        ),
        (
            {"position_max": [0.0, np.nan, 1.0]},
            "position_max[1] must be finite",
        ),
        (
            {"position_min": [0.0, 0.0, 2.0], "position_max": [1.0, 1.0, 1.0]},
            "position_min must be <= position_max elementwise",
        ),
        (
            {"gripper_min": np.inf},
            "gripper_min must be finite",
        ),
        (
            {"gripper_min": 0.08, "gripper_max": 0.01},
            "gripper_min must be <= gripper_max",
        ),
        (
            {"max_position_step": 0.0},
            "max_position_step must be positive",
        ),
        (
            {"max_position_step": np.nan},
            "max_position_step must be finite",
        ),
        (
            {"max_position_step": True},
            "max_position_step must be a finite float",
        ),
    ]
    for kwargs, expected in invalid_cases:
        try:
            GazeWamSafetyConfig(**kwargs)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"Expected invalid safety config {kwargs} to fail.")


def test_gaze_wam_deployment_runner_dispatches_commands_to_sink():
    class FakeAdapter:
        def predict_action(self, **kwargs):
            action_abs = np.zeros((1, 2, 10), dtype=np.float32)
            action_abs[0, :, 3] = 1.0
            action_abs[0, :, 7] = 1.0
            return {"action_abs": action_abs}

    class Sink:
        def __init__(self):
            self.commands = None

        def schedule_commands(self, commands):
            self.commands = commands

    sink = Sink()
    runner = GazeWamDeploymentRunner(
        adapter=FakeAdapter(),
        image_provider=lambda: np.zeros((8, 8, 3), dtype=np.uint8),
        state_provider=lambda: {"tcp_pose": np.zeros(6, dtype=np.float32), "gripper_width": 0.02},
        command_sink=sink,
        command_dt=0.05,
        command_start_delay=0.0,
        dry_run=False,
    )

    output = runner.step(now=3.0)

    assert sink.commands is output["commands"]
    assert len(sink.commands) == 2
    assert np.allclose([cmd.target_time for cmd in sink.commands], [3.0, 3.05])
    assert all(not cmd.was_clipped for cmd in sink.commands)


def test_gaze_wam_deployment_runner_rejects_bad_runtime_inputs():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    state = {"tcp_pose": np.zeros(6, dtype=np.float32), "gripper_width": 0.02}

    class StaticAdapter:
        def __init__(self, prediction):
            self.prediction = prediction

        def predict_action(self, **kwargs):
            return self.prediction

    valid_action = np.zeros((1, 2, 10), dtype=np.float32)
    valid_action[0, :, 3] = 1.0
    valid_action[0, :, 7] = 1.0

    bad_runner_kwargs = [
        ({"command_dt": np.inf}, "command_dt must be finite"),
        ({"command_dt": True}, "command_dt must be finite"),
        ({"command_start_delay": np.nan}, "command_start_delay must be finite"),
        ({"max_commands_per_step": 0}, "max_commands_per_step must be a positive integer"),
        ({"max_commands_per_step": 1.5}, "max_commands_per_step must be a positive integer"),
        ({"max_commands_per_step": True}, "max_commands_per_step must be a positive integer"),
    ]
    for kwargs, expected in bad_runner_kwargs:
        try:
            GazeWamDeploymentRunner(
                adapter=StaticAdapter({"action_abs": valid_action}),
                image_provider=lambda: image,
                state_provider=lambda: state,
                **kwargs,
            )
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"Expected bad runner kwargs {kwargs} to fail.")

    bad_predictions = [
        ({}, RuntimeError, "requires absolute action predictions"),
        ({"action_abs": np.zeros((1, 0, 10), dtype=np.float32)}, ValueError, "at least one timestep"),
        ({"action_abs": np.zeros((1, 2, 9), dtype=np.float32)}, ValueError, "at least 10"),
        (
            {"action_abs": np.full((1, 2, 10), np.nan, dtype=np.float32)},
            ValueError,
            "only finite values",
        ),
        (
            {
                "action_abs": valid_action,
                "action_pred_relative": np.zeros((1, 1, 10), dtype=np.float32),
            },
            ValueError,
            "action_pred_relative length must match action_abs length",
        ),
    ]
    for prediction, expected_exc, expected_message in bad_predictions:
        runner = GazeWamDeploymentRunner(
            adapter=StaticAdapter(prediction),
            image_provider=lambda: image,
            state_provider=lambda: state,
            dry_run=True,
        )
        try:
            runner.step(now=1.0)
        except expected_exc as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError(f"Expected bad prediction {prediction.keys()} to fail.")


def test_gaze_wam_deployment_runner_reports_timing_and_late_commands():
    class FakeAdapter:
        def predict_action(self, **kwargs):
            action_abs = np.zeros((1, 2, 10), dtype=np.float32)
            action_abs[0, :, 3] = 1.0
            action_abs[0, :, 7] = 1.0
            return {"action_abs": action_abs}

    clock_values = iter([0.0, 0.0, 0.15, 0.15, 0.15])
    runner = GazeWamDeploymentRunner(
        adapter=FakeAdapter(),
        image_provider=lambda: np.zeros((8, 8, 3), dtype=np.uint8),
        state_provider=lambda: {"tcp_pose": np.zeros(6, dtype=np.float32), "gripper_width": 0.02},
        command_dt=0.1,
        command_start_delay=0.05,
        dry_run=True,
        clock=lambda: next(clock_values),
    )

    output = runner.step()

    assert np.isclose(output["timing"]["prediction_latency"], 0.15)
    assert np.isclose(output["timing"]["command_available_time"], 0.15)
    assert np.allclose(output["timing"]["command_lead_times"], [-0.1, 0.0])
    assert np.isclose(output["timing"]["min_command_lead_time"], -0.1)
    assert output["timing"]["num_late_commands"] == 1


def test_gaze_wam_jsonl_command_sink_appends_queue_records():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_jsonl = Path(tmpdir) / "commands.jsonl"
        sink = GazeWamJsonlCommandSink(
            output_jsonl=str(output_jsonl),
            append=False,
        )
        commands = [
            GazeWamScheduledCommand(
                step_index=0,
                target_time=1.0,
                action_abs=np.ones(10, dtype=np.float32),
                raw_action_abs=np.zeros(10, dtype=np.float32),
                action_pred_relative=np.full(10, 0.5, dtype=np.float32),
                was_clipped=True,
            )
        ]

        sink.schedule_commands(commands)
        sink.schedule_commands(commands)

        rows = [
            json.loads(line)
            for line in output_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows) == 2
        assert rows[0]["batch_index"] == 0
        assert rows[0]["command_index_in_batch"] == 0
        assert rows[0]["global_command_index"] == 0
        assert rows[1]["batch_index"] == 1
        assert rows[1]["global_command_index"] == 1
        assert rows[0]["action_abs"] == [1.0] * 10
        assert rows[0]["action_pred_relative"] == [0.5] * 10
        assert sink.to_dict()["num_commands"] == 2
        assert sink.flush() == str(output_jsonl)


def test_gaze_wam_deployment_bindings_build_zarr_runner_and_recording_sink():
    class FakeAdapter:
        def __init__(self):
            self.calls = []

        def reset(self):
            self.calls.clear()

        def predict_action(self, **kwargs):
            self.calls.append(kwargs)
            action_abs = np.zeros((1, 3, 10), dtype=np.float32)
            action_abs[0, :, 0] = [0.0, 0.2, 0.4]
            action_abs[0, :, 3] = 1.0
            action_abs[0, :, 7] = 1.0
            action_abs[0, :, 9] = [0.02, 0.03, 0.04]
            return {"action_abs": action_abs}

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        robot_path = _write_linear_action_zarr(root / "robot.zarr", length=6)
        sink_json = root / "commands.json"
        adapter = FakeAdapter()
        runner = build_gaze_wam_deployment_runner_from_config(
            adapter=adapter,
            config={
                "source": {
                    "type": "zarr_replay",
                    "dataset_path": str(robot_path),
                    "start_offset": 1,
                    "max_steps": 2,
                    "missing_gaze": True,
                },
                "command_sink": {
                    "type": "recording",
                    "output_json": str(sink_json),
                },
                "safety": {
                    "gripper_min": 0.025,
                    "gripper_max": 0.035,
                },
                "command_dt": 0.2,
                "command_start_delay": 0.1,
                "max_commands_per_step": 2,
                "dry_run": False,
                "cfg_scale": 1.1,
            },
            clock=lambda: 5.0,
        )

        output = runner.step()

        assert output["dry_run"] is False
        assert len(output["commands"]) == 2
        assert len(adapter.calls) == 1
        assert adapter.calls[0]["gaze_xy"] is None
        assert adapter.calls[0]["cfg_scale"] == 1.1
        assert np.isclose(output["commands"][0].target_time, 5.1)
        assert np.isclose(output["commands"][0].action_abs[9], 0.025)
        assert np.isclose(output["commands"][1].action_abs[9], 0.03)
        assert isinstance(runner.command_sink_handle, GazeWamRecordingCommandSink)
        assert len(runner.command_sink_handle.commands) == 2
        assert runner.command_sink_handle.flush() == str(sink_json)
        payload = json.loads(sink_json.read_text(encoding="utf-8"))
        assert payload["num_batches"] == 1
        assert payload["num_commands"] == 2
        assert payload["batches"][0][0]["was_clipped"] is True


def test_gaze_wam_deployment_bindings_build_zarr_runner_and_jsonl_sink():
    class FakeAdapter:
        def predict_action(self, **kwargs):
            action_abs = np.zeros((1, 2, 10), dtype=np.float32)
            action_abs[0, :, 0] = [0.0, 0.1]
            action_abs[0, :, 3] = 1.0
            action_abs[0, :, 7] = 1.0
            action_abs[0, :, 9] = 0.04
            return {"action_abs": action_abs}

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        robot_path = _write_linear_action_zarr(root / "robot.zarr", length=6)
        sink_path = root / "command_queue.jsonl"
        runner = build_gaze_wam_deployment_runner_from_config(
            adapter=FakeAdapter(),
            config={
                "source": {
                    "type": "zarr_replay",
                    "dataset_path": str(robot_path),
                    "max_steps": 1,
                },
                "command_sink": {
                    "type": "jsonl",
                    "output_jsonl": str(sink_path),
                    "append": False,
                },
                "command_dt": 0.05,
                "command_start_delay": 0.0,
                "dry_run": False,
            },
            clock=lambda: 7.0,
        )

        output = runner.step()

        assert isinstance(runner.command_sink_handle, GazeWamJsonlCommandSink)
        assert len(output["commands"]) == 2
        rows = [
            json.loads(line)
            for line in sink_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows) == 2
        assert rows[0]["target_time"] == 7.0
        assert rows[1]["target_time"] == 7.05
        assert rows[0]["batch_index"] == 0
        assert rows[1]["global_command_index"] == 1


def test_gaze_wam_opencv_camera_provider_reads_rgb_frames_and_loops():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        video_path = root / "camera.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (4, 3),
        )
        frame0 = np.zeros((3, 4, 3), dtype=np.uint8)
        frame0[..., 0] = 10
        frame0[..., 1] = 20
        frame0[..., 2] = 240
        writer.write(frame0)
        writer.release()

        provider = build_gaze_wam_image_provider(
            {
                "type": "opencv_video",
                "source": str(video_path),
                "loop": True,
                "convert_bgr_to_rgb": True,
            }
        )
        try:
            assert isinstance(provider, GazeWamOpenCVCameraProvider)
            rgb0 = provider.get_image()
            rgb1 = provider.get_image()
        finally:
            provider.release()

        assert rgb0.ndim == 3 and rgb0.shape[-1] == 3
        assert rgb1.ndim == 3 and rgb1.shape[-1] == 3
        assert int(rgb0[..., 0].mean()) > int(rgb0[..., 2].mean())
        assert int(rgb0[..., 1].mean()) >= 10


def test_gaze_wam_jsonl_gaze_provider_normalizes_missing_and_eof():
    with tempfile.TemporaryDirectory() as tmpdir:
        gaze_path = Path(tmpdir) / "gaze.jsonl"
        rows = [
            {"gaze_x": 32, "gaze_y": 24, "image_width": 64, "image_height": 48},
            {"event": "blink"},
        ]
        gaze_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

        provider = build_gaze_wam_gaze_provider(
            {
                "type": "jsonl_replay",
                "path": str(gaze_path),
                "gaze_is_normalized": False,
                "missing_gaze": "hold_last",
                "eof": "hold_last",
            }
        )

        assert isinstance(provider, GazeWamJsonlGazeProvider)
        assert np.allclose(provider.get_gaze(), [0.5, 0.5])
        assert np.allclose(provider.get_gaze(), [0.5, 0.5])
        assert np.allclose(provider.get_gaze(), [0.5, 0.5])
        provider.reset()
        assert np.allclose(provider.read_gaze(), [0.5, 0.5])


def test_gaze_wam_jsonl_state_provider_reads_base_and_tcp_rows():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state.jsonl"
        rows = [
            {"state": {"base": [0, 1, 2, 1, 0, 0, 0, 1, 0, 0.03]}},
            {"tcp_pose_6d": [0.1, 0.2, 0.3, 0, 0, 0], "gripper_width": 0.04},
            {"state": {"base": [0, 1, 2, 1, 0, 0, 0, 1, 0]}, "gripper_width": 0.05},
        ]
        state_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

        provider = GazeWamJsonlStateProvider(
            path=str(state_path),
            action_base_abs_key="state.base",
            eof="hold_last",
        )

        state0 = provider.get_state()
        state1 = provider.get_state()
        state2 = provider.get_state()
        state3 = provider.get_state()

        assert np.allclose(np.asarray(state0.action_base_abs), rows[0]["state"]["base"])
        assert state0.tcp_pose is None
        assert np.allclose(np.asarray(state1.tcp_pose), rows[1]["tcp_pose_6d"])
        assert np.isclose(state1.gripper_width, 0.04)
        assert np.allclose(np.asarray(state2.action_base_abs), rows[2]["state"]["base"])
        assert np.isclose(state2.gripper_width, 0.05)
        assert state3 is state2
        provider.reset()
        assert np.allclose(np.asarray(provider.read_state().action_base_abs), rows[0]["state"]["base"])


def test_gaze_wam_jsonl_state_provider_requires_gripper_for_direct_pose9_base():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state_missing_gripper.jsonl"
        state_path.write_text(
            json.dumps({"action_base_abs": [0, 1, 2, 1, 0, 0, 0, 1, 0]}) + "\n",
            encoding="utf-8",
        )
        provider = GazeWamJsonlStateProvider(path=str(state_path))
        try:
            provider.get_state()
        except ValueError as exc:
            assert "9D action_base_abs requires gripper_width" in str(exc)
        else:
            raise AssertionError("Expected 9D JSONL action_base_abs without gripper_width to fail.")

        state_path.write_text(
            json.dumps(
                {
                    "action_base_abs": [0, 1, 2, 1, 0, 0, 0, 1, 0],
                    "gripper_width": [0.01, 0.02],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        provider = GazeWamJsonlStateProvider(path=str(state_path))
        try:
            provider.get_state()
        except ValueError as exc:
            assert "gripper_width must contain exactly one scalar" in str(exc)
        else:
            raise AssertionError("Expected multi-value JSONL gripper_width to fail.")


def test_gaze_wam_static_state_provider_normalizes_direct_action_base():
    provider = GazeWamStaticStateProvider(
        action_base_abs=[0, 1, 2, 1, 0, 0, 0, 1, 0],
        gripper_width=0.05,
    )
    state = provider.get_state()
    assert np.asarray(state.action_base_abs).shape == (10,)
    assert np.isclose(np.asarray(state.action_base_abs)[9], 0.05)

    try:
        GazeWamStaticStateProvider(action_base_abs=[0, 1, 2, 1, 0, 0, 0, 1, 0])
    except ValueError as exc:
        assert "9D action_base_abs requires gripper_width" in str(exc)
    else:
        raise AssertionError("Expected 9D static action_base_abs without gripper_width to fail.")

    try:
        GazeWamStaticStateProvider(
            action_base_abs=[0, 1, 2, 1, 0, 0, 0, 1, 0],
            gripper_width=[0.01, 0.02],
        )
    except ValueError as exc:
        assert "gripper_width must contain exactly one scalar" in str(exc)
    else:
        raise AssertionError("Expected multi-value static gripper_width to fail.")


def test_gaze_wam_static_state_provider_validates_tcp_pose_width():
    provider = GazeWamStaticStateProvider(
        tcp_pose=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        gripper_width=0.04,
    )
    state = provider.get_state()
    assert np.asarray(state.tcp_pose).shape == (6,)
    assert np.isclose(state.gripper_width, 0.04)

    try:
        GazeWamStaticStateProvider(tcp_pose=[0.0, 1.0, 2.0, 3.0])
    except ValueError as exc:
        assert "tcp_pose must contain 6, 9, or 10 values" in str(exc)
    else:
        raise AssertionError("Expected invalid static tcp_pose width to fail.")

    try:
        GazeWamStaticStateProvider(tcp_pose=[0.0, 1.0, 2.0, np.nan, 0.0, 0.0])
    except ValueError as exc:
        assert "tcp_pose must contain only finite values" in str(exc)
    else:
        raise AssertionError("Expected non-finite static tcp_pose to fail.")


def test_gaze_wam_deployment_bindings_build_split_providers_runner():
    class FakeAdapter:
        def __init__(self):
            self.calls = []

        def predict_action(self, **kwargs):
            self.calls.append(kwargs)
            action_abs = np.zeros((1, 1, 10), dtype=np.float32)
            action_abs[0, 0, 3] = 1.0
            action_abs[0, 0, 7] = 1.0
            action_abs[0, 0, 9] = 0.04
            return {"action_abs": action_abs}

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        video_path = root / "camera.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (4, 3),
        )
        frame = np.zeros((3, 4, 3), dtype=np.uint8)
        frame[..., 1] = 128
        writer.write(frame)
        writer.release()
        gaze_path = root / "gaze.jsonl"
        gaze_path.write_text(
            json.dumps({"gaze_xy": [0.25, 0.75]}) + "\n",
            encoding="utf-8",
        )
        state_path = root / "state.jsonl"
        state_path.write_text(
            json.dumps({"state": {"base": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0.03]}}) + "\n",
            encoding="utf-8",
        )

        adapter = FakeAdapter()
        runner = build_gaze_wam_deployment_runner_from_config(
            adapter=adapter,
            config={
                "image_provider": {
                    "type": "opencv_video",
                    "source": str(video_path),
                    "convert_bgr_to_rgb": True,
                },
                "state_provider": {
                    "type": "jsonl_replay",
                    "path": str(state_path),
                    "action_base_abs_key": "state.base",
                    "eof": "hold_last",
                },
                "gaze_provider": {
                    "type": "jsonl_replay",
                    "path": str(gaze_path),
                    "eof": "hold_last",
                },
                "command_sink": {"type": "recording"},
                "dry_run": False,
                "command_start_delay": 0.0,
                "command_dt": 0.1,
            },
            clock=lambda: 2.0,
        )
        try:
            output = runner.step()
        finally:
            for handle in getattr(runner, "provider_handles", []):
                if hasattr(handle, "close"):
                    handle.close()

        assert output["dry_run"] is False
        assert len(output["commands"]) == 1
        assert np.isclose(output["commands"][0].target_time, 2.0)
        assert adapter.calls[0]["image"].ndim == 3
        assert adapter.calls[0]["image"].shape[-1] == 3
        assert np.allclose(adapter.calls[0]["gaze_xy"], [0.25, 0.75])
        assert np.asarray(adapter.calls[0]["action_base_abs"]).shape == (10,)
        assert isinstance(runner.command_sink_handle, GazeWamRecordingCommandSink)
        assert len(runner.command_sink_handle.commands) == 1


def test_make_gaze_wam_split_deployment_config_builds_runner_config():
    class FakeAdapter:
        def __init__(self):
            self.calls = []

        def predict_action(self, **kwargs):
            self.calls.append(kwargs)
            action_abs = np.zeros((1, 1, 10), dtype=np.float32)
            action_abs[0, 0, 3] = 1.0
            action_abs[0, 0, 7] = 1.0
            action_abs[0, 0, 9] = 0.04
            return {"action_abs": action_abs}

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        video_path = root / "camera.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (4, 3),
        )
        writer.write(np.zeros((3, 4, 3), dtype=np.uint8))
        writer.release()
        state_path = root / "state.jsonl"
        state_path.write_text(
            json.dumps({"action_base_abs": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0.03]}) + "\n",
            encoding="utf-8",
        )
        gaze_path = root / "gaze.jsonl"
        gaze_path.write_text(json.dumps({"gaze_xy": [0.4, 0.6]}) + "\n", encoding="utf-8")
        command_jsonl = root / "commands.jsonl"

        config = build_gaze_wam_split_deployment_config(
            image_source=str(video_path),
            state_path=str(state_path),
            gaze_path=str(gaze_path),
            command_output_jsonl=str(command_jsonl),
            max_commands_per_step=1,
            dry_run=False,
        )
        runner = build_gaze_wam_deployment_runner_from_config(
            adapter=FakeAdapter(),
            config=config,
            clock=lambda: 3.0,
        )
        try:
            output = runner.step()
        finally:
            for handle in getattr(runner, "provider_handles", []):
                if hasattr(handle, "close"):
                    handle.close()

        assert config["image_provider"]["type"] == "opencv_video"
        assert config["state_provider"]["eof"] == "hold_last"
        assert config["gaze_provider"]["path"] == str(gaze_path)
        assert config["command_sink"]["append"] is False
        assert output["dry_run"] is False
        assert command_jsonl.exists()


def test_make_gaze_wam_split_deployment_config_cli_writes_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        output_json = root / "split_config.json"
        command_jsonl = root / "commands.jsonl"
        old_argv = sys.argv
        try:
            sys.argv = [
                "make_gaze_wam_split_deployment_config.py",
                "--image-source",
                "camera.mp4",
                "--state-provider-type",
                "static",
                "--static-action-base-abs",
                "[0,0,0,1,0,0,0,1,0,0.03]",
                "--gaze-provider-type",
                "none",
                "--command-output-jsonl",
                str(command_jsonl),
                "--max-commands-per-step",
                "2",
                "--output-json",
                str(output_json),
            ]
            make_gaze_wam_split_deployment_config_main()
        finally:
            sys.argv = old_argv

        config = json.loads(output_json.read_text(encoding="utf-8"))
        assert config["state_provider"]["type"] == "static"
        assert config["state_provider"]["action_base_abs"][-1] == 0.03
        assert config["gaze_provider"]["type"] == "none"
        assert config["command_sink"]["output_jsonl"] == str(command_jsonl)
        assert config["max_commands_per_step"] == 2


def test_make_gaze_wam_split_deployment_config_validates_runtime_numeric_fields():
    valid_config = build_gaze_wam_split_deployment_config(
        image_source="camera.mp4",
        state_provider_type="static",
        static_action_base_abs=[0, 1, 2, 1, 0, 0, 0, 1, 0, 0.03],
        gaze_provider_type="none",
        command_output_jsonl="commands.jsonl",
        command_dt=np.float64(0.125),
        command_start_delay=np.float64(0.0),
        max_commands_per_step=np.int64(2),
        cfg_scale=np.float64(1.5),
    )
    assert valid_config["command_dt"] == 0.125
    assert valid_config["command_start_delay"] == 0.0
    assert valid_config["max_commands_per_step"] == 2
    assert valid_config["cfg_scale"] == 1.5

    invalid_cases = [
        ({"command_dt": 0.0}, "command_dt must be positive"),
        ({"command_dt": -0.1}, "command_dt must be positive"),
        ({"command_dt": np.inf}, "command_dt must be finite"),
        ({"command_start_delay": -0.1}, "command_start_delay must be non-negative"),
        ({"command_start_delay": np.nan}, "command_start_delay must be finite"),
        ({"max_commands_per_step": 0}, "max_commands_per_step must be a positive integer"),
        ({"max_commands_per_step": -1}, "max_commands_per_step must be a positive integer"),
        ({"max_commands_per_step": 1.5}, "max_commands_per_step must be a positive integer"),
        ({"max_commands_per_step": True}, "max_commands_per_step must be a positive integer"),
        ({"cfg_scale": -0.1}, "cfg_scale must be non-negative"),
        ({"cfg_scale": np.inf}, "cfg_scale must be finite"),
    ]
    for overrides, expected_message in invalid_cases:
        try:
            build_gaze_wam_split_deployment_config(
                image_source="camera.mp4",
                state_provider_type="static",
                static_action_base_abs=[0, 1, 2, 1, 0, 0, 0, 1, 0, 0.03],
                gaze_provider_type="none",
                command_output_jsonl="commands.jsonl",
                **overrides,
            )
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError(f"Expected invalid split deployment override {overrides} to fail.")


def test_make_gaze_wam_split_deployment_config_validates_static_state_contract():
    try:
        build_gaze_wam_split_deployment_config(
            image_source="camera.mp4",
            state_provider_type="static",
            static_action_base_abs=[0, 1, 2, 1, 0, 0, 0, 1, 0],
            gaze_provider_type="none",
            command_output_jsonl="commands.jsonl",
        )
    except ValueError as exc:
        assert "9D static_action_base_abs requires static_gripper_width" in str(exc)
    else:
        raise AssertionError("Expected 9D static action base without gripper to fail.")

    try:
        build_gaze_wam_split_deployment_config(
            image_source="camera.mp4",
            state_provider_type="static",
            static_action_base_abs=[0, 1, 2, 1, 0, 0, 0, 1],
            static_gripper_width=0.02,
            gaze_provider_type="none",
            command_output_jsonl="commands.jsonl",
        )
    except ValueError as exc:
        assert "static_action_base_abs must contain 9 pose-only or 10 pose+gripper values" in str(exc)
    else:
        raise AssertionError("Expected invalid static action base width to fail.")

    try:
        build_gaze_wam_split_deployment_config(
            image_source="camera.mp4",
            state_provider_type="static",
            static_tcp_pose=[0.0, 1.0, 2.0, 3.0],
            gaze_provider_type="none",
            command_output_jsonl="commands.jsonl",
        )
    except ValueError as exc:
        assert "static_tcp_pose must contain 6, 9, or 10 values" in str(exc)
    else:
        raise AssertionError("Expected invalid static TCP pose width to fail.")

    try:
        build_gaze_wam_split_deployment_config(
            image_source="camera.mp4",
            state_provider_type="static",
            static_tcp_pose=[0.0, 1.0, 2.0, np.nan, 0.0, 0.0],
            gaze_provider_type="none",
            command_output_jsonl="commands.jsonl",
        )
    except ValueError as exc:
        assert "static_tcp_pose must contain only finite values" in str(exc)
    else:
        raise AssertionError("Expected non-finite static TCP pose to fail.")

    config = build_gaze_wam_split_deployment_config(
        image_source="camera.mp4",
        state_provider_type="static",
        static_action_base_abs=[0, 1, 2, 1, 0, 0, 0, 1, 0],
        static_gripper_width=[0.02],
        gaze_provider_type="none",
        command_output_jsonl="commands.jsonl",
    )
    assert config["state_provider"]["gripper_width"] == 0.02


def test_gaze_wam_split_deployment_rehearsal_dispatches_to_jsonl_sink():
    class FakeAdapter:
        def __init__(self):
            self.calls = []

        def predict_action(self, **kwargs):
            self.calls.append(kwargs)
            action_abs = np.zeros((1, 2, 10), dtype=np.float32)
            action_abs[0, :, 3] = 1.0
            action_abs[0, :, 7] = 1.0
            action_abs[0, :, 9] = 0.04
            return {"action_abs": action_abs}

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        video_path = root / "camera.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (4, 3),
        )
        frame = np.zeros((3, 4, 3), dtype=np.uint8)
        frame[..., 0] = 16
        frame[..., 2] = 220
        writer.write(frame)
        writer.release()
        gaze_path = root / "gaze.jsonl"
        gaze_path.write_text(json.dumps({"gaze_xy": [0.2, 0.8]}) + "\n", encoding="utf-8")
        state_path = root / "state.jsonl"
        state_path.write_text(
            json.dumps({"action_base_abs": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0.03]}) + "\n",
            encoding="utf-8",
        )
        command_jsonl = root / "commands.jsonl"
        summary_json = root / "summary.json"

        summary = run_gaze_wam_split_deployment_rehearsal(
            adapter=FakeAdapter(),
            deployment_config={
                "image_provider": {
                    "type": "opencv_video",
                    "source": str(video_path),
                    "loop": True,
                },
                "state_provider": {
                    "type": "jsonl_replay",
                    "path": str(state_path),
                    "eof": "hold_last",
                },
                "gaze_provider": {
                    "type": "jsonl_replay",
                    "path": str(gaze_path),
                    "eof": "hold_last",
                },
                "command_sink": {
                    "type": "jsonl",
                    "output_jsonl": str(command_jsonl),
                    "append": False,
                },
                "command_start_delay": 0.0,
                "command_dt": 0.1,
            },
            output_json=str(summary_json),
            max_steps=2,
            start_time=4.0,
            step_dt=0.5,
            dispatch=True,
        )

        assert summary["dispatch"] is True
        assert summary["num_steps"] == 2
        assert summary["num_commands"] == 4
        assert summary_json.exists()
        rows = [
            json.loads(line)
            for line in command_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows) == 4
        assert rows[0]["target_time"] == 4.0
        assert rows[2]["target_time"] == 4.5
        assert rows[-1]["global_command_index"] == 3


def test_gaze_wam_config_split_deployment_rehearsal_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data_dir = root / "debug_data"
        generate_gaze_wam_debug_data(
            output_dir=str(data_dir),
            num_episodes=1,
            episode_length=18,
            image_size=256,
            seed=7,
        )
        video_path = root / "camera.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (16, 16),
        )
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
        writer.release()
        state_path = root / "state.jsonl"
        state_path.write_text(
            json.dumps({"action_base_abs": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0.03]}) + "\n",
            encoding="utf-8",
        )
        command_jsonl = root / "commands.jsonl"
        summary_json = root / "summary.json"

        summary = run_gaze_wam_config_split_deployment_rehearsal(
            config_name="train_gaze_wam_debug_workspace",
            robot_dataset_path=str(data_dir / "robot.zarr"),
            deployment_config={
                "image_provider": {
                    "type": "opencv_video",
                    "source": str(video_path),
                    "loop": True,
                },
                "state_provider": {
                    "type": "jsonl_replay",
                    "path": str(state_path),
                    "eof": "hold_last",
                },
                "gaze_provider": {"type": "none"},
                "command_sink": {
                    "type": "jsonl",
                    "output_jsonl": str(command_jsonl),
                    "append": False,
                },
                "command_start_delay": 0.0,
                "command_dt": 0.1,
                "max_commands_per_step": 1,
            },
            output_json=str(summary_json),
            device="cpu",
            overrides=[
                f"task.robot_dataset_path={data_dir / 'robot.zarr'}",
                f"task.open_dataset_path={data_dir / 'open.zarr'}",
                "policy.obs_encoder.pretrained=false",
            ],
            num_inference_steps=2,
            max_steps=1,
            dispatch=True,
        )

        assert summary["policy_source"] == "config"
        assert summary["num_steps"] == 1
        assert summary["num_commands"] == 1
        assert command_jsonl.exists()
        assert json.loads(summary_json.read_text(encoding="utf-8"))["config_name"] == "train_gaze_wam_debug_workspace"


def test_gaze_wam_zarr_replay_source_reads_provider_streams():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_linear_action_zarr(Path(tmpdir) / "robot.zarr", length=6)
        source = GazeWamZarrReplaySource(
            dataset_path=str(robot_path),
            start_offset=1,
            max_steps=2,
            stride=2,
        )

        assert len(source) == 2
        assert source.current_sample().source_index == 1
        assert source.current_sample().episode_offset == 1
        assert source.get_image().shape == (16, 16, 3)
        assert np.allclose(source.get_gaze(), [0.5, 0.5])
        state = source.get_state()
        assert np.asarray(state.action_base_abs).shape == (10,)
        assert np.isclose(np.asarray(state.action_base_abs)[9], 0.01)
        assert np.isclose(source.replay_now(base_time=2.0, replay_dt=0.25), 2.0)

        source.advance()
        assert source.current_sample().source_index == 3
        assert np.isclose(source.replay_now(base_time=2.0, replay_dt=0.25), 2.25)


def test_gaze_wam_zarr_replay_direct_pose9_base_requires_gripper():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_linear_action_zarr(Path(tmpdir) / "robot.zarr", length=6)
        data = zarr.open(str(robot_path), mode="a")["data"]
        direct_pose9 = np.asarray(data["tcp_pose_abs"][:], dtype=np.float32)
        data.array("action_base_pose9", direct_pose9, shape=direct_pose9.shape, dtype=direct_pose9.dtype)

        source = GazeWamZarrReplaySource(
            dataset_path=str(robot_path),
            action_base_abs_key="action_base_pose9",
            start_offset=1,
            max_steps=1,
        )
        state = source.get_state()
        assert np.asarray(state.action_base_abs).shape == (10,)
        assert np.isclose(np.asarray(state.action_base_abs)[9], 0.01)

        del data["gripper_width"]
        missing_gripper_source = GazeWamZarrReplaySource(
            dataset_path=str(robot_path),
            action_base_abs_key="action_base_pose9",
            start_offset=1,
            max_steps=1,
        )
        try:
            missing_gripper_source.get_state()
        except KeyError as exc:
            assert "gripper_width" in str(exc)
        else:
            raise AssertionError("Expected 9D direct action_base_abs replay to require gripper_width.")


def test_gaze_wam_zarr_replay_rejects_multi_column_gripper():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_linear_action_zarr(Path(tmpdir) / "robot_bad_gripper.zarr", length=6)
        data = zarr.open(str(robot_path), mode="a")["data"]
        gripper = np.asarray(data["gripper_width"][:], dtype=np.float32)
        del data["gripper_width"]
        bad_gripper = np.concatenate([gripper.reshape(-1, 1), gripper.reshape(-1, 1)], axis=-1)
        data.array("gripper_width", bad_gripper, shape=bad_gripper.shape, dtype=bad_gripper.dtype)

        source = GazeWamZarrReplaySource(
            dataset_path=str(robot_path),
            start_offset=1,
            max_steps=1,
        )
        try:
            source.get_state()
        except ValueError as exc:
            assert "must provide exactly one scalar per replay step" in str(exc)
        else:
            raise AssertionError("Expected zarr replay with multi-column gripper_width to fail.")


def test_gaze_wam_zarr_replay_allows_null_gaze_but_training_validation_rejects_it():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_robot_heatmap_only_zarr(Path(tmpdir) / "robot_heatmap.zarr")

        source = GazeWamZarrReplaySource(
            dataset_path=str(robot_path),
            gaze_key=None,
            max_steps=1,
        )
        assert source.get_gaze() is None

        with pytest.raises(ValueError, match="normalized point gaze key"):
            _validate_rehearsal_robot_zarr(
                dataset_path=str(robot_path),
                gaze_key=None,
                heatmap_key="gaze_heatmap",
                n_obs_steps=2,
                action_horizon=3,
                image_size=(16, 16),
                heatmap_token_grid=(4, 4),
            )


def test_gaze_wam_zarr_deployment_rehearsal_records_json_and_missing_gaze():
    class FakeAdapter:
        def __init__(self):
            self.calls = []

        def predict_action(self, **kwargs):
            self.calls.append(kwargs)
            action_abs = np.zeros((1, 3, 10), dtype=np.float32)
            action_abs[0, :, 0] = [0.0, 0.2, 0.4]
            action_abs[0, :, 3] = 1.0
            action_abs[0, :, 7] = 1.0
            action_abs[0, :, 9] = [0.01, 0.02, 0.03]
            action_pred_relative = np.full((1, 3, 10), 0.5, dtype=np.float32)
            return {
                "action_abs": action_abs,
                "action_pred_relative": action_pred_relative,
            }

    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_linear_action_zarr(Path(tmpdir) / "robot.zarr", length=6)
        output_json = Path(tmpdir) / "rehearsal.json"
        adapter = FakeAdapter()

        summary = run_gaze_wam_zarr_deployment_rehearsal(
            adapter=adapter,
            dataset_path=str(robot_path),
            output_json=str(output_json),
            max_steps=2,
            missing_gaze=True,
            command_dt=0.2,
            command_start_delay=0.05,
            max_commands_per_step=2,
            replay_base_time=4.0,
            replay_dt=0.5,
        )

        assert output_json.exists()
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert summary["num_steps"] == 2
        assert payload["num_commands"] == 4
        assert payload["missing_gaze"] is True
        assert payload["replay_config"] == {
            "dataset_path": str(robot_path),
            "camera_key": "camera0_rgb",
            "tcp_pose_key": "tcp_pose_abs",
            "gripper_key": "gripper_width",
            "gaze_key": "gaze_xy",
            "heatmap_key": "gaze_heatmap",
            "action_base_abs_key": None,
            "timestamp_key": None,
            "episode_index": 0,
            "start_offset": 0,
            "max_steps": 2,
            "stride": 1,
            "missing_gaze": True,
            "command_dt": 0.2,
            "command_start_delay": 0.05,
            "max_commands_per_step": 2,
            "dry_run": True,
            "cfg_scale": None,
            "replay_base_time": 4.0,
            "replay_dt": 0.5,
            "include_prediction_summary": True,
            "fail_on_late_commands": False,
        }
        assert len(adapter.calls) == 2
        assert adapter.calls[0]["gaze_xy"] is None
        assert "action_base_abs" in adapter.calls[0]
        assert payload["records"][0]["source_index"] == 0
        assert payload["records"][0]["now"] == 4.0
        assert payload["records"][0]["commands"][0]["target_time"] == 4.05
        assert payload["records"][0]["commands"][1]["target_time"] == 4.25
        assert payload["records"][0]["commands"][0]["action_pred_relative"][0] == 0.5
        assert payload["records"][0]["prediction_summary"]["action_abs"]["shape"] == [1, 3, 10]
        assert payload["records"][0]["timing"]["num_late_commands"] == 0
        assert payload["timing_summary"]["num_late_commands"] == 0
        assert payload["timing_summary"]["min_command_lead_time"] >= 0.0
        assert payload["records"][1]["now"] == 4.5


def test_gaze_wam_zarr_deployment_rehearsal_can_fail_on_late_commands():
    class FakeAdapter:
        def predict_action(self, **kwargs):
            action_abs = np.zeros((1, 2, 10), dtype=np.float32)
            action_abs[0, :, 3] = 1.0
            action_abs[0, :, 7] = 1.0
            return {"action_abs": action_abs}

    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_linear_action_zarr(Path(tmpdir) / "robot.zarr", length=6)
        clock_values = iter([0.0, 0.0, 0.2, 0.2, 0.2])

        try:
            run_gaze_wam_zarr_deployment_rehearsal(
                adapter=FakeAdapter(),
                dataset_path=str(robot_path),
                max_steps=1,
                command_dt=0.05,
                command_start_delay=0.0,
                max_commands_per_step=2,
                fail_on_late_commands=True,
                clock=lambda: next(clock_values),
            )
        except RuntimeError as exc:
            assert "late scheduled commands" in str(exc)
        else:
            raise AssertionError("Expected late scheduled commands to fail rehearsal.")


def test_gaze_wam_zarr_deployment_rehearsal_applies_safety_config():
    class FakeAdapter:
        def predict_action(self, **kwargs):
            action_abs = np.zeros((1, 3, 10), dtype=np.float32)
            action_abs[0, :, 0] = [0.0, 0.2, 0.4]
            action_abs[0, :, 3] = 1.0
            action_abs[0, :, 7] = 1.0
            action_abs[0, :, 9] = [0.01, 0.05, 0.09]
            return {
                "action_abs": action_abs,
                "action_pred_relative": np.zeros((1, 3, 10), dtype=np.float32),
            }

    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_linear_action_zarr(Path(tmpdir) / "robot.zarr", length=6)
        output_json = Path(tmpdir) / "safe_rehearsal.json"
        safety_path = Path(tmpdir) / "safety.json"
        safety_path.write_text(
            json.dumps(
                {
                    "position_min": [-0.05, -1.0, -1.0],
                    "position_max": [0.15, 1.0, 1.0],
                    "gripper_min": 0.0,
                    "gripper_max": 0.04,
                    "max_position_step": 0.1,
                }
            ),
            encoding="utf-8",
        )
        safety = safety_config_from_json(str(safety_path))

        summary = run_gaze_wam_zarr_deployment_rehearsal(
            adapter=FakeAdapter(),
            dataset_path=str(robot_path),
            output_json=str(output_json),
            max_steps=1,
            max_commands_per_step=3,
            command_dt=0.1,
            command_start_delay=0.0,
            safety=safety,
        )

        payload = json.loads(output_json.read_text(encoding="utf-8"))
        commands = payload["records"][0]["commands"]
        assert summary["num_clipped_commands"] == 2
        assert payload["num_clipped_commands"] == 2
        assert payload["safety"]["position_max"] == [0.15, 1.0, 1.0]
        assert payload["safety"]["gripper_max"] == 0.04
        assert commands[0]["was_clipped"] is False
        assert commands[1]["was_clipped"] is True
        assert commands[2]["was_clipped"] is True
        assert commands[1]["action_abs"][0] <= 0.100001
        assert commands[2]["action_abs"][0] <= 0.150001
        assert np.isclose(commands[2]["action_abs"][9], 0.04)

        bad_safety_path = Path(tmpdir) / "bad_safety.json"
        bad_safety_path.write_text(json.dumps({"unknown": 1}), encoding="utf-8")
        try:
            safety_config_from_json(str(bad_safety_path))
        except ValueError as exc:
            assert "Unknown safety JSON keys" in str(exc)
        else:
            raise AssertionError("Expected unknown safety key to fail.")

        bad_value_safety_path = Path(tmpdir) / "bad_value_safety.json"
        bad_value_safety_path.write_text(
            json.dumps({"position_min": [0.2, 0.0, 0.0], "position_max": [0.1, 1.0, 1.0]}),
            encoding="utf-8",
        )
        try:
            safety_config_from_json(str(bad_value_safety_path))
        except ValueError as exc:
            assert "position_min must be <= position_max elementwise" in str(exc)
        else:
            raise AssertionError("Expected invalid safety JSON values to fail.")


def test_gaze_wam_config_zarr_deployment_rehearsal_without_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_gaze_wam_zarr(
            Path(tmpdir) / "robot.zarr",
            include_action=True,
            image_hw=(16, 16),
        )
        output_json = Path(tmpdir) / "config_rehearsal.json"

        summary = run_gaze_wam_config_zarr_deployment_rehearsal(
            config_name="train_gaze_wam_debug_workspace",
            dataset_path=str(robot_path),
            output_json=str(output_json),
            device="cpu",
            max_steps=1,
            max_commands_per_step=1,
            command_dt=0.1,
            command_start_delay=0.0,
            include_prediction_summary=True,
            overrides=[
                "policy.obs_encoder.pretrained=false",
                "policy.num_inference_steps=2",
            ],
        )

        payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert summary["policy_source"] == "config"
        assert summary["checkpoint_path"] is None
        assert payload["policy_source"] == "config"
        assert payload["config_name"] == "train_gaze_wam_debug_workspace"
        assert payload["num_steps"] == 1
        assert payload["num_commands"] == 1
        assert payload["zarr_validation"]["valid"] is True
        assert payload["zarr_validation"]["dataset_type"] == "robot"
        assert payload["zarr_validation"]["sample"]["action_roundtrip_max_error"] < 1e-6
        assert payload["records"][0]["commands"][0]["action_abs"]
        assert payload["records"][0]["prediction_summary"]["action_abs"]["shape"] == [1, 16, 10]


def test_gaze_wam_config_zarr_rehearsal_validates_timestamps_before_running():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_linear_action_zarr(Path(tmpdir) / "robot.zarr", length=8)
        _add_timestamp_arrays(robot_path)

        try:
            run_gaze_wam_config_zarr_deployment_rehearsal(
                config_name="train_gaze_wam_robot_only_debug_workspace",
                dataset_path=str(robot_path),
                output_json=str(Path(tmpdir) / "rehearsal.json"),
                device="cpu",
                max_steps=1,
                max_commands_per_step=1,
                validate_zarr=True,
                require_timestamps=True,
                timestamp_max_step=0.01,
                num_inference_steps=2,
                overrides=[
                    "policy.obs_encoder.pretrained=false",
                ],
            )
        except ValueError as exc:
            assert "Rehearsal robot zarr validation failed" in str(exc)
            assert "max_step" in str(exc)
        else:
            raise AssertionError("Expected timestamp validation to block zarr rehearsal.")


def test_gaze_wam_config_split_rehearsal_validates_normalizer_zarr_timestamps():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        robot_path = _write_linear_action_zarr(root / "robot.zarr", length=8)
        _add_timestamp_arrays(robot_path)
        video_path = root / "camera.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (4, 3),
        )
        writer.write(np.zeros((3, 4, 3), dtype=np.uint8))
        writer.release()
        state_path = root / "state.jsonl"
        state_path.write_text(
            json.dumps({"action_base_abs": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0.03]}) + "\n",
            encoding="utf-8",
        )

        try:
            run_gaze_wam_config_split_deployment_rehearsal(
                config_name="train_gaze_wam_robot_only_debug_workspace",
                robot_dataset_path=str(robot_path),
                deployment_config={
                    "image_provider": {
                        "type": "opencv_video",
                        "source": str(video_path),
                        "loop": True,
                    },
                    "state_provider": {
                        "type": "jsonl_replay",
                        "path": str(state_path),
                        "eof": "hold_last",
                    },
                    "gaze_provider": {"type": "none"},
                    "command_sink": {"type": "recording"},
                },
                output_json=str(root / "split_rehearsal.json"),
                device="cpu",
                overrides=[
                    "policy.obs_encoder.pretrained=false",
                ],
                num_inference_steps=2,
                max_steps=1,
                validate_zarr=True,
                require_timestamps=True,
                timestamp_max_step=0.01,
            )
        except ValueError as exc:
            assert "Rehearsal robot zarr validation failed" in str(exc)
            assert "max_step" in str(exc)
        else:
            raise AssertionError("Expected timestamp validation to block split rehearsal.")


def test_gaze_wam_policy_gaze_dependency_ratio(tmp_path):
    torch.manual_seed(4)
    shape_meta = {
        "action": {
            "shape": [10],
            "horizon": 4,
        }
    }
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    encoder_path, decoder_path = _write_fake_cosmos_jit_pair(
        tmp_path,
        image_size=(16, 16),
        token_grid=(2, 4),
        latent_channels=1,
    )
    policy = GazeWamPolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        obs_encoder=FakeTokenObsEncoder(num_tokens=8, embed_dim=32),
        model=JointGazeWamTransformer(
            action_dim=10,
            heatmap_dim=1,
            action_horizon=4,
            heatmap_num_tokens=8,
            max_image_tokens=8,
            n_layer=1,
            n_head=4,
            n_emb=32,
            p_drop_emb=0.0,
            p_drop_attn=0.0,
        ),
        gaze_encoder=GazeConditionEncoder(embed_dim=32),
        num_inference_steps=2,
        input_pertub=0.0,
        heatmap_num_tokens=8,
        heatmap_token_grid=(2, 4),
        heatmap_image_size=(16, 16),
        heatmap_cosmos_encoder_path=encoder_path,
        heatmap_cosmos_decoder_path=decoder_path,
        n_emb=32,
    )
    normalizer = LinearNormalizer()
    normalizer["camera0_rgb"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)

    result = policy.compute_gaze_dependency_ratio(
        {
            "camera0_rgb": torch.randn(3, 2, 3, 16, 16),
            "gaze_xy": torch.tensor(
                [
                    [0.5, 0.5],
                    [0.2, 0.8],
                    [0.7, 0.3],
                ],
                dtype=torch.float32,
            ),
            "has_gaze_label": torch.tensor([True, True, True]),
        },
        noisy_action=torch.zeros(3, 4, 10),
        timestep=torch.zeros(3, dtype=torch.long),
    )

    assert result["feature_gdr"].shape == (3,)
    assert result["output_gdr"].shape == (3,)
    assert torch.all(result["feature_gdr"] >= 0)
    assert torch.all(result["output_gdr"] >= 0)
    assert result["feature_gdr_mean"].ndim == 0
    assert result["output_gdr_mean"].ndim == 0

    try:
        policy.compute_gaze_dependency_ratio(
            {
                "camera0_rgb": torch.randn(2, 2, 3, 16, 16),
                "gaze_xy": torch.tensor(
                    [
                        [0.5, 0.5],
                        [0.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
                "has_gaze_label": torch.tensor([True, False]),
            },
            noisy_action=torch.zeros(2, 4, 10),
            timestep=torch.zeros(2, dtype=torch.long),
        )
    except ValueError as exc:
        assert "has_gaze_condition" in str(exc)
        assert "filter eligible gaze rows" in str(exc)
    else:
        raise AssertionError("Expected GDR rows without point gaze labels to fail.")

    try:
        policy.compute_gaze_dependency_ratio(
            {
                "camera0_rgb": torch.randn(1, 2, 3, 16, 16),
                "gaze_xy": torch.tensor([[1.2, 0.5]], dtype=torch.float32),
                "has_gaze_label": torch.tensor([True]),
            },
            noisy_action=torch.zeros(1, 4, 10),
            timestep=torch.zeros(1, dtype=torch.long),
        )
    except ValueError as exc:
        assert "compute_gaze_dependency_ratio gaze_xy rows with has_gaze_label=True" in str(exc)
    else:
        raise AssertionError("Expected out-of-range GDR gaze input to fail.")

    outside_result = policy.compute_gaze_dependency_ratio(
        {
            "camera0_rgb": torch.randn(1, 2, 3, 16, 16),
            "gaze_xy": torch.tensor([[1.2, 0.5]], dtype=torch.float32),
            "has_gaze_condition": torch.tensor([True]),
            "has_gaze_label": torch.tensor([False]),
        },
        noisy_action=torch.zeros(1, 4, 10),
        timestep=torch.zeros(1, dtype=torch.long),
    )
    assert outside_result["feature_gdr"].shape == (1,)

    try:
        policy.compute_gaze_dependency_ratio(
            {
                "camera0_rgb": torch.randn(1, 2, 3, 16, 16),
                "gaze_xy": torch.tensor([[0.5, 0.5]], dtype=torch.float32),
                "has_gaze_label": torch.tensor([True]),
            },
            noisy_action=torch.zeros(1, 3, 10),
            timestep=torch.zeros(1, dtype=torch.long),
        )
    except ValueError as exc:
        assert "compute_gaze_dependency_ratio noisy_action must have shape" in str(exc)
    else:
        raise AssertionError("Expected bad GDR noisy_action shape to fail.")

    try:
        policy.compute_gaze_dependency_ratio(
            {
                "camera0_rgb": torch.randn(1, 2, 3, 16, 16),
                "gaze_xy": torch.tensor([[0.5, 0.5]], dtype=torch.float32),
                "has_gaze_label": torch.tensor([True]),
            },
            noisy_action=torch.zeros(1, 4, 10),
            timestep=torch.zeros(2, dtype=torch.long),
        )
    except ValueError as exc:
        assert "compute_gaze_dependency_ratio timestep" in str(exc)
        assert "B=1" in str(exc)
    else:
        raise AssertionError("Expected bad GDR timestep shape to fail.")

    try:
        policy.compute_gaze_dependency_ratio(
            {
                "camera0_rgb": torch.randn(1, 2, 3, 16, 16),
                "gaze_xy": torch.tensor([[0.5, 0.5]], dtype=torch.float32),
                "has_gaze_label": torch.tensor([True]),
            },
            noisy_action=torch.zeros(1, 4, 10),
            timestep=torch.tensor([10]),
        )
    except ValueError as exc:
        assert "compute_gaze_dependency_ratio timestep" in str(exc)
        assert "[0, 9]" in str(exc)
    else:
        raise AssertionError("Expected out-of-range GDR timestep to fail.")


def test_gaze_wam_policy_predict_heatmap_visualization(tmp_path):
    torch.manual_seed(5)
    shape_meta = {
        "action": {
            "shape": [10],
            "horizon": 4,
        }
    }
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    encoder_path, decoder_path = _write_fake_cosmos_jit_pair(
        tmp_path,
        image_size=(16, 32),
        token_grid=(2, 4),
        latent_channels=1,
    )
    policy = GazeWamPolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        obs_encoder=FakeTokenObsEncoder(num_tokens=8, embed_dim=32),
        model=JointGazeWamTransformer(
            action_dim=10,
            heatmap_dim=1,
            action_horizon=4,
            heatmap_num_tokens=8,
            max_image_tokens=8,
            n_layer=1,
            n_head=4,
            n_emb=32,
            p_drop_emb=0.0,
            p_drop_attn=0.0,
        ),
        gaze_encoder=GazeConditionEncoder(embed_dim=32),
        num_inference_steps=2,
        input_pertub=0.0,
        heatmap_num_tokens=8,
        heatmap_token_grid=(2, 4),
        heatmap_image_size=(16, 32),
        heatmap_cosmos_encoder_path=encoder_path,
        heatmap_cosmos_decoder_path=decoder_path,
        n_emb=32,
    )
    normalizer = LinearNormalizer()
    normalizer["camera0_rgb"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)

    result = policy.predict_heatmap(
        {
            "camera0_rgb": torch.randn(3, 2, 3, 16, 16),
            "gaze_xy": torch.tensor(
                [
                    [0.5, 0.5],
                    [0.2, 0.8],
                    [0.7, 0.3],
                ],
                dtype=torch.float32,
            ),
            "has_gaze_label": torch.tensor([True, True, True]),
        },
        timestep=torch.zeros(3, dtype=torch.long),
    )

    assert result["heatmap_tokens"].shape == (3, 8, 1)
    assert result["heatmap_tokens_raw"].shape == (3, 8, 1)
    assert result["heatmap_model_output"].shape == (3, 8, 1)
    assert result["heatmap_features"].shape == (3, 8, 32)
    assert result["heatmap_image"].shape == (3, 16, 32)
    assert result["heatmap_image"].min() >= 0
    assert result["heatmap_image"].max() <= 1.00001

    try:
        policy.predict_heatmap(
            {
                "camera0_rgb": torch.randn(1, 2, 3, 16, 16),
                "gaze_xy": torch.tensor([[-0.1, 0.5]], dtype=torch.float32),
                "has_gaze_label": torch.tensor([True]),
            },
            timestep=torch.zeros(1, dtype=torch.long),
        )
    except ValueError as exc:
        assert "predict_heatmap gaze_xy rows with has_gaze_label=True" in str(exc)
    else:
        raise AssertionError("Expected out-of-range predict_heatmap gaze input to fail.")

    valid_obs = {
        "camera0_rgb": torch.randn(1, 2, 3, 16, 16),
        "gaze_xy": torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        "has_gaze_label": torch.tensor([True]),
    }
    invalid_noisy_cases = [
        (
            {"noisy_action": torch.zeros(1, 3, 10)},
            "predict_heatmap noisy_action must have shape",
        ),
        (
            {"noisy_heatmap": torch.zeros(1, 7, 1)},
            "predict_heatmap noisy_heatmap must have shape",
        ),
        (
            {"noisy_heatmap": torch.full((1, 8, 1), float("nan"))},
            "predict_heatmap noisy_heatmap must contain only finite values",
        ),
    ]
    for kwargs, expected in invalid_noisy_cases:
        try:
            policy.predict_heatmap(
                valid_obs,
                timestep=torch.zeros(1, dtype=torch.long),
                **kwargs,
            )
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"Expected invalid predict_heatmap input to fail: {expected}.")

    invalid_timestep_cases = [
        (torch.zeros(2, dtype=torch.long), "B=1"),
        (torch.zeros(1, 1, dtype=torch.long), "scalar or 1D"),
        (torch.tensor([0.5]), "integer diffusion timesteps"),
        (torch.tensor([-1]), "[0, 9]"),
        (torch.tensor([10]), "[0, 9]"),
    ]
    for bad_timestep, expected in invalid_timestep_cases:
        try:
            policy.predict_heatmap(valid_obs, timestep=bad_timestep)
        except ValueError as exc:
            assert "predict_heatmap timestep" in str(exc)
            assert expected in str(exc)
        else:
            raise AssertionError("Expected invalid predict_heatmap timestep to fail.")


def test_gaze_wam_mixed_batch_builder_masks_and_placeholders():
    robot_batch = {
        "obs": {
            "camera0_rgb": torch.randn(3, 2, 3, 16, 16),
        },
        "action": torch.randn(3, 4, 10),
        "action_abs": torch.randn(3, 4, 10),
        "action_base_abs": torch.randn(3, 10),
        "heatmap": torch.randn(3, 1, 8, 1),
        "gaze_xy": torch.rand(3, 2),
        "has_action_abs": torch.tensor([True, False, True]),
        "has_action_base_abs": torch.tensor([False, True, True]),
    }
    open_batch = {
        "obs": {
            "camera0_rgb": torch.randn(1, 2, 3, 16, 16),
        },
        "action": torch.ones(1, 4, 10),
        "heatmap": torch.randn(1, 1, 8, 1),
        "heatmap_image": torch.ones(1, 1, 16, 16),
        "gaze_xy": torch.rand(1, 2),
        "has_gaze_label": torch.tensor([False]),
    }

    no_dropout = build_gaze_wam_mixed_batch(
        robot_batch,
        open_batch,
        robot_gaze_dropout_prob=0.0,
        shuffle=False,
    )
    all_dropout = build_gaze_wam_mixed_batch(
        robot_batch,
        open_batch,
        robot_gaze_dropout_prob=1.0,
        shuffle=False,
    )

    assert no_dropout["obs"]["camera0_rgb"].shape == (4, 2, 3, 16, 16)
    assert no_dropout["action"].shape == (4, 4, 10)
    assert no_dropout["action_abs"].shape == (4, 4, 10)
    assert no_dropout["action_base_abs"].shape == (4, 10)
    assert no_dropout["has_action_abs"].tolist() == [True, False, True, False]
    assert no_dropout["has_action_base_abs"].tolist() == [False, True, True, False]
    assert no_dropout["heatmap_image"].shape == (4, 1, 16, 16)
    assert no_dropout["has_heatmap_image"].tolist() == [False, False, False, True]
    assert torch.allclose(no_dropout["action"][3], torch.zeros_like(no_dropout["action"][3]))
    assert torch.allclose(no_dropout["action_abs"][3], torch.zeros_like(no_dropout["action_abs"][3]))
    assert torch.allclose(
        no_dropout["action_base_abs"][3],
        torch.zeros_like(no_dropout["action_base_abs"][3]),
    )

    assert no_dropout["is_open"].tolist() == [False, False, False, True]
    assert no_dropout["has_action"].tolist() == [True, True, True, False]
    assert no_dropout["has_heatmap"].tolist() == [False, False, False, True]
    assert no_dropout["has_gaze_label"].tolist() == [True, True, True, False]
    assert torch.allclose(no_dropout["gaze_xy"][3], torch.zeros_like(no_dropout["gaze_xy"][3]))
    assert torch.allclose(no_dropout["heatmap"][:3], torch.zeros_like(no_dropout["heatmap"][:3]))
    assert torch.allclose(no_dropout["heatmap"][3], open_batch["heatmap"][0])
    assert no_dropout["use_gaze_condition"].tolist() == [True, True, True, False]
    assert no_dropout["is_gaze_condition_dropped"].tolist() == [False, False, False, True]

    assert all_dropout["has_action"].tolist() == [True, True, True, False]
    assert all_dropout["has_heatmap"].tolist() == [True, True, True, True]
    assert torch.allclose(all_dropout["heatmap"][:3], robot_batch["heatmap"])
    assert torch.allclose(all_dropout["heatmap"][3], open_batch["heatmap"][0])
    assert all_dropout["use_gaze_condition"].tolist() == [False, False, False, False]
    assert all_dropout["is_gaze_condition_dropped"].tolist() == [True, True, True, True]


def test_gaze_wam_mixed_batch_builder_supports_open_only():
    open_batch = {
        "obs": {
            "camera0_rgb": torch.randn(2, 2, 3, 16, 16),
        },
        "action": torch.ones(2, 4, 10),
        "action_abs": torch.ones(2, 4, 10),
        "heatmap": torch.randn(2, 1, 8, 1),
        "heatmap_image": torch.ones(2, 1, 16, 16),
        "gaze_xy": torch.tensor([[0.25, 0.75], [0.5, 0.5]]),
        "has_gaze_label": torch.tensor([True, False]),
        "has_heatmap_image": torch.tensor([True, False]),
    }

    batch = build_gaze_wam_mixed_batch(
        robot_batch=None,
        open_batch=open_batch,
        shuffle=False,
    )

    assert batch["obs"]["camera0_rgb"].shape == (2, 2, 3, 16, 16)
    assert torch.allclose(batch["action"], torch.zeros_like(batch["action"]))
    assert torch.allclose(batch["action_abs"], torch.zeros_like(batch["action_abs"]))
    assert batch["has_action_abs"].tolist() == [False, False]
    assert torch.allclose(batch["heatmap"][0], open_batch["heatmap"][0])
    assert torch.allclose(batch["heatmap_image"][1], torch.zeros_like(batch["heatmap_image"][1]))
    assert batch["has_heatmap_image"].tolist() == [True, False]
    assert batch["is_open"].tolist() == [True, True]
    assert batch["has_action"].tolist() == [False, False]
    assert batch["has_heatmap"].tolist() == [True, True]
    assert batch["has_gaze_label"].tolist() == [True, False]
    assert torch.allclose(batch["gaze_xy"][0], open_batch["gaze_xy"][0])
    assert torch.allclose(batch["gaze_xy"][1], torch.zeros_like(batch["gaze_xy"][1]))
    assert batch["use_gaze_condition"].tolist() == [False, False]
    assert batch["is_gaze_condition_dropped"].tolist() == [True, True]


def test_gaze_wam_mixed_batch_builder_normalizes_routing_config_values():
    robot_batch = {
        "obs": {
            "camera0_rgb": torch.randn(2, 2, 3, 16, 16),
        },
        "action": torch.randn(2, 4, 10),
        "heatmap": torch.randn(2, 1, 8, 1),
        "gaze_xy": torch.rand(2, 2),
    }
    open_batch = {
        "obs": {
            "camera0_rgb": torch.randn(1, 2, 3, 16, 16),
        },
        "heatmap": torch.randn(1, 1, 8, 1),
        "gaze_xy": torch.rand(1, 2),
    }

    no_robot_heatmap = build_gaze_wam_mixed_batch(
        robot_batch,
        open_batch,
        robot_gaze_dropout_prob="1.0",
        robot_heatmap_on_gaze_dropout="false",
        shuffle=False,
    )
    assert no_robot_heatmap["use_gaze_condition"].tolist() == [False, False, False]
    assert no_robot_heatmap["has_heatmap"].tolist() == [False, False, True]
    assert torch.allclose(
        no_robot_heatmap["heatmap"][:2],
        torch.zeros_like(no_robot_heatmap["heatmap"][:2]),
    )
    assert torch.allclose(no_robot_heatmap["heatmap"][2], open_batch["heatmap"][0])

    invalid_cases = [
        {"robot_gaze_dropout_prob": True},
        {"robot_gaze_dropout_prob": "1.5"},
        {"robot_gaze_dropout_prob": "-0.1"},
        {"robot_heatmap_on_gaze_dropout": "maybe"},
        {"robot_heatmap_on_gaze_dropout": 1},
    ]
    for kwargs in invalid_cases:
        try:
            build_gaze_wam_mixed_batch(
                robot_batch,
                open_batch,
                shuffle=False,
                **kwargs,
            )
        except ValueError as exc:
            assert (
                "robot_gaze_dropout_prob" in str(exc)
                or "robot_heatmap_on_gaze_dropout" in str(exc)
            )
        else:
            raise AssertionError(f"Expected invalid mixed-batch routing config {kwargs!r} to fail.")


def test_gaze_wam_mixed_batch_builder_preserves_optional_presence_masks():
    robot_batch = {
        "obs": {
            "camera0_rgb": torch.randn(2, 2, 3, 16, 16),
        },
        "action": torch.randn(2, 4, 10),
        "action_abs": torch.randn(2, 4, 10),
        "action_base_abs": torch.randn(2, 10),
        "heatmap": torch.randn(2, 1, 8, 1),
        "heatmap_image": torch.ones(2, 1, 16, 16),
        "gaze_xy": torch.rand(2, 2),
        "has_action_abs": torch.tensor([True, False]),
        "has_action_base_abs": torch.tensor([False, True]),
        "has_heatmap_image": torch.tensor([True, False]),
    }
    open_batch = {
        "obs": {
            "camera0_rgb": torch.randn(2, 2, 3, 16, 16),
        },
        "action_abs": torch.randn(2, 4, 10),
        "action_base_abs": torch.randn(2, 10),
        "heatmap": torch.randn(2, 1, 8, 1),
        "heatmap_image": torch.ones(2, 1, 16, 16),
        "gaze_xy": torch.rand(2, 2),
        "has_action_abs": torch.tensor([False, True]),
        "has_action_base_abs": torch.tensor([True, False]),
        "has_heatmap_image": torch.tensor([False, True]),
    }

    mixed = build_gaze_wam_mixed_batch(
        robot_batch,
        open_batch,
        robot_gaze_dropout_prob=0.0,
        shuffle=False,
    )

    assert mixed["has_action_abs"].tolist() == [True, False, False, False]
    assert mixed["has_action_base_abs"].tolist() == [False, True, False, False]
    assert mixed["has_heatmap_image"].tolist() == [True, False, False, True]
    assert torch.allclose(mixed["action_abs"][1], torch.zeros_like(mixed["action_abs"][1]))
    assert torch.allclose(mixed["action_abs"][2:], torch.zeros_like(mixed["action_abs"][2:]))
    assert torch.allclose(
        mixed["action_base_abs"][0],
        torch.zeros_like(mixed["action_base_abs"][0]),
    )
    assert torch.allclose(
        mixed["action_base_abs"][2:],
        torch.zeros_like(mixed["action_base_abs"][2:]),
    )
    assert torch.allclose(
        mixed["heatmap_image"][1],
        torch.zeros_like(mixed["heatmap_image"][1]),
    )
    assert torch.allclose(
        mixed["heatmap_image"][2],
        torch.zeros_like(mixed["heatmap_image"][2]),
    )
    assert torch.allclose(mixed["heatmap_image"][3], open_batch["heatmap_image"][1])


def test_loss_routing_summary_counts_source_supervision():
    mixed = {
        "is_open": torch.tensor([False, False, True, True]),
        "has_action": torch.tensor([True, True, False, False]),
        "has_heatmap": torch.tensor([False, True, True, True]),
        "has_gaze_label": torch.tensor([True, True, True, False]),
        "use_gaze_condition": torch.tensor([True, False, False, False]),
        "is_gaze_condition_dropped": torch.tensor([False, True, True, True]),
    }
    summary = loss_routing_summary(
        mixed=mixed,
        action_loss_mask=torch.tensor([True, True, False, False]),
        heatmap_loss_mask=torch.tensor([False, True, True, True]),
        use_distributed_counts=True,
    )

    assert summary == {
        "robot_rows": 2,
        "open_rows": 2,
        "has_action_rows": 2,
        "has_heatmap_rows": 3,
        "has_gaze_condition_rows": 3,
        "has_gaze_label_rows": 3,
        "use_gaze_condition_rows": 1,
        "dropped_gaze_condition_rows": 3,
        "robot_real_gaze_rows": 1,
        "robot_masked_gaze_rows": 1,
        "open_action_loss_count": 0,
        "open_heatmap_loss_count": 2,
        "robot_action_loss_count": 2,
        "robot_heatmap_loss_count": 1,
        "robot_real_gaze_action_loss_count": 1,
        "robot_real_gaze_heatmap_loss_count": 0,
        "robot_masked_gaze_action_loss_count": 1,
        "robot_masked_gaze_heatmap_loss_count": 1,
    }


def test_loss_routing_summary_rejects_non_bool_or_misshaped_masks():
    mixed = {
        "is_open": torch.tensor([False, False, True, True]),
        "has_action": torch.tensor([True, True, False, False]),
        "has_heatmap": torch.tensor([False, True, True, True]),
        "has_gaze_label": torch.tensor([True, True, True, False]),
        "use_gaze_condition": torch.tensor([True, False, False, False]),
        "is_gaze_condition_dropped": torch.tensor([False, True, True, True]),
    }
    action_loss_mask = torch.tensor([True, True, False, False])
    heatmap_loss_mask = torch.tensor([False, True, True, True])

    bad_source = dict(mixed)
    bad_source["has_heatmap"] = torch.tensor([0.0, 1.0, 1.0, 1.0])
    try:
        loss_routing_summary(
            mixed=bad_source,
            action_loss_mask=action_loss_mask,
            heatmap_loss_mask=heatmap_loss_mask,
        )
        assert False, "Expected non-bool source mask to fail."
    except ValueError as exc:
        assert "mixed['has_heatmap'] must be a BoolTensor" in str(exc)

    try:
        loss_routing_summary(
            mixed=mixed,
            action_loss_mask=torch.tensor([1, 1, 0, 0]),
            heatmap_loss_mask=heatmap_loss_mask,
        )
        assert False, "Expected non-bool action loss mask to fail."
    except ValueError as exc:
        assert "action_loss_mask must be a BoolTensor" in str(exc)

    try:
        loss_routing_summary(
            mixed=mixed,
            action_loss_mask=action_loss_mask[:, None],
            heatmap_loss_mask=heatmap_loss_mask,
        )
        assert False, "Expected misshaped action loss mask to fail."
    except ValueError as exc:
        assert "action_loss_mask must have shape [B]" in str(exc)


def test_gaze_wam_mixed_batch_builder_robot_only_baseline():
    robot_batch = {
        "obs": {
            "camera0_rgb": torch.randn(3, 2, 3, 16, 16),
        },
        "action": torch.randn(3, 4, 10),
        "action_abs": torch.randn(3, 4, 10),
        "action_base_abs": torch.randn(3, 10),
        "heatmap": torch.randn(3, 1, 8, 1),
        "gaze_xy": torch.rand(3, 2),
    }

    batch = build_gaze_wam_mixed_batch(
        robot_batch=robot_batch,
        open_batch=None,
        robot_gaze_dropout_prob=0.0,
        shuffle=False,
    )

    assert batch["obs"]["camera0_rgb"].shape == (3, 2, 3, 16, 16)
    assert batch["action"].shape == (3, 4, 10)
    assert batch["is_open"].tolist() == [False, False, False]
    assert batch["has_action"].tolist() == [True, True, True]
    assert batch["has_heatmap"].tolist() == [False, False, False]
    assert torch.allclose(batch["heatmap"], torch.zeros_like(batch["heatmap"]))
    assert batch["use_gaze_condition"].tolist() == [True, True, True]
    assert batch["is_gaze_condition_dropped"].tolist() == [False, False, False]
    assert batch["has_action_abs"].tolist() == [True, True, True]
    assert batch["has_action_base_abs"].tolist() == [True, True, True]
    assert torch.allclose(batch["action_abs"], robot_batch["action_abs"])
    assert torch.allclose(batch["action_base_abs"], robot_batch["action_base_abs"])


def test_gaze_wam_mixed_batch_builder_out_of_frame_gaze_conditions_action_only():
    robot_batch = {
        "obs": {
            "camera0_rgb": torch.randn(3, 2, 3, 16, 16),
        },
        "action": torch.randn(3, 4, 10),
        "heatmap": torch.randn(3, 1, 8, 1),
        "gaze_xy": torch.tensor([[0.2, 0.3], [1.25, 0.4], [0.7, 0.8]]),
        "has_gaze_condition": torch.tensor([True, True, True]),
        "has_gaze_label": torch.tensor([True, False, True]),
    }

    batch = build_gaze_wam_mixed_batch(
        robot_batch=robot_batch,
        open_batch=None,
        robot_gaze_dropout_prob=0.0,
        robot_heatmap_on_gaze_dropout=True,
        shuffle=False,
    )

    assert batch["has_action"].tolist() == [True, True, True]
    assert batch["has_gaze_condition"].tolist() == [True, True, True]
    assert batch["has_gaze_label"].tolist() == [True, False, True]
    assert batch["use_gaze_condition"].tolist() == [True, True, True]
    assert batch["is_gaze_condition_dropped"].tolist() == [False, False, False]
    assert batch["has_heatmap"].tolist() == [False, False, False]
    assert torch.allclose(batch["gaze_xy"][1], robot_batch["gaze_xy"][1])
    assert torch.allclose(batch["heatmap"], torch.zeros_like(batch["heatmap"]))

    no_heatmap_batch = build_gaze_wam_mixed_batch(
        robot_batch=robot_batch,
        open_batch=None,
        robot_gaze_dropout_prob=1.0,
        robot_heatmap_on_gaze_dropout=True,
        shuffle=False,
    )

    assert no_heatmap_batch["use_gaze_condition"].tolist() == [False, False, False]
    assert no_heatmap_batch["is_gaze_condition_dropped"].tolist() == [True, True, True]
    assert no_heatmap_batch["has_heatmap"].tolist() == [True, False, True]
    assert torch.allclose(
        no_heatmap_batch["heatmap"][1],
        torch.zeros_like(no_heatmap_batch["heatmap"][1]),
    )


def test_gaze_wam_mixed_batch_builder_no_gaze_action_baseline():
    robot_batch = {
        "obs": {
            "camera0_rgb": torch.randn(3, 2, 3, 16, 16),
        },
        "action": torch.randn(3, 4, 10),
        "heatmap": torch.randn(3, 1, 8, 1),
        "gaze_xy": torch.rand(3, 2),
    }

    batch = build_gaze_wam_mixed_batch(
        robot_batch=robot_batch,
        open_batch=None,
        robot_gaze_dropout_prob=1.0,
        robot_heatmap_on_gaze_dropout=False,
        shuffle=False,
    )

    assert batch["is_open"].tolist() == [False, False, False]
    assert batch["has_action"].tolist() == [True, True, True]
    assert batch["has_heatmap"].tolist() == [False, False, False]
    assert torch.allclose(batch["heatmap"], torch.zeros_like(batch["heatmap"]))
    assert batch["use_gaze_condition"].tolist() == [False, False, False]
    assert batch["is_gaze_condition_dropped"].tolist() == [True, True, True]


def test_gaze_wam_mixed_batch_builder_rejects_shape_mismatches():
    robot_batch = {
        "obs": {
            "camera0_rgb": torch.randn(3, 2, 3, 16, 16),
        },
        "action": torch.randn(3, 4, 10),
        "heatmap": torch.randn(3, 1, 8, 1),
        "gaze_xy": torch.rand(3, 2),
    }
    open_batch = {
        "obs": {
            "camera0_rgb": torch.randn(1, 2, 3, 16, 16),
        },
        "heatmap": torch.randn(1, 1, 8, 1),
        "gaze_xy": torch.rand(1, 2),
    }

    bad_obs = {
        **open_batch,
        "obs": {"camera0_rgb": torch.randn(1, 2, 3, 20, 16)},
    }
    try:
        build_gaze_wam_mixed_batch(
            robot_batch=robot_batch,
            open_batch=bad_obs,
            shuffle=False,
        )
    except ValueError as exc:
        assert "open_batch['obs']['camera0_rgb'] tail shape must match" in str(exc)
    else:
        raise AssertionError("Expected mixed batch builder to reject mismatched obs shapes.")

    bad_heatmap = {
        **open_batch,
        "heatmap": torch.randn(1, 1, 16, 1),
    }
    try:
        build_gaze_wam_mixed_batch(
            robot_batch=robot_batch,
            open_batch=bad_heatmap,
            shuffle=False,
        )
    except ValueError as exc:
        assert "open_batch['heatmap'] tail shape must match" in str(exc)
    else:
        raise AssertionError("Expected mixed batch builder to reject mismatched heatmap shapes.")

    bad_robot_mask = {
        **robot_batch,
        "has_gaze_label": torch.ones(3, 1, dtype=torch.bool),
    }
    try:
        build_gaze_wam_mixed_batch(
            robot_batch=bad_robot_mask,
            open_batch=open_batch,
            shuffle=False,
        )
    except ValueError as exc:
        assert "robot_batch['has_gaze_label'] must have shape [3]" in str(exc)
    else:
        raise AssertionError("Expected mixed batch builder to reject non-vector robot masks.")

    bad_robot_binary_mask = {
        **robot_batch,
        "has_gaze_label": torch.tensor([1, 2, 0]),
    }
    try:
        build_gaze_wam_mixed_batch(
            robot_batch=bad_robot_binary_mask,
            open_batch=open_batch,
            shuffle=False,
        )
    except ValueError as exc:
        assert "robot_batch['has_gaze_label'] must be a BoolTensor" in str(exc)
    else:
        raise AssertionError("Expected mixed batch builder to reject non-bool robot masks.")

    bad_open_binary_mask = {
        **open_batch,
        "has_gaze_label": torch.tensor([2]),
    }
    try:
        build_gaze_wam_mixed_batch(
            robot_batch=robot_batch,
            open_batch=bad_open_binary_mask,
            shuffle=False,
        )
    except ValueError as exc:
        assert "open_batch['has_gaze_label'] must be a BoolTensor" in str(exc)
    else:
        raise AssertionError("Expected mixed batch builder to reject non-bool open masks.")

    bad_optional_mask = {
        **open_batch,
        "heatmap_image": torch.ones(1, 1, 16, 16),
        "has_heatmap_image": torch.tensor([2]),
    }
    try:
        build_gaze_wam_mixed_batch(
            robot_batch=robot_batch,
            open_batch=bad_optional_mask,
            shuffle=False,
        )
    except ValueError as exc:
        assert "open_batch['has_heatmap_image'] must be a BoolTensor" in str(exc)
    else:
        raise AssertionError("Expected mixed batch builder to reject non-bool optional masks.")


def test_train_gaze_wam_workspace_rejects_empty_train_datasets_before_dataloader():
    class DummyDataset:
        def __init__(self, length):
            self.length = int(length)

        def __len__(self):
            return self.length

    lengths = _check_training_dataset_lengths(
        robot_dataset=DummyDataset(2),
        robot_val_dataset=DummyDataset(0),
        open_dataset=DummyDataset(1),
        open_val_dataset=DummyDataset(0),
        open_batch_size=1,
    )
    assert lengths == {
        "robot_train_samples": 2,
        "robot_val_samples": 0,
        "open_train_samples": 1,
        "open_val_samples": 0,
    }

    string_lengths = _check_training_dataset_lengths(
        robot_dataset=DummyDataset(2),
        robot_val_dataset=DummyDataset(0),
        open_dataset=DummyDataset(1),
        open_val_dataset=DummyDataset(0),
        open_batch_size="1",
    )
    assert string_lengths == lengths
    open_only_lengths = _check_training_dataset_lengths(
        robot_dataset=None,
        robot_val_dataset=None,
        open_dataset=DummyDataset(3),
        open_val_dataset=DummyDataset(1),
        robot_batch_size=0,
        open_batch_size=2,
    )
    assert open_only_lengths == {
        "robot_train_samples": 0,
        "robot_val_samples": 0,
        "open_train_samples": 3,
        "open_val_samples": 1,
    }

    for bad_open_batch_size in (True, 0.5, float("inf"), -1, "0.5"):
        try:
            _check_training_dataset_lengths(
                robot_dataset=DummyDataset(2),
                robot_val_dataset=DummyDataset(0),
                open_dataset=DummyDataset(1),
                open_val_dataset=DummyDataset(0),
                open_batch_size=bad_open_batch_size,
            )
        except ValueError as exc:
            assert "open_batch_size must be a non-negative integer" in str(exc)
        else:
            raise AssertionError(
                f"Expected invalid open_batch_size={bad_open_batch_size!r} to fail."
            )

    try:
        _check_training_dataset_lengths(
            robot_dataset=DummyDataset(0),
            robot_val_dataset=DummyDataset(0),
            open_dataset=DummyDataset(1),
            open_val_dataset=DummyDataset(0),
            open_batch_size=1,
        )
    except ValueError as exc:
        assert "Robot train dataset produced zero samples" in str(exc)
        assert "robot_train_samples=0" in str(exc)
    else:
        raise AssertionError("Expected empty robot train dataset to fail.")

    try:
        _check_training_dataset_lengths(
            robot_dataset=DummyDataset(2),
            robot_val_dataset=DummyDataset(0),
            open_dataset=DummyDataset(0),
            open_val_dataset=DummyDataset(0),
            open_batch_size=1,
        )
    except ValueError as exc:
        assert "Open-source train dataset is enabled but produced zero samples" in str(exc)
        assert "open_train_samples=0" in str(exc)
    else:
        raise AssertionError("Expected enabled empty open train dataset to fail.")


def test_gaze_wam_dataloader_length_gate_uses_shared_int_normalizer():
    text = Path("diffusion_policy/common/gaze_wam_dataloader_checks.py").read_text(
        encoding="utf-8"
    )

    assert "normalize_gaze_wam_nonnegative_int_field" in text
    assert "import math" not in text
    assert "math.isfinite" not in text
    assert "return normalize_gaze_wam_nonnegative_int_field(name, value)" in text


def test_train_gaze_wam_workspace_rejects_zero_train_dataloader_batches():
    class DummyLoader:
        def __init__(self, length):
            self.length = int(length)

        def __len__(self):
            return self.length

    lengths = _check_training_dataloader_lengths(
        robot_dataloader=DummyLoader(3),
        robot_val_dataloader=DummyLoader(0),
        open_dataloader=DummyLoader(1),
        open_val_dataloader=DummyLoader(0),
        open_batch_size=1,
    )
    assert lengths == {
        "robot_train_batches": 3,
        "robot_val_batches": 0,
        "open_train_batches": 1,
        "open_val_batches": 0,
    }

    string_lengths = _check_training_dataloader_lengths(
        robot_dataloader=DummyLoader(3),
        robot_val_dataloader=DummyLoader(0),
        open_dataloader=DummyLoader(1),
        open_val_dataloader=DummyLoader(0),
        open_batch_size="1",
    )
    assert string_lengths == lengths
    open_only_lengths = _check_training_dataloader_lengths(
        robot_dataloader=None,
        robot_val_dataloader=None,
        open_dataloader=DummyLoader(2),
        open_val_dataloader=DummyLoader(1),
        robot_batch_size=0,
        open_batch_size=2,
    )
    assert open_only_lengths == {
        "robot_train_batches": 0,
        "robot_val_batches": 0,
        "open_train_batches": 2,
        "open_val_batches": 1,
    }

    for bad_open_batch_size in (True, 0.5, float("inf"), -1, "0.5"):
        try:
            _check_training_dataloader_lengths(
                robot_dataloader=DummyLoader(3),
                robot_val_dataloader=DummyLoader(0),
                open_dataloader=DummyLoader(1),
                open_val_dataloader=DummyLoader(0),
                open_batch_size=bad_open_batch_size,
            )
        except ValueError as exc:
            assert "open_batch_size must be a non-negative integer" in str(exc)
        else:
            raise AssertionError(
                f"Expected invalid open_batch_size={bad_open_batch_size!r} to fail."
            )

    try:
        _check_training_dataloader_lengths(
            robot_dataloader=DummyLoader(0),
            robot_val_dataloader=DummyLoader(0),
            open_dataloader=DummyLoader(1),
            open_val_dataloader=DummyLoader(0),
            open_batch_size=1,
        )
    except ValueError as exc:
        assert "Robot train dataloader produced zero batches" in str(exc)
        assert "robot_train_batches=0" in str(exc)
    else:
        raise AssertionError("Expected zero robot train batches to fail.")

    try:
        _check_training_dataloader_lengths(
            robot_dataloader=DummyLoader(2),
            robot_val_dataloader=DummyLoader(0),
            open_dataloader=DummyLoader(0),
            open_val_dataloader=DummyLoader(0),
            open_batch_size=1,
        )
    except ValueError as exc:
        assert "Open-source train dataloader is enabled but produced zero batches" in str(exc)
        assert "open_train_batches=0" in str(exc)
    else:
        raise AssertionError("Expected zero open train batches to fail when open is enabled.")


def test_gaze_wam_zarr_datasets_emit_contract():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_gaze_wam_zarr(Path(tmpdir) / "robot.zarr", include_action=True)
        open_path = _write_gaze_wam_zarr(Path(tmpdir) / "open.zarr", include_action=False)

        robot_dataset = GazeWamRobotDataset(
            dataset_path=str(robot_path),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            action_padding=True,
        )
        open_dataset = GazeWamOpenDataset(
            dataset_path=str(open_path),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
        )

        robot_sample = robot_dataset[1]
        open_sample = open_dataset[1]

        assert len(robot_dataset) == 6
        assert robot_sample["obs"]["camera0_rgb"].shape == (2, 3, 16, 16)
        assert robot_sample["action"].shape == (3, 10)
        assert robot_sample["action_abs"].shape == (3, 10)
        assert robot_sample["action_base_abs"].shape == (10,)
        assert robot_sample["heatmap"].shape == (1, 16, 1)
        assert robot_sample["has_action"].item() is True
        assert robot_sample["has_heatmap"].item() is False
        assert torch.allclose(robot_sample["action"][:, 9], robot_sample["action_abs"][:, 9])

        assert open_sample["obs"]["camera0_rgb"].shape == (2, 3, 16, 16)
        assert open_sample["action"].shape == (3, 10)
        assert torch.allclose(open_sample["action"], torch.zeros_like(open_sample["action"]))
        assert open_sample["heatmap"].shape == (1, 16, 1)
        assert open_sample["is_open"].item() is True
        assert open_sample["has_action"].item() is False
        assert open_sample["has_heatmap"].item() is True

        normalizer = robot_dataset.get_normalizer()
        assert "camera0_rgb" in normalizer.params_dict
        assert "action" in normalizer.params_dict
        assert normalizer["action"].params_dict["scale"].shape[0] == 10


def test_gaze_wam_dataset_patchifies_ambiguous_dense_full_resolution_heatmap():
    with tempfile.TemporaryDirectory() as tmpdir:
        open_path = _write_gaze_wam_zarr(
            Path(tmpdir) / "open_dense_fullres.zarr",
            include_action=False,
            image_hw=(256, 256),
        )
        zroot = zarr.open(str(open_path), mode="r")
        target = torch.from_numpy(
            np.asarray(zroot["data"]["gaze_heatmap"][0], dtype=np.float32)
        )

        dataset = GazeWamOpenDataset(
            dataset_path=str(open_path),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(256, 256),
            heatmap_token_grid=(16, 16),
            heatmap_dim=256,
            action_padding=True,
        )
        sample = dataset[0]

        assert dataset.heatmap_codec.patch_area == 256
        assert sample["heatmap"].shape == (1, 256, 256)
        assert sample["heatmap_image"].shape == (1, 256, 256)
        assert torch.allclose(sample["heatmap_image"][0], target, atol=1e-6)
        decoded = dataset.heatmap_codec.decode_tokens(sample["heatmap"][0])
        assert torch.allclose(decoded, target, atol=1e-6)


def test_gaze_wam_dataset_rejects_scalar_tokens_for_full_resolution_heatmap_dim():
    with tempfile.TemporaryDirectory() as tmpdir:
        open_path = _write_gaze_wam_zarr(
            Path(tmpdir) / "open_scalar_tokens.zarr",
            include_action=False,
            image_hw=(256, 256),
        )
        zroot = zarr.open(str(open_path), mode="a")
        gaze_xy = torch.from_numpy(np.asarray(zroot["data"]["gaze_xy"][:], dtype=np.float32))
        scalar_tokens = HeatmapTokenCodec(
            token_grid=(16, 16),
            image_size=(256, 256),
        ).encode_points(gaze_xy).numpy().astype(np.float32)
        _replace_zarr_array(zroot["data"], "gaze_heatmap", scalar_tokens)

        dataset = GazeWamOpenDataset(
            dataset_path=str(open_path),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(256, 256),
            heatmap_token_grid=(16, 16),
            heatmap_dim=256,
            action_padding=True,
        )

        try:
            dataset[0]
        except ValueError as exc:
            assert "Scalar token heatmap zarr rows" in str(exc)
            assert "heatmap_dim=256" in str(exc)
        else:
            raise AssertionError(
                "Expected scalar token heatmap rows to fail for heatmap_dim=256."
            )


def test_convert_hot3d_processed_to_open_zarr_outputs_dataset_contract():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        processed_root = root / "hot3d_processed"
        _write_hot3d_processed_sequence(processed_root, frames=8, image_hw=(32, 32))
        output_zarr = root / "hot3d_open.zarr"
        preview_dir = root / "preview"

        summary = convert_hot3d_processed_to_open_zarr(
            processed_root=str(processed_root),
            output_zarr=str(output_zarr),
            image_size=(16, 16),
            heatmap_method="gaussian_point",
            heatmap_storage="token",
            heatmap_token_grid=(4, 4),
            point_heatmap_sigma_tokens=0.5,
            point_heatmap_window=2,
            preview_overlay_dir=str(preview_dir),
            preview_overlay_max_frames=4,
            preview_sigma_compare_tokens=(0.5, 1.0),
            overwrite=True,
            validate=True,
        )

        assert summary["num_sequences"] == 1
        assert summary["num_frames"] == 8
        assert summary["episode_ends"] == [8]
        assert summary["heatmap_method"] == "gaussian_point"
        assert summary["heatmap_storage"] == "token"
        assert summary["heatmap_shape"] == [8, 16, 1]
        assert summary["point_heatmap_sigma_tokens"] == 0.5
        assert summary["point_heatmap_sigma_px"] == 2.0
        assert summary["point_heatmap_sigma_source"] == "tokens"
        assert summary["validation"]["valid"] is True
        assert summary["validation"]["heatmap"]["storage"] == "token"
        assert Path(summary["preview_overlay"]["contact_sheet"]).is_file()
        assert Path(summary["preview_overlay"]["video"]).is_file()
        assert summary["preview_overlay"]["video_kind"] == (
            "red_overlay_left_bw_heatmap_right"
        )
        assert summary["preview_overlay"]["video_codec"] in {
            "h264_yuv420p",
            "opencv_mp4v",
        }
        assert summary["preview_overlay"]["num_frames"] == 4
        assert Path(summary["preview_sigma_compare"]["contact_sheet"]).is_file()
        assert Path(summary["preview_sigma_compare"]["video"]).is_file()
        assert summary["preview_sigma_compare"]["sigma_px_values"] == [2.0, 4.0]
        assert summary["preview_sigma_compare"]["sigma_labels"] == [
            "sigma=0.5tok",
            "sigma=1tok",
        ]
        assert summary["preview_sigma_compare"]["num_frames"] == 4
        assert Path(summary["preview_token_decode_comparison"]["output"]).is_file()
        assert summary["preview_token_decode_comparison"]["token_shape"] == [16, 1]

        zarr_root = zarr.open(str(output_zarr), mode="r")
        assert zarr_root["meta"].attrs["dataset_type"] == "open"
        assert zarr_root["meta"].attrs["source_dataset"] == "HOT3D Aria"
        assert zarr_root["meta"].attrs["heatmap_storage"] == "token"
        assert zarr_root["meta"].attrs["heatmap_method"] == "gaussian_point"
        assert zarr_root["meta"].attrs["heatmap_source"] == (
            "generated_from_gaze_xy:gaussian_point"
        )
        assert zarr_root["meta"].attrs["point_heatmap_sigma_tokens"] == 0.5
        assert zarr_root["meta"].attrs["point_heatmap_sigma_px"] == 2.0
        assert zarr_root["meta"].attrs["point_heatmap_sigma_source"] == "tokens"
        assert zarr_root["meta"].attrs["gaze_coordinate_source"] == (
            "upright_x_norm/upright_y_norm"
        )
        assert zarr_root["data"]["camera0_rgb"].shape == (8, 16, 16, 3)
        assert zarr_root["data"]["gaze_xy"].shape == (8, 2)
        assert zarr_root["data"]["gaze_heatmap"].shape == (8, 16, 1)
        assert zarr_root["data"]["timestamp_ns"].shape == (8,)

        dataset = GazeWamOpenDataset(
            dataset_path=str(output_zarr),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            action_padding=True,
        )
        sample = dataset[0]
        assert sample["obs"]["camera0_rgb"].shape == (2, 3, 16, 16)
        assert sample["action"].shape == (3, 10)
        assert torch.allclose(sample["action"], torch.zeros_like(sample["action"]))
        assert sample["gaze_xy"].shape == (2,)
        assert sample["heatmap"].shape == (1, 16, 1)
        assert sample["has_heatmap"].item() is True


def test_convert_hot3d_processed_to_open_zarr_supports_xy_only_heatmap_none():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        processed_root = root / "hot3d_processed"
        _write_hot3d_processed_sequence(processed_root, frames=6, image_hw=(32, 32))
        output_zarr = root / "hot3d_open_xy_only.zarr"

        summary = convert_hot3d_processed_to_open_zarr(
            processed_root=str(processed_root),
            output_zarr=str(output_zarr),
            image_size=(16, 16),
            heatmap_storage="none",
            heatmap_token_grid=(4, 4),
            overwrite=True,
            validate=True,
        )

        assert summary["num_frames"] == 6
        assert summary["heatmap_storage"] == "none"
        assert summary["heatmap_shape"] is None
        assert summary["validation"]["valid"] is True
        assert summary["validation"]["heatmap"] is None

        zarr_root = zarr.open(str(output_zarr), mode="r")
        assert "gaze_heatmap" not in zarr_root["data"]
        assert zarr_root["data"]["camera0_rgb"].chunks[0] == 6
        assert zarr_root["meta"].attrs["heatmap_storage"] == "none"
        assert zarr_root["meta"].attrs["image_chunk_frames"] == 16
        assert zarr_root["meta"].attrs["heatmap_source"] == (
            "absent:xy_only_dsnt_supervision"
        )
        assert zarr_root["data"]["has_heatmap_image"][:].tolist() == [False] * 6

        dataset = GazeWamOpenDataset(
            dataset_path=str(output_zarr),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            heatmap_dim=16,
            action_padding=True,
        )
        sample = dataset[0]
        assert sample["gaze_xy"].shape == (2,)
        assert sample["heatmap"].shape == (1, 16, 16)
        assert torch.allclose(sample["heatmap"], torch.zeros_like(sample["heatmap"]))
        assert sample["heatmap_image"].shape == (1, 16, 16)
        assert torch.allclose(
            sample["heatmap_image"],
            torch.zeros_like(sample["heatmap_image"]),
        )
        assert sample["has_heatmap"].item() is True
        assert sample["has_heatmap_image"].item() is False


def test_gaze_wam_dataset_normalizes_string_bool_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        pixel_gaze = np.asarray([[8.0, 8.0]] * 6, dtype=np.float32)
        robot_path = _write_gaze_wam_zarr(
            Path(tmpdir) / "robot.zarr",
            include_action=True,
        )
        robot_data = zarr.open(str(robot_path), mode="a")["data"]
        _replace_zarr_array(robot_data, "gaze_xy", pixel_gaze)

        dataset = GazeWamRobotDataset(
            dataset_path=str(robot_path),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            gaze_is_normalized="false",
            action_padding="off",
        )

        sample = dataset[0]
        assert dataset.gaze_is_normalized is False
        assert dataset.action_padding is False
        assert len(dataset) == 4
        assert torch.allclose(sample["gaze_xy"], torch.tensor([0.5, 0.5]))

        try:
            GazeWamRobotDataset(
                dataset_path=str(robot_path),
                n_obs_steps=2,
                action_horizon=3,
                image_size=(16, 16),
                heatmap_token_grid=(4, 4),
                action_padding="maybe",
            )
        except ValueError as exc:
            assert "action_padding must be a boolean" in str(exc)
        else:
            raise AssertionError("Expected invalid action_padding string to fail.")

        try:
            GazeWamRobotDataset(
                dataset_path=str(robot_path),
                n_obs_steps=2,
                action_horizon=3,
                image_size=(16, 16),
                heatmap_token_grid=(4, 4),
                gaze_is_normalized="maybe",
            )
        except ValueError as exc:
            assert "gaze_is_normalized must be a boolean" in str(exc)
        else:
            raise AssertionError("Expected invalid gaze_is_normalized string to fail.")


def test_gaze_wam_training_audit_summaries_normalize_string_action_padding():
    dataset_cfg = {
        "n_obs_steps": "2",
        "obs_downsample_steps": "1",
        "action_horizon": "16",
        "n_latency_steps": "0",
        "action_downsample_steps": "1",
        "action_padding": "0",
    }

    workspace_summary = train_gaze_wam_workspace_module._dataset_sampling_summary(
        dataset_cfg
    )
    preflight_summary = preflight_gaze_wam_module._dataset_sampling_summary(dataset_cfg)

    assert workspace_summary["n_obs_steps"] == 2
    assert workspace_summary["obs_downsample_steps"] == 1
    assert workspace_summary["action_horizon"] == 16
    assert workspace_summary["n_latency_steps"] == 0
    assert workspace_summary["action_downsample_steps"] == 1
    assert workspace_summary["action_padding"] is False
    assert preflight_summary["n_obs_steps"] == 2
    assert preflight_summary["obs_downsample_steps"] == 1
    assert preflight_summary["action_horizon"] == 16
    assert preflight_summary["n_latency_steps"] == 0
    assert preflight_summary["action_downsample_steps"] == 1
    assert preflight_summary["action_padding"] is False

    invalid_cases = [
        ({"n_obs_steps": True}, "dataset.n_obs_steps", "positive integer"),
        ({"obs_downsample_steps": 0.5}, "dataset.obs_downsample_steps", "positive integer"),
        ({"action_horizon": float("inf")}, "dataset.action_horizon", "positive integer"),
        ({"n_latency_steps": True}, "dataset.n_latency_steps", "non-negative integer"),
        ({"n_latency_steps": "0.5"}, "dataset.n_latency_steps", "non-negative integer"),
        (
            {"action_downsample_steps": "1.5"},
            "dataset.action_downsample_steps",
            "positive integer",
        ),
    ]
    for overrides, field_name, expected in invalid_cases:
        bad_cfg = {**dataset_cfg, **overrides}
        for summary_fn in (
            train_gaze_wam_workspace_module._dataset_sampling_summary,
            preflight_gaze_wam_module._dataset_sampling_summary,
        ):
            try:
                summary_fn(bad_cfg)
            except ValueError as exc:
                assert field_name in str(exc)
                assert expected in str(exc)
            else:
                raise AssertionError(f"Expected invalid sampling field {field_name} to fail.")


def test_gaze_wam_integer_config_helpers_reject_silent_truncation():
    assert normalize_gaze_wam_positive_int_field("task.n_obs_steps", "2") == 2
    assert normalize_gaze_wam_nonnegative_int_field("task.n_latency_steps", "0") == 0
    assert normalize_gaze_wam_nonnegative_int_field("task.n_latency_steps", 2.0) == 2
    assert normalize_gaze_wam_positive_int_sequence(
        "task.heatmap_token_grid",
        "[16, 16]",
        length=2,
    ) == [16, 16]
    assert train_gaze_wam_workspace_module._cfg_positive_int_sequence(
        "task.image_shape",
        ("3", "256", "256"),
        length=3,
    ) == [3, 256, 256]
    assert _validate_gaze_wam_dataset_positive_int("n_obs_steps", "2") == 2
    assert _validate_gaze_wam_dataset_nonnegative_int("n_latency_steps", "0") == 0
    assert _validate_gaze_wam_dataset_positive_int_pair(
        "image_size",
        ("16", "16"),
    ) == (16, 16)

    invalid_positive_cases = [True, 0, 1.5, float("inf"), "1.5"]
    for value in invalid_positive_cases:
        try:
            normalize_gaze_wam_positive_int_field("task.n_obs_steps", value)
        except ValueError as exc:
            assert "task.n_obs_steps" in str(exc)
            assert "positive integer" in str(exc)
        else:
            raise AssertionError(f"Expected invalid positive integer value {value!r} to fail.")

    invalid_nonnegative_cases = [True, -1, 0.5, float("nan"), "0.5"]
    for value in invalid_nonnegative_cases:
        try:
            normalize_gaze_wam_nonnegative_int_field("task.n_latency_steps", value)
        except ValueError as exc:
            assert "task.n_latency_steps" in str(exc)
            assert "non-negative integer" in str(exc)
        else:
            raise AssertionError(f"Expected invalid non-negative integer value {value!r} to fail.")

    invalid_sequence_cases = [
        (True, "sequence of positive integers"),
        ([16, True], "task.heatmap_token_grid[1]"),
        ([16, 16.5], "task.heatmap_token_grid[1]"),
        ([16, float("inf")], "task.heatmap_token_grid[1]"),
        ("[16, 16.5]", "task.heatmap_token_grid[1]"),
        ([16, 16, 16], "2 positive integers"),
    ]
    for value, expected in invalid_sequence_cases:
        try:
            normalize_gaze_wam_positive_int_sequence(
                "task.heatmap_token_grid",
                value,
                length=2,
            )
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"Expected invalid positive int sequence {value!r} to fail.")

    dataset_invalid_cases = [
        (
            lambda: _validate_gaze_wam_dataset_positive_int("n_obs_steps", True),
            "n_obs_steps",
            "positive integer",
        ),
        (
            lambda: _validate_gaze_wam_dataset_nonnegative_int("n_latency_steps", 0.5),
            "n_latency_steps",
            "non-negative integer",
        ),
        (
            lambda: _validate_gaze_wam_dataset_positive_int_pair("image_size", (16, "16.5")),
            "image_size[1]",
            "positive integer",
        ),
    ]
    for call, field_name, expected in dataset_invalid_cases:
        try:
            call()
        except ValueError as exc:
            assert field_name in str(exc)
            assert expected in str(exc)
        else:
            raise AssertionError(f"Expected invalid dataset integer field {field_name} to fail.")


def test_gaze_wam_nonnegative_float_config_helper_rejects_boolean_values():
    assert normalize_gaze_wam_nonnegative_float_field(
        "policy.cfg_scale",
        "1.25",
        default=1.0,
    ) == 1.25
    assert normalize_gaze_wam_positive_float_field(
        "heatmap_sigma_tokens",
        "1.25",
        default=1.0,
    ) == 1.25

    invalid_cases = [
        True,
        False,
        -0.1,
        float("inf"),
        "nan",
        "oops",
    ]
    for value in invalid_cases:
        try:
            normalize_gaze_wam_nonnegative_float_field(
                "policy.cfg_scale",
                value,
                default=1.0,
            )
        except ValueError as exc:
            assert "policy.cfg_scale" in str(exc)
            assert "finite non-negative float" in str(exc)
        else:
            raise AssertionError(f"Expected invalid cfg_scale={value!r} to fail.")

    invalid_positive_cases = [True, False, 0.0, -0.1, float("inf"), "nan", "oops"]
    for value in invalid_positive_cases:
        try:
            normalize_gaze_wam_positive_float_field(
                "heatmap_sigma_tokens",
                value,
                default=1.0,
            )
        except ValueError as exc:
            assert "heatmap_sigma_tokens" in str(exc)
            assert "finite positive float" in str(exc)
        else:
            raise AssertionError(f"Expected invalid heatmap_sigma_tokens={value!r} to fail.")


def test_gaze_wam_launch_float_helper_rejects_boolean_policy_scalars():
    assert launch_gaze_wam_training_module._as_float(
        "1.25",
        name="policy.cfg_scale",
    ) == 1.25

    try:
        launch_gaze_wam_training_module._as_float(
            True,
            name="policy.cfg_scale",
        )
    except ValueError as exc:
        assert "policy.cfg_scale" in str(exc)
        assert "finite non-negative float" in str(exc)
    else:
        raise AssertionError("Expected boolean launcher cfg_scale to fail.")


def test_gaze_wam_training_contract_rejects_boolean_policy_scalars():
    policy_cfg = {
        "action_loss_weight": "1.0",
        "heatmap_loss_weight": 1,
        "heatmap_token_kl_loss_weight": 0,
        "cfg_scale": np.float64(1.5),
    }

    assert train_gaze_wam_workspace_module._cfg_get_nonnegative_float(
        policy_cfg,
        "action_loss_weight",
        0.0,
        "policy.action_loss_weight",
    ) == 1.0
    assert train_gaze_wam_workspace_module._cfg_get_nonnegative_float(
        policy_cfg,
        "heatmap_loss_weight",
        0.0,
        "policy.heatmap_loss_weight",
    ) == 1.0
    assert train_gaze_wam_workspace_module._cfg_get_nonnegative_float(
        policy_cfg,
        "heatmap_token_kl_loss_weight",
        0.0,
        "policy.heatmap_token_kl_loss_weight",
    ) == 0.0
    assert train_gaze_wam_workspace_module._cfg_get_nonnegative_float(
        policy_cfg,
        "cfg_scale",
        1.0,
        "policy.cfg_scale",
    ) == 1.5

    invalid_cases = [
        ("action_loss_weight", True),
        ("heatmap_loss_weight", False),
        ("heatmap_token_kl_loss_weight", True),
        ("cfg_scale", True),
    ]
    for key, value in invalid_cases:
        try:
            train_gaze_wam_workspace_module._cfg_get_nonnegative_float(
                {key: value},
                key,
                1.0,
                f"policy.{key}",
            )
        except ValueError as exc:
            assert f"policy.{key}" in str(exc)
            assert "finite non-negative float" in str(exc)
        else:
            raise AssertionError(f"Expected boolean policy.{key} to fail.")


def test_gaze_wam_action_normalizer_contract_documents_robot_relative_source():
    contract = gaze_wam_action_normalizer_contract(
        action_dim="10",
        camera_key="front_rgb",
    )

    assert contract["source"] == "robot_dataset_relative_actions_only"
    assert contract["action_normalizer_source"] == "GazeWamRobotDataset.get_all_actions"
    assert contract["image_normalizer_source"] == "identity_image_normalizer"
    assert contract["normalizer_keys"] == ["front_rgb", "action"]
    assert contract["camera_key"] == "front_rgb"
    assert contract["action_key"] == "action"
    assert contract["action_dim"] == 10
    assert (
        contract["action_representation"]
        == "relative_tcp_from_latest_observed_absolute_base"
    )
    assert contract["robot_zarr_action_storage"] == "absolute_tcp_trajectory"
    assert contract["excludes_open_source_dummy_actions"] is True
    assert contract["open_source_get_normalizer_allowed"] is False
    assert contract["open_source_actions_are_zero_placeholders"] is True

    try:
        gaze_wam_action_normalizer_contract(action_dim=False)
    except ValueError as exc:
        assert "action_dim" in str(exc)
        assert "positive integer" in str(exc)
    else:
        raise AssertionError("Expected invalid action_dim to fail.")


def test_gaze_wam_loss_routing_validation_guardrail_helper():
    flags = gaze_wam_required_loss_routing_validation_flags()
    assert "inactive_optional_metadata_rows_must_be_zero_placeholders" in flags
    assert len(flags) == len(set(flags))

    contract = {"validation": {key: True for key in flags}}
    assert gaze_wam_loss_routing_validation_guardrails_ok(contract) is True

    missing_contract = {"validation": {key: True for key in flags[:-1]}}
    assert gaze_wam_loss_routing_validation_guardrails_ok(missing_contract) is False

    false_contract = copy.deepcopy(contract)
    false_contract["validation"]["inactive_heatmap_rows_must_be_zero_placeholders"] = False
    assert gaze_wam_loss_routing_validation_guardrails_ok(false_contract) is False

    assert gaze_wam_loss_routing_validation_guardrails_ok(None) is False
    assert gaze_wam_loss_routing_validation_guardrails_ok({"validation": []}) is False


def test_launcher_reads_preflight_loss_routing_validation_guardrails():
    flags = gaze_wam_required_loss_routing_validation_flags()
    preflight = {
        "policy_contract": {
            "loss_routing_contract": {
                "validation": {key: True for key in flags},
            },
        },
    }
    assert _preflight_loss_routing_validation_guardrails_ok(preflight) is True

    missing_contract = copy.deepcopy(preflight)
    missing_contract["policy_contract"].pop("loss_routing_contract")
    assert _preflight_loss_routing_validation_guardrails_ok(missing_contract) is False

    false_contract = copy.deepcopy(preflight)
    false_contract["policy_contract"]["loss_routing_contract"]["validation"][
        "inactive_heatmap_rows_must_be_zero_placeholders"
    ] = False
    assert _preflight_loss_routing_validation_guardrails_ok(false_contract) is False

    assert _preflight_loss_routing_validation_guardrails_ok(None) is False


def test_gaze_wam_data_stream_contract_documents_two_zarr_online_mixing():
    contract = gaze_wam_data_stream_contract(
        robot_dataset_path="data/robot.zarr",
        open_dataset_path="data/open.zarr",
        robot_dataset_class=(
            "diffusion_policy.dataset.gaze_wam_dataset.GazeWamRobotDataset"
        ),
        open_dataset_class=(
            "diffusion_policy.dataset.gaze_wam_dataset.GazeWamOpenDataset"
        ),
        robot_batch_size="48",
        open_batch_size=16,
    )

    assert contract["source"] == "two_zarr_two_dataset_online_mixed_batch"
    assert contract["separate_zarr_sources"] is True
    assert contract["offline_merged_zarr"] is False
    assert contract["robot"]["dataset_path"] == "data/robot.zarr"
    assert contract["open"]["dataset_path"] == "data/open.zarr"
    assert contract["robot"]["dataset_class_matches_expected"] is True
    assert contract["open"]["dataset_class_matches_expected"] is True
    assert contract["robot"]["dataloader"] == "robot_dataloader"
    assert contract["open"]["dataloader"] == "open_dataloader"
    assert contract["robot"]["batch_size_per_process"] == 48
    assert contract["open"]["batch_size_per_process"] == 16
    assert contract["robot"]["enabled"] is True
    assert contract["robot"]["drives_epoch"] is True
    assert contract["open"]["drives_epoch"] is False
    assert contract["open"]["has_action"] is False
    assert contract["open"]["action_values_used_for_loss"] is False
    assert (
        contract["mixing"]["builder"]
        == "diffusion_policy.dataset.gaze_wam_mixing.build_gaze_wam_mixed_batch"
    )
    assert contract["mixing"]["mode"] == "online_per_step_concat_after_fetch"
    assert contract["mixing"]["primary_epoch_driver"] == "robot_dataloader"
    assert (
        contract["mixing"]["ratio_source"]
        == "robot_dataloader.batch_size/open_dataloader.batch_size"
    )
    assert contract["mixing"]["shuffle_after_concat"] is True
    assert contract["mixing"]["open_iterator_policy"] == "restart_on_exhaustion"
    assert contract["mixing"]["robot_ratio_per_process"] == 0.75
    assert contract["mixing"]["open_ratio_per_process"] == 0.25

    same_path_contract = gaze_wam_data_stream_contract(
        robot_dataset_path="data/shared.zarr",
        open_dataset_path="data/shared.zarr",
        robot_dataset_class=(
            "diffusion_policy.dataset.gaze_wam_dataset.GazeWamRobotDataset"
        ),
        open_dataset_class=(
            "diffusion_policy.dataset.gaze_wam_dataset.GazeWamOpenDataset"
        ),
        robot_batch_size=48,
        open_batch_size=16,
    )
    assert same_path_contract["separate_zarr_sources"] is False

    open_only_contract = gaze_wam_data_stream_contract(
        robot_dataset_path="data/robot.zarr",
        open_dataset_path="data/open.zarr",
        robot_dataset_class=(
            "diffusion_policy.dataset.gaze_wam_dataset.GazeWamRobotDataset"
        ),
        open_dataset_class=(
            "diffusion_policy.dataset.gaze_wam_dataset.GazeWamOpenDataset"
        ),
        robot_batch_size=0,
        open_batch_size=4,
    )
    assert open_only_contract["robot"]["enabled"] is False
    assert open_only_contract["robot"]["drives_epoch"] is False
    assert open_only_contract["open"]["drives_epoch"] is True
    assert open_only_contract["mixing"]["primary_epoch_driver"] == "open_dataloader"
    assert open_only_contract["mixing"]["robot_ratio_per_process"] == 0.0
    assert open_only_contract["mixing"]["open_ratio_per_process"] == 1.0

    try:
        gaze_wam_data_stream_contract(
            robot_dataset_path="data/robot.zarr",
            open_dataset_path="data/open.zarr",
            robot_dataset_class="robot",
            open_dataset_class="open",
            robot_batch_size=False,
            open_batch_size=16,
        )
    except ValueError as exc:
        assert "robot_batch_size" in str(exc)
        assert "non-negative integer" in str(exc)
    else:
        raise AssertionError("Expected invalid robot_batch_size to fail.")


def test_gaze_wam_zarr_dataset_rejects_non_integer_geometry_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_gaze_wam_zarr(Path(tmpdir) / "robot.zarr", include_action=True)
        open_path = _write_gaze_wam_zarr(Path(tmpdir) / "open.zarr", include_action=False)

        valid_robot_kwargs = {
            "dataset_path": str(robot_path),
            "n_obs_steps": 2,
            "action_horizon": 3,
            "image_size": (16, 16),
            "heatmap_token_grid": (4, 4),
        }
        valid_open_kwargs = {
            "dataset_path": str(open_path),
            "n_obs_steps": 2,
            "action_horizon": 3,
            "image_size": (16, 16),
            "heatmap_token_grid": (4, 4),
        }

        robot_string_cfg = GazeWamRobotDataset(
            **{
                **valid_robot_kwargs,
                "n_obs_steps": "2",
                "action_horizon": "3",
                "n_latency_steps": "0",
                "image_size": ("16", "16"),
                "heatmap_token_grid": ("4", "4"),
            }
        )
        assert robot_string_cfg.n_obs_steps == 2
        assert robot_string_cfg.action_horizon == 3
        assert robot_string_cfg.n_latency_steps == 0
        assert robot_string_cfg.image_size == (16, 16)
        assert robot_string_cfg.heatmap_codec.token_grid == (4, 4)

        invalid_cases = [
            (GazeWamRobotDataset, {**valid_robot_kwargs, "n_obs_steps": True}, "n_obs_steps"),
            (GazeWamRobotDataset, {**valid_robot_kwargs, "action_horizon": 3.5}, "action_horizon"),
            (GazeWamRobotDataset, {**valid_robot_kwargs, "n_latency_steps": True}, "n_latency_steps"),
            (GazeWamRobotDataset, {**valid_robot_kwargs, "n_latency_steps": 0.5}, "n_latency_steps"),
            (
                GazeWamRobotDataset,
                {**valid_robot_kwargs, "n_latency_steps": float("inf")},
                "n_latency_steps",
            ),
            (GazeWamRobotDataset, {**valid_robot_kwargs, "n_latency_steps": "0.5"}, "n_latency_steps"),
            (GazeWamRobotDataset, {**valid_robot_kwargs, "image_size": (16, "16.5")}, "image_size[1]"),
            (
                GazeWamRobotDataset,
                {**valid_robot_kwargs, "heatmap_token_grid": (4, float("inf"))},
                "heatmap_token_grid[1]",
            ),
            (GazeWamOpenDataset, {**valid_open_kwargs, "action_dim": False}, "action_dim"),
        ]
        for dataset_cls, kwargs, expected in invalid_cases:
            try:
                dataset_cls(**kwargs)
            except ValueError as exc:
                assert expected in str(exc)
                expected_integer_message = (
                    "non-negative integer"
                    if expected == "n_latency_steps"
                    else "positive integer"
                )
                assert expected_integer_message in str(exc)
            else:
                raise AssertionError(f"Expected invalid dataset config for {expected} to fail.")


def test_gaze_wam_action_normalizer_uses_robot_relative_actions_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_gaze_wam_zarr(Path(tmpdir) / "robot.zarr", include_action=True)
        open_path = _write_gaze_wam_zarr(Path(tmpdir) / "open.zarr", include_action=False)

        robot_dataset = GazeWamRobotDataset(
            dataset_path=str(robot_path),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            action_padding=True,
        )
        open_dataset = GazeWamOpenDataset(
            dataset_path=str(open_path),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
        )

        relative_actions = robot_dataset.get_all_actions()
        assert relative_actions.shape == (6, 3, 10)
        assert not torch.allclose(relative_actions, torch.zeros_like(relative_actions))
        normalizer = robot_dataset.get_normalizer()
        assert normalizer["action"].params_dict["scale"].shape[0] == 10

        try:
            open_dataset.get_normalizer()
        except RuntimeError as exc:
            assert "must not fit the action normalizer" in str(exc)
            assert "GazeWamRobotDataset relative actions" in str(exc)
        else:
            raise AssertionError("Expected open-source dummy action normalizer fitting to fail.")


def test_gaze_wam_robot_get_all_actions_skips_observation_loading():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_gaze_wam_zarr(Path(tmpdir) / "robot.zarr", include_action=True)
        robot_dataset = GazeWamRobotDataset(
            dataset_path=str(robot_path),
            n_obs_steps=2,
            action_horizon=3,
            n_latency_steps=1,
            action_downsample_steps=2,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            action_padding=True,
        )
        expected = torch.stack(
            [robot_dataset[idx]["action"] for idx in range(len(robot_dataset))],
            dim=0,
        )

        def fail_if_observations_are_loaded(_idx):
            raise AssertionError("get_all_actions must not decode observations")

        robot_dataset._sample_obs_and_gaze = fail_if_observations_are_loaded
        actual = robot_dataset.get_all_actions()

        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=1e-6)


def test_gaze_wam_robot_action_normalizer_rejects_empty_train_samples():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_gaze_wam_zarr(Path(tmpdir) / "robot.zarr", include_action=True)
        robot_dataset = GazeWamRobotDataset(
            dataset_path=str(robot_path),
            n_obs_steps=2,
            action_horizon=64,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            action_padding=False,
        )

        assert len(robot_dataset) == 0
        try:
            robot_dataset.get_normalizer()
        except ValueError as exc:
            assert "zero relative-action samples" in str(exc)
            assert "cannot fit the Gaze-WAM action normalizer" in str(exc)
        else:
            raise AssertionError("Expected empty robot normalizer fitting to fail.")


def test_gaze_wam_dataset_rejects_invalid_temporal_parameters():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_gaze_wam_zarr(Path(tmpdir) / "robot.zarr", include_action=True)

        bad_cases = [
            ("n_obs_steps", {"n_obs_steps": 0}, "positive integer"),
            ("action_horizon", {"action_horizon": 0}, "positive integer"),
            ("obs_downsample_steps", {"obs_downsample_steps": 0}, "positive integer"),
            ("action_downsample_steps", {"action_downsample_steps": 0}, "positive integer"),
            ("n_latency_steps", {"n_latency_steps": -1}, "non-negative integer"),
            ("image_size", {"image_size": (0, 16)}, "positive integer"),
            ("heatmap_token_grid", {"heatmap_token_grid": (0, 4)}, "positive integer"),
        ]
        base_kwargs = {
            "dataset_path": str(robot_path),
            "n_obs_steps": 2,
            "action_horizon": 3,
            "image_size": (16, 16),
            "heatmap_token_grid": (4, 4),
        }
        for name, overrides, expected in bad_cases:
            try:
                kwargs = dict(base_kwargs)
                kwargs.update(overrides)
                GazeWamRobotDataset(**kwargs)
            except ValueError as exc:
                assert name in str(exc)
                assert expected in str(exc)
            else:
                raise AssertionError(f"Expected invalid {name} to fail.")

        summary = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=0,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_dataset_sample=True,
        )
        assert summary["valid"] is False
        assert any("n_obs_steps" in message and "positive integer" in message for message in summary["errors"])

        bad_geometry_summary = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(0, 16),
            heatmap_token_grid=(4, 0),
            check_dataset_sample=False,
        )
        assert bad_geometry_summary["valid"] is False
        assert any("image_size[0]" in message and "positive integer" in message for message in bad_geometry_summary["errors"])
        assert any(
            "heatmap_token_grid[1]" in message and "positive integer" in message
            for message in bad_geometry_summary["errors"]
        )


def test_gaze_wam_dataset_requires_point_gaze_key_with_dense_heatmap():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_robot_heatmap_only_zarr(Path(tmpdir) / "robot_heatmap.zarr")
        open_path = _write_gaze_wam_zarr(
            Path(tmpdir) / "open_point.zarr",
            include_action=False,
        )

        assert as_optional_gaze_wam_key(None) is None
        assert as_optional_gaze_wam_key("") is None
        assert as_optional_gaze_wam_key("None") is None
        assert as_optional_gaze_wam_key("null") is None
        assert as_optional_gaze_wam_key("gaze_xy") == "gaze_xy"

        for dataset_cls, dataset_path, gaze_key in (
            (GazeWamRobotDataset, robot_path, None),
            (GazeWamOpenDataset, open_path, "null"),
        ):
            try:
                dataset_cls(
                    dataset_path=str(dataset_path),
                    gaze_key=gaze_key,
                    heatmap_key="gaze_heatmap",
                    n_obs_steps=2,
                    action_horizon=3,
                    image_size=(16, 16),
                    heatmap_token_grid=(4, 4),
                )
            except ValueError as exc:
                assert "non-null point gaze key" in str(exc)
            else:
                raise AssertionError("Expected null gaze_key to be rejected.")

        try:
            GazeWamOpenDataset(
                dataset_path=str(robot_path),
                gaze_key="gaze_xy",
                heatmap_key="gaze_heatmap",
                n_obs_steps=2,
                action_horizon=3,
                image_size=(16, 16),
                heatmap_token_grid=(4, 4),
            )
        except KeyError as exc:
            assert "missing required point gaze key" in str(exc)
        else:
            raise AssertionError("Expected dense-heatmap-only zarr to be rejected.")


def test_gaze_wam_dataset_respects_row_has_gaze_label_mask():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_mixed_point_dense_gaze_zarr(Path(tmpdir) / "mixed_gaze.zarr")
        dataset = GazeWamRobotDataset(
            dataset_path=str(robot_path),
            gaze_key="gaze_xy",
            heatmap_key="gaze_heatmap",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
        )

        dense_sample = dataset[1]
        point_sample = dataset[2]

        assert dense_sample["has_gaze_label"].item() is False
        assert dense_sample["has_gaze_condition"].item() is True
        assert dense_sample["use_gaze_condition"].item() is True
        assert dense_sample["is_gaze_condition_dropped"].item() is False
        assert torch.allclose(dense_sample["gaze_xy"], torch.tensor([1.25, 0.4]))
        assert dense_sample["heatmap"].shape == (1, 16, 1)
        assert dense_sample["heatmap"].max() > 0
        assert dense_sample["heatmap_image"].shape == (1, 16, 16)

        assert point_sample["has_gaze_label"].item() is True
        assert point_sample["use_gaze_condition"].item() is True
        assert point_sample["is_gaze_condition_dropped"].item() is False
        assert torch.allclose(
            point_sample["gaze_xy"],
            torch.tensor([0.3, 0.4], dtype=torch.float32),
        )
        assert point_sample["heatmap_image"].shape == (1, 16, 16)
        assert point_sample["has_heatmap_image"].item() is True


def test_gaze_wam_zarr_dataset_emits_optional_presence_masks():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_gaze_wam_zarr(Path(tmpdir) / "robot.zarr", include_action=True)
        robot_data = zarr.open(str(robot_path), mode="a")["data"]
        has_action_abs = np.asarray([1, 1, 0, 1, 1, 1], dtype=np.uint8)
        has_action_base_abs = np.asarray([1, 0, 1, 1, 1, 1], dtype=np.uint8)
        robot_data.array(
            "has_action_abs",
            has_action_abs,
            shape=has_action_abs.shape,
            dtype=has_action_abs.dtype,
        )
        robot_data.array(
            "has_action_base_abs",
            has_action_base_abs[:, None],
            shape=(has_action_base_abs.shape[0], 1),
            dtype=has_action_base_abs.dtype,
        )
        robot_dataset = GazeWamRobotDataset(
            dataset_path=str(robot_path),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
        )
        robot_sample = robot_dataset[1]

        assert robot_sample["has_action_abs"].item() is False
        assert robot_sample["has_action_base_abs"].item() is False

        open_path = _write_gaze_wam_zarr(
            Path(tmpdir) / "open_point.zarr",
            include_action=False,
        )
        open_data = zarr.open(str(open_path), mode="a")["data"]
        has_heatmap_image = np.asarray([1, 0, 1, 1, 1, 1], dtype=np.uint8)
        _replace_zarr_array(
            open_data,
            "has_heatmap_image",
            has_heatmap_image[:, None],
        )
        open_dataset = GazeWamOpenDataset(
            dataset_path=str(open_path),
            gaze_key="gaze_xy",
            heatmap_key="gaze_heatmap",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
        )
        open_sample = open_dataset[1]

        assert "heatmap_image" in open_sample
        assert open_sample["has_heatmap_image"].item() is False


def test_gaze_wam_dataset_ignores_dense_heatmap_mask_when_heatmap_key_is_disabled():
    with tempfile.TemporaryDirectory() as tmpdir:
        open_path = _write_gaze_wam_zarr(
            Path(tmpdir) / "open_point.zarr",
            include_action=False,
        )
        dataset = GazeWamOpenDataset(
            dataset_path=str(open_path),
            gaze_key="gaze_xy",
            heatmap_key=None,
            temporal_heatmap_mode="off",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
        )

        sample = dataset[1]

        assert sample["has_gaze_label"].item() is True
        assert sample["has_heatmap_image"].item() is False
        assert torch.count_nonzero(sample["heatmap_image"]).item() == 0

        temporal_dataset = GazeWamOpenDataset(
            dataset_path=str(open_path),
            gaze_key="gaze_xy",
            heatmap_key=None,
            temporal_heatmap_mode="causal",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
        )
        temporal_sample = temporal_dataset[1]

        assert temporal_sample["has_heatmap_image"].item() is True
        assert temporal_sample["heatmap_image"].sum().item() > 0.0


def test_gaze_wam_open_dataset_can_alias_one_source_camera_to_two_model_inputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        open_path = _write_gaze_wam_zarr(
            Path(tmpdir) / "open_point.zarr",
            include_action=False,
        )
        dataset = GazeWamOpenDataset(
            dataset_path=str(open_path),
            camera_key="camera0_rgb",
            camera_keys=("camera0_rgb", "camera1_rgb"),
            camera_key_map={
                "camera0_rgb": "camera0_rgb",
                "camera1_rgb": "camera0_rgb",
            },
            n_obs_steps=1,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
        )

        sample = dataset[1]

        assert set(sample["obs"]) == {"camera0_rgb", "camera1_rgb"}
        assert torch.equal(sample["obs"]["camera0_rgb"], sample["obs"]["camera1_rgb"])


def test_validate_gaze_wam_zarr_robot_open_and_missing_key():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_gaze_wam_zarr(Path(tmpdir) / "robot.zarr", include_action=True)
        robot_heatmap_path = _write_robot_heatmap_only_zarr(Path(tmpdir) / "robot_heatmap.zarr")
        open_path = _write_gaze_wam_zarr(
            Path(tmpdir) / "open_point.zarr",
            include_action=False,
        )

        robot_summary = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
        )
        robot_heatmap_summary = validate_gaze_wam_zarr(
            dataset_path=str(robot_heatmap_path),
            dataset_type="robot",
            gaze_key="missing_gaze_xy",
            heatmap_key="gaze_heatmap",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
        )
        open_summary = validate_gaze_wam_zarr(
            dataset_path=str(open_path),
            dataset_type="open",
            gaze_key="gaze_xy",
            heatmap_key="gaze_heatmap",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
        )

        assert robot_summary["valid"] is True
        assert robot_summary["image_resize_mode"] == "stretch"
        assert robot_summary["sample"]["action_shape"] == [3, 10]
        assert robot_summary["sample"]["action_abs_shape"] == [3, 10]
        assert robot_summary["sample"]["action_base_abs_shape"] == [10]
        assert robot_summary["sample"]["use_gaze_condition"] is True
        assert robot_summary["sample"]["is_gaze_condition_dropped"] is False
        assert robot_summary["episode_lengths"]["lengths"] == [6]
        assert robot_summary["episode_lengths"]["num_unpadded_action_starts"] == 3
        assert robot_summary["episode_lengths"]["action_target_start_offset_steps"] == 1
        assert robot_summary["image"]["layout"] == "NHWC"
        assert robot_summary["image"]["range_kind"] == "uint8_0_255_like"
        assert robot_summary["image"]["min"] == 0.0
        assert robot_summary["image"]["max"] == 50.0
        assert set(robot_summary["presence_masks"]).issuperset(
            {"has_gaze_label", "has_heatmap_image"}
        )
        assert robot_summary["robot_numeric"]["action_abs"]["shape"] == [6, 10]
        assert robot_summary["robot_numeric"]["gripper"]["finite"] is True
        assert robot_heatmap_summary["valid"] is False
        assert any(
            "normalized point gaze key" in message
            for message in robot_heatmap_summary["errors"]
        )
        assert open_summary["valid"] is True
        assert open_summary["image_resize_mode"] == "stretch"
        assert open_summary["sample"]["has_gaze_label"] is True
        assert open_summary["sample"]["use_gaze_condition"] is False
        assert open_summary["sample"]["is_gaze_condition_dropped"] is True
        assert open_summary["sample"]["heatmap_shape"] == [1, 16, 16]
        assert open_summary["sample"]["heatmap_image_shape"] == [1, 16, 16]
        assert open_summary["image"]["range_kind"] == "uint8_0_255_like"
        assert open_summary["image"]["max"] == 50.0
        assert open_summary["heatmap"]["shape"] == [6, 16, 16]
        assert open_summary["heatmap"]["finite"] is True

        broken_path = Path(tmpdir) / "broken_robot.zarr"
        root = zarr.open(str(broken_path), mode="w")
        data = root.create_group("data")
        meta = root.create_group("meta")
        source_root = zarr.open(str(robot_path), mode="r")
        meta.array(
            "episode_ends",
            source_root["meta/episode_ends"][:],
            shape=source_root["meta/episode_ends"].shape,
            dtype=source_root["meta/episode_ends"].dtype,
        )
        for key in ("camera0_rgb", "gaze_xy", "tcp_pose_abs", "gripper_width"):
            value = source_root[f"data/{key}"][:]
            data.array(key, value, shape=value.shape, dtype=value.dtype)

        broken_summary = validate_gaze_wam_zarr(
            dataset_path=str(broken_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            heatmap_key="gaze_heatmap",
            check_dataset_sample=False,
        )
        assert broken_summary["valid"] is False
        assert any("action_abs_tcp" in message for message in broken_summary["errors"])

        missing_label_path = Path(tmpdir) / "missing_robot_gaze_label.zarr"
        _write_robot_heatmap_only_zarr(missing_label_path)
        missing_label_data = zarr.open(str(missing_label_path), mode="a")["data"]
        del missing_label_data["gaze_heatmap"]
        missing_label_summary = validate_gaze_wam_zarr(
            dataset_path=str(missing_label_path),
            dataset_type="robot",
            gaze_key="missing_gaze_xy",
            heatmap_key="gaze_heatmap",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_dataset_sample=False,
        )
        assert missing_label_summary["valid"] is False
        assert any("Robot zarr must contain" in message for message in missing_label_summary["errors"])

        zarr.open(str(robot_path), mode="a")["meta"].attrs["image_resize_mode"] = "letterbox"
        letterbox_summary = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            image_resize_mode="letterbox",
            heatmap_token_grid=(4, 4),
            check_dataset_sample=False,
        )
        assert letterbox_summary["valid"] is True
        assert letterbox_summary["image_resize_mode"] == "letterbox"


def test_validate_gaze_wam_zarr_checks_optional_presence_masks():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_gaze_wam_zarr(Path(tmpdir) / "robot.zarr", include_action=True)
        data = zarr.open(str(robot_path), mode="a")["data"]
        good_mask = np.asarray([1, 0, 1, 1, 0, 1], dtype=np.uint8)
        gaze_label_mask = np.asarray([1, 0, 1, 1, 1, 1], dtype=np.uint8)
        heatmap = np.zeros((6, 16, 16), dtype=np.float32)
        heatmap[:, 3:7, 8:12] = 1.0
        if "gaze_heatmap" in data:
            del data["gaze_heatmap"]
        data.array("gaze_heatmap", heatmap, shape=heatmap.shape, dtype=heatmap.dtype)
        data.array(
            "has_action_base_abs",
            good_mask,
            shape=good_mask.shape,
            dtype=good_mask.dtype,
        )
        if "has_gaze_label" in data:
            del data["has_gaze_label"]
        data.array(
            "has_gaze_label",
            gaze_label_mask[:, None],
            shape=(gaze_label_mask.shape[0], 1),
            dtype=gaze_label_mask.dtype,
        )

        summary = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            heatmap_key="gaze_heatmap",
            check_dataset_sample=False,
        )

        assert summary["valid"] is True
        assert summary["presence_masks"]["has_action_base_abs"]["true_count"] == 4
        assert summary["presence_masks"]["has_action_base_abs"]["false_count"] == 2
        assert summary["presence_masks"]["has_gaze_label"]["true_count"] == 5
        assert summary["presence_masks"]["has_gaze_label"]["false_count"] == 1

        sample_summary = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            heatmap_key="gaze_heatmap",
            sample_index=1,
            check_dataset_sample=True,
        )
        assert sample_summary["valid"] is True
        assert sample_summary["sample"]["has_action_base_abs"] is False
        assert sample_summary["sample"]["has_gaze_label"] is False
        assert sample_summary["sample"]["use_gaze_condition"] is False
        assert sample_summary["sample"]["action_abs_shape"] == [3, 10]
        assert sample_summary["sample"]["action_base_abs_shape"] == [10]

        data["has_action_base_abs"][1] = 2
        bad_value = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            heatmap_key="gaze_heatmap",
            check_dataset_sample=False,
        )
        assert bad_value["valid"] is False
        assert any("has_action_base_abs" in message and "0/1" in message for message in bad_value["errors"])

        open_path = _write_gaze_wam_zarr(
            Path(tmpdir) / "open_point.zarr",
            include_action=False,
        )
        open_data = zarr.open(str(open_path), mode="a")["data"]
        bad_shape = np.ones((6, 2), dtype=np.uint8)
        _replace_zarr_array(
            open_data,
            "has_heatmap_image",
            bad_shape,
        )
        bad_shape_summary = validate_gaze_wam_zarr(
            dataset_path=str(open_path),
            dataset_type="open",
            gaze_key="gaze_xy",
            heatmap_key="gaze_heatmap",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_dataset_sample=False,
        )
        assert bad_shape_summary["valid"] is False
        assert any("has_heatmap_image must be [N] or [N,1]" in message for message in bad_shape_summary["errors"])

        del open_data["has_heatmap_image"]
        good_heatmap_mask = np.asarray([1, 0, 1, 1, 1, 1], dtype=np.uint8)
        open_data.array(
            "has_heatmap_image",
            good_heatmap_mask[:, None],
            shape=(good_heatmap_mask.shape[0], 1),
            dtype=good_heatmap_mask.dtype,
        )
        open_sample_summary = validate_gaze_wam_zarr(
            dataset_path=str(open_path),
            dataset_type="open",
            gaze_key="gaze_xy",
            heatmap_key="gaze_heatmap",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            sample_index=1,
            check_dataset_sample=True,
        )
        assert open_sample_summary["valid"] is True
        assert open_sample_summary["sample"]["has_heatmap_image"] is False
        assert open_sample_summary["sample"]["heatmap_image_shape"] == [1, 16, 16]

        no_heatmap_path = _write_gaze_wam_zarr(
            Path(tmpdir) / "point_gaze_with_false_mask.zarr",
            include_action=True,
        )
        no_heatmap_data = zarr.open(str(no_heatmap_path), mode="a")["data"]
        false_gaze_mask = np.asarray([1, 0, 1, 1, 1, 1], dtype=np.uint8)
        _replace_zarr_array(
            no_heatmap_data,
            "has_gaze_label",
            false_gaze_mask,
        )
        no_heatmap_summary = validate_gaze_wam_zarr(
            dataset_path=str(no_heatmap_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_dataset_sample=False,
        )
        assert no_heatmap_summary["valid"] is True

        missing_gaze_dataset = GazeWamRobotDataset(
            dataset_path=str(no_heatmap_path),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
        )
        missing_gaze_sample = missing_gaze_dataset[1]
        assert missing_gaze_sample["has_gaze_label"].item() is False
        assert missing_gaze_sample["use_gaze_condition"].item() is False
        assert missing_gaze_sample["is_gaze_condition_dropped"].item() is True
        assert torch.allclose(missing_gaze_sample["gaze_xy"], torch.zeros(2))

        no_point_path = _write_open_heatmap_only_zarr(Path(tmpdir) / "heatmap_with_true_mask.zarr")
        no_point_data = zarr.open(str(no_point_path), mode="a")["data"]
        true_gaze_mask = np.asarray([0, 1, 0, 0, 0, 0], dtype=np.uint8)
        _replace_zarr_array(
            no_point_data,
            "has_gaze_label",
            true_gaze_mask,
        )
        no_point_summary = validate_gaze_wam_zarr(
            dataset_path=str(no_point_path),
            dataset_type="open",
            gaze_key="missing_gaze_xy",
            heatmap_key="gaze_heatmap",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_dataset_sample=False,
        )
        assert no_point_summary["valid"] is False
        assert any(
            "has_gaze_label marks 1 row(s) with point gaze labels" in message
            for message in no_point_summary["errors"]
        )


def test_gaze_wam_open_dataset_rejects_nonpositive_action_dim():
    with tempfile.TemporaryDirectory() as tmpdir:
        open_path = _write_gaze_wam_zarr(
            Path(tmpdir) / "open_point.zarr",
            include_action=False,
        )

        summary = validate_gaze_wam_zarr(
            dataset_path=str(open_path),
            dataset_type="open",
            gaze_key="gaze_xy",
            heatmap_key="gaze_heatmap",
            n_obs_steps=2,
            action_horizon=3,
            action_dim=0,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_dataset_sample=False,
        )
        assert summary["valid"] is False
        assert any("action_dim must be a positive integer" in message for message in summary["errors"])

        try:
            GazeWamOpenDataset(
                dataset_path=str(open_path),
                gaze_key="gaze_xy",
                heatmap_key="gaze_heatmap",
                n_obs_steps=2,
                action_horizon=3,
                action_dim=0,
                image_size=(16, 16),
                heatmap_token_grid=(4, 4),
            )
        except ValueError as exc:
            assert "action_dim must be a positive integer" in str(exc)
        else:
            raise AssertionError("Expected nonpositive open action_dim to fail.")


def test_validate_gaze_wam_zarr_rejects_invalid_scalar_arguments():
    with tempfile.TemporaryDirectory() as tmpdir:
        open_path = _write_open_heatmap_only_zarr(Path(tmpdir) / "open.zarr")

        summary = validate_gaze_wam_zarr(
            dataset_path=str(open_path),
            dataset_type="open",
            gaze_key="missing_gaze_xy",
            heatmap_key="gaze_heatmap",
            n_obs_steps=2,
            action_horizon=3,
            action_dim="bad",
            sample_index=-1,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_dataset_sample=False,
        )

        assert summary["valid"] is False
        assert any("action_dim must be a positive integer" in message for message in summary["errors"])
        assert any("sample_index must be a non-negative integer" in message for message in summary["errors"])


def test_validate_gaze_wam_zarr_reports_short_episode_and_image_range_warnings():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_linear_action_zarr(Path(tmpdir) / "short_robot.zarr", length=2)
        root = zarr.open(str(robot_path), mode="a")
        del root["data"]["camera0_rgb"]
        image = np.full((2, 16, 16, 3), 300.0, dtype=np.float32)
        root["data"].array("camera0_rgb", image, shape=image.shape, dtype=image.dtype)

        summary = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=3,
            action_horizon=4,
            n_latency_steps=1,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_dataset_sample=False,
        )

    assert summary["valid"] is True
    assert summary["episode_lengths"]["lengths"] == [2]
    assert summary["episode_lengths"]["num_short_for_obs"] == 1
    assert summary["episode_lengths"]["num_short_for_action"] == 1
    assert summary["episode_lengths"]["num_unpadded_action_starts"] == 0
    assert summary["image"]["range_kind"] == "out_of_expected_image_range"
    assert summary["image"]["max"] == 300.0
    assert any("shorter than n_obs_steps" in warning for warning in summary["warnings"])
    assert any("No unpadded action chunks" in warning for warning in summary["warnings"])
    assert any("exceeds 255" in warning for warning in summary["warnings"])


def test_validate_gaze_wam_zarr_robot_action_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_linear_action_zarr(Path(tmpdir) / "robot.zarr", length=6)

        good_summary = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
        )

        assert good_summary["valid"] is True
        assert good_summary["sample"]["action_roundtrip_max_error"] < 1e-6

        robot_root = zarr.open(str(robot_path), mode="a")
        robot_root["data/action_abs_tcp"][1, 3:9] = np.asarray(
            [2.0, 0.0, 0.0, 0.0, 0.5, 0.0],
            dtype=np.float32,
        )

        bad_summary = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
        )
        skipped_summary = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_action_roundtrip=False,
        )

        assert bad_summary["valid"] is False
        assert any("roundtrip" in message for message in bad_summary["errors"])
        assert bad_summary["sample"]["action_roundtrip_max_error"] > 0.1
        assert skipped_summary["valid"] is True
        assert "action_roundtrip_max_error" not in skipped_summary["sample"]


def test_validate_gaze_wam_zarr_timestamp_alignment():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_linear_action_zarr(Path(tmpdir) / "robot.zarr", length=6)
        _add_timestamp_arrays(robot_path)

        summary = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            timestamp_key="timestamp",
            image_timestamp_key="image_timestamp",
            robot_state_timestamp_key="robot_state_timestamp",
            action_timestamp_key="action_timestamp",
            gaze_timestamp_key="gaze_timestamp",
            timestamp_max_delta=0.01,
            timestamp_max_step=0.06,
        )

        assert summary["valid"] is True
        assert summary["timestamps"]["checked"] is True
        assert "image_timestamp" in summary["timestamps"]["alignment"]
        assert summary["timestamps"]["alignment"]["action_timestamp"]["max_abs_delta"] < 0.005
        assert summary["timestamps"]["intervals"]["timestamp"]["count"] == 5
        assert summary["timestamps"]["intervals"]["timestamp"]["max_step"] <= 0.051
        assert summary["timestamps"]["intervals"]["gaze_timestamp"]["mean_step"] <= 0.051

        root = zarr.open(str(robot_path), mode="a")
        root["data/gaze_timestamp"][:] = root["data/timestamp"][:] + 0.2
        bad_delta = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            timestamp_key="timestamp",
            gaze_timestamp_key="gaze_timestamp",
            timestamp_max_delta=0.01,
        )
        assert bad_delta["valid"] is False
        assert any("gaze_timestamp" in message and "max_abs_delta" in message for message in bad_delta["errors"])

        root["data/gaze_timestamp"][:] = root["data/timestamp"][:] + 0.003
        root["data/timestamp"][3:] = root["data/timestamp"][3:] + 0.2
        bad_step = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            timestamp_key="timestamp",
            timestamp_max_step=0.06,
            check_dataset_sample=False,
        )
        assert bad_step["valid"] is False
        assert bad_step["timestamps"]["intervals"]["timestamp"]["max_step"] > 0.2
        assert any("timestamp" in message and "max_step" in message for message in bad_step["errors"])

        root["data/timestamp"][:] = np.arange(6, dtype=np.float64) * 0.05
        root["data/timestamp"][2] = root["data/timestamp"][1] - 0.01
        bad_order = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            timestamp_key="timestamp",
        )
        assert bad_order["valid"] is False
        assert any("timestamp must be nondecreasing" in message for message in bad_order["errors"])

        missing = validate_gaze_wam_zarr(
            dataset_path=str(_write_linear_action_zarr(Path(tmpdir) / "no_timestamp.zarr", length=6)),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            require_timestamps=True,
        )
        assert missing["valid"] is False
        assert any("timestamp" in message for message in missing["errors"])


def test_validate_gaze_wam_zarr_infers_nanosecond_timestamp_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        open_path = _write_gaze_wam_zarr(
            Path(tmpdir) / "open.zarr",
            include_action=False,
        )
        root = zarr.open(str(open_path), mode="a")
        root["meta"].attrs.update(
            {
                "dataset_type": "open",
                "image_resize_mode": "stretch",
                "image_size": [16, 16],
                "timestamp_key": "timestamp_ns",
                "timestamp_unit": "ns",
            }
        )
        timestamp_ns = np.arange(6, dtype=np.int64) * 50_000_000
        root["data"].array(
            "timestamp_ns",
            timestamp_ns,
            shape=timestamp_ns.shape,
            dtype=timestamp_ns.dtype,
        )

        summary = validate_gaze_wam_zarr(
            dataset_path=str(open_path),
            dataset_type="open",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            image_resize_mode="stretch",
            heatmap_token_grid=(4, 4),
            require_timestamps=True,
            timestamp_max_step=0.06,
        )

        assert summary["valid"] is True
        assert summary["timestamps"]["base_key"] == "timestamp_ns"
        assert summary["timestamps"]["timestamp_unit"] == "ns"
        assert summary["timestamps"]["scale_to_seconds"] == 1e-9
        assert summary["timestamps"]["intervals"]["timestamp_ns"]["max_step"] <= 0.051


def test_validate_gaze_wam_zarr_timestamp_steps_ignore_episode_boundaries():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_linear_action_zarr(Path(tmpdir) / "robot.zarr", length=6)
        _add_timestamp_arrays(robot_path)
        root = zarr.open(str(robot_path), mode="a")
        _replace_zarr_array(root["meta"], "episode_ends", np.asarray([3, 6], dtype=np.int64))
        for key in (
            "timestamp",
            "image_timestamp",
            "robot_state_timestamp",
            "action_timestamp",
            "gaze_timestamp",
        ):
            values = np.asarray(root[f"data/{key}"][:], dtype=np.float64)
            values[3:] += 100.0
            root[f"data/{key}"][:] = values

        summary = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            timestamp_key="timestamp",
            image_timestamp_key="image_timestamp",
            robot_state_timestamp_key="robot_state_timestamp",
            action_timestamp_key="action_timestamp",
            gaze_timestamp_key="gaze_timestamp",
            timestamp_max_delta=0.01,
            timestamp_max_step=0.06,
        )

        assert summary["valid"] is True
        assert summary["timestamps"]["intervals"]["timestamp"]["count"] == 4
        assert summary["timestamps"]["intervals"]["timestamp"]["max_step"] <= 0.051


def test_validate_gaze_wam_zarr_allows_independent_gaze_timestamp_gaps():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_linear_action_zarr(Path(tmpdir) / "robot.zarr", length=6)
        _add_timestamp_arrays(robot_path)
        root = zarr.open(str(robot_path), mode="a")
        root["data/gaze_timestamp"][3:] += 0.2

        allowed = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            timestamp_key="timestamp",
            gaze_timestamp_key="gaze_timestamp",
            timestamp_max_step=0.06,
            check_dataset_sample=False,
        )
        constrained = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            timestamp_key="timestamp",
            gaze_timestamp_key="gaze_timestamp",
            timestamp_max_step=0.06,
            gaze_timestamp_max_step=0.06,
            check_dataset_sample=False,
        )

        assert allowed["valid"] is True
        assert allowed["timestamps"]["intervals"]["gaze_timestamp"]["max_step"] > 0.2
        assert constrained["valid"] is False
        assert any(
            "gaze_timestamp" in message and "max_step" in message
            for message in constrained["errors"]
        )


def test_validate_gaze_wam_zarr_rejects_invalid_point_gaze():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        robot_path = _write_gaze_wam_zarr(root / "robot.zarr", include_action=True)
        robot_root = zarr.open(str(robot_path), mode="a")
        robot_root["data/gaze_xy"][0] = np.asarray([1.2, 0.5], dtype=np.float32)

        robot_summary = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_dataset_sample=False,
        )

        assert robot_summary["valid"] is False
        assert any("gaze_xy" in message and "Out-of-frame" in message for message in robot_summary["errors"])
        assert robot_summary["warnings"] == []

        open_path = _write_gaze_wam_zarr(root / "open_point.zarr", include_action=False)
        open_root = zarr.open(str(open_path), mode="a")
        open_root["data/gaze_xy"][1] = np.asarray([np.nan, 0.5], dtype=np.float32)

        open_summary = validate_gaze_wam_zarr(
            dataset_path=str(open_path),
            dataset_type="open",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_dataset_sample=False,
        )

        assert open_summary["valid"] is False
        assert any("gaze_xy" in message and "Invalid gaze point" in message for message in open_summary["errors"])


def test_validate_gaze_wam_zarr_accepts_out_of_frame_gaze_condition_without_label():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_gaze_wam_zarr(Path(tmpdir) / "robot.zarr", include_action=True)
        data = zarr.open(str(robot_path), mode="a")["data"]
        gaze_xy = np.asarray(data["gaze_xy"][:], dtype=np.float32)
        gaze_xy[1] = np.asarray([1.25, -0.1], dtype=np.float32)
        has_gaze_condition = np.ones((gaze_xy.shape[0],), dtype=np.bool_)
        has_gaze_label = np.ones((gaze_xy.shape[0],), dtype=np.bool_)
        has_gaze_label[1] = False
        _replace_zarr_array(data, "gaze_xy", gaze_xy)
        _replace_zarr_array(data, "has_gaze_condition", has_gaze_condition)
        _replace_zarr_array(data, "has_gaze_label", has_gaze_label)

        summary = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_dataset_sample=False,
        )

        assert summary["valid"] is True
        assert summary["presence_masks"]["has_gaze_condition"]["true_count"] == 6
        assert summary["presence_masks"]["has_gaze_label"]["false_count"] == 1


def test_validate_gaze_wam_zarr_rejects_nonfinite_actions_and_negative_heatmap():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        robot_path = _write_gaze_wam_zarr(root / "robot.zarr", include_action=True)
        robot_root = zarr.open(str(robot_path), mode="a")
        robot_root["data/action_abs_tcp"][2, 0] = np.nan

        robot_summary = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_dataset_sample=False,
        )

        assert robot_summary["valid"] is False
        assert any(
            "action_abs_tcp" in message and "finite values" in message
            for message in robot_summary["errors"]
        )

        open_path = _write_open_heatmap_only_zarr(root / "open_heatmap.zarr")
        open_root = zarr.open(str(open_path), mode="a")
        open_root["data/gaze_heatmap"][1, 2, 3] = -1.0

        open_summary = validate_gaze_wam_zarr(
            dataset_path=str(open_path),
            dataset_type="open",
            gaze_key="missing_gaze_xy",
            heatmap_key="gaze_heatmap",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_dataset_sample=False,
        )

        assert open_summary["valid"] is False
        assert any(
            "gaze_heatmap" in message and "non-negative" in message
            for message in open_summary["errors"]
        )


def test_gaze_wam_dataset_rejects_nonfinite_actions_and_negative_heatmap_on_sample():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        robot_path = _write_gaze_wam_zarr(root / "robot.zarr", include_action=True)
        robot_root = zarr.open(str(robot_path), mode="a")
        robot_root["data/action_abs_tcp"][2, 0] = np.nan

        robot_dataset = GazeWamRobotDataset(
            dataset_path=str(robot_path),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
        )
        try:
            _ = robot_dataset[1]
        except ValueError as exc:
            assert "robot action_abs must contain only finite values" in str(exc)
        else:
            raise AssertionError("Expected robot dataset sample to reject non-finite action_abs.")

        open_path = _write_gaze_wam_zarr(
            root / "open_heatmap.zarr",
            include_action=False,
        )
        open_root = zarr.open(str(open_path), mode="a")
        open_root["data/gaze_heatmap"][1, 2, 3] = -1.0

        open_dataset = GazeWamOpenDataset(
            dataset_path=str(open_path),
            gaze_key="gaze_xy",
            heatmap_key="gaze_heatmap",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
        )
        try:
            _ = open_dataset[1]
        except ValueError as exc:
            assert "dense gaze heatmap must be non-negative" in str(exc)
        else:
            raise AssertionError("Expected open dataset sample to reject negative heatmap.")


def test_inspect_gaze_wam_zarr_suggests_robot_key_mapping():
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = _write_noncanonical_robot_zarr(Path(tmpdir) / "raw_robot.zarr")

        summary = inspect_gaze_wam_zarr(str(raw_path), dataset_type="auto", max_items=4, top_k=2)

        assert summary["guessed_dataset_type"] == "robot"
        assert summary["episode_ends"]["path"] == "meta/episode_ends"
        assert summary["episode_ends"]["last"] == 6
        assert summary["episode_ends"]["strictly_increasing"] is True
        assert summary["suggestions"]["camera"][0]["key"] == "front_rgb"
        assert summary["suggestions"]["action"][0]["key"] == "future_tcp_pose"
        assert summary["suggestions"]["tcp_pose"][0]["key"] == "current_tcp_pose"
        assert summary["suggestions"]["gripper"][0]["key"] == "jaw_width"
        assert summary["suggestions"]["gaze"][0]["key"] == "eye_pixel_xy"
        assert summary["suggestions"]["timestamp"][0]["key"] == "sensor_time"
        assert summary["source_key_map"] == {
            "camera": "front_rgb",
            "action": "future_tcp_pose",
            "tcp_pose": "current_tcp_pose",
            "gripper": "jaw_width",
            "gaze": "eye_pixel_xy",
            "heatmap": None,
            "timestamp": "sensor_time",
        }
        assert summary["mapping_status"]["ready_for_robot_canonicalization"] is True
        assert summary["mapping_status"]["missing_required_roles"] == []
        assert summary["canonicalizer_args"] == [
            "--camera-key",
            "front_rgb",
            "--action-key",
            "future_tcp_pose",
            "--tcp-pose-key",
            "current_tcp_pose",
            "--gripper-key",
            "jaw_width",
            "--gaze-key",
            "eye_pixel_xy",
            "--timestamp-key",
            "sensor_time",
        ]
        assert "scripts/canonicalize_robot_gaze_wam_zarr.py" in summary["canonicalizer_command_template"]
        assert "--input" in summary["canonicalizer_command_template"]
        assert "raw_robot.zarr" in summary["canonicalizer_command_template"]
        assert "<output_robot.zarr>" in summary["canonicalizer_command_template"]


def test_inspect_gaze_wam_zarr_suggests_open_heatmap_key():
    with tempfile.TemporaryDirectory() as tmpdir:
        open_path = _write_open_heatmap_only_zarr(Path(tmpdir) / "open.zarr")

        summary = inspect_gaze_wam_zarr(str(open_path), dataset_type="open", max_items=4, top_k=2)

        assert summary["guessed_dataset_type"] == "open"
        assert summary["suggestions"]["camera"][0]["key"] == "camera0_rgb"
        assert summary["suggestions"]["heatmap"][0]["key"] == "gaze_heatmap"
        assert summary["suggestions"]["gaze"] == []
        assert summary["mapping_status"]["ready_for_robot_canonicalization"] is False
        assert summary["canonicalizer_command_template"] is None
        assert summary["warnings"] == []


def test_inspect_gaze_wam_zarr_suggests_robot_heatmap_only_key_mapping():
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = _write_noncanonical_robot_heatmap_only_zarr(Path(tmpdir) / "raw_robot.zarr")

        summary = inspect_gaze_wam_zarr(str(raw_path), dataset_type="auto", max_items=4, top_k=2)

        assert summary["guessed_dataset_type"] == "robot"
        assert summary["suggestions"]["gaze"] == []
        assert summary["suggestions"]["heatmap"][0]["key"] == "eye_gaze_heatmap"
        assert summary["mapping_status"]["ready_for_robot_canonicalization"] is False
        assert summary["mapping_status"]["missing_required_roles"] == ["gaze"]
        assert summary["mapping_status"]["has_dense_heatmap"] is True
        assert summary["canonicalizer_args"] is None
        assert summary["canonicalizer_command_template"] is None
        assert any("infer all robot canonicalizer keys" in item for item in summary["warnings"])


def test_preview_gaze_wam_dataset_writes_robot_overlay_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_gaze_wam_zarr(Path(tmpdir) / "robot.zarr", include_action=True)
        output_dir = Path(tmpdir) / "preview_robot"

        summary = preview_gaze_wam_dataset(
            dataset_path=str(robot_path),
            dataset_type="robot",
            output_dir=str(output_dir),
            sample_index=1,
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
        )

        assert summary["dataset_type"] == "robot"
        assert summary["image_resize_mode"] == "stretch"
        assert summary["sample_index"] == 1
        assert summary["has_gaze_label"] is True
        assert summary["has_action"] is True
        assert summary["use_gaze_condition"] is True
        assert summary["is_gaze_condition_dropped"] is False
        assert summary["heatmap_token_shape"] == [16, 1]
        assert summary["heatmap_image_shape"] == [16, 16]
        assert summary["action_shape"] == [3, 10]
        for name in ("rgb", "rgb_gaze", "heatmap", "overlay", "summary"):
            path = Path(summary["paths"][name])
            assert path.exists()
        overlay = cv2.imread(summary["paths"]["overlay"], cv2.IMREAD_UNCHANGED)
        heatmap = cv2.imread(summary["paths"]["heatmap"], cv2.IMREAD_UNCHANGED)
        saved_summary = json.loads(Path(summary["paths"]["summary"]).read_text(encoding="utf-8"))
        assert overlay.shape[:2] == (16, 16)
        assert heatmap.shape[:2] == (16, 16)
        assert saved_summary["heatmap_token_argmax"] == summary["heatmap_token_argmax"]


def test_preview_gaze_wam_dataset_writes_open_dense_heatmap_preview():
    with tempfile.TemporaryDirectory() as tmpdir:
        open_path = _write_gaze_wam_zarr(
            Path(tmpdir) / "open.zarr",
            include_action=False,
            image_hw=(32, 32),
        )
        open_data = zarr.open(str(open_path), mode="a")["data"]
        _replace_zarr_array(
            open_data,
            "has_gaze_label",
            np.zeros(6, dtype=np.bool_),
        )
        output_dir = Path(tmpdir) / "preview_open"

        summary = preview_gaze_wam_dataset(
            dataset_path=str(open_path),
            dataset_type="open",
            output_dir=str(output_dir),
            sample_index=2,
            gaze_key="gaze_xy",
            heatmap_key="gaze_heatmap",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
        )

        assert summary["dataset_type"] == "open"
        assert summary["image_resize_mode"] == "stretch"
        assert summary["has_gaze_label"] is False
        assert summary["has_action"] is False
        assert summary["has_heatmap"] is True
        assert summary["use_gaze_condition"] is False
        assert summary["is_gaze_condition_dropped"] is True
        assert summary["gaze_xy"] == [0.0, 0.0]
        assert summary["heatmap_image_source"] == "sample_heatmap_image"
        assert Path(summary["paths"]["overlay"]).exists()
        overlay = cv2.imread(summary["paths"]["overlay"], cv2.IMREAD_UNCHANGED)
        assert overlay.shape[:2] == (16, 16)


def test_canonicalize_robot_gaze_wam_zarr_key_mapping_and_pixel_gaze():
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = _write_noncanonical_robot_zarr(
            Path(tmpdir) / "raw_robot.zarr",
            image_hw=(16, 16),
        )
        raw_data = zarr.open(str(raw_path), mode="a")["data"]
        has_action_abs = np.asarray([1, 1, 0, 1, 1, 1], dtype=np.uint8)
        has_action_base_abs = np.asarray([1, 0, 1, 1, 1, 1], dtype=np.uint8)
        has_gaze_label = np.asarray([1, 1, 1, 1, 1, 1], dtype=np.uint8)
        raw_data.array(
            "has_action_abs",
            has_action_abs,
            shape=has_action_abs.shape,
            dtype=has_action_abs.dtype,
        )
        raw_data.array(
            "has_action_base_abs",
            has_action_base_abs[:, None],
            shape=(has_action_base_abs.shape[0], 1),
            dtype=has_action_base_abs.dtype,
        )
        raw_data.array(
            "has_gaze_label",
            has_gaze_label,
            shape=has_gaze_label.shape,
            dtype=has_gaze_label.dtype,
        )
        canonical_path = Path(tmpdir) / "canonical_robot.zarr"

        summary = canonicalize_robot_gaze_wam_zarr(
            input_path=str(raw_path),
            output_path=str(canonical_path),
            camera_key="front_rgb",
            action_key="future_tcp_pose",
            tcp_pose_key="current_tcp_pose",
            gripper_key="jaw_width",
            gaze_key="eye_pixel_xy",
            timestamp_key="sensor_time",
            image_timestamp_key="image_timestamp",
            robot_state_timestamp_key="robot_state_timestamp",
            action_timestamp_key="action_timestamp",
            gaze_timestamp_key="gaze_timestamp",
            gaze_is_normalized=False,
            overwrite=True,
        )

        assert summary["validated"] is True
        assert summary["keys"] == [
            "action_abs_tcp",
            "action_timestamp",
            "camera0_rgb",
            "gaze_heatmap",
            "gaze_timestamp",
            "gaze_xy",
            "gripper_width",
            "has_action_abs",
            "has_action_base_abs",
            "has_gaze_condition",
            "has_gaze_label",
            "has_heatmap_image",
            "image_timestamp",
            "robot_state_timestamp",
            "tcp_pose_abs",
            "timestamp",
        ]
        assert summary["presence_mask_keys"] == [
            "has_action_abs",
            "has_action_base_abs",
            "has_gaze_condition",
            "has_gaze_label",
            "has_heatmap_image",
        ]
        assert summary["validation"]["presence_masks"]["has_action_abs"]["true_count"] == 5
        assert summary["validation"]["presence_masks"]["has_action_base_abs"]["false_count"] == 1
        assert summary["validation"]["presence_masks"]["has_gaze_label"]["true_count"] == 6
        assert summary["timestamp_stream_keys"] == {
            "action_timestamp": {
                "output_key": "action_timestamp",
                "source_key": "action_timestamp",
            },
            "gaze_timestamp": {
                "output_key": "gaze_timestamp",
                "source_key": "gaze_timestamp",
            },
            "image_timestamp": {
                "output_key": "image_timestamp",
                "source_key": "image_timestamp",
            },
            "robot_state_timestamp": {
                "output_key": "robot_state_timestamp",
                "source_key": "robot_state_timestamp",
            },
        }
        root = zarr.open(str(canonical_path), mode="r")
        assert root["data/action_abs_tcp"].shape == (6, 10)
        assert root["data/tcp_pose_abs"].shape == (6, 9)
        assert root["meta"].attrs["image_size"] == [16, 16]
        assert root["meta"].attrs["image_resize_mode"] == "stretch"
        assert summary["image_size"] == [16, 16]
        assert summary["image_resize_mode"] == "stretch"
        assert summary["validation"]["metadata_attrs"]["image_size"] == [16, 16]
        assert summary["validation"]["metadata_attrs"]["image_resize_mode"] == "stretch"
        assert np.allclose(root["data/timestamp"][:], np.arange(6, dtype=np.float64) * 0.05)
        assert np.allclose(root["data/image_timestamp"][:], root["data/timestamp"][:] + 0.001)
        assert np.allclose(root["data/gaze_timestamp"][:], root["data/timestamp"][:] + 0.003)
        assert np.allclose(root["data/action_abs_tcp"][:, 9], root["data/gripper_width"][:])
        assert np.all(root["data/gaze_xy"][:] >= 0)
        assert np.all(root["data/gaze_xy"][:] <= 1)
        assert np.allclose(root["data/gaze_xy"][0], np.asarray([0.0, 0.0], dtype=np.float32))
        assert np.allclose(root["data/has_action_abs"][:], has_action_abs)
        assert np.allclose(root["data/has_action_base_abs"][:, 0], has_action_base_abs)
        assert np.allclose(root["data/has_gaze_label"][:], has_gaze_label)

        dataset = GazeWamRobotDataset(
            dataset_path=str(canonical_path),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            action_padding=True,
        )
        sample = dataset[1]
        assert sample["obs"]["camera0_rgb"].shape == (2, 3, 16, 16)
        assert sample["action"].shape == (3, 10)
        assert sample["action_abs"].shape == (3, 10)
        assert sample["action_base_abs"].shape == (10,)
        assert sample["has_action_abs"].item() is False
        assert sample["has_action_base_abs"].item() is False


def test_canonicalize_robot_gaze_wam_zarr_rejects_dense_heatmap_without_gaze_point():
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = _write_noncanonical_robot_heatmap_only_zarr(Path(tmpdir) / "raw_robot.zarr")
        with pytest.raises(KeyError, match="point gaze key"):
            canonicalize_robot_gaze_wam_zarr(
                input_path=str(raw_path),
                output_path=str(Path(tmpdir) / "canonical_robot.zarr"),
                camera_key="front_rgb",
                action_key="future_tcp_pose",
                tcp_pose_key="current_tcp_pose",
                gripper_key="jaw_width",
                gaze_key="missing_eye_pixel_xy",
                heatmap_key="eye_gaze_heatmap",
                timestamp_key="sensor_time",
                overwrite=True,
            )


def test_canonicalize_robot_gaze_wam_zarr_rejects_null_like_gaze_key():
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = _write_noncanonical_robot_heatmap_only_zarr(Path(tmpdir) / "raw_robot.zarr")
        with pytest.raises(KeyError, match="point gaze key"):
            canonicalize_robot_gaze_wam_zarr(
                input_path=str(raw_path),
                output_path=str(Path(tmpdir) / "canonical_robot.zarr"),
                camera_key="front_rgb",
                action_key="future_tcp_pose",
                tcp_pose_key="current_tcp_pose",
                gripper_key="jaw_width",
                gaze_key="null",
                heatmap_key="eye_gaze_heatmap",
                timestamp_key="None",
                overwrite=True,
            )


def test_canonicalize_robot_gaze_wam_zarr_rejects_out_of_bounds_gaze_by_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = _write_noncanonical_robot_zarr(Path(tmpdir) / "raw_robot.zarr")
        root = zarr.open(str(raw_path), mode="a")
        root["data"]["eye_pixel_xy"][:] = np.asarray(
            [[50.0, 5.0], [10.0, 5.0], [10.0, 5.0], [10.0, 5.0], [10.0, 5.0], [10.0, 5.0]],
            dtype=np.float32,
        )

        try:
            canonicalize_robot_gaze_wam_zarr(
                input_path=str(raw_path),
                output_path=str(Path(tmpdir) / "canonical_robot.zarr"),
                camera_key="front_rgb",
                action_key="future_tcp_pose",
                tcp_pose_key="current_tcp_pose",
                gripper_key="jaw_width",
                gaze_key="eye_pixel_xy",
                gaze_is_normalized=False,
                overwrite=True,
                validate_output=False,
            )
        except ValueError as exc:
            assert "Out-of-frame gaze point" in str(exc)
        else:
            raise AssertionError("Expected out-of-bounds robot gaze to fail by default.")

        clipped_path = Path(tmpdir) / "canonical_robot_clipped.zarr"
        summary = canonicalize_robot_gaze_wam_zarr(
            input_path=str(raw_path),
            output_path=str(clipped_path),
            camera_key="front_rgb",
            action_key="future_tcp_pose",
            tcp_pose_key="current_tcp_pose",
            gripper_key="jaw_width",
            gaze_key="eye_pixel_xy",
            gaze_is_normalized=False,
            gaze_bounds_policy="clip",
            overwrite=True,
            validate_output=False,
        )
        clipped = zarr.open(str(clipped_path), mode="r")["data"]["gaze_xy"][:]
        assert summary["gaze_bounds_policy"] == "clip"
        assert np.allclose(clipped[0], [1.0, 0.25])


def test_canonicalize_robot_gaze_wam_zarr_rejects_multi_column_gripper():
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = _write_noncanonical_robot_zarr(Path(tmpdir) / "raw_robot.zarr")
        data = zarr.open(str(raw_path), mode="a")["data"]
        pose9 = np.asarray(data["future_tcp_pose"][:], dtype=np.float32)
        del data["future_tcp_pose"]
        action10 = np.concatenate([pose9, np.full((pose9.shape[0], 1), 0.02, dtype=np.float32)], axis=-1)
        data.array(
            "future_tcp_pose",
            action10,
            shape=action10.shape,
            dtype=action10.dtype,
        )
        gripper = np.asarray(data["jaw_width"][:], dtype=np.float32)
        del data["jaw_width"]
        bad_gripper = np.stack([gripper, gripper + 0.01], axis=-1)
        data.array(
            "jaw_width",
            bad_gripper,
            shape=bad_gripper.shape,
            dtype=bad_gripper.dtype,
        )

        try:
            canonicalize_robot_gaze_wam_zarr(
                input_path=str(raw_path),
                output_path=str(Path(tmpdir) / "canonical_robot.zarr"),
                camera_key="front_rgb",
                action_key="future_tcp_pose",
                tcp_pose_key="current_tcp_pose",
                gripper_key="jaw_width",
                gaze_key="eye_pixel_xy",
                gaze_is_normalized=False,
                overwrite=True,
                validate_output=False,
            )
        except ValueError as exc:
            assert "jaw_width must provide exactly one gripper scalar" in str(exc)
        else:
            raise AssertionError("Expected multi-column canonicalizer gripper input to fail.")


def test_canonicalize_robot_gaze_wam_zarr_rejects_nonfinite_gripper():
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = _write_noncanonical_robot_zarr(Path(tmpdir) / "raw_robot.zarr")
        data = zarr.open(str(raw_path), mode="a")["data"]
        data["jaw_width"][2] = np.nan

        try:
            canonicalize_robot_gaze_wam_zarr(
                input_path=str(raw_path),
                output_path=str(Path(tmpdir) / "canonical_robot.zarr"),
                camera_key="front_rgb",
                action_key="future_tcp_pose",
                tcp_pose_key="current_tcp_pose",
                gripper_key="jaw_width",
                gaze_key="eye_pixel_xy",
                gaze_is_normalized=False,
                overwrite=True,
                validate_output=False,
            )
        except ValueError as exc:
            assert "jaw_width must contain only finite gripper scalar values" in str(exc)
        else:
            raise AssertionError("Expected non-finite canonicalizer gripper input to fail.")


def test_canonicalize_robot_gaze_wam_zarr_rejects_nonfinite_action_or_tcp():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        raw_action_path = _write_noncanonical_robot_zarr(root / "raw_action_nan.zarr")
        raw_action_data = zarr.open(str(raw_action_path), mode="a")["data"]
        raw_action_data["future_tcp_pose"][1, 0] = np.nan

        try:
            canonicalize_robot_gaze_wam_zarr(
                input_path=str(raw_action_path),
                output_path=str(root / "canonical_action_nan.zarr"),
                camera_key="front_rgb",
                action_key="future_tcp_pose",
                tcp_pose_key="current_tcp_pose",
                gripper_key="jaw_width",
                gaze_key="eye_pixel_xy",
                gaze_is_normalized=False,
                overwrite=True,
                validate_output=False,
            )
        except ValueError as exc:
            assert "future_tcp_pose must contain only finite values" in str(exc)
        else:
            raise AssertionError("Expected non-finite canonicalizer action input to fail.")

        raw_tcp_path = _write_noncanonical_robot_zarr(root / "raw_tcp_inf.zarr")
        raw_tcp_data = zarr.open(str(raw_tcp_path), mode="a")["data"]
        raw_tcp_data["current_tcp_pose"][1, 0] = np.inf

        try:
            canonicalize_robot_gaze_wam_zarr(
                input_path=str(raw_tcp_path),
                output_path=str(root / "canonical_tcp_inf.zarr"),
                camera_key="front_rgb",
                action_key="future_tcp_pose",
                tcp_pose_key="current_tcp_pose",
                gripper_key="jaw_width",
                gaze_key="eye_pixel_xy",
                gaze_is_normalized=False,
                overwrite=True,
                validate_output=False,
            )
        except ValueError as exc:
            assert "current_tcp_pose must contain only finite values" in str(exc)
        else:
            raise AssertionError("Expected non-finite canonicalizer TCP input to fail.")


def test_prepare_robot_gaze_wam_zarr_pipeline_autoinfers_and_previews():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        raw_path = _write_noncanonical_robot_zarr(root / "raw_robot.zarr")
        raw_data = zarr.open(str(raw_path), mode="a")["data"]
        has_action_abs = np.asarray([1, 1, 0, 1, 1, 1], dtype=np.uint8)
        raw_data.array(
            "has_action_abs",
            has_action_abs,
            shape=has_action_abs.shape,
            dtype=has_action_abs.dtype,
        )
        canonical_path = root / "prepared_robot.zarr"
        preview_dir = root / "prepared_preview"
        report_path = root / "prepare_report.json"

        summary = prepare_robot_gaze_wam_zarr(
            input_path=str(raw_path),
            output_path=str(canonical_path),
            report_json=str(report_path),
            preview_dir=str(preview_dir),
            gaze_is_normalized=False,
            overwrite=True,
            n_obs_steps=2,
            action_horizon=3,
            image_size=(20, 40),
            heatmap_token_grid=(4, 4),
        )

        assert summary["ok"] is True
        assert summary["key_map"] == {
            "camera_key": "front_rgb",
            "action_key": "future_tcp_pose",
            "tcp_pose_key": "current_tcp_pose",
            "gripper_key": "jaw_width",
            "gaze_key": "eye_pixel_xy",
            "heatmap_key": None,
            "timestamp_key": "sensor_time",
            "image_timestamp_key": "image_timestamp",
            "robot_state_timestamp_key": "robot_state_timestamp",
            "action_timestamp_key": "action_timestamp",
            "gaze_timestamp_key": "gaze_timestamp",
        }
        assert summary["canonicalize"]["validated"] is False
        assert "scripts/canonicalize_robot_gaze_wam_zarr.py" in summary["canonicalizer_command"]
        assert str(raw_path) in summary["canonicalizer_command"]
        assert str(canonical_path) in summary["canonicalizer_command"]
        assert "--no-gaze-is-normalized" in summary["canonicalizer_command"]
        assert summary["validation"]["valid"] is True
        assert summary["canonicalize"]["presence_mask_keys"] == [
            "has_action_abs",
            "has_gaze_condition",
            "has_gaze_label",
            "has_heatmap_image",
        ]
        assert summary["validation"]["presence_masks"]["has_action_abs"]["false_count"] == 1
        assert sorted(summary["validation"]["timestamps"]["keys"]) == [
            "action_timestamp",
            "gaze_timestamp",
            "image_timestamp",
            "robot_state_timestamp",
            "timestamp",
        ]
        assert summary["validation"]["timestamps"]["alignment"]["gaze_timestamp"]["max_abs_delta"] < 0.004
        assert summary["preview"]["dataset_type"] == "robot"
        assert report_path.exists()
        assert Path(summary["preview"]["paths"]["overlay"]).exists()
        prepared = zarr.open(str(canonical_path), mode="r")
        assert np.allclose(prepared["data/timestamp"][:], np.arange(6, dtype=np.float64) * 0.05)

        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["ok"] is True
        assert payload["key_map"]["timestamp_key"] == "sensor_time"
        assert payload["key_map"]["gaze_timestamp_key"] == "gaze_timestamp"
        assert payload["canonicalizer_command"] == summary["canonicalizer_command"]
        assert payload["validation"]["sample"]["action_shape"] == [3, 10]
        prepared = zarr.open(str(canonical_path), mode="r")
        assert prepared["data/action_abs_tcp"].shape == (6, 10)
        assert np.allclose(prepared["data/has_action_abs"][:], has_action_abs)
        assert "image_timestamp" in prepared["data"]
        assert np.all(prepared["data/gaze_xy"][:] <= 1)


def test_prepare_robot_gaze_wam_zarr_dry_run_resolves_keys_without_writing_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        raw_path = _write_noncanonical_robot_zarr(root / "raw_robot.zarr")
        canonical_path = root / "prepared_robot.zarr"
        report_path = root / "prepare_dry_run_report.json"

        summary = prepare_robot_gaze_wam_zarr(
            input_path=str(raw_path),
            output_path=str(canonical_path),
            report_json=str(report_path),
            gaze_is_normalized=False,
            overwrite=True,
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            dry_run=True,
        )

        assert summary["ok"] is True
        assert summary["dry_run"] is True
        assert summary["canonicalize"] is None
        assert summary["validation"] is None
        assert summary["preview"] is None
        assert summary["key_map"]["camera_key"] == "front_rgb"
        assert summary["key_map"]["action_key"] == "future_tcp_pose"
        assert summary["key_map"]["tcp_pose_key"] == "current_tcp_pose"
        assert summary["key_map"]["gripper_key"] == "jaw_width"
        assert summary["key_map"]["gaze_key"] == "eye_pixel_xy"
        assert "scripts/canonicalize_robot_gaze_wam_zarr.py" in summary["canonicalizer_command"]
        assert str(raw_path) in summary["canonicalizer_command"]
        assert str(canonical_path) in summary["canonicalizer_command"]
        assert "--no-gaze-is-normalized" in summary["canonicalizer_command"]
        assert not canonical_path.exists()
        assert report_path.exists()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["dry_run"] is True
        assert payload["canonicalize"] is None
        assert payload["validation"] is None
        assert payload["key_map"] == summary["key_map"]


def test_prepare_robot_gaze_wam_zarr_rejects_dense_heatmap_without_gaze_point():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        raw_path = _write_noncanonical_robot_heatmap_only_zarr(root / "raw_robot.zarr")
        with pytest.raises(ValueError, match="required robot point gaze key"):
            prepare_robot_gaze_wam_zarr(
                input_path=str(raw_path),
                output_path=str(root / "prepared_robot.zarr"),
                preview_dir=str(root / "prepared_preview"),
                overwrite=True,
                n_obs_steps=2,
                action_horizon=3,
                image_size=(16, 16),
                heatmap_token_grid=(4, 4),
            )


def test_prepare_robot_gaze_wam_zarr_rejects_null_like_explicit_gaze_key():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        raw_path = _write_noncanonical_robot_heatmap_only_zarr(root / "raw_robot.zarr")
        with pytest.raises(ValueError, match="required robot point gaze key"):
            prepare_robot_gaze_wam_zarr(
                input_path=str(raw_path),
                output_path=str(root / "prepared_robot.zarr"),
                gaze_key="None",
                heatmap_key="eye_gaze_heatmap",
                timestamp_key="null",
                overwrite=True,
                n_obs_steps=2,
                action_horizon=3,
                image_size=(16, 16),
                heatmap_token_grid=(4, 4),
                skip_preview=True,
            )


def test_prepare_robot_gaze_wam_zarr_threads_timestamp_validation_options():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        raw_path = _write_noncanonical_robot_zarr(root / "raw_robot.zarr")

        summary = prepare_robot_gaze_wam_zarr(
            input_path=str(raw_path),
            output_path=str(root / "canonical_robot.zarr"),
            timestamp_key="sensor_time",
            gaze_is_normalized=False,
            image_size=(16, 16),
            n_obs_steps=2,
            action_horizon=3,
            heatmap_token_grid=(4, 4),
            timestamp_max_step=0.01,
            overwrite=True,
            skip_preview=True,
        )

        assert summary["ok"] is False
        assert summary["validation"]["timestamps"]["checked"] is True
        assert summary["validation"]["timestamps"]["base_key"] == "timestamp"
        assert summary["validation"]["timestamps"]["intervals"]["timestamp"]["max_step"] > 0.01
        assert any("max_step" in message for message in summary["validation"]["errors"])


def test_gaze_wam_robot_dataset_action_chunk_starts_after_observation():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_linear_action_zarr(Path(tmpdir) / "robot.zarr", length=8)
        dataset = GazeWamRobotDataset(
            dataset_path=str(robot_path),
            n_obs_steps=2,
            action_horizon=3,
            n_latency_steps=0,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            action_padding=False,
        )

        sample = dataset[1]

        assert len(dataset) == 5
        assert torch.allclose(
            sample["action_abs"][:, 0],
            torch.tensor([2.0, 3.0, 4.0]),
        )
        assert torch.allclose(sample["action_base_abs"][0], torch.tensor(1.0))
        assert torch.allclose(
            sample["action"][:, 0],
            torch.tensor([1.0, 2.0, 3.0]),
        )


def test_gaze_wam_robot_dataset_latency_skips_future_action_rows():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_linear_action_zarr(Path(tmpdir) / "robot.zarr", length=8)
        dataset = GazeWamRobotDataset(
            dataset_path=str(robot_path),
            n_obs_steps=2,
            action_horizon=3,
            n_latency_steps=2,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            action_padding=False,
        )

        sample = dataset[1]

        assert len(dataset) == 3
        assert sample["action_abs"].shape == (3, 10)
        assert sample["action_base_abs"].shape == (10,)
        assert torch.allclose(
            sample["action_abs"][:, 0],
            torch.tensor([4.0, 5.0, 6.0]),
        )
        assert torch.allclose(sample["action_base_abs"][0], torch.tensor(1.0))
        assert torch.allclose(
            sample["action"][:, 9],
            sample["action_abs"][:, 9],
        )


def test_gaze_wam_robot_dataset_composes_pose_only_action_with_gripper():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_linear_action_pose_only_zarr(Path(tmpdir) / "robot_pose9.zarr", length=8)
        dataset = GazeWamRobotDataset(
            dataset_path=str(robot_path),
            n_obs_steps=2,
            action_horizon=3,
            n_latency_steps=2,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            action_padding=False,
        )

        sample = dataset[1]

        assert sample["action_abs"].shape == (3, 10)
        assert sample["action"].shape == (3, 10)
        assert torch.allclose(
            sample["action_abs"][:, 0],
            torch.tensor([4.0, 5.0, 6.0]),
        )
        assert torch.allclose(
            sample["action_abs"][:, 9],
            torch.tensor([0.04, 0.05, 0.06]),
        )
        assert torch.allclose(sample["action"][:, 9], sample["action_abs"][:, 9])

        summary = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            n_latency_steps=2,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_dataset_sample=True,
        )
        assert summary["valid"] is True
        assert summary["robot_numeric"]["action_abs"]["shape"] == [8, 9]
        assert summary["sample"]["action_shape"] == [3, 10]
        assert summary["sample"]["action_roundtrip_max_error"] < 1e-6


def test_gaze_wam_robot_dataset_rejects_multi_column_gripper_for_pose_only_action():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_linear_action_pose_only_zarr(Path(tmpdir) / "robot_pose9_bad_gripper.zarr", length=8)
        data = zarr.open(str(robot_path), mode="a")["data"]
        gripper = np.asarray(data["gripper_width"][:], dtype=np.float32)
        del data["gripper_width"]
        bad_gripper = np.concatenate([gripper.reshape(-1, 1), gripper.reshape(-1, 1)], axis=-1)
        data.array("gripper_width", bad_gripper, shape=bad_gripper.shape, dtype=bad_gripper.dtype)

        summary = validate_gaze_wam_zarr(
            dataset_path=str(robot_path),
            dataset_type="robot",
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            check_dataset_sample=False,
        )
        assert summary["valid"] is False
        assert any("gripper_width must be [N] or [N,1]" in message for message in summary["errors"])

        dataset = GazeWamRobotDataset(
            dataset_path=str(robot_path),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            action_padding=False,
        )
        try:
            dataset[0]
        except ValueError as exc:
            assert "requires exactly one gripper scalar" in str(exc)
        else:
            raise AssertionError("Expected multi-column gripper_width to fail for pose-only action.")


def test_gaze_wam_dataset_resizes_images_and_normalizes_pixel_gaze_from_source_size():
    with tempfile.TemporaryDirectory() as tmpdir:
        pixel_gaze = np.asarray(
            [
                [24.0, 16.0],
                [12.0, 8.0],
                [36.0, 24.0],
                [0.0, 0.0],
                [48.0, 32.0],
                [6.0, 4.0],
            ],
            dtype=np.float32,
        )
        robot_path = _write_gaze_wam_zarr(
            Path(tmpdir) / "robot.zarr",
            include_action=True,
            image_hw=(32, 48),
        )
        _replace_zarr_array(
            zarr.open(str(robot_path), mode="a")["data"],
            "gaze_xy",
            pixel_gaze,
        )
        dataset = GazeWamRobotDataset(
            dataset_path=str(robot_path),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            gaze_is_normalized=False,
            action_padding=True,
        )

        sample = dataset[0]

        assert sample["obs"]["camera0_rgb"].shape == (2, 3, 16, 16)
        assert torch.allclose(
            sample["gaze_xy"],
            torch.tensor([0.5, 0.5], dtype=torch.float32),
        )

        letterbox_dataset = GazeWamRobotDataset(
            dataset_path=str(robot_path),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            image_resize_mode="letterbox",
            heatmap_token_grid=(4, 4),
            gaze_is_normalized=False,
            action_padding=True,
        )
        assert torch.allclose(
            letterbox_dataset[0]["gaze_xy"],
            torch.tensor([0.5, 0.5], dtype=torch.float32),
        )


def test_gaze_wam_open_dataset_accepts_dense_heatmap_with_masked_point_gaze():
    with tempfile.TemporaryDirectory() as tmpdir:
        open_path = _write_gaze_wam_zarr(
            Path(tmpdir) / "open.zarr",
            include_action=False,
            image_hw=(32, 32),
        )
        _replace_zarr_array(
            zarr.open(str(open_path), mode="a")["data"],
            "has_gaze_label",
            np.zeros(6, dtype=np.bool_),
        )
        dataset = GazeWamOpenDataset(
            dataset_path=str(open_path),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            gaze_key="gaze_xy",
            heatmap_key="gaze_heatmap",
        )

        sample = dataset[1]

        assert sample["obs"]["camera0_rgb"].shape == (2, 3, 16, 16)
        assert sample["gaze_xy"].shape == (2,)
        assert torch.allclose(sample["gaze_xy"], torch.zeros(2))
        assert sample["has_gaze_label"].item() is False
        assert sample["heatmap"].shape == (1, 16, 1)
        assert sample["heatmap"].max() > 0
        assert sample["has_heatmap"].item() is True
        assert sample["use_gaze_condition"].item() is False
        assert sample["heatmap_image"].shape == (1, 16, 16)
        assert sample["heatmap_image"].max() > 0


def test_gaze_wam_dataset_validation_split_is_episode_level_and_stable():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_multi_episode_gaze_wam_zarr(
            Path(tmpdir) / "robot.zarr",
            include_action=True,
            episode_lengths=(4, 5, 6, 7),
        )
        open_path = _write_multi_episode_gaze_wam_zarr(
            Path(tmpdir) / "open.zarr",
            include_action=False,
            episode_lengths=(4, 5, 6, 7),
        )
        robot_dataset = GazeWamRobotDataset(
            dataset_path=str(robot_path),
            n_obs_steps=2,
            action_horizon=2,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            action_padding=False,
            seed=123,
            val_ratio=0.5,
        )
        robot_val = robot_dataset.get_validation_dataset()
        repeated_robot_dataset = GazeWamRobotDataset(
            dataset_path=str(robot_path),
            n_obs_steps=2,
            action_horizon=2,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            action_padding=False,
            seed=123,
            val_ratio=0.5,
        )
        open_dataset = GazeWamOpenDataset(
            dataset_path=str(open_path),
            n_obs_steps=2,
            action_horizon=2,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            action_padding=False,
            seed=123,
            val_ratio=0.5,
        )
        open_val = open_dataset.get_validation_dataset()

        train_episodes = set(robot_dataset.indices[:, 3].tolist())
        val_episodes = set(robot_val.indices[:, 3].tolist())
        open_train_episodes = set(open_dataset.indices[:, 3].tolist())
        open_val_episodes = set(open_val.indices[:, 3].tolist())

        assert train_episodes
        assert val_episodes
        assert train_episodes.isdisjoint(val_episodes)
        assert train_episodes | val_episodes == {0, 1, 2, 3}
        assert np.array_equal(robot_dataset.val_mask, repeated_robot_dataset.val_mask)
        assert np.array_equal(robot_dataset.indices, repeated_robot_dataset.indices)
        assert open_train_episodes.isdisjoint(open_val_episodes)
        assert open_train_episodes | open_val_episodes == {0, 1, 2, 3}
        assert len(robot_dataset) + len(robot_val) == sum(
            max(0, length - 2) for length in (4, 5, 6, 7)
        )


def test_gaze_wam_dataset_rejects_invalid_val_ratio():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_multi_episode_gaze_wam_zarr(
            Path(tmpdir) / "robot.zarr",
            include_action=True,
            episode_lengths=(4, 5, 6, 7),
        )

        valid_dataset = GazeWamRobotDataset(
            dataset_path=str(robot_path),
            n_obs_steps=2,
            action_horizon=2,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            action_padding=False,
            seed="123",
            val_ratio="0.5",
        )
        assert valid_dataset.seed == 123
        assert valid_dataset.val_ratio == 0.5
        assert normalize_gaze_wam_unit_interval_float_field("val_ratio", "0.25") == 0.25

        for bad_ratio in (-0.1, 1.0, float("nan"), True, False, "oops"):
            try:
                GazeWamRobotDataset(
                    dataset_path=str(robot_path),
                    n_obs_steps=2,
                    action_horizon=2,
                    image_size=(16, 16),
                    heatmap_token_grid=(4, 4),
                    action_padding=False,
                    val_ratio=bad_ratio,
                )
            except ValueError as exc:
                assert "val_ratio must be in [0, 1)" in str(exc)
            else:
                raise AssertionError(f"Expected invalid val_ratio={bad_ratio!r} to fail.")

        for bad_seed in (True, 1.5, float("inf"), "1.5", -1):
            try:
                GazeWamRobotDataset(
                    dataset_path=str(robot_path),
                    n_obs_steps=2,
                    action_horizon=2,
                    image_size=(16, 16),
                    heatmap_token_grid=(4, 4),
                    action_padding=False,
                    seed=bad_seed,
                    val_ratio=0.5,
                )
            except ValueError as exc:
                assert "seed must be a non-negative integer" in str(exc)
            else:
                raise AssertionError(f"Expected invalid seed={bad_seed!r} to fail.")


def test_gaze_wam_dataset_val_ratio_zero_uses_all_episodes_for_training():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_multi_episode_gaze_wam_zarr(
            Path(tmpdir) / "robot.zarr",
            include_action=True,
            episode_lengths=(4, 5, 6),
        )
        dataset = GazeWamRobotDataset(
            dataset_path=str(robot_path),
            n_obs_steps=2,
            action_horizon=2,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            action_padding=False,
            val_ratio=0.0,
        )
        val_dataset = dataset.get_validation_dataset()

        assert set(dataset.indices[:, 3].tolist()) == {0, 1, 2}
        assert len(dataset) == sum(length - 2 for length in (4, 5, 6))
        assert len(val_dataset) == 0


def test_convert_open_gaze_manifest_point_labels_to_zarr():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_dir = root / "images"
        image_dir.mkdir()
        rows = []
        for idx in range(3):
            image = np.zeros((10, 20, 3), dtype=np.uint8)
            image[..., 0] = idx * 20
            image_path = image_dir / f"frame_{idx}.png"
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            rows.append(
                {
                    "episode_id": "episode_a" if idx < 2 else "episode_b",
                    "image_path": f"images/frame_{idx}.png",
                    "timestamp": f"{idx * 0.05:.3f}",
                    "gaze_x": "10",
                    "gaze_y": "5",
                    "image_width": "20",
                    "image_height": "10",
                }
            )
        manifest_path = root / "manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            f.write("episode_id,image_path,timestamp,gaze_x,gaze_y,image_width,image_height\n")
            for row in rows:
                f.write(
                    f"{row['episode_id']},{row['image_path']},{row['timestamp']},{row['gaze_x']},"
                    f"{row['gaze_y']},{row['image_width']},{row['image_height']}\n"
                )
        output_path = root / "open_point.zarr"

        summary = convert_open_gaze_manifest(
            manifest_path=str(manifest_path),
            output_path=str(output_path),
            image_size=(16, 16),
            gaze_is_normalized=False,
            overwrite=True,
        )

        assert summary["num_frames"] == 3
        assert summary["num_episodes"] == 2
        assert summary["dataset_type"] == "open"
        assert summary["label_mode"] == "point"
        assert summary["image_resize_mode"] == "stretch"
        zroot = zarr.open(str(output_path), mode="r")
        assert zroot["data/camera0_rgb"].shape == (3, 16, 16, 3)
        assert zroot["data/gaze_xy"].shape == (3, 2)
        assert zroot["meta"].attrs["dataset_type"] == "open"
        assert zroot["meta"].attrs["image_resize_mode"] == "stretch"
        assert zroot["data/timestamp"].shape == (3,)
        assert np.allclose(zroot["data/timestamp"][:], np.asarray([0.0, 0.05, 0.1]))
        assert np.allclose(zroot["data/gaze_xy"][0], np.asarray([0.5, 0.5], dtype=np.float32))
        assert np.allclose(zroot["meta/episode_ends"][:], np.asarray([2, 3], dtype=np.int64))

        dataset = GazeWamOpenDataset(
            dataset_path=str(output_path),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
        )
        sample = dataset[1]
        assert sample["has_gaze_label"].item() is True
        assert torch.allclose(sample["gaze_xy"], torch.tensor([0.5, 0.5]))
        assert sample["heatmap"].shape == (1, 16, 1)


def test_convert_open_gaze_manifest_gaze_bounds_policy():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_dir = root / "images"
        image_dir.mkdir()
        for idx in range(2):
            image = np.zeros((10, 20, 3), dtype=np.uint8)
            cv2.imwrite(str(image_dir / f"frame_{idx}.png"), image)
        manifest_path = root / "manifest.csv"
        manifest_path.write_text(
            "\n".join(
                [
                    "episode_id,image_path,gaze_x,gaze_y,image_width,image_height",
                    "episode_a,images/frame_0.png,10,5,20,10",
                    "episode_a,images/frame_1.png,30,5,20,10",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        try:
            convert_open_gaze_manifest(
                manifest_path=str(manifest_path),
                output_path=str(root / "open_error.zarr"),
                image_size=(16, 16),
                gaze_is_normalized=False,
                overwrite=True,
            )
        except ValueError as exc:
            assert "Out-of-frame gaze point" in str(exc)
        else:
            raise AssertionError("Expected out-of-bounds open gaze to fail by default.")

        clipped_summary = convert_open_gaze_manifest(
            manifest_path=str(manifest_path),
            output_path=str(root / "open_clip.zarr"),
            image_size=(16, 16),
            gaze_is_normalized=False,
            gaze_bounds_policy="clip",
            overwrite=True,
        )
        clipped = zarr.open(str(root / "open_clip.zarr"), mode="r")["data/gaze_xy"][:]
        assert clipped_summary["gaze_bounds_policy"] == "clip"
        assert np.allclose(clipped[1], [1.0, 0.5])

        heatmap_dir = root / "heatmaps"
        heatmap_dir.mkdir()
        for idx in range(2):
            cv2.imwrite(str(heatmap_dir / f"heatmap_{idx}.png"), np.ones((10, 20), dtype=np.uint8))
        manifest_path.write_text(
            "\n".join(
                [
                    "episode_id,image_path,gaze_x,gaze_y,image_width,image_height,heatmap_path",
                    "episode_a,images/frame_0.png,10,5,20,10,heatmaps/heatmap_0.png",
                    "episode_a,images/frame_1.png,30,5,20,10,heatmaps/heatmap_1.png",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="label_mode='heatmap' requires"):
            convert_open_gaze_manifest(
                manifest_path=str(manifest_path),
                output_path=str(root / "open_drop_heatmap.zarr"),
                image_size=(16, 16),
                gaze_is_normalized=False,
                gaze_bounds_policy="drop",
                label_mode="heatmap",
                overwrite=True,
            )


def test_convert_open_gaze_manifest_rejects_non_stretch_resize_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_dir = root / "images"
        image_dir.mkdir()
        cv2.imwrite(str(image_dir / "frame.png"), np.zeros((10, 20, 3), dtype=np.uint8))
        manifest_path = root / "manifest.csv"
        manifest_path.write_text(
            "\n".join(
                [
                    "episode_id,image_path,gaze_x,gaze_y,image_width,image_height",
                    "episode,images/frame.png,10,5,20,10",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        try:
            convert_open_gaze_manifest(
                manifest_path=str(manifest_path),
                output_path=str(root / "open.zarr"),
                image_size=(16, 16),
                gaze_is_normalized=False,
                image_resize_mode="letterbox",
                overwrite=True,
            )
        except ValueError as exc:
            assert "stretch resize" in str(exc)
            assert "letterbox" in str(exc)
        else:
            raise AssertionError("Expected unsupported image_resize_mode to fail.")


def test_convert_open_gaze_manifest_dense_heatmap_labels_to_zarr():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_dir = root / "images"
        heatmap_dir = root / "heatmaps"
        image_dir.mkdir()
        heatmap_dir.mkdir()
        manifest_path = root / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            for idx in range(3):
                image = np.zeros((12, 12, 3), dtype=np.uint8)
                image[..., 1] = 30 + idx
                image_path = image_dir / f"frame_{idx}.png"
                cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                heatmap = np.zeros((12, 12), dtype=np.uint8)
                heatmap[3:8, 4:9] = 255
                heatmap_path = heatmap_dir / f"heatmap_{idx}.png"
                cv2.imwrite(str(heatmap_path), heatmap)
                f.write(
                    json.dumps(
                        {
                            "episode_id": "episode_a",
                            "image_path": str(image_path.relative_to(root)),
                            "gaze_x": 0.5,
                            "gaze_y": 0.5,
                            "heatmap_path": str(heatmap_path.relative_to(root)),
                        }
                    )
                    + "\n"
                )
        output_path = root / "open_heatmap.zarr"

        summary = convert_open_gaze_manifest(
            manifest_path=str(manifest_path),
            output_path=str(output_path),
            image_size=(16, 16),
            label_mode="heatmap",
            root_dir=str(root),
            overwrite=True,
        )

        assert summary["label_mode"] == "heatmap"
        assert summary["presence_mask_keys"] == ["has_gaze_label", "has_heatmap_image"]
        zroot = zarr.open(str(output_path), mode="r")
        assert zroot["data/camera0_rgb"].shape == (3, 16, 16, 3)
        assert zroot["data/gaze_xy"].shape == (3, 2)
        assert zroot["data/gaze_heatmap"].shape == (3, 16, 16)
        assert zroot["data/has_gaze_label"][:].tolist() == [True, True, True]
        assert zroot["data/has_heatmap_image"].shape == (3,)
        assert zroot["data/has_heatmap_image"][:].tolist() == [True, True, True]
        assert zroot["meta"].attrs["presence_mask_keys"] == [
            "has_gaze_label",
            "has_heatmap_image",
        ]
        assert np.isclose(float(zroot["data/gaze_heatmap"][:].max()), 1.0, atol=1e-6)

        dataset = GazeWamOpenDataset(
            dataset_path=str(output_path),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            heatmap_key="gaze_heatmap",
        )
        sample = dataset[1]
        assert sample["has_gaze_label"].item() is True
        assert torch.allclose(sample["gaze_xy"], torch.tensor([0.5, 0.5]))
        assert sample["heatmap"].max() > 0
        assert sample["has_heatmap_image"].item() is True

        try:
            convert_open_gaze_manifest(
                manifest_path=str(manifest_path),
                output_path=str(root / "bad_point_null_key.zarr"),
                image_size=(16, 16),
                label_mode="point",
                root_dir=str(root),
                gaze_key="null",
                overwrite=True,
            )
        except ValueError as exc:
            assert "non-null gaze_key" in str(exc)
        else:
            raise AssertionError("Expected point-label conversion with null gaze_key to fail.")


def test_export_video_gaze_manifest_to_open_zarr():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        video_path = root / "clip.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (24, 16),
        )
        for idx in range(4):
            frame_rgb = np.zeros((16, 24, 3), dtype=np.uint8)
            frame_rgb[..., 0] = idx * 40
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        writer.release()

        metadata_path = root / "video_gaze.csv"
        with metadata_path.open("w", encoding="utf-8", newline="") as f:
            f.write("episode_id,video_path,frame_idx,timestamp,gaze_x,gaze_y,image_width,image_height\n")
            f.write("clip_a,clip.mp4,0,0.0,12,8,24,16\n")
            f.write("clip_a,clip.mp4,2,0.4,6,4,24,16\n")

        manifest_path = root / "open_video_manifest.csv"
        frames_dir = root / "frames"
        output_zarr = root / "open_video.zarr"
        summary = export_video_gaze_manifest(
            metadata_path=str(metadata_path),
            output_manifest=str(manifest_path),
            frames_dir=str(frames_dir),
            root_dir=str(root),
            image_size=(16, 24),
            gaze_is_normalized=False,
            overwrite=True,
            output_zarr=str(output_zarr),
            zarr_image_size=(16, 16),
        )

        assert summary["num_frames"] == 2
        assert summary["num_videos"] == 1
        assert summary["image_resize_mode"] == "stretch"
        assert summary["zarr"]["label_mode"] == "point"
        assert summary["zarr"]["image_resize_mode"] == "stretch"
        manifest_text = manifest_path.read_text(encoding="utf-8")
        assert "image_path" in manifest_text
        assert "timestamp" in manifest_text
        assert "source_video" in manifest_text
        assert len(list(frames_dir.glob("clip_a/*.png"))) == 2

        zroot = zarr.open(str(output_zarr), mode="r")
        assert zroot["data/camera0_rgb"].shape == (2, 16, 16, 3)
        assert np.allclose(zroot["data/timestamp"][:], np.asarray([0.0, 0.4]))
        assert np.allclose(zroot["data/gaze_xy"][0], np.asarray([0.5, 0.5], dtype=np.float32))

        dataset = GazeWamOpenDataset(
            dataset_path=str(output_zarr),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
        )
        sample = dataset[1]
        assert sample["obs"]["camera0_rgb"].shape == (2, 3, 16, 16)
        assert sample["has_gaze_label"].item() is True
        assert sample["heatmap"].shape == (1, 16, 1)


def test_prepare_open_gaze_wam_zarr_from_manifest_validates_and_previews():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_dir = root / "images"
        image_dir.mkdir()
        manifest_path = root / "manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            f.write("episode_id,image_path,gaze_x,gaze_y,image_width,image_height\n")
            for idx in range(3):
                image = np.zeros((12, 16, 3), dtype=np.uint8)
                image[..., 2] = 50 + idx
                image_path = image_dir / f"frame_{idx}.png"
                cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                f.write(f"open_a,{image_path.relative_to(root)},{idx + 1},6,16,12\n")

        output_zarr = root / "prepared_open.zarr"
        preview_dir = root / "prepared_open_preview"
        report_path = root / "prepared_open_report.json"
        summary = prepare_open_gaze_wam_zarr(
            manifest_path=str(manifest_path),
            output_zarr=str(output_zarr),
            report_json=str(report_path),
            preview_dir=str(preview_dir),
            root_dir=str(root),
            gaze_is_normalized=False,
            image_size=(16, 16),
            n_obs_steps=2,
            action_horizon=3,
            heatmap_token_grid=(4, 4),
            overwrite=True,
        )

        assert summary["ok"] is True
        assert summary["mode"] == "manifest"
        assert summary["convert"]["label_mode"] == "point"
        assert summary["validation"]["valid"] is True
        assert summary["preview"]["dataset_type"] == "open"
        assert Path(summary["preview"]["paths"]["overlay"]).exists()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["validation"]["sample"]["has_gaze_label"] is True


def test_prepare_open_gaze_wam_zarr_manifest_dry_run_does_not_write_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        manifest_path = root / "manifest.csv"
        manifest_path.write_text(
            "episode_id,image_path,gaze_x,gaze_y,image_width,image_height\n"
            "open_a,images/frame_0.png,0.5,0.5,16,16\n",
            encoding="utf-8",
        )
        output_zarr = root / "prepared_open.zarr"
        preview_dir = root / "prepared_open_preview"
        report_path = root / "prepared_open_dry_run_report.json"

        summary = prepare_open_gaze_wam_zarr(
            manifest_path=str(manifest_path),
            output_zarr=str(output_zarr),
            report_json=str(report_path),
            preview_dir=str(preview_dir),
            root_dir=str(root),
            image_size=(16, 16),
            n_obs_steps=2,
            action_horizon=3,
            heatmap_token_grid=(4, 4),
            dry_run=True,
        )

        assert summary["ok"] is True
        assert summary["dry_run"] is True
        assert summary["mode"] == "manifest"
        assert summary["convert"] is None
        assert summary["validation"] is None
        assert summary["preview"] is None
        assert "convert" in summary["planned_commands"]
        assert "validate_gaze_wam_zarr.py" in summary["planned_commands"]["validation"]
        assert "preview_gaze_wam_dataset.py" in summary["planned_commands"]["preview"]
        assert str(manifest_path) in summary["planned_commands"]["convert"]
        assert str(output_zarr) in summary["planned_commands"]["convert"]
        assert not output_zarr.exists()
        assert not preview_dir.exists()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["dry_run"] is True
        assert payload["convert"] is None
        assert payload["planned_commands"] == summary["planned_commands"]


def test_prepare_open_gaze_wam_zarr_heatmap_manifest_requires_point_gaze():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_dir = root / "images"
        heatmap_dir = root / "heatmaps"
        image_dir.mkdir()
        heatmap_dir.mkdir()
        manifest_path = root / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            for idx in range(3):
                image = np.zeros((12, 12, 3), dtype=np.uint8)
                image_path = image_dir / f"frame_{idx}.png"
                cv2.imwrite(str(image_path), image)
                heatmap = np.zeros((12, 12), dtype=np.uint8)
                heatmap[2:8, 3:9] = 255
                heatmap_path = heatmap_dir / f"heatmap_{idx}.png"
                cv2.imwrite(str(heatmap_path), heatmap)
                f.write(
                    json.dumps(
                        {
                            "episode_id": "episode_heatmap",
                            "image_path": str(image_path.relative_to(root)),
                            "gaze_x": 0.5,
                            "gaze_y": 0.5,
                            "heatmap_path": str(heatmap_path.relative_to(root)),
                        }
                    )
                    + "\n"
                )

        output_zarr = root / "prepared_open_heatmap.zarr"
        summary = prepare_open_gaze_wam_zarr(
            manifest_path=str(manifest_path),
            output_zarr=str(output_zarr),
            root_dir=str(root),
            label_mode="heatmap",
            heatmap_key="gaze_heatmap",
            image_size=(16, 16),
            n_obs_steps=2,
            action_horizon=3,
            heatmap_token_grid=(4, 4),
            overwrite=True,
        )

        assert summary["ok"] is True
        assert summary["convert"]["label_mode"] == "heatmap"
        assert summary["convert"]["presence_mask_keys"] == [
            "has_gaze_label",
            "has_heatmap_image",
        ]
        assert summary["validation"]["presence_masks"]["has_heatmap_image"]["true_count"] == 3
        assert summary["validation"]["sample"]["has_gaze_label"] is True
        assert summary["preview"]["has_gaze_label"] is True
        zroot = zarr.open(str(output_zarr), mode="r")
        assert "gaze_xy" in zroot["data"]
        assert "gaze_heatmap" in zroot["data"]
        assert zroot["data/has_heatmap_image"][:].tolist() == [True, True, True]


def test_prepare_open_gaze_wam_zarr_threads_timestamp_validation_options():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_dir = root / "images"
        image_dir.mkdir()
        manifest_path = root / "manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            f.write("episode_id,image_path,gaze_x,gaze_y,image_width,image_height\n")
            for idx in range(3):
                image = np.zeros((12, 16, 3), dtype=np.uint8)
                image_path = image_dir / f"frame_{idx}.png"
                cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                f.write(f"open_a,{image_path.relative_to(root)},{idx + 1},6,16,12\n")

        summary = prepare_open_gaze_wam_zarr(
            manifest_path=str(manifest_path),
            output_zarr=str(root / "prepared_open.zarr"),
            root_dir=str(root),
            gaze_is_normalized=False,
            image_size=(16, 16),
            n_obs_steps=2,
            action_horizon=3,
            heatmap_token_grid=(4, 4),
            require_timestamps=True,
            overwrite=True,
            skip_preview=True,
        )

        assert summary["ok"] is False
        assert summary["validation"]["valid"] is False
        assert any("timestamp" in message for message in summary["validation"]["errors"])


def test_inspect_open_video_gaze_metadata_suggests_nested_key_map():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        metadata_path = root / "ego_like.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "annotations": [
                        {
                            "clip": {"path": "clip_a.mp4", "uid": "clip_a"},
                            "frame": {"number": 3, "width": 640, "height": 480},
                            "gaze": {"point": [320, 240]},
                            "split": "train",
                        },
                        {
                            "clip": {"path": "clip_b.mp4", "uid": "clip_b"},
                            "frame": {"number": 7, "width": 640, "height": 480},
                            "gaze": {"point": [120, 80]},
                            "split": "val",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        summary = inspect_open_video_gaze_metadata(
            metadata_path=str(metadata_path),
            sample_rows=10,
            top_k=3,
        )

        expected_key_map = {
            "episode_id": "clip.uid",
            "frame_idx": "frame.number",
            "gaze_x": "gaze.point.0",
            "gaze_y": "gaze.point.1",
            "image_height": "frame.height",
            "image_width": "frame.width",
            "video_path": "clip.path",
        }
        assert summary["num_rows"] == 2
        assert summary["suggested_key_map"] == expected_key_map
        assert summary["adapter_args"] == [
            "--key-map",
            json.dumps(expected_key_map, sort_keys=True, separators=(",", ":")),
        ]
        assert summary["mapping_status"]["ready_for_metadata_adapter"] is True
        assert summary["mapping_status"]["missing_required_roles"] == []
        assert summary["mapping_status"]["has_frame_idx"] is True
        assert summary["mapping_status"]["has_image_size"] is True
        assert "scripts/adapt_open_video_gaze_metadata.py" in summary["adapter_command_template"]
        assert "--metadata" in summary["adapter_command_template"]
        assert "ego_like.json" in summary["adapter_command_template"]
        assert "<canonical_open_video_gaze.csv>" in summary["adapter_command_template"]
        assert "<open_gaze_wam.zarr>" in summary["adapter_command_template"]
        assert summary["suggested_filter_candidates"][0]["filter_args"] == [
            "split=train",
            "split=val",
        ]
        assert not summary["warnings"]


def test_inspect_open_video_gaze_metadata_reports_missing_adapter_mapping():
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = Path(tmpdir) / "missing_video.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "annotations": [
                        {
                            "frame": {"number": 3},
                            "gaze": {"point": [0.5, 0.4]},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        summary = inspect_open_video_gaze_metadata(
            metadata_path=str(metadata_path),
            sample_rows=10,
            top_k=3,
        )

        assert summary["suggested_key_map"] is None
        assert summary["adapter_args"] is None
        assert summary["adapter_command_template"] is None
        assert summary["mapping_status"]["ready_for_metadata_adapter"] is False
        assert "video_path" in summary["mapping_status"]["missing_required_roles"]
        assert summary["warnings"]


def test_adapt_open_video_gaze_metadata_nested_mapping_and_filters():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        metadata_path = root / "ego_like.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "annotations": [
                        {
                            "clip": {"path": "clip_a.mp4", "uid": "clip_a"},
                            "frame": {"number": 0, "width": 640, "height": 480},
                            "gaze": {"point": [320, 240]},
                            "split": "train",
                        },
                        {
                            "clip": {"path": "clip_b.mp4", "uid": "clip_b"},
                            "frame": {"number": 2, "width": 640, "height": 480},
                            "gaze": {"point": [10, 20]},
                            "split": "val",
                        },
                        {
                            "clip": {"path": "clip_a.mp4", "uid": "clip_a"},
                            "frame": {"number": 1, "width": 640, "height": 480},
                            "gaze": {"point": [160, 120]},
                            "split": "train",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        output_metadata = root / "canonical.csv"

        summary = adapt_open_video_gaze_metadata(
            metadata_path=str(metadata_path),
            output_metadata=str(output_metadata),
            key_map={
                "video_path": "clip.path",
                "episode_id": "clip.uid",
                "frame_idx": "frame.number",
                "gaze_x": "gaze.point.0",
                "gaze_y": "gaze.point.1",
                "image_width": "frame.width",
                "image_height": "frame.height",
            },
            filters=["split=train"],
            limit=1,
            overwrite=True,
        )

        assert summary["num_input_rows"] == 3
        assert summary["num_output_rows"] == 1
        assert summary["num_skipped_rows"] == 0
        text = output_metadata.read_text(encoding="utf-8")
        assert "episode_id,video_path,frame_idx,timestamp,gaze_x,gaze_y,image_width,image_height" in text
        assert "clip_a,clip_a.mp4,0,,320,240,640,480" in text


def test_adapt_open_video_gaze_metadata_exports_manifest_and_zarr():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        video_path = root / "clip.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (24, 16),
        )
        for idx in range(3):
            frame_rgb = np.zeros((16, 24, 3), dtype=np.uint8)
            frame_rgb[:, :, 0] = idx * 60
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        writer.release()

        metadata_path = root / "source.jsonl"
        with metadata_path.open("w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "video": {"file": "clip.mp4", "id": "clip_nested"},
                        "sample": {"frame": 1},
                        "eye": {"x": 12, "y": 8},
                        "size": {"w": 24, "h": 16},
                    }
                )
                + "\n"
            )
        output_metadata = root / "canonical.csv"
        output_manifest = root / "manifest.csv"
        frames_dir = root / "frames"
        output_zarr = root / "open.zarr"

        summary = adapt_open_video_gaze_metadata(
            metadata_path=str(metadata_path),
            output_metadata=str(output_metadata),
            video_key="video.file",
            episode_key="video.id",
            frame_key="sample.frame",
            gaze_x_key="eye.x",
            gaze_y_key="eye.y",
            width_key="size.w",
            height_key="size.h",
            output_manifest=str(output_manifest),
            frames_dir=str(frames_dir),
            root_dir=str(root),
            image_size=(16, 24),
            gaze_is_normalized=False,
            output_zarr=str(output_zarr),
            zarr_image_size=(16, 16),
            overwrite=True,
        )

        assert summary["num_output_rows"] == 1
        assert summary["export"]["num_frames"] == 1
        assert output_manifest.exists()
        assert len(list(frames_dir.glob("clip_nested/*.png"))) == 1
        zroot = zarr.open(str(output_zarr), mode="r")
        assert zroot["data/camera0_rgb"].shape == (1, 16, 16, 3)
        assert np.allclose(zroot["data/gaze_xy"][0], np.asarray([0.5, 0.5], dtype=np.float32))


def test_prepare_open_gaze_wam_zarr_from_nested_video_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        video_path = root / "clip.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (24, 16),
        )
        for idx in range(3):
            frame_rgb = np.zeros((16, 24, 3), dtype=np.uint8)
            frame_rgb[:, :, 1] = idx * 70
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        writer.release()

        metadata_path = root / "source.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "annotations": [
                        {
                            "video": {"file": "clip.mp4", "id": "clip_nested"},
                            "sample": {"frame": 1},
                            "eye": {"x": 12, "y": 8},
                            "size": {"w": 24, "h": 16},
                            "split": "train",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        output_zarr = root / "prepared_open_video.zarr"
        output_manifest = root / "prepared_manifest.csv"
        frames_dir = root / "prepared_frames"
        adapted_metadata = root / "prepared_canonical.csv"
        preview_dir = root / "prepared_video_preview"
        report_path = root / "prepared_video_report.json"
        inspect_path = root / "prepared_video_metadata_inspect.json"

        summary = prepare_open_gaze_wam_zarr(
            video_metadata_path=str(metadata_path),
            output_zarr=str(output_zarr),
            report_json=str(report_path),
            preview_dir=str(preview_dir),
            adapted_metadata_path=str(adapted_metadata),
            metadata_inspect_json=str(inspect_path),
            output_manifest=str(output_manifest),
            frames_dir=str(frames_dir),
            root_dir=str(root),
            key_map={
                "video_path": "video.file",
                "episode_id": "video.id",
                "frame_idx": "sample.frame",
                "gaze_x": "eye.x",
                "gaze_y": "eye.y",
                "image_width": "size.w",
                "image_height": "size.h",
            },
            filters=["split=train"],
            image_size=(16, 24),
            gaze_is_normalized=False,
            n_obs_steps=2,
            action_horizon=3,
            heatmap_token_grid=(4, 4),
            overwrite=True,
        )

        assert summary["ok"] is True
        assert summary["mode"] == "video_metadata"
        assert summary["adapt"]["num_output_rows"] == 1
        assert summary["export"]["num_frames"] == 1
        assert summary["convert"]["label_mode"] == "point"
        assert summary["validation"]["valid"] is True
        assert output_manifest.exists()
        assert adapted_metadata.exists()
        assert inspect_path.exists()
        assert summary["metadata_inspect"]["suggested_key_map"] == {
            "episode_id": "video.id",
            "frame_idx": "sample.frame",
            "gaze_x": "eye.x",
            "gaze_y": "eye.y",
            "image_height": "size.h",
            "image_width": "size.w",
            "video_path": "video.file",
        }
        assert "scripts/adapt_open_video_gaze_metadata.py" in summary["metadata_inspect"]["adapter_command_template"]
        assert str(adapted_metadata) in summary["metadata_inspect"]["adapter_command_template"]
        assert str(output_manifest) in summary["metadata_inspect"]["adapter_command_template"]
        assert str(frames_dir) in summary["metadata_inspect"]["adapter_command_template"]
        assert str(output_zarr) in summary["metadata_inspect"]["adapter_command_template"]
        assert len(list(frames_dir.glob("clip_nested/*.png"))) == 1
        assert Path(summary["preview"]["paths"]["overlay"]).exists()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["ok"] is True
        inspect_payload = json.loads(inspect_path.read_text(encoding="utf-8"))
        assert inspect_payload["suggested_key_map"] == summary["metadata_inspect"]["suggested_key_map"]
        assert inspect_payload["adapter_command_template"] == summary["metadata_inspect"]["adapter_command_template"]
        zroot = zarr.open(str(output_zarr), mode="r")
        assert np.allclose(zroot["data/gaze_xy"][0], np.asarray([0.5, 0.5], dtype=np.float32))


def test_prepare_open_gaze_wam_zarr_video_metadata_dry_run_reports_plan_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        metadata_path = root / "source.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "annotations": [
                        {
                            "video": {"file": "clip.mp4", "id": "clip_nested"},
                            "sample": {"frame": 1},
                            "eye": {"x": 12, "y": 8},
                            "size": {"w": 24, "h": 16},
                            "split": "train",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        output_zarr = root / "prepared_open_video.zarr"
        output_manifest = root / "prepared_manifest.csv"
        frames_dir = root / "prepared_frames"
        adapted_metadata = root / "prepared_canonical.csv"
        report_path = root / "prepared_video_dry_run_report.json"
        inspect_path = root / "prepared_video_metadata_inspect.json"

        summary = prepare_open_gaze_wam_zarr(
            video_metadata_path=str(metadata_path),
            output_zarr=str(output_zarr),
            report_json=str(report_path),
            adapted_metadata_path=str(adapted_metadata),
            metadata_inspect_json=str(inspect_path),
            output_manifest=str(output_manifest),
            frames_dir=str(frames_dir),
            root_dir=str(root),
            key_map={
                "video_path": "video.file",
                "episode_id": "video.id",
                "frame_idx": "sample.frame",
                "gaze_x": "eye.x",
                "gaze_y": "eye.y",
                "image_width": "size.w",
                "image_height": "size.h",
            },
            filters=["split=train"],
            image_size=(16, 24),
            gaze_is_normalized=False,
            n_obs_steps=2,
            action_horizon=3,
            heatmap_token_grid=(4, 4),
            dry_run=True,
        )

        assert summary["ok"] is True
        assert summary["dry_run"] is True
        assert summary["mode"] == "video_metadata"
        assert summary["adapt"] is None
        assert summary["convert"] is None
        assert summary["validation"] is None
        assert summary["preview"] is None
        assert summary["metadata_inspect"]["suggested_key_map"] == {
            "episode_id": "video.id",
            "frame_idx": "sample.frame",
            "gaze_x": "eye.x",
            "gaze_y": "eye.y",
            "image_height": "size.h",
            "image_width": "size.w",
            "video_path": "video.file",
        }
        assert "adapt_open_video_gaze_metadata.py" in summary["planned_commands"]["adapt"]
        assert "export_video_gaze_manifest.py" in summary["planned_commands"]["export"]
        assert "convert_open_gaze_manifest.py" in summary["planned_commands"]["convert"]
        assert "--no-gaze-is-normalized" in summary["planned_commands"]["export"]
        assert str(adapted_metadata) in summary["planned_commands"]["export"]
        assert str(output_manifest) in summary["planned_commands"]["convert"]
        assert not adapted_metadata.exists()
        assert not output_manifest.exists()
        assert not frames_dir.exists()
        assert not output_zarr.exists()
        assert inspect_path.exists()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["dry_run"] is True
        assert payload["planned_commands"] == summary["planned_commands"]


def test_review_gaze_wam_data_onboarding_runs_robot_and_open_dry_runs(monkeypatch):
    calls = []

    def fake_robot_prepare(**kwargs):
        calls.append(("robot", kwargs))
        assert kwargs["dry_run"] is True
        assert kwargs["input_path"] == "raw_robot.zarr"
        assert kwargs["output_path"] == "robot.zarr"
        assert kwargs["image_size"] == (256, 256)
        return {
            "ok": True,
            "dry_run": True,
            "canonicalizer_command": "py scripts/canonicalize_robot_gaze_wam_zarr.py ...",
        }

    def fake_open_prepare(**kwargs):
        calls.append(("open", kwargs))
        assert kwargs["dry_run"] is True
        assert kwargs["manifest_path"] == "open_manifest.csv"
        assert kwargs["output_zarr"] == "open.zarr"
        assert kwargs["image_size"] == (256, 256)
        return {
            "ok": True,
            "dry_run": True,
            "planned_commands": {"convert": "py scripts/convert_open_gaze_manifest.py ..."},
        }

    monkeypatch.setattr(
        review_gaze_wam_data_onboarding_module,
        "_ensure_onboarding_runtime",
        lambda needs_robot, needs_open: None,
    )
    monkeypatch.setattr(
        review_gaze_wam_data_onboarding_module,
        "prepare_robot_gaze_wam_zarr",
        fake_robot_prepare,
    )
    monkeypatch.setattr(
        review_gaze_wam_data_onboarding_module,
        "prepare_open_gaze_wam_zarr",
        fake_open_prepare,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = Path(tmpdir) / "review.json"
        summary = review_gaze_wam_data_onboarding(
            output_json=str(output_json),
            robot_input_path="raw_robot.zarr",
            robot_output_path="robot.zarr",
            open_manifest_path="open_manifest.csv",
            open_output_zarr="open.zarr",
            require_timestamps=True,
            timestamp_max_step=0.08,
        )

        assert summary["ok"] is True
        assert summary["dry_run"] is True
        assert summary["policy_training_scope"] is True
        assert summary["deployment_runner_scope"] == "deferred"
        assert summary["selected"] == {"robot": True, "open": True}
        assert summary["contract"]["require_timestamps"] is True
        assert summary["contract"]["timestamp_max_step"] == 0.08
        assert summary["robot"]["dry_run"] is True
        assert summary["open"]["dry_run"] is True
        assert [name for name, _ in calls] == ["robot", "open"]
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert payload["ok"] is True
        assert payload["robot"]["canonicalizer_command"].startswith("py scripts/canonicalize")
        assert payload["open"]["planned_commands"]["convert"].startswith("py scripts/convert")


def test_review_gaze_wam_data_onboarding_rejects_incomplete_sources(monkeypatch):
    monkeypatch.setattr(
        review_gaze_wam_data_onboarding_module,
        "_ensure_onboarding_runtime",
        lambda needs_robot, needs_open: (_ for _ in ()).throw(
            AssertionError("Invalid source review should not load prepare runtimes.")
        ),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = Path(tmpdir) / "bad_review.json"
        summary = review_gaze_wam_data_onboarding(
            output_json=str(output_json),
            robot_input_path="raw_robot.zarr",
        )

        assert summary["ok"] is False
        assert any("Robot onboarding requires" in message for message in summary["errors"])
        assert output_json.exists()
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert payload["ok"] is False
        assert payload["robot"] is None
        assert payload["open"] is None


def test_review_gaze_wam_data_onboarding_uses_strict_stage_ok_bool(monkeypatch):
    def fake_robot_prepare(**kwargs):
        return {
            "ok": "false",
            "dry_run": True,
        }

    monkeypatch.setattr(
        review_gaze_wam_data_onboarding_module,
        "_ensure_onboarding_runtime",
        lambda needs_robot, needs_open: None,
    )
    monkeypatch.setattr(
        review_gaze_wam_data_onboarding_module,
        "prepare_robot_gaze_wam_zarr",
        fake_robot_prepare,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = Path(tmpdir) / "review.json"
        summary = review_gaze_wam_data_onboarding(
            output_json=str(output_json),
            robot_input_path="raw_robot.zarr",
            robot_output_path="robot.zarr",
        )

        assert summary["robot"]["ok"] == "false"
        assert summary["ok"] is False
        assert "Robot dry-run onboarding failed." in summary["errors"]
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert payload["ok"] is False


def test_review_gaze_wam_training_readiness_writes_three_stage_bundle(monkeypatch):
    calls = {
        "onboarding": None,
        "dino": None,
        "launch": None,
    }

    def fake_onboarding(**kwargs):
        calls["onboarding"] = kwargs
        assert kwargs["robot_input_path"] == "raw_robot.zarr"
        assert kwargs["robot_output_path"] == "robot.zarr"
        assert kwargs["open_manifest_path"] == "open.csv"
        assert kwargs["open_output_zarr"] == "open.zarr"
        assert kwargs["require_timestamps"] is True
        return {
            "ok": True,
            "dry_run": True,
            "policy_training_scope": True,
            "deployment_runner_scope": "deferred",
            "robot": {"output_path": "robot.zarr"},
            "open": {"output_zarr": "open.zarr"},
        }

    def fake_dino(**kwargs):
        calls["dino"] = kwargs
        assert kwargs["checkpoint_path"] == "dinov3.ckpt"
        assert kwargs["image_size"] == (256, 256)
        assert kwargs["heatmap_token_grid"] == (16, 16)
        return {
            "ok": True,
            "errors": [],
            "warnings": [],
            "dino_source": {
                "model_name": "vit_base_patch16_dinov3",
                "pretrained": True,
                "checkpoint_path": "dinov3.ckpt",
                "cache_dir": "",
            },
            "geometry": {
                "image_size": [256, 256],
                "patch_size": 16,
                "expected_tokens_per_frame": 256,
                "heatmap_token_grid": [16, 16],
                "heatmap_num_tokens": 256,
            },
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        }

    def fake_launch(**kwargs):
        calls["launch"] = kwargs
        assert kwargs["real_data"] is True
        assert kwargs["run"] is False
        assert kwargs["data_onboarding_review_json"].endswith("_onboarding.json")
        assert kwargs["require_data_onboarding_review"] is True
        assert kwargs["preflight_require_timestamps"] is True
        return {
            "ok": True,
            "errors": [],
            "warnings": [],
            "preflight_routing_validation_guardrails_ok": True,
            "real_data_readiness": {
                "ok": True,
                "dino_source_verifier": {
                    "ok": True,
                    "errors": [],
                    "warnings": [],
                    "dino_source": {
                        "model_name": "vit_base_patch16_dinov3",
                        "pretrained": True,
                        "checkpoint_path": "dinov3.ckpt",
                        "cache_dir": "",
                    },
                    "geometry": {
                        "image_size": [256, 256],
                        "patch_size": 16,
                        "expected_tokens_per_frame": 256,
                        "heatmap_token_grid": [16, 16],
                        "heatmap_num_tokens": 256,
                    },
                    "normalization": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                },
            },
        }

    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "_ensure_onboarding_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "_ensure_dino_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "_ensure_launcher_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "review_gaze_wam_data_onboarding",
        fake_onboarding,
    )
    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "verify_gaze_wam_dino_source",
        fake_dino,
    )
    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "launch_gaze_wam_training",
        fake_launch,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = Path(tmpdir) / "readiness.json"
        summary = review_gaze_wam_training_readiness(
            output_json=str(output_json),
            robot_input_path="raw_robot.zarr",
            robot_output_path="robot.zarr",
            open_manifest_path="open.csv",
            open_output_zarr="open.zarr",
            dino_checkpoint_path="dinov3.ckpt",
            preflight_timestamp_max_step=0.08,
        )

        assert summary["ok"] is True
        assert summary["policy_training_scope"] is True
        assert summary["deployment_runner_scope"] == "deferred"
        assert summary["stages"]["data_onboarding_review"]["ran"] is True
        assert summary["stages"]["dino_source_verifier"]["ran"] is True
        assert summary["stages"]["launch_dry_run"]["ran"] is True
        assert summary["cross_checks"]["dino_matches_launch"]["ok"] is True
        assert summary["cross_checks"]["launch_preflight_routing_guardrails"]["ok"] is True
        assert (
            summary["cross_checks"]["launch_preflight_routing_guardrails"][
                "preflight_routing_validation_guardrails_ok"
            ]
            is True
        )
        assert summary["artifacts"]["data_onboarding_review_json"].endswith("_onboarding.json")
        assert summary["artifacts"]["dino_report_json"].endswith("_dino.json")
        assert summary["artifacts"]["launch_report_json"].endswith("_launch.json")
        assert calls["onboarding"] is not None
        assert calls["dino"] is not None
        assert calls["launch"] is not None
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert payload["ok"] is True
        assert payload["cross_checks"]["launch_preflight_routing_guardrails"]["ok"] is True
        assert payload["stages"]["launch_dry_run"]["summary"]["real_data_readiness"]["ok"] is True


def test_review_gaze_wam_training_readiness_parses_string_dino_pretrained():
    def dino_report(pretrained):
        return {
            "dino_source": {
                "model_name": "vit_base_patch16_dinov3",
                "pretrained": pretrained,
                "checkpoint_path": "dinov3.ckpt",
                "cache_dir": "",
            },
            "geometry": {
                "image_size": [256, 256],
                "patch_size": 16,
                "expected_tokens_per_frame": 256,
                "heatmap_token_grid": [16, 16],
                "heatmap_num_tokens": 256,
            },
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        }

    summary = review_gaze_wam_training_readiness_module._dino_report_match_check(
        dino_report("false"),
        {"real_data_readiness": {"dino_source_verifier": dino_report(False)}},
    )

    assert summary["ok"] is True
    assert summary["standalone_signature"]["pretrained"] is False
    assert summary["launch_signature"]["pretrained"] is False
    assert summary["mismatched_fields"] == []


def test_review_gaze_wam_training_readiness_uses_strict_stage_ok_bool(monkeypatch):
    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "_ensure_onboarding_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "review_gaze_wam_data_onboarding",
        lambda **kwargs: {"ok": "false", "errors": [], "warnings": []},
    )

    summary = review_gaze_wam_training_readiness(
        output_json=None,
        require_data_onboarding_review=True,
        require_dino_ok=False,
        require_launch_ok=False,
        run_dino_verifier=False,
        run_launch_dry_run=False,
        robot_input_path="raw_robot.zarr",
        robot_output_path="robot.zarr",
        open_manifest_path="open.csv",
        open_output_zarr="open.zarr",
        preflight_timestamp_max_step=0.08,
    )

    stage = summary["stages"]["data_onboarding_review"]
    assert stage["ran"] is True
    assert stage["summary"]["ok"] == "false"
    assert stage["ok"] is False
    assert summary["ok"] is False
    assert "Data onboarding review stage is not ok." in summary["errors"]


def test_review_gaze_wam_training_readiness_blocks_dino_launch_mismatch(monkeypatch):
    def fake_dino(**kwargs):
        return {
            "ok": True,
            "errors": [],
            "warnings": [],
            "dino_source": {
                "model_name": "vit_base_patch16_dinov3",
                "pretrained": True,
                "checkpoint_path": "standalone.ckpt",
                "cache_dir": "",
            },
            "geometry": {
                "image_size": [256, 256],
                "patch_size": 16,
                "expected_tokens_per_frame": 256,
                "heatmap_token_grid": [16, 16],
                "heatmap_num_tokens": 256,
            },
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        }

    def fake_launch(**kwargs):
        return {
            "ok": True,
            "errors": [],
            "warnings": [],
            "preflight_routing_validation_guardrails_ok": True,
            "real_data_readiness": {
                "ok": True,
                "dino_source_verifier": {
                    "ok": True,
                    "errors": [],
                    "warnings": [],
                    "dino_source": {
                        "model_name": "vit_base_patch16_dinov3",
                        "pretrained": True,
                        "checkpoint_path": "launcher.ckpt",
                        "cache_dir": "",
                    },
                    "geometry": {
                        "image_size": [256, 256],
                        "patch_size": 16,
                        "expected_tokens_per_frame": 256,
                        "heatmap_token_grid": [16, 16],
                        "heatmap_num_tokens": 256,
                    },
                    "normalization": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                },
            },
        }

    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "_ensure_onboarding_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "_ensure_dino_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "_ensure_launcher_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "verify_gaze_wam_dino_source",
        fake_dino,
    )
    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "launch_gaze_wam_training",
        fake_launch,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = Path(tmpdir) / "readiness.json"
        summary = review_gaze_wam_training_readiness(
            output_json=str(output_json),
            require_data_onboarding_review=False,
            dino_checkpoint_path="standalone.ckpt",
            preflight_timestamp_max_step=0.08,
        )

        assert summary["ok"] is False
        dino_check = summary["cross_checks"]["dino_matches_launch"]
        assert dino_check["enabled"] is True
        assert dino_check["ok"] is False
        assert dino_check["mismatched_fields"] == ["checkpoint_path"]
        assert "checkpoint_path" in summary["errors"][0]


def test_review_gaze_wam_training_readiness_blocks_launch_routing_guardrail_failure(monkeypatch):
    dino_report = {
        "ok": True,
        "errors": [],
        "warnings": [],
        "dino_source": {
            "model_name": "vit_base_patch16_dinov3",
            "pretrained": True,
            "checkpoint_path": "dinov3.ckpt",
            "cache_dir": "",
        },
        "geometry": {
            "image_size": [256, 256],
            "patch_size": 16,
            "expected_tokens_per_frame": 256,
            "heatmap_token_grid": [16, 16],
            "heatmap_num_tokens": 256,
        },
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }

    def fake_launch(**kwargs):
        return {
            "ok": True,
            "errors": [],
            "warnings": [],
            "preflight_routing_validation_guardrails_ok": False,
            "real_data_readiness": {
                "ok": True,
                "dino_source_verifier": dino_report,
            },
        }

    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "_ensure_onboarding_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "_ensure_dino_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "_ensure_launcher_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "verify_gaze_wam_dino_source",
        lambda **kwargs: copy.deepcopy(dino_report),
    )
    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "launch_gaze_wam_training",
        fake_launch,
    )

    summary = review_gaze_wam_training_readiness(
        output_json=None,
        require_data_onboarding_review=False,
        dino_checkpoint_path="dinov3.ckpt",
        preflight_timestamp_max_step=0.08,
    )

    routing_check = summary["cross_checks"]["launch_preflight_routing_guardrails"]
    assert summary["ok"] is False
    assert summary["stages"]["launch_dry_run"]["ok"] is True
    assert routing_check["enabled"] is True
    assert routing_check["ok"] is False
    assert routing_check["preflight_routing_validation_guardrails_ok"] is False
    assert "preflight_routing_validation_guardrails_ok" in summary["errors"][0]


def test_review_gaze_wam_training_readiness_does_not_pass_missing_onboarding_path(monkeypatch):
    calls = {"launch": None}

    def fail_onboarding():
        raise AssertionError("No onboarding runtime should load without source paths.")

    def fake_launch(**kwargs):
        calls["launch"] = kwargs
        assert kwargs["data_onboarding_review_json"] is None
        assert kwargs["require_data_onboarding_review"] is False
        return {
            "ok": True,
            "errors": [],
            "warnings": [],
            "preflight_routing_validation_guardrails_ok": True,
            "real_data_readiness": {
                "dino_source_verifier": {
                    "ok": True,
                    "errors": [],
                    "warnings": [],
                },
            },
        }

    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "_ensure_onboarding_runtime",
        fail_onboarding,
    )
    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "_ensure_dino_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "_ensure_launcher_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "verify_gaze_wam_dino_source",
        lambda **kwargs: {"ok": True, "errors": [], "warnings": []},
    )
    monkeypatch.setattr(
        review_gaze_wam_training_readiness_module,
        "launch_gaze_wam_training",
        fake_launch,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = Path(tmpdir) / "readiness.json"
        summary = review_gaze_wam_training_readiness(
            output_json=str(output_json),
            require_data_onboarding_review=False,
            require_dino_ok=True,
            require_launch_ok=True,
            preflight_timestamp_max_step=0.08,
        )

        assert summary["ok"] is True
        assert summary["stages"]["data_onboarding_review"]["enabled"] is False
        assert summary["artifacts"]["data_onboarding_review_json"] is None
        assert calls["launch"] is not None


def test_prepare_open_gaze_wam_zarr_video_metadata_timestamp_key_forces_adapt():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        video_path = root / "clip.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (24, 16),
        )
        for idx in range(2):
            frame_rgb = np.zeros((16, 24, 3), dtype=np.uint8)
            frame_rgb[:, :, 2] = idx * 80
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        writer.release()

        metadata_path = root / "source.jsonl"
        with metadata_path.open("w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "clip": {"file": "clip.mp4", "id": "clip_nested"},
                        "sample": {"frame": 0},
                        "eye": {"x": 12, "y": 8},
                        "size": {"w": 24, "h": 16},
                        "sensor_time": 0.0,
                    }
                )
                + "\n"
            )

        output_zarr = root / "prepared_open_video.zarr"
        output_manifest = root / "prepared_manifest.csv"
        frames_dir = root / "prepared_frames"
        adapted_metadata = root / "prepared_canonical.csv"

        summary = prepare_open_gaze_wam_zarr(
            video_metadata_path=str(metadata_path),
            output_zarr=str(output_zarr),
            adapted_metadata_path=str(adapted_metadata),
            output_manifest=str(output_manifest),
            frames_dir=str(frames_dir),
            root_dir=str(root),
            video_key="clip.file",
            episode_key="clip.id",
            frame_key="sample.frame",
            timestamp_key="sensor_time",
            gaze_x_key="eye.x",
            gaze_y_key="eye.y",
            width_key="size.w",
            height_key="size.h",
            image_size=(16, 24),
            gaze_is_normalized=False,
            n_obs_steps=2,
            action_horizon=3,
            heatmap_token_grid=(4, 4),
            overwrite=True,
        )

        assert summary["mode"] == "video_metadata"
        assert summary["adapt"]["keys"]["timestamp"] == "sensor_time"
        assert summary["export"]["num_frames"] == 1
        assert adapted_metadata.exists()
        assert output_manifest.exists()
        zroot = zarr.open(str(output_zarr), mode="r")
        assert np.allclose(zroot["data/timestamp"][:], np.asarray([0.0]))


def test_gaze_wam_offline_metric_evaluator_robot_and_open_sources(tmp_path):
    torch.manual_seed(9)
    shape_meta = {
        "action": {
            "shape": [10],
            "horizon": 3,
        }
    }
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    encoder_path, decoder_path = _write_fake_cosmos_jit_pair(
        tmp_path,
        image_size=(16, 16),
        token_grid=(4, 4),
        latent_channels=1,
    )
    policy = GazeWamPolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        obs_encoder=FakeTokenObsEncoder(num_tokens=8, embed_dim=32),
        model=JointGazeWamTransformer(
            action_dim=10,
            heatmap_dim=1,
            action_horizon=3,
            heatmap_num_tokens=16,
            max_image_tokens=8,
            n_layer=1,
            n_head=4,
            n_emb=32,
            p_drop_emb=0.0,
            p_drop_attn=0.0,
        ),
        gaze_encoder=GazeConditionEncoder(embed_dim=32),
        num_inference_steps=2,
        input_pertub=0.0,
        heatmap_num_tokens=16,
        heatmap_token_grid=(4, 4),
        heatmap_image_size=(16, 16),
        heatmap_cosmos_encoder_path=encoder_path,
        heatmap_cosmos_decoder_path=decoder_path,
        n_emb=32,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_gaze_wam_zarr(Path(tmpdir) / "robot.zarr", include_action=True)
        open_path = _write_gaze_wam_zarr(Path(tmpdir) / "open.zarr", include_action=False)
        robot_root = zarr.open(str(robot_path), mode="a")
        robot_data = robot_root["data"]
        robot_heatmap = np.zeros_like(robot_data["gaze_heatmap"][:], dtype=np.float32)
        _replace_zarr_array(robot_data, "gaze_heatmap", robot_heatmap)
        robot_heatmap_presence = np.zeros_like(
            robot_data["has_heatmap_image"][:],
            dtype=np.bool_,
        )
        _replace_zarr_array(
            robot_data,
            "has_heatmap_image",
            robot_heatmap_presence,
        )
        robot_dataset = GazeWamRobotDataset(
            dataset_path=str(robot_path),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            action_padding=True,
        )
        open_dataset = GazeWamOpenDataset(
            dataset_path=str(open_path),
            n_obs_steps=2,
            action_horizon=3,
            image_size=(16, 16),
            heatmap_token_grid=(4, 4),
            action_dim=10,
        )
        policy.set_normalizer(robot_dataset.get_normalizer())

        robot_metrics = evaluate_gaze_wam_dataset(
            policy=policy,
            dataset=robot_dataset,
            batch_size=2,
            max_batches=1,
            device=torch.device("cpu"),
            source_name="robot",
        )
        robot_dropout_metrics = evaluate_gaze_wam_dataset(
            policy=policy,
            dataset=robot_dataset,
            batch_size=2,
            max_batches=1,
            device=torch.device("cpu"),
            source_name="robot_dropout",
            source_is_robot=True,
            robot_gaze_dropout_prob=1.0,
            robot_heatmap_on_gaze_dropout=True,
            robot_gaze_dropout_seed=0,
        )
        open_metrics = evaluate_gaze_wam_dataset(
            policy=policy,
            dataset=open_dataset,
            batch_size=2,
            max_batches=1,
            device=torch.device("cpu"),
            source_name="open",
        )

    assert robot_metrics["robot_num_samples"] == 2.0
    assert robot_metrics["robot_action_supervision_count"] == 2.0
    assert robot_metrics["robot_action_abs_supervision_count"] == 2.0
    assert robot_metrics["robot_action_abs_metric_eligible_count"] == 2.0
    assert robot_metrics["robot_has_action_abs_count"] == 2.0
    assert robot_metrics["robot_has_action_base_abs_count"] == 2.0
    assert robot_metrics["robot_has_heatmap_image_count"] == 0.0
    assert robot_metrics["robot_heatmap_supervision_count"] == 0.0
    assert robot_metrics["robot_denoise_action_mask_count"] == 2.0
    assert robot_metrics["robot_denoise_heatmap_mask_count"] == 0.0
    assert robot_metrics["robot_gdr_eligible_count"] == 2.0
    assert "robot_action_mse" in robot_metrics
    assert "robot_action_abs_mse" in robot_metrics
    assert "robot_feature_gdr" in robot_metrics
    assert "robot_output_gdr" in robot_metrics
    assert "robot_heatmap_mse" not in robot_metrics

    assert robot_dropout_metrics["robot_dropout_num_samples"] == 2.0
    assert robot_dropout_metrics["robot_dropout_action_supervision_count"] == 2.0
    assert robot_dropout_metrics["robot_dropout_action_abs_metric_eligible_count"] == 2.0
    assert robot_dropout_metrics["robot_dropout_heatmap_supervision_count"] == 2.0
    assert robot_dropout_metrics["robot_dropout_has_heatmap_image_count"] == 0.0
    assert robot_dropout_metrics["robot_dropout_denoise_action_mask_count"] == 2.0
    assert robot_dropout_metrics["robot_dropout_denoise_heatmap_mask_count"] == 2.0
    assert robot_dropout_metrics["robot_dropout_gaze_condition_count"] == 0.0
    assert robot_dropout_metrics["robot_dropout_gdr_eligible_count"] == 0.0
    assert "robot_dropout_action_mse" in robot_dropout_metrics
    assert "robot_dropout_heatmap_mse" in robot_dropout_metrics
    assert "robot_dropout_feature_gdr" not in robot_dropout_metrics

    assert open_metrics["open_num_samples"] == 2.0
    assert open_metrics["open_action_supervision_count"] == 0.0
    assert open_metrics["open_action_abs_supervision_count"] == 0.0
    assert open_metrics["open_action_abs_metric_eligible_count"] == 0.0
    assert open_metrics["open_has_action_abs_count"] == 0.0
    assert open_metrics["open_has_action_base_abs_count"] == 0.0
    assert open_metrics["open_heatmap_supervision_count"] == 2.0
    assert open_metrics["open_has_heatmap_image_count"] == 2.0
    assert open_metrics["open_denoise_action_mask_count"] == 0.0
    assert open_metrics["open_denoise_heatmap_mask_count"] == 2.0
    assert open_metrics["open_gdr_eligible_count"] == 0.0
    assert "open_heatmap_mse" in open_metrics
    assert "open_heatmap_kl" in open_metrics
    assert "open_heatmap_argmax_l2" in open_metrics
    assert "open_action_mse" not in open_metrics


def test_gaze_wam_offline_metric_evaluator_slices_action_base_presence_mask():
    class TinyEvalDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            action = torch.zeros(3, 10)
            action_abs = torch.zeros(3, 10)
            action_base_abs = torch.zeros(10)
            return {
                "obs": {"camera0_rgb": torch.zeros(2, 3, 16, 16)},
                "action": action,
                "action_abs": action_abs,
                "action_base_abs": action_base_abs,
                "heatmap": torch.zeros(1, 16, 1),
                "gaze_xy": torch.tensor([0.5, 0.5], dtype=torch.float32),
                "is_open": torch.tensor(False),
                "has_action": torch.tensor(index == 0),
                "has_heatmap": torch.tensor(False),
                "has_gaze_label": torch.tensor(True),
                "use_gaze_condition": torch.tensor(True),
                "is_gaze_condition_dropped": torch.tensor(False),
                "has_action_abs": torch.tensor(True),
                "has_action_base_abs": torch.tensor(True),
            }

    class FakeEvalPolicy:
        def __init__(self):
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.training = True
            self.predict_calls = []

        def eval(self):
            self.training = False
            return self

        def train(self):
            self.training = True
            return self

        def to(self, device):
            self.device = torch.device(device)
            return self

        def predict_action(self, obs_dict, cfg_scale=None):
            self.predict_calls.append(obs_dict)
            batch_size = obs_dict["gaze_xy"].shape[0]
            pred = torch.zeros(batch_size, 3, 10)
            result = {
                "action_pred_relative": pred,
                "action": pred,
                "action_pred": pred,
            }
            if "action_base_abs" in obs_dict:
                assert torch.all(obs_dict["has_action_base_abs"])
                result["action_pred_abs"] = pred
                result["action_abs"] = pred
            return result

    policy = FakeEvalPolicy()
    metrics = evaluate_gaze_wam_dataset(
        policy=policy,
        dataset=TinyEvalDataset(),
        batch_size=2,
        max_batches=1,
        device=torch.device("cpu"),
        source_name="fake",
        compute_denoising_loss=False,
        compute_heatmap=False,
        compute_gdr=False,
    )

    assert metrics["fake_action_supervision_count"] == 1.0
    assert metrics["fake_action_abs_supervision_count"] == 1.0
    assert metrics["fake_action_abs_metric_eligible_count"] == 1.0
    assert metrics["fake_has_action_abs_count"] == 1.0
    assert metrics["fake_has_action_base_abs_count"] == 1.0
    assert metrics["fake_has_heatmap_image_count"] == 0.0
    assert metrics["fake_action_mse_count"] == 1.0
    assert metrics["fake_action_abs_mse_count"] == 1.0
    assert len(policy.predict_calls) == 2
    assert "action_base_abs" not in policy.predict_calls[0]
    assert policy.predict_calls[0]["gaze_xy"].shape[0] == 2
    assert policy.predict_calls[1]["action_base_abs"].shape[0] == 1
    assert policy.predict_calls[1]["has_action_base_abs"].tolist() == [True]


def test_gaze_wam_offline_metric_evaluator_rejects_non_bool_presence_masks():
    has_action = torch.tensor([True, False])
    batch = {
        "action_abs": torch.zeros(2, 3, 10),
        "action_base_abs": torch.zeros(2, 10),
        "has_action_abs": torch.tensor([1, 0]),
        "has_action_base_abs": torch.tensor([True, True]),
    }
    try:
        _action_abs_mask(batch, has_action)
        assert False, "Expected non-bool action presence mask to fail."
    except ValueError as exc:
        assert "batch['has_action_abs'] must be a BoolTensor" in str(exc)

    bad_shape_batch = {
        "action_abs": torch.zeros(2, 3, 10),
        "action_base_abs": torch.zeros(2, 10),
        "has_action_abs": torch.tensor([[True], [False]]),
        "has_action_base_abs": torch.tensor([True, True]),
    }
    try:
        _action_abs_mask(bad_shape_batch, has_action)
        assert False, "Expected misshaped action presence mask to fail."
    except ValueError as exc:
        assert "batch['has_action_abs'] must have shape [B]" in str(exc)

    optional_batch = {
        "heatmap_image": torch.zeros(2, 1, 16, 16),
        "has_heatmap_image": torch.tensor([1.0, 0.0]),
    }
    try:
        _optional_presence_mask(
            optional_batch,
            "heatmap_image",
            torch.tensor([True, True]),
        )
        assert False, "Expected non-bool heatmap presence mask to fail."
    except ValueError as exc:
        assert "batch['has_heatmap_image'] must be a BoolTensor" in str(exc)


def test_compare_gaze_wam_ablation_metrics_rows_and_csv(monkeypatch):
    def fake_load_policy_for_eval(
        cfg,
        checkpoint,
        device,
        use_ema,
        overrides=None,
        trust_checkpoint=False,
    ):
        assert checkpoint is None
        return object(), cfg

    def fake_evaluate_gaze_wam_sources(policy, cfg, **kwargs):
        return {
            "robot_denoise_loss": 0.25,
            "robot_num_samples": 2.0,
            "robot_action_supervision_count": 2.0,
            "robot_heatmap_supervision_count": 0.0,
            "robot_zarr_validation": {
                "valid": True,
                "dataset_type": "robot",
                "sample": {"action_roundtrip_max_error": 0.0},
            },
        }

    monkeypatch.setattr(
        eval_gaze_wam_metrics_module,
        "load_policy_for_eval",
        fake_load_policy_for_eval,
    )
    monkeypatch.setattr(
        eval_gaze_wam_metrics_module,
        "evaluate_gaze_wam_sources",
        fake_evaluate_gaze_wam_sources,
    )

    rows = compare_gaze_wam_ablation_metrics(
        variants=[
            "main_debug=train_gaze_wam_debug_workspace",
            "robot_only_debug=train_gaze_wam_robot_only_debug_workspace",
        ],
        device="cpu",
        batch_size=2,
        max_batches=1,
        sources=("robot",),
        compute_sampling=False,
        compute_heatmap=False,
        compute_gdr=False,
    )

    assert [row["variant"] for row in rows] == [
        "main_debug",
        "robot_only_debug",
    ]
    assert rows[0]["config_name"] == "train_gaze_wam_debug_workspace"
    assert rows[1]["config_name"] == "train_gaze_wam_robot_only_debug_workspace"
    assert rows[0]["global_overrides"] == ""
    assert rows[0]["variant_overrides"] == ""
    assert rows[0]["eval_sources"] == "robot"
    assert rows[0]["eval_batch_size"] == 2
    assert rows[0]["eval_max_batches"] == 1
    assert rows[0]["eval_cfg_scale"] == ""
    assert rows[0]["policy_cfg_scale"] == 1.0
    assert rows[0]["effective_cfg_scale"] == 1.0
    assert rows[0]["cfg_scale"] == 1.0
    assert rows[0]["provenance_contract_version"] == PROVENANCE_CONTRACT_VERSION
    assert rows[0]["provenance_contract_id"] == provenance_contract_id(rows[0])
    assert len(rows[0]["provenance_contract_id"]) == 16
    assert rows[0]["robot_batch_size"] == 3
    assert rows[0]["open_batch_size"] == 1
    assert rows[0]["robot_ratio"] == 0.75
    assert rows[0]["open_ratio"] == 0.25
    assert rows[0]["training_stage"] == "mixed_train"
    assert rows[0]["batch_size_source"] == "ratio"
    assert rows[0]["requested_batch_size_source"] == "auto"
    assert rows[0]["total_batch_size_per_process"] == 4
    assert rows[0]["requested_total_batch_size_per_process"] == 4
    assert rows[0]["requested_robot_ratio"] == 0.75
    assert rows[0]["requested_open_ratio"] == 0.25
    assert rows[0]["gradient_accumulate_every"] == 1
    assert rows[0]["num_processes"] == 1
    assert rows[0]["mixed_precision"] == "no"
    assert rows[0]["distributed_type"] == "NO"
    assert rows[0]["effective_robot_batch_size_per_optimizer_step"] == 3
    assert rows[0]["effective_open_batch_size_per_optimizer_step"] == 1
    assert rows[0]["effective_train_batch_size_per_optimizer_step"] == 4
    assert rows[0]["n_latency_steps"] == 0
    assert rows[0]["robot_obs_downsample_steps"] == 1
    assert rows[0]["robot_action_downsample_steps"] == 1
    assert rows[0]["robot_action_padding"] is True
    assert rows[0]["open_obs_downsample_steps"] == 1
    assert rows[0]["open_action_downsample_steps"] == 1
    assert rows[0]["open_action_padding"] is True
    assert rows[1]["open_batch_size"] == 0
    assert rows[1]["robot_ratio"] == 1.0
    assert rows[0]["use_block_attention_mask"] is True
    assert rows[0]["heatmap_objective"] == "dsnt_js"
    assert rows[0]["heatmap_token_grid"] == "16x16"
    assert rows[0]["image_shape"] == "3x256x256"
    assert rows[0]["image_resize_mode"] == "stretch"
    assert rows[0]["robot_image_resize_mode"] == "stretch"
    assert rows[0]["open_image_resize_mode"] == "stretch"
    assert rows[0]["obs_encoder_model_name"] == "vit_base_patch16_dinov3"
    assert rows[0]["obs_encoder_pretrained"] is False
    assert rows[0]["obs_encoder_checkpoint_path"] == ""
    assert rows[0]["obs_encoder_checkpoint_path_exists"] is False
    assert rows[0]["obs_encoder_checkpoint_path_is_file"] is False
    assert rows[0]["obs_encoder_cache_dir"] == ""
    assert rows[0]["obs_encoder_cache_dir_exists"] is False
    assert rows[0]["obs_encoder_cache_dir_is_dir"] is False
    assert rows[0]["obs_encoder_local_weight_source_configured"] is False
    assert rows[0]["obs_encoder_local_weight_source_valid"] is True
    assert rows[0]["checkpoint_provided"] is False
    assert "robot_denoise_loss" in rows[0]
    assert "robot_num_samples" in rows[0]
    assert "robot_action_supervision_count" in rows[0]
    assert "robot_heatmap_supervision_count" in rows[0]
    assert rows[0]["robot_zarr_validation"]["valid"] is True
    assert rows[0]["robot_zarr_validation"]["dataset_type"] == "robot"
    assert rows[0]["robot_zarr_validation"]["sample"]["action_roundtrip_max_error"] < 1e-6

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "metrics.csv"
        write_metrics_csv(rows, str(csv_path))
        text = csv_path.read_text(encoding="utf-8")
    assert text.startswith(
        "variant,config_name,checkpoint,checkpoint_provided,"
        "global_overrides,variant_overrides,provenance_contract_version,"
        "provenance_contract_id,training_stage,batch_size_source,"
        "requested_batch_size_source,"
        "total_batch_size_per_process,requested_total_batch_size_per_process,"
        "requested_robot_ratio,requested_open_ratio,eval_sources,"
        "eval_batch_size,eval_max_batches,"
        "eval_cfg_scale,policy_cfg_scale,effective_cfg_scale,cfg_scale,robot_batch_size"
    )
    assert "gradient_accumulate_every,num_processes,mixed_precision" in text
    assert "effective_train_batch_size_per_optimizer_step" in text
    assert "provenance_contract_version" in text
    assert "provenance_contract_id" in text
    assert "robot_action_supervision_count" in text
    assert "robot_heatmap_supervision_count" in text
    assert "image_resize_mode" in text
    assert "robot_image_resize_mode" in text
    assert "open_image_resize_mode" in text
    assert "obs_encoder_checkpoint_path_is_file" in text
    assert "obs_encoder_cache_dir_is_dir" in text
    assert "main_debug" in text
    assert "robot_only_debug" in text


def test_compare_gaze_wam_ablation_metrics_records_dino_source_path_types(monkeypatch):
    def fake_load_policy_for_eval(
        cfg,
        checkpoint,
        device,
        use_ema,
        overrides=None,
        trust_checkpoint=False,
    ):
        return object(), cfg

    def fake_evaluate_gaze_wam_sources(policy, cfg, **kwargs):
        return {"robot_num_samples": 0.0}

    monkeypatch.setattr(
        eval_gaze_wam_metrics_module,
        "load_policy_for_eval",
        fake_load_policy_for_eval,
    )
    monkeypatch.setattr(
        eval_gaze_wam_metrics_module,
        "evaluate_gaze_wam_sources",
        fake_evaluate_gaze_wam_sources,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        checkpoint = root / "dinov3.pt"
        cache_dir = root / "dino_cache"
        checkpoint.write_text("fake checkpoint", encoding="utf-8")
        cache_dir.mkdir()
        rows = compare_gaze_wam_ablation_metrics(
            variants=["main_debug=train_gaze_wam_debug_workspace"],
            overrides=[
                f"policy.obs_encoder.checkpoint_path={checkpoint}",
                f"policy.obs_encoder.cache_dir={cache_dir}",
            ],
            device="cpu",
            batch_size=2,
            max_batches=1,
            sources=("robot",),
            compute_sampling=False,
            compute_heatmap=False,
            compute_gdr=False,
        )

    row = rows[0]
    assert row["obs_encoder_pretrained"] is False
    assert row["obs_encoder_checkpoint_path"] == str(checkpoint)
    assert row["obs_encoder_checkpoint_path_exists"] is True
    assert row["obs_encoder_checkpoint_path_is_file"] is True
    assert row["obs_encoder_cache_dir"] == str(cache_dir)
    assert row["obs_encoder_cache_dir_exists"] is True
    assert row["obs_encoder_cache_dir_is_dir"] is True
    assert row["obs_encoder_local_weight_source_configured"] is True
    assert row["obs_encoder_local_weight_source_valid"] is True


def test_compare_gaze_wam_ablation_metrics_applies_overrides_to_checkpoint_cfg(monkeypatch):
    calls = {}

    def fake_load_policy_for_eval(
        cfg,
        checkpoint,
        device,
        use_ema,
        overrides=None,
        trust_checkpoint=False,
    ):
        calls["cfg_arg"] = cfg
        calls["checkpoint"] = checkpoint
        calls["device"] = device
        calls["use_ema"] = use_ema
        calls["overrides"] = list(overrides or [])
        assert cfg is None
        cfg_after_overrides = load_cfg(
            "train_gaze_wam_robot_only_debug_workspace",
            overrides=overrides,
        )
        with open_dict(cfg_after_overrides.training):
            cfg_after_overrides.training.num_processes = 8
            cfg_after_overrides.training.mixed_precision = "bf16"
            cfg_after_overrides.training.distributed_type = "MULTI_GPU"
            cfg_after_overrides.training.robot_batch_size_per_process = 48
            cfg_after_overrides.training.open_batch_size_per_process = 16
            cfg_after_overrides.training.train_batch_size_per_process = 64
            cfg_after_overrides.training.robot_ratio = 0.75
            cfg_after_overrides.training.open_ratio = 0.25
            cfg_after_overrides.training.effective_robot_batch_size_per_optimizer_step = 384
            cfg_after_overrides.training.effective_open_batch_size_per_optimizer_step = 128
            cfg_after_overrides.training.effective_train_batch_size_per_optimizer_step = 512
        return object(), cfg_after_overrides

    def fake_evaluate_gaze_wam_sources(policy, cfg, **kwargs):
        calls["eval_robot_path"] = str(cfg.task.robot_dataset.dataset_path)
        calls["eval_robot_batch"] = int(cfg.robot_dataloader.batch_size)
        calls["eval_kwargs"] = kwargs
        return {"robot_num_samples": 0.0}

    monkeypatch.setattr(
        eval_gaze_wam_metrics_module,
        "load_policy_for_eval",
        fake_load_policy_for_eval,
    )
    monkeypatch.setattr(
        eval_gaze_wam_metrics_module,
        "evaluate_gaze_wam_sources",
        fake_evaluate_gaze_wam_sources,
    )

    rows = compare_gaze_wam_ablation_metrics(
        variants=["ckpt=train_gaze_wam_robot_only_debug_workspace:data/ckpt/latest.ckpt"],
        overrides=[
            "task.robot_dataset_path=data/real_robot_eval.zarr",
            "robot_dataloader.batch_size=7",
        ],
        device="cpu",
        batch_size=2,
        max_batches=1,
        sources=("robot",),
        compute_sampling=False,
        compute_heatmap=False,
        compute_gdr=False,
    )

    assert calls["cfg_arg"] is None
    assert calls["checkpoint"] == "data/ckpt/latest.ckpt"
    assert calls["overrides"] == [
        "task.robot_dataset_path=data/real_robot_eval.zarr",
        "robot_dataloader.batch_size=7",
    ]
    assert calls["eval_robot_path"] == "data/real_robot_eval.zarr"
    assert calls["eval_robot_batch"] == 7
    assert rows[0]["checkpoint_provided"] is True
    assert rows[0]["robot_batch_size"] == 48
    assert rows[0]["open_batch_size"] == 16
    assert rows[0]["robot_ratio"] == 0.75
    assert rows[0]["open_ratio"] == 0.25
    assert rows[0]["gradient_accumulate_every"] == 1
    assert rows[0]["num_processes"] == 8
    assert rows[0]["mixed_precision"] == "bf16"
    assert rows[0]["distributed_type"] == "MULTI_GPU"
    assert rows[0]["effective_robot_batch_size_per_optimizer_step"] == 384
    assert rows[0]["effective_open_batch_size_per_optimizer_step"] == 128
    assert rows[0]["effective_train_batch_size_per_optimizer_step"] == 512
    assert rows[0]["provenance_contract_id"] == provenance_contract_id(rows[0])


def test_compare_gaze_wam_ablation_metrics_applies_variant_overrides(monkeypatch):
    load_calls = []

    def fake_load_policy_for_eval(
        cfg,
        checkpoint,
        device,
        use_ema,
        overrides=None,
        trust_checkpoint=False,
    ):
        load_calls.append(list(overrides or []))
        cfg_after_overrides = load_cfg("train_gaze_wam_debug_workspace", overrides=overrides)
        return object(), cfg_after_overrides

    def fake_evaluate_gaze_wam_sources(policy, cfg, **kwargs):
        return {"robot_num_samples": float(cfg.robot_dataloader.batch_size)}

    monkeypatch.setattr(
        eval_gaze_wam_metrics_module,
        "load_policy_for_eval",
        fake_load_policy_for_eval,
    )
    monkeypatch.setattr(
        eval_gaze_wam_metrics_module,
        "evaluate_gaze_wam_sources",
        fake_evaluate_gaze_wam_sources,
    )

    rows = compare_gaze_wam_ablation_metrics(
        variants=[
            "main=train_gaze_wam_debug_workspace",
            "open_ratio_90_10=train_gaze_wam_debug_workspace",
        ],
        overrides=["task.robot_dataset_path=data/eval_robot.zarr"],
        variant_overrides=[
            ("open_ratio_90_10", "robot_dataloader.batch_size=9"),
            ("open_ratio_90_10", "open_dataloader.batch_size=1"),
        ],
        device="cpu",
        batch_size=2,
        max_batches=1,
        sources=("robot",),
        compute_sampling=False,
        compute_heatmap=False,
        compute_gdr=False,
    )

    assert load_calls[0] == ["task.robot_dataset_path=data/eval_robot.zarr"]
    assert load_calls[1] == [
        "task.robot_dataset_path=data/eval_robot.zarr",
        "robot_dataloader.batch_size=9",
        "open_dataloader.batch_size=1",
    ]
    assert rows[0]["robot_batch_size"] == 3
    assert rows[0]["open_batch_size"] == 1
    assert rows[0]["global_overrides"] == "task.robot_dataset_path=data/eval_robot.zarr"
    assert rows[0]["variant_overrides"] == ""
    assert rows[1]["robot_batch_size"] == 9
    assert rows[1]["open_batch_size"] == 1
    assert rows[1]["robot_ratio"] == 0.9
    assert rows[1]["open_ratio"] == 0.1
    assert rows[1]["global_overrides"] == "task.robot_dataset_path=data/eval_robot.zarr"
    assert rows[1]["variant_overrides"] == (
        "robot_dataloader.batch_size=9 open_dataloader.batch_size=1"
    )
    assert rows[0]["provenance_contract_id"] != rows[1]["provenance_contract_id"]


def test_compare_gaze_wam_ablation_metrics_uses_strict_bool_provenance(monkeypatch):
    def fake_load_policy_for_eval(
        cfg,
        checkpoint,
        device,
        use_ema,
        overrides=None,
        trust_checkpoint=False,
    ):
        cfg_after_overrides = load_cfg("train_gaze_wam_debug_workspace")
        cfg_after_overrides.task.robot_heatmap_on_gaze_dropout = "false"
        cfg_after_overrides.task.robot_dataset.action_padding = "off"
        cfg_after_overrides.task.open_dataset.action_padding = "0"
        cfg_after_overrides.policy.use_block_attention_mask = "false"
        cfg_after_overrides.policy.obs_encoder.pretrained = "on"
        cfg_after_overrides.policy.cfg_scale = "1.25"
        return object(), cfg_after_overrides

    def fake_evaluate_gaze_wam_sources(policy, cfg, **kwargs):
        return {"robot_num_samples": 0.0}

    monkeypatch.setattr(
        eval_gaze_wam_metrics_module,
        "load_policy_for_eval",
        fake_load_policy_for_eval,
    )
    monkeypatch.setattr(
        eval_gaze_wam_metrics_module,
        "evaluate_gaze_wam_sources",
        fake_evaluate_gaze_wam_sources,
    )

    rows = compare_gaze_wam_ablation_metrics(
        variants=["main=train_gaze_wam_debug_workspace"],
        device="cpu",
        batch_size=2,
        max_batches=1,
        sources=("robot",),
        compute_sampling=False,
        compute_heatmap=False,
        compute_gdr=False,
    )

    row = rows[0]
    assert row["robot_heatmap_on_gaze_dropout"] is False
    assert row["robot_action_padding"] is False
    assert row["open_action_padding"] is False
    assert row["use_block_attention_mask"] is False
    assert row["obs_encoder_pretrained"] is True
    assert row["policy_cfg_scale"] == 1.25
    assert row["effective_cfg_scale"] == 1.25


def test_compare_gaze_wam_ablation_metrics_rejects_boolean_cfg_scale(monkeypatch):
    def fake_load_policy_for_eval(
        cfg,
        checkpoint,
        device,
        use_ema,
        overrides=None,
        trust_checkpoint=False,
    ):
        cfg_after_overrides = load_cfg("train_gaze_wam_debug_workspace")
        cfg_after_overrides.policy.cfg_scale = True
        return object(), cfg_after_overrides

    monkeypatch.setattr(
        eval_gaze_wam_metrics_module,
        "load_policy_for_eval",
        fake_load_policy_for_eval,
    )

    try:
        compare_gaze_wam_ablation_metrics(
            variants=["main=train_gaze_wam_debug_workspace"],
            device="cpu",
            batch_size=2,
            max_batches=1,
            sources=("robot",),
            compute_sampling=False,
            compute_heatmap=False,
            compute_gdr=False,
        )
    except ValueError as exc:
        assert "policy.cfg_scale" in str(exc)
        assert "finite non-negative float" in str(exc)
    else:
        raise AssertionError("Expected boolean policy.cfg_scale to fail in compare provenance.")


def test_gaze_wam_eval_sources_uses_strict_robot_heatmap_bool(monkeypatch):
    cfg = load_cfg("train_gaze_wam_debug_workspace")
    cfg.task.robot_heatmap_on_gaze_dropout = "false"
    captured = {}

    def fake_evaluate_gaze_wam_dataset(**kwargs):
        captured["robot_heatmap_on_gaze_dropout"] = kwargs[
            "robot_heatmap_on_gaze_dropout"
        ]
        return {"robot_num_samples": 0.0}

    monkeypatch.setattr(
        eval_gaze_wam_metrics_module.hydra.utils,
        "instantiate",
        lambda dataset_cfg: object(),
    )
    monkeypatch.setattr(
        eval_gaze_wam_metrics_module,
        "evaluate_gaze_wam_dataset",
        fake_evaluate_gaze_wam_dataset,
    )

    metrics = eval_gaze_wam_metrics_module.evaluate_gaze_wam_sources(
        policy=object(),
        cfg=cfg,
        sources=("robot",),
        batch_size=2,
        max_batches=1,
        device="cpu",
        validate_zarr=False,
    )

    assert metrics["robot_num_samples"] == 0.0
    assert captured["robot_heatmap_on_gaze_dropout"] is False


def test_compare_gaze_wam_ablation_metrics_rejects_unknown_variant_override():
    try:
        compare_gaze_wam_ablation_metrics(
            variants=["main=train_gaze_wam_debug_workspace"],
            variant_overrides=[("typo", "robot_dataloader.batch_size=9")],
            device="cpu",
            batch_size=2,
            max_batches=1,
            sources=("robot",),
            compute_sampling=False,
            compute_heatmap=False,
            compute_gdr=False,
        )
    except ValueError as exc:
        assert "unknown variant" in str(exc)
        assert "typo" in str(exc)
    else:
        raise AssertionError("Expected unknown variant-specific override to fail.")


def test_compare_gaze_wam_ablation_metrics_threads_zarr_timestamp_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_linear_action_zarr(Path(tmpdir) / "robot.zarr", length=8)
        _add_timestamp_arrays(robot_path)

        try:
            compare_gaze_wam_ablation_metrics(
                variants=["robot_only_debug=train_gaze_wam_robot_only_debug_workspace"],
                overrides=[
                    f"task.robot_dataset_path={robot_path}",
                ],
                device="cpu",
                batch_size=2,
                max_batches=1,
                sources=("robot",),
                compute_denoising_loss=False,
                compute_sampling=False,
                compute_heatmap=False,
                compute_gdr=False,
                validate_zarr=True,
                require_timestamps=True,
                timestamp_max_step=0.01,
            )
        except ValueError as exc:
            assert "robot zarr validation failed" in str(exc)
            assert "max_step" in str(exc)
        else:
            raise AssertionError("Expected timestamp validation to block ablation metrics.")


def test_plan_gaze_wam_experiments_builds_debug_plan_and_outputs():
    plan = build_gaze_wam_experiment_plan(
        debug=True,
        use_accelerate=True,
        train_overrides=["training.max_train_steps=1"],
        eval_overrides=["task.robot_dataset_path=data/debug_gaze_wam/robot.zarr"],
        checkpoint_template="data/outputs/{name}/checkpoints/latest.ckpt",
        eval_device="cpu",
        eval_batch_size=2,
        eval_max_batches=1,
        eval_sources=("robot",),
        skip_sampling=True,
        skip_heatmap=True,
        skip_gdr=True,
        validate_zarr=True,
        timestamp_key="timestamp",
        require_timestamps=True,
        timestamp_max_step=0.08,
        metrics_json="tmp/metrics.json",
        metrics_csv="tmp/metrics.csv",
    )

    assert plan["mode"] == "debug"
    assert plan["contract_summary"]["image_resize_mode"] == "stretch"
    assert plan["contract_summary"]["image_shape"] == "3x256x256"
    assert plan["contract_summary"]["visual_token_count"] == 512
    assert len(plan["train_jobs"]) == 6
    assert [job["variant"] for job in plan["train_jobs"]] == [
        "robot_only_debug",
        "mixed_debug",
        "open_ratio_100_0_debug",
        "open_ratio_90_10_debug",
        "open_ratio_75_25_debug",
        "open_ratio_50_50_debug",
    ]
    robot_only_job = plan["train_jobs"][0]
    mixed_job = plan["train_jobs"][1]
    assert robot_only_job["config_name"] == "train_gaze_wam_robot_only_debug_workspace"
    assert robot_only_job["robot_batch_size"] == 3
    assert robot_only_job["open_batch_size"] == 0
    assert robot_only_job["robot_ratio"] == 1.0
    assert robot_only_job["open_ratio"] == 0.0
    assert robot_only_job["training_stage"] == "robot_only"
    assert robot_only_job["total_batch_size_per_process"] == 3
    assert robot_only_job["effective_robot_batch_size_per_optimizer_step"] == 24
    assert robot_only_job["effective_open_batch_size_per_optimizer_step"] == 0
    assert robot_only_job["effective_train_batch_size_per_optimizer_step"] == 24
    assert mixed_job["config_name"] == "train_gaze_wam_debug_workspace"
    assert mixed_job["robot_batch_size"] == 3
    assert mixed_job["open_batch_size"] == 1
    assert mixed_job["robot_ratio"] == 0.75
    assert mixed_job["open_ratio"] == 0.25
    assert mixed_job["training_stage"] == "mixed_train"
    assert mixed_job["gradient_accumulate_every"] == 1
    assert mixed_job["num_processes"] == 8
    assert mixed_job["mixed_precision"] == "bf16"
    assert mixed_job["distributed_type"] == "MULTI_GPU"
    assert mixed_job["effective_train_batch_size_per_optimizer_step"] == 32
    assert mixed_job["robot_gaze_dropout_prob"] == 0.2
    assert mixed_job["robot_heatmap_on_gaze_dropout"] is True
    assert mixed_job["use_block_attention_mask"] is True
    assert mixed_job["heatmap_objective"] == "dsnt_js"
    assert mixed_job["image_resize_mode"] == "stretch"
    assert mixed_job["robot_image_resize_mode"] == "stretch"
    assert mixed_job["open_image_resize_mode"] == "stretch"
    assert mixed_job["obs_encoder_model_name"] == "vit_base_patch16_dinov3"
    assert mixed_job["obs_encoder_pretrained"] is False
    assert mixed_job["obs_encoder_checkpoint_path"] == ""
    assert mixed_job["obs_encoder_checkpoint_path_exists"] is False
    assert mixed_job["obs_encoder_checkpoint_path_is_file"] is False
    assert mixed_job["obs_encoder_cache_dir"] == ""
    assert mixed_job["obs_encoder_cache_dir_exists"] is False
    assert mixed_job["obs_encoder_cache_dir_is_dir"] is False
    assert mixed_job["obs_encoder_local_weight_source_configured"] is False
    assert mixed_job["obs_encoder_local_weight_source_valid"] is True
    assert mixed_job["provenance_contract_version"] == PROVENANCE_CONTRACT_VERSION
    assert mixed_job["provenance_contract_id"] == provenance_contract_id(mixed_job)
    assert robot_only_job["command"][:4] == [
        "accelerate",
        "launch",
        "--config_file",
        "accelerate/8gpu-amp.yaml",
    ]
    assert "training.max_train_steps=1" in robot_only_job["command"]
    eval_command = plan["eval_job"]["command"]
    assert eval_command[:2] == ["py", "scripts/compare_gaze_wam_ablation_metrics.py"]
    assert "--variant" in eval_command
    assert (
        "robot_only_debug=train_gaze_wam_robot_only_debug_workspace:"
        "data/outputs/robot_only_debug/checkpoints/latest.ckpt"
    ) in eval_command
    assert (
        "mixed_debug=train_gaze_wam_debug_workspace:"
        "data/outputs/mixed_debug/checkpoints/latest.ckpt"
    ) in eval_command
    assert "--skip-sampling" in eval_command
    assert "--skip-heatmap" in eval_command
    assert "--skip-gdr" in eval_command
    assert "--validate-zarr" in eval_command
    assert "--timestamp-key" in eval_command
    assert "timestamp" in eval_command
    assert "--require-timestamps" in eval_command
    assert "--timestamp-max-step" in eval_command
    assert "0.08" in eval_command
    assert "--variant-override" in eval_command
    assert len(plan["eval_job"]["variant_jobs"]) == 6
    assert plan["eval_job"]["variant_jobs"][0]["variant"] == "robot_only_debug"
    assert plan["eval_job"]["variant_jobs"][0]["checkpoint"] == (
        "data/outputs/robot_only_debug/checkpoints/latest.ckpt"
    )
    assert plan["eval_job"]["variant_jobs"][0]["obs_encoder_pretrained"] is False
    assert plan["eval_job"]["variant_jobs"][0]["obs_encoder_local_weight_source_valid"] is True
    assert plan["eval_job"]["variant_jobs"][0]["provenance_contract_id"] == robot_only_job[
        "provenance_contract_id"
    ]
    assert plan["eval_validation"] == {
        "validate_zarr": True,
        "timestamp_key": "timestamp",
        "require_timestamps": True,
        "timestamp_max_delta": None,
        "timestamp_max_step": 0.08,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "plan.csv"
        script_path = Path(tmpdir) / "run_plan.sh"
        write_plan_csv(plan, str(csv_path))
        write_plan_script(plan, str(script_path))
        csv_text = csv_path.read_text(encoding="utf-8")
        script_text = script_path.read_text(encoding="utf-8")

    assert (
        "job_type,variant,config_name,checkpoint,provenance_contract_version,"
        "provenance_contract_id,robot_batch_size,open_batch_size"
    ) in csv_text
    assert "gradient_accumulate_every,num_processes,mixed_precision" in csv_text
    assert "effective_train_batch_size_per_optimizer_step" in csv_text
    assert "image_resize_mode,robot_image_resize_mode,open_image_resize_mode" in csv_text
    assert "obs_encoder_pretrained,obs_encoder_checkpoint_path" in csv_text
    assert "obs_encoder_cache_dir_is_dir,obs_encoder_local_weight_source_configured" in csv_text
    assert "robot_only_debug" in csv_text
    assert "mixed_debug" in csv_text
    assert "compare_gaze_wam_ablation_metrics.py" in csv_text
    assert "set -euo pipefail" in script_text
    assert "training.max_train_steps=1" in script_text
    assert "evaluating ablations" in script_text


def test_plan_gaze_wam_experiments_accepts_custom_variants_without_checkpoints():
    plan = build_gaze_wam_experiment_plan(
        variants=["tiny=train_gaze_wam_debug_workspace"],
        checkpoint_template="",
        use_accelerate=False,
        eval_sources=("robot", "open"),
        validate_zarr=False,
    )

    assert len(plan["train_jobs"]) == 1
    assert plan["train_jobs"][0]["command"][:3] == ["py", "train.py", "--config-name"]
    assert plan["train_jobs"][0]["robot_batch_size"] == 3
    assert plan["train_jobs"][0]["open_batch_size"] == 1
    assert plan["train_jobs"][0]["image_resize_mode"] == "stretch"
    assert plan["train_jobs"][0]["obs_encoder_pretrained"] is False
    assert plan["train_jobs"][0]["obs_encoder_local_weight_source_configured"] is False
    assert plan["train_jobs"][0]["obs_encoder_local_weight_source_valid"] is True
    assert plan["eval_job"]["command"].count("--variant") == 1
    assert "tiny=train_gaze_wam_debug_workspace" in plan["eval_job"]["command"]
    assert "tiny=train_gaze_wam_debug_workspace:" not in plan["eval_job"]["command"]
    assert "--variant-override" not in plan["eval_job"]["command"]
    assert len(plan["eval_job"]["variant_jobs"]) == 1
    assert plan["eval_job"]["variant_jobs"][0]["variant"] == "tiny"
    assert plan["eval_job"]["variant_jobs"][0]["checkpoint"] == ""
    assert plan["eval_job"]["variant_jobs"][0]["obs_encoder_pretrained"] is False
    assert "--no-validate-zarr" in plan["eval_job"]["command"]
    assert plan["eval_validation"]["validate_zarr"] is False


def test_plan_gaze_wam_experiments_training_scale_changes_provenance():
    base = build_gaze_wam_experiment_plan(
        variants=["main=train_gaze_wam_debug_workspace"],
        use_accelerate=True,
        checkpoint_template="",
        validate_zarr=False,
    )
    accumulated = build_gaze_wam_experiment_plan(
        variants=["main=train_gaze_wam_debug_workspace"],
        use_accelerate=True,
        train_overrides=["training.gradient_accumulate_every=2"],
        checkpoint_template="",
        validate_zarr=False,
    )

    base_job = base["train_jobs"][0]
    accumulated_job = accumulated["train_jobs"][0]
    assert base_job["gradient_accumulate_every"] == 1
    assert accumulated_job["gradient_accumulate_every"] == 2
    assert base_job["num_processes"] == 8
    assert accumulated_job["num_processes"] == 8
    assert accumulated_job["effective_train_batch_size_per_optimizer_step"] == 64
    assert accumulated_job["provenance_contract_id"] == provenance_contract_id(accumulated_job)
    assert base_job["provenance_contract_id"] != accumulated_job["provenance_contract_id"]


def test_plan_gaze_wam_experiments_uses_strict_bool_overrides():
    plan = build_gaze_wam_experiment_plan(
        variants=["main=train_gaze_wam_debug_workspace"],
        use_accelerate=False,
        train_overrides=[
            "task.robot_heatmap_on_gaze_dropout=off",
            "policy.use_block_attention_mask=false",
            "policy.obs_encoder.pretrained=on",
        ],
        checkpoint_template="",
        validate_zarr=False,
    )

    job = plan["train_jobs"][0]
    eval_job = plan["eval_job"]["variant_jobs"][0]
    assert job["robot_heatmap_on_gaze_dropout"] is False
    assert job["use_block_attention_mask"] is False
    assert job["obs_encoder_pretrained"] is True
    assert eval_job["robot_heatmap_on_gaze_dropout"] is False
    assert eval_job["use_block_attention_mask"] is False
    assert eval_job["obs_encoder_pretrained"] is True


def test_plan_gaze_wam_experiments_can_route_train_jobs_through_real_data_launcher():
    plan = build_gaze_wam_experiment_plan(
        variants=["main=train_gaze_wam_workspace"],
        use_accelerate=True,
        train_via_launcher=True,
        real_data_launch=True,
        launcher_report_template="data/outputs/{name}/launch_report.json",
        data_onboarding_review_template="data/reviews/{name}_onboarding.json",
        require_data_onboarding_review=True,
        train_overrides=["policy.obs_encoder.pretrained=true"],
        include_sweeps=["open_ratio"],
        checkpoint_template="",
        validate_zarr=True,
        require_timestamps=True,
        timestamp_max_step=0.08,
        train_fail_on_zarr_warning=True,
    )

    assert plan["train_launch"] == {
        "train_via_launcher": True,
        "real_data_launch": True,
        "real_data_contract": "ablation",
        "launcher_report_template": "data/outputs/{name}/launch_report.json",
        "data_onboarding_review_template": "data/reviews/{name}_onboarding.json",
        "require_data_onboarding_review": True,
        "train_fail_on_zarr_warning": True,
    }
    main_command = plan["train_jobs"][0]["command"]
    assert main_command[:5] == [
        "py",
        "scripts/launch_gaze_wam_training.py",
        "--config-name",
        "train_gaze_wam_workspace",
        "--task",
    ]
    assert "--accelerate" in main_command
    assert "--accelerate-config" in main_command
    assert "--real-data" in main_command
    assert "--run" == main_command[-1]
    assert "--preflight-require-timestamps" in main_command
    assert "--preflight-timestamp-max-step" in main_command
    assert "0.08" in main_command
    assert "--preflight-fail-on-zarr-warning" in main_command
    assert "--data-onboarding-review-json" in main_command
    assert "data/reviews/main_onboarding.json" in main_command
    assert "--require-data-onboarding-review" in main_command
    assert "--output-json" in main_command
    assert "data/outputs/main/launch_report.json" in main_command
    assert "--real-data-contract" in main_command
    assert "ablation" in main_command
    assert "--override" in main_command
    assert "policy.obs_encoder.pretrained=true" in main_command

    open_ratio_job = next(job for job in plan["train_jobs"] if job["variant"] == "open_ratio_90_10")
    open_ratio_command = open_ratio_job["command"]
    assert "data/outputs/open_ratio_90_10/launch_report.json" in open_ratio_command
    assert "data/reviews/open_ratio_90_10_onboarding.json" in open_ratio_command
    assert "--override" in open_ratio_command
    assert "robot_dataloader.batch_size=58" in open_ratio_command
    assert "open_dataloader.batch_size=6" in open_ratio_command
    assert open_ratio_job["robot_ratio"] == 58 / 64
    assert open_ratio_job["open_ratio"] == 6 / 64
    assert "--variant-override" in plan["eval_job"]["command"]


def test_plan_gaze_wam_experiments_adds_preset_sweep_variants():
    plan = build_gaze_wam_experiment_plan(
        debug=True,
        include_sweeps=["gaze_dropout", "open_ratio"],
        checkpoint_template="data/outputs/{name}/latest.ckpt",
        use_accelerate=False,
        validate_zarr=False,
    )

    names = [job["variant"] for job in plan["train_jobs"]]
    assert "gaze_dropout_0p1_debug" in names
    assert "gaze_dropout_0p3_debug" in names
    assert "open_ratio_100_0_debug" in names
    assert "open_ratio_50_50_debug" in names
    assert plan["include_sweeps"] == ["gaze_dropout", "open_ratio"]

    dropout_job = next(job for job in plan["train_jobs"] if job["variant"] == "gaze_dropout_0p1_debug")
    assert dropout_job["config_name"] == "train_gaze_wam_debug_workspace"
    assert dropout_job["overrides"] == ["task.robot_gaze_dropout_prob=0.1"]
    assert "task.robot_gaze_dropout_prob=0.1" in dropout_job["command"]
    dropout_eval_job = next(
        job for job in plan["eval_job"]["variant_jobs"] if job["variant"] == "gaze_dropout_0p1_debug"
    )
    assert dropout_eval_job["robot_gaze_dropout_prob"] == 0.1
    assert dropout_eval_job["provenance_contract_id"] == dropout_job["provenance_contract_id"]

    open_ratio_job = next(job for job in plan["train_jobs"] if job["variant"] == "open_ratio_90_10_debug")
    assert "robot_dataloader.batch_size=9" in open_ratio_job["overrides"]
    assert "open_dataloader.batch_size=1" in open_ratio_job["overrides"]
    assert open_ratio_job["robot_batch_size"] == 9
    assert open_ratio_job["open_batch_size"] == 1
    assert open_ratio_job["robot_ratio"] == 0.9
    assert open_ratio_job["open_ratio"] == 0.1
    open_ratio_eval_job = next(
        job for job in plan["eval_job"]["variant_jobs"] if job["variant"] == "open_ratio_90_10_debug"
    )
    assert open_ratio_eval_job["provenance_contract_id"] == open_ratio_job["provenance_contract_id"]
    assert open_ratio_eval_job["robot_ratio"] == 0.9
    assert open_ratio_eval_job["open_ratio"] == 0.1
    eval_command = plan["eval_job"]["command"]
    assert "--variant-override" in eval_command
    assert "gaze_dropout_0p1_debug" in eval_command
    assert "task.robot_gaze_dropout_prob=0.1" in eval_command
    assert "open_ratio_90_10_debug" in eval_command
    assert "robot_dataloader.batch_size=9" in eval_command
    assert "open_dataloader.batch_size=1" in eval_command


def test_plan_gaze_wam_experiments_records_dino_source_overrides():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Path(tmpdir) / "dinov3.ckpt"
        checkpoint.write_text("fake checkpoint", encoding="utf-8")
        cache_dir = Path(tmpdir) / "cache"
        cache_dir.mkdir()

        plan = build_gaze_wam_experiment_plan(
            variants=["main=train_gaze_wam_workspace"],
            train_overrides=[
                "policy.obs_encoder.pretrained=true",
                f"policy.obs_encoder.checkpoint_path={checkpoint}",
                f"policy.obs_encoder.cache_dir={cache_dir}",
            ],
            eval_overrides=[
                "policy.obs_encoder.pretrained=true",
                f"policy.obs_encoder.checkpoint_path={checkpoint}",
                f"policy.obs_encoder.cache_dir={cache_dir}",
            ],
            checkpoint_template="",
            validate_zarr=False,
        )

    job = plan["train_jobs"][0]
    assert job["obs_encoder_pretrained"] is True
    assert job["obs_encoder_checkpoint_path"] == str(checkpoint)
    assert job["obs_encoder_checkpoint_path_exists"] is True
    assert job["obs_encoder_checkpoint_path_is_file"] is True
    assert job["obs_encoder_cache_dir"] == str(cache_dir)
    assert job["obs_encoder_cache_dir_exists"] is True
    assert job["obs_encoder_cache_dir_is_dir"] is True
    assert job["obs_encoder_local_weight_source_configured"] is True
    assert job["obs_encoder_local_weight_source_valid"] is True
    eval_job = plan["eval_job"]["variant_jobs"][0]
    assert eval_job["obs_encoder_pretrained"] is True
    assert eval_job["obs_encoder_checkpoint_path"] == str(checkpoint)
    assert eval_job["obs_encoder_checkpoint_path_is_file"] is True
    assert eval_job["obs_encoder_cache_dir"] == str(cache_dir)
    assert eval_job["obs_encoder_cache_dir_is_dir"] is True


def test_preflight_gaze_wam_debug_config_runs_loss_smoke(tmp_path):
    encoder_path, decoder_path = _write_fake_cosmos_jit_pair(
        tmp_path,
        image_size=(256, 256),
        token_grid=(16, 16),
        latent_channels=16,
    )
    summary = preflight_gaze_wam(
        config_name="train_gaze_wam_debug_workspace",
        overrides=[
            f"policy.heatmap_cosmos_encoder_path={encoder_path}",
            f"policy.heatmap_cosmos_decoder_path={decoder_path}",
        ],
        device="cpu",
        validate_zarr=True,
        run_loss_smoke=True,
    )

    assert summary["ok"] is True
    assert summary["errors"] == []
    assert summary["config"]["n_obs_steps"] == 2
    assert summary["config"]["n_latency_steps"] == 0
    assert summary["config"]["heatmap_num_tokens"] == 256
    assert summary["image_geometry"] == {
        "image_shape": [3, 256, 256],
        "task_image_size": [256, 256],
        "robot_image_size": [256, 256],
        "open_image_size": [256, 256],
        "task_image_resize_mode": "stretch",
        "robot_image_resize_mode": "stretch",
        "open_image_resize_mode": "stretch",
        "resize_modes": {
            "task": "stretch",
            "robot_dataset": "stretch",
            "open_dataset": "stretch",
        },
        "image_sizes": {
            "task": [256, 256],
            "robot_dataset": [256, 256],
            "open_dataset": [256, 256],
        },
        "supported": True,
        "all_stretch": True,
        "consistent": True,
        "image_size_consistent": True,
    }
    assert summary["sampling_contract"] == {
        "task": {
            "n_obs_steps": 2,
            "action_horizon": 48,
            "n_latency_steps": 0,
        },
        "robot_dataset": {
            "n_obs_steps": 2,
            "obs_downsample_steps": 1,
            "action_horizon": 48,
            "n_latency_steps": 0,
            "action_downsample_steps": 1,
            "action_padding": True,
        },
        "open_dataset": {
            "n_obs_steps": 2,
            "obs_downsample_steps": 1,
            "action_horizon": 48,
            "n_latency_steps": 0,
            "action_downsample_steps": 1,
            "action_padding": True,
        },
        "compare_keys": ["n_obs_steps", "action_horizon", "n_latency_steps"],
        "robot_matches_task": True,
        "open_matches_task": True,
    }
    assert summary["config"]["robot_dataset_sampling"] == {
        "n_obs_steps": 2,
        "obs_downsample_steps": 1,
        "action_horizon": 48,
        "n_latency_steps": 0,
        "action_downsample_steps": 1,
        "action_padding": True,
    }
    assert summary["config"]["open_dataset_sampling"] == {
        "n_obs_steps": 2,
        "obs_downsample_steps": 1,
        "action_horizon": 48,
        "n_latency_steps": 0,
        "action_downsample_steps": 1,
        "action_padding": True,
    }
    assert summary["robot_dataset"]["length"] > 0
    assert summary["robot_dataset"]["val_length"] >= 0
    assert summary["open_dataset"]["length"] > 0
    assert summary["open_dataset"]["val_length"] >= 0
    assert summary["dataset_lengths"] == {
        "robot_train_samples": summary["robot_dataset"]["length"],
        "robot_val_samples": summary["robot_dataset"]["val_length"],
        "open_train_samples": summary["open_dataset"]["length"],
        "open_val_samples": summary["open_dataset"]["val_length"],
    }
    assert summary["dataloader_batches"]["robot_train_batches"] > 0
    assert summary["dataloader_batches"]["open_train_batches"] > 0
    assert summary["dataloader_batches"]["robot_val_batches"] == 0
    assert summary["dataloader_batches"]["open_val_batches"] == 0
    assert summary["data_stream_contract"]["source"] == "two_zarr_two_dataset_online_mixed_batch"
    assert summary["data_stream_contract"]["separate_zarr_sources"] is True
    assert summary["data_stream_contract"]["offline_merged_zarr"] is False
    assert summary["data_stream_contract"]["robot"]["dataset_class_matches_expected"] is True
    assert summary["data_stream_contract"]["open"]["dataset_class_matches_expected"] is True
    assert summary["data_stream_contract"]["robot"]["dataloader"] == "robot_dataloader"
    assert summary["data_stream_contract"]["open"]["dataloader"] == "open_dataloader"
    assert summary["data_stream_contract"]["robot"]["batch_size_per_process"] == 3
    assert summary["data_stream_contract"]["open"]["batch_size_per_process"] == 1
    assert (
        summary["data_stream_contract"]["mixing"]["builder"]
        == "diffusion_policy.dataset.gaze_wam_mixing.build_gaze_wam_mixed_batch"
    )
    assert (
        summary["data_stream_contract"]["mixing"]["mode"]
        == "online_per_step_concat_after_fetch"
    )
    assert (
        summary["data_stream_contract"]["mixing"]["ratio_source"]
        == (
            "data_mixing.total_batch_size_per_process+"
            "data_mixing.robot_ratio+data_mixing.open_ratio"
        )
    )
    assert summary["data_stream_contract"]["mixing"]["robot_ratio_per_process"] == 0.75
    assert summary["data_stream_contract"]["mixing"]["open_ratio_per_process"] == 0.25
    assert summary["robot_dataset"]["sample"]["action_abs"] == [48, 10]
    assert summary["robot_dataset"]["sample"]["action_base_abs"] == [10]
    if "heatmap_image" in summary["open_dataset"]["sample"]:
        assert summary["open_dataset"]["sample"]["heatmap_image"][-2:] == [256, 256]
    assert summary["robot_zarr_validation"]["valid"] is True
    assert summary["open_zarr_validation"]["valid"] is True
    assert summary["zarr_presence_masks"]["robot"]["validation_key"] == "robot_zarr_validation"
    assert summary["zarr_presence_masks"]["open"]["validation_key"] == "open_zarr_validation"
    assert set(summary["zarr_presence_masks"]["robot"]["mask_keys"]).issubset(
        {"has_action_abs", "has_action_base_abs", "has_heatmap_image", "has_gaze_label"}
    )
    assert set(summary["zarr_presence_masks"]["open"]["mask_keys"]).issubset(
        {"has_action_abs", "has_action_base_abs", "has_heatmap_image", "has_gaze_label"}
    )
    assert summary["policy_contract"]["obs_encoder_model_name"] == "vit_base_patch16_dinov3"
    assert summary["policy_contract"]["obs_encoder_pretrained"] is False
    assert summary["policy_contract"]["obs_encoder_pretrained_cfg"]["hf_hub_id"] == (
        "timm/vit_base_patch16_dinov3.lvd1689m"
    )
    assert summary["policy_contract"]["obs_encoder_checkpoint_path"] == ""
    assert summary["policy_contract"]["obs_encoder_checkpoint_path_exists"] is False
    assert summary["policy_contract"]["obs_encoder_cache_dir"] == ""
    assert summary["policy_contract"]["obs_encoder_cache_dir_exists"] is False
    assert summary["policy_contract"]["obs_encoder_local_weight_source_configured"] is False
    assert summary["policy_contract"]["obs_encoder_local_weight_source_exists"] is True
    assert summary["policy_contract"]["obs_encoder_local_weight_source_valid"] is True
    assert summary["policy_contract"]["obs_encoder_feature_aggregation"] == "patch"
    assert summary["policy_contract"]["obs_encoder_downsample_ratio"] == 16
    assert summary["policy_contract"]["obs_encoder_patch_size"] == [16, 16]
    assert summary["policy_contract"]["obs_encoder_model_input_size"] == [256, 256]
    transforms = summary["policy_contract"]["obs_encoder_transforms"]["camera0_rgb"]
    assert [item["type"] for item in transforms] == ["ColorJitter", "Normalize"]
    assert transforms[-1]["mean"] == [0.485, 0.456, 0.406]
    assert transforms[-1]["std"] == [0.229, 0.224, 0.225]
    assert summary["policy_contract"]["obs_output_shape"] == [1, 512, 768]
    assert summary["policy_contract"]["visual_tokens_per_batch_item"] == 512
    assert summary["policy_contract"]["visual_tokens_per_frame"] == 256.0
    assert summary["policy_contract"]["expected_visual_tokens_per_frame"] == 256
    assert summary["policy_contract"]["expected_visual_tokens_total"] == 512
    assert summary["policy_contract"]["visual_embed_dim"] == 768
    assert summary["policy_contract"]["image_size"] == [256, 256]
    assert summary["policy_contract"]["inferred_patch_size_from_heatmap_grid"] == 16
    assert summary["policy_contract"]["gaze_mask_token_shape"] == [1, 1, 768]
    assert summary["policy_contract"]["gaze_encoder_grid"] == [8, 8]
    assert summary["policy_contract"]["model_action_dim"] == 10
    assert summary["policy_contract"]["model_action_horizon"] == 48
    assert summary["policy_contract"]["model_heatmap_num_tokens"] == 256
    assert summary["policy_contract"]["action_head_out_features"] == 10
    assert summary["policy_contract"]["model_heatmap_dim"] == 16
    assert summary["policy_contract"]["heatmap_head_out_features"] == 16
    assert summary["policy_contract"]["use_block_attention_mask"] is True
    attention_contract = summary["policy_contract"]["attention_contract"]
    assert {
        key: attention_contract[key]
        for key in (
            "use_block_attention_mask",
            "num_image_tokens",
            "gaze_token_count",
            "action_horizon",
            "heatmap_num_tokens",
            "train_sequence_tokens",
            "inference_sequence_tokens",
            "condition_reads_targets",
            "action_reads_heatmap",
            "heatmap_reads_action",
            "action_inference_drops_heatmap",
        )
    } == {
        "use_block_attention_mask": True,
        "num_image_tokens": 512,
        "gaze_token_count": 1,
        "action_horizon": 48,
        "heatmap_num_tokens": 256,
        "train_sequence_tokens": 817,
        "inference_sequence_tokens": 561,
        "condition_reads_targets": False,
        "action_reads_heatmap": False,
        "heatmap_reads_action": False,
        "action_inference_drops_heatmap": True,
    }
    assert attention_contract["architecture"] == "cached_dual_stream"
    assert attention_contract["shared_world_kv_cache"] is True
    assert attention_contract["world_cache_consumed_by_action"] is True
    assert attention_contract["action_reads_heatmap_world_cache"] is True
    assert attention_contract["action_reads_noisy_heatmap"] is False
    assert attention_contract["action_reads_heatmap_target"] is False
    assert summary["policy_contract"]["loss_routing_contract"]["source"] == "policy"
    assert summary["policy_contract"]["loss_routing_contract"][
        "dynamic_head_freezing"
    ] is False
    assert summary["policy_contract"]["loss_routing_contract"][
        "action_loss_mask"
    ] == "(~is_open) & has_action"
    assert summary["policy_contract"]["loss_routing_contract"][
        "heatmap_loss_mask"
    ] == "has_heatmap & has_gaze_label"
    assert summary["policy_contract"]["loss_routing_contract"]["open_rows"][
        "trains_action"
    ] is False
    assert summary["policy_contract"]["loss_routing_contract"]["open_rows"][
        "trains_heatmap"
    ] == "xy DSNT plus generated Gaussian JS target"
    assert summary["policy_contract"]["loss_routing_contract"]["open_rows"][
        "use_gaze_condition"
    ] is False
    assert summary["policy_contract"]["loss_routing_contract"]["robot_real_gaze_rows"][
        "trains_action"
    ] is True
    assert summary["policy_contract"]["loss_routing_contract"]["robot_real_gaze_rows"][
        "trains_heatmap"
    ] == "has_heatmap & has_gaze_label"
    assert summary["policy_contract"]["loss_routing_contract"]["robot_masked_gaze_rows"][
        "trains_action"
    ] is True
    assert summary["policy_contract"]["loss_routing_contract"]["robot_masked_gaze_rows"][
        "trains_heatmap"
    ] == "has_heatmap & has_gaze_label"
    assert summary["policy_contract"]["loss_routing_contract"]["validation"][
        "inactive_action_rows_must_be_zero_placeholders"
    ] is True
    assert summary["policy_contract"]["loss_routing_contract"]["validation"][
        "inactive_heatmap_rows_must_be_zero_placeholders"
    ] is True
    assert summary["policy_contract"]["loss_routing_contract"]["validation"][
        "inactive_gaze_rows_must_be_zero_placeholders"
    ] is True
    assert summary["policy_contract"]["loss_routing_contract"]["validation"][
        "inactive_optional_metadata_rows_must_be_zero_placeholders"
    ] is True
    assert summary["policy_contract"]["loss_routing_contract"]["validation"][
        "robot_real_gaze_rows_must_not_have_heatmap_loss"
    ] is True
    assert (
        summary["policy_contract"]["normalizer_contract"]["source"]
        == "robot_dataset_relative_actions_only"
    )
    assert (
        summary["policy_contract"]["normalizer_contract"]["action_normalizer_source"]
        == "GazeWamRobotDataset.get_all_actions"
    )
    assert summary["policy_contract"]["normalizer_contract"]["normalizer_keys"] == [
        "camera0_rgb",
        "action",
    ]
    assert summary["policy_contract"]["normalizer_contract"]["action_dim"] == 10
    assert (
        summary["policy_contract"]["normalizer_contract"][
            "excludes_open_source_dummy_actions"
        ]
        is True
    )
    assert (
        summary["policy_contract"]["normalizer_contract"][
            "open_source_get_normalizer_allowed"
        ]
        is False
    )
    assert summary["policy_contract"]["heatmap_codec_token_grid"] == [16, 16]
    assert summary["policy_contract"]["heatmap_codec_image_size"] == [256, 256]
    assert summary["policy_contract"]["heatmap_objective"] == "dsnt_js"
    assert summary["policy_contract"]["heatmap_token_kl_loss_weight"] == 0.0
    assert summary["policy_contract"]["heatmap_xy_loss_weight"] == 1.0
    assert summary["policy_contract"]["heatmap_point_nll_loss_weight"] == 0.0
    assert summary["policy_contract"]["heatmap_js_loss_weight"] == 1.0
    assert summary["policy_contract"]["heatmap_dsnt_temperature"] == 0.1
    assert summary["policy_contract"]["heatmap_distribution_mode"] == "intensity_softplus"
    assert summary["policy_contract"]["num_inference_steps"] == 2
    assert summary["policy_contract"]["robot_batch_size"] == 3
    assert summary["policy_contract"]["open_batch_size"] == 1
    assert summary["policy_contract"]["robot_ratio"] == 0.75
    assert summary["policy_contract"]["open_ratio"] == 0.25
    assert summary["loss_smoke"]["mixed_batch_size"] == 4
    assert summary["loss_smoke"]["mixed_action"] == [4, 48, 10]
    assert summary["loss_smoke"]["mixed_heatmap"] == [4, 1, 256, 16]
    assert summary["loss_smoke"]["action_loss_mask_count"] == 3.0
    assert summary["loss_smoke"]["heatmap_loss_mask_count"] >= 1.0
    assert summary["loss_smoke"]["heatmap_xy_loss_mask_count"] >= 1.0
    assert summary["loss_smoke"]["heatmap_xy_loss"] >= 0.0
    assert summary["loss_smoke"]["heatmap_js_loss"] >= 0.0
    assert summary["loss_smoke"]["heatmap_token_kl_loss"] >= 0.0
    assert summary["loss_smoke"]["heatmap_token_kl_loss_weight"] == 0.0
    assert summary["loss_smoke"]["heatmap_xy_loss_weight"] == 1.0
    assert summary["loss_smoke"]["heatmap_point_nll_loss_weight"] == 0.0
    assert summary["loss_smoke"]["heatmap_js_loss_weight"] == 1.0
    assert summary["loss_smoke"]["heatmap_dsnt_temperature"] == 0.1
    assert summary["loss_smoke"]["heatmap_distribution_mode"] == "intensity_softplus"
    assert summary["loss_smoke"]["routing"]["robot_rows"] == 3
    assert summary["loss_smoke"]["routing"]["open_rows"] == 1
    assert summary["loss_smoke"]["routing"]["robot_action_loss_count"] == 3
    assert summary["loss_smoke"]["routing"]["open_action_loss_count"] == 0
    assert summary["loss_smoke"]["routing"]["open_heatmap_loss_count"] == 1
    assert summary["loss_smoke"]["routing"]["robot_real_gaze_heatmap_loss_count"] == 0
    assert (
        summary["loss_smoke"]["routing"]["robot_masked_gaze_rows"]
        == summary["loss_smoke"]["routing"]["robot_heatmap_loss_count"]
    )
    assert (
        summary["loss_smoke"]["routing"]["robot_real_gaze_rows"]
        + summary["loss_smoke"]["routing"]["robot_masked_gaze_rows"]
        == 3
    )


def test_preflight_gaze_wam_robot_only_skips_open_dataset():
    summary = preflight_gaze_wam(
        config_name="train_gaze_wam_robot_only_debug_workspace",
        overrides=["task.robot_gaze_dropout_prob=1.0"],
        device="cpu",
        validate_zarr=False,
        run_loss_smoke=True,
    )

    assert summary["ok"] is True
    assert summary["open_dataset"]["skipped"] == "open_dataloader.batch_size <= 0"
    assert summary["dataset_lengths"]["robot_train_samples"] == summary["robot_dataset"]["length"]
    assert summary["dataset_lengths"]["robot_val_samples"] == summary["robot_dataset"]["val_length"]
    assert summary["dataset_lengths"]["open_train_samples"] == 0
    assert summary["dataset_lengths"]["open_val_samples"] == 0
    assert "open_zarr_validation" not in summary
    assert summary["policy_contract"]["robot_batch_size"] == 3
    assert summary["policy_contract"]["open_batch_size"] == 0
    assert summary["policy_contract"]["robot_ratio"] == 1.0
    assert summary["policy_contract"]["open_ratio"] == 0.0
    assert summary["policy_contract"]["normalizer_contract"]["action_dim"] == 10
    assert (
        summary["policy_contract"]["normalizer_contract"][
            "excludes_open_source_dummy_actions"
        ]
        is True
    )
    assert summary["loss_smoke"]["mixed_batch_size"] == 3
    assert summary["loss_smoke"]["action_loss_mask_count"] == 3.0
    assert summary["loss_smoke"]["heatmap_loss_mask_count"] == 3.0
    assert summary["loss_smoke"]["routing"]["robot_rows"] == 3
    assert summary["loss_smoke"]["routing"]["open_rows"] == 0
    assert summary["loss_smoke"]["routing"]["robot_action_loss_count"] == 3
    assert summary["loss_smoke"]["routing"]["open_action_loss_count"] == 0
    assert summary["loss_smoke"]["routing"]["robot_heatmap_loss_count"] == 3
    assert summary["loss_smoke"]["routing"]["open_heatmap_loss_count"] == 0
    assert summary["loss_smoke"]["routing"]["robot_real_gaze_heatmap_loss_count"] == 0


def test_preflight_gaze_wam_contract_runs_when_loss_smoke_is_skipped():
    summary = preflight_gaze_wam(
        config_name="train_gaze_wam_debug_workspace",
        device="cpu",
        validate_zarr=False,
        run_loss_smoke=False,
    )

    assert summary["ok"] is True
    assert summary["errors"] == []
    assert "loss_smoke" not in summary
    assert "robot_zarr_validation" not in summary
    assert "open_zarr_validation" not in summary
    assert summary["policy_contract"]["obs_output_shape"] == [1, 512, 768]
    assert summary["policy_contract"]["model_action_dim"] == 10
    assert summary["policy_contract"]["model_heatmap_num_tokens"] == 256
    assert summary["policy_contract"]["robot_ratio"] == 0.75
    assert summary["policy_contract"]["open_ratio"] == 0.25
    assert (
        summary["policy_contract"]["normalizer_contract"]["source"]
        == "robot_dataset_relative_actions_only"
    )


def test_preflight_gaze_wam_rejects_invalid_training_loop_config():
    summary = preflight_gaze_wam(
        config_name="train_gaze_wam_debug_workspace",
        overrides=[
            "training.gradient_accumulate_every=0",
            "training.checkpoint_every=0",
            "training.max_train_steps=0",
            "robot_dataloader.batch_size=0",
            "open_dataloader.batch_size=0",
        ],
        device="cpu",
        validate_zarr=False,
        run_loss_smoke=False,
    )

    assert summary["ok"] is False
    assert summary["training_config"]["valid"] is False
    assert summary["training_config"]["gradient_accumulate_every"] == 0
    assert summary["training_config"]["train_batch_size_per_process"] == 0
    assert any("training.gradient_accumulate_every" in error for error in summary["errors"])
    assert any("training.checkpoint_every" in error for error in summary["errors"])
    assert any("training.max_train_steps" in error for error in summary["errors"])
    assert any("robot_dataloader.batch_size" in error for error in summary["errors"])
    assert any("batch_size + open_dataloader.batch_size" in error for error in summary["errors"])
    assert summary["robot_dataset"] == {"skipped": "invalid training_config"}
    assert summary["open_dataset"] == {"skipped": "invalid training_config"}
    assert summary["data_stream_contract"] == {"skipped": "invalid training_config"}
    assert summary["dataset_lengths"] == {"skipped": "invalid training_config"}
    assert summary["dataloader_batches"] == {"skipped": "invalid training_config"}
    assert summary["policy_contract"] == {"skipped": "invalid training_config"}
    assert "loss_smoke" not in summary
    assert "Robot dataset check failed" not in "\n".join(summary["errors"])
    assert "Policy contract check failed" not in "\n".join(summary["errors"])


def test_preflight_gaze_wam_reports_training_config_parse_errors_without_downstream_noise():
    summary = preflight_gaze_wam(
        config_name="train_gaze_wam_debug_workspace",
        overrides=[
            "open_dataloader.batch_size=oops",
        ],
        device="cpu",
        validate_zarr=True,
        run_loss_smoke=True,
    )

    errors = "\n".join(summary["errors"])

    assert summary["ok"] is False
    assert summary["training_config"]["valid"] is False
    assert summary["training_config"]["open_batch_size"] == 0
    assert summary["config"]["open_batch_size"] == 0
    assert "open_dataloader.batch_size must be an integer" in errors
    assert summary["robot_dataset"] == {"skipped": "invalid training_config"}
    assert summary["open_dataset"] == {"skipped": "invalid training_config"}
    assert summary["data_stream_contract"] == {"skipped": "invalid training_config"}
    assert summary["dataset_lengths"] == {"skipped": "invalid training_config"}
    assert summary["dataloader_batches"] == {"skipped": "invalid training_config"}
    assert summary["robot_zarr_validation"] == {"skipped": "invalid training_config"}
    assert summary["open_zarr_validation"] == {"skipped": "invalid training_config"}
    assert summary["policy_contract"] == {"skipped": "invalid training_config"}
    assert summary["zarr_presence_masks"] == {}
    assert "loss_smoke" not in summary
    assert "Robot dataset check failed" not in errors
    assert "Open dataset check failed" not in errors
    assert "Policy contract check failed" not in errors


def test_preflight_gaze_wam_normalizes_valid_training_loop_config():
    summary = preflight_gaze_wam(
        config_name="train_gaze_wam_debug_workspace",
        overrides=[
            "training.gradient_accumulate_every='2'",
            "training.max_train_steps='1'",
            "training.max_val_steps='1'",
            "training.tqdm_interval_sec='0.25'",
            "robot_dataloader.batch_size='3'",
            "open_dataloader.batch_size='1'",
            "robot_dataloader.num_workers='0'",
            "robot_dataloader.pin_memory='false'",
            "robot_dataloader.persistent_workers='false'",
            "robot_dataloader.drop_last='false'",
            "open_dataloader.num_workers='0'",
            "open_dataloader.pin_memory='false'",
            "open_dataloader.persistent_workers='false'",
            "open_dataloader.drop_last='false'",
        ],
        device="cpu",
        validate_zarr=False,
        run_loss_smoke=False,
    )

    assert summary["ok"] is True
    assert summary["training_config"]["valid"] is True
    assert summary["training_config"]["gradient_accumulate_every"] == 2
    assert summary["training_config"]["tqdm_interval_sec"] == 0.25
    assert summary["config"]["robot_batch_size"] == 3
    assert summary["config"]["open_batch_size"] == 1
    assert summary["dataloader_batches"]["robot_train_batches"] > 0
    assert summary["dataloader_batches"]["open_train_batches"] > 0
    assert summary["policy_contract"]["robot_ratio"] == 0.75
    assert summary["policy_contract"]["open_ratio"] == 0.25


def test_preflight_gaze_wam_reports_zero_train_dataloader_batches():
    summary = preflight_gaze_wam(
        config_name="train_gaze_wam_debug_workspace",
        overrides=[
            "data_mixing.batch_size_source=dataloader",
            "robot_dataloader.batch_size=999",
            "open_dataloader.batch_size=999",
            "data_mixing.robot_tail_policy=drop",
            "data_mixing.open_tail_policy=drop",
        ],
        device="cpu",
        validate_zarr=False,
        run_loss_smoke=False,
    )

    assert summary["ok"] is False
    assert summary["dataloader_batches"]["error"].startswith("ValueError:")
    assert any("Preflight dataloader batch-count check failed" in error for error in summary["errors"])
    assert any("robot_train_batches=0" in error for error in summary["errors"])
    assert any("open_train_batches=0" in error for error in summary["errors"])


def test_preflight_gaze_wam_reports_zero_train_dataset_samples():
    summary = preflight_gaze_wam(
        config_name="train_gaze_wam_debug_workspace",
        overrides=[
            "task.action_horizon=64",
            "task.action_padding=false",
        ],
        device="cpu",
        validate_zarr=False,
        run_loss_smoke=False,
    )

    assert summary["ok"] is False
    assert summary["dataset_lengths"]["error"].startswith("ValueError:")
    assert any("Preflight dataset-length check failed" in error for error in summary["errors"])
    assert any("Robot train dataset produced zero samples" in error for error in summary["errors"])
    assert any("Open-source train dataset is enabled but produced zero samples" in error for error in summary["errors"])
    assert any("robot_train_samples=0" in error for error in summary["errors"])
    assert any("open_train_samples=0" in error for error in summary["errors"])


def test_preflight_gaze_wam_policy_contract_requires_weight_source_when_pretrained():
    summary = preflight_gaze_wam(
        config_name="train_gaze_wam_debug_workspace",
        device="cpu",
        validate_zarr=False,
        run_loss_smoke=False,
    )
    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train_gaze_wam_debug_workspace")
    contract = dict(summary["policy_contract"])
    contract.update(
        {
            "obs_encoder_pretrained": True,
            "obs_encoder_checkpoint_path": "",
            "obs_encoder_checkpoint_path_exists": False,
            "obs_encoder_cache_dir": "",
            "obs_encoder_cache_dir_exists": False,
            "obs_encoder_local_weight_source_configured": False,
            "obs_encoder_local_weight_source_exists": False,
            "obs_encoder_local_weight_source_valid": False,
            "obs_encoder_hf_hub_id": "",
        }
    )

    errors = _check_policy_contract(contract, cfg)

    assert any("requires a DINO weight source" in error for error in errors)


def test_preflight_gaze_wam_policy_contract_uses_strict_bool_fields():
    summary = preflight_gaze_wam(
        config_name="train_gaze_wam_debug_workspace",
        device="cpu",
        validate_zarr=False,
        run_loss_smoke=False,
    )
    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train_gaze_wam_debug_workspace")
    contract = dict(summary["policy_contract"])
    contract.update(
        {
            "obs_encoder_pretrained": "false",
            "obs_encoder_local_weight_source_configured": "false",
            "obs_encoder_local_weight_source_exists": "false",
            "obs_encoder_local_weight_source_valid": "false",
            "obs_encoder_checkpoint_path_is_file": "false",
            "obs_encoder_cache_dir_is_dir": "false",
        }
    )

    errors = _check_policy_contract(contract, cfg)

    assert not any("requires a local DINO weight source" in error for error in errors)
    assert not any("Configured local DINO weight source" in error for error in errors)


def test_preflight_gaze_wam_policy_contract_requires_loss_validation_guardrails():
    summary = preflight_gaze_wam(
        config_name="train_gaze_wam_debug_workspace",
        device="cpu",
        validate_zarr=False,
        run_loss_smoke=False,
    )
    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train_gaze_wam_debug_workspace")
    contract = copy.deepcopy(summary["policy_contract"])
    contract["loss_routing_contract"]["validation"][
        "inactive_optional_metadata_rows_must_be_zero_placeholders"
    ] = False

    errors = _check_policy_contract(contract, cfg)

    assert any("Loss routing validation guardrails" in error for error in errors)


def test_preflight_gaze_wam_policy_contract_requires_normalizer_contract():
    summary = preflight_gaze_wam(
        config_name="train_gaze_wam_debug_workspace",
        device="cpu",
        validate_zarr=False,
        run_loss_smoke=False,
    )
    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train_gaze_wam_debug_workspace")

    missing_contract = dict(summary["policy_contract"])
    missing_contract.pop("normalizer_contract")
    missing_errors = _check_policy_contract(missing_contract, cfg)
    assert any("normalizer contract summary is missing" in error for error in missing_errors)

    wrong_contract = dict(summary["policy_contract"])
    wrong_contract["normalizer_contract"] = dict(wrong_contract["normalizer_contract"])
    wrong_contract["normalizer_contract"]["source"] = "open_dataset_dummy_actions"
    wrong_contract["normalizer_contract"]["excludes_open_source_dummy_actions"] = False
    wrong_errors = _check_policy_contract(wrong_contract, cfg)
    assert any("robot-relative-only" in error for error in wrong_errors)


def test_preflight_gaze_wam_policy_contract_reports_missing_local_dino_source():
    summary = preflight_gaze_wam(
        config_name="train_gaze_wam_debug_workspace",
        device="cpu",
        validate_zarr=False,
        run_loss_smoke=False,
    )
    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train_gaze_wam_debug_workspace")
    contract = dict(summary["policy_contract"])
    contract.update(
        {
            "obs_encoder_pretrained": True,
            "obs_encoder_checkpoint_path": "W:/gaze-wam/missing-dinov3.ckpt",
            "obs_encoder_checkpoint_path_exists": False,
            "obs_encoder_cache_dir": "",
            "obs_encoder_cache_dir_exists": False,
            "obs_encoder_local_weight_source_configured": True,
            "obs_encoder_local_weight_source_exists": False,
            "obs_encoder_local_weight_source_valid": False,
        }
    )

    errors = _check_policy_contract(contract, cfg)

    assert any("Configured local DINO weight source does not exist" in error for error in errors)


def test_preflight_gaze_wam_policy_contract_checks_local_dino_path_types():
    summary = preflight_gaze_wam(
        config_name="train_gaze_wam_debug_workspace",
        device="cpu",
        validate_zarr=False,
        run_loss_smoke=False,
    )
    config_dir = str(Path("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train_gaze_wam_debug_workspace")
    contract = dict(summary["policy_contract"])
    contract.update(
        {
            "obs_encoder_pretrained": True,
            "obs_encoder_checkpoint_path": "W:/gaze-wam/dino_cache_dir",
            "obs_encoder_checkpoint_path_exists": True,
            "obs_encoder_checkpoint_path_is_file": False,
            "obs_encoder_cache_dir": "W:/gaze-wam/dinov3.ckpt",
            "obs_encoder_cache_dir_exists": True,
            "obs_encoder_cache_dir_is_dir": False,
            "obs_encoder_local_weight_source_configured": True,
            "obs_encoder_local_weight_source_exists": True,
            "obs_encoder_local_weight_source_valid": False,
        }
    )

    errors = _check_policy_contract(contract, cfg)

    assert any("not structurally valid" in error for error in errors)
    assert any("checkpoint_path must point to a file" in error for error in errors)
    assert any("cache_dir must point to a directory" in error for error in errors)


def test_preflight_gaze_wam_allows_source_resize_modes_but_blocks_size_mismatch():
    summary = preflight_gaze_wam(
        config_name="train_gaze_wam_debug_workspace",
        overrides=[
            "task.open_dataset.image_resize_mode=letterbox",
            "task.robot_dataset.image_size=[224,224]",
        ],
        device="cpu",
        validate_zarr=False,
        run_loss_smoke=False,
    )

    assert summary["ok"] is False
    assert summary["image_geometry"]["task_image_resize_mode"] == "stretch"
    assert summary["image_geometry"]["robot_image_resize_mode"] == "stretch"
    assert summary["image_geometry"]["open_image_resize_mode"] == "letterbox"
    assert summary["image_geometry"]["task_image_size"] == [256, 256]
    assert summary["image_geometry"]["robot_image_size"] == [224, 224]
    assert summary["image_geometry"]["open_image_size"] == [256, 256]
    assert summary["image_geometry"]["all_stretch"] is False
    assert summary["image_geometry"]["consistent"] is False
    assert summary["image_geometry"]["supported"] is True
    assert summary["image_geometry"]["image_size_consistent"] is False
    assert not any("image_resize_mode" in error for error in summary["errors"])
    assert any("image_size" in error and "must match" in error for error in summary["errors"])


def test_preflight_gaze_wam_geometry_summary_uses_strict_integer_parser():
    cfg = OmegaConf.create(
        {
            "task": {
                "image_shape": ["3", "256", "256"],
                "image_resize_mode": "stretch",
                "robot_dataset": {
                    "image_size": ["256", "256"],
                    "image_resize_mode": "stretch",
                },
                "open_dataset": {
                    "image_size": ["256", "256"],
                    "image_resize_mode": "stretch",
                },
            },
        }
    )

    summary = preflight_gaze_wam_module._image_geometry_summary(cfg)

    assert summary["image_shape"] == [3, 256, 256]
    assert summary["task_image_size"] == [256, 256]
    assert summary["robot_image_size"] == [256, 256]
    assert summary["open_image_size"] == [256, 256]
    assert summary["image_size_consistent"] is True

    mixed_mode_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    mixed_mode_cfg.task.robot_dataset.image_resize_mode = "letterbox"
    mixed_mode_summary = preflight_gaze_wam_module._image_geometry_summary(mixed_mode_cfg)
    assert mixed_mode_summary["supported"] is True
    assert mixed_mode_summary["consistent"] is False
    assert preflight_gaze_wam_module._check_image_geometry_contract(mixed_mode_summary) == []

    invalid_overrides = [
        ("task.image_shape", [3, 256.5, 256], "task.image_shape[1]"),
        ("task.robot_dataset.image_size", [True, 256], "task.robot_dataset.image_size[0]"),
        ("task.open_dataset.image_size", [256, float("inf")], "task.open_dataset.image_size[1]"),
    ]
    for key, value, expected in invalid_overrides:
        bad_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.update(bad_cfg, key, value, merge=False)
        try:
            preflight_gaze_wam_module._image_geometry_summary(bad_cfg)
        except ValueError as exc:
            assert expected in str(exc)
            assert "positive integer" in str(exc)
        else:
            raise AssertionError(f"Expected invalid geometry field {key} to fail.")


def test_preflight_gaze_wam_blocks_dataset_sampling_mismatch():
    summary = preflight_gaze_wam(
        config_name="train_gaze_wam_debug_workspace",
        overrides=[
            "task.robot_dataset.action_horizon=8",
            "task.open_dataset.n_latency_steps=1",
        ],
        device="cpu",
        validate_zarr=False,
        run_loss_smoke=False,
    )

    assert summary["ok"] is False
    assert summary["sampling_contract"]["task"] == {
        "n_obs_steps": 2,
        "action_horizon": 48,
        "n_latency_steps": 0,
    }
    assert summary["sampling_contract"]["robot_dataset"]["action_horizon"] == 8
    assert summary["sampling_contract"]["open_dataset"]["n_latency_steps"] == 1
    assert summary["sampling_contract"]["robot_matches_task"] is False
    assert summary["sampling_contract"]["open_matches_task"] is False
    assert any("Robot dataset temporal sampling" in error for error in summary["errors"])
    assert any("Open dataset temporal sampling" in error for error in summary["errors"])


def test_preflight_gaze_wam_contract_runs_when_robot_dataset_fails():
    summary = preflight_gaze_wam(
        config_name="train_gaze_wam_debug_workspace",
        overrides=[
            "task.robot_dataset_path=data/debug_gaze_wam/missing_robot.zarr",
        ],
        device="cpu",
        validate_zarr=False,
        run_loss_smoke=True,
    )

    assert summary["ok"] is False
    assert any("Robot dataset check failed" in error for error in summary["errors"])
    assert "robot_dataset" not in summary
    assert "loss_smoke" not in summary
    assert summary["policy_contract"]["obs_output_shape"] == [1, 512, 768]
    assert summary["policy_contract"]["model_action_dim"] == 10
    assert summary["policy_contract"]["model_heatmap_num_tokens"] == 256
    assert summary["policy_contract"]["robot_ratio"] == 0.75
    assert summary["policy_contract"]["open_ratio"] == 0.25


def test_preflight_gaze_wam_reports_token_grid_contract_mismatch():
    summary = preflight_gaze_wam(
        config_name="train_gaze_wam_debug_workspace",
        overrides=[
            "task.heatmap_num_tokens=128",
            "policy.heatmap_num_tokens=128",
            "policy.heatmap_token_grid=[8,16]",
        ],
        device="cpu",
        validate_zarr=False,
        run_loss_smoke=False,
    )

    assert summary["ok"] is False
    assert "Visual token count does not match n_obs_steps * heatmap_num_tokens." in summary["errors"]
    assert "Visual tokens per frame do not match heatmap tokens per frame." in summary["errors"]
    assert summary["policy_contract"]["visual_tokens_per_batch_item"] == 512
    assert summary["policy_contract"]["visual_tokens_per_frame"] == 256.0
    assert summary["policy_contract"]["expected_visual_tokens_per_frame"] == 128
    assert summary["policy_contract"]["expected_visual_tokens_total"] == 256
    assert summary["policy_contract"]["heatmap_codec_token_grid"] == [8, 16]


def test_preflight_gaze_wam_reports_patch_size_contract_mismatch():
    summary = preflight_gaze_wam(
        config_name="train_gaze_wam_debug_workspace",
        overrides=[
            "task.heatmap_token_grid=[32,8]",
            "policy.heatmap_token_grid=[32,8]",
        ],
        device="cpu",
        validate_zarr=False,
        run_loss_smoke=False,
    )

    assert summary["ok"] is False
    assert "Obs encoder patch size does not match heatmap-grid patch size." in summary["errors"]
    assert summary["policy_contract"]["obs_encoder_patch_size"] == [16, 16]
    assert summary["policy_contract"]["inferred_patch_size_from_heatmap_grid"] == [8, 32]


def test_preflight_gaze_wam_threads_timestamp_validation_options():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_linear_action_zarr(Path(tmpdir) / "robot.zarr", length=8)
        _add_timestamp_arrays(robot_path)

        summary = preflight_gaze_wam(
            config_name="train_gaze_wam_robot_only_debug_workspace",
            overrides=[
                f"task.robot_dataset_path={robot_path}",
            ],
            device="cpu",
            validate_zarr=True,
            run_loss_smoke=False,
            require_timestamps=True,
            timestamp_max_step=0.01,
        )

    assert summary["ok"] is False
    assert summary["robot_zarr_validation"]["timestamps"]["checked"] is True
    assert summary["robot_zarr_validation"]["timestamps"]["intervals"]["timestamp"]["max_step"] > 0.01
    assert "loss_smoke" not in summary
    assert any("Robot zarr validation failed" in error for error in summary["errors"])


def test_preflight_gaze_wam_can_fail_on_zarr_warnings():
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_path = _write_linear_action_zarr(Path(tmpdir) / "short_robot.zarr", length=2)

        summary = preflight_gaze_wam(
            config_name="train_gaze_wam_robot_only_debug_workspace",
            overrides=[
                f"task.robot_dataset_path={robot_path}",
            ],
            device="cpu",
            validate_zarr=True,
            run_loss_smoke=False,
            fail_on_zarr_warning=True,
        )

    assert summary["ok"] is False
    assert any("Robot zarr warning" in warning for warning in summary["warnings"])
    assert any("Robot zarr validation produced" in error for error in summary["errors"])


def test_launch_gaze_wam_training_preflights_and_writes_command_report():
    command = build_gaze_wam_train_command(
        config_name="train_gaze_wam_debug_workspace",
        task="gaze_wam",
        overrides=["policy.obs_encoder.pretrained=false"],
        use_accelerate=True,
        accelerate_config="accelerate/8gpu-amp.yaml",
    )
    assert command[:5] == [
        "accelerate",
        "launch",
        "--config_file",
        "accelerate/8gpu-amp.yaml",
        "train.py",
    ]
    assert "--config-name" in command
    assert "policy.obs_encoder.pretrained=false" in command

    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = Path(tmpdir) / "launch_report.json"
        summary = launch_gaze_wam_training(
            config_name="train_gaze_wam_debug_workspace",
            overrides=[
                "task.robot_dataset_path=data/debug_gaze_wam/robot.zarr",
                "task.open_dataset_path=data/debug_gaze_wam/open.zarr",
                "policy.obs_encoder.pretrained=false",
            ],
            use_accelerate=False,
            preflight_device="cpu",
            output_json=str(output_json),
            run=False,
        )

        assert summary["ok"] is True
        assert summary["run"] is False
        assert summary["returncode"] is None
        assert summary["preflight"]["ok"] is True
        assert summary["preflight_routing_validation_guardrails_ok"] is True
        assert summary["command"][:3] == ["py", "train.py", "--config-name"]
        assert "task.robot_dataset_path=data/debug_gaze_wam/robot.zarr" in summary["command"]
        assert summary["acceleration"]["use_accelerate"] is False
        assert summary["acceleration"]["num_processes"] == 1
        assert summary["acceleration"]["mixed_precision"] == "no"
        assert summary["acceleration"]["robot_batch_size_per_process"] == 3
        assert summary["acceleration"]["open_batch_size_per_process"] == 1
        assert summary["acceleration"]["effective_robot_batch_size_per_optimizer_step"] == 3
        assert summary["acceleration"]["effective_open_batch_size_per_optimizer_step"] == 1
        assert summary["acceleration"]["effective_train_batch_size_per_optimizer_step"] == 4
        assert summary["acceleration"]["effective_train_batch_size"] == 4
        assert output_json.exists()
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert payload["command_str"].startswith("py train.py")
        assert payload["preflight"]["loss_smoke"]["mixed_batch_size"] == 4
        assert "zarr_presence_masks" in payload["preflight"]
        assert payload["preflight_routing_validation_guardrails_ok"] is True
        assert payload["acceleration"]["effective_train_batch_size_per_optimizer_step"] == 4
        assert payload["acceleration"]["effective_train_batch_size"] == 4


def test_launch_gaze_wam_training_blocks_failed_preflight_routing_guardrails():
    flags = gaze_wam_required_loss_routing_validation_flags()
    validation = {key: True for key in flags}
    validation["inactive_heatmap_rows_must_be_zero_placeholders"] = False

    def fake_preflight_gaze_wam(**kwargs):
        return {
            "ok": True,
            "policy_contract": {
                "loss_routing_contract": {
                    "validation": validation,
                },
            },
        }

    original_preflight = preflight_gaze_wam_module.preflight_gaze_wam
    try:
        preflight_gaze_wam_module.preflight_gaze_wam = fake_preflight_gaze_wam
        summary = launch_gaze_wam_training(
            config_name="train_gaze_wam_debug_workspace",
            use_accelerate=False,
            run=True,
        )
    finally:
        preflight_gaze_wam_module.preflight_gaze_wam = original_preflight

    assert summary["ok"] is False
    assert summary["preflight"]["ok"] is True
    assert summary["preflight_routing_validation_guardrails_ok"] is False
    assert summary["returncode"] is None
    assert summary["run_skipped"] == "launch_checks_failed"
    assert "Preflight loss-routing validation guardrails failed." in summary["errors"]


def test_launch_gaze_wam_training_threads_timestamp_preflight_options():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        robot_path = _write_linear_action_zarr(root / "robot.zarr", length=8)
        _add_timestamp_arrays(robot_path)
        output_json = root / "launch_report.json"

        summary = launch_gaze_wam_training(
            config_name="train_gaze_wam_robot_only_debug_workspace",
            overrides=[
                f"task.robot_dataset_path={robot_path}",
            ],
            use_accelerate=False,
            preflight_device="cpu",
            skip_loss_smoke=True,
            preflight_require_timestamps=True,
            preflight_timestamp_max_step=0.01,
            preflight_fail_on_zarr_warning=True,
            output_json=str(output_json),
            run=True,
        )

        assert summary["ok"] is False
        assert summary["returncode"] is None
        assert summary["run_skipped"] == "launch_checks_failed"
        assert summary["preflight_options"]["require_timestamps"] is True
        assert summary["preflight_options"]["timestamp_max_step"] == 0.01
        assert summary["preflight_options"]["fail_on_zarr_warning"] is True
        assert summary["preflight"]["robot_zarr_validation"]["timestamps"]["checked"] is True
        assert (
            summary["preflight"]["robot_zarr_validation"]["timestamps"]["intervals"]["timestamp"]["max_step"]
            > 0.01
        )
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert payload["ok"] is False
        assert payload["run_skipped"] == "launch_checks_failed"
        assert any("Preflight failed" in error for error in payload["errors"])


def test_launch_gaze_wam_training_reports_accelerate_effective_batch():
    summary = launch_gaze_wam_training(
        config_name="train_gaze_wam_workspace",
        overrides=[
            "data_mixing.batch_size_source=dataloader",
            "robot_dataloader.batch_size=6",
            "open_dataloader.batch_size=2",
            "training.gradient_accumulate_every=2",
        ],
        use_accelerate=True,
        accelerate_config="accelerate/8gpu-amp.yaml",
        skip_preflight=True,
        run=False,
    )

    assert summary["ok"] is True
    assert summary["preflight"] == {"skipped": True}
    assert summary["acceleration"]["accelerate_config"]["exists"] is True
    assert summary["acceleration"]["accelerate_config"]["distributed_type"] == "MULTI_GPU"
    assert summary["acceleration"]["mixed_precision"] == "bf16"
    assert summary["acceleration"]["num_processes"] == 8
    assert summary["acceleration"]["train_batch_size_per_process"] == 8
    assert summary["acceleration"]["gradient_accumulate_every"] == 2
    assert summary["acceleration"]["effective_robot_batch_size_per_optimizer_step"] == 96
    assert summary["acceleration"]["effective_open_batch_size_per_optimizer_step"] == 32
    assert summary["acceleration"]["effective_train_batch_size_per_optimizer_step"] == 128
    assert summary["acceleration"]["effective_train_batch_size"] == 128
    assert summary["warnings"] == []


def test_launch_gaze_wam_training_ratio_quota_ignores_legacy_batch_overrides():
    summary = launch_gaze_wam_training(
        config_name="train_gaze_wam_workspace",
        overrides=[
            "robot_dataloader.batch_size=6",
            "open_dataloader.batch_size=2",
            "training.gradient_accumulate_every=2",
        ],
        use_accelerate=True,
        accelerate_config="accelerate/8gpu-amp.yaml",
        skip_preflight=True,
        run=False,
    )

    assert summary["ok"] is True
    acceleration = summary["acceleration"]
    assert acceleration["batch_size_source"] == "ratio"
    assert acceleration["requested_total_batch_size_per_process"] == 64
    assert acceleration["requested_robot_ratio"] == pytest.approx(0.75)
    assert acceleration["requested_open_ratio"] == pytest.approx(0.25)
    assert acceleration["robot_batch_size_per_process"] == 48
    assert acceleration["open_batch_size_per_process"] == 16
    assert acceleration["train_batch_size_per_process"] == 64
    assert acceleration["effective_robot_batch_size_per_optimizer_step"] == 768
    assert acceleration["effective_open_batch_size_per_optimizer_step"] == 256
    assert acceleration["effective_train_batch_size_per_optimizer_step"] == 1024
    assert acceleration["warnings"] == []


def test_launch_gaze_wam_training_blocks_invalid_training_config_without_preflight():
    summary = launch_gaze_wam_training(
        config_name="train_gaze_wam_debug_workspace",
        overrides=[
            "training.gradient_accumulate_every=0",
            "robot_dataloader.batch_size=0",
            "open_dataloader.batch_size=0",
        ],
        use_accelerate=False,
        skip_preflight=True,
        run=True,
    )

    assert summary["ok"] is False
    assert summary["preflight"] == {"skipped": True}
    assert summary["run_skipped"] == "launch_checks_failed"
    assert summary["returncode"] is None
    assert summary["acceleration"]["training_config"]["valid"] is False
    assert summary["acceleration"]["gradient_accumulate_every"] == 0
    assert any(
        "training.gradient_accumulate_every" in error
        for error in summary["acceleration"]["errors"]
    )
    assert any(
        "Training acceleration config invalid" in error
        for error in summary["errors"]
    )


def test_launch_gaze_wam_training_writes_report_before_run(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = Path(tmpdir) / "launch_report.json"
        observed = {}

        def fake_run(command, cwd, env, check):
            observed["command"] = list(command)
            assert output_json.exists()
            pre_run_payload = json.loads(output_json.read_text(encoding="utf-8"))
            assert pre_run_payload["pre_run_report_written"] is True
            assert pre_run_payload["ok"] is True
            assert pre_run_payload["returncode"] is None
            assert pre_run_payload["run"] is True

            class Completed:
                returncode = 0

            return Completed()

        monkeypatch.setattr(
            launch_gaze_wam_training_module.subprocess,
            "run",
            fake_run,
        )

        summary = launch_gaze_wam_training(
            config_name="train_gaze_wam_debug_workspace",
            overrides=["policy.obs_encoder.pretrained=false"],
            use_accelerate=False,
            skip_preflight=True,
            output_json=str(output_json),
            run=True,
        )

        assert observed["command"][:3] == ["py", "train.py", "--config-name"]
        assert summary["ok"] is True
        assert summary["returncode"] == 0
        assert summary["pre_run_report_written"] is True
        final_payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert final_payload["returncode"] == 0
        assert final_payload["pre_run_report_written"] is True


def test_launch_gaze_wam_training_real_data_readiness_blocks_unsafe_debug_launch():
    summary = launch_gaze_wam_training(
        config_name="train_gaze_wam_debug_workspace",
        overrides=[
            "policy.obs_encoder.pretrained=false",
        ],
        use_accelerate=False,
        skip_preflight=True,
        skip_zarr_validation=True,
        skip_loss_smoke=True,
        real_data=True,
        run=True,
    )

    assert summary["ok"] is False
    assert summary["run_skipped"] == "launch_checks_failed"
    assert summary["preflight"] == {"skipped": True}
    readiness = summary["real_data_readiness"]
    assert readiness["enabled"] is True
    failed_names = {
        check["name"]
        for check in readiness["checks"]
        if not check["ok"]
    }
    assert "non_debug_config_name" in failed_names
    assert "training_debug_false" in failed_names
    assert "preflight_enabled" in failed_names
    assert "zarr_validation_enabled" in failed_names
    assert "loss_smoke_enabled" in failed_names
    assert "timestamps_required" in failed_names
    assert "timestamp_threshold_configured" in failed_names
    assert "zarr_warnings_block" in failed_names
    assert "launch_report_path" in failed_names
    assert "obs_encoder_pretrained" in failed_names
    assert "accelerate_enabled" in failed_names


def test_real_data_readiness_accepts_full_gated_launch_settings():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        robot_path = root / "real_robot.zarr"
        open_path = root / "real_open.zarr"
        dino_cache = root / "dino_cache"
        _write_real_data_readiness_zarr_metadata(
            robot_path,
            "robot",
            image_resize_mode="letterbox",
        )
        _write_real_data_readiness_zarr_metadata(open_path, "open")
        dino_cache.mkdir()
        (dino_cache / "fake_dinov3_cache.bin").write_bytes(b"fake-local-dinov3-cache")
        cfg = load_cfg(
            "train_gaze_wam_workspace",
            overrides=[
                "task=gaze_wam",
                f"task.robot_dataset_path={robot_path}",
                f"task.open_dataset_path={open_path}",
                f"policy.obs_encoder.cache_dir={dino_cache}",
            ],
        )
        summary = launch_gaze_wam_training(
            config_name="train_gaze_wam_workspace",
            use_accelerate=True,
            accelerate_config="accelerate/8gpu-amp.yaml",
            skip_preflight=True,
            run=False,
        )
        readiness = check_real_data_readiness(
            config_name="train_gaze_wam_workspace",
            cfg=cfg,
            acceleration=summary["acceleration"],
            output_json="data/outputs/gaze_wam_launch_report.json",
            skip_preflight=False,
            skip_zarr_validation=False,
            skip_loss_smoke=False,
            preflight_require_timestamps=True,
            preflight_timestamp_max_delta=None,
            preflight_timestamp_max_step=0.08,
            preflight_fail_on_zarr_warning=True,
            use_accelerate=True,
        )

        assert readiness["ok"] is True
        assert readiness["errors"] == []
        assert all(check["ok"] for check in readiness["checks"])
        assert readiness["dino_source_verifier"]["ok"] is True
        assert readiness["dino_source_verifier"]["dino_source"]["cache_dir"] == str(dino_cache)
        assert readiness["dino_source_verifier"]["geometry"]["expected_tokens_per_frame"] == 256
        assert readiness["dino_source_verifier"]["normalization"]["mean"] == [0.485, 0.456, 0.406]
        assert readiness["training_config"]["valid"] is True
        assert readiness["training_config"]["gradient_accumulate_every"] == 1
        assert readiness["training_config"]["train_batch_size_per_process"] == 64
        assert readiness["data_onboarding_review"]["enabled"] is False


def test_real_data_readiness_blocks_invalid_training_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        robot_path = root / "real_robot.zarr"
        open_path = root / "real_open.zarr"
        dino_cache = root / "dino_cache"
        _write_real_data_readiness_zarr_metadata(robot_path, "robot")
        _write_real_data_readiness_zarr_metadata(open_path, "open")
        dino_cache.mkdir()
        (dino_cache / "fake_dinov3_cache.bin").write_bytes(b"fake-local-dinov3-cache")
        overrides = [
            "task=gaze_wam",
            f"task.robot_dataset_path={robot_path}",
            f"task.open_dataset_path={open_path}",
            f"policy.obs_encoder.cache_dir={dino_cache}",
            "training.gradient_accumulate_every=0",
        ]
        cfg = load_cfg("train_gaze_wam_workspace", overrides=overrides)
        summary = launch_gaze_wam_training(
            config_name="train_gaze_wam_workspace",
            overrides=overrides[1:],
            use_accelerate=True,
            accelerate_config="accelerate/8gpu-amp.yaml",
            skip_preflight=True,
            run=False,
        )
        readiness = check_real_data_readiness(
            config_name="train_gaze_wam_workspace",
            cfg=cfg,
            acceleration=summary["acceleration"],
            output_json=str(root / "launch_report.json"),
            skip_preflight=False,
            skip_zarr_validation=False,
            skip_loss_smoke=False,
            preflight_require_timestamps=True,
            preflight_timestamp_max_delta=None,
            preflight_timestamp_max_step=0.08,
            preflight_fail_on_zarr_warning=True,
            use_accelerate=True,
        )

    failed_names = {
        check["name"]
        for check in readiness["checks"]
        if not check["ok"]
    }
    assert readiness["ok"] is False
    assert "training_config_valid" in failed_names
    assert readiness["training_config"]["valid"] is False
    assert readiness["training_config"]["gradient_accumulate_every"] == 0
    assert any("gradient_accumulate_every" in error for error in readiness["training_config"]["errors"])
    assert any("training-loop config is invalid" in error for error in readiness["errors"])


def test_real_data_readiness_accepts_matching_data_onboarding_review():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        robot_path = root / "real_robot.zarr"
        open_path = root / "real_open.zarr"
        dino_cache = root / "dino_cache"
        review_path = root / "onboarding_review.json"
        _write_real_data_readiness_zarr_metadata(
            robot_path,
            "robot",
            image_resize_mode="letterbox",
        )
        _write_real_data_readiness_zarr_metadata(open_path, "open")
        dino_cache.mkdir()
        (dino_cache / "fake_dinov3_cache.bin").write_bytes(b"fake-local-dinov3-cache")
        review_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "dry_run": True,
                    "policy_training_scope": True,
                    "deployment_runner_scope": "deferred",
                    "selected": {"robot": True, "open": True},
                    "contract": {
                        "image_size": [256, 256],
                        "image_resize_mode": "stretch",
                        "robot_image_resize_mode": "letterbox",
                        "open_image_resize_mode": "stretch",
                        "n_obs_steps": 2,
                        "action_horizon": 48,
                        "n_latency_steps": 0,
                        "heatmap_token_grid": [16, 16],
                        "require_timestamps": True,
                        "timestamp_max_delta": None,
                        "timestamp_max_step": 0.08,
                    },
                    "robot": {"output_path": str(robot_path), "dry_run": True, "ok": True},
                    "open": {"output_zarr": str(open_path), "dry_run": True, "ok": True},
                }
            ),
            encoding="utf-8",
        )
        cfg = load_cfg(
            "train_gaze_wam_workspace",
            overrides=[
                "task=gaze_wam",
                f"task.robot_dataset_path={robot_path}",
                f"task.open_dataset_path={open_path}",
                f"policy.obs_encoder.cache_dir={dino_cache}",
            ],
        )
        summary = launch_gaze_wam_training(
            config_name="train_gaze_wam_workspace",
            use_accelerate=True,
            accelerate_config="accelerate/8gpu-amp.yaml",
            skip_preflight=True,
            run=False,
        )
        readiness = check_real_data_readiness(
            config_name="train_gaze_wam_workspace",
            cfg=cfg,
            acceleration=summary["acceleration"],
            output_json=str(root / "launch_report.json"),
            skip_preflight=False,
            skip_zarr_validation=False,
            skip_loss_smoke=False,
            preflight_require_timestamps=True,
            preflight_timestamp_max_delta=None,
            preflight_timestamp_max_step=0.08,
            preflight_fail_on_zarr_warning=True,
            use_accelerate=True,
            data_onboarding_review_json=str(review_path),
            require_data_onboarding_review=True,
        )

        assert readiness["ok"] is True
        review = readiness["data_onboarding_review"]
        assert review["enabled"] is True
        assert review["required"] is True
        assert review["exists"] is True
        assert review["ok"] is True
        assert review["report"]["robot_output_path"] == str(robot_path)
        assert review["report"]["open_output_zarr"] == str(open_path)
        failed_names = {
            check["name"]
            for check in review["checks"]
            if not check["ok"]
        }
        assert failed_names == set()


def test_data_onboarding_review_uses_strict_bool_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        robot_path = root / "real_robot.zarr"
        open_path = root / "real_open.zarr"
        review_path = root / "onboarding_review.json"
        review_path.write_text(
            json.dumps(
                {
                    "ok": "true",
                    "dry_run": "true",
                    "policy_training_scope": "false",
                    "deployment_runner_scope": "deferred",
                    "selected": {"robot": "false", "open": "0"},
                    "contract": {
                        "image_size": [256, 256],
                        "image_resize_mode": "stretch",
                        "n_obs_steps": 2,
                        "action_horizon": 16,
                        "n_latency_steps": 0,
                        "heatmap_token_grid": [16, 16],
                        "require_timestamps": "false",
                    },
                    "robot": {"output_path": str(robot_path), "dry_run": True, "ok": True},
                    "open": {"output_zarr": str(open_path), "dry_run": True, "ok": True},
                }
            ),
            encoding="utf-8",
        )

        summary = launch_gaze_wam_training_module._read_data_onboarding_review(
            path=str(review_path),
            required=True,
            robot_dataset_path=str(robot_path),
            open_dataset_path=str(open_path),
            image_size=[256, 256],
            image_resize_mode="stretch",
            task_sampling={"n_obs_steps": 2, "action_horizon": 16, "n_latency_steps": 0},
            heatmap_token_grid=[16, 16],
            preflight_require_timestamps=False,
            preflight_timestamp_max_delta=None,
            preflight_timestamp_max_step=None,
        )

    failed_names = {
        check["name"]
        for check in summary["checks"]
        if not check["ok"]
    }
    assert summary["ok"] is False
    assert summary["report"]["ok"] is True
    assert summary["report"]["dry_run"] is True
    assert summary["report"]["policy_training_scope"] is False
    assert "data_onboarding_review_policy_scope" in failed_names
    assert "data_onboarding_review_selected_robot" in failed_names
    assert "data_onboarding_review_selected_open" in failed_names
    assert "data_onboarding_review_timestamp_requirement_matches_launch" not in failed_names


def test_real_data_readiness_blocks_zarr_metadata_mismatch():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        robot_path = root / "real_robot.zarr"
        open_path = root / "real_open.zarr"
        dino_cache = root / "dino_cache"
        _write_real_data_readiness_zarr_metadata(
            robot_path,
            "open",
            image_size=(224, 224),
            image_resize_mode="stretch",
        )
        _write_real_data_readiness_zarr_metadata(
            open_path,
            "robot",
            image_size=(128, 256),
            image_resize_mode="letterbox",
        )
        dino_cache.mkdir()
        (dino_cache / "fake_dinov3_cache.bin").write_bytes(b"fake-local-dinov3-cache")
        cfg = load_cfg(
            "train_gaze_wam_workspace",
            overrides=[
                "task=gaze_wam",
                f"task.robot_dataset_path={robot_path}",
                f"task.open_dataset_path={open_path}",
                f"policy.obs_encoder.cache_dir={dino_cache}",
            ],
        )
        summary = launch_gaze_wam_training(
            config_name="train_gaze_wam_workspace",
            use_accelerate=True,
            accelerate_config="accelerate/8gpu-amp.yaml",
            skip_preflight=True,
            run=False,
        )
        readiness = check_real_data_readiness(
            config_name="train_gaze_wam_workspace",
            cfg=cfg,
            acceleration=summary["acceleration"],
            output_json=str(root / "launch_report.json"),
            skip_preflight=False,
            skip_zarr_validation=False,
            skip_loss_smoke=False,
            preflight_require_timestamps=True,
            preflight_timestamp_max_delta=None,
            preflight_timestamp_max_step=0.08,
            preflight_fail_on_zarr_warning=True,
            use_accelerate=True,
        )

    failed_names = {
        check["name"]
        for check in readiness["checks"]
        if not check["ok"]
    }
    assert readiness["ok"] is False
    assert "robot_zarr_metadata_dataset_type" in failed_names
    assert "robot_zarr_metadata_image_resize_mode" in failed_names
    assert "robot_zarr_metadata_image_size" in failed_names
    assert "open_zarr_metadata_dataset_type" in failed_names
    assert "open_zarr_metadata_image_resize_mode" in failed_names
    assert "open_zarr_metadata_image_size" in failed_names
    assert readiness["zarr_metadata"]["robot"]["metadata_attrs"]["dataset_type"] == "open"
    assert readiness["zarr_metadata"]["open"]["metadata_attrs"]["dataset_type"] == "robot"
    assert any("meta.attrs.dataset_type" in error for error in readiness["errors"])
    assert any("meta.attrs.image_size" in error for error in readiness["errors"])


def test_real_data_readiness_can_require_data_onboarding_review():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        robot_path = root / "real_robot.zarr"
        open_path = root / "real_open.zarr"
        dino_cache = root / "dino_cache"
        _write_real_data_readiness_zarr_metadata(robot_path, "robot")
        _write_real_data_readiness_zarr_metadata(open_path, "open")
        dino_cache.mkdir()
        (dino_cache / "fake_dinov3_cache.bin").write_bytes(b"fake-local-dinov3-cache")
        cfg = load_cfg(
            "train_gaze_wam_workspace",
            overrides=[
                "task=gaze_wam",
                f"task.robot_dataset_path={robot_path}",
                f"task.open_dataset_path={open_path}",
                f"policy.obs_encoder.cache_dir={dino_cache}",
            ],
        )
        summary = launch_gaze_wam_training(
            config_name="train_gaze_wam_workspace",
            use_accelerate=True,
            accelerate_config="accelerate/8gpu-amp.yaml",
            skip_preflight=True,
            run=False,
        )
        readiness = check_real_data_readiness(
            config_name="train_gaze_wam_workspace",
            cfg=cfg,
            acceleration=summary["acceleration"],
            output_json=str(root / "launch_report.json"),
            skip_preflight=False,
            skip_zarr_validation=False,
            skip_loss_smoke=False,
            preflight_require_timestamps=True,
            preflight_timestamp_max_delta=None,
            preflight_timestamp_max_step=0.08,
            preflight_fail_on_zarr_warning=True,
            use_accelerate=True,
            require_data_onboarding_review=True,
        )

    failed_names = {
        check["name"]
        for check in readiness["data_onboarding_review"]["checks"]
        if not check["ok"]
    }
    assert readiness["ok"] is False
    assert "data_onboarding_review_path_configured" in failed_names
    assert any("data-onboarding-review-json" in error for error in readiness["errors"])


def test_real_data_readiness_blocks_mismatched_data_onboarding_review():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        robot_path = root / "real_robot.zarr"
        open_path = root / "real_open.zarr"
        dino_cache = root / "dino_cache"
        review_path = root / "bad_onboarding_review.json"
        _write_real_data_readiness_zarr_metadata(robot_path, "robot")
        _write_real_data_readiness_zarr_metadata(open_path, "open")
        dino_cache.mkdir()
        (dino_cache / "fake_dinov3_cache.bin").write_bytes(b"fake-local-dinov3-cache")
        review_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "dry_run": True,
                    "policy_training_scope": True,
                    "deployment_runner_scope": "deferred",
                    "selected": {"robot": True, "open": True},
                    "contract": {
                        "image_size": [224, 224],
                        "image_resize_mode": "stretch",
                        "n_obs_steps": 2,
                        "action_horizon": 16,
                        "n_latency_steps": 0,
                        "heatmap_token_grid": [16, 16],
                        "require_timestamps": True,
                        "timestamp_max_delta": None,
                        "timestamp_max_step": 0.04,
                    },
                    "robot": {"output_path": str(root / "other_robot.zarr"), "dry_run": True, "ok": True},
                    "open": {"output_zarr": str(open_path), "dry_run": True, "ok": True},
                }
            ),
            encoding="utf-8",
        )
        cfg = load_cfg(
            "train_gaze_wam_workspace",
            overrides=[
                "task=gaze_wam",
                f"task.robot_dataset_path={robot_path}",
                f"task.open_dataset_path={open_path}",
                f"policy.obs_encoder.cache_dir={dino_cache}",
            ],
        )
        summary = launch_gaze_wam_training(
            config_name="train_gaze_wam_workspace",
            use_accelerate=True,
            accelerate_config="accelerate/8gpu-amp.yaml",
            skip_preflight=True,
            run=False,
        )
        readiness = check_real_data_readiness(
            config_name="train_gaze_wam_workspace",
            cfg=cfg,
            acceleration=summary["acceleration"],
            output_json=str(root / "launch_report.json"),
            skip_preflight=False,
            skip_zarr_validation=False,
            skip_loss_smoke=False,
            preflight_require_timestamps=True,
            preflight_timestamp_max_delta=None,
            preflight_timestamp_max_step=0.08,
            preflight_fail_on_zarr_warning=True,
            use_accelerate=True,
            data_onboarding_review_json=str(review_path),
            require_data_onboarding_review=True,
        )

    failed_names = {
        check["name"]
        for check in readiness["data_onboarding_review"]["checks"]
        if not check["ok"]
    }
    assert readiness["ok"] is False
    assert "data_onboarding_review_robot_output_matches_config" in failed_names
    assert "data_onboarding_review_image_size_matches_config" in failed_names
    assert "data_onboarding_review_timestamp_max_step_matches_launch" in failed_names
    assert any("robot output_path" in error for error in readiness["errors"])
    assert any("image_size" in error for error in readiness["errors"])


def test_real_data_readiness_blocks_empty_cache_only_dino_source():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        robot_path = root / "real_robot.zarr"
        open_path = root / "real_open.zarr"
        dino_cache = root / "empty_dino_cache"
        _write_real_data_readiness_zarr_metadata(robot_path, "robot")
        _write_real_data_readiness_zarr_metadata(open_path, "open")
        dino_cache.mkdir()
        cfg = load_cfg(
            "train_gaze_wam_workspace",
            overrides=[
                "task=gaze_wam",
                f"task.robot_dataset_path={robot_path}",
                f"task.open_dataset_path={open_path}",
                "policy.obs_encoder.checkpoint_path=''",
                f"policy.obs_encoder.cache_dir={dino_cache}",
            ],
        )
        summary = launch_gaze_wam_training(
            config_name="train_gaze_wam_workspace",
            use_accelerate=True,
            accelerate_config="accelerate/8gpu-amp.yaml",
            skip_preflight=True,
            run=False,
        )
        readiness = check_real_data_readiness(
            config_name="train_gaze_wam_workspace",
            cfg=cfg,
            acceleration=summary["acceleration"],
            output_json="data/outputs/gaze_wam_launch_report.json",
            skip_preflight=False,
            skip_zarr_validation=False,
            skip_loss_smoke=False,
            preflight_require_timestamps=True,
            preflight_timestamp_max_delta=None,
            preflight_timestamp_max_step=0.08,
            preflight_fail_on_zarr_warning=True,
            use_accelerate=True,
        )

    failed_names = {
        check["name"]
        for check in readiness["checks"]
        if not check["ok"]
    }
    assert readiness["ok"] is False
    assert "obs_encoder_cache_dir_contains_files_when_cache_only" in failed_names
    assert readiness["dino_source_verifier"]["ok"] is True
    assert readiness["dino_source_verifier"]["dino_source"]["cache_dir_contains_files"] is False
    assert any("cache directory contains no files" in error for error in readiness["errors"])


def test_real_data_readiness_contracts_allow_planned_ratio_variants():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        robot_path = root / "real_robot.zarr"
        open_path = root / "real_open.zarr"
        dino_cache = root / "dino_cache"
        _write_real_data_readiness_zarr_metadata(robot_path, "robot")
        _write_real_data_readiness_zarr_metadata(open_path, "open")
        dino_cache.mkdir()
        (dino_cache / "fake_dinov3_cache.bin").write_bytes(b"fake-local-dinov3-cache")
        cfg = load_cfg(
            "train_gaze_wam_workspace",
            overrides=[
                "task=gaze_wam",
                f"task.robot_dataset_path={robot_path}",
                f"task.open_dataset_path={open_path}",
                f"policy.obs_encoder.cache_dir={dino_cache}",
                "robot_dataloader.batch_size=58",
                "open_dataloader.batch_size=6",
                "val_robot_dataloader.batch_size=58",
                "val_open_dataloader.batch_size=6",
            ],
        )
        summary = launch_gaze_wam_training(
            config_name="train_gaze_wam_workspace",
            use_accelerate=True,
            accelerate_config="accelerate/8gpu-amp.yaml",
            skip_preflight=True,
            run=False,
        )
        kwargs = dict(
            config_name="train_gaze_wam_workspace",
            cfg=cfg,
            acceleration=summary["acceleration"],
            output_json="data/outputs/gaze_wam_launch_report.json",
            skip_preflight=False,
            skip_zarr_validation=False,
            skip_loss_smoke=False,
            preflight_require_timestamps=True,
            preflight_timestamp_max_delta=None,
            preflight_timestamp_max_step=0.08,
            preflight_fail_on_zarr_warning=True,
            use_accelerate=True,
        )
        main_readiness = check_real_data_readiness(contract="main", **kwargs)
        ablation_readiness = check_real_data_readiness(contract="ablation", **kwargs)

    main_failed_names = {
        check["name"]
        for check in main_readiness["checks"]
        if not check["ok"]
    }
    ablation_failed_names = {
        check["name"]
        for check in ablation_readiness["checks"]
        if not check["ok"]
    }
    assert main_readiness["ok"] is True
    assert "source_ratio_75_25" not in main_failed_names
    assert ablation_readiness["ok"] is True
    assert ablation_readiness["contract"] == "ablation"
    assert "source_ratio_75_25" not in ablation_failed_names


def test_real_data_readiness_requires_local_dino_weight_source():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        robot_path = root / "real_robot.zarr"
        open_path = root / "real_open.zarr"
        _write_real_data_readiness_zarr_metadata(robot_path, "robot")
        _write_real_data_readiness_zarr_metadata(open_path, "open")
        cfg = load_cfg(
            "train_gaze_wam_workspace",
            overrides=[
                "task=gaze_wam",
                f"task.robot_dataset_path={robot_path}",
                f"task.open_dataset_path={open_path}",
                "policy.obs_encoder.checkpoint_path=''",
                "policy.obs_encoder.cache_dir=''",
            ],
        )
        summary = launch_gaze_wam_training(
            config_name="train_gaze_wam_workspace",
            use_accelerate=True,
            accelerate_config="accelerate/8gpu-amp.yaml",
            skip_preflight=True,
            run=False,
        )
        readiness = check_real_data_readiness(
            config_name="train_gaze_wam_workspace",
            cfg=cfg,
            acceleration=summary["acceleration"],
            output_json="data/outputs/gaze_wam_launch_report.json",
            skip_preflight=False,
            skip_zarr_validation=False,
            skip_loss_smoke=False,
            preflight_require_timestamps=True,
            preflight_timestamp_max_delta=None,
            preflight_timestamp_max_step=0.08,
            preflight_fail_on_zarr_warning=True,
            use_accelerate=True,
        )

    failed_names = {
        check["name"]
        for check in readiness["checks"]
        if not check["ok"]
    }
    assert readiness["ok"] is False
    assert "obs_encoder_local_weight_source_configured" in failed_names
    assert "dino_source_verifier_ok" in failed_names
    assert readiness["dino_source_verifier"]["ok"] is False
    assert any("local DINO weight source" in error for error in readiness["errors"])


def test_real_data_readiness_blocks_missing_local_dino_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        robot_path = root / "real_robot.zarr"
        open_path = root / "real_open.zarr"
        _write_real_data_readiness_zarr_metadata(robot_path, "robot")
        _write_real_data_readiness_zarr_metadata(open_path, "open")
        missing_checkpoint = root / "missing_dinov3.ckpt"
        missing_cache = root / "missing_dino_cache"
        cfg = load_cfg(
            "train_gaze_wam_workspace",
            overrides=[
                "task=gaze_wam",
                f"task.robot_dataset_path={robot_path}",
                f"task.open_dataset_path={open_path}",
                f"policy.obs_encoder.checkpoint_path={missing_checkpoint}",
                f"policy.obs_encoder.cache_dir={missing_cache}",
            ],
        )
        summary = launch_gaze_wam_training(
            config_name="train_gaze_wam_workspace",
            use_accelerate=True,
            accelerate_config="accelerate/8gpu-amp.yaml",
            skip_preflight=True,
            run=False,
        )
        readiness = check_real_data_readiness(
            config_name="train_gaze_wam_workspace",
            cfg=cfg,
            acceleration=summary["acceleration"],
            output_json="data/outputs/gaze_wam_launch_report.json",
            skip_preflight=False,
            skip_zarr_validation=False,
            skip_loss_smoke=False,
            preflight_require_timestamps=True,
            preflight_timestamp_max_delta=None,
            preflight_timestamp_max_step=0.08,
            preflight_fail_on_zarr_warning=True,
            use_accelerate=True,
        )

    failed_names = {
        check["name"]
        for check in readiness["checks"]
        if not check["ok"]
    }
    assert readiness["ok"] is False
    assert "obs_encoder_checkpoint_path_exists" in failed_names
    assert "obs_encoder_cache_dir_exists" in failed_names
    assert any("checkpoint_path does not exist" in error for error in readiness["errors"])
    assert any("cache_dir does not exist" in error for error in readiness["errors"])


def test_real_data_readiness_blocks_wrong_local_dino_path_types():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        robot_path = root / "real_robot.zarr"
        open_path = root / "real_open.zarr"
        checkpoint_dir = root / "checkpoint_dir"
        cache_file = root / "cache_file"
        _write_real_data_readiness_zarr_metadata(robot_path, "robot")
        _write_real_data_readiness_zarr_metadata(open_path, "open")
        checkpoint_dir.mkdir()
        cache_file.write_text("not a cache directory", encoding="utf-8")
        cfg = load_cfg(
            "train_gaze_wam_workspace",
            overrides=[
                "task=gaze_wam",
                f"task.robot_dataset_path={robot_path}",
                f"task.open_dataset_path={open_path}",
                f"policy.obs_encoder.checkpoint_path={checkpoint_dir}",
                f"policy.obs_encoder.cache_dir={cache_file}",
            ],
        )
        summary = launch_gaze_wam_training(
            config_name="train_gaze_wam_workspace",
            use_accelerate=True,
            accelerate_config="accelerate/8gpu-amp.yaml",
            skip_preflight=True,
            run=False,
        )
        readiness = check_real_data_readiness(
            config_name="train_gaze_wam_workspace",
            cfg=cfg,
            acceleration=summary["acceleration"],
            output_json="data/outputs/gaze_wam_launch_report.json",
            skip_preflight=False,
            skip_zarr_validation=False,
            skip_loss_smoke=False,
            preflight_require_timestamps=True,
            preflight_timestamp_max_delta=None,
            preflight_timestamp_max_step=0.08,
            preflight_fail_on_zarr_warning=True,
            use_accelerate=True,
        )

    failed_names = {
        check["name"]
        for check in readiness["checks"]
        if not check["ok"]
    }
    assert readiness["ok"] is False
    assert "obs_encoder_checkpoint_path_is_file" in failed_names
    assert "obs_encoder_cache_dir_is_dir" in failed_names
    assert "obs_encoder_local_weight_source_valid" in failed_names
    assert any("not structurally valid" in error for error in readiness["errors"])
    assert any("checkpoint_path must point to a file" in error for error in readiness["errors"])
    assert any("cache_dir must point to a directory" in error for error in readiness["errors"])


def test_launch_gaze_wam_training_real_data_requires_output_json():
    summary = launch_gaze_wam_training(
        config_name="train_gaze_wam_workspace",
        use_accelerate=True,
        accelerate_config="accelerate/8gpu-amp.yaml",
        skip_preflight=True,
        real_data=True,
        preflight_require_timestamps=True,
        preflight_fail_on_zarr_warning=True,
        run=True,
    )

    failed_names = {
        check["name"]
        for check in summary["real_data_readiness"]["checks"]
        if not check["ok"]
    }
    assert summary["ok"] is False
    assert summary["run_skipped"] == "launch_checks_failed"
    assert "launch_report_path" in failed_names
    assert "timestamp_threshold_configured" in failed_names
    assert any("--output-json" in error for error in summary["errors"])


def test_real_data_readiness_blocks_bad_output_json_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        robot_path = root / "real_robot.zarr"
        open_path = root / "real_open.zarr"
        dino_cache = root / "dino_cache"
        output_dir_path = root / "report_dir"
        blocked_parent = root / "blocked_parent"
        _write_real_data_readiness_zarr_metadata(robot_path, "robot")
        _write_real_data_readiness_zarr_metadata(open_path, "open")
        dino_cache.mkdir()
        output_dir_path.mkdir()
        blocked_parent.write_text("not a directory", encoding="utf-8")
        cfg = load_cfg(
            "train_gaze_wam_workspace",
            overrides=[
                "task=gaze_wam",
                f"task.robot_dataset_path={robot_path}",
                f"task.open_dataset_path={open_path}",
                f"policy.obs_encoder.cache_dir={dino_cache}",
            ],
        )
        summary = launch_gaze_wam_training(
            config_name="train_gaze_wam_workspace",
            use_accelerate=True,
            accelerate_config="accelerate/8gpu-amp.yaml",
            skip_preflight=True,
            run=False,
        )

        readiness_dir = check_real_data_readiness(
            config_name="train_gaze_wam_workspace",
            cfg=cfg,
            acceleration=summary["acceleration"],
            output_json=str(output_dir_path),
            skip_preflight=False,
            skip_zarr_validation=False,
            skip_loss_smoke=False,
            preflight_require_timestamps=True,
            preflight_timestamp_max_delta=None,
            preflight_timestamp_max_step=0.08,
            preflight_fail_on_zarr_warning=True,
            use_accelerate=True,
        )
        readiness_blocked_parent = check_real_data_readiness(
            config_name="train_gaze_wam_workspace",
            cfg=cfg,
            acceleration=summary["acceleration"],
            output_json=str(blocked_parent / "report.json"),
            skip_preflight=False,
            skip_zarr_validation=False,
            skip_loss_smoke=False,
            preflight_require_timestamps=True,
            preflight_timestamp_max_delta=None,
            preflight_timestamp_max_step=0.08,
            preflight_fail_on_zarr_warning=True,
            use_accelerate=True,
        )

    dir_failed_names = {
        check["name"]
        for check in readiness_dir["checks"]
        if not check["ok"]
    }
    blocked_failed_names = {
        check["name"]
        for check in readiness_blocked_parent["checks"]
        if not check["ok"]
    }
    assert readiness_dir["ok"] is False
    assert readiness_blocked_parent["ok"] is False
    assert "launch_report_path_writable_candidate" in dir_failed_names
    assert "launch_report_path_writable_candidate" in blocked_failed_names
    assert any("writable report file path" in error for error in readiness_dir["errors"])
    assert any(
        "writable report file path" in error
        for error in readiness_blocked_parent["errors"]
    )


def test_launch_gaze_wam_training_real_data_requires_timestamp_threshold():
    summary = launch_gaze_wam_training(
        config_name="train_gaze_wam_workspace",
        use_accelerate=True,
        accelerate_config="accelerate/8gpu-amp.yaml",
        skip_preflight=True,
        real_data=True,
        preflight_require_timestamps=True,
        preflight_fail_on_zarr_warning=True,
        output_json="data/outputs/gaze_wam_launch_report.json",
        run=True,
    )

    failed_names = {
        check["name"]
        for check in summary["real_data_readiness"]["checks"]
        if not check["ok"]
    }
    assert summary["ok"] is False
    assert summary["run_skipped"] == "launch_checks_failed"
    assert "timestamp_threshold_configured" in failed_names
    assert "launch_report_path" not in failed_names
    assert any("timestamp threshold" in error for error in summary["errors"])


def test_launch_gaze_wam_positive_finite_helper_rejects_bool():
    assert launch_gaze_wam_training_module._is_positive_finite(0.08) is True
    assert launch_gaze_wam_training_module._is_positive_finite("0.08") is True
    assert launch_gaze_wam_training_module._is_positive_finite(True) is False
    assert launch_gaze_wam_training_module._is_positive_finite(False) is False
    assert launch_gaze_wam_training_module._is_positive_finite(0.0) is False
    assert launch_gaze_wam_training_module._is_positive_finite(-0.1) is False
    assert launch_gaze_wam_training_module._is_positive_finite(float("nan")) is False
    assert launch_gaze_wam_training_module._is_positive_finite(float("inf")) is False


def test_real_data_debug_path_classifier_checks_zarr_store_context():
    assert _looks_like_debug_data_path("data/debug_gaze_wam/robot.zarr") is True
    assert _looks_like_debug_data_path("data/open_smoke/open.zarr") is True
    assert _looks_like_debug_data_path("data/robot_synthetic.zarr") is True
    assert _looks_like_debug_data_path("data/temp/robot.zarr") is True
    assert _looks_like_debug_data_path("C:/Users/yibo/AppData/Local/Temp/real_robot.zarr") is False
    assert _looks_like_debug_data_path("/tmp/real_robot.zarr") is False
    assert _looks_like_debug_data_path("W:/real_data/tmp_collection/real_robot.zarr") is False


def test_real_data_readiness_blocks_debug_dataset_paths():
    cfg = load_cfg(
        "train_gaze_wam_workspace",
        overrides=[
            "task=gaze_wam",
            "task.robot_dataset_path=data/debug_gaze_wam/robot.zarr",
            "task.open_dataset_path=data/smoke_open/open.zarr",
        ],
    )
    summary = launch_gaze_wam_training(
        config_name="train_gaze_wam_workspace",
        use_accelerate=True,
        accelerate_config="accelerate/8gpu-amp.yaml",
        skip_preflight=True,
        run=False,
    )
    readiness = check_real_data_readiness(
        config_name="train_gaze_wam_workspace",
        cfg=cfg,
        acceleration=summary["acceleration"],
        output_json="data/outputs/gaze_wam_launch_report.json",
        skip_preflight=False,
        skip_zarr_validation=False,
        skip_loss_smoke=False,
        preflight_require_timestamps=True,
        preflight_timestamp_max_delta=None,
        preflight_timestamp_max_step=0.08,
        preflight_fail_on_zarr_warning=True,
        use_accelerate=True,
    )

    failed_names = {
        check["name"]
        for check in readiness["checks"]
        if not check["ok"]
    }
    assert readiness["ok"] is False
    assert "robot_dataset_path_not_debug" in failed_names
    assert "open_dataset_path_not_debug" in failed_names
    assert any("debug/smoke/synthetic/temp" in error for error in readiness["errors"])


def test_real_data_readiness_blocks_bad_dataset_path_shape():
    cfg = load_cfg(
        "train_gaze_wam_workspace",
        overrides=[
            "task=gaze_wam",
            "task.robot_dataset_path=data/real_mixed_dataset",
            "task.open_dataset_path=data/real_mixed_dataset",
        ],
    )
    summary = launch_gaze_wam_training(
        config_name="train_gaze_wam_workspace",
        use_accelerate=True,
        accelerate_config="accelerate/8gpu-amp.yaml",
        skip_preflight=True,
        run=False,
    )
    readiness = check_real_data_readiness(
        config_name="train_gaze_wam_workspace",
        cfg=cfg,
        acceleration=summary["acceleration"],
        output_json="data/outputs/gaze_wam_launch_report.json",
        skip_preflight=False,
        skip_zarr_validation=False,
        skip_loss_smoke=False,
        preflight_require_timestamps=True,
        preflight_timestamp_max_delta=None,
        preflight_timestamp_max_step=0.08,
        preflight_fail_on_zarr_warning=True,
        use_accelerate=True,
    )

    failed_names = {
        check["name"]
        for check in readiness["checks"]
        if not check["ok"]
    }
    assert readiness["ok"] is False
    assert "robot_dataset_path_zarr" in failed_names
    assert "open_dataset_path_zarr" in failed_names
    assert "robot_open_dataset_paths_distinct" in failed_names
    assert any(".zarr" in error for error in readiness["errors"])
    assert any("distinct robot and open-source" in error for error in readiness["errors"])


def test_real_data_readiness_blocks_missing_dataset_paths():
    cfg = load_cfg(
        "train_gaze_wam_workspace",
        overrides=[
            "task=gaze_wam",
            "task.robot_dataset_path=data/real_robot_missing.zarr",
            "task.open_dataset_path=data/real_open_missing.zarr",
        ],
    )
    summary = launch_gaze_wam_training(
        config_name="train_gaze_wam_workspace",
        use_accelerate=True,
        accelerate_config="accelerate/8gpu-amp.yaml",
        skip_preflight=True,
        run=False,
    )
    readiness = check_real_data_readiness(
        config_name="train_gaze_wam_workspace",
        cfg=cfg,
        acceleration=summary["acceleration"],
        output_json="data/outputs/gaze_wam_launch_report.json",
        skip_preflight=False,
        skip_zarr_validation=False,
        skip_loss_smoke=False,
        preflight_require_timestamps=True,
        preflight_timestamp_max_delta=None,
        preflight_timestamp_max_step=0.08,
        preflight_fail_on_zarr_warning=True,
        use_accelerate=True,
    )

    failed_names = {
        check["name"]
        for check in readiness["checks"]
        if not check["ok"]
    }
    assert readiness["ok"] is False
    assert "robot_dataset_path_exists" in failed_names
    assert "open_dataset_path_exists" in failed_names
    assert any("does not exist" in error for error in readiness["errors"])


def test_real_data_readiness_blocks_geometry_mismatch():
    cfg = load_cfg(
        "train_gaze_wam_workspace",
        overrides=[
            "task=gaze_wam",
            "task.open_dataset.image_resize_mode=letterbox",
            "task.robot_dataset.image_size=[224,224]",
            "task.robot_dataset.action_horizon=8",
            "task.open_dataset.n_latency_steps=1",
        ],
    )
    summary = launch_gaze_wam_training(
        config_name="train_gaze_wam_workspace",
        use_accelerate=True,
        accelerate_config="accelerate/8gpu-amp.yaml",
        skip_preflight=True,
        run=False,
    )
    readiness = check_real_data_readiness(
        config_name="train_gaze_wam_workspace",
        cfg=cfg,
        acceleration=summary["acceleration"],
        output_json="data/outputs/gaze_wam_launch_report.json",
        skip_preflight=False,
        skip_zarr_validation=False,
        skip_loss_smoke=False,
        preflight_require_timestamps=True,
        preflight_timestamp_max_delta=None,
        preflight_timestamp_max_step=0.08,
        preflight_fail_on_zarr_warning=True,
        use_accelerate=True,
    )

    failed_names = {
        check["name"]
        for check in readiness["checks"]
        if not check["ok"]
    }
    assert readiness["ok"] is False
    assert "image_resize_modes_supported" not in failed_names
    assert "image_size_consistent" in failed_names
    assert "robot_sampling_matches_task" in failed_names
    assert "open_sampling_matches_task" in failed_names
    assert not any("requires direct-stretch" in error for error in readiness["errors"])
    assert not any("share the same image_resize_mode" in error for error in readiness["errors"])
    assert any("image_size" in error and "must match" in error for error in readiness["errors"])
    assert any("robot dataset n_obs_steps" in error for error in readiness["errors"])
    assert any("open dataset n_obs_steps" in error for error in readiness["errors"])


def test_real_data_readiness_blocks_core_contract_mismatch():
    cfg = load_cfg(
        "train_gaze_wam_workspace",
        overrides=[
            "task=gaze_wam",
            "task.image_shape=[3,224,224]",
            "task.n_obs_steps=1",
            "task.action_horizon=8",
            "task.n_latency_steps=1",
            "task.action_dim=9",
            "task.heatmap_token_grid=[8,16]",
            "task.heatmap_num_tokens=128",
            "policy.obs_encoder.model_name=vit_base_patch16_224",
            "policy.use_block_attention_mask=false",
        ],
    )
    summary = launch_gaze_wam_training(
        config_name="train_gaze_wam_workspace",
        use_accelerate=True,
        accelerate_config="accelerate/8gpu-amp.yaml",
        skip_preflight=True,
        run=False,
    )
    readiness = check_real_data_readiness(
        config_name="train_gaze_wam_workspace",
        cfg=cfg,
        acceleration=summary["acceleration"],
        output_json="data/outputs/gaze_wam_launch_report.json",
        skip_preflight=False,
        skip_zarr_validation=False,
        skip_loss_smoke=False,
        preflight_require_timestamps=True,
        preflight_timestamp_max_delta=None,
        preflight_timestamp_max_step=0.08,
        preflight_fail_on_zarr_warning=True,
        use_accelerate=True,
    )

    failed_names = {
        check["name"]
        for check in readiness["checks"]
        if not check["ok"]
    }
    assert readiness["ok"] is False
    assert "image_shape_256" in failed_names
    assert "n_obs_steps_2" in failed_names
    assert "action_horizon_48" in failed_names
    assert "n_latency_steps_0" in failed_names
    assert "action_dim_10" in failed_names
    assert "heatmap_token_grid_16x16" in failed_names
    assert "heatmap_num_tokens_256" in failed_names
    assert "obs_encoder_dinov3_vit16" in failed_names
    assert "block_attention_mask_enabled" in failed_names


def test_real_data_readiness_blocks_main_training_routing_mismatch():
    cfg = load_cfg(
        "train_gaze_wam_workspace",
        overrides=[
            "task=gaze_wam",
            "robot_dataloader.batch_size=64",
            "open_dataloader.batch_size=0",
            "task.robot_gaze_dropout_prob=0.0",
            "task.robot_heatmap_on_gaze_dropout=false",
            "policy.heatmap_objective=clean_token",
            "policy.action_loss_weight=0.5",
            "policy.heatmap_loss_weight=0.25",
            "policy.heatmap_token_kl_loss_weight=0.1",
            "policy.heatmap_xy_loss_weight=0.2",
            "policy.heatmap_point_nll_loss_weight=0.4",
            "policy.heatmap_js_loss_weight=0.3",
        ],
    )
    summary = launch_gaze_wam_training(
        config_name="train_gaze_wam_workspace",
        use_accelerate=True,
        accelerate_config="accelerate/8gpu-amp.yaml",
        skip_preflight=True,
        run=False,
    )
    readiness = check_real_data_readiness(
        config_name="train_gaze_wam_workspace",
        cfg=cfg,
        acceleration=summary["acceleration"],
        output_json="data/outputs/gaze_wam_launch_report.json",
        skip_preflight=False,
        skip_zarr_validation=False,
        skip_loss_smoke=False,
        preflight_require_timestamps=True,
        preflight_timestamp_max_delta=None,
        preflight_timestamp_max_step=0.08,
        preflight_fail_on_zarr_warning=True,
        use_accelerate=True,
    )

    failed_names = {
        check["name"]
        for check in readiness["checks"]
        if not check["ok"]
    }
    assert readiness["ok"] is False
    assert "open_batch_positive" not in failed_names
    assert "source_ratio_75_25" not in failed_names
    assert "robot_gaze_dropout_prob_0p2" in failed_names
    assert "robot_heatmap_on_gaze_dropout_enabled" in failed_names
    assert "heatmap_objective_dsnt_js" in failed_names
    assert "action_loss_weight_1" in failed_names
    assert "heatmap_loss_weight_1" in failed_names
    assert "heatmap_token_kl_loss_weight_0" in failed_names
    assert "heatmap_point_nll_loss_weight_0" in failed_names
    assert "heatmap_xy_loss_weight_1" in failed_names
    assert "heatmap_js_loss_weight_1" in failed_names
    assert not any("75% robot / 25% open-source gaze" in error for error in readiness["errors"])


def test_gaze_wam_smoke_pipeline_generates_artifacts_and_runs_rehearsal():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        summary = run_gaze_wam_smoke_pipeline(
            config_name="train_gaze_wam_debug_workspace",
            output_dir=str(root / "outputs"),
            debug_data_dir=str(root / "debug_data"),
            num_episodes=1,
            episode_length=18,
            image_size=256,
            seed=77,
            device="cpu",
            max_rehearsal_steps=1,
            max_commands_per_step=1,
            num_inference_steps=2,
            run_deployment_rehearsal=True,
            run_split_rehearsal=True,
            extra_overrides=["policy.obs_encoder.pretrained=false"],
        )

        assert summary["ok"] is True
        assert Path(summary["debug_data"]["robot_path"]).exists()
        assert Path(summary["debug_data"]["open_path"]).exists()
        assert summary["debug_data"]["image_resize_mode"] == "stretch"
        assert Path(summary["preflight_json"]).exists()
        assert Path(summary["rehearsal_json"]).exists()
        assert Path(summary["split_rehearsal_config_json"]).exists()
        assert Path(summary["split_rehearsal_json"]).exists()
        assert Path(summary["split_command_jsonl"]).exists()
        assert Path(summary["summary_json"]).exists()
        assert summary["robot_zarr_validation"]["valid"] is True
        assert summary["open_zarr_validation"]["valid"] is True
        assert summary["preflight"]["ok"] is True
        assert summary["rehearsal"]["policy_source"] == "config"
        assert summary["rehearsal"]["num_commands"] == 1
        assert summary["split_rehearsal"]["policy_source"] == "config"
        assert summary["split_rehearsal"]["num_commands"] == 1
        split_config = json.loads(Path(summary["split_rehearsal_config_json"]).read_text(encoding="utf-8"))
        assert split_config["image_provider"]["type"] == "opencv_video"
        split_commands = [
            json.loads(line)
            for line in Path(summary["split_command_jsonl"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(split_commands) == 1
        payload = json.loads(Path(summary["summary_json"]).read_text(encoding="utf-8"))
        assert payload["ok"] is True
        assert payload["rehearsal_record_count"] == 1
        assert payload["split_rehearsal"]["num_commands"] == 1
        robot_root = zarr.open(summary["debug_data"]["robot_path"], mode="r")
        open_root = zarr.open(summary["debug_data"]["open_path"], mode="r")
        assert robot_root["meta"].attrs["dataset_type"] == "robot"
        assert open_root["meta"].attrs["dataset_type"] == "open"
        assert robot_root["meta"].attrs["image_resize_mode"] == "stretch"
        assert open_root["meta"].attrs["image_resize_mode"] == "stretch"


def test_gaze_wam_smoke_pipeline_threads_timestamp_validation_options():
    calls = {
        "validation": [],
        "preflight": None,
    }

    def fake_validate_gaze_wam_zarr(**kwargs):
        calls["validation"].append(kwargs)
        return {
            "valid": True,
            "errors": [],
            "warnings": [],
            "timestamps": {
                "checked": bool(kwargs["require_timestamps"]),
                "intervals": {},
                "alignment": {},
            },
        }

    def fake_preflight_gaze_wam(**kwargs):
        calls["preflight"] = kwargs
        return {"ok": True, "errors": []}

    def fake_rehearsal(**kwargs):
        return {
            "num_steps": 1,
            "num_commands": 1,
            "records": [{"step": 0}],
            "policy_source": "config",
        }

    original_validate = gaze_wam_smoke_pipeline_module.validate_gaze_wam_zarr
    original_preflight = gaze_wam_smoke_pipeline_module.preflight_gaze_wam
    original_rehearsal = gaze_wam_smoke_pipeline_module.run_gaze_wam_config_zarr_deployment_rehearsal
    try:
        gaze_wam_smoke_pipeline_module.validate_gaze_wam_zarr = fake_validate_gaze_wam_zarr
        gaze_wam_smoke_pipeline_module.preflight_gaze_wam = fake_preflight_gaze_wam
        gaze_wam_smoke_pipeline_module.run_gaze_wam_config_zarr_deployment_rehearsal = fake_rehearsal

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = gaze_wam_smoke_pipeline_module.run_gaze_wam_smoke_pipeline(
                config_name="train_gaze_wam_debug_workspace",
                output_dir=str(Path(tmpdir) / "outputs"),
                generate_debug_data=False,
                robot_dataset_path="robot.zarr",
                open_dataset_path="open.zarr",
                image_size=16,
                image_resize_mode="stretch",
                run_split_rehearsal=False,
                require_timestamps=True,
                timestamp_max_delta=0.02,
                timestamp_max_step=0.08,
                fail_on_zarr_warning=True,
            )
    finally:
        gaze_wam_smoke_pipeline_module.validate_gaze_wam_zarr = original_validate
        gaze_wam_smoke_pipeline_module.preflight_gaze_wam = original_preflight
        gaze_wam_smoke_pipeline_module.run_gaze_wam_config_zarr_deployment_rehearsal = original_rehearsal

    assert summary["ok"] is True
    assert summary["timestamp_validation_options"] == {
        "require_timestamps": True,
        "timestamp_max_delta": 0.02,
        "timestamp_max_step": 0.08,
        "fail_on_zarr_warning": True,
    }
    assert len(calls["validation"]) == 2
    assert {call["dataset_type"] for call in calls["validation"]} == {"robot", "open"}
    for call in calls["validation"]:
        assert call["require_timestamps"] is True
        assert call["timestamp_max_delta"] == 0.02
        assert call["timestamp_max_step"] == 0.08
        assert call["image_resize_mode"] == "stretch"
    assert calls["preflight"]["require_timestamps"] is True
    assert calls["preflight"]["timestamp_max_delta"] == 0.02
    assert calls["preflight"]["timestamp_max_step"] == 0.08
    assert calls["preflight"]["fail_on_zarr_warning"] is True


def test_gaze_wam_smoke_pipeline_policy_only_skips_deployment_rehearsal():
    calls = {
        "validation": [],
        "preflight": None,
        "rehearsal": 0,
    }

    def fake_validate_gaze_wam_zarr(**kwargs):
        calls["validation"].append(kwargs)
        return {"valid": True, "errors": [], "warnings": []}

    def fake_preflight_gaze_wam(**kwargs):
        calls["preflight"] = kwargs
        return {"ok": True, "errors": []}

    def fail_rehearsal(**kwargs):
        calls["rehearsal"] += 1
        raise AssertionError("Policy-only smoke should not run deployment rehearsal.")

    original_validate = gaze_wam_smoke_pipeline_module.validate_gaze_wam_zarr
    original_preflight = gaze_wam_smoke_pipeline_module.preflight_gaze_wam
    original_rehearsal = gaze_wam_smoke_pipeline_module.run_gaze_wam_config_zarr_deployment_rehearsal
    try:
        gaze_wam_smoke_pipeline_module.validate_gaze_wam_zarr = fake_validate_gaze_wam_zarr
        gaze_wam_smoke_pipeline_module.preflight_gaze_wam = fake_preflight_gaze_wam
        gaze_wam_smoke_pipeline_module.run_gaze_wam_config_zarr_deployment_rehearsal = fail_rehearsal

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = gaze_wam_smoke_pipeline_module.run_gaze_wam_smoke_pipeline(
                config_name="train_gaze_wam_debug_workspace",
                output_dir=str(Path(tmpdir) / "outputs"),
                generate_debug_data=False,
                robot_dataset_path="robot.zarr",
                open_dataset_path="open.zarr",
                image_size=16,
            )
            payload = json.loads(Path(summary["summary_json"]).read_text(encoding="utf-8"))
    finally:
        gaze_wam_smoke_pipeline_module.validate_gaze_wam_zarr = original_validate
        gaze_wam_smoke_pipeline_module.preflight_gaze_wam = original_preflight
        gaze_wam_smoke_pipeline_module.run_gaze_wam_config_zarr_deployment_rehearsal = original_rehearsal

    assert summary["ok"] is True
    assert summary["run_deployment_rehearsal"] is False
    assert summary["run_split_rehearsal"] is False
    assert summary["rehearsal"] is None
    assert summary["rehearsal_record_count"] == 0
    assert summary["rehearsal_json"] is None
    assert summary["split_rehearsal"] is None
    assert summary["split_rehearsal_config_json"] is None
    assert any("policy-only" in warning for warning in summary["warnings"])
    assert calls["rehearsal"] == 0
    assert len(calls["validation"]) == 2
    assert calls["preflight"] is not None
    assert payload["run_deployment_rehearsal"] is False
    assert payload["rehearsal"] is None


def test_dinov3_patch_obs_encoder_returns_256_tokens_per_frame():
    shape_meta = {
        "obs": {
            "camera0_rgb": {
                "shape": [3, 256, 256],
                "type": "rgb",
                "horizon": 2,
            }
        }
    }
    encoder = TransformerObsEncoder(
        shape_meta=shape_meta,
        model_name="vit_base_patch16_dinov3",
        n_emb=768,
        pretrained=False,
        feature_aggregation="patch",
        transforms=None,
    )

    assert encoder.output_shape() == torch.Size([1, 512, 768])


def test_verify_gaze_wam_dino_source_reports_local_source_and_geometry():
    summary = verify_gaze_wam_dino_source()

    assert summary["ok"] is False
    assert summary["dino_source"]["model_name"] == "vit_base_patch16_dinov3"
    assert summary["geometry"]["expected_tokens_per_frame"] == 256
    assert summary["geometry"]["heatmap_num_tokens"] == 256
    assert summary["geometry"]["heatmap_token_grid"] == [16, 16]
    assert summary["normalization"]["mean"] == [0.485, 0.456, 0.406]
    assert summary["normalization"]["std"] == [0.229, 0.224, 0.225]
    assert any("checkpoint_path or cache_dir" in error for error in summary["errors"])

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "dinov3.ckpt"
        ckpt.write_bytes(b"fake-local-dinov3-weights")
        good = verify_gaze_wam_dino_source(checkpoint_path=str(ckpt))

    assert good["ok"] is True
    assert good["dino_source"]["checkpoint_path_is_file"] is True
    assert good["dino_source"]["checkpoint_path_file_size"] > 0
    assert good["dino_source"]["local_source_valid"] is True
