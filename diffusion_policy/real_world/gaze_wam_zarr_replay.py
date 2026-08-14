from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from diffusion_policy.real_world.gaze_wam_runner import (
    GazeWamDeploymentRunner,
    GazeWamRobotState,
    GazeWamSafetyConfig,
    GazeWamScheduledCommand,
)
from diffusion_policy.common.gaze_utils import as_optional_gaze_wam_key

if TYPE_CHECKING:
    from diffusion_policy.real_world.gaze_wam_inference import GazeWamInferenceAdapter


def _import_hydra_and_omegaconf():
    import hydra
    from omegaconf import OmegaConf
    from diffusion_policy.common.omegaconf_resolvers import (
        register_safe_omegaconf_resolvers,
    )

    register_safe_omegaconf_resolvers()
    return hydra, OmegaConf


def _import_torch():
    import torch

    return torch


def _import_zarr():
    import zarr

    return zarr


def _import_inference_adapter():
    from diffusion_policy.real_world.gaze_wam_inference import (
        GazeWamInferenceAdapter,
    )

    return GazeWamInferenceAdapter


def _action_base_abs_to_10d(action_base_abs, gripper_width=None):
    from diffusion_policy.real_world.gaze_wam_action_base import action_base_abs_to_10d

    return action_base_abs_to_10d(action_base_abs, gripper_width=gripper_width)


def _validate_gaze_wam_zarr(**kwargs):
    from diffusion_policy.scripts.validate_gaze_wam_zarr import validate_gaze_wam_zarr

    return validate_gaze_wam_zarr(**kwargs)


def _open_zarr_root(dataset_path: str):
    zarr = _import_zarr()
    if str(dataset_path).endswith(".zip"):
        store = zarr.ZipStore(dataset_path, mode="r")
        return zarr.group(store=store), store
    return zarr.open(dataset_path, mode="r"), None


def _resolve_data_group(root):
    if "data" in root:
        data_group = root["data"]
        if "meta" in root and "episode_ends" in root["meta"]:
            return data_group, np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if "episode_ends" in root:
        return root, np.asarray(root["episode_ends"][:], dtype=np.int64)
    raise KeyError(
        "Expected either a diffusion_policy-style zarr with data/meta/episode_ends "
        "or a flat zarr group containing episode_ends."
    )


def _as_float_list(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def _array_summary(value: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(value)
    summary = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            summary.update(
                {
                    "min": float(finite.min()),
                    "max": float(finite.max()),
                    "mean": float(finite.mean()),
                }
            )
    return summary


def safety_config_to_dict(safety: Optional[GazeWamSafetyConfig]) -> Optional[Dict[str, Any]]:
    if safety is None:
        return None
    return {
        "position_min": _as_float_list(safety.position_min),
        "position_max": _as_float_list(safety.position_max),
        "gripper_min": None if safety.gripper_min is None else float(safety.gripper_min),
        "gripper_max": None if safety.gripper_max is None else float(safety.gripper_max),
        "max_position_step": (
            None if safety.max_position_step is None else float(safety.max_position_step)
        ),
    }


def safety_config_from_json(path: Optional[str]) -> Optional[GazeWamSafetyConfig]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("Safety JSON must contain an object.")
    allowed = {
        "position_min",
        "position_max",
        "gripper_min",
        "gripper_max",
        "max_position_step",
    }
    unknown = sorted(set(payload.keys()) - allowed)
    if unknown:
        raise ValueError(f"Unknown safety JSON keys: {unknown}.")
    return GazeWamSafetyConfig(**payload)


def load_gaze_wam_config(config_name: str, overrides: Optional[Sequence[str]] = None):
    hydra, OmegaConf = _import_hydra_and_omegaconf()
    config_dir = Path(__file__).resolve().parents[1] / "config"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=list(overrides or []))
    OmegaConf.resolve(cfg)
    return cfg


def build_config_rehearsal_adapter(
    config_name: str,
    dataset_path: str,
    device: str = "cpu",
    overrides: Optional[Sequence[str]] = None,
    num_inference_steps: Optional[int] = None,
    adapter_camera_key: str = "camera0_rgb",
    cfg_scale: float = 1.0,
) -> Tuple[GazeWamInferenceAdapter, object]:
    """Instantiate an untrained Gaze-WAM policy from config for deployment smoke tests."""
    hydra, _ = _import_hydra_and_omegaconf()
    torch = _import_torch()
    GazeWamInferenceAdapter = _import_inference_adapter()
    cfg_overrides = list(overrides or [])
    cfg_overrides.append(f"task.robot_dataset_path={dataset_path}")
    cfg = load_gaze_wam_config(config_name=config_name, overrides=cfg_overrides)
    robot_dataset = hydra.utils.instantiate(cfg.task.robot_dataset)
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(robot_dataset.get_normalizer())
    if num_inference_steps is not None:
        policy.num_inference_steps = int(num_inference_steps)
    policy.eval().to(torch.device(device))
    adapter = GazeWamInferenceAdapter(
        policy=policy,
        shape_meta=cfg.task.shape_meta,
        camera_key=adapter_camera_key,
        device=device,
        cfg_scale=cfg_scale,
    )
    return adapter, cfg


def _validation_error(summary: Dict[str, Any]) -> ValueError:
    errors = summary.get("errors", [])
    message = "; ".join(str(error) for error in errors) if errors else "unknown validation error"
    return ValueError(f"Rehearsal robot zarr validation failed: {message}")


def _validate_rehearsal_robot_zarr(
    dataset_path: str,
    camera_key: str = "camera0_rgb",
    gaze_key: Optional[str] = "gaze_xy",
    heatmap_key: Optional[str] = "gaze_heatmap",
    action_abs_key: str = "action_abs_tcp",
    tcp_pose_key: str = "tcp_pose_abs",
    gripper_key: str = "gripper_width",
    n_obs_steps: int = 2,
    action_horizon: int = 48,
    n_latency_steps: int = 0,
    image_size: Sequence[int] = (256, 256),
    image_resize_mode: str = "stretch",
    heatmap_token_grid: Sequence[int] = (16, 16),
    action_dim: int = 10,
    timestamp_key: Optional[str] = None,
    require_timestamps: bool = False,
    timestamp_max_delta: Optional[float] = None,
    timestamp_max_step: Optional[float] = None,
    check_dataset_sample: bool = True,
) -> Dict[str, Any]:
    summary = _validate_gaze_wam_zarr(
        dataset_path=dataset_path,
        dataset_type="robot",
        camera_key=camera_key,
        gaze_key=gaze_key,
        heatmap_key=heatmap_key,
        action_abs_key=action_abs_key,
        tcp_pose_key=tcp_pose_key,
        gripper_key=gripper_key,
        n_obs_steps=n_obs_steps,
        action_horizon=action_horizon,
        n_latency_steps=n_latency_steps,
        image_size=image_size,
        image_resize_mode=image_resize_mode,
        heatmap_token_grid=heatmap_token_grid,
        action_dim=action_dim,
        timestamp_key=timestamp_key,
        require_timestamps=require_timestamps,
        timestamp_max_delta=timestamp_max_delta,
        timestamp_max_step=timestamp_max_step,
        check_dataset_sample=check_dataset_sample,
    )
    if not summary["valid"]:
        raise _validation_error(summary)
    return summary


def validate_config_rehearsal_robot_zarr(
    config_name: str,
    dataset_path: str,
    overrides: Optional[Sequence[str]] = None,
    timestamp_key: Optional[str] = None,
    require_timestamps: bool = False,
    timestamp_max_delta: Optional[float] = None,
    timestamp_max_step: Optional[float] = None,
) -> Dict[str, Any]:
    cfg_overrides = list(overrides or [])
    cfg_overrides.append(f"task.robot_dataset_path={dataset_path}")
    cfg = load_gaze_wam_config(config_name=config_name, overrides=cfg_overrides)
    robot_cfg = cfg.task.robot_dataset
    return _validate_rehearsal_robot_zarr(
        dataset_path=dataset_path,
        camera_key=str(robot_cfg.camera_key),
        gaze_key=as_optional_gaze_wam_key(robot_cfg.get("gaze_key", None)),
        heatmap_key=as_optional_gaze_wam_key(robot_cfg.get("heatmap_key", None)),
        heatmap_token_grid=cfg.task.heatmap_token_grid,
        action_abs_key=str(robot_cfg.action_abs_key),
        tcp_pose_key=str(robot_cfg.tcp_pose_key),
        gripper_key=str(robot_cfg.gripper_key),
        n_obs_steps=int(cfg.task.n_obs_steps),
        action_horizon=int(cfg.task.action_horizon),
        n_latency_steps=int(cfg.task.n_latency_steps),
        image_size=robot_cfg.image_size,
        image_resize_mode=str(robot_cfg.get("image_resize_mode", "stretch")),
        action_dim=int(cfg.task.action_dim),
        timestamp_key=timestamp_key,
        require_timestamps=require_timestamps,
        timestamp_max_delta=timestamp_max_delta,
        timestamp_max_step=timestamp_max_step,
    )


def scheduled_command_to_dict(command: GazeWamScheduledCommand) -> Dict[str, Any]:
    return {
        "step_index": int(command.step_index),
        "target_time": float(command.target_time),
        "action_abs": np.asarray(command.action_abs, dtype=np.float32).tolist(),
        "raw_action_abs": np.asarray(command.raw_action_abs, dtype=np.float32).tolist(),
        "action_pred_relative": (
            None
            if command.action_pred_relative is None
            else np.asarray(command.action_pred_relative, dtype=np.float32).tolist()
        ),
        "was_clipped": bool(command.was_clipped),
    }


@dataclasses.dataclass
class GazeWamZarrReplaySample:
    replay_step: int
    source_index: int
    episode_index: int
    episode_offset: int
    source_timestamp: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "replay_step": int(self.replay_step),
            "source_index": int(self.source_index),
            "episode_index": int(self.episode_index),
            "episode_offset": int(self.episode_offset),
            "source_timestamp": self.source_timestamp,
        }


class GazeWamZarrReplaySource:
    """Read canonical robot zarr rows through deployment-style provider methods."""

    def __init__(
        self,
        dataset_path: str,
        camera_key: str = "camera0_rgb",
        tcp_pose_key: str = "tcp_pose_abs",
        gripper_key: str = "gripper_width",
        gaze_key: Optional[str] = "gaze_xy",
        action_base_abs_key: Optional[str] = None,
        timestamp_key: Optional[str] = None,
        episode_index: int = 0,
        start_offset: int = 0,
        max_steps: Optional[int] = None,
        stride: int = 1,
        missing_gaze: bool = False,
    ) -> None:
        if stride <= 0:
            raise ValueError("stride must be positive.")
        if start_offset < 0:
            raise ValueError("start_offset must be non-negative.")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive when configured.")

        self.dataset_path = str(dataset_path)
        self.camera_key = camera_key
        self.tcp_pose_key = tcp_pose_key
        self.gripper_key = gripper_key
        self.gaze_key = as_optional_gaze_wam_key(gaze_key)
        self.action_base_abs_key = action_base_abs_key
        self.timestamp_key = timestamp_key
        self.missing_gaze = bool(missing_gaze)

        root, store = _open_zarr_root(self.dataset_path)
        self._zarr_store = store
        self.root = root
        self.data_group, self.episode_ends = _resolve_data_group(root)

        if not 0 <= episode_index < len(self.episode_ends):
            raise ValueError(
                f"episode_index={episode_index} out of range for {len(self.episode_ends)} episodes."
            )
        self.episode_index = int(episode_index)
        episode_start = 0 if episode_index == 0 else int(self.episode_ends[episode_index - 1])
        episode_end = int(self.episode_ends[episode_index])
        first = episode_start + int(start_offset)
        if first >= episode_end:
            raise ValueError(
                f"start_offset={start_offset} starts at {first}, beyond episode end {episode_end}."
            )
        indices = np.arange(first, episode_end, int(stride), dtype=np.int64)
        if max_steps is not None:
            indices = indices[: int(max_steps)]
        if indices.size == 0:
            raise ValueError("Replay source has no indices to replay.")
        self.indices = indices
        self.episode_start = episode_start
        self.episode_end = episode_end
        self.cursor = 0

        self._require_key(self.camera_key)
        if self.action_base_abs_key is not None:
            self._require_key(self.action_base_abs_key)
        else:
            self._require_key(self.tcp_pose_key)
            self._require_key(self.gripper_key)
        if self.timestamp_key is not None:
            self._require_key(self.timestamp_key)

    def _require_key(self, key: str) -> None:
        if key not in self.data_group:
            raise KeyError(f"Replay zarr is missing key '{key}'.")

    def __len__(self) -> int:
        return int(self.indices.size)

    @property
    def source_index(self) -> int:
        return int(self.indices[self.cursor])

    def reset(self) -> None:
        self.cursor = 0

    def has_next(self) -> bool:
        return self.cursor < len(self.indices)

    def advance(self) -> None:
        self.cursor += 1

    def current_sample(self) -> GazeWamZarrReplaySample:
        source_index = self.source_index
        timestamp = None
        if self.timestamp_key is not None:
            timestamp = float(np.asarray(self.data_group[self.timestamp_key][source_index]).reshape(-1)[0])
        return GazeWamZarrReplaySample(
            replay_step=self.cursor,
            source_index=source_index,
            episode_index=self.episode_index,
            episode_offset=source_index - self.episode_start,
            source_timestamp=timestamp,
        )

    def replay_now(self, base_time: float = 0.0, replay_dt: float = 0.1) -> float:
        sample = self.current_sample()
        if sample.source_timestamp is not None:
            return float(sample.source_timestamp)
        return float(base_time + sample.replay_step * replay_dt)

    def get_image(self) -> np.ndarray:
        return np.asarray(self.data_group[self.camera_key][self.source_index])

    def get_gaze(self):
        if self.missing_gaze or self.gaze_key is None or self.gaze_key not in self.data_group:
            return None
        return np.asarray(self.data_group[self.gaze_key][self.source_index], dtype=np.float32)

    def _read_gripper(self, source_index: int) -> float:
        gripper = np.asarray(self.data_group[self.gripper_key][source_index], dtype=np.float32)
        flat = gripper.reshape(-1)
        if flat.size != 1:
            raise ValueError(
                f"{self.gripper_key} must provide exactly one scalar per replay step, "
                f"got shape {gripper.shape} at source index {source_index}."
            )
        return float(flat[0])

    def _compose_action_base_abs(self, source_index: int) -> np.ndarray:
        if self.action_base_abs_key is not None:
            base = np.asarray(self.data_group[self.action_base_abs_key][source_index], dtype=np.float32)
            gripper_width = None
            if base.reshape(-1).shape[-1] == 9:
                self._require_key(self.gripper_key)
                gripper_width = self._read_gripper(source_index)
            return _action_base_abs_to_10d(
                base.reshape(-1),
                gripper_width=gripper_width,
            )

        tcp_pose = np.asarray(self.data_group[self.tcp_pose_key][source_index], dtype=np.float32).reshape(-1)
        if tcp_pose.shape[-1] == 10:
            base = tcp_pose.copy()
        elif tcp_pose.shape[-1] == 9:
            base = np.concatenate(
                [tcp_pose, np.asarray([self._read_gripper(source_index)], dtype=np.float32)],
                axis=0,
            )
        else:
            raise ValueError(
                f"{self.tcp_pose_key} must be 9D or 10D for zarr replay, got {tcp_pose.shape}."
            )
        return base.astype(np.float32)

    def get_state(self) -> GazeWamRobotState:
        source_index = self.source_index
        return GazeWamRobotState(
            action_base_abs=self._compose_action_base_abs(source_index),
        )


class GazeWamReplayCommandRecorder:
    """Collect serializable command records from zarr deployment rehearsal."""

    def __init__(self, include_prediction_summary: bool = True) -> None:
        self.include_prediction_summary = bool(include_prediction_summary)
        self.records: List[Dict[str, Any]] = []

    def record_step(
        self,
        source_sample: GazeWamZarrReplaySample,
        runner_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        prediction = runner_output.get("prediction", {})
        record = {
            **source_sample.as_dict(),
            "now": float(runner_output["now"]),
            "dry_run": bool(runner_output["dry_run"]),
            "timing": runner_output.get("timing", {}),
            "commands": [
                scheduled_command_to_dict(command)
                for command in runner_output.get("commands", [])
            ],
        }
        if self.include_prediction_summary:
            record["prediction_summary"] = {
                key: _array_summary(value)
                for key, value in prediction.items()
                if isinstance(value, np.ndarray)
            }
        self.records.append(record)
        return record

    def summary(self) -> Dict[str, Any]:
        num_commands = sum(len(record["commands"]) for record in self.records)
        num_clipped = sum(
            int(command["was_clipped"])
            for record in self.records
            for command in record["commands"]
        )
        timing_records = [
            record.get("timing", {})
            for record in self.records
            if isinstance(record.get("timing", {}), dict)
        ]
        prediction_latencies = [
            float(timing["prediction_latency"])
            for timing in timing_records
            if timing.get("prediction_latency") is not None
        ]
        min_leads = [
            float(timing["min_command_lead_time"])
            for timing in timing_records
            if timing.get("min_command_lead_time") is not None
        ]
        late_commands = sum(
            int(timing.get("num_late_commands", 0))
            for timing in timing_records
        )
        timing_summary = {
            "max_prediction_latency": (
                max(prediction_latencies) if prediction_latencies else None
            ),
            "mean_prediction_latency": (
                float(np.mean(prediction_latencies)) if prediction_latencies else None
            ),
            "min_command_lead_time": min(min_leads) if min_leads else None,
            "num_late_commands": int(late_commands),
        }
        return {
            "num_steps": len(self.records),
            "num_commands": int(num_commands),
            "num_clipped_commands": int(num_clipped),
            "timing_summary": timing_summary,
            "records": self.records,
        }

    def write_json(self, output_path: str) -> Dict[str, Any]:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = self.summary()
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return payload


def run_gaze_wam_zarr_deployment_rehearsal(
    adapter: Any,
    dataset_path: str,
    output_json: Optional[str] = None,
    camera_key: str = "camera0_rgb",
    tcp_pose_key: str = "tcp_pose_abs",
    gripper_key: str = "gripper_width",
    gaze_key: Optional[str] = "gaze_xy",
    heatmap_key: Optional[str] = "gaze_heatmap",
    action_base_abs_key: Optional[str] = None,
    timestamp_key: Optional[str] = None,
    episode_index: int = 0,
    start_offset: int = 0,
    max_steps: Optional[int] = None,
    stride: int = 1,
    missing_gaze: bool = False,
    command_dt: float = 0.1,
    command_start_delay: float = 0.05,
    max_commands_per_step: Optional[int] = None,
    dry_run: bool = True,
    cfg_scale: Optional[float] = None,
    replay_base_time: float = 0.0,
    replay_dt: float = 0.1,
    safety: Optional[GazeWamSafetyConfig] = None,
    include_prediction_summary: bool = True,
    fail_on_late_commands: bool = False,
    clock: Optional[Callable[[], float]] = None,
) -> Dict[str, Any]:
    """Replay a canonical robot zarr through ``GazeWamDeploymentRunner`` offline."""
    source = GazeWamZarrReplaySource(
        dataset_path=dataset_path,
        camera_key=camera_key,
        tcp_pose_key=tcp_pose_key,
        gripper_key=gripper_key,
        gaze_key=gaze_key,
        action_base_abs_key=action_base_abs_key,
        timestamp_key=timestamp_key,
        episode_index=episode_index,
        start_offset=start_offset,
        max_steps=max_steps,
        stride=stride,
        missing_gaze=missing_gaze,
    )
    recorder = GazeWamReplayCommandRecorder(
        include_prediction_summary=include_prediction_summary,
    )
    runner = GazeWamDeploymentRunner(
        adapter=adapter,
        image_provider=source.get_image,
        state_provider=source.get_state,
        gaze_provider=source.get_gaze,
        command_sink=None,
        safety=safety,
        command_dt=command_dt,
        command_start_delay=command_start_delay,
        max_commands_per_step=max_commands_per_step,
        dry_run=dry_run,
        cfg_scale=cfg_scale,
        clock=clock,
    )

    while source.has_next():
        sample = source.current_sample()
        now = source.replay_now(base_time=replay_base_time, replay_dt=replay_dt)
        output = runner.step(now=now)
        recorder.record_step(sample, output)
        source.advance()

    payload = recorder.summary()
    payload["dataset_path"] = str(dataset_path)
    payload["missing_gaze"] = bool(missing_gaze)
    payload["command_dt"] = float(command_dt)
    payload["command_start_delay"] = float(command_start_delay)
    payload["max_commands_per_step"] = max_commands_per_step
    payload["replay_config"] = {
        "dataset_path": str(dataset_path),
        "camera_key": str(camera_key),
        "tcp_pose_key": str(tcp_pose_key),
        "gripper_key": str(gripper_key),
        "gaze_key": as_optional_gaze_wam_key(gaze_key),
        "heatmap_key": as_optional_gaze_wam_key(heatmap_key),
        "action_base_abs_key": action_base_abs_key,
        "timestamp_key": timestamp_key,
        "episode_index": int(episode_index),
        "start_offset": int(start_offset),
        "max_steps": None if max_steps is None else int(max_steps),
        "stride": int(stride),
        "missing_gaze": bool(missing_gaze),
        "command_dt": float(command_dt),
        "command_start_delay": float(command_start_delay),
        "max_commands_per_step": (
            None if max_commands_per_step is None else int(max_commands_per_step)
        ),
        "dry_run": bool(dry_run),
        "cfg_scale": None if cfg_scale is None else float(cfg_scale),
        "replay_base_time": float(replay_base_time),
        "replay_dt": float(replay_dt),
        "include_prediction_summary": bool(include_prediction_summary),
        "fail_on_late_commands": bool(fail_on_late_commands),
    }
    payload["policy_source"] = "adapter"
    payload["safety"] = safety_config_to_dict(safety)
    if fail_on_late_commands and payload["timing_summary"]["num_late_commands"] > 0:
        raise RuntimeError(
            "Zarr deployment rehearsal produced "
            f"{payload['timing_summary']['num_late_commands']} late scheduled commands; "
            "increase command_start_delay, reduce inference latency, or lower command rate."
        )
    if output_json is not None:
        output = Path(output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return payload


def run_gaze_wam_checkpoint_zarr_deployment_rehearsal(
    checkpoint_path: str,
    dataset_path: str,
    output_json: Optional[str] = None,
    device: str = "cuda:0",
    use_ema: bool = True,
    num_inference_steps: Optional[int] = None,
    adapter_camera_key: str = "camera0_rgb",
    cfg_scale: float = 1.0,
    validate_zarr: bool = True,
    timestamp_key: Optional[str] = None,
    require_timestamps: bool = False,
    timestamp_max_delta: Optional[float] = None,
    timestamp_max_step: Optional[float] = None,
    trust_checkpoint: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    GazeWamInferenceAdapter = _import_inference_adapter()
    validation = None
    if validate_zarr:
        validation = _validate_rehearsal_robot_zarr(
            dataset_path=dataset_path,
            camera_key=str(kwargs.get("camera_key", "camera0_rgb")),
            gaze_key=as_optional_gaze_wam_key(kwargs.get("gaze_key", "gaze_xy")),
            heatmap_key=as_optional_gaze_wam_key(kwargs.get("heatmap_key", "gaze_heatmap")),
            tcp_pose_key=str(kwargs.get("tcp_pose_key", "tcp_pose_abs")),
            gripper_key=str(kwargs.get("gripper_key", "gripper_width")),
            timestamp_key=timestamp_key,
            require_timestamps=require_timestamps,
            timestamp_max_delta=timestamp_max_delta,
            timestamp_max_step=timestamp_max_step,
            check_dataset_sample=False,
        )
    adapter = GazeWamInferenceAdapter.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        use_ema=use_ema,
        num_inference_steps=num_inference_steps,
        camera_key=adapter_camera_key,
        cfg_scale=cfg_scale,
        trust_checkpoint=trust_checkpoint,
    )
    payload = run_gaze_wam_zarr_deployment_rehearsal(
        adapter=adapter,
        dataset_path=dataset_path,
        output_json=output_json,
        cfg_scale=cfg_scale,
        timestamp_key=timestamp_key,
        **kwargs,
    )
    payload["checkpoint_path"] = str(checkpoint_path)
    payload["policy_source"] = "checkpoint"
    payload["zarr_validation"] = validation if validation is not None else {"skipped": True}
    if output_json is not None:
        output = Path(output_json)
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return payload


def run_gaze_wam_config_zarr_deployment_rehearsal(
    config_name: str,
    dataset_path: str,
    output_json: Optional[str] = None,
    device: str = "cpu",
    overrides: Optional[Sequence[str]] = None,
    num_inference_steps: Optional[int] = None,
    adapter_camera_key: str = "camera0_rgb",
    cfg_scale: float = 1.0,
    validate_zarr: bool = True,
    timestamp_key: Optional[str] = None,
    require_timestamps: bool = False,
    timestamp_max_delta: Optional[float] = None,
    timestamp_max_step: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    validation = None
    if validate_zarr:
        validation = validate_config_rehearsal_robot_zarr(
            config_name=config_name,
            dataset_path=dataset_path,
            overrides=overrides,
            timestamp_key=timestamp_key,
            require_timestamps=require_timestamps,
            timestamp_max_delta=timestamp_max_delta,
            timestamp_max_step=timestamp_max_step,
        )
    adapter, cfg = build_config_rehearsal_adapter(
        config_name=config_name,
        dataset_path=dataset_path,
        device=device,
        overrides=overrides,
        num_inference_steps=num_inference_steps,
        adapter_camera_key=adapter_camera_key,
        cfg_scale=cfg_scale,
    )
    payload = run_gaze_wam_zarr_deployment_rehearsal(
        adapter=adapter,
        dataset_path=dataset_path,
        output_json=output_json,
        cfg_scale=cfg_scale,
        timestamp_key=timestamp_key,
        **kwargs,
    )
    payload["config_name"] = config_name
    payload["checkpoint_path"] = None
    payload["policy_source"] = "config"
    payload["resolved_task_name"] = str(cfg.task.name)
    payload["zarr_validation"] = validation if validation is not None else {"skipped": True}
    if output_json is not None:
        output = Path(output_json)
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return payload
