import dataclasses
import json
import pathlib
from typing import Any, Dict, List, Optional

import numpy as np

from diffusion_policy.real_world.gaze_wam_runner import (
    GazeWamDeploymentRunner,
    GazeWamRobotState,
    GazeWamSafetyConfig,
    GazeWamScheduledCommand,
)
from diffusion_policy.real_world.gaze_wam_inference import action_base_abs_to_10d
from diffusion_policy.real_world.gaze_wam_zarr_replay import (
    GazeWamZarrReplaySource,
    scheduled_command_to_dict,
)


class GazeWamOpenCVCameraProvider:
    """OpenCV-backed RGB image provider for deployment smoke tests and simple cameras."""

    def __init__(
        self,
        source: Any = 0,
        backend: Optional[int] = None,
        convert_bgr_to_rgb: bool = True,
        loop: bool = False,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        warmup_reads: int = 0,
    ) -> None:
        import cv2

        if warmup_reads < 0:
            raise ValueError("warmup_reads must be non-negative.")
        self.cv2 = cv2
        self.source = self._coerce_source(source)
        self.backend = None if backend is None else int(backend)
        self.convert_bgr_to_rgb = bool(convert_bgr_to_rgb)
        self.loop = bool(loop)
        self.capture = self._open_capture()
        if width is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if fps is not None:
            self.capture.set(cv2.CAP_PROP_FPS, float(fps))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open OpenCV video source {source!r}.")
        for _ in range(int(warmup_reads)):
            self.capture.read()

    @staticmethod
    def _coerce_source(source: Any) -> Any:
        if isinstance(source, str) and source.isdigit():
            return int(source)
        return source

    def _open_capture(self):
        if self.backend is None:
            return self.cv2.VideoCapture(self.source)
        return self.cv2.VideoCapture(self.source, self.backend)

    def get_image(self) -> np.ndarray:
        ok, frame = self.capture.read()
        if (not ok or frame is None) and self.loop:
            self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
        if (not ok or frame is None) and self.loop:
            self.capture.release()
            self.capture = self._open_capture()
            ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame from OpenCV video source {self.source!r}.")

        if self.convert_bgr_to_rgb and frame.ndim == 3 and frame.shape[-1] >= 3:
            if frame.shape[-1] == 3:
                frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            elif frame.shape[-1] == 4:
                frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGRA2RGBA)
        return np.asarray(frame)

    read = get_image

    def release(self) -> None:
        self.capture.release()

    close = release

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


class GazeWamJsonlGazeProvider:
    """Replay normalized gaze points from JSONL rows."""

    def __init__(
        self,
        path: str,
        gaze_key: Optional[str] = "gaze_xy",
        x_key: str = "gaze_x",
        y_key: str = "gaze_y",
        gaze_is_normalized: bool = True,
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
        image_width_key: str = "image_width",
        image_height_key: str = "image_height",
        missing_gaze: str = "none",
        eof: str = "none",
        clip: bool = True,
    ) -> None:
        self.path = str(path)
        self.gaze_key = gaze_key
        self.x_key = x_key
        self.y_key = y_key
        self.gaze_is_normalized = bool(gaze_is_normalized)
        self.image_width = image_width
        self.image_height = image_height
        self.image_width_key = image_width_key
        self.image_height_key = image_height_key
        self.missing_gaze = missing_gaze
        self.eof = eof
        self.clip = bool(clip)
        if self.missing_gaze not in ("none", "hold_last", "error"):
            raise ValueError("missing_gaze must be one of: none, hold_last, error.")
        if self.eof not in ("none", "hold_last", "loop", "error"):
            raise ValueError("eof must be one of: none, hold_last, loop, error.")

        self.rows = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"JSONL gaze provider found no rows in {self.path}.")
        self.cursor = 0
        self.last_gaze: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.cursor = 0
        self.last_gaze = None

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _get_path(row: Dict[str, Any], key: Optional[str], default: Any = None) -> Any:
        if key is None:
            return default
        if key in row:
            return row[key]
        value = row
        for part in str(key).split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value

    def _handle_missing(self):
        if self.missing_gaze == "error":
            raise ValueError(f"Missing gaze value at JSONL row {self.cursor - 1} in {self.path}.")
        if self.missing_gaze == "hold_last" and self.last_gaze is not None:
            return self.last_gaze.copy()
        return None

    def _handle_eof(self):
        if self.eof == "loop":
            self.cursor = 0
            return self.get_gaze()
        if self.eof == "hold_last" and self.last_gaze is not None:
            return self.last_gaze.copy()
        if self.eof == "error":
            raise EOFError(f"JSONL gaze provider exhausted {self.path}.")
        return None

    def _extract_gaze(self, row: Dict[str, Any]) -> Optional[np.ndarray]:
        value = self._get_path(row, self.gaze_key, default=None)
        if value is not None:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size < 2:
                return None
            gaze = arr[:2].astype(np.float32)
        else:
            x = self._get_path(row, self.x_key, default=None)
            y = self._get_path(row, self.y_key, default=None)
            if x is None or y is None:
                return None
            gaze = np.asarray([x, y], dtype=np.float32)

        if not np.all(np.isfinite(gaze)):
            return None
        if not self.gaze_is_normalized:
            width = self.image_width
            height = self.image_height
            if width is None:
                width = self._get_path(row, self.image_width_key, default=None)
            if height is None:
                height = self._get_path(row, self.image_height_key, default=None)
            if width is None or height is None:
                raise ValueError(
                    "Pixel-space gaze replay requires image_width/image_height in the row "
                    "or provider config."
                )
            width = float(width)
            height = float(height)
            if width <= 0 or height <= 0:
                raise ValueError("image_width and image_height must be positive.")
            gaze = gaze / np.asarray([width, height], dtype=np.float32)
        if self.clip:
            gaze = np.clip(gaze, 0.0, 1.0)
        return gaze.astype(np.float32)

    def get_gaze(self):
        if self.cursor >= len(self.rows):
            return self._handle_eof()
        row = self.rows[self.cursor]
        self.cursor += 1
        gaze = self._extract_gaze(row)
        if gaze is None:
            return self._handle_missing()
        self.last_gaze = gaze.copy()
        return gaze

    read = get_gaze
    read_gaze = get_gaze


class _JsonlReplayMixin:
    def _load_jsonl_rows(self, path: str) -> List[Dict[str, Any]]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            raise ValueError(f"JSONL provider found no rows in {path}.")
        return rows

    @staticmethod
    def _get_path(row: Dict[str, Any], key: Optional[str], default: Any = None) -> Any:
        if key is None:
            return default
        if key in row:
            return row[key]
        value = row
        for part in str(key).split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value


class GazeWamJsonlStateProvider(_JsonlReplayMixin):
    """Replay robot state snapshots from JSONL rows."""

    def __init__(
        self,
        path: str,
        action_base_abs_key: str = "action_base_abs",
        tcp_pose_key: str = "tcp_pose",
        tcp_pose_6d_key: str = "tcp_pose_6d",
        gripper_width_key: str = "gripper_width",
        eof: str = "error",
    ) -> None:
        self.path = str(path)
        self.action_base_abs_key = action_base_abs_key
        self.tcp_pose_key = tcp_pose_key
        self.tcp_pose_6d_key = tcp_pose_6d_key
        self.gripper_width_key = gripper_width_key
        self.eof = eof
        if self.eof not in ("hold_last", "loop", "error"):
            raise ValueError("eof must be one of: hold_last, loop, error.")
        self.rows = self._load_jsonl_rows(self.path)
        self.cursor = 0
        self.last_state: Optional[GazeWamRobotState] = None

    def reset(self) -> None:
        self.cursor = 0
        self.last_state = None

    def __len__(self) -> int:
        return len(self.rows)

    def _handle_eof(self) -> GazeWamRobotState:
        if self.eof == "loop":
            self.cursor = 0
            return self.get_state()
        if self.eof == "hold_last" and self.last_state is not None:
            return self.last_state
        raise EOFError(f"JSONL state provider exhausted {self.path}.")

    def _array_or_none(self, row: Dict[str, Any], key: Optional[str]) -> Optional[np.ndarray]:
        value = self._get_path(row, key, default=None)
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return None
        return arr

    def _float_or_none(self, row: Dict[str, Any], key: Optional[str]) -> Optional[float]:
        value = self._get_path(row, key, default=None)
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size != 1:
            raise ValueError(f"{key} must contain exactly one scalar, got {arr.size} value(s).")
        if not np.isfinite(arr[0]):
            return None
        return float(arr[0])

    def _extract_state(self, row: Dict[str, Any]) -> GazeWamRobotState:
        action_base_abs = self._array_or_none(row, self.action_base_abs_key)
        if action_base_abs is not None:
            if action_base_abs.shape[-1] not in (9, 10):
                raise ValueError(
                    f"{self.action_base_abs_key} must contain 9 or 10 values, "
                    f"got {action_base_abs.shape[-1]}."
                )
            gripper_width = self._float_or_none(row, self.gripper_width_key)
            if action_base_abs.shape[-1] == 9 and gripper_width is None:
                raise ValueError(
                    f"9D {self.action_base_abs_key} requires {self.gripper_width_key} "
                    "so deployment can build the 10D Gaze-WAM action base."
                )
            return GazeWamRobotState(
                action_base_abs=action_base_abs.astype(np.float32),
                gripper_width=gripper_width,
            )

        tcp_pose = self._array_or_none(row, self.tcp_pose_key)
        if tcp_pose is None:
            tcp_pose = self._array_or_none(row, self.tcp_pose_6d_key)
        if tcp_pose is None:
            raise ValueError(
                f"JSONL state row must contain {self.action_base_abs_key!r}, "
                f"{self.tcp_pose_key!r}, or {self.tcp_pose_6d_key!r}."
            )
        if tcp_pose.shape[-1] not in (6, 9, 10):
            raise ValueError(
                f"tcp pose state must contain 6, 9, or 10 values, got {tcp_pose.shape[-1]}."
            )
        return GazeWamRobotState(
            tcp_pose=tcp_pose.astype(np.float32),
            gripper_width=self._float_or_none(row, self.gripper_width_key),
        )

    def get_state(self) -> GazeWamRobotState:
        if self.cursor >= len(self.rows):
            return self._handle_eof()
        row = self.rows[self.cursor]
        self.cursor += 1
        state = self._extract_state(row)
        self.last_state = state
        return state

    read = get_state
    read_state = get_state


class GazeWamStaticStateProvider:
    """Static robot state provider for bench tests and split-provider config smoke runs."""

    def __init__(
        self,
        action_base_abs: Optional[Any] = None,
        tcp_pose: Optional[Any] = None,
        gripper_width: Optional[float] = None,
    ) -> None:
        if action_base_abs is None and tcp_pose is None:
            raise ValueError("Static state provider requires action_base_abs or tcp_pose.")
        gripper_width = self._normalize_gripper_width(gripper_width)
        if action_base_abs is not None:
            action_base_abs = action_base_abs_to_10d(
                action_base_abs,
                gripper_width=gripper_width,
            )
        if tcp_pose is not None:
            tcp_pose = self._normalize_tcp_pose(tcp_pose)
        self.state = GazeWamRobotState(
            action_base_abs=None if action_base_abs is None else np.asarray(action_base_abs, dtype=np.float32),
            tcp_pose=tcp_pose,
            gripper_width=gripper_width,
        )

    @staticmethod
    def _normalize_gripper_width(gripper_width: Optional[float]) -> Optional[float]:
        if gripper_width is None:
            return None
        arr = np.asarray(gripper_width, dtype=np.float32).reshape(-1)
        if arr.size != 1:
            raise ValueError(f"gripper_width must contain exactly one scalar, got {arr.size} value(s).")
        if not np.isfinite(arr[0]):
            return None
        return float(arr[0])

    @staticmethod
    def _normalize_tcp_pose(tcp_pose: Any) -> np.ndarray:
        arr = np.asarray(tcp_pose, dtype=np.float32).reshape(-1)
        if arr.size not in (6, 9, 10):
            raise ValueError(f"tcp_pose must contain 6, 9, or 10 values, got {arr.size}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("tcp_pose must contain only finite values.")
        return arr.astype(np.float32)

    def get_state(self) -> GazeWamRobotState:
        return self.state

    read = get_state


class GazeWamRecordingCommandSink:
    """Command sink that records scheduled commands and can write them to JSON."""

    def __init__(self, output_json: Optional[str] = None) -> None:
        self.output_json = output_json
        self.batches: List[List[GazeWamScheduledCommand]] = []

    @property
    def commands(self) -> List[GazeWamScheduledCommand]:
        return [command for batch in self.batches for command in batch]

    def schedule_commands(self, commands: List[GazeWamScheduledCommand]) -> None:
        self.batches.append(list(commands))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_batches": len(self.batches),
            "num_commands": len(self.commands),
            "batches": [
                [scheduled_command_to_dict(command) for command in batch]
                for batch in self.batches
            ],
        }

    def flush(self) -> Optional[str]:
        if self.output_json is None:
            return None
        output = pathlib.Path(self.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return str(output)


class GazeWamJsonlCommandSink:
    """Append scheduled commands to a JSONL file for queue-style dry integration."""

    def __init__(
        self,
        output_jsonl: Optional[str] = None,
        path: Optional[str] = None,
        append: bool = True,
        flush_each_batch: bool = True,
        include_batch_index: bool = True,
    ) -> None:
        if output_jsonl is None and path is None:
            raise ValueError("GazeWamJsonlCommandSink requires output_jsonl or path.")
        self.output_jsonl = str(output_jsonl if output_jsonl is not None else path)
        self.append = bool(append)
        self.flush_each_batch = bool(flush_each_batch)
        self.include_batch_index = bool(include_batch_index)
        self.num_batches = 0
        self.num_commands = 0
        output = pathlib.Path(self.output_jsonl)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not self.append:
            output.write_text("", encoding="utf-8")

    def _record_for_command(
        self,
        command: GazeWamScheduledCommand,
        command_index_in_batch: int,
    ) -> Dict[str, Any]:
        record = scheduled_command_to_dict(command)
        if self.include_batch_index:
            record["batch_index"] = int(self.num_batches)
            record["command_index_in_batch"] = int(command_index_in_batch)
            record["global_command_index"] = int(self.num_commands + command_index_in_batch)
        return record

    def schedule_commands(self, commands: List[GazeWamScheduledCommand]) -> None:
        output = pathlib.Path(self.output_jsonl)
        with output.open("a", encoding="utf-8") as f:
            for i, command in enumerate(commands):
                record = self._record_for_command(command, command_index_in_batch=i)
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if self.flush_each_batch:
                f.flush()
        self.num_batches += 1
        self.num_commands += len(commands)

    send_commands = schedule_commands

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_jsonl": self.output_jsonl,
            "num_batches": int(self.num_batches),
            "num_commands": int(self.num_commands),
            "append": bool(self.append),
        }

    def flush(self) -> str:
        return self.output_jsonl


def _as_dict(value: Optional[Any]) -> Dict[str, Any]:
    if value is None:
        return {}
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    raise TypeError(f"Expected dict-like config, got {type(value).__name__}.")


def build_gaze_wam_zarr_replay_source(config: Any) -> GazeWamZarrReplaySource:
    cfg = _as_dict(config)
    provider_type = cfg.pop("type", "zarr_replay")
    if provider_type != "zarr_replay":
        raise ValueError(f"Unsupported replay source type {provider_type!r}.")
    return GazeWamZarrReplaySource(**cfg)


def build_gaze_wam_image_provider(config: Any):
    if callable(config) or hasattr(config, "get_image"):
        return config
    cfg = _as_dict(config)
    provider_type = cfg.pop("type", "opencv_camera")
    if provider_type in ("opencv_camera", "opencv_video", "opencv"):
        return GazeWamOpenCVCameraProvider(**cfg)
    if provider_type == "zarr_replay":
        return build_gaze_wam_zarr_replay_source(cfg)
    raise ValueError(f"Unsupported image provider type {provider_type!r}.")


def build_gaze_wam_state_provider(config: Any):
    if callable(config) or hasattr(config, "get_state"):
        return config
    cfg = _as_dict(config)
    provider_type = cfg.pop("type", None)
    if provider_type is None:
        provider_type = "zarr_replay" if "dataset_path" in cfg else "static"
    if provider_type == "zarr_replay":
        return build_gaze_wam_zarr_replay_source(cfg)
    if provider_type in ("jsonl_replay", "jsonl"):
        return GazeWamJsonlStateProvider(**cfg)
    if provider_type == "static":
        return GazeWamStaticStateProvider(**cfg)
    raise ValueError(f"Unsupported state provider type {provider_type!r}.")


def build_gaze_wam_gaze_provider(config: Optional[Any]):
    if config is None:
        return None
    if callable(config) or hasattr(config, "get_gaze"):
        return config
    cfg = _as_dict(config)
    provider_type = cfg.pop("type", "jsonl_replay")
    if provider_type == "none":
        return None
    if provider_type in ("jsonl_replay", "jsonl"):
        return GazeWamJsonlGazeProvider(**cfg)
    if provider_type == "zarr_replay":
        return build_gaze_wam_zarr_replay_source(cfg)
    raise ValueError(f"Unsupported gaze provider type {provider_type!r}.")


def build_gaze_wam_command_sink(config: Optional[Any]):
    cfg = _as_dict(config)
    sink_type = cfg.pop("type", "recording")
    if sink_type == "none":
        return None
    if sink_type == "recording":
        return GazeWamRecordingCommandSink(**cfg)
    if sink_type in ("jsonl", "jsonl_queue"):
        return GazeWamJsonlCommandSink(**cfg)
    raise ValueError(f"Unsupported command sink type {sink_type!r}.")


def build_gaze_wam_deployment_runner_from_config(
    adapter,
    config: Any,
    clock=None,
) -> GazeWamDeploymentRunner:
    """Build a deployment runner from provider-style config.

    The first concrete provider type is ``zarr_replay`` for offline hardware-binding rehearsal.
    Real camera, gaze, state, and command providers can be added here without changing the runner.
    """
    cfg = _as_dict(config)
    providers_cfg = cfg.get("providers")
    providers = _as_dict(providers_cfg) if providers_cfg is not None else {}
    source_cfg = cfg.get("source")
    if source_cfg is None and providers_cfg is not None:
        if "type" in providers or "dataset_path" in providers:
            source_cfg = providers_cfg

    provider_handles = []
    source = None
    if source_cfg is not None:
        source = build_gaze_wam_zarr_replay_source(source_cfg)
        image_provider = source.get_image
        state_provider = source.get_state
        gaze_provider = source.get_gaze
        provider_handles.append(source)
    else:
        image_cfg = cfg.get("image_provider", providers.get("image_provider", providers.get("image")))
        state_cfg = cfg.get("state_provider", providers.get("state_provider", providers.get("state")))
        gaze_cfg = cfg.get("gaze_provider", providers.get("gaze_provider", providers.get("gaze")))
        if image_cfg is None or state_cfg is None:
            raise ValueError(
                "Deployment config must contain source/providers, or both image_provider "
                "and state_provider sections."
            )
        image_provider = build_gaze_wam_image_provider(image_cfg)
        state_provider = build_gaze_wam_state_provider(state_cfg)
        gaze_provider = build_gaze_wam_gaze_provider(gaze_cfg)
        provider_handles.extend(
            provider
            for provider in (image_provider, state_provider, gaze_provider)
            if provider is not None
        )

    sink = build_gaze_wam_command_sink(cfg.get("command_sink", {"type": "recording"}))
    safety_cfg = cfg.get("safety")
    safety = GazeWamSafetyConfig(**_as_dict(safety_cfg)) if safety_cfg is not None else None
    runner = GazeWamDeploymentRunner(
        adapter=adapter,
        image_provider=image_provider,
        state_provider=state_provider,
        gaze_provider=gaze_provider,
        command_sink=sink,
        safety=safety,
        command_dt=float(cfg.get("command_dt", 0.1)),
        command_start_delay=float(cfg.get("command_start_delay", 0.05)),
        max_commands_per_step=cfg.get("max_commands_per_step"),
        dry_run=bool(cfg.get("dry_run", True)),
        cfg_scale=cfg.get("cfg_scale"),
        clock=clock,
    )
    runner.provider_source = source
    runner.provider_handles = provider_handles
    runner.command_sink_handle = sink
    return runner
