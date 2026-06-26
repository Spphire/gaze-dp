import dataclasses
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from diffusion_policy.real_world.gaze_wam_inference import GazeWamInferenceAdapter


def _validate_safety_float(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite float, got {value!r}.")
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite float, got {value!r}.")
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return scalar


def _validate_finite_float(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return scalar


def _validate_positive_int(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if int_value != value and not (
        isinstance(value, np.integer) and int_value == int(value)
    ):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if int_value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int_value


def _validate_position_bound(
    name: str,
    value: Optional[Sequence[float]],
) -> Optional[List[float]]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain exactly 3 finite floats, got {value!r}.")
    if arr.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 finite floats, got {value!r}.")
    return [
        _validate_safety_float(f"{name}[{idx}]", item)
        for idx, item in enumerate(arr.tolist())
    ]


@dataclasses.dataclass
class GazeWamRobotState:
    """Hardware-neutral state snapshot consumed by the Gaze-WAM runner."""

    tcp_pose: Optional[Sequence[float]] = None
    gripper_width: Optional[float] = None
    action_base_abs: Optional[Sequence[float]] = None


@dataclasses.dataclass
class GazeWamSafetyConfig:
    """Simple absolute-action safety limits for scheduled 10D TCP commands."""

    position_min: Optional[Sequence[float]] = None
    position_max: Optional[Sequence[float]] = None
    gripper_min: Optional[float] = None
    gripper_max: Optional[float] = None
    max_position_step: Optional[float] = None

    def __post_init__(self):
        self.position_min = _validate_position_bound("position_min", self.position_min)
        self.position_max = _validate_position_bound("position_max", self.position_max)
        if self.position_min is not None and self.position_max is not None:
            pos_min = np.asarray(self.position_min, dtype=np.float64)
            pos_max = np.asarray(self.position_max, dtype=np.float64)
            if np.any(pos_min > pos_max):
                raise ValueError("position_min must be <= position_max elementwise.")

        if self.gripper_min is not None:
            self.gripper_min = _validate_safety_float("gripper_min", self.gripper_min)
        if self.gripper_max is not None:
            self.gripper_max = _validate_safety_float("gripper_max", self.gripper_max)
        if (
            self.gripper_min is not None
            and self.gripper_max is not None
            and self.gripper_min > self.gripper_max
        ):
            raise ValueError("gripper_min must be <= gripper_max.")

        if self.max_position_step is not None:
            self.max_position_step = _validate_safety_float(
                "max_position_step",
                self.max_position_step,
            )
            if self.max_position_step <= 0:
                raise ValueError("max_position_step must be positive when configured.")

    def clip_actions(
        self,
        action_abs: np.ndarray,
        current_action_base_abs: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        actions = np.asarray(action_abs, dtype=np.float32).copy()
        if actions.ndim != 2 or actions.shape[-1] < 10:
            raise ValueError(f"Expected absolute actions [T, >=10], got {actions.shape}.")

        if self.max_position_step is not None:
            max_step = float(self.max_position_step)
            if max_step <= 0:
                raise ValueError("max_position_step must be positive when configured.")
            if current_action_base_abs is None:
                prev_pos = actions[0, :3].copy()
            else:
                prev_pos = np.asarray(current_action_base_abs, dtype=np.float32).reshape(-1)[:3]
            for i in range(actions.shape[0]):
                delta = actions[i, :3] - prev_pos
                norm = float(np.linalg.norm(delta))
                if norm > max_step:
                    actions[i, :3] = prev_pos + delta / max(norm, 1e-12) * max_step
                prev_pos = actions[i, :3].copy()

        if self.position_min is not None or self.position_max is not None:
            pos_min = (
                np.asarray(self.position_min, dtype=np.float32)
                if self.position_min is not None
                else np.full(3, -np.inf, dtype=np.float32)
            )
            pos_max = (
                np.asarray(self.position_max, dtype=np.float32)
                if self.position_max is not None
                else np.full(3, np.inf, dtype=np.float32)
            )
            actions[:, :3] = np.clip(actions[:, :3], pos_min.reshape(1, 3), pos_max.reshape(1, 3))

        if self.gripper_min is not None or self.gripper_max is not None:
            g_min = -np.inf if self.gripper_min is None else float(self.gripper_min)
            g_max = np.inf if self.gripper_max is None else float(self.gripper_max)
            actions[:, 9] = np.clip(actions[:, 9], g_min, g_max)

        return actions


@dataclasses.dataclass
class GazeWamScheduledCommand:
    """One timestamped absolute action command."""

    step_index: int
    target_time: float
    action_abs: np.ndarray
    raw_action_abs: np.ndarray
    action_pred_relative: Optional[np.ndarray] = None
    was_clipped: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "target_time": self.target_time,
            "action_abs": self.action_abs,
            "raw_action_abs": self.raw_action_abs,
            "action_pred_relative": self.action_pred_relative,
            "was_clipped": self.was_clipped,
        }


class GazeWamDeploymentRunner:
    """Hardware-agnostic real-time runner around ``GazeWamInferenceAdapter``.

    The runner intentionally speaks callback/provider interfaces instead of a concrete robot SDK.
    A deployment can bind these hooks to RealSense, eye tracker, RTDE, or another controller.
    """

    def __init__(
        self,
        adapter: "GazeWamInferenceAdapter",
        image_provider: Any,
        state_provider: Any,
        command_sink: Optional[Any] = None,
        gaze_provider: Optional[Any] = None,
        safety: Optional[GazeWamSafetyConfig] = None,
        command_dt: float = 0.1,
        command_start_delay: float = 0.05,
        max_commands_per_step: Optional[int] = None,
        dry_run: bool = False,
        cfg_scale: Optional[float] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        command_dt = _validate_finite_float("command_dt", command_dt)
        command_start_delay = _validate_finite_float(
            "command_start_delay",
            command_start_delay,
        )
        if command_dt <= 0:
            raise ValueError("command_dt must be positive.")
        if command_start_delay < 0:
            raise ValueError("command_start_delay must be non-negative.")
        if max_commands_per_step is not None:
            max_commands_per_step = _validate_positive_int(
                "max_commands_per_step",
                max_commands_per_step,
            )
        self.adapter = adapter
        self.image_provider = image_provider
        self.state_provider = state_provider
        self.command_sink = command_sink
        self.gaze_provider = gaze_provider
        self.safety = safety
        self.command_dt = float(command_dt)
        self.command_start_delay = float(command_start_delay)
        self.max_commands_per_step = max_commands_per_step
        self.dry_run = bool(dry_run)
        self.cfg_scale = self._validate_optional_nonnegative_float("cfg_scale", cfg_scale)
        self.clock = clock or time.monotonic
        self.last_prediction: Optional[Dict[str, np.ndarray]] = None
        self.last_commands: List[GazeWamScheduledCommand] = []

    @staticmethod
    def _validate_optional_nonnegative_float(name: str, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative float, got {value!r}.")
        return value

    def reset(self) -> None:
        if hasattr(self.adapter, "reset"):
            self.adapter.reset()
        self.last_prediction = None
        self.last_commands = []

    def step(self, now: Optional[float] = None) -> Dict[str, Any]:
        timing_start = float(self.clock())
        now = timing_start if now is None else float(now)
        image = self._read_provider(
            self.image_provider,
            method_names=("get_image", "read_image", "get_rgb", "read"),
            provider_name="image_provider",
        )
        state = self._read_provider(
            self.state_provider,
            method_names=("get_state", "read_state", "get_robot_state", "read"),
            provider_name="state_provider",
        )
        gaze_xy = None
        if self.gaze_provider is not None:
            gaze_xy = self._read_provider(
                self.gaze_provider,
                method_names=("get_gaze", "get_gaze_xy", "read_gaze", "read"),
                provider_name="gaze_provider",
            )

        adapter_kwargs = self._state_to_adapter_kwargs(state)
        prediction_start = float(self.clock())
        prediction = self.adapter.predict_action(
            image=image,
            gaze_xy=gaze_xy,
            cfg_scale=self.cfg_scale,
            **adapter_kwargs,
        )
        prediction_end = float(self.clock())
        raw_action_abs = self._extract_absolute_actions(prediction)
        if self.max_commands_per_step is not None:
            raw_action_abs = raw_action_abs[: self.max_commands_per_step]

        current_base = adapter_kwargs.get("action_base_abs")
        if current_base is None:
            current_base = self._state_get(state, "action_base_abs")
        clipped_action_abs = (
            self.safety.clip_actions(raw_action_abs, current_action_base_abs=current_base)
            if self.safety is not None
            else raw_action_abs.copy()
        )
        action_pred_relative = prediction.get("action_pred_relative")
        if action_pred_relative is not None:
            action_pred_relative = self._as_action_sequence(action_pred_relative)
            if self.max_commands_per_step is not None:
                action_pred_relative = action_pred_relative[: self.max_commands_per_step]
            if action_pred_relative.shape[0] != raw_action_abs.shape[0]:
                raise ValueError(
                    "action_pred_relative length must match action_abs length after command "
                    f"truncation, got {action_pred_relative.shape[0]} and {raw_action_abs.shape[0]}."
                )

        commands = self._build_commands(
            raw_action_abs=raw_action_abs,
            action_abs=clipped_action_abs,
            action_pred_relative=action_pred_relative,
            now=now,
        )
        dispatch_start = float(self.clock())
        if not self.dry_run:
            self._dispatch_commands(commands)
        dispatch_end = float(self.clock())

        prediction_latency = max(0.0, prediction_end - prediction_start)
        dispatch_latency = max(0.0, dispatch_end - dispatch_start)
        total_step_latency = max(0.0, dispatch_end - timing_start)
        command_available_time = now + max(0.0, prediction_end - timing_start)
        command_lead_times = [
            float(command.target_time - command_available_time)
            for command in commands
        ]
        num_late_commands = sum(1 for lead_time in command_lead_times if lead_time < 0.0)
        timing = {
            "step_start_time": now,
            "prediction_latency": float(prediction_latency),
            "dispatch_latency": float(dispatch_latency),
            "total_step_latency": float(total_step_latency),
            "command_available_time": float(command_available_time),
            "command_lead_times": command_lead_times,
            "min_command_lead_time": (
                min(command_lead_times) if command_lead_times else None
            ),
            "num_late_commands": int(num_late_commands),
        }

        self.last_prediction = prediction
        self.last_commands = commands
        return {
            "prediction": prediction,
            "commands": commands,
            "dry_run": self.dry_run,
            "now": now,
            "timing": timing,
        }

    def _build_commands(
        self,
        raw_action_abs: np.ndarray,
        action_abs: np.ndarray,
        action_pred_relative: Optional[np.ndarray],
        now: float,
    ) -> List[GazeWamScheduledCommand]:
        commands = []
        start_time = now + self.command_start_delay
        for i in range(action_abs.shape[0]):
            rel = None if action_pred_relative is None else action_pred_relative[i].copy()
            commands.append(
                GazeWamScheduledCommand(
                    step_index=i,
                    target_time=start_time + i * self.command_dt,
                    action_abs=action_abs[i].copy(),
                    raw_action_abs=raw_action_abs[i].copy(),
                    action_pred_relative=rel,
                    was_clipped=not np.allclose(action_abs[i], raw_action_abs[i]),
                )
            )
        return commands

    def _dispatch_commands(self, commands: List[GazeWamScheduledCommand]) -> None:
        if self.command_sink is None:
            raise RuntimeError("command_sink is required unless dry_run=True.")
        sink = self.command_sink
        if callable(sink):
            sink(commands)
            return
        for method_name in ("schedule_commands", "send_commands"):
            if hasattr(sink, method_name):
                getattr(sink, method_name)(commands)
                return
        if hasattr(sink, "send"):
            for command in commands:
                sink.send(command)
            return
        raise TypeError(
            "command_sink must be callable or expose schedule_commands, send_commands, or send."
        )

    @staticmethod
    def _read_provider(provider: Any, method_names: Sequence[str], provider_name: str) -> Any:
        if callable(provider):
            return provider()
        for method_name in method_names:
            if hasattr(provider, method_name):
                return getattr(provider, method_name)()
        raise TypeError(f"{provider_name} must be callable or expose one of {method_names}.")

    @staticmethod
    def _state_get(state: Any, key: str, default: Any = None) -> Any:
        if isinstance(state, dict):
            return state.get(key, default)
        return getattr(state, key, default)

    def _state_to_adapter_kwargs(self, state: Any) -> Dict[str, Any]:
        action_base_abs = self._state_get(state, "action_base_abs")
        if action_base_abs is not None:
            return {
                "action_base_abs": action_base_abs,
                "gripper_width": self._state_get(state, "gripper_width"),
            }
        tcp_pose = self._state_get(state, "tcp_pose")
        if tcp_pose is None:
            tcp_pose = self._state_get(state, "tcp_pose_6d")
        if tcp_pose is None:
            raise ValueError(
                "state_provider must provide action_base_abs or tcp_pose/tcp_pose_6d so "
                "the runner can schedule absolute commands."
            )
        return {
            "tcp_pose": tcp_pose,
            "gripper_width": self._state_get(state, "gripper_width"),
        }

    @staticmethod
    def _as_action_sequence(action: np.ndarray) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32)
        if arr.ndim == 3:
            if arr.shape[0] != 1:
                raise ValueError(f"Runner expects batch size 1 predictions, got {arr.shape}.")
            arr = arr[0]
        elif arr.ndim == 1:
            arr = arr[None]
        if arr.ndim != 2:
            raise ValueError(f"Expected action sequence [T,D] or [1,T,D], got {arr.shape}.")
        if arr.shape[0] <= 0:
            raise ValueError("Expected action sequence to contain at least one timestep.")
        if arr.shape[-1] <= 0:
            raise ValueError(f"Expected action sequence feature dim to be positive, got {arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("action sequence must contain only finite values.")
        return arr

    def _extract_absolute_actions(self, prediction: Dict[str, np.ndarray]) -> np.ndarray:
        for key in ("action_abs", "action_pred_abs"):
            if key in prediction:
                action_abs = self._as_action_sequence(prediction[key])
                if action_abs.shape[-1] < 10:
                    raise ValueError(
                        f"{key} must have at least 10 action dimensions, got {action_abs.shape}."
                    )
                return action_abs
        raise RuntimeError(
            "GazeWamDeploymentRunner requires absolute action predictions. Ensure the state "
            "provider supplies action_base_abs or tcp_pose so the adapter can return action_abs."
        )
