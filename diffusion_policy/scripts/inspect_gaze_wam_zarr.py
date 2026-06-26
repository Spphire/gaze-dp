from __future__ import annotations

import argparse
import json
import pathlib
import shlex
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _ensure_inspector_runtime():
    global np
    global zarr
    try:
        return zarr
    except NameError:
        import numpy as _np
        import zarr as _zarr

        np = _np
        zarr = _zarr
        return zarr


CANONICAL_KEYS = {
    "camera": "camera0_rgb",
    "action": "action_abs_tcp",
    "tcp_pose": "tcp_pose_abs",
    "gripper": "gripper_width",
    "gaze": "gaze_xy",
    "heatmap": "gaze_heatmap",
    "timestamp": "timestamp",
}

ALIASES = {
    "camera": (
        "camera0_rgb",
        "rgb",
        "image",
        "images",
        "front_rgb",
        "wrist_rgb",
        "head_rgb",
        "color",
    ),
    "action": (
        "action_abs_tcp",
        "action",
        "actions",
        "future_tcp_pose",
        "target_tcp_pose",
        "tcp_action",
        "eef_action",
    ),
    "tcp_pose": (
        "tcp_pose_abs",
        "tcp_pose",
        "current_tcp_pose",
        "robot_tcp_pose",
        "eef_pose",
        "ee_pose",
    ),
    "gripper": (
        "gripper_width",
        "gripper",
        "jaw_width",
        "gripper_position",
        "gripper_pos",
    ),
    "gaze": (
        "gaze_xy",
        "gaze",
        "eye_xy",
        "eye_pixel_xy",
        "gaze_point",
        "fixation_xy",
    ),
    "heatmap": (
        "gaze_heatmap",
        "heatmap",
        "gaze_map",
        "attention_map",
        "mask",
    ),
    "timestamp": (
        "timestamp",
        "time",
        "sensor_time",
        "camera_timestamp",
        "camera_receive_timestamp",
        "robot_receive_timestamp",
        "robot_state_timestamp",
        "gaze_timestamp",
        "action_timestamp",
    ),
}


def _open_root(path: str):
    _ensure_inspector_runtime()
    if str(path).endswith(".zip"):
        store = zarr.ZipStore(path, mode="r")
        return zarr.group(store=store), store
    return zarr.open(path, mode="r"), None


def _is_array(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and not hasattr(value, "keys")


def _walk_arrays(group, prefix: str = "") -> List[Tuple[str, Any]]:
    arrays: List[Tuple[str, Any]] = []
    for key in sorted(group.keys()):
        value = group[key]
        path = f"{prefix}/{key}" if prefix else key
        if _is_array(value):
            arrays.append((path, value))
        elif hasattr(value, "keys"):
            arrays.extend(_walk_arrays(value, path))
    return arrays


def _data_key(path: str) -> str:
    return path[len("data/") :] if path.startswith("data/") else path


def _sample_array(array, max_items: int) -> np.ndarray:
    _ensure_inspector_runtime()
    shape = tuple(int(v) for v in array.shape)
    if len(shape) == 0:
        return np.asarray(array[()])
    slices = []
    for dim in shape:
        slices.append(slice(0, min(int(dim), max_items)))
    return np.asarray(array[tuple(slices)])


def _numeric_stats(array, max_items: int) -> Dict[str, Any]:
    _ensure_inspector_runtime()
    stats = {
        "min": None,
        "max": None,
        "mean": None,
        "finite": None,
    }
    if not np.issubdtype(array.dtype, np.number):
        return stats
    sample = _sample_array(array, max_items=max_items).astype(np.float64, copy=False)
    if sample.size == 0:
        return stats
    finite = np.isfinite(sample)
    stats["finite"] = bool(finite.all())
    if finite.any():
        valid = sample[finite]
        stats["min"] = float(valid.min())
        stats["max"] = float(valid.max())
        stats["mean"] = float(valid.mean())
    return stats


def _array_summary(path: str, array, max_items: int) -> Dict[str, Any]:
    return {
        "path": path,
        "key": _data_key(path),
        "shape": [int(v) for v in array.shape],
        "rank": int(len(array.shape)),
        "dtype": str(array.dtype),
        "chunks": [int(v) for v in getattr(array, "chunks", ())],
        "stats": _numeric_stats(array, max_items=max_items),
    }


def _score_candidate(summary: Dict[str, Any], role: str) -> int:
    key = str(summary["key"]).lower()
    shape = summary["shape"]
    rank = summary["rank"]
    score = 0
    for alias in ALIASES[role]:
        alias_l = alias.lower()
        if key == alias_l:
            score += 50
        elif alias_l in key:
            score += 20

    if role == "camera":
        if rank == 4:
            if shape[-1] in (1, 3, 4) or (len(shape) > 1 and shape[1] in (1, 3, 4)):
                score += 40
        if any(token in key for token in ("rgb", "image", "camera", "color")):
            score += 10
    elif role in ("action", "tcp_pose"):
        if rank == 2 and shape[-1] in (9, 10):
            score += 40
        if role == "action" and any(token in key for token in ("action", "future", "target")):
            score += 15
        if role == "tcp_pose" and any(token in key for token in ("tcp", "pose", "current", "eef")):
            score += 15
    elif role == "gripper":
        if rank in (1, 2) and (rank == 1 or shape[-1] <= 2):
            score += 30
        if any(token in key for token in ("gripper", "jaw")):
            score += 20
    elif role == "gaze":
        if rank == 2 and shape[-1] == 2:
            score += 40
        else:
            return 0
        if any(token in key for token in ("gaze", "eye", "fixation")):
            score += 20
    elif role == "heatmap":
        if rank in (3, 4):
            score += 25
        if any(token in key for token in ("heatmap", "map", "mask")):
            score += 20
    elif role == "timestamp":
        if rank in (1, 2) and (rank == 1 or shape[-1] == 1):
            score += 35
        else:
            return 0
        if key in ("timestamp", "time", "sensor_time"):
            score += 60
        if any(
            token in key
            for token in (
                "image_timestamp",
                "robot_state_timestamp",
                "action_timestamp",
                "gaze_timestamp",
            )
        ):
            score -= 45
        if any(token in key for token in ("timestamp", "time")):
            score += 25
        if "sensor_time" in key:
            score += 20
    return int(score)


def _best_candidates(arrays: Sequence[Dict[str, Any]], role: str, top_k: int) -> List[Dict[str, Any]]:
    scored = []
    for summary in arrays:
        score = _score_candidate(summary, role)
        if score > 0:
            scored.append(
                {
                    "key": summary["key"],
                    "path": summary["path"],
                    "score": score,
                    "shape": summary["shape"],
                    "dtype": summary["dtype"],
                    "stats": summary["stats"],
                }
            )
    scored.sort(key=lambda item: (-int(item["score"]), str(item["key"])))
    return scored[:top_k]


def _episode_ends_info(arrays: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    for summary in arrays:
        if summary["path"] in ("meta/episode_ends", "episode_ends") or summary["key"] == "episode_ends":
            info = {
                "path": summary["path"],
                "shape": summary["shape"],
                "dtype": summary["dtype"],
                "valid_shape": summary["rank"] == 1,
                "last": None,
                "strictly_increasing": None,
            }
            try:
                root_path = summary["path"]
                info["path"] = root_path
            except Exception:
                pass
            return info
    return {
        "path": None,
        "shape": None,
        "dtype": None,
        "valid_shape": False,
        "last": None,
        "strictly_increasing": None,
    }


def _read_episode_values(root, episode_path: Optional[str]) -> Dict[str, Any]:
    _ensure_inspector_runtime()
    if episode_path is None:
        return {}
    try:
        array = root
        for part in episode_path.split("/"):
            array = array[part]
        values = np.asarray(array[:], dtype=np.int64)
        return {
            "last": int(values[-1]) if values.size > 0 else None,
            "strictly_increasing": bool(np.all(np.diff(values) > 0)) if values.size > 1 else True,
            "values_preview": [int(v) for v in values[:10].tolist()],
        }
    except Exception as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}


def _canonicalizer_args(suggestions: Dict[str, List[Dict[str, Any]]]) -> Optional[List[str]]:
    required = ("camera", "action", "tcp_pose", "gripper")
    if any(not suggestions.get(role) for role in required):
        return None
    has_gaze_label = bool(suggestions.get("gaze"))
    has_heatmap_label = bool(suggestions.get("heatmap"))
    if not has_gaze_label and not has_heatmap_label:
        return None
    args = [
        "--camera-key",
        suggestions["camera"][0]["key"],
        "--action-key",
        suggestions["action"][0]["key"],
        "--tcp-pose-key",
        suggestions["tcp_pose"][0]["key"],
        "--gripper-key",
        suggestions["gripper"][0]["key"],
    ]
    if has_gaze_label:
        args.extend(["--gaze-key", suggestions["gaze"][0]["key"]])
    if has_heatmap_label:
        args.extend(["--heatmap-key", suggestions["heatmap"][0]["key"]])
    if suggestions.get("timestamp"):
        args.extend([
            "--timestamp-key",
            suggestions["timestamp"][0]["key"],
        ])
    return args


def _source_key_map(suggestions: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Optional[str]]:
    return {
        role: suggestions[role][0]["key"] if suggestions.get(role) else None
        for role in ("camera", "action", "tcp_pose", "gripper", "gaze", "heatmap", "timestamp")
    }


def _mapping_status(
    suggestions: Dict[str, List[Dict[str, Any]]],
    guessed_type: str,
    canonicalizer_args: Optional[Sequence[str]],
) -> Dict[str, Any]:
    required_robot_roles = ("camera", "action", "tcp_pose", "gripper")
    missing_required = [
        role
        for role in required_robot_roles
        if guessed_type == "robot" and not suggestions.get(role)
    ]
    has_any_gaze_label = bool(suggestions.get("gaze") or suggestions.get("heatmap"))
    if guessed_type == "robot" and not has_any_gaze_label:
        missing_required.append("gaze_or_heatmap")
    optional_missing = [
        role
        for role in ("gaze", "heatmap", "timestamp")
        if not suggestions.get(role)
    ]
    return {
        "ready_for_robot_canonicalization": bool(guessed_type == "robot" and canonicalizer_args),
        "missing_required_roles": missing_required,
        "missing_optional_roles": optional_missing,
        "has_point_gaze": bool(suggestions.get("gaze")),
        "has_dense_heatmap": bool(suggestions.get("heatmap")),
        "has_timestamp": bool(suggestions.get("timestamp")),
    }


def _canonicalizer_command_template(
    dataset_path: str,
    canonicalizer_args: Optional[Sequence[str]],
) -> Optional[str]:
    if canonicalizer_args is None:
        return None
    command = [
        "py",
        "scripts/canonicalize_robot_gaze_wam_zarr.py",
        "--input",
        str(dataset_path),
        "--output",
        "<output_robot.zarr>",
        *canonicalizer_args,
    ]
    return " ".join(shlex.quote(str(part)) for part in command)


def inspect_gaze_wam_zarr(
    dataset_path: str,
    dataset_type: str = "auto",
    max_items: int = 32,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Inspect a raw or canonical zarr and suggest Gaze-WAM key mappings."""
    _ensure_inspector_runtime()
    if dataset_type not in ("auto", "robot", "open"):
        raise ValueError("dataset_type must be auto, robot, or open.")
    root, store = _open_root(dataset_path)
    try:
        arrays = [_array_summary(path, array, max_items=max_items) for path, array in _walk_arrays(root)]
        suggestions = {
            role: _best_candidates(arrays, role=role, top_k=top_k)
            for role in ("camera", "action", "tcp_pose", "gripper", "gaze", "heatmap", "timestamp")
        }
        if dataset_type == "auto":
            guessed_type = "robot" if suggestions["action"] and suggestions["tcp_pose"] else "open"
        else:
            guessed_type = dataset_type
        episode_info = _episode_ends_info(arrays)
        episode_info.update(_read_episode_values(root, episode_info.get("path")))
        canonical_args = _canonicalizer_args(suggestions)
        source_key_map = _source_key_map(suggestions)
        mapping_status = _mapping_status(
            suggestions=suggestions,
            guessed_type=guessed_type,
            canonicalizer_args=canonical_args,
        )
        canonicalizer_command_template = _canonicalizer_command_template(
            dataset_path=dataset_path,
            canonicalizer_args=canonical_args,
        )
        warnings = []
        if not episode_info.get("path"):
            warnings.append("No episode_ends array found; dataset adapters require episode boundaries.")
        if guessed_type == "robot" and canonical_args is None:
            warnings.append("Could not infer all robot canonicalizer keys with confidence.")
        if guessed_type == "open" and not (suggestions["gaze"] or suggestions["heatmap"]):
            warnings.append("Could not infer open gaze point or dense heatmap label key.")

        return {
            "dataset_path": str(dataset_path),
            "dataset_type": dataset_type,
            "guessed_dataset_type": guessed_type,
            "num_arrays": int(len(arrays)),
            "arrays": arrays,
            "episode_ends": episode_info,
            "suggestions": suggestions,
            "source_key_map": source_key_map,
            "mapping_status": mapping_status,
            "canonicalizer_args": canonical_args,
            "canonicalizer_command_template": canonicalizer_command_template,
            "warnings": warnings,
        }
    finally:
        if store is not None:
            store.close()


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Inspect a raw/canonical zarr and suggest Gaze-WAM key mappings."
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-type", choices=("auto", "robot", "open"), default="auto")
    parser.add_argument("--max-items", type=int, default=32, help="Max prefix size sampled per axis for stats.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of candidates to report per role.")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    summary = inspect_gaze_wam_zarr(
        dataset_path=args.dataset_path,
        dataset_type=args.dataset_type,
        max_items=args.max_items,
        top_k=args.top_k,
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        output = pathlib.Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
