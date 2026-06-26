import argparse
import csv
import json
import pathlib
import re
import shlex
from typing import Any, Dict, List, Optional, Sequence


CANONICAL_ROLES = (
    "video_path",
    "episode_id",
    "frame_idx",
    "timestamp",
    "gaze_x",
    "gaze_y",
    "image_width",
    "image_height",
)


def _is_present(value: Any) -> bool:
    return value is not None and str(value) != ""


def _read_rows(path: pathlib.Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    elif suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("frames", data.get("items", data.get("rows", data.get("annotations"))))
        if not isinstance(data, list):
            raise ValueError("JSON metadata must be a list or contain frames/items/rows/annotations list.")
        rows = list(data)
    else:
        raise ValueError(f"Unsupported metadata suffix '{path.suffix}'. Use .csv, .json, or .jsonl.")
    if not rows:
        raise ValueError(f"Metadata '{path}' has no rows.")
    return rows


def _flatten_row(
    value: Any,
    prefix: str = "",
    max_list_items: int = 8,
) -> Dict[str, Any]:
    if isinstance(value, dict):
        flat: Dict[str, Any] = {}
        for key, item in value.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_row(item, child_key, max_list_items=max_list_items))
        return flat
    if isinstance(value, list):
        flat = {}
        for idx, item in enumerate(value[:max_list_items]):
            child_key = f"{prefix}.{idx}" if prefix else str(idx)
            flat.update(_flatten_row(item, child_key, max_list_items=max_list_items))
        if not flat and prefix:
            flat[prefix] = value
        return flat
    return {prefix: value} if prefix else {}


def _as_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _path_tokens(path: str) -> List[str]:
    return [token for token in re.split(r"[^a-zA-Z0-9]+", path.lower()) if token]


def _looks_like_video_path(value: Any) -> bool:
    text = str(value).lower()
    return any(text.endswith(ext) for ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"))


def _field_summary(path: str, values: Sequence[Any], num_rows: int) -> Dict[str, Any]:
    present = [value for value in values if _is_present(value)]
    numeric_values = [number for number in (_as_float(value) for value in present) if number is not None]
    examples: List[Any] = []
    seen = set()
    for value in present:
        key = json.dumps(value, sort_keys=True, default=str)
        if key in seen:
            continue
        examples.append(value)
        seen.add(key)
        if len(examples) >= 4:
            break
    summary: Dict[str, Any] = {
        "path": path,
        "present_count": int(len(present)),
        "missing_count": int(num_rows - len(present)),
        "numeric_count": int(len(numeric_values)),
        "video_path_like_count": int(sum(_looks_like_video_path(value) for value in present)),
        "examples": examples,
    }
    if numeric_values:
        summary["numeric_min"] = float(min(numeric_values))
        summary["numeric_max"] = float(max(numeric_values))
    return summary


def _summarize_fields(rows: Sequence[Dict[str, Any]], max_list_items: int) -> List[Dict[str, Any]]:
    flat_rows = [
        _flatten_row(row, max_list_items=max_list_items)
        for row in rows
    ]
    paths = sorted({path for flat in flat_rows for path in flat.keys()})
    num_rows = len(rows)
    summaries = []
    for path in paths:
        values = [flat.get(path) for flat in flat_rows]
        summaries.append(_field_summary(path, values, num_rows=num_rows))
    summaries.sort(key=lambda item: str(item["path"]))
    return summaries


def _has_any(tokens: Sequence[str], options: Sequence[str]) -> bool:
    return any(option in tokens for option in options)


def _score_field(field: Dict[str, Any], role: str) -> int:
    path = str(field["path"])
    tokens = _path_tokens(path)
    leaf = tokens[-1] if tokens else ""
    score = 0
    numeric_count = int(field["numeric_count"])
    present_count = int(field["present_count"])
    video_like = int(field["video_path_like_count"])

    if role == "video_path":
        if _has_any(tokens, ("video", "clip", "movie")):
            score += 25
        if _has_any(tokens, ("path", "file", "filename", "url", "source")):
            score += 35
        if leaf in ("mp4", "video_path", "clip_path"):
            score += 30
        if video_like > 0:
            score += 45
        if numeric_count == present_count and present_count > 0:
            score -= 35
    elif role == "episode_id":
        if _has_any(tokens, ("episode", "sequence", "clip", "video", "session", "take")):
            score += 30
        if leaf in ("id", "uid", "uuid", "episode_id", "clip_id", "video_id", "sequence_id"):
            score += 35
        if _has_any(tokens, ("path", "file", "frame", "timestamp", "time", "gaze")):
            score -= 20
    elif role == "frame_idx":
        if _has_any(tokens, ("frame", "image")):
            score += 35
        if leaf in ("idx", "index", "number", "frame_idx", "frame_index", "frame_number"):
            score += 35
        if numeric_count > 0:
            score += 25
        if _has_any(tokens, ("width", "height", "timestamp", "time", "gaze")):
            score -= 35
    elif role == "timestamp":
        if not _has_any(tokens, ("timestamp", "time", "pts", "sec", "seconds")):
            return 0
        score += 45
        if numeric_count > 0:
            score += 25
        if _has_any(tokens, ("width", "height", "frame")):
            score -= 20
    elif role in ("gaze_x", "gaze_y"):
        want_y = role == "gaze_y"
        if _has_any(tokens, ("gaze", "eye", "fixation", "point", "xy")):
            score += 35
        if numeric_count > 0:
            score += 25
        positive_leaf = ("y", "v", "gy", "1") if want_y else ("x", "u", "gx", "0")
        negative_leaf = ("x", "u", "gx", "0") if want_y else ("y", "v", "gy", "1")
        if leaf in positive_leaf:
            score += 45
        if leaf in negative_leaf:
            score -= 50
        if _has_any(tokens, ("width", "height", "frame", "timestamp", "time")):
            score -= 35
    elif role == "image_width":
        if _has_any(tokens, ("image", "frame", "video", "size", "resolution")):
            score += 20
        if leaf in ("width", "w", "image_width"):
            score += 55
        if numeric_count > 0:
            score += 20
        if leaf in ("height", "h", "y", "1"):
            score -= 45
    elif role == "image_height":
        if _has_any(tokens, ("image", "frame", "video", "size", "resolution")):
            score += 20
        if leaf in ("height", "h", "image_height"):
            score += 55
        if numeric_count > 0:
            score += 20
        if leaf in ("width", "w", "x", "0"):
            score -= 45
    return max(int(score), 0)


def _role_candidates(fields: Sequence[Dict[str, Any]], role: str, top_k: int) -> List[Dict[str, Any]]:
    candidates = []
    for field in fields:
        score = _score_field(field, role)
        if score <= 0:
            continue
        candidates.append(
            {
                "path": field["path"],
                "score": score,
                "present_count": field["present_count"],
                "numeric_count": field["numeric_count"],
                "video_path_like_count": field["video_path_like_count"],
                "examples": field["examples"],
            }
        )
    candidates.sort(key=lambda item: (-int(item["score"]), str(item["path"])))
    return candidates[:top_k]


def _suggest_key_map(suggestions: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, str]]:
    required = ("video_path", "gaze_x", "gaze_y")
    if any(not suggestions.get(role) for role in required):
        return None
    if not (suggestions.get("frame_idx") or suggestions.get("timestamp")):
        return None
    key_map = {
        "video_path": str(suggestions["video_path"][0]["path"]),
        "gaze_x": str(suggestions["gaze_x"][0]["path"]),
        "gaze_y": str(suggestions["gaze_y"][0]["path"]),
    }
    for role in ("episode_id", "frame_idx", "timestamp", "image_width", "image_height"):
        if suggestions.get(role):
            key_map[role] = str(suggestions[role][0]["path"])
    return key_map


def _split_filter_candidates(fields: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for field in fields:
        path = str(field["path"])
        tokens = _path_tokens(path)
        if not _has_any(tokens, ("split", "subset", "partition")):
            continue
        examples = [str(value) for value in field["examples"]]
        filters = [f"{path}={value}" for value in examples if value]
        result.append(
            {
                "path": path,
                "examples": examples,
                "filter_args": filters,
            }
        )
    return result


def _mapping_status(
    suggestions: Dict[str, List[Dict[str, Any]]],
    key_map: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    missing_required = [
        role
        for role in ("video_path", "gaze_x", "gaze_y")
        if not suggestions.get(role)
    ]
    if not (suggestions.get("frame_idx") or suggestions.get("timestamp")):
        missing_required.append("frame_idx_or_timestamp")
    optional_missing = [
        role
        for role in ("episode_id", "image_width", "image_height")
        if not suggestions.get(role)
    ]
    return {
        "ready_for_metadata_adapter": key_map is not None,
        "missing_required_roles": missing_required,
        "missing_optional_roles": optional_missing,
        "has_frame_idx": bool(suggestions.get("frame_idx")),
        "has_timestamp": bool(suggestions.get("timestamp")),
        "has_image_size": bool(suggestions.get("image_width") and suggestions.get("image_height")),
    }


def _adapter_command_template(
    metadata_path: str,
    key_map: Optional[Dict[str, str]],
    *,
    output_metadata: str,
    frames_dir: str,
    output_manifest: str,
    output_zarr: str,
    root_dir: Optional[str],
) -> Optional[str]:
    if key_map is None:
        return None
    command = [
        "py",
        "scripts/adapt_open_video_gaze_metadata.py",
        "--metadata",
        str(metadata_path),
        "--output-metadata",
        output_metadata,
        "--key-map",
        json.dumps(key_map, sort_keys=True, separators=(",", ":")),
        "--output-manifest",
        output_manifest,
        "--frames-dir",
        frames_dir,
        "--output-zarr",
        output_zarr,
        "--zarr-image-size",
        "256",
        "256",
        "--overwrite",
    ]
    if root_dir is not None:
        command.extend(["--root-dir", root_dir])
    return " ".join(shlex.quote(str(part)) for part in command)


def inspect_open_video_gaze_metadata(
    metadata_path: str,
    sample_rows: int = 200,
    top_k: int = 5,
    max_list_items: int = 8,
    output_metadata_template: str = "<canonical_open_video_gaze.csv>",
    frames_dir_template: str = "<open_frames_dir>",
    output_manifest_template: str = "<open_manifest.csv>",
    output_zarr_template: str = "<open_gaze_wam.zarr>",
    root_dir_template: Optional[str] = "<video_root_dir>",
) -> Dict[str, Any]:
    """Inspect raw open-video gaze metadata and suggest adapter key mappings."""
    source = pathlib.Path(metadata_path)
    rows = _read_rows(source)
    sampled = rows[: max(1, int(sample_rows))]
    fields = _summarize_fields(sampled, max_list_items=max_list_items)
    suggestions = {
        role: _role_candidates(fields, role=role, top_k=top_k)
        for role in CANONICAL_ROLES
    }
    key_map = _suggest_key_map(suggestions)
    filters = _split_filter_candidates(fields)
    mapping_status = _mapping_status(suggestions, key_map)
    warnings = []
    if key_map is None:
        warnings.append(
            "Could not infer a complete key_map; provide explicit mappings to "
            "adapt_open_video_gaze_metadata.py."
        )
    if not (suggestions.get("frame_idx") or suggestions.get("timestamp")):
        warnings.append("No frame_idx or timestamp candidate was found.")

    adapter_args = None
    if key_map is not None:
        adapter_args = [
            "--key-map",
            json.dumps(key_map, sort_keys=True, separators=(",", ":")),
        ]
    adapter_command_template = _adapter_command_template(
        metadata_path=str(source),
        key_map=key_map,
        output_metadata=output_metadata_template,
        frames_dir=frames_dir_template,
        output_manifest=output_manifest_template,
        output_zarr=output_zarr_template,
        root_dir=root_dir_template,
    )

    return {
        "metadata_path": str(source),
        "num_rows": int(len(rows)),
        "sampled_rows": int(len(sampled)),
        "num_fields": int(len(fields)),
        "fields": fields,
        "suggestions": suggestions,
        "suggested_key_map": key_map,
        "mapping_status": mapping_status,
        "suggested_filter_candidates": filters,
        "adapter_args": adapter_args,
        "adapter_command_template": adapter_command_template,
        "warnings": warnings,
    }


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Inspect raw open-video gaze metadata and suggest a Gaze-WAM key_map before "
            "running adapt_open_video_gaze_metadata.py or prepare_open_gaze_wam_zarr.py."
        )
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--sample-rows", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-list-items", type=int, default=8)
    parser.add_argument("--canonical-metadata", default="<canonical_open_video_gaze.csv>")
    parser.add_argument("--frames-dir", default="<open_frames_dir>")
    parser.add_argument("--output-manifest", default="<open_manifest.csv>")
    parser.add_argument("--output-zarr", default="<open_gaze_wam.zarr>")
    parser.add_argument("--root-dir", default="<video_root_dir>")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    summary = inspect_open_video_gaze_metadata(
        metadata_path=args.metadata,
        sample_rows=args.sample_rows,
        top_k=args.top_k,
        max_list_items=args.max_list_items,
        output_metadata_template=args.canonical_metadata,
        frames_dir_template=args.frames_dir,
        output_manifest_template=args.output_manifest,
        output_zarr_template=args.output_zarr,
        root_dir_template=args.root_dir,
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
