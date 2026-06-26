from __future__ import annotations

import argparse
import json
import pathlib
import shutil
from typing import Dict, Optional, Sequence, Tuple


def _ensure_canonicalizer_runtime(needs_validator: bool = False):
    global as_optional_gaze_wam_key
    global check_gaze_bounds
    global gaussian_heatmaps_from_points
    global np
    global validate_gaze_wam_zarr
    global zarr
    try:
        zarr
    except NameError:
        import numpy as _np
        import zarr as _zarr
        from diffusion_policy.common.gaze_utils import as_optional_gaze_wam_key as _as_optional_gaze_wam_key
        from diffusion_policy.common.gaze_utils import check_gaze_bounds as _check_gaze_bounds
        from diffusion_policy.common.gaze_utils import (
            gaussian_heatmaps_from_points as _gaussian_heatmaps_from_points,
        )

        as_optional_gaze_wam_key = _as_optional_gaze_wam_key
        check_gaze_bounds = _check_gaze_bounds
        gaussian_heatmaps_from_points = _gaussian_heatmaps_from_points
        np = _np
        zarr = _zarr
    if needs_validator:
        try:
            validate_gaze_wam_zarr
        except NameError:
            from diffusion_policy.scripts.validate_gaze_wam_zarr import (
                validate_gaze_wam_zarr as _validate_gaze_wam_zarr,
            )

            validate_gaze_wam_zarr = _validate_gaze_wam_zarr


OPTIONAL_PRESENCE_MASK_KEYS = (
    "has_action_abs",
    "has_action_base_abs",
    "has_heatmap_image",
    "has_gaze_label",
)


def _open_input(path: str):
    _ensure_canonicalizer_runtime()
    if str(path).endswith(".zip"):
        store = zarr.ZipStore(path, mode="r")
        return zarr.group(store=store), store
    return zarr.open(path, mode="r"), None


def _resolve_input_groups(root):
    _ensure_canonicalizer_runtime()
    if "data" in root:
        data = root["data"]
        if "meta" in root and "episode_ends" in root["meta"]:
            return data, np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
        raise ValueError("Input zarr has data/ but no meta/episode_ends.")
    if "episode_ends" in root:
        return root, np.asarray(root["episode_ends"][:], dtype=np.int64)
    raise ValueError("Input zarr must contain either data/meta/episode_ends or flat episode_ends.")


def _array_shape(array) -> Tuple[int, ...]:
    return tuple(int(v) for v in array.shape)


def _copy_array(src_group, dst_group, src_key: str, dst_key: str):
    if src_key not in src_group:
        raise KeyError(f"Input zarr missing key '{src_key}'.")
    src = src_group[src_key]
    dst_group.array(
        dst_key,
        src[:],
        shape=src.shape,
        chunks=getattr(src, "chunks", None),
        dtype=src.dtype,
    )
    return _array_shape(src)


def _copy_optional_array(src_group, dst_group, src_key: Optional[str], dst_key: str) -> Optional[Tuple[int, ...]]:
    if src_key is None:
        return None
    return _copy_array(src_group, dst_group, src_key, dst_key)


def _copy_optional_presence_masks(src_group, dst_group) -> Dict[str, Tuple[int, ...]]:
    copied = {}
    for key in OPTIONAL_PRESENCE_MASK_KEYS:
        if key not in src_group:
            continue
        copied[key] = _copy_array(src_group, dst_group, key, key)
    return copied


def _read_gripper_scalar_column(src_group, gripper_key: str, n_steps: int) -> np.ndarray:
    _ensure_canonicalizer_runtime()
    if gripper_key not in src_group:
        raise KeyError(f"Input zarr missing key '{gripper_key}'.")
    gripper = np.asarray(src_group[gripper_key][:], dtype=np.float32)
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    elif gripper.ndim == 2 and gripper.shape[-1] == 1:
        pass
    else:
        raise ValueError(
            f"{gripper_key} must provide exactly one gripper scalar per timestep as "
            f"[N] or [N,1], got shape {gripper.shape}."
        )
    if gripper.shape[0] != int(n_steps):
        raise ValueError(
            f"{gripper_key}.shape[0]={gripper.shape[0]} must match pose/action rows "
            f"{int(n_steps)}."
        )
    if not np.all(np.isfinite(gripper)):
        raise ValueError(f"{gripper_key} must contain only finite gripper scalar values.")
    return gripper


def _require_finite_numeric_array(name: str, array: np.ndarray) -> None:
    _ensure_canonicalizer_runtime()
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")


def _copy_or_compose_action_abs(
    src_group,
    dst_group,
    action_key: str,
    gripper_key: str,
    dst_key: str = "action_abs_tcp",
) -> Tuple[int, ...]:
    _ensure_canonicalizer_runtime()
    if action_key not in src_group:
        raise KeyError(f"Input zarr missing key '{action_key}'.")
    action = np.asarray(src_group[action_key][:], dtype=np.float32)
    if action.ndim != 2:
        raise ValueError(f"{action_key} must be [N,D], got {action.shape}.")
    _require_finite_numeric_array(action_key, action)
    if action.shape[-1] == 10:
        action_abs = action
    elif action.shape[-1] == 9:
        if gripper_key not in src_group:
            raise KeyError(
                f"{action_key} is 9D, so gripper key '{gripper_key}' is required to compose "
                "canonical 10D action_abs_tcp."
            )
        gripper = _read_gripper_scalar_column(src_group, gripper_key, action.shape[0])
        action_abs = np.concatenate([action, gripper], axis=-1)
    else:
        raise ValueError(f"{action_key} must be 9D or 10D, got {action.shape}.")
    dst_group.array(dst_key, action_abs, shape=action_abs.shape, dtype=action_abs.dtype)
    return _array_shape(action_abs)


def _copy_or_compose_tcp_pose(
    src_group,
    dst_group,
    tcp_key: str,
    gripper_key: str,
    keep_tcp_dim: int,
    dst_key: str = "tcp_pose_abs",
) -> Tuple[int, ...]:
    _ensure_canonicalizer_runtime()
    if keep_tcp_dim not in (9, 10):
        raise ValueError("keep_tcp_dim must be 9 or 10.")
    if tcp_key not in src_group:
        raise KeyError(f"Input zarr missing key '{tcp_key}'.")
    tcp = np.asarray(src_group[tcp_key][:], dtype=np.float32)
    if tcp.ndim != 2 or tcp.shape[-1] not in (9, 10):
        raise ValueError(f"{tcp_key} must be [N,9] or [N,10], got {tcp.shape}.")
    _require_finite_numeric_array(tcp_key, tcp)
    if keep_tcp_dim == 9:
        tcp_out = tcp[:, :9]
    elif tcp.shape[-1] == 10:
        tcp_out = tcp
    else:
        if gripper_key not in src_group:
            raise KeyError(
                f"{tcp_key} is 9D, so gripper key '{gripper_key}' is required to compose "
                "10D tcp_pose_abs."
            )
        gripper = _read_gripper_scalar_column(src_group, gripper_key, tcp.shape[0])
        tcp_out = np.concatenate([tcp, gripper], axis=-1)
    dst_group.array(dst_key, tcp_out, shape=tcp_out.shape, dtype=tcp_out.dtype)
    return _array_shape(tcp_out)


def _infer_image_hw(image_array) -> Tuple[int, int]:
    if image_array.ndim != 4:
        raise ValueError(f"Expected image [N,H,W,C] or [N,C,H,W], got {image_array.shape}.")
    if image_array.shape[-1] in (1, 3, 4):
        return int(image_array.shape[1]), int(image_array.shape[2])
    if image_array.shape[1] in (1, 3, 4):
        return int(image_array.shape[2]), int(image_array.shape[3])
    raise ValueError(f"Cannot infer image height/width from shape {image_array.shape}.")


def _normalize_gaze(
    gaze: np.ndarray,
    image_hw: Tuple[int, int],
    gaze_is_normalized: bool,
    gaze_bounds_policy: str = "error",
) -> np.ndarray:
    _ensure_canonicalizer_runtime()
    gaze = np.asarray(gaze, dtype=np.float32)
    if gaze.ndim != 2 or gaze.shape[-1] != 2:
        raise ValueError(f"gaze_xy must be [N,2], got {gaze.shape}.")
    if not gaze_is_normalized:
        image_h, image_w = image_hw
        gaze = gaze / np.asarray([image_w, image_h], dtype=np.float32)
    checked = []
    for idx, point in enumerate(gaze):
        normalized = check_gaze_bounds(point, policy=gaze_bounds_policy, row_idx=idx)
        if normalized is None:
            raise ValueError("Robot canonicalization does not support dropping gaze rows.")
        checked.append(normalized)
    return np.stack(checked, axis=0).astype(np.float32)


def _write_episode_ends(meta_group, episode_ends: np.ndarray) -> None:
    _ensure_canonicalizer_runtime()
    episode_ends = np.asarray(episode_ends, dtype=np.int64)
    meta_group.array(
        "episode_ends",
        episode_ends,
        shape=episode_ends.shape,
        dtype=episode_ends.dtype,
    )


def canonicalize_robot_gaze_wam_zarr(
    input_path: str,
    output_path: str,
    camera_key: str = "camera0_rgb",
    action_key: str = "action_abs_tcp",
    tcp_pose_key: str = "tcp_pose_abs",
    gripper_key: str = "gripper_width",
    gaze_key: Optional[str] = "gaze_xy",
    heatmap_key: Optional[str] = None,
    output_camera_key: str = "camera0_rgb",
    output_action_key: str = "action_abs_tcp",
    output_tcp_pose_key: str = "tcp_pose_abs",
    output_gripper_key: str = "gripper_width",
    output_gaze_key: str = "gaze_xy",
    output_heatmap_key: str = "gaze_heatmap",
    timestamp_key: Optional[str] = None,
    output_timestamp_key: str = "timestamp",
    image_timestamp_key: Optional[str] = None,
    robot_state_timestamp_key: Optional[str] = None,
    action_timestamp_key: Optional[str] = None,
    gaze_timestamp_key: Optional[str] = None,
    output_image_timestamp_key: str = "image_timestamp",
    output_robot_state_timestamp_key: str = "robot_state_timestamp",
    output_action_timestamp_key: str = "action_timestamp",
    output_gaze_timestamp_key: str = "gaze_timestamp",
    gaze_is_normalized: bool = True,
    gaze_bounds_policy: str = "error",
    image_size_for_pixel_gaze: Optional[Sequence[int]] = None,
    point_heatmap_sigma_px: float = 20.0,
    point_heatmap_window: int = 1,
    keep_tcp_dim: int = 9,
    overwrite: bool = False,
    validate_output: bool = True,
) -> Dict[str, object]:
    """Map a robot zarr into the canonical Gaze-WAM robot schema."""
    _ensure_canonicalizer_runtime(needs_validator=validate_output)
    gaze_key = as_optional_gaze_wam_key(gaze_key)
    heatmap_key = as_optional_gaze_wam_key(heatmap_key)
    timestamp_key = as_optional_gaze_wam_key(timestamp_key)
    image_timestamp_key = as_optional_gaze_wam_key(image_timestamp_key)
    robot_state_timestamp_key = as_optional_gaze_wam_key(robot_state_timestamp_key)
    action_timestamp_key = as_optional_gaze_wam_key(action_timestamp_key)
    gaze_timestamp_key = as_optional_gaze_wam_key(gaze_timestamp_key)

    output = pathlib.Path(output_path)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output '{output}' already exists. Pass overwrite=True.")
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()

    input_root, input_store = _open_input(input_path)
    try:
        input_data, episode_ends = _resolve_input_groups(input_root)

        store = zarr.DirectoryStore(str(output))
        root = zarr.group(store=store, overwrite=True)
        data = root.create_group("data")
        meta = root.create_group("meta")

        shapes = {}
        shapes[output_camera_key] = _copy_array(input_data, data, camera_key, output_camera_key)
        image_array = data[output_camera_key]
        output_image_hw = _infer_image_hw(image_array)
        gaze_image_hw = (
            (int(image_size_for_pixel_gaze[0]), int(image_size_for_pixel_gaze[1]))
            if image_size_for_pixel_gaze is not None
            else output_image_hw
        )
        _read_gripper_scalar_column(input_data, gripper_key, int(image_array.shape[0]))
        shapes[output_gripper_key] = _copy_array(input_data, data, gripper_key, output_gripper_key)
        shapes[output_action_key] = _copy_or_compose_action_abs(
            input_data,
            data,
            action_key=action_key,
            gripper_key=gripper_key,
            dst_key=output_action_key,
        )
        shapes[output_tcp_pose_key] = _copy_or_compose_tcp_pose(
            input_data,
            data,
            tcp_key=tcp_pose_key,
            gripper_key=gripper_key,
            keep_tcp_dim=keep_tcp_dim,
            dst_key=output_tcp_pose_key,
        )
        copied_gaze_key = None
        copied_heatmap_key = None
        has_gaze = gaze_key is not None and gaze_key in input_data
        has_heatmap = heatmap_key is not None and heatmap_key in input_data
        if not has_gaze:
            raise KeyError(
                "Input robot zarr must contain a point gaze key so canonical zarr can write "
                f"normalized {output_gaze_key!r}. Requested gaze_key={gaze_key!r}."
            )
        gaze = _normalize_gaze(
            input_data[gaze_key][:],
            image_hw=gaze_image_hw,
            gaze_is_normalized=gaze_is_normalized,
            gaze_bounds_policy=gaze_bounds_policy,
        )
        data.array(output_gaze_key, gaze, shape=gaze.shape, dtype=gaze.dtype)
        shapes[output_gaze_key] = _array_shape(gaze)
        copied_gaze_key = output_gaze_key
        if has_heatmap:
            copied_shape = _copy_optional_array(input_data, data, heatmap_key, output_heatmap_key)
            if copied_shape is not None:
                shapes[output_heatmap_key] = copied_shape
                copied_heatmap_key = output_heatmap_key
        else:
            generated_heatmap = gaussian_heatmaps_from_points(
                gaze,
                image_size=output_image_hw,
                sigma_px=point_heatmap_sigma_px,
                window_size=point_heatmap_window,
                episode_ends=episode_ends,
            )
            data.array(
                output_heatmap_key,
                generated_heatmap,
                shape=generated_heatmap.shape,
                dtype=generated_heatmap.dtype,
            )
            shapes[output_heatmap_key] = _array_shape(generated_heatmap)
            copied_heatmap_key = output_heatmap_key
        if timestamp_key is not None:
            shapes[output_timestamp_key] = _copy_array(
                input_data,
                data,
                timestamp_key,
                output_timestamp_key,
            )
        timestamp_streams = {
            "image_timestamp": (image_timestamp_key, output_image_timestamp_key),
            "robot_state_timestamp": (robot_state_timestamp_key, output_robot_state_timestamp_key),
            "action_timestamp": (action_timestamp_key, output_action_timestamp_key),
            "gaze_timestamp": (gaze_timestamp_key, output_gaze_timestamp_key),
        }
        copied_timestamp_streams = {}
        for role, (src_key, dst_key) in timestamp_streams.items():
            if src_key is None:
                continue
            shapes[dst_key] = _copy_array(input_data, data, src_key, dst_key)
            copied_timestamp_streams[role] = {
                "source_key": src_key,
                "output_key": dst_key,
            }
        copied_presence_masks = _copy_optional_presence_masks(input_data, data)
        n_steps = int(image_array.shape[0])
        for mask_key in ("has_gaze_label", "has_heatmap_image"):
            if mask_key not in data:
                mask = np.ones((n_steps,), dtype=np.bool_)
                data.array(mask_key, mask, shape=mask.shape, dtype=mask.dtype)
                copied_presence_masks[mask_key] = _array_shape(mask)
        shapes.update(copied_presence_masks)

        _write_episode_ends(meta, episode_ends)
        meta.attrs["source_zarr"] = str(input_path)
        meta.attrs["dataset_type"] = "robot"
        meta.attrs["canonical_schema"] = "gaze_wam_robot_v1"
        meta.attrs["gaze_is_normalized"] = True
        meta.attrs["gaze_bounds_policy"] = gaze_bounds_policy
        meta.attrs["heatmap_source"] = "input" if has_heatmap else "generated_from_gaze_xy"
        meta.attrs["point_heatmap_sigma_px"] = float(point_heatmap_sigma_px)
        meta.attrs["point_heatmap_window"] = int(point_heatmap_window)
        meta.attrs["image_size"] = [int(output_image_hw[0]), int(output_image_hw[1])]
        meta.attrs["image_resize_mode"] = "stretch"
        meta.attrs["source_key_map"] = {
            "camera": camera_key,
            "action": action_key,
            "tcp_pose": tcp_pose_key,
            "gripper": gripper_key,
            "gaze": gaze_key,
            "heatmap": heatmap_key,
            "timestamp": timestamp_key,
            "image_timestamp": image_timestamp_key,
            "robot_state_timestamp": robot_state_timestamp_key,
            "action_timestamp": action_timestamp_key,
            "gaze_timestamp": gaze_timestamp_key,
        }
        meta.attrs["timestamp_key"] = output_timestamp_key if timestamp_key is not None else None
        meta.attrs["timestamp_stream_keys"] = copied_timestamp_streams
        meta.attrs["presence_mask_keys"] = sorted(copied_presence_masks.keys())

        summary: Dict[str, object] = {
            "input_path": str(input_path),
            "output_path": str(output),
            "num_steps": int(data[output_camera_key].shape[0]),
            "num_episodes": int(len(episode_ends)),
            "episode_ends": [int(v) for v in episode_ends.tolist()],
            "keys": sorted(list(data.keys())),
            "shapes": {key: list(value) for key, value in shapes.items()},
            "dataset_type": "robot",
            "gaze_is_normalized": True,
            "gaze_bounds_policy": gaze_bounds_policy,
            "heatmap_source": "input" if has_heatmap else "generated_from_gaze_xy",
            "point_heatmap_sigma_px": float(point_heatmap_sigma_px),
            "point_heatmap_window": int(point_heatmap_window),
            "image_size": [int(output_image_hw[0]), int(output_image_hw[1])],
            "image_resize_mode": "stretch",
            "output_gaze_key": copied_gaze_key,
            "output_heatmap_key": copied_heatmap_key,
            "timestamp_key": output_timestamp_key if timestamp_key is not None else None,
            "timestamp_stream_keys": copied_timestamp_streams,
            "presence_mask_keys": sorted(copied_presence_masks.keys()),
            "validated": False,
        }
        if validate_output:
            validation = validate_gaze_wam_zarr(
                dataset_path=str(output),
                dataset_type="robot",
                camera_key=output_camera_key,
                gaze_key=output_gaze_key,
                heatmap_key=output_heatmap_key,
                action_abs_key=output_action_key,
                tcp_pose_key=output_tcp_pose_key,
                gripper_key=output_gripper_key,
                image_size=output_image_hw,
                image_resize_mode="stretch",
                heatmap_dim=int(output_image_hw[0] // 16) * int(output_image_hw[1] // 16),
                timestamp_key=output_timestamp_key if timestamp_key is not None else None,
                image_timestamp_key=(
                    output_image_timestamp_key if image_timestamp_key is not None else None
                ),
                robot_state_timestamp_key=(
                    output_robot_state_timestamp_key if robot_state_timestamp_key is not None else None
                ),
                action_timestamp_key=(
                    output_action_timestamp_key if action_timestamp_key is not None else None
                ),
                gaze_timestamp_key=(
                    output_gaze_timestamp_key if gaze_timestamp_key is not None else None
                ),
                check_dataset_sample=True,
            )
            summary["validation"] = validation
            summary["validated"] = bool(validation["valid"])
            if not validation["valid"]:
                raise ValueError(f"Canonicalized zarr failed validation: {validation['errors']}")
        return summary
    finally:
        if input_store is not None:
            input_store.close()


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Map a robot zarr into the canonical Gaze-WAM robot schema."
    )
    parser.add_argument("--input", required=True, help="Input robot zarr directory or zip.")
    parser.add_argument("--output", required=True, help="Output canonical zarr directory.")
    parser.add_argument("--camera-key", default="camera0_rgb")
    parser.add_argument("--action-key", default="action_abs_tcp")
    parser.add_argument("--tcp-pose-key", default="tcp_pose_abs")
    parser.add_argument("--gripper-key", default="gripper_width")
    parser.add_argument("--gaze-key", default="gaze_xy")
    parser.add_argument("--heatmap-key", default=None)
    parser.add_argument("--output-heatmap-key", default="gaze_heatmap")
    parser.add_argument("--timestamp-key", default=None)
    parser.add_argument("--output-timestamp-key", default="timestamp")
    parser.add_argument("--image-timestamp-key", default=None)
    parser.add_argument("--robot-state-timestamp-key", default=None)
    parser.add_argument("--action-timestamp-key", default=None)
    parser.add_argument("--gaze-timestamp-key", default=None)
    parser.add_argument("--output-image-timestamp-key", default="image_timestamp")
    parser.add_argument("--output-robot-state-timestamp-key", default="robot_state_timestamp")
    parser.add_argument("--output-action-timestamp-key", default="action_timestamp")
    parser.add_argument("--output-gaze-timestamp-key", default="gaze_timestamp")
    parser.add_argument("--gaze-is-normalized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gaze-bounds-policy",
        choices=("error", "clip"),
        default="error",
        help="How to handle gaze outside [0,1] after normalization.",
    )
    parser.add_argument(
        "--image-size-for-pixel-gaze",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Source image size used to normalize pixel gaze labels when not inferable.",
    )
    parser.add_argument("--point-heatmap-sigma-px", type=float, default=20.0)
    parser.add_argument(
        "--point-heatmap-window",
        type=int,
        default=1,
        help="Trailing in-episode frame window used when generating dense heatmaps from gaze_xy.",
    )
    parser.add_argument("--keep-tcp-dim", type=int, choices=(9, 10), default=9)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    summary = canonicalize_robot_gaze_wam_zarr(
        input_path=args.input,
        output_path=args.output,
        camera_key=args.camera_key,
        action_key=args.action_key,
        tcp_pose_key=args.tcp_pose_key,
        gripper_key=args.gripper_key,
        gaze_key=args.gaze_key,
        heatmap_key=args.heatmap_key,
        output_heatmap_key=args.output_heatmap_key,
        timestamp_key=args.timestamp_key,
        output_timestamp_key=args.output_timestamp_key,
        image_timestamp_key=args.image_timestamp_key,
        robot_state_timestamp_key=args.robot_state_timestamp_key,
        action_timestamp_key=args.action_timestamp_key,
        gaze_timestamp_key=args.gaze_timestamp_key,
        output_image_timestamp_key=args.output_image_timestamp_key,
        output_robot_state_timestamp_key=args.output_robot_state_timestamp_key,
        output_action_timestamp_key=args.output_action_timestamp_key,
        output_gaze_timestamp_key=args.output_gaze_timestamp_key,
        gaze_is_normalized=args.gaze_is_normalized,
        gaze_bounds_policy=args.gaze_bounds_policy,
        image_size_for_pixel_gaze=args.image_size_for_pixel_gaze,
        point_heatmap_sigma_px=args.point_heatmap_sigma_px,
        point_heatmap_window=args.point_heatmap_window,
        keep_tcp_dim=args.keep_tcp_dim,
        overwrite=args.overwrite,
        validate_output=not args.skip_validation,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
