from __future__ import annotations

import argparse
import pathlib
from typing import Dict


def _ensure_debug_data_runtime():
    global gaussian_heatmaps_from_points
    global Rotation
    global mat_to_rot6d
    global np
    global zarr
    try:
        return zarr
    except NameError:
        import numpy as _np
        import zarr as _zarr
        from diffusion_policy.common.gaze_utils import (
            gaussian_heatmaps_from_points as _gaussian_heatmaps_from_points,
        )
        from scipy.spatial.transform import Rotation as _Rotation
        from diffusion_policy.common.pose_util import mat_to_rot6d as _mat_to_rot6d

        gaussian_heatmaps_from_points = _gaussian_heatmaps_from_points
        Rotation = _Rotation
        mat_to_rot6d = _mat_to_rot6d
        np = _np
        zarr = _zarr
        return zarr


def random_pose10(rng: np.random.Generator, n_steps: int) -> np.ndarray:
    _ensure_debug_data_runtime()
    pos = rng.normal(loc=0.0, scale=0.15, size=(n_steps, 3))
    rot = Rotation.random(n_steps, random_state=rng).as_matrix()
    rot6d = mat_to_rot6d(rot)
    gripper = rng.uniform(0.0, 0.08, size=(n_steps, 1))
    return np.concatenate([pos, rot6d, gripper], axis=-1).astype(np.float32)


def make_episode_ends(num_episodes: int, episode_length: int) -> np.ndarray:
    _ensure_debug_data_runtime()
    return np.arange(1, num_episodes + 1, dtype=np.int64) * int(episode_length)


def write_common_arrays(
    data,
    rng: np.random.Generator,
    n_steps: int,
    image_size: int,
) -> None:
    _ensure_debug_data_runtime()
    images = rng.integers(
        low=0,
        high=255,
        size=(n_steps, image_size, image_size, 3),
        dtype=np.uint8,
    )
    gaze_xy = rng.uniform(0.05, 0.95, size=(n_steps, 2)).astype(np.float32)
    gaze_heatmap = gaussian_heatmaps_from_points(
        gaze_xy,
        image_size=(image_size, image_size),
        sigma_px=20.0,
        window_size=1,
    )
    has_gaze_label = np.ones((n_steps,), dtype=np.bool_)
    has_heatmap_image = np.ones((n_steps,), dtype=np.bool_)
    data.array("camera0_rgb", images, shape=images.shape, dtype=images.dtype)
    data.array("gaze_xy", gaze_xy, shape=gaze_xy.shape, dtype=gaze_xy.dtype)
    data.array("gaze_heatmap", gaze_heatmap, shape=gaze_heatmap.shape, dtype=gaze_heatmap.dtype)
    data.array(
        "has_gaze_label",
        has_gaze_label,
        shape=has_gaze_label.shape,
        dtype=has_gaze_label.dtype,
    )
    data.array(
        "has_heatmap_image",
        has_heatmap_image,
        shape=has_heatmap_image.shape,
        dtype=has_heatmap_image.dtype,
    )


def write_dataset(
    path: pathlib.Path,
    include_action: bool,
    num_episodes: int,
    episode_length: int,
    image_size: int,
    seed: int,
    image_resize_mode: str = "stretch",
) -> None:
    _ensure_debug_data_runtime()
    rng = np.random.default_rng(seed)
    n_steps = num_episodes * episode_length
    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    meta.attrs["dataset_type"] = "robot" if include_action else "open"
    meta.attrs["image_resize_mode"] = str(image_resize_mode)
    meta.attrs["image_size"] = [int(image_size), int(image_size)]
    meta.attrs["gaze_is_normalized"] = True
    episode_ends = make_episode_ends(num_episodes, episode_length)
    meta.array("episode_ends", episode_ends, shape=episode_ends.shape, dtype=episode_ends.dtype)

    write_common_arrays(data, rng, n_steps=n_steps, image_size=image_size)

    if include_action:
        action_abs = random_pose10(rng, n_steps)
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


def generate_gaze_wam_debug_data(
    output_dir: str,
    num_episodes: int = 2,
    episode_length: int = 24,
    image_size: int = 256,
    image_resize_mode: str = "stretch",
    seed: int = 42,
) -> Dict[str, str]:
    _ensure_debug_data_runtime()
    output = pathlib.Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    robot_path = output / "robot.zarr"
    open_path = output / "open.zarr"
    write_dataset(
        robot_path,
        include_action=True,
        num_episodes=num_episodes,
        episode_length=episode_length,
        image_size=image_size,
        image_resize_mode=image_resize_mode,
        seed=seed,
    )
    write_dataset(
        open_path,
        include_action=False,
        num_episodes=num_episodes,
        episode_length=episode_length,
        image_size=image_size,
        image_resize_mode=image_resize_mode,
        seed=seed + 1,
    )
    return {
        "robot_path": str(robot_path),
        "open_path": str(open_path),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/debug_gaze_wam")
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--episode-length", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--image-resize-mode",
        choices=("stretch",),
        default="stretch",
        help="Image/gaze geometric contract. Only direct stretch resize is currently supported.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = generate_gaze_wam_debug_data(
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        image_size=args.image_size,
        image_resize_mode=args.image_resize_mode,
        seed=args.seed,
    )
    print(f"Wrote robot debug data to {result['robot_path']}")
    print(f"Wrote open debug data to {result['open_path']}")
    return result


if __name__ == "__main__":
    main()
