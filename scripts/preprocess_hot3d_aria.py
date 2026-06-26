import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm


REQUIRED_DATA_TYPES = ("main_vrs", "mps_eye_gaze", "mps_slam_calibration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and preprocess HOT3D Aria RGB+gaze sequences into compact packages."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--hot3d-repo", required=True, type=Path)
    parser.add_argument(
        "--work-root",
        "--out-root",
        dest="work_root",
        required=True,
        type=Path,
        help="Processing output root, preferably on a fast local/WSL filesystem.",
    )
    parser.add_argument(
        "--final-root",
        type=Path,
        default=None,
        help="Optional final copy root, for example a Windows drive mounted in WSL.",
    )
    parser.add_argument("--temp-root", required=True, type=Path)
    parser.add_argument(
        "--download-cache-root",
        type=Path,
        default=None,
        help="Optional per-sequence download cache root. If omitted, downloads stay only in temp.",
    )
    parser.add_argument("--sequence", "--sequence-id", dest="sequence", nargs="*", default=None)
    parser.add_argument("--sequences", "--sequence-ids", dest="sequences", nargs="*", default=None)
    parser.add_argument("--sequence-file", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="Process all manifest sequences. Default when no sequence is given.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--continue-on-error", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stream-id", default="214-1")
    parser.add_argument(
        "--rotate",
        choices=("cw90", "ccw90", "rot180", "none"),
        default="cw90",
        help="Final video orientation. cw90 matches HOT3D preview_rgb.mp4 for Aria RGB.",
    )
    parser.add_argument("--overlay-scale", type=float, default=0.5)
    parser.add_argument("--trail", type=int, default=12)
    parser.add_argument("--radius", type=int, default=13)
    parser.add_argument("--raw-crf", type=int, default=24)
    parser.add_argument("--overlay-crf", type=int, default=25)
    parser.add_argument("--ffmpeg-preset", default="veryfast")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument(
        "--worker-sequence",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--download-cache-dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def format_bytes(num: int) -> str:
    value = float(num)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024 or unit == "GiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{value:.2f} GiB"


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, expected_size: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if target.exists() and target.stat().st_size == expected_size:
        print(f"[skip] {target.name} already complete")
        return

    print(f"[download] {target.name} ({format_bytes(expected_size)})")
    start = time.time()
    last_report = 0.0
    attempts = 0

    while True:
        current_size = partial.stat().st_size if partial.exists() else 0
        if current_size == expected_size:
            break
        if current_size > expected_size:
            partial.unlink()
            current_size = 0

        attempts += 1
        if attempts > 300:
            raise RuntimeError(f"Too many interrupted attempts for {target.name}")

        headers = {"User-Agent": "hot3d-sequence-pipeline/1.0"}
        if current_size:
            headers["Range"] = f"bytes={current_size}-"

        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                status = getattr(response, "status", None)
                mode = "ab" if current_size and status == 206 else "wb"
                if current_size and status != 206:
                    print("  server ignored Range header; restarting partial file")
                    current_size = 0
                with partial.open(mode) as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        current_size += len(chunk)
                        now = time.time()
                        if now - last_report >= 5:
                            pct = 100 * current_size / expected_size if expected_size else 0
                            speed = current_size / max(now - start, 1)
                            print(
                                f"  {pct:5.1f}% {format_bytes(current_size)} at {format_bytes(speed)}/s",
                                flush=True,
                            )
                            last_report = now
        except Exception as exc:
            print(f"  interrupted attempt {attempts}: {exc}; retrying", flush=True)
            time.sleep(min(30, 2 + attempts // 5))
            continue

        actual_size = partial.stat().st_size if partial.exists() else 0
        if actual_size < expected_size:
            print(f"  connection ended at {format_bytes(actual_size)}; resuming", flush=True)
            time.sleep(min(10, 1 + attempts // 20))
            continue
        if actual_size > expected_size:
            partial.unlink()
            raise RuntimeError(
                f"Downloaded too many bytes for {target.name}: expected {expected_size}, got {actual_size}"
            )

    actual_size = partial.stat().st_size
    if expected_size and actual_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {target.name}: expected {expected_size}, got {actual_size}"
        )
    partial.replace(target)
    print(f"[done] {target.name}")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialize_file(item: dict, target: Path, cache_dir: Path | None = None) -> None:
    expected_size = int(item["file_size_bytes"])
    filename = item["filename"]
    cached = cache_dir / filename if cache_dir is not None else None
    if cached is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not cached.exists() or cached.stat().st_size != expected_size:
            download(item["download_url"], cached, expected_size)
        print(f"[cache] {filename} -> {target}")
        link_or_copy(cached, target)
    else:
        download(item["download_url"], target, expected_size)

    if target.stat().st_size != expected_size:
        raise RuntimeError(f"Size mismatch for {filename}")
    expected_sha1 = item.get("sha1sum")
    if expected_sha1:
        actual_sha1 = sha1_file(target)
        if actual_sha1.lower() != expected_sha1.lower():
            raise RuntimeError(
                f"SHA1 mismatch for {filename}: expected {expected_sha1}, got {actual_sha1}"
            )


def extract_zip_flat(zip_path: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if member.endswith("/") or "__MACOSX" in Path(member).parts:
                continue
            out_path = destination / Path(member).name
            with archive.open(member) as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            copied.append(out_path.name)
    return sorted(copied)


def rotate_image(image: np.ndarray, rotate: str) -> np.ndarray:
    if rotate == "cw90":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotate == "ccw90":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate == "rot180":
        return cv2.rotate(image, cv2.ROTATE_180)
    return image


def rotate_point(x: float, y: float, raw_w: int, raw_h: int, rotate: str) -> tuple[float, float, int, int]:
    if rotate == "cw90":
        return raw_h - 1 - y, x, raw_h, raw_w
    if rotate == "ccw90":
        return y, raw_w - 1 - x, raw_h, raw_w
    if rotate == "rot180":
        return raw_w - 1 - x, raw_h - 1 - y, raw_w, raw_h
    return x, y, raw_w, raw_h


def even_dimension(value: int) -> int:
    return max(2, value - (value % 2))


def resize_for_scale(image: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return image
    height, width = image.shape[:2]
    out_w = even_dimension(int(round(width * scale)))
    out_h = even_dimension(int(round(height * scale)))
    return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)


class FFmpegVideoWriter:
    def __init__(self, output: Path, width: int, height: int, fps: float, crf: int, preset: str) -> None:
        self.output = output
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame_rgb: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin closed")
        frame = np.ascontiguousarray(frame_rgb)
        self.process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed for {self.output} with code {return_code}")


def draw_gaze(frame_rgb: np.ndarray, gaze_xy, trail: deque, radius: int) -> np.ndarray:
    frame = np.ascontiguousarray(frame_rgb.copy())
    height, width = frame.shape[:2]
    if gaze_xy is not None:
        x, y = float(gaze_xy[0]), float(gaze_xy[1])
        if np.isfinite(x) and np.isfinite(y) and -width < x < 2 * width and -height < y < 2 * height:
            trail.append((int(round(x)), int(round(y))))

    for idx, (x, y) in enumerate(trail):
        alpha = (idx + 1) / max(len(trail), 1)
        color = (int(255 * alpha), int(80 + 120 * alpha), 0)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), max(2, int(radius * alpha * 0.75)), color, -1)

    if trail:
        x, y = trail[-1]
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), radius + 3, (255, 255, 255), 2)
            cv2.circle(frame, (x, y), radius, (255, 40, 0), -1)
            cv2.circle(frame, (x, y), max(2, radius // 3), (255, 255, 255), -1)
    return frame


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def assert_child_path(child: Path, parent: Path) -> None:
    child_resolved = child.resolve()
    parent_resolved = parent.resolve()
    if parent_resolved not in child_resolved.parents and child_resolved != parent_resolved:
        raise RuntimeError(f"Refusing to delete {child_resolved}; it is not under {parent_resolved}")


def copy_tree_contents(src: Path, dst: Path) -> list[str]:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for item in sorted(src.iterdir()):
        if item.is_file():
            shutil.copy2(item, dst / item.name)
            copied.append(item.name)
    return copied


def get_eye_gaze_source(provider, gaze_dir: Path) -> str:
    try:
        if provider._mps_data_provider.has_personalized_eyegaze():
            return "personalized"
        if provider._mps_data_provider.has_general_eyegaze():
            return "general"
    except Exception:
        pass
    return "personalized" if (gaze_dir / "personalized_eye_gaze.csv").exists() else "general"


def tracking_timestamp_to_us(value) -> str | int | float:
    if value is None:
        return ""
    total_seconds = getattr(value, "total_seconds", None)
    if callable(total_seconds):
        return int(round(total_seconds() * 1_000_000))
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def projection_row(
    sequence: str,
    frame_index: int,
    timestamp_ns: int,
    raw_w: int,
    raw_h: int,
    rotate: str,
    gaze_xy,
    eye_gaze,
    source: str,
) -> dict:
    row = {
        "sequence": sequence,
        "frame_index": frame_index,
        "timecode_timestamp_ns": timestamp_ns,
        "raw_width": raw_w,
        "raw_height": raw_h,
        "gaze_source": source,
        "gaze_available": False,
        "raw_x_px": "",
        "raw_y_px": "",
        "raw_x_norm": "",
        "raw_y_norm": "",
        "in_raw_bounds": False,
        "upright_x_px": "",
        "upright_y_px": "",
        "upright_x_norm": "",
        "upright_y_norm": "",
        "upright_width": "",
        "upright_height": "",
        "tracking_timestamp_us": "",
        "yaw_rad": "",
        "pitch_rad": "",
        "depth_m": "",
        "yaw_low_rad": "",
        "yaw_high_rad": "",
        "pitch_low_rad": "",
        "pitch_high_rad": "",
        "session_uid": "",
    }
    if gaze_xy is None:
        return row

    x = float(gaze_xy[0])
    y = float(gaze_xy[1])
    upright_x, upright_y, upright_w, upright_h = rotate_point(x, y, raw_w, raw_h, rotate)
    row.update(
        {
            "gaze_available": True,
            "raw_x_px": x,
            "raw_y_px": y,
            "raw_x_norm": x / raw_w,
            "raw_y_norm": y / raw_h,
            "in_raw_bounds": 0 <= x < raw_w and 0 <= y < raw_h,
            "upright_x_px": upright_x,
            "upright_y_px": upright_y,
            "upright_x_norm": upright_x / upright_w,
            "upright_y_norm": upright_y / upright_h,
            "upright_width": upright_w,
            "upright_height": upright_h,
        }
    )
    if eye_gaze is not None:
        row.update(
            {
                "tracking_timestamp_us": tracking_timestamp_to_us(
                    getattr(eye_gaze, "tracking_timestamp", None)
                ),
                "yaw_rad": getattr(eye_gaze, "yaw", ""),
                "pitch_rad": getattr(eye_gaze, "pitch", ""),
                "depth_m": getattr(eye_gaze, "depth", ""),
                "yaw_low_rad": getattr(eye_gaze, "yaw_low", ""),
                "yaw_high_rad": getattr(eye_gaze, "yaw_high", ""),
                "pitch_low_rad": getattr(eye_gaze, "pitch_low", ""),
                "pitch_high_rad": getattr(eye_gaze, "pitch_high", ""),
                "session_uid": getattr(eye_gaze, "session_uid", ""),
            }
        )
    return row


def write_csv(path: Path, rows: Iterable[dict]) -> int:
    rows = list(rows)
    if not rows:
        raise RuntimeError("No projection rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_temp_sequence(args: argparse.Namespace, sequence_data: dict, temp_sequence_dir: Path) -> dict:
    sequence_dir = temp_sequence_dir / "sequence"
    downloads_dir = temp_sequence_dir / "downloads"
    gaze_dir = sequence_dir / "mps" / "eye_gaze"
    slam_dir = sequence_dir / "mps" / "slam"
    sequence_dir.mkdir(parents=True, exist_ok=True)

    main_vrs = sequence_data["main_vrs"]
    materialize_file(
        main_vrs,
        sequence_dir / "recording.vrs",
        args.download_cache_dir,
    )

    gaze_zip = downloads_dir / sequence_data["mps_eye_gaze"]["filename"]
    materialize_file(sequence_data["mps_eye_gaze"], gaze_zip, args.download_cache_dir)
    gaze_files = extract_zip_flat(gaze_zip, gaze_dir)

    slam_zip = downloads_dir / sequence_data["mps_slam_calibration"]["filename"]
    materialize_file(sequence_data["mps_slam_calibration"], slam_zip, args.download_cache_dir)
    slam_files = extract_zip_flat(slam_zip, slam_dir)

    return {
        "sequence_dir": sequence_dir,
        "gaze_dir": gaze_dir,
        "slam_dir": slam_dir,
        "gaze_files": gaze_files,
        "slam_files": slam_files,
    }


def process_sequence(args: argparse.Namespace, temp_info: dict, output_dir: Path) -> dict:
    sys.path.insert(0, str(args.hot3d_repo / "hot3d"))
    from data_loaders.AriaDataProvider import AriaDataProvider
    from projectaria_tools.core.calibration import FISHEYE624
    from projectaria_tools.core.stream_id import StreamId

    sequence_dir = temp_info["sequence_dir"]
    gaze_dir = temp_info["gaze_dir"]
    provider = AriaDataProvider(str(sequence_dir / "recording.vrs"), str(sequence_dir / "mps"))
    stream_id = StreamId(args.stream_id)
    timestamps = provider.get_sequence_timestamps(stream_id)
    if not timestamps:
        raise RuntimeError(f"No timestamps found for stream {args.stream_id}")

    first_raw = provider.get_image(timestamps[0], stream_id)
    if first_raw is None:
        raise RuntimeError("Could not read first RGB frame")
    raw_h, raw_w = first_raw.shape[:2]
    first_upright = rotate_image(first_raw, args.rotate)
    upright_h, upright_w = first_upright.shape[:2]
    overlay_first = resize_for_scale(first_upright, args.overlay_scale)
    overlay_h, overlay_w = overlay_first.shape[:2]
    duration_s = (timestamps[-1] - timestamps[0]) / 1e9 if len(timestamps) > 1 else 1.0
    fps = (len(timestamps) - 1) / duration_s if duration_s > 0 else 30.0

    raw_writer = FFmpegVideoWriter(
        output_dir / "raw_rgb.mp4",
        upright_w,
        upright_h,
        fps,
        args.raw_crf,
        args.ffmpeg_preset,
    )
    overlay_writer = FFmpegVideoWriter(
        output_dir / "overlay_preview_half.mp4",
        overlay_w,
        overlay_h,
        fps,
        args.overlay_crf,
        args.ffmpeg_preset,
    )

    gaze_source = get_eye_gaze_source(provider, gaze_dir)
    rows = []
    trail = deque(maxlen=max(args.trail, 0))
    valid_gaze = 0
    visible_gaze = 0

    try:
        for frame_index, timestamp_ns in enumerate(tqdm(timestamps, desc=f"process {args.sequence}")):
            frame_raw = provider.get_image(timestamp_ns, stream_id)
            if frame_raw is None:
                continue
            gaze_xy = provider.get_eye_gaze_in_camera(
                stream_id,
                timestamp_ns,
                camera_model=FISHEYE624,
            )
            eye_gaze = provider.get_eye_gaze(timestamp_ns)

            row = projection_row(
                args.sequence,
                frame_index,
                timestamp_ns,
                raw_w,
                raw_h,
                args.rotate,
                gaze_xy,
                eye_gaze,
                gaze_source,
            )
            rows.append(row)
            if row["gaze_available"]:
                valid_gaze += 1
            if row["in_raw_bounds"]:
                visible_gaze += 1

            raw_video_frame = rotate_image(frame_raw, args.rotate)
            raw_writer.write(raw_video_frame)

            overlay_frame = draw_gaze(frame_raw, gaze_xy, trail, args.radius)
            overlay_frame = rotate_image(overlay_frame, args.rotate)
            overlay_frame = resize_for_scale(overlay_frame, args.overlay_scale)
            overlay_writer.write(overlay_frame)
    finally:
        raw_writer.close()
        overlay_writer.close()

    projection_rows = write_csv(output_dir / "gaze_projected_raw_rgb_normalized.csv", rows)
    raw_gaze_files = copy_tree_contents(gaze_dir, output_dir / "gaze_raw")

    return {
        "frames": len(timestamps),
        "projection_rows": projection_rows,
        "fps": fps,
        "duration_s": duration_s,
        "raw_width": raw_w,
        "raw_height": raw_h,
        "upright_width": upright_w,
        "upright_height": upright_h,
        "overlay_width": overlay_w,
        "overlay_height": overlay_h,
        "valid_gaze_frames": valid_gaze,
        "visible_gaze_frames": visible_gaze,
        "gaze_source": gaze_source,
        "raw_gaze_files": raw_gaze_files,
    }


def file_size_map(output_dir: Path) -> dict[str, int]:
    return {
        str(path.relative_to(output_dir)).replace("\\", "/"): path.stat().st_size
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if "sequences" not in manifest or not isinstance(manifest["sequences"], dict):
        raise RuntimeError(f"Manifest does not contain a sequences object: {path}")
    return manifest


def read_sequence_file(path: Path) -> list[str]:
    sequences = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            sequences.append(value)
    return sequences


def resolve_sequences(args: argparse.Namespace, manifest: dict) -> list[str]:
    available = list(manifest["sequences"].keys())
    requested = []
    if args.sequence:
        requested.extend(args.sequence)
    if args.sequences:
        requested.extend(args.sequences)
    if args.sequence_file:
        requested.extend(read_sequence_file(args.sequence_file))
    if args.all or not requested:
        return available

    unique_requested = list(dict.fromkeys(requested))
    missing = [sequence for sequence in unique_requested if sequence not in manifest["sequences"]]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise RuntimeError(f"{len(missing)} requested sequence ids are not in the manifest: {preview}{suffix}")
    return unique_requested


def summary_is_done(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    required_outputs = (
        "raw_rgb.mp4",
        "overlay_preview_half.mp4",
        "gaze_projected_raw_rgb_normalized.csv",
        "processing_summary.json",
    )
    output_dir = summary_path.parent
    if not all((output_dir / name).exists() for name in required_outputs):
        return False
    return bool(summary.get("temp_deleted") or summary.get("keep_temp"))


def sequence_done(root: Path | None, sequence: str) -> bool:
    if root is None:
        return False
    return summary_is_done(root / sequence / "processing_summary.json")


def same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def sync_sequence_output(sequence: str, work_root: Path, final_root: Path | None, overwrite: bool) -> bool:
    if final_root is None or same_path(work_root, final_root):
        return True

    src = work_root / sequence
    if not summary_is_done(src / "processing_summary.json"):
        return False

    final_root.mkdir(parents=True, exist_ok=True)
    dst = final_root / sequence
    if dst.exists():
        if summary_is_done(dst / "processing_summary.json"):
            return True
        if not overwrite:
            return False
        assert_child_path(dst, final_root)
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    return True


def sync_finished_outputs(sequences: list[str], work_root: Path, final_root: Path | None, overwrite: bool) -> int:
    if final_root is None or same_path(work_root, final_root):
        return sum(1 for sequence in sequences if sequence_done(work_root, sequence))

    copied = 0
    for sequence in sequences:
        if sync_sequence_output(sequence, work_root, final_root, overwrite):
            copied += 1
    return copied


def build_worker_command(args: argparse.Namespace, sequence: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--manifest",
        str(args.manifest),
        "--hot3d-repo",
        str(args.hot3d_repo),
        "--work-root",
        str(args.work_root),
        "--temp-root",
        str(args.temp_root),
        "--worker-sequence",
        sequence,
        "--stream-id",
        args.stream_id,
        "--rotate",
        args.rotate,
        "--overlay-scale",
        str(args.overlay_scale),
        "--trail",
        str(args.trail),
        "--radius",
        str(args.radius),
        "--raw-crf",
        str(args.raw_crf),
        "--overlay-crf",
        str(args.overlay_crf),
        "--ffmpeg-preset",
        args.ffmpeg_preset,
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.keep_temp:
        command.append("--keep-temp")
    if args.download_cache_root is not None:
        command.extend(["--download-cache-dir", str(args.download_cache_root / sequence)])
    return command


def run_worker_subprocess(args: argparse.Namespace, sequence: str, log_root: Path) -> dict:
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{sequence}.log"
    command = build_worker_command(args, sequence)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[started] {timestamp()}\n")
        log.write("[command] " + " ".join(command) + "\n\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\n[finished] {timestamp()} return_code={result.returncode}\n")

    return {
        "sequence": sequence,
        "return_code": result.returncode,
        "seconds": time.time() - start,
        "log_path": log_path,
    }


def run_worker(args: argparse.Namespace) -> int:
    if not args.worker_sequence:
        raise RuntimeError("--worker-sequence is required in worker mode")
    args.sequence = args.worker_sequence

    manifest = load_manifest(args.manifest)
    if args.sequence not in manifest["sequences"]:
        raise KeyError(f"Sequence {args.sequence} not found in manifest")
    sequence_data = manifest["sequences"][args.sequence]
    for data_type in REQUIRED_DATA_TYPES:
        if data_type not in sequence_data:
            raise KeyError(f"{args.sequence} lacks required data type {data_type}")

    output_dir = args.work_root / args.sequence
    if output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"Output exists: {output_dir}. Pass --overwrite to replace it.")
        assert_child_path(output_dir, args.work_root)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args.temp_root.mkdir(parents=True, exist_ok=True)
    temp_sequence_dir = args.temp_root / args.sequence
    assert_child_path(temp_sequence_dir, args.temp_root)
    if temp_sequence_dir.exists():
        shutil.rmtree(temp_sequence_dir)
    temp_sequence_dir.mkdir(parents=True)

    summary = {
        "sequence": args.sequence,
        "stream_id": args.stream_id,
        "rotate": args.rotate,
        "overlay_scale": args.overlay_scale,
        "required_data_types": list(REQUIRED_DATA_TYPES),
        "keep_temp": args.keep_temp,
        "source_files": {
            data_type: {
                "filename": sequence_data[data_type]["filename"],
                "file_size_bytes": int(sequence_data[data_type]["file_size_bytes"]),
                "sha1sum": sequence_data[data_type].get("sha1sum", ""),
            }
            for data_type in REQUIRED_DATA_TYPES
        },
        "outputs": {},
    }

    success = False
    try:
        temp_info = build_temp_sequence(args, sequence_data, temp_sequence_dir)
        summary["temp_mps_files"] = {
            "eye_gaze": temp_info["gaze_files"],
            "slam": temp_info["slam_files"],
        }
        process_summary = process_sequence(args, temp_info, output_dir)
        summary.update(process_summary)
        summary["outputs"] = file_size_map(output_dir)
        summary["temp_deleted"] = False
        (output_dir / "processing_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        success = True
    finally:
        if success and not args.keep_temp:
            assert_child_path(temp_sequence_dir, args.temp_root)
            shutil.rmtree(temp_sequence_dir, ignore_errors=True)
            summary["temp_deleted"] = True
            summary["outputs"] = file_size_map(output_dir)
            (output_dir / "processing_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        elif args.keep_temp:
            print(f"[keep-temp] {temp_sequence_dir}")

    print(f"[ready] {output_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def run_batch(args: argparse.Namespace) -> int:
    if args.jobs < 1:
        raise RuntimeError("--jobs must be >= 1")

    manifest = load_manifest(args.manifest)
    sequences = resolve_sequences(args, manifest)
    log_root = args.log_root or (args.work_root / "_logs")

    args.work_root.mkdir(parents=True, exist_ok=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)
    if args.download_cache_root is not None:
        args.download_cache_root.mkdir(parents=True, exist_ok=True)
    if args.final_root is not None:
        args.final_root.mkdir(parents=True, exist_ok=True)

    if args.final_root is not None:
        sync_finished_outputs(sequences, args.work_root, args.final_root, args.overwrite)

    if args.overwrite:
        pending = sequences
        skipped = []
    else:
        pending = [
            sequence
            for sequence in sequences
            if not sequence_done(args.work_root, sequence) and not sequence_done(args.final_root, sequence)
        ]
        skipped = [sequence for sequence in sequences if sequence not in pending]

    print(
        f"[batch] {timestamp()} selected={len(sequences)} pending={len(pending)} "
        f"skipped_done={len(skipped)} jobs={args.jobs}",
        flush=True,
    )
    if args.final_root is not None:
        print(f"[batch] work_root={args.work_root}", flush=True)
        print(f"[batch] final_root={args.final_root}", flush=True)
    print(f"[batch] log_root={log_root}", flush=True)

    if not pending:
        copied = sync_finished_outputs(sequences, args.work_root, args.final_root, args.overwrite)
        print(f"[batch] nothing to process; synced={copied}/{len(sequences)}", flush=True)
        return 0

    completed = 0
    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_to_sequence = {
            executor.submit(run_worker_subprocess, args, sequence, log_root): sequence
            for sequence in pending
        }
        for future in as_completed(future_to_sequence):
            sequence = future_to_sequence[future]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:
                failed.append((sequence, str(exc), None))
                print(
                    f"[batch] fail {completed}/{len(pending)} {sequence}: {exc}",
                    flush=True,
                )
                continue

            if result["return_code"] == 0:
                synced = sync_sequence_output(sequence, args.work_root, args.final_root, args.overwrite)
                sync_label = "synced" if synced else "not-synced"
                print(
                    f"[batch] done {completed}/{len(pending)} {sequence} "
                    f"{result['seconds']:.1f}s {sync_label}",
                    flush=True,
                )
            else:
                failed.append((sequence, f"return_code={result['return_code']}", result["log_path"]))
                print(
                    f"[batch] fail {completed}/{len(pending)} {sequence} "
                    f"return_code={result['return_code']} log={result['log_path']}",
                    flush=True,
                )

    copied = sync_finished_outputs(sequences, args.work_root, args.final_root, args.overwrite)
    print(
        f"[batch] {timestamp()} finished selected={len(sequences)} "
        f"processed_ok={len(pending) - len(failed)} failed={len(failed)} synced={copied}/{len(sequences)}",
        flush=True,
    )
    if failed:
        for sequence, reason, log_path in failed[:20]:
            suffix = f" log={log_path}" if log_path else ""
            print(f"[batch] failed {sequence}: {reason}{suffix}", flush=True)
        return 1
    return 0


def main() -> int:
    args = parse_args()
    if args.worker_sequence:
        return run_worker(args)
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
