from __future__ import annotations

import argparse
import ast
import json
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple


ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_YAML = ROOT_DIR / "diffusion_policy" / "config" / "train_gaze_wam_workspace.yaml"
DEFAULT_TASK_YAML = ROOT_DIR / "diffusion_policy" / "config" / "task" / "gaze_wam.yaml"
DEFAULT_MODEL_NAME = "vit_base_patch16_dinov3"
DEFAULT_IMAGE_SIZE = [256, 256]
DEFAULT_PATCH_SIZE = 16
DEFAULT_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
DEFAULT_NORMALIZE_STD = [0.229, 0.224, 0.225]


def _strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _parse_scalar(value: str):
    value = value.strip()
    if value == "":
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if value.startswith("${"):
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        pass
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def _extract_path_value(text: str, path: Sequence[str]):
    stack: List[Tuple[int, str]] = []
    target = list(path)
    for raw_line in text.splitlines():
        line = _strip_inline_comment(raw_line).rstrip()
        if not line.strip() or line.lstrip().startswith("-"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        current_path = [item_key for _, item_key in stack] + [key]
        if current_path == target:
            return _parse_scalar(value)
        if value == "":
            stack.append((indent, key))
    return None


def _extract_normalize_stats(text: str) -> Dict[str, Optional[List[float]]]:
    active = False
    item_indent = 0
    mean = None
    std = None
    for raw_line in text.splitlines():
        line = _strip_inline_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped.startswith("-"):
            active = "torchvision.transforms.Normalize" in stripped or stripped.endswith(".Normalize")
            item_indent = indent
            continue
        if active and indent <= item_indent:
            active = False
        if not active or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        parsed = _parse_scalar(value)
        if key.strip() == "mean":
            mean = _float_list(parsed)
        elif key.strip() == "std":
            std = _float_list(parsed)
    return {"mean": mean, "std": std}


def _float_list(value) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, str):
        value = _parse_scalar(value)
    try:
        return [float(item) for item in list(value)]
    except (TypeError, ValueError):
        return None


def _int_list(value) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, str):
        value = _parse_scalar(value)
    try:
        return [int(item) for item in list(value)]
    except (TypeError, ValueError):
        return None


def _as_bool(value, default: bool = False) -> bool:
    from diffusion_policy.common.gaze_wam_training_config import (
        normalize_gaze_wam_bool_field,
    )

    return normalize_gaze_wam_bool_field("value", value, default=default)


def _path_str(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _path_exists(value: str) -> bool:
    return bool(value) and pathlib.Path(value).exists()


def _path_is_file(value: str) -> bool:
    return bool(value) and pathlib.Path(value).is_file()


def _path_is_dir(value: str) -> bool:
    return bool(value) and pathlib.Path(value).is_dir()


def _path_file_size(value: str) -> Optional[int]:
    if not _path_is_file(value):
        return None
    return int(pathlib.Path(value).stat().st_size)


def _cache_contains_files(value: str, limit: int = 1000) -> Optional[bool]:
    if not _path_is_dir(value):
        return None
    count = 0
    for item in pathlib.Path(value).rglob("*"):
        if item.is_file():
            return True
        count += 1
        if count >= limit:
            break
    return False


def _lists_close(left: Optional[Sequence[float]], right: Sequence[float], atol: float = 1e-9) -> bool:
    if left is None or len(left) != len(right):
        return False
    return all(abs(float(a) - float(b)) <= atol for a, b in zip(left, right))


def _pair(value, default: Sequence[int]) -> List[int]:
    parsed = _int_list(value)
    if parsed is None or len(parsed) != 2:
        return [int(v) for v in default]
    return [int(parsed[0]), int(parsed[1])]


def _last_hw_from_image_shape(value) -> Optional[List[int]]:
    parsed = _int_list(value)
    if parsed is None or len(parsed) < 2:
        return None
    return [int(parsed[-2]), int(parsed[-1])]


def _read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def verify_gaze_wam_dino_source(
    *,
    config_yaml: str = str(DEFAULT_CONFIG_YAML),
    task_yaml: str = str(DEFAULT_TASK_YAML),
    model_name: Optional[str] = None,
    expected_model_name: str = DEFAULT_MODEL_NAME,
    pretrained: Optional[bool] = None,
    checkpoint_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    image_size: Optional[Sequence[int]] = None,
    patch_size: int = DEFAULT_PATCH_SIZE,
    heatmap_token_grid: Optional[Sequence[int]] = None,
    heatmap_num_tokens: Optional[int] = None,
    normalize_mean: Optional[Sequence[float]] = None,
    normalize_std: Optional[Sequence[float]] = None,
    expected_normalize_mean: Sequence[float] = DEFAULT_NORMALIZE_MEAN,
    expected_normalize_std: Sequence[float] = DEFAULT_NORMALIZE_STD,
    require_local_source: bool = True,
    require_cache_files: bool = False,
) -> Dict[str, object]:
    config_path = pathlib.Path(config_yaml)
    task_path = pathlib.Path(task_yaml)
    config_text = _read_text(config_path)
    task_text = _read_text(task_path)
    normalize_stats = _extract_normalize_stats(config_text)

    config_model_name = _extract_path_value(config_text, ["policy", "obs_encoder", "model_name"])
    config_pretrained = _extract_path_value(config_text, ["policy", "obs_encoder", "pretrained"])
    config_checkpoint_path = _extract_path_value(
        config_text,
        ["policy", "obs_encoder", "checkpoint_path"],
    )
    config_cache_dir = _extract_path_value(config_text, ["policy", "obs_encoder", "cache_dir"])
    config_downsample_ratio = _extract_path_value(
        config_text,
        ["policy", "obs_encoder", "downsample_ratio"],
    )
    config_image_tokens_per_frame = _extract_path_value(
        config_text,
        ["policy", "image_tokens_per_frame"],
    )
    config_heatmap_image_size = _extract_path_value(config_text, ["policy", "heatmap_image_size"])

    task_image_hw = _last_hw_from_image_shape(_extract_path_value(task_text, ["image_shape"]))
    task_heatmap_grid = _int_list(_extract_path_value(task_text, ["heatmap_token_grid"]))
    task_heatmap_tokens = _extract_path_value(task_text, ["heatmap_num_tokens"])

    model_name = str(model_name if model_name is not None else config_model_name or "")
    pretrained = _as_bool(config_pretrained, False) if pretrained is None else _as_bool(pretrained)
    checkpoint_path = _path_str(
        checkpoint_path if checkpoint_path is not None else config_checkpoint_path
    )
    cache_dir = _path_str(cache_dir if cache_dir is not None else config_cache_dir)
    image_size = [int(v) for v in (image_size or task_image_hw or DEFAULT_IMAGE_SIZE)]
    heatmap_token_grid = [
        int(v) for v in (heatmap_token_grid or task_heatmap_grid or [16, 16])
    ]
    heatmap_num_tokens = int(
        heatmap_num_tokens if heatmap_num_tokens is not None else task_heatmap_tokens or 256
    )
    normalize_mean = (
        [float(v) for v in normalize_mean]
        if normalize_mean is not None
        else normalize_stats["mean"]
    )
    normalize_std = (
        [float(v) for v in normalize_std]
        if normalize_std is not None
        else normalize_stats["std"]
    )
    heatmap_image_size = _pair(config_heatmap_image_size, image_size)

    checkpoint_exists = _path_exists(checkpoint_path)
    checkpoint_is_file = _path_is_file(checkpoint_path)
    checkpoint_file_size = _path_file_size(checkpoint_path)
    cache_exists = _path_exists(cache_dir)
    cache_is_dir = _path_is_dir(cache_dir)
    cache_contains_files = _cache_contains_files(cache_dir)
    local_source_configured = bool(checkpoint_path or cache_dir)
    local_source_exists = (
        (not checkpoint_path or checkpoint_exists) and (not cache_dir or cache_exists)
    )
    local_source_valid = (
        (not checkpoint_path or checkpoint_is_file)
        and (not cache_dir or cache_is_dir)
        and (not checkpoint_path or (checkpoint_file_size or 0) > 0)
    )

    expected_grid = [
        image_size[0] // patch_size if patch_size > 0 and image_size[0] % patch_size == 0 else -1,
        image_size[1] // patch_size if patch_size > 0 and image_size[1] % patch_size == 0 else -1,
    ]
    expected_tokens = expected_grid[0] * expected_grid[1] if min(expected_grid) > 0 else -1
    grid_product = heatmap_token_grid[0] * heatmap_token_grid[1]

    checks: List[Dict[str, object]] = []

    def add_check(name: str, ok: bool, message: str, *, severity: str = "error") -> None:
        ok = bool(ok)
        checks.append(
            {
                "name": name,
                "ok": ok,
                "severity": severity,
                "message": "ok" if ok else message,
                "failure_message": message,
            }
        )

    add_check(
        "config_yaml_exists",
        config_path.exists(),
        f"Config YAML does not exist: {config_path}",
    )
    add_check("task_yaml_exists", task_path.exists(), f"Task YAML does not exist: {task_path}")
    add_check(
        "model_name",
        model_name == expected_model_name,
        f"DINO model_name must be {expected_model_name!r}, got {model_name!r}.",
    )
    add_check(
        "image_size_256",
        list(image_size) == DEFAULT_IMAGE_SIZE,
        f"Gaze-WAM main config expects image_size={DEFAULT_IMAGE_SIZE}, got {image_size!r}.",
    )
    add_check(
        "patch_size_16",
        int(patch_size) == DEFAULT_PATCH_SIZE,
        f"Gaze-WAM DINOv3 ViT/16 path expects patch_size={DEFAULT_PATCH_SIZE}, got {patch_size!r}.",
    )
    add_check(
        "heatmap_grid_matches_image_patch_grid",
        heatmap_token_grid == expected_grid,
        "Heatmap token grid must match the image/patch grid.",
    )
    add_check(
        "heatmap_num_tokens_matches_grid",
        heatmap_num_tokens == grid_product,
        "heatmap_num_tokens must equal heatmap_token_grid product.",
    )
    add_check(
        "heatmap_num_tokens_256",
        heatmap_num_tokens == expected_tokens == 256,
        "DINOv3/16 at 256x256 expects 256 image tokens and 256 heatmap tokens per frame.",
    )
    if config_downsample_ratio is not None:
        add_check(
            "obs_encoder_downsample_ratio_matches_patch",
            int(config_downsample_ratio) == int(patch_size),
            "policy.obs_encoder.downsample_ratio must match the ViT patch size.",
        )
    if config_image_tokens_per_frame is not None:
        add_check(
            "image_tokens_per_frame",
            int(config_image_tokens_per_frame) == expected_tokens,
            "policy.image_tokens_per_frame must match the DINO image-token count per frame.",
        )
    add_check(
        "heatmap_image_size_matches_image_size",
        heatmap_image_size == list(image_size),
        "policy.heatmap_image_size must match task image H/W.",
    )
    if pretrained:
        add_check(
            "normalize_mean",
            _lists_close(normalize_mean, expected_normalize_mean),
            "Normalize mean does not match the expected DINOv3/ImageNet statistics.",
        )
        add_check(
            "normalize_std",
            _lists_close(normalize_std, expected_normalize_std),
            "Normalize std does not match the expected DINOv3/ImageNet statistics.",
        )
    if pretrained and require_local_source:
        add_check(
            "local_source_configured",
            local_source_configured,
            "Pretrained DINOv3 requires policy.obs_encoder.checkpoint_path or cache_dir.",
        )
        add_check(
            "local_source_exists",
            local_source_exists,
            "Configured DINOv3 checkpoint/cache path does not exist.",
        )
        add_check(
            "local_source_valid",
            local_source_valid,
            "checkpoint_path must be a non-empty file and cache_dir must be a directory.",
        )
        if checkpoint_path:
            add_check(
                "checkpoint_path_is_file",
                checkpoint_is_file,
                f"checkpoint_path must point to a file, got {checkpoint_path!r}.",
            )
            add_check(
                "checkpoint_path_nonempty",
                (checkpoint_file_size or 0) > 0,
                f"checkpoint_path must not be empty, got {checkpoint_path!r}.",
            )
        if cache_dir:
            add_check(
                "cache_dir_is_dir",
                cache_is_dir,
                f"cache_dir must point to a directory, got {cache_dir!r}.",
            )
            add_check(
                "cache_dir_contains_files",
                bool(cache_contains_files),
                (
                    "cache_dir contains no files; this cannot prove DINOv3 weights are cached. "
                    "Use --require-cache-files to make this launch-blocking."
                ),
                severity="error" if require_cache_files else "warning",
            )

    errors = [
        check["failure_message"]
        for check in checks
        if not check["ok"] and str(check["severity"]) == "error"
    ]
    warnings = [
        check["failure_message"]
        for check in checks
        if not check["ok"] and str(check["severity"]) == "warning"
    ]
    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "config": {
            "config_yaml": str(config_path),
            "config_yaml_exists": config_path.exists(),
            "task_yaml": str(task_path),
            "task_yaml_exists": task_path.exists(),
        },
        "dino_source": {
            "model_name": model_name,
            "expected_model_name": expected_model_name,
            "pretrained": pretrained,
            "checkpoint_path": checkpoint_path,
            "checkpoint_path_exists": checkpoint_exists,
            "checkpoint_path_is_file": checkpoint_is_file,
            "checkpoint_path_file_size": checkpoint_file_size,
            "cache_dir": cache_dir,
            "cache_dir_exists": cache_exists,
            "cache_dir_is_dir": cache_is_dir,
            "cache_dir_contains_files": cache_contains_files,
            "local_source_configured": local_source_configured,
            "local_source_exists": local_source_exists,
            "local_source_valid": local_source_valid,
        },
        "geometry": {
            "image_size": list(image_size),
            "patch_size": int(patch_size),
            "expected_patch_grid": expected_grid,
            "expected_tokens_per_frame": expected_tokens,
            "heatmap_token_grid": list(heatmap_token_grid),
            "heatmap_num_tokens": int(heatmap_num_tokens),
            "heatmap_grid_product": int(grid_product),
            "heatmap_image_size": heatmap_image_size,
            "obs_encoder_downsample_ratio": config_downsample_ratio,
            "image_tokens_per_frame": config_image_tokens_per_frame,
        },
        "normalization": {
            "mean": normalize_mean,
            "std": normalize_std,
            "expected_mean": [float(v) for v in expected_normalize_mean],
            "expected_std": [float(v) for v in expected_normalize_std],
        },
    }


def _write_json(path: Optional[str], payload: Dict[str, object]) -> None:
    if not path:
        return
    output = pathlib.Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Lightweight Gaze-WAM DINOv3 source/preprocess verifier. "
            "This checks config paths and tensor geometry without importing torch, timm, or Hydra."
        )
    )
    parser.add_argument("--config-yaml", default=str(DEFAULT_CONFIG_YAML))
    parser.add_argument("--task-yaml", default=str(DEFAULT_TASK_YAML))
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--expected-model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--image-size", nargs=2, type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--heatmap-token-grid", nargs=2, type=int, default=None)
    parser.add_argument("--heatmap-num-tokens", type=int, default=None)
    parser.add_argument("--normalize-mean", nargs=3, type=float, default=None)
    parser.add_argument("--normalize-std", nargs=3, type=float, default=None)
    parser.add_argument("--expected-normalize-mean", nargs=3, type=float, default=DEFAULT_NORMALIZE_MEAN)
    parser.add_argument("--expected-normalize-std", nargs=3, type=float, default=DEFAULT_NORMALIZE_STD)
    parser.add_argument("--require-local-source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-cache-files", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args(argv)

    summary = verify_gaze_wam_dino_source(
        config_yaml=args.config_yaml,
        task_yaml=args.task_yaml,
        model_name=args.model_name,
        expected_model_name=args.expected_model_name,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint_path,
        cache_dir=args.cache_dir,
        image_size=args.image_size,
        patch_size=args.patch_size,
        heatmap_token_grid=args.heatmap_token_grid,
        heatmap_num_tokens=args.heatmap_num_tokens,
        normalize_mean=args.normalize_mean,
        normalize_std=args.normalize_std,
        expected_normalize_mean=args.expected_normalize_mean,
        expected_normalize_std=args.expected_normalize_std,
        require_local_source=args.require_local_source,
        require_cache_files=args.require_cache_files,
    )
    _write_json(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
