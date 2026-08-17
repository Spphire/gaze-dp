from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import cv2


SUPPORTED_IMAGE_RESIZE_MODES = ("stretch", "letterbox")


def _validate_image_size(image_size: Optional[Sequence[int]]) -> Optional[Tuple[int, int]]:
    if image_size is None:
        return None
    values = tuple(int(value) for value in image_size)
    if len(values) != 2 or any(value <= 0 for value in values):
        raise ValueError(f"image_size must contain two positive integers, got {image_size!r}.")
    return values


def validate_image_resize_mode(image_resize_mode: str) -> str:
    """Validate the geometry used to align RGB frames with normalized gaze."""
    mode = str(image_resize_mode).strip().lower()
    if mode not in SUPPORTED_IMAGE_RESIZE_MODES:
        raise ValueError(
            "image_resize_mode must be one of: stretch, letterbox; "
            f"got {image_resize_mode!r}."
        )
    return mode


def letterbox_geometry(
    source_size: Sequence[int],
    target_size: Sequence[int],
) -> dict:
    """Return the exact resize and padding geometry for H/W image sizes."""
    source_h, source_w = (int(source_size[0]), int(source_size[1]))
    target_h, target_w = (int(target_size[0]), int(target_size[1]))
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError("source and target image dimensions must be positive")

    scale = min(float(target_w) / float(source_w), float(target_h) / float(source_h))
    resized_w = min(target_w, max(1, int(round(float(source_w) * scale))))
    resized_h = min(target_h, max(1, int(round(float(source_h) * scale))))
    pad_left = (target_w - resized_w) // 2
    pad_top = (target_h - resized_h) // 2
    return {
        "source_size": [source_h, source_w],
        "target_size": [target_h, target_w],
        "resized_size": [resized_h, resized_w],
        "scale_xy": [
            float(resized_w) / float(source_w),
            float(resized_h) / float(source_h),
        ],
        "padding_ltrb": [
            pad_left,
            pad_top,
            target_w - resized_w - pad_left,
            target_h - resized_h - pad_top,
        ],
    }


def remap_normalized_gaze_xy(
    gaze_xy: Sequence[float],
    *,
    source_image_size: Sequence[int],
    target_image_size: Sequence[int],
    image_resize_mode: str,
) -> np.ndarray:
    """Map raw-camera normalized gaze into the model's aligned image frame."""
    point = np.asarray(gaze_xy, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("gaze_xy must contain two finite normalized coordinates")
    mode = validate_image_resize_mode(image_resize_mode)
    if mode == "stretch":
        return point.astype(np.float32)

    geometry = letterbox_geometry(source_image_size, target_image_size)
    source_h, source_w = geometry["source_size"]
    target_h, target_w = geometry["target_size"]
    scale_x, scale_y = geometry["scale_xy"]
    pad_left, pad_top, _, _ = geometry["padding_ltrb"]
    output_x = float(point[0]) * float(source_w) * float(scale_x) + float(pad_left)
    output_y = float(point[1]) * float(source_h) * float(scale_y) + float(pad_top)
    return np.asarray(
        [output_x / float(target_w), output_y / float(target_h)],
        dtype=np.float32,
    )


def _resize_image_sequence(
    image: torch.Tensor,
    target_size: Tuple[int, int],
    image_resize_mode: str,
) -> torch.Tensor:
    mode = validate_image_resize_mode(image_resize_mode)
    source_size = (int(image.shape[-2]), int(image.shape[-1]))
    if source_size == target_size:
        return image
    if mode == "stretch":
        return F.interpolate(
            image,
            size=target_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    geometry = letterbox_geometry(source_size, target_size)
    resized_h, resized_w = geometry["resized_size"]
    pad_left, pad_top, _, _ = geometry["padding_ltrb"]
    # The training converter uses cv2.INTER_AREA for both resize modes. Keep
    # the letterbox path pixel-compatible with that converter, including its
    # zero-valued padding canvas.
    image_np = np.ascontiguousarray(image.detach().cpu().numpy())
    output_np = np.zeros(
        (image_np.shape[0], image_np.shape[1], target_size[0], target_size[1]),
        dtype=image_np.dtype,
    )
    for frame_index, frame_chw in enumerate(image_np):
        frame_hwc = np.moveaxis(frame_chw, 0, -1)
        resized_hwc = cv2.resize(
            frame_hwc,
            (resized_w, resized_h),
            interpolation=cv2.INTER_AREA,
        )
        output_np[
            frame_index,
            :,
            pad_top : pad_top + resized_h,
            pad_left : pad_left + resized_w,
        ] = np.moveaxis(resized_hwc, -1, 0)
    return torch.from_numpy(output_np).to(device=image.device, dtype=image.dtype)


def image_sequence_to_chw_float(
    image: np.ndarray,
    image_size: Optional[Sequence[int]] = None,
    *,
    image_resize_mode: str = "stretch",
    name: str = "camera image",
) -> np.ndarray:
    """Apply the canonical Gaze-WAM RGB image preprocessing to a sequence."""
    image_resize_mode = validate_image_resize_mode(image_resize_mode)
    image = np.asarray(image)
    if image.ndim != 4:
        raise ValueError(f"Expected image sequence [T,H,W,C] or [T,C,H,W], got {image.shape}.")
    if image.shape[-1] in (1, 3, 4):
        image = np.moveaxis(image, -1, 1)
    elif image.shape[1] not in (1, 3, 4):
        raise ValueError(f"Cannot infer channel dimension for image shape {image.shape}.")
    image = image.astype(np.float32)
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{name} must contain only finite values.")
    if image.max(initial=0.0) > 1.5:
        image = image / 255.0
    image = image[:, :3]
    if image.shape[1] == 1:
        image = np.repeat(image, 3, axis=1)

    validated_size = _validate_image_size(image_size)
    if validated_size is not None and image.shape[-2:] != validated_size:
        image_tensor = torch.from_numpy(np.ascontiguousarray(image))
        image_tensor = _resize_image_sequence(
            image_tensor,
            target_size=validated_size,
            image_resize_mode=image_resize_mode,
        )
        image = image_tensor.numpy()
    return np.ascontiguousarray(image, dtype=np.float32)


def image_to_chw_float(
    image: np.ndarray,
    image_size: Optional[Sequence[int]] = None,
    *,
    image_resize_mode: str = "stretch",
    name: str = "image",
) -> np.ndarray:
    """Apply canonical preprocessing to one HWC or CHW image."""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image [H,W,C] or [C,H,W], got {image.shape}.")
    return image_sequence_to_chw_float(
        image[None],
        image_size=image_size,
        image_resize_mode=image_resize_mode,
        name=name,
    )[0]


__all__ = [
    "SUPPORTED_IMAGE_RESIZE_MODES",
    "image_sequence_to_chw_float",
    "image_to_chw_float",
    "letterbox_geometry",
    "remap_normalized_gaze_xy",
    "validate_image_resize_mode",
]
