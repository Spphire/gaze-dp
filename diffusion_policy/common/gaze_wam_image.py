from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _validate_image_size(image_size: Optional[Sequence[int]]) -> Optional[Tuple[int, int]]:
    if image_size is None:
        return None
    values = tuple(int(value) for value in image_size)
    if len(values) != 2 or any(value <= 0 for value in values):
        raise ValueError(f"image_size must contain two positive integers, got {image_size!r}.")
    return values


def image_sequence_to_chw_float(
    image: np.ndarray,
    image_size: Optional[Sequence[int]] = None,
    *,
    name: str = "camera image",
) -> np.ndarray:
    """Apply the canonical Gaze-WAM RGB image preprocessing to a sequence."""
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
        image_tensor = F.interpolate(
            image_tensor,
            size=validated_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        image = image_tensor.numpy()
    return np.ascontiguousarray(image, dtype=np.float32)


def image_to_chw_float(
    image: np.ndarray,
    image_size: Optional[Sequence[int]] = None,
    *,
    name: str = "image",
) -> np.ndarray:
    """Apply canonical preprocessing to one HWC or CHW image."""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image [H,W,C] or [C,H,W], got {image.shape}.")
    return image_sequence_to_chw_float(
        image[None],
        image_size=image_size,
        name=name,
    )[0]


__all__ = ["image_sequence_to_chw_float", "image_to_chw_float"]
