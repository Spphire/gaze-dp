from __future__ import annotations

import numpy as np
import torch

from diffusion_policy.common.gaze_wam_image import (
    image_sequence_to_chw_float,
    letterbox_geometry,
    remap_normalized_gaze_xy,
)
from diffusion_policy.real_world.gaze_wam_inference import GazeWamInferenceAdapter


def test_letterbox_matches_robot_training_geometry() -> None:
    geometry = letterbox_geometry((720, 1280), (256, 256))
    assert geometry["resized_size"] == [144, 256]
    assert geometry["padding_ltrb"] == [0, 56, 0, 56]

    image = np.full((1, 720, 1280, 3), 255, dtype=np.uint8)
    processed = image_sequence_to_chw_float(
        image,
        image_size=(256, 256),
        image_resize_mode="letterbox",
    )
    assert processed.shape == (1, 3, 256, 256)
    np.testing.assert_array_equal(processed[:, :, :56], 0.0)
    np.testing.assert_array_equal(processed[:, :, 200:], 0.0)
    assert float(processed[:, :, 56:200].mean()) > 0.99


def test_letterbox_remaps_gaze_into_padded_frame() -> None:
    np.testing.assert_allclose(
        remap_normalized_gaze_xy(
            [0.5, 0.0],
            source_image_size=(720, 1280),
            target_image_size=(256, 256),
            image_resize_mode="letterbox",
        ),
        [0.5, 0.21875],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        remap_normalized_gaze_xy(
            [0.5, 1.0],
            source_image_size=(720, 1280),
            target_image_size=(256, 256),
            image_resize_mode="letterbox",
        ),
        [0.5, 0.78125],
        atol=1e-7,
    )


class _FakePolicy:
    device = torch.device("cpu")

    @staticmethod
    def _validate_nonnegative_float(name: str, value: float) -> float:
        del name
        return float(value)

    def eval(self):
        return self

    def to(self, device):
        self.device = torch.device(device)
        return self


def test_inference_adapter_applies_same_image_and_gaze_geometry() -> None:
    adapter = GazeWamInferenceAdapter(
        policy=_FakePolicy(),
        shape_meta={
            "obs": {
                "camera0_rgb": {
                    "shape": [3, 256, 256],
                    "horizon": 2,
                }
            }
        },
        camera_key="camera0_rgb",
        image_resize_mode="letterbox",
        device="cpu",
    )
    adapter.push_image(np.full((720, 1280, 3), 255, dtype=np.uint8))
    obs = adapter.build_obs(gaze_xy=[0.5, 0.0])

    np.testing.assert_allclose(obs["gaze_xy"].numpy(), [[0.5, 0.21875]], atol=1e-7)
    assert tuple(obs["camera0_rgb"].shape) == (1, 2, 3, 256, 256)
    np.testing.assert_array_equal(obs["camera0_rgb"].numpy()[:, :, :, :56], 0.0)
    assert float(obs["camera0_rgb"].numpy()[:, :, :, 56:200].mean()) > 0.99
