#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import zarr


GAZE_PROJECTION_VALID = 1
GAZE_PROJECTION_OUT_OF_FRAME = 3


def verify_gaze_condition_contract(
    dataset_path: str,
    *,
    require_out_of_frame: bool = False,
) -> Dict[str, object]:
    path = Path(dataset_path).expanduser().resolve()
    root = zarr.open(str(path), mode="r")
    if "data" not in root:
        raise KeyError(f"Zarr is missing the data group: {path}")
    data = root["data"]
    required = (
        "gaze_xy",
        "has_gaze_condition",
        "has_gaze_label",
        "gaze_projection_status",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Zarr is missing gaze condition contract arrays: {missing}")

    gaze_xy = np.asarray(data["gaze_xy"][:], dtype=np.float32)
    has_condition = np.asarray(data["has_gaze_condition"][:]).reshape(-1).astype(bool)
    has_label = np.asarray(data["has_gaze_label"][:]).reshape(-1).astype(bool)
    status = np.asarray(data["gaze_projection_status"][:]).reshape(-1)
    rows = int(gaze_xy.shape[0])
    if gaze_xy.shape != (rows, 2):
        raise ValueError(f"gaze_xy must have shape [N,2], got {gaze_xy.shape}.")
    for name, value in (
        ("has_gaze_condition", has_condition),
        ("has_gaze_label", has_label),
        ("gaze_projection_status", status),
    ):
        if value.shape != (rows,):
            raise ValueError(f"{name} must have shape [N], got {value.shape}.")
    if not np.all(np.isfinite(gaze_xy[has_condition])):
        raise ValueError("Conditioned gaze_xy rows must contain only finite values.")

    expected_condition = np.isin(
        status,
        (GAZE_PROJECTION_VALID, GAZE_PROJECTION_OUT_OF_FRAME),
    )
    expected_label = status == GAZE_PROJECTION_VALID
    if not np.array_equal(has_condition, expected_condition):
        mismatch = int(np.count_nonzero(has_condition != expected_condition))
        raise ValueError(
            "has_gaze_condition must be true exactly for valid and out-of-frame "
            f"projections; mismatched rows={mismatch}."
        )
    if not np.array_equal(has_label, expected_label):
        mismatch = int(np.count_nonzero(has_label != expected_label))
        raise ValueError(
            "has_gaze_label must be true exactly for in-frame valid projections; "
            f"mismatched rows={mismatch}."
        )
    if np.any(gaze_xy[~has_condition] != 0):
        raise ValueError("Rows without a gaze condition must use zero gaze_xy placeholders.")

    out_of_frame = status == GAZE_PROJECTION_OUT_OF_FRAME
    out_of_frame_rows = int(np.count_nonzero(out_of_frame))
    if require_out_of_frame and out_of_frame_rows == 0:
        raise ValueError("Dataset has no out-of-frame gaze rows to condition on.")

    return {
        "dataset_path": str(path),
        "valid": True,
        "rows": rows,
        "has_gaze_condition_rows": int(np.count_nonzero(has_condition)),
        "has_gaze_label_rows": int(np.count_nonzero(has_label)),
        "out_of_frame_condition_rows": out_of_frame_rows,
        "missing_or_invalid_rows": int(np.count_nonzero(~has_condition)),
        "routing": {
            "action_condition": "has_gaze_condition",
            "heatmap_supervision": "has_gaze_label",
            "out_of_frame_action_condition": True,
            "out_of_frame_heatmap_supervision": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the split gaze condition/heatmap-label zarr contract."
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--require-out-of-frame", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Dict[str, object]:
    args = parse_args(argv)
    summary = verify_gaze_condition_contract(
        args.dataset_path,
        require_out_of_frame=args.require_out_of_frame,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
