#!/usr/bin/env bash
# Verify a Gaze-WAM open-gaze zarr is complete and self-consistent.
#
# Checks performed:
#   1. Every required field directory exists.
#   2. Every required field has a .zarray (zarr v2 metadata).
#   3. The number of chunk files on disk equals the product of
#      ceil(shape[i]/chunks[i]) derived from .zarray — i.e. no missing chunks.
#
# This catches the failure mode seen on 2026-06-27 where a partially-deleted
# output directory caused the converter to "succeed" while leaving 192 chunks
# missing from camera0_rgb and four fields with no .zarray file.
#
# Usage:
#   scripts/ops/verify_open_zarr.sh <zarr-path>
#
# Exit codes:
#   0 = OK
#   1 = bad arguments
#   2 = one or more fields broken (details printed to stdout)

set -euo pipefail

ZARR=${1:-}
if [[ -z "$ZARR" ]]; then
  echo "usage: $0 <zarr-path>" >&2
  exit 1
fi
if [[ ! -d "$ZARR" ]]; then
  echo "zarr path not found or not a directory: $ZARR" >&2
  exit 1
fi

PYTHON=${PYTHON:-.venv/bin/python}
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

"$PYTHON" - "$ZARR" <<'PYEOF'
import json, math, os, sys

zarr_path = sys.argv[1]
REQUIRED = [
    "gaze_xy",
    "gaze_heatmap",
    "has_gaze_label",
    "has_heatmap_image",
    "camera0_rgb",
    "source_sequence_index",
    "timestamp_ns",
]
data_dir = os.path.join(zarr_path, "data")
meta_dir = os.path.join(zarr_path, "meta")

problems = []

if not os.path.isdir(data_dir):
    problems.append(f"missing data/ directory under {zarr_path}")
if not os.path.isdir(meta_dir):
    problems.append(f"missing meta/ directory under {zarr_path}")
if not os.path.isfile(os.path.join(meta_dir, "episode_ends", ".zarray")):
    problems.append("missing meta/episode_ends/.zarray")

for k in REQUIRED:
    p = os.path.join(data_dir, k)
    if not os.path.isdir(p):
        problems.append(f"missing field directory: data/{k}")
        continue
    zarr_meta_path = os.path.join(p, ".zarray")
    if not os.path.isfile(zarr_meta_path):
        problems.append(f"missing data/{k}/.zarray (chunks may be orphans)")
        continue
    z = json.load(open(zarr_meta_path))
    shape = z["shape"]
    chunks = z["chunks"]
    expected = 1
    for s, c in zip(shape, chunks):
        expected *= math.ceil(s / c)
    actual = sum(
        1 for f in os.listdir(p)
        if not f.startswith(".")
    )
    if actual != expected:
        problems.append(
            f"data/{k}: shape={shape} chunks={chunks} → expected {expected} chunk files, got {actual}"
        )
    else:
        print(f"  OK data/{k}: shape={shape} files={actual}")

if problems:
    print("--- VERIFY FAILED ---")
    for p in problems:
        print(f"  {p}")
    sys.exit(2)
print(f"--- VERIFY OK: {zarr_path} ---")
PYEOF
