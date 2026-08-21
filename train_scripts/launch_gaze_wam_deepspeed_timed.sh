#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the shared Hydra run directory.}"
MACHINE_RANK="${MACHINE_RANK:?Set the zero-based host rank.}"
TIMING_FILE="$OUTPUT_DIR/run_timing.json"
LOG_FILE="$OUTPUT_DIR/launcher_rank${MACHINE_RANK}.log"

mkdir -p "$OUTPUT_DIR"
if [[ "$MACHINE_RANK" == "0" ]]; then
  start_epoch="$(date -u +%s)"
  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python3 - "$TIMING_FILE" "$start_epoch" "$start_utc" <<'PY'
import json
import sys

path, epoch, utc = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "status": "running",
            "started_at_utc": utc,
            "started_at_epoch_seconds": int(epoch),
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY
fi

set +e
"$ROOT/train_scripts/train_gaze_wam_deepspeed_multinode.sh" 2>&1 | tee -a "$LOG_FILE"
launcher_status=${PIPESTATUS[0]}
set -e

if [[ "$MACHINE_RANK" == "0" ]]; then
  end_epoch="$(date -u +%s)"
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python3 - "$TIMING_FILE" "$end_epoch" "$end_utc" "$launcher_status" <<'PY'
import json
import sys

path, epoch, utc, exit_code = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    timing = json.load(handle)
timing.update(
    {
        "status": "finished" if int(exit_code) == 0 else "failed",
        "finished_at_utc": utc,
        "finished_at_epoch_seconds": int(epoch),
        "elapsed_seconds": int(epoch) - int(timing["started_at_epoch_seconds"]),
        "launcher_exit_code": int(exit_code),
    }
)
with open(path, "w", encoding="utf-8") as handle:
    json.dump(timing, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
fi

exit "$launcher_status"
