#!/usr/bin/env bash
# Launch a command inside a fresh tmux session, detached.
# Stdout/stderr are tee'd to a log file so progress can be tailed without
# attaching; the session itself remains live for interactive attach.
#
# Usage:
#   scripts/ops/run_in_tmux.sh <session-name> <log-path> <cmd...>
#
# Example:
#   scripts/ops/run_in_tmux.sh \
#     gaze_train_zarr  data/outputs/logs/train_zarr_$(date +%Y%m%d_%H%M).log \
#     bash scripts/ops/prepare_hot3d_zarr.sh data/hot3d_train_sequences.txt data/hot3d_open_train.zarr
#
# Inspect later:
#   tmux ls
#   tmux attach -t gaze_train_zarr        (Ctrl-b d to detach)
#   tail -f <log-path>                    (any time, no tmux involvement)
#
# Stop:
#   tmux kill-session -t gaze_train_zarr

set -euo pipefail

SESSION=${1:-}
LOG=${2:-}
shift 2 || true

if [[ -z "$SESSION" || -z "$LOG" || $# -eq 0 ]]; then
  echo "usage: $0 <session-name> <log-path> <cmd...>" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$SESSION' already exists. Kill it or choose another name." >&2
  echo "  tmux kill-session -t $SESSION" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG")"

# Build a quoted command line so set -x in tmux is faithful.
CMD=""
for arg in "$@"; do
  CMD+=" $(printf '%q' "$arg")"
done

# Run under bash so set -e -o pipefail apply; tee output to log file.
WRAP="set -o pipefail; echo \"=== tmux:$SESSION start \$(date) ===\"; { ${CMD# }; } 2>&1 | tee \"$LOG\"; RC=\${PIPESTATUS[0]}; echo \"=== tmux:$SESSION exit \$RC \$(date) ===\"; exit \$RC"

tmux new-session -d -s "$SESSION" "bash -lc $(printf '%q' "$WRAP")"
echo "started tmux session: $SESSION"
echo "  log: $LOG"
echo "  attach: tmux attach -t $SESSION"
