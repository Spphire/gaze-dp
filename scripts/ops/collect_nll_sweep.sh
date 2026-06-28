#!/usr/bin/env bash
# Wait for an NLL sweep to finish, then run the multimodal heatmap eval on every
# run's final checkpoint (in parallel across the now-free GPUs) and assemble a
# comparison table.
#
# Usage:
#   bash scripts/ops/collect_nll_sweep.sh [sweep-dir]
#   (defaults to the dir in data/outputs/.last_nll_sweep)
#
# Best launched detached so it fires when training completes:
#   bash scripts/ops/run_in_tmux.sh nll_collect \
#     data/outputs/logs/nll_collect.log \
#     bash scripts/ops/collect_nll_sweep.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
PYTHON=${PYTHON:-.venv/bin/python}
VAL_ZARR=${VAL_ZARR:-data/hot3d_open_val.zarr}
N_SAMPLES=${N_SAMPLES:-128}

SWEEP_DIR=${1:-$(cat data/outputs/.last_nll_sweep 2>/dev/null)}
if [[ -z "$SWEEP_DIR" || ! -d "$SWEEP_DIR" ]]; then
  echo "sweep dir not found: $SWEEP_DIR" >&2; exit 1
fi
echo "=== collect_nll_sweep $(date) ==="
echo "  sweep: $SWEEP_DIR"

# 1. wait until no sweep_* tmux sessions remain
echo "--- waiting for training to finish ---"
while true; do
  live=$(tmux ls 2>/dev/null | grep -c '^sweep_' || true)
  echo "[$(date +%T)] live sweep sessions: $live"
  [[ "$live" -eq 0 ]] && break
  sleep 120
done
echo "--- training done, syncing ---"
sync; sleep 5

# 2. eval each run on a dedicated GPU, in parallel
run_dirs=()
for d in "$SWEEP_DIR"/*/; do
  [[ -f "$d/checkpoints/latest.ckpt" ]] && run_dirs+=("$d")
done
echo "--- found ${#run_dirs[@]} checkpoints ---"
if [[ ${#run_dirs[@]} -eq 0 ]]; then
  echo "NO CHECKPOINTS — check training logs" >&2; exit 2
fi

gpu=0
pids=()
for d in "${run_dirs[@]}"; do
  name=$(basename "$d")
  ck="$d/checkpoints/latest.ckpt"
  out="$d/multimodal_eval.json"
  log="$SWEEP_DIR/logs/eval_${name}.log"
  echo "[gpu $gpu] eval $name"
  CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 "$PYTHON" scripts/ops/eval_multimodal_heatmap.py \
    --checkpoint "$ck" --val-zarr "$VAL_ZARR" --n-samples "$N_SAMPLES" \
    --device cuda:0 --output-json "$out" > "$log" 2>&1 &
  pids+=($!)
  gpu=$(( (gpu + 1) % 8 ))
done
echo "--- waiting for ${#pids[@]} evals ---"
for p in "${pids[@]}"; do wait "$p" || echo "eval pid $p returned nonzero"; done

# 3. assemble comparison table
echo "--- assembling RESULTS.md ---"
"$PYTHON" - "$SWEEP_DIR" <<'PYEOF'
import json, glob, os, sys, re
sweep = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(sweep, "*", "multimodal_eval.json"))):
    try:
        r = json.load(open(f))
    except Exception as e:
        print(f"skip {f}: {e}"); continue
    name = os.path.basename(os.path.dirname(f))
    # pull role + purpose from the run's EXPERIMENT.md
    role = purpose = ""
    card = os.path.join(os.path.dirname(f), "EXPERIMENT.md")
    if os.path.exists(card):
        txt = open(card, encoding="utf-8").read()
        mr = re.search(r"\*\*role\*\*:\s*(.+)", txt)
        mp = re.search(r"\*\*purpose\*\*:\s*(.+)", txt)
        role = mr.group(1).strip() if mr else ""
        purpose = mp.group(1).strip() if mp else ""
    rows.append((name, role, purpose, r))

def g(r, k):
    return r.get(k, float("nan"))

lines = []
lines.append("# NLL sweep — multimodal heatmap eval\n")
lines.append(f"sweep: `{sweep}`  |  val: hot3d_open_val.zarr\n")
lines.append("| run | role | argmax_l2↓ | point_nll↓ | peak_count | %multi | entropy |")
lines.append("|---|---|---|---|---|---|---|")
for name, role, purpose, r in rows:
    lines.append("| {} | {} | {:.4f} | {:.3f} | {:.2f} | {:.2f} | {:.3f} |".format(
        name, role, g(r,"argmax_l2_mean"), g(r,"point_nll_mean"),
        g(r,"peak_count_mean"), g(r,"peak_count_frac_multi"), g(r,"entropy_mean")))
lines.append("")
lines.append("## 每个 run 的目的")
for name, role, purpose, r in rows:
    lines.append(f"- **{name}** ({role}): {purpose}")
lines.append("")
lines.append("Read: lower argmax_l2 + lower point_nll = better localization/coverage; "
             "peak_count>1 and entropy not collapsing = multimodality preserved. "
             "The sweet spot maximizes localization WITHOUT collapsing peaks.")
out = os.path.join(sweep, "RESULTS.md")
open(out, "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
print(f"\nwrote {out}")
PYEOF

echo "=== collect done $(date) ==="
