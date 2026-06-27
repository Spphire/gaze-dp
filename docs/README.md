# Gaze-WAM Documentation Index

See the top-level [README.md](../README.md) for setup, data pipeline, and
training commands. This index points to the deeper docs.

## guides/ — how-to

Day-to-day usage references (Chinese).

- [local_usage_zh.md](guides/local_usage_zh.md) — local development workflow
- [test_guide_zh.md](guides/test_guide_zh.md) — quickest path to confirm code
  compiles, key unit tests pass, and `preflight_gaze_wam.py` runs
- [training_review_zh.md](guides/training_review_zh.md) — walk-through of the
  training-loop code for reviewers

## design/ — design and research

- [design_plan.md](design/design_plan.md) — full design and implementation plan
  (3.4 k lines, partially historical: predates UMI/SLAM removal and includes
  rationale references to the original fork — treat those sections as
  historical context, not as current architecture)
- [single_point_to_multimodal_research.md](design/single_point_to_multimodal_research.md)
  — research notes on going from single-point gaze supervision to a full
  multimodal heatmap target

## reviews/ — code / architecture reviews

- [expert_review_20260607.md](reviews/expert_review_20260607.md) — 2026-06-07
  expert architecture review
- [fullres_fastwam_review.md](reviews/fullres_fastwam_review.md) — review of
  the full-resolution FastWAM heatmap path

## experiments/ — empirical results

- [temporal_window_report.md](experiments/temporal_window_report.md) —
  temporal-window ablation experiment report
- [gaze_baseline_hot3d_val.md](experiments/gaze_baseline_hot3d_val.md) —
  numeric L2 lower bounds (center, prior-mean, prev-frame, …) on HOT3D val.
  **Any trained model must clear these to be considered learning.**
- [smoke_mixed_nll_diagnostic.md](experiments/smoke_mixed_nll_diagnostic.md) —
  130-step smoke training on the `mixed_nll` config. Key finding: the
  `point_nll` term at weight 0.001 is effectively dead — the config is closer
  to `diffusion + 0.1·JS` than its name suggests.

## _archive/ — historical dev logs

Kept because they record the chronology of decisions, but they are not the
current source of truth — they reference deleted modules (UMI/SLAM tooling)
and use paths that no longer exist after the 2026-06-27 cleanup. Read with
that caveat.

- [implementation_log.md](_archive/implementation_log.md) — 22 k-line
  chronological dev log
- [server_run_status.md](_archive/server_run_status.md) — 5 k-line server run
  status log
