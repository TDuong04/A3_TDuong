---
id: A3-007
title: Implement the intrinsic reward experiment on level 6
type: feature
status: done
priority: P0
rubric: F
points_at_risk: 3.0
area: gridworld
github: https://github.com/TDuong04/A3_TDuong/issues/7
owner: unassigned
blocks: [A3-014]
blocked_by: [A3-005]
created: 2026-08-19
updated: 2026-08-20
---

# A3-007 — Implement the intrinsic reward experiment on level 6

## Context

Level 6 is deliberately exploration-hostile — a distant chest behind a long corridor with the
key down a dead-end branch — so a visit-count bonus has a real problem to solve and the comparison
curve shows a visible difference.

## Acceptance criteria

- [x] `r_i = strength / sqrt(n(s) + 1)` implemented exactly as the brief states
- [x] `n(s)` counts visits within the current episode and resets on reset
- [x] Environment rewards unchanged; the intrinsic term is added only inside the update
- [x] Two runs on level 6 at strengths 0.0 and 0.5, plotted on one axis
- [x] Short written explanation of the improvement for the report

## Notes

Config block `intrinsic_experiment` already exists in `config/gridworld.yaml`.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/7
- 2026-08-20 — implemented in `gridworld/algorithms.py` (`intrinsic_reward`, `EpisodeVisitCounts`,
  `train_with_intrinsic_reward`); sweep artifacts in `results/intrinsic_level6_q.*`
- 2026-08-20 — verified: all five headline numbers recomputed from the raw per-episode CSVs and
  matched exactly (final-500 solved 1.0000 at strength 0 against 0.3280 at 0.5; all-episode 0.8886
  against 0.3953). The identical-first-success artefact reproduces — every non-zero strength first
  opens the chest on the same episodes, seed for seed, against a different set for the baseline.
  Four mutants applied to the intrinsic code (sqrt offset, counter never reset, shaped return
  logged instead of the environment return, bonus keyed on the state left) and all four were
  killed by `tests/test_gridworld_intrinsic.py`. Closed.
