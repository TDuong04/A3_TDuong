---
id: A3-027
title: Delete orphaned level 6 intrinsic-reward artefacts from an earlier sweep
type: chore
status: open
priority: P2
rubric: none
points_at_risk: 0
area: gridworld
part: 1
github: https://github.com/TDuong04/A3_TDuong/issues/30
owner: member-3
blocks: []
blocked_by: []
created: 2026-08-29
updated: 2026-08-29
---

# A3-027 — Delete orphaned level 6 intrinsic-reward artefacts from an earlier sweep

## Context

A rubric audit (2026-08-29) found `results/` still holds per-seed raw files from an earlier
intrinsic-reward sweep over five strengths, but `config/gridworld.yaml`'s
`intrinsic_experiment.strengths` now lists only `[0.0, 0.5]`. The files for `strength0` and
`strength0p5` match the current config; the files for `strength0p1`, `strength0p25` and `strength1`
do not correspond to any value the config can currently produce.

This carries no rubric row of its own, but it is not harmless: a marker or a report author could
cite one of these orphaned runs, and nobody could reproduce it from a clean checkout because the
config no longer contains that strength value.

## Acceptance criteria

- [ ] `results/history_level6_q_strength0p1_seed*.csv` deleted (5 files, seeds 0-4)
- [ ] `results/history_level6_q_strength0p25_seed*.csv` deleted (5 files, seeds 0-4)
- [ ] `results/history_level6_q_strength1_seed*.csv` deleted (5 files, seeds 0-4)
- [ ] `results/qtable_level6_q_strength0p1_seed*.npz` deleted (5 files, seeds 0-4)
- [ ] `results/qtable_level6_q_strength0p25_seed*.npz` deleted (5 files, seeds 0-4)
- [ ] `results/qtable_level6_q_strength1_seed*.npz` deleted (5 files, seeds 0-4)
- [ ] `results/intrinsic_level6_q.json`, `results/intrinsic_level6_q.md`,
      `results/intrinsic_level6_q_curve.png` and `results/intrinsic_level6_q_sweep.png` are left
      untouched
- [ ] `results/history_level6_q_strength0_seed*.csv` and `*strength0p5_seed*.csv` (and their
      matching `qtable_*` files) are left untouched — these match the current config
- [ ] Full test suite still passes after deletion (`pytest`)

## Notes

Files likely touched: `results/history_level6_q_strength0p1_seed*.csv`,
`results/history_level6_q_strength0p25_seed*.csv`, `results/history_level6_q_strength1_seed*.csv`,
and the matching `results/qtable_level6_q_strength{0p1,0p25,1}_seed*.npz`. Nothing in `src`/`train`/
`eval` references these filenames directly — confirm with a grep before deleting, since A3-024's
Part I report draft cites `intrinsic_level6_q_sweep.png`, not the raw per-seed files, so the report
is unaffected.

## Log

- 2026-08-29 — created from a rubric audit finding: results/ holds artefacts from a 5-strength
  sweep (0, 0.1, 0.25, 0.5, 1) but the config now only declares 2 strengths (0.0, 0.5); the other
  3 strengths' raw files are unreproducible from the current config and a citation risk.
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/30
