---
id: A3-024
title: Draft the Part I report sections and figures
type: task
status: open
priority: P1
rubric: R
points_at_risk: 2.5
area: report
part: 1
github: https://github.com/TDuong04/A3_TDuong/issues/25
owner: member-3
blocks: [A3-014]
blocked_by: []
created: 2026-08-29
updated: 2026-08-29
---

# A3-024 — Draft the Part I report sections and figures

## Context

`report/` holds nothing but a `.gitkeep`, and A3-014 is blocked behind the Part II training run —
so under the current board nobody can write a word of the report until Part II lands. That is
false: Part I is finished and every figure it needs is already sitting in `results/`.

Drafting the Part I half now converts the last week from "write ten pages" into "paste in the
Part II results", and it is the only report work that can start today.

## Acceptance criteria

- [ ] Environment and mechanics section written, citing `gridworld/constants.py` for the fixed
      rewards rather than restating numbers that could drift
- [ ] Q-learning vs SARSA section built on `results/comparison_level1_seed0.md` — the 11-step
      fire-hugging route against the 13-step safe route, with the figure embedded
- [ ] Levels 2-5 section reporting completion and death rates across seeds 0-2 from
      `results/summary_level*.json`
- [ ] Intrinsic reward section using the level 6 sweep — 5 strengths by 5 seeds, with
      `intrinsic_level6_q_sweep.png` and a stated conclusion about which strength won and why
- [ ] Every claim carries the seed and run that produced it
- [ ] Part I draft fits its share of the 10-page budget, images included

## Notes

Figures already exist in `results/`; do not regenerate them, and do not retrain to recover a plot.
Hands over to A3-014, which assembles the full document.

## Log

- 2026-08-29 — created; split the Part I half of the report out so it is not blocked on Part II
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/25
