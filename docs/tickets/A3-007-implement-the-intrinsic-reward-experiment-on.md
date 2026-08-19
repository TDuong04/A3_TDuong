---
id: A3-007
title: Implement the intrinsic reward experiment on level 6
type: feature
status: open
priority: P0
rubric: F
points_at_risk: 3.0
area: gridworld
github: https://github.com/TDuong04/A3_TDuong/issues/7
owner: unassigned
blocks: [A3-014]
blocked_by: [A3-005]
created: 2026-08-19
updated: 2026-08-19
---

# A3-007 — Implement the intrinsic reward experiment on level 6

## Context

Level 6 is deliberately exploration-hostile — a distant chest behind a long corridor with the
key down a dead-end branch — so a visit-count bonus has a real problem to solve and the comparison
curve shows a visible difference.

## Acceptance criteria

- [ ] `r_i = strength / sqrt(n(s) + 1)` implemented exactly as the brief states
- [ ] `n(s)` counts visits within the current episode and resets on reset
- [ ] Environment rewards unchanged; the intrinsic term is added only inside the update
- [ ] Two runs on level 6 at strengths 0.0 and 0.5, plotted on one axis
- [ ] Short written explanation of the improvement for the report

## Notes

Config block `intrinsic_experiment` already exists in `config/gridworld.yaml`.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/7
