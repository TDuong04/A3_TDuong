---
id: A3-003
title: Implement Q-learning and demonstrate it on level 0
type: feature
status: done
priority: P0
rubric: B
points_at_risk: 2.5
area: gridworld
github: https://github.com/TDuong04/A3_TDuong/issues/3
owner: unassigned
blocks: [A3-004, A3-005]
blocked_by: [A3-001]
created: 2026-08-19
updated: 2026-08-20
---

# A3-003 — Implement Q-learning and demonstrate it on level 0

## Context

First learning agent. Level 0 is deliberately open with apples on the right, so the optimal
policy is a clean shortest path and the demonstration is unambiguous.

## Acceptance criteria

- [ ] Epsilon-greedy selection using the shared `LinearEpsilon` from `common.schedules`
- [ ] Off-policy update: target uses max over next-state Q, and is just `r` on terminal steps
- [ ] Random tie-breaking via `np.flatnonzero(...)` — `np.argmax` is a graded failure
- [ ] Epsilon bounds read from `config/gridworld.yaml`, no literals at the call site
- [ ] Learned policy reaches all level-0 apples by a shortest path, shown via policy arrows
- [ ] Per-episode history written to `results/` and a training curve plotted

## Notes

Files: `gridworld/algorithms.py`, `train/train_gridworld.py`.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/3
- 2026-08-20 — verified and committed in e77bcef; greedy rollout 17 steps against an independently computed BFS optimum of 17
