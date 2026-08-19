---
id: A3-005
title: Extend both algorithms to levels 2-3
type: task
status: done
priority: P0
rubric: D
points_at_risk: 3.0
area: gridworld
github: https://github.com/TDuong04/A3_TDuong/issues/5
owner: unassigned
blocks: [A3-006, A3-007]
blocked_by: [A3-003, A3-004]
created: 2026-08-19
updated: 2026-08-20
---

# A3-005 — Extend both algorithms to levels 2-3

## Context

Levels 2-3 add multiple apples, a key and a chest, which is where the state representation
stops being just a position and the ordering of objectives starts to matter.

## Acceptance criteria

- [x] Both algorithms solve levels 2 and 3 with correct termination
- [x] Reward accounting matches the brief exactly across a full episode
- [x] Agent learns to collect the key before the chest
- [x] Training curves for both levels saved to `results/`

## Notes

Depends on the state key from A3-001 being correct.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/5
- 2026-08-20 — PASS. Committed in 4322e27. Evaluator independently reproduced the level-2 collection order on all six runs (apple, key at step 11, apple, chest at step 21, apple; 25 steps, 5.0/5.0) and confirmed the chest is genuinely key-gated by walking to it without the key: 0.0 reward, chest still on the grid. Levels 0-3 re-verified unregressed. All six production mutants killed.
