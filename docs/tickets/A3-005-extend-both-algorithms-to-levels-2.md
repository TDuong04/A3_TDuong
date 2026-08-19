---
id: A3-005
title: Extend both algorithms to levels 2-3
type: task
status: open
priority: P0
rubric: D
points_at_risk: 3.0
area: gridworld
owner: unassigned
blocks: [A3-006, A3-007]
blocked_by: [A3-003, A3-004]
created: 2026-08-19
updated: 2026-08-19
---

# A3-005 — Extend both algorithms to levels 2-3

## Context

Levels 2-3 add multiple apples, a key and a chest, which is where the state representation
stops being just a position and the ordering of objectives starts to matter.

## Acceptance criteria

- [ ] Both algorithms solve levels 2 and 3 with correct termination
- [ ] Reward accounting matches the brief exactly across a full episode
- [ ] Agent learns to collect the key before the chest
- [ ] Training curves for both levels saved to `results/`

## Notes

Depends on the state key from A3-001 being correct.

## Log

- 2026-08-19 — created during project setup
