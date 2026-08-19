---
id: A3-004
title: Implement SARSA and produce the level 1 comparison
type: feature
status: open
priority: P0
rubric: C
points_at_risk: 3.0
area: gridworld
owner: unassigned
blocks: [A3-005]
blocked_by: [A3-003]
created: 2026-08-19
updated: 2026-08-19
---

# A3-004 — Implement SARSA and produce the level 1 comparison

## Context

SARSA differs from Q-learning in exactly one term, and the assignment wants evidence that
this changes behaviour. Level 1 is a cliff-walk built for this: fire between start and goal, so
Q-learning learns the fast edge route and SARSA, accounting for its own exploration, detours.

The comparison figure is also the strongest thing to show in the video for Part I.

## Acceptance criteria

- [ ] On-policy update uses the action actually taken next, never a max
- [ ] Uses the same `LinearEpsilon` instance construction as Q-learning — provable by inspection
- [ ] Both algorithms run on level 1 under identical seeds and schedules
- [ ] Figure showing both greedy policies over the cliff, saved to `results/`
- [ ] Short written comparison of the two routes for the report

## Notes

Files: `gridworld/algorithms.py`, `eval/play_gridworld.py` (`--compare` mode).

## Log

- 2026-08-19 — created during project setup
