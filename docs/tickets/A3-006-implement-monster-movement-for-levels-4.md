---
id: A3-006
title: Implement monster movement for levels 4-5
type: feature
status: done
priority: P1
rubric: none
points_at_risk: 0.0
area: gridworld
part: 1
github: https://github.com/TDuong04/A3_TDuong/issues/6
owner: unassigned
blocks: []
blocked_by: [A3-005]
created: 2026-08-19
updated: 2026-08-20
---

# A3-006 — Implement monster movement for levels 4-5

## Context

Required by the brief but — worth knowing — it carries no rubric row of its own. It still
shows up through the level 4-5 training curves and the video, so it cannot be skipped, but it should
not displace work on rows that do carry points.

The two-sided collision check is the part that gets missed: the player dies both when it steps onto
a monster and when a monster steps onto it.

## Acceptance criteria

- [x] Each monster moves after each agent action with probability 0.4
- [x] Movement chooses uniformly among directions that are not rocks or off-grid
- [x] Death is detected both when the agent enters a monster tile and when a monster enters the agent's tile
- [x] Q-learning and SARSA both handle the resulting stochastic transitions
- [x] Training curves for levels 4 and 5 saved to `results/`

## Notes

Stochastic levels need more episodes; overrides are already in `config/gridworld.yaml`.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/6
- 2026-08-20 — PASS. Committed in 4322e27. Evaluator re-measured monster movement at 0.40013 on both levels over 200,000 observations each (z=+0.11), confirmed uniformity with a monster confined to two legal neighbours (0.2001/0.1996, staying put 0.6003) and zero illegal landings, and reproduced both death checks with the agent provably stationary. Non-monotone budget recovery reproduced and found worse than reported: 0% success, not partial degradation.
