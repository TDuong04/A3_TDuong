---
id: A3-006
title: Implement monster movement for levels 4-5
type: feature
status: open
priority: P1
rubric: none
points_at_risk: 0.0
area: gridworld
owner: unassigned
blocks: []
blocked_by: [A3-005]
created: 2026-08-19
updated: 2026-08-19
---

# A3-006 — Implement monster movement for levels 4-5

## Context

Required by the brief but — worth knowing — it carries no rubric row of its own. It still
shows up through the level 4-5 training curves and the video, so it cannot be skipped, but it should
not displace work on rows that do carry points.

The two-sided collision check is the part that gets missed: the player dies both when it steps onto
a monster and when a monster steps onto it.

## Acceptance criteria

- [ ] Each monster moves after each agent action with probability 0.4
- [ ] Movement chooses uniformly among directions that are not rocks or off-grid
- [ ] Death is detected both when the agent enters a monster tile and when a monster enters the agent's tile
- [ ] Q-learning and SARSA both handle the resulting stochastic transitions
- [ ] Training curves for levels 4 and 5 saved to `results/`

## Notes

Stochastic levels need more episodes; overrides are already in `config/gridworld.yaml`.

## Log

- 2026-08-19 — created during project setup
