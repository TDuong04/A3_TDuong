---
id: A3-008
title: Build arena entities and physics
type: feature
status: open
priority: P0
rubric: G
points_at_risk: 4.5
area: arena
github: https://github.com/TDuong04/A3_TDuong/issues/8
owner: unassigned
blocks: [A3-009, A3-010]
blocked_by: []
created: 2026-08-19
updated: 2026-08-19
---

# A3-008 — Build arena entities and physics

## Context

Row G is the largest single row in the whole rubric and needs no machine learning. Build and
bank it before touching Stable Baselines3.

Fixed timestep is non-negotiable: wall-clock delta time makes training and evaluation diverge
silently and destroys reproducibility.

## Acceptance criteria

- [ ] Player, Enemy, Spawner and Bullet integrate against `FIXED_DT`, never wall-clock
- [ ] Player supports both control styles over identical physics
- [ ] Enemies navigate toward the player; spawners emit on an interval and stop when destroyed
- [ ] Circle-vs-circle collisions; live bullet count is capped
- [ ] Player invulnerability window after a hit
- [ ] No pygame import anywhere in this module

## Notes

Files: `arena/entities.py`. Tunables live in `config/arena.yaml`.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/8
