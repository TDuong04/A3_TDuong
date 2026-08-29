---
id: A3-008
title: Build arena entities and physics
type: feature
status: done
priority: P0
rubric: G
points_at_risk: 4.5
area: arena
part: 2
github: https://github.com/TDuong04/A3_TDuong/issues/8
owner: unassigned
blocks: [A3-009, A3-010]
blocked_by: []
created: 2026-08-19
updated: 2026-08-20
---

# A3-008 — Build arena entities and physics

## Context

Row G is the largest single row in the whole rubric and needs no machine learning. Build and
bank it before touching Stable Baselines3.

Fixed timestep is non-negotiable: wall-clock delta time makes training and evaluation diverge
silently and destroys reproducibility.

## Acceptance criteria

- [x] Player, Enemy, Spawner and Bullet integrate against `FIXED_DT`, never wall-clock
- [x] Player supports both control styles over identical physics
- [x] Enemies navigate toward the player; spawners emit on an interval and stop when destroyed
- [x] Circle-vs-circle collisions; live bullet count is capped
- [x] Player invulnerability window after a hit
- [x] No pygame import anywhere in this module

## Notes

Files: `arena/entities.py`. Tunables live in `config/arena.yaml`.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/8
- 2026-08-20 - implemented in `arena/entities.py` (590 lines): `Entity` -> `DestructibleEntity` ->
  `Player`/`Enemy`/`Spawner`, plus `Bullet`, with frozen config dataclasses read from
  `config/arena.yaml`. New `entities:` block added there for radii, damage, bullet lifetime and the
  bullet cap; no existing value changed. 66 new tests, suite 379 -> 445.
- 2026-08-20 - evaluated: PASS on all six criteria, verified by execution rather than by reading
  the tests. Both control styles proven to share one integrator structurally, not by fixture
  coincidence - `thrust`/`rotate` and `move` only write acceleration, and there is exactly one
  `_integrate()`. Five mutants designed independently of the implementer's own nine, aimed at
  control-style equivalence, the drag/dt interpretation, the spawner timer epsilon and the
  collision boundary in the directions the suite's own fixtures never construct: all five killed,
  none survived. No symmetric-fixture collapse of the kind that let a wrong SARSA target pass 216
  tests earlier in this project - the physics tests pin exact float literals.
- 2026-08-20 - playability confirmed by headless simulation, which matters because a broken arena
  costs row G and every deep-RL row after it: top speed 220 px/s reached in 35 frames, arena
  crossed in 4.52s, player outruns a phase-3 enemy in open space (gap 3.1px -> 279px and widening),
  and a phase-1 spawner dies at frame 49, before its first enemy emission at 3.0s. Closed.
