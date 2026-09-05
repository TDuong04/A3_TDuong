---
id: A3-021
title: Build the core arena renderer
type: feature
status: done
priority: P0
rubric: G
points_at_risk: 4.5
area: arena
owner: unassigned
blocks: [A3-009, A3-022, A3-023]
blocked_by: [A3-008]
created: 2026-09-05
updated: 2026-09-05
---

# A3-021 — Build the core arena renderer

## Context

Replace the ArenaRenderer stub with the visual foundation for parent A3-009 (#9).
The existing ArenaEnv already owns simulation state and calls ArenaRenderer(), draw(env), and close().
Both control styles must use this same renderer; no trained model is needed to demonstrate it.

Parent: [A3-009](A3-009-build-the-arena-renderer-hud-and.md), mirrored as
[GitHub #9](https://github.com/TDuong04/A3_TDuong/issues/9).
The rubric points are shared with the parent, not additional marks.

## Acceptance criteria

- [x] Implement the existing ArenaRenderer(), draw(env), and close() contract; ArenaEnv.render() displays a frame and close() releases its display resources.
- [x] Draw a heading-aligned player ship, enemies, spawners, and bullets as clearly distinguishable shapes at their simulation positions.
- [x] Support an offscreen surface for automated rendering checks without opening a window; keep display ownership explicit.
- [x] Provide a small manual rendering demonstration with seeded scripted actions for both control styles, including spawning, movement, shooting, and collisions.
- [x] Repeated draw calls leave entity state, environment counters, and the environment RNG state unchanged; step() never invokes rendering.
- [x] Add meaningful checks for visible entity pixels and heading changes; the existing real-renderer integration test in tests/test_arena_env.py runs instead of skipping.

## Notes

Primary file: arena/render.py. Reference gridworld/render.py for surface lifecycle and headless rendering.
Add focused checks in tests/test_arena_render.py. A small demonstration may live in this module;
full trained-policy evaluation and human controls remain in A3-012.
Health bars/HUD belong to A3-022; observation graphics belong to A3-023.
Particles, muzzle flashes, and screen shake remain in A3-018.

## Log

- 2026-09-05 — created by splitting A3-009 into three implementation tasks at the user's request.
- 2026-09-05 — GitHub mirror pending: the connected account returns 404 for the parent issue.
- 2026-09-05 — implemented core ArenaRenderer with the existing draw/close contract,
  offscreen surfaces, explicit single-window ownership, heading-aligned ship, enemy circles,
  spawner squares, and heading-aligned projectiles. No simulation or configuration changes.
- 2026-09-05 — added seeded scripted demos: `python -m arena.render --style direct --seed 0`
  and `python -m arena.render --style rotation --seed 0`. Both exercise movement, spawning,
  shooting, and collisions through env.step(); ESC/window close exits. Offscreen runs use
  `--headless --frames 1200`. These controllers are explicitly labeled as scripted, not trained.
- 2026-09-05 — verified 85 focused tests and the full suite: 602 passed, none skipped.
  New pixel assertions cover shapes, positions, heading, clearing dead entities, and bullet
  orientation. Full environment/entity/RNG snapshots and paired seeded trajectories prove
  drawing does not affect simulation. Display ownership, presentation, and cleanup verified
  with SDL's dummy driver; offscreen frames from both styles inspected visually.
  Dependencies installed from requirements.txt into /private/tmp/a3-render-venv.
  No desktop-window manual session was performed. No commit or push made.
