---
id: A3-023
title: Add the arena observation overlay
type: feature
status: done
priority: P0
rubric: G
points_at_risk: 4.5
area: arena
owner: unassigned
blocks: [A3-009]
blocked_by: [A3-021]
created: 2026-09-05
updated: 2026-09-05
---

# A3-023 — Add the arena observation overlay

## Context

Explain the numeric observation presented to the agent, completing parent A3-009 (#9).
The 21-feature vector and ship-local coordinate transform already exist in arena/observation.py.
The overlay must reflect that implementation rather than maintaining a separate observation model.

Parent: [A3-009](A3-009-build-the-arena-renderer-hud-and.md), mirrored as
[GitHub #9](https://github.com/TDuong04/A3_TDuong/issues/9).
The rubric points are shared with the parent, not additional marks.

## Acceptance criteria

- [x] Expose a toggle_observation_overlay() method and initialize visibility from evaluation.show_observation_overlay in config/arena.yaml.
- [x] Wire a documented keyboard toggle in the renderer demonstration; event handling remains with the application loop so later evaluation can reuse it.
- [x] Draw distinguishable lines from the player to the nearest living enemy and nearest living spawner selected using the existing observation helpers.
- [x] Draw and label ship-local forward and lateral axes consistent with to_ship_local(); local +x points along the nose and local +y is clockwise on screen.
- [x] Display relevant normalized observation values, including health/cooldown and target offsets/distances, using feature names from describe() rather than a duplicate index definition.
- [x] Clear target lines and values when no living target exists; coincident targets and empty scenes render without errors or stale markers.
- [x] Check overlay alignment at nonzero headings and off-axis targets, toggle behavior in both directions, and agreement with env.observation(); rendering leaves simulation and RNG state unchanged.

## Notes

Primary file: arena/render.py; reuse arena/observation.py helpers and env.observation().
Use tests/test_arena_render.py for focused pixel and state checks. The renderer may store overlay
visibility but must not advance physics. Demonstrate the same overlay under both control styles.
A3-012 owns integration into the final evaluation application; A3-018 may reuse this completed feature.

## Log

- 2026-09-05 — created by splitting A3-009 into three implementation tasks at the user's request.
- 2026-09-05 — GitHub mirror pending: the connected account returns 404 for the parent issue.
- 2026-09-05 — Completed alongside A3-018: config-default O toggle, nearest living target links, local axes, normalized values from describe()/observation(), and ship-local perception compass. Focused checks cover nonzero headings, absent/coincident targets, toggling, and state/RNG invariance. Full suite: 621 passed.
