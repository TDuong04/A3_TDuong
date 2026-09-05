---
id: A3-022
title: Add the arena HUD and phase feedback
type: feature
status: open
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

# A3-022 — Add the arena HUD and phase feedback

## Context

Make the arena state and progression legible for parent A3-009 (#9).
Read health, phase, step count, last action, and accumulated reward from the existing environment;
this task adds presentation, not a second scoring or phase system.

Parent: [A3-009](A3-009-build-the-arena-renderer-hud-and.md), mirrored as
[GitHub #9](https://github.com/TDuong04/A3_TDuong/issues/9).
The rubric points are shared with the parent, not additional marks.

## Acceptance criteria

- [ ] Draw player and living-spawner health bars using current health and maximum health; full, damaged, and zero-health states display correctly.
- [ ] Show phase, player health, agent step count, and the last action name for both rotation and direct controls.
- [ ] Use env.episode_reward for the parent ticket's score field and label it Return (cumulative reward); do not introduce an independent score accumulator.
- [ ] Show a visible phase-transition banner when env.phase advances; its duration and prior-phase tracking are renderer-owned visual state.
- [ ] Clear banner state on episode reset or environment replacement, so a new episode does not retain the previous episode's notification.
- [ ] Verify HUD changes against known environment values and inspect a frame at the intended window size for clipping, overlap, and legibility.
- [ ] Health bars, text, and banner timing never modify simulation state or use the environment RNG.

## Notes

Primary file: arena/render.py; focused checks in tests/test_arena_render.py.
Use env.action_name, env.steps, env.phase, env.episode_reward, and entity health properties.
Keep visual durations configurable when added. Any reset/lifecycle hook needed by the renderer
must preserve the existing headless training behavior. Capture representative frames for review.

## Log

- 2026-09-05 — created by splitting A3-009 into three implementation tasks at the user's request.
- 2026-09-05 — GitHub mirror pending: the connected account returns 404 for the parent issue.
