---
id: A3-009
title: Build the arena renderer, HUD and observation overlay
type: feature
status: open
priority: P0
rubric: G
points_at_risk: 4.5
area: arena
github: https://github.com/TDuong04/A3_TDuong/issues/9
owner: unassigned
blocks: [A3-018]
blocked_by: [A3-008, A3-021, A3-022, A3-023]
created: 2026-08-19
updated: 2026-09-05
---

# A3-009 — Build the arena renderer, HUD and observation overlay

## Context

Completes row G and supplies most of the video's Part II footage.

The observation overlay pays three times over: creativity marks, the report's observation-design
figure, and the most persuasive thirty seconds of the video, because it shows the agent reacting to
something specific rather than flailing.

## Acceptance criteria

- [ ] Clear shapes for ship, enemies, spawners and bullets; health bars on player and spawners
- [ ] HUD showing phase, health, score, step count and current action name
- [x] Observation overlay toggle drawing lines to nearest enemy and spawner plus the local heading
- [ ] Phase-transition banner visible on screen
- [ ] Purely visual effects only — nothing here may alter simulation state

## Subtasks

Complete A3-021 first; A3-022 and A3-023 build on its rendering lifecycle.

- [x] [A3-021 — Build the core arena renderer](A3-021-build-the-core-arena-renderer.md)
- [ ] [A3-022 — Add the arena HUD and phase feedback](A3-022-add-the-arena-hud-and-phase-feedback.md)
- [x] [A3-023 — Add the arena observation overlay](A3-023-add-the-arena-observation-overlay.md)

Keep this parent open until all three subtasks and the original acceptance criteria pass together
through ArenaEnv.render(), for both control styles. Visual polish remains in A3-018;
trained-policy evaluation and human-play integration remain in A3-012.
The HUD score is the existing cumulative episode reward, labeled Return; no new scoring rule is added.

## Notes

Files: `arena/render.py`. Called only from `ArenaEnv.render()`.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/9
- 2026-09-05 — split implementation into A3-021 (core renderer), A3-022 (HUD/phase feedback), and A3-023 (observation overlay); parent scope and acceptance criteria retained. GitHub synchronization pending repository access.
- 2026-09-05 — A3-021 core renderer implemented and verified (602 tests passed). Parent remains open for A3-022 HUD/phase feedback and A3-023 observation overlay.
- 2026-09-05 — A3-018 completed A3-023 observation overlay and renderer-owned combat feedback, with simulation/RNG invariance checks. Parent remains open for A3-022 health bars, HUD and phase banners.
