---
id: A3-009
title: Build the arena renderer, HUD and observation overlay
type: feature
status: open
priority: P0
rubric: G
points_at_risk: 4.5
area: arena
owner: unassigned
blocks: [A3-018]
blocked_by: [A3-008]
created: 2026-08-19
updated: 2026-08-19
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
- [ ] Observation overlay toggle drawing lines to nearest enemy and spawner plus the local heading
- [ ] Phase-transition banner visible on screen
- [ ] Purely visual effects only — nothing here may alter simulation state

## Notes

Files: `arena/render.py`. Called only from `ArenaEnv.render()`.

## Log

- 2026-08-19 — created during project setup
