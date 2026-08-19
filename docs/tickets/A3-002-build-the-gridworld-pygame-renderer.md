---
id: A3-002
title: Build the gridworld Pygame renderer
type: feature
status: open
priority: P0
rubric: A1
points_at_risk: 2.0
area: gridworld
owner: unassigned
blocks: [A3-015]
blocked_by: [A3-001]
created: 2026-08-19
updated: 2026-08-19
---

# A3-002 — Build the gridworld Pygame renderer

## Context

Rubric A1 requires the gridworld to be visually rendered, animated and interactive. Console
or text output scores zero, so this is worth 2 points on its own and needs no machine learning.

The policy-arrow overlay is not decoration: it is how "learned a shortest-path policy" gets
demonstrated in both the report and the video.

## Acceptance criteria

- [ ] Simple shapes per tile type, agent visually distinct while holding the key
- [ ] Agent animates between cells rather than teleporting
- [ ] Policy-arrow overlay toggles on a keypress
- [ ] Q-value heatmap overlay toggles on a keypress
- [ ] Keyboard playback control: pause, single-step, speed, reset, switch level
- [ ] Human-play mode for testing the env by hand

## Notes

Files: `gridworld/render.py`. Nothing here may be imported from `env.py`.

## Log

- 2026-08-19 — created during project setup
