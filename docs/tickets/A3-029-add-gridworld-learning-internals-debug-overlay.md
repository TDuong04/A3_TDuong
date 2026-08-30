---
id: A3-029
title: Add gridworld learning-internals debug overlay
type: feature
status: open
priority: P1
rubric: Creativity
points_at_risk: 5.0
area: gridworld
part: 1
github: https://github.com/TDuong04/A3_TDuong/issues/28
owner: unassigned
blocks: []
blocked_by: []
created: 2026-08-29
updated: 2026-08-29
---

# A3-029 — Add gridworld learning-internals debug overlay

## Context

Priority derivation: P1 — Creativity is a rubric row worth 5 points (row shared with A3-018, so
the index counts it once), below the ≥3-point-blocking P0 threshold and not blocking anyone else's
work, so P1 per the board's own rule.

CLAUDE.md's VISIBILITY FIRST golden rule states prior assignments lost marks because algorithm
logic was only observable with a debugger. `gridworld/render.py:314-511` already ships substantial
visualisation — policy arrows, a Q-value heatmap, the TAB algorithm swap between Q-learning and
SARSA tables, human play, and an on-screen key legend. None of that is being rebuilt here. What is
still invisible to a marker without a debugger is the algorithm's internal update math: the exact
state/action/reward transition just taken, the Q-value before and after the update, the TD target
and error, and — for row F — the intrinsic-reward computation. This ticket adds a second overlay
panel for exactly that, and nothing else.

This issue started as GitHub issue #28, hand-created with an empty body and no labels under the
title "A3-027: Add debug mode for Part 1". A3-027 was already taken locally by the orphaned
level-6 artefact cleanup ticket, so this work was renumbered to A3-029 and the existing issue
reused rather than duplicated.

## Acceptance criteria

- [ ] Toggled by a dedicated key that does not collide with existing bindings (`SPACE`, `N`, `+/-`,
      `R`, `0-6`, `P`, `Q/H`, `TAB`, `WASD`/arrows, `ESC`), and listed in the on-screen key legend
- [ ] For the most recent step, displays: state `s`, action `a`, reward `r`, next state `s'`, the
      chosen action's Q-value before and after the update, the TD target, the TD error, and the
      current epsilon, alpha and gamma
- [ ] The TD target line renders the algorithm-specific term explicitly and labelled —
      `max_a' Q(s',a') = …` for Q-learning versus `Q(s', a'_chosen) = …` for SARSA — so the
      off-policy/on-policy distinction (rows B and C) is legible on screen without a debugger. This
      is the single most valuable item in this ticket: it is the one thing a marker cannot get from
      the existing arrow/heatmap overlays or from watching the agent move.
- [ ] Marks whether the action just taken was greedy or exploratory, showing the epsilon roll
      against the current epsilon
- [ ] When intrinsic reward is active (level 6), shows `n(s)` for the current cell, the computed
      `r_i = strength / sqrt(n(s)+1)`, and the environment reward and shaped reward as two separate
      labelled numbers, never conflated — this is row F's evidence made visible
- [ ] A per-cell episode visit-count heatmap, separately toggleable from the debug panel above
- [ ] Works inside the existing `--compare` TAB-swap mode: the panel follows whichever algorithm's
      table is currently displayed
- [ ] Pause and single-step (reusing the existing `SPACE`/`N` bindings), so a frame can be held
      while the numbers are read aloud on camera
- [ ] Renders headless under `SDL_VIDEODRIVER=dummy` across all seven levels with no window,
      covered by a test alongside the existing renderer tests
- [ ] Read-only: a test asserts identical environment transitions and identical Q-tables for the
      same seed with the overlay on versus off

## Notes

Builds on, does not replace: `gridworld/render.py:314-511` (policy arrows, Q-value heatmap, TAB
swap, human play, key legend) — reuse the existing font/panel plumbing rather than duplicating it.

Cross-reference A3-018 (the creativity row this feeds — a debug overlay is a beyond-brief mechanic
candidate, not required polish) and A3-026 (the video shot list should include a debug-overlay
moment demonstrating the on-policy/off-policy TD-target distinction on camera).

Files likely touched: `gridworld/render.py`, `eval/play_gridworld.py`, `tests/` (new headless
render test, new read-only/non-mutation test).

## Log

- 2026-08-29 — created: refined from GitHub issue #28, which was hand-created empty with no body
  and no labels under a title colliding with the already-taken A3-027. Renumbered to A3-029
  (A3-027 stays with the orphaned-artefact cleanup ticket) and the existing issue reused rather
  than duplicated.
