---
id: A3-030
title: Add arena policy and reward debug overlay
type: feature
status: open
priority: P1
rubric: Creativity
points_at_risk: 5.0
area: arena
part: 2
github: https://github.com/TDuong04/A3_TDuong/issues/29
owner: member-2
blocks: []
blocked_by: [A3-023]
created: 2026-08-29
updated: 2026-08-29
---

# A3-030 — Add arena policy and reward debug overlay

## Context

Priority derivation: P1 — Creativity is a rubric row worth 5 points (row shared with A3-018, so
the index counts it once), below the ≥3-point-blocking P0 threshold, so P1 per the board's own
rule.

`arena/render.py:175-393` already ships the observation-overlay panel and a HUD with phase,
health, action name and step count — that visualisation is not being rebuilt here. What a marker
still cannot see without a debugger is (1) whether the agent is following a learned policy or
acting randomly, and (2) which reward terms are firing and how much each contributes, moment to
moment. This ticket adds those two panels plus a physics-debug layer, all read-only on top of the
existing renderer.

This issue started as GitHub issue #29, hand-created with an empty body and no labels under the
title "A3-028: Add debug mode for Part 2". A3-028 was already taken locally by the Part II report
figures ticket, so this work was renumbered to A3-030 and the existing issue reused rather than
duplicated.

Owner note: assigned to member-2 provisionally, but member-2 already owns A3-023 and A3-022, both
on the critical path (A3-023 blocks A3-012 and A3-022; A3-022 blocks A3-012 and A3-015). Loading a
fourth ticket onto the same person may need reassigning once A3-017 settles ownership — raise this
at the next stand-up rather than defaulting to it silently.

## Acceptance criteria

- [ ] Toggled by a dedicated key distinct from the existing `O` / `TAB` observation-overlay
      binding, and listed in the key legend
- [ ] Policy panel: per-action probabilities from the loaded SB3 model, drawn as one labelled bar
      per action name for the active control style, plus the value estimate `V(s)`. This is what
      proves "a learned policy, not random actions" for the video rubric.
- [ ] States honestly on screen when no model is loaded and the policy is random, rather than
      drawing a misleading uniform distribution unlabelled
- [ ] Reward decomposition panel: which reward terms fired this step with their values, plus
      cumulative per-term totals for the episode, reading the weights from
      `arena/constants.py:46-56` rather than restating them. This doubles as the evidence A3-014
      needs to justify each weight per term, and is the fastest way to see reward hacking as it
      happens.
- [ ] Physics debug: collision radii drawn for player, enemies, spawners and bullets; the
      spawner spawn-timer countdown; and enemy target lines
- [ ] Pause and single-step
- [ ] Read-only: a test asserts identical `env.step()` outputs for the same seed and action
      sequence with the debug overlay on versus off. Cross-reference this against A3-018's
      non-mutation criterion (visual effects provably never mutating simulation state) — one test
      can satisfy both; it should not be built twice.
- [ ] Renders headless under `SDL_VIDEODRIVER=dummy` with no window

## Notes

Builds on, does not replace: `arena/render.py:175-393` (observation overlay, phase/health/
action/step HUD).

Sequencing: the policy-probability panel is the only part needing a trained model, so it lands
after A3-012. The reward-decomposition and physics-debug panels are buildable and demonstrable
against a random policy today and do not need to wait.

`eval/play_arena.py` is the host script for this overlay and is currently a stub (hence
`blocked_by: [A3-023]`) — do not build a second entry point.

Cross-reference A3-018 (the creativity row this feeds, and its shared non-mutation test), A3-026
(the video shot list should include a debug-overlay moment showing the policy bars and reward
decomposition live), and A3-014's reward-justification section (the per-term cumulative totals
this panel prints are exactly the evidence that section needs, per weight, without re-deriving
them by hand).

Files likely touched: `eval/play_arena.py`, `arena/render.py` (read-only use), `tests/`.

## Log

- 2026-08-29 — created: refined from GitHub issue #29, which was hand-created empty with no body
  and no labels under a title colliding with the already-taken A3-028. Renumbered to A3-030
  (A3-028 stays with the Part II report-figures ticket) and the existing issue reused rather than
  duplicated. Set `blocked_by: [A3-023]` since `eval/play_arena.py` is the host script and is
  currently a stub. Owner set to member-2 with a reassignment flag noted above.
