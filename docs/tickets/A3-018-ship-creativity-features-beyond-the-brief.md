---
id: A3-018
title: Ship creativity features beyond the brief
type: feature
status: open
priority: P1
rubric: Creativity
points_at_risk: 5.0
area: arena
github: https://github.com/TDuong04/A3_TDuong/issues/18
owner: unassigned
blocks: []
blocked_by: [A3-014]
created: 2026-08-19
updated: 2026-09-10
---

# A3-018 — Ship creativity features beyond the brief

## Context

The Creativity row requires additions beyond the brief. Observation overlays and visual
polish support the demonstration but do not establish two additional gameplay systems.

## Acceptance criteria

- [x] Observation overlay shipped and toggleable (shared with A3-009; supporting evidence)
- [x] Two additional mechanics: collectible one-hit shields and elite chargers with a distinct
      pursuit/wind-up/charge/recovery attack pattern; enable with `--mechanics`
- [x] Existing `eval/play_arena.py --human` uses the same action space and mechanics as training
- [x] Muzzle flash, hit flicker, explosion particles and screen shake remain renderer-owned;
      seeded rendered/unrendered tests cover both baseline and new mechanics
- [ ] Final report contains a paragraph naming each item and explaining why it exceeds the brief
      (section draft available in `report/creativity.md`; integration belongs to A3-014)

## Notes

Purely visual effects must never touch simulation state, or evaluation stops matching training. Ideas beyond these are welcome — raise them as separate P3 tickets.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/18
- 2026-09-08 — Integrated creativity features with completed A3-009/A3-012 on main: retain HUD, phase, observation, policy and trained/human playback; add renderer-owned combat feedback and pilot-perception compass. See docs/arena-creativity.md for integration differences, controls and originality rationale.
- 2026-09-08 — Verification: 65 focused and 721 full-suite tests passed; both shipped PPO models reached phase 2 at seed 0; offscreen playback inspected for both styles. Desktop keyboard testing remains manual.

- 2026-09-10 — Restored the stricter two-systems criteria after review. Implemented shield
  pickups and elite chargers in optional mechanics mode for both human and agent play.
  Added 36-feature observations, training/evaluation integration, distinct artifact directories,
  and gameplay/invariance/model compatibility tests. See `docs/arena-mechanics.md`.
  Reopened pending final report integration; existing trained models remain baseline models.

- 2026-09-10 — Integrated latest main (`c65acb2`) on the A3-018 branch, preserving its
  retrained 20-feature PPO models. Mechanics mode now has 36 features. Kept A3-030
  debugging separate. Verification: 286 arena tests passed; both shipped baseline
  models loaded and rendered, and mechanics PPO smoke training/save/reload passed
  for both control styles. Full-suite rerun was interrupted; no full-suite result
  is claimed for this integration.
