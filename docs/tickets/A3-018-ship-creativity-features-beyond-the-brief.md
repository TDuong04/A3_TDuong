---
id: A3-018
title: Ship creativity features beyond the brief
type: feature
status: open
priority: P1
rubric: Creativity
points_at_risk: 5.0
area: arena
part: 2
github: https://github.com/TDuong04/A3_TDuong/issues/18
owner: member-2
blocks: []
blocked_by: []
created: 2026-08-19
updated: 2026-08-29
---

# A3-018 — Ship creativity features beyond the brief

## Context

5 points — tied for the largest single row in the rubric.

A rubric audit (2026-08-29) found that every "extra" currently on disk — the gridworld Q-value
heatmap and policy-arrow overlays, the TAB algorithm swap, the arena observation-debug HUD, and
the Welch confidence intervals in the comparison stats — is polish on a feature the brief already
requires, not something that goes beyond it. The team has been implicitly treating these as
already earning this row; they do not. This is currently the least-defended row on the board and
needs work that is demonstrably outside A3_Brief.pdf.

## Acceptance criteria

- [x] Observation overlay shipped and toggleable (shared with A3-009, done) — required polish, not
      itself sufficient for this row; kept as a supporting item, not the answer.
- [ ] At least two mechanics or systems added that do not appear anywhere in A3_Brief.pdf. Pick two
      (or three) and finish them rather than half-building six. Candidates: a boss or elite enemy
      with a distinct attack pattern; a second arena layout with a different spawner/obstacle
      arrangement; a pickup or power-up (e.g. shield, rapid-fire, speed boost) the player can
      collect mid-run; an enemy type with different navigation (e.g. flanking or ranged kiting
      instead of direct pursuit). Final choice is the owner's; log which two were built.
- [ ] `eval/play_arena.py --human` is playable on the identical action space used for training
      (this is A3-023's scope — do not rebuild it here, just confirm it and use it as the vehicle
      for demonstrating the new mechanics)
- [ ] Visual feedback effects shipped — muzzle flash, hit flicker, explosion particles, screen
      shake — each provably never mutating simulation state (e.g. a test or an assertion showing
      identical `env.step()` outputs with rendering on vs. off for the same seed and action
      sequence)
- [ ] One paragraph in the report (feeds A3-014) naming each shipped item individually and
      justifying, per item, why it goes beyond the brief rather than restating the brief's
      requirements

## Notes

Purely visual effects must never touch simulation state, or evaluation stops matching training.
Ideas beyond the candidates above are welcome — raise them as separate P3 tickets rather than
scope-creeping this one.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/18
- 2026-08-29 — unblocked — A3-009 is done and the observation overlay already ships. Assigned to member-2
- 2026-08-29 — priority P2 -> P1 (5-point row, tied for largest in the rubric, currently the
  least-defended). Acceptance criteria rewritten: rubric audit found every existing "extra" is
  polish on a required feature, not beyond-brief work — see Context. Old wishlist-style items
  replaced with concrete, checkable criteria naming specific beyond-brief candidates, the shared
  `--human` mode (A3-023), non-mutating visual effects, and a report paragraph per item.
