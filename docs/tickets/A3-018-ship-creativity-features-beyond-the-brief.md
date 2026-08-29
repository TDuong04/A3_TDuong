---
id: A3-018
title: Ship creativity features beyond the brief
type: feature
status: open
priority: P2
rubric: Creativity
points_at_risk: 5.0
area: arena
github: https://github.com/TDuong04/A3_TDuong/issues/18
owner: member-2
blocks: []
blocked_by: []
created: 2026-08-19
updated: 2026-08-29
---

# A3-018 — Ship creativity features beyond the brief

## Context

5 points — tied for the largest single row in the rubric, and the cheapest to earn, because
it is graded on going beyond expectations rather than on correctness.

The observation debug overlay is the strongest single item and is already scoped inside A3-009: it
earns creativity marks, supplies the report's observation-design figure, and gives the video its
most persuasive moment. Everything below it is optional polish; pick two or three and finish them
rather than half-building six.

## Acceptance criteria

- [x] Observation overlay shipped and toggleable (shared with A3-009, done)
- [ ] Human-playable mode on the same env used for training
- [ ] Visual feedback: muzzle flash, hit flicker, explosion particles, screen shake
- [ ] At least one item the report can point at and call original, with a sentence justifying it

## Notes

Purely visual effects must never touch simulation state, or evaluation stops matching training. Ideas beyond these are welcome — raise them as separate P3 tickets.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/18
- 2026-08-29 — unblocked — A3-009 is done and the observation overlay already ships. Assigned to member-2
