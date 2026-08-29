---
id: A3-026
title: Storyboard the video and schedule the recording session
type: task
status: open
priority: P0
rubric: V
points_at_risk: 5.0
area: video
part: both
github: https://github.com/TDuong04/A3_TDuong/issues/27
owner: member-3
blocks: [A3-015]
blocked_by: []
created: 2026-08-29
updated: 2026-08-29
---

# A3-026 — Storyboard the video and schedule the recording session

## Context

A3-015 is worth 5 points, needs all three members on camera, and is blocked to the very end of
the project — which is exactly how a group video ends up recorded at midnight with one member
missing and half the required moments not shown.

The planning half does not depend on Part II. Doing it now means the recording session is an hour
of reading a shot list, not an evening of deciding what to say.

## Acceptance criteria

- [ ] Shot list under 10 minutes total, with a time budget per segment
- [ ] Every rubric V moment mapped to a specific shot: gridworld window with correct item and
      monster behaviour, evidence of a learned policy via the arrow overlay, arena with enemies,
      projectiles, collisions and one phase progression, both control schemes
- [ ] Each of the three members assigned at least one segment to present, by name
- [ ] The exact command to type on camera written next to each shot, verified to run
- [ ] A recording date agreed with all three members, at least five days before the 19 September
      deadline
- [ ] Screen recording tool chosen and tested once, audio included

## Notes

The Part I shots are recordable today with `python -m eval.play_gridworld --level 1 --compare`.
Record them early rather than waiting for Part II — a banked segment is a segment that cannot go
wrong. Hands over to A3-015 for the recording and edit.

## Log

- 2026-08-29 — created; the planning half of A3-015, which needs no Part II dependency
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/27
- 2026-08-29 — priority P1 -> P0: it blocks A3-015, a P0 ticket worth 5 rubric points, and is the
  only V-row work that can be done before Part II lands. Blocking a P0 makes this P0 under the
  board's own rule.
