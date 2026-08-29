---
id: A3-022
title: Tune phase 1 so it is clearable in 20-30 seconds
type: task
status: open
priority: P1
rubric: none
points_at_risk: 0
area: arena
github: https://github.com/TDuong04/A3_TDuong/issues/23
owner: member-2
blocks: [A3-012, A3-015]
blocked_by: [A3-023]
created: 2026-08-29
updated: 2026-08-29
---

# A3-022 — Tune phase 1 so it is clearable in 20-30 seconds

## Context

Both the video (row V) and A3-012's acceptance require a visible phase progression. If phase 1
takes four minutes of perfect play to clear, no agent will clear it inside an episode and no clip
will show it — and the fix at that point is retraining, which is the expensive way to discover a
config problem.

This is currently one sentence in A3-012's notes. It is config work that can be settled before
training starts, using `--human` mode from A3-023 as the measuring instrument.

## Acceptance criteria

- [ ] A competent human clears phase 1 in 20-30 seconds, measured over 5 attempts, both styles
- [ ] The numbers changed are spawner count, spawner health, spawn interval and enemy health in
      `config/arena.yaml` only — no constant in `arena/constants.py` is touched
- [ ] `MAX_EPISODE_STEPS` allows at least three phases at that pace
- [ ] A random policy still fails to clear phase 1, so the task is not trivially easy
- [ ] The before and after numbers are recorded here for the report's env-design paragraph

## Notes

Frozen rewards and brief-fixed mechanics stay in `constants.py` and are test-guarded; only
tunables move. If a test fails, the change was in the wrong file.

## Log

- 2026-08-29 — created; promoted out of A3-012's notes into its own ticket
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/23
