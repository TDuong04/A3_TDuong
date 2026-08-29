---
id: A3-013
title: Run the hyperparameter sweep and tabulate the results
type: task
status: open
priority: P0
rubric: J3
points_at_risk: 3.0
area: training
part: 2
github: https://github.com/TDuong04/A3_TDuong/issues/13
owner: member-1
blocks: []
blocked_by: [A3-011]
created: 2026-08-19
updated: 2026-08-29
---

# A3-013 — Run the hyperparameter sweep and tabulate the results

## Context

Defaults-only training is explicitly marked down, and the report needs tuning evidence as
tables or plots. The sweep axes are already listed in `config/arena.yaml`.

## Acceptance criteria

- [ ] One axis varied at a time from the documented baseline at 100k timesteps
- [ ] Top two configs re-run on 3 seeds — a one-seed win is not a result
- [ ] Table ranked by mean phase reached, then by ep_rew_mean, paste-ready into the report
- [ ] Any config where reward rose while phase progression stayed flat is flagged as reward hacking
- [ ] Winner retrained at full budget and saved

## Notes

Delegate to the `sweep-runner` agent; it keeps the SB3 output out of the main context.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/13
- 2026-08-29 — assigned to member-1
- 2026-08-29 — priority P1 -> P0: this ticket carries rubric row J3 worth 3 points, meeting the
  board's own P0 rule ("blocks a rubric row worth 3+ points"). It was mispriced as P1.
