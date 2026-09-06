---
id: A3-013
title: Run the hyperparameter sweep and tabulate the results
type: task
status: done
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
updated: 2026-09-05
---

# A3-013 — Run the hyperparameter sweep and tabulate the results

## Context

Defaults-only training is explicitly marked down, and the report needs tuning evidence as
tables or plots. The sweep axes are already listed in `config/arena.yaml`.

## Acceptance criteria

- [x] One axis varied at a time from the documented baseline at 100k timesteps
- [x] Top two configs re-run on 3 seeds — a one-seed win is not a result
- [x] Table ranked by mean phase reached, then by ep_rew_mean, paste-ready into the report
- [x] Any config where reward rose while phase progression stayed flat is flagged as reward hacking
- [x] Winner retrained at full budget and saved

## Notes

Implemented as `train/sweep_arena.py` — config-driven from the `sweep` section of
`config/arena.yaml`, three stages (`explore`, `confirm`, `final`), artefacts in
`results/arena_sweep/`. Each run is a `train.train_arena` subprocess whose exact command line is
recorded in `sweep_runs.json`, so any row of the table can be reproduced by hand. SB3's per-rollout
output goes to `logs/arena_sweep/<run>/train.log` rather than the console.

Two things the first run got wrong, both fixed:

- The reward-hacking flag was also used to filter the shortlist, which promoted `baseline` over
  `learning_rate=0.001` — a config that had tripled phase progression but by less than the
  tolerance. The flag now annotates the row only; ranking phase before reward is what keeps a
  reward farmer off the top of the table.
- `reward_hack_phase_tolerance` was 0.05, wider than the entire spread of phase reached at this
  budget, so it fired on 6 of 9 rows. It is now 0.02, one measured seed-to-seed sd of the baseline.

## Result

Run against `main`'s arena env (phase is 1-based there: `phase == 1` means the agent is still in
phase 1). Winner is `n_steps=512`, 1.027 +- 0.012 mean phase reached over seeds 0-2, against the
runner-up `gamma=0.95` at 1.020 +- 0.008 and the baseline at 1.010. Retrained at the full 400k
budget into `models/ppo_direct_sweep.zip`.

**The winner was not promoted, and should not be.** Compared with the incumbent
`models/ppo_direct.zip` from A3-012 over 5 seeded episodes at `deterministic=True`:

| Model | n_steps | Cleared phase 1 | Spawners destroyed | Episode reward |
|-------|--------:|----------------:|--------------------|----------------|
| `ppo_direct.zip` (A3-012 incumbent) | 2048 | 4 / 5 | 2, 0, 3, 5, 5 | +14.7, -9.7, +21.7, +34.5, +34.7 |
| `ppo_direct_sweep.zip` (sweep winner) | 512 | 0 / 5 | 0, 0, 0, 0, 0 | -6.4, -6.8, -5.8, -6.3, -9.8 |

The incumbent uses `n_steps=2048` — the sweep's own baseline. The sweep preferred 512 on a margin
of 0.007 mean phase, which is smaller than either config's seed-to-seed standard deviation. That is
tuning on noise, and following it would have shipped a large regression over a teammate's better
model. `train/sweep_arena.py` therefore saves the tuned model under its own `_sweep` name and never
writes `models/<algo>_<style>.zip`; promotion is a manual, evidence-gated step.

Two caveats the report must state rather than hide:

- **The 100k budget does not separate the axes.** Every exploratory run lands between 1.000 and
  1.040 mean phase reached, and the top two are 0.007 apart with standard deviations of 0.012 and
  0.008. The ranking is honest about what it measured, but it is not evidence that the winner is
  better. The sweep's value here is the negative result, not the recommendation.
- **The full-budget tuned run still does not clear phase 1 deterministically** (0 of 5 seeds, 0
  spawners), even though its training rollouts average phase 1.290. A3-022 (phase 1 clearable in
  20-30s) is still open and is the more likely fix than more timesteps.

To make this sweep decisive, raise `sweep.budget_timesteps` — the whole run costs about 5 minutes
on a 14-core machine, so 200k-400k per run is affordable and is where the task starts being learned.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/13
- 2026-08-29 — assigned to member-1
- 2026-08-29 — priority P1 -> P0: this ticket carries rubric row J3 worth 3 points, meeting the
  board's own P0 rule ("blocks a rubric row worth 3+ points"). It was mispriced as P1.
- 2026-09-05 — done. `train/sweep_arena.py` plus tests; sweep run end to end against `main`,
  winner `n_steps=512` retrained at 400k and saved as `models/ppo_direct_sweep.zip`. Deliberately
  NOT promoted: it loses badly to the A3-012 incumbent under deterministic evaluation, because the
  100k budget separates the axes by less than their seed noise. Raised for A3-022.
