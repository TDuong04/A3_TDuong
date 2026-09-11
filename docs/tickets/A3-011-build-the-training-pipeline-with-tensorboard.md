---
id: A3-011
title: Build the training pipeline with TensorBoard behavioural logging
type: feature
status: done
priority: P0
rubric: J
points_at_risk: 3.0
area: training
github: https://github.com/TDuong04/A3_TDuong/issues/11
owner: unassigned
blocks: [A3-012, A3-013]
blocked_by: []
created: 2026-08-19
updated: 2026-09-12
---

# A3-011 — Build the training pipeline with TensorBoard behavioural logging

## Context

Set the callback up before the first real run. Retraining to recover a metric nobody logged
costs hours, and reward alone cannot tell a policy that is progressing from one farming a shaping
term.

## Acceptance criteria

- [x] `train/train_arena.py` trains either control style from `config/arena.yaml`
- [x] SubprocVecEnv with 8 envs, behind an `if __name__ == "__main__":` guard
- [x] `SDL_VIDEODRIVER=dummy` set before pygame import; no worker opens a window
- [x] Monitor wrapper with info_keywords; TensorBoard logs under `logs/`
- [x] Callback logs phase_reached, spawners_destroyed, enemies_killed, damage_taken
- [x] Checkpoints saved every 50k timesteps to `models/checkpoints/`

## Notes

Files: `train/train_arena.py`, `train/callbacks.py`.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/11
- 2026-09-12 — closed as stale board drift, not real open work. Every acceptance criterion is
  already met on `main`, landed by commit d25c069 / 045b4e0 (merged via PR #40,
  4806832) and extended by cc54432 (A3-012): `train/train_arena.py` trains either style from
  `config/arena.yaml`; `SubprocVecEnv` with 8 envs sits behind `if __name__ == "__main__":`
  (train/train_arena.py:143, :387); `SDL_VIDEODRIVER=dummy` is set before the pygame import at
  train/train_arena.py:40 and again per-worker at :110; the `Monitor` wrapper with
  `info_keywords` is at train/train_arena.py:114 and real TensorBoard event files exist under
  `logs/arena_seed_selection/`, `logs/arena_sweep/`, `logs/ppo_direct_shipped/`,
  `logs/ppo_rotation_shipped/`; `train/callbacks.py:44-54` logs `phase_reached`,
  `spawners_destroyed`, `enemies_killed`, `damage_taken` (plus `survival_rate`); checkpoints in
  `models/checkpoints/` land in 50k-step increments (e.g.
  `final_n_steps-512_s0_50000_steps.zip` through `..._400000_steps.zip`). Also corrected
  `blocked_by: [A3-010]` to `[]` — A3-010 is `status: done`; this stale link was independently
  flagged by a rubric-auditor pass and a solution-evaluator pass (both 2026-09-12). No code
  touched; `pytest` (860+ passing) was already green and unaffected.
