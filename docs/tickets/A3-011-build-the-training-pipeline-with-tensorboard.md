---
id: A3-011
title: Build the training pipeline with TensorBoard behavioural logging
type: feature
status: done
priority: P0
rubric: J
points_at_risk: 3.0
area: training
part: 2
github: https://github.com/TDuong04/A3_TDuong/issues/11
owner: member-1
blocks: [A3-012, A3-013]
blocked_by: [A3-021]
created: 2026-08-19
updated: 2026-08-30
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
- 2026-08-29 — blocked_by moved from A3-010 (done) to A3-021, the pre-flight env validation; assigned to member-1
- 2026-08-30 — closed — `train/train_arena.py` trains either style from `config/arena.yaml` with per-axis CLI overrides; `SubprocVecEnv` at `n_envs > 1` (8 by config) behind the `__main__` guard, `DummyVecEnv` at 1; `SDL_VIDEODRIVER=dummy` set at import and again inside each worker, with a subprocess test asserting pygame never enters `sys.modules`; every env wrapped in `Monitor(info_keywords=('phase','spawners_destroyed','enemies_killed','damage_taken'))`; `BehaviourLoggingCallback` writes the four `behaviour/*` scalars plus `behaviour/episode_length` as 100-episode rolling means, read back out of a real event file by the tests; checkpoints every 50k *total* timesteps to `models/checkpoints/`; each run also leaves `logs/<run>/run.json` with the resolved config, seed and git commit. 26 new tests in `tests/test_train_arena.py`, full suite 552 passing
