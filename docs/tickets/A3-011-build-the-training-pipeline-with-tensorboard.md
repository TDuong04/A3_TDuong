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
blocked_by: [A3-010]
created: 2026-08-19
updated: 2026-08-19
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
- 2026-09-05 - implemented. `train/callbacks.py` carries `BehaviourLoggingCallback` (rolling means
  over a 100-episode window, matching SB3's own `ep_rew_mean` window so the curves can be read
  together) and a console progress line. `train/train_arena.py` trains either style from
  `config/arena.yaml`, with every hyperparameter overridable by flag so A3-013's sweep can vary one
  axis without editing the file.
- 2026-09-05 - the callback logs a survival rate, which needed `terminated` adding to the env's
  `info`. Without it `Monitor` would have forwarded nothing and the callback would have defaulted
  the key to False, logging a constant 1.0 for an entire run with no error anywhere. A test now
  asserts every key the callback logs is a key the env actually emits.
- 2026-09-05 - the Windows guards are tested structurally rather than trusted: an AST check that
  the entry point sits behind `if __name__ == "__main__"`, and a source-order check that
  `SDL_VIDEODRIVER=dummy` is assigned above the first arena import. Both failures are catastrophic
  and neither shows up in a normal test run.
- 2026-09-05 - measured 2,877 fps with 8 envs, so a 400k-timestep run takes about 2.5 minutes.
  Closed.
