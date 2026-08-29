---
id: A3-011
title: Build the training pipeline with TensorBoard behavioural logging
type: feature
status: open
priority: P0
rubric: J
points_at_risk: 3.0
area: training
github: https://github.com/TDuong04/A3_TDuong/issues/11
owner: member-1
blocks: [A3-012, A3-013]
blocked_by: [A3-021]
created: 2026-08-19
updated: 2026-08-29
---

# A3-011 — Build the training pipeline with TensorBoard behavioural logging

## Context

Set the callback up before the first real run. Retraining to recover a metric nobody logged
costs hours, and reward alone cannot tell a policy that is progressing from one farming a shaping
term.

## Acceptance criteria

- [ ] `train/train_arena.py` trains either control style from `config/arena.yaml`
- [ ] SubprocVecEnv with 8 envs, behind an `if __name__ == "__main__":` guard
- [ ] `SDL_VIDEODRIVER=dummy` set before pygame import; no worker opens a window
- [ ] Monitor wrapper with info_keywords; TensorBoard logs under `logs/`
- [ ] Callback logs phase_reached, spawners_destroyed, enemies_killed, damage_taken
- [ ] Checkpoints saved every 50k timesteps to `models/checkpoints/`

## Notes

Files: `train/train_arena.py`, `train/callbacks.py`.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/11
- 2026-08-29 — blocked_by moved from A3-010 (done) to A3-021, the pre-flight env validation; assigned to member-1
