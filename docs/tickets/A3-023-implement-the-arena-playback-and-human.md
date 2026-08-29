---
id: A3-023
title: Implement the arena playback and human-play script
type: feature
status: open
priority: P0
rubric: I
points_at_risk: 4.0
area: eval
github: https://github.com/TDuong04/A3_TDuong/issues/24
owner: member-2
blocks: [A3-012, A3-022]
blocked_by: []
created: 2026-08-29
updated: 2026-08-29
---

# A3-023 — Implement the arena playback and human-play script

## Context

`eval/play_arena.py` is still a docstring and a `NotImplementedError`, and it was scoped inside
A3-012 behind the training run. That serialises three people behind one training job for no
reason: the env and renderer are both finished, so the playback script can be written and
demonstrated today against a random policy, then have the trained model dropped into it.

Splitting it out is what makes A3-022 (phase-1 tuning) and A3-018 (creativity) possible in
parallel with A3-011.

## Acceptance criteria

- [ ] `python -m eval.play_arena --style {rotation,direct}` opens a Pygame window and runs a
      policy, with `deterministic=True` for a loaded model
- [ ] Runs against a random policy when `models/` is empty, so it is testable before A3-012
- [ ] A missing model file exits with the `train.train_arena` command that would produce it —
      never trains on demand
- [ ] HUD shows phase, player health, current action name and cumulative reward
- [ ] Observation overlay toggleable at runtime, reusing `arena/render.py`
- [ ] `--human` plays the same env from the keyboard, on the identical action space
- [ ] `--episodes N` prints mean and std return, mean phase reached, spawners destroyed and
      survival time over N seeded episodes
- [ ] Nothing in the render or input path mutates simulation state

## Notes

Files: `eval/play_arena.py`, and read-only use of `arena/render.py`. The `--episodes` summary is
the table report row R6 needs, so print it in a paste-ready form. Split out of A3-012.

## Log

- 2026-08-29 — created; split the eval script out of A3-012 to unblock parallel work
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/24
