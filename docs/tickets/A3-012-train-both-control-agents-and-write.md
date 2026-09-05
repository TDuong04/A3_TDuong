---
id: A3-012
title: Train both control agents and write the evaluation scripts
type: feature
status: open
priority: P0
rubric: I
points_at_risk: 4.0
area: eval
github: https://github.com/TDuong04/A3_TDuong/issues/12
owner: unassigned
blocks: [A3-014, A3-015]
blocked_by: [A3-011]
created: 2026-08-19
updated: 2026-09-05
---

# A3-012 — Train both control agents and write the evaluation scripts

## Context

Row I is a checklist row: two styles, two saved models, working eval scripts. None of it
depends on the agent being especially good, so it is 4 points of low-risk marks — but only if the
scripts actually run from a clean checkout.

Train direct movement first. It learns faster, so it proves the reward function works before the
harder control style is attempted.

## Acceptance criteria

- [ ] A trained model per style saved in `models/` under that exact folder name
- [ ] `eval/play_arena.py --style {rotation,direct}` visually runs each with deterministic=True
- [ ] Both agents clear phase 1 at least once during evaluation
- [ ] Summary printed over 5 seeded episodes: return, phase reached, spawners destroyed, survival
- [x] `--human` mode plays the same env from the keyboard

## Notes

If an agent cannot clear phase 1, tune phase 1 in config rather than adding timesteps — the video requires a visible phase progression.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/12
- 2026-09-05 — A3-018 implemented keyboard play through python -m eval.play_arena --human for both styles, using the unmodified ArenaEnv. Training, model loading, and evaluation summaries remain outstanding; this ticket stays open.
