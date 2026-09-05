---
id: A3-012
title: Train both control agents and write the evaluation scripts
type: feature
status: done
priority: P0
rubric: I
points_at_risk: 4.0
area: eval
github: https://github.com/TDuong04/A3_TDuong/issues/12
owner: unassigned
blocks: [A3-014, A3-015]
blocked_by: [A3-011]
created: 2026-08-19
updated: 2026-08-19
---

# A3-012 — Train both control agents and write the evaluation scripts

## Context

Row I is a checklist row: two styles, two saved models, working eval scripts. None of it
depends on the agent being especially good, so it is 4 points of low-risk marks — but only if the
scripts actually run from a clean checkout.

Train direct movement first. It learns faster, so it proves the reward function works before the
harder control style is attempted.

## Acceptance criteria

- [x] A trained model per style saved in `models/` under that exact folder name
- [x] `eval/play_arena.py --style {rotation,direct}` visually runs each with deterministic=True
- [x] Both agents clear phase 1 at least once during evaluation
- [x] Summary printed over 5 seeded episodes: return, phase reached, spawners destroyed, survival
- [x] `--human` mode plays the same env from the keyboard

## Notes

If an agent cannot clear phase 1, tune phase 1 in config rather than adding timesteps — the video requires a visible phase progression.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/12
- 2026-09-05 - both agents trained at the config's 400k timesteps and saved to `models/`:
  `ppo_direct.zip` and `ppo_rotation.zip`. Direct was trained first, as the ticket advised, because
  it has no rotation to learn and so proves the reward function works before the harder style is
  attempted. Mean episode reward moved from -12.6 under a random policy to +15.0 for direct.
- 2026-09-05 - `eval/play_arena.py` plays either style with `deterministic=True`, prints the
  five-column summary report row R6 tabulates, and has a `--human` mode that drives the same
  `env.step()` from the keyboard. Human input reads `pygame.key.get_pressed()` rather than the
  event queue, so it never competes with the renderer for events - and held keys are the right
  model for a real-time game anyway.
- 2026-09-05 - measured over 5 seeded episodes, both styles meeting the same arenas:
  direct returns +19.18 +/- 16.36, phase 2.20 mean and 3 best, clearing a phase in 4 of 5 episodes;
  rotation returns +2.79 +/- 11.83, phase 1.60 mean and 2 best, clearing a phase in 3 of 5. The
  acceptance criterion that both agents clear phase 1 at least once is met with room to spare, so
  the video's required phase progression is safe to record.
- 2026-09-05 - the gap between the styles is the expected one and is worth the report saying
  plainly: rotation has to learn to aim before it can learn to shoot, so it spends much of its
  budget on a control problem direct never has.
- 2026-09-05 - a mutant that reset every episode to the same seed survived the first version of the
  tests. Five episodes would then have been five replays of one arena, with a structurally zero
  standard deviation - a spread quoted in the report that was never measured. A test now asserts
  each episode receives its own seed. Closed.
