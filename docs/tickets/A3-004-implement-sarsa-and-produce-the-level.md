---
id: A3-004
title: Implement SARSA and produce the level 1 comparison
type: feature
status: open
priority: P0
rubric: C
points_at_risk: 3.0
area: gridworld
github: https://github.com/TDuong04/A3_TDuong/issues/4
owner: unassigned
blocks: [A3-005]
blocked_by: [A3-003]
created: 2026-08-19
updated: 2026-08-20
---

# A3-004 — Implement SARSA and produce the level 1 comparison

## Context

SARSA differs from Q-learning in exactly one term, and the assignment wants evidence that
this changes behaviour. Level 1 is a cliff-walk built for this: fire between start and goal, so
Q-learning learns the fast edge route and SARSA, accounting for its own exploration, detours.

The comparison figure is also the strongest thing to show in the video for Part I.

## Acceptance criteria

- [ ] On-policy update uses the action actually taken next, never a max
- [ ] Uses the same `LinearEpsilon` instance construction as Q-learning — provable by inspection
- [ ] Both algorithms run on level 1 under identical seeds and schedules
- [ ] Figure showing both greedy policies over the cliff, saved to `results/`
- [ ] Short written comparison of the two routes for the report

## Notes

Files: `gridworld/algorithms.py`, `eval/play_gridworld.py` (`--compare` mode).

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/4
- 2026-08-20 — SARSA implemented in `gridworld/algorithms.py`, sharing `make_q_table`,
  `epsilon_schedule` and `select_action` with Q-learning unchanged. `--compare` added to
  `train/train_gridworld.py`. Level 1 comparison on seeds 0 and 1: Q-learning takes the 11-step
  cliff-edge route along row 7, SARSA detours via row 6 in 13 steps; death rate under the
  converged behaviour policy (eps 0.05, 500 rollouts) 11.4% vs 1.8% on seed 0 and 12.0% vs 1.8%
  on seed 1. Uncommitted, pending evaluation.
- 2026-08-20 — diagnosis while producing the figure: at the original level-1 budget of 4000
  episodes both algorithms detoured and the comparison showed nothing. `epsilon_end` was not the
  cause; Q-learning simply had not converged on row 7, whose cells are only reached by an agent
  that has already survived several steps beside the fire. `level_overrides[1]` raised to
  8000 episodes / 6000 decay; `epsilon_end` left at the shared 0.05 so level 0 is untouched.
