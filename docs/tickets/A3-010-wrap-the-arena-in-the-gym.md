---
id: A3-010
title: Wrap the arena in the Gym-style API with the observation vector
type: feature
status: done
priority: P0
rubric: H
points_at_risk: 2.5
area: arena
github: https://github.com/TDuong04/A3_TDuong/issues/10
owner: unassigned
blocks: [A3-011]
blocked_by: [A3-008]
created: 2026-08-19
updated: 2026-08-20
---

# A3-010 — Wrap the arena in the Gym-style API with the observation vector

## Context

One env parameterised by control style, not two forks. Forking doubles every bug fix and
makes the two agents incomparable, which also costs the report's comparison section.

The observation layout is fully specified in the `arena/observation.py` docstring.

## Acceptance criteria

- [x] `ArenaEnv(control_style=...)` gives Discrete(5) for rotation and Discrete(6) for direct, with the brief's exact action indices
- [x] Observation is a fixed-size float32 vector of 21 features, no pixels
- [x] Relative positions rotated into the ship-local frame; angles as sin/cos
- [x] Empty entity slots zeroed with their validity flag cleared; no NaN at zero distance
- [x] `terminated` (death) and `truncated` (step cap) kept distinct
- [x] `info` carries phase, spawners_destroyed, enemies_killed, damage_taken
- [x] `stable_baselines3.common.env_checker.check_env` passes
- [x] `LegacyGymAPI` verified to present the brief's 4-tuple signature

## Notes

`arena/legacy_api.py` is already written. Run `env-validator` before closing this.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/10
- 2026-08-20 - implemented. `arena/observation.py` builds the 21-feature float32 vector;
  `arena/env.py` is one `ArenaEnv(control_style=...)` subclassing `gymnasium.Env`, holding each
  action for `ACTION_REPEAT` frames and importing the renderer lazily inside `render()` so
  training never loads pygame. Two new config blocks (`observation`, `episode`); nothing existing
  changed. Suite 445 -> 566 passed, 1 skipped. The skip is the real-renderer draw test, gated on
  A3-009 landing rather than on the stub's current behaviour, so it unskips itself.
- 2026-08-20 - verified by execution in the main session after the evaluator agent hit a session
  limit mid-run. All eight criteria PASS. The brief's literal API confirmed on `LegacyGymAPI`:
  `reset()` takes no required arguments and returns a bare `(21,)` float32 array, `step()` returns
  exactly four values with a real `bool` done, and the Gymnasium 5-tuple does not leak through it.
  The ship-local rotation was checked against an independently derived `cmath.exp(-1j*heading)` at
  headings 37, 121, -84 and 179 degrees with an off-axis target - exact agreement, and a `+heading`
  sign flip produces visibly different numbers at every one of them, so the sign is genuinely
  under test rather than hidden by an identity rotation. Death on the exact step the cap is
  reached yields `terminated=True, truncated=False`. `check_env` is clean with zero warnings on
  both styles. Adversarial states (all slots empty, zero distance) stay finite and in range.
- 2026-08-20 - five mutants, all killed: rotation sign flip, phase advancing when ANY spawner dies
  rather than all, `np.clip` removed, distance self-normalised to a constant 1.0, and the phase
  feature off by one. A sixth arrived by accident and is the most informative of the set - the
  evaluator died mid-mutation and left its scaling mutant (`max_relative_speed` = enemy speed
  alone, 150 instead of 370) live in the working tree, and the suite caught it immediately with
  two named failures. That is unplanned proof the scaling test bites.
- 2026-08-20 - the closing-velocity decision checked numerically and upheld: a player at its
  220 px/s cap meeting the fastest 150 px/s enemy head-on closes at 370 px/s. Dividing by the
  enemy speed alone gives -2.47, which clips to -1 and destroys the gradient exactly when an
  enemy is most dangerous. The shipped divisor reaches -1.0 only at that true maximum. Closed.
