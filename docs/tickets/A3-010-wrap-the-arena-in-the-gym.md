---
id: A3-010
title: Wrap the arena in the Gym-style API with the observation vector
type: feature
status: open
priority: P0
rubric: H
points_at_risk: 2.5
area: arena
owner: unassigned
blocks: [A3-011]
blocked_by: [A3-008]
created: 2026-08-19
updated: 2026-08-19
---

# A3-010 — Wrap the arena in the Gym-style API with the observation vector

## Context

One env parameterised by control style, not two forks. Forking doubles every bug fix and
makes the two agents incomparable, which also costs the report's comparison section.

The observation layout is fully specified in the `arena/observation.py` docstring.

## Acceptance criteria

- [ ] `ArenaEnv(control_style=...)` gives Discrete(5) for rotation and Discrete(6) for direct, with the brief's exact action indices
- [ ] Observation is a fixed-size float32 vector of 21 features, no pixels
- [ ] Relative positions rotated into the ship-local frame; angles as sin/cos
- [ ] Empty entity slots zeroed with their validity flag cleared; no NaN at zero distance
- [ ] `terminated` (death) and `truncated` (step cap) kept distinct
- [ ] `info` carries phase, spawners_destroyed, enemies_killed, damage_taken
- [ ] `stable_baselines3.common.env_checker.check_env` passes
- [ ] `LegacyGymAPI` verified to present the brief's 4-tuple signature

## Notes

`arena/legacy_api.py` is already written. Run `env-validator` before closing this.

## Log

- 2026-08-19 — created during project setup
