---
id: A3-021
title: Validate the arena environment before the first training run
type: task
status: open
priority: P0
rubric: H
points_at_risk: 2.5
area: training
github: https://github.com/TDuong04/A3_TDuong/issues/22
owner: member-1
blocks: [A3-011]
blocked_by: []
created: 2026-08-29
updated: 2026-08-29
---

# A3-021 — Validate the arena environment before the first training run

## Context

`ArenaEnv` now steps cleanly under a random policy in both control styles, but nothing has yet
checked the properties that make a training run fail silently rather than loudly. A run that
learns nothing because one observation feature is unnormalised costs a day, and the symptom is
identical to a badly tuned reward.

Do this once, before A3-011 burns any GPU hours. It is also the evidence rubric row H wants for
"fixed-size normalised numeric vector" — an assertion that ran beats a sentence in the report.

## Acceptance criteria

- [ ] Observation is `(21,)` float32, finite, and every feature stays within `[-1, 1]` across
      10,000 random steps in both control styles
- [ ] No single feature is constant across a run, and none saturates at a bound for more than
      50% of steps — a dead or clipped feature is reported by name
- [ ] `reset(seed=n)` twice with the same action sequence gives byte-identical observations and
      rewards, for at least 3 seeds
- [ ] `action_space` is `Discrete(5)` for rotation and `Discrete(6)` for direct
- [ ] Every episode ends within `MAX_EPISODE_STEPS`, with `terminated` and `truncated` never both
      set, over 200 episodes
- [ ] `render_mode=None` imports and runs with `SDL_VIDEODRIVER=dummy` and opens no window
- [ ] Step throughput recorded in steps/second on one core, so A3-011 can size its budget

## Notes

Delegate to the `env-validator` agent — it writes its harness to the scratchpad, not the repo.
Anything it finds is a bug ticket against `arena/env.py` or `arena/observation.py`, not a
tolerance to widen.

## Log

- 2026-08-29 — created; the arena env is built but never stress-tested
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/22
