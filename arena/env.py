"""Arena environment — NOT YET IMPLEMENTED.

ONE env class for both control schemes. `ArenaEnv(control_style="rotation"|"direct")` switches only
the action space and `_apply_action`; physics, spawning, rewards, observation and rendering are
shared. Forking this into two files is the fastest way to lose rubric row G — every bug gets fixed
twice, the two agents stop being comparable, and report row R6's comparison becomes meaningless.

    action_space = Discrete(5) for rotation, Discrete(6) for direct   # exact indices in constants.py
    observation_space = Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=float32)

`step(action)` runs `ACTION_REPEAT` (3) physics frames of `FIXED_DT` with the action held, then
builds the observation. That cuts the episode horizon threefold for free and makes credit
assignment tractable.

Per-step sequence: apply action; advance bullets, enemies and spawners; resolve collisions;
accumulate rewards; check phase advance (all spawners destroyed) and set up the next phase from
`config/arena.yaml`; check termination.

Termination is on player death (`terminated`) or `MAX_EPISODE_STEPS` (`truncated`). Keeping those
distinct matters — bootstrapping through a time-limit cutoff as though it were death teaches the
agent that running out of clock is fatal.

`info` must carry `phase`, `spawners_destroyed`, `enemies_killed` and `damage_taken`. Those are the
behavioural metrics the diagnostician and the report depend on; reward alone cannot distinguish a
policy that is progressing from one that is farming a shaping term.

Two API surfaces, both required:
  - This class is Gymnasium 1.3 native — `reset(seed=None)` returns `(obs, info)` and `step` returns
    the 5-tuple — because that is what Stable Baselines3 requires.
  - `LegacyGymAPI` in `arena/legacy_api.py` wraps it to the signature the brief names literally:
    `reset()` returns obs, `step(action)` returns `(obs, reward, done, info)`, plus `render()`.
    Rubric H1 quotes that 4-tuple, so ship the adapter and name it in the report.

`render_mode=None` must import and run with no display. Nothing in this file may call
`pygame.draw`, `blit`, `display` or `clock.tick` — that all belongs in `render.py`, invoked only
from `render()`. Rendering inside `step()` silently multiplies training time by an order of
magnitude and is the first thing `env-validator` greps for.
"""

from __future__ import annotations


class ArenaEnv:
    def __init__(self, control_style: str = "direct", render_mode: str | None = None) -> None:
        raise NotImplementedError("see module docstring for the contract")
