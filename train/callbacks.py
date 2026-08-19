"""TensorBoard callbacks — NOT YET IMPLEMENTED.

An SB3 `BaseCallback` that reads the `info` dicts of finished episodes and records, as TensorBoard
scalars alongside `ep_rew_mean`:

    behaviour/phase_reached
    behaviour/spawners_destroyed
    behaviour/enemies_killed
    behaviour/damage_taken
    behaviour/episode_length

These four curves are the difference between "here is a reward graph" and "here is evidence the
agent learned to progress", and they are what `training-diagnostician` reads to tell reward hacking
apart from genuine learning. Reward rising while phase progression stays flat means the agent found
a shaping exploit, and no reward curve alone will ever show you that.

Also useful, and nearly free: a checkpoint callback saving to `models/checkpoints/` every 50k
timesteps, so a crash six hours in does not cost the whole run.
"""

from __future__ import annotations


class BehaviourLoggingCallback:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("see module docstring for the contract")
