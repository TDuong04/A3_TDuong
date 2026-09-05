"""TensorBoard callbacks for arena training.

`BehaviourLoggingCallback` records what the agent *did*, alongside what it scored:

    behaviour/phase_reached      behaviour/spawners_destroyed
    behaviour/enemies_killed     behaviour/damage_taken
    behaviour/episode_length     behaviour/survival_rate

These are the difference between "here is a reward graph" and "here is evidence the agent learned
to progress". A reward curve alone cannot tell an agent that is genuinely clearing phases from one
that has found a way to farm a shaping term -- both are a line going up. Reward rising while
`phase_reached` stays flat is the signature of the second, and nothing in `ep_rew_mean` will ever
show it to you.

They are also what `training-diagnostician` reads, and what report row R6 compares between the two
control styles.

The metrics arrive through `Monitor(env, info_keywords=...)`, which copies the named keys out of
the final `info` dict of each episode into the `episode` record that SB3 puts on `infos`. That is
why the keyword list here and the one in `train_arena.py` have to agree -- a key Monitor was not
told to keep simply never arrives, silently, and the curve stays empty for the whole run.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

#: Keys copied out of the env's `info` dict by `Monitor`, and logged here. Keep in step with
#: `train_arena.py`'s Monitor construction; a mismatch is silent.
INFO_KEYWORDS: tuple[str, ...] = (
    "phase",
    "spawners_destroyed",
    "enemies_killed",
    "damage_taken",
    "terminated",
)

#: How many finished episodes each mean is taken over. Matches SB3's own `ep_rew_mean` window, so
#: the behavioural curves and the reward curve are smoothed identically and can be read together.
WINDOW = 100


class BehaviourLoggingCallback(BaseCallback):
    """Log per-episode behavioural metrics to TensorBoard as a rolling mean.

    Rolling rather than instantaneous because a single arena episode is extremely noisy -- an agent
    that clears phase 2 half the time produces a phase curve that alternates between 1 and 3, and
    reading a trend off that is guesswork.
    """

    def __init__(self, window: int = WINDOW, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.window = int(window)
        self._buffers: dict[str, deque[float]] = {}
        self.episodes_seen = 0

    def _record(self, name: str, value: float) -> None:
        buffer = self._buffers.setdefault(name, deque(maxlen=self.window))
        buffer.append(float(value))

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is None:
                continue  # mid-episode step; Monitor only attaches `episode` on the last one
            self.episodes_seen += 1
            self._record("episode_length", episode.get("l", 0))
            self._record("phase_reached", info.get("phase", 1))
            self._record("spawners_destroyed", info.get("spawners_destroyed", 0))
            self._record("enemies_killed", info.get("enemies_killed", 0))
            self._record("damage_taken", info.get("damage_taken", 0))
            # `terminated` is death; its complement is "still alive when the clock ran out". An
            # agent whose survival rate climbs while its phase count does not has learned to hide.
            self._record("survival_rate", 0.0 if info.get("terminated", False) else 1.0)
        return True

    def _on_rollout_end(self) -> None:
        """Write the current means. Once per rollout, not once per step: the buffers only change
        when an episode finishes, and logging a mean that has not moved costs disk for nothing."""
        for name, buffer in self._buffers.items():
            if buffer:
                self.logger.record(f"behaviour/{name}", float(np.mean(buffer)))
        self.logger.record("behaviour/episodes_finished", float(self.episodes_seen))

    def summary(self) -> dict[str, float]:
        """The current means, for a caller that wants them without reading TensorBoard."""
        return {name: float(np.mean(buf)) for name, buf in self._buffers.items() if buf}


class ConsoleProgressCallback(BaseCallback):
    """A readable one-line progress report every `every` timesteps.

    SB3's own table is printed per rollout and is easy to lose in a long run; this prints the two
    numbers that actually decide whether the run is worth continuing -- mean reward, and whether
    any phase has ever been cleared.
    """

    def __init__(self, every: int = 25_000, behaviour: BehaviourLoggingCallback | None = None,
                 verbose: int = 0) -> None:
        super().__init__(verbose)
        self.every = int(every)
        self.behaviour = behaviour
        self._next = self.every

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next:
            return True
        self._next += self.every
        stats: dict[str, Any] = {}
        if self.behaviour is not None:
            stats = self.behaviour.summary()
        phase = stats.get("phase_reached")
        spawners = stats.get("spawners_destroyed")
        parts = [f"{self.num_timesteps:>7,} steps"]
        if phase is not None:
            parts.append(f"phase {phase:.2f}")
        if spawners is not None:
            parts.append(f"spawners {spawners:.2f}")
        print("  " + " | ".join(parts), flush=True)
        return True
