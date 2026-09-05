"""TensorBoard callbacks for the arena training runs.

`BehaviourLoggingCallback` reads the `info` dicts of finished episodes and records, as TensorBoard
scalars alongside `rollout/ep_rew_mean`:

    behaviour/phase_reached
    behaviour/spawners_destroyed
    behaviour/enemies_killed
    behaviour/damage_taken
    behaviour/episode_length

These curves are the difference between "here is a reward graph" and "here is evidence the agent
learned to progress", and they are what `training-diagnostician` reads to tell reward hacking apart
from genuine learning. Reward rising while phase progression stays flat means the agent found a
shaping exploit, and no reward curve alone will ever show you that.

The values arrive here because each env is wrapped in `Monitor(env, info_keywords=...)`, which
copies those keys out of the terminal step's `info` and into `info["episode"]`. Everything is
reported as a rolling mean over the last `window` episodes for the same reason SB3 smooths
`ep_rew_mean`: a single arena episode is far too noisy to read a trend from.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

#: The keys `arena.env.ArenaEnv._info` promises, in the order they are passed to `Monitor`.
BEHAVIOUR_INFO_KEYS: tuple[str, ...] = (
    "phase",
    "spawners_destroyed",
    "enemies_killed",
    "damage_taken",
)

#: `info` key -> TensorBoard tag. `phase` is logged as `phase_reached` because what the episode
#: record holds is the phase the agent got to before it died, not a phase it merely occupied.
BEHAVIOUR_TAGS: dict[str, str] = {
    "phase": "behaviour/phase_reached",
    "spawners_destroyed": "behaviour/spawners_destroyed",
    "enemies_killed": "behaviour/enemies_killed",
    "damage_taken": "behaviour/damage_taken",
}

#: Monitor's own episode-length key, logged next to the four behavioural ones so a run that is
#: surviving longer without progressing is visible at a glance.
EPISODE_LENGTH_KEY = "l"
EPISODE_LENGTH_TAG = "behaviour/episode_length"

#: Episodes behind each reported mean. Matches SB3's own 100-episode window for `ep_rew_mean`, so
#: the behavioural curves and the reward curve are smoothed identically and can be read together.
DEFAULT_WINDOW = 100

#: Rubric-neutral default: a crash six hours into a 400k run should cost minutes, not the run.
DEFAULT_CHECKPOINT_FREQ = 50_000


class BehaviourLoggingCallback(BaseCallback):
    """Log the arena's behavioural metrics as TensorBoard scalars.

    Args:
        window: episodes averaged per reported point.
        verbose: SB3 verbosity; 1 prints each rollout's means to stdout.
    """

    def __init__(self, window: int = DEFAULT_WINDOW, verbose: int = 0) -> None:
        super().__init__(verbose)
        if window < 1:
            raise ValueError(f"window must be at least 1 episode, got {window}")
        self.window = window
        self.episodes_seen = 0
        self._buffers: dict[str, deque[float]] = {
            tag: deque(maxlen=window) for tag in (*BEHAVIOUR_TAGS.values(), EPISODE_LENGTH_TAG)
        }

    # --- SB3 hooks -------------------------------------------------------------------------

    def _on_step(self) -> bool:
        """Absorb every episode that finished during this step of the vectorised env."""
        recorded = False
        for info in self.locals.get("infos", ()):
            episode = info.get("episode") if isinstance(info, dict) else None
            if episode is None:
                continue
            self._absorb(episode)
            recorded = True
        # Recorded here rather than only at rollout end: DQN dumps its logs from inside
        # `collect_rollouts`, so waiting for `_on_rollout_end` would leave the first dumps empty.
        if recorded:
            self._record()
        return True

    def _on_rollout_end(self) -> None:
        """Re-record before the algorithm's own dump, so a quiet rollout still reports a value."""
        self._record()

    # --- internals -------------------------------------------------------------------------

    def _absorb(self, episode: dict[str, Any]) -> None:
        """Push one Monitor episode record into the rolling windows."""
        for key, tag in BEHAVIOUR_TAGS.items():
            value = episode.get(key)
            if value is not None:
                self._buffers[tag].append(float(value))
        length = episode.get(EPISODE_LENGTH_KEY)
        if length is not None:
            self._buffers[EPISODE_LENGTH_TAG].append(float(length))
        self.episodes_seen += 1

    def _record(self) -> None:
        means = self.latest()
        for tag, mean in means.items():
            self.logger.record(tag, mean)
        if self.verbose > 0 and means:
            summary = ", ".join(f"{tag.split('/')[-1]}={mean:.2f}" for tag, mean in means.items())
            print(f"[behaviour] episodes={self.episodes_seen} {summary}")

    def latest(self) -> dict[str, float]:
        """The current rolling mean per tag, skipping tags no episode has reported yet."""
        return {
            tag: sum(values) / len(values)
            for tag, values in self._buffers.items()
            if values
        }


def make_checkpoint_callback(
    save_path: Path | str,
    name_prefix: str,
    save_freq: int = DEFAULT_CHECKPOINT_FREQ,
    n_envs: int = 1,
    verbose: int = 1,
) -> CheckpointCallback:
    """A `CheckpointCallback` that fires every `save_freq` *total* timesteps.

    SB3 counts `save_freq` in per-env steps, so a run with 8 workers would otherwise checkpoint
    eight times more often than the number asked for. Dividing here keeps the flag meaning what it
    says regardless of `n_envs`.
    """
    if save_freq < 1:
        raise ValueError(f"save_freq must be positive, got {save_freq}")
    if n_envs < 1:
        raise ValueError(f"n_envs must be positive, got {n_envs}")
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    return CheckpointCallback(
        save_freq=max(1, save_freq // n_envs),
        save_path=str(save_path),
        name_prefix=name_prefix,
        verbose=verbose,
    )
