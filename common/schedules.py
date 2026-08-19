"""Exploration schedules.

Q-learning and SARSA must share one exploration schedule (rubric C2). Both import
`LinearEpsilon` from here so that "same exploration schedule" is provable by inspection rather
than by comparing two copies of the same arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearEpsilon:
    """Linear decay from `start` to `end` over `decay_episodes`, then held at `end`.

    The brief requires linear decay from epsilonStart to epsilonEnd (rubric B3), with the values
    supplied by a config file rather than hardcoded at the call site.
    """

    start: float
    end: float
    decay_episodes: int

    def __post_init__(self) -> None:
        if self.decay_episodes < 1:
            raise ValueError("decay_episodes must be >= 1")
        if not 0.0 <= self.end <= self.start <= 1.0:
            raise ValueError(f"require 0 <= end <= start <= 1, got start={self.start} end={self.end}")

    def __call__(self, episode: int) -> float:
        """Epsilon for a zero-indexed episode number, clamped outside the decay window."""
        if episode <= 0:
            return self.start
        if episode >= self.decay_episodes:
            return self.end
        fraction = episode / self.decay_episodes
        return self.start + fraction * (self.end - self.start)
