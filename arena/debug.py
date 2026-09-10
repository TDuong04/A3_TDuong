"""Immutable arena evidence; no rendering, action selection or random draws."""
from dataclasses import dataclass

from . import constants
from .policy_view import PolicyView

# Labels and weights are shared by telemetry, the panel and accounting tests.
REWARD_TERMS = {
    "step": ("Step penalty", constants.REWARD_PER_STEP),
    "enemy": ("Enemy destroyed", constants.REWARD_ENEMY_DESTROYED),
    "spawner": ("Spawner destroyed", constants.REWARD_SPAWNER_DESTROYED),
    "phase": ("Phase advance", constants.REWARD_PHASE_ADVANCE),
    "damage": ("Damage event", constants.REWARD_DAMAGE_TAKEN),
    "death": ("Death", constants.REWARD_DEATH),
}


@dataclass(frozen=True)
class RewardSnapshot:
    step: int
    action: int
    contributions: tuple[tuple[str, float], ...]
    totals: tuple[tuple[str, float], ...]
    reward: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class ArenaDebugSnapshot:
    """Policy from s, action a, and rewards from the same completed transition."""
    observation: tuple[float, ...]
    next_observation: tuple[float, ...]
    policy: PolicyView | None
    policy_status: str
    reward: RewardSnapshot
