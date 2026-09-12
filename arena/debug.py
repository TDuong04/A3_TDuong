"""Immutable arena evidence; no rendering, action selection or random draws."""
from dataclasses import dataclass

from common.config import load_yaml

from . import constants
from .policy_view import PolicyView

# The two terms here not sourced from constants.py: `shaping.*` are tunables (config/arena.yaml),
# not brief-fixed rewards, so they cannot live in constants.py under rule 1. Shown as coefficients,
# not per-event constants -- unlike the rows above them, these terms' actual values change every
# step with the ship's aim and range, which is exactly the point of them.
_SHAPING = load_yaml("arena")["shaping"]
_AIM_STRENGTH = float(_SHAPING["aim_strength"])
_SAFETY_STRENGTH = float(_SHAPING["safety_strength"])

# Labels and weights are shared by telemetry, the panel and accounting tests.
REWARD_TERMS = {
    "step": ("Step penalty", constants.REWARD_PER_STEP),
    "enemy": ("Enemy destroyed", constants.REWARD_ENEMY_DESTROYED),
    "spawner": ("Spawner destroyed", constants.REWARD_SPAWNER_DESTROYED),
    "phase": ("Phase advance", constants.REWARD_PHASE_ADVANCE),
    "damage": ("Damage event", constants.REWARD_DAMAGE_TAKEN),
    "death": ("Death", constants.REWARD_DEATH),
    "aim_shaping": ("Aim alignment (shaping coef)", _AIM_STRENGTH),
    "safety_shaping": ("Range keeping (shaping coef)", _SAFETY_STRENGTH),
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
