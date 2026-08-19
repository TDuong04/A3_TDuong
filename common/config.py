"""Config loading.

Training parameters live in `config/*.yaml`, never as literals in algorithm code — rubric B3
requires config-driven epsilon decay, and the hyperparameter sweep needs one place to vary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_yaml(name: str) -> dict[str, Any]:
    """Load `config/<name>.yaml` (the `.yaml` suffix is optional)."""
    path = Path(name)
    if not path.suffix:
        path = CONFIG_DIR / f"{name}.yaml"
    elif not path.is_absolute():
        path = CONFIG_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass(frozen=True)
class TabularConfig:
    """Hyperparameters shared by Q-learning and SARSA."""

    episodes: int
    alpha: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_episodes: int
    max_steps_per_episode: int
    seed: int
    intrinsic_reward_strength: float = 0.0

    @classmethod
    def from_yaml(cls, name: str = "gridworld", section: str = "training") -> TabularConfig:
        data = load_yaml(name)[section]
        return cls(**data)


@dataclass(frozen=True)
class ArenaTrainConfig:
    """Deep RL training hyperparameters for the arena."""

    algorithm: str
    total_timesteps: int
    n_envs: int
    learning_rate: float
    n_steps: int
    batch_size: int
    gamma: float
    ent_coef: float
    net_arch: list[int] = field(default_factory=lambda: [64, 64])
    seed: int = 0

    @classmethod
    def from_yaml(cls, name: str = "arena", section: str = "training") -> ArenaTrainConfig:
        data = load_yaml(name)[section]
        return cls(**data)
