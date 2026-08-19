"""Frozen arena constants.

The action index mappings below are dictated by the brief and are checked by rubric rows I1/I2 —
the *indices* matter, not just the set of available actions. `tests/test_spec_constants.py` guards
them.
"""

from __future__ import annotations

from enum import IntEnum


class RotationAction(IntEnum):
    """Control style 1 — rotation movement and thrust. Discrete(5)."""

    NOOP = 0
    THRUST = 1
    ROTATE_LEFT = 2
    ROTATE_RIGHT = 3
    SHOOT = 4


class DirectAction(IntEnum):
    """Control style 2 — direct directional movement. Discrete(6)."""

    NOOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    SHOOT = 5


CONTROL_STYLES = ("rotation", "direct")

ACTION_ENUMS = {
    "rotation": RotationAction,
    "direct": DirectAction,
}

N_ACTIONS = {
    "rotation": len(RotationAction),  # 5
    "direct": len(DirectAction),      # 6
}

# --- Rewards ---
# Rubric J1 requires enemy kill, spawner kill, phase progress, damage penalty and death penalty.
# The step penalty is what stops the agent learning to hide in a corner: never pay it to survive.
# Any additional shaping must be potential-based (F = gamma * phi(s') - phi(s)) so that the
# optimal policy is provably unchanged, and must be justified in the report.
REWARD_ENEMY_DESTROYED = 1.0
REWARD_SPAWNER_DESTROYED = 5.0
REWARD_PHASE_ADVANCE = 10.0
REWARD_DAMAGE_TAKEN = -0.5
REWARD_DEATH = -10.0
REWARD_PER_STEP = -0.01

# --- Arena geometry and pacing ---
ARENA_WIDTH = 960
ARENA_HEIGHT = 680
FPS = 60
FIXED_DT = 1.0 / FPS  # physics timestep; never use wall-clock delta or runs stop reproducing
ACTION_REPEAT = 3     # agent decides every 3rd physics frame, cutting the episode horizon 3x
MAX_EPISODE_STEPS = 2000  # agent steps, not physics frames

# --- Observation ---
# Fixed-size numeric vector, no pixels (brief requirement). See docs in arena/observation.py.
OBS_DIM = 21
MAX_TRACKED_ENEMIES = 1   # observation reports the nearest enemy only
MAX_TRACKED_SPAWNERS = 1  # ... and the nearest spawner
