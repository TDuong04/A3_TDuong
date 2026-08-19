"""Frozen gridworld constants taken from A3_Brief.pdf.

The brief states you may add helper functions but must NOT alter rewards or mechanics. Rubric row
A2 checks these values directly, and `tests/test_spec_constants.py` guards them. Do not edit
anything in this module without a corresponding change in the brief.
"""

from __future__ import annotations

from enum import IntEnum

# --- Rewards (fixed by the brief) ---
REWARD_APPLE = 1.0
REWARD_KEY = 0.0  # no reward, but unlocks chests
REWARD_CHEST = 2.0
REWARD_STEP = 0.0
REWARD_DEATH = 0.0  # death ends the episode; the brief specifies no explicit death penalty

# --- Mechanics (fixed by the brief) ---
MONSTER_MOVE_PROBABILITY = 0.4  # chance a monster moves after each agent action


class Action(IntEnum):
    """The only four legal moves. Attempting to enter a rock results in no movement."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


# (row_delta, col_delta) in grid coordinates, row 0 at the top.
ACTION_DELTAS: dict[Action, tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}

N_ACTIONS = len(Action)


class Tile:
    """ASCII tile codes used by `levels.py`."""

    EMPTY = "."
    ROCK = "#"      # blocks movement, no displacement, no reward change
    FIRE = "F"      # instant death on entry
    MONSTER = "M"   # instant death on entry; moves after each agent action
    APPLE = "A"     # +1
    KEY = "K"       # 0 reward, unlocks chests
    CHEST = "C"     # +2, only openable while holding a key
    START = "S"     # agent spawn


LEGAL_TILES = frozenset(
    {Tile.EMPTY, Tile.ROCK, Tile.FIRE, Tile.MONSTER, Tile.APPLE, Tile.KEY, Tile.CHEST, Tile.START}
)

#: Tiles that are collected and count toward "all collectible rewards obtained".
COLLECTIBLE_TILES = frozenset({Tile.APPLE, Tile.KEY, Tile.CHEST})

#: Tiles that kill the agent instantly on entry.
LETHAL_TILES = frozenset({Tile.FIRE, Tile.MONSTER})

TILE_REWARDS: dict[str, float] = {
    Tile.APPLE: REWARD_APPLE,
    Tile.KEY: REWARD_KEY,
    Tile.CHEST: REWARD_CHEST,
}
