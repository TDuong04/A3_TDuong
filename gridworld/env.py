"""Gridworld environment.

`GridWorld(level_index, rng)` holds the mutable grid, the agent position, `has_key`, and the set of
remaining collectibles.

`reset()` reloads the level from `levels.load_level` (never mutate the module-level strings) and
returns the initial state.

`state` must be a hashable key for the Q-table. Position alone is NOT Markov on levels 2-6: two
visits to the same tile differ by which collectibles remain and whether the key is held. The key
used here is `(row, col, has_key, collected_mask)`, where `collected_mask` is an int bitmask over
the level's collectible cells. Bit order is fixed once at construction from the *sorted* collectible
coordinates, so a state key means the same thing in every episode and across processes — a mask
built from iteration order would silently relabel states between runs and corrupt a saved Q-table.

`step(action)` order of operations, which the rubric checks:
  1. Compute the target cell. If it is a rock or outside the grid, the agent does not move —
     no displacement, no reward change, no penalty.
  2. If the target is fire or a monster, the agent dies: episode over.
  3. Collect any item on the target cell. Apple `+1`. Key `0` but sets `has_key`. Chest `+2` only
     when `has_key` is set, otherwise the chest stays put and pays nothing.
  4. Move monsters: each independently moves with probability `MONSTER_MOVE_PROBABILITY` (0.4),
     choosing uniformly among directions that are not rocks or outside the grid.
  5. If a monster has moved onto the agent, the agent dies. This second check is mandatory — the
     brief kills the player both when it enters a monster tile and when a monster enters its tile.
  6. Episode ends when all collectibles are obtained or the agent has died.

Returns `(state, reward, done, info)`; `info` carries `died`, `collected` and `steps` so the
training scripts can log behavioural metrics rather than reward alone.

Monsters are tracked as entities in `self.monsters` rather than as `M` characters in the grid.
Once a monster moves, the character it left behind would have to be repaired every step, and the
tile it stands on (fire, an apple) would be destroyed by overwriting it. Separating them keeps the
grid purely static after `reset()`.

Truncation: `max_steps` bounds an episode so a policy that has learnt to stand still cannot hang
training. It is a cap, not a rule of the game, so `info["terminated"]` stays False when it fires
while `info["truncated"]` goes True; `done` is the disjunction of the two. The default comes from
`config/gridworld.yaml` (including the per-level overrides) because tuned numbers do not belong in
code.

Keep this file free of pygame imports — rendering lives in `render.py` so training runs headless.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from common.config import load_yaml
from common.seeding import make_rng

from .constants import (
    ACTION_DELTAS,
    COLLECTIBLE_TILES,
    LETHAL_TILES,
    MONSTER_MOVE_PROBABILITY,
    N_ACTIONS,
    REWARD_DEATH,
    REWARD_STEP,
    TILE_REWARDS,
    Action,
    Tile,
)
from .levels import load_level

Coord = tuple[int, int]
State = tuple[int, int, bool, int]


@lru_cache(maxsize=None)
def _configured_max_steps(level_index: int) -> int:
    """The episode cap for a level, from config rather than a literal in this file.

    Levels 4-5 are stochastic and are given a longer cap in `level_overrides`; reading the same
    file the training scripts read stops the env and the trainer from disagreeing about when an
    episode should have ended.
    """
    config = load_yaml("gridworld")
    overrides = config.get("level_overrides") or {}
    level_config = overrides.get(level_index) or {}
    return int(
        level_config.get(
            "max_steps_per_episode", config["training"]["max_steps_per_episode"]
        )
    )


class GridWorld:
    """A tabular gridworld: one agent, static hazards, collectibles and roaming monsters."""

    def __init__(
        self,
        level_index: int = 0,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        max_steps: int | None = None,
    ) -> None:
        """Build the env for `level_index`.

        `rng` is injected (or built from `seed`) rather than taken from global numpy state so two
        envs in one process cannot consume each other's random numbers, and so a run is
        reproducible from the seed recorded in its results filename.
        """
        self.level_index = int(level_index)
        self.rng = rng if rng is not None else make_rng(seed)
        if max_steps is None:
            max_steps = _configured_max_steps(self.level_index)
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")

        layout = load_level(self.level_index)
        self.n_rows = len(layout)
        self.n_cols = len(layout[0])

        # Bit order is fixed here, once, from sorted coordinates: see the module docstring.
        self.collectible_cells: tuple[Coord, ...] = tuple(
            sorted(
                (r, c)
                for r, row in enumerate(layout)
                for c, tile in enumerate(row)
                if tile in COLLECTIBLE_TILES
            )
        )
        self._bit_of: dict[Coord, int] = {
            cell: bit for bit, cell in enumerate(self.collectible_cells)
        }
        self.full_mask = (1 << len(self.collectible_cells)) - 1

        self.reset()

    # --- episode lifecycle -------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> State:
        """Reload the level and return the initial state. Optionally re-seed for a repeat run."""
        if seed is not None:
            self.rng = make_rng(seed)

        # `load_level` returns a fresh mutable copy, so the module-level strings are never touched.
        grid = load_level(self.level_index)
        self.monsters: list[Coord] = []
        agent: Coord | None = None
        for r, row in enumerate(grid):
            for c, tile in enumerate(row):
                if tile == Tile.MONSTER:
                    self.monsters.append((r, c))
                    row[c] = Tile.EMPTY
                elif tile == Tile.START:
                    agent = (r, c)
                    row[c] = Tile.EMPTY
        assert agent is not None, "load_level guarantees exactly one start tile"

        self.grid = grid
        self.agent_pos: Coord = agent
        self.has_key = False
        self.collected_mask = 0
        self.steps = 0
        self.died = False
        self.truncated = False
        self.episode_return = 0.0
        return self.state

    @property
    def state(self) -> State:
        """The Q-table key. Held-key and remaining collectibles are part of it by necessity."""
        return (self.agent_pos[0], self.agent_pos[1], self.has_key, self.collected_mask)

    @property
    def terminated(self) -> bool:
        """True once the game itself is over: everything collected, or the agent is dead."""
        return self.died or self.collected_mask == self.full_mask

    @property
    def done(self) -> bool:
        """What a training loop should break on — a real ending or the step cap."""
        return self.terminated or self.truncated

    # --- observation helpers for the renderer and the training loop ---------------------------

    @property
    def monster_positions(self) -> tuple[Coord, ...]:
        """A copy, so a renderer cannot mutate the simulation by holding onto the list."""
        return tuple(self.monsters)

    @property
    def remaining_collectibles(self) -> frozenset[Coord]:
        """Cells that still hold an uncollected item (an unopened chest counts as remaining)."""
        return frozenset(
            cell for cell, bit in self._bit_of.items() if not self.collected_mask >> bit & 1
        )

    @property
    def n_collected(self) -> int:
        return int(self.collected_mask.bit_count())

    def tile_at(self, row: int, col: int) -> str:
        """The static tile character; monsters and the agent are not part of the grid."""
        return self.grid[row][col]

    def in_bounds(self, row: int, col: int) -> bool:
        """Grid edges act as walls — the levels carry no rock border."""
        return 0 <= row < self.n_rows and 0 <= col < self.n_cols

    def is_blocked(self, row: int, col: int) -> bool:
        """Only rocks and the grid edge block. Items and fire are all enterable."""
        return not self.in_bounds(row, col) or self.grid[row][col] == Tile.ROCK

    def is_legal_move(self, action: int | Action) -> bool:
        """Whether `action` would actually displace the agent from where it stands.

        "Legal" here means "not walled off": a rock or the grid edge ahead makes the move a no-op.
        It is a query for the renderer and for debugging, never a filter on the action set — see
        `available_actions`.
        """
        d_row, d_col = ACTION_DELTAS[Action(int(action))]
        return not self.is_blocked(self.agent_pos[0] + d_row, self.agent_pos[1] + d_col)

    def available_actions(self) -> tuple[Action, ...]:
        """The action set an agent may choose from — always all four moves, in every state.

        Deliberately *not* filtered by `is_legal_move`. The brief makes bumping a rock a legal
        action with a no-op outcome, and a tabular agent needs column `a` of its Q-table to mean
        the same action in every row; pruning blocked directions would both relabel the table
        per-state and hide the fact that the agent has to learn not to walk into walls.
        """
        return tuple(Action)

    def moving_actions(self) -> tuple[Action, ...]:
        """The subset of `available_actions` that would change the agent's position."""
        return tuple(action for action in Action if self.is_legal_move(action))

    # --- the step function -------------------------------------------------------------------

    def step(self, action: int | Action) -> tuple[State, float, bool, dict[str, Any]]:
        """Apply one action in the order fixed by the brief. See the module docstring."""
        if self.done:
            raise RuntimeError("step() called after the episode ended; call reset() first")
        action = int(action)
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"action must be in 0..{N_ACTIONS - 1}, got {action}")

        self.steps += 1
        reward = REWARD_STEP

        # 1. Blocked move: no displacement, no reward change, no penalty.
        row, col = self.agent_pos
        d_row, d_col = ACTION_DELTAS[Action(action)]
        target = (row + d_row, col + d_col)
        if not self.is_blocked(*target):
            self.agent_pos = target

        # 2. First death check: the agent walked into fire or into a monster.
        if self._is_lethal(self.agent_pos):
            self.died = True
            reward += REWARD_DEATH
        else:
            # 3. Collect whatever is on the tile now occupied.
            reward += self._collect(self.agent_pos)

            # 4. Monsters move only while the episode is still running. Collecting the last item
            #    ends it at step 3, so no monster gets a free posthumous kill on the winning move.
            if not self.terminated:
                self._move_monsters()

                # 5. Second death check: a monster moved onto a stationary agent. The one that
                #    implementations miss, and the reason the check appears on both sides.
                if self.agent_pos in self.monsters:
                    self.died = True
                    reward += REWARD_DEATH

        # 6. Termination is `self.terminated`; the cap is a separate, non-game ending.
        if not self.terminated and self.steps >= self.max_steps:
            self.truncated = True

        self.episode_return += reward
        return self.state, reward, self.done, self._info()

    def _is_lethal(self, cell: Coord) -> bool:
        """Fire is static, monsters are entities — both kill on entry."""
        return self.grid[cell[0]][cell[1]] in LETHAL_TILES or cell in self.monsters

    def _collect(self, cell: Coord) -> float:
        """Pick up the item on `cell` and return its reward.

        The chest is the exception to "step onto it and it is yours": without the key it stays on
        the grid and pays nothing, so the same tile can be re-entered later once the key is held.
        """
        tile = self.grid[cell[0]][cell[1]]
        if tile not in COLLECTIBLE_TILES:
            return 0.0
        if tile == Tile.CHEST and not self.has_key:
            return 0.0

        if tile == Tile.KEY:
            self.has_key = True
        self.grid[cell[0]][cell[1]] = Tile.EMPTY
        self.collected_mask |= 1 << self._bit_of[cell]
        return TILE_REWARDS[tile]

    def _move_monsters(self) -> None:
        """Each monster independently moves with p = 0.4, uniformly among legal directions.

        Legal means in-grid and not a rock; the brief imposes no other restriction, so monsters may
        stand on fire and may share a cell. Monsters are drawn one at a time from the env's own
        generator, which is what makes a seeded episode replayable.
        """
        for index, (row, col) in enumerate(self.monsters):
            if self.rng.random() >= MONSTER_MOVE_PROBABILITY:
                continue
            choices = [
                (row + d_row, col + d_col)
                for d_row, d_col in ACTION_DELTAS.values()
                if not self.is_blocked(row + d_row, col + d_col)
            ]
            if not choices:
                continue  # walled in; nothing legal to pick from
            self.monsters[index] = choices[int(self.rng.integers(len(choices)))]

    def _info(self) -> dict[str, Any]:
        """Behavioural metrics for the training logs — reward alone hides why an episode ended."""
        return {
            "died": self.died,
            "collected": self.n_collected,
            "steps": self.steps,
            "has_key": self.has_key,
            "remaining": len(self.collectible_cells) - self.n_collected,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "episode_return": self.episode_return,
        }

    def __repr__(self) -> str:
        return (
            f"GridWorld(level={self.level_index}, agent={self.agent_pos}, has_key={self.has_key}, "
            f"collected={self.n_collected}/{len(self.collectible_cells)}, steps={self.steps})"
        )
