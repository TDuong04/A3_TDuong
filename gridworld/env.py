"""Gridworld environment — NOT YET IMPLEMENTED.

Contract for whoever builds this:

`GridWorld(level_index, rng)` holds the mutable grid, the agent position, `has_key`, and the set of
remaining collectibles.

`reset()` reloads the level from `levels.load_level` (never mutate the module-level strings) and
returns the initial state.

`state` must be a hashable key for the Q-table. Position alone is NOT Markov on levels 2-6: two
visits to the same tile differ by which collectibles remain and whether the key is held. Use
`(row, col, has_key, frozenset(remaining_collectibles))` or an equivalent bitmask. Getting this
wrong makes the chest levels unlearnable in a way that looks like a hyperparameter problem.

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

Returns `(state, reward, done, info)`; put `died`, `collected` and `steps` in `info` so the
training scripts can log behavioural metrics rather than reward alone.

Keep this file free of pygame imports — rendering lives in `render.py` so training runs headless.
"""

from __future__ import annotations


class GridWorld:
    def __init__(self, level_index: int, rng=None) -> None:
        raise NotImplementedError("see module docstring for the contract")
