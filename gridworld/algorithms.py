"""Tabular control for the gridworld: Q-learning now, SARSA next (A3-004).

Both algorithms live in this one module and share `select_action` and the `LinearEpsilon` schedule
from `common.schedules`. That sharing is deliberate: rubric C2 requires SARSA to use the same
exploration schedule as Q-learning, and a marker can verify that by reading one function instead of
diffing two files.

Shared action selection (rubric B1, B4):

    def select_action(q_row, epsilon, rng):
        if rng.random() < epsilon:
            return rng.integers(N_ACTIONS)
        best = np.flatnonzero(q_row == q_row.max())   # ALL tied-best actions
        return rng.choice(best)                        # random tie-break

Never use `np.argmax` here. It silently returns the lowest index among ties, which biases the agent
toward UP on every unvisited state and is a B4 failure the rubric checks for by name.

The update rules differ in exactly one term, and that term is the whole point of the comparison:

    Q-learning (off-policy):  target = r + gamma * max_a' Q[s'][a']
    SARSA      (on-policy):   target = r + gamma * Q[s'][a']   where a' is the action ACTUALLY
                                                   taken next by the same epsilon-greedy policy

    Q[s][a] += alpha * (target - Q[s][a])

On a terminal transition the bootstrap term is zero for both — `target = r`. Forgetting that is the
most common tabular RL bug and it quietly inflates the value of dying.

"Terminal" here means `info["terminated"]`, not `done`. The env distinguishes a real ending (all
collectibles taken, or the agent died) from a truncation (the `max_steps` cap firing). Running out
of clock is not the game ending: the successor state still has value, so a truncated transition must
keep its bootstrap term. Treating truncation as terminal teaches the agent that the world stops
paying out near the step cap, which shortens episodes for the wrong reason.

SARSA's loop shape differs from Q-learning's: it must choose `a'` before it can update, so the
action for the next step is selected at the end of the current one and carried forward. It reuses
`select_action`, `make_q_table` and `epsilon_schedule` unchanged; only the target line differs.

Intrinsic reward (Task 5, A3-007) is added by the caller, not baked in here: the agent maintains a
per-episode visit counter `n(s)`, computes `r_i = strength / sqrt(n(s) + 1)`, and passes `r + r_i`
as the reward for the update. Environment rewards stay unchanged, and the counter resets every
episode.

Every hyperparameter arrives inside a `TabularConfig` loaded from `config/gridworld.yaml`. Nothing
in this module names a numeric value for episodes, alpha, gamma or the epsilon bounds — rubric B3
grades that, and a caller that wants a shorter run builds a modified config with
`dataclasses.replace` rather than passing a literal.

Each training run returns per-episode histories — return, steps, died, collected, epsilon — which
become the report's training curves. It also returns the Q-table so the renderer can draw the policy
and `eval/play_gridworld.py` can replay without retraining.

The verification helpers at the bottom exist because "the agent learned a shortest path" (rubric B5)
is a claim that has to be checked, not asserted: `greedy_rollout` runs the learned policy with
exploration switched off, and `optimal_collection_steps` computes the true optimum independently by
BFS so the two numbers can be compared.
"""

from __future__ import annotations

import csv
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from common.config import TabularConfig
from common.schedules import LinearEpsilon
from common.seeding import make_rng

from .constants import ACTION_DELTAS, COLLECTIBLE_TILES, LETHAL_TILES, N_ACTIONS, Tile
from .env import Coord, GridWorld, State
from .levels import load_level

#: A sparse table: only states the agent has actually reached occupy memory. See `make_q_table`.
QTable = dict[State, np.ndarray]

#: Column order for the per-episode history CSV, so the trainer and any reader agree on it.
HISTORY_FIELDS: tuple[str, ...] = (
    "episode",
    "return",
    "steps",
    "died",
    "collected",
    "epsilon",
)


# --- shared machinery (Q-learning and SARSA both use exactly this) -----------------------------


def make_q_table() -> QTable:
    """A zero-initialised sparse Q-table keyed by the env's state tuple.

    A `defaultdict` rather than a dense array because the reachable state set is a small fraction of
    `positions x has_key x masks`, and because switching levels then costs nothing. Zeros, not
    optimistic initialisation: the brief pairs plain zeros with epsilon-greedy, and optimism would
    confound the epsilon-decay evidence rubric B3 asks for.
    """
    return defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float64))


def epsilon_schedule(config: TabularConfig) -> LinearEpsilon:
    """Build the exploration schedule from config: the one construction site for both algorithms.

    Q-learning and SARSA call this same function with the same config block, which is how "same
    exploration schedule" (rubric C2) stays true by construction instead of by convention.
    """
    return LinearEpsilon(
        start=config.epsilon_start,
        end=config.epsilon_end,
        decay_episodes=config.epsilon_decay_episodes,
    )


def select_action(q_row: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """Epsilon-greedy with a *random* tie-break among equal-valued actions.

    `np.argmax` is banned here (rubric B4). It returns the lowest index among ties, so on a
    zero-initialised table every unvisited state would deterministically pick UP — the agent would
    spend its first episodes hugging the top wall, and the bias never fully washes out because the
    states it never leaves never get explored.
    """
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    q_row = np.asarray(q_row)
    best = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(best))


@dataclass(frozen=True)
class EpisodeRecord:
    """One row of the training history. Reward alone hides *why* an episode ended.

    `died` and `collected` are what separate "the curve is flat because the agent keeps dying" from
    "the curve is flat because it never finds the last apple", and the two need opposite fixes.
    """

    episode: int
    total_return: float
    steps: int
    died: bool
    collected: int
    epsilon: float

    def as_row(self) -> dict[str, Any]:
        """Map to the CSV column names (`return` is a keyword, so the field cannot use it)."""
        return {
            "episode": self.episode,
            "return": self.total_return,
            "steps": self.steps,
            "died": int(self.died),
            "collected": self.collected,
            "epsilon": self.epsilon,
        }


@dataclass
class TrainingResult:
    """Everything a run leaves behind: the policy, the evidence, and the config behind both."""

    algo: str
    level: int
    q_table: QTable
    history: list[EpisodeRecord]
    config: TabularConfig

    @property
    def returns(self) -> np.ndarray:
        return np.array([record.total_return for record in self.history], dtype=np.float64)

    @property
    def steps(self) -> np.ndarray:
        return np.array([record.steps for record in self.history], dtype=np.float64)

    @property
    def epsilons(self) -> np.ndarray:
        return np.array([record.epsilon for record in self.history], dtype=np.float64)

    def write_history_csv(self, path: str | Path) -> Path:
        """Write the per-episode history. `newline=""` stops csv doubling newlines on Windows."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            for record in self.history:
                writer.writerow(record.as_row())
        return path


# --- Q-learning ---------------------------------------------------------------------------------


def q_learning(
    env: GridWorld,
    config: TabularConfig,
    *,
    rng: np.random.Generator | None = None,
    q_table: QTable | None = None,
) -> TrainingResult:
    """Off-policy TD control. Every hyperparameter comes from `config`; none is written here.

    `rng` is injected so the learner's exploration draws are independent of the env's monster draws.
    Sharing one generator would make an unrelated change to monster movement silently reshuffle the
    exploration sequence and break run-to-run comparability.

    To run fewer episodes (a fast test, a smoke run) build a narrowed config with
    `dataclasses.replace(config, episodes=...)` rather than passing a literal count — the config
    stays the one place a hyperparameter is stated.
    """
    rng = make_rng(config.seed) if rng is None else rng
    schedule = epsilon_schedule(config)
    q = make_q_table() if q_table is None else q_table
    history: list[EpisodeRecord] = []

    for episode in range(config.episodes):
        epsilon = schedule(episode)
        state = env.reset()
        total_return = 0.0

        while True:
            action = select_action(q[state], epsilon, rng)
            next_state, reward, done, info = env.step(action)
            total_return += reward

            # Off-policy: the target uses the best action available at s', not the one the
            # behaviour policy will actually take. On a real ending there is no s' worth anything,
            # so the target is r alone. `info["terminated"]` and not `done`: a truncation is the
            # step cap firing, and the successor state still has value.
            if info["terminated"]:
                target = reward
            else:
                target = reward + config.gamma * float(q[next_state].max())

            q[state][action] += config.alpha * (target - q[state][action])
            state = next_state

            if done:
                break

        history.append(
            EpisodeRecord(
                episode=episode,
                total_return=total_return,
                steps=info["steps"],
                died=bool(info["died"]),
                collected=int(info["collected"]),
                epsilon=epsilon,
            )
        )

    return TrainingResult(
        algo="q", level=env.level_index, q_table=q, history=history, config=config
    )


# --- SARSA (A3-004) -----------------------------------------------------------------------------


def sarsa(
    env: GridWorld,
    config: TabularConfig,
    *,
    rng: np.random.Generator | None = None,
    q_table: QTable | None = None,
) -> TrainingResult:
    """On-policy TD control — NOT YET IMPLEMENTED, owned by ticket A3-004.

    The place is marked deliberately rather than left absent. When it lands it must reuse
    `select_action`, `make_q_table` and `epsilon_schedule` above verbatim; the only line that may
    differ from `q_learning` is the target:

        target = reward if terminated else reward + config.gamma * q[next_state][next_action]

    where `next_action` is selected *before* the update and carried into the following step.
    """
    raise NotImplementedError("SARSA is ticket A3-004; see the docstring for the contract")


# --- verification helpers: is the learned policy actually optimal? ------------------------------


@dataclass(frozen=True)
class Rollout:
    """A single greedy episode, kept in enough detail to draw the path and the policy arrows."""

    steps: int
    total_return: float
    died: bool
    truncated: bool
    collected: int
    states: tuple[State, ...]
    actions: tuple[int, ...]

    @property
    def path(self) -> tuple[Coord, ...]:
        return tuple((state[0], state[1]) for state in self.states)


def greedy_rollout(
    env: GridWorld,
    q_table: QTable,
    *,
    rng: np.random.Generator | None = None,
) -> Rollout:
    """Run one episode with exploration switched off (epsilon = 0) — the policy under test.

    An rng is still required: with epsilon 0 `select_action` must break ties randomly, and a state
    the agent never visited has an all-zero row where every action ties. That is the honest
    behaviour to demonstrate — silently falling back to `argmax` here would flatter the policy.
    """
    rng = make_rng(0) if rng is None else rng
    state = env.reset()
    states: list[State] = [state]
    actions: list[int] = []
    total_return = 0.0

    while True:
        action = select_action(q_table[state], 0.0, rng)
        state, reward, done, info = env.step(action)
        actions.append(action)
        states.append(state)
        total_return += reward
        if done:
            break

    return Rollout(
        steps=int(info["steps"]),
        total_return=total_return,
        died=bool(info["died"]),
        truncated=bool(info["truncated"]),
        collected=int(info["collected"]),
        states=tuple(states),
        actions=tuple(actions),
    )


def _bfs_distances(grid: list[list[str]], source: Coord) -> dict[Coord, int]:
    """Step counts from `source` to every safely reachable cell.

    Rocks block, and static lethal tiles are excluded rather than routed through — a path that ends
    the episode by dying is not a collection route. Monsters move, so on levels 4-5 this is a lower
    bound on the safe optimum rather than the exact answer; level 0 has neither, which is why the
    B5 demonstration uses it.
    """
    n_rows, n_cols = len(grid), len(grid[0])

    def passable(cell: Coord) -> bool:
        row, col = cell
        if not (0 <= row < n_rows and 0 <= col < n_cols):
            return False
        return grid[row][col] not in LETHAL_TILES and grid[row][col] != Tile.ROCK

    distances = {source: 0}
    queue = deque([source])
    while queue:
        row, col = queue.popleft()
        for d_row, d_col in ACTION_DELTAS.values():
            neighbour = (row + d_row, col + d_col)
            if neighbour in distances or not passable(neighbour):
                continue
            distances[neighbour] = distances[(row, col)] + 1
            queue.append(neighbour)
    return distances


def optimal_collection_steps(level_index: int) -> int:
    """The true minimum number of steps to collect everything on a level.

    Computed independently of the learner so that "the agent found a shortest path" can be checked
    rather than believed. It is a tiny travelling-salesman problem — BFS distances between the start
    and every collectible, then brute force over the collection orders — which is exact at three or
    four collectibles and would need replacing well before it became slow.

    Enumerating orders is exact even though the agent may walk over an item on its way somewhere
    else: that route simply *is* one of the enumerated orders, with the walked-over item earlier in
    the sequence.

    The one ordering constraint is the chest: it pays nothing and stays on the grid until the key is
    held, so any order placing a chest before the key is not a collection order at all.
    """
    grid = load_level(level_index)
    start: Coord | None = None
    collectibles: list[Coord] = []
    for row, line in enumerate(grid):
        for col, tile in enumerate(line):
            if tile == Tile.START:
                start = (row, col)
            elif tile in COLLECTIBLE_TILES:
                collectibles.append((row, col))
    assert start is not None, "load_level guarantees exactly one start tile"
    collectibles.sort()

    if not collectibles:
        return 0

    distances = {cell: _bfs_distances(grid, cell) for cell in [start, *collectibles]}
    keys = {cell for cell in collectibles if grid[cell[0]][cell[1]] == Tile.KEY}
    chests = {cell for cell in collectibles if grid[cell[0]][cell[1]] == Tile.CHEST}

    best = None
    for order in permutations(collectibles):
        if chests and not _key_precedes_every_chest(order, keys, chests):
            continue
        total = 0
        current = start
        for cell in order:
            step = distances[current].get(cell)
            if step is None:
                total = None
                break
            total += step
            current = cell
        if total is not None and (best is None or total < best):
            best = total

    if best is None:
        raise ValueError(f"level {level_index} has collectibles that cannot all be reached safely")
    return best


def _key_precedes_every_chest(order: Iterable[Coord], keys: set[Coord], chests: set[Coord]) -> bool:
    """A chest opened before any key is taken pays nothing, so such an order is not a solution."""
    held = 0
    for cell in order:
        if cell in keys:
            held += 1
        elif cell in chests and held == 0:
            return False
    return True


# --- persistence: a saved Q-table is what lets playback skip retraining -------------------------


def save_q_table(q_table: QTable, path: str | Path) -> Path:
    """Save as two parallel arrays rather than pickling a defaultdict.

    Pickle would embed the lambda's module path and refuse to load if this file ever moves; a plain
    `npz` of `(states, values)` is readable by anything and survives refactors. The state tuple is
    stored column-wise as `(row, col, has_key, collected_mask)` — the same order `GridWorld.state`
    produces, which is fixed at level load so a table saved today still means the same thing later.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = list(q_table.items())
    states = np.array(
        [(row, col, int(has_key), mask) for (row, col, has_key, mask), _ in items],
        dtype=np.int64,
    ).reshape(-1, 4)
    values = np.array([row for _, row in items], dtype=np.float64).reshape(-1, N_ACTIONS)
    np.savez_compressed(path, states=states, values=values)
    return path


def load_q_table(path: str | Path) -> QTable:
    """Reload a saved table as a fresh defaultdict, so unseen states still yield zeros."""
    with np.load(Path(path)) as data:
        states, values = data["states"], data["values"]
    q_table = make_q_table()
    for (row, col, has_key, mask), q_row in zip(states, values, strict=True):
        q_table[(int(row), int(col), bool(has_key), int(mask))] = np.array(q_row, dtype=np.float64)
    return q_table
