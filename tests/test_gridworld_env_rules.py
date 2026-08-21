"""Rule-by-rule tests for `gridworld.env.GridWorld`.

Each test names the brief rule it pins down. The layouts are tiny hand-written levels injected in
place of `load_level`, because the shipped 10x10 levels cannot isolate a single rule — proving that
a monster kills a *stationary* agent needs a monster with exactly one legal move.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from gridworld import env as env_module
from gridworld.constants import MONSTER_MOVE_PROBABILITY, Action, Tile
from gridworld.env import GridWorld
from gridworld.levels import parse_level

REPO_ROOT = Path(__file__).resolve().parents[1]


class StubRng:
    """A deterministic stand-in for `np.random.Generator` covering the two calls the env makes.

    Monster movement is the only randomness in the env, so forcing `random()` to 0.0 or 1.0 makes
    "the monster always moves" and "the monster never moves" exact rather than seed-hunted.
    """

    def __init__(self, move: bool, choice: int = 0) -> None:
        self._value = 0.0 if move else 1.0
        self._choice = choice

    def random(self) -> float:
        return self._value

    def integers(self, n: int) -> int:
        return self._choice % n


@pytest.fixture
def make_env(monkeypatch):
    """Build a `GridWorld` over a hand-written layout, bypassing `levels.LEVELS`."""

    def _make(layout: str, **kwargs) -> GridWorld:
        monkeypatch.setattr(env_module, "load_level", lambda index: parse_level(layout))
        kwargs.setdefault("max_steps", 100)
        return GridWorld(level_index=0, **kwargs)

    return _make


# --- rocks and grid edges ---------------------------------------------------------------------


def test_rock_blocks_with_no_displacement_and_no_penalty(make_env):
    env = make_env("S#A\n...\n...\n")
    state, reward, done, info = env.step(Action.RIGHT)

    assert env.agent_pos == (0, 0), "the agent must not be displaced by a blocked move"
    assert reward == 0.0, "bumping a rock is neither rewarded nor penalised"
    assert done is False
    assert info["died"] is False
    assert state == (0, 0, False, 0)


def test_grid_edge_blocks_like_a_rock(make_env):
    env = make_env("S.A\n...\n...\n")
    _, reward, done, _ = env.step(Action.UP)

    assert env.agent_pos == (0, 0)
    assert reward == 0.0
    assert done is False


def test_rock_blocks_before_the_death_check(make_env):
    """Step 1 runs before step 2: a monster behind a rock is unreachable, so nothing dies."""
    env = make_env("S#M\nA..\n", rng=StubRng(move=False))
    _, reward, done, info = env.step(Action.RIGHT)

    assert env.agent_pos == (0, 0)
    assert info["died"] is False
    assert reward == 0.0
    assert done is False


# --- collectibles -----------------------------------------------------------------------------


def test_apple_pays_one(make_env):
    env = make_env("SA\n..\n")
    _, reward, _, info = env.step(Action.RIGHT)

    assert reward == 1.0
    assert info["collected"] == 1


def test_key_pays_nothing_and_sets_has_key(make_env):
    env = make_env("KA\nS.\n")
    _, reward, done, info = env.step(Action.UP)

    assert reward == 0.0
    assert env.has_key is True
    assert info["has_key"] is True
    assert done is False, "an apple is still uncollected"


def test_chest_pays_nothing_and_stays_put_without_the_key(make_env):
    env = make_env("CA\nS.\n")
    _, reward, done, info = env.step(Action.UP)

    assert env.agent_pos == (0, 0), "the chest does not block, only rocks do"
    assert reward == 0.0
    assert env.tile_at(0, 0) == Tile.CHEST, "an unopened chest stays on the grid"
    assert info["collected"] == 0
    assert (0, 0) in env.remaining_collectibles
    assert done is False


def test_chest_pays_two_while_the_key_is_held(make_env):
    env = make_env("KC\nS.\n")
    env.step(Action.UP)  # take the key at (0, 0)
    _, reward, done, info = env.step(Action.RIGHT)  # open the chest at (0, 1)

    assert reward == 2.0
    assert done is True, "key and chest were the only collectibles"
    assert info["collected"] == 2
    assert info["died"] is False


def test_chest_pays_on_a_second_visit_once_the_key_is_taken(make_env):
    """The no-key visit must not consume the chest, or the level becomes unwinnable."""
    env = make_env("CK\nS.\n")
    _, first, _, _ = env.step(Action.UP)  # onto the chest, no key
    env.step(Action.RIGHT)  # take the key
    _, second, done, _ = env.step(Action.LEFT)  # back onto the chest, now holding it

    assert first == 0.0
    assert second == 2.0
    assert done is True


def test_level_2_chest_pays_nothing_until_the_key_is_taken():
    """The plan's M1 gate, on the shipped level rather than a fixture.

    The route walks down the left edge, along row 7 and up to the chest at (5,8), then detours to
    the key at (8,1) and comes back — so the same chest tile is entered twice, once each way.
    """
    env = GridWorld(level_index=2, seed=0)

    def walk(actions) -> float:
        return sum(env.step(action)[1] for action in actions)

    without_key = walk([Action.DOWN] * 7 + [Action.RIGHT] * 8 + [Action.UP] * 2)
    assert env.agent_pos == (5, 8)
    assert without_key == 0.0
    assert env.tile_at(5, 8) == Tile.CHEST, "the chest stays put when the key is not held"

    walk([Action.DOWN] * 3 + [Action.LEFT] * 7)  # the apple at (8,7) and the key at (8,1)
    assert env.has_key is True

    with_key = walk([Action.RIGHT] * 7 + [Action.UP] * 3)
    assert env.agent_pos == (5, 8)
    assert with_key == 2.0
    assert env.tile_at(5, 8) == Tile.EMPTY


# --- termination ------------------------------------------------------------------------------


def test_episode_ends_when_the_last_collectible_is_taken(make_env):
    env = make_env("SA\n..\n")
    _, _, done, info = env.step(Action.RIGHT)

    assert done is True
    assert info["terminated"] is True
    assert info["died"] is False
    assert env.remaining_collectibles == frozenset()


def test_episode_ends_on_stepping_into_fire(make_env):
    env = make_env("SF\nA.\n")
    _, reward, done, info = env.step(Action.RIGHT)

    assert done is True
    assert info["died"] is True
    assert reward == 0.0, "the brief specifies no explicit death penalty"
    assert info["collected"] == 0


def test_episode_ends_on_stepping_into_a_monster(make_env):
    env = make_env("SM\nA.\n", rng=StubRng(move=False))
    _, _, done, info = env.step(Action.RIGHT)

    assert done is True
    assert info["died"] is True


def test_monster_moving_onto_a_stationary_agent_kills_it(make_env):
    """The second death check. The agent bumps the edge and never moves; the monster comes to it."""
    env = make_env("S.A\nM#.\n#..\n", rng=StubRng(move=True))
    assert env.monster_positions == ((1, 0),), "only (0,0) is a legal move for this monster"

    _, _, done, info = env.step(Action.LEFT)  # off-grid: the agent stays on (0, 0)

    assert env.agent_pos == (0, 0)
    assert env.monster_positions == ((0, 0),)
    assert done is True
    assert info["died"] is True


def test_dying_on_a_tile_forfeits_the_item_standing_on_it(make_env):
    """Ordering: the death check (step 2) precedes collection (step 3).

    A monster parked on an apple is the only situation that separates the two — if collection ran
    first the agent would bank the apple on the move that kills it, looting the tile it dies on.
    """
    env = make_env("SA\n..\n", rng=StubRng(move=False))
    env.monsters = [(0, 1)]  # the monster stands on the apple

    _, reward, done, info = env.step(Action.RIGHT)

    assert info["died"] is True
    assert info["collected"] == 0, "the agent must not loot the tile it dies on"
    assert env.tile_at(0, 1) == Tile.APPLE, "the apple stays on the grid"
    assert reward == 0.0, "death pays nothing, and the forfeited apple pays nothing either"
    assert (0, 1) in env.remaining_collectibles
    assert done is True


def test_collecting_the_last_item_ends_the_episode_before_monsters_move(make_env):
    """Ordering: collection (step 3) precedes monster movement (step 4)."""
    env = make_env("SA\n.M\n", rng=StubRng(move=True, choice=0))
    _, reward, done, info = env.step(Action.RIGHT)

    assert reward == 1.0
    assert done is True
    assert info["died"] is False
    assert env.monster_positions == ((1, 1),), "a finished episode does not move monsters"


def test_episode_truncates_at_max_steps_without_terminating(make_env):
    env = make_env("SA\n..\n", max_steps=5)
    for _ in range(5):
        _, _, done, info = env.step(Action.UP)  # off-grid every time: the episode cannot end

    assert done is True
    assert info["steps"] == 5
    assert info["truncated"] is True
    assert info["terminated"] is False
    assert info["died"] is False


@pytest.fixture
def clear_max_steps_cache():
    """Isolate `env._configured_max_steps`, which is `lru_cache`d for the life of the process.

    Without this, a value read while the config is patched would be cached and served to every
    later test in the session — the cache is cleared on the way in as well as on the way out.
    """
    env_module._configured_max_steps.cache_clear()
    yield
    env_module._configured_max_steps.cache_clear()


def test_max_steps_is_read_from_config_including_the_level_override(
    monkeypatch, clear_max_steps_cache
):
    """Rubric B3: the episode cap is a tuned number, so it must come from `config/gridworld.yaml`.

    The values are deliberately unlike anything the real config or the code could hold, so a
    literal baked into `_configured_max_steps` cannot coincidentally satisfy this.
    """
    patched_config = {
        "training": {"max_steps_per_episode": 137},
        "level_overrides": {5: {"max_steps_per_episode": 911}},
    }
    monkeypatch.setattr(env_module, "load_yaml", lambda name: patched_config)

    assert GridWorld(level_index=0).max_steps == 137, "the base cap must come from `training`"
    assert GridWorld(level_index=5).max_steps == 911, "a per-level override must win"


def test_step_after_the_episode_ended_is_rejected(make_env):
    env = make_env("SA\n..\n")
    env.step(Action.RIGHT)
    with pytest.raises(RuntimeError):
        env.step(Action.UP)


def test_illegal_action_is_rejected(make_env):
    env = make_env("SA\n..\n")
    with pytest.raises(ValueError):
        env.step(4)


# --- the action set ---------------------------------------------------------------------------


def test_available_actions_is_the_full_action_set_in_every_state(make_env):
    """A tabular Q-table needs column `a` to mean the same action in every row."""
    env = make_env("S#\n#A\n")  # boxed in: both moves out of the start are blocked
    assert env.available_actions() == tuple(Action)
    env.step(Action.DOWN)
    assert env.available_actions() == tuple(Action)


def test_is_legal_move_reports_rocks_and_grid_edges(make_env):
    env = make_env("S#.\n...\n...\n")
    assert not env.is_legal_move(Action.UP)  # top edge
    assert not env.is_legal_move(Action.LEFT)  # left edge
    assert not env.is_legal_move(Action.RIGHT)  # rock
    assert env.is_legal_move(Action.DOWN)


def test_moving_actions_is_the_subset_that_displaces_the_agent(make_env):
    env = make_env("S#.\n.A.\n...\n")
    assert env.moving_actions() == (Action.DOWN,)
    env.step(Action.DOWN)
    assert set(env.moving_actions()) == {Action.UP, Action.DOWN, Action.RIGHT}


def test_a_blocked_move_is_still_accepted_by_step(make_env):
    """`available_actions` promises it, so `step` must honour it: no-op, not an error."""
    env = make_env("S#\n.A\n")
    before = env.agent_pos
    _, reward, done, _ = env.step(Action.RIGHT)
    assert env.agent_pos == before
    assert reward == 0.0
    assert not done


# --- the state key ----------------------------------------------------------------------------


def test_state_key_distinguishes_the_same_tile_with_and_without_the_key(make_env):
    env = make_env("KA\nS.\n")
    before = env.state
    env.step(Action.UP)  # take the key
    after = env.step(Action.DOWN)[0]  # return to the starting tile

    assert before[:2] == after[:2], "same tile"
    assert before != after, "position alone is not Markov once a key exists"
    assert before[2] is False and after[2] is True


def test_state_key_distinguishes_remaining_collectibles(make_env):
    env = make_env("AA\nS.\n")
    before = env.state
    env.step(Action.UP)  # take one apple
    after = env.step(Action.DOWN)[0]

    assert before[:2] == after[:2]
    assert before[3] != after[3], "the collected mask must be part of the key"


def test_collected_mask_bit_order_is_sorted_and_stable_across_episodes(make_env):
    env = make_env("AAA\n.S.\n...\n")
    order = env.collectible_cells
    assert order == tuple(sorted(order))

    env.step(Action.UP)
    mask_first_episode = env.collected_mask
    assert mask_first_episode != 0, "the step above must have collected something"

    env.reset()
    env.step(Action.UP)

    assert env.collectible_cells == order
    assert env.collected_mask == mask_first_episode


# --- info payload and reward accounting -------------------------------------------------------


def test_info_carries_died_collected_and_steps(make_env):
    env = make_env("SA\n..\n")
    _, _, _, info = env.step(Action.DOWN)

    assert {"died", "collected", "steps"} <= info.keys()
    assert info["died"] is False
    assert info["collected"] == 0
    assert info["steps"] == 1


def test_summed_step_rewards_equal_the_episode_return(make_env):
    env = make_env("KC\nSA\n")
    total = 0.0
    for action in (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT):
        _, reward, done, info = env.step(action)
        total += reward
        if done:
            break

    assert total == pytest.approx(env.episode_return)
    assert total == pytest.approx(info["episode_return"])
    assert total == pytest.approx(3.0), "apple +1 and chest +2, key 0"


# --- monster movement -------------------------------------------------------------------------


def test_monsters_move_at_roughly_the_configured_probability(make_env):
    """The rock column walls the monster off, so the sample is not cut short by the agent dying."""
    env = make_env("S#.......\nA#.M.....\n.#.......\n", seed=7, max_steps=10_000)
    moves = 0
    trials = 5_000
    for _ in range(trials):
        before = env.monster_positions
        env.step(Action.UP)  # off-grid: the agent stays put and never collects
        if env.monster_positions != before:
            moves += 1

    assert moves / trials == pytest.approx(MONSTER_MOVE_PROBABILITY, abs=0.03)


def test_monsters_never_enter_rocks_or_leave_the_grid(make_env):
    env = make_env("S#M\nA#.\n.#.\n", seed=3, max_steps=10_000)
    for _ in range(500):
        env.step(Action.RIGHT)  # blocked by the rock column; the agent never moves
        for row, col in env.monster_positions:
            assert env.in_bounds(row, col)
            assert env.tile_at(row, col) != Tile.ROCK


# --- headless guarantee -----------------------------------------------------------------------


def test_env_module_source_has_no_pygame_import():
    source = (REPO_ROOT / "gridworld" / "env.py").read_text(encoding="utf-8")
    assert re.search(r"^\s*(import|from)\s+pygame", source, re.MULTILINE) is None


def test_importing_the_env_does_not_pull_in_pygame():
    """Guards the whole import path, not just this file — training must run headless."""
    result = subprocess.run(
        [sys.executable, "-c", "import gridworld.env, sys; assert 'pygame' not in sys.modules"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
