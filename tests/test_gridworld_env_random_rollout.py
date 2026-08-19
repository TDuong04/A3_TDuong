"""The M1 gate: a random policy across all seven shipped levels.

These are the tests the plan names as the gate for the environment core — no crash, every episode
terminates inside the cap, and the episode return equals the value of the items actually taken off
the grid (which is also what the per-step rewards sum to). They run on
the real levels rather than fixtures, so they also catch a level that has become unplayable.
"""

from __future__ import annotations

import pytest

from gridworld.constants import N_ACTIONS, TILE_REWARDS
from gridworld.env import GridWorld
from gridworld.levels import N_LEVELS

STEPS_PER_LEVEL = 10_000


@pytest.mark.parametrize("level", range(N_LEVELS))
def test_random_policy_survives_10000_steps_and_always_terminates(level):
    """No crash, and every episode ends — by termination or by the cap — within `max_steps`."""
    env = GridWorld(level_index=level, seed=level)
    rng = env.rng
    episodes = 0
    steps_this_episode = 0

    for _ in range(STEPS_PER_LEVEL):
        _, _, done, info = env.step(int(rng.integers(N_ACTIONS)))
        steps_this_episode += 1
        assert info["steps"] == steps_this_episode
        assert steps_this_episode <= env.max_steps, "episode ran past its cap"
        if done:
            assert info["terminated"] or info["truncated"]
            episodes += 1
            steps_this_episode = 0
            env.reset()

    assert episodes > 0, f"level {level} never finished an episode in {STEPS_PER_LEVEL} steps"


@pytest.mark.parametrize("level", range(N_LEVELS))
def test_episode_return_equals_the_value_of_what_was_actually_collected(level):
    """Reward accounting: a training curve plotted from returns is only meaningful if this holds.

    The expected total is derived from the grid — which collectibles vanished, priced by
    `TILE_REWARDS` — and not from the rewards `step` handed back. Comparing the summed rewards to
    `episode_return` alone cannot fail, because `step` accumulates the same value it returns; the
    independent total is what makes this a real check on both the per-step reward and the running
    return. Every other reward in the game is zero, so the value of the items taken *is* the return.
    """
    env = GridWorld(level_index=level, seed=100 + level)
    rng = env.rng

    for _ in range(20):
        env.reset()
        # Snapshot the level's collectibles before anything is taken: the grid is cleared as items
        # are picked up, so their tile characters have to be read now.
        value_of = {cell: TILE_REWARDS[env.tile_at(*cell)] for cell in env.collectible_cells}

        total = 0.0
        done = False
        while not done:
            _, reward, done, info = env.step(int(rng.integers(N_ACTIONS)))
            total += reward

        taken = set(value_of) - env.remaining_collectibles
        expected = sum(value_of[cell] for cell in taken)

        assert total == pytest.approx(expected), "a step paid something the grid cannot account for"
        assert env.episode_return == pytest.approx(expected)
        assert info["episode_return"] == pytest.approx(expected)


@pytest.mark.parametrize("level", range(N_LEVELS))
def test_episode_return_never_exceeds_the_levels_total_reward(level):
    """An agent cannot bank more than the level holds — a re-collectable item would show up here."""
    env = GridWorld(level_index=level, seed=200 + level)
    rng = env.rng
    best_possible = sum(TILE_REWARDS[env.tile_at(r, c)] for r, c in env.collectible_cells)

    for _ in range(20):
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(int(rng.integers(N_ACTIONS)))
        assert env.episode_return <= best_possible + 1e-9


@pytest.mark.parametrize("level", range(N_LEVELS))
def test_reset_restores_the_level_to_its_starting_layout(level):
    """`load_level` must hand back a fresh grid; a mutated module string would leak between runs."""
    env = GridWorld(level_index=level, seed=300 + level)
    starting_grid = [row[:] for row in env.grid]
    starting_collectibles = env.remaining_collectibles
    rng = env.rng

    while not env.done:
        env.step(int(rng.integers(N_ACTIONS)))
    env.reset()

    assert env.grid == starting_grid
    assert env.remaining_collectibles == starting_collectibles
    assert env.collected_mask == 0
    assert env.has_key is False
    assert env.steps == 0
    assert env.died is False
