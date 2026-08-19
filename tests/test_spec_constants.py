"""Guards against drift from the brief.

The brief forbids altering rewards or mechanics, and rubric rows A2, I1 and I2 check these values
directly. If one of these tests fails, the fix is to restore the constant — not to update the test.
"""

from __future__ import annotations

from arena import constants as arena_c
from gridworld import constants as grid_c


class TestGridworldSpec:
    def test_rewards(self):
        assert grid_c.REWARD_APPLE == 1.0
        assert grid_c.REWARD_KEY == 0.0
        assert grid_c.REWARD_CHEST == 2.0

    def test_monster_move_probability(self):
        assert grid_c.MONSTER_MOVE_PROBABILITY == 0.4

    def test_exactly_four_actions(self):
        assert grid_c.N_ACTIONS == 4
        assert set(grid_c.ACTION_DELTAS) == set(grid_c.Action)

    def test_hazards_are_lethal(self):
        assert grid_c.Tile.FIRE in grid_c.LETHAL_TILES
        assert grid_c.Tile.MONSTER in grid_c.LETHAL_TILES

    def test_key_is_collectible_but_unrewarded(self):
        assert grid_c.Tile.KEY in grid_c.COLLECTIBLE_TILES
        assert grid_c.TILE_REWARDS[grid_c.Tile.KEY] == 0.0


class TestArenaSpec:
    def test_rotation_action_indices(self):
        a = arena_c.RotationAction
        assert (a.NOOP, a.THRUST, a.ROTATE_LEFT, a.ROTATE_RIGHT, a.SHOOT) == (0, 1, 2, 3, 4)
        assert arena_c.N_ACTIONS["rotation"] == 5

    def test_direct_action_indices(self):
        a = arena_c.DirectAction
        assert (a.NOOP, a.UP, a.DOWN, a.LEFT, a.RIGHT, a.SHOOT) == (0, 1, 2, 3, 4, 5)
        assert arena_c.N_ACTIONS["direct"] == 6

    def test_reward_signs(self):
        assert arena_c.REWARD_ENEMY_DESTROYED > 0
        assert arena_c.REWARD_SPAWNER_DESTROYED > arena_c.REWARD_ENEMY_DESTROYED
        assert arena_c.REWARD_PHASE_ADVANCE > arena_c.REWARD_SPAWNER_DESTROYED
        assert arena_c.REWARD_DAMAGE_TAKEN < 0
        assert arena_c.REWARD_DEATH < arena_c.REWARD_DAMAGE_TAKEN

    def test_no_positive_survival_reward(self):
        """A positive per-step reward teaches the agent to hide in a corner forever."""
        assert arena_c.REWARD_PER_STEP <= 0

    def test_observation_is_within_brief_guidance(self):
        assert 10 <= arena_c.OBS_DIM <= 30
