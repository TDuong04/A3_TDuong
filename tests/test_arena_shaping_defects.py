"""The four shaping defects behind the reported agent behaviour, pinned as tests.

Each test names a behaviour a marker (or a player) actually sees, and asserts the property of
`phi(s)` that has to hold for a policy to be able to learn it. They are unit tests on the
potentials rather than rollouts because the potentials are what the gradient sees: a rollout can
only tell you the agent misbehaves, not which term taught it to.

Measured behaviour of the shipped `models/ppo_rotation.zip` that motivated each one, over 6
deterministic episodes under the current config:

* 40.7% of all steps had no enemy alive, and the agent spent them idling rather than finishing
  the spawners -- `_aim_potential` was flat (returned 0.0) for every one of those steps.
* corner dwelling rose 0.9% -> 15.0% after the rebalance; `_safety_potential` is maximised in a
  corner, because the farthest points from anything in a rectangle are its corners.
* the agent fired at 32.1% of steps against a 33% cooldown ceiling, median 33.6 degrees off
  target: it holds the trigger and never aims.
* one episode crossed the nose 392 times without settling; a cosine potential is stationary
  exactly where an enemy chasing directly astern sits.
"""

from __future__ import annotations

import math

import pytest

from arena.entities import Enemy, Spawner
from arena.env import ArenaEnv


def quiet_arena(seed: int = 1) -> ArenaEnv:
    """A rotation arena with the phase's own spawners parked far away and muted."""
    env = ArenaEnv(control_style="rotation", seed=seed)
    for index, spawner in enumerate(env.spawners):
        spawner.x, spawner.y = 900.0, 40.0 + index * 40.0
        spawner.spawn_timer = 1.0e6
    env.player.x, env.player.y = 480.0, 340.0
    env.player.vx = env.player.vy = 0.0
    env.player.heading = 0.0
    env.enemies = []
    return env


class TestItLooksAtSpawnersOnceTheEnemiesAreGone:
    """Symptom: with the arena cleared the ship shoots into empty space instead of finishing the
    spawners, and sits out 40.7% of the episode doing it."""

    def test_the_aim_potential_is_not_flat_when_only_spawners_remain(self):
        env = quiet_arena()
        env.spawners = [Spawner(env.player.x + 200.0, env.player.y, env.phase_settings)]

        env.player.heading = 0.0                      # pointed straight at it
        aligned = env._aim_potential()
        env.player.heading = math.pi                  # pointed straight away
        away = env._aim_potential()

        assert aligned > away, "no gradient toward a spawner: the agent cannot learn to face one"

    def test_a_living_enemy_still_outranks_a_spawner(self):
        env = quiet_arena()
        env.spawners = [Spawner(env.player.x + 100.0, env.player.y, env.phase_settings)]
        env.enemies = [Enemy(env.player.x - 300.0, env.player.y, health=1, speed=0.0)]

        env.player.heading = math.pi                  # facing the enemy, away from the spawner
        facing_enemy = env._aim_potential()
        env.player.heading = 0.0                      # facing the spawner, away from the enemy
        facing_spawner = env._aim_potential()

        assert facing_enemy > facing_spawner, "enemies kill the player; they come first"


class TestTheLeadPointIsOneABulletCanActuallyReach:
    """Symptom: the agent will not hit an enemy in front of it. A bullet is given a fixed
    world-frame velocity and inherits nothing from the ship, so a lead solved against the
    target's velocity *relative to the ship* aims at a point no bullet ever flies through --
    and the error grows with ship speed, which the rebalance raised 220 -> 380."""

    def test_the_lead_ignores_the_ships_own_velocity(self):
        env = quiet_arena()
        env.enemies = [Enemy(env.player.x + 300.0, env.player.y, health=1, speed=0.0)]
        env.enemies[0].vx, env.enemies[0].vy = 0.0, 120.0

        # The correct firing solution depends only on the target's motion.
        env.player.vx, env.player.vy = 0.0, 0.0
        still = _best_heading(env)
        env.player.vx, env.player.vy = 0.0, 300.0     # same target, ship now sprinting
        sprinting = _best_heading(env)

        assert abs(math.remainder(still - sprinting, math.tau)) < 1e-6, (
            "the aim point moved because the ship moved; bullets do not inherit ship velocity"
        )


class TestAnEnemyDirectlyAsternIsEscapable:
    """Symptom: an enemy chases from directly behind and the ship rotates one way forever without
    ever bringing it into the sights. A cosine potential is stationary at 180 degrees -- turning
    either way changes it only to second order, so there is almost nothing to learn from."""

    def test_turning_off_dead_astern_pays_first_order(self):
        env = quiet_arena()
        env.enemies = [Enemy(env.player.x - 300.0, env.player.y, health=1, speed=0.0)]

        env.player.heading = 0.0                      # enemy exactly behind
        astern = env._aim_potential()
        env.player.heading = math.radians(10.0)       # a single agent decision's worth of turn
        turned = env._aim_potential()

        gain = turned - astern
        # cos gives 1 - cos(10 deg) = 0.015. A potential with a real gradient here gives ~10x that.
        assert gain > 0.04, (
            f"only {gain:.4f} of potential for turning off dead astern: too flat to learn from"
        )


class TestRetreatingStopsPayingBeforeTheCorner:
    """Symptom: the agent parks in corners (0.9% -> 15.0% of steps after the rebalance). A
    potential that rises with distance-to-enemy all the way to the arena diagonal is maximised
    exactly in the corners, so corner-camping is not a failure to learn -- it is the reward."""

    def test_extra_distance_beyond_safe_range_is_not_rewarded(self):
        env = quiet_arena()
        env.spawners = []

        env.player.x, env.player.y = 480.0, 340.0     # mid-arena
        env.enemies = [Enemy(80.0, 80.0, health=1, speed=0.0)]
        middle = env._safety_potential()

        env.player.x, env.player.y = 940.0, 660.0     # jammed in the far corner
        corner = env._safety_potential()

        assert corner <= middle + 1e-9, (
            "retreating into the corner scores higher than standing in open space"
        )


def _best_heading(env: ArenaEnv) -> float:
    """The heading that maximises `_aim_potential`, found by scan -- i.e. where it says to point."""
    best, best_phi = 0.0, -math.inf
    for step in range(3600):
        env.player.heading = -math.pi + step * math.tau / 3600
        phi = env._aim_potential()
        if phi > best_phi:
            best, best_phi = env.player.heading, phi
    return best


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
