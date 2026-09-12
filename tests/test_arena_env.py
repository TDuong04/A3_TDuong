"""Tests for `arena.env` and `arena.legacy_api` — the Gym-style API (rubric row H).

The arena is deterministic once seeded, so almost every assertion here is on an exact value: a
reward equal to the sum of named constants, a position equal to what `entities.py` produces from
the same start state, a heading equal to the configured turn rate times three frames. "Something
moved" and "reward changed" would both survive most of the bugs this file exists to catch.

Reward *sources* are checked by monkeypatching the constant in `arena.env` and watching the
returned reward follow it. That is the only way to prove the env reads `arena/constants.py` rather
than a literal that happens to hold the same value today.
"""

from __future__ import annotations

import ast
import math
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from arena.constants import (
    ACTION_REPEAT,
    ARENA_HEIGHT,
    ARENA_WIDTH,
    CONTROL_STYLES,
    FIXED_DT,
    MAX_EPISODE_STEPS,
    OBS_DIM,
    REWARD_DAMAGE_TAKEN,
    REWARD_DEATH,
    REWARD_ENEMY_DESTROYED,
    REWARD_PER_STEP,
    REWARD_PHASE_ADVANCE,
    REWARD_SPAWNER_DESTROYED,
    DirectAction,
    RotationAction,
)
from arena.entities import Enemy, Player, Spawner, player_config
from arena.env import ArenaEnv, episode_config
from arena.legacy_api import LegacyGymAPI
from arena.observation import to_ship_local
from common.config import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_SOURCE = (REPO_ROOT / "arena" / "env.py").read_text(encoding="utf-8")
RENDER_SOURCE = (REPO_ROOT / "arena" / "render.py").read_text(encoding="utf-8")

ARENA_YAML = load_yaml("arena")
PLAYER_YAML = ARENA_YAML["player"]
PHASES_YAML = ARENA_YAML["phases"]
EPISODE_YAML = ARENA_YAML["episode"]

RENDERER_IS_STILL_A_STUB = "NOT YET IMPLEMENTED" in RENDER_SOURCE


# --- helpers -------------------------------------------------------------------------------------


def park_spawners(env: ArenaEnv) -> None:
    """Push the phase's spawners into a far corner and mute them.

    Tests that want to observe one specific interaction need the rest of the arena to hold still;
    they still need *a* live spawner, because destroying the last one advances the phase.
    """
    for index, spawner in enumerate(env.spawners):
        spawner.x = ARENA_WIDTH - 40.0
        spawner.y = 40.0 + index * 60.0
        spawner.spawn_timer = 1.0e6


def quiet_env(control_style: str = "rotation", seed: int = 1, **kwargs) -> ArenaEnv:
    env = ArenaEnv(control_style=control_style, seed=seed, **kwargs)
    park_spawners(env)
    # This file asserts reward as an exact sum of the frozen constants in `constants.py`. The
    # optional shaping terms (`ShapingConfig`) are config-driven and, unlike those, vary with the
    # ship's own aim and range every step -- exercised on their own in `TestAimShaping` and
    # `TestSafetyShaping` instead of smuggled into every other assertion here.
    env.shaping = replace(env.shaping, aim_strength=0.0, safety_strength=0.0)
    return env


def ahead_of(env: ArenaEnv, distance: float) -> tuple[float, float]:
    """A point `distance` pixels down the ship's nose — wherever the seed pointed it."""
    return (
        env.player.x + math.cos(env.player.heading) * distance,
        env.player.y + math.sin(env.player.heading) * distance,
    )


def place_enemy_ahead(env: ArenaEnv, distance: float) -> Enemy:
    x, y = ahead_of(env, distance)
    enemy = Enemy.from_phase(x, y, env.phase_settings)
    env.enemies.append(enemy)
    return enemy


def overlap_player_with_enemy(env: ArenaEnv) -> Enemy:
    enemy = Enemy.from_phase(env.player.x, env.player.y, env.phase_settings)
    env.enemies.append(enemy)
    return enemy


def clone_player(player: Player) -> Player:
    """A standalone player at the same state, for comparing against raw `entities.py` physics."""
    return Player(player.x, player.y, heading=player.heading)


def _aim_and_shoot(env: ArenaEnv) -> int:
    """A crude hand-written `direct`-style policy: face the nearest spawner, then shoot it.

    Style 2 snaps the heading to the direction it moves, so moving toward the target is also how
    it aims.
    """
    from arena.observation import nearest_alive

    target = nearest_alive(env.player, env.spawners)
    if target is None:
        return int(DirectAction.NOOP)
    dx, dy = target.x - env.player.x, target.y - env.player.y
    aimed = abs(math.atan2(dy, dx) - env.player.heading) < 0.15
    if aimed and math.hypot(dx, dy) < 260.0:
        return int(DirectAction.SHOOT)
    if abs(dx) > abs(dy):
        return int(DirectAction.RIGHT if dx > 0 else DirectAction.LEFT)
    return int(DirectAction.DOWN if dy > 0 else DirectAction.UP)


def rollout(env: ArenaEnv, actions, stop_on_done: bool = True):
    """Run a fixed action sequence and return the per-step (obs, reward, terminated, truncated)."""
    steps = []
    for action in actions:
        obs, reward, terminated, truncated, _info = env.step(action)
        steps.append((obs, reward, terminated, truncated))
        if stop_on_done and (terminated or truncated):
            break
    return steps


# --- module hygiene -------------------------------------------------------------------------------


def test_env_source_has_no_module_level_pygame_import():
    assert re.search(r"^\s*(import|from)\s+pygame", ENV_SOURCE, re.MULTILINE) is None


def test_env_imports_nothing_graphical_at_module_scope():
    """The only `arena.render` import must be inside `render()`, so a headless training worker
    never loads pygame at all."""
    tree = ast.parse(ENV_SOURCE)
    top_level_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.add((node.module or "").split(".")[0])
    assert top_level_imports.isdisjoint({"pygame", "time", "datetime"}), top_level_imports
    assert "render" not in top_level_imports, "arena.render must be imported lazily"

    # ... and it must genuinely be imported somewhere deeper, or `render()` could not work.
    nested_render_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "render" and node not in tree.body
    ]
    assert nested_render_imports, "render() should import arena.render inside the method"


def test_importing_the_env_does_not_pull_in_pygame():
    result = subprocess.run(
        [sys.executable, "-c", "import arena.env, sys; assert 'pygame' not in sys.modules"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_a_whole_episode_runs_with_pygame_banned_from_the_import_system():
    """Transitive proof: not merely "pygame is not imported yet" but "the simulation never needs
    it", which is what keeps training headless."""
    script = """
import sys

class BanPygame:
    def find_spec(self, name, path=None, target=None):
        if name == 'pygame' or name.startswith('pygame.'):
            raise ImportError('pygame is banned in simulation modules')
        return None

sys.meta_path.insert(0, BanPygame())
from arena.env import ArenaEnv

env = ArenaEnv(control_style='rotation', seed=3)
obs, info = env.reset(seed=3)
for step in range(300):
    obs, reward, terminated, truncated, info = env.step(step % int(env.action_space.n))
    if terminated or truncated:
        break
assert 'pygame' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_every_reward_constant_is_referenced_by_name():
    """No reward literal anywhere in the env: the constants file is the single source."""
    for name in (
        "REWARD_ENEMY_DESTROYED",
        "REWARD_SPAWNER_DESTROYED",
        "REWARD_PHASE_ADVANCE",
        "REWARD_DAMAGE_TAKEN",
        "REWARD_DEATH",
        "REWARD_PER_STEP",
    ):
        assert ENV_SOURCE.count(name) >= 2, name  # imported and used


def test_episode_tunables_come_from_the_config():
    config = episode_config()
    assert config.max_enemies == EPISODE_YAML["max_enemies"]
    assert config.min_player_clearance == EPISODE_YAML["min_player_clearance"]
    assert config.placement_attempts == EPISODE_YAML["placement_attempts"]


# --- one env, two control styles -----------------------------------------------------------------


class TestControlStyles:
    def test_one_class_serves_both_styles(self):
        assert set(CONTROL_STYLES) == {"rotation", "direct"}
        assert all(type(ArenaEnv(control_style=style)) is ArenaEnv for style in CONTROL_STYLES)

    def test_action_space_sizes(self):
        assert ArenaEnv(control_style="rotation").action_space.n == 5
        assert ArenaEnv(control_style="direct").action_space.n == 6

    def test_observation_space_is_the_same_box_for_both(self):
        rotation = ArenaEnv(control_style="rotation").observation_space
        direct = ArenaEnv(control_style="direct").observation_space
        assert rotation.shape == direct.shape == (OBS_DIM,)
        assert rotation.dtype == direct.dtype == np.float32
        assert float(rotation.low.min()) == -1.0 and float(rotation.high.max()) == 1.0

    def test_an_unknown_style_is_rejected_loudly(self):
        with pytest.raises(ValueError, match="control_style"):
            ArenaEnv(control_style="twin_stick")

    def test_out_of_range_actions_are_rejected(self):
        env = quiet_env("rotation")
        with pytest.raises(ValueError):
            env.step(5)  # valid in `direct`, out of range here
        with pytest.raises(ValueError):
            env.step(-1)

    def test_rotation_action_indices_do_what_the_brief_says(self):
        rate = player_config().rotation_speed_radians * FIXED_DT * ACTION_REPEAT

        env = quiet_env("rotation")
        start = env.player.heading
        env.step(RotationAction.NOOP)
        assert env.player.heading == pytest.approx(start, abs=1e-12)

        env = quiet_env("rotation")
        start = env.player.heading
        env.step(RotationAction.ROTATE_LEFT)
        assert env.player.heading == pytest.approx(start - rate, abs=1e-9)

        env = quiet_env("rotation")
        start = env.player.heading
        env.step(RotationAction.ROTATE_RIGHT)
        assert env.player.heading == pytest.approx(start + rate, abs=1e-9)

        env = quiet_env("rotation")
        reference = clone_player(env.player)
        for _ in range(ACTION_REPEAT):
            reference.thrust()
            reference.update()
        env.step(RotationAction.THRUST)
        assert (env.player.x, env.player.y) == pytest.approx((reference.x, reference.y), abs=1e-9)

        env = quiet_env("rotation")
        env.step(RotationAction.SHOOT)
        assert len(env.bullets) == 1

    def test_direct_action_indices_do_what_the_brief_says(self):
        speed_cap = PLAYER_YAML["speed"]

        env = quiet_env("direct")
        env.step(DirectAction.NOOP)
        assert (env.player.vx, env.player.vy) == (0.0, 0.0)
        assert len(env.bullets) == 0

        for action, (dx, dy) in (
            (DirectAction.UP, (0.0, -1.0)),
            (DirectAction.DOWN, (0.0, 1.0)),
            (DirectAction.LEFT, (-1.0, 0.0)),
            (DirectAction.RIGHT, (1.0, 0.0)),
        ):
            env = quiet_env("direct")
            reference = clone_player(env.player)
            for _ in range(ACTION_REPEAT):
                reference.move(dx, dy)
                reference.update()
            env.step(action)
            assert (env.player.x, env.player.y) == pytest.approx(
                (reference.x, reference.y), abs=1e-9
            ), action
            assert env.player.heading == pytest.approx(math.atan2(dy, dx), abs=1e-9), action
            assert math.hypot(env.player.vx, env.player.vy) <= speed_cap + 1e-9

        env = quiet_env("direct")
        env.step(DirectAction.SHOOT)
        assert len(env.bullets) == 1

    def test_index_4_means_different_things_in_the_two_styles(self):
        """The single most expensive index bug available: `SHOOT` is 4 for rotation and 5 for
        direct, and 4 in direct is `RIGHT`."""
        rotation = quiet_env("rotation")
        rotation.step(4)
        assert len(rotation.bullets) == 1

        direct = quiet_env("direct")
        direct.step(4)
        assert len(direct.bullets) == 0
        assert direct.player.vx > 0.0
        assert direct.player.heading == pytest.approx(0.0, abs=1e-9)

    def test_action_name_reports_the_last_action_for_the_hud(self):
        env = quiet_env("direct")
        env.step(DirectAction.LEFT)
        assert env.action_name == "LEFT"
        env.step(DirectAction.SHOOT)
        assert env.action_name == "SHOOT"


# --- the action repeat ----------------------------------------------------------------------------


class TestActionRepeat:
    def test_a_step_advances_exactly_action_repeat_physics_frames(self):
        env = quiet_env("rotation")
        three_frames = clone_player(env.player)
        two_frames = clone_player(env.player)
        for _ in range(ACTION_REPEAT):
            three_frames.thrust()
            three_frames.update()
        for _ in range(ACTION_REPEAT - 1):
            two_frames.thrust()
            two_frames.update()

        env.step(RotationAction.THRUST)
        assert env.player.x == pytest.approx(three_frames.x, abs=1e-9)
        assert env.player.x != pytest.approx(two_frames.x, abs=1e-6)

    def test_the_bullet_flies_three_frames_per_step(self):
        env = quiet_env("rotation")
        env.step(RotationAction.SHOOT)
        bullet = env.bullets[0]
        start = (bullet.x, bullet.y)
        env.step(RotationAction.NOOP)
        travelled = math.hypot(bullet.x - start[0], bullet.y - start[1])
        assert travelled == pytest.approx(
            PLAYER_YAML["bullet_speed"] * FIXED_DT * ACTION_REPEAT, abs=1e-6
        )

    def test_the_step_penalty_is_paid_once_per_agent_step_not_once_per_frame(self):
        env = quiet_env("rotation")
        _obs, reward, _terminated, _truncated, _info = env.step(RotationAction.NOOP)
        assert reward == pytest.approx(REWARD_PER_STEP)
        assert reward != pytest.approx(REWARD_PER_STEP * ACTION_REPEAT)

    def test_the_cooldown_survives_the_repeat(self):
        """Three frames of a held SHOOT must not become three bullets, and the cooldown must
        still unblock on schedule once it genuinely elapses.

        The schedule is derived from config rather than written in: one agent step drains
        `ACTION_REPEAT * FIXED_DT` of cooldown, so a held trigger is blocked for
        `ceil(shoot_cooldown / that)` steps and fires again on the next one. Hard-coding the
        answer for one `shoot_cooldown` is how this test broke when the value was retuned -- the
        invariant is the ratio, not the number.
        """
        env = quiet_env("rotation")
        drained_per_step = ACTION_REPEAT * FIXED_DT
        blocked_steps = math.ceil(env.player.config.shoot_cooldown / drained_per_step)

        env.step(RotationAction.SHOOT)
        assert env.bullets_fired == 1

        for _ in range(blocked_steps - 1):
            env.step(RotationAction.SHOOT)
        assert env.bullets_fired == 1, "a held trigger beat the cooldown"

        env.step(RotationAction.SHOOT)
        assert env.bullets_fired == 2, "the cooldown never cleared"


# --- rewards --------------------------------------------------------------------------------------


class TestRewards:
    def test_step_penalty_reads_the_constant(self, monkeypatch):
        import arena.env as env_module

        monkeypatch.setattr(env_module, "REWARD_PER_STEP", -0.777)
        env = quiet_env("rotation")
        _obs, reward, *_ = env.step(RotationAction.NOOP)
        assert reward == pytest.approx(-0.777)

    def test_destroying_an_enemy_pays_the_enemy_reward(self):
        env = quiet_env("rotation")
        enemy = place_enemy_ahead(env, 110.0)
        total = 0.0
        steps = 0
        _obs, reward, *_ = env.step(RotationAction.SHOOT)
        total += reward
        steps += 1
        while enemy.alive and steps < 10:
            _obs, reward, *_ = env.step(RotationAction.NOOP)
            total += reward
            steps += 1

        assert not enemy.alive
        assert env.enemies_killed == 1
        assert total == pytest.approx(steps * REWARD_PER_STEP + REWARD_ENEMY_DESTROYED)

    def test_the_enemy_reward_reads_the_constant(self, monkeypatch):
        import arena.env as env_module

        monkeypatch.setattr(env_module, "REWARD_ENEMY_DESTROYED", 42.0)
        env = quiet_env("rotation")
        enemy = place_enemy_ahead(env, 110.0)
        total, steps = 0.0, 0
        _obs, reward, *_ = env.step(RotationAction.SHOOT)
        total += reward
        steps += 1
        while enemy.alive and steps < 10:
            _obs, reward, *_ = env.step(RotationAction.NOOP)
            total += reward
            steps += 1
        assert total == pytest.approx(steps * REWARD_PER_STEP + 42.0)

    def test_taking_damage_is_paid_once_per_invulnerability_window(self):
        env = quiet_env("rotation")
        overlap_player_with_enemy(env)
        health_before = env.player.health

        _obs, first, *_ = env.step(RotationAction.NOOP)
        assert env.damage_taken == 1
        assert env.player.health == health_before - 1
        assert first == pytest.approx(REWARD_PER_STEP + REWARD_DAMAGE_TAKEN)

        # Still overlapping, still invulnerable: three more frames of contact cost nothing.
        _obs, second, *_ = env.step(RotationAction.NOOP)
        assert env.damage_taken == 1
        assert second == pytest.approx(REWARD_PER_STEP)

    def test_the_damage_and_death_rewards_read_their_constants(self, monkeypatch):
        import arena.env as env_module

        monkeypatch.setattr(env_module, "REWARD_DAMAGE_TAKEN", -3.0)
        monkeypatch.setattr(env_module, "REWARD_DEATH", -60.0)
        env = quiet_env("rotation")
        env.player.health = 1
        overlap_player_with_enemy(env)

        _obs, reward, terminated, truncated, _info = env.step(RotationAction.NOOP)
        assert terminated and not truncated
        assert reward == pytest.approx(REWARD_PER_STEP - 3.0 - 60.0)

    def test_death_pays_the_death_penalty_on_top_of_the_damage_penalty(self):
        env = quiet_env("rotation")
        env.player.health = 1
        overlap_player_with_enemy(env)
        _obs, reward, terminated, _truncated, info = env.step(RotationAction.NOOP)
        assert terminated
        assert reward == pytest.approx(REWARD_PER_STEP + REWARD_DAMAGE_TAKEN + REWARD_DEATH)
        assert info["health"] == 0

    def test_destroying_the_last_spawner_pays_the_spawner_and_phase_rewards(self):
        env = quiet_env("rotation")
        x, y = ahead_of(env, 150.0)
        env.spawners = [Spawner(x, y, env.phase_settings)]
        env.spawners[0].spawn_timer = 1.0e6

        total, steps = 0.0, 0
        while env.phase == 1 and steps < 60:
            _obs, reward, *_ = env.step(RotationAction.SHOOT)
            total += reward
            steps += 1

        assert env.spawners_destroyed == 1
        assert env.phase == 2
        assert total == pytest.approx(
            steps * REWARD_PER_STEP + REWARD_SPAWNER_DESTROYED + REWARD_PHASE_ADVANCE
        )

    def test_the_spawner_and_phase_rewards_read_their_constants(self, monkeypatch):
        import arena.env as env_module

        monkeypatch.setattr(env_module, "REWARD_SPAWNER_DESTROYED", 7.5)
        monkeypatch.setattr(env_module, "REWARD_PHASE_ADVANCE", 21.0)
        env = quiet_env("rotation")
        x, y = ahead_of(env, 150.0)
        env.spawners = [Spawner(x, y, env.phase_settings)]
        env.spawners[0].spawn_timer = 1.0e6

        total, steps = 0.0, 0
        while env.phase == 1 and steps < 60:
            _obs, reward, *_ = env.step(RotationAction.SHOOT)
            total += reward
            steps += 1
        assert total == pytest.approx(steps * REWARD_PER_STEP + 7.5 + 21.0)


class TestAimShaping:
    """`ShapingConfig`'s potential-based aim term: `F = gamma * phi(s') - phi(s)`.

    The frozen-constant reward tests above all run with `aim_strength` forced to 0.0 (see
    `quiet_env`); these turn it on deliberately and check the shape of the term itself rather than
    any particular trained behaviour.
    """

    def test_config_loads_a_valid_strength(self):
        """Not a claim about what the value should be -- that's `config/arena.yaml`'s call,
        justified there and in the report -- just that shaping_config() reads a real, finite
        number rather than silently defaulting to something `from_yaml` never actually returned."""
        from arena.env import shaping_config

        assert shaping_config().aim_strength >= 0.0

    def test_it_tracks_the_spawners_once_the_enemies_are_gone(self):
        """With the arena cleared there is still something to shoot. This term used to go flat
        the moment the last enemy died, which left nothing pointing the ship at the spawners it
        had to destroy to advance -- measured, the shipped model idled through 40.7% of every
        episode that way."""
        env = quiet_env("rotation")
        env.shaping = replace(env.shaping, aim_strength=0.05)
        assert not env.enemies and env.spawners

        _obs, reward, *_ = env.step(RotationAction.ROTATE_LEFT)

        assert reward != pytest.approx(REWARD_PER_STEP), "still flat with spawners left to kill"

    def test_is_zero_with_nothing_left_alive_at_all(self):
        """Asserted on the potential rather than through `step()`: an arena with no spawners left
        advances the phase on the same step and repopulates, so the state cannot be observed from
        a reward."""
        env = quiet_env("rotation")
        env.enemies = []
        env.spawners = []

        assert env._aim_potential() == 0.0

    def test_rewards_turning_to_face_an_enemy_and_penalises_turning_away(self):
        """Place an enemy 90 degrees off the nose. Turning toward it must earn strictly more
        shaping reward over the same number of steps than turning away, with everything else
        about the two runs -- seed, enemy, spawners -- identical."""

        def run(action) -> float:
            env = quiet_env("rotation")
            env.shaping = replace(env.shaping, aim_strength=0.05)
            # +90 degrees off the nose: heading increases toward it (`_wrap_angle(heading + spin *
            # rate * dt)`, spin=+1 for ROTATE_RIGHT), so ROTATE_RIGHT is "toward" here and
            # ROTATE_LEFT is "away" -- not a universal convention, just this placement's.
            perp = env.player.heading + math.pi / 2.0
            ex = env.player.x + math.cos(perp) * 300.0
            ey = env.player.y + math.sin(perp) * 300.0
            env.enemies = [Enemy.from_phase(ex, ey, env.phase_settings)]
            env.enemies[0].speed = 0.0  # hold still so only the ship's own turning matters
            total_shaping = 0.0
            for _ in range(5):
                env.step(action)
                total_shaping += dict(env.latest_reward.contributions)["aim_shaping"]
            return total_shaping

        toward = run(RotationAction.ROTATE_RIGHT)
        away = run(RotationAction.ROTATE_LEFT)
        assert toward > away

    def test_matches_the_potential_based_formula_for_one_step(self):
        """Direct correctness check: the recorded contribution for one step must equal
        `strength * (gamma * phi(s') - phi(s))`, with phi computed independently here from the
        player's heading and the enemy's position rather than by calling `_aim_potential`."""

        def phi(env: ArenaEnv) -> float:
            enemy = env.enemies[0]
            dx, dy = enemy.x - env.player.x, enemy.y - env.player.y
            local_x, local_y = to_ship_local(env.player.heading, dx, dy)
            return 1.0 - abs(math.atan2(local_y, local_x)) / math.pi

        env = quiet_env("rotation")
        env.shaping = replace(env.shaping, aim_strength=0.05)
        perp = env.player.heading + math.pi / 2.0
        ex = env.player.x + math.cos(perp) * 300.0
        ey = env.player.y + math.sin(perp) * 300.0
        env.enemies = [Enemy.from_phase(ex, ey, env.phase_settings)]
        env.enemies[0].speed = 0.0

        phi_before = phi(env)
        env.step(RotationAction.ROTATE_LEFT)
        phi_after = phi(env)

        expected = env.shaping.aim_strength * (env.shaping.gamma * phi_after - phi_before)
        actual = dict(env.latest_reward.contributions)["aim_shaping"]
        assert actual == pytest.approx(expected, abs=1e-9)


class TestSafetyShaping:
    """`ShapingConfig`'s potential-based range term: same nearest enemy as `TestAimShaping`, but on
    distance instead of angle, and deliberately independent of it -- see `ShapingConfig`'s
    docstring for why the two are kept separate rather than combined into one term."""

    def test_is_a_small_constant_with_no_enemy_alive_not_zero(self):
        """Unlike `_aim_potential` (phi=0 with no enemy, so a still-empty arena contributes
        exactly nothing), `_safety_potential` reads phi=1.0 with no enemy -- "as safe as maximally
        far away". A *constant* nonzero potential still isn't free under discounting: it pays
        `strength * (gamma - 1) * phi` every step, a small structural cost of gamma < 1, not a
        bug. This checks the env matches that formula exactly, not that the term vanishes."""
        env = quiet_env("rotation")
        env.shaping = replace(env.shaping, safety_strength=0.05)
        assert not env.enemies
        _obs, reward, *_ = env.step(RotationAction.NOOP)
        expected_shaping = env.shaping.safety_strength * (env.shaping.gamma - 1.0) * 1.0
        assert reward == pytest.approx(REWARD_PER_STEP + expected_shaping)

    def test_rewards_increasing_distance_and_penalises_closing_in(self):
        """An enemy dead ahead. Thrusting forward (closing in) must earn strictly less shaping
        reward than thrusting the opposite way (opening the range), all else identical."""

        def run(rotate_first: bool) -> float:
            env = quiet_env("rotation")
            env.shaping = replace(env.shaping, safety_strength=0.05)
            ex, ey = ahead_of(env, 300.0)
            env.enemies = [Enemy.from_phase(ex, ey, env.phase_settings)]
            env.enemies[0].speed = 0.0
            if rotate_first:
                # Face the opposite way first so "thrust" opens the range instead of closing it;
                # a full about-turn well within the 5 steps this test then spends thrusting.
                for _ in range(20):
                    env.step(RotationAction.ROTATE_LEFT)
            total_shaping = 0.0
            for _ in range(5):
                env.step(RotationAction.THRUST)
                total_shaping += dict(env.latest_reward.contributions)["safety_shaping"]
            return total_shaping

        opening_range = run(rotate_first=True)
        closing_in = run(rotate_first=False)
        assert opening_range > closing_in

    def test_matches_the_potential_based_formula_for_one_step(self):
        """Direct correctness check: the recorded contribution for one step must equal
        `strength * (gamma * phi(s') - phi(s))`, with phi computed independently here from the
        player's and enemy's positions rather than by calling `_safety_potential`."""

        def phi(env: ArenaEnv) -> float:
            enemy = env.enemies[0]
            distance = math.hypot(enemy.x - env.player.x, enemy.y - env.player.y)
            return min(1.0, distance / env.shaping.safe_distance)

        env = quiet_env("rotation")
        env.shaping = replace(env.shaping, safety_strength=0.05)
        ex, ey = ahead_of(env, 300.0)
        env.enemies = [Enemy.from_phase(ex, ey, env.phase_settings)]
        env.enemies[0].speed = 0.0

        phi_before = phi(env)
        env.step(RotationAction.THRUST)
        phi_after = phi(env)

        expected = env.shaping.safety_strength * (env.shaping.gamma * phi_after - phi_before)
        actual = dict(env.latest_reward.contributions)["safety_shaping"]
        assert actual == pytest.approx(expected, abs=1e-9)


# --- the phase system -----------------------------------------------------------------------------


class TestPhases:
    def test_an_episode_starts_at_phase_one_with_its_configured_spawners(self):
        env = ArenaEnv(control_style="direct", seed=5)
        assert env.phase == 1
        assert len(env.spawners) == PHASES_YAML[0]["spawners"]
        assert all(s.max_health == PHASES_YAML[0]["spawner_health"] for s in env.spawners)

    def test_clearing_the_spawners_sets_up_the_next_phase_from_the_config(self):
        env = quiet_env("direct")
        for spawner in env.spawners:
            spawner.kill()
        env.step(DirectAction.NOOP)

        assert env.phase == 2
        assert len(env.spawners) == PHASES_YAML[1]["spawners"]
        assert all(s.max_health == PHASES_YAML[1]["spawner_health"] for s in env.spawners)
        assert env.phase_settings.enemy_speed == PHASES_YAML[1]["enemy_speed"]

    def test_the_difficulty_ladder_stops_climbing_after_the_last_phase(self):
        env = quiet_env("direct")
        for _ in range(len(PHASES_YAML) + 2):
            for spawner in env.spawners:
                spawner.kill()
            env.step(DirectAction.NOOP)
        assert env.phase == len(PHASES_YAML) + 3
        assert len(env.spawners) == PHASES_YAML[-1]["spawners"]

    def test_enemies_already_loose_survive_the_phase_transition(self):
        """A free wipe would make the transition the safest moment in the episode."""
        env = quiet_env("direct")
        place_enemy_ahead(env, 300.0)
        for spawner in env.spawners:
            spawner.kill()
        env.step(DirectAction.NOOP)
        assert env.phase == 2
        assert len(env.enemies) == 1

    def test_spawners_are_placed_clear_of_the_player(self):
        clearance = EPISODE_YAML["min_player_clearance"]
        for seed in range(30):
            env = ArenaEnv(control_style="direct", seed=seed)
            for spawner in env.spawners:
                assert env.player.distance_to(spawner) >= clearance, seed
                assert EPISODE_YAML["arena_margin"] <= spawner.x <= ARENA_WIDTH
                assert EPISODE_YAML["arena_margin"] <= spawner.y <= ARENA_HEIGHT

    def test_spawners_emit_enemies_that_seek_the_player(self):
        env = ArenaEnv(control_style="direct", seed=2)
        interval_steps = int(PHASES_YAML[0]["spawn_interval"] / (FIXED_DT * ACTION_REPEAT)) + 2
        for _ in range(interval_steps):
            env.step(DirectAction.NOOP)
        assert len(env.enemies) >= 1

        enemy = env.enemies[0]
        before = env.player.distance_to(enemy)
        for _ in range(10):
            env.step(DirectAction.NOOP)
        assert env.player.distance_to(enemy) < before

    def test_a_scripted_policy_can_reach_phase_two_through_the_action_interface(self):
        """End-to-end reachability, using nothing but `step()`.

        Every other phase test arranges the world by hand; this one proves an agent can actually
        get there, which is what the video's "at least one phase progression" depends on. The
        script aims at the nearest spawner and fires. It dies on some seeds — that is the point of
        training an agent — so the bar is a majority of seeds, not all of them.
        """
        cleared = 0
        for seed in range(6):
            env = ArenaEnv(control_style="direct", seed=seed)
            env.reset(seed=seed)
            for _ in range(400):
                _obs, _reward, terminated, truncated, info = env.step(_aim_and_shoot(env))
                if info["phase"] > 1:
                    cleared += 1
                    break
                if terminated or truncated:
                    break
        assert cleared >= 3, f"only {cleared}/6 seeds cleared phase 1"

    def test_the_live_enemy_cap_bounds_the_arena(self):
        env = quiet_env("direct")
        env.episode_config = replace(env.episode_config, max_enemies=3)
        for spawner in env.spawners:
            spawner.spawn_interval = FIXED_DT
            spawner.spawn_timer = FIXED_DT
        for _ in range(30):
            env.step(DirectAction.NOOP)
        assert len(env.enemies) <= 3


# --- termination and truncation -------------------------------------------------------------------


class TestTerminationAndTruncation:
    def test_the_default_step_cap_is_the_constant(self):
        assert ArenaEnv(control_style="direct").max_episode_steps == MAX_EPISODE_STEPS

    def test_running_out_of_clock_truncates_and_does_not_terminate(self):
        env = quiet_env("direct", max_episode_steps=3)
        steps = rollout(env, [DirectAction.NOOP] * 5)
        assert len(steps) == 3
        _obs, _reward, terminated, truncated = steps[-1]
        assert truncated is True
        assert terminated is False
        assert env.player.alive

    def test_earlier_steps_are_neither_terminated_nor_truncated(self):
        env = quiet_env("direct", max_episode_steps=3)
        steps = rollout(env, [DirectAction.NOOP] * 5)
        for _obs, _reward, terminated, truncated in steps[:-1]:
            assert not terminated and not truncated

    def test_death_terminates_and_does_not_truncate(self):
        env = quiet_env("rotation")
        env.player.health = 1
        overlap_player_with_enemy(env)
        _obs, _reward, terminated, truncated, _info = env.step(RotationAction.NOOP)
        assert terminated is True
        assert truncated is False
        assert not env.player.alive

    def test_dying_on_the_last_allowed_step_is_death_not_a_time_limit(self):
        """The adversarial case: both conditions true at once. Conflating them teaches the agent
        that running out of time is fatal."""
        env = quiet_env("rotation", max_episode_steps=1)
        env.player.health = 1
        overlap_player_with_enemy(env)
        _obs, reward, terminated, truncated, _info = env.step(RotationAction.NOOP)
        assert terminated is True
        assert truncated is False
        assert reward == pytest.approx(REWARD_PER_STEP + REWARD_DAMAGE_TAKEN + REWARD_DEATH)

    def test_reset_clears_the_previous_episode(self):
        env = quiet_env("rotation", max_episode_steps=2)
        env.player.health = 1
        overlap_player_with_enemy(env)
        env.step(RotationAction.NOOP)
        assert not env.player.alive

        obs, info = env.reset(seed=9)
        assert env.player.alive
        assert env.steps == 0
        assert env.phase == 1
        assert env.enemies == [] and env.bullets == []
        assert info["damage_taken"] == 0 and info["enemies_killed"] == 0
        assert env.observation_space.contains(obs)


# --- info -----------------------------------------------------------------------------------------


class TestInfo:
    REQUIRED = ("phase", "spawners_destroyed", "enemies_killed", "damage_taken")

    def test_reset_info_carries_the_behavioural_metrics(self):
        _obs, info = ArenaEnv(control_style="direct", seed=4).reset(seed=4)
        for key in self.REQUIRED:
            assert key in info, key
        assert info["phase"] == 1
        assert (info["spawners_destroyed"], info["enemies_killed"], info["damage_taken"]) == (
            0,
            0,
            0,
        )

    def test_step_info_carries_the_behavioural_metrics(self):
        env = quiet_env("direct")
        _obs, _reward, _terminated, _truncated, info = env.step(DirectAction.NOOP)
        for key in self.REQUIRED:
            assert key in info, key

    def test_the_counters_are_cumulative_and_never_decrease(self):
        env = ArenaEnv(control_style="rotation", seed=7)
        previous = {key: 0 for key in self.REQUIRED}
        for step in range(400):
            _obs, _reward, terminated, truncated, info = env.step(step % 5)
            for key in self.REQUIRED:
                assert info[key] >= previous[key], key
                previous[key] = info[key]
            if terminated or truncated:
                break

    def test_info_terminated_matches_the_returned_flag(self):
        """`behaviour/survival_rate` is this key averaged, so it has to agree with the step tuple.

        Checked on both endings: a step-cap truncation is not a death, and reporting it as one
        would show a policy learning to survive when it had only learned to run out the clock.
        """
        env = quiet_env("rotation", max_episode_steps=2)
        _obs, _reward, terminated, truncated, info = env.step(RotationAction.NOOP)
        assert info["terminated"] is terminated is False

        _obs, _reward, terminated, truncated, info = env.step(RotationAction.NOOP)
        assert truncated is True
        assert info["terminated"] is terminated is False

        env = quiet_env("rotation")
        env.player.health = 1
        overlap_player_with_enemy(env)
        _obs, _reward, terminated, _truncated, info = env.step(RotationAction.NOOP)
        assert info["terminated"] is terminated is True

    def test_the_counters_track_what_actually_happened(self):
        env = quiet_env("rotation")
        enemy = place_enemy_ahead(env, 110.0)
        env.step(RotationAction.SHOOT)
        for _ in range(9):
            if not enemy.alive:
                break
            env.step(RotationAction.NOOP)
        _obs, _reward, _terminated, _truncated, info = env.step(RotationAction.NOOP)
        assert info["enemies_killed"] == 1
        assert info["damage_taken"] == 0
        assert info["spawners_destroyed"] == 0


# --- the observation the env hands out ------------------------------------------------------------


class TestEnvObservation:
    def test_observations_always_lie_inside_the_declared_box(self):
        env = ArenaEnv(control_style="rotation", seed=11)
        obs, _info = env.reset(seed=11)
        assert env.observation_space.contains(obs)
        for step in range(400):
            obs, _reward, terminated, truncated, _info = env.step(step % 5)
            assert obs.dtype == np.float32
            assert np.all(np.isfinite(obs))
            assert env.observation_space.contains(obs), step
            if terminated or truncated:
                break

    def test_both_styles_observe_the_same_world_identically(self):
        """Same seed, same world; only the action space differs. If these ever diverge, the two
        trained agents stop being comparable and report row R6 has nothing to compare."""
        rotation, _ = ArenaEnv(control_style="rotation", seed=13).reset(seed=13)
        direct, _ = ArenaEnv(control_style="direct", seed=13).reset(seed=13)
        assert np.array_equal(rotation, direct)

    def test_a_shared_world_state_produces_one_observation(self):
        """Stronger than the seeded version: one arena, two envs looking at it."""
        source = ArenaEnv(control_style="direct", seed=17)
        for _ in range(40):
            source.step(DirectAction.RIGHT)

        viewer = ArenaEnv(control_style="rotation", seed=99)
        viewer.player = source.player
        viewer.enemies = source.enemies
        viewer.spawners = source.spawners
        viewer.phase = source.phase
        assert np.array_equal(viewer.observation(), source.observation())

    def test_the_observation_reflects_the_arena_rather_than_the_clock(self):
        env = quiet_env("rotation")
        empty = env.observation()
        place_enemy_ahead(env, 200.0)
        occupied = env.observation()
        assert not np.array_equal(empty, occupied)


# --- reproducibility ------------------------------------------------------------------------------


class TestReproducibility:
    ACTIONS = [0, 1, 1, 2, 4, 1, 3, 3, 1, 4, 0, 1, 2, 4, 1]

    def test_the_same_seed_and_actions_give_an_identical_trajectory(self):
        first = ArenaEnv(control_style="rotation")
        second = ArenaEnv(control_style="rotation")
        obs_a, _ = first.reset(seed=101)
        obs_b, _ = second.reset(seed=101)
        assert np.array_equal(obs_a, obs_b)

        for action in self.ACTIONS * 6:
            step_a = first.step(action)
            step_b = second.step(action)
            assert np.array_equal(step_a[0], step_b[0])
            assert step_a[1:] == step_b[1:]

    def test_resetting_the_same_env_with_the_same_seed_repeats_the_episode(self):
        env = ArenaEnv(control_style="direct")
        obs_a, _ = env.reset(seed=55)
        for action in self.ACTIONS:
            env.step(action)
        obs_b, _ = env.reset(seed=55)
        assert np.array_equal(obs_a, obs_b)

    def test_different_seeds_give_different_episodes(self):
        """Guards the opposite failure: a seed that is quietly ignored would pass every test
        above."""
        obs_a, _ = ArenaEnv(control_style="rotation").reset(seed=1)
        obs_b, _ = ArenaEnv(control_style="rotation").reset(seed=2)
        assert not np.array_equal(obs_a, obs_b)

    def test_the_starting_heading_is_not_always_the_same(self):
        """A fixed start heading would also make the ship-local frame the identity at every reset,
        which is how a rotation bug survives to training day."""
        headings = {round(ArenaEnv(seed=seed).player.heading, 6) for seed in range(10)}
        assert len(headings) > 1


# --- Stable Baselines3 compatibility --------------------------------------------------------------


@pytest.mark.parametrize("control_style", CONTROL_STYLES)
def test_sb3_env_checker_passes(control_style):
    import warnings

    from stable_baselines3.common.env_checker import check_env

    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        check_env(ArenaEnv(control_style=control_style, seed=0), warn=True)
    # SB3 reports observation-scaling and API problems as warnings rather than failures, so a
    # clean run means clean, not merely "did not crash".
    assert [str(warning.message) for warning in raised] == []


@pytest.mark.parametrize("control_style", CONTROL_STYLES)
def test_the_env_is_a_gymnasium_env(control_style):
    import gymnasium as gym

    assert isinstance(ArenaEnv(control_style=control_style), gym.Env)


# --- the brief's own API surface ------------------------------------------------------------------


class TestLegacyGymAPI:
    """Rubric H1 quotes `reset()`, `step(action) -> (obs, reward, done, info)` and `render()`."""

    def make(self, control_style: str = "direct", **kwargs) -> LegacyGymAPI:
        return LegacyGymAPI(ArenaEnv(control_style=control_style, seed=0, **kwargs))

    def test_reset_takes_no_arguments_and_returns_a_bare_observation(self):
        import inspect

        api = self.make()
        parameters = list(inspect.signature(api.reset).parameters)
        assert parameters == []

        obs = api.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (OBS_DIM,)
        assert not isinstance(obs, tuple)

    def test_step_returns_exactly_four_values(self):
        api = self.make()
        api.reset()
        result = api.step(0)
        assert isinstance(result, tuple)
        assert len(result) == 4

        obs, reward, done, info = result
        assert isinstance(obs, np.ndarray) and obs.shape == (OBS_DIM,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_done_is_the_disjunction_but_the_cause_survives_in_info(self):
        api = self.make(control_style="rotation", max_episode_steps=2)
        api.reset()
        _obs, _reward, done, info = api.step(RotationAction.NOOP)
        assert done is False

        _obs, _reward, done, info = api.step(RotationAction.NOOP)
        assert done is True
        assert info["truncated"] is True
        assert info["terminated"] is False

    def test_done_on_death_reports_termination(self):
        api = self.make(control_style="rotation")
        api.reset()
        park_spawners(api.env)
        api.env.player.health = 1
        overlap_player_with_enemy(api.env)
        _obs, _reward, done, info = api.step(RotationAction.NOOP)
        assert done is True
        assert info["terminated"] is True
        assert info["truncated"] is False

    def test_info_still_carries_the_behavioural_metrics(self):
        api = self.make()
        api.reset()
        _obs, _reward, _done, info = api.step(0)
        for key in ("phase", "spawners_destroyed", "enemies_killed", "damage_taken"):
            assert key in info, key

    def test_the_spaces_pass_through(self):
        api = self.make(control_style="rotation")
        assert api.action_space.n == 5
        assert api.observation_space.shape == (OBS_DIM,)

    def test_render_is_delegated_to_the_env(self):
        calls = []
        api = LegacyGymAPI(_FakeEnv(calls))
        api.render()
        assert calls == ["render"]


class _FakeEnv:
    """Minimal stand-in so the adapter's delegation is tested without a display."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def render(self):
        self.calls.append("render")
        return "frame"


# --- rendering stays outside the simulation -------------------------------------------------------


class _FakeRenderer:
    instances: list[_FakeRenderer] = []

    def __init__(self) -> None:
        self.drawn: list[ArenaEnv] = []
        self.closed = False
        _FakeRenderer.instances.append(self)

    def draw(self, env):
        self.drawn.append(env)
        return "frame"

    def close(self):
        self.closed = True


class TestRendering:
    def test_render_without_a_render_mode_fails_loudly(self):
        env = ArenaEnv(control_style="direct", seed=0)
        with pytest.raises(RuntimeError, match="render_mode"):
            env.render()

    def test_an_unknown_render_mode_is_rejected(self):
        with pytest.raises(ValueError, match="render_mode"):
            ArenaEnv(control_style="direct", render_mode="rgb_array")

    def test_render_builds_the_renderer_once_and_hands_it_the_env(self, monkeypatch):
        import arena.render

        _FakeRenderer.instances.clear()
        monkeypatch.setattr(arena.render, "ArenaRenderer", _FakeRenderer)

        env = ArenaEnv(control_style="direct", seed=0, render_mode="human")
        env.render()
        env.render()

        assert len(_FakeRenderer.instances) == 1
        renderer = _FakeRenderer.instances[0]
        assert renderer.drawn == [env, env]

        env.close()
        assert renderer.closed

    def test_stepping_never_touches_the_renderer(self, monkeypatch):
        """Rendering inside `step()` multiplies training time by an order of magnitude."""
        import arena.render

        _FakeRenderer.instances.clear()
        monkeypatch.setattr(arena.render, "ArenaRenderer", _FakeRenderer)

        env = ArenaEnv(control_style="direct", seed=0, render_mode="human")
        for _ in range(20):
            env.step(0)
        assert _FakeRenderer.instances == []

    @pytest.mark.skipif(
        RENDERER_IS_STILL_A_STUB,
        reason="arena/render.py is still the A3-009 stub; this test unskips itself when it lands",
    )
    def test_render_draws_a_frame_with_the_real_renderer(self):
        """Run in a subprocess under the dummy video driver so no window and no pygame import
        leaks into the rest of the suite."""
        script = """
from arena.env import ArenaEnv

env = ArenaEnv(control_style='rotation', seed=0, render_mode='human')
env.reset(seed=0)
for _ in range(3):
    env.step(1)
    env.render()
env.close()
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env={**dict(__import__("os").environ), "SDL_VIDEODRIVER": "dummy"},
        )
        assert result.returncode == 0, result.stderr
