"""Behaviour tests for `arena.env` and `arena.observation` — rubric rows H and J1.

The acceptance criteria of A3-010 are the spine of this file: exact action indices per control
style, a 21-feature float32 observation inside its `Box`, relative positions in the ship-local
frame, cleared slots for absent entities, `terminated` kept distinct from `truncated`, the four
behavioural `info` keys, and `check_env` passing for both styles.

Reward assertions are written against the constants rather than against literals, so a test failing
here means the env changed — not that a magic number in the test went stale.
"""

from __future__ import annotations

import ast
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from arena.constants import (
    ACTION_REPEAT,
    ARENA_HEIGHT,
    ARENA_WIDTH,
    FIXED_DT,
    MAX_EPISODE_STEPS,
    OBS_DIM,
    REWARD_DAMAGE_TAKEN,
    REWARD_DEATH,
    REWARD_ENEMY_DESTROYED,
    REWARD_PER_STEP,
    REWARD_PHASE_ADVANCE,
    DirectAction,
    RotationAction,
)
from arena.entities import Enemy, Player
from arena.env import ArenaEnv
from arena.legacy_api import LegacyGymAPI
from arena.observation import ARENA_DIAGONAL, build_observation, describe

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_SOURCE = (REPO_ROOT / "arena" / "env.py").read_text(encoding="utf-8")

CENTRE_X, CENTRE_Y = ARENA_WIDTH / 2, ARENA_HEIGHT / 2


@pytest.fixture
def env() -> ArenaEnv:
    world = ArenaEnv("direct")
    world.reset(seed=0)
    return world


# --- module hygiene: the simulation path never imports pygame ----------------------------------


def test_env_source_has_no_pygame_import_at_module_scope():
    """The docstring names pygame deliberately; an *import* of it is what must not exist."""
    module_level = ENV_SOURCE.split("def render", 1)[0]
    assert re.search(r"^\s*(import|from)\s+pygame", module_level, re.MULTILINE) is None


def test_env_imports_no_clock_or_display_module():
    imported: set[str] = set()
    for node in ast.walk(ast.parse(ENV_SOURCE)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"time", "datetime", "pygame"}), imported


def test_stepping_a_headless_env_never_pulls_in_pygame():
    """`render_mode=None` must import and run with no display anywhere in the process."""
    script = """
import sys
from arena.env import ArenaEnv
env = ArenaEnv('rotation')
env.reset(seed=0)
for _ in range(50):
    env.step(env.action_space.sample())
assert 'pygame' not in sys.modules, sorted(m for m in sys.modules if 'pygame' in m)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# --- spaces: the brief's exact action indices ---------------------------------------------------


def test_rotation_style_action_space():
    world = ArenaEnv("rotation")
    assert world.action_space.n == len(RotationAction) == 5
    assert world.action_enum is RotationAction


def test_direct_style_action_space():
    world = ArenaEnv("direct")
    assert world.action_space.n == len(DirectAction) == 6
    assert world.action_enum is DirectAction


def test_observation_space_is_the_normalized_fixed_size_vector():
    world = ArenaEnv("direct")
    space = world.observation_space
    assert space.shape == (OBS_DIM,)
    assert space.dtype == np.float32
    assert float(space.low.min()) == -1.0 and float(space.high.max()) == 1.0


def test_an_unknown_control_style_is_rejected():
    with pytest.raises(ValueError, match="control_style"):
        ArenaEnv("joystick")


def test_an_out_of_range_action_is_rejected(env):
    with pytest.raises(ValueError, match="outside"):
        env.step(99)


# --- observation ---------------------------------------------------------------------------------


def test_observation_is_float32_and_inside_the_box(env):
    obs, _info = env.reset(seed=1)
    assert obs.dtype == np.float32 and obs.shape == (OBS_DIM,)
    for _ in range(200):
        obs, _r, terminated, truncated, _i = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(obs).all()
        if terminated or truncated:
            break


def test_feature_names_cover_every_slot_in_order():
    assert len(describe()) == OBS_DIM
    assert len(set(describe())) == OBS_DIM


def test_relative_positions_are_in_the_ship_local_frame():
    """An enemy dead ahead reads as +x whatever the ship's heading is."""
    for heading in (0.0, math.pi / 2, -math.pi / 3, 2.5):
        player = Player(CENTRE_X, CENTRE_Y, heading=heading)
        ahead = Enemy(
            CENTRE_X + math.cos(heading) * 100.0,
            CENTRE_Y + math.sin(heading) * 100.0,
            health=1,
            speed=90.0,
        )
        obs = build_observation(player, [ahead], [], 0)
        assert obs[8] * ARENA_DIAGONAL == pytest.approx(100.0, abs=1e-3)
        assert obs[9] * ARENA_DIAGONAL == pytest.approx(0.0, abs=1e-3)
        assert obs[10] * ARENA_DIAGONAL == pytest.approx(100.0, abs=1e-3)


def test_absent_entities_zero_their_slots_and_clear_their_flags():
    obs = build_observation(Player(CENTRE_X, CENTRE_Y), [], [], 0)
    assert obs[13] == 0.0 and obs[17] == 0.0
    assert np.all(obs[8:13] == 0.0)
    assert np.all(obs[14:17] == 0.0)


def test_a_dead_entity_is_not_reported_as_present():
    """Callers hand over lists that may still hold something killed earlier this frame."""
    enemy = Enemy(CENTRE_X + 40.0, CENTRE_Y, health=1, speed=90.0)
    enemy.kill()
    obs = build_observation(Player(CENTRE_X, CENTRE_Y), [enemy], [], 0)
    assert obs[13] == 0.0


def test_an_entity_exactly_on_the_player_produces_no_nan():
    """A single NaN propagates through the network and kills the run silently."""
    player = Player(CENTRE_X, CENTRE_Y)
    obs = build_observation(player, [Enemy(CENTRE_X, CENTRE_Y, health=1, speed=90.0)], [], 0)
    assert np.isfinite(obs).all()
    assert obs[10] == 0.0 and obs[13] == 1.0


def test_heading_is_reported_as_sin_and_cos():
    player = Player(CENTRE_X, CENTRE_Y, heading=math.pi / 3)
    obs = build_observation(player, [], [], 0)
    assert obs[4] == pytest.approx(math.sin(math.pi / 3), abs=1e-6)
    assert obs[5] == pytest.approx(math.cos(math.pi / 3), abs=1e-6)


# --- the step contract ----------------------------------------------------------------------------


def test_one_agent_step_advances_exactly_action_repeat_physics_frames(env):
    env.step(int(DirectAction.SHOOT))
    assert env.bullets, "the shoot action should have produced a bullet"
    bullet = env.bullets[0]
    before = (bullet.x, bullet.y)
    env.step(int(DirectAction.NOOP))
    travelled = math.dist(before, (bullet.x, bullet.y))
    assert travelled == pytest.approx(bullet.speed * ACTION_REPEAT * FIXED_DT, rel=1e-6)


def test_the_step_penalty_is_charged_once_per_agent_step_not_per_frame(env):
    """`ACTION_REPEAT` sets the horizon; it must not also triple the cost of living."""
    _obs, reward, _term, _trunc, _info = env.step(int(DirectAction.NOOP))
    assert reward == pytest.approx(REWARD_PER_STEP)


def test_info_carries_the_behavioural_metrics(env):
    _obs, _r, _term, _trunc, info = env.step(int(DirectAction.NOOP))
    assert set(info) == {"phase", "spawners_destroyed", "enemies_killed", "damage_taken"}


def test_death_terminates_and_is_not_a_truncation(env):
    env.player.health = 1
    env.enemies.append(Enemy(env.player.x, env.player.y, health=1, speed=90.0))
    _obs, reward, terminated, truncated, info = env.step(int(DirectAction.NOOP))
    assert terminated and not truncated
    assert reward == pytest.approx(REWARD_PER_STEP + REWARD_DAMAGE_TAKEN + REWARD_DEATH)
    assert info["damage_taken"] == 1


def test_the_step_cap_truncates_without_terminating(env):
    env.player.max_health = env.player.health = 10**6  # outlive every enemy, not the clock
    env.steps = MAX_EPISODE_STEPS - 1
    _obs, _r, terminated, truncated, _info = env.step(int(DirectAction.NOOP))
    assert truncated and not terminated
    assert env.steps == MAX_EPISODE_STEPS


def test_destroying_every_spawner_advances_the_phase(env):
    for spawner in env.spawners:
        spawner.take_damage(spawner.health)
    _obs, reward, _term, _trunc, info = env.step(int(DirectAction.NOOP))
    assert env.phase == 1 and info["phase"] == 1
    assert reward == pytest.approx(REWARD_PER_STEP + REWARD_PHASE_ADVANCE)
    assert env.spawners, "the next phase must field its own spawners"


def test_an_enemy_that_crashes_into_the_ship_pays_no_kill_reward(env):
    """Otherwise parking in a corner and absorbing contact out-earns shooting."""
    env.enemies.append(Enemy(env.player.x, env.player.y, health=1, speed=90.0))
    _obs, reward, _term, _trunc, info = env.step(int(DirectAction.NOOP))
    assert info["enemies_killed"] == 0
    assert reward == pytest.approx(REWARD_PER_STEP + REWARD_DAMAGE_TAKEN)
    assert reward < REWARD_ENEMY_DESTROYED


def test_the_same_seed_reproduces_the_same_episode():
    first, second = ArenaEnv("rotation"), ArenaEnv("rotation")
    obs_a, _ = first.reset(seed=7)
    obs_b, _ = second.reset(seed=7)
    assert np.array_equal(obs_a, obs_b)
    for action in (1, 1, 4, 2, 4, 0, 3, 4):
        step_a = first.step(action)
        step_b = second.step(action)
        assert np.array_equal(step_a[0], step_b[0])
        assert step_a[1] == step_b[1]


def test_reset_clears_the_counters_of_the_previous_episode(env):
    for _ in range(60):
        env.step(env.action_space.sample())
    env.reset(seed=2)
    assert (env.steps, env.enemies_killed, env.spawners_destroyed, env.damage_taken) == (0, 0, 0, 0)
    assert env.phase == 0 and env.enemies == [] and env.bullets == []
    assert env.episode_return == 0.0


# --- both API surfaces the brief and SB3 ask for -----------------------------------------------


@pytest.mark.parametrize("style", ["rotation", "direct"])
def test_sb3_env_checker_passes(style):
    from stable_baselines3.common.env_checker import check_env

    check_env(ArenaEnv(style), warn=True)


def test_legacy_adapter_presents_the_briefs_four_tuple():
    legacy = LegacyGymAPI(ArenaEnv("direct"))
    obs = legacy.reset()
    assert isinstance(obs, np.ndarray) and obs.shape == (OBS_DIM,)

    result = legacy.step(int(DirectAction.NOOP))
    assert len(result) == 4
    _obs, _reward, done, info = result
    assert isinstance(done, bool)
    # `done` collapses the two, so both must survive separately for a caller that needs them.
    assert info["terminated"] is False and info["truncated"] is False
