"""Tests for `arena.observation` — the agent's entire view of the world (rubric row H2).

Every expected value here is either derived from `config/arena.yaml` or computed by an independent
route (complex-number rotation rather than the module's own sin/cos expression), so a test passing
means the layout is right rather than that the test copied the implementation.

Two fixtures are deliberately avoided throughout, because both hide whole classes of bug:

  * a heading of 0, which makes the ship-local rotation the identity matrix and lets a rotation by
    `+heading`, by `-heading` or by nothing at all produce the same numbers;
  * a target placed dead ahead or dead abeam, which zeroes one local component and hides a swapped
    or sign-flipped axis.

`HEADING` below is 50 degrees and the targets are off-axis for exactly that reason.
"""

from __future__ import annotations

import ast
import cmath
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from arena.constants import ARENA_HEIGHT, ARENA_WIDTH, OBS_DIM
from arena.entities import Enemy, PhaseConfig, Player, Spawner, n_phases, phase_config
from arena.observation import (
    FEATURE_NAMES,
    build_observation,
    describe,
    nearest_alive,
    observation_scales,
    to_ship_local,
)
from common.config import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
OBSERVATION_SOURCE = (REPO_ROOT / "arena" / "observation.py").read_text(encoding="utf-8")

ARENA_YAML = load_yaml("arena")
PLAYER_YAML = ARENA_YAML["player"]
OBSERVATION_YAML = ARENA_YAML["observation"]
PHASES_YAML = ARENA_YAML["phases"]

DIAGONAL = math.hypot(ARENA_WIDTH, ARENA_HEIGHT)
MAX_RELATIVE_SPEED = max(p["enemy_speed"] for p in PHASES_YAML) + PLAYER_YAML["speed"]

CENTRE_X, CENTRE_Y = ARENA_WIDTH / 2.0, ARENA_HEIGHT / 2.0
HEADING = math.radians(50.0)  # never 0: see the module docstring

# Named indices, so a test reads like the layout table rather than like a magic offset.
IDX = {name: index for index, name in enumerate(FEATURE_NAMES)}


def make_player(x=CENTRE_X, y=CENTRE_Y, heading=HEADING) -> Player:
    return Player(x, y, heading=heading)


def make_enemy(x: float, y: float, phase_index: int = 0) -> Enemy:
    return Enemy.from_phase(x, y, phase_config(phase_index))


def make_spawner(x: float, y: float, phase_index: int = 0) -> Spawner:
    return Spawner(x, y, phase_config(phase_index))


def rotate_into_local(heading: float, dx: float, dy: float) -> tuple[float, float]:
    """Independent implementation of the ship-local rotation, via complex multiplication."""
    local = complex(dx, dy) * cmath.exp(-1j * heading)
    return local.real, local.imag


# --- module hygiene ----------------------------------------------------------------------------


def test_observation_source_has_no_pygame_import():
    assert re.search(r"^\s*(import|from)\s+pygame", OBSERVATION_SOURCE, re.MULTILINE) is None


def test_observation_imports_nothing_graphical_or_wall_clock():
    imported: set[str] = set()
    for node in ast.walk(ast.parse(OBSERVATION_SOURCE)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"pygame", "time", "datetime"}), imported


def test_importing_observation_does_not_pull_in_pygame():
    result = subprocess.run(
        [sys.executable, "-c", "import arena.observation, sys; assert 'pygame' not in sys.modules"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# --- shape, dtype and labels -------------------------------------------------------------------


class TestVectorShape:
    def test_is_a_fixed_size_float32_vector(self):
        obs = build_observation(make_player(), [], [], phase=1)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        assert OBS_DIM == 20

    def test_shape_is_identical_however_crowded_the_arena_is(self):
        """Fixed size means fixed size: 40 enemies must not lengthen the vector."""
        player = make_player()
        enemies = [make_enemy(100.0 + 5 * i, 90.0 + 3 * i) for i in range(40)]
        spawners = [make_spawner(800.0, 600.0), make_spawner(120.0, 610.0)]
        crowded = build_observation(player, enemies, spawners, phase=2)
        empty = build_observation(player, [], [], phase=1)
        assert crowded.shape == empty.shape == (OBS_DIM,)

    def test_there_is_no_spawner_exists_flag(self):
        """A dead input is capacity the network pays for and learns nothing from.

        A3-021's validation found `spawner_exists` constant at 1.0 across 20,000 random steps and
        both trained policies, and the cause is structural, not statistical:
        `ArenaEnv._maybe_advance_phase()` lays out the next phase in the same frame the last
        spawner dies, so the flag could never read 0.0. It is gone, and the symmetry with
        `enemy_exists` at index 13 is not a reason to put it back -- enemies genuinely run out,
        spawners never do.
        """
        assert "spawner_exists" not in FEATURE_NAMES
        assert "enemy_exists" in FEATURE_NAMES

    def test_describe_labels_every_index_once(self):
        names = describe()
        assert len(names) == OBS_DIM
        assert len(set(names)) == OBS_DIM
        assert names == list(FEATURE_NAMES)

    def test_describe_returns_a_copy_the_caller_cannot_corrupt(self):
        names = describe()
        names[0] = "mutated"
        assert describe()[0] == "player_x"


class TestScalesComeFromConfig:
    def test_counts_saturate_at_the_configured_caps(self):
        scales = observation_scales()
        assert scales.enemy_count_cap == OBSERVATION_YAML["enemy_count_cap"]
        assert scales.spawner_count_cap == OBSERVATION_YAML["spawner_count_cap"]

    def test_relative_speed_scale_covers_a_head_on_approach(self):
        """Dividing by the enemy speed alone would saturate this feature whenever the player is
        moving toward the enemy at speed, which is most of an episode."""
        scales = observation_scales()
        assert scales.max_relative_speed == MAX_RELATIVE_SPEED
        assert scales.max_relative_speed > max(p["enemy_speed"] for p in PHASES_YAML)

    def test_diagonal_is_the_arena_diagonal(self):
        assert observation_scales().diagonal == pytest.approx(DIAGONAL)


# --- player block ------------------------------------------------------------------------------


class TestPlayerFeatures:
    def test_position_is_centred_on_the_arena(self):
        obs = build_observation(make_player(x=CENTRE_X, y=CENTRE_Y), [], [], phase=1)
        assert obs[IDX["player_x"]] == pytest.approx(0.0)
        assert obs[IDX["player_y"]] == pytest.approx(0.0)

        corner = build_observation(make_player(x=0.0, y=ARENA_HEIGHT), [], [], phase=1)
        assert corner[IDX["player_x"]] == pytest.approx(-1.0)
        assert corner[IDX["player_y"]] == pytest.approx(1.0)

    def test_position_axes_are_not_swapped(self):
        """x and y normalize by different arena dimensions, so the same pixel offset must produce
        different feature values."""
        offset = 100.0
        obs = build_observation(
            make_player(x=CENTRE_X + offset, y=CENTRE_Y + offset), [], [], phase=1
        )
        assert obs[IDX["player_x"]] == pytest.approx(2.0 * offset / ARENA_WIDTH, abs=1e-6)
        assert obs[IDX["player_y"]] == pytest.approx(2.0 * offset / ARENA_HEIGHT, abs=1e-6)
        assert obs[IDX["player_x"]] != pytest.approx(obs[IDX["player_y"]], abs=1e-4)

    def test_velocity_is_scaled_by_the_configured_speed_cap(self):
        player = make_player()
        player.vx, player.vy = 110.0, -55.0
        obs = build_observation(player, [], [], phase=1)
        assert obs[IDX["player_vx"]] == pytest.approx(110.0 / PLAYER_YAML["speed"], abs=1e-6)
        assert obs[IDX["player_vy"]] == pytest.approx(-55.0 / PLAYER_YAML["speed"], abs=1e-6)

    def test_heading_is_carried_as_a_sin_cos_pair(self):
        obs = build_observation(make_player(heading=HEADING), [], [], phase=1)
        assert obs[IDX["heading_sin"]] == pytest.approx(math.sin(HEADING), abs=1e-6)
        assert obs[IDX["heading_cos"]] == pytest.approx(math.cos(HEADING), abs=1e-6)

    def test_no_feature_carries_the_raw_angle(self):
        """A raw radian value would wrap at +-pi and the network could not fit across the seam."""
        heading = math.radians(200.0)  # wraps to -160 degrees inside Player
        player = make_player(heading=heading)
        obs = build_observation(player, [], [], phase=1)
        assert not np.any(np.isclose(obs, player.heading, atol=1e-6))

    def test_heading_features_are_continuous_across_the_wrap(self):
        eps = 1e-3
        before = build_observation(make_player(heading=math.pi - eps), [], [], phase=1)
        after = build_observation(make_player(heading=-math.pi + eps), [], [], phase=1)
        assert np.max(np.abs(before - after)) < 1e-2

    def test_health_is_a_fraction_and_reaches_zero_on_death(self):
        player = make_player()
        assert build_observation(player, [], [], phase=1)[IDX["health"]] == pytest.approx(1.0)

        player.take_damage(1)
        expected = (PLAYER_YAML["max_health"] - 1) / PLAYER_YAML["max_health"]
        assert build_observation(player, [], [], phase=1)[IDX["health"]] == pytest.approx(expected)

        player.invulnerable_remaining = 0.0
        player.take_damage(PLAYER_YAML["max_health"])
        obs = build_observation(player, [], [], phase=1)
        assert obs[IDX["health"]] == 0.0
        assert np.all(np.isfinite(obs))

    def test_phase_spans_zero_to_one_and_saturates_beyond_the_ladder(self):
        first = build_observation(make_player(), [], [], phase=1)[IDX["phase"]]
        last = build_observation(make_player(), [], [], phase=n_phases())[IDX["phase"]]
        beyond = build_observation(make_player(), [], [], phase=n_phases() + 5)[IDX["phase"]]
        assert first == pytest.approx(0.0)
        assert last == pytest.approx(1.0)
        assert beyond == pytest.approx(1.0)

    def test_shoot_cooldown_is_a_fraction_of_the_configured_cooldown(self):
        player = make_player()
        assert build_observation(player, [], [], phase=1)[IDX["shoot_cooldown"]] == 0.0

        player.shoot()
        full = build_observation(player, [], [], phase=1)[IDX["shoot_cooldown"]]
        assert full == pytest.approx(1.0)

        player.shoot_cooldown_remaining = PLAYER_YAML["shoot_cooldown"] / 4.0
        quarter = build_observation(player, [], [], phase=1)[IDX["shoot_cooldown"]]
        assert quarter == pytest.approx(0.25)


# --- the ship-local rotation ---------------------------------------------------------------------


class TestShipLocalFrame:
    @pytest.mark.parametrize("heading_degrees", [50.0, 137.0, -95.0, 179.0])
    @pytest.mark.parametrize("offset", [(120.0, -60.0), (-30.0, 200.0), (-140.0, -80.0)])
    def test_matches_an_independent_rotation(self, heading_degrees, offset):
        heading = math.radians(heading_degrees)
        expected = rotate_into_local(heading, *offset)
        assert to_ship_local(heading, *offset) == pytest.approx(expected, abs=1e-9)

    def test_a_target_dead_ahead_reads_as_pure_forward(self):
        distance = 250.0
        offset = (distance * math.cos(HEADING), distance * math.sin(HEADING))
        local_x, local_y = to_ship_local(HEADING, *offset)
        assert local_x == pytest.approx(distance, abs=1e-9)
        assert local_y == pytest.approx(0.0, abs=1e-9)

    def test_a_target_off_the_nose_lands_on_the_lateral_axis(self):
        """90 degrees clockwise of the nose on screen is local +y; anticlockwise is local -y."""
        distance = 250.0
        clockwise = HEADING + math.pi / 2.0
        local_x, local_y = to_ship_local(
            HEADING, distance * math.cos(clockwise), distance * math.sin(clockwise)
        )
        assert local_x == pytest.approx(0.0, abs=1e-9)
        assert local_y == pytest.approx(distance, abs=1e-9)

        anticlockwise = HEADING - math.pi / 2.0
        _, other_y = to_ship_local(
            HEADING, distance * math.cos(anticlockwise), distance * math.sin(anticlockwise)
        )
        assert other_y == pytest.approx(-distance, abs=1e-9)

    def test_rotation_preserves_distance(self):
        local_x, local_y = to_ship_local(HEADING, 120.0, -60.0)
        assert math.hypot(local_x, local_y) == pytest.approx(math.hypot(120.0, -60.0))

    def test_the_frame_turns_with_the_ship(self):
        """Turn the ship and the world offset by the same angle and the local view is unchanged —
        the invariance the rotation agent depends on."""
        turn = math.radians(37.0)
        base = to_ship_local(HEADING, 120.0, -60.0)
        turned_offset = complex(120.0, -60.0) * cmath.exp(1j * turn)
        turned = to_ship_local(HEADING + turn, turned_offset.real, turned_offset.imag)
        assert turned == pytest.approx(base, abs=1e-9)

    def test_rotating_by_the_wrong_sign_would_be_visible(self):
        """Guards the sign of the rotation: `+heading` and `-heading` must differ here."""
        right_way = to_ship_local(HEADING, 120.0, -60.0)
        wrong_way = to_ship_local(-HEADING, 120.0, -60.0)
        assert right_way[1] != pytest.approx(wrong_way[1], abs=1.0)


# --- nearest enemy and spawner slots --------------------------------------------------------------


class TestEntitySlots:
    def test_enemy_slot_holds_the_rotated_offset_and_distance(self):
        player = make_player()
        dx, dy = 180.0, -95.0
        enemy = make_enemy(player.x + dx, player.y + dy)
        obs = build_observation(player, [enemy], [], phase=1)

        local_x, local_y = rotate_into_local(HEADING, dx, dy)
        assert obs[IDX["enemy_local_dx"]] == pytest.approx(local_x / DIAGONAL, abs=1e-6)
        assert obs[IDX["enemy_local_dy"]] == pytest.approx(local_y / DIAGONAL, abs=1e-6)
        assert obs[IDX["enemy_distance"]] == pytest.approx(math.hypot(dx, dy) / DIAGONAL, abs=1e-6)
        assert obs[IDX["enemy_exists"]] == 1.0

    def test_enemy_offset_is_not_the_raw_world_offset(self):
        """The whole point of the local frame: at a non-zero heading the stored numbers must
        differ from the world-frame offset."""
        player = make_player()
        dx, dy = 180.0, -95.0
        obs = build_observation(player, [make_enemy(player.x + dx, player.y + dy)], [], phase=1)
        assert obs[IDX["enemy_local_dx"]] != pytest.approx(dx / DIAGONAL, abs=1e-3)
        assert obs[IDX["enemy_local_dy"]] != pytest.approx(dy / DIAGONAL, abs=1e-3)

    def test_spawner_slot_holds_the_rotated_offset_and_distance(self):
        player = make_player()
        dx, dy = -220.0, 140.0
        spawner = make_spawner(player.x + dx, player.y + dy)
        obs = build_observation(player, [], [spawner], phase=1)

        local_x, local_y = rotate_into_local(HEADING, dx, dy)
        assert obs[IDX["spawner_local_dx"]] == pytest.approx(local_x / DIAGONAL, abs=1e-6)
        assert obs[IDX["spawner_local_dy"]] == pytest.approx(local_y / DIAGONAL, abs=1e-6)
        assert obs[IDX["spawner_distance"]] == pytest.approx(
            math.hypot(dx, dy) / DIAGONAL, abs=1e-6
        )
        # There is no spawner-exists flag: it could never read 0. `spawners_alive` carries the
        # same "is one there" information with a range the network can actually use.
        assert obs[IDX["spawners_alive"]] > 0.0

    def test_enemy_and_spawner_slots_are_independent(self):
        """A copy-paste bug that filled both slots from the same entity would pass a test with one
        entity of each at mirror-image positions; these are deliberately asymmetric."""
        player = make_player()
        enemy = make_enemy(player.x + 180.0, player.y - 95.0)
        spawner = make_spawner(player.x - 220.0, player.y + 140.0)
        obs = build_observation(player, [enemy], [spawner], phase=1)
        assert obs[IDX["enemy_local_dx"]] != pytest.approx(obs[IDX["spawner_local_dx"]], abs=1e-3)
        assert obs[IDX["enemy_distance"]] != pytest.approx(obs[IDX["spawner_distance"]], abs=1e-3)

    def test_relative_velocity_is_the_closing_velocity_in_the_local_frame(self):
        player = make_player()
        player.vx, player.vy = 40.0, -25.0
        enemy = make_enemy(player.x + 180.0, player.y - 95.0)
        enemy.vx, enemy.vy = -70.0, 30.0

        obs = build_observation(player, [enemy], [], phase=1)
        expected_x, expected_y = rotate_into_local(
            HEADING, enemy.vx - player.vx, enemy.vy - player.vy
        )
        assert obs[IDX["enemy_local_rel_vx"]] == pytest.approx(
            expected_x / MAX_RELATIVE_SPEED, abs=1e-6
        )
        assert obs[IDX["enemy_local_rel_vy"]] == pytest.approx(
            expected_y / MAX_RELATIVE_SPEED, abs=1e-6
        )

    def test_relative_velocity_uses_the_player_velocity_too(self):
        """A stationary enemy still has a non-zero relative velocity while the player moves."""
        player = make_player()
        player.vx, player.vy = 150.0, 0.0
        enemy = make_enemy(player.x + 180.0, player.y - 95.0)
        obs = build_observation(player, [enemy], [], phase=1)
        assert obs[IDX["enemy_local_rel_vx"]] != pytest.approx(0.0, abs=1e-3)

    def test_the_nearest_enemy_wins_regardless_of_list_order(self):
        player = make_player()
        far = make_enemy(player.x + 400.0, player.y + 200.0)
        near = make_enemy(player.x + 60.0, player.y - 35.0)
        # Far one first: a slot filled from `enemies[0]` would pass with the order reversed.
        obs = build_observation(player, [far, near], [], phase=1)
        assert obs[IDX["enemy_distance"]] == pytest.approx(
            math.hypot(60.0, -35.0) / DIAGONAL, abs=1e-6
        )
        assert nearest_alive(player, [far, near]) is near

    def test_dead_entities_are_invisible(self):
        player = make_player()
        corpse = make_enemy(player.x + 30.0, player.y + 20.0)
        corpse.kill()
        live = make_enemy(player.x + 300.0, player.y + 150.0)
        obs = build_observation(player, [corpse, live], [], phase=1)
        assert obs[IDX["enemy_distance"]] == pytest.approx(
            math.hypot(300.0, 150.0) / DIAGONAL, abs=1e-6
        )
        assert obs[IDX["enemies_alive"]] == pytest.approx(
            1.0 / OBSERVATION_YAML["enemy_count_cap"], abs=1e-6
        )

    def test_counts_are_scaled_and_saturate(self):
        player = make_player()
        cap = OBSERVATION_YAML["enemy_count_cap"]
        enemies = [make_enemy(100.0 + i, 90.0 + i) for i in range(3)]
        obs = build_observation(player, enemies, [], phase=1)
        assert obs[IDX["enemies_alive"]] == pytest.approx(3.0 / cap, abs=1e-6)

        swarm = [make_enemy(100.0 + i, 90.0 + i) for i in range(cap * 3)]
        assert build_observation(player, swarm, [], phase=1)[IDX["enemies_alive"]] == 1.0

    def test_spawner_count_is_scaled_by_its_own_cap(self):
        player = make_player()
        spawners = [make_spawner(200.0, 200.0), make_spawner(700.0, 500.0)]
        obs = build_observation(player, [], spawners, phase=1)
        assert obs[IDX["spawners_alive"]] == pytest.approx(
            2.0 / OBSERVATION_YAML["spawner_count_cap"], abs=1e-6
        )


# --- adversarial states ---------------------------------------------------------------------------


class TestAdversarialStates:
    EMPTY_ENEMY_SLOT = ("enemy_local_dx", "enemy_local_dy", "enemy_distance",
                        "enemy_local_rel_vx", "enemy_local_rel_vy", "enemy_exists")
    EMPTY_SPAWNER_SLOT = ("spawner_local_dx", "spawner_local_dy", "spawner_distance")

    def test_empty_slots_are_zeroed_and_their_flags_cleared(self):
        obs = build_observation(make_player(), [], [], phase=1)
        for name in (*self.EMPTY_ENEMY_SLOT, *self.EMPTY_SPAWNER_SLOT):
            assert obs[IDX[name]] == 0.0, name
        assert obs[IDX["enemies_alive"]] == 0.0
        assert obs[IDX["spawners_alive"]] == 0.0

    def test_an_emptied_slot_does_not_keep_the_previous_frames_values(self):
        player = make_player()
        enemy = make_enemy(player.x + 150.0, player.y + 90.0)
        occupied = build_observation(player, [enemy], [], phase=1)
        assert occupied[IDX["enemy_exists"]] == 1.0

        enemy.kill()
        emptied = build_observation(player, [enemy], [], phase=1)
        for name in self.EMPTY_ENEMY_SLOT:
            assert emptied[IDX[name]] == 0.0, name

    def test_zero_distance_produces_no_nan_and_no_division_by_zero(self):
        player = make_player()
        enemy = make_enemy(player.x, player.y)
        spawner = make_spawner(player.x, player.y)
        obs = build_observation(player, [enemy], [spawner], phase=1)

        assert np.all(np.isfinite(obs))
        assert obs[IDX["enemy_distance"]] == 0.0
        assert obs[IDX["enemy_local_dx"]] == 0.0
        assert obs[IDX["enemy_local_dy"]] == 0.0
        assert obs[IDX["spawner_distance"]] == 0.0
        # The entity is there; only its offset is degenerate, so the flag must stay set.
        assert obs[IDX["enemy_exists"]] == 1.0
        assert obs[IDX["spawners_alive"]] > 0.0

    def test_opposite_corners_saturate_the_distance_feature_without_leaving_the_box(self):
        player = make_player(x=0.0, y=0.0)
        enemy = make_enemy(ARENA_WIDTH, ARENA_HEIGHT)
        obs = build_observation(player, [enemy], [], phase=1)
        assert obs[IDX["enemy_distance"]] == pytest.approx(1.0, abs=1e-6)
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)

    def test_a_dead_player_with_an_empty_arena_is_still_a_valid_observation(self):
        player = make_player()
        player.take_damage(PLAYER_YAML["max_health"])
        obs = build_observation(player, [], [], phase=1)
        assert not player.alive
        assert np.all(np.isfinite(obs))
        assert obs[IDX["health"]] == 0.0
        assert np.all(np.abs(obs) <= 1.0)

    def test_overspeed_and_extreme_states_stay_inside_the_box(self):
        """Random adversarial worlds: positions outside the walls, speeds past the cap, entities
        stacked on the player. Every one must stay inside Box(-1, 1) and finite."""
        rng = np.random.default_rng(20260820)
        for _ in range(200):
            player = make_player(
                x=float(rng.uniform(-50.0, ARENA_WIDTH + 50.0)),
                y=float(rng.uniform(-50.0, ARENA_HEIGHT + 50.0)),
                heading=float(rng.uniform(-math.pi, math.pi)),
            )
            player.vx = float(rng.uniform(-2.0, 2.0)) * PLAYER_YAML["speed"]
            player.vy = float(rng.uniform(-2.0, 2.0)) * PLAYER_YAML["speed"]
            player.shoot_cooldown_remaining = float(rng.uniform(0.0, 2.0))

            enemies = []
            for _ in range(int(rng.integers(0, 5))):
                enemy = make_enemy(
                    float(rng.uniform(0.0, ARENA_WIDTH)), float(rng.uniform(0.0, ARENA_HEIGHT))
                )
                enemy.vx = float(rng.uniform(-1.0, 1.0)) * 200.0
                enemy.vy = float(rng.uniform(-1.0, 1.0)) * 200.0
                enemies.append(enemy)
            spawners = [
                make_spawner(
                    float(rng.uniform(0.0, ARENA_WIDTH)), float(rng.uniform(0.0, ARENA_HEIGHT))
                )
                for _ in range(int(rng.integers(0, 4)))
            ]

            obs = build_observation(player, enemies, spawners, phase=int(rng.integers(1, 6)))
            assert np.all(np.isfinite(obs))
            assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
            assert obs.dtype == np.float32

    def test_an_unusual_phase_config_still_normalizes(self):
        """A one-phase config would divide by `n_phases() - 1 == 0` without the guard."""
        scales = observation_scales()
        assert scales.max_phase_index >= 1.0
        solo = PhaseConfig(**PHASES_YAML[0])
        spawner = Spawner(300.0, 300.0, solo)
        obs = build_observation(make_player(), [], [spawner], phase=1)
        assert np.all(np.isfinite(obs))
