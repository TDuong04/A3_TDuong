"""Behaviour tests for `arena.entities` — the Part II simulation (rubric row G).

Every number the entities move by is either read from `config/arena.yaml` inside the test or
derived from `FIXED_DT`, so a test failing here means the simulation changed, not that a magic
number in the test went stale. Where the physics is deterministic the assertions are on exact
integrated values rather than on "the position changed": a seek that moves the enemy one pixel per
minute would still change the position.
"""

from __future__ import annotations

import ast
import inspect
import math
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from arena.constants import ARENA_HEIGHT, ARENA_WIDTH, FIXED_DT, FPS
from arena.entities import (
    Bullet,
    Enemy,
    EntityConfig,
    PhaseConfig,
    Player,
    PlayerConfig,
    Spawner,
    entity_config,
    n_phases,
    phase_config,
    player_config,
)
from common.config import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTITIES_SOURCE = (REPO_ROOT / "arena" / "entities.py").read_text(encoding="utf-8")

ARENA_YAML = load_yaml("arena")
PLAYER_YAML = ARENA_YAML["player"]
ENTITIES_YAML = ARENA_YAML["entities"]
PHASES_YAML = ARENA_YAML["phases"]

CENTRE_X, CENTRE_Y = ARENA_WIDTH / 2, ARENA_HEIGHT / 2


def frames(seconds: float) -> int:
    """How many fixed physics frames a duration in simulated seconds occupies."""
    return round(seconds / FIXED_DT)


def coast(entity, n: int) -> None:
    for _ in range(n):
        entity.update()


# --- module hygiene: no pygame, no wall clock --------------------------------------------------


def test_entities_source_has_no_pygame_import():
    assert re.search(r"^\s*(import|from)\s+pygame", ENTITIES_SOURCE, re.MULTILINE) is None
    assert "pygame" not in ENTITIES_SOURCE.replace("No pygame imports", "")


def test_entities_imports_no_clock_module():
    """`time`, `datetime` and `pygame` are the three ways wall-clock physics sneaks in."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(ENTITIES_SOURCE)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"time", "datetime", "pygame"}), imported


def test_importing_entities_does_not_pull_in_pygame():
    result = subprocess.run(
        [sys.executable, "-c", "import arena.entities, sys; assert 'pygame' not in sys.modules"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_entities_import_survives_pygame_being_unimportable():
    """Proves the independence transitively: with pygame banned from the import system, the
    simulation module must still import and simulate."""
    script = """
import sys

class BanPygame:
    def find_spec(self, name, path=None, target=None):
        if name == 'pygame' or name.startswith('pygame.'):
            raise ImportError('pygame is banned in simulation modules')
        return None

sys.meta_path.insert(0, BanPygame())
from arena.entities import Player
player = Player(100.0, 100.0)
player.thrust()
player.update()
assert player.x > 100.0
assert 'pygame' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("cls", [Player, Enemy, Spawner, Bullet])
def test_update_never_accepts_a_delta_time(cls):
    """FIXED_DT is the only clock; an entity that could be handed a dt would eventually be handed
    a wall-clock one, and training and evaluation would stop matching."""
    names = list(inspect.signature(cls.update).parameters)
    assert not any(re.search(r"dt|delta|time|elapsed", name) for name in names), names


# --- configuration is the only source of numbers ----------------------------------------------


def test_player_defaults_come_from_the_yaml():
    config = player_config()
    assert config == PlayerConfig(**PLAYER_YAML)
    player = Player(CENTRE_X, CENTRE_Y)
    assert player.max_health == PLAYER_YAML["max_health"]
    assert player.health == PLAYER_YAML["max_health"]
    assert player.radius == PLAYER_YAML["radius"]


def test_entity_defaults_come_from_the_yaml():
    assert entity_config() == EntityConfig(**ENTITIES_YAML)
    enemy = Enemy(0.0, 0.0, health=1, speed=10.0)
    assert enemy.radius == ENTITIES_YAML["enemy_radius"]
    assert enemy.contact_damage == ENTITIES_YAML["enemy_contact_damage"]
    bullet = Bullet(0.0, 0.0, heading=0.0, speed=1.0)
    assert bullet.radius == ENTITIES_YAML["bullet_radius"]
    assert bullet.damage == ENTITIES_YAML["bullet_damage"]
    assert bullet.lifetime_remaining == ENTITIES_YAML["bullet_lifetime"]


def test_phase_config_reads_the_configured_ladder():
    assert n_phases() == len(PHASES_YAML)
    for index, block in enumerate(PHASES_YAML):
        assert phase_config(index) == PhaseConfig(**block)


def test_phase_beyond_the_ladder_clamps_to_the_hardest_phase():
    """An agent that clears the last configured phase keeps playing, at the hardest settings."""
    assert phase_config(n_phases() + 5) == phase_config(n_phases() - 1)


def test_negative_phase_index_is_rejected():
    with pytest.raises(IndexError):
        PhaseConfig.from_yaml(-1)


# --- player physics: fixed timestep -----------------------------------------------------------


def test_one_thrust_frame_integrates_exactly_one_fixed_dt():
    config = player_config()
    player = Player(CENTRE_X, CENTRE_Y)
    player.thrust()
    player.update()

    expected_v = (config.thrust * FIXED_DT) * config.drag**FIXED_DT
    assert player.vx == pytest.approx(expected_v, rel=1e-12)
    assert player.vy == 0.0
    assert player.x == pytest.approx(CENTRE_X + expected_v * FIXED_DT, rel=1e-12)
    # The literal is what the shipped config produces; it is here so that a silent change to the
    # integrator (drag applied before acceleration, dt squared, ...) cannot pass unnoticed.
    assert player.vx == pytest.approx(6.654970215244861, rel=1e-9)


def test_thrust_follows_the_documented_recurrence_over_many_frames():
    config = player_config()
    player = Player(CENTRE_X, CENTRE_Y)
    drag_per_frame = config.drag**FIXED_DT

    v, x = 0.0, CENTRE_X
    for _ in range(30):
        player.thrust()
        player.update()
        v = (v + config.thrust * FIXED_DT) * drag_per_frame
        x += v * FIXED_DT

    assert player.vx == pytest.approx(v, rel=1e-12)
    assert player.x == pytest.approx(x, rel=1e-12)


def test_top_speed_is_reached_on_the_frame_the_closed_form_predicts():
    """Pins the whole model at once: FIXED_DT, thrust, drag and the speed cap together decide
    which frame the ship saturates on, and the closed form of that geometric series is computed
    here independently of the implementation."""
    config = player_config()
    retained = config.drag**FIXED_DT
    per_frame = config.thrust * FIXED_DT * retained
    terminal = per_frame / (1.0 - retained)  # speed the ship would reach with no cap
    predicted = math.ceil(math.log(1.0 - config.speed / terminal) / math.log(retained))

    player = Player(CENTRE_X, CENTRE_Y)
    saturated_on = None
    for frame in range(1, 10 * FPS + 1):
        player.thrust()
        player.update()
        player.y = CENTRE_Y  # the wall is not what this test is about
        if saturated_on is None and player.speed >= config.speed - 1e-9:
            saturated_on = frame
    assert saturated_on == predicted


def test_coasting_retains_the_configured_drag_fraction_per_second():
    player = Player(CENTRE_X, CENTRE_Y)
    player.vx = 100.0
    coast(player, FPS)
    assert player.vx == pytest.approx(100.0 * PLAYER_YAML["drag"], rel=1e-9)


def test_speed_is_capped_at_the_configured_maximum():
    """Two seconds of burning downward from the top wall: far enough to saturate, not far enough
    to reach the bottom wall, so the cap is the only thing that can be limiting the ship."""
    cap = player_config().speed
    player = Player(CENTRE_X, 20.0, heading=math.pi / 2)
    for _ in range(2 * FPS):
        player.thrust()
        player.update()
    assert player.y < ARENA_HEIGHT - player.radius
    assert player.vx == pytest.approx(0.0, abs=1e-9)
    assert player.vy == pytest.approx(cap, rel=1e-9)
    assert player.speed == pytest.approx(cap, rel=1e-9)


def test_thrust_magnitude_is_config_driven():
    """Double the configured thrust, double the first frame's velocity."""
    config = player_config()
    doubled = replace(config, thrust=config.thrust * 2)
    a, b = Player(CENTRE_X, CENTRE_Y), Player(CENTRE_X, CENTRE_Y, config=doubled)
    a.thrust()
    a.update()
    b.thrust()
    b.update()
    assert b.vx == pytest.approx(2 * a.vx, rel=1e-12)


def test_rotation_turns_at_the_configured_degrees_per_second():
    player = Player(CENTRE_X, CENTRE_Y, heading=0.0)
    half_second = FPS // 2
    for _ in range(half_second):
        player.rotate(+1)
        player.update()
    assert player.heading_degrees == pytest.approx(PLAYER_YAML["rotation_speed"] / 2, rel=1e-9)

    for _ in range(half_second):
        player.rotate(-1)
        player.update()
    assert player.heading_degrees == pytest.approx(0.0, abs=1e-9)


def test_rotation_alone_never_moves_the_ship():
    player = Player(CENTRE_X, CENTRE_Y)
    for _ in range(FPS):
        player.rotate(-1)
        player.update()
    assert (player.x, player.y) == (CENTRE_X, CENTRE_Y)
    assert (player.vx, player.vy) == (0.0, 0.0)


def test_heading_stays_wrapped_into_pi_range():
    player = Player(CENTRE_X, CENTRE_Y)
    for _ in range(10 * FPS):
        player.rotate(+1)
        player.update()
    assert -math.pi <= player.heading < math.pi


def test_inputs_do_not_persist_into_the_next_frame():
    """One action per frame: a thrust must not keep accelerating the ship forever after."""
    player = Player(CENTRE_X, CENTRE_Y)
    player.thrust()
    player.update()
    accelerated = player.vx
    player.update()
    assert player.vx < accelerated  # only drag acts now
    assert player.vx == pytest.approx(accelerated * PLAYER_YAML["drag"] ** FIXED_DT, rel=1e-12)


# --- the two control styles share one integrator ----------------------------------------------


def test_both_control_styles_produce_identical_trajectories_on_axis():
    """Rubric I / report R6: the two agents must be compared over one dynamical system."""
    rotation_style = Player(CENTRE_X, CENTRE_Y, heading=0.0)
    direct_style = Player(CENTRE_X, CENTRE_Y)

    for _ in range(2 * FPS):
        rotation_style.thrust()  # style 1: burn along the current heading
        direct_style.move(1.0, 0.0)  # style 2: burn to the right
        rotation_style.update()
        direct_style.update()

    assert rotation_style.position == direct_style.position
    assert rotation_style.velocity == direct_style.velocity
    assert rotation_style.heading == direct_style.heading


def test_both_control_styles_agree_off_axis():
    rotation_style = Player(CENTRE_X, CENTRE_Y, heading=math.pi / 4)
    direct_style = Player(CENTRE_X, CENTRE_Y)

    for _ in range(FPS):
        rotation_style.thrust()
        direct_style.move(1.0, 1.0)
        rotation_style.update()
        direct_style.update()

    assert rotation_style.x == pytest.approx(direct_style.x, rel=1e-9)
    assert rotation_style.y == pytest.approx(direct_style.y, rel=1e-9)
    assert rotation_style.speed == pytest.approx(direct_style.speed, rel=1e-9)


def test_direct_movement_has_inertia_rather_than_teleporting():
    """If `move` set the velocity directly the two styles would not be comparable at all."""
    player = Player(CENTRE_X, CENTRE_Y)
    player.move(1.0, 0.0)
    player.update()
    # One frame of a movement command is one frame of acceleration, not a jump to top speed.
    assert 0.0 < player.vx < player_config().speed / 10

    for _ in range(29):
        player.move(1.0, 0.0)
        player.update()
    cruising = player.vx

    player.move(-1.0, 0.0)
    player.update()
    # Momentum carries the ship right for a while after the command reverses.
    assert 0.0 < player.vx < cruising


def test_direct_movement_faces_the_commanded_direction():
    """Style 2 has no rotate action, so `move` is the only way it can aim a shot."""
    player = Player(CENTRE_X, CENTRE_Y)
    player.move(0.0, -1.0)
    assert player.heading == pytest.approx(-math.pi / 2, rel=1e-12)
    bullet = player.shoot()
    assert bullet is not None
    assert bullet.vx == pytest.approx(0.0, abs=1e-9)
    assert bullet.vy == pytest.approx(-PLAYER_YAML["bullet_speed"], rel=1e-12)


def test_null_direction_is_a_noop_and_never_produces_nan():
    player = Player(CENTRE_X, CENTRE_Y, heading=1.0)
    player.move(0.0, 0.0)
    player.update()
    assert player.heading == pytest.approx(1.0)
    assert (player.vx, player.vy) == (0.0, 0.0)
    assert math.isfinite(player.x) and math.isfinite(player.y)


# --- arena bounds -------------------------------------------------------------------------------


def test_walls_clamp_the_player_and_stop_it_dead():
    player = Player(player_config().radius + 5.0, CENTRE_Y)
    for _ in range(FPS):
        player.move(-1.0, 0.0)
        player.update()
    assert player.x == player_config().radius
    assert player.vx == 0.0
    assert 0.0 <= player.y <= ARENA_HEIGHT


def test_the_player_can_never_leave_the_arena():
    player = Player(CENTRE_X, CENTRE_Y)
    for corner in ((-1.0, -1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0)):
        for _ in range(3 * FPS):
            player.move(*corner)
            player.update()
            assert player.radius <= player.x <= ARENA_WIDTH - player.radius
            assert player.radius <= player.y <= ARENA_HEIGHT - player.radius


# --- shooting: cooldown and the live bullet cap -----------------------------------------------


def test_shooting_respects_the_configured_cooldown_on_both_sides():
    player = Player(CENTRE_X, CENTRE_Y)
    assert player.shoot() is not None

    ready_after = frames(PLAYER_YAML["shoot_cooldown"])
    coast(player, ready_after - 1)
    assert not player.can_shoot
    assert player.shoot() is None

    player.update()
    assert player.can_shoot
    assert player.shoot() is not None


def test_cooldown_length_is_config_driven():
    quick = replace(player_config(), shoot_cooldown=PLAYER_YAML["shoot_cooldown"] / 5)
    player = Player(CENTRE_X, CENTRE_Y, config=quick)
    player.shoot()
    coast(player, frames(quick.shoot_cooldown) - 1)
    assert player.shoot() is None
    player.update()
    assert player.shoot() is not None


def test_the_live_bullet_cap_holds_under_sustained_fire():
    """Fire every single frame for ten simulated seconds; the list must never exceed the cap."""
    cap = 3
    entities = replace(entity_config(), max_bullets=cap)
    config = replace(player_config(), shoot_cooldown=2 * FIXED_DT)
    player = Player(CENTRE_X, CENTRE_Y, config=config, entities=entities)

    bullets: list[Bullet] = []
    high_water = 0
    refused_by_cap = 0
    for _ in range(10 * FPS):
        was_ready = player.can_shoot
        bullet = player.shoot(len(bullets))
        if bullet is None:
            refused_by_cap += was_ready
        else:
            bullets.append(bullet)
        for live in bullets:
            live.update()
        bullets = [live for live in bullets if live.alive]
        player.update()
        high_water = max(high_water, len(bullets))
        assert len(bullets) <= cap

    assert high_water == cap, "the cap never bound, so the test proved nothing"
    assert refused_by_cap > 0


def test_a_shot_refused_by_the_cap_does_not_burn_the_cooldown():
    player = Player(CENTRE_X, CENTRE_Y)
    assert player.shoot(entity_config().max_bullets) is None
    assert player.can_shoot
    assert player.shoot(entity_config().max_bullets - 1) is not None
    assert not player.can_shoot


def test_a_dead_player_cannot_shoot():
    player = Player(CENTRE_X, CENTRE_Y)
    for _ in range(PLAYER_YAML["max_health"]):
        player.take_damage(1)
        coast(player, frames(PLAYER_YAML["invulnerability_time"]))
    assert not player.alive
    assert player.shoot() is None


def test_the_bullet_leaves_the_muzzle_not_the_ship_centre():
    player = Player(CENTRE_X, CENTRE_Y, heading=0.0)
    bullet = player.shoot()
    assert bullet is not None
    assert bullet.x == pytest.approx(CENTRE_X + player.radius + bullet.radius, rel=1e-12)
    assert bullet.y == pytest.approx(CENTRE_Y, abs=1e-12)


# --- bullets ------------------------------------------------------------------------------------


def test_bullet_travels_at_the_configured_speed_along_its_heading():
    speed = PLAYER_YAML["bullet_speed"]
    bullet = Bullet(100.0, 200.0, heading=math.pi / 2, speed=speed)
    steps = 30
    for _ in range(steps):
        bullet.update()
    assert bullet.x == pytest.approx(100.0, abs=1e-9)
    assert bullet.y == pytest.approx(200.0 + speed * steps * FIXED_DT, rel=1e-9)
    assert bullet.alive


def test_bullet_expires_exactly_at_the_configured_lifetime():
    lifetime = ENTITIES_YAML["bullet_lifetime"]
    bullet = Bullet(50.0, CENTRE_Y, heading=0.0, speed=PLAYER_YAML["bullet_speed"])
    coast(bullet, frames(lifetime) - 1)
    assert bullet.alive
    assert not bullet.is_outside_arena, "this test must isolate ageing from leaving the arena"
    bullet.update()
    assert not bullet.alive


def test_bullet_lifetime_is_config_driven():
    entities = replace(entity_config(), bullet_lifetime=ENTITIES_YAML["bullet_lifetime"] / 4)
    bullet = Bullet(CENTRE_X, CENTRE_Y, heading=math.pi, speed=1.0, config=entities)
    coast(bullet, frames(entities.bullet_lifetime) - 1)
    assert bullet.alive
    bullet.update()
    assert not bullet.alive


def test_bullet_dies_when_it_leaves_the_arena():
    bullet = Bullet(ARENA_WIDTH - 1.0, CENTRE_Y, heading=0.0, speed=PLAYER_YAML["bullet_speed"])
    bullet.update()
    assert not bullet.alive
    assert bullet.lifetime_remaining > 0.0, "it should have died from the wall, not from age"


def test_a_dead_bullet_is_inert():
    bullet = Bullet(CENTRE_X, CENTRE_Y, heading=0.0, speed=500.0)
    bullet.kill()
    before = bullet.position
    bullet.update()
    assert bullet.position == before


# --- circle-vs-circle collision ----------------------------------------------------------------


def test_collision_triggers_exactly_at_the_sum_of_the_radii():
    enemy = Enemy(CENTRE_X, CENTRE_Y, health=1, speed=0.0)
    reach = enemy.radius + entity_config().bullet_radius

    overlapping = Bullet(CENTRE_X + reach - 1.0, CENTRE_Y, heading=0.0, speed=0.0)
    touching = Bullet(CENTRE_X + reach, CENTRE_Y, heading=0.0, speed=0.0)
    clear = Bullet(CENTRE_X + reach + 1e-6, CENTRE_Y, heading=0.0, speed=0.0)

    assert enemy.collides_with(overlapping)
    assert enemy.collides_with(touching)
    assert not enemy.collides_with(clear)
    # Symmetric, so the env may test from either side.
    assert clear.collides_with(enemy) is False
    assert touching.collides_with(enemy) is True


def test_collision_uses_the_diagonal_distance_not_a_bounding_box():
    player = Player(CENTRE_X, CENTRE_Y)
    reach = player.radius + entity_config().enemy_radius
    offset = reach * 0.72  # inside the box corner, outside the circle
    corner = Enemy(CENTRE_X + offset, CENTRE_Y + offset, health=1, speed=0.0)
    assert math.hypot(offset, offset) > reach
    assert not player.collides_with(corner)


def test_dead_entities_stop_colliding():
    player = Player(CENTRE_X, CENTRE_Y)
    enemy = Enemy(CENTRE_X, CENTRE_Y, health=1, speed=0.0)
    assert player.collides_with(enemy)
    enemy.kill()
    assert not player.collides_with(enemy)
    assert not enemy.collides_with(player)


# --- invulnerability window ---------------------------------------------------------------------


def test_a_hit_opens_the_invulnerability_window():
    player = Player(CENTRE_X, CENTRE_Y)
    assert player.take_damage(1) is True
    assert player.health == PLAYER_YAML["max_health"] - 1
    assert player.is_invulnerable
    # Contact damage is applied every frame the enemy overlaps; without the window one touch would
    # drain the whole health bar in five frames.
    assert player.take_damage(1) is False
    assert player.health == PLAYER_YAML["max_health"] - 1


def test_invulnerability_expires_on_the_exact_configured_frame():
    window = frames(PLAYER_YAML["invulnerability_time"])
    player = Player(CENTRE_X, CENTRE_Y)
    player.take_damage(1)

    coast(player, window - 1)
    assert player.is_invulnerable
    assert player.take_damage(1) is False
    assert player.health == PLAYER_YAML["max_health"] - 1

    player.update()
    assert not player.is_invulnerable
    assert player.take_damage(1) is True
    assert player.health == PLAYER_YAML["max_health"] - 2


def test_invulnerability_length_is_config_driven():
    brief = replace(player_config(), invulnerability_time=PLAYER_YAML["invulnerability_time"] / 6)
    player = Player(CENTRE_X, CENTRE_Y, config=brief)
    player.take_damage(1)
    coast(player, frames(brief.invulnerability_time) - 1)
    assert player.take_damage(1) is False
    player.update()
    assert player.take_damage(1) is True


def test_the_player_dies_at_zero_health_and_absorbs_nothing_after():
    window = frames(PLAYER_YAML["invulnerability_time"])
    player = Player(CENTRE_X, CENTRE_Y)
    for _ in range(PLAYER_YAML["max_health"]):
        assert player.alive
        assert player.take_damage(1) is True
        coast(player, window)
    assert player.health == 0
    assert not player.alive
    assert player.take_damage(1) is False
    assert player.health == 0


# --- enemies seek the player ---------------------------------------------------------------------


def test_enemy_moves_straight_at_the_player_at_the_phase_speed():
    phase = phase_config(0)
    enemy = Enemy.from_phase(100.0, 300.0, phase)
    enemy.update(500.0, 300.0)
    assert enemy.x == pytest.approx(100.0 + phase.enemy_speed * FIXED_DT, rel=1e-12)
    assert enemy.y == 300.0
    assert enemy.velocity == pytest.approx((phase.enemy_speed, 0.0))


def test_enemy_closes_the_gap_by_speed_times_elapsed_simulated_time():
    phase = phase_config(0)
    enemy = Enemy.from_phase(200.0, 200.0, phase)
    target = (600.0, 500.0)
    start = math.hypot(target[0] - 200.0, target[1] - 200.0)

    steps = FPS
    for _ in range(steps):
        enemy.update(*target)

    travelled = start - math.hypot(target[0] - enemy.x, target[1] - enemy.y)
    assert travelled == pytest.approx(phase.enemy_speed * steps * FIXED_DT, rel=1e-9)


def test_enemy_re_aims_at_a_moving_player():
    enemy = Enemy(300.0, 300.0, health=1, speed=100.0)
    enemy.update(900.0, 300.0)
    assert enemy.vx > 0 and enemy.vy == 0
    enemy.update(300.0, 900.0)
    assert enemy.vx < 0 and enemy.vy > 0


def test_enemy_speed_comes_from_the_phase_it_was_spawned_in():
    early = Enemy.from_phase(0.0, 0.0, phase_config(0))
    late = Enemy.from_phase(0.0, 0.0, phase_config(len(PHASES_YAML) - 1))
    assert early.speed == PHASES_YAML[0]["enemy_speed"]
    assert late.speed == PHASES_YAML[-1]["enemy_speed"]
    assert late.speed > early.speed


def test_enemy_on_top_of_the_player_produces_no_nan():
    """A zero-distance seek is the classic divide-by-zero that poisons the observation vector."""
    enemy = Enemy(CENTRE_X, CENTRE_Y, health=1, speed=100.0)
    enemy.update(CENTRE_X, CENTRE_Y)
    assert (enemy.x, enemy.y) == (CENTRE_X, CENTRE_Y)
    assert enemy.velocity == (0.0, 0.0)
    assert math.isfinite(enemy.x) and math.isfinite(enemy.y)


def test_enemy_health_comes_from_the_phase_and_dies_when_it_runs_out():
    phase = phase_config(len(PHASES_YAML) - 1)
    enemy = Enemy.from_phase(CENTRE_X, CENTRE_Y, phase)
    assert enemy.health == phase.enemy_health >= 2
    for remaining in range(phase.enemy_health - 1, 0, -1):
        assert enemy.take_damage(1) is True
        assert enemy.alive
        assert enemy.health == remaining
    assert enemy.take_damage(1) is True
    assert not enemy.alive
    assert enemy.health == 0
    assert enemy.take_damage(1) is False, "a corpse must not absorb a second bullet"


def test_a_dead_enemy_stops_moving():
    enemy = Enemy(100.0, 100.0, health=1, speed=200.0)
    enemy.kill()
    enemy.update(900.0, 100.0)
    assert enemy.position == (100.0, 100.0)


def test_enemies_stay_inside_the_arena():
    enemy = Enemy(50.0, 50.0, health=1, speed=400.0)
    for _ in range(FPS):
        enemy.update(-500.0, -500.0)
    assert enemy.x == enemy.radius
    assert enemy.y == enemy.radius


# --- spawners -------------------------------------------------------------------------------------


def test_spawner_emits_on_the_configured_interval_and_not_between():
    phase = phase_config(0)
    spawner = Spawner(CENTRE_X, CENTRE_Y, phase)
    interval = frames(phase.spawn_interval)

    emitted_on = [frame for frame in range(1, 3 * interval + 1) if spawner.update() is not None]
    assert emitted_on == [interval, 2 * interval, 3 * interval]
    assert spawner.enemies_spawned == 3


def test_spawn_interval_is_read_from_the_phase():
    late = phase_config(len(PHASES_YAML) - 1)
    assert late.spawn_interval < phase_config(0).spawn_interval
    spawner = Spawner(CENTRE_X, CENTRE_Y, late)
    interval = frames(late.spawn_interval)
    emitted_on = [frame for frame in range(1, 2 * interval + 1) if spawner.update() is not None]
    assert emitted_on == [interval, 2 * interval]


def test_the_first_enemy_arrives_a_full_interval_after_the_phase_starts():
    phase = phase_config(0)
    spawner = Spawner(CENTRE_X, CENTRE_Y, phase)
    assert spawner.time_to_next_spawn == phase.spawn_interval
    for _ in range(frames(phase.spawn_interval) - 1):
        assert spawner.update() is None
    assert spawner.update() is not None


def test_a_destroyed_spawner_never_emits_again():
    phase = phase_config(0)
    spawner = Spawner(CENTRE_X, CENTRE_Y, phase)
    interval = frames(phase.spawn_interval)

    for _ in range(interval - 1):
        spawner.update()  # parked one frame short of an emit, the worst case for this bug
    spawner.kill()

    assert spawner.enemies_spawned == 0
    for _ in range(10 * interval):
        assert spawner.update() is None
    assert spawner.enemies_spawned == 0


def test_a_spawner_shot_down_mid_cycle_stops_emitting():
    phase = phase_config(0)
    spawner = Spawner(CENTRE_X, CENTRE_Y, phase)
    for _ in range(frames(phase.spawn_interval)):
        spawner.update()
    assert spawner.enemies_spawned == 1

    for _ in range(phase.spawner_health):
        assert spawner.take_damage(1) is True
    assert not spawner.alive
    assert spawner.health == 0

    for _ in range(20 * frames(phase.spawn_interval)):
        assert spawner.update() is None
    assert spawner.enemies_spawned == 1


def test_spawner_health_comes_from_the_phase():
    for index, block in enumerate(PHASES_YAML):
        spawner = Spawner(CENTRE_X, CENTRE_Y, phase_config(index))
        assert spawner.health == block["spawner_health"]
        assert spawner.max_health == block["spawner_health"]
        for _ in range(block["spawner_health"] - 1):
            spawner.take_damage(1)
        assert spawner.alive, "a spawner must survive until the last point of health is gone"
        spawner.take_damage(1)
        assert not spawner.alive


def test_spawners_are_stationary():
    spawner = Spawner(300.0, 400.0, phase_config(0))
    for _ in range(10 * frames(phase_config(0).spawn_interval)):
        spawner.update()
    assert spawner.position == (300.0, 400.0)


def test_emitted_enemies_match_the_phase_and_start_clear_of_the_rim():
    phase = phase_config(1)
    spawner = Spawner(CENTRE_X, CENTRE_Y, phase)
    interval = frames(phase.spawn_interval)

    enemies: list[Enemy] = []
    for _ in range(3 * interval):
        enemy = spawner.update()
        if enemy is not None:
            enemies.append(enemy)

    assert len(enemies) == 3
    for enemy in enemies:
        assert enemy.health == phase.enemy_health
        assert enemy.speed == phase.enemy_speed
        assert spawner.distance_to(enemy) == pytest.approx(
            spawner.radius + enemy.radius, rel=1e-9
        )
    positions = {(round(e.x, 6), round(e.y, 6)) for e in enemies}
    assert len(positions) == 3, "successive enemies must not stack on one point"


def test_emitted_enemies_are_placed_inside_the_arena():
    phase = phase_config(0)
    corner = Spawner(0.0, 0.0, phase)
    interval = frames(phase.spawn_interval)
    for _ in range(5 * interval):
        enemy = corner.update()
        if enemy is not None:
            assert enemy.radius <= enemy.x <= ARENA_WIDTH - enemy.radius
            assert enemy.radius <= enemy.y <= ARENA_HEIGHT - enemy.radius


# --- a whole-simulation smoke run ----------------------------------------------------------------


def test_a_headless_minute_of_simulation_stays_finite_and_bounded():
    """Everything wired together the way `ArenaEnv` will wire it, for one simulated minute."""
    phase = phase_config(0)
    player = Player(CENTRE_X, CENTRE_Y)
    spawners = [Spawner(120.0, 120.0, phase), Spawner(840.0, 560.0, phase)]
    enemies: list[Enemy] = []
    bullets: list[Bullet] = []
    hits = 0

    for frame in range(60 * FPS):
        player.rotate(1 if frame % 120 < 60 else -1)
        player.thrust()
        bullet = player.shoot(len(bullets))
        if bullet is not None:
            bullets.append(bullet)
        player.update()

        for spawner in spawners:
            spawned = spawner.update()
            if spawned is not None:
                enemies.append(spawned)

        for enemy in enemies:
            enemy.update(player.x, player.y)
        for live in bullets:
            live.update()

        for live in bullets:
            for target in [*enemies, *spawners]:
                if live.alive and live.collides_with(target):
                    target.take_damage(live.damage)
                    live.kill()
        for enemy in enemies:
            if enemy.collides_with(player) and player.take_damage(enemy.contact_damage):
                hits += 1

        enemies = [enemy for enemy in enemies if enemy.alive]
        bullets = [live for live in bullets if live.alive]

        assert len(bullets) <= entity_config().max_bullets
        assert math.isfinite(player.x) and math.isfinite(player.y)
        assert player.radius <= player.x <= ARENA_WIDTH - player.radius
        assert player.radius <= player.y <= ARENA_HEIGHT - player.radius
        if not player.alive:
            break

    assert hits > 0, "60 seconds of seeking enemies should land at least one hit"
    assert hits <= 60 / PLAYER_YAML["invulnerability_time"] + 1, "the window was not enforced"
