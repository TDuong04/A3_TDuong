"""Headless tests for `arena.render` — rubric row G's visual half.

Pygame normally needs a display, which a CI box and a marker's laptop may not have. Every test here
runs under `SDL_VIDEODRIVER=dummy` (set below, before pygame touches the video subsystem) and draws
to an off-screen `pygame.Surface` via `ArenaRenderer(headless=True)`, so the whole renderer is
exercised without a window ever opening — the same arrangement `test_gridworld_render.py` uses.

What these tests can and cannot do: they read the pixels the renderer actually produced, so they
prove each entity type reaches the screen in its own colour and that the overlay and banner appear
only when they should. They cannot prove the picture is *pretty* — that is ten minutes of watching
`eval/play_arena.py` by hand.

The load-bearing test is `test_drawing_never_mutates_the_simulation`: an effect that wrote back into
the env would make evaluation stop matching training, and the drift would look like a training bug
for as long as it took to find.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402  (must follow the driver assignment above)

from arena.constants import ARENA_HEIGHT, ARENA_WIDTH, DirectAction  # noqa: E402
from arena.entities import Enemy  # noqa: E402
from arena.env import ArenaEnv  # noqa: E402
from arena.observation import describe  # noqa: E402
from arena.render import (  # noqa: E402
    BANNER_SECONDS,
    COLOR_BULLET,
    COLOR_ENEMY,
    COLOR_ENEMY_EYE,
    COLOR_ENGINE_CORE,
    COLOR_HEALTH,
    COLOR_OVERLAY_HEADING,
    COLOR_PLAYER,
    COLOR_PLAYER_CANOPY,
    COLOR_SPAWNER,
    ArenaRenderConfig,
    ArenaRenderer,
    _make_starfield,
)


@pytest.fixture
def env() -> ArenaEnv:
    world = ArenaEnv("direct")
    world.reset(seed=0)
    return world


@pytest.fixture
def renderer() -> ArenaRenderer:
    return ArenaRenderer(headless=True)


def pixels(surface: pygame.Surface) -> np.ndarray:
    """The surface as an (H, W, 3) array, so a colour can be counted across the whole frame."""
    return pygame.surfarray.array3d(surface).transpose(1, 0, 2)


def count_color(surface: pygame.Surface, color: tuple[int, int, int]) -> int:
    return int((pixels(surface) == np.array(color, dtype=np.uint8)).all(axis=2).sum())


def test_the_model_label_reaches_the_hud_and_only_when_set(env: ArenaEnv) -> None:
    """In a checkpoint time-lapse the step count is the one thing telling 100k from 400k on camera,
    so it has to be pixels in the HUD — and human play, which has no network, must draw nothing."""
    width = ArenaRenderer(headless=True).surface_size[0]
    corner = (slice(0, 40), slice(width - 420, width))

    bare = ArenaRenderer(headless=True)
    labelled = ArenaRenderer(headless=True)
    labelled.model_label = "200,000 training steps"
    labelled.model_source = "ppo_direct_shipped_200000_steps.zip"

    without = pixels(bare.draw(env))[corner]
    with_label = pixels(labelled.draw(env))[corner]
    assert int((without != with_label).any(axis=2).sum()) > 200


def test_the_longest_model_label_clears_the_hud_fields() -> None:
    """A checkpoint name at HUD size once ran ~550px and printed over the STYLE field."""
    renderer = ArenaRenderer(headless=True)
    right = renderer.surface_size[0] - 14
    fields_end = 14 + 5 * 132 + renderer.font_hud.size("rotation")[0]
    longest_source = "ppo_rotation_mechanics_2000000_steps.zip"
    assert right - renderer.font_hud.size("2,000,000 training steps")[0] > fields_end
    assert right - renderer.font_small.size(longest_source)[0] > fields_end


class RecordingFont:
    """Wraps a `pygame.font.Font` and records every string it is asked to render.

    A proxy rather than a monkeypatch because `Font.render` is a read-only attribute; everything
    the renderer calls on a font is delegated through `__getattr__`.
    """

    def __init__(self, font: pygame.font.Font) -> None:
        self.font = font
        self.drawn: list[str] = []

    def render(self, text: str, *args: Any, **kwargs: Any) -> pygame.Surface:
        self.drawn.append(text)
        return self.font.render(text, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.font, name)


# --- surface and configuration ------------------------------------------------------------------


def test_headless_renderer_opens_no_display(env, renderer):
    renderer.draw(env)
    assert not pygame.display.get_init()


def test_surface_is_the_arena_plus_hud_and_panel(renderer):
    width, height = renderer.surface_size
    assert width == ARENA_WIDTH + renderer.OVERLAY_PANEL_WIDTH
    assert height == ARENA_HEIGHT + renderer.HUD_HEIGHT


def test_playfield_coordinates_sit_below_the_hud(renderer):
    assert renderer.to_screen(0.0, 0.0) == (0, renderer.HUD_HEIGHT)
    assert renderer.to_screen(10.0, 20.0) == (10, 20 + renderer.HUD_HEIGHT)


def test_render_config_comes_from_the_yaml():
    from common.config import load_yaml

    block = load_yaml("arena")["evaluation"]
    config = ArenaRenderConfig.from_yaml()
    assert config.fps == block["fps"]
    assert config.show_observation_overlay == block["show_observation_overlay"]


# --- entities reach the screen --------------------------------------------------------------------


def test_every_entity_type_is_drawn_in_its_own_colour(env, renderer):
    env.enemies.append(Enemy(env.player.x + 90.0, env.player.y, health=1, speed=90.0))
    env.step(int(DirectAction.SHOOT))
    surface = renderer.draw(env)

    assert count_color(surface, COLOR_PLAYER) > 0, "ship missing"
    assert count_color(surface, COLOR_ENEMY) > 0, "enemy missing"
    assert count_color(surface, COLOR_SPAWNER) > 0, "spawner missing"
    assert count_color(surface, COLOR_BULLET) > 0, "bullet missing"


def test_the_ship_is_a_triangle_pointing_along_its_heading(renderer, env):
    env.player.heading = 0.0
    points = renderer._ship_points(env.player)
    assert len(points) == 3
    nose = points[0]
    # The nose leads on +x when the heading is 0, and reaches further than the rear corners do.
    assert nose[0] > points[1][0] and nose[0] > points[2][0]


def test_a_destroyed_player_is_not_drawn(env, renderer):
    baseline = count_color(renderer.draw(env), COLOR_PLAYER)
    assert baseline > 0
    env.player.kill()
    surface = renderer.draw(env)
    # The canopy and the flame are drawn by the same method as the hull, so a wreck that kept
    # either of them would leave a cockpit light burning over an empty arena.
    assert count_color(surface, COLOR_PLAYER) == 0
    assert count_color(surface, COLOR_PLAYER_CANOPY) == 0
    assert count_color(surface, COLOR_ENGINE_CORE) == 0


# --- character detail: the parts that are read from live entity state ---------------------------


def test_the_thruster_flame_burns_only_while_the_ship_is_moving(env, renderer):
    """The flame is driven by `player.speed`, not by "was thrust pressed" — so a ship coasting to
    a stop cools down, and a stationary ship is unmistakably stationary."""
    assert env.player.speed == 0.0
    assert count_color(renderer.draw(env), COLOR_ENGINE_CORE) == 0

    for _ in range(4):
        env.step(int(DirectAction.RIGHT))
    assert env.player.speed > 0.0
    assert count_color(renderer.draw(env), COLOR_ENGINE_CORE) > 0


def test_the_enemy_eye_tracks_the_direction_it_is_travelling(env):
    """The eye is what makes a grunt read as a creature rather than a disc, and it is only worth
    drawing if it points somewhere true: the direction the enemy is actually moving."""
    # Clear of the player, which spawns centre-arena and is drawn on top of anything under it,
    # and with the overlay off so its target line cannot clip the pixels being counted.
    enemy = Enemy(env.player.x - 200.0, env.player.y - 120.0, health=1, speed=90.0)
    enemy.vx, enemy.vy = 90.0, 0.0  # travelling due right
    env.enemies = [enemy]
    renderer = ArenaRenderer(
        headless=True, config=ArenaRenderConfig(show_observation_overlay=False)
    )

    frame = pixels(renderer.draw(env))
    ys, xs = np.nonzero((frame == np.array(COLOR_ENEMY_EYE, dtype=np.uint8)).all(axis=2))
    assert xs.size > 0, "eye missing"
    assert xs.mean() > enemy.x, "eye should sit on the leading side of the body"
    assert abs(ys.mean() - (enemy.y + renderer.HUD_HEIGHT)) < enemy.radius


def test_only_a_wounded_grunt_wears_a_health_bar(env, renderer):
    """A full bar over every one-hit enemy in a swarm of forty is noise, and it buries the bars
    that do carry information — so the bar appears when, and only when, the grunt has been hit."""
    enemy = Enemy(env.player.x - 200.0, env.player.y - 120.0, health=2, speed=90.0)
    env.enemies = [enemy]
    unwounded = count_color(renderer.draw(env), COLOR_HEALTH)

    enemy.take_damage(1)
    assert count_color(renderer.draw(env), COLOR_HEALTH) > unwounded


def test_the_starfield_is_fixed_and_private_to_the_renderer():
    """Set dressing must not cost reproducibility: the stars come from a private generator, so
    building a renderer cannot shift the global stream a seeded episode draws from."""
    import random

    random.seed(1234)
    expected = [random.random() for _ in range(3)]

    random.seed(1234)
    first = ArenaRenderer(headless=True)
    second = ArenaRenderer(headless=True)
    assert [random.random() for _ in range(3)] == expected
    assert first._stars == second._stars == _make_starfield(ARENA_WIDTH, ARENA_HEIGHT)


# --- overlays and the HUD ---------------------------------------------------------------------------


def test_observation_overlay_toggles(env, renderer):
    renderer.show_observation_overlay = True
    with_overlay = count_color(renderer.draw(env), COLOR_OVERLAY_HEADING)
    assert renderer.toggle_observation_overlay() is False
    without = count_color(renderer.draw(env), COLOR_OVERLAY_HEADING)
    assert with_overlay > 0 and without == 0
    assert renderer.toggle_observation_overlay() is True


def test_the_overlay_panel_is_driven_by_describe(env, renderer):
    """Labels are read from `observation.describe()`, so they cannot drift from the layout."""
    renderer.show_observation_overlay = True
    font = RecordingFont(renderer.font_small)
    renderer.font_small = font
    renderer.draw(env)
    assert set(describe()).issubset(set(font.drawn))


def test_hud_reports_the_quantities_the_rubric_asks_to_see(env, renderer):
    env.step(int(DirectAction.RIGHT))
    renderer.font_small = RecordingFont(renderer.font_small)
    renderer.font_hud = RecordingFont(renderer.font_hud)
    renderer.draw(env)

    drawn = renderer.font_small.drawn + renderer.font_hud.drawn
    for label in ("PHASE", "HEALTH", "SCORE", "STEP", "ACTION"):
        assert label in drawn
    assert env.action_name in drawn
    assert str(env.steps) in drawn


def test_the_hud_prints_the_phase_number_the_env_is_actually_in(env, renderer):
    """The value, not just the label. `env.phase` is 1-based, so a renderer that adds one prints
    PHASE 2 over the first phase and PHASE 3 on the first banner -- wrong in exactly the
    phase-progression shot the video rubric asks for, and invisible to a test that only checks the
    word "PHASE" was drawn somewhere."""
    assert env.phase == 1
    renderer.font_hud = RecordingFont(renderer.font_hud)
    renderer.draw(env)
    assert "1" in renderer.font_hud.drawn
    assert "2" not in renderer.font_hud.drawn

    for spawner in env.spawners:
        spawner.take_damage(spawner.health)
    env.step(int(DirectAction.NOOP))
    assert env.phase == 2

    renderer.font_hud = RecordingFont(renderer.font_hud)
    renderer.font_banner = RecordingFont(renderer.font_banner)
    renderer.draw(env)
    assert "2" in renderer.font_hud.drawn
    assert "PHASE 2" in renderer.font_banner.drawn, renderer.font_banner.drawn


def test_the_game_over_banner_shows_on_death_but_not_on_survival(env, renderer):
    """No countdown to latch here, unlike the phase banner: `player.alive` stays False for the
    rest of the episode, so the banner should hold for every frame death draws, not just one."""
    renderer.font_banner = RecordingFont(renderer.font_banner)
    renderer.draw(env)
    assert "GAME OVER" not in renderer.font_banner.drawn

    env.player.kill()
    renderer.font_banner = RecordingFont(renderer.font_banner)
    renderer.draw(env)
    assert "GAME OVER" in renderer.font_banner.drawn

    renderer.font_banner = RecordingFont(renderer.font_banner)
    for _ in range(5):
        renderer.draw(env)
    assert "GAME OVER" in renderer.font_banner.drawn, "should not need a per-frame trigger"


def test_phase_banner_latches_for_its_own_countdown(env, renderer):
    """`phase_just_advanced` is true for one step; the banner must outlive it on screen."""
    for spawner in env.spawners:
        spawner.take_damage(spawner.health)
    env.step(int(DirectAction.NOOP))
    assert env.phase_just_advanced

    renderer.draw(env)
    assert renderer._banner_remaining == pytest.approx(BANNER_SECONDS)

    env.step(int(DirectAction.NOOP))
    assert not env.phase_just_advanced
    renderer.draw(env)
    assert 0.0 < renderer._banner_remaining < BANNER_SECONDS


# --- the load-bearing guarantee ----------------------------------------------------------------------


def test_drawing_never_mutates_the_simulation(env, renderer):
    """Every effect in the renderer is cosmetic, or evaluation stops matching training."""
    env.enemies.append(Enemy(env.player.x + 60.0, env.player.y, health=2, speed=90.0))
    env.step(int(DirectAction.SHOOT))

    def snapshot() -> tuple:
        return (
            env.player.x, env.player.y, env.player.vx, env.player.vy,
            env.player.heading, env.player.health, env.player.shoot_cooldown_remaining,
            env.player.invulnerable_remaining,
            tuple((e.x, e.y, e.health) for e in env.enemies),
            tuple((s.x, s.y, s.health, s.spawn_timer) for s in env.spawners),
            tuple((b.x, b.y, b.lifetime_remaining) for b in env.bullets),
            env.phase, env.steps, env.enemies_killed, env.spawners_destroyed,
            env.damage_taken, env.episode_return,
        )

    before = snapshot()
    for _ in range(10):
        renderer.draw(env)
    assert snapshot() == before


def test_the_last_frame_is_held_so_the_death_explosion_can_play(env):
    """A death drawn once is a death shown for a sixtieth of a second: on the video the ship
    simply blinks out. The hold redraws the finished episode without stepping it."""
    windowed = ArenaRenderer(headless=False)  # SDL_VIDEODRIVER=dummy: a window without a screen
    try:
        env.player.kill()
        before = (env.steps, env.player.health, len(env.enemies))
        assert windowed.hold(env, seconds=0.05) >= 2
        assert (env.steps, env.player.health, len(env.enemies)) == before
    finally:
        windowed.close()


def test_a_headless_run_holds_nothing(env, renderer):
    """Headless callers — the eval tables, the figure capture — count their frames, so the hold
    must not quietly add any."""
    assert renderer.hold(env) == 0


def test_a_full_episode_renders_without_error(env, renderer):
    """Exercises every branch a real playback hits: spawns, kills, damage, phase change, death."""
    for _ in range(400):
        _obs, _r, terminated, truncated, _info = env.step(env.action_space.sample())
        renderer.draw(env)
        if terminated or truncated:
            break
    renderer.close()
