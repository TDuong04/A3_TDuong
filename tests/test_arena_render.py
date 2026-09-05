"""A3-021: actual entity pixels, display ownership, and simulation isolation."""
from __future__ import annotations

import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pygame
import pytest

from arena.constants import ARENA_HEIGHT, ARENA_WIDTH, CONTROL_STYLES
from arena.entities import Bullet, Enemy, Spawner
from arena.env import ArenaEnv
from arena.render import (
    COLOR_BACKGROUND,
    COLOR_BULLET,
    COLOR_ENEMY,
    COLOR_PLAYER,
    COLOR_PLAYER_DEAD,
    COLOR_SPAWNER,
    ArenaRenderer,
    scripted_action,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scene():
    env = ArenaEnv(seed=0)
    env.player.x, env.player.y, env.player.heading = 480.0, 340.0, 0.0
    env.enemies = [Enemy(200, 200, health=1, speed=90)]
    env.spawners = [Spawner(700, 450, env.phase_settings)]
    env.bullets = [Bullet(350, 200, heading=0, speed=520)]
    return env


def rgb(surface, position):
    return tuple(surface.get_at(position)[:3])


def test_entities_have_distinct_shapes_at_their_world_positions(scene):
    renderer = ArenaRenderer(headless=True, show_observation_overlay=False, effects=False)
    surface = renderer.draw(scene)
    assert surface.get_size() == (ARENA_WIDTH, ARENA_HEIGHT)
    assert rgb(surface, (480, 340)) == COLOR_PLAYER
    assert rgb(surface, (200, 200)) == COLOR_ENEMY
    assert rgb(surface, (700, 450)) == COLOR_SPAWNER
    assert rgb(surface, (350, 200)) == COLOR_BULLET
    assert rgb(surface, (100, 100)) == COLOR_BACKGROUND
    # Circle corners stay empty, whereas square corners are filled.
    assert rgb(surface, (210, 210)) == COLOR_BACKGROUND
    assert rgb(surface, (715, 465)) == COLOR_SPAWNER
    assert rgb(surface, (344, 200)) == COLOR_BULLET
    assert rgb(surface, (350, 206)) == COLOR_BACKGROUND
    renderer.close()


@pytest.mark.parametrize('heading, nose, behind', [
    (0, (492, 340), (468, 340)),
    (math.pi / 2, (480, 352), (480, 328)),
    (math.pi, (468, 340), (492, 340)),
    (-math.pi / 2, (480, 328), (480, 352)),
    (math.pi / 4, (488, 348), (472, 332)),
])
def test_ship_nose_follows_heading(scene, heading, nose, behind):
    scene.player.heading = heading
    renderer = ArenaRenderer(headless=True, show_observation_overlay=False, effects=False)
    surface = renderer.draw(scene)
    assert rgb(surface, nose) == COLOR_PLAYER
    assert rgb(surface, behind) == COLOR_BACKGROUND
    renderer.close()


def test_new_frame_clears_old_positions_and_dead_entities(scene):
    renderer = ArenaRenderer(headless=True, show_observation_overlay=False, effects=False)
    renderer.draw(scene)
    scene.enemies[0].x = 250
    scene.spawners[0].kill()
    scene.bullets[0].kill()
    scene.player.kill()
    surface = renderer.draw(scene)
    assert rgb(surface, (200, 200)) == COLOR_BACKGROUND
    assert rgb(surface, (250, 200)) == COLOR_ENEMY
    assert rgb(surface, (700, 450)) == COLOR_BACKGROUND
    assert rgb(surface, (350, 200)) == COLOR_BACKGROUND
    assert rgb(surface, (480, 340)) == COLOR_PLAYER_DEAD
    scene.enemies.clear()
    scene.spawners.clear()
    scene.bullets.clear()
    assert rgb(renderer.draw(scene), (250, 200)) == COLOR_BACKGROUND
    renderer.close()


def test_bullet_heading_changes_its_visible_direction(scene):
    renderer = ArenaRenderer(headless=True, show_observation_overlay=False, effects=False)
    scene.bullets[0].heading = math.pi / 2
    surface = renderer.draw(scene)
    assert rgb(surface, (350, 194)) == COLOR_BULLET
    assert rgb(surface, (344, 200)) == COLOR_BACKGROUND
    renderer.close()


def test_drawing_preserves_all_environment_entity_and_rng_state(scene):
    renderer = ArenaRenderer(headless=True, show_observation_overlay=False, effects=False)
    before = pickle.dumps(scene.__dict__)
    for _ in range(5):
        renderer.draw(scene)
    assert pickle.dumps(scene.__dict__) == before
    renderer.close()


@pytest.mark.parametrize('style', CONTROL_STYLES)
def test_rendering_does_not_change_a_seeded_trajectory(style):
    env = ArenaEnv(control_style=style, seed=7)
    reference = ArenaEnv(control_style=style, seed=7)
    renderer = ArenaRenderer(headless=True, show_observation_overlay=False, effects=False)
    for step in range(180):
        action = step % int(env.action_space.n)
        actual = env.step(action)
        expected = reference.step(action)
        renderer.draw(env)
        renderer.draw(env)
        assert np.array_equal(actual[0], expected[0])
        assert actual[1:] == expected[1:]
        assert pickle.dumps(env.__dict__) == pickle.dumps(reference.__dict__)
        if actual[2] or actual[3]:
            break
    renderer.close()


@pytest.mark.parametrize('style', CONTROL_STYLES)
def test_scripted_demo_exercises_movement_spawning_shooting_and_collisions(style):
    env = ArenaEnv(control_style=style, seed=0)
    start = env.player.position
    moved = spawned = fired = collided = False
    for _ in range(600):
        _, _, terminated, truncated, _ = env.step(scripted_action(env))
        moved |= env.player.position != start
        spawned |= any(s.enemies_spawned > 0 for s in env.spawners)
        fired |= bool(env.bullets) or env.enemies_killed > 0
        collided |= env.enemies_killed > 0 or env.damage_taken > 0
        if terminated or truncated:
            break
    assert moved and spawned and fired and collided


def test_display_lifecycle_and_headless_isolation_in_a_fresh_process():
    script = '''
import pygame
from arena.env import ArenaEnv
from arena.render import ArenaRenderer, COLOR_PLAYER

env = ArenaEnv(seed=0, render_mode='human')
assert not pygame.display.get_init()
offscreen = ArenaRenderer(headless=True, show_observation_overlay=False, effects=False)
offscreen.draw(env)
assert not pygame.display.get_init()
offscreen.close()
offscreen.close()
assert not pygame.display.get_init()

flips = []
original_flip = pygame.display.flip
def flip():
    flips.append(True)
    original_flip()
pygame.display.flip = flip
surface = env.render()
assert pygame.display.get_surface() is surface
assert tuple(surface.get_at((480, 340))[:3]) == COLOR_PLAYER
assert len(flips) == 1
for _ in range(3):
    env.step(0)
assert len(flips) == 1, 'step must not present or render'

second = ArenaRenderer()
try:
    second.draw(env)
except RuntimeError:
    pass
else:
    raise AssertionError('must not replace an existing window')
second.close()
offscreen.draw(env)
offscreen.close()
assert pygame.display.get_surface() is surface
assert len(flips) == 1
pygame.font.init()
env.close()
env.close()
assert not pygame.display.get_init()
assert pygame.font.get_init(), 'close must not quit unrelated subsystems'
assert env.render() is pygame.display.get_surface()
env.close()
pygame.font.quit()
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                            env={**os.environ, 'SDL_VIDEODRIVER': 'dummy'},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('style', CONTROL_STYLES)
def test_demo_cli_runs_offscreen(style):
    result = subprocess.run(
        [sys.executable, '-m', 'arena.render', '--style', style, '--seed', '0',
         '--headless', '--frames', '600'], cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert 'not a trained policy' in result.stdout
    assert 'Rendered 600 frames' in result.stdout
