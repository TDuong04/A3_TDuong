"""Perception, human input and purely visual combat feedback (A3-018)."""
from __future__ import annotations

import math
import pickle
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pygame
import pytest

from arena.constants import ACTION_REPEAT, CONTROL_STYLES, FIXED_DT, DirectAction, RotationAction
from arena.entities import Enemy, Spawner
from arena.env import ArenaEnv
from arena.play import ArenaApp, human_action
from arena.render import ArenaRenderer
from arena.visuals import ENEMY_LINK, FLASH, FORWARD, LATERAL, CombatFeedback, PerceptionOverlay
from common.config import load_yaml

ROOT = Path(__file__).resolve().parents[1]
INTERVAL = FIXED_DT * ACTION_REPEAT


def key(kind, code):
    return pygame.event.Event(kind, key=code)


def pixels(surface, color):
    return int(np.all(pygame.surfarray.array3d(surface) == color, axis=-1).sum())


@pytest.fixture
def scene():
    env = ArenaEnv(seed=11)
    env.player.x, env.player.y, env.player.heading = 480.0, 340.0, 0.7
    env.enemies = [Enemy(600, 420, health=2, speed=0)]
    env.spawners = [Spawner(740, 510, env.phase_settings)]
    return env


def test_overlay_default_and_toggle_really_change_pixels(scene):
    renderer = ArenaRenderer(headless=True, effects=False)
    assert renderer.show_observation_overlay == load_yaml('arena')['evaluation']['show_observation_overlay']
    renderer.show_observation_overlay = True
    on = renderer.draw(scene).copy()
    assert pixels(on, ENEMY_LINK) > 0
    assert renderer.toggle_observation_overlay() is False
    off = renderer.draw(scene).copy()
    assert pixels(off, ENEMY_LINK) == 0
    assert not np.array_equal(pygame.surfarray.array3d(on), pygame.surfarray.array3d(off))
    assert renderer.toggle_observation_overlay() is True
    assert np.array_equal(pygame.surfarray.array3d(on), pygame.surfarray.array3d(renderer.draw(scene)))


@pytest.mark.parametrize('heading', [0.4, 1.9, -2.7])
def test_compass_uses_actual_local_observation_and_world_axes(scene, heading):
    scene.player.heading = heading
    renderer = ArenaRenderer(headless=True, effects=False)
    values = renderer.overlay.values(scene)
    # Independently rotate an off-axis displacement with complex multiplication.
    relative = complex(120, 80) * complex(math.cos(-heading), math.sin(-heading))
    diagonal = math.hypot(960, 680)
    assert values['enemy_local_dx'] == pytest.approx(relative.real / diagonal)
    assert values['enemy_local_dy'] == pytest.approx(relative.imag / diagonal)
    frame = renderer.draw(scene)
    marker = (round(90 + 56 * values['enemy_local_dx']),
              round(136 + 56 * values['enemy_local_dy']))
    assert tuple(frame.get_at(marker)[:3]) == ENEMY_LINK
    forward, lateral = PerceptionOverlay.axes(scene.player)
    assert forward == pytest.approx((480 + 48 * math.cos(heading), 340 + 48 * math.sin(heading)))
    assert lateral == pytest.approx((480 - 48 * math.sin(heading), 340 + 48 * math.cos(heading)))
    assert pixels(frame, FORWARD) > 0 and pixels(frame, LATERAL) > 0


def test_overlay_clears_absent_slots_and_ignores_dead_targets(scene):
    renderer = ArenaRenderer(headless=True, effects=False)
    renderer.draw(scene)
    scene.enemies[0].kill()
    assert renderer.overlay.values(scene)['enemy_exists'] == 0
    assert pixels(renderer.draw(scene), ENEMY_LINK) > 0  # 'none' label remains, not a target marker
    raw = pygame.Surface((960, 680))
    renderer.overlay.draw_world(raw, scene)
    assert pixels(raw, ENEMY_LINK) == 0
    scene.enemies = [Enemy(480, 340, health=1, speed=0)]
    values = renderer.overlay.values(scene)
    assert values['enemy_distance'] == values['enemy_local_dx'] == values['enemy_local_dy'] == 0
    renderer.draw(scene)  # a coincident target must not divide by its distance


def test_links_select_nearest_living_entity(scene):
    scene.enemies.insert(0, Enemy(800, 600, health=1, speed=0))
    dead = Enemy(485, 345, health=1, speed=0)
    dead.kill()
    scene.enemies.insert(0, dead)
    canvas = pygame.Surface((960, 680))
    PerceptionOverlay().draw_world(canvas, scene)
    assert tuple(canvas.get_at((540, 380))[:3]) == ENEMY_LINK
    assert tuple(canvas.get_at((800, 600))[:3]) != ENEMY_LINK


@pytest.mark.parametrize('style', CONTROL_STYLES)
def test_human_key_mapping_and_one_action_priority(style):
    assert human_action(style, set()) == 0
    assert human_action(style, {pygame.K_SPACE, pygame.K_w}) == (
        RotationAction.SHOOT if style == 'rotation' else DirectAction.SHOOT)
    if style == 'rotation':
        expected = [(pygame.K_w, RotationAction.THRUST), (pygame.K_LEFT, RotationAction.ROTATE_LEFT),
                    (pygame.K_d, RotationAction.ROTATE_RIGHT)]
        assert human_action(style, {pygame.K_s}) == 0
    else:
        expected = [(pygame.K_w, DirectAction.UP), (pygame.K_DOWN, DirectAction.DOWN),
                    (pygame.K_a, DirectAction.LEFT), (pygame.K_RIGHT, DirectAction.RIGHT)]
    for code, action in expected:
        assert human_action(style, {code}) == action


@pytest.mark.parametrize('style', CONTROL_STYLES)
def test_human_app_matches_training_environment_for_the_same_actions(style):
    app = ArenaApp(style=style, seed=5, headless=True)
    reference = ArenaEnv(control_style=style, seed=5)
    assert type(app.env) is type(reference)
    for code in (pygame.K_w, pygame.K_a, pygame.K_d, pygame.K_SPACE):
        app.handle_event(key(pygame.KEYDOWN, code))
        action = human_action(style, {code})
        for _ in range(8):
            app.update(INTERVAL)
            reference.step(action)
            app.renderer.draw(app.env)
            assert pickle.dumps(app.env.__dict__) == pickle.dumps(reference.__dict__)
        app.handle_event(key(pygame.KEYUP, code))
    app.close()


def test_input_pause_focus_reset_and_toggles():
    app = ArenaApp(headless=True)
    app.handle_event(key(pygame.KEYDOWN, pygame.K_w))
    app.update(INTERVAL)
    assert app.env.steps == 1
    app.handle_event(key(pygame.KEYDOWN, pygame.K_p))
    app.update(1)
    assert app.env.steps == 1
    app.handle_event(key(pygame.KEYUP, pygame.K_p))
    app.handle_event(key(pygame.KEYDOWN, pygame.K_p))
    app.handle_event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
    assert not app.held and app.paused
    overlay = app.renderer.show_observation_overlay
    app.handle_event(key(pygame.KEYDOWN, pygame.K_o))
    app.handle_event(key(pygame.KEYDOWN, pygame.K_o))  # repeated key must not toggle again
    assert app.renderer.show_observation_overlay is not overlay
    app.handle_event(key(pygame.KEYDOWN, pygame.K_e))
    assert not app.renderer.effects_enabled
    app.handle_event(key(pygame.KEYDOWN, pygame.K_r))
    assert app.env.steps == 0 and not app.done and not app.renderer.feedback.events
    app.handle_event(key(pygame.KEYDOWN, pygame.K_ESCAPE))
    assert not app.running
    app.close()


def combat_scene(style):
    env = ArenaEnv(control_style=style, seed=0)
    env.player.heading = 0
    # First enemy is hit by a projectile within one agent step; second causes contact damage.
    env.enemies = [Enemy(env.player.x + 42, env.player.y, health=1, speed=0),
                   Enemy(env.player.x, env.player.y, health=2, speed=0)]
    return env


@pytest.mark.parametrize('style', CONTROL_STYLES)
def test_real_combat_creates_bounded_transient_feedback_without_retrigger(style):
    env = combat_scene(style)
    renderer = ArenaRenderer(headless=True, show_observation_overlay=False)
    renderer.draw(env)
    assert renderer.feedback.events == []
    env.step(RotationAction.SHOOT if style == 'rotation' else DirectAction.SHOOT)
    before = pickle.dumps(env.__dict__)
    frame = renderer.draw(env).copy()
    kinds = [event.kind for event in renderer.feedback.events]
    assert {'muzzle', 'hit', 'explosion'} <= set(kinds)
    assert renderer.feedback.offset != (0, 0)
    assert pixels(frame, FLASH) > 0
    assert env.enemies_killed == 1 and env.damage_taken == 1
    for _ in range(5):
        renderer.draw(env)
    assert [e.kind for e in renderer.feedback.events] == kinds
    assert pickle.dumps(env.__dict__) == before
    renderer.advance(0.13)
    mid = renderer.draw(env).copy()
    assert not np.array_equal(pygame.surfarray.array3d(frame), pygame.surfarray.array3d(mid))
    renderer.advance(2)
    assert renderer.feedback.events == [] and renderer.feedback.offset == (0, 0)
    assert pickle.dumps(env.__dict__) == before
    renderer.close()


def test_effects_clear_on_reset_replacement_and_toggle():
    env = combat_scene('direct')
    renderer = ArenaRenderer(headless=True)
    renderer.draw(env)
    env.step(DirectAction.SHOOT)
    renderer.draw(env)
    assert renderer.feedback.events
    env.reset(seed=0)
    renderer.draw(env)
    assert not renderer.feedback.events and renderer.feedback.offset == (0, 0)
    renderer.draw(combat_scene('rotation'))
    assert not renderer.feedback.events
    renderer.toggle_effects()
    assert not renderer.effects_enabled and not renderer.feedback.events
    renderer.close()


def test_effect_capacity_and_disappearing_living_entities(scene):
    feedback = CombatFeedback()
    feedback.observe(scene)
    scene.enemies.clear()
    scene.steps += 1
    feedback.observe(scene)
    assert not feedback.events, 'phase cleanup is not a kill'
    for _ in range(100):
        feedback._emit('hit', (20, 20), 0.2)
    assert len(feedback.events) == feedback.config.max_events
    feedback.advance(1)
    assert not feedback.events


@pytest.mark.parametrize('style', CONTROL_STYLES)
def test_effects_and_overlay_preserve_world_and_global_random_streams(style):
    env = combat_scene(style)
    reference = combat_scene(style)
    renderer = ArenaRenderer(headless=True)
    renderer.draw(env)
    python_rng = random.getstate()
    numpy_rng = pickle.dumps(np.random.get_state())
    for step in range(35):
        action = int(RotationAction.SHOOT if style == 'rotation' else DirectAction.SHOOT)
        actual, expected = env.step(action), reference.step(action)
        renderer.observe(env)
        renderer.advance(0.013)
        renderer.draw(env)
        renderer.draw(env)
        if step % 7 == 0:
            renderer.toggle_observation_overlay()
            renderer.toggle_effects()
        assert np.array_equal(actual[0], expected[0])
        assert actual[1:] == expected[1:]
        assert pickle.dumps(env.__dict__) == pickle.dumps(reference.__dict__)
        if actual[2] or actual[3]:
            break
    assert random.getstate() == python_rng
    assert pickle.dumps(np.random.get_state()) == numpy_rng
    renderer.close()


@pytest.mark.parametrize('style', CONTROL_STYLES)
def test_human_eval_cli_and_no_implicit_trained_policy(style):
    result = subprocess.run([sys.executable, '-m', 'eval.play_arena', '--human', '--style', style,
                             '--headless', '--frames', '10'], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'Human play on the training environment' in result.stdout
    assert 'O: perception' in result.stdout
    result = subprocess.run([sys.executable, '-m', 'eval.play_arena', '--style', style], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and 'A3-012' in result.stderr
