"""A3-018 integration with the A3-009 renderer and existing evaluation application."""
import math
import pickle
import random

import numpy as np
import pygame
import pytest

from arena.constants import CONTROL_STYLES, DirectAction, RotationAction
from arena.entities import Enemy, Spawner
from arena.env import ArenaEnv
from arena.render import ArenaRenderer, ArenaRenderConfig
from arena.visuals import ENEMY_LINK, FLASH, CombatFeedback


def pixels(surface, color):
    return int(np.all(pygame.surfarray.array3d(surface) == color, axis=-1).sum())


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
    renderer = ArenaRenderer(headless=True, config=ArenaRenderConfig(show_observation_overlay=False))
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
    renderer.feedback.advance(0.13)
    mid = renderer.draw(env).copy()
    assert not np.array_equal(pygame.surfarray.array3d(frame), pygame.surfarray.array3d(mid))
    renderer.feedback.advance(2)
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


def test_an_effect_fades_by_alpha_rather_than_smearing_the_sprite_it_covers():
    """A fading effect must lighten toward whatever is beneath it, never toward black.

    Scaling the colour toward black reads as a fade only against the empty playfield. Drawn over
    the ship — which is exactly where a muzzle flash lands, every single shot — the same pixels
    paint a dirty grey wedge across the hull for the whole of the flash's life.
    """
    hull = (74, 148, 236)  # arena.render.COLOR_PLAYER: darker in green than FLASH is
    surface = pygame.Surface((120, 120))
    surface.fill(hull)
    feedback = CombatFeedback()
    feedback._emit('hit', (60, 60), 0.2)
    feedback.advance(0.1)  # half spent: washing out, not darkening
    feedback.draw(surface)

    green = pygame.surfarray.array3d(surface)[:, :, 1]
    assert green.min() >= hull[1], 'a fading effect darkened the sprite underneath it'
    assert green.max() > hull[1], 'the effect never reached the surface at all'
    assert pixels(surface, FLASH) == 0, 'a half-spent effect should not still be fully opaque'


def test_effect_capacity_and_disappearing_living_entities():
    scene = combat_scene("direct")
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
        renderer.feedback.advance(0.013)
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
@pytest.mark.parametrize('heading', [0.4, 1.9, -2.7])
def test_compass_alignment_and_toggle(style, heading):
    env = ArenaEnv(control_style=style, seed=11)
    env.player.heading = heading
    env.enemies = [Enemy(env.player.x + 120, env.player.y + 80, health=2, speed=0)]
    renderer = ArenaRenderer(headless=True, effects=False)
    values = renderer.perception.values(env)
    relative = complex(120, 80) * complex(math.cos(-heading), math.sin(-heading))
    assert values['enemy_local_dx'] == pytest.approx(relative.real / math.hypot(960, 680))
    assert values['enemy_local_dy'] == pytest.approx(relative.imag / math.hypot(960, 680))
    surface = renderer.draw(env)
    center = (70, surface.get_height() - 72)
    point = (round(center[0] + 40 * values['enemy_local_dx']),
             round(center[1] + 40 * values['enemy_local_dy']))
    assert tuple(surface.get_at(point)[:3]) == ENEMY_LINK
    renderer.toggle_observation_overlay()
    assert pixels(renderer.draw(env), ENEMY_LINK) == 0
    renderer.toggle_observation_overlay()
    assert pixels(renderer.draw(env), ENEMY_LINK) > 0
    env.enemies.clear()
    assert renderer.perception.values(env)['enemy_exists'] == 0
    renderer.draw(env)
    env.enemies = [Enemy(env.player.x, env.player.y, health=1, speed=0)]
    assert renderer.perception.values(env)['enemy_distance'] == 0
    renderer.draw(env)
    renderer.close()


def test_shake_preserves_hud_and_side_panel():
    env = combat_scene('direct')
    renderer = ArenaRenderer(headless=True)
    renderer.draw(env)
    env.step(DirectAction.SHOOT)
    shaken = renderer.draw(env).copy()
    renderer.feedback.shake_remaining = 0
    still = renderer.draw(env)
    a, b = pygame.surfarray.array3d(shaken), pygame.surfarray.array3d(still)
    assert np.array_equal(a[:, :renderer.HUD_HEIGHT], b[:, :renderer.HUD_HEIGHT])
    assert np.array_equal(a[960:], b[960:])
    assert not np.array_equal(a[:960, renderer.HUD_HEIGHT:], b[:960, renderer.HUD_HEIGHT:])
    renderer.close()
