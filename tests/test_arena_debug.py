"""Reward accounting and the F3 debug panel, ported from A3-030 onto the current renderer.

A3-030's own commit also introduced a pause/single-step/reset control layer (`ArenaPlayback` in
`eval/play_arena.py`, `P`/`N`/`R` keys, `renderer.handle_event`) bundled into the same diff as the
debug panel. That layer is deliberately NOT ported here: it would need its own design against
`wait_for_start()` and the title screen, neither of which existed when it was written, and forcing
it through today would be a second full rewrite rather than a gentle one. This file therefore
covers the two things that were ported -- the reward-accounting instrumentation in `arena/env.py`
and the F3 panel in `arena/render.py` -- reusing A3-030's own reward-accounting tests, which had no
dependency on the descoped control layer, and adding fresh pixel-based tests for the panel itself.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from arena import constants as C
from arena.debug import REWARD_TERMS
from arena.entities import Bullet, Enemy, Spawner
from arena.env import ArenaEnv
from arena.render import ArenaRenderer


def check_accounting(env: ArenaEnv, returned: float) -> dict[str, float]:
    """The snapshot's own arithmetic has to agree with what `step()` actually returned."""
    snapshot = env.latest_reward
    assert sum(dict(snapshot.contributions).values()) == returned
    assert snapshot.reward == returned
    assert sum(dict(snapshot.totals).values()) == pytest.approx(env.episode_reward)
    assert snapshot.step == env.steps
    return dict(snapshot.contributions)


# --- reward accounting, ported from A3-030 -------------------------------------------------------


@pytest.mark.parametrize("style", C.CONTROL_STYLES)
def test_actual_enemy_spawner_phase_and_step_rewards(style):
    env = ArenaEnv(style, seed=0)
    env.enemies = [Enemy(100, 100, health=1, speed=0)]
    env.spawners = [Spawner(700, 500, env.phase_settings)]
    env.spawners[0].health = 1
    # Stationary bullets put each actual collision site on this agent step.
    env.bullets = [Bullet(100, 100, 0, speed=0), Bullet(700, 500, 0, speed=0)]
    for bullet in env.bullets:
        bullet.speed = 0
    _, reward, _, _, _ = env.step(0)
    terms = check_accounting(env, reward)
    assert terms == dict(step=C.REWARD_PER_STEP, enemy=C.REWARD_ENEMY_DESTROYED,
                         spawner=C.REWARD_SPAWNER_DESTROYED, phase=C.REWARD_PHASE_ADVANCE,
                         damage=0, death=0)
    previous = env.latest_reward
    _, reward, _, _, _ = env.step(0)
    terms = check_accounting(env, reward)
    assert terms["step"] == C.REWARD_PER_STEP  # once, not ACTION_REPEAT times
    assert terms["enemy"] == terms["spawner"] == terms["phase"] == 0
    assert dict(previous.contributions)["phase"] == C.REWARD_PHASE_ADVANCE


@pytest.mark.parametrize("style", C.CONTROL_STYLES)
@pytest.mark.parametrize("fatal", [False, True])
def test_damage_is_per_accepted_event_not_health_point_or_frame(style, fatal):
    env = ArenaEnv(style, seed=0)
    env.player.health = 2 if fatal else env.player.max_health
    enemy = Enemy(env.player.x, env.player.y, health=5, speed=0)
    enemy.contact_damage = 2
    env.enemies = [enemy]
    _, reward, terminated, _, _ = env.step(0)
    terms = check_accounting(env, reward)
    assert terms["damage"] == C.REWARD_DAMAGE_TAKEN
    assert terms["death"] == (C.REWARD_DEATH if fatal else 0)
    assert terminated == fatal
    if not fatal:
        _, reward, _, _, _ = env.step(0)
        assert check_accounting(env, reward)["damage"] == 0


def test_a_reward_term_that_stopped_recording_would_be_dropped_not_lost_silently():
    """The property the mutation check in this project's commit history relies on: `step()`'s
    returned reward is the canonical sum of what `_record_reward` actually saw, not a locally
    accumulated total -- so a site that forgets to record does not double-count, it disappears
    from both the return value and the snapshot together, which is what makes it visible on the
    debug panel instead of silently inflating the score."""
    env = ArenaEnv("direct", seed=0)
    env.enemies = [Enemy(100, 100, health=1, speed=0)]
    env.bullets = [Bullet(100, 100, 0, speed=0)]
    _, reward, _, _, _ = env.step(0)
    contributions = dict(env.latest_reward.contributions)
    assert contributions["enemy"] == C.REWARD_ENEMY_DESTROYED
    assert reward == pytest.approx(sum(contributions.values()))


# --- the F3 panel -----------------------------------------------------------------------------


@pytest.fixture
def env() -> ArenaEnv:
    world = ArenaEnv("direct", seed=0)
    world.reset(seed=0)
    return world


@pytest.fixture
def renderer() -> ArenaRenderer:
    return ArenaRenderer(headless=True)


def pixels(surface: pygame.Surface) -> np.ndarray:
    return pygame.surfarray.array3d(surface).transpose(1, 0, 2)


def test_the_panel_is_absent_until_f3_is_toggled(env, renderer):
    before = renderer.surface_size
    surface = renderer.draw(env)
    assert surface.get_size() == before

    renderer.toggle_debug()
    after = renderer.surface_size
    assert after[0] > before[0], "the debug column should widen the window once shown"
    surface = renderer.draw(env)
    assert surface.get_size() == after


def test_toggle_debug_flips_and_reports_its_new_state(renderer):
    """Matches this file's existing convention for O/E/V: the toggle method is the unit under
    test, not the keyboard plumbing that calls it -- see `test_arena_render.py`."""
    assert renderer.show_debug is False
    assert renderer.toggle_debug() is True
    assert renderer.show_debug is True
    assert renderer.toggle_debug() is False


def test_the_panel_shows_no_transition_before_the_first_step(env, renderer):
    renderer.toggle_debug()
    renderer.font_small = _RecordingFont(renderer.font_small)
    renderer.draw(env)
    assert any("No transition yet" in text for text in renderer.font_small.drawn)


def test_the_panel_reports_the_transition_that_actually_happened(env, renderer):
    renderer.toggle_debug()
    env.step(int(C.DirectAction.SHOOT))
    renderer.font_small = _RecordingFont(renderer.font_small)
    renderer.draw(env)
    drawn = " ".join(renderer.font_small.drawn)
    assert "SHOOT" in drawn
    assert f"Step {env.steps}" in drawn
    for label, _weight in REWARD_TERMS.values():
        assert label in drawn


def test_physics_overlay_draws_a_collision_ring_around_every_entity(env, renderer):
    env.enemies = [Enemy(env.player.x - 150.0, env.player.y, health=1, speed=0)]
    renderer.toggle_debug()
    surface = renderer.draw(env)
    from arena.render import COLOR_OVERLAY_HEADING

    array = pixels(surface)
    matches = (array == np.array(COLOR_OVERLAY_HEADING, dtype=np.uint8)).all(axis=-1)
    assert matches.sum() > 0, "no collision-radius rings drawn"


def test_toggling_debug_off_removes_the_panel_and_its_column(env, renderer):
    renderer.toggle_debug()
    assert renderer.surface_size[0] > ArenaRenderer(headless=True).surface_size[0]
    renderer.toggle_debug()
    assert renderer.surface_size == ArenaRenderer(headless=True).surface_size


class _RecordingFont:
    """Wraps a `pygame.font.Font` and records every string it is asked to render."""

    def __init__(self, font: pygame.font.Font) -> None:
        self.font = font
        self.drawn: list[str] = []

    def render(self, text: str, *args, **kwargs):
        self.drawn.append(text)
        return self.font.render(text, *args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.font, name)
