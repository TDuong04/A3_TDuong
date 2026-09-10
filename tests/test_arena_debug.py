"""A3-030: real reward accounting, aligned snapshots and observational debugging."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pygame
import pytest
import torch

from arena import constants as C
from arena.debug import REWARD_TERMS
from arena.entities import Bullet, Enemy, Spawner
from arena.env import ArenaEnv
from arena.policy_view import probe
from arena.render import ArenaRenderer
from eval.play_arena import ArenaPlayback, RandomPolicy, load_agent, load_policy, play


def make_app(style="direct", *, policy="random", agent=None, max_steps=15):
    env = ArenaEnv(style, seed=9, max_episode_steps=max_steps)
    renderer = ArenaRenderer(headless=True)
    renderer.show_debug = True
    if agent is None and policy == "random":
        agent = RandomPolicy(C.N_ACTIONS[style], 5)
    return ArenaPlayback(env, renderer, agent, policy)


def key(app, value):
    app.renderer.handle_event(pygame.event.Event(pygame.KEYDOWN, key=value))


def check_accounting(env, returned):
    snapshot = env.latest_reward
    assert sum(dict(snapshot.contributions).values()) == returned
    assert snapshot.reward == returned
    assert sum(dict(snapshot.totals).values()) == pytest.approx(env.episode_reward)
    assert snapshot.step == env.steps
    return dict(snapshot.contributions)


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


@pytest.mark.parametrize("style", C.CONTROL_STYLES)
def test_random_snapshot_pause_repeated_draw_and_reset(style):
    app = make_app(style)
    key(app, pygame.K_p)
    before = pickle.dumps(app.agent._rng.bit_generator.state)
    assert not app.advance()
    assert app.snapshot is None
    key(app, pygame.K_n)
    observation = app.env.last_observation.copy()
    assert app.advance()
    snapshot = app.snapshot
    np.testing.assert_array_equal(snapshot.observation, observation)
    np.testing.assert_array_equal(snapshot.next_observation, app.env.last_observation)
    assert snapshot.reward.action == app.env.last_action
    assert snapshot.policy is None and "RANDOM" in snapshot.policy_status
    assert not app.advance()
    rng_after_step = pickle.dumps(app.agent._rng.bit_generator.state)
    assert rng_after_step != before
    env_after_step = pickle.dumps(app.env.__dict__)
    for _ in range(5):
        app.draw()
        key(app, pygame.K_F3)
        app.draw()
    assert app.snapshot is snapshot
    assert pickle.dumps(app.env.__dict__) == env_after_step
    assert pickle.dumps(app.agent._rng.bit_generator.state) == rng_after_step
    with pytest.raises(FrozenInstanceError):
        snapshot.reward.reward = 123
    app.renderer._banner_remaining = 1.0
    key(app, pygame.K_r)
    assert not app.advance()
    assert app.renderer._banner_remaining == 0
    assert app.snapshot is None and app.env.latest_reward is None
    assert all(value == 0 for value in app.env.reward_totals.values())
    assert app.env.steps == 0 and app.renderer.step_requests == 0
    app.renderer.close()


@pytest.mark.parametrize("style", C.CONTROL_STYLES)
def test_loaded_policy_snapshot_uses_action_input_and_drawing_never_probes(style):
    model = load_agent(style)
    app = make_app(style, agent=model, policy="ppo")
    observation = app.obs.copy()
    expected_action = int(model.predict(observation, deterministic=True)[0])
    torch_rng = torch.get_rng_state().clone()
    expected_view = probe(model, observation, style, expected_action)
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert app.advance()
    snapshot = app.snapshot
    assert snapshot.policy == expected_view
    assert snapshot.policy.chosen == snapshot.reward.action == expected_action
    assert snapshot.policy.action_names == tuple(action.name for action in C.ACTION_ENUMS[style])
    assert sum(snapshot.policy.scores) == pytest.approx(1)
    np.testing.assert_array_equal(snapshot.observation, observation)
    model.predict = lambda *args, **kwargs: pytest.fail("draw must not select an action")
    model.policy.get_distribution = lambda *args: pytest.fail("draw must not probe")
    before = pickle.dumps(app.env.__dict__)
    for _ in range(3):
        app.draw()
    assert pickle.dumps(app.env.__dict__) == before
    assert torch.equal(torch_rng, torch.get_rng_state())
    app.renderer.close()


@pytest.mark.parametrize("style", C.CONTROL_STYLES)
def test_terminal_frame_holds_and_never_predicts_again(style):
    app = make_app(style, max_steps=1)
    key(app, pygame.K_n)
    assert app.advance() and app.done
    final = app.snapshot
    assert final.reward.truncated
    app.agent.predict = lambda *args, **kwargs: pytest.fail("terminal must not select")
    key(app, pygame.K_n)
    assert not app.advance()
    app.draw()
    assert app.snapshot is final
    app.renderer.close()


def test_human_and_unavailable_diagnostics_are_honest():
    human = make_app(policy="human")
    key(human, pygame.K_n)
    human.advance(lambda: int(C.DirectAction.SHOOT))
    assert human.snapshot.policy is None
    assert "HUMAN" in human.snapshot.policy_status
    human.renderer.close()
    class OpaqueModel:
        def predict(self, obs, deterministic):
            assert deterministic
            return 0, None
    opaque = make_app(policy="ppo", agent=OpaqueModel())
    opaque.advance()
    assert opaque.snapshot.policy is None
    assert "diagnostics unavailable" in opaque.snapshot.policy_status
    assert "RANDOM" not in opaque.snapshot.policy_status
    opaque.renderer.close()


@pytest.mark.parametrize("style", C.CONTROL_STYLES)
def test_headless_debug_and_physics_have_visible_evidence(style):
    pygame.display.quit()
    app = make_app(style)
    app.env.enemies = [Enemy(200, 150, health=2, speed=0)]
    app.env.bullets = [Bullet(300, 300, 0, speed=0)]
    app.advance()
    captured = []
    original = app.renderer._text
    def recording(surface, text, x, y, color=(154, 160, 174)):
        captured.append((text, x, y))
        original(surface, text, x, y, color)
    app.renderer._text = recording
    visible = app.draw().copy()
    assert not pygame.display.get_init()
    text = "\n".join(row[0] for row in captured)
    assert "RANDOM" in text and "spawn " in text and "step() reward" in text
    assert "Probabilities / V(s): unavailable" in text
    assert all(y + app.renderer.font_small.get_height() <= visible.get_height()
               for _, _, y in captured)
    app.renderer.show_debug = False
    hidden = app.draw()
    field = pygame.Rect(0, app.renderer.HUD_HEIGHT, C.ARENA_WIDTH, C.ARENA_HEIGHT)
    assert not np.array_equal(pygame.surfarray.array3d(visible.subsurface(field)),
                              pygame.surfarray.array3d(hidden.subsurface(field)))
    app.renderer.close()


def test_empty_models_random_fallback_and_headless_human(tmp_path):
    agent, name = load_policy("direct", models_dir=tmp_path)
    assert name == "random" and isinstance(agent, RandomPolicy)
    result = play("rotation", human=True, headless=True, max_frames=3, debug=True)
    assert result["frames"] == 3
