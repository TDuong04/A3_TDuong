"""Shield and elite gameplay, model compatibility, and read-only presentation."""
import math
import pickle
import random
from dataclasses import replace

import numpy as np
import pytest

from arena.constants import CONTROL_STYLES, DirectAction, RotationAction, FIXED_DT, OBS_DIM, REWARD_PER_STEP
from arena.entities import Bullet, Enemy, Spawner
from arena.env import ArenaEnv
from arena.mechanics import EliteCharger, MechanicsConfig, ShieldPickup
from arena.observation import describe
from arena.render import ArenaRenderer
from eval.play_arena import ArenaEvalError, play, validate_policy_space, write_results


@pytest.mark.parametrize('style', CONTROL_STYLES)
def test_real_spawner_kill_drop_collection_and_one_block(style):
    env = ArenaEnv(control_style=style, seed=0, mechanics=True)
    # This test asserts reward as an exact sum of the frozen constants in `constants.py`; the
    # optional shaping terms (config-driven, arena/env.py's ShapingConfig) are exercised on their
    # own in test_arena_env.py's TestAimShaping/TestSafetyShaping instead of smuggled in here.
    env.shaping = replace(env.shaping, aim_strength=0.0, safety_strength=0.0)
    env.player.heading = 0
    target = Spawner(env.player.x + 48, env.player.y, env.phase_settings)
    target.health = 1
    env.spawners = [target]
    env.step(RotationAction.SHOOT if style == 'rotation' else DirectAction.SHOOT)
    assert env.spawners_destroyed == 1 and len(env.pickups) == 1
    assert env.phase == 2  # drop survives a transition
    assert len([e for e in env.enemies if isinstance(e, EliteCharger)]) == 1
    pickup = env.pickups[0]
    assert pickup.position == target.position
    env.player.x, env.player.y = pickup.position
    env.enemies = [Enemy(*pickup.position, health=2, speed=0) for _ in range(2)]
    health = env.player.health
    _, reward, _, _, info = env.step(0)
    assert info['pickups_collected'] == 1 and info['shield_blocks'] == 1
    assert not env.pickups and env.player.shield_charge == 0
    assert env.player.health == health and env.damage_taken == 0
    assert reward == pytest.approx(REWARD_PER_STEP)
    env.player.invulnerable_remaining = 0
    env.step(0)
    assert env.player.health == health - 1 and env.damage_taken == 1


def test_pickups_expire_and_do_not_stack_or_consume_when_full():
    env = ArenaEnv(mechanics=True)
    pickup = ShieldPickup(*env.player.position, env.mechanics_config)
    env.pickups = [pickup]
    env.player.shield_charge = 1
    env.step(0)
    assert env.pickups == [pickup] and env.pickups_collected == 0
    pickup.remaining = FIXED_DT
    env.step(0)
    assert not env.pickups and env.player.shield_charge == 1
    env.reset(seed=1)
    assert not env.pickups and env.player.shield_charge == env.player.shield_blocks == 0


def test_elite_locks_direction_cycles_and_recovers_at_boundary():
    cfg = replace(MechanicsConfig.from_yaml(), pursuit_seconds=FIXED_DT,
                  windup_seconds=2 * FIXED_DT, charge_seconds=2 * FIXED_DT,
                  recovery_seconds=FIXED_DT)
    elite = EliteCharger(200, 200, cfg)
    elite.update(500, 200)
    assert elite.state == 'windup' and (elite.charge_dx, elite.charge_dy) == (1, 0)
    position = elite.position
    for _ in range(2):
        elite.update(200, 500)
    assert elite.state == 'charge' and elite.position == position
    elite.update(200, 500)
    assert elite.x > position[0] and elite.y == position[1]
    elite.update(200, 500)
    assert elite.state == 'recovery' and elite.vx == elite.vy == 0
    elite.update(200, 500)
    assert elite.state == 'pursuit'
    elite._enter('charge')
    elite.charge_dx, elite.charge_dy = -1, 0
    elite.x = elite.radius
    elite.update(500, 500)
    assert elite.state == 'recovery' and elite.x == elite.radius
    elite.kill()
    before = pickle.dumps(elite)
    elite.update(500, 500)
    assert pickle.dumps(elite) == before


def test_elite_gates_phase_and_real_projectile_kill_releases_it():
    env = ArenaEnv(mechanics=True, seed=0)
    env.phase = 2
    env._begin_phase()
    elite = next(e for e in env.enemies if isinstance(e, EliteCharger))
    env.spawners.clear()
    env.step(0)
    assert env.phase == 2
    elite.health = 1
    env.bullets = [Bullet(elite.x, elite.y, 0, speed=0)]
    env._resolve_bullet_hits()
    env._drop_dead()
    env._maybe_advance_phase()
    assert env.phase == 3 and env.elites_killed == env.enemies_killed == 1
    assert len([e for e in env.enemies if isinstance(e, EliteCharger)]) == 1
    env.reset(seed=0)
    assert env.phase == 1 and not env.enemies and env.elites_killed == 0


@pytest.mark.parametrize('style', CONTROL_STYLES)
def test_observation_local_frame_absence_and_action_compatibility(style):
    env = ArenaEnv(control_style=style, mechanics=True)
    baseline = ArenaEnv(control_style=style)
    assert env.action_space == baseline.action_space
    assert baseline.observation_space.shape == (OBS_DIM,)
    env.player.heading = math.pi / 2
    env.pickups = [ShieldPickup(env.player.x, env.player.y + 100, env.mechanics_config)]
    elite = EliteCharger(env.player.x, env.player.y + 200, env.mechanics_config)
    elite._enter('windup')
    elite.charge_dx, elite.charge_dy = 0, 1
    env.enemies = [elite]
    obs = env.observation()
    values = dict(zip(describe(True), obs, strict=True))
    assert len(obs) == len(describe(True)) and env.observation_space.contains(obs)
    assert values['pickup_local_dx'] == pytest.approx(100 / math.hypot(960, 680))
    assert values['pickup_local_dy'] == pytest.approx(0, abs=1e-7)
    assert values['elite_windup'] == values['elite_timer'] == values['charge_local_dx'] == 1
    env.pickups.clear()
    env.enemies.clear()
    assert not env.observation()[OBS_DIM + 1:].any()


@pytest.mark.parametrize('style', CONTROL_STYLES)
def test_mechanics_rendered_and_unrendered_trajectories_match(style):
    def scene():
        env = ArenaEnv(control_style=style, mechanics=True, seed=7)
        env.phase = 2
        env._begin_phase()
        env.player.shield_charge = 1
        env.pickups = [ShieldPickup(env.player.x + 150, env.player.y, env.mechanics_config)]
        return env
    env, reference = scene(), scene()
    renderer = ArenaRenderer(headless=True)
    python_rng, numpy_rng = random.getstate(), pickle.dumps(np.random.get_state())
    states = set()
    for step in range(100):
        actual, expected = env.step(0), reference.step(0)
        states.update(e.state for e in env.enemies if isinstance(e, EliteCharger))
        renderer.draw(env)
        renderer.draw(env)
        if step % 9 == 0:
            renderer.toggle_effects()
            renderer.toggle_observation_overlay()
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1:] == expected[1:]
        assert pickle.dumps(env.__dict__) == pickle.dumps(reference.__dict__)
    assert {'windup', 'charge', 'recovery'} <= states
    assert random.getstate() == python_rng
    assert pickle.dumps(np.random.get_state()) == numpy_rng
    renderer.close()


@pytest.mark.parametrize('style', CONTROL_STYLES)
def test_human_playback_with_mechanics(style):
    result = play(style, human=True, mechanics=True, headless=True, max_frames=4)
    assert result['mechanics'] and result['human'] and result['frames'] == 4


def test_baseline_checkpoint_rejected_for_mechanics():
    class BaselineAgent:
        observation_space = ArenaEnv().observation_space
    with pytest.raises(ArenaEvalError, match='spaces do not match'):
        validate_policy_space(BaselineAgent(), ArenaEnv(mechanics=True))


@pytest.mark.parametrize('style', CONTROL_STYLES)
def test_new_model_can_train_save_reload_and_play(tmp_path, style):
    from stable_baselines3 import PPO
    from train.train_arena import build_vec_env
    venv = build_vec_env(style, 1, 0, mechanics=True)
    model = PPO('MlpPolicy', venv, n_steps=8, batch_size=8, n_epochs=1,
                policy_kwargs={'net_arch': [8]}, device='cpu', verbose=0)
    model.learn(16)
    directory = tmp_path / 'mechanics'
    directory.mkdir()
    model.save(directory / f'ppo_{style}')
    venv.close()
    result = play(style, mechanics=True, models_dir=tmp_path, headless=True, max_frames=3)
    assert result['mechanics'] and result['policy'] == 'ppo'


def test_cli_training_records_mechanics_and_preserves_baseline(tmp_path):
    import json
    from train.train_arena import parse_args, train
    baseline = tmp_path / 'models' / 'ppo_direct.zip'
    baseline.parent.mkdir()
    baseline.write_bytes(b'baseline sentinel')
    path = train(parse_args([
        '--style', 'direct', '--mechanics', '--timesteps', '16', '--n-envs', '1',
        '--n-steps', '8', '--batch-size', '8', '--device', 'cpu', '--verbose', '0',
        '--log-dir', str(tmp_path / 'logs'), '--models-dir', str(baseline.parent),
        '--run-name', 'smoke',
    ]))
    assert path == baseline.parent / 'mechanics' / 'ppo_direct.zip'
    assert path.exists() and baseline.read_bytes() == b'baseline sentinel'
    metadata = json.loads((tmp_path / 'logs' / 'smoke' / 'run.json').read_text())
    assert metadata['mechanics'] and metadata['observation_dim'] == len(describe(True))
    assert metadata['mechanics_config']['elite_first_phase'] == 2


def test_evaluation_results_cannot_mix_rules_or_overwrite_baseline(tmp_path):
    from eval.play_arena import evaluate
    base = evaluate('direct', episodes=0, random_policy=True)
    extra = evaluate('direct', episodes=0, random_policy=True, mechanics=True)
    baseline_paths = write_results([base], tmp_path)
    original = baseline_paths['comparison'].read_bytes()
    paths = write_results([extra], tmp_path)
    assert paths['comparison'].parent == tmp_path / 'mechanics'
    assert '--mechanics' in paths['comparison'].read_text()
    assert baseline_paths['comparison'].read_bytes() == original
    with pytest.raises(ValueError, match='same mechanics'):
        write_results([base, extra], tmp_path)
