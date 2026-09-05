"""Tests for `train.train_arena` and `train.callbacks` — A3-011's acceptance criteria.

The pipeline is worth testing for the same reason it is worth building early: a run that trains
correctly but logs the wrong thing is only discovered hours later, and the fix is to train again.
So these pin the wiring rather than the learning — that the config is what reaches the model, that
`Monitor` carries the four behavioural keys through the episode boundary, that the callback turns
them into `behaviour/*` scalars in a real event file, and that a run leaves a model, a checkpoint
and its own resolved config behind.

The training here is deliberately tiny (a few hundred timesteps, one or two envs) so the suite
stays fast; the full-length 400k run lives in the CLI, not in pytest.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from arena.constants import CONTROL_STYLES, MAX_EPISODE_STEPS, N_ACTIONS
from common.config import ArenaTrainConfig
from train import train_arena
from train.callbacks import (
    BEHAVIOUR_INFO_KEYS,
    BEHAVIOUR_TAGS,
    EPISODE_LENGTH_TAG,
    BehaviourLoggingCallback,
    make_checkpoint_callback,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scratch_dir() -> Iterator[Path]:
    """Not `tmp_path`: the shared `pytest-of-<user>` base dir is unreadable on this machine."""
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


def tiny_args(scratch: Path, **overrides) -> object:
    """CLI args for a run short enough to sit inside a test."""
    argv = [
        "--style", "direct",
        "--timesteps", "256",
        "--n-envs", "1",
        "--n-steps", "64",
        "--batch-size", "64",
        "--checkpoint-freq", "128",
        "--device", "cpu",
        "--verbose", "0",
        "--log-dir", str(scratch / "logs"),
        "--models-dir", str(scratch / "models"),
    ]
    for flag, value in overrides.items():
        argv += [f"--{flag.replace('_', '-')}", str(value)]
    return train_arena.parse_args(argv)


# --- headless guarantees -----------------------------------------------------------------------


def test_sdl_videodriver_is_set_by_importing_the_module():
    """Set at import, before anything downstream could reach a display driver."""
    import os

    assert os.environ["SDL_VIDEODRIVER"] == "dummy"


def test_training_path_never_imports_pygame():
    """A worker that imports pygame has a window's worth of overhead per env, times eight.

    Run in a subprocess so the check is not fooled by another test having imported the renderer.
    """
    script = (
        "import sys;"
        "from train.train_arena import build_vec_env;"
        "venv = build_vec_env('direct', 1, 0);"
        "venv.reset();"
        "venv.step(__import__('numpy').zeros(1, dtype=int));"
        "venv.close();"
        "assert 'pygame' not in sys.modules, sorted(m for m in sys.modules if 'pygame' in m);"
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_module_guards_subprocvecenv_behind_a_main_block():
    """Without the guard, `spawn` re-imports this module in every worker and forks recursively."""
    source = (REPO_ROOT / "train" / "train_arena.py").read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in source


# --- env construction --------------------------------------------------------------------------


@pytest.mark.parametrize("style", CONTROL_STYLES)
def test_make_env_wraps_a_monitor_carrying_the_behavioural_keys(style: str):
    env = train_arena.make_env(style, seed=0, rank=0)()
    try:
        assert isinstance(env, Monitor)
        assert env.info_keywords == BEHAVIOUR_INFO_KEYS
        assert env.action_space.n == N_ACTIONS[style]
    finally:
        env.close()


def test_monitor_puts_the_behavioural_keys_into_the_episode_record():
    """The path the callback depends on: info keys -> `info['episode']` at the episode boundary."""
    env = train_arena.make_env("direct", seed=0, rank=0)()
    try:
        env.reset(seed=0)
        episode = None
        for _ in range(MAX_EPISODE_STEPS + 1):
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                episode = info["episode"]
                break
        assert episode is not None, "no episode finished within the horizon"
        for key in BEHAVIOUR_INFO_KEYS:
            assert key in episode
        assert {"r", "l", "t"} <= set(episode)
    finally:
        env.close()


def test_build_vec_env_stays_in_process_for_one_env():
    venv = train_arena.build_vec_env("direct", n_envs=1, seed=0)
    try:
        assert isinstance(venv, DummyVecEnv)
        assert venv.num_envs == 1
    finally:
        venv.close()


def test_build_vec_env_uses_subprocesses_above_one_env(scratch_dir: Path):
    monitor_dir = scratch_dir / "monitor"
    venv = train_arena.build_vec_env("direct", n_envs=2, seed=0, monitor_dir=monitor_dir)
    try:
        assert isinstance(venv, SubprocVecEnv)
        assert venv.num_envs == 2
        venv.reset()
    finally:
        venv.close()
    assert sorted(p.name for p in monitor_dir.glob("*.csv")) == [
        "worker_0.monitor.csv", "worker_1.monitor.csv",
    ]


def test_workers_are_seeded_apart():
    """Eight arenas replaying identical spawner layouts would be one arena with eight times the
    sample count and none of the diversity."""
    first = train_arena.make_env("direct", seed=0, rank=0)()
    second = train_arena.make_env("direct", seed=0, rank=1)()
    try:
        assert not (first.unwrapped.last_observation == second.unwrapped.last_observation).all()
    finally:
        first.close()
        second.close()


# --- config ----------------------------------------------------------------------------------


def test_config_defaults_come_from_the_yaml_not_from_code():
    baseline = ArenaTrainConfig.from_yaml("arena")
    resolved = train_arena.resolve_config(
        train_arena.parse_args(["--style", "direct"])
    )
    assert resolved == baseline


def test_only_the_flags_passed_override_the_baseline():
    """What lets the sweep vary one axis at a time and attribute the difference to it."""
    baseline = ArenaTrainConfig.from_yaml("arena")
    resolved = train_arena.resolve_config(
        train_arena.parse_args(["--style", "direct", "--learning-rate", "0.001",
                                "--net-arch", "128", "128"])
    )
    assert resolved.learning_rate == 0.001
    assert resolved.net_arch == [128, 128]
    for field in ("algorithm", "total_timesteps", "n_envs", "n_steps", "batch_size", "gamma",
                  "ent_coef", "seed"):
        assert getattr(resolved, field) == getattr(baseline, field)


def test_config_asks_for_eight_parallel_envs():
    assert ArenaTrainConfig.from_yaml("arena").n_envs == 8


def test_algorithm_class_is_case_insensitive_and_rejects_the_unknown():
    assert train_arena.algorithm_class("ppo") is PPO
    assert train_arena.algorithm_class("DQN") is DQN
    with pytest.raises(ValueError, match="unsupported algorithm"):
        train_arena.algorithm_class("A2C")


def test_run_name_carries_algorithm_style_and_tag():
    config = ArenaTrainConfig.from_yaml("arena")
    name = train_arena.run_name(config, "rotation", tag="lr1e-3")
    assert name.startswith("ppo_rotation_lr1e-3_")


def test_build_model_uses_the_config_and_a_hidden_layer_mlp():
    venv = train_arena.build_vec_env("rotation", n_envs=1, seed=0)
    try:
        config = ArenaTrainConfig(
            algorithm="PPO", total_timesteps=256, n_envs=1, learning_rate=1e-3, n_steps=64,
            batch_size=64, gamma=0.95, ent_coef=0.02, net_arch=[32, 16], seed=7,
        )
        model = train_arena.build_model(config, venv, tensorboard_log=None, device="cpu",
                                        verbose=0)
        assert isinstance(model, PPO)
        assert model.learning_rate == 1e-3
        assert model.n_steps == 64
        assert model.gamma == 0.95
        assert model.ent_coef == 0.02
        assert model.policy_kwargs["net_arch"] == [32, 16]
    finally:
        venv.close()


def test_dqn_declares_the_on_policy_flags_it_cannot_use():
    config = ArenaTrainConfig.from_yaml("arena")
    assert train_arena.ignored_hyperparameters(config) == []
    assert train_arena.ignored_hyperparameters(replace(config, algorithm="DQN")) == [
        "n_steps", "ent_coef",
    ]


# --- callback ----------------------------------------------------------------------------------


class RecordingLogger:
    """The two lines of `stable_baselines3.common.logger.Logger` the callback actually touches."""

    def __init__(self) -> None:
        self.records: dict[str, float] = {}

    def record(self, key: str, value: float, exclude=None) -> None:
        self.records[key] = value


def attach(callback: BehaviourLoggingCallback) -> RecordingLogger:
    logger = RecordingLogger()
    callback.model = SimpleNamespace(logger=logger)
    return logger


def episode_info(**values) -> dict:
    record = {"r": 0.0, "l": values.pop("l", 100)}
    record.update(values)
    return {"episode": record}


def test_callback_records_every_behavioural_tag():
    callback = BehaviourLoggingCallback()
    logger = attach(callback)
    callback.locals = {"infos": [episode_info(phase=2, spawners_destroyed=5,
                                              enemies_killed=9, damage_taken=1, l=400)]}
    assert callback._on_step() is True

    assert logger.records[BEHAVIOUR_TAGS["phase"]] == 2
    assert logger.records[BEHAVIOUR_TAGS["spawners_destroyed"]] == 5
    assert logger.records[BEHAVIOUR_TAGS["enemies_killed"]] == 9
    assert logger.records[BEHAVIOUR_TAGS["damage_taken"]] == 1
    assert logger.records[EPISODE_LENGTH_TAG] == 400
    assert callback.episodes_seen == 1


def test_callback_reports_a_rolling_mean_over_its_window():
    callback = BehaviourLoggingCallback(window=2)
    logger = attach(callback)
    for phase in (0, 2, 4):
        callback.locals = {"infos": [episode_info(phase=phase, spawners_destroyed=0,
                                                  enemies_killed=0, damage_taken=0)]}
        callback._on_step()
    # Window of 2 keeps the last two episodes only: mean(2, 4), not mean(0, 2, 4).
    assert logger.records[BEHAVIOUR_TAGS["phase"]] == 3
    assert callback.episodes_seen == 3


def test_callback_absorbs_every_env_that_finished_on_the_same_step():
    callback = BehaviourLoggingCallback()
    attach(callback)
    callback.locals = {"infos": [
        episode_info(phase=1, spawners_destroyed=1, enemies_killed=1, damage_taken=0),
        {"phase": 9},  # mid-episode: no `episode` key, must be ignored
        episode_info(phase=3, spawners_destroyed=3, enemies_killed=1, damage_taken=0),
    ]}
    callback._on_step()
    assert callback.episodes_seen == 2
    assert callback.latest()[BEHAVIOUR_TAGS["phase"]] == 2


def test_callback_records_nothing_before_the_first_episode_ends():
    callback = BehaviourLoggingCallback()
    logger = attach(callback)
    callback.locals = {"infos": [{}, {"phase": 0}]}
    callback._on_step()
    callback._on_rollout_end()
    assert logger.records == {}


def test_callback_rejects_an_empty_window():
    with pytest.raises(ValueError, match="window"):
        BehaviourLoggingCallback(window=0)


def test_checkpoint_frequency_is_in_total_timesteps_not_per_env(scratch_dir: Path):
    """SB3 counts `save_freq` per env, so eight workers would otherwise checkpoint 8x too often."""
    callback = make_checkpoint_callback(scratch_dir / "checkpoints", "run", save_freq=50_000,
                                        n_envs=8, verbose=0)
    assert callback.save_freq == 6250
    assert (scratch_dir / "checkpoints").is_dir()


def test_checkpoint_callback_rejects_nonsense():
    with pytest.raises(ValueError):
        make_checkpoint_callback("/tmp/x", "run", save_freq=0)
    with pytest.raises(ValueError):
        make_checkpoint_callback("/tmp/x", "run", save_freq=10, n_envs=0)


# --- end to end --------------------------------------------------------------------------------


def test_a_run_leaves_model_logs_checkpoints_and_its_own_config(scratch_dir: Path):
    args = tiny_args(scratch_dir, run_name="unit")
    model_path = train_arena.train(args)

    run_dir = scratch_dir / "logs" / "unit"
    assert model_path == scratch_dir / "models" / "ppo_direct.zip"
    assert model_path.is_file()
    assert list(run_dir.glob("tb_1/events.out.tfevents.*")), "no TensorBoard event file"
    assert (run_dir / "monitor" / "worker_0.monitor.csv").is_file()
    assert list((scratch_dir / "models" / "checkpoints").glob("unit_*_steps.zip"))

    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert metadata["control_style"] == "direct"
    assert metadata["total_timesteps"] == 256
    assert metadata["policy"] == "MlpPolicy"
    assert metadata["net_arch"] == [64, 64]
    assert "finished" in metadata and "wall_clock_seconds" in metadata


def test_the_event_file_carries_the_behavioural_scalars(scratch_dir: Path):
    """The criterion in full: the four metrics are readable in TensorBoard, not just in stdout."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    train_arena.train(tiny_args(scratch_dir, run_name="tb", timesteps=1024))

    accumulator = EventAccumulator(str(scratch_dir / "logs" / "tb" / "tb_1"))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    assert set(BEHAVIOUR_TAGS.values()) | {EPISODE_LENGTH_TAG} <= tags
    assert "rollout/ep_rew_mean" in tags
    for tag in BEHAVIOUR_TAGS.values():
        assert accumulator.Scalars(tag), f"{tag} has no points"


def test_the_module_runs_as_a_script(scratch_dir: Path):
    """`python -m train.train_arena` is the documented entry point, so run it as one."""
    result = subprocess.run(
        [sys.executable, "-m", "train.train_arena",
         "--style", "rotation", "--timesteps", "256", "--n-envs", "2", "--n-steps", "64",
         "--batch-size", "64", "--device", "cpu", "--verbose", "0", "--run-name", "cli",
         "--log-dir", str(scratch_dir / "logs"), "--models-dir", str(scratch_dir / "models")],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert (scratch_dir / "models" / "ppo_rotation.zip").is_file()
