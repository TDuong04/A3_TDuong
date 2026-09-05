"""Tests for the arena training pipeline.

Training a real agent takes minutes, so nothing here trains one. What these tests protect is
everything *around* the learning: that the behavioural metrics reach TensorBoard, and that the
two Windows-specific guards are actually in place. Both of those failures are catastrophic and
neither shows up in an ordinary test run.

The behavioural metrics matter more than they look. A key that `Monitor` was never told to keep
simply never arrives -- no error, no warning, just a curve that stays empty for the whole run, which
is discovered hours later when someone opens TensorBoard looking for it.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from arena.env import ArenaEnv  # noqa: E402
from train.callbacks import (  # noqa: E402
    INFO_KEYWORDS,
    BehaviourLoggingCallback,
    ConsoleProgressCallback,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- the metrics that make a reward curve interpretable -------------------------------------------


class TestBehaviourLogging:
    def test_every_logged_keyword_is_actually_produced_by_the_env(self):
        """The silent failure this guards: Monitor only forwards keys it was told to keep, and a
        key the env never emits arrives as nothing at all, with no error anywhere."""
        env = ArenaEnv(control_style="direct")
        _, info = env.reset(seed=0)
        missing = [key for key in INFO_KEYWORDS if key not in info]
        assert not missing, f"train/callbacks.py logs keys the env never emits: {missing}"

    def test_the_metrics_that_distinguish_progress_from_reward_hacking_are_logged(self):
        """Reward alone cannot tell an agent clearing phases from one farming a shaping term."""
        for required in ("phase", "spawners_destroyed"):
            assert required in INFO_KEYWORDS

    def test_a_finished_episode_updates_every_buffer(self):
        callback = BehaviourLoggingCallback()
        callback.locals = {
            "infos": [
                {
                    "episode": {"r": 4.0, "l": 120},
                    "phase": 3,
                    "spawners_destroyed": 5,
                    "enemies_killed": 11,
                    "damage_taken": 2,
                    "terminated": False,
                }
            ]
        }
        callback._on_step()
        summary = callback.summary()
        assert summary["phase_reached"] == 3.0
        assert summary["spawners_destroyed"] == 5.0
        assert summary["enemies_killed"] == 11.0
        assert summary["damage_taken"] == 2.0
        assert summary["episode_length"] == 120.0
        assert callback.episodes_seen == 1

    def test_survival_rate_distinguishes_death_from_the_step_cap(self):
        """An agent whose survival rate climbs while its phase count does not has learned to hide,
        which is a real outcome in this env and invisible in the reward curve."""
        callback = BehaviourLoggingCallback()
        for terminated in (True, True, False, False):
            callback.locals = {
                "infos": [{"episode": {"r": 0.0, "l": 10}, "terminated": terminated}]
            }
            callback._on_step()
        assert callback.summary()["survival_rate"] == pytest.approx(0.5)

    def test_a_step_without_a_finished_episode_records_nothing(self):
        """`episode` is attached by Monitor only on the final step; counting mid-episode infos
        would let a long episode dominate the mean."""
        callback = BehaviourLoggingCallback()
        callback.locals = {"infos": [{"phase": 9, "spawners_destroyed": 9}]}
        callback._on_step()
        assert callback.summary() == {}
        assert callback.episodes_seen == 0

    def test_the_rolling_mean_is_bounded_by_its_window(self):
        callback = BehaviourLoggingCallback(window=3)
        for phase in (1, 1, 1, 4, 4, 4):
            callback.locals = {"infos": [{"episode": {"r": 0.0, "l": 1}, "phase": phase}]}
            callback._on_step()
        assert callback.summary()["phase_reached"] == pytest.approx(4.0)

    def test_metrics_reach_tensorboard_under_a_behaviour_prefix(self):
        recorded: dict[str, float] = {}

        class FakeLogger:
            def record(self, key, value):
                recorded[key] = value

        callback = BehaviourLoggingCallback()
        callback.locals = {"infos": [{"episode": {"r": 1.0, "l": 5}, "phase": 2}]}
        callback._on_step()
        # `BaseCallback.logger` is a read-only property returning `self.model.logger`, so the
        # logger has to be injected through a stand-in model rather than assigned directly.
        callback.model = SimpleNamespace(logger=FakeLogger())
        callback._on_rollout_end()
        assert "behaviour/phase_reached" in recorded
        assert recorded["behaviour/phase_reached"] == pytest.approx(2.0)

    def test_progress_callback_survives_having_no_behaviour_callback(self):
        callback = ConsoleProgressCallback(every=1, behaviour=None)
        callback.num_timesteps = 10
        assert callback._on_step() is True


# --- the Windows guards, which are not optional ---------------------------------------------------


class TestTrainingEntryPointGuards:
    def _source(self) -> str:
        return (REPO_ROOT / "train" / "train_arena.py").read_text(encoding="utf-8")

    def test_subprocvecenv_is_behind_a_main_guard(self):
        """Without the guard, each SubprocVecEnv worker re-imports this module and forks again,
        recursively, until the machine stops responding."""
        tree = ast.parse(self._source())
        guards = [
            node for node in tree.body
            if isinstance(node, ast.If) and "__main__" in ast.dump(node.test)
        ]
        assert guards, "train_arena.py must guard its entry point with if __name__ == '__main__'"

    def test_the_dummy_video_driver_is_set_before_any_import_that_could_pull_in_pygame(self):
        """A worker that opens a window during training is both slow and, on a headless machine,
        fatal. The assignment has to be set above the arena imports, not merely present."""
        source = self._source()
        driver = source.index("SDL_VIDEODRIVER")
        first_arena_import = source.index("from arena.env import")
        assert driver < first_arena_import

    def test_models_land_in_the_folder_the_brief_names(self):
        from train.train_arena import MODELS_DIR, model_path

        assert MODELS_DIR.name == "models", "the brief names this folder exactly"
        assert model_path("direct").name == "ppo_direct.zip"
        assert model_path("rotation").name == "ppo_rotation.zip"

    def test_both_styles_are_trainable_and_an_unknown_one_is_refused(self):
        from train.train_arena import STYLES, train

        assert set(STYLES) == {"direct", "rotation"}
        with pytest.raises(ValueError, match="unknown control style"):
            train("sideways", timesteps=1)

    def test_a_single_env_avoids_the_subprocess_machinery(self):
        """One env in a subprocess costs more in IPC than it gains in parallelism, and it makes
        every traceback cross a process boundary."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        from train.train_arena import build_vec_env

        vec = build_vec_env("direct", n_envs=1, seed=0)
        try:
            assert isinstance(vec, DummyVecEnv)
        finally:
            vec.close()
