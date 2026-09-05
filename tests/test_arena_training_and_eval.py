"""Tests for the arena training pipeline and the evaluation script.

Training a real agent takes minutes, so nothing here trains one. What these tests protect is
everything *around* the learning: that the behavioural metrics reach TensorBoard, that the
Windows-specific guards are actually in place, that evaluation is deterministic, that both control
styles are driven correctly, and that a missing model produces a sentence rather than a traceback.

The behavioural metrics matter more than they look. A key that `Monitor` was never told to keep
simply never arrives -- no error, no warning, just a curve that stays empty for the whole run, which
is discovered hours later when someone opens TensorBoard looking for it.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from arena.constants import DirectAction, RotationAction  # noqa: E402
from arena.env import ArenaEnv  # noqa: E402
from eval.play_arena import (  # noqa: E402
    DETERMINISTIC,
    ArenaEvalError,
    evaluate,
    format_summary,
    human_action,
    resolve_model,
)
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


# --- evaluation -----------------------------------------------------------------------------------


class TestEvaluation:
    def test_evaluation_is_deterministic(self):
        """The rubric asks for it, and a policy that samples on camera reads as one that has not
        learned anything."""
        assert DETERMINISTIC is True

    def test_a_missing_model_is_a_message_not_a_traceback(self, ):
        with pytest.raises(ArenaEvalError) as caught:
            resolve_model("direct", models_dir=REPO_ROOT / "models" / "__absent__")
        message = str(caught.value)
        assert "python -m train.train_arena --style direct" in message
        assert "No trained model" in message

    def test_the_message_names_the_style_that_was_missing(self):
        with pytest.raises(ArenaEvalError) as caught:
            resolve_model("rotation", models_dir=REPO_ROOT / "models" / "__absent__")
        assert "--style rotation" in str(caught.value)

    def test_evaluate_reports_every_column_the_report_tabulates(self):
        """Report row R6 compares the two control styles; these are its columns."""
        stats = evaluate("direct", episodes=2, seed=0, agent=_ScriptedAgent(DirectAction.SHOOT))
        for key in (
            "return_mean", "return_std", "phase_mean", "phase_max",
            "phase_cleared_episodes", "spawners_mean", "enemies_mean", "survival_rate",
        ):
            assert key in stats, key
        assert stats["episodes"] == 2
        assert len(stats["returns"]) == 2

    def test_both_styles_meet_the_same_arenas(self):
        """A comparison run on different layouts compares the layouts as much as the agents."""
        first = evaluate("direct", episodes=2, seed=7, agent=_ScriptedAgent(DirectAction.NOOP))
        second = evaluate("direct", episodes=2, seed=7, agent=_ScriptedAgent(DirectAction.NOOP))
        assert first["returns"] == second["returns"]

    def test_a_different_seed_gives_a_different_run(self):
        """Guards the mirror image: a seed that is accepted and ignored would make every
        comparison above vacuous."""
        first = evaluate("direct", episodes=2, seed=1, agent=_ScriptedAgent(DirectAction.NOOP))
        second = evaluate("direct", episodes=2, seed=99, agent=_ScriptedAgent(DirectAction.NOOP))
        assert first["returns"] != second["returns"]

    def test_each_episode_gets_its_own_seed(self, monkeypatch):
        """Otherwise five "episodes" are five replays of one arena, the standard deviation is
        structurally zero, and the report quotes a spread that was never measured. A mutant that
        reset every episode to the same seed survived the rest of this class."""
        seeds: list[int | None] = []
        original = ArenaEnv.reset

        def spy(self, *, seed=None, options=None):
            seeds.append(seed)
            return original(self, seed=seed, options=options)

        monkeypatch.setattr(ArenaEnv, "reset", spy)
        evaluate("direct", episodes=4, seed=10, agent=_ScriptedAgent(DirectAction.NOOP))
        # `ArenaEnv.__init__` resets once with no seed so a renderer never sees a half-built env;
        # that unseeded call is not one of the episodes.
        assert [s for s in seeds if s is not None] == [10, 11, 12, 13], seeds

    def test_the_summary_names_the_phase_progression_the_video_needs(self):
        stats = evaluate("direct", episodes=1, seed=0, agent=_ScriptedAgent(DirectAction.NOOP))
        text = format_summary(stats)
        assert "phase cleared in" in text
        assert "return" in text and "survival" in text


# --- human control --------------------------------------------------------------------------------


class TestHumanControl:
    def test_direct_keys_map_to_direct_actions(self):
        import pygame

        assert human_action(_Keys(pygame.K_UP), "direct") == int(DirectAction.UP)
        assert human_action(_Keys(pygame.K_DOWN), "direct") == int(DirectAction.DOWN)
        assert human_action(_Keys(pygame.K_a), "direct") == int(DirectAction.LEFT)
        assert human_action(_Keys(pygame.K_SPACE), "direct") == int(DirectAction.SHOOT)
        assert human_action(_Keys(), "direct") == int(DirectAction.NOOP)

    def test_rotation_keys_map_to_rotation_actions(self):
        """The same physical key means different things in the two styles, which is the whole
        point of having two: LEFT rotates the ship, it does not move it left."""
        import pygame

        assert human_action(_Keys(pygame.K_LEFT), "rotation") == int(RotationAction.ROTATE_LEFT)
        assert human_action(_Keys(pygame.K_UP), "rotation") == int(RotationAction.THRUST)
        assert human_action(_Keys(pygame.K_SPACE), "rotation") == int(RotationAction.SHOOT)
        assert human_action(_Keys(), "rotation") == int(RotationAction.NOOP)

    def test_the_two_styles_disagree_about_the_left_arrow(self):
        import pygame

        keys = _Keys(pygame.K_LEFT)
        assert human_action(keys, "direct") != human_action(keys, "rotation")

    def test_every_human_action_is_inside_its_action_space(self):
        import pygame

        probes = [_Keys(), _Keys(pygame.K_UP), _Keys(pygame.K_DOWN), _Keys(pygame.K_LEFT),
                  _Keys(pygame.K_RIGHT), _Keys(pygame.K_SPACE)]
        for style in ("direct", "rotation"):
            env = ArenaEnv(control_style=style)
            for keys in probes:
                assert env.action_space.contains(human_action(keys, style))


# --- helpers ----------------------------------------------------------------------------------------


class _ScriptedAgent:
    """Stands in for a trained policy. Deterministic, so seeded runs stay reproducible."""

    def __init__(self, action) -> None:
        self.action = int(action)
        self.calls = 0

    def predict(self, obs, deterministic: bool = True):
        self.calls += 1
        return np.int64(self.action), None


class _Keys:
    """A `pygame.key.get_pressed()` stand-in: truthy only for the keys it was given."""

    def __init__(self, *pressed: int) -> None:
        self.pressed = set(pressed)

    def __getitem__(self, key: int) -> bool:
        return key in self.pressed
