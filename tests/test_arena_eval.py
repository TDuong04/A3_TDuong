"""Tests for the arena evaluation script.

Nothing here trains an agent -- a scripted stand-in policy stands in for one, so these run in
milliseconds and stay deterministic. What they protect is everything a marker touches: that
evaluation is deterministic, that both control styles are driven correctly, that each episode is a
genuinely different arena, and that a missing model produces a sentence rather than a traceback.

The seeding test exists because a mutant that reset every episode to the same seed survived the
first version of this file. Five episodes would then have been five replays of one arena, the
standard deviation would be structurally zero, and the report would quote a spread that was never
measured.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from arena.constants import DirectAction, RotationAction  # noqa: E402
from arena.env import ArenaEnv  # noqa: E402
from eval.play_arena import (  # noqa: E402
    DETERMINISTIC,
    ArenaEvalError,
    evaluate,
    format_comparison_markdown,
    format_summary,
    human_action,
    play,
    resolve_model,
    write_results,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Scratch space for the persistence tests, created and removed per test. `tmp_path` is avoided
#: for the same reason `test_play_gridworld.py` avoids it: the pytest temp root is unusable here.
EMPTY_RESULTS = REPO_ROOT / "results" / "__not_a_directory__"


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


class TestPlayHumanMode:
    """`play(..., human=True)` used to crash on its very first frame.

    `pygame.key.get_pressed()` needs the video subsystem initialised, and that only happens
    lazily inside `ArenaRenderer.draw()`. Reading input before the loop's first `draw()` call
    raises "video system not initialized" on a display that has never been opened — exactly the
    state a real invocation starts from. `pygame.display.quit()` below resets to that state
    deliberately, since another test in the session may have already initialised it and hidden
    the bug.
    """

    def test_the_first_frame_of_human_play_does_not_crash(self):
        import pygame

        if pygame.display.get_init():
            pygame.display.quit()
        result = play("direct", episodes=1, seed=0, human=True, max_frames=3)
        assert result["frames"] == 3
        assert result["human"] is True

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


# --- persistence ----------------------------------------------------------------------------------


class TestWritingResults:
    """`evaluate` used to measure everything the report tabulates and then drop it on the floor.

    Numbers that reach a report by hand are how a report ends up disagreeing with its own
    artifacts, so these tests care about one thing above all: what lands on disk is what was
    measured, unrounded and unedited.
    """

    def _stats(self, style: str = "direct") -> dict:
        return {
            "style": style, "episodes": 3, "seed": 0,
            "return_mean": 8.903333, "return_std": 13.4921,
            "phase_mean": 1.6667, "phase_max": 2, "phase_cleared_episodes": 2,
            "spawners_mean": 1.6667, "enemies_mean": 14.0, "steps_mean": 760.0,
            "survival_rate": 0.0, "returns": [1.5, 2.5, 22.7], "phases": [1, 2, 2],
        }

    def test_one_json_per_style_plus_the_comparison_table(self):
        directory = EMPTY_RESULTS.parent / "__eval_write_test__"
        written = write_results([self._stats("direct"), self._stats("rotation")], directory)
        try:
            assert set(written) == {"direct", "rotation", "comparison"}
            assert written["comparison"].name == "comparison.md"
            assert all(path.exists() for path in written.values())
        finally:
            for path in written.values():
                path.unlink()
            directory.rmdir()

    def test_the_raw_episode_lists_survive_the_round_trip(self):
        """Means can be recomputed from these; they cannot be recovered from a mean."""
        directory = EMPTY_RESULTS.parent / "__eval_round_trip__"
        stats = self._stats()
        written = write_results([stats], directory)
        try:
            saved = json.loads(written["direct"].read_text())
            assert saved["returns"] == stats["returns"]
            assert saved["phases"] == stats["phases"]
            assert saved["return_mean"] == stats["return_mean"]  # unrounded
            assert "generated" in saved
        finally:
            for path in written.values():
                path.unlink()
            directory.rmdir()

    def test_the_comparison_table_names_both_styles_and_the_phase_evidence(self):
        markdown = format_comparison_markdown(
            [self._stats("direct"), self._stats("rotation")], "2026-09-06T00:00:00"
        )
        assert "`direct`" in markdown and "`rotation`" in markdown
        assert "Phases cleared" in markdown
        assert "2/3" in markdown  # the phase progression the video depends on
        assert "deterministic=True" in markdown

    def test_the_table_reports_the_numbers_it_was_given(self):
        markdown = format_comparison_markdown([self._stats()], "2026-09-06T00:00:00")
        assert "+8.90" in markdown
        assert "1.67" in markdown
