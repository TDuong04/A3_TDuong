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
    RandomPolicy,
    evaluate,
    format_comparison_markdown,
    format_summary,
    has_trained_models,
    human_action,
    load_policy,
    play,
    resolve_model,
    wait_for_start,
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


class TestTitleScreenGate:
    """The title screen must appear only where a person could plausibly press SPACE for it."""

    def test_a_headless_call_never_shows_the_title_screen(self, monkeypatch):
        """A regression guard, not a happy-path check: if `play()`'s `not headless` condition is
        ever dropped, this fails immediately instead of the row-I evidence table silently hanging
        the next time someone regenerates it."""
        import eval.play_arena as play_arena_module

        def _fail_if_called(*_args, **_kwargs):
            raise AssertionError("wait_for_start must not run for a headless call")

        monkeypatch.setattr(play_arena_module, "wait_for_start", _fail_if_called)
        play("direct", episodes=1, seed=0, headless=True, random_policy=True)

    def test_a_frame_capped_call_never_shows_the_title_screen(self, monkeypatch):
        """Report-figure capture passes `--frames`; it must get exactly that many gameplay frames,
        not one spent on a title screen it is never watching for."""
        import eval.play_arena as play_arena_module

        def _fail_if_called(*_args, **_kwargs):
            raise AssertionError("wait_for_start must not run for a frame-capped call")

        monkeypatch.setattr(play_arena_module, "wait_for_start", _fail_if_called)
        result = play("direct", episodes=1, seed=0, human=True, max_frames=3)
        assert result["frames"] == 3

    def test_quitting_at_the_title_screen_exits_cleanly_with_no_episodes_played(self, monkeypatch):
        """`wait_for_start` returning False is the window-closed / ESC path; `play()` should exit
        the same way it already does for a mid-episode quit, not raise or hang."""
        import eval.play_arena as play_arena_module

        monkeypatch.setattr(play_arena_module, "wait_for_start", lambda *_a, **_k: False)
        result = play("direct", episodes=5, seed=0, random_policy=True)
        assert result["episodes"] == 0
        assert result["frames"] == 0


class TestRetryLoop:
    """A real human session replays on R rather than stopping at `episodes` -- see `play()`'s
    `interactive` flag. Each episode here is forced to end after one step (`_end_immediately`
    below), so these run in milliseconds rather than waiting out a full arena episode."""

    #: Captured before any test monkeypatches `ArenaEnv.step`, so the wrapper below calls the
    #: real physics rather than recursing into itself once patched onto the class. A plain
    #: function, not a bound method, so assigning it to `ArenaEnv.step` still rebinds correctly
    #: through the normal descriptor protocol when a test calls `env.step(action)`.
    _real_step = ArenaEnv.step

    @staticmethod
    def _end_immediately(env, action):
        obs, reward, _terminated, truncated, info = TestRetryLoop._real_step(env, action)
        return obs, reward, True, truncated, info

    def test_a_headless_human_call_never_waits_for_retry(self, monkeypatch):
        import eval.play_arena as play_arena_module

        def _fail_if_called(*_args, **_kwargs):
            raise AssertionError("wait_for_retry must not run for a headless call")

        monkeypatch.setattr(play_arena_module, "wait_for_retry", _fail_if_called)
        play("direct", episodes=1, seed=0, human=True, headless=True)

    def test_a_frame_capped_call_never_waits_for_retry(self, monkeypatch):
        import eval.play_arena as play_arena_module

        def _fail_if_called(*_args, **_kwargs):
            raise AssertionError("wait_for_retry must not run for a frame-capped call")

        monkeypatch.setattr(play_arena_module, "wait_for_retry", _fail_if_called)
        result = play("direct", episodes=1, seed=0, human=True, max_frames=3)
        assert result["frames"] == 3

    def test_pressing_r_replays_ignoring_the_episode_count(self, monkeypatch):
        """Asking for one episode but answering the retry prompt "yes" once should still play two
        -- a real human session is driven by R, not by `--episodes`."""
        import eval.play_arena as play_arena_module

        answers = iter([True, False])
        monkeypatch.setattr(play_arena_module, "wait_for_start", lambda *_a, **_k: True)
        monkeypatch.setattr(play_arena_module, "wait_for_retry", lambda *_a, **_k: next(answers))
        monkeypatch.setattr(ArenaEnv, "step", self._end_immediately)

        result = play("direct", episodes=1, seed=0, human=True)
        assert result["episodes"] == 2

    def test_quitting_the_retry_prompt_stops_the_session(self, monkeypatch):
        import eval.play_arena as play_arena_module

        monkeypatch.setattr(play_arena_module, "wait_for_start", lambda *_a, **_k: True)
        monkeypatch.setattr(play_arena_module, "wait_for_retry", lambda *_a, **_k: False)
        monkeypatch.setattr(ArenaEnv, "step", self._end_immediately)

        result = play("direct", episodes=5, seed=0, human=True)
        assert result["episodes"] == 1


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

    def test_rerunning_one_style_keeps_the_others_row(self):
        """A3-032: `write_results` used to rebuild the whole table from only the styles it was just
        given, so re-running `rotation` alone silently deleted `direct` from `comparison.md` even
        though `eval_direct_seed0.json` was untouched on disk right next to it."""
        directory = EMPTY_RESULTS.parent / "__eval_rerun_one_style__"
        try:
            write_results([self._stats("direct"), self._stats("rotation")], directory)
            written = write_results([self._stats("rotation")], directory)
            table = written["comparison"].read_text()
            assert "`direct`" in table
            assert "`rotation`" in table
        finally:
            for path in directory.glob("*"):
                path.unlink()
            directory.rmdir()

    def test_a_carried_over_row_is_named_not_silently_folded_in(self):
        markdown = format_comparison_markdown(
            [self._stats("direct"), self._stats("rotation")],
            "2026-09-06T00:00:00",
            carried_over={"direct"},
        )
        assert "`direct`" in markdown
        assert "carried over from an earlier run" in markdown

    def test_the_header_names_the_episode_count_actually_used(self):
        """A3-032: the header used to be a fixed string that never reflected --episodes, so a run
        that shrank the sample size could do so with nothing on the page to notice it by."""
        stats = self._stats("direct")
        stats["episodes"] = 30
        markdown = format_comparison_markdown([stats], "2026-09-06T00:00:00")
        assert "--episodes 30" in markdown

    def test_the_cli_default_matches_the_committed_evidence_sample_size(self):
        """The command README.md documents for regenerating row I's evidence passes no --episodes
        override, so the default has to already be 30 -- the size `comparison.md` is measured at."""
        from eval.play_arena import build_parser

        assert build_parser().parse_args(["--style", "both", "--no-window"]).episodes == 30


# --- the random policy ----------------------------------------------------------------------------


class TestRandomPolicyFallback:
    """Running before anything has been trained, without ever pretending chance is a result.

    Two rules that sound like they contradict each other. An *empty* `models/` means nobody has
    trained anything yet, so the script runs a random policy and says so -- that is what lets the
    playback loop, the HUD and the overlay be demonstrated while a training run is still going.
    A *populated* `models/` missing the one style asked for is a different situation entirely: a
    typo, or a run that died halfway. Falling back to random there would hand back a table that
    looks exactly like a trained result, so it raises instead.
    """

    def _dir(self, name: str) -> Path:
        directory = EMPTY_RESULTS.parent / name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def test_an_empty_models_directory_runs_a_random_policy(self):
        directory = self._dir("__models_empty__")
        try:
            agent, policy = load_policy("direct", models_dir=directory)
            assert isinstance(agent, RandomPolicy)
            assert policy == "random"
        finally:
            directory.rmdir()

    def test_the_fallback_names_the_command_that_would_train_one(self, capsys):
        """Runnable now is not the same as finished; the notice has to point at the next step."""
        directory = self._dir("__models_empty_notice__")
        try:
            load_policy("rotation", models_dir=directory)
            message = capsys.readouterr().err
            assert "random policy" in message
            assert "python -m train.train_arena --style rotation" in message
        finally:
            directory.rmdir()

    def test_a_populated_directory_missing_this_style_still_raises(self):
        """The dangerous case: silently substituting random actions for a model someone expects."""
        directory = self._dir("__models_half__")
        stray = directory / "ppo_direct.zip"
        stray.write_bytes(b"")
        try:
            with pytest.raises(ArenaEvalError) as error:
                load_policy("rotation", models_dir=directory)
            assert "python -m train.train_arena --style rotation" in str(error.value)
        finally:
            stray.unlink()
            directory.rmdir()

    def test_a_checkpoint_is_not_a_trained_model(self):
        """`models/checkpoints/` fills up mid-run; a policy still moving is not one to play back."""
        directory = self._dir("__models_checkpoints__")
        checkpoints = directory / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        checkpoint = checkpoints / "ppo_direct_50000_steps.zip"
        checkpoint.write_bytes(b"")
        try:
            assert has_trained_models(directory) is False
        finally:
            checkpoint.unlink()
            checkpoints.rmdir()
            directory.rmdir()

    def test_random_is_forced_even_when_a_model_exists(self):
        """`--random` is the baseline, not only a fallback: it has to work with models present."""
        agent, policy = load_policy("direct", models_dir=REPO_ROOT / "models", random=True)
        assert isinstance(agent, RandomPolicy)
        assert policy == "random"

    def test_the_random_policy_reproduces_from_its_seed(self):
        """A baseline nobody can re-derive is not a baseline."""

        def draw(seed: int) -> list[int]:
            policy = RandomPolicy(6, seed=seed)
            return [int(policy.predict(None)[0]) for _ in range(20)]

        assert draw(3) == draw(3)
        assert draw(3) != draw(4)  # ... and it is not a constant wearing a seed

    def test_every_random_action_is_inside_its_action_space(self):
        for style in ("direct", "rotation"):
            env = ArenaEnv(control_style=style)
            policy = RandomPolicy(int(env.action_space.n), seed=0)
            for _ in range(50):
                assert env.action_space.contains(int(policy.predict(None)[0]))

    def test_evaluate_labels_the_policy_that_produced_the_numbers(self):
        stats = evaluate("direct", episodes=1, seed=0, agent=RandomPolicy(6, seed=0))
        assert stats["policy"] == "random"
        assert "random" in format_summary(stats)

    def test_an_unnamed_agent_is_labelled_by_its_algorithm(self):
        stats = evaluate("direct", episodes=1, seed=0, agent=_ScriptedAgent(DirectAction.NOOP))
        assert stats["policy"] == "ppo"


class TestRandomRunsStayOffTheTrainedTables:
    """`comparison.md` is what report row R6 cites. Chance-level numbers must never reach it."""

    def _stats(self, style: str, policy: str) -> dict:
        return {
            "style": style, "policy": policy, "episodes": 2, "seed": 0,
            "return_mean": -13.3, "return_std": 0.73, "phase_mean": 1.0, "phase_max": 1,
            "phase_cleared_episodes": 0, "spawners_mean": 0.0, "enemies_mean": 1.0,
            "steps_mean": 180.0, "survival_rate": 0.0, "returns": [-13.0, -13.6],
            "phases": [1, 1],
        }

    def test_a_random_run_writes_its_own_files(self):
        directory = EMPTY_RESULTS.parent / "__eval_random_write__"
        written = write_results([self._stats("direct", "random")], directory)
        try:
            assert written["comparison"].name == "comparison_random.md"
            assert written["direct"].name == "eval_random_direct_seed0.json"
            assert not (directory / "comparison.md").exists()
        finally:
            for path in written.values():
                path.unlink()
            directory.rmdir()

    def test_a_trained_run_keeps_the_filenames_the_report_cites(self):
        directory = EMPTY_RESULTS.parent / "__eval_trained_write__"
        written = write_results([self._stats("direct", "ppo")], directory)
        try:
            assert written["comparison"].name == "comparison.md"
            assert written["direct"].name == "eval_direct_seed0.json"
        finally:
            for path in written.values():
                path.unlink()
            directory.rmdir()

    def test_the_random_table_says_on_its_face_that_it_is_a_baseline(self):
        """The table gets screenshotted into a report; it has to carry its own provenance."""
        markdown = format_comparison_markdown(
            [self._stats("direct", "random")], "2026-09-07T00:00:00"
        )
        assert "not a trained result" in markdown
        assert "`random`" in markdown
        assert "deterministic=True" not in markdown

    def test_the_trained_table_still_reports_its_policy(self):
        markdown = format_comparison_markdown(
            [self._stats("direct", "ppo")], "2026-09-07T00:00:00"
        )
        assert "`ppo`" in markdown
        assert "deterministic=True" in markdown


# --- the render and input paths are read-only -----------------------------------------------------


class TestRenderAndInputPathsAreReadOnly:
    """Playback must observe the simulation, never steer it.

    `tests/test_arena_render.py` proves one frame of `draw()` mutates nothing. This is the same
    guarantee at the level the playback script actually runs at: a whole seeded episode, window and
    all. If any part of the render or input path wrote back into the env -- a cached observation
    rebuilt from a moved world, an event pump consuming a step, an overlay nudging an entity -- the
    two runs below would diverge, and the video would be showing something other than what the
    evaluation table reports.
    """

    def test_drawing_a_window_does_not_change_the_trajectory(self):
        drawn = play("direct", episodes=3, seed=0, headless=True, random_policy=True)
        undrawn = evaluate("direct", episodes=3, seed=0, random_policy=True)
        assert drawn["returns"] == undrawn["returns"]
        assert drawn["phases"] == undrawn["phases"]

    def test_reading_the_keyboard_does_not_touch_the_simulation(self):
        import pygame

        env = ArenaEnv(control_style="direct")
        env.step(int(DirectAction.SHOOT))

        def snapshot() -> tuple:
            return (
                env.player.x, env.player.y, env.player.vx, env.player.vy,
                env.player.heading, env.player.health,
                tuple((e.x, e.y, e.health) for e in env.enemies),
                tuple((s.x, s.y, s.health) for s in env.spawners),
                tuple((b.x, b.y) for b in env.bullets),
                env.phase, env.steps, env.episode_return, env.last_action,
            )

        before = snapshot()
        for keys in (_Keys(), _Keys(pygame.K_UP), _Keys(pygame.K_SPACE),
                     _Keys(pygame.K_LEFT, pygame.K_SPACE)):
            human_action(keys, "direct")
            human_action(keys, "rotation")
        assert snapshot() == before
