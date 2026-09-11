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

import hashlib
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
    format_timelapse_markdown,
    has_trained_models,
    human_action,
    load_policy,
    main,
    model_label,
    play,
    resolve_model,
    shipped_checkpoints,
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

    def test_the_command_names_the_sample_and_seed_it_measured(self):
        """A 3-episode table and a 30-episode one are different evidence (A3-032)."""
        markdown = format_comparison_markdown([self._stats()], "2026-09-06T00:00:00")
        assert "--episodes 3 --seed 0`" in markdown

    def test_mechanics_tables_carry_the_shield_and_elite_columns(self):
        stats = {**self._stats(), "mechanics": True, "pickups_collected_mean": 1.77,
                 "shield_blocks_mean": 1.67, "elites_killed_mean": 0.8}
        markdown = format_comparison_markdown([stats], "2026-09-11T00:00:00")
        assert "Shields collected" in markdown and "Elite kills" in markdown
        assert "| 1.77 | 1.67 | 0.80 |" in markdown
        assert "--mechanics" in markdown

    def test_baseline_tables_keep_their_columns(self):
        markdown = format_comparison_markdown([self._stats()], "2026-09-06T00:00:00")
        assert "Shields collected" not in markdown


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


#: A 20-feature snapshot from the shipped direct run -- the first frame of the learning time-lapse.
SHIPPED_CHECKPOINT = REPO_ROOT / "models" / "checkpoints" / "ppo_direct_shipped_50000_steps.zip"
#: A 21-feature snapshot from before `spawner_exists` was dropped: a world that no longer exists.
STALE_CHECKPOINT = REPO_ROOT / "models" / "checkpoints" / "ppo_direct_50000_steps.zip"


@pytest.mark.skipif(not SHIPPED_CHECKPOINT.exists(), reason="shipped checkpoints not on disk")
class TestExactCheckpointPlayback:
    """`--model` plays one exact snapshot, for the learning time-lapse the video shows.

    The risk it adds is playing the wrong thing convincingly: a checkpoint from an older world, or
    a step count on screen that the network never reached. These pin both shut.
    """

    def test_the_exact_file_is_loaded_not_the_final_model(self):
        agent, policy = load_policy("direct", model_path=SHIPPED_CHECKPOINT)
        assert policy == "ppo"
        assert agent.num_timesteps == 50_000

    def test_the_hud_label_reads_the_step_count_from_the_model(self):
        agent, _ = load_policy("direct", model_path=SHIPPED_CHECKPOINT)
        assert model_label(agent) == "50,000 training steps"

    def test_human_and_random_play_carry_no_model_label(self):
        assert model_label(None) is None
        assert model_label(RandomPolicy(6, seed=0)) is None

    def test_a_missing_checkpoint_is_a_message_not_a_traceback(self):
        with pytest.raises(ArenaEvalError, match="does not exist"):
            load_policy("direct", model_path=SHIPPED_CHECKPOINT.with_name("absent.zip"))

    @pytest.mark.skipif(not STALE_CHECKPOINT.exists(), reason="stale checkpoint not on disk")
    def test_a_checkpoint_from_the_old_observation_is_refused(self):
        with pytest.raises(ArenaEvalError, match="spaces do not match"):
            evaluate("direct", episodes=1, seed=0, model_path=STALE_CHECKPOINT)

    def test_random_still_wins_over_an_explicit_checkpoint(self):
        agent, policy = load_policy("direct", model_path=SHIPPED_CHECKPOINT, random=True)
        assert isinstance(agent, RandomPolicy)
        assert policy == "random"

    def test_a_checkpoint_cannot_be_played_as_both_styles(self, capsys):
        assert main(["--style", "both", "--model", str(SHIPPED_CHECKPOINT), "--no-save"]) == 2
        assert "one style's checkpoint" in capsys.readouterr().err

    def test_evaluate_plays_the_checkpoint_it_was_given(self):
        stats = evaluate("direct", episodes=1, seed=0, model_path=SHIPPED_CHECKPOINT)
        assert stats["policy"] == "ppo"
        assert stats["training_steps"] == 50_000
        assert stats["model"] == "models/checkpoints/ppo_direct_shipped_50000_steps.zip"

    def test_random_with_a_checkpoint_is_filed_as_random(self):
        stats = evaluate("direct", episodes=1, seed=0, model_path=SHIPPED_CHECKPOINT,
                         random_policy=True)
        assert stats["policy"] == "random"
        assert "model" not in stats

    def test_a_checkpoint_run_never_writes_the_row_i_table(self):
        """`comparison.md` is row I's evidence; a half-trained snapshot must land beside it."""
        directory = EMPTY_RESULTS.parent / "__eval_checkpoint_write__"
        stats = evaluate("direct", episodes=1, seed=0, model_path=SHIPPED_CHECKPOINT)
        written = write_results([stats], directory)
        try:
            assert all(path.parent.name == "checkpoints" for path in written.values())
            assert not (directory / "comparison.md").exists()
            table = written["comparison"].read_text()
            assert "--model models/checkpoints/ppo_direct_shipped_50000_steps.zip" in table
            assert "not the shipped model" in table
        finally:
            for path in written.values():
                path.unlink()
            (directory / "checkpoints").rmdir()
            directory.rmdir()


@pytest.mark.skipif(not SHIPPED_CHECKPOINT.exists(), reason="shipped checkpoints not on disk")
class TestLearningTimelapse:
    """The table behind the video's "watch it learn" shot, and the files that shot replays."""

    def test_snapshots_are_ordered_by_training_steps_not_by_name(self):
        steps = [int(path.stem.rsplit("_", 2)[-2])
                 for path in shipped_checkpoints("direct") if "_shipped_" in path.name]
        assert steps == sorted(steps)
        assert steps[0] == 50_000  # as text, "100000" would sort first

    def test_no_model_appears_twice(self):
        digests = [hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in shipped_checkpoints("direct")]
        assert len(digests) == len(set(digests))

    def test_the_table_starts_at_the_random_floor(self):
        rows = [evaluate("direct", episodes=1, seed=0, random_policy=True),
                evaluate("direct", episodes=1, seed=0, model_path=SHIPPED_CHECKPOINT)]
        markdown = format_timelapse_markdown("direct", rows, "2026-09-11T00:00:00")
        table = [line for line in markdown.splitlines() if line.startswith("| ")]
        assert table[1].startswith("| 0 (random policy) |")  # table[0] is the header row
        assert table[2].startswith("| 50,000 | `ppo_direct_shipped_50000_steps.zip` |")
        assert "--timelapse --episodes 1 --seed 0`" in markdown

    def test_the_timelapse_refuses_flags_it_would_ignore(self, capsys):
        assert main(["--timelapse", "--mechanics", "--no-save"]) == 2
        assert "drop --mechanics" in capsys.readouterr().err


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
