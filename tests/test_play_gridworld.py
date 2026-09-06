"""Headless tests for `eval.play_gridworld`.

This is the script the video records for Part I, and the video is worth 5 rubric points, so the
things asserted here are the things a viewer has to be able to see: that the arrows are on by
default, that they actually reach the screen, that `--compare` really runs two independent
policies rather than drawing the same one twice, and that a marker who runs a README command on a
fresh clone without trained tables gets a sentence rather than a traceback.

Every test runs under `SDL_VIDEODRIVER=dummy` and draws to an off-screen surface, following
`tests/test_gridworld_render.py`. That file's history is the reason the assertions below read
pixels: 63 of its tests once asserted only `surface.get_size() == renderer.surface_size(...)`, a
tautology that let six visual regressions ship green. "The surface has the expected size" is not
evidence that anything was drawn on it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402  (must follow the driver assignment above)

from gridworld.algorithms import make_q_table, save_q_table  # noqa: E402
from gridworld.constants import Action, N_ACTIONS  # noqa: E402
from gridworld.env import GridWorld  # noqa: E402
from gridworld.render import COLOR_ARROW  # noqa: E402

from eval.play_gridworld import (  # noqa: E402
    COMPARE_LEVEL,
    GREEDY_EPSILON,
    GreedyPolicy,
    PlaybackError,
    build_app,
    build_compare_app,
    find_q_table,
    load_learner_info,
    load_policy_tables,
    resolve_q_table,
    trained_seeds,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"

#: A directory with no Q-tables in it. It deliberately does not exist: `glob` on a missing
#: directory yields nothing, which is exactly the "fresh clone, nothing trained yet" case, and
#: `tmp_path` is unusable on this machine (WinError 5 on the pytest temp root).
EMPTY_RESULTS = REPO_ROOT / "results" / "__not_a_directory__"


@pytest.fixture(scope="module", autouse=True)
def _pygame_session():
    pygame.init()
    yield
    pygame.quit()


def _colour_pixels(surface: pygame.Surface, colour: tuple[int, int, int]) -> int:
    """Exact-match pixel count. Safe because every shape here is drawn with `pygame.draw`, which
    does not antialias; the only antialiased pixels on the surface are font glyphs."""
    array = pygame.surfarray.array3d(surface)
    return int(np.count_nonzero(np.all(array == np.array(colour), axis=-1)))


def _nonzero_pixels(surface: pygame.Surface) -> int:
    """How many pixels are not the background — a crude but honest "something was drawn"."""
    array = pygame.surfarray.array3d(surface)
    background = array[0, 0]
    return int(np.count_nonzero(np.any(array != background, axis=-1)))


# --- locating trained tables ---------------------------------------------------------------------


class TestResolvingTables:
    def test_the_lowest_trained_seed_wins_when_none_is_asked_for(self):
        """The README commands name no seed, so they must work without anyone remembering which
        seeds were run."""
        seeds = trained_seeds(RESULTS, COMPARE_LEVEL, "q")
        assert seeds, "level 1 Q-learning should be trained in results/"
        assert seeds == sorted(seeds)
        chosen = find_q_table(RESULTS, COMPARE_LEVEL, "q")
        assert chosen is not None
        assert f"_seed{seeds[0]}.npz" in chosen.name

    def test_an_untrained_table_is_a_message_not_a_traceback(self):
        """Criterion 5. A marker on a fresh clone hits this path, and `FileNotFoundError` tells
        them nothing about what to do next."""
        with pytest.raises(PlaybackError) as caught:
            resolve_q_table(EMPTY_RESULTS, 0, "q")
        message = str(caught.value)
        assert "python -m train.train_gridworld --level 0 --algo q" in message, message
        assert "No trained Q-table" in message

    def test_the_message_distinguishes_an_untrained_seed_from_an_untrained_level(self):
        """Asking for a seed that was never run is a different mistake from a level that was never
        trained, and the fix is a different command."""
        with pytest.raises(PlaybackError) as caught:
            resolve_q_table(RESULTS, COMPARE_LEVEL, "q", seed=997)
        message = str(caught.value)
        assert "997" in message
        assert "--seed 997" in message, "the suggested command must reproduce the seed asked for"

    def test_loading_never_raises_a_bare_oserror(self):
        with pytest.raises(PlaybackError):
            load_policy_tables(EMPTY_RESULTS, "sarsa", required_level=0)


# --- the greedy policy ---------------------------------------------------------------------------


class TestGreedyPolicy:
    def test_playback_is_greedy(self):
        """Playback shows the learned policy; an exploring agent on camera is exactly the
        "acting randomly" the video rubric warns about."""
        assert GREEDY_EPSILON == 0.0

    def test_the_greedy_action_is_the_argmax_of_the_row(self):
        env = GridWorld(level_index=0)
        env.reset()
        table = make_q_table()
        row = np.full(N_ACTIONS, -5.0)
        row[int(Action.RIGHT)] = 9.0
        table[env.state] = row
        policy = GreedyPolicy({0: table})
        assert policy(env) == int(Action.RIGHT)

    def test_an_unseen_state_does_not_grow_the_trained_table(self):
        """`make_q_table` returns a defaultdict, so indexing it for an unseen state would silently
        add a zero row per step — the table on screen would stop being the table that was trained.
        """
        env = GridWorld(level_index=0)
        env.reset()
        table = make_q_table()
        table[env.state] = np.zeros(N_ACTIONS)
        policy = GreedyPolicy({0: table})

        # Move somewhere the table has never seen. Asserting the state really is absent first is
        # the point: an earlier version of this test parked the agent on a cell that was already
        # in the table, so a mutant using `table[state]` instead of `table.get(state)` survived.
        row, col = env.agent_pos
        env.agent_pos = (row + 1, col) if row + 1 < env.n_rows else (row - 1, col)
        assert env.state not in table

        before = len(table)
        for _ in range(25):
            policy(env)
        assert len(table) == before, "playback grew the trained table with zero rows"

    def test_a_level_without_a_trained_table_is_reported_not_faked(self):
        policy = GreedyPolicy({0: make_q_table()})
        assert policy.has_policy_for(0)
        assert not policy.has_policy_for(5)
        assert policy.table_for(5) is None


# --- what reaches the screen ---------------------------------------------------------------------


class TestArrowOverlay:
    def test_arrows_are_on_by_default(self):
        """Criterion 2. The overlay is the evidence that the agent follows a learned policy, so it
        has to be visible without anyone pressing a key during a recording."""
        app = build_app(COMPARE_LEVEL, "q", results_dir=RESULTS, headless=True)
        assert app.renderer.show_policy_arrows is True

    @pytest.mark.parametrize("wanted", [True, False])
    def test_the_arrows_flag_is_honoured_in_both_directions(self, wanted):
        """Both directions, deliberately. Asserting only the `False` case lets an implementation
        that ignores the argument and always turns arrows off pass -- a mutant that did exactly
        that survived the first version of this test."""
        app = build_app(
            COMPARE_LEVEL, "q", results_dir=RESULTS, headless=True, show_arrows=wanted
        )
        assert app.renderer.show_policy_arrows is wanted

    def test_toggling_the_overlay_changes_the_pixels(self):
        """Not "the flag flipped" — the picture has to differ, or the toggle is decorative."""
        app = build_app(COMPARE_LEVEL, "q", results_dir=RESULTS, headless=True)
        app.renderer.show_policy_arrows = True
        with_arrows = _colour_pixels(app.draw(), COLOR_ARROW)
        app.renderer.show_policy_arrows = False
        without_arrows = _colour_pixels(app.draw(), COLOR_ARROW)
        assert with_arrows > 0, "no arrow-coloured pixels reached the screen at all"
        assert without_arrows == 0, without_arrows


# --- the comparison, animated --------------------------------------------------------------------


class TestCompareApp:
    def test_the_window_is_wide_enough_for_both_panels(self):
        app = build_compare_app(COMPARE_LEVEL, results_dir=RESULTS, headless=True)
        panels = app.panel_sizes()
        assert len(panels) == 2
        width, height = app.surface_size()
        assert width >= sum(size[0] for size in panels)
        assert height > max(size[1] for size in panels), "the title strip needs its own height"

    def test_both_panels_advance_and_they_are_independent_envs(self):
        """`--compare` is worthless if it draws one policy twice. Distinct env objects, and both
        actually stepping."""
        app = build_compare_app(COMPARE_LEVEL, results_dir=RESULTS, headless=True)
        first, second = app.apps
        assert first.env is not second.env
        assert first.policy is not second.policy
        for _ in range(12):
            first.policy_step()
            second.policy_step()
        assert first.env.steps > 0 and second.env.steps > 0

    def test_the_two_panels_run_the_two_different_algorithms(self):
        """Rubric row C is graded on the two algorithms differing, so the labels on screen must not
        both say the same thing."""
        app = build_compare_app(COMPARE_LEVEL, results_dir=RESULTS, headless=True)
        labels = {a.label for a in app.apps}
        assert labels == {"Q-LEARNING", "SARSA"}, labels

    def test_both_halves_of_the_window_have_something_drawn_on_them(self):
        app = build_compare_app(COMPARE_LEVEL, results_dir=RESULTS, headless=True)
        surface = app.draw()
        width, height = surface.get_size()
        left = surface.subsurface((0, 0, width // 2, height)).copy()
        right = surface.subsurface((width // 2, 0, width - width // 2, height)).copy()
        assert _nonzero_pixels(left) > 0
        assert _nonzero_pixels(right) > 0


# --- monsters ------------------------------------------------------------------------------------


class TestMonsterLevels:
    @pytest.mark.parametrize("level", [4, 5])
    def test_a_monster_level_is_playable(self, level):
        """Criterion 4. These are the levels whose stochastic transitions the video has to show."""
        app = build_app(level, "sarsa", results_dir=RESULTS, headless=True)
        assert app.env.monsters, f"level {level} should have monsters"
        for _ in range(20):
            if app.env.done:
                break
            app.policy_step()
        assert _nonzero_pixels(app.draw()) > 0

    def test_monsters_move_across_replays_when_the_env_seed_is_free(self):
        """With no `--env-seed` the monsters take a fresh walk each replay while the policy stays
        fixed, which is what makes the stochasticity visible on camera rather than a fixed film."""
        positions = set()
        for _ in range(8):
            app = build_app(4, "sarsa", results_dir=RESULTS, headless=True)
            for _ in range(12):
                if app.env.done:
                    break
                app.policy_step()
            positions.add(tuple(sorted(app.env.monsters)))
        assert len(positions) > 1, "monster positions never differed across replays"


# --- the learner block's data source ------------------------------------------------------------


class TestLearnerInfoLoading:
    """What the HUD says about the policy has to come from the run that produced it."""

    def test_the_training_summary_is_read_from_beside_the_table(self):
        info = load_learner_info(RESULTS / "qtable_level1_q_seed0.npz")
        assert info is not None
        rows = dict(info.lines())
        assert rows["algorithm"] == "Q-learning"
        assert "alpha" in rows and "gamma" in rows and "epsilon" in rows

    def test_the_schedule_shown_is_the_one_that_trained_the_table(self):
        """Not `config/gridworld.yaml`. The config says what a run today would use, which becomes
        a different — and false — claim the moment anyone edits it."""
        summary = json.loads((RESULTS / "summary_level1_q_seed0.json").read_text())
        rows = dict(load_learner_info(RESULTS / "qtable_level1_q_seed0.npz").lines())
        assert rows["epsilon"] == f"{summary['epsilon_start']:g} -> {summary['epsilon_end']:g}"
        assert rows["alpha"] == f"{summary['alpha']:g}"

    def test_an_intrinsic_run_falls_back_to_what_the_filename_proves(self):
        """The level-6 sweep writes no per-run summary, but its filename carries the strength —
        which is the one number rubric row F is about."""
        info = load_learner_info(RESULTS / "qtable_level6_q_strength0p25_seed0.npz")
        rows = dict(info.lines())
        assert rows["intrinsic"] == "0.25"
        assert rows["algorithm"] == "Q-learning"
        assert "alpha" not in rows  # never borrowed from config

    def test_a_table_that_says_nothing_gets_no_panel(self):
        assert load_learner_info(EMPTY_RESULTS / "qtable_level9_zzz_seed0.npz") is None

    def test_a_corrupt_summary_costs_the_panel_not_the_demo(self, monkeypatch):
        broken = RESULTS / "qtable_level1_q_seed0.npz"
        monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: "{not json")
        assert load_learner_info(broken) is not None  # falls back to the filename

    def test_playback_puts_the_learner_on_screen(self):
        """End to end: the app built for a marker to watch carries the block."""
        app = build_app(1, "sarsa", seed=0, headless=True)
        assert app.status()["learner"] is not None
        assert dict(app.status()["learner"].lines())["algorithm"] == "SARSA"

    def test_both_compare_panels_show_their_own_algorithm(self):
        """Rubric C2 is "the same exploration schedule": two panels, two algorithm names, one
        identical epsilon line — visible side by side rather than asserted in prose."""
        app = build_compare_app(COMPARE_LEVEL, seed=0, headless=True)
        rows = [dict(panel.status()["learner"].lines()) for panel in app.apps]
        assert [r["algorithm"] for r in rows] == ["Q-learning", "SARSA"]
        assert rows[0]["epsilon"] == rows[1]["epsilon"]
