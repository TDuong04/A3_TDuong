"""Tests for `eval.play_gridworld` — the script the Part I video records.

The window itself cannot be asserted on, so these tests check the two things that decide whether the
footage shows what rubric row V asks for: that the app is wired to the *trained* tables (arrows on,
both algorithms under `--compare`, the acting policy reading the selected one), and that the greedy
policy actually solves the level rather than wandering. Everything runs under
`SDL_VIDEODRIVER=dummy` with `headless=True`, so no window ever opens.

The subprocess smoke tests at the end run the real CLI end to end with a frame budget, because a
module that imports cleanly can still fail on its first frame.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from common.seeding import make_rng  # noqa: E402
from eval import play_gridworld  # noqa: E402
from eval.play_gridworld import (  # noqa: E402
    ALGOS,
    COMPARE_LEVEL,
    ComparePlaybackApp,
    MissingQTableError,
    build_app,
    load_level_tables,
    parse_args,
    q_table_path,
)
from gridworld.env import GridWorld  # noqa: E402
from gridworld.levels import N_LEVELS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

#: The seed every shipped level-0..5 artifact was trained under.
TRAINED_SEED = 0

#: A level with monsters, whose movement is the stochasticity the video has to show.
MONSTER_LEVEL = 4

#: Enough frames for the loop to tick, draw and present several times without slowing the suite.
SMOKE_FRAMES = 20

#: Level 1 is 11 steps under the trained Q-learning policy; the cap only fires on a broken one.
MAX_POLICY_STEPS = 300


def _skip_without(level: int, algo: str) -> None:
    path = q_table_path(level, algo, TRAINED_SEED, RESULTS_DIR)
    if not path.is_file():
        pytest.skip(f"{path.name} has not been trained in this checkout")


@pytest.fixture
def compare_app() -> Iterator[ComparePlaybackApp]:
    """The `--compare` wiring, headless: level 1, both algorithms, a deterministic policy rng."""
    for algo in ALGOS:
        _skip_without(COMPARE_LEVEL, algo)
    app = build_app(
        COMPARE_LEVEL,
        ALGOS,
        seed=TRAINED_SEED,
        env_seed=0,
        results_dir=RESULTS_DIR,
        headless=True,
        cell_size=24,
        policy_rng=make_rng(0),
        app_class=ComparePlaybackApp,
    )
    yield app
    app.renderer.close()


# --- the CLI ------------------------------------------------------------------------------------


def test_defaults_play_one_algorithm_on_level_zero():
    args = parse_args([])

    assert (args.level, args.algo, args.algos) == (0, "q", ["q"])
    assert args.seed == TRAINED_SEED
    assert args.epsilon == 0.0, "playback is greedy unless asked otherwise"
    assert args.no_arrows is False, "the arrow overlay is the point; it starts on"


def test_compare_selects_both_algorithms_and_defaults_to_the_cliff_level():
    args = parse_args(["--compare"])

    assert args.algos == list(ALGOS)
    assert args.level == COMPARE_LEVEL


def test_an_explicit_level_survives_compare():
    assert parse_args(["--compare", "--level", "3"]).level == 3


@pytest.mark.parametrize("algo", ALGOS)
def test_algo_choices_are_the_two_trained_algorithms(algo):
    assert parse_args(["--algo", algo]).algos == [algo]


@pytest.mark.parametrize("bad", (["--level", str(N_LEVELS)], ["--level", "-1"], ["--epsilon", "2"]))
def test_out_of_range_arguments_are_rejected_by_the_parser(bad):
    with pytest.raises(SystemExit):
        parse_args(bad)


def test_q_table_path_matches_what_training_writes():
    """The naming scheme is shared with `train.train_gridworld.stem_for`, not guessed at."""
    path = q_table_path(1, "sarsa", 0, RESULTS_DIR)

    assert path.name == "qtable_level1_sarsa_seed0.npz"
    assert path.is_file(), "the shipped level 1 SARSA table should be where playback looks for it"


# --- the untrained-table path -------------------------------------------------------------


def test_loading_an_untrained_table_raises_with_the_training_command(tmp_path):
    with pytest.raises(MissingQTableError) as caught:
        load_level_tables(1, "q", 4242, results_dir=tmp_path)

    message = str(caught.value)
    assert "python -m train.train_gridworld" in message
    assert "--seed 4242" in message


def test_main_reports_a_missing_table_as_a_message_not_a_traceback(tmp_path, capsys):
    argv = ["--level", "1", "--results-dir", str(tmp_path), "--frames", "1"]
    exit_code = play_gridworld.main(argv)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "no trained Q-table" in captured.err
    assert "Traceback" not in captured.err


def test_the_message_lists_seeds_that_do_exist():
    """Pointing at the seeds on disk is what turns the error into an instruction."""
    message = play_gridworld._missing_message(1, "q", 4242, RESULTS_DIR)

    assert "qtable_level1_q_seed0.npz" in message


# --- the wiring the video depends on -------------------------------------------------------


def test_policy_arrows_start_on(compare_app):
    assert compare_app.renderer.show_policy_arrows is True


def test_no_arrows_flag_starts_the_overlay_off():
    _skip_without(0, "q")
    app = build_app(
        0, ["q"], seed=TRAINED_SEED, results_dir=RESULTS_DIR, headless=True, show_arrows=False
    )
    try:
        assert app.renderer.show_policy_arrows is False
        app.renderer.toggle_policy_arrows()
        assert app.renderer.show_policy_arrows is True, "the overlay stays toggleable"
    finally:
        app.renderer.close()


def test_compare_loads_both_algorithms_as_separate_overlays(compare_app):
    assert compare_app.overlay_algos == ALGOS
    assert compare_app.algo == "q"
    assert compare_app.overlays["q"][COMPARE_LEVEL] is not compare_app.overlays["sarsa"][
        COMPARE_LEVEL
    ]


def test_tab_swaps_both_the_overlay_and_the_acting_policy(compare_app):
    """A route walked by one table under the other table's arrows would be a lie."""
    before = compare_app.q_table
    compare_app.cycle_overlay()

    assert compare_app.algo == "sarsa"
    assert compare_app.q_table is compare_app.overlays["sarsa"][COMPARE_LEVEL]
    assert compare_app.q_table is not before
    assert compare_app.policy.app is compare_app, "the policy reads the app's current selection"


def test_tab_restarts_the_episode_so_both_routes_start_from_the_same_state(compare_app):
    start = compare_app.env.agent_pos
    for _ in range(3):
        compare_app.policy_step()
    assert compare_app.env.agent_pos != start

    compare_app.cycle_overlay()

    assert compare_app.env.agent_pos == start
    assert compare_app.env.steps == 0


def test_the_two_algorithms_walk_different_routes_on_level_one(compare_app):
    """Row C3's claim, animated: SARSA detours from the cliff edge Q-learning hugs."""
    routes = {}
    for algo in ALGOS:
        while compare_app.algo != algo:
            compare_app.cycle_overlay()
        compare_app.reset()
        routes[algo] = _run_episode(compare_app)

    assert routes["q"] != routes["sarsa"]
    assert len(routes["q"]) < len(routes["sarsa"]), "Q-learning takes the shorter, riskier route"


def _run_episode(app) -> list[tuple[int, int]]:
    """Step the app's own policy to the end of the episode, returning the cells visited."""
    path = [app.env.agent_pos]
    for _ in range(MAX_POLICY_STEPS):
        if app.env.done:
            break
        app.policy_step()
        path.append(app.env.agent_pos)
    return path


# --- does it actually play a learned policy? -----------------------------------------------


@pytest.mark.parametrize("algo", ALGOS)
def test_the_greedy_policy_solves_level_one_rather_than_wandering(algo):
    _skip_without(COMPARE_LEVEL, algo)
    app = build_app(
        COMPARE_LEVEL,
        [algo],
        seed=TRAINED_SEED,
        env_seed=0,
        results_dir=RESULTS_DIR,
        headless=True,
        policy_rng=make_rng(0),
    )
    try:
        _run_episode(app)
        assert app.env.died is False
        assert app.env.truncated is False
        assert app.env.collected_mask == app.env.full_mask, "every collectible was taken"
    finally:
        app.renderer.close()


def test_switching_to_a_level_with_no_table_says_so_instead_of_pretending(tmp_path):
    """The HUD must not present a random walk as a learned policy."""
    _skip_without(0, "q")
    app = build_app(
        0, ["q"], seed=TRAINED_SEED, env_seed=0, results_dir=RESULTS_DIR, headless=True
    )
    try:
        app.overlays["q"] = {}  # as if only some levels had been trained
        app.policy_step()
        assert "no Q-table" in app.message
    finally:
        app.renderer.close()


# --- monsters -----------------------------------------------------------------------------


def test_playback_runs_a_level_with_monsters():
    _skip_without(MONSTER_LEVEL, "q")
    app = build_app(
        MONSTER_LEVEL,
        ["q"],
        seed=TRAINED_SEED,
        env_seed=1,
        results_dir=RESULTS_DIR,
        headless=True,
        policy_rng=make_rng(0),
    )
    try:
        assert app.env.monster_positions, "level 4 is the monster level"
        _run_episode(app)
        assert app.env.done is True
    finally:
        app.renderer.close()


def test_monster_movement_stays_stochastic_across_env_seeds():
    """`--env-seed` is what a viewer changes to see a different draw; the same policy must not
    produce the same monster trace."""
    _skip_without(MONSTER_LEVEL, "q")
    traces = []
    for env_seed in (0, 1):
        app = build_app(
            MONSTER_LEVEL,
            ["q"],
            seed=TRAINED_SEED,
            env_seed=env_seed,
            results_dir=RESULTS_DIR,
            headless=True,
            policy_rng=make_rng(0),
        )
        try:
            trace = []
            for _ in range(10):
                app.policy_step()
                trace.append(app.env.monster_positions)
            traces.append(trace)
        finally:
            app.renderer.close()

    assert traces[0] != traces[1]


def test_an_unseeded_env_leaves_monster_movement_to_chance():
    """`env_seed=None` is the default so consecutive demo runs are not identical."""
    _skip_without(MONSTER_LEVEL, "q")
    env = GridWorld(level_index=MONSTER_LEVEL, seed=None)
    assert env.monster_positions


# --- the real CLI, one frame budget at a time ----------------------------------------------

SMOKE_ENV = {**os.environ, "SDL_VIDEODRIVER": "dummy", "PYTHONPATH": str(REPO_ROOT)}


def _run_cli(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "eval.play_gridworld", *argv],
        cwd=REPO_ROOT,
        env=SMOKE_ENV,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.parametrize(
    "argv",
    (
        ("--level", "1", "--algo", "q"),
        ("--level", "1", "--algo", "sarsa"),
        ("--level", str(MONSTER_LEVEL), "--algo", "q"),
        ("--compare",),
    ),
)
def test_the_documented_commands_run_a_window_loop(argv):
    result = _run_cli(*argv, "--frames", str(SMOKE_FRAMES))

    assert result.returncode == 0, result.stderr
    assert "Controls:" in result.stdout, "the key map is printed so a viewer knows what to press"
