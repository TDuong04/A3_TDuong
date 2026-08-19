"""Tests for `train.train_gridworld` — the artifacts are half the deliverable.

A run that trains correctly but leaves nothing behind cannot be reported: the plan's own risk list
says a curve without a seed in its filename is not evidence. These tests pin the config merge, the
filenames and the fact that every promised file is actually written.

The training here is deliberately tiny (a few hundred episodes on level 0) so the suite stays fast;
the full-length run lives in the CLI, not in pytest.
"""

from __future__ import annotations

import csv
import json
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from common.config import TabularConfig, load_yaml
from gridworld.algorithms import HISTORY_FIELDS, load_q_table, optimal_collection_steps
from train import train_gridworld

#: A shortened but not fragile run. 600 episodes reaches the optimum on most seeds and misses on
#: some; 1200 with a proportionally shortened decay window reaches it on every seed tried (0-5),
#: which is what a test needs. The full-length numbers stay in `config/gridworld.yaml`.
FAST = dict(episodes=1200, epsilon_decay_episodes=800, seed=0)


@pytest.fixture
def scratch_dir() -> Iterator[Path]:
    """Not `tmp_path`: the shared `pytest-of-<user>` base dir is unreadable on this machine."""
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


def test_config_for_level_merges_level_overrides():
    """Level 1's longer decay window must survive the merge, and untouched fields must not move."""
    raw = load_yaml("gridworld")
    base, override = raw["training"], raw["level_overrides"][1]

    merged = train_gridworld.config_for_level(1)
    assert merged.episodes == override["episodes"]
    assert merged.epsilon_decay_episodes == override["epsilon_decay_episodes"]
    assert merged.alpha == base["alpha"]
    assert merged.gamma == base["gamma"]


def test_config_for_level_without_overrides_is_the_training_block():
    assert train_gridworld.config_for_level(0) == TabularConfig.from_yaml()


def test_cli_overrides_travel_inside_a_config_object():
    """`--episodes`/`--seed` narrow the config rather than being passed as loose literals."""
    args = train_gridworld.parse_args(["--level", "0", "--seed", "7", "--episodes", "12"])
    config = train_gridworld.build_config(args)
    assert isinstance(config, TabularConfig)
    assert (config.seed, config.episodes) == (7, 12)
    assert config.epsilon_start == TabularConfig.from_yaml().epsilon_start


def test_filenames_record_the_seed():
    """A curve that cannot name its seed cannot be reproduced, so the seed is in the name."""
    assert train_gridworld.stem_for(level=0, algo="q", seed=3) == "level0_q_seed3"


def test_run_writes_every_promised_artifact_and_reaches_the_optimum(scratch_dir):
    config = replace(TabularConfig.from_yaml(), **FAST)
    summary = train_gridworld.run(0, "q", config, results_dir=scratch_dir)

    # B5: the greedy policy matches the independently computed shortest route.
    assert summary["bfs_optimum_steps"] == optimal_collection_steps(0)
    assert summary["greedy_collected"] == summary["n_collectibles"] == 3
    assert summary["greedy_steps"] == summary["bfs_optimum_steps"]
    assert summary["optimal"] is True

    # B6: the evidence exists on disk, and every filename carries level, algorithm and seed.
    expected = {"history_csv", "q_table", "training_curve", "policy_arrows", "summary_json"}
    assert set(summary["artifacts"]) == expected
    assert set(summary["paths"]) == expected
    for path in summary["paths"].values():
        assert path.exists() and path.stat().st_size > 0
        assert "level0_q_seed0" in path.name

    history = summary["paths"]["history_csv"].read_text(encoding="utf-8").splitlines()
    rows = list(csv.DictReader(history))
    assert len(rows) == config.episodes
    assert list(rows[0]) == list(HISTORY_FIELDS)

    saved = json.loads(summary["paths"]["summary_json"].read_text(encoding="utf-8"))
    assert saved["seed"] == 0 and saved["greedy_steps"] == saved["bfs_optimum_steps"]

    # The Q-table has to reload, or playback would need a retrain it cannot afford in a demo.
    assert len(load_q_table(summary["paths"]["q_table"])) == len(summary["result"].q_table)


# --- C3: the level 1 comparison ------------------------------------------------------------------


def test_death_rate_counts_deaths_under_the_policy_it_is_given():
    """The death rate has to be a property of the table, or the C3 claim measures nothing.

    Two synthetic tables with known outcomes on level 1: RIGHT walks off the start straight into
    the fire on every rollout, UP walks away from it and never dies. A measurement that could not
    tell those apart could not tell Q-learning from SARSA either.
    """
    config = TabularConfig.from_yaml()
    into_the_fire = defaultdict(lambda: np.array([0.0, 0.0, 0.0, 1.0]))  # RIGHT
    away_from_it = defaultdict(lambda: np.array([1.0, 0.0, 0.0, 0.0]))   # UP

    fatal = train_gridworld.death_rate(
        1, into_the_fire, epsilon=0.0, max_steps=config.max_steps_per_episode, rollouts=25
    )
    safe = train_gridworld.death_rate(
        1, away_from_it, epsilon=0.0, max_steps=config.max_steps_per_episode, rollouts=25
    )
    assert fatal["death_rate"] == 1.0 and fatal["deaths"] == fatal["rollouts"] == 25
    assert safe["death_rate"] == 0.0 and safe["deaths"] == 0


def test_death_rate_rises_with_epsilon_on_the_cliff():
    """Exploration is the risk SARSA prices in, so the measurement must respond to epsilon.

    The table hugs the fire (RIGHT along row 7); greedy it never falls in, and with exploration
    switched on it sometimes does. A rate that ignored epsilon would report the same number for
    both algorithms and the comparison would be vacuous.
    """
    config = TabularConfig.from_yaml()
    hug_the_edge = defaultdict(lambda: np.array([0.0, 0.0, 0.0, 1.0]))  # RIGHT
    hug_the_edge.update({(7, col, False, 0): np.array([0.0, 0.0, 0.0, 1.0]) for col in range(10)})
    # From the start, go up once, then right along row 7 — the cliff-edge route.
    hug_the_edge[(8, 0, False, 0)] = np.array([1.0, 0.0, 0.0, 0.0])
    hug_the_edge[(7, 9, False, 0)] = np.array([0.0, 1.0, 0.0, 0.0])

    greedy = train_gridworld.death_rate(
        1, hug_the_edge, epsilon=0.0, max_steps=config.max_steps_per_episode, rollouts=50
    )
    exploring = train_gridworld.death_rate(
        1, hug_the_edge, epsilon=config.epsilon_end,
        max_steps=config.max_steps_per_episode, rollouts=200, seed_offset=1000,
    )
    assert greedy["death_rate"] == 0.0, "the greedy edge route never enters the fire"
    assert exploring["death_rate"] > 0.0, (
        "walking beside the fire with epsilon > 0 must sometimes kill the agent; "
        f"got {exploring}"
    )


def test_compare_trains_both_algorithms_under_one_identical_config(scratch_dir):
    """C3's structural half: one config object, both algorithms, every artifact on disk.

    Deliberately a short run — whether the two routes actually separate is a question about level 1
    at full length, answered by the figure in `results/` and asserted nowhere here, because a test
    that needs 8000 episodes per algorithm is a test nobody runs.
    """
    config = replace(TabularConfig.from_yaml(), episodes=150, epsilon_decay_episodes=100, seed=4)
    comparison = train_gridworld.compare(1, config, results_dir=scratch_dir, rollouts=10)

    assert set(comparison["algorithms"]) == {"q", "sarsa"}
    assert comparison["seed"] == 4 and comparison["episodes"] == 150
    assert comparison["bfs_optimum_steps"] == optimal_collection_steps(1) == 11

    # Identical seeds and identical schedules: both runs carry the same config object.
    configs = {algo: summary["result"].config for algo, summary in comparison["summaries"].items()}
    assert configs["q"] == configs["sarsa"] == config
    epsilons = {algo: summary["result"].epsilons for algo, summary in
                comparison["summaries"].items()}
    assert epsilons["q"] == pytest.approx(epsilons["sarsa"])
    assert comparison["summaries"]["sarsa"]["result"].algo == "sarsa"

    for entry in comparison["algorithms"].values():
        for sample in entry["death_rates"].values():
            assert sample["rollouts"] == 10
            assert 0.0 <= sample["death_rate"] <= 1.0

    expected = {"comparison_figure", "comparison_markdown", "comparison_json"}
    assert set(comparison["artifacts"]) == expected
    for path in comparison["paths"].values():
        assert path.exists() and path.stat().st_size > 0
        assert "level1_seed4" in path.name

    saved = json.loads(comparison["paths"]["comparison_json"].read_text(encoding="utf-8"))
    assert set(saved["algorithms"]) == {"q", "sarsa"}
    assert "summaries" not in saved  # the live objects must not leak into the JSON

    markdown = comparison["paths"]["comparison_markdown"].read_text(encoding="utf-8")
    assert "SARSA" in markdown and "Q-learning" in markdown
    assert "death rate" in markdown


def test_compare_cli_reaches_the_comparison_path(scratch_dir, capsys):
    train_gridworld.main([
        "--level", "1", "--compare",
        "--episodes", "120", "--epsilon-decay-episodes", "90",
        "--death-rate-rollouts", "5", "--seed", "2",
        "--results-dir", str(scratch_dir),
    ])
    out = capsys.readouterr().out
    assert "comparison" in out and "seed 2" in out
    assert "identical for both algorithms" in out
    assert (scratch_dir / "compare_level1_seed2.png").exists()
    assert (scratch_dir / "comparison_level1_seed2.md").exists()


def test_epsilon_end_override_travels_inside_the_config():
    """The knob that decides whether the comparison is meaningful is still config-shaped."""
    args = train_gridworld.parse_args(["--level", "1", "--epsilon-end", "0.2"])
    config = train_gridworld.build_config(args)
    assert isinstance(config, TabularConfig)
    assert config.epsilon_end == 0.2
    assert config.epsilon_start == TabularConfig.from_yaml().epsilon_start


def test_main_runs_end_to_end_via_the_documented_cli(scratch_dir, capsys):
    train_gridworld.main([
        "--level", "0", "--algo", "q",
        "--episodes", "1200", "--epsilon-decay-episodes", "800",
        "--seed", "1", "--results-dir", str(scratch_dir),
    ])
    out = capsys.readouterr().out
    assert "seed 1" in out
    assert "BFS optimum:    17" in out
    assert "verdict: OPTIMAL" in out
    assert (scratch_dir / "curve_level0_q_seed1.png").exists()
