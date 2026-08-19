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
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

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
