"""Levels 2-5: the key/chest ordering, reward accounting, termination, and monster mechanics.

These are the mechanically checkable halves of A3-005 (rubric row D) and A3-006. The parts that
cannot live here — whether a full-length run converges, what the training curves look like — are
answered by the artifacts in `results/`, because a test that needs 12,000 episodes per algorithm is
a test nobody runs.

Every training run is shortened through `dataclasses.replace` on a `TabularConfig` read from
`config/gridworld.yaml`, never by editing the file and never by passing a bare literal into an
algorithm. `LEVEL_2_FAST` is the exception worth explaining: level 2 reaches the 25-step optimum on
both algorithms at 800/500, and 1200/800 was chosen for margin after checking seeds 0-3.

`tmp_path` is deliberately unused: the shared `pytest-of-<user>` base directory is unreadable on
this machine, so a local `tempfile.TemporaryDirectory` fixture stands in for it, matching
`tests/test_train_gridworld.py`.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from common.config import TabularConfig
from common.seeding import make_rng
from gridworld.algorithms import (
    QTable,
    greedy_rollout,
    optimal_collection_steps,
    q_learning,
    sarsa,
)
from gridworld.constants import COLLECTIBLE_TILES, MONSTER_MOVE_PROBABILITY, Tile
from gridworld.env import GridWorld
from gridworld.levels import load_level
from train import train_gridworld

#: Enough to reach level 2's 25-step optimum on both algorithms (verified on seeds 0-3) and small
#: enough to keep the suite quick. The full-length numbers stay in `config/gridworld.yaml`.
LEVEL_2_FAST = dict(episodes=1200, epsilon_decay_episodes=800, seed=0)

#: A token run: long enough to exercise the loop on every level, far too short to learn anything.
#: The tests using it are about termination and bookkeeping, not about policy quality.
TOKEN_RUN = dict(episodes=25, epsilon_decay_episodes=20, seed=0)

#: The reward a complete episode pays out on each level, worked out by hand from the brief's fixed
#: values (apple +1, key 0, chest +2, step 0, no death penalty) and from `gridworld/levels.py`.
#: Written out rather than computed so that a change to either the constants or a layout has to
#: break a test instead of quietly moving the target it is compared against.
EXPECTED_TOTALS = {
    2: 5.0,   # apples at (1,8), (3,2), (8,7) = +3, key at (8,1) = 0, chest at (5,8) = +2
    3: 4.0,   # apples at (3,9), (6,1) = +2, key at (6,9) = 0, chest at (9,7) = +2
    4: 3.0,   # apples at (5,2), (5,9), (9,8) = +3, no key or chest
    5: 4.0,   # apples at (2,8), (8,2) = +2, key at (6,1) = 0, chest at (9,7) = +2
}

LEVELS = (2, 3, 4, 5)
ALGORITHMS = (q_learning, sarsa)


@pytest.fixture
def scratch_dir() -> Iterator[Path]:
    """Not `tmp_path`: the shared `pytest-of-<user>` base dir is unreadable on this machine."""
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


def _train(level: int, algo, overrides: dict) -> tuple[QTable, TabularConfig]:
    """Train `algo` on `level` under a config narrowed by `dataclasses.replace`."""
    config = replace(train_gridworld.config_for_level(level), **overrides)
    env = GridWorld(level_index=level, rng=make_rng(config.seed + 5_000),
                    max_steps=config.max_steps_per_episode)
    result = algo(env, config, rng=make_rng(config.seed))
    return result.q_table, config


# --- A3-005: the key/chest ordering --------------------------------------------------------------


def test_level_2_layout_is_the_one_these_tests_assume():
    """Guards the hand-written expectations below against a level edit."""
    grid = load_level(2)
    items = {(row, col): tile
             for row, line in enumerate(grid)
             for col, tile in enumerate(line)
             if tile in COLLECTIBLE_TILES}
    assert items == {(1, 8): Tile.APPLE, (3, 2): Tile.APPLE, (8, 7): Tile.APPLE,
                     (5, 8): Tile.CHEST, (8, 1): Tile.KEY}


@pytest.mark.parametrize("algo", ALGORITHMS, ids=["q", "sarsa"])
def test_trained_level_2_policy_takes_the_key_before_the_chest(algo):
    """A3-005: the ordering claim, checked against the order the policy actually collects in.

    The reason this needs the *order* and not the return: level 2's chest pays +2 only while the
    key is held, and otherwise stays on the grid paying nothing. A policy that walked over the
    chest first would not crash, would not die, and would still be able to come back for it later —
    so "it scored 5" is compatible with a chest-first route, but a `C` before any `K` in the
    sequence is not.
    """
    q_table, config = _train(2, algo, LEVEL_2_FAST)
    env = GridWorld(level_index=2, rng=make_rng(config.seed + 9_000),
                    max_steps=config.max_steps_per_episode)
    rollout = greedy_rollout(env, q_table, rng=make_rng(config.seed + 9_000))
    order = train_gridworld.collection_order(2, rollout)

    tiles = [item["tile"] for item in order]
    assert tiles.count(Tile.KEY) == 1 and tiles.count(Tile.CHEST) == 1
    assert tiles.index(Tile.KEY) < tiles.index(Tile.CHEST), (
        f"the chest was opened before the key was taken: "
        f"{train_gridworld.describe_collection_order(order)}"
    )
    assert train_gridworld.key_precedes_chest(order) is True

    # "Solved" has to mean everything was collected, not merely that the agent survived.
    assert rollout.collected == len(env.collectible_cells) == 5
    assert not rollout.died and not rollout.truncated
    assert rollout.total_return == EXPECTED_TOTALS[2]
    assert rollout.steps == optimal_collection_steps(2) == 25


@pytest.mark.parametrize("algo", ALGORITHMS, ids=["q", "sarsa"])
def test_trained_level_3_policy_takes_the_key_before_the_chest(algo):
    """Level 3 is the same dependency across a rock maze, and the same evidence."""
    q_table, config = _train(3, algo, LEVEL_2_FAST)
    env = GridWorld(level_index=3, rng=make_rng(config.seed + 9_000),
                    max_steps=config.max_steps_per_episode)
    rollout = greedy_rollout(env, q_table, rng=make_rng(config.seed + 9_000))
    order = train_gridworld.collection_order(3, rollout)

    assert train_gridworld.key_precedes_chest(order) is True, (
        train_gridworld.describe_collection_order(order)
    )
    assert rollout.collected == len(env.collectible_cells) == 4
    assert rollout.total_return == EXPECTED_TOTALS[3]


def test_key_precedes_chest_reports_none_when_there_is_no_chest():
    """A chestless level must not silently score a tick for a check that was never run."""
    order = [{"tile": Tile.APPLE, "cell": [0, 0], "step": 1}]
    assert train_gridworld.key_precedes_chest(order) is None


def test_key_precedes_chest_catches_a_chest_first_order():
    """The check has to be able to fail, or asserting on it proves nothing."""
    bad = [{"tile": Tile.CHEST, "cell": [5, 8], "step": 3},
           {"tile": Tile.KEY, "cell": [8, 1], "step": 9}]
    assert train_gridworld.key_precedes_chest(bad) is False


def test_collection_order_decodes_the_mask_in_the_env_s_own_bit_order():
    """`collectible_cells` must agree with the order `GridWorld` fixes at construction.

    They are computed in two places — the env sorts its collectibles once at construction, the
    report recomputes them from the layout — and if the two ever disagreed, every decoded
    collection order would name the wrong tiles while still looking plausible.
    """
    for level in LEVELS:
        env = GridWorld(level_index=level, rng=make_rng(0))
        assert list(env.collectible_cells) == [
            tuple(cell) for cell in train_gridworld.collectible_cells(level)
        ]


# --- A3-005: reward accounting across a full episode ---------------------------------------------


def test_level_total_reward_matches_the_hand_worked_totals():
    for level, total in EXPECTED_TOTALS.items():
        assert train_gridworld.level_total_reward(level) == total


@pytest.mark.parametrize("algo", ALGORITHMS, ids=["q", "sarsa"])
def test_a_full_level_2_episode_sums_to_the_level_total(algo):
    """A3-005: reward accounting across a whole episode, summed step by step.

    Rewards are accumulated from `step()` here rather than read off `episode_return`, so the test
    would still fail if the env's own accumulator and the rewards it hands back drifted apart.
    """
    q_table, config = _train(2, algo, LEVEL_2_FAST)
    env = GridWorld(level_index=2, rng=make_rng(config.seed + 9_000),
                    max_steps=config.max_steps_per_episode)
    rollout = greedy_rollout(env, q_table, rng=make_rng(config.seed + 9_000))

    # Replay the greedy actions and add the rewards up independently of the rollout's own total.
    replay = GridWorld(level_index=2, rng=make_rng(config.seed + 9_000),
                       max_steps=config.max_steps_per_episode)
    replay.reset()
    total = 0.0
    payouts = []
    for action in rollout.actions:
        _, reward, _, _ = replay.step(action)
        total += reward
        if reward:
            payouts.append(reward)

    assert total == EXPECTED_TOTALS[2] == train_gridworld.level_total_reward(2)
    assert total == replay.episode_return == rollout.total_return
    # Three apples at +1 and one chest at +2; the key pays 0 and every other step pays 0.
    assert sorted(payouts) == [1.0, 1.0, 1.0, 2.0]
    assert replay.terminated and not replay.truncated and not replay.died


# --- termination on all four levels --------------------------------------------------------------


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("algo", ALGORITHMS, ids=["q", "sarsa"])
def test_both_algorithms_terminate_correctly_on_every_level(level, algo):
    """Every episode must end for a reason the env can name, and inside the step cap.

    The three legal endings are: everything collected, the agent died, or the step cap fired. An
    episode that ended for none of those, or that ran past `max_steps`, would mean the training
    loop and the env disagree about when to stop — which on levels 4-5 is easy to miss, because a
    monster can end an episode on a step the agent thought was safe.
    """
    q_table, config = _train(level, algo, TOKEN_RUN)
    n_collectibles = len(train_gridworld.collectible_cells(level))

    for offset in range(5):
        env = GridWorld(level_index=level, rng=make_rng(offset + 20_000),
                        max_steps=config.max_steps_per_episode)
        rollout = greedy_rollout(env, q_table, rng=make_rng(offset + 30_000))
        assert 1 <= rollout.steps <= config.max_steps_per_episode
        assert env.done
        assert env.terminated or env.truncated
        assert rollout.died or rollout.truncated or rollout.collected == n_collectibles
        # A truncation is not the game ending, so it must not be reported as one.
        assert not (env.truncated and env.terminated)
        # Nothing can be collected after the agent is dead.
        assert rollout.collected <= n_collectibles


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("algo", ALGORITHMS, ids=["q", "sarsa"])
def test_training_records_one_row_per_episode_within_the_step_cap(level, algo):
    config = replace(train_gridworld.config_for_level(level), **TOKEN_RUN)
    env = GridWorld(level_index=level, rng=make_rng(1), max_steps=config.max_steps_per_episode)
    result = algo(env, config, rng=make_rng(2))
    assert len(result.history) == config.episodes
    for record in result.history:
        assert 1 <= record.steps <= config.max_steps_per_episode
        assert 0 <= record.collected <= len(train_gridworld.collectible_cells(level))


# --- A3-006: the monster mechanics, measured against the shipped env ------------------------------


@pytest.mark.parametrize("level", (4, 5))
def test_measured_monster_move_rate_matches_the_brief(level):
    """A3-006 criteria 1 and 2, measured through `GridWorld.step()` rather than asserted.

    20,000 observations put the standard error of the rate near 0.0035, so a tolerance of 0.02 is
    roughly six standard errors — tight enough to catch a wrong constant (0.5, or a per-step rather
    than per-monster draw) and loose enough never to flake. The full 200,000-sample measurement
    that the report cites lives in the `--levels` run, not here.
    """
    measured = train_gridworld.measure_monster_movement(level, samples=20_000, seed=3)
    assert measured["measured_move_probability"] == pytest.approx(
        MONSTER_MOVE_PROBABILITY, abs=0.02
    )
    assert measured["landings_on_rock"] == 0
    assert measured["landings_off_grid"] == 0
    # Uniform over the legal directions: on an open cell all four are legal and each takes a
    # quarter of the moves. The tolerance absorbs the cells near the walls, where fewer are.
    for share in measured["direction_shares"].values():
        assert share == pytest.approx(0.25, abs=0.05)


@pytest.mark.parametrize("level", (4, 5))
def test_a_cornered_monster_splits_its_move_probability_over_the_legal_directions_only(level):
    """A3-006 criterion 2, at a cell where illegal directions actually exist.

    In open ground all four directions are legal, so a monster that ignored the rule entirely
    would produce the same near-uniform histogram as one that obeyed it. The cells used here have
    exactly two legal neighbours — blocked by the grid edge on level 4, which carries no rocks at
    all, and by the edge plus a rock on level 5 — so the whole 0.4 must land on those two at 0.2
    each, with 0.6 left staying put. 20,000 draws put the standard error of a 0.2 share near
    0.0028, so the 0.02 tolerance is about seven standard errors.
    """
    measured = train_gridworld.measure_constrained_monster_choice(
        level, samples=20_000, seed=level
    )
    assert len(measured["legal_neighbours"]) == 2
    assert measured["illegal_landings"] == 0
    assert measured["measured_move_probability"] == pytest.approx(
        MONSTER_MOVE_PROBABILITY, abs=0.02
    )
    for neighbour in measured["legal_neighbours"]:
        share = measured["landing_shares"][f"{neighbour[0]},{neighbour[1]}"]
        assert share == pytest.approx(MONSTER_MOVE_PROBABILITY / 2, abs=0.02)
    stayed = measured["landing_shares"][
        f"{measured['cell'][0]},{measured['cell'][1]}"
    ]
    assert stayed == pytest.approx(1 - MONSTER_MOVE_PROBABILITY, abs=0.02)


@pytest.mark.parametrize("level", (4, 5))
def test_both_death_checks_fire_on_the_real_levels(level):
    """A3-006 criterion 3: the agent dies stepping onto a monster *and* being stepped onto.

    The second half is the one implementations miss, so it is demonstrated rather than inspected:
    the agent presses UP into the grid edge — a blocked move, so it does not shift a cell — and the
    monster alongside it is the only thing that moves.
    """
    checks = train_gridworld.verify_death_checks(level)
    assert checks["agent_entered_monster"]["died"] is True
    assert checks["agent_entered_monster"]["terminated"] is True

    entered_by = checks["monster_entered_agent"]
    assert entered_by is not None, "no seed drew a monster onto the stationary agent"
    assert entered_by["agent_before"] == entered_by["agent_after"], (
        "the agent must not have moved, or this is the first death check again"
    )
    assert entered_by["monster_after"] == entered_by["agent_after"]
    assert entered_by["died"] is True and entered_by["terminated"] is True
    assert checks["both_checks_fire"] is True


# --- the levels report ---------------------------------------------------------------------------


def test_death_rate_reports_success_and_truncation_alongside_deaths():
    """On a stochastic level the death rate alone is not the complement of success.

    An episode can also run out of clock, so the three rates are measured separately rather than
    inferred from one another.
    """
    q_table, config = _train(4, q_learning, TOKEN_RUN)
    stats = train_gridworld.death_rate(
        4, q_table, epsilon=0.0, max_steps=config.max_steps_per_episode, rollouts=20
    )
    assert stats["rollouts"] == 20
    assert stats["n_collectibles"] == 3
    # Every episode ends in exactly one of the three ways: everything collected and survived, dead,
    # or out of clock. The env makes them mutually exclusive — `truncated` is only set when the
    # episode has not terminated — so the three counts must add up to the sample size exactly.
    assert stats["successes"] + stats["deaths"] + stats["truncations"] == stats["rollouts"]
    for key in ("success_rate", "death_rate", "truncation_rate"):
        assert 0.0 <= stats[key] <= 1.0
    assert 0.0 <= stats["mean_collected"] <= 3.0


def test_levels_report_writes_the_markdown_and_the_json(scratch_dir):
    """The deliverable half of both tickets: the summary exists and names what it measured."""
    report = train_gridworld.levels_report(
        [2, 4],
        [0],
        results_dir=scratch_dir,
        rollouts=5,
        monster_samples=2_000,
        overrides=dict(episodes=40, epsilon_decay_episodes=30),
    )

    assert [entry["level"] for entry in report["runs"]] == [2, 2, 4, 4]
    assert {entry["algo"] for entry in report["runs"]} == {"q", "sarsa"}
    # Every run's config came from the file, narrowed through `replace` — level 4 keeps its own
    # step cap even though the episode count was shortened.
    assert all(entry["episodes"] == 40 for entry in report["runs"])
    assert set(report["monsters"]) == {4}, "level 2 has no monsters to measure"

    for path in report["paths"].values():
        assert path.exists() and path.stat().st_size > 0
        assert "levels2-4_seeds0" in path.name

    markdown = report["paths"]["levels_markdown"].read_text(encoding="utf-8")
    assert "collection order" in markdown
    assert "move probability" in markdown
    assert "key before chest" in markdown

    saved = json.loads(report["paths"]["levels_json"].read_text(encoding="utf-8"))
    assert "run_objects" not in saved, "the live objects must not leak into the JSON"
    assert saved["monsters"]["4"]["death_checks"]["both_checks_fire"] is True

    # Curves for both algorithms on both levels, each naming its own seed.
    for level in (2, 4):
        for algo in ("q", "sarsa"):
            assert (scratch_dir / f"curve_level{level}_{algo}_seed0.png").exists()


def test_levels_cli_reaches_the_report_path(scratch_dir, capsys):
    train_gridworld.main([
        "--levels", "2",
        "--seeds", "0",
        "--episodes", "40", "--epsilon-decay-episodes", "30",
        "--death-rate-rollouts", "5",
        "--results-dir", str(scratch_dir),
    ])
    out = capsys.readouterr().out
    assert "level 2" in out and "order:" in out
    assert (scratch_dir / "levels2-2_seeds0.md").exists()
