"""Tests for Task 5 — the intrinsic reward (A3-007, rubric row F).

Four claims are made about this feature, and all four are the kind that a learning curve cannot
distinguish from its neighbour:

  1. the bonus is exactly `strength / sqrt(n(s) + 1)`, not `1/sqrt(n)`, not `1/(n+1)`;
  2. `n(s)` is a **within-episode** count that resets at every `env.reset()` — a lifetime counter
     produces a curve of the same shape and a completely different algorithm;
  3. the environment reward is untouched, so a run at strength 0 must be *numerically identical*
     to a plain `q_learning`/`sarsa` run on the same seed;
  4. the return that is logged, written to CSV and plotted is the **environment** return, with the
     bonus excluded.

The fourth is the one that would quietly ruin the F-row evidence. The bonus is paid on nearly every
step, so a curve of shaped returns puts the strength-0.5 arm above the strength-0.0 arm by roughly
(steps per episode x bonus) whether or not the agent ever reached the chest. The figure would
separate, the report would claim a result, and the experiment would have measured nothing. Two
tests below pin it: one on a stub env with known rewards, one against the env's own accumulator.

`ArrivalTraceEnv` exists because the visit-count sequence has to be *predicted*, not observed. Its
successor state does not depend on the action, so the state trace is a fixed list whatever the
exploring policy does, and the exact sequence of `n(s)` values the loop should produce can be
written down. That sequence also separates the two readings of `n(s)` — the state arrived in
against the state being left — which is the choice the module docstring commits to.
"""

from __future__ import annotations

import ast
import csv
import json
import tempfile
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from common.config import TabularConfig, load_yaml
from gridworld import algorithms
from gridworld.algorithms import (
    EpisodeVisitCounts,
    intrinsic_reward,
    q_learning,
    sarsa,
    train_with_intrinsic_reward,
)
from gridworld.constants import N_ACTIONS
from gridworld.env import GridWorld
from train import train_gridworld

BASE_CONFIG = TabularConfig.from_yaml()

#: A run short enough for a test and long enough to exercise many episodes of counter lifetime.
FAST = dict(episodes=120, epsilon_decay_episodes=80)


@pytest.fixture
def scratch_dir() -> Iterator[Path]:
    """Not `tmp_path`: the shared `pytest-of-<user>` base dir is unreadable on this machine."""
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


class ArrivalTraceEnv:
    """An env whose state trace is fixed regardless of the action, so visit counts are predictable.

    Every episode walks `S0 -> S1 -> S2 -> S1 -> S2 -> ...` for `length` steps and then terminates.
    The successor ignores the action, which is the point: the agent underneath is a real
    epsilon-greedy learner drawing real random actions, and the sequence of states it occupies is
    still a constant that the test can write down.

    `reward_on_last` is paid on the final step and nowhere else, so the environment return of every
    episode is exactly that number — the quantity the logged return must equal.
    """

    level_index = 6

    S0 = (0, 0, False, 0)
    S1 = (0, 1, False, 0)
    S2 = (0, 2, False, 0)

    def __init__(self, *, length: int = 5, reward_on_last: float = 1.0) -> None:
        self.length = int(length)
        self.reward_on_last = float(reward_on_last)
        self.steps = 0

    #: The states the episode arrives in, in order. `S0` is the start and is never *arrived* in.
    @property
    def arrivals(self) -> list[tuple[int, int, bool, int]]:
        return [self.S1 if index % 2 == 0 else self.S2 for index in range(self.length)]

    def reset(self):
        self.steps = 0
        return self.S0

    def step(self, action: int):
        assert 0 <= int(action) < N_ACTIONS
        state = self.arrivals[self.steps]
        self.steps += 1
        terminated = self.steps >= self.length
        reward = self.reward_on_last if terminated else 0.0
        info = {
            "steps": self.steps,
            "died": False,
            "collected": 0,
            "terminated": terminated,
            "truncated": False,
        }
        return state, reward, terminated, info


def _config(**changes: Any) -> TabularConfig:
    """A narrowed config. Every value still originates in `config/gridworld.yaml`."""
    return replace(BASE_CONFIG, **{**FAST, **changes})


def _spy_on_the_bonus(monkeypatch) -> list[tuple[float, int]]:
    """Record every `(strength, n(s))` pair the training loop asks the formula for, in order."""
    calls: list[tuple[float, int]] = []
    real = algorithms.intrinsic_reward

    def recorder(strength: float, visits: int) -> float:
        calls.append((float(strength), int(visits)))
        return real(strength, visits)

    monkeypatch.setattr(algorithms, "intrinsic_reward", recorder)
    return calls


# --- criterion 1: the formula is exactly strength / sqrt(n + 1) ---------------------------------


@pytest.mark.parametrize("visits", [0, 1, 2, 3, 4, 8, 15, 24, 99])
@pytest.mark.parametrize("strength", [0.0, 0.1, 0.5, 1.0, 2.5])
def test_bonus_matches_the_brief_formula_at_many_visit_counts(strength: float, visits: int):
    """`r_i = strength / sqrt(n(s) + 1)`, checked against the arithmetic rather than a fixture.

    Several counts, not one: `strength / sqrt(n)`, `strength / (n + 1)` and
    `strength / sqrt(n) + 1` all agree with the correct formula at some single value of `n` and
    disagree everywhere else.
    """
    assert intrinsic_reward(strength, visits) == pytest.approx(strength / np.sqrt(visits + 1))


def test_bonus_at_a_never_visited_state_is_the_full_strength():
    """`n(s) = 0` on first arrival, so the denominator is 1 and the bonus is `strength` exactly."""
    assert intrinsic_reward(0.5, 0) == 0.5
    assert intrinsic_reward(2.0, 0) == 2.0


def test_bonus_decays_with_the_square_root_and_never_reaches_zero():
    """The shape claim: halving requires four times the visits, and the bonus stays positive."""
    assert intrinsic_reward(1.0, 3) == pytest.approx(0.5)
    assert intrinsic_reward(1.0, 15) == pytest.approx(0.25)
    values = [intrinsic_reward(0.5, n) for n in range(50)]
    assert all(later < earlier for earlier, later in zip(values, values[1:], strict=False))
    assert values[-1] > 0.0


def test_zero_strength_is_exactly_zero_at_every_count():
    """Not "small" — exactly 0.0, or the strength-0 arm is not a control (criterion 3)."""
    assert all(intrinsic_reward(0.0, n) == 0.0 for n in range(200))


def test_negative_visit_counts_are_rejected():
    with pytest.raises(ValueError):
        intrinsic_reward(0.5, -1)


# --- criterion 2: n(s) is a within-episode count and resets ------------------------------------


def test_visit_counter_counts_occupancies_and_returns_the_count_before_each():
    counts = EpisodeVisitCounts()
    state = (1, 2, False, 0)
    assert [counts.record(state) for _ in range(4)] == [0, 1, 2, 3]
    assert counts.visits(state) == 4


def test_visit_counter_keys_on_the_full_state_not_the_position():
    """Holding the key changes the situation, so it must change `n(s)` (see the state-key note)."""
    counts = EpisodeVisitCounts()
    without_key, with_key = (4, 6, False, 0), (4, 6, True, 1)
    counts.record(without_key)
    counts.record(without_key)
    assert counts.record(with_key) == 0, "a different state key must have its own count"
    assert counts.visits(without_key) == 2


def test_visit_counter_reset_clears_every_count():
    counts = EpisodeVisitCounts()
    for state in [(0, 0, False, 0), (0, 1, False, 0), (0, 0, False, 0)]:
        counts.record(state)
    assert len(counts) == 2
    counts.reset()
    assert len(counts) == 0
    assert counts.record((0, 0, False, 0)) == 0, "the count must start again from zero"


def test_the_training_loop_restarts_the_counter_every_episode(monkeypatch):
    """The behavioural half of criterion 2, on a trace whose visit counts are known in advance.

    `ArrivalTraceEnv` visits S1, S2, S1, S2, S1 every episode, so the `n(s)` the loop should ask
    for is `[0, 0, 1, 1, 2]` — and then the *same* list again on the next episode, because the
    counter resets. A lifetime counter would produce `[0, 0, 1, 1, 2, 3, 2, 4, 3, 5]` across two
    episodes, which is what this asserts against.
    """
    calls = _spy_on_the_bonus(monkeypatch)
    config = _config(episodes=3, intrinsic_reward_strength=0.5, seed=0)
    env = ArrivalTraceEnv(length=5)

    train_with_intrinsic_reward(env, config, algo="q")

    per_episode = [0, 0, 1, 1, 2]
    assert [visits for _, visits in calls] == per_episode * 3
    assert {strength for strength, _ in calls} == {0.5}


def test_the_counted_state_is_the_one_arrived_in_not_the_one_left(monkeypatch):
    """The documented choice, pinned. `n(s)` scores the arrival; the alternative gives another list.

    On `ArrivalTraceEnv` the arrival trace S1, S2, S1, S2, S1 yields `[0, 0, 1, 1, 2]`. Scoring the
    state being *left* — S0, S1, S2, S1, S2, with S0 already recorded at reset — would yield
    `[1, 0, 0, 1, 1]`. The two lists differ in their first entry alone at a glance, which is
    exactly why this is asserted rather than trusted to a comment.
    """
    calls = _spy_on_the_bonus(monkeypatch)
    config = _config(episodes=1, intrinsic_reward_strength=0.5, seed=0)

    train_with_intrinsic_reward(ArrivalTraceEnv(length=5), config, algo="q")

    observed = [visits for _, visits in calls]
    assert observed == [0, 0, 1, 1, 2], "the bonus must score the state the transition arrives in"
    assert observed != [1, 0, 0, 1, 1], "this is the state-being-left reading, which was rejected"


def test_the_start_state_is_recorded_so_returning_to_it_is_a_revisit(monkeypatch):
    """`n(s)` counts occupancies, and the episode occupies its start state at t = 0."""

    class ReturnsHomeEnv(ArrivalTraceEnv):
        """S0 -> S1 -> S0, so the second arrival is back at the start cell."""

        @property
        def arrivals(self):
            return [self.S1, self.S0][: self.length]

    calls = _spy_on_the_bonus(monkeypatch)
    config = _config(episodes=1, intrinsic_reward_strength=0.5, seed=0)

    train_with_intrinsic_reward(ReturnsHomeEnv(length=2), config, algo="q")

    assert [visits for _, visits in calls] == [0, 1], (
        "arriving back at the start state must count as a revisit, not as fresh ground"
    )


# --- criterion 3: environment rewards are untouched ---------------------------------------------


@pytest.mark.parametrize("algo,plain", [("q", q_learning), ("sarsa", sarsa)])
@pytest.mark.parametrize("level", [0, 4])
def test_zero_strength_is_numerically_identical_to_the_plain_algorithm(algo, plain, level):
    """The control has to be a control: at strength 0 this must be the *same computation*.

    Not "close" — identical, float for float and draw for draw. The two runs get envs built from
    the same seed and learners built from the same seed, so any divergence at all would mean the
    intrinsic loop consumes random numbers in a different order or applies a different update, and
    the baseline arm of the F-row experiment would silently stop being the algorithm it claims to
    be. Level 4 is included because it has monsters: it is the case where a stray draw from the
    env's generator would show up.
    """
    config = _config(seed=5, intrinsic_reward_strength=0.0)

    reference = plain(
        GridWorld(level_index=level, seed=11, max_steps=config.max_steps_per_episode),
        config,
        rng=algorithms.make_rng(7),
    )
    shaped = train_with_intrinsic_reward(
        GridWorld(level_index=level, seed=11, max_steps=config.max_steps_per_episode),
        config,
        algo=algo,
        rng=algorithms.make_rng(7),
    )

    assert shaped.algo == reference.algo == algo
    assert list(shaped.returns) == list(reference.returns)
    assert list(shaped.steps) == list(reference.steps)
    assert set(shaped.q_table) == set(reference.q_table)
    for state, row in reference.q_table.items():
        assert np.array_equal(shaped.q_table[state], row), f"tables diverge at {state}"


def test_a_positive_strength_actually_changes_the_update():
    """The other half of the previous test: if the bonus changed nothing, it would also pass.

    Same env seed, same learner seed, same everything but the strength. The Q-tables must differ,
    or the bonus is being computed and thrown away.
    """
    level = 6
    tables = {}
    for strength in (0.0, 0.5):
        config = _config(seed=5, intrinsic_reward_strength=strength)
        result = train_with_intrinsic_reward(
            GridWorld(level_index=level, seed=11, max_steps=config.max_steps_per_episode),
            config,
            algo="q",
            rng=algorithms.make_rng(7),
        )
        tables[strength] = result.q_table

    assert any(
        not np.array_equal(tables[0.5][state], row) for state, row in tables[0.0].items()
    ), "the intrinsic bonus never reached the Q-update"


def test_the_environment_reward_never_changes_shape_under_a_bonus():
    """Level 0 pays apples worth +1 each, so an episode return is an integer 0..3 — bonus or not.

    A shaped return would be a fractional number on essentially every episode, so this catches the
    bonus leaking into the environment reward without needing to know which episodes succeeded.
    """
    config = _config(episodes=60, epsilon_decay_episodes=40, seed=2,
                     intrinsic_reward_strength=0.5)
    result = train_with_intrinsic_reward(
        GridWorld(level_index=0, seed=1, max_steps=config.max_steps_per_episode),
        config,
        algo="q",
        rng=algorithms.make_rng(3),
    )
    observed = sorted(set(result.returns))
    assert set(observed) <= {0.0, 1.0, 2.0, 3.0}, (
        f"episode returns must be sums of the level's own rewards, got {observed}"
    )


# --- criterion 4: the logged and plotted return is the environment return -----------------------


@pytest.mark.parametrize("strength", [0.0, 0.5, 2.0])
def test_logged_return_is_the_environment_return_and_excludes_the_bonus(strength: float):
    """The single most damaging possible bug in this feature, pinned on a known-reward env.

    `ArrivalTraceEnv` pays exactly 1.0 per episode, on the last of its five steps. The intrinsic
    bonus over those five steps is `strength * (1 + 1 + 1/sqrt(2) + 1/sqrt(2) + 1/sqrt(3))`, which
    at strength 0.5 is about 1.99 — comparable to the environment reward itself. If the shaped
    reward were being logged, the recorded return would move with the strength; it must not.
    """
    config = _config(episodes=4, intrinsic_reward_strength=strength, seed=0)
    result = train_with_intrinsic_reward(ArrivalTraceEnv(length=5), config, algo="q")

    assert list(result.returns) == [1.0] * config.episodes, (
        f"the logged return moved with the strength ({strength}); the bonus is being logged"
    )


def test_logged_return_equals_the_environments_own_accumulator():
    """A second, independent reading of criterion 4, against `GridWorld.episode_return`.

    `env.episode_return` is summed inside `step()` from the env's own rewards and knows nothing
    about the learner. After a single-episode run the logged return must equal it exactly.
    """
    config = _config(episodes=1, intrinsic_reward_strength=0.75, seed=4)
    env = GridWorld(level_index=0, seed=9, max_steps=config.max_steps_per_episode)
    result = train_with_intrinsic_reward(env, config, algo="q", rng=algorithms.make_rng(9))

    assert result.history[0].total_return == env.episode_return


def test_history_csv_carries_the_environment_return(scratch_dir):
    """The CSV is what the report's curve is drawn from, so the check follows it to disk."""
    config = _config(episodes=6, intrinsic_reward_strength=1.0, seed=0)
    result = train_with_intrinsic_reward(ArrivalTraceEnv(length=5), config, algo="q")

    path = result.write_history_csv(scratch_dir / "history.csv")
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    assert [float(row["return"]) for row in rows] == [1.0] * config.episodes


def test_the_plotted_series_is_the_history_and_nothing_else(scratch_dir):
    """`plot_intrinsic_comparison` draws `TrainingResult.returns`, which is the environment return.

    Asserted at the source, because a figure cannot be read back numerically: the plotting function
    must not reach for any shaped or bonus-carrying quantity. `TrainingResult` exposes exactly one
    return series and it is the environment one, so the check is that no other name appears.
    """
    source = ast.parse(Path(train_gridworld.__file__).read_text(encoding="utf-8"))
    node = next(
        n for n in ast.walk(source)
        if isinstance(n, ast.FunctionDef) and n.name == "plot_intrinsic_comparison"
    )
    attributes = {a.attr for a in ast.walk(node) if isinstance(a, ast.Attribute)}
    assert "returns" in attributes, "the figure must plot the recorded episode returns"
    for banned in ("shaped", "shaped_returns", "bonus", "intrinsic_returns"):
        assert banned not in attributes


# --- B3: the strength comes from the config, not from a literal ---------------------------------


def test_the_strength_is_read_from_the_config_object():
    """No literal at the call site: the loop must take its strength from `TabularConfig`."""
    node = next(
        n for n in ast.walk(ast.parse(Path(algorithms.__file__).read_text(encoding="utf-8")))
        if isinstance(n, ast.FunctionDef) and n.name == "train_with_intrinsic_reward"
    )
    attributes = {a.attr for a in ast.walk(node) if isinstance(a, ast.Attribute)}
    assert "intrinsic_reward_strength" in attributes
    assert {"alpha", "gamma", "episodes"} <= attributes, (
        "every hyperparameter in this loop must come off the config object"
    )


def test_the_config_file_still_names_the_experiment_the_ticket_asks_for():
    """A3-007 names strengths 0.0 and 0.5 on level 6; the figure's legend comes from this block."""
    block = load_yaml("gridworld")["intrinsic_experiment"]
    assert block["level"] == 6
    assert [float(value) for value in block["strengths"]] == [0.0, 0.5]
    assert int(block["episodes"]) > 0
    assert train_gridworld.intrinsic_experiment_settings() == {
        "level": 6,
        "strengths": [0.0, 0.5],
        "episodes": int(block["episodes"]),
    }


def test_level_6_keeps_its_own_step_cap_and_leaves_levels_0_to_5_alone():
    """The level 6 budget is an override of its own; nothing else in the file may have moved."""
    raw = load_yaml("gridworld")
    overrides = raw["level_overrides"]
    assert overrides[6]["max_steps_per_episode"] == 800
    assert overrides[1]["episodes"] == 8000 and overrides[1]["epsilon_decay_episodes"] == 6000
    assert overrides[4]["episodes"] == 12000 and overrides[5]["episodes"] == 16000
    assert overrides[5]["max_steps_per_episode"] == 400
    assert raw["training"]["epsilon_end"] == 0.05
    assert "epsilon_end" not in overrides.get(6, {})
    assert train_gridworld.config_for_level(6).max_steps_per_episode == 800


def test_a_bad_algorithm_name_is_rejected():
    with pytest.raises(ValueError):
        train_with_intrinsic_reward(ArrivalTraceEnv(), _config(episodes=1), algo="dqn")


# --- the statistics the write-up quotes ---------------------------------------------------------


def test_summary_reports_the_standard_error_not_the_range():
    """A mean quoted with a spread has to be quoted with one that shrinks as seeds are added."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = train_gridworld.summarise_across_seeds(values)
    assert stats["n"] == 5
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["sd"] == pytest.approx(np.std(values, ddof=1))
    assert stats["sem"] == pytest.approx(np.std(values, ddof=1) / np.sqrt(5))
    assert stats["sem"] < stats["sd"], "the standard error must be the tighter of the two"


def test_single_seed_reports_no_spread_rather_than_a_nan():
    stats = train_gridworld.summarise_across_seeds([2.0])
    assert stats["mean"] == 2.0 and stats["sd"] == 0.0 and stats["sem"] == 0.0


def test_difference_of_means_is_a_difference_of_means():
    """Not a max-pairwise gap: the statistic must be the gap between the two arm means."""
    treatment, baseline = [3.0, 4.0, 5.0], [1.0, 2.0, 3.0]
    entry = train_gridworld.difference_of_means(treatment, baseline)
    assert entry["difference"] == pytest.approx(2.0)
    assert entry["standard_error"] == pytest.approx(np.sqrt(2 * np.var(treatment, ddof=1) / 3))
    assert entry["ci95_low"] < entry["difference"] < entry["ci95_high"]


def test_an_overlapping_comparison_is_not_called_significant():
    """Two arms drawn from obviously overlapping spreads must not clear the 95% interval."""
    entry = train_gridworld.difference_of_means(
        [1.0, 5.0, 2.0, 6.0, 3.0], [2.0, 4.0, 3.0, 5.0, 4.0]
    )
    assert entry["significant_at_95"] is False
    assert entry["ci95_low"] < 0.0 < entry["ci95_high"]


def test_a_wide_separation_is_reported_as_clearing_zero():
    entry = train_gridworld.difference_of_means(
        [10.0, 10.1, 9.9, 10.2, 9.8], [0.0, 0.1, -0.1, 0.2, 0.0]
    )
    assert entry["significant_at_95"] is True
    assert entry["ci95_low"] > 0.0


def test_two_constant_arms_report_a_point_interval_rather_than_a_false_negative():
    """The degenerate case the level 6 run actually hit, and got wrong before this test existed.

    Every seed of the baseline arm ended with a greedy policy that solves the level and every seed
    of the treatment arm ended with one that does not, so both arms have zero variance, the Welch
    standard error is 0 and the interval collapses to the point [-1, -1]. An earlier guard of
    `standard_error > 0` labelled that "includes zero" and printed it directly beside an interval
    that plainly does not — the flag contradicting its own numbers. The flag now reads off the
    interval, and `zero_variance` marks the case so the prose can call it what it is.
    """
    entry = train_gridworld.difference_of_means([0.0] * 5, [1.0] * 5)
    assert entry["difference"] == pytest.approx(-1.0)
    assert entry["standard_error"] == 0.0
    assert entry["zero_variance"] is True
    assert entry["ci95_low"] == entry["ci95_high"] == pytest.approx(-1.0)
    assert entry["significant_at_95"] is True, "an interval of [-1, -1] does not include zero"


def test_two_identical_arms_are_never_reported_as_a_difference():
    """The other degenerate case: zero variance *and* zero difference is not an effect."""
    entry = train_gridworld.difference_of_means([1.0] * 5, [1.0] * 5)
    assert entry["difference"] == 0.0
    assert entry["zero_variance"] is True
    assert entry["significant_at_95"] is False


@pytest.mark.parametrize(
    "treatment,baseline",
    [
        ([0.0] * 5, [1.0] * 5),
        ([1.0] * 5, [1.0] * 5),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        ([9.0, 9.1, 8.9], [0.0, 0.1, -0.1]),
        ([1.0, 5.0, 2.0], [2.0, 4.0, 3.0]),
    ],
)
def test_the_significance_flag_never_contradicts_its_own_interval(treatment, baseline):
    """Whatever the arms look like, the flag and the printed interval must agree."""
    entry = train_gridworld.difference_of_means(treatment, baseline)
    excludes_zero = entry["ci95_low"] > 0.0 or entry["ci95_high"] < 0.0
    assert entry["significant_at_95"] is excludes_zero


# --- the experiment wiring end to end -----------------------------------------------------------


def test_intrinsic_run_reports_the_metrics_a_sparse_level_needs(scratch_dir):
    # Built from `config_for_level(6)` and not from the bare training block: merging the level's
    # own overrides is the caller's job, and level 6's step cap is one of them.
    config = replace(
        train_gridworld.config_for_level(6),
        episodes=25, epsilon_decay_episodes=15, seed=0, intrinsic_reward_strength=0.5,
    )
    summary = train_gridworld.intrinsic_run(6, "q", config, results_dir=scratch_dir)

    assert summary["intrinsic_reward_strength"] == 0.5
    assert summary["max_steps_per_episode"] == 800
    assert 0.0 <= summary["solved_fraction"] <= 1.0
    assert summary["solved_fraction_final"] == pytest.approx(summary["solved_fraction"])
    assert summary["first_success_censored"] == (summary["episodes_to_first_success"] is None)
    assert summary["mean_return"] == pytest.approx(summary["result"].returns.mean())
    # Level 6 pays only the chest's +2, so a mean return is that times the success rate.
    assert summary["mean_return"] == pytest.approx(2.0 * summary["solved_fraction"])
    for path in summary["paths"].values():
        assert path.exists() and path.stat().st_size > 0
        assert "level6_q_strength0p5_seed0" in path.name


def test_the_experiment_writes_both_curves_on_one_axis_and_the_explanation(scratch_dir):
    """Criteria 4 and 5 structurally: the two config strengths, several seeds, figure and prose."""
    experiment = train_gridworld.intrinsic_experiment(
        seeds=(0, 1), episodes=25, epsilon_decay_episodes=15, results_dir=scratch_dir
    )

    assert experiment["headline_strengths"] == [0.0, 0.5]
    assert experiment["seeds"] == [0, 1]
    assert experiment["level"] == 6
    assert set(experiment["runs"]) == {"0", "0.5"}
    assert all(len(arm) == 2 for arm in experiment["runs"].values())
    assert experiment["plotted_quantity"].startswith("environment return")

    expected = {"comparison_figure", "explanation_markdown", "experiment_json"}
    assert set(experiment["artifacts"]) == expected
    for path in experiment["paths"].values():
        assert path.exists() and path.stat().st_size > 0

    saved = json.loads(experiment["paths"]["experiment_json"].read_text(encoding="utf-8"))
    assert "run_objects" not in saved and "paths" not in saved
    assert saved["formula"] == "r_i = intrinsic_reward_strength / sqrt(n(s) + 1)"

    markdown = experiment["paths"]["explanation_markdown"].read_text(encoding="utf-8")
    assert "sqrt(n(s) + 1)" in markdown
    assert "environment" in markdown and "excluded" in markdown
    assert "standard error" in markdown
    # The write-up must state its own sample size rather than implying a significance it lacks.
    assert "significance" in markdown


def test_the_experiment_can_run_a_supporting_strength_sweep(scratch_dir):
    """Extra strengths go into a second figure; the headline pair stays the config's."""
    experiment = train_gridworld.intrinsic_experiment(
        seeds=(0,), episodes=15, epsilon_decay_episodes=10, extra_strengths=(0.25,),
        results_dir=scratch_dir,
    )
    assert experiment["headline_strengths"] == [0.0, 0.5]
    assert experiment["extra_strengths"] == [0.25]
    assert set(experiment["runs"]) == {"0", "0.5", "0.25"}
    assert experiment["paths"]["sweep_figure"].exists()


def test_the_documented_cli_reaches_the_experiment(scratch_dir, capsys):
    train_gridworld.main([
        "--level", "6", "--intrinsic-sweep",
        "--episodes", "20", "--epsilon-decay-episodes", "12",
        "--intrinsic-seeds", "0", "1",
        "--results-dir", str(scratch_dir),
    ])
    out = capsys.readouterr().out
    assert "intrinsic reward" in out
    assert "ENVIRONMENT return" in out
    assert "sqrt(n(s) + 1)" in out
    assert "significance claim" in out
    assert (scratch_dir / "intrinsic_level6_q_curve.png").exists()
    assert (scratch_dir / "intrinsic_level6_q.md").exists()
