"""Tests for `gridworld.algorithms` — the five bugs that are invisible from a training curve.

Each of these failure modes produces a curve that still rises, which is why they need tests rather
than eyeballing:

  - `np.argmax` tie-breaking biases every unvisited state toward UP (rubric B4);
  - bootstrapping through a terminal transition inflates the value of the state that ended it;
  - *not* bootstrapping through a truncation teaches the agent the world stops paying near the cap;
  - the target bootstrapping from `Q[s'][a]` instead of `max(Q[s'])` — Q-learning silently turned
    on-policy, i.e. SARSA wearing Q-learning's name (rubric C);
  - an epsilon literal at the call site silently detaches the run from `config/gridworld.yaml` (B3).

The fourth is the one that survives the longest. It cannot be seen on level 0 at all — SARSA also
converges to the shortest path on a deterministic hazard-free grid — so it is caught here at the
level of a single update, with a successor state whose four action-values are deliberately
different. That distinction is load-bearing for A3-004: when SARSA lands in this same module, the
suite has to be able to prove the two algorithms are not the same code.

SARSA has since landed, and the C1/C2 section near the bottom is the mirror image of the C section
above it — same successor rows, opposite expectations. Q-learning's update must equal the `max`
target and must not vary with the action taken; SARSA's must equal the `Q[s'][a']` target and must
vary. Neither set can pass against the other algorithm's update rule, so a copy-paste that
collapses the two into one implementation cannot leave the suite green.

The last test is the B5 demonstration in miniature: train briefly on level 0 and check the greedy
path length against an independently computed BFS optimum, rather than against a number typed here.
"""

from __future__ import annotations

import ast
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from common.config import TabularConfig
from common.schedules import LinearEpsilon
from common.seeding import make_rng
from gridworld import algorithms
from gridworld.algorithms import (
    epsilon_schedule,
    greedy_rollout,
    make_q_table,
    optimal_collection_steps,
    q_learning,
    select_action,
)
from gridworld.constants import N_ACTIONS
from gridworld.env import GridWorld

BASE_CONFIG = TabularConfig.from_yaml()


@pytest.fixture
def scratch_dir() -> Iterator[Path]:
    """A throwaway directory.

    Deliberately not pytest's `tmp_path`: on this machine the shared `pytest-of-<user>` base
    directory is unreadable (WinError 5) and every `tmp_path` test errors during collection of the
    fixture. `tempfile` sidesteps the poisoned base directory without touching global config.
    """
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


class OneStepEnv:
    """An env whose every episode is exactly one transition, with a chosen ending.

    The real `GridWorld` cannot isolate the terminal-vs-truncated distinction: reaching a real
    ending and hitting the step cap need different levels, different episode lengths and a policy
    that cooperates. Here the ending is a constructor argument.
    """

    level_index = 0

    START = (0, 0, False, 0)
    NEXT = (0, 1, False, 0)

    def __init__(self, *, terminated: bool, truncated: bool, reward: float = 1.0) -> None:
        self.terminated = terminated
        self.truncated = truncated
        self.reward = reward

    def reset(self) -> tuple[int, int, bool, int]:
        return self.START

    def step(self, action: int):
        info = {
            "steps": 1,
            "died": False,
            "collected": 0,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }
        return self.NEXT, self.reward, self.terminated or self.truncated, info


#: The successor state's Q-row. Deliberately **non-uniform**, and that is the whole point.
#:
#: An earlier version of this file seeded the row with `np.full(N_ACTIONS, next_value)` — four
#: identical entries — which made `max(Q[s'])` and `Q[s'][a]` numerically the same number for every
#: action. Under that row the off-policy target and the on-policy (SARSA) target are indistinguish-
#: able, so swapping one for the other in `q_learning` left the whole suite green. With distinct
#: entries the two targets separate, and every test below that bootstraps through this row can tell
#: which rule produced the update. Keep the four values distinct.
NEXT_Q_ROW = np.array([1.0, 7.0, -3.0, 4.0])

#: The greedy action at NEXT, i.e. the index `max(Q[s'])` reads. Only this one action agrees with
#: the SARSA target; the other three disagree, which is what makes the distinction testable.
NEXT_GREEDY_ACTION = 1

#: A second, more extreme successor row used by the explicit off-policy test. The gap between the
#: best action and the rest is huge and signed, so the two candidate targets land far apart
#: (+9.6 vs -3.7 at the config's alpha/gamma) and no floating-point tolerance can blur them.
OFF_POLICY_NEXT_ROW = np.array([-40.0, -40.0, 100.0, -40.0])
OFF_POLICY_GREEDY_ACTION = 2


class Transition(NamedTuple):
    """One recorded update: the config it ran under, the action taken, and the resulting Q-value.

    `action` is returned rather than discarded because a test that cannot say *which* action was
    taken cannot state the SARSA target it is ruling out.
    """

    config: TabularConfig
    action: int
    value: float


class SarsaTransition(NamedTuple):
    """The same recording for SARSA, which needs one more field than Q-learning does.

    Q-learning's target is a property of `s'` alone, so naming the action taken *out of* `s` is
    enough to state what the update should not have been. SARSA's target is `Q[s'][a']`, so the test
    cannot predict anything without knowing `a'` — the action drawn at the successor state and
    carried into the next step. That is precisely the quantity that makes the update on-policy, so
    it is recorded rather than inferred.
    """

    config: TabularConfig
    action: int
    next_action: int | None
    value: float


class RecordingRng:
    """A `np.random.Generator` proxy that remembers every action handed back to `select_action`.

    Reproducing the draw order by hand (`random()`, then `integers()`, twice) would re-implement
    `select_action` inside the test file and go stale the moment the selection logic is touched.
    Wrapping the generator instead records what was actually chosen, in order, whichever branch
    produced it: `integers` for an exploratory draw, `choice` for a greedy one.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.actions: list[int] = []

    def random(self, *args, **kwargs):
        return self._rng.random(*args, **kwargs)

    def integers(self, *args, **kwargs):
        value = self._rng.integers(*args, **kwargs)
        self.actions.append(int(value))
        return value

    def choice(self, *args, **kwargs):
        value = self._rng.choice(*args, **kwargs)
        self.actions.append(int(value))
        return value


def one_episode_config(**changes) -> TabularConfig:
    """A single-episode config. Values still come from the yaml; only the run length is narrowed."""
    return replace(BASE_CONFIG, episodes=1, **changes)


def _run_one(
    algo,
    *,
    terminated: bool,
    truncated: bool,
    next_row: np.ndarray,
    epsilon: float | None,
    seed: int,
) -> tuple[TabularConfig, tuple[int, ...], int, float]:
    """Drive `algo` for a single transition and report the config, the actions it drew and the
    one Q-entry that moved."""
    changes = {} if epsilon is None else {"epsilon_start": epsilon, "epsilon_end": epsilon}
    config = one_episode_config(**changes)
    env = OneStepEnv(terminated=terminated, truncated=truncated)
    q_table = make_q_table()
    q_table[OneStepEnv.NEXT] = np.array(next_row, dtype=np.float64)

    rng = RecordingRng(make_rng(seed))
    result = algo(env, config, rng=rng, q_table=q_table)
    updated = result.q_table[OneStepEnv.START]
    touched = np.flatnonzero(updated != 0.0)
    assert len(touched) == 1, "exactly one action was taken, so exactly one entry may move"
    return config, tuple(rng.actions), int(touched[0]), float(updated[touched[0]])


def run_one_transition(
    *,
    terminated: bool,
    truncated: bool,
    next_row: np.ndarray = NEXT_Q_ROW,
    epsilon: float | None = None,
    seed: int = 0,
) -> Transition:
    """Run one Q-learning transition into a next state whose Q-row is known and non-uniform.

    `epsilon` pins both ends of the schedule when given, so the behaviour policy at episode 0 is
    forced rather than inherited; `seed` picks which action the forced exploration draws.
    """
    config, _, action, value = _run_one(
        q_learning,
        terminated=terminated,
        truncated=truncated,
        next_row=next_row,
        epsilon=epsilon,
        seed=seed,
    )
    return Transition(config, action, value)


def run_one_sarsa_transition(
    *,
    terminated: bool,
    truncated: bool,
    next_row: np.ndarray = NEXT_Q_ROW,
    epsilon: float | None = None,
    seed: int = 0,
) -> SarsaTransition:
    """The mirror of `run_one_transition` for SARSA, which draws a second action at `s'`.

    On a terminated transition there is no successor action to draw and `next_action` is None; on a
    truncated one the agent is still playing, so SARSA must have drawn `a'` in order to bootstrap
    and exactly two actions must appear in the recording.
    """
    config, actions, action, value = _run_one(
        algorithms.sarsa,
        terminated=terminated,
        truncated=truncated,
        next_row=next_row,
        epsilon=epsilon,
        seed=seed,
    )
    expected_draws = 1 if terminated else 2
    assert len(actions) == expected_draws, (
        f"SARSA should draw {expected_draws} action(s) on a "
        f"{'terminated' if terminated else 'truncated'} one-step episode "
        f"(a at s, then a' at s' unless the game ended), but drew {actions}"
    )
    assert actions[0] == action, (
        f"the entry that moved belongs to action {action} but the first action drawn was "
        f"{actions[0]}: the update was applied to a different action than the one taken"
    )
    next_action = None if terminated else actions[1]
    return SarsaTransition(config, action, next_action, value)


# --- B4: random tie-breaking ---------------------------------------------------------------------


def test_tie_break_is_random_not_argmax():
    """With epsilon 0 and an all-equal row, every action must appear across many calls.

    `np.argmax` would return 0 (UP) on all 400 of them, which is precisely the graded failure.
    """
    rng = make_rng(0)
    q_row = np.zeros(N_ACTIONS)
    seen = {select_action(q_row, 0.0, rng) for _ in range(400)}
    assert seen == set(range(N_ACTIONS))


def test_tie_break_covers_only_the_tied_best_actions():
    """A partial tie must split between the tied winners and never pick a losing action."""
    rng = make_rng(1)
    q_row = np.array([0.0, 5.0, 5.0, 1.0])
    seen = {select_action(q_row, 0.0, rng) for _ in range(400)}
    assert seen == {1, 2}


def test_greedy_selection_picks_the_unique_best():
    rng = make_rng(2)
    q_row = np.array([0.0, 0.0, 0.0, 7.0])
    assert all(select_action(q_row, 0.0, rng) == 3 for _ in range(50))


def test_exploration_reaches_every_action():
    """Epsilon 1 must sample uniformly, not defer to the greedy action."""
    rng = make_rng(3)
    q_row = np.array([0.0, 0.0, 0.0, 7.0])
    counts = np.bincount([select_action(q_row, 1.0, rng) for _ in range(2000)],
                         minlength=N_ACTIONS)
    assert counts.min() > 300, f"expected roughly uniform exploration, got {counts}"


def test_module_source_never_calls_argmax():
    """A direct read of B4: no executable reference to `argmax` anywhere in the module.

    Parsed rather than grepped, because the docstrings name `argmax` in order to ban it and a plain
    substring search would fire on the prohibition itself.
    """
    tree = ast.parse(Path(algorithms.__file__).read_text(encoding="utf-8"))
    names = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]
    names += [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
    assert "argmax" not in names


# --- B2: the target on terminal and truncated transitions -----------------------------------------


def test_terminal_target_has_no_bootstrap_term():
    """`target = r` when the episode really ended, however valuable the successor state looks."""
    config, _, updated = run_one_transition(terminated=True, truncated=False)
    expected = config.alpha * 1.0
    assert updated == pytest.approx(expected)


def test_terminal_target_ignores_the_next_state_value_entirely():
    """Changing the successor's Q-row must not move a terminal update by a single float.

    Both rows are non-uniform, and their *orderings* differ as well as their magnitudes, so this
    also rules out a terminal update that quietly indexes the successor row by the action taken.
    """
    _, _, low = run_one_transition(
        terminated=True, truncated=False, next_row=np.array([0.0, -0.5, 0.25, -0.75])
    )
    _, _, high = run_one_transition(
        terminated=True, truncated=False, next_row=np.array([1000.0, -250.0, 500.0, 750.0])
    )
    assert low == pytest.approx(high)


def test_truncation_does_not_suppress_bootstrapping():
    """The step cap is not the game ending: the successor still has value, so the term stays.

    The bootstrap term is `max(Q[s'])` — 7.0, the best entry of `NEXT_Q_ROW` — and not the entry
    belonging to whichever action the behaviour policy happened to take.
    """
    config, action, updated = run_one_transition(terminated=False, truncated=True)
    expected = config.alpha * (1.0 + config.gamma * float(NEXT_Q_ROW.max()))
    assert updated == pytest.approx(expected)
    assert updated > config.alpha * 1.0, "a truncated transition must not be treated as terminal"
    if action != NEXT_GREEDY_ACTION:
        on_policy = config.alpha * (1.0 + config.gamma * float(NEXT_Q_ROW[action]))
        assert updated != pytest.approx(on_policy), (
            f"action {action} was taken and its successor value is {NEXT_Q_ROW[action]}, so an "
            f"update of {on_policy} would mean the target bootstrapped from Q[s'][a] (SARSA) "
            f"instead of max(Q[s']) (Q-learning)"
        )


def test_truncated_and_terminal_targets_differ():
    """The two endings must produce different updates: the whole point of the distinction."""
    _, _, terminal = run_one_transition(terminated=True, truncated=False)
    _, _, truncated = run_one_transition(terminated=False, truncated=True)
    assert terminal != pytest.approx(truncated)


# --- C: the update is off-policy, and is not SARSA ------------------------------------------------


def test_successor_rows_are_non_uniform_so_the_two_targets_can_differ():
    """A guard on the fixtures themselves: a flat successor row disarms every test below it.

    If `NEXT_Q_ROW` were ever flattened back to four equal entries, `max(Q[s'])` and `Q[s'][a]`
    would coincide and the off-policy tests would keep passing against an on-policy implementation.
    The distinction is only observable while these rows have a unique maximum and a real spread.
    """
    for name, row, greedy in (
        ("NEXT_Q_ROW", NEXT_Q_ROW, NEXT_GREEDY_ACTION),
        ("OFF_POLICY_NEXT_ROW", OFF_POLICY_NEXT_ROW, OFF_POLICY_GREEDY_ACTION),
    ):
        assert row.shape == (N_ACTIONS,), f"{name} must give every action a value"
        assert int(np.argmax(row)) == greedy, f"{name}: recorded greedy action is stale"
        assert np.count_nonzero(row == row.max()) == 1, f"{name}: the maximum must be unique"
        assert float(row.max() - row.min()) > 1.0, f"{name}: the spread is too small to see"


def test_update_bootstraps_from_the_max_not_from_the_action_taken():
    """The surviving-mutant test: swap `max(Q[s'])` for `Q[s'][a]` and this must go red.

    Epsilon is pinned at 1.0 so the behaviour policy is pure exploration and the action taken is
    usually *not* the greedy one; the seeds are fixed so the sample is reproducible rather than
    lucky. Whatever action is drawn, an off-policy update targets `r + gamma * max(Q[s'])` — 96.0
    here, an update of +9.6 — while an on-policy (SARSA) update targets `r + gamma * Q[s'][a]`,
    which is -37.0 and an update of -3.7 for three of the four actions. The two predictions differ
    in sign, so no tolerance and no rounding can confuse them.
    """
    seeds = tuple(range(24))
    off_policy_target = None
    actions: list[int] = []

    for seed in seeds:
        config, action, updated = run_one_transition(
            terminated=False,
            truncated=True,
            next_row=OFF_POLICY_NEXT_ROW,
            epsilon=1.0,
            seed=seed,
        )
        actions.append(action)
        off_policy_target = config.alpha * (
            1.0 + config.gamma * float(OFF_POLICY_NEXT_ROW.max())
        )
        on_policy_target = config.alpha * (
            1.0 + config.gamma * float(OFF_POLICY_NEXT_ROW[action])
        )
        on_policy_alternatives = {
            index: config.alpha * (1.0 + config.gamma * float(value))
            for index, value in enumerate(OFF_POLICY_NEXT_ROW)
            if index != OFF_POLICY_GREEDY_ACTION
        }
        matched = [
            index
            for index, candidate in on_policy_alternatives.items()
            if updated == pytest.approx(candidate)
        ]
        assert updated == pytest.approx(off_policy_target), (
            f"seed {seed}: the behaviour policy took action {action} and Q[s'] is "
            f"{OFF_POLICY_NEXT_ROW.tolist()}.\n"
            f"  off-policy (Q-learning) target alpha * (r + gamma * max(Q[s'])) = "
            f"{off_policy_target}\n"
            f"  on-policy  (SARSA)      target alpha * (r + gamma * Q[s'][a]) = "
            f"{on_policy_target} for the action taken, "
            f"{sorted(set(on_policy_alternatives.values()))} for any non-greedy action\n"
            f"  observed = {updated}\n"
            + (
                f"  -> the observed value is the SARSA target for action(s) {matched}: the update "
                f"is bootstrapping from an action's own value at s', not from the best action "
                f"available there. Q-learning must be off-policy."
                if matched
                else "  -> the update does not match the off-policy target."
            )
        )

    exploratory = [action for action in actions if action != OFF_POLICY_GREEDY_ACTION]
    assert len(exploratory) >= len(seeds) // 2, (
        f"this test only separates the two update rules on transitions where the action taken is "
        f"not the greedy one at s'; only {len(exploratory)} of {len(seeds)} seeds explored, so the "
        f"sample is too greedy to prove anything. Actions: {actions}"
    )
    assert off_policy_target is not None and off_policy_target > 0.0


def test_greedy_and_exploratory_transitions_receive_the_same_update():
    """An off-policy target does not depend on the behaviour policy — so the update cannot either.

    Every seed produces the same number even though the action taken varies, which is the property
    that fails the moment the target reads `Q[s'][a]`.
    """
    updates = {}
    for seed in range(24):
        _, action, updated = run_one_transition(
            terminated=False,
            truncated=True,
            next_row=OFF_POLICY_NEXT_ROW,
            epsilon=1.0,
            seed=seed,
        )
        updates.setdefault(action, []).append(updated)

    assert len(updates) == N_ACTIONS, f"forced exploration should reach every action, got {updates}"
    values = [value for group in updates.values() for value in group]
    assert max(values) - min(values) == pytest.approx(0.0), (
        f"the update varied with the action taken ({updates}), which is the signature of an "
        f"on-policy (SARSA) target; an off-policy target is identical for all four actions"
    )


# --- B3: epsilon comes from the config file ------------------------------------------------------


def test_epsilon_schedule_is_built_from_config():
    schedule = epsilon_schedule(BASE_CONFIG)
    assert isinstance(schedule, LinearEpsilon)
    assert schedule.start == BASE_CONFIG.epsilon_start
    assert schedule.end == BASE_CONFIG.epsilon_end
    assert schedule.decay_episodes == BASE_CONFIG.epsilon_decay_episodes


def test_recorded_epsilons_follow_the_config_not_a_literal():
    """Every logged epsilon must equal the config-built schedule, episode by episode.

    Written against a *modified* config so that a hardcoded 1.0 -> 0.05 in the loop would fail:
    matching the yaml's own numbers would not distinguish the two.
    """
    config = replace(
        BASE_CONFIG,
        episodes=40,
        epsilon_start=0.8,
        epsilon_end=0.2,
        epsilon_decay_episodes=20,
    )
    env = GridWorld(level_index=0, seed=config.seed, max_steps=config.max_steps_per_episode)
    result = q_learning(env, config, rng=make_rng(config.seed))

    schedule = LinearEpsilon(start=0.8, end=0.2, decay_episodes=20)
    assert result.epsilons == pytest.approx([schedule(e) for e in range(config.episodes)])
    assert result.epsilons[0] == pytest.approx(0.8)
    assert result.epsilons[-1] == pytest.approx(0.2)


def test_episode_count_comes_from_config():
    config = replace(BASE_CONFIG, episodes=7)
    env = GridWorld(level_index=0, seed=config.seed, max_steps=config.max_steps_per_episode)
    result = q_learning(env, config, rng=make_rng(config.seed))
    assert len(result.history) == 7


# --- B5: the learned policy is a shortest path ---------------------------------------------------


def test_bfs_optimum_for_level_0_matches_a_hand_calculation():
    """Start (0,0); apples (2,8), (5,7), (7,8). 10 + 4 + 3 = 17 on a grid with no obstacles."""
    assert optimal_collection_steps(0) == 17


def test_short_training_run_reaches_the_bfs_optimum_on_level_0():
    """The B5 demonstration, shrunk to a few seconds.

    The decay window is narrowed alongside the episode count: leaving it at the yaml's 2000 while
    running 1200 episodes would hold epsilon near 1.0 for the whole run and test nothing but chance.
    Both values still travel inside a `TabularConfig`. 1200/800 was checked to reach the optimum on
    seeds 0-5; 600/400 misses on some, which would make this test a coin flip rather than a check.
    """
    config = replace(BASE_CONFIG, episodes=1200, epsilon_decay_episodes=800, seed=0)
    env = GridWorld(level_index=0, seed=config.seed, max_steps=config.max_steps_per_episode)
    result = q_learning(env, config, rng=make_rng(config.seed))

    rollout = greedy_rollout(env, result.q_table, rng=make_rng(config.seed))
    assert rollout.collected == len(env.collectible_cells) == 3
    assert not rollout.died and not rollout.truncated
    assert rollout.total_return == pytest.approx(3.0)
    assert rollout.steps == optimal_collection_steps(0)


def test_q_table_roundtrips_through_disk(scratch_dir):
    """Playback must not need retraining, so a saved table has to reload value-for-value."""
    config = replace(BASE_CONFIG, episodes=20)
    env = GridWorld(level_index=0, seed=config.seed, max_steps=config.max_steps_per_episode)
    result = q_learning(env, config, rng=make_rng(config.seed))

    path = algorithms.save_q_table(result.q_table, scratch_dir / "q.npz")
    reloaded = algorithms.load_q_table(path)
    assert set(reloaded) == set(result.q_table)
    for state, row in result.q_table.items():
        assert reloaded[state] == pytest.approx(row)


def test_history_csv_has_the_documented_columns(scratch_dir):
    config = replace(BASE_CONFIG, episodes=5)
    env = GridWorld(level_index=0, seed=config.seed, max_steps=config.max_steps_per_episode)
    result = q_learning(env, config, rng=make_rng(config.seed))

    path = result.write_history_csv(scratch_dir / "history.csv")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].split(",") == list(algorithms.HISTORY_FIELDS)
    assert len(lines) == 6  # header plus one row per episode


# --- C1/C2: SARSA is on-policy, and shares Q-learning's schedule --------------------------------
#
# Every test below is the mirror image of one above it. Where the Q-learning tests assert the update
# equals `max(Q[s'])` and is invariant to the action taken, these assert it equals `Q[s'][a']` and
# *varies* with it. Both sets run against the same successor rows, so if the two implementations
# ever converge on one update rule, one side or the other goes red.


def test_sarsa_update_bootstraps_from_the_action_taken_not_the_max():
    """The mirror of `test_update_bootstraps_from_the_max_not_from_the_action_taken`.

    Epsilon is pinned at 1.0 so the action drawn at `s'` is pure exploration and usually not the
    greedy one; the seeds are fixed so the sample is reproducible rather than lucky. Under
    `OFF_POLICY_NEXT_ROW` the two candidate targets are 96.0 (off-policy, an update of +9.6) and
    -37.0 (on-policy for any of the three non-greedy actions, an update of -3.7). They differ in
    sign, so no tolerance can blur them, and an implementation that reached for `max` here would
    fail on every exploratory seed.
    """
    seeds = tuple(range(24))
    next_actions: list[int] = []

    for seed in seeds:
        config, action, next_action, updated = run_one_sarsa_transition(
            terminated=False,
            truncated=True,
            next_row=OFF_POLICY_NEXT_ROW,
            epsilon=1.0,
            seed=seed,
        )
        assert next_action is not None
        next_actions.append(next_action)

        on_policy_target = config.alpha * (
            1.0 + config.gamma * float(OFF_POLICY_NEXT_ROW[next_action])
        )
        off_policy_target = config.alpha * (
            1.0 + config.gamma * float(OFF_POLICY_NEXT_ROW.max())
        )
        message = (
            f"seed {seed}: the behaviour policy took action {action} at s and then drew "
            f"a' = {next_action} at s', where Q[s'] is {OFF_POLICY_NEXT_ROW.tolist()}.\n"
            f"  on-policy  (SARSA)      target alpha * (r + gamma * Q[s'][a']) = "
            f"{on_policy_target}\n"
            f"  off-policy (Q-learning) target alpha * (r + gamma * max(Q[s'])) = "
            f"{off_policy_target}\n"
            f"  observed = {updated}"
        )
        assert updated == pytest.approx(on_policy_target), message
        if next_action != OFF_POLICY_GREEDY_ACTION:
            assert updated != pytest.approx(off_policy_target), (
                message + "\n  -> the observed value is the off-policy target: SARSA bootstrapped "
                "from the best action available at s' instead of the action it actually took "
                "next. SARSA must be on-policy."
            )

    exploratory = [action for action in next_actions if action != OFF_POLICY_GREEDY_ACTION]
    assert len(exploratory) >= len(seeds) // 2, (
        f"this test only separates the two update rules on transitions where a' is not the greedy "
        f"action at s'; only {len(exploratory)} of {len(seeds)} seeds explored, so the sample is "
        f"too greedy to prove anything. Next actions: {next_actions}"
    )


def test_sarsa_update_varies_with_the_action_actually_taken():
    """The mirror of `test_greedy_and_exploratory_transitions_receive_the_same_update`.

    Varying with `a'` *is* the definition of on-policy, and it is the one property no off-policy
    target can reproduce: Q-learning's answer is the same number for all four actions. So this
    asserts the opposite of its twin — four distinct updates, one per action drawn at `s'`.
    """
    updates: dict[int, list[float]] = {}
    for seed in range(24):
        _, _, next_action, updated = run_one_sarsa_transition(
            terminated=False,
            truncated=True,
            next_row=OFF_POLICY_NEXT_ROW,
            epsilon=1.0,
            seed=seed,
        )
        assert next_action is not None
        updates.setdefault(next_action, []).append(updated)

    assert len(updates) == N_ACTIONS, (
        f"forced exploration should draw every action at s', got {updates}"
    )
    for next_action, values in updates.items():
        assert max(values) - min(values) == pytest.approx(0.0), (
            f"a' = {next_action} produced inconsistent updates {values}; the target is a function "
            f"of a' alone"
        )
    spread = max(v for group in updates.values() for v in group) - min(
        v for group in updates.values() for v in group
    )
    assert spread > 1.0, (
        f"the update did not vary with the action taken ({updates}); an on-policy target must "
        f"track Q[s'][a'], and an update that is identical for all four actions is the signature "
        f"of an off-policy (Q-learning) target"
    )


def test_sarsa_terminal_target_has_no_bootstrap_term():
    """`target = r` on a real ending, for SARSA too: there is no a' to take after the game ends."""
    config, _, next_action, updated = run_one_sarsa_transition(terminated=True, truncated=False)
    assert next_action is None
    assert updated == pytest.approx(config.alpha * 1.0)


def test_sarsa_truncation_still_bootstraps():
    """The step cap is not the game ending, so SARSA draws a' at s' and keeps the bootstrap."""
    config, _, next_action, updated = run_one_sarsa_transition(terminated=False, truncated=True)
    assert next_action is not None
    expected = config.alpha * (1.0 + config.gamma * float(NEXT_Q_ROW[next_action]))
    assert updated == pytest.approx(expected)
    assert updated != pytest.approx(config.alpha * 1.0), (
        "a truncated transition must not be treated as terminal"
    )


def _function_source(name: str) -> ast.FunctionDef:
    tree = ast.parse(Path(algorithms.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in {algorithms.__file__}")


def _called_names(node: ast.AST) -> set[str]:
    return {
        call.func.id
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }


def test_sarsa_body_never_takes_a_max_over_the_successor_row():
    """C1 read straight off the source: no `max` anywhere inside `sarsa`.

    The behavioural tests above already pin the target, but this is the check that survives a
    refactor — `q_learning`'s `float(q[next_state].max())` copy-pasted into `sarsa` is the single
    most likely way this file loses its second algorithm, and it would leave the level 1 comparison
    showing two identical policies with no error anywhere.
    """
    node = _function_source("sarsa")
    attributes = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    assert "max" not in attributes and "max" not in names, (
        "sarsa references `max`; its target must be Q[s'][a'], not max(Q[s'])"
    )
    q_attributes = {
        n.attr for n in ast.walk(_function_source("q_learning")) if isinstance(n, ast.Attribute)
    }
    assert "max" in q_attributes, (
        "q_learning stopped taking a max, so this test is no longer checking a real difference"
    )


def test_both_algorithms_build_the_schedule_at_the_same_construction_site():
    """C2 by inspection: neither loop constructs a `LinearEpsilon` of its own.

    Both call `epsilon_schedule(config)`, and that function is the only place in the module where a
    schedule is built — so "SARSA uses the same exploration schedule as Q-learning" is a fact about
    one function rather than a promise about two.
    """
    for name in ("q_learning", "sarsa"):
        called = _called_names(_function_source(name))
        assert "epsilon_schedule" in called, f"{name} must build its schedule from the config"
        assert "select_action" in called, f"{name} must use the shared epsilon-greedy selector"
        assert "make_q_table" in called, f"{name} must use the shared table constructor"
        assert "LinearEpsilon" not in called, (
            f"{name} constructs its own schedule; C2 requires the shared one"
        )

    module = ast.parse(Path(algorithms.__file__).read_text(encoding="utf-8"))
    constructions = [
        call
        for call in ast.walk(module)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "LinearEpsilon"
    ]
    assert len(constructions) == 1, (
        f"expected exactly one LinearEpsilon construction site (inside `epsilon_schedule`), found "
        f"{len(constructions)}"
    )


def test_both_algorithms_follow_the_same_epsilon_sequence():
    """C2 behaviourally: the epsilon logged episode-by-episode is identical for the two runs."""
    config = replace(
        BASE_CONFIG,
        episodes=40,
        epsilon_start=0.9,
        epsilon_end=0.1,
        epsilon_decay_episodes=25,
    )
    schedule = LinearEpsilon(start=0.9, end=0.1, decay_episodes=25)
    expected = [schedule(episode) for episode in range(config.episodes)]

    for algo in (q_learning, algorithms.sarsa):
        env = GridWorld(level_index=1, seed=config.seed, max_steps=config.max_steps_per_episode)
        result = algo(env, config, rng=make_rng(config.seed))
        assert result.epsilons == pytest.approx(expected)


def test_sarsa_reports_itself_and_records_one_row_per_episode():
    config = replace(BASE_CONFIG, episodes=9)
    env = GridWorld(level_index=1, seed=config.seed, max_steps=config.max_steps_per_episode)
    result = algorithms.sarsa(env, config, rng=make_rng(config.seed))
    assert result.algo == "sarsa"
    assert result.level == 1
    assert len(result.history) == config.episodes


def test_sarsa_learns_a_safe_route_to_the_apple_on_the_cliff_level():
    """The C3 claim in miniature: SARSA solves level 1 without walking into the fire.

    Not a check that it is *shorter* than Q-learning's route — it is not, and it should not be. The
    check is that the on-policy update still converges to a policy that collects the apple, which
    is what makes the route comparison a comparison of two working agents.
    """
    config = replace(BASE_CONFIG, episodes=2000, epsilon_decay_episodes=1400, seed=0)
    env = GridWorld(level_index=1, seed=config.seed, max_steps=config.max_steps_per_episode)
    result = algorithms.sarsa(env, config, rng=make_rng(config.seed))

    rollout = greedy_rollout(env, result.q_table, rng=make_rng(config.seed))
    assert rollout.collected == len(env.collectible_cells) == 1
    assert not rollout.died and not rollout.truncated
    assert rollout.total_return == pytest.approx(1.0)
    assert rollout.steps >= optimal_collection_steps(1)


def test_policy_rollout_honours_its_epsilon():
    """`policy_rollout` must actually use its epsilon — the death-rate measurement depends on it.

    Run against a hand-built table that prefers RIGHT in every state, which on level 1 walks
    straight off the start into the fire. Greedy must do exactly that on every seed; at epsilon 1
    the table is ignored and some seed must do something else. A synthetic table rather than a
    trained one on purpose: with `REWARD_STEP = 0` a briefly trained table is all zeros away from
    the apple, every action ties, and a greedy rollout is already a random walk — so it would agree
    with an exploring one by accident and the test would prove nothing.
    """
    table = defaultdict(lambda: np.array([0.0, 0.0, 0.0, 1.0]))  # RIGHT is the unique best
    env = GridWorld(level_index=1, seed=0, max_steps=BASE_CONFIG.max_steps_per_episode)

    greedy = {algorithms.greedy_rollout(env, table, rng=make_rng(seed)).path for seed in range(8)}
    assert greedy == {((8, 0), (8, 1))}, "greedy must follow the table into the fire, every time"

    explored = [
        algorithms.policy_rollout(env, table, epsilon=1.0, rng=make_rng(seed)).path
        for seed in range(8)
    ]
    assert any(path != ((8, 0), (8, 1)) for path in explored), (
        f"epsilon 1.0 produced the greedy path on all 8 seeds ({explored}); the epsilon argument "
        f"is being ignored"
    )
