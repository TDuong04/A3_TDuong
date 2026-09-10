"""A3-029: true update evidence and observational overlays."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from dataclasses import replace

import numpy as np
import pygame
import pytest

from common.config import TabularConfig
from eval.play_gridworld import CompareApp, LearningApp, build_parser
from gridworld.algorithms import (
    LiveLearner, make_q_table, q_learning, sarsa, select_action, train_with_intrinsic_reward,
)
from gridworld.env import GridWorld
from gridworld.render import GridRenderer


def config(**kwargs):
    return replace(TabularConfig.from_yaml(), episodes=2, max_steps_per_episode=12,
                   **kwargs)


def app(level=0, algo="q", **kwargs):
    return LearningApp(level, algo, headless=True, seed=7, policy_seed=11,
                       restart_seconds=0.01, learning_config=config(), **kwargs)


def assert_tables_equal(left, right):
    assert left.keys() == right.keys()
    for state in left:
        np.testing.assert_array_equal(left[state], right[state])


@pytest.mark.parametrize("epsilon", [0.0, 0.5, 1.0])
def test_selection_trace_preserves_rng_and_records_branch(epsilon):
    plain, traced = np.random.default_rng(31), np.random.default_rng(31)
    for _ in range(40):
        records = []
        a = select_action(np.array([0., 2., 2., 1.]), epsilon, plain)
        b = select_action(np.array([0., 2., 2., 1.]), epsilon, traced, trace=records)
        assert a == b == records[0].action
        assert records[0].exploratory == (records[0].roll < epsilon)
        assert plain.bit_generator.state == traced.bit_generator.state


@pytest.mark.parametrize("algo", ["q", "sarsa"])
@pytest.mark.parametrize("strength", [0.0, 0.5])
@pytest.mark.parametrize("level", range(7))
def test_live_updates_match_existing_training(algo, strength, level):
    cfg = config(intrinsic_reward_strength=strength)
    training_env = GridWorld(level, seed=9, max_steps=12)
    live_env = GridWorld(level, seed=9, max_steps=12)
    transitions = [[], []]
    for env, log in zip((training_env, live_env), transitions):
        original = env.step
        def tracked(action, original=original, log=log):
            result = original(action)
            log.append((action, result))
            return result
        env.step = tracked
    if strength:
        trained = train_with_intrinsic_reward(training_env, cfg, algo=algo)
    else:
        trained = (q_learning if algo == "q" else sarsa)(training_env, cfg)
    live_env.reset()
    learner = LiveLearner(live_env, cfg, algo)
    for episode in range(cfg.episodes):
        if episode:
            live_env.reset()
            learner.begin_episode()
        while not live_env.done:
            u = learner.step()
            assert u.target == u.shaped_reward + cfg.gamma * u.bootstrap
            assert u.error == u.target - u.q_before
            assert u.q_after == u.q_before + cfg.alpha * u.error
            assert u.intrinsic_bonus == pytest.approx(strength / np.sqrt(u.prior_visits + 1))
    assert transitions[0] == transitions[1]
    assert_tables_equal(trained.q_table, learner.q)


@pytest.mark.parametrize("algo", ["q", "sarsa"])
@pytest.mark.parametrize("level", range(7))
def test_overlays_are_read_only_and_render_all_levels(algo, level):
    hidden, shown = app(level, algo), app(level, algo)
    hidden.renderer.show_debug = False
    shown.renderer.show_visits = True
    for _ in range(20):
        for a in (hidden, shown):
            if a.env.done:
                a.reset()
            a.policy_step()
        # Repeated draws and toggles must never consume RNG, insert table rows,
        # recount a visit or modify an immutable snapshot.
        previous = shown.live.latest
        for _ in range(3):
            shown.draw()
        assert shown.live.latest is previous
        assert hidden.env.state == shown.env.state
        assert hidden.env.monster_positions == shown.env.monster_positions
        assert hidden.live.latest == shown.live.latest
        assert hidden.live.cell_visits == shown.live.cell_visits
        assert hidden.live.rng.bit_generator.state == shown.live.rng.bit_generator.state
        assert_tables_equal(hidden.live.q, shown.live.q)
    assert shown.draw().get_size() == shown.renderer.surface_size(shown.env.n_rows, shown.env.n_cols)


@pytest.mark.parametrize("algo", ["q", "sarsa"])
def test_bootstrap_uses_original_successor_values_and_sarsa_carries_action(algo):
    env = GridWorld(0, seed=3, max_steps=5)
    q = make_q_table()
    # Cover every possible successor, including blocked moves back into s.
    for row in range(env.n_rows):
        for col in range(env.n_cols):
            for mask in range(1 << len(env.collectible_cells)):
                q[(row, col, False, mask)] = np.array([1., 3., 8., 2.])
    learner = LiveLearner(env, config(epsilon_start=1., epsilon_end=1.), algo,
                          rng=np.random.default_rng(4), q_table=q)
    before = {state: values.copy() for state, values in q.items()}
    u = learner.step()
    expected = (before[u.next_state].max() if algo == "q"
                else before[u.next_state][u.next_action])
    assert u.bootstrap == expected
    assert u.q_before == before[u.state][u.action]
    lines = "\n".join(GridRenderer.debug_lines(u))
    assert ("max_a' Q(s',a')" if algo == "q" else "Q(s', a'_chosen)") in lines
    assert f"= {u.bootstrap:+.9g}" in lines
    assert f"= {u.target:+.9g}" in lines
    assert f"= {u.error:+.9g}" in lines
    assert f"= {u.q_after:+.9g}" in lines
    if algo == "sarsa":
        pending = learner.pending
        following = learner.step()
        assert following.action == u.next_action
        assert following.selection is pending


@pytest.mark.parametrize("algo", ["q", "sarsa"])
@pytest.mark.parametrize("terminal", [False, True])
def test_terminal_suppresses_bootstrap_but_truncation_keeps_it(algo, terminal):
    env = GridWorld(0, seed=3)
    state = env.state
    def ending(action):
        return state, 2.0, True, {"terminated": terminal, "truncated": not terminal}
    env.step = ending
    q = make_q_table()
    q[state] = np.array([1., 3., 8., 2.])
    learner = LiveLearner(env, config(epsilon_start=0., epsilon_end=0.), algo, q_table=q)
    u = learner.step()
    assert u.bootstrap == (0 if terminal else 8)
    assert u.target == 2 + learner.config.gamma * u.bootstrap
    assert (u.next_action is None) == (terminal or algo == "q")
    assert ("not used (terminal)" in "\n".join(GridRenderer.debug_lines(u))) == terminal


def test_intrinsic_text_and_counts_use_prior_full_arrival_state():
    a = app(6, intrinsic_strength=0.5)
    start = a.env.state
    prior = a.live.counts.visits(start)
    assert prior == 1
    a.policy_step()
    u = a.live.latest
    assert a.live.counts.visits(u.next_state) == u.prior_visits + 1
    lines = GridRenderer.debug_lines(u)
    assert f"Environment reward = {u.reward:+.9g}" in lines
    assert f"Shaped reward = r_env + r_i = {u.shaped_reward:+.9g}" in lines
    assert f"Intrinsic reward r_i = {u.intrinsic_bonus:+.9g}" in lines
    assert sum(a.live.cell_visits.values()) == 2
    # No rendered line should be clipped at the debug panel boundary.
    assert max(a.renderer.font_small.size(line)[0] for line in lines) <= a.renderer.debug_width - 24


def key(a, value):
    a.handle_event(pygame.event.Event(pygame.KEYDOWN, key=value))


def test_controls_reset_terminal_hold_and_comparison_selection():
    apps = [app(algo=algo) for algo in ("q", "sarsa")]
    compare = CompareApp(apps, headless=True, restart_seconds=0.01)
    key(compare, pygame.K_n)
    assert all(a.env.steps == 1 and a.paused for a in apps)
    snapshots = [a.live.latest for a in apps]
    compare.update(10)
    assert [a.live.latest for a in apps] == snapshots
    for expected in (0, 1, None):
        key(compare, pygame.K_TAB)
        assert compare.selected_index == expected
        assert compare.visible_apps == (tuple(apps) if expected is None else (apps[expected],))
        compare.draw()
    key(compare, pygame.K_i)
    key(compare, pygame.K_v)
    assert all(not a.renderer.show_debug and a.renderer.show_visits for a in apps)
    for a in apps:
        while not a.env.done:
            a.policy_step()
    final = [a.live.latest for a in apps]
    compare.update(10)
    assert [a.live.latest for a in apps] == final
    key(compare, pygame.K_r)
    assert all(a.live.latest is None and a.env.steps == 0 for a in apps)
    assert all(sum(a.live.cell_visits.values()) == 1 for a in apps)
    key(compare, pygame.K_6)
    assert all(a.env.level_index == 6 and a.live.latest is None for a in apps)
    assert all(a.live.env is a.env for a in apps)


def test_frozen_evaluation_remains_default():
    assert not build_parser().parse_args([]).learn
    assert "Frozen-policy playback does not update Q values." in GridRenderer.debug_lines(None)


@pytest.mark.parametrize("cell_size", [12, 34, 100])
def test_debug_panel_fits_all_lines_at_custom_cell_sizes(cell_size):
    a = app(6, algo="sarsa", intrinsic_strength=0.5, cell_size=cell_size)
    a.policy_step()
    renderer = a.renderer
    lines = renderer.debug_lines(a.live.latest)
    width, height = a.draw().get_size()
    assert max(renderer.font_small.size(line)[0] for line in lines) <= renderer.debug_width - 24
    bottom = (renderer.hud_height + 10 + renderer.font_title.get_height() + 12
              + len(lines) * (renderer.font_small.get_height() + 8))
    assert bottom <= height


def test_manual_moves_clear_update_and_next_learning_action_is_valid():
    a = app(algo="sarsa")
    a.policy_step()
    key(a, pygame.K_RIGHT)
    assert a.live.latest is None
    assert sum(a.live.cell_visits.values()) == a.env.steps + 1
    state = a.env.state
    pending = a.live.pending
    key(a, pygame.K_n)
    assert a.live.latest.state == state
    assert a.live.latest.selection is pending


def test_single_app_holds_terminal_update_when_paused():
    a = app()
    while not a.env.done:
        key(a, pygame.K_n)
    final = a.live.latest
    a.update(10)
    assert a.live.latest is final
    assert a.env.done
