"""Tests for the arena policy overlay — the Part II half of the visibility rule.

Nothing here loads a trained model. A stub standing in for an SB3 policy keeps these in
milliseconds and, more usefully, lets the tests state exactly what the network returned, which is
the only way to prove the panel reports it rather than recomputing something of its own.

Two things are load-bearing. `probe` must never raise: it runs once per frame inside the playback
loop, so an SB3 whose internals moved has to cost the panel and not the demo. And the printed
number must be the network's own — a panel that quietly softmaxes Q-values, or rescales a
probability to make a bar look tidier, is putting a figure on screen the agent never computed,
which is worse than showing nothing.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402  (must follow the driver assignment above)
import torch  # noqa: E402

from arena.constants import OBS_DIM, DirectAction, RotationAction  # noqa: E402
from arena.env import ArenaEnv  # noqa: E402
from arena.policy_view import PolicyView, action_names, probe  # noqa: E402
from arena.render import (  # noqa: E402
    COLOR_POLICY_BAR,
    COLOR_POLICY_BAR_CHOSEN,
    ArenaRenderer,
)


class _StubDistribution:
    def __init__(self, probs: list[float]) -> None:
        self.distribution = self
        self.probs = torch.tensor([probs])


class _StubPolicy:
    """The three attributes `probe` touches on an SB3 `ActorCriticPolicy`."""

    def __init__(self, probs: list[float], value: float) -> None:
        self._probs, self._value = probs, value

    def obs_to_tensor(self, observation):
        return torch.tensor(np.asarray(observation, dtype=np.float32)).unsqueeze(0), None

    def get_distribution(self, _tensor):
        return _StubDistribution(self._probs)

    def predict_values(self, _tensor):
        return torch.tensor([[self._value]])


class _StubModel:
    def __init__(self, probs: list[float], value: float = 0.0) -> None:
        self.policy = _StubPolicy(probs, value)


class _StubQNet:
    def __init__(self, q_values: list[float]) -> None:
        self._q = q_values

    def __call__(self, _tensor):
        return torch.tensor([self._q])


class _StubDQN:
    def __init__(self, q_values: list[float]) -> None:
        self.policy = _StubPolicy([0.0] * len(q_values), 0.0)
        self.q_net = _StubQNet(q_values)


OBSERVATION = np.zeros(OBS_DIM, dtype=np.float32)


# --- action names ---------------------------------------------------------------------------


class TestActionNames:
    def test_names_come_from_the_brief_fixed_enums_in_index_order(self):
        """The panel labels a bar by index, so a reordering here would mislabel every row."""
        assert action_names("rotation") == tuple(m.name for m in RotationAction)
        assert action_names("direct") == tuple(m.name for m in DirectAction)

    def test_the_two_styles_have_different_action_counts(self):
        assert len(action_names("rotation")) == 5
        assert len(action_names("direct")) == 6

    def test_an_unknown_style_is_rejected(self):
        with pytest.raises(ValueError, match="unknown control style"):
            action_names("keyboard")


# --- the probe ------------------------------------------------------------------------------


class TestProbe:
    def test_ppo_probabilities_are_reported_as_the_network_produced_them(self):
        probs = [0.05, 0.10, 0.15, 0.20, 0.20, 0.30]
        view = probe(_StubModel(probs, value=1.25), OBSERVATION, "direct", chosen=5)
        assert view is not None
        assert view.scores == pytest.approx(probs)
        assert view.value == pytest.approx(1.25)
        assert view.score_label == "ACTION PROBABILITY"

    def test_probabilities_pass_through_to_bars_unscaled(self):
        """A probability already lives in [0, 1]; rescaling it would put a length on screen that
        does not match the number printed beside it."""
        probs = [0.05, 0.10, 0.15, 0.20, 0.20, 0.30]
        view = probe(_StubModel(probs), OBSERVATION, "direct", chosen=0)
        assert view.bars == pytest.approx(probs)

    def test_dqn_reports_raw_q_values_not_a_softmax(self):
        """DQN computes no distribution. Softmaxing its Q-values would invent a probability the
        network never produced, so the panel shows Q and says so."""
        q_values = [-3.0, -1.0, 0.0, 1.0, 5.0, 2.0]
        view = probe(_StubDQN(q_values), OBSERVATION, "direct", chosen=4, algo="dqn")
        assert view.scores == pytest.approx(q_values)
        assert view.score_label == "Q-VALUE"
        assert view.value == pytest.approx(5.0)  # max_a Q(s, a)

    def test_q_values_are_min_max_scaled_for_bar_length_only(self):
        q_values = [-3.0, -1.0, 0.0, 1.0, 5.0, 2.0]
        view = probe(_StubDQN(q_values), OBSERVATION, "direct", chosen=4, algo="dqn")
        assert view.bars[0] == pytest.approx(0.0)  # the smallest Q
        assert view.bars[4] == pytest.approx(1.0)  # the largest
        assert view.scores[0] == pytest.approx(-3.0)  # but the printed number is untouched

    def test_an_all_equal_row_does_not_divide_by_zero(self):
        """An untrained net, or a genuine tie. Half-width bars, not a crash."""
        view = probe(_StubDQN([2.0] * 6), OBSERVATION, "direct", chosen=0, algo="dqn")
        assert view.bars == pytest.approx([0.5] * 6)

    def test_a_broken_model_costs_the_panel_and_not_the_demo(self):
        class Exploding:
            @property
            def policy(self):
                raise RuntimeError("SB3 moved this attribute")

        assert probe(Exploding(), OBSERVATION, "direct", chosen=0) is None

    def test_human_play_has_no_model_and_gets_no_panel(self):
        assert probe(None, OBSERVATION, "direct", chosen=0) is None

    def test_a_width_mismatch_is_refused_rather_than_mislabelled(self):
        """Five scores against six action names would silently shift every label by one."""
        assert probe(_StubModel([0.2] * 5), OBSERVATION, "direct", chosen=0) is None

    def test_the_chosen_action_is_named_for_the_highlight_row(self):
        view = probe(_StubModel([0.1] * 6), OBSERVATION, "direct", chosen=int(DirectAction.SHOOT))
        assert view.chosen_name == "SHOOT"


# --- the panel ------------------------------------------------------------------------------


class _Recorder:
    """A font that remembers the strings drawn through it."""

    def __init__(self, font) -> None:
        self.font, self.drawn = font, []

    def render(self, text, *args, **kwargs):
        self.drawn.append(text)
        return self.font.render(text, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.font, name)


def _count(surface: pygame.Surface, color) -> int:
    pixels = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
    return int(np.all(pixels == np.array(color), axis=-1).sum())


@pytest.fixture
def env() -> ArenaEnv:
    world = ArenaEnv("direct")
    world.reset(seed=0)
    return world


class TestPolicyPanel:
    def test_every_action_name_and_score_reaches_the_screen(self, env):
        renderer = ArenaRenderer(headless=True)
        renderer.font_small = _Recorder(renderer.font_small)
        view = probe(_StubModel([0.05, 0.10, 0.15, 0.20, 0.20, 0.30], 1.25),
                     OBSERVATION, "direct", chosen=5)
        renderer.draw(env, view)

        drawn = renderer.font_small.drawn
        for name in action_names("direct"):
            assert name in drawn
        assert "+0.30" in drawn      # the chosen action's probability
        assert "VALUE V(s)" in drawn
        assert "+1.25" in drawn      # the critic's estimate

    def test_the_chosen_action_is_drawn_in_its_own_colour(self, env):
        """Under `deterministic=True` the argmax is what actually ran, and on a near-tie the
        viewer cannot pick it out by bar length alone."""
        renderer = ArenaRenderer(headless=True)
        view = probe(_StubModel([0.17, 0.17, 0.16, 0.17, 0.16, 0.17]),
                     OBSERVATION, "direct", chosen=3)
        surface = renderer.draw(env, view)
        assert _count(surface, COLOR_POLICY_BAR_CHOSEN) > 0
        assert _count(surface, COLOR_POLICY_BAR) > 0

    def test_the_panel_toggles_off(self, env):
        renderer = ArenaRenderer(headless=True)
        view = probe(_StubModel([0.1] * 6), OBSERVATION, "direct", chosen=0)
        with_panel = _count(renderer.draw(env, view), COLOR_POLICY_BAR)
        assert renderer.toggle_policy_overlay() is False
        without = _count(renderer.draw(env, view), COLOR_POLICY_BAR)
        assert with_panel > 0 and without == 0
        assert renderer.toggle_policy_overlay() is True

    def test_no_view_means_no_panel_and_no_crash(self, env):
        """Human play passes no model, so `draw` must stay callable with one argument."""
        renderer = ArenaRenderer(headless=True)
        surface = renderer.draw(env)
        assert _count(surface, COLOR_POLICY_BAR) == 0

    def test_the_panel_survives_the_observation_overlay_being_off(self, env):
        """The two overlays share a side panel but are independently toggled."""
        renderer = ArenaRenderer(headless=True)
        renderer.show_observation_overlay = False
        view = probe(_StubModel([0.1] * 6), OBSERVATION, "direct", chosen=0)
        assert _count(renderer.draw(env, view), COLOR_POLICY_BAR) > 0

    def test_drawing_the_panel_never_mutates_the_simulation(self, env):
        """The renderer reads the env and never writes to it, or evaluation stops matching
        training and the drift looks like a training bug."""
        renderer = ArenaRenderer(headless=True)
        view = probe(_StubModel([0.1] * 6), OBSERVATION, "direct", chosen=0)
        before = (env.steps, env.phase, env.player.health, float(env.episode_reward))
        renderer.draw(env, view)
        assert (env.steps, env.phase, env.player.health, float(env.episode_reward)) == before


class TestPolicyViewContract:
    def test_an_out_of_range_choice_does_not_raise(self):
        view = PolicyView(("NOOP",), (1.0,), (1.0,), chosen=9, value=None,
                          score_label="ACTION PROBABILITY", algo="PPO")
        assert view.chosen_name == "?"


class TestHudAgreesWithThePanel:
    """One frame must describe one decision.

    `env.last_action` is the *previous* step's action, so a HUD reading it sat beside a policy
    panel highlighting the next one and the two disagreed by a frame. On camera that reads as a
    contradiction — the exact opposite of what putting the policy on screen is for.
    """

    def test_the_hud_names_the_action_the_panel_highlights(self, env):
        renderer = ArenaRenderer(headless=True)
        renderer.font_hud = _Recorder(renderer.font_hud)
        # A policy that is certain about SHOOT, on an env whose last action was something else.
        env.step(int(DirectAction.LEFT))
        probs = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        view = probe(_StubModel(probs), OBSERVATION, "direct", chosen=int(DirectAction.SHOOT))
        renderer.draw(env, view)
        assert "SHOOT" in renderer.font_hud.drawn
        assert "LEFT" not in renderer.font_hud.drawn

    def test_human_play_keeps_the_environments_own_action(self, env):
        """No model, no view — the HUD must still report what was actually executed."""
        renderer = ArenaRenderer(headless=True)
        renderer.font_hud = _Recorder(renderer.font_hud)
        env.step(int(DirectAction.LEFT))
        renderer.draw(env)
        assert env.action_name in renderer.font_hud.drawn
