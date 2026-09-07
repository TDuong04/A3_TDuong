"""What the trained policy is thinking, read out of an SB3 model for the renderer.

The HUD in `render.py` shows what the *environment* is doing — phase, health, score — and the
observation overlay shows what the agent *sees*. Neither shows what the network actually computed,
so on camera a good policy and a stuck one look identical: both are just a ship moving. This module
closes that gap by extracting, for a single observation, the quantity the algorithm ranks actions
by and the critic's opinion of the current state.

That is the CLAUDE.md "visibility first" rule applied to Part II: the marker sees SHOOT rise to
0.9 as an enemy lines up, and sees the value estimate fall as health drops. It is the Part II
counterpart of the gridworld's Q-value heatmap and policy arrows.

Contract: this module reads a model and an observation and returns a plain dataclass. It imports
neither pygame nor the renderer, so `eval` can build a `PolicyView` in a headless run and the
renderer stays a pure consumer. Nothing here writes to the model or the env.

Algorithm differences are deliberately preserved rather than smoothed over:

  - PPO is a stochastic policy, so the honest quantity is the action *probability* from its
    categorical distribution, plus the critic's `V(s)`.
  - DQN has no distribution and no critic. Its analogue is the raw Q-value per action, and its
    "value" is `max_a Q(s, a)`. Printing softmaxed Q-values would invent a probability the network
    never computed.

`score_label` carries which of the two is on screen so the panel can never mislabel them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import ACTION_ENUMS


@dataclass(frozen=True)
class PolicyView:
    """One frame of policy internals, ready to draw.

    `scores` are the raw numbers to print (probabilities for PPO, Q-values for DQN) and `bars` are
    those same numbers mapped to [0, 1] purely for bar length. They are kept separate because a
    Q-value of -3.2 has no sensible bar length of its own, and rescaling the printed number to make
    the bar work would put a figure on screen that the network never produced.
    """

    action_names: tuple[str, ...]
    scores: tuple[float, ...]
    bars: tuple[float, ...]
    chosen: int
    value: float | None
    score_label: str
    algo: str

    @property
    def chosen_name(self) -> str:
        """The name of the action actually taken, for the panel's highlight row."""
        if 0 <= self.chosen < len(self.action_names):
            return self.action_names[self.chosen]
        return "?"


def action_names(control_style: str) -> tuple[str, ...]:
    """Action names for a control style, in index order, from the brief-fixed enums."""
    try:
        enum = ACTION_ENUMS[control_style]
    except KeyError:
        raise ValueError(
            f"unknown control style {control_style!r}; expected one of {tuple(ACTION_ENUMS)}"
        ) from None
    return tuple(member.name for member in enum)


def _bars_from_scores(scores: np.ndarray) -> tuple[float, ...]:
    """Map arbitrary scores onto [0, 1] bar lengths.

    Probabilities already live in [0, 1] and are passed through, so a 0.9 draws as 90% of the
    track. Q-values are min-max scaled across the row instead: their absolute magnitude is not
    meaningful on a fixed track, but their *ordering* is exactly what the viewer needs to see.
    An all-equal row (an untrained net, or a genuine tie) maps to a half-width bar rather than
    dividing by zero.
    """
    if scores.size == 0:
        return ()
    low, high = float(scores.min()), float(scores.max())
    if 0.0 <= low and high <= 1.0:
        return tuple(float(value) for value in scores)
    span = high - low
    if span <= 1e-12:
        return tuple(0.5 for _ in scores)
    return tuple(float((value - low) / span) for value in scores)


def probe(
    model: Any,
    observation: np.ndarray,
    control_style: str,
    chosen: int,
    *,
    algo: str = "ppo",
) -> PolicyView | None:
    """Read the policy's internals for one observation, or return None if they are unavailable.

    Returns None rather than raising: this runs once per frame inside the playback loop, and an
    SB3 version whose private attributes moved should cost the panel, not the demo. `--human`
    passes no model and lands here too.
    """
    if model is None:
        return None
    try:
        import torch

        names = action_names(control_style)
        obs_tensor, _ = model.policy.obs_to_tensor(np.asarray(observation))
        with torch.no_grad():
            if algo == "dqn":
                q_values = model.q_net(obs_tensor).cpu().numpy().reshape(-1)
                scores = q_values.astype(np.float64)
                value = float(scores.max())
                label = "Q-VALUE"
            else:
                distribution = model.policy.get_distribution(obs_tensor)
                scores = (
                    distribution.distribution.probs.cpu().numpy().reshape(-1).astype(np.float64)
                )
                value = float(model.policy.predict_values(obs_tensor).cpu().numpy().reshape(-1)[0])
                label = "ACTION PROBABILITY"
    except Exception:  # noqa: BLE001 - a missing panel must never take the demo down
        return None

    if scores.size != len(names):
        return None
    return PolicyView(
        action_names=names,
        scores=tuple(float(value_) for value_ in scores),
        bars=_bars_from_scores(scores),
        chosen=int(chosen),
        value=value,
        score_label=label,
        algo=str(algo).upper(),
    )
