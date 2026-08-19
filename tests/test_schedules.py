from __future__ import annotations

import pytest

from common.schedules import LinearEpsilon


def test_endpoints():
    schedule = LinearEpsilon(start=1.0, end=0.05, decay_episodes=1000)
    assert schedule(0) == pytest.approx(1.0)
    assert schedule(1000) == pytest.approx(0.05)


def test_is_linear_at_the_midpoint():
    schedule = LinearEpsilon(start=1.0, end=0.0, decay_episodes=100)
    assert schedule(50) == pytest.approx(0.5)
    assert schedule(25) == pytest.approx(0.75)


def test_held_at_end_after_decay_window():
    schedule = LinearEpsilon(start=1.0, end=0.1, decay_episodes=10)
    assert schedule(50) == pytest.approx(0.1)
    assert schedule(10_000) == pytest.approx(0.1)


def test_monotonically_decreasing():
    schedule = LinearEpsilon(start=0.9, end=0.02, decay_episodes=500)
    values = [schedule(e) for e in range(600)]
    assert all(b <= a for a, b in zip(values, values[1:]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": 0.1, "end": 0.9, "decay_episodes": 10},   # end above start
        {"start": 1.2, "end": 0.1, "decay_episodes": 10},   # start above 1
        {"start": 1.0, "end": -0.1, "decay_episodes": 10},  # end below 0
        {"start": 1.0, "end": 0.1, "decay_episodes": 0},    # empty decay window
    ],
)
def test_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        LinearEpsilon(**kwargs)
