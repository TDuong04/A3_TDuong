"""How long does each phase take to clear, and by whom?

    python -m eval.measure_pacing                       # every player, both styles
    python -m eval.measure_pacing --episodes 20         # tighter intervals
    python -m eval.measure_pacing --player random       # just the floor

A3-022 asks a question about *pacing* rather than about correctness: phase 1 has to be beatable
quickly enough that a phase progression fits inside one episode, because the video has to show one
and A3-012's acceptance depends on it. Phase 1 that takes four minutes of perfect play means no
agent ever clears it, no clip ever shows it, and the fix at that point is retraining -- the
expensive way to discover a config problem.

That question is only answerable by measurement, and a number nobody can reproduce is a number
nobody can check. Hence a script rather than a paragraph.

What "too easy" and "too hard" mean here
----------------------------------------
Both ends are failures, and they fail differently:

  too hard   no agent clears phase 1, the video has no progression to show, and the difficulty
             ladder in `config/arena.yaml` may as well not exist
  too easy   a random policy clears it, the phase system demonstrates nothing about learning, and
             the reward for clearing a phase is being handed out for free

So this reports the trained agents *and* a random policy. The random row is not padding: it is the
floor that makes the trained rows mean something.

Time is reported in seconds of simulated game time, not wall clock. One agent step is
`ACTION_REPEAT` physics frames of `FIXED_DT`, so the conversion is exact and identical whether the
sim ran at 40,000 steps a second headless or at 60 fps on screen.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402

from arena.constants import ACTION_REPEAT, FIXED_DT, MAX_EPISODE_STEPS  # noqa: E402
from arena.env import ArenaEnv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STYLES = ("direct", "rotation")
PLAYERS = ("trained", "random")

#: Seconds of simulated game time per agent step. Exact, and independent of how fast the machine
#: actually ran the simulation.
SECONDS_PER_STEP = ACTION_REPEAT * FIXED_DT

#: The window A3-022 asks phase 1 to fall inside, in seconds.
TARGET_LOW, TARGET_HIGH = 20.0, 30.0


def seconds(steps: float) -> float:
    return float(steps) * SECONDS_PER_STEP


def episode_budget_seconds() -> float:
    return seconds(MAX_EPISODE_STEPS)


def random_policy(style: str, seed: int = 0) -> Callable[[Any, ArenaEnv], int]:
    """The floor. If this clears phase 1, the phase system is demonstrating nothing."""
    rng = np.random.default_rng(seed)

    def act(_obs: Any, env: ArenaEnv) -> int:
        return int(rng.integers(int(env.action_space.n)))

    return act


def trained_policy(style: str, models_dir: Path | None = None) -> Callable[[Any, ArenaEnv], int]:
    """The trained agent, acting deterministically, as the stand-in for competent play.

    A stand-in and not a substitute: A3-022's criterion names a competent *human*, and a human is
    plausibly faster than either agent. This measures the floor a marker will actually see on
    video, which is the number the video depends on.
    """
    from eval.play_arena import load_agent

    agent = load_agent(style, "ppo", models_dir)

    def act(obs: Any, _env: ArenaEnv) -> int:
        return int(agent.predict(obs, deterministic=True)[0])

    return act


def measure(
    style: str,
    policy: Callable[[Any, ArenaEnv], int],
    *,
    episodes: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """Play `episodes` seeded episodes and record when each phase was first entered."""
    env = ArenaEnv(control_style=style, render_mode=None)
    phase_1_steps: list[int] = []
    best_phase, deaths = 0, 0

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        cleared_at: int | None = None
        terminated = False
        done = False
        while not done:
            obs, _, terminated, truncated, _ = env.step(policy(obs, env))
            if cleared_at is None and env.phase > 1:
                cleared_at = env.steps
            done = terminated or truncated
        best_phase = max(best_phase, env.phase)
        deaths += int(terminated)
        if cleared_at is not None:
            phase_1_steps.append(cleared_at)
    env.close()

    times = [seconds(s) for s in phase_1_steps]
    return {
        "style": style,
        "episodes": episodes,
        "cleared": len(times),
        "clear_rate": len(times) / episodes if episodes else 0.0,
        "phase_1_seconds_mean": statistics.fmean(times) if times else None,
        "phase_1_seconds_min": min(times) if times else None,
        "phase_1_seconds_max": max(times) if times else None,
        "best_phase": best_phase,
        "death_rate": deaths / episodes if episodes else 0.0,
    }


def verdict(row: dict[str, Any], player: str) -> str:
    """One word per row, so the table answers the ticket rather than merely informing it."""
    mean = row["phase_1_seconds_mean"]
    if player == "random":
        return "OK (floor holds)" if row["cleared"] == 0 else "TOO EASY (random clears it)"
    if mean is None:
        return "TOO HARD (never cleared)"
    if mean < TARGET_LOW:
        return "fast (under target, harmless)"
    if mean > TARGET_HIGH:
        return "slow (over target)"
    return "OK (in target)"


def format_table(rows: list[tuple[str, dict[str, Any]]]) -> str:
    budget = episode_budget_seconds()
    out = [
        f"episode budget   {MAX_EPISODE_STEPS} agent steps = {budget:.0f}s of game time",
        f"phase 1 target   {TARGET_LOW:.0f}-{TARGET_HIGH:.0f}s",
        "",
        f"{'player':<20}{'cleared':>9}{'phase 1 (s)':>14}{'range':>16}"
        f"{'best':>6}{'died':>7}  verdict",
    ]
    for player, row in rows:
        mean = row["phase_1_seconds_mean"]
        mean_s = f"{mean:.1f}" if mean is not None else "-"
        span = (
            f"{row['phase_1_seconds_min']:.1f}-{row['phase_1_seconds_max']:.1f}"
            if mean is not None else "-"
        )
        out.append(
            f"{player + ': ' + row['style']:<20}"
            f"{row['cleared']}/{row['episodes']:<7}"
            f"{mean_s:>14}{span:>16}"
            f"{row['best_phase']:>6}{row['death_rate']:>6.0%}  {verdict(row, player)}"
        )
    slowest = max(
        (r["phase_1_seconds_mean"] for _, r in rows if r["phase_1_seconds_mean"] is not None),
        default=None,
    )
    if slowest is not None:
        out += [
            "",
            f"three phases at the slowest measured pace: {3 * slowest:.0f}s against a "
            f"{budget:.0f}s budget"
            + ("  -- fits" if 3 * slowest <= budget else "  -- DOES NOT FIT"),
        ]
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure how long each phase takes to clear.")
    parser.add_argument("--style", choices=(*STYLES, "both"), default="both")
    parser.add_argument("--player", choices=(*PLAYERS, "both"), default="both")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    styles = list(STYLES) if args.style == "both" else [args.style]
    players = list(PLAYERS) if args.player == "both" else [args.player]

    rows: list[tuple[str, dict[str, Any]]] = []
    for player in players:
        for style in styles:
            factory = trained_policy if player == "trained" else random_policy
            try:
                policy = factory(style)
            except Exception as error:  # a missing model is a message, not a traceback
                print(f"{player}: {style} skipped -- {error}", file=sys.stderr)
                continue
            rows.append((player, measure(style, policy, episodes=args.episodes, seed=args.seed)))

    print(format_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
