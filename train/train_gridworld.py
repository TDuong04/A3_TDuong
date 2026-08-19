"""Part I training entry point.

    python -m train.train_gridworld --level 0 --algo q
    python -m train.train_gridworld --level 1 --algo sarsa        # A3-004
    python -m train.train_gridworld --level 6 --intrinsic-sweep   # A3-007

Reads `config/gridworld.yaml`, merges any `level_overrides` for the chosen level, seeds via
`common.seeding.seed_everything`, runs the algorithm headless, then writes to `results/`:

  - the learned Q-table (so `eval/play_gridworld.py` can replay the policy without retraining),
  - a per-episode history CSV: return, steps, died, collected, epsilon,
  - a smoothed training curve PNG,
  - a policy-arrow figure, which is how rubric B5 ("learned a shortest-path policy") is shown,
  - a JSON summary carrying the greedy rollout length against the independently computed optimum.

Every filename carries the level, the algorithm and the seed. A curve without a seed cannot be
reproduced and therefore cannot be cited in the report, so the seed is part of the name rather than
something to remember.

Never render during training — matplotlib is forced onto the Agg backend below and pygame is not
imported at all here. Rendering belongs to `eval/play_gridworld.py`.

The comparisons the rubric asks for are produced here:
  - C3: run Q-learning and SARSA on level 1 under identical seeds and schedules, then plot both
    greedy policies over the cliff. The figure is the evidence, and it is the same figure the video
    should show.
  - F: run level 6 twice, once at `intrinsic_reward_strength: 0.0` and once at `0.5`, and plot both
    curves on one axis.

Report `--seeds 0 1 2` and average, or state the single seed used. An unreported seed makes a
comparison unfalsifiable.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

# Must precede the pyplot import: training runs headless (and in CI), where the default interactive
# backend would either fail to find a display or leak a window into the run.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.config import TabularConfig, load_yaml  # noqa: E402
from common.seeding import make_rng, seed_everything  # noqa: E402
from gridworld.algorithms import (  # noqa: E402
    Rollout,
    TrainingResult,
    greedy_rollout,
    optimal_collection_steps,
    policy_rollout,
    q_learning,
    sarsa,
    save_q_table,
)
from gridworld.constants import ACTION_DELTAS, COLLECTIBLE_TILES, Action, Tile  # noqa: E402
from gridworld.env import GridWorld  # noqa: E402
from gridworld.levels import load_level  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

#: Levels whose transitions are stochastic (monsters). A single greedy rollout there is a sample,
#: not a proof, so the optimum comparison is reported but not asserted.
STOCHASTIC_LEVELS = frozenset({4, 5})

ALGORITHMS = {"q": q_learning, "sarsa": sarsa}

#: How many rollouts back a reported death rate. Not a hyperparameter — nothing here reaches the
#: learner — but a sample size, and 500 keeps the standard error of a rate near 10% under 1.5
#: points, which is finer than the difference the C3 comparison is claiming.
DEATH_RATE_ROLLOUTS = 500

#: Seed offsets for the two death-rate measurements, so the greedy and behaviour samples are drawn
#: from different streams and neither reuses the training or rollout seeds.
GREEDY_SAMPLE_OFFSET = 100_000
BEHAVIOUR_SAMPLE_OFFSET = 200_000

#: Keeps a measurement's env stream clear of its policy stream. Level 1 is deterministic and never
#: draws from the env's generator at all, but levels 4-5 move monsters from it: seeding the two
#: with the same number there would correlate monster movement with exploration, which is exactly
#: the coupling `GridWorld` takes an injected rng to avoid.
ENV_STREAM_OFFSET = 500_000


# --- config -------------------------------------------------------------------------------------


def config_for_level(level: int, name: str = "gridworld") -> TabularConfig:
    """`training` from the config file, with `level_overrides[level]` merged on top.

    The overrides exist because the levels genuinely differ — the cliff needs a longer decay window
    for SARSA's risk aversion to show, the monster levels need more episodes to average out — and
    merging here means the env and the trainer read the same numbers from the same file rather than
    disagreeing about how long an episode may run.
    """
    data = load_yaml(name)
    training = dict(data["training"])
    overrides = (data.get("level_overrides") or {}).get(level) or {}
    training.update(overrides)
    return TabularConfig(**training)


def build_config(args: argparse.Namespace) -> TabularConfig:
    """Apply the CLI's narrow overrides on top of the file, keeping config the single source.

    `--episodes` exists for smoke runs and tests; it goes through `dataclasses.replace` so the value
    still lives in a `TabularConfig` and no algorithm ever sees a bare literal.
    """
    config = config_for_level(args.level)
    changes: dict[str, Any] = {}
    if args.seed is not None:
        changes["seed"] = int(args.seed)
    if args.episodes is not None:
        changes["episodes"] = int(args.episodes)
    if args.epsilon_decay_episodes is not None:
        changes["epsilon_decay_episodes"] = int(args.epsilon_decay_episodes)
    if getattr(args, "epsilon_end", None) is not None:
        changes["epsilon_end"] = float(args.epsilon_end)
    return replace(config, **changes) if changes else config


def _reportable(path: Path) -> str:
    """Repo-relative where possible: an absolute path baked into a JSON summary names this machine,
    which is noise in a submitted zip and useless to whoever opens it next."""
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def stem_for(level: int, algo: str, seed: int) -> str:
    """Level, algorithm and seed in every name: an artifact that cannot name its seed is not
    evidence, and nobody remembers which run produced which PNG a week later."""
    return f"level{level}_{algo}_seed{seed}"


# --- plotting -----------------------------------------------------------------------------------


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centred-free trailing mean. Returns the raw series when the window does not fit."""
    window = max(1, min(int(window), len(values)))
    if window <= 1:
        return np.asarray(values, dtype=np.float64)
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="valid")


def plot_training_curve(result: TrainingResult, path: Path, *, optimum: int | None = None) -> Path:
    """Smoothed return and episode length on one figure, with epsilon behind them.

    Return alone is ambiguous on this env: with `REWARD_STEP = 0` the undiscounted return is just
    the number of items collected, so it saturates at 3 long before the *path* stops being wasteful.
    Steps-per-episode is the series that actually shows the shortest path being found, and the
    optimum line makes "converged" a reading rather than an impression.
    """
    returns = result.returns
    steps = result.steps
    window = max(1, len(returns) // 100)
    smooth_returns = moving_average(returns, window)
    smooth_steps = moving_average(steps, window)
    x_smooth = np.arange(len(smooth_returns)) + window - 1

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    config = result.config

    axes[0].plot(returns, color="tab:blue", alpha=0.15, linewidth=0.6, label="per episode")
    axes[0].plot(x_smooth, smooth_returns, color="tab:blue", linewidth=1.8,
                 label=f"mean over {window} episodes")
    axes[0].set_ylabel("episode return")
    axes[0].legend(loc="lower center", fontsize=8)
    axes[0].grid(alpha=0.25)

    epsilon_axis = axes[0].twinx()
    epsilon_axis.plot(result.epsilons, color="tab:grey", linestyle="--", linewidth=1.0,
                      label="epsilon")
    epsilon_axis.set_ylabel("epsilon", color="tab:grey")
    epsilon_axis.set_ylim(0.0, 1.05)
    epsilon_axis.tick_params(axis="y", colors="tab:grey")

    axes[1].plot(steps, color="tab:orange", alpha=0.15, linewidth=0.6, label="per episode")
    axes[1].plot(x_smooth, smooth_steps, color="tab:orange", linewidth=1.8,
                 label=f"mean over {window} episodes")
    if optimum is not None:
        axes[1].axhline(optimum, color="tab:green", linestyle=":", linewidth=1.5,
                        label=f"BFS optimum = {optimum}")
    axes[1].set_ylabel("steps per episode")
    axes[1].set_xlabel("episode")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.25)

    figure.suptitle(
        f"{result.algo.upper()} on level {result.level} — seed {config.seed}, "
        f"alpha={config.alpha}, gamma={config.gamma}, "
        f"eps {config.epsilon_start}->{config.epsilon_end} over {config.epsilon_decay_episodes}"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_policy_arrows(level: int, result: TrainingResult, rollout: Rollout, path: Path) -> Path:
    """One panel per collection phase, each showing the greedy action in every known state.

    The policy is a function of `(row, col, has_key, collected_mask)`, so a single arrow field would
    be a lie: after the first apple the agent follows a different slice of the table. Each panel
    fixes one `(has_key, mask)` and draws the arrows for that slice, with the greedy rollout's own
    route for that phase overlaid, which is what makes "shortest path" visible not asserted.
    """
    grid = load_level(level)
    n_rows, n_cols = len(grid), len(grid[0])
    # Same bit order the env fixes at construction: sorted collectible coordinates. Recomputing it
    # from the layout avoids building a second env just to read one tuple.
    bit_cells = sorted(
        (row, col)
        for row, line in enumerate(grid)
        for col, tile in enumerate(line)
        if tile in COLLECTIBLE_TILES
    )

    # Group the rollout by the phase (has_key, mask) each step was taken in.
    phases: list[tuple[bool, int]] = []
    segments: dict[tuple[bool, int], list[tuple[int, int]]] = {}
    for index, state in enumerate(rollout.states):
        row, col, has_key, mask = state
        phase = (has_key, mask)
        if phase not in segments:
            segments[phase] = []
            phases.append(phase)
        segments[phase].append((row, col))
        # The step that changes phase belongs to both: draw it as the arrival point of the old one.
        if index + 1 < len(rollout.states):
            following = rollout.states[index + 1]
            if (following[2], following[3]) != phase:
                segments[phase].append((following[0], following[1]))

    # The final phase is the terminal state the episode ended in: nothing left to collect, no action
    # ever taken from it, so a panel for it would be blank. Drop phases the rollout never moved in.
    phases = [phase for phase in phases if len(segments[phase]) > 1] or phases[:1]

    figure, axes = plt.subplots(1, len(phases), figsize=(4.2 * len(phases), 4.8), squeeze=False)
    for axis, phase in zip(axes[0], phases, strict=True):
        has_key, mask = phase
        _draw_level_background(axis, grid, bit_cells, mask, n_rows, n_cols)

        xs, ys, us, vs = [], [], [], []
        for row in range(n_rows):
            for col in range(n_cols):
                if grid[row][col] == Tile.ROCK:
                    continue
                state = (row, col, has_key, mask)
                if state not in result.q_table:
                    continue  # never visited in this phase: an arrow here would be invented
                q_row = result.q_table[state]
                if float(q_row.max()) == float(q_row.min()):
                    continue  # all actions tied — the state carries no learned preference
                # Draw every tied-best action, not the first index. The policy the agent follows
                # breaks ties at random, so a single arrow would misreport a state where two moves
                # are genuinely equally short — which on an open field is most of them.
                for best in np.flatnonzero(q_row == q_row.max()):
                    d_row, d_col = ACTION_DELTAS[Action(int(best))]
                    xs.append(col)
                    ys.append(row)
                    us.append(d_col * 0.34)
                    vs.append(d_row * 0.34)
        if xs:
            axis.quiver(xs, ys, us, vs, angles="xy", scale_units="xy", scale=1.0,
                        color="tab:blue", width=0.007)

        route = segments[phase]
        axis.plot([c for _, c in route], [r for r, _ in route], color="tab:red", linewidth=2.0,
                  alpha=0.7, zorder=3)
        collected = int(mask.bit_count())
        axis.set_title(f"after {collected} collected" + (" · holding key" if has_key else ""),
                       fontsize=10)

    figure.suptitle(
        f"{result.algo.upper()} greedy policy, level {level}, seed {result.config.seed} — "
        f"rollout {rollout.steps} steps"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _draw_level_background(axis, grid, bit_cells, mask: int, n_rows: int, n_cols: int) -> None:
    """Static tiles plus the items still uncollected in this phase."""
    markers = {Tile.APPLE: ("o", "tab:green"), Tile.KEY: ("*", "goldenrod"),
               Tile.CHEST: ("s", "saddlebrown")}
    for row in range(n_rows):
        for col in range(n_cols):
            tile = grid[row][col]
            if tile == Tile.ROCK:
                axis.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color="0.35"))
            elif tile == Tile.FIRE:
                axis.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color="tab:red",
                                             alpha=0.5))
            elif tile == Tile.START:
                axis.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color="tab:blue",
                                             alpha=0.15))
    for bit, cell in enumerate(bit_cells):
        if mask >> bit & 1:
            continue  # already collected in this phase
        tile = grid[cell[0]][cell[1]]
        marker, colour = markers.get(tile, ("o", "tab:green"))
        axis.plot(cell[1], cell[0], marker=marker, color=colour, markersize=11, zorder=2)

    axis.set_xlim(-0.5, n_cols - 0.5)
    axis.set_ylim(-0.5, n_rows - 0.5)
    axis.invert_yaxis()
    axis.set_xticks(range(n_cols))
    axis.set_yticks(range(n_rows))
    axis.set_aspect("equal")
    axis.grid(alpha=0.2)
    axis.tick_params(labelsize=7)


# --- run ----------------------------------------------------------------------------------------


def run(
    level: int,
    algo: str,
    config: TabularConfig,
    *,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """Train, verify the greedy policy against the BFS optimum, and write every artifact.

    Artifacts are written the moment they are produced. Regenerating a curve later means retraining,
    and by then the config may have moved — at which point the figure in the report and the numbers
    in the repository describe different experiments.
    """
    if algo not in ALGORITHMS:
        raise ValueError(f"unknown algo {algo!r}; expected one of {sorted(ALGORITHMS)}")

    seed_everything(config.seed)
    # Separate streams for the env and the learner: a change to monster movement must not reshuffle
    # the exploration sequence, or two runs that differ only in level become incomparable.
    env_seed, learner_seed, rollout_seed = np.random.SeedSequence(config.seed).spawn(3)
    env = GridWorld(
        level_index=level,
        rng=make_rng(env_seed),
        max_steps=config.max_steps_per_episode,
    )

    result = ALGORITHMS[algo](env, config, rng=make_rng(learner_seed))
    rollout = greedy_rollout(env, result.q_table, rng=make_rng(rollout_seed))
    optimum = optimal_collection_steps(level)
    n_collectibles = len(env.collectible_cells)

    summary: dict[str, Any] = {
        "level": level,
        "algo": algo,
        "seed": config.seed,
        "episodes": config.episodes,
        "alpha": config.alpha,
        "gamma": config.gamma,
        "epsilon_start": config.epsilon_start,
        "epsilon_end": config.epsilon_end,
        "epsilon_decay_episodes": config.epsilon_decay_episodes,
        "max_steps_per_episode": config.max_steps_per_episode,
        "greedy_steps": rollout.steps,
        "bfs_optimum_steps": optimum,
        "greedy_collected": rollout.collected,
        "n_collectibles": n_collectibles,
        "greedy_return": rollout.total_return,
        "greedy_died": rollout.died,
        "greedy_truncated": rollout.truncated,
        "optimal": rollout.collected == n_collectibles and rollout.steps == optimum,
        "stochastic_level": level in STOCHASTIC_LEVELS,
        "mean_return_last_100": float(np.mean(result.returns[-100:])),
        "mean_steps_last_100": float(np.mean(result.steps[-100:])),
        "artifacts": {},
    }

    if write_artifacts:
        results_dir = Path(results_dir)
        stem = stem_for(level, algo, config.seed)
        history_path = result.write_history_csv(results_dir / f"history_{stem}.csv")
        qtable_path = save_q_table(result.q_table, results_dir / f"qtable_{stem}.npz")
        curve_path = plot_training_curve(result, results_dir / f"curve_{stem}.png", optimum=optimum)
        policy_path = plot_policy_arrows(level, result, rollout, results_dir / f"policy_{stem}.png")
        summary["artifacts"] = {
            "history_csv": _reportable(history_path),
            "q_table": _reportable(qtable_path),
            "training_curve": _reportable(curve_path),
            "policy_arrows": _reportable(policy_path),
        }
        summary_path = results_dir / f"summary_{stem}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["artifacts"]["summary_json"] = _reportable(summary_path)
        summary["paths"] = {
            "history_csv": history_path,
            "q_table": qtable_path,
            "training_curve": curve_path,
            "policy_arrows": policy_path,
            "summary_json": summary_path,
        }

    summary["result"] = result
    summary["rollout"] = rollout
    return summary


# --- C3: the level 1 comparison -----------------------------------------------------------------


def death_rate(
    level: int,
    q_table: dict,
    *,
    epsilon: float,
    max_steps: int,
    rollouts: int = DEATH_RATE_ROLLOUTS,
    seed_offset: int = 0,
) -> dict[str, float]:
    """Run the learned policy `rollouts` times and report how often it ended in the fire.

    This is the quantitative form of "more conservative around hazards". The picture shows one
    route; a rate over many seeded episodes shows how often that route survives, which is the claim
    the report is actually making.

    `epsilon` is what makes the measurement mean something. At epsilon 0 both algorithms walk a
    safe route and both score zero — the cliff only bites an agent that is still exploring, and
    SARSA's whole argument is about the policy it follows *including* its exploration. Measuring at
    the schedule's `epsilon_end` asks the question the two update rules actually disagree about.

    Every rollout gets its own seed, and the offsets keep those seeds clear of the training and
    greedy-rollout streams so a death rate is never an echo of the run that produced the table.
    """
    deaths = 0
    steps: list[int] = []
    returns: list[float] = []
    for index in range(rollouts):
        env = GridWorld(
            level_index=level,
            rng=make_rng(ENV_STREAM_OFFSET + seed_offset + index),
            max_steps=max_steps,
        )
        rollout = policy_rollout(
            env, q_table, epsilon=epsilon, rng=make_rng(seed_offset + index)
        )
        deaths += int(rollout.died)
        steps.append(rollout.steps)
        returns.append(rollout.total_return)
    return {
        "epsilon": float(epsilon),
        "rollouts": int(rollouts),
        "deaths": int(deaths),
        "death_rate": deaths / rollouts,
        "mean_steps": float(np.mean(steps)),
        "mean_return": float(np.mean(returns)),
    }


def _route_rows(rollout: Rollout) -> list[int]:
    """The grid rows the greedy route touches — the shortest honest summary of "which way round"."""
    return sorted({row for row, _ in rollout.path})


def _fire_distances(grid: list[list[str]]) -> dict[tuple[int, int], int]:
    """Manhattan distance from every cell to the nearest lethal tile."""
    fires = [
        (row, col)
        for row, line in enumerate(grid)
        for col, tile in enumerate(line)
        if tile == Tile.FIRE
    ]
    return {
        (row, col): min(abs(row - f_row) + abs(col - f_col) for f_row, f_col in fires)
        for row in range(len(grid))
        for col in range(len(grid[0]))
    } if fires else {}


def retreat_from_hazard(level: int, result: TrainingResult, phase: tuple[bool, int]) -> dict:
    """Of the cells next to the fire, how many does the greedy policy step *away* from it?

    The figure shows this and the eye reads it immediately — Q-learning's arrows run parallel to the
    fire, SARSA's point up out of the danger row — but the report needs it as a number. Counted over
    the cells directly adjacent to a lethal tile, a cell is "retreating" when every tied-best action
    increases its distance from the nearest fire. Parallel counts as not retreating: it keeps the
    agent in the row where a single exploratory step is fatal, which is exactly the risk at issue.
    """
    grid = load_level(level)
    distances = _fire_distances(grid)
    has_key, mask = phase
    adjacent = [cell for cell, distance in distances.items() if distance == 1]
    retreating = 0
    considered = 0
    for row, col in adjacent:
        if grid[row][col] == Tile.ROCK:
            continue
        q_row = result.q_table.get((row, col, has_key, mask))
        if q_row is None or float(q_row.max()) == float(q_row.min()):
            continue  # no learned preference here, so nothing to report either way
        considered += 1
        moves = [
            (row + ACTION_DELTAS[Action(int(best))][0], col + ACTION_DELTAS[Action(int(best))][1])
            for best in np.flatnonzero(q_row == q_row.max())
        ]
        if all(distances.get(move, 0) > 1 for move in moves):
            retreating += 1
    return {
        "fire_adjacent_cells": len(adjacent),
        "cells_with_a_preference": considered,
        "retreating_cells": retreating,
    }


def plot_policy_comparison(
    level: int,
    summaries: dict[str, dict[str, Any]],
    path: Path,
) -> Path:
    """The C3 figure: both greedy policies over the cliff, side by side, on identical axes.

    One panel per algorithm, same grid, same arrow convention, so the only thing that can differ
    between the two halves is the policy. Fire is drawn as a solid hatched band and labelled,
    because a reader who cannot see the hazard cannot see the point of the comparison.

    Arrows are drawn for the collection phase the route runs in, and every tied-best action is
    drawn rather than the first index — the agent breaks ties at random, so a single arrow would
    misreport a state where two moves are genuinely equal.
    """
    grid = load_level(level)
    n_rows, n_cols = len(grid), len(grid[0])
    colours = {"q": "tab:blue", "sarsa": "tab:purple"}
    titles = {"q": "Q-learning (off-policy)", "sarsa": "SARSA (on-policy)"}

    figure, axes = plt.subplots(1, len(summaries), figsize=(6.0 * len(summaries), 6.4),
                                squeeze=False)
    for axis, (algo, summary) in zip(axes[0], summaries.items(), strict=True):
        result: TrainingResult = summary["result"]
        rollout: Rollout = summary["rollout"]
        colour = colours.get(algo, "tab:blue")

        # The phase the route runs in: level 1 has a single apple, so this is the whole episode.
        _, _, has_key, mask = rollout.states[0]
        _draw_level_background(axis, grid, [], mask, n_rows, n_cols)
        for row in range(n_rows):
            for col in range(n_cols):
                if grid[row][col] == Tile.FIRE:
                    axis.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor="tab:red",
                                                 edgecolor="darkred", hatch="xx", alpha=0.75,
                                                 zorder=1))
        for cell in (
            (row, col)
            for row, line in enumerate(grid)
            for col, tile in enumerate(line)
            if tile in COLLECTIBLE_TILES
        ):
            axis.plot(cell[1], cell[0], marker="o", color="tab:green", markersize=13, zorder=4)

        xs, ys, us, vs = [], [], [], []
        for row in range(n_rows):
            for col in range(n_cols):
                if grid[row][col] == Tile.ROCK:
                    continue
                state = (row, col, has_key, mask)
                if state not in result.q_table:
                    continue  # never visited: an arrow here would be invented
                q_row = result.q_table[state]
                if float(q_row.max()) == float(q_row.min()):
                    continue  # no learned preference
                for best in np.flatnonzero(q_row == q_row.max()):
                    d_row, d_col = ACTION_DELTAS[Action(int(best))]
                    xs.append(col)
                    ys.append(row)
                    us.append(d_col * 0.34)
                    vs.append(d_row * 0.34)
        if xs:
            axis.quiver(xs, ys, us, vs, angles="xy", scale_units="xy", scale=1.0, color=colour,
                        width=0.006, zorder=3)

        route = rollout.path
        axis.plot([c for _, c in route], [r for r, _ in route], color=colour, linewidth=3.0,
                  alpha=0.85, zorder=5, label="greedy route")
        axis.plot(route[0][1], route[0][0], marker="s", color=colour, markersize=10, zorder=6)

        greedy = summary["death_rates"]["greedy"]
        behaviour = summary["death_rates"]["behaviour"]
        axis.set_title(
            f"{titles.get(algo, algo)}\n"
            f"greedy route {rollout.steps} steps via rows {_route_rows(rollout)}\n"
            f"death rate: {greedy['death_rate']:.1%} greedy, "
            f"{behaviour['death_rate']:.1%} at eps={behaviour['epsilon']}",
            fontsize=10,
        )
        axis.legend(loc="upper right", fontsize=8)

    any_summary = next(iter(summaries.values()))
    config: TabularConfig = any_summary["result"].config
    figure.suptitle(
        f"Level {level} cliff walk — greedy policy after identical training (seed {config.seed})",
        fontsize=13,
    )
    # Two short lines rather than one long one: at this figure width a single caption runs off
    # both edges, and the caption is what makes the panels readable at report scale.
    figure.text(
        0.5,
        0.012,
        f"both algorithms: {config.episodes} episodes, alpha={config.alpha}, "
        f"gamma={config.gamma}, epsilon {config.epsilon_start} -> {config.epsilon_end} "
        f"over {config.epsilon_decay_episodes} episodes, identical seeds and schedules\n"
        f"hatched red = fire (instant death) · arrows = greedy action(s) in every visited cell · "
        f"square = start · circle = apple",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.075, 1.0, 0.955))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def write_comparison_markdown(
    level: int,
    summaries: dict[str, dict[str, Any]],
    figure_path: Path,
    path: Path,
) -> Path:
    """The written half of C3 — a few paragraphs the report can lift verbatim.

    Numbers are formatted from the run that just happened rather than typed, so the prose cannot
    drift away from the figure sitting next to it.
    """
    config: TabularConfig = next(iter(summaries.values()))["result"].config
    q_summary, sarsa_summary = summaries["q"], summaries["sarsa"]
    q_roll, sarsa_roll = q_summary["rollout"], sarsa_summary["rollout"]
    q_behaviour = q_summary["death_rates"]["behaviour"]
    sarsa_behaviour = sarsa_summary["death_rates"]["behaviour"]
    q_greedy = q_summary["death_rates"]["greedy"]
    sarsa_greedy = sarsa_summary["death_rates"]["greedy"]
    q_hazard, sarsa_hazard = q_summary["hazard"], sarsa_summary["hazard"]
    optimum = q_summary["bfs_optimum_steps"]

    # Table cells are formatted up front so no row of the markdown table has to be split across
    # source lines — a wrapped row stops being a table.
    rollouts = q_behaviour["rollouts"]
    eps_end = q_behaviour["epsilon"]
    q_greedy_rate = f"{q_greedy['death_rate']:.1%}"
    sarsa_greedy_rate = f"{sarsa_greedy['death_rate']:.1%}"
    q_beh_rate = f"{q_behaviour['death_rate']:.1%}"
    sarsa_beh_rate = f"{sarsa_behaviour['death_rate']:.1%}"
    q_beh_return = f"{q_behaviour['mean_return']:.3f}"
    sarsa_beh_return = f"{sarsa_behaviour['mean_return']:.3f}"
    q_retreat = f"{q_hazard['retreating_cells']}/{q_hazard['cells_with_a_preference']}"
    sarsa_retreat = f"{sarsa_hazard['retreating_cells']}/{sarsa_hazard['cells_with_a_preference']}"

    text = f"""# Q-learning vs SARSA on level {level} (the cliff walk)

*Generated by `python -m train.train_gridworld --level {level} --compare --seed {config.seed}`.
Figure: `{_reportable(figure_path)}`.*

Both algorithms were trained on level {level} under **identical settings**: seed {config.seed},
{config.episodes} episodes, alpha {config.alpha}, gamma {config.gamma}, and the same linear epsilon
schedule ({config.epsilon_start} -> {config.epsilon_end} over {config.epsilon_decay_episodes}
episodes) built by the same `epsilon_schedule()` function, from the same config block. The two runs
differ in exactly one line of code: the bootstrap term of the update.

Level {level} places a wall of fire along row 8 between the start at (8,0) and the apple at (8,9).
The shortest safe route is {optimum} steps and runs along row 7, the row directly above the fire;
any route further from the fire is longer.

## What the two agents learned

| | Q-learning (off-policy) | SARSA (on-policy) |
|---|---|---|
| greedy route length | {q_roll.steps} steps | {sarsa_roll.steps} steps |
| rows used | {_route_rows(q_roll)} | {_route_rows(sarsa_roll)} |
| BFS optimum | {optimum} steps | {optimum} steps |
| death rate, greedy (eps=0, {rollouts} rollouts) | {q_greedy_rate} | {sarsa_greedy_rate} |
| death rate at eps={eps_end} ({rollouts} rollouts) | {q_beh_rate} | {sarsa_beh_rate} |
| mean return at eps={eps_end} | {q_beh_return} | {sarsa_beh_return} |
| fire-adjacent cells retreating from the fire | {q_retreat} | {sarsa_retreat} |

## Why the routes differ

Q-learning's target is `r + gamma * max_a' Q[s',a']`. The max is taken over the *best* action
available at the successor state, so the value it learns is the value of a policy that never
explores. Standing next to the fire is therefore free: the agent evaluates itself as if it will
always choose to walk sideways, never down. It converges on the shortest route, which hugs row 7.

SARSA's target is `r + gamma * Q[s',a']`, where `a'` is the action the epsilon-greedy behaviour
policy actually goes on to take. Some fraction of the time that action is a random one, and next to
the fire a random action is sometimes fatal, ending the episode with the apple uncollected. That
cost is folded straight into the value of the cell it happened in, so the cells along row 7 are
worth less to SARSA than they are to Q-learning, and the detour one row further up becomes the
better trade. SARSA is not more cautious by design; it is evaluating the policy it is actually
following, exploration included, and that policy is genuinely more dangerous near the cliff.

The arrow field says the same thing cell by cell. Of the cells that sit directly against the fire
and carry a learned preference, Q-learning's greedy action steps away from the hazard in
{q_hazard['retreating_cells']} of {q_hazard['cells_with_a_preference']}; SARSA's does in
{sarsa_hazard['retreating_cells']} of {sarsa_hazard['cells_with_a_preference']}. Q-learning's arrows
run *parallel* to the fire — the fastest way along it — while SARSA's point up, out of the row where
one unlucky exploratory step ends the episode.

The death rates are the same statement without the picture. With exploration switched off both
tables are safe, because neither greedy route ever steps into the fire. Restore the exploration the
agent trained under and the routes separate: Q-learning dies in {q_beh_rate} of episodes against
SARSA's {sarsa_beh_rate}. SARSA pays {sarsa_roll.steps - q_roll.steps} extra steps for that.

This is also why the effect depends on `epsilon_end`. As epsilon approaches zero the behaviour
policy converges on the greedy policy, `Q[s',a']` converges on `max_a' Q[s',a']`, and the two
algorithms converge on the same answer — the risk SARSA is pricing in stops existing. The schedule
here settles at {config.epsilon_end}, which keeps the risk real at convergence and keeps the
comparison meaningful.

## Reproducing it

`python -m train.train_gridworld --level {level} --compare --seed {config.seed}`, with every
hyperparameter read from `config/gridworld.yaml`.

Two conditions have to hold before this comparison shows anything, and both are properties of the
config rather than of the code. Epsilon at convergence must be high enough for the risk to be real,
for the reason above. And the run must be long enough for **Q-learning** to have converged along
row 7: those cells are the last states in the level to be visited often, because reaching the
middle of that row means having already survived several steps beside the fire. Until they
converge, Q-learning's own greedy policy avoids the row it should be hugging, both algorithms
detour, and the figure shows two identical policies — which is easy to misread as SARSA being
broken. The episode budget in `level_overrides` is set past that point.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_rewrap_prose(text), encoding="utf-8")
    return path


def _rewrap_prose(text: str, width: int = 96) -> str:
    """Reflow the prose paragraphs, leaving headings and the table exactly as written.

    The source is an f-string, so its line breaks fall wherever the substituted numbers happened to
    end. Markdown renders that identically either way, but this file is meant to be read and lifted
    into a report by a person, and a paragraph that breaks after "dies in" reads as a mistake.
    """
    blocks = []
    for block in text.split("\n\n"):
        if block.startswith(("#", "|", "*", "-")) or block.lstrip().startswith("|"):
            blocks.append(block)
        else:
            blocks.append(textwrap.fill(block, width=width))
    return "\n\n".join(blocks).rstrip("\n") + "\n"


def compare(
    level: int,
    config: TabularConfig,
    *,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    rollouts: int = DEATH_RATE_ROLLOUTS,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """Train both algorithms on `level` under one config and produce the C3 evidence.

    Both runs go through `run()` with the *same* `TabularConfig` object, so they share the seed, the
    episode count and the epsilon schedule by construction — there is no second place where either
    could be set differently. Anything that separates the two policies afterwards came from the
    update rule, which is the entire claim.
    """
    summaries: dict[str, dict[str, Any]] = {}
    for algo in ("q", "sarsa"):
        summary = run(level, algo, config, results_dir=results_dir,
                      write_artifacts=write_artifacts)
        q_table = summary["result"].q_table
        summary["death_rates"] = {
            "greedy": death_rate(
                level, q_table, epsilon=0.0, max_steps=config.max_steps_per_episode,
                rollouts=rollouts, seed_offset=GREEDY_SAMPLE_OFFSET + config.seed,
            ),
            "behaviour": death_rate(
                level, q_table, epsilon=config.epsilon_end,
                max_steps=config.max_steps_per_episode, rollouts=rollouts,
                seed_offset=BEHAVIOUR_SAMPLE_OFFSET + config.seed,
            ),
        }
        # The phase the route runs in: on level 1 the single apple means one phase per episode.
        _, _, has_key, mask = summary["rollout"].states[0]
        summary["hazard"] = retreat_from_hazard(level, summary["result"], (has_key, mask))
        summaries[algo] = summary

    comparison: dict[str, Any] = {
        "level": level,
        "seed": config.seed,
        "episodes": config.episodes,
        "alpha": config.alpha,
        "gamma": config.gamma,
        "epsilon_start": config.epsilon_start,
        "epsilon_end": config.epsilon_end,
        "epsilon_decay_episodes": config.epsilon_decay_episodes,
        "bfs_optimum_steps": summaries["q"]["bfs_optimum_steps"],
        "identical_config": True,
        "algorithms": {
            algo: {
                "greedy_steps": summary["rollout"].steps,
                "greedy_died": summary["rollout"].died,
                "greedy_return": summary["rollout"].total_return,
                "route_rows": _route_rows(summary["rollout"]),
                "route": [list(cell) for cell in summary["rollout"].path],
                "death_rates": summary["death_rates"],
                "hazard": summary["hazard"],
            }
            for algo, summary in summaries.items()
        },
        "routes_differ": (
            [list(c) for c in summaries["q"]["rollout"].path]
            != [list(c) for c in summaries["sarsa"]["rollout"].path]
        ),
        "artifacts": {},
    }
    comparison["summaries"] = summaries

    if write_artifacts:
        results_dir = Path(results_dir)
        stem = f"level{level}_seed{config.seed}"
        figure_path = plot_policy_comparison(
            level, summaries, results_dir / f"compare_{stem}.png"
        )
        markdown_path = write_comparison_markdown(
            level, summaries, figure_path, results_dir / f"comparison_{stem}.md"
        )
        comparison["artifacts"] = {
            "comparison_figure": _reportable(figure_path),
            "comparison_markdown": _reportable(markdown_path),
        }
        json_path = results_dir / f"compare_{stem}.json"
        json_path.write_text(
            json.dumps({k: v for k, v in comparison.items() if k != "summaries"}, indent=2),
            encoding="utf-8",
        )
        comparison["artifacts"]["comparison_json"] = _reportable(json_path)
        comparison["paths"] = {
            "comparison_figure": figure_path,
            "comparison_markdown": markdown_path,
            "comparison_json": json_path,
        }

    return comparison


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", type=int, default=0, help="level index 0-6")
    parser.add_argument("--algo", choices=sorted(ALGORITHMS), default="q")
    parser.add_argument("--seed", type=int, default=None,
                        help="overrides config; recorded in every output filename")
    parser.add_argument("--episodes", type=int, default=None,
                        help="overrides config for smoke runs; the full number lives in the yaml")
    parser.add_argument("--epsilon-decay-episodes", type=int, default=None,
                        help="overrides config; only useful alongside a shortened --episodes")
    parser.add_argument("--epsilon-end", type=float, default=None,
                        help="overrides config; the floor of the linear schedule. As it approaches "
                             "0 SARSA and Q-learning converge to the same policy, so the level 1 "
                             "comparison is only meaningful while it is well above 0")
    parser.add_argument("--compare", action="store_true",
                        help="C3: train both algorithms under this one config and write the "
                             "side-by-side policy figure, the death rates and the comparison")
    parser.add_argument("--death-rate-rollouts", type=int, default=DEATH_RATE_ROLLOUTS,
                        help="sample size behind each reported death rate (--compare only)")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--no-artifacts", action="store_true",
                        help="train without writing to results/ (smoke runs only)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    if args.compare:
        main_compare(args, config)
        return
    summary = run(
        args.level,
        args.algo,
        config,
        results_dir=args.results_dir,
        write_artifacts=not args.no_artifacts,
    )

    # ASCII only: the Windows console this is marked on defaults to cp1252 and mangles the rest.
    print(f"level {summary['level']} | {summary['algo']} | seed {summary['seed']} | "
          f"{summary['episodes']} episodes")
    print(f"  mean return  (last 100): {summary['mean_return_last_100']:.3f}")
    print(f"  mean steps   (last 100): {summary['mean_steps_last_100']:.1f}")
    print(f"  greedy rollout: {summary['greedy_steps']} steps, "
          f"{summary['greedy_collected']}/{summary['n_collectibles']} collected"
          + (" (DIED)" if summary["greedy_died"] else ""))
    print(f"  BFS optimum:    {summary['bfs_optimum_steps']} steps")
    verdict = "OPTIMAL" if summary["optimal"] else "NOT OPTIMAL"
    if summary["stochastic_level"]:
        verdict += " (stochastic level: one rollout is a sample, not a proof)"
    print(f"  verdict: {verdict}")
    for name, path in summary["artifacts"].items():
        print(f"  wrote {name}: {path}")


def main_compare(args: argparse.Namespace, config: TabularConfig) -> None:
    """`--compare`: the C3 run. ASCII only — the console this is marked on is cp1252."""
    comparison = compare(
        args.level,
        config,
        results_dir=args.results_dir,
        rollouts=int(args.death_rate_rollouts),
        write_artifacts=not args.no_artifacts,
    )
    print(f"level {comparison['level']} comparison | seed {comparison['seed']} | "
          f"{comparison['episodes']} episodes | eps {comparison['epsilon_start']}->"
          f"{comparison['epsilon_end']} over {comparison['epsilon_decay_episodes']} "
          f"(identical for both algorithms)")
    print(f"  BFS optimum: {comparison['bfs_optimum_steps']} steps")
    for algo, entry in comparison["algorithms"].items():
        greedy = entry["death_rates"]["greedy"]
        behaviour = entry["death_rates"]["behaviour"]
        print(f"  {algo:5s}: greedy {entry['greedy_steps']:3d} steps via rows "
              f"{entry['route_rows']}"
              + (" (DIED)" if entry["greedy_died"] else ""))
        print(f"         death rate {greedy['death_rate']:.1%} greedy, "
              f"{behaviour['death_rate']:.1%} at eps={behaviour['epsilon']} "
              f"({behaviour['rollouts']} rollouts)")
        hazard = entry["hazard"]
        print(f"         retreats from the fire in {hazard['retreating_cells']} of "
              f"{hazard['cells_with_a_preference']} fire-adjacent cells")
    verdict = "ROUTES DIFFER" if comparison["routes_differ"] else "ROUTES IDENTICAL"
    print(f"  verdict: {verdict}")
    if not comparison["routes_differ"]:
        print("  (identical routes usually mean epsilon_end is too low for the risk to matter at "
              "convergence, or the run is too short for Q-learning to have found the edge route)")
    for name, path in comparison["artifacts"].items():
        print(f"  wrote {name}: {path}")


if __name__ == "__main__":
    main()
