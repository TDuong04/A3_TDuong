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
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--no-artifacts", action="store_true",
                        help="train without writing to results/ (smoke runs only)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
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


if __name__ == "__main__":
    main()
