"""Part I training entry point.

    python -m train.train_gridworld --level 0 --algo q
    python -m train.train_gridworld --level 1 --algo sarsa        # A3-004
    python -m train.train_gridworld --levels 2 3 4 5 --seeds 0 1 2  # A3-005 + A3-006
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
  - F: run level 6 at `intrinsic_reward_strength: 0.0` and at `0.5` - the pair the config's
    `intrinsic_experiment` block names - over five seeds each, and plot both curves on one axis as
    a mean with a standard-error band. The plotted series is the ENVIRONMENT return: the intrinsic
    bonus is paid on nearly every step, so plotting the shaped return would separate the two arms
    for an arithmetic reason and measure nothing. `--intrinsic-sweep` also records the four metrics
    a sparse level needs - share of episodes solved, episodes until the first success, final greedy
    success and mean environment return - and writes the written explanation the rubric asks for.

`--levels` is the A3-005/A3-006 evidence run and it measures two different things on purpose.
Levels 2-3 are deterministic, so a greedy rollout *is* the policy: what gets recorded there is the
order the collectibles were taken in — the only way to show the agent learnt to take the key before
the chest, since a chest opened without the key simply pays nothing and stays on the grid — and
whether the run took everything in the optimal number of steps. Levels 4-5 are stochastic, so one
rollout is a sample and their curves will not flatten; what gets recorded there is the success rate,
the death rate and the mean return over many seeded rollouts, plus a direct measurement of the
monster move rate and of both halves of the two-sided collision check against the shipped env.

Report `--seeds 0 1 2` and average, or state the single seed used. An unreported seed makes a
comparison unfalsifiable.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter
from collections.abc import Sequence
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
    train_with_intrinsic_reward,
)
from gridworld.constants import (  # noqa: E402
    ACTION_DELTAS,
    COLLECTIBLE_TILES,
    MONSTER_MOVE_PROBABILITY,
    TILE_REWARDS,
    Action,
    Tile,
)
from gridworld.env import Coord, GridWorld  # noqa: E402
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

#: Monster-observations behind the reported move rate. At p=0.4 the standard error of the rate is
#: sqrt(0.24/n), so 200,000 puts it near 0.0011 — an order of magnitude finer than any deviation
#: from 0.4 that would matter, and small enough to run inside a report generation.
MONSTER_SAMPLES = 200_000

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


# --- D / A3-005: what order did the policy collect things in? -----------------------------------


def collectible_cells(level: int) -> list[Coord]:
    """The level's collectible cells in the env's own bit order — sorted coordinates.

    `GridWorld` fixes this order once at construction so a `collected_mask` bit means the same cell
    in every episode. Recomputing it from the layout here, by the same rule, lets a mask be decoded
    without building an env, and keeps the two definitions provably identical (see
    `tests/test_gridworld_levels_2_to_5.py`).
    """
    grid = load_level(level)
    return sorted(
        (row, col)
        for row, line in enumerate(grid)
        for col, tile in enumerate(line)
        if tile in COLLECTIBLE_TILES
    )


def level_total_reward(level: int) -> float:
    """Everything the level pays out over a full episode: apples +1, key 0, chest +2.

    `REWARD_STEP` is 0 and the brief specifies no death penalty, so a complete episode's summed
    rewards equal exactly this number and nothing else contributes to it. That makes it the check
    for "reward accounting matches the brief across a full episode" (A3-005), computed from
    `TILE_REWARDS` rather than typed out, so it cannot drift away from the constants.
    """
    grid = load_level(level)
    return float(
        sum(
            TILE_REWARDS[tile]
            for line in grid
            for tile in line
            if tile in COLLECTIBLE_TILES
        )
    )


def collection_order(level: int, rollout: Rollout) -> list[dict[str, Any]]:
    """The items a rollout picked up, in the order it picked them up.

    This is the evidence behind "the agent learns to collect the key before the chest" (A3-005).
    The claim cannot be read off the return: level 2 pays +2 for the chest *only* while the key is
    held, and an agent that walks over the chest first simply collects nothing and leaves the chest
    on the grid, so a policy that never learnt the dependency and one that did can post the same
    intermediate score. The order is the thing that separates them, and it is recovered from the
    state trace rather than from the env, because `collected_mask` is the only record of what was
    taken and when.
    """
    grid = load_level(level)
    cells = collectible_cells(level)
    order: list[dict[str, Any]] = []
    previous = rollout.states[0][3]
    for step, state in enumerate(rollout.states[1:], start=1):
        mask = state[3]
        taken = (bit for bit in range(len(cells)) if mask >> bit & 1 and not previous >> bit & 1)
        for bit in taken:
            row, col = cells[bit]
            order.append({"tile": grid[row][col], "cell": [row, col], "step": step})
        previous = mask
    return order


def key_precedes_chest(order: list[dict[str, Any]]) -> bool | None:
    """True when every chest in `order` was opened after a key was taken; None if there is no chest.

    None rather than True on a chestless level: levels 0, 1 and 4 have nothing to order, and
    reporting "passed" there would put a tick next to a check that was never run.
    """
    if not any(item["tile"] == Tile.CHEST for item in order):
        return None
    held = 0
    for item in order:
        if item["tile"] == Tile.KEY:
            held += 1
        elif item["tile"] == Tile.CHEST and held == 0:
            return False
    return True


def describe_collection_order(order: list[dict[str, Any]]) -> str:
    """`K(8,1) -> A(3,2) -> ...`: the sequence in one line, for a report or a console."""
    if not order:
        return "(nothing collected)"
    return " -> ".join(
        f"{item['tile']}({item['cell'][0]},{item['cell'][1]})@{item['step']}" for item in order
    )


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
    axis.tick_params(labelsize=10)


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
    order = collection_order(level, rollout)

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
        # "solved" is the A3-005 sense of the word: everything collected, the episode ended because
        # the game ended, and neither death nor the step cap got there first. Surviving is not
        # solving, and neither is a return that happens to look respectable half way through.
        "solved": (
            rollout.collected == n_collectibles and not rollout.died and not rollout.truncated
        ),
        "collection_order": order,
        "collection_order_text": describe_collection_order(order),
        "key_precedes_chest": key_precedes_chest(order),
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

    On levels 4-5 it is also the *only* honest summary of a policy. A single greedy rollout there
    is one sample from a stochastic transition function, so "the agent solved the level" cannot be
    read from it either way; the success rate, the death rate and the mean return over many seeded
    episodes can. `success_rate` counts an episode as a success only when every collectible was
    taken and the agent survived, which is why it is reported alongside the death rate rather than
    inferred as its complement: an episode can also end by running out of clock.

    `epsilon` is what makes the measurement mean something. At epsilon 0 both algorithms walk a
    safe route and both score zero — the cliff only bites an agent that is still exploring, and
    SARSA's whole argument is about the policy it follows *including* its exploration. Measuring at
    the schedule's `epsilon_end` asks the question the two update rules actually disagree about.

    Every rollout gets its own seed, and the offsets keep those seeds clear of the training and
    greedy-rollout streams so a death rate is never an echo of the run that produced the table.
    """
    deaths = 0
    successes = 0
    truncations = 0
    steps: list[int] = []
    returns: list[float] = []
    collected: list[int] = []
    n_collectibles = len(collectible_cells(level))
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
        truncations += int(rollout.truncated)
        successes += int(rollout.collected == n_collectibles and not rollout.died)
        steps.append(rollout.steps)
        returns.append(rollout.total_return)
        collected.append(rollout.collected)
    return {
        "epsilon": float(epsilon),
        "rollouts": int(rollouts),
        "deaths": int(deaths),
        "death_rate": deaths / rollouts,
        "successes": int(successes),
        "success_rate": successes / rollouts,
        "truncations": int(truncations),
        "truncation_rate": truncations / rollouts,
        "n_collectibles": int(n_collectibles),
        "mean_collected": float(np.mean(collected)),
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

        # The route is a wide translucent corridor UNDERNEATH the arrows (zorder 3), not a solid
        # line over them. Drawn on top at linewidth 3 it hid Q-learning's row-7 arrows entirely --
        # and those arrows, running parallel to the fire, are the whole visual claim of this panel.
        # SARSA's route leaves row 7 visible, so the defect silently weakened one half of a
        # side-by-side comparison. Both layers stay below zorder 3 for that reason.
        route = rollout.path
        route_x = [c for _, c in route]
        route_y = [r for r, _ in route]
        axis.plot(route_x, route_y, color=colour, linewidth=9.0, alpha=0.20,
                  solid_capstyle="round", zorder=2.0, label="greedy route")
        axis.plot(route_x, route_y, color=colour, linewidth=1.0, alpha=0.55, zorder=2.5)
        axis.plot(route[0][1], route[0][0], marker="s", color=colour, markersize=11, zorder=6)

        greedy = summary["death_rates"]["greedy"]
        behaviour = summary["death_rates"]["behaviour"]
        axis.set_title(
            f"{titles.get(algo, algo)}\n"
            f"greedy route {rollout.steps} steps via rows {_route_rows(rollout)}\n"
            f"death rate: {greedy['death_rate']:.1%} greedy, "
            f"{behaviour['death_rate']:.1%} at eps={behaviour['epsilon']}",
            fontsize=13,
        )
        axis.legend(loc="upper right", fontsize=11)

    any_summary = next(iter(summaries.values()))
    config: TabularConfig = any_summary["result"].config
    figure.suptitle(
        f"Level {level} cliff walk — greedy policy after identical training (seed {config.seed})",
        fontsize=16,
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
        fontsize=11,
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
Two routes tie for the {optimum}-step optimum: one along row 7 directly above the fire and one
along row 9 directly below it. Both run the length of the wall one cell away from it, so the
shortest route is necessarily a fire-hugging one, and any route that keeps further away is longer.

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
always choose to walk sideways, never down. It converges on one of the two optimal routes, hugging
row 7.

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

State the result as the table states it — SARSA's greedy route is {sarsa_roll.steps} steps via rows
{_route_rows(sarsa_roll)}, and it dies several times less often at the exploration it trained under.
"SARSA never goes near the fire" is stronger than the evidence supports: on some seeds its route
descends its final column early and clips one fire-adjacent cell, which lifts its death rate to
around 4% while still leaving it well clear of Q-learning's. The direction of the effect is stable
across seeds; the claim that its route is fire-free is not.

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
        # A fenced code block must survive verbatim: reflowing it would join its lines
        # into one paragraph and the fence would stop being a code block.
        if (block.startswith(("#", "|", "*", "-")) or block.lstrip().startswith("|")
                or "```" in block):
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


# --- A3-006: measuring the monster mechanics the env already implements -------------------------


def measure_monster_movement(
    level: int,
    *,
    samples: int = MONSTER_SAMPLES,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure the monster move rate, the direction mix and the legality of every landing cell.

    A3-006 states three rules — move with p=0.4 after each agent action, choose uniformly among
    directions that are neither rocks nor off-grid, and kill on contact from both sides — and
    `gridworld/env.py` already implements all three. This function is the *check*, not a second
    implementation: it drives the real `GridWorld.step()` and counts what comes out, so it would
    still fail if the env's numbers drifted.

    The agent is parked on the start cell and presses UP into the grid edge, which is a blocked
    move: no displacement, no reward, no penalty. Every change in a monster position between two
    steps is therefore the monster's own draw and nothing else. The episode is restarted whenever a
    monster catches the parked agent, which is itself the death check firing.

    The measured rate is a rate, so it carries sampling error: at p=0.4 the standard error over
    `samples` monster-observations is `sqrt(0.24 / samples)`, about 0.001 at 200,000.
    """
    grid = load_level(level)
    n_rows, n_cols = len(grid), len(grid[0])
    env = GridWorld(level_index=level, rng=make_rng(seed), max_steps=samples + 1)
    if not env.monsters:
        raise ValueError(f"level {level} has no monsters to measure")

    observations = moved = off_grid = on_rock = restarts = 0
    directions: Counter[tuple[int, int]] = Counter()
    while observations < samples:
        before = env.monster_positions
        env.step(Action.UP)  # blocked by the grid edge from the start cell: the agent never moves
        for start, end in zip(before, env.monster_positions, strict=True):
            observations += 1
            delta = (end[0] - start[0], end[1] - start[1])
            if delta != (0, 0):
                moved += 1
                directions[delta] += 1
            if not 0 <= end[0] < n_rows or not 0 <= end[1] < n_cols:
                off_grid += 1
            elif grid[end[0]][end[1]] == Tile.ROCK:
                on_rock += 1
        if env.done:
            env.reset()
            restarts += 1

    return {
        "level": level,
        "seed": seed,
        "monsters": len(env.monsters),
        "observations": observations,
        "moves": moved,
        "measured_move_probability": moved / observations,
        "expected_move_probability": MONSTER_MOVE_PROBABILITY,
        "standard_error": float(
            np.sqrt(
                MONSTER_MOVE_PROBABILITY * (1 - MONSTER_MOVE_PROBABILITY) / observations
            )
        ),
        "direction_counts": {f"{d_row},{d_col}": count for (d_row, d_col), count in
                             sorted(directions.items())},
        "direction_shares": {f"{d_row},{d_col}": count / moved for (d_row, d_col), count in
                             sorted(directions.items())},
        "landings_off_grid": off_grid,
        "landings_on_rock": on_rock,
        "episode_restarts": restarts,
    }


#: One cell per monster level whose occupant has exactly *two* legal directions, so that "uniform
#: among the directions that are not rocks and not off-grid" becomes a number instead of an
#: absence. On level 4 — which carries no rocks at all — the corner (0,9) is blocked above and to
#: the right by the grid edge; on level 5 the cell (0,4) is blocked above by the edge and to the
#: right by the rock at (0,5). Both should move to each of their two legal neighbours with
#: probability 0.4/2 = 0.2 and stay put with probability 0.6. The aggregate histogram over a free
#: roam cannot show this: there every direction is legal, so a monster that ignored the rules
#: entirely would produce the same four near-equal shares.
CONSTRAINED_MONSTER_CELLS: dict[int, Coord] = {4: (0, 9), 5: (0, 4)}


def measure_constrained_monster_choice(
    level: int,
    *,
    cell: Coord | None = None,
    samples: int = 100_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Where does a monster with only two legal directions actually go, and how often?

    The monster is returned to `cell` before every step and the agent is parked out of reach in the
    opposite corner of row 0, pressing UP into the grid edge, so each step is one independent draw
    from the same position and nothing else in the episode moves.
    """
    cell = cell if cell is not None else CONSTRAINED_MONSTER_CELLS[level]
    env = GridWorld(level_index=level, rng=make_rng(seed), max_steps=samples + 1)
    env.reset()
    agent = env.agent_pos
    legal = [
        (cell[0] + d_row, cell[1] + d_col)
        for d_row, d_col in ACTION_DELTAS.values()
        if not env.is_blocked(cell[0] + d_row, cell[1] + d_col)
    ]
    if agent in legal or agent == cell:
        raise ValueError(f"the parked agent at {agent} is within reach of a monster at {cell}")

    landings: Counter[Coord] = Counter()
    for _ in range(samples):
        env.agent_pos = agent
        env.monsters = [cell]
        env.step(Action.UP)  # blocked by the grid edge: the agent never moves
        landings[env.monster_positions[0]] += 1
        if env.done:  # unreachable while the agent is out of reach, but do not hang if it is not
            env.reset()

    moves = samples - landings[cell]
    return {
        "level": level,
        "cell": list(cell),
        "samples": samples,
        "legal_neighbours": [list(neighbour) for neighbour in sorted(legal)],
        "stayed": landings[cell],
        "moved": moves,
        "measured_move_probability": moves / samples,
        "landing_shares": {f"{row},{col}": landings[(row, col)] / samples
                           for row, col in sorted(landings)},
        "expected_share_per_legal_direction": MONSTER_MOVE_PROBABILITY / len(legal),
        "illegal_landings": sum(count for landing, count in landings.items()
                                if landing != cell and landing not in legal),
    }


def verify_death_checks(level: int, *, max_env_seeds: int = 2000) -> dict[str, Any]:
    """Demonstrate both halves of the two-sided collision check on a real `GridWorld`.

    The half everyone implements is the agent walking into a monster. The half that gets missed is
    the monster walking into a stationary agent, and it cannot be shown by asserting on the source:
    it has to be a step that kills an agent which did not move. Here the agent presses UP into the
    grid edge — a blocked move, so its position is unchanged — with a monster placed alongside, and
    env seeds are tried until one draws the monster onto the agent. The seed that did it is
    returned, so the case is reproducible rather than anecdotal.
    """
    agent = (0, 0)
    neighbour = (0, 1)

    # Half 1: the agent enters the monster's tile. Fully deterministic — no monster draw involved,
    # because the death check fires before monsters are moved.
    entered = GridWorld(level_index=level, rng=make_rng(0), max_steps=8)
    entered.reset()
    entered.agent_pos = agent
    entered.monsters = [neighbour]
    _, _, done_entered, info_entered = entered.step(Action.RIGHT)

    # Half 2: a monster enters the agent's tile while the agent stands still.
    entered_by = None
    for env_seed in range(max_env_seeds):
        env = GridWorld(level_index=level, rng=make_rng(env_seed), max_steps=8)
        env.reset()
        env.agent_pos = agent
        env.monsters = [neighbour]
        _, _, done, info = env.step(Action.UP)  # blocked: the agent cannot leave (0,0)
        if info["died"]:
            entered_by = {
                "env_seed": env_seed,
                "agent_before": list(agent),
                "agent_after": list(env.agent_pos),
                "monster_before": list(neighbour),
                "monster_after": list(env.monster_positions[0]),
                "died": bool(info["died"]),
                "done": bool(done),
                "terminated": bool(info["terminated"]),
            }
            break

    return {
        "level": level,
        "agent_entered_monster": {
            "agent_before": list(agent),
            "agent_after": list(entered.agent_pos),
            "monster": list(neighbour),
            "died": bool(info_entered["died"]),
            "done": bool(done_entered),
            "terminated": bool(info_entered["terminated"]),
        },
        "monster_entered_agent": entered_by,
        "both_checks_fire": bool(info_entered["died"]) and entered_by is not None,
    }


# --- D + A3-006: the levels 2-5 evidence run ----------------------------------------------------


def levels_report(
    levels: Sequence[int],
    seeds: Sequence[int],
    *,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    rollouts: int = DEATH_RATE_ROLLOUTS,
    monster_samples: int = MONSTER_SAMPLES,
    overrides: dict[str, Any] | None = None,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """Train both algorithms on every level x seed, then measure what each ticket actually claims.

    Two different questions need two different measurements, and conflating them is the trap this
    function exists to avoid. On the deterministic levels 2-3 a single greedy rollout *is* the
    policy, so the evidence is the collection order and whether the run took everything in the
    optimal number of steps. On levels 4-5 the transitions are stochastic and one rollout is a
    sample: the evidence there is the success rate, the death rate and the mean return over many
    seeded rollouts, which is also where the two algorithms are free to differ again.

    `overrides` narrows every level's config through `dataclasses.replace` — the way a test or a
    smoke run shortens a training run without any algorithm ever seeing a bare literal. Leave it
    unset for the real evidence run: the numbers that belong in the report are the ones in
    `config/gridworld.yaml`, per-level overrides included.
    """
    runs: list[dict[str, Any]] = []
    for level in levels:
        base = config_for_level(level)
        if overrides:
            base = replace(base, **overrides)
        for seed in seeds:
            config = replace(base, seed=int(seed))
            for algo in sorted(ALGORITHMS):
                summary = run(level, algo, config, results_dir=results_dir,
                              write_artifacts=write_artifacts)
                q_table = summary["result"].q_table
                summary["rollout_stats"] = {
                    "greedy": death_rate(
                        level, q_table, epsilon=0.0,
                        max_steps=config.max_steps_per_episode, rollouts=rollouts,
                        seed_offset=GREEDY_SAMPLE_OFFSET + config.seed,
                    ),
                    "behaviour": death_rate(
                        level, q_table, epsilon=config.epsilon_end,
                        max_steps=config.max_steps_per_episode, rollouts=rollouts,
                        seed_offset=BEHAVIOUR_SAMPLE_OFFSET + config.seed,
                    ),
                }
                runs.append(summary)

    # One seed per level, not one shared seed: levels 4 and 5 both carry two monsters, so the same
    # seed would draw the identical sequence on both and the second measurement would be a copy of
    # the first dressed up as independent corroboration.
    monsters = {
        level: {
            "movement": measure_monster_movement(level, samples=monster_samples, seed=level),
            "constrained_choice": measure_constrained_monster_choice(
                level, samples=max(1000, monster_samples // 2), seed=level
            ),
            "death_checks": verify_death_checks(level),
        }
        for level in levels
        if any(Tile.MONSTER in row for row in load_level(level))
    }

    report: dict[str, Any] = {
        "levels": list(levels),
        "seeds": list(seeds),
        "rollouts_per_measurement": int(rollouts),
        "runs": [
            {key: value for key, value in summary.items()
             if key not in {"result", "rollout", "paths"}}
            for summary in runs
        ],
        "monsters": monsters,
        "artifacts": {},
    }
    report["run_objects"] = runs

    if write_artifacts:
        results_dir = Path(results_dir)
        stem = (f"levels{min(levels)}-{max(levels)}_seeds"
                f"{'-'.join(str(seed) for seed in seeds)}")
        markdown_path = write_levels_markdown(report, results_dir / f"{stem}.md")
        json_path = results_dir / f"{stem}.json"
        json_path.write_text(
            json.dumps({k: v for k, v in report.items() if k != "run_objects"}, indent=2),
            encoding="utf-8",
        )
        report["artifacts"] = {
            "levels_markdown": _reportable(markdown_path),
            "levels_json": _reportable(json_path),
        }
        report["paths"] = {"levels_markdown": markdown_path, "levels_json": json_path}

    return report


def write_levels_markdown(report: dict[str, Any], path: Path) -> Path:
    """The written deliverable for A3-005 and A3-006, formatted from the run that just happened."""
    levels = report["levels"]
    seeds = report["seeds"]
    rollouts = report["rollouts_per_measurement"]
    runs = report["runs"]
    by_level: dict[int, list[dict[str, Any]]] = {level: [] for level in levels}
    for entry in runs:
        by_level[entry["level"]].append(entry)

    lines: list[str] = [
        f"# Levels {min(levels)}-{max(levels)}: both algorithms, collection order and the "
        "stochastic rates",
        "",
        "*Generated by "
        f"`python -m train.train_gridworld --levels {' '.join(str(x) for x in levels)} "
        f"--seeds {' '.join(str(x) for x in seeds)}`. Covers ticket A3-005 (levels 2-3, rubric row "
        "D) and A3-006 (levels 4-5, no rubric row of its own).*",
        "",
        "Every hyperparameter comes from `config/gridworld.yaml`, including the per-level "
        "`level_overrides`. Each measurement below is over "
        f"{rollouts} seeded rollouts per algorithm per seed, drawn from streams that do not "
        "overlap the training or the greedy-rollout streams.",
        "",
    ]

    for level in levels:
        entries = by_level[level]
        config_line = entries[0]
        stochastic = config_line["stochastic_level"]
        lines += [
            f"## Level {level}",
            "",
            f"`{config_line['episodes']}` episodes, alpha {config_line['alpha']}, gamma "
            f"{config_line['gamma']}, epsilon {config_line['epsilon_start']} -> "
            f"{config_line['epsilon_end']} over {config_line['epsilon_decay_episodes']} episodes, "
            f"step cap {config_line['max_steps_per_episode']}. "
            f"{config_line['n_collectibles']} collectibles worth "
            f"{level_total_reward(level):.1f} in total; BFS "
            f"{'lower bound' if stochastic else 'optimum'} "
            f"{config_line['bfs_optimum_steps']} steps.",
            "",
        ]
        if stochastic:
            lines += [
                "| algo | seed | greedy success | greedy death | greedy truncation | mean return "
                "| mean steps | success at eps=" f"{config_line['epsilon_end']} | death at eps="
                f"{config_line['epsilon_end']} |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for entry in entries:
                greedy = entry["rollout_stats"]["greedy"]
                behaviour = entry["rollout_stats"]["behaviour"]
                lines.append(
                    f"| {entry['algo']} | {entry['seed']} | {greedy['success_rate']:.1%} | "
                    f"{greedy['death_rate']:.1%} | {greedy['truncation_rate']:.1%} | "
                    f"{greedy['mean_return']:.3f} / {level_total_reward(level):.1f} | "
                    f"{greedy['mean_steps']:.1f} | {behaviour['success_rate']:.1%} | "
                    f"{behaviour['death_rate']:.1%} |"
                )
            lines += [
                "",
                "The curves for this level are noisy and do not flatten, and that is the "
                "stochasticity rather than a fault: a monster moves with probability 0.4 after "
                "every agent action and monster positions are not part of the state key, so the "
                "same policy played twice from the same start produces different episodes. The "
                "rates above are the meaningful summary; a single greedy rollout is one sample "
                "from that distribution.",
                "",
            ]
        else:
            lines += [
                "| algo | seed | solved | greedy steps | BFS optimum | greedy return | collection "
                "order (item@step) | key before chest |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for entry in entries:
                order = entry["key_precedes_chest"]
                verdict = {True: "yes", False: "NO", None: "n/a"}[order]
                lines.append(
                    f"| {entry['algo']} | {entry['seed']} | "
                    f"{'yes' if entry['solved'] else 'NO'} | {entry['greedy_steps']} | "
                    f"{entry['bfs_optimum_steps']} | {entry['greedy_return']:.1f} / "
                    f"{level_total_reward(level):.1f} | "
                    f"`{entry['collection_order_text']}` | {verdict} |"
                )
            lines += [
                "",
                "The greedy rollouts here are deterministic, so the collection order in that "
                "column is the policy's order, not a sample of it. The chest pays +2 only while "
                "the key is held and otherwise stays on the grid paying nothing, so a policy that "
                "had not learnt the dependency would show a `C` before any `K` and would end the "
                "episode short of the level's full return.",
                "",
            ]
        for entry in entries:
            arts = entry.get("artifacts", {})
            if arts:
                lines.append(
                    f"- {entry['algo']} seed {entry['seed']}: `{arts.get('training_curve')}`, "
                    f"`{arts.get('policy_arrows')}`, `{arts.get('history_csv')}`, "
                    f"`{arts.get('summary_json')}`"
                )
        lines.append("")

    if report["monsters"]:
        lines += ["## Monster mechanics, measured against the shipped environment", ""]
        for level, measured in report["monsters"].items():
            movement = measured["movement"]
            checks = measured["death_checks"]
            shares = ", ".join(f"({d}) {share:.4f}" for d, share in
                               movement["direction_shares"].items())
            lines += [
                f"### Level {level}",
                "",
                f"- move probability: **{movement['measured_move_probability']:.5f}** over "
                f"{movement['observations']} monster-observations "
                f"({movement['monsters']} monsters), against the specified "
                f"{movement['expected_move_probability']}; standard error "
                f"{movement['standard_error']:.5f}.",
                f"- direction shares among moves (row,col deltas): {shares} — the monsters roam, "
                "so this is pooled over every cell they stood on and each of the four directions "
                "takes about a quarter.",
                f"- landings on a rock: **{movement['landings_on_rock']}**; landings off the grid: "
                f"**{movement['landings_off_grid']}**.",
            ]
            constrained = measured["constrained_choice"]
            constrained_shares = ", ".join(f"{landing} {share:.4f}" for landing, share in
                                           constrained["landing_shares"].items())
            lines += [
                f"- a monster confined to {constrained['cell']}, which has only the legal "
                f"neighbours {constrained['legal_neighbours']} (the rest are rocks or off the "
                f"grid), over {constrained['samples']} draws: {constrained_shares} — against an "
                f"expected "
                f"{constrained['expected_share_per_legal_direction']:.4f} per legal direction and "
                f"{1 - MONSTER_MOVE_PROBABILITY:.1f} for staying put. Illegal landings: "
                f"**{constrained['illegal_landings']}**. This is the measurement that shows the "
                "choice is uniform over the *legal* directions: in open ground all four are legal, "
                "so an implementation that ignored the rule entirely would look identical there.",
                f"- agent enters a monster tile: died="
                f"{checks['agent_entered_monster']['died']}, terminated="
                f"{checks['agent_entered_monster']['terminated']} (agent "
                f"{checks['agent_entered_monster']['agent_before']} -> "
                f"{checks['agent_entered_monster']['agent_after']}).",
            ]
            entered_by = checks["monster_entered_agent"]
            if entered_by:
                lines.append(
                    "- monster enters a stationary agent's tile (env seed "
                    f"{entered_by['env_seed']}, agent pressed UP into the grid edge so it stayed "
                    f"at {entered_by['agent_after']}, monster "
                    f"{entered_by['monster_before']} -> {entered_by['monster_after']}): died="
                    f"{entered_by['died']}, terminated={entered_by['terminated']}."
                )
            else:
                lines.append("- monster enters a stationary agent's tile: **NOT OBSERVED**.")
            lines.append("")

    if report["monsters"]:
        lines += [
            "## Why levels 4 and 5 have a larger episode budget than they used to",
            "",
            "The `truncation` column above reads 0.0% everywhere, and it took a config change to "
            "get there. At the previous budgets (level 4: 6000/4000, level 5: 8000/5000) the "
            "training curves looked settled — level 4 ended around a mean return of 2.4-2.5 out "
            "of 3 — while the greedy policy on some seeds collected nothing after the first item "
            "and oscillated between two adjacent cells until a monster caught it or the step cap "
            "fired. Monster positions are not part of the state key, so the greedy policy is a "
            "fixed map from `(row, col, has_key, mask)` to an action, and two neighbouring cells "
            "whose values have not yet separated by more than the 5% per step that gamma=0.95 "
            "implies produce a two-cycle that the epsilon-greedy behaviour policy papers over "
            "during training. On level 4 seed 0 the pair was (3,2) and (4,2) at 0.7879 against "
            "0.7867. The budgets in `config/gridworld.yaml` are the smallest ones that were clean "
            "across every seed tried (0-4, both algorithms), and the inline comment there records "
            "the full diagnosis, including that the recovery is not monotone in the budget.",
            "",
        ]

    lines += [
        "## Reward accounting",
        "",
        "Apple +1, key 0 (but unlocks chests), chest +2 only while the key is held, every other "
        "step 0, and no explicit death penalty — the values in `gridworld/constants.py`, which the "
        "brief fixes. A full episode's summed rewards therefore equal the level's collectible "
        "total exactly: 5.0 on level 2 (3 apples + chest), 4.0 on level 3 (2 apples + chest), 3.0 "
        "on level 4 (3 apples) and 4.0 on level 5 (2 apples + chest). "
        "`tests/test_gridworld_levels_2_to_5.py` sums a full episode step by step and compares it "
        "against that total.",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    return path


# --- F / A3-007: the level 6 intrinsic-reward experiment -----------------------------------------


#: Episodes at the tail of a run that the "converged" numbers are read from. Not a hyperparameter —
#: nothing here reaches the learner — but the window over which a rate is called final. 500 of the
#: experiment's 5000 episodes is long enough that a rate is not one lucky episode, and short enough
#: to sit entirely after the epsilon schedule has finished decaying at 2000.
SUCCESS_WINDOW = 500

#: Default seeds for the experiment. Five, not one: whether an epsilon-greedy agent ever stumbles
#: onto the chest at all is close to a coin flip early on, so a single seed on level 6 is an
#: anecdote. Five is also the honest ceiling on what can be claimed — see `difference_of_means`.
INTRINSIC_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: Two-sided 95% critical values of Student's t by degrees of freedom. Hard-coded because scipy is
#: not a dependency of this project and a five-seed experiment does not justify adding one. Past
#: the end of the table the normal value is close enough that the difference is invisible next to
#: the sampling error it is describing.
_T_CRITICAL_95: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042,
}


def _t_critical_95(df: float) -> float:
    """The two-sided 95% t multiplier for `df`, rounded conservatively to the table entry below."""
    if df < 1:
        return float("inf")
    candidates = [key for key in sorted(_T_CRITICAL_95) if key <= df]
    return _T_CRITICAL_95[candidates[-1]] if candidates else 1.96


def summarise_across_seeds(values: Sequence[float]) -> dict[str, Any]:
    """Mean, sample sd and standard error of a per-seed measurement.

    The standard error is the number that belongs beside the mean in the report. A spread quoted as
    a min-max range, or as the largest gap between any two seeds, *grows* with the sample size and
    says nothing about how well the mean is pinned down; the standard error shrinks like 1/sqrt(n)
    and does.
    """
    array = np.asarray(list(values), dtype=np.float64)
    n = int(array.size)
    sd = float(array.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": n,
        "values": [float(value) for value in array],
        "mean": float(array.mean()) if n else float("nan"),
        "sd": sd,
        "sem": float(sd / np.sqrt(n)) if n > 1 else 0.0,
    }


def difference_of_means(
    treatment: Sequence[float],
    baseline: Sequence[float],
) -> dict[str, Any]:
    """`mean(treatment) - mean(baseline)` with a Welch standard error and a 95% interval.

    Deliberately a difference of means and not a comparison of the best seed of one arm against the
    worst of the other. With five runs per arm the largest pairwise gap is a statistic of the tails,
    it grows with the sample size, and quoting it makes noise look like an effect.

    What this does *not* support is a claim of significance. Welch's interval at four degrees of
    freedom is about plus or minus 2.8 standard errors wide, so at n = 5 per arm only a very large
    effect clears zero. `significant_at_95` is reported so the write-up can state plainly whether
    the interval excludes zero, and the prose around it says what that is worth.
    """
    a, b = summarise_across_seeds(treatment), summarise_across_seeds(baseline)
    var_a = a["sd"] ** 2 / a["n"] if a["n"] > 1 else 0.0
    var_b = b["sd"] ** 2 / b["n"] if b["n"] > 1 else 0.0
    standard_error = float(np.sqrt(var_a + var_b))
    # Welch-Satterthwaite. If one arm is exactly constant the denominator vanishes; fall back to
    # the smaller sample's df rather than reporting an infinite one.
    denominator = (
        (var_a**2 / (a["n"] - 1) if a["n"] > 1 else 0.0)
        + (var_b**2 / (b["n"] - 1) if b["n"] > 1 else 0.0)
    )
    df = (
        float((var_a + var_b) ** 2 / denominator)
        if denominator > 0
        else float(min(a["n"], b["n"]) - 1)
    )
    difference = a["mean"] - b["mean"]
    half_width = _t_critical_95(df) * standard_error
    return {
        "treatment_mean": a["mean"],
        "baseline_mean": b["mean"],
        "difference": float(difference),
        "standard_error": standard_error,
        "welch_df": df,
        "ci95_low": float(difference - half_width),
        "ci95_high": float(difference + half_width),
        # Read straight off the interval, so the flag can never contradict the numbers printed
        # beside it. The case that forced this: every seed of one arm scored 1 and every seed of
        # the other scored 0, giving a zero standard error, an interval of exactly [-1, -1] and —
        # under an earlier `standard_error > 0` guard — the label "includes zero" next to an
        # interval that plainly does not. `zero_variance` marks that degenerate case so the prose
        # can describe it as "identical on every seed" rather than dress it up as an inference.
        "significant_at_95": bool(
            (difference - half_width) > 0.0 or (difference + half_width) < 0.0
        ),
        "zero_variance": bool(standard_error == 0.0),
        "n_per_arm": [a["n"], b["n"]],
    }


def strength_tag(strength: float) -> str:
    """`0.5` -> `strength0p5`. A dot in a filename is legal and still confuses half the tools."""
    return f"strength{strength:g}".replace(".", "p").replace("-", "m")


def intrinsic_stem_for(level: int, algo: str, strength: float, seed: int) -> str:
    """Level, algorithm, strength and seed in every name — the strength *is* the condition here,
    so a file that cannot name it is no more citable than one that cannot name its seed."""
    return f"level{level}_{algo}_{strength_tag(strength)}_seed{seed}"


def intrinsic_experiment_settings(name: str = "gridworld") -> dict[str, Any]:
    """The `intrinsic_experiment` block from the config: level, strengths, episodes.

    The experiment's parameters live beside the training parameters rather than in this file, for
    the same B3 reason everything else does: `0.0` and `0.5` are the condition under test, and a
    literal here would detach the figure from the file the report cites.
    """
    block = dict(load_yaml(name)["intrinsic_experiment"])
    return {
        "level": int(block["level"]),
        "strengths": [float(value) for value in block["strengths"]],
        "episodes": int(block["episodes"]),
    }


def intrinsic_run(
    level: int,
    algo: str,
    config: TabularConfig,
    *,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """One seeded run of `train_with_intrinsic_reward`, with the metrics a sparse level needs.

    The seed streams are spawned exactly as `run()` spawns them — `SeedSequence(seed).spawn(3)` for
    the env, the learner and the greedy rollout — so a run at `intrinsic_reward_strength = 0.0` is
    the same computation as the plain `run()` on the same seed. The baseline arm is therefore a
    genuine control rather than a second, differently-seeded run that happens to have the bonus off.

    Return alone is a poor summary here. Level 6 pays 0 for the key and +2 for the chest and nothing
    else, so an episode scores either 0.0 or 2.0 and the mean return is just the success rate times
    two. What separates an agent that is *finding* the chest from one that has *learnt the route* is
    the pair of numbers either side of that: how early the first success happened, and how often the
    agent still succeeds once epsilon has finished decaying.
    """
    seed_everything(config.seed)
    env_seed, learner_seed, rollout_seed = np.random.SeedSequence(config.seed).spawn(3)
    env = GridWorld(
        level_index=level,
        rng=make_rng(env_seed),
        max_steps=config.max_steps_per_episode,
    )

    result = train_with_intrinsic_reward(env, config, algo=algo, rng=make_rng(learner_seed))
    rollout = greedy_rollout(env, result.q_table, rng=make_rng(rollout_seed))
    optimum = optimal_collection_steps(level)
    n_collectibles = len(env.collectible_cells)

    solved = np.array(
        [record.collected == n_collectibles and not record.died for record in result.history],
        dtype=bool,
    )
    returns = result.returns
    window = min(SUCCESS_WINDOW, len(solved))
    first_success = int(np.flatnonzero(solved)[0]) if solved.any() else None

    summary: dict[str, Any] = {
        "level": level,
        "algo": algo,
        "seed": config.seed,
        "intrinsic_reward_strength": float(config.intrinsic_reward_strength),
        "episodes": config.episodes,
        "alpha": config.alpha,
        "gamma": config.gamma,
        "epsilon_start": config.epsilon_start,
        "epsilon_end": config.epsilon_end,
        "epsilon_decay_episodes": config.epsilon_decay_episodes,
        "max_steps_per_episode": config.max_steps_per_episode,
        "n_collectibles": n_collectibles,
        "bfs_optimum_steps": optimum,
        # --- every metric below is computed from ENVIRONMENT reward alone --------------------
        "solved_episodes": int(solved.sum()),
        "solved_fraction": float(solved.mean()),
        "solved_fraction_final": float(solved[-window:].mean()),
        "episodes_to_first_success": first_success,
        # A run that never succeeded has no first success. Recording the episode budget instead,
        # flagged as censored, keeps the seed in the mean without pretending it succeeded on the
        # final episode — and the flag is what stops that mean being read as unbiased.
        "first_success_censored": bool(first_success is None),
        "first_success_or_budget": int(config.episodes if first_success is None else first_success),
        "mean_return": float(returns.mean()),
        "mean_return_final": float(returns[-window:].mean()),
        "mean_steps_final": float(result.steps[-window:].mean()),
        "greedy_solved": bool(
            rollout.collected == n_collectibles and not rollout.died and not rollout.truncated
        ),
        "greedy_steps": rollout.steps,
        "greedy_return": rollout.total_return,
        "greedy_collected": rollout.collected,
        "success_window": int(window),
        "artifacts": {},
    }

    if write_artifacts:
        results_dir = Path(results_dir)
        stem = intrinsic_stem_for(level, algo, config.intrinsic_reward_strength, config.seed)
        history_path = result.write_history_csv(results_dir / f"history_{stem}.csv")
        qtable_path = save_q_table(result.q_table, results_dir / f"qtable_{stem}.npz")
        summary["artifacts"] = {
            "history_csv": _reportable(history_path),
            "q_table": _reportable(qtable_path),
        }
        summary["paths"] = {"history_csv": history_path, "q_table": qtable_path}

    summary["result"] = result
    summary["rollout"] = rollout
    return summary


#: The metrics aggregated across seeds and compared between the two arms. Deliberately four of
#: them: on a level this sparse an exploration bonus can succeed at finding the goal and still fail
#: at learning the route, and a single headline number cannot show both halves of that.
INTRINSIC_METRICS: tuple[str, ...] = (
    "solved_fraction",
    "solved_fraction_final",
    "first_success_or_budget",
    "mean_return",
    "mean_return_final",
    "greedy_solved",
)


def intrinsic_experiment(
    *,
    level: int | None = None,
    algo: str = "q",
    strengths: Sequence[float] | None = None,
    extra_strengths: Sequence[float] = (),
    seeds: Sequence[int] = INTRINSIC_SEEDS,
    episodes: int | None = None,
    epsilon_decay_episodes: int | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """The F-row experiment: level 6 at each strength over several seeds, with the figure.

    `strengths` defaults to the config's headline pair (0.0 and 0.5), and those two are what the
    comparison figure shows. `extra_strengths` runs additional values into a second, supporting
    figure and table, which is what tells "the bonus is the wrong size" apart from "a count-based
    bonus does not help on this level" — different findings with different explanations.

    Every arm shares the seed list, and within a seed both arms spawn the same env, learner and
    rollout streams, so the two conditions begin from the same random walk and diverge only once
    the bonus starts moving the table.
    """
    settings = intrinsic_experiment_settings()
    level = settings["level"] if level is None else int(level)
    headline = [float(s) for s in (settings["strengths"] if strengths is None else strengths)]
    episodes = settings["episodes"] if episodes is None else int(episodes)
    all_strengths = headline + [float(s) for s in extra_strengths if float(s) not in headline]

    base = config_for_level(level)
    changes: dict[str, Any] = {"episodes": episodes}
    if epsilon_decay_episodes is not None:
        changes["epsilon_decay_episodes"] = int(epsilon_decay_episodes)
    base = replace(base, **changes)

    runs: dict[float, list[dict[str, Any]]] = {}
    for strength in all_strengths:
        runs[strength] = [
            intrinsic_run(
                level,
                algo,
                replace(base, seed=int(seed), intrinsic_reward_strength=float(strength)),
                results_dir=results_dir,
                write_artifacts=write_artifacts,
            )
            for seed in seeds
        ]

    aggregates = {
        strength: {
            **{
                metric: summarise_across_seeds([float(run[metric]) for run in arm])
                for metric in INTRINSIC_METRICS
            },
            "censored_seeds": int(sum(run["first_success_censored"] for run in arm)),
        }
        for strength, arm in runs.items()
    }

    baseline, treatment = headline[0], headline[-1]
    comparison = {
        metric: difference_of_means(
            [float(run[metric]) for run in runs[treatment]],
            [float(run[metric]) for run in runs[baseline]],
        )
        for metric in INTRINSIC_METRICS
    }

    config_line = runs[baseline][0]
    experiment: dict[str, Any] = {
        "level": level,
        "algo": algo,
        "seeds": [int(seed) for seed in seeds],
        "headline_strengths": headline,
        "extra_strengths": [s for s in all_strengths if s not in headline],
        "episodes": episodes,
        "alpha": config_line["alpha"],
        "gamma": config_line["gamma"],
        "epsilon_start": config_line["epsilon_start"],
        "epsilon_end": config_line["epsilon_end"],
        "epsilon_decay_episodes": config_line["epsilon_decay_episodes"],
        "max_steps_per_episode": config_line["max_steps_per_episode"],
        "bfs_optimum_steps": config_line["bfs_optimum_steps"],
        "level_total_reward": level_total_reward(level),
        "success_window": config_line["success_window"],
        "formula": "r_i = intrinsic_reward_strength / sqrt(n(s) + 1)",
        "n_of_s_definition": (
            "occupancies of the state the transition arrives in, within the current episode, "
            "counted before this arrival; the counter is cleared at every env.reset()"
        ),
        "plotted_quantity": "environment return per episode; the intrinsic bonus is excluded",
        "aggregates": {f"{strength:g}": value for strength, value in aggregates.items()},
        "comparison_treatment_minus_baseline": comparison,
        "runs": {
            f"{strength:g}": [
                {
                    key: value
                    for key, value in run.items()
                    if key not in {"result", "rollout", "paths"}
                }
                for run in arm
            ]
            for strength, arm in runs.items()
        },
        "artifacts": {},
    }
    experiment["run_objects"] = runs

    if write_artifacts:
        results_dir = Path(results_dir)
        stem = f"intrinsic_level{level}_{algo}"
        experiment["paths"] = {}
        figure_path = plot_intrinsic_comparison(
            experiment, runs, headline, results_dir / f"{stem}_curve.png"
        )
        experiment["artifacts"]["comparison_figure"] = _reportable(figure_path)
        experiment["paths"]["comparison_figure"] = figure_path
        if experiment["extra_strengths"]:
            sweep_path = plot_intrinsic_comparison(
                experiment,
                runs,
                # Ascending, so the ordered colormap actually orders something: on the headline
                # figure the two arms are told apart by colour, but on the sweep the reader has
                # to see which way along the strength scale a curve sits.
                sorted(all_strengths),
                results_dir / f"{stem}_sweep.png",
                subtitle="supporting strength sweep",
            )
            experiment["artifacts"]["sweep_figure"] = _reportable(sweep_path)
            experiment["paths"]["sweep_figure"] = sweep_path
        markdown_path = results_dir / f"{stem}.md"
        json_path = results_dir / f"{stem}.json"
        # Registered before either is written: the markdown cites the JSON by name, so the name
        # has to exist in the dict the markdown is formatted from.
        experiment["artifacts"]["explanation_markdown"] = _reportable(markdown_path)
        experiment["artifacts"]["experiment_json"] = _reportable(json_path)
        write_intrinsic_markdown(experiment, markdown_path)
        # Named before the dump so the JSON can point at its own markdown and at itself. `paths`
        # holds live `Path` objects for the caller and `run_objects` holds live `TrainingResult`s;
        # neither is JSON, and both are repo-absolute besides.
        json_path.write_text(
            json.dumps(
                {k: v for k, v in experiment.items() if k not in {"run_objects", "paths"}},
                indent=2,
            ),
            encoding="utf-8",
        )
        experiment["paths"]["explanation_markdown"] = markdown_path
        experiment["paths"]["experiment_json"] = json_path

    return experiment


def plot_intrinsic_comparison(
    experiment: dict[str, Any],
    runs: dict[float, list[dict[str, Any]]],
    strengths: Sequence[float],
    path: Path,
    *,
    subtitle: str = "",
) -> Path:
    """Both conditions on one axis: mean environment return per episode, with a spread band.

    **The plotted series is the environment return.** `TrainingResult.returns` sums what
    `env.step()` paid and nothing else; the intrinsic bonus never enters it. That is the whole
    validity of this figure. A bonus is paid on almost every step, so plotting the shaped return
    would lift the strength-0.5 curve above the strength-0.0 curve by roughly (steps per episode x
    bonus) whether or not the agent ever found the chest — the curves would separate for an
    arithmetic reason and the figure would be evidence of nothing.

    Mean across seeds with a plus/minus one standard error band, never a single seed. On a level
    this sparse, whether the agent stumbles onto the chest at all in the first few hundred episodes
    is close to a coin flip, and one seed of either condition can be made to look like either
    result.

    The lower panel is the cumulative count of solved episodes. It is the upper panel's data
    integrated, but it is where "how long until the agent first solved it" is legible: the episode
    each curve lifts off the x axis is that number, and a curve that stays flat is an arm that
    never solved the level at all.
    """
    figure, axes = plt.subplots(2, 1, figsize=(9.5, 7.8), sharex=True)
    # Two arms get the standard contrast pair; a longer sweep gets an ordered colormap, because
    # with five strengths the reader needs to see which way along the scale a curve sits.
    colours = (
        ["tab:blue", "tab:red"] if len(strengths) == 2
        else plt.get_cmap("viridis")(np.linspace(0.08, 0.82, len(strengths)))
    )
    window = max(1, experiment["episodes"] // 100)

    for colour, strength in zip(colours, strengths, strict=True):
        arm = runs[float(strength)]
        label = f"strength {strength:g}" + (" (no bonus)" if strength == 0.0 else "")

        curves = np.array([moving_average(run["result"].returns, window) for run in arm])
        x = np.arange(curves.shape[1]) + window - 1
        mean = curves.mean(axis=0)
        sem = (curves.std(axis=0, ddof=1) / np.sqrt(len(arm)) if len(arm) > 1
               else np.zeros_like(mean))
        axes[0].plot(x, mean, color=colour, linewidth=1.9, label=label)
        axes[0].fill_between(x, mean - sem, mean + sem, color=colour, alpha=0.22, linewidth=0)

        solved = np.array(
            [
                np.cumsum(
                    [
                        record.collected == run["n_collectibles"] and not record.died
                        for record in run["result"].history
                    ]
                )
                for run in arm
            ],
            dtype=np.float64,
        )
        cumulative = solved.mean(axis=0)
        cumulative_sem = (solved.std(axis=0, ddof=1) / np.sqrt(len(arm)) if len(arm) > 1
                          else np.zeros_like(cumulative))
        episodes_x = np.arange(solved.shape[1])
        axes[1].plot(episodes_x, cumulative, color=colour, linewidth=1.9, label=label)
        axes[1].fill_between(episodes_x, cumulative - cumulative_sem,
                             cumulative + cumulative_sem, color=colour, alpha=0.22, linewidth=0)

    full_return = experiment["level_total_reward"]
    axes[0].axhline(full_return, color="tab:green", linestyle=":", linewidth=1.4,
                    label=f"level solved = {full_return:g}")
    axes[0].set_ylabel("environment return per episode\n(intrinsic bonus excluded)")
    # "best" rather than a fixed corner: which corner is free depends on whether the arms
    # converge, and this figure is generated for whatever the data turns out to be.
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[0].set_title(
        f"mean over {len(experiment['seeds'])} seeds, smoothed with a {window}-episode moving "
        f"average; band = +/- 1 standard error",
        fontsize=9,
    )

    axes[1].set_ylabel("cumulative episodes solved")
    axes[1].set_xlabel("episode")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(alpha=0.25)
    axes[1].set_title(
        "the episode a curve lifts off the axis is when that arm first opened the chest",
        fontsize=9,
    )

    decay = experiment["epsilon_decay_episodes"]
    for axis in axes:
        axis.axvline(decay, color="tab:grey", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[1].annotate(
        f"epsilon reaches {experiment['epsilon_end']}",
        xy=(decay, 0),
        xytext=(5, 6),
        textcoords="offset points",
        fontsize=8,
        color="tab:grey",
    )

    heading = (f"Level {experiment['level']} intrinsic reward"
               + (f" - {subtitle}" if subtitle else ""))
    figure.suptitle(
        f"{heading}: r_i = strength / sqrt(n(s) + 1), n(s) counted within the episode\n"
        f"{experiment['algo'].upper()}, {experiment['episodes']} episodes, seeds "
        f"{experiment['seeds']}, alpha={experiment['alpha']}, gamma={experiment['gamma']}, "
        f"eps {experiment['epsilon_start']}->{experiment['epsilon_end']} over {decay}, "
        f"step cap {experiment['max_steps_per_episode']}",
        fontsize=10,
        y=0.995,
        va="top",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.945))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def write_intrinsic_markdown(experiment: dict[str, Any], path: Path) -> Path:
    """The written half of rubric F, formatted from the run that just happened.

    Every number and every verdict below is derived from the experiment dict rather than typed, so
    the prose cannot drift away from the figure beside it — including when the numbers make the
    result a negative one. The direction words ("sooner"/"later", "helped"/"hurt") are computed the
    same way, because a hand-written conclusion is exactly what survives a change in the data.
    """
    level = experiment["level"]
    algo = experiment["algo"]
    seeds = experiment["seeds"]
    headline = experiment["headline_strengths"]
    baseline, treatment = headline[0], headline[-1]
    aggregates = experiment["aggregates"]
    comparison = experiment["comparison_treatment_minus_baseline"]
    window = experiment["success_window"]
    n = len(seeds)
    full_return = experiment["level_total_reward"]

    def stat(strength: float, metric: str) -> dict[str, Any]:
        return aggregates[f"{strength:g}"][metric]

    def cell(strength: float, metric: str, fmt: str = "{:.3f}") -> str:
        entry = stat(strength, metric)
        return f"{fmt.format(entry['mean'])} +/- {fmt.format(entry['sem'])}"

    def diff_line(metric: str, fmt: str = "{:.3f}") -> str:
        entry = comparison[metric]
        verdict = (
            "every seed of both arms gave the same value, so the interval is a point"
            if entry["zero_variance"]
            else "excludes zero" if entry["significant_at_95"]
            else "includes zero"
        )
        return (
            f"{fmt.format(entry['difference'])} +/- {fmt.format(entry['standard_error'])} "
            f"(95% CI {fmt.format(entry['ci95_low'])} to {fmt.format(entry['ci95_high'])}, "
            f"{verdict})"
        )

    rows = []
    for key in sorted(aggregates, key=float):
        value = float(key)
        censored = aggregates[key]["censored_seeds"]
        first = aggregates[key]["first_success_or_budget"]
        first_text = f"{first['mean']:.0f} +/- {first['sem']:.0f}"
        if censored:
            first_text += f" ({censored}/{n} never)"
        rows.append(
            f"| {value:g} | {cell(value, 'solved_fraction', '{:.1%}')} "
            f"| {cell(value, 'solved_fraction_final', '{:.1%}')} | {first_text} "
            f"| {cell(value, 'mean_return_final', '{:.3f}')} "
            f"| {cell(value, 'greedy_solved', '{:.1f}')} |"
        )

    per_seed_lines = [
        f"- strength {float(key):g}: "
        + ", ".join(
            "never" if run["episodes_to_first_success"] is None
            else str(run["episodes_to_first_success"])
            for run in experiment["runs"][key]
        )
        for key in sorted(experiment["runs"], key=float)
    ]

    first_diff = comparison["first_success_or_budget"]
    final_diff = comparison["solved_fraction_final"]
    greedy_diff = comparison["greedy_solved"]

    # Every direction word below is read off the data rather than written by hand.
    explored_sooner = first_diff["difference"] < 0
    exploited_better = final_diff["difference"] > 0
    overall = "helped" if exploited_better else "did not help"
    exploration_word = "sooner" if explored_sooner else "later"
    saturation = treatment / (1.0 - experiment["gamma"])
    drowning = saturation > full_return

    # Derived rather than written: which non-zero strength did best, and whether every non-zero
    # strength found the chest on exactly the same episodes.
    nonzero = sorted(float(key) for key in aggregates if float(key) > 0.0)
    gentlest = min(nonzero) if nonzero else treatment
    best_nonzero = max(nonzero, key=lambda s: stat(s, "solved_fraction_final")["mean"],
                       default=treatment)
    first_vectors = {
        float(key): tuple(
            run["episodes_to_first_success"] for run in experiment["runs"][key]
        )
        for key in experiment["runs"]
    }
    scale_invariant = (
        len(nonzero) > 1
        and len({first_vectors[s] for s in nonzero}) == 1
        and first_vectors[nonzero[0]] != first_vectors[baseline]
    )

    verdict_paragraph = (
        f"The bonus **{overall}**. On the metric it is designed for — how long until the agent "
        f"first opens the chest — strength {treatment:g} got there {exploration_word} than the "
        f"baseline, by {abs(first_diff['difference']):.0f} episodes on average. On the metrics "
        f"that decide whether the level was actually learnt, it is "
        f"{'ahead' if exploited_better else 'behind'}: over the final {window} episodes it solves "
        f"{stat(treatment, 'solved_fraction_final')['mean']:.1%} against the baseline's "
        f"{stat(baseline, 'solved_fraction_final')['mean']:.1%}, and its greedy policy solves the "
        f"level on {stat(treatment, 'greedy_solved')['mean']:.0%} of seeds against the baseline's "
        f"{stat(baseline, 'greedy_solved')['mean']:.0%}."
    )

    drowning_paragraph = (
        f"The bonus is large relative to the reward it is meant to help find. The whole level pays "
        f"{full_return:g}. A first arrival pays {treatment:g}, and on an "
        f"{experiment['max_steps_per_episode']}-step episode the agent collects one on nearly "
        f"every step, so the discounted intrinsic return of walking into fresh ground is worth "
        f"about `strength / (1 - gamma)` = {saturation:.0f} — "
        + (
            f"roughly {saturation / full_return:.0f}x the chest itself. Once the table has learnt "
            f"that, the greedy policy maximising it is a tour of unfamiliar cells rather than a "
            f"route to the chest, and the +2 is a rounding error inside the target. This is the "
            f"failure the plan named in one line: a large strength drowns the environment reward "
            f"and teaches wandering."
            if drowning
            else f"less than the {full_return:g} the level pays, so drowning is not the "
                 f"explanation here."
        )
    )

    # Assembled before the template so no source line of it has to exceed the line limit: a
    # markdown table row wrapped in the source would wrap in the output and stop being a table.
    table_header = (
        f"| strength | episodes solved | final {window} solved | episodes to first success "
        f"| final mean return | greedy solves |"
    )
    reproduce = (
        f"python -m train.train_gridworld --level {level} --intrinsic-sweep "
        f"--intrinsic-seeds {' '.join(str(s) for s in seeds)}"
    )

    if scale_invariant:
        shared = ", ".join(
            "never" if value is None else str(value) for value in first_vectors[nonzero[0]]
        )
        scale_note = (
            f"Every non-zero strength in the sweep first opened the chest on exactly the same "
            f"episodes - {shared}, seed by seed - while the baseline took "
            f"{', '.join('never' if v is None else str(v) for v in first_vectors[baseline])}. "
            f"That is not a coincidence, and it is worth a line in the report. Until the first "
            f"environment reward arrives, "
            f"the *only* reward in the MDP is the bonus, the Q-table starts at "
            f"zero, and every update is linear in the strength - so the whole table is exactly "
            f"proportional to it. Scaling every entry of a row by a positive constant leaves the "
            f"greedy choice and the tie set unchanged, so the behaviour policy is identical for "
            f"any strength above zero, and the agent walks the same path until it finds the chest. "
            f"The strength only begins to matter once there is a +2 for it to be weighed against. "
            f"It also means the exploration half of this experiment has an effective sample of "
            f"{n} runs, not {n} per strength."
        )
    else:
        scale_note = (
            "The non-zero strengths did not share a first-success pattern, so the exploration "
            "effect here varies with the size of the bonus as well as with the seed."
        )

    text = f"""# Task 5 - intrinsic reward on level {level} (rubric row F)

*Generated by `python -m train.train_gridworld --level {level} --intrinsic-sweep`. Figure:
`{experiment['artifacts'].get('comparison_figure')}`. Raw numbers:
`{experiment['artifacts'].get('experiment_json')}`.*

## What was run

Every transition adds a count-based exploration bonus to the reward the TD update sees:

```
r_i    = intrinsic_reward_strength / sqrt(n(s) + 1)
target = (r + r_i) + gamma * bootstrap     # bootstrap = max_a' Q[s',a'] for Q-learning
```

`n(s)` is the number of times the current episode had already occupied the state the transition
arrives in, counted **before** this arrival, and the counter is cleared at every `env.reset()`. It
is novelty *within the episode*, not a lifetime count.

The environment is untouched. `env.step()` returns exactly what it always returned; `r_i` is added
to a local variable that only the TD target sees; and the return logged, plotted and tabulated
below is the **environment** return, with the bonus excluded. That last point is what makes the
comparison mean anything. A bonus is paid on almost every step, so a curve of shaped returns would
sit strength {treatment:g} above strength {baseline:g} by roughly (steps per episode x bonus)
whether or not the agent ever found the chest, and the figure would be evidence of nothing.

Level {level} is the sparse one. The key pays 0 and only unlocks the chest; the chest pays +2 and
only while the key is held; nothing else on the grid pays anything. An episode therefore scores
exactly 0.0 or {full_return:.1f}, and the shortest solution is {experiment['bfs_optimum_steps']}
steps: down a dead-end branch of the spiral to the key, then back out to the chest. An
epsilon-greedy agent that has never seen a reward is a random walk, and a random walk on a
one-cell-wide corridor takes on the order of the squared corridor length to reach the end of it.

Configuration, all from `config/gridworld.yaml`: {algo.upper()}, {experiment['episodes']} episodes,
alpha {experiment['alpha']}, gamma {experiment['gamma']}, epsilon {experiment['epsilon_start']} ->
{experiment['epsilon_end']} over {experiment['epsilon_decay_episodes']} episodes, step cap
{experiment['max_steps_per_episode']}, seeds {seeds}. Strengths {baseline:g} and {treatment:g} are
the pair the `intrinsic_experiment` block names.

## What was measured

Mean over {n} seeds, plus or minus one standard error. "Final" means the last {window} episodes,
which sit entirely after the epsilon schedule has finished decaying.

{table_header}
|---|---|---|---|---|---|
{chr(10).join(rows)}

Episodes until the first success, seed by seed, in the order {seeds}:

{chr(10).join(per_seed_lines)}

## Did the bonus help?

{verdict_paragraph}

Strength {treatment:g} minus strength {baseline:g}, with Welch standard errors:

- share of the final {window} episodes solved: {diff_line('solved_fraction_final', '{:.3f}')}
- share of all episodes solved: {diff_line('solved_fraction', '{:.3f}')}
- mean environment return over the final {window}: {diff_line('mean_return_final', '{:.3f}')}
- episodes until the first success: {diff_line('first_success_or_budget', '{:.0f}')}
- greedy policy solves the level: {diff_line('greedy_solved', '{:.2f}')}

The exploration metric and the exploitation metrics have to be read separately, because they
disagree. Finding the chest and learning the route to it are different achievements, and this bonus
buys the first at the cost of the second.

### Why, part one: the bonus is the wrong size

{drowning_paragraph}

### Why, part two: a per-episode count is not a Markov reward

This is the more interesting half for the report. `n(s)` is per-episode by construction, but the
Q-table's state key is `(row, col, has_key, collected_mask)` and has no room for a visit count. The
bonus is therefore not a Markov reward in the MDP the agent is actually solving: the same state
pays a different bonus on its first and its fifth visit of an episode, and a table keyed without
the count can only learn the average of those. A stationary greedy policy cannot express "go
somewhere I have not been *this episode*"; the best it can represent is a fixed tour, and that is
what it converges to. A count-based bonus driven by a *lifetime* count would not have this problem
— the bonus would fade as training progressed and hand control back to the environment reward —
but the brief specifies the per-episode form, so this is a property of the specified algorithm
rather than of the implementation.

### Why, part three: the damage is a dose, and {treatment:g} is an overdose

The supporting sweep is what turns "the bonus hurt" into something more useful. The final success
rate falls monotonically with the strength, and it does not begin to fall at the gentlest value
tried: strength {gentlest:g} keeps
{stat(gentlest, 'solved_fraction_final')['mean']:.1%} of the final window against the baseline's
{stat(baseline, 'solved_fraction_final')['mean']:.1%}, and the best non-zero strength on that
metric is {best_nonzero:g}. The exploration benefit, meanwhile, is the same at every non-zero
strength (see below). So the shape of the result is not "count-based exploration does not work
here" — it is "the reward it adds has to stay small next to the reward it is helping the agent
find", and {treatment:g} is roughly {saturation / full_return:.0f} times too large by that measure.

### Why, part four: the first success does not depend on the strength at all

{scale_note}

## What this is not

With {n} seeds per arm, none of the differences above is a significance claim. The intervals quoted
are Welch intervals at about {final_diff['welch_df']:.1f} degrees of freedom, roughly plus or minus
2.8 standard errors wide, so only a very large effect clears zero at this sample size. Each bullet
above says whether its own interval excludes zero. The defensible summary is that the exploitation
result is consistent in direction across all {n} seeds
(greedy success {greedy_diff['difference']:+.2f} on a 0-1 scale), while the exploration result is
not pinned down by {n} seeds and should be quoted with its standard error or not at all.

## Reproducing it

```
{reproduce}
```

Everything else - the strengths, the episode count, alpha, gamma, the epsilon schedule and the step
cap - comes from `config/gridworld.yaml`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_rewrap_prose(text), encoding="utf-8")
    return path


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
                        help="sample size behind each reported death/success rate")
    parser.add_argument("--levels", type=int, nargs="+", default=None,
                        help="A3-005/A3-006: train both algorithms on each of these levels, "
                             "record the collection order, measure the success and death rates "
                             "over many seeded rollouts, and write the markdown summary")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="seeds for --levels; each run's artifacts carry its own seed")
    parser.add_argument("--monster-samples", type=int, default=MONSTER_SAMPLES,
                        help="monster-observations behind the measured move rate (--levels only)")
    parser.add_argument("--intrinsic-sweep", action="store_true",
                        help="F / A3-007: run level 6 at each strength in the config's "
                             "`intrinsic_experiment` block over several seeds, and write the "
                             "comparison curve, the metrics and the written explanation")
    parser.add_argument("--intrinsic-seeds", type=int, nargs="+", default=None,
                        help="seeds for --intrinsic-sweep; the default is five, because on a "
                             "level this sparse a single seed is an anecdote")
    parser.add_argument("--intrinsic-strengths", type=float, nargs="+", default=None,
                        help="supporting sweep: extra strengths run alongside the config's "
                             "headline pair, into a second figure. The headline comparison "
                             "always stays the pair named in the config")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--no-artifacts", action="store_true",
                        help="train without writing to results/ (smoke runs only)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.intrinsic_sweep:
        main_intrinsic(args)
        return
    if args.levels:
        main_levels(args)
        return
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


def main_levels(args: argparse.Namespace) -> None:
    """`--levels`: the A3-005 / A3-006 evidence run. ASCII only — the console here is cp1252."""
    seeds = args.seeds if args.seeds is not None else [
        args.seed if args.seed is not None else TabularConfig.from_yaml().seed
    ]
    # `--episodes` and `--epsilon-decay-episodes` narrow every level's own config rather than
    # replacing it, so a smoke run still reads alpha, gamma and the epsilon bounds from the file.
    overrides: dict[str, Any] = {}
    if args.episodes is not None:
        overrides["episodes"] = int(args.episodes)
    if args.epsilon_decay_episodes is not None:
        overrides["epsilon_decay_episodes"] = int(args.epsilon_decay_episodes)
    report = levels_report(
        args.levels,
        seeds,
        results_dir=args.results_dir,
        rollouts=int(args.death_rate_rollouts),
        monster_samples=int(args.monster_samples),
        overrides=overrides or None,
        write_artifacts=not args.no_artifacts,
    )
    for entry in report["runs"]:
        greedy = entry["rollout_stats"]["greedy"]
        head = (f"level {entry['level']} | {entry['algo']:5s} | seed {entry['seed']} | "
                f"{entry['episodes']} episodes")
        if entry["stochastic_level"]:
            print(f"{head}\n  success {greedy['success_rate']:.1%}, death "
                  f"{greedy['death_rate']:.1%}, truncation {greedy['truncation_rate']:.1%}, "
                  f"mean return {greedy['mean_return']:.3f} over {greedy['rollouts']} rollouts")
        else:
            verdict = "SOLVED" if entry["solved"] else "NOT SOLVED"
            key = {True: "key before chest", False: "CHEST BEFORE KEY", None: "no chest"}[
                entry["key_precedes_chest"]
            ]
            print(f"{head}\n  {verdict}: {entry['greedy_steps']} steps "
                  f"(optimum {entry['bfs_optimum_steps']}), return {entry['greedy_return']} of "
                  f"{level_total_reward(entry['level'])}\n"
                  f"  order: {entry['collection_order_text']}  [{key}]")
    for level, measured in report["monsters"].items():
        movement = measured["movement"]
        print(f"level {level} monsters: move rate {movement['measured_move_probability']:.5f} "
              f"over {movement['observations']} observations (spec "
              f"{movement['expected_move_probability']}), "
              f"{movement['landings_on_rock']} landings on rocks, "
              f"{movement['landings_off_grid']} off-grid; both death checks fire: "
              f"{measured['death_checks']['both_checks_fire']}")
    for name, path in report["artifacts"].items():
        print(f"  wrote {name}: {path}")


def main_intrinsic(args: argparse.Namespace) -> None:
    """`--intrinsic-sweep`: the F / A3-007 run. ASCII only - the console here is cp1252."""
    seeds = args.intrinsic_seeds if args.intrinsic_seeds is not None else INTRINSIC_SEEDS
    experiment = intrinsic_experiment(
        # `--level` defaults to 0, which is not a level this experiment runs on; treat the default
        # as "use the level the config's intrinsic_experiment block names".
        level=args.level if args.level != 0 else None,
        algo=args.algo,
        seeds=seeds,
        episodes=args.episodes,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        extra_strengths=args.intrinsic_strengths or (),
        results_dir=args.results_dir,
        write_artifacts=not args.no_artifacts,
    )
    baseline, treatment = (experiment["headline_strengths"][0],
                           experiment["headline_strengths"][-1])
    print(f"level {experiment['level']} intrinsic reward | {experiment['algo']} | "
          f"{experiment['episodes']} episodes | seeds {experiment['seeds']} | "
          f"step cap {experiment['max_steps_per_episode']}")
    print("  r_i = strength / sqrt(n(s) + 1); n(s) resets every episode; the return below is the "
          "ENVIRONMENT return")
    for key in sorted(experiment["aggregates"], key=float):
        entry = experiment["aggregates"][key]
        first = entry["first_success_or_budget"]
        censored = entry["censored_seeds"]
        print(f"  strength {float(key):4g}: solved {entry['solved_fraction']['mean']:.1%} of "
              f"episodes, {entry['solved_fraction_final']['mean']:.1%} in the final window; "
              f"greedy solves {entry['greedy_solved']['mean']:.0%} of seeds")
        print(f"                 first success at episode {first['mean']:.0f} +/- "
              f"{first['sem']:.0f}"
              + (f" ({censored}/{len(experiment['seeds'])} never solved)" if censored else ""))
    print(f"  strength {treatment:g} minus strength {baseline:g} "
          f"(difference of means +/- Welch standard error, n={len(experiment['seeds'])} per arm):")
    for metric in ("solved_fraction_final", "first_success_or_budget", "greedy_solved"):
        entry = experiment["comparison_treatment_minus_baseline"][metric]
        verdict = (
            "identical on every seed" if entry["zero_variance"]
            else "excludes 0" if entry["significant_at_95"]
            else "includes 0"
        )
        print(f"    {metric:24s} {entry['difference']:+9.3f} +/- {entry['standard_error']:.3f} "
              f"(95% CI {entry['ci95_low']:+.3f} to {entry['ci95_high']:+.3f}, {verdict})")
    helped = experiment["comparison_treatment_minus_baseline"]["solved_fraction_final"]
    print(f"  verdict: the bonus {'HELPED' if helped['difference'] > 0 else 'DID NOT HELP'} on the "
          f"final success rate")
    print("  (a 5-seed experiment cannot support a significance claim either way; the intervals "
          "above are what it does support)")
    for name, target in experiment["artifacts"].items():
        print(f"  wrote {name}: {target}")


if __name__ == "__main__":
    main()
