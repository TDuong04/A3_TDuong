"""Part II hyperparameter sweep (A3-013, rubric J3).

    python -m train.sweep_arena                      # explore, confirm, then retrain the winner
    python -m train.sweep_arena --stages explore     # just the one-axis-at-a-time pass
    python -m train.sweep_arena --dry-run            # print the run plan and exit

Defaults-only training is explicitly marked down, so this produces the tuning evidence the report
needs: a ranked table over the axes in `config/arena.yaml`, a multi-seed confirmation of the top
configs, and one full-budget model trained from the winner.

Three stages, because each answers a different objection to the one before it:

    explore   one axis varied at a time from the baseline at `sweep.budget_timesteps`. Varying one
              axis keeps the comparison attributable — a grid would tell you which corner won, not
              which knob did it.
    confirm   the best `sweep.confirm_top_k` configs re-run on every seed in `sweep.confirm_seeds`.
              PPO on a stochastic arena is noisy enough that a one-seed win is a coin toss, so
              nothing reaches the report until it survives three seeds.
    final     the confirmed winner retrained at `training.total_timesteps` and saved under its own
              name, `models/<algo>_<style>_sweep.zip`.

Every run is a plain `train.train_arena` subprocess with per-axis flags, so any row of the table
can be reproduced by hand from the command printed in `sweep_runs.json` — and so no run can
contaminate the next one through torch's global state. Sweep runs write their models and
checkpoints inside the sweep directory, and the final run saves under `_sweep`, so nothing here
ever overwrites `models/<algo>_<style>.zip`.

Promotion to that canonical name is deliberately NOT automatic. A sweep only ever compares configs
against each other at a reduced budget, and when the axes separate by less than their seed noise
the winner is a coin toss dressed as a result — promoting it on rank alone is how a tuning run
ships a regression over a better incumbent. Promote by hand, after a deterministic evaluation shows
the tuned model actually beats the model already in `models/`.

Metrics come from the `Monitor` CSVs rather than the TensorBoard event files: they hold the same
behavioural keys, they are already per-episode, and parsing them needs no TB reader. A run is
scored on the mean over its last `sweep.summary_episodes` episodes — the tail, not the whole run,
because what matters is where a config ended up, not how it got there.

Ranking is by mean phase reached, then by mean episode reward. That order is deliberate: reward is
the thing being optimised and therefore the thing an agent can game, while phase progression is the
behaviour the brief actually asks for — so ranking on phase first is what keeps a reward farmer
off the top of the table. Any config whose reward beat the baseline while its phase progression
did not is additionally flagged as suspected reward hacking — an annotation on the row rather than
a disqualification, because the tolerance sits at one seed-noise width and is far too blunt to
decide a shortlist with.
"""

from __future__ import annotations

import os

# Set before anything can import pygame, for the same reason `train_arena` does it: a sweep launches
# many training processes and not one of them may open a window.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field, replace  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from common.config import ArenaTrainConfig, load_yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_DIR = REPO_ROOT / "logs" / "arena_sweep"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "arena_sweep"
DEFAULT_MODELS_DIR = REPO_ROOT / "models"

#: Stage names in the order they must run; `--stages` accepts any prefix of this sequence.
STAGES: tuple[str, ...] = ("explore", "confirm", "final")

#: Config fields the sweep may vary, mapped to the `train_arena` flag that overrides each one.
AXIS_FLAGS: dict[str, str] = {
    "learning_rate": "--learning-rate",
    "n_steps": "--n-steps",
    "batch_size": "--batch-size",
    "gamma": "--gamma",
    "ent_coef": "--ent-coef",
    "net_arch": "--net-arch",
}

#: Monitor CSV column -> reported metric name. `r` and `l` are Monitor's own reward and length.
METRIC_COLUMNS: dict[str, str] = {
    "phase": "phase_reached",
    "r": "ep_rew_mean",
    "spawners_destroyed": "spawners_destroyed",
    "enemies_killed": "enemies_killed",
    "damage_taken": "damage_taken",
    "l": "ep_len_mean",
}

#: Sort key for the ranked table: phase progression first, reward only as the tie-break. See the
#: module docstring — reward is the gameable quantity, so it must not be the primary ranking.
RANK_METRICS: tuple[str, ...] = ("phase_reached", "ep_rew_mean")


# --- sweep configuration ---------------------------------------------------------------------


@dataclass(frozen=True)
class SweepConfig:
    """The `sweep` section of `config/arena.yaml`."""

    budget_timesteps: int
    control_style: str
    axes: dict[str, list[Any]]
    confirm_seeds: list[int]
    confirm_top_k: int
    summary_episodes: int
    reward_hack_phase_tolerance: float

    @classmethod
    def from_yaml(cls, name: str = "arena", section: str = "sweep") -> SweepConfig:
        config = cls(**load_yaml(name)[section])
        unknown = sorted(set(config.axes) - set(AXIS_FLAGS))
        if unknown:
            raise ValueError(
                f"config/{name}.yaml sweep.axes names fields the trainer cannot override: "
                f"{unknown}; known axes are {sorted(AXIS_FLAGS)}"
            )
        if config.confirm_top_k < 1:
            raise ValueError(f"sweep.confirm_top_k must be at least 1, got {config.confirm_top_k}")
        if not config.confirm_seeds:
            raise ValueError("sweep.confirm_seeds must list at least one seed")
        return config


@dataclass(frozen=True)
class Run:
    """One planned training run: what it varies, and everything needed to launch and label it."""

    name: str
    stage: str
    axis: str
    value: Any
    seed: int
    timesteps: int
    overrides: dict[str, Any] = field(default_factory=dict)
    is_baseline: bool = False

    @property
    def label(self) -> str:
        """How the run is named in the report table, e.g. `learning_rate=0.001`."""
        return "baseline" if self.is_baseline else f"{self.axis}={_format_value(self.value)}"


def _format_value(value: Any) -> str:
    """Render an axis value compactly and identically in run names, tables and JSON."""
    if isinstance(value, (list, tuple)):
        return "x".join(str(v) for v in value)
    return str(value)


def _same_value(value: Any, current: Any) -> bool:
    """Whether an axis value is the baseline's, comparing YAML lists element-wise."""
    if isinstance(value, (list, tuple)) or isinstance(current, (list, tuple)):
        return list(value) == list(current)
    return bool(value == current)


def _slug(text: str) -> str:
    """A filesystem- and TensorBoard-safe fragment of a run label."""
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in text).strip("-")


# --- run planning ----------------------------------------------------------------------------


def plan_explore(baseline: ArenaTrainConfig, sweep: SweepConfig) -> list[Run]:
    """One run per axis value, plus the baseline itself, at the reduced budget.

    A value equal to the baseline's is skipped rather than trained twice: the baseline row already
    is that configuration, and running it again would put two rows in the table that differ only by
    the noise between two identical commands.
    """
    runs = [
        Run(
            name=f"baseline_s{baseline.seed}",
            stage="explore",
            axis="baseline",
            value=None,
            seed=baseline.seed,
            timesteps=sweep.budget_timesteps,
            is_baseline=True,
        )
    ]
    for axis, values in sweep.axes.items():
        current = getattr(baseline, axis)
        for value in values:
            if _same_value(value, current):
                continue
            runs.append(
                Run(
                    name=f"{_slug(axis)}-{_slug(_format_value(value))}_s{baseline.seed}",
                    stage="explore",
                    axis=axis,
                    value=value,
                    seed=baseline.seed,
                    timesteps=sweep.budget_timesteps,
                    overrides={axis: value},
                )
            )
    return runs


def plan_confirm(winners: list[Run], sweep: SweepConfig, baseline_seed: int) -> list[Run]:
    """Re-run each shortlisted config on every confirmation seed except the one already done."""
    runs = []
    for winner in winners:
        for seed in sweep.confirm_seeds:
            if seed == baseline_seed:
                continue  # the explore stage already produced this exact run
            runs.append(replace(winner, name=f"{winner.name.rsplit('_s', 1)[0]}_s{seed}",
                                stage="confirm", seed=seed))
    return runs


# --- launching -------------------------------------------------------------------------------


def build_command(
    run: Run,
    sweep: SweepConfig,
    log_dir: Path,
    models_dir: Path,
    config_name: str,
    device: str,
) -> list[str]:
    """The exact `train.train_arena` command line, recorded so a row can be re-run by hand."""
    command = [
        sys.executable, "-m", "train.train_arena",
        "--style", sweep.control_style,
        "--config", config_name,
        "--timesteps", str(run.timesteps),
        "--seed", str(run.seed),
        "--run-name", run.name,
        "--log-dir", str(log_dir),
        "--models-dir", str(models_dir),
        "--device", device,
    ]
    for axis, value in run.overrides.items():
        flag = AXIS_FLAGS[axis]
        if isinstance(value, (list, tuple)):
            command.extend([flag, *(str(v) for v in value)])
        else:
            command.extend([flag, str(value)])
    return command


def launch(command: list[str], stdout_path: Path) -> float:
    """Run one training subprocess with its output tee'd to a file. Returns wall-clock seconds.

    SB3 prints a table per rollout; a nine-run sweep is thousands of lines of it, so it goes to
    `<run>/train.log` where the diagnostician can read it and the console stays legible.
    """
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=handle,
                                stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        tail = "".join(stdout_path.read_text(encoding="utf-8").splitlines(keepends=True)[-20:])
        raise RuntimeError(
            f"training run failed ({result.returncode}): {' '.join(command)}\n--- tail ---\n{tail}"
        )
    return elapsed


# --- scoring ---------------------------------------------------------------------------------


def read_episodes(run_dir: Path) -> list[dict[str, float]]:
    """Every episode every worker recorded, ordered by the time it finished.

    Monitor writes one CSV per worker with a JSON header line, so the files are concatenated and
    re-sorted on the `t` column to recover the order the episodes actually completed in — which is
    what makes "the last N episodes" mean the tail of the run rather than the tail of worker 7.
    """
    episodes: list[dict[str, float]] = []
    for path in sorted(run_dir.glob("monitor/*.monitor.csv")):
        with path.open(encoding="utf-8") as handle:
            if not handle.readline().startswith("#"):
                handle.seek(0)  # no Monitor header: treat the first line as the column names
            for row in csv.DictReader(handle):
                try:
                    episodes.append({key: float(value) for key, value in row.items()
                                     if key in METRIC_COLUMNS or key == "t"})
                except (TypeError, ValueError):
                    continue  # a partially written final line, from a run killed mid-episode
    episodes.sort(key=lambda episode: episode.get("t", 0.0))
    return episodes


def summarise(run_dir: Path, episodes_window: int) -> dict[str, float]:
    """Mean of each behavioural metric over the last `episodes_window` episodes of a run."""
    episodes = read_episodes(run_dir)
    if not episodes:
        raise RuntimeError(f"no Monitor episodes under {run_dir}; did the run finish?")
    tail = episodes[-episodes_window:]
    summary = {
        metric: statistics.fmean(episode[column] for episode in tail if column in episode)
        for column, metric in METRIC_COLUMNS.items()
        if any(column in episode for episode in tail)
    }
    summary["episodes"] = float(len(episodes))
    summary["episodes_averaged"] = float(len(tail))
    return summary


def rank_key(result: dict[str, Any]) -> tuple[float, ...]:
    """Sort descending on phase reached, then on episode reward. See `RANK_METRICS`."""
    return tuple(-float(result["metrics"].get(metric, float("-inf"))) for metric in RANK_METRICS)


def flag_reward_hacking(
    results: list[dict[str, Any]],
    baseline: dict[str, Any],
    tolerance: float,
) -> None:
    """Mark, in place, every config that bought reward without buying phase progression.

    The acceptance criterion this implements is not a nicety: the arena's reward has shaping terms
    for damage avoided and enemies killed, and a policy that circles the arena shooting spawned
    enemies without ever clearing a spawner can out-earn one that progresses.

    The flag annotates; it does not disqualify. Ranking on phase before reward already stops a
    farmer from winning, and `tolerance` has to be set near the seed-to-seed noise in phase reached,
    which makes the flag too blunt to decide the shortlist with: a config that genuinely improved
    progression by less than one noise width would be thrown out along with the farmers.
    """
    base_phase = baseline["metrics"].get("phase_reached", 0.0)
    base_reward = baseline["metrics"].get("ep_rew_mean", 0.0)
    for result in results:
        phase_gain = result["metrics"].get("phase_reached", 0.0) - base_phase
        reward_gain = result["metrics"].get("ep_rew_mean", 0.0) - base_reward
        result["reward_hacking_suspected"] = bool(
            not result["is_baseline"] and reward_gain > 0.0 and phase_gain <= tolerance
        )


# --- reporting -------------------------------------------------------------------------------


def _cell(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def explore_table(results: list[dict[str, Any]]) -> str:
    """The ranked one-axis-at-a-time table, ready to paste into the report."""
    header = (
        "| Rank | Config | Phase reached | Ep reward | Spawners | Enemies | Damage | Ep len "
        "| Flag |\n"
        "|-----:|--------|--------------:|----------:|---------:|--------:|-------:|-------:"
        "|------|"
    )
    rows = []
    for rank, result in enumerate(results, start=1):
        metrics = result["metrics"]
        flag = "⚠ reward hacking" if result.get("reward_hacking_suspected") else ""
        rows.append(
            f"| {rank} | `{result['label']}` | {_cell(metrics.get('phase_reached', 0.0), 3)} "
            f"| {_cell(metrics.get('ep_rew_mean', 0.0))} "
            f"| {_cell(metrics.get('spawners_destroyed', 0.0))} "
            f"| {_cell(metrics.get('enemies_killed', 0.0))} "
            f"| {_cell(metrics.get('damage_taken', 0.0))} "
            f"| {_cell(metrics.get('ep_len_mean', 0.0), 0)} | {flag} |"
        )
    return "\n".join([header, *rows])


def confirm_table(confirmed: list[dict[str, Any]]) -> str:
    """Mean ± population sd across the confirmation seeds, per shortlisted config."""
    header = (
        "| Rank | Config | Seeds | Phase reached | Ep reward | Spawners | Damage |\n"
        "|-----:|--------|-------|--------------:|----------:|---------:|-------:|"
    )
    rows = []
    for rank, entry in enumerate(confirmed, start=1):
        metrics, spread = entry["metrics"], entry["spread"]
        rows.append(
            f"| {rank} | `{entry['label']}` | {', '.join(str(s) for s in entry['seeds'])} "
            f"| {_cell(metrics['phase_reached'], 3)} ± {_cell(spread['phase_reached'], 3)} "
            f"| {_cell(metrics['ep_rew_mean'])} ± {_cell(spread['ep_rew_mean'])} "
            f"| {_cell(metrics.get('spawners_destroyed', 0.0))} "
            f"| {_cell(metrics.get('damage_taken', 0.0))} |"
        )
    return "\n".join([header, *rows])


def render_report(report: dict[str, Any]) -> str:
    """The whole `sweep_table.md`: both tables, the flags, and the winner."""
    sweep, baseline = report["sweep"], report["baseline_config"]
    lines = [
        "# Arena hyperparameter sweep (A3-013, rubric J3)",
        "",
        f"Generated {report['generated']} from commit `{report.get('git_commit') or 'unknown'}` "
        f"by `python -m train.sweep_arena`.",
        "",
        f"Control style `{sweep['control_style']}`, {sweep['budget_timesteps']:,} timesteps per "
        f"exploratory run, one axis varied at a time from the baseline below. Each row is the mean "
        f"over that run's last {sweep['summary_episodes']} episodes. Ranked by mean phase reached, "
        f"then by mean episode reward.",
        "",
        "## Baseline",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        *(f"| `{key}` | {_format_value(value)} |" for key, value in baseline.items()),
        "",
        "## Stage 1 — one axis at a time",
        "",
        explore_table(report["explore"]),
        "",
    ]

    flagged = [r["label"] for r in report["explore"] if r.get("reward_hacking_suspected")]
    lines += [
        "Configs where episode reward rose above the baseline while mean phase reached did not "
        f"(gain \u2264 {sweep['reward_hack_phase_tolerance']}) are flagged as suspected reward "
        "hacking: "
        + (", ".join(f"`{label}`" for label in flagged) + "." if flagged else "none.")
        + (" Reward bought without phase progression is the shaping terms being farmed. The flag "
           "annotates the row rather than disqualifying it — ranking on phase before reward is "
           "what keeps a farmer off the top of the table." if flagged else ""),
        "",
    ]

    if report.get("confirm"):
        lines += [
            "## Stage 2 — top configs across seeds",
            "",
            f"The best {len(report['confirm'])} configs re-run on seeds "
            f"{', '.join(str(s) for s in sweep['confirm_seeds'])}. A one-seed win is not a result.",
            "",
            confirm_table(report["confirm"]),
            "",
        ]

    if report.get("winner"):
        winner = report["winner"]
        lines += [
            "## Winner",
            "",
            f"`{winner['label']}` — mean phase reached "
            f"{_cell(winner['metrics']['phase_reached'], 3)}"
            f", mean episode reward {_cell(winner['metrics']['ep_rew_mean'])} across "
            f"{len(winner['seeds'])} seeds.",
            "",
        ]
    if report.get("final"):
        final = report["final"]
        lines += [
            f"Retrained at the full budget of {final['timesteps']:,} timesteps and saved to "
            f"`{final['model_path']}` ({final['wall_clock_seconds']:.0f}s). Final-run means over "
            f"its last {sweep['summary_episodes']} episodes: phase reached "
            f"{_cell(final['metrics']['phase_reached'], 3)}, episode reward "
            f"{_cell(final['metrics']['ep_rew_mean'])}.",
            "",
            f"This is **not** promoted to `{final.get('incumbent_path', 'models/')}` "
            "automatically. "
            "Compare the two under `deterministic=True` first and promote only if the tuned model "
            "actually wins — when the axes separate by less than their seed noise, the ranking "
            "above is not evidence that it will.",
            "",
        ]
    return "\n".join(lines)


# --- orchestration ---------------------------------------------------------------------------


def _repo_relative(path: Path) -> str:
    """`path` as written in the report: relative to the repo when it is inside it.

    An absolute path bakes whoever ran the sweep into a committed artefact, and the report is read
    on someone else's machine.
    """
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def execute(
    run: Run,
    sweep: SweepConfig,
    log_dir: Path,
    models_dir: Path,
    config_name: str,
    device: str,
    index: int,
    total: int,
) -> dict[str, Any]:
    """Launch one run, score it, and return its table row."""
    command = build_command(run, sweep, log_dir, models_dir, config_name, device)
    print(f"[sweep] ({index}/{total}) {run.stage}: {run.label} seed={run.seed}", flush=True)
    elapsed = launch(command, log_dir / run.name / "train.log")
    metrics = summarise(log_dir / run.name, sweep.summary_episodes)
    print(
        f"[sweep]   phase={metrics.get('phase_reached', 0.0):.3f} "
        f"reward={metrics.get('ep_rew_mean', 0.0):.2f} ({elapsed:.0f}s)",
        flush=True,
    )
    return {
        "name": run.name,
        "label": run.label,
        "stage": run.stage,
        "axis": run.axis,
        "value": run.value,
        "seed": run.seed,
        "timesteps": run.timesteps,
        "is_baseline": run.is_baseline,
        "overrides": run.overrides,
        "command": " ".join(command),
        "wall_clock_seconds": round(elapsed, 1),
        "metrics": metrics,
    }


def aggregate(label: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse one config's per-seed runs into a mean and a spread per metric."""
    metrics = sorted({metric for run in runs for metric in run["metrics"]})
    return {
        "label": label,
        "seeds": [run["seed"] for run in runs],
        "runs": [run["name"] for run in runs],
        "overrides": runs[0]["overrides"],
        "is_baseline": runs[0]["is_baseline"],
        "metrics": {m: statistics.fmean(r["metrics"][m] for r in runs if m in r["metrics"])
                    for m in metrics},
        "spread": {m: statistics.pstdev([r["metrics"][m] for r in runs if m in r["metrics"]])
                   for m in metrics},
    }


def rerender(results_dir: Path) -> dict[str, Any]:
    """Rebuild `sweep_table.md` from the recorded runs, without training anything again.

    The JSON is the record of what happened; the Markdown is a view of it. Editing the wording of a
    report should never mean spending another sweep to see the change.
    """
    report = json.loads((results_dir / "sweep_runs.json").read_text(encoding="utf-8"))
    (results_dir / "sweep_table.md").write_text(render_report(report), encoding="utf-8")
    print(f"[sweep] re-rendered {results_dir / 'sweep_table.md'}")
    return report


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Run the requested stages end to end and write the JSON and Markdown artefacts."""
    if args.report_only:
        return rerender(Path(args.results_dir))

    baseline = ArenaTrainConfig.from_yaml(args.config)
    sweep = SweepConfig.from_yaml(args.config)
    if args.style is not None:
        sweep = replace(sweep, control_style=args.style)
    if args.budget is not None:
        sweep = replace(sweep, budget_timesteps=args.budget)

    stages = args.stages
    sweep_dir = Path(args.sweep_dir)
    results_dir = Path(args.results_dir)
    scratch_models = sweep_dir / "models"

    explore_runs = plan_explore(baseline, sweep)
    if args.dry_run:
        print(f"[sweep] {len(explore_runs)} exploratory runs at {sweep.budget_timesteps:,} steps:")
        for run in explore_runs:
            print("  " + " ".join(
                build_command(run, sweep, sweep_dir, scratch_models, args.config, args.device)))
        print(f"[sweep] then {sweep.confirm_top_k} configs x {len(sweep.confirm_seeds)} seeds, "
              f"then one run at {baseline.total_timesteps:,} steps")
        return {}

    report: dict[str, Any] = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "stages": list(stages),
        "sweep": {
            "budget_timesteps": sweep.budget_timesteps,
            "control_style": sweep.control_style,
            "confirm_seeds": sweep.confirm_seeds,
            "confirm_top_k": sweep.confirm_top_k,
            "summary_episodes": sweep.summary_episodes,
            "reward_hack_phase_tolerance": sweep.reward_hack_phase_tolerance,
        },
        "baseline_config": {
            "algorithm": baseline.algorithm, "n_envs": baseline.n_envs,
            "learning_rate": baseline.learning_rate, "n_steps": baseline.n_steps,
            "batch_size": baseline.batch_size, "gamma": baseline.gamma,
            "ent_coef": baseline.ent_coef, "net_arch": baseline.net_arch, "seed": baseline.seed,
        },
        "explore": [], "confirm": [], "runs": [],
    }

    # --- stage 1: one axis at a time ---
    total = len(explore_runs)
    explored = [execute(run, sweep, sweep_dir, scratch_models, args.config, args.device, i, total)
                for i, run in enumerate(explore_runs, start=1)]
    report["runs"].extend(explored)
    baseline_result = next(r for r in explored if r["is_baseline"])
    flag_reward_hacking(explored, baseline_result, sweep.reward_hack_phase_tolerance)
    explored.sort(key=rank_key)
    report["explore"] = explored

    # --- stage 2: the shortlist across seeds ---
    if "confirm" in stages:
        shortlist = explored[: sweep.confirm_top_k]
        by_name = {run.name: run for run in explore_runs}
        confirm_runs = plan_confirm([by_name[r["name"]] for r in shortlist], sweep, baseline.seed)
        total = len(confirm_runs)
        confirmed = [execute(run, sweep, sweep_dir, scratch_models, args.config, args.device,
                             i, total)
                     for i, run in enumerate(confirm_runs, start=1)]
        report["runs"].extend(confirmed)

        aggregates = []
        for result in shortlist:
            seeds = [result] + [c for c in confirmed if c["label"] == result["label"]]
            aggregates.append(aggregate(result["label"], seeds))
        aggregates.sort(key=rank_key)
        report["confirm"] = aggregates
        report["winner"] = aggregates[0]
    else:
        report["winner"] = aggregate(explored[0]["label"], [explored[0]])

    # --- stage 3: the winner at full budget ---
    if "final" in stages:
        winner = report["winner"]
        final_run = Run(
            name=f"final_{_slug(winner['label'])}_s{baseline.seed}",
            stage="final",
            axis=next(iter(winner["overrides"]), "baseline"),
            value=next(iter(winner["overrides"].values()), None),
            seed=baseline.seed,
            timesteps=baseline.total_timesteps,
            overrides=winner["overrides"],
            is_baseline=winner["is_baseline"],
        )
        # Trained into the sweep's own directory, then copied out under a `_sweep` name. The
        # trainer derives its filename from `--style`, which must stay a real control style, so
        # keeping the canonical `models/<algo>_<style>.zip` untouched means never pointing the
        # final run at `models/` in the first place.
        staging = sweep_dir / "final"
        final = execute(final_run, sweep, sweep_dir, staging, args.config, args.device, 1, 1)
        stem = f"{baseline.algorithm.lower()}_{sweep.control_style}"
        tuned = Path(args.models_dir) / f"{stem}_sweep.zip"
        tuned.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staging / f"{stem}.zip", tuned)
        final["model_path"] = _repo_relative(tuned)
        final["incumbent_path"] = _repo_relative(Path(args.models_dir) / f"{stem}.zip")
        report["runs"].append(final)
        report["final"] = final

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "sweep_runs.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    (results_dir / "sweep_table.md").write_text(render_report(report), encoding="utf-8")
    print(f"\n[sweep] wrote {results_dir / 'sweep_table.md'}")
    print(f"[sweep] wrote {results_dir / 'sweep_runs.json'}")
    print(f"[sweep] tensorboard --logdir {sweep_dir}")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="arena", help="config file under config/")
    parser.add_argument("--style", default=None,
                        help="override sweep.control_style")
    parser.add_argument("--budget", type=int, default=None,
                        help="override sweep.budget_timesteps for the exploratory runs")
    parser.add_argument("--stages", default=",".join(STAGES),
                        help=f"comma-separated prefix of {','.join(STAGES)}")
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR,
                        help="TensorBoard logs, monitor CSVs and throwaway models for sweep runs")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="where sweep_table.md and sweep_runs.json are written")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR,
                        help="where the final full-budget model is saved")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"],
                        help="cpu is faster than gpu for a 64x64 MLP on this observation size")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the run plan and exit without training")
    parser.add_argument("--report-only", action="store_true",
                        help="re-render sweep_table.md from an existing sweep_runs.json")
    args = parser.parse_args(argv)

    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    if stages != list(STAGES[: len(stages)]):
        parser.error(f"--stages must be a prefix of {','.join(STAGES)}, got {args.stages!r}")
    args.stages = stages
    return args


def main(argv: list[str] | None = None) -> None:
    run_sweep(parse_args(argv))


if __name__ == "__main__":
    main()
