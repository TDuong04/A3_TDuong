"""Part II training entry point.

    python -m train.train_arena --style direct
    python -m train.train_arena --style rotation --timesteps 400000
    python -m train.train_arena --style direct --learning-rate 0.001 --timesteps 100000 --tag lr1e-3

Trains one SB3 agent per control style and saves it to `models/<algo>_<style>.zip`. That folder
name is required by the brief; do not rename it or gitignore it.

Everything a run produces lands under one directory, `logs/<algo>_<style>[_<tag>]_<timestamp>/`:

    tb_1/                TensorBoard event files — the rubric's evidence that training happened
    monitor/*.csv        per-worker episode records, including the behavioural info keys
    run.json             the fully resolved hyperparameters, seed, git commit and wall-clock time

Nothing here reads a hyperparameter from a literal: the baseline is `config/arena.yaml` and every
flag below overrides exactly one field of it, which is what lets `sweep-runner` vary one axis at a
time. Defaults-only training is explicitly marked down (J3), and a run whose config was not
recorded cannot be cited in the report, hence `run.json`.

Behavioural metrics are logged alongside reward by `train.callbacks.BehaviourLoggingCallback`.
Reward alone cannot distinguish an agent that is progressing through phases from one that has found
a way to farm a shaping term, and that distinction is what rubric J and report row R6 are asking
about. Set this up before the first real run: retraining to recover a metric you forgot to log
costs hours.

WINDOWS/macOS: `SubprocVecEnv` starts its workers with `spawn` or `forkserver`, which re-imports
this module in each one, so everything that launches it sits behind `if __name__ == "__main__":` or
the process forks recursively. `SDL_VIDEODRIVER=dummy` is set below before any import can reach
pygame, and envs are built with `render_mode=None`, so no worker ever opens a window.
"""

from __future__ import annotations

import os

# Before any import can reach pygame. `arena.env` does not import it — rendering is loaded lazily
# inside `render()` and nowhere else — but a worker that somehow does must still get the null
# video driver rather than a window, or eight training processes fight over the display.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from dataclasses import asdict, replace  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import gymnasium as gym  # noqa: E402
from stable_baselines3 import DQN, PPO  # noqa: E402
from stable_baselines3.common.base_class import BaseAlgorithm  # noqa: E402
from stable_baselines3.common.callbacks import CallbackList  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv  # noqa: E402

from arena.constants import CONTROL_STYLES  # noqa: E402
from arena.env import ArenaEnv  # noqa: E402
from arena.mechanics import MechanicsConfig  # noqa: E402
from arena.observation import describe  # noqa: E402
from common.config import ArenaTrainConfig  # noqa: E402
from common.seeding import seed_everything  # noqa: E402
from train.callbacks import (  # noqa: E402
    BEHAVIOUR_INFO_KEYS,
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_WINDOW,
    BehaviourLoggingCallback,
    make_checkpoint_callback,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO_ROOT / "logs"
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
CHECKPOINT_SUBDIR = "checkpoints"

#: SB3 classes the brief permits. Keyed by the uppercase name `config/arena.yaml` uses.
ALGORITHMS: dict[str, type[BaseAlgorithm]] = {"PPO": PPO, "DQN": DQN}

#: Hyperparameters that only exist on the on-policy learner. Named rather than silently dropped so
#: a DQN run that inherited them from the PPO baseline says so instead of pretending they applied.
ON_POLICY_ONLY = ("n_steps", "ent_coef")

#: Rubric J2 asks for an MLP with at least one hidden layer; the policy is chosen here rather than
#: in config because the observation is a flat float vector and nothing else would be valid.
POLICY = "MlpPolicy"


# --- environment construction ----------------------------------------------------------------


def make_env(
    control_style: str,
    seed: int,
    rank: int,
    monitor_dir: Path | None = None,
    mechanics: bool = False,
) -> Callable[[], gym.Env]:
    """Build the thunk `SubprocVecEnv` calls inside each worker.

    Each worker gets `seed + rank` so the eight arenas do not replay the same spawner layouts, and
    a `Monitor` carrying `BEHAVIOUR_INFO_KEYS` so the behavioural metrics survive the episode
    boundary and reach the callback.
    """

    def _init() -> gym.Env:
        # Repeated inside the worker: `spawn` starts a fresh interpreter, and on the off-chance a
        # worker imports pygame this is the only place left to say "no display".
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        env: gym.Env = ArenaEnv(control_style=control_style, render_mode=None, mechanics=mechanics)
        filename = None if monitor_dir is None else str(monitor_dir / f"worker_{rank}")
        env = Monitor(env, filename=filename, info_keywords=BEHAVIOUR_INFO_KEYS + (
            ("pickups_collected", "shield_blocks", "elites_killed") if mechanics else ()))
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env

    return _init


def build_vec_env(
    control_style: str,
    n_envs: int,
    seed: int,
    monitor_dir: Path | None = None,
    mechanics: bool = False,
) -> VecEnv:
    """`n_envs` monitored arenas. `SubprocVecEnv` above one env, `DummyVecEnv` at one.

    A single arena in a subprocess costs a process and an IPC round trip per step and buys nothing,
    so one env stays in-process — which is also what makes a short run debuggable and testable.
    """
    if n_envs < 1:
        raise ValueError(f"n_envs must be at least 1, got {n_envs}")
    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)

    factories = [make_env(control_style, seed, rank, monitor_dir, mechanics) for rank in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories)


# --- model construction ----------------------------------------------------------------------


def algorithm_class(name: str) -> type[BaseAlgorithm]:
    """Look up the SB3 class for a config's `algorithm` field, case-insensitively."""
    try:
        return ALGORITHMS[name.upper()]
    except KeyError:
        raise ValueError(
            f"unsupported algorithm {name!r}; config/arena.yaml must name one of "
            f"{sorted(ALGORITHMS)}"
        ) from None


def ignored_hyperparameters(config: ArenaTrainConfig) -> list[str]:
    """Config fields the chosen algorithm cannot use, so `run.json` never implies they applied."""
    return [] if algorithm_class(config.algorithm) is PPO else list(ON_POLICY_ONLY)


def build_model(
    config: ArenaTrainConfig,
    venv: VecEnv,
    tensorboard_log: Path | str | None,
    device: str = "auto",
    verbose: int = 1,
) -> BaseAlgorithm:
    """Instantiate PPO or DQN from the resolved config.

    `n_steps` and `ent_coef` are on-policy concepts and are dropped for DQN — with a printed note,
    because silently ignoring a flag someone passed is how a sweep ends up comparing two identical
    runs.
    """
    cls = algorithm_class(config.algorithm)
    kwargs: dict[str, Any] = {
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "gamma": config.gamma,
        "policy_kwargs": {"net_arch": list(config.net_arch)},
        "seed": config.seed,
        "verbose": verbose,
        "tensorboard_log": None if tensorboard_log is None else str(tensorboard_log),
        "device": device,
    }
    if cls is PPO:
        kwargs["n_steps"] = config.n_steps
        kwargs["ent_coef"] = config.ent_coef
    else:
        print(f"[train_arena] {config.algorithm.upper()} ignores {', '.join(ON_POLICY_ONLY)}")
    return cls(POLICY, venv, **kwargs)


# --- run bookkeeping -------------------------------------------------------------------------


def resolve_config(args: argparse.Namespace) -> ArenaTrainConfig:
    """`config/arena.yaml` as the baseline, with one field replaced per flag the user passed."""
    config = ArenaTrainConfig.from_yaml(args.config)
    overrides = {
        "algorithm": args.algo,
        "total_timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "ent_coef": args.ent_coef,
        "net_arch": args.net_arch,
        "seed": args.seed,
    }
    return replace(config, **{k: v for k, v in overrides.items() if v is not None})


def run_name(config: ArenaTrainConfig, control_style: str, tag: str | None = None) -> str:
    """`<algo>_<style>[_<tag>]_<timestamp>` — unique per run and self-describing in TensorBoard.

    The timestamp is part of the name so a second run never overwrites the first one's curves, and
    `--tag` is what a sweep uses to say which axis a run varied.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [config.algorithm.lower(), control_style]
    if tag:
        parts.append(tag)
    parts.append(stamp)
    return "_".join(parts)


def git_commit() -> str | None:
    """The commit the run was launched from, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def write_run_metadata(
    path: Path,
    config: ArenaTrainConfig,
    control_style: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record the resolved config next to the curves it produced, and return what was written."""
    metadata: dict[str, Any] = {
        "control_style": control_style,
        "policy": POLICY,
        "started": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        **asdict(config),
    }
    metadata.update(extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


# --- training --------------------------------------------------------------------------------


def train(args: argparse.Namespace) -> Path:
    """Run one training job end to end. Returns the path of the saved model."""
    if args.style not in CONTROL_STYLES:
        raise ValueError(f"--style must be one of {CONTROL_STYLES}, got {args.style!r}")

    config = resolve_config(args)
    seed_everything(config.seed)

    name = args.run_name or run_name(config, args.style, args.tag)
    run_dir = Path(args.log_dir) / name
    monitor_dir = run_dir / "monitor"
    mechanics = args.mechanics
    models_dir = Path(args.models_dir)
    if mechanics:
        models_dir = models_dir / "mechanics"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{config.algorithm.lower()}_{args.style}"

    metadata = write_run_metadata(
        run_dir / "run.json",
        config,
        args.style,
        extra={
            "run_name": name,
            "mechanics": mechanics,
            "observation_dim": len(describe(mechanics)),
            "mechanics_config": asdict(MechanicsConfig.from_yaml()) if mechanics else None,
            "model_path": str(model_path.with_suffix(".zip")),
            "ignored_hyperparameters": ignored_hyperparameters(config),
        },
    )
    print(f"[train_arena] {name}")
    print(f"[train_arena] {json.dumps(metadata, indent=2)}")

    venv = build_vec_env(args.style, config.n_envs, config.seed, monitor_dir, mechanics)
    try:
        model = build_model(config, venv, run_dir, device=args.device, verbose=args.verbose)
        callbacks = CallbackList([
            BehaviourLoggingCallback(window=args.log_window, verbose=max(0, args.verbose - 1)),
            make_checkpoint_callback(
                models_dir / CHECKPOINT_SUBDIR,
                name_prefix=name,
                save_freq=args.checkpoint_freq,
                n_envs=config.n_envs,
                verbose=args.verbose,
            ),
        ])
        started = time.perf_counter()
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            tb_log_name="tb",
            progress_bar=args.progress_bar,
        )
        elapsed = time.perf_counter() - started
        model.save(model_path)
    finally:
        venv.close()

    write_run_metadata(
        run_dir / "run.json",
        config,
        args.style,
        extra={
            "run_name": name,
            "mechanics": mechanics,
            "observation_dim": len(describe(mechanics)),
            "mechanics_config": asdict(MechanicsConfig.from_yaml()) if mechanics else None,
            "model_path": str(model_path.with_suffix(".zip")),
            "ignored_hyperparameters": ignored_hyperparameters(config),
            "finished": datetime.now().isoformat(timespec="seconds"),
            "wall_clock_seconds": round(elapsed, 1),
        },
    )
    print(f"[train_arena] saved {model_path.with_suffix('.zip')} in {elapsed:.1f}s")
    print(f"[train_arena] tensorboard --logdir {Path(args.log_dir)}")
    return model_path.with_suffix(".zip")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--style", choices=sorted(CONTROL_STYLES), required=True,
                        help="control scheme to train; one model is trained per style")
    parser.add_argument("--mechanics", action="store_true",
                        help="train shield pickups and elite chargers (separate models/mechanics)")
    parser.add_argument("--config", default="arena", help="config file under config/")
    parser.add_argument("--algo", choices=sorted(ALGORITHMS), default=None,
                        help="override config/arena.yaml training.algorithm")
    parser.add_argument("--timesteps", type=int, default=None, help="override total_timesteps")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="parallel arenas; >1 uses SubprocVecEnv")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=None, help="PPO rollout length per env")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None, help="PPO entropy bonus")
    parser.add_argument("--net-arch", type=int, nargs="+", default=None,
                        help="hidden layer widths, e.g. --net-arch 128 128")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", default=None,
                        help="short label folded into the run name, e.g. the swept axis")
    parser.add_argument("--run-name", default=None,
                        help="use this exact run directory name instead of the generated one")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--checkpoint-freq", type=int, default=DEFAULT_CHECKPOINT_FREQ,
                        help="total timesteps between checkpoints written to models/checkpoints/")
    parser.add_argument("--log-window", type=int, default=DEFAULT_WINDOW,
                        help="episodes averaged per behavioural TensorBoard point")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--progress-bar", action="store_true",
                        help="show a tqdm progress bar (needs tqdm and rich installed)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
