"""Part II training entry point.

    python -m train.train_arena --style direct
    python -m train.train_arena --style rotation --timesteps 400000

Trains one SB3 agent per control style and saves to `models/ppo_<style>.zip`. That folder name is
required by the brief and by the submission checklist; it is not gitignored and it ships in the zip.

Windows, and not optional
-------------------------
`SubprocVecEnv` spawns fresh interpreters that re-import this module, so every line that starts a
subprocess sits behind `if __name__ == "__main__":`. Without it the process forks recursively until
the machine gives up. `SDL_VIDEODRIVER=dummy` is set at import time, before pygame can be pulled in
by any worker, and envs are built with `render_mode=None`, so no training process ever opens a
window. Training is headless; only `eval/play_arena.py` draws.

What is logged, and why it is not just reward
---------------------------------------------
Each env is wrapped in `Monitor(env, info_keywords=INFO_KEYWORDS)` so the behavioural metrics reach
the callback, and `BehaviourLoggingCallback` writes them to TensorBoard beside `ep_rew_mean`. A
reward curve alone cannot distinguish an agent clearing phases from one farming a shaping term.
Those event files under `logs/` are the rubric's evidence of training, so they ship too.

Hyperparameters come from `config/arena.yaml` and every one is overridable by flag, so A3-013's
sweep can vary a single axis without editing the file. Defaults-only training is explicitly marked
down (rubric J3), which is why the config's baseline is a starting point rather than an answer.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Before pygame can be imported anywhere downstream, including inside a SubprocVecEnv worker.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from stable_baselines3 import DQN, PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor  # noqa: E402

from arena.env import ArenaEnv  # noqa: E402
from common.config import ArenaTrainConfig, load_yaml  # noqa: E402
from train.callbacks import (  # noqa: E402
    INFO_KEYWORDS,
    BehaviourLoggingCallback,
    ConsoleProgressCallback,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"          # REQUIRED NAME - see the brief's submission checklist
LOGS_DIR = REPO_ROOT / "logs"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

STYLES = ("direct", "rotation")
ALGORITHMS = {"PPO": PPO, "DQN": DQN}

#: Saved model path per style. Fixed rather than timestamped: `eval/play_arena.py` and the marker
#: both need to find the model without being told where it is.
def model_path(style: str, algo: str = "PPO") -> Path:
    return MODELS_DIR / f"{algo.lower()}_{style}.zip"


def make_env(style: str, seed: int, rank: int = 0):
    """A factory SubprocVecEnv can call in a fresh interpreter.

    Returns a closure rather than an env, because the subprocess has to build its own -- an env
    constructed here would have to be pickled across the process boundary, and pygame surfaces do
    not pickle.
    """

    def _init():
        env = ArenaEnv(control_style=style, render_mode=None, seed=seed + rank)
        # Per-env Monitor so `info_keywords` are captured before vectorisation flattens the infos.
        return Monitor(env, info_keywords=INFO_KEYWORDS)

    return _init


def build_vec_env(style: str, n_envs: int, seed: int, force_dummy: bool = False):
    """Vectorised envs. `DummyVecEnv` when there is one env or when asked, else `SubprocVecEnv`.

    One env in a subprocess is strictly worse than one env in-process: the IPC costs more than the
    parallelism saves. It also makes debugging far easier, which is why `--n-envs 1` is the first
    thing to try when training behaves strangely.
    """
    factories = [make_env(style, seed, rank) for rank in range(n_envs)]
    if n_envs == 1 or force_dummy:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method="spawn")


def run_name(algo: str, style: str) -> str:
    return f"{algo.lower()}_{style}_{datetime.now():%Y%m%d-%H%M%S}"


def train(
    style: str,
    *,
    timesteps: int | None = None,
    n_envs: int | None = None,
    learning_rate: float | None = None,
    n_steps: int | None = None,
    batch_size: int | None = None,
    gamma: float | None = None,
    ent_coef: float | None = None,
    seed: int | None = None,
    algo: str | None = None,
    checkpoint_every: int = 50_000,
    out_path: Path | None = None,
    tag: str | None = None,
    progress: bool = True,
) -> dict:
    """Train one style and save the model. Returns a summary dict for the caller and the report."""
    if style not in STYLES:
        raise ValueError(f"unknown control style {style!r}; expected one of {STYLES}")

    config = ArenaTrainConfig.from_yaml()
    algo_name = (algo or config.algorithm).upper()
    if algo_name not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algo_name!r}; expected one of {sorted(ALGORITHMS)}")

    total_timesteps = int(timesteps if timesteps is not None else config.total_timesteps)
    workers = int(n_envs if n_envs is not None else config.n_envs)
    run_seed = int(seed if seed is not None else config.seed)

    vec = build_vec_env(style, workers, run_seed)
    # VecMonitor on top of the per-env Monitors gives SB3 the episode statistics it prints; the
    # inner Monitors are what carry `info_keywords` through.
    vec = VecMonitor(vec)

    # Named but NOT created: SB3 makes the directory itself, and pre-creating it makes it append
    # a `_1` suffix to avoid the collision, leaving an empty directory beside the real one in the
    # submitted zip.
    log_name = tag or run_name(algo_name, style)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        policy="MlpPolicy",
        env=vec,
        learning_rate=float(learning_rate if learning_rate is not None else config.learning_rate),
        gamma=float(gamma if gamma is not None else config.gamma),
        seed=run_seed,
        verbose=1,
        tensorboard_log=str(LOGS_DIR),
        # At least one hidden layer is a rubric requirement (J2); the config's two 64-unit layers
        # are comfortably above it and small enough to train on CPU.
        policy_kwargs={"net_arch": list(config.net_arch)},
    )
    if algo_name == "PPO":
        kwargs.update(
            n_steps=int(n_steps if n_steps is not None else config.n_steps),
            batch_size=int(batch_size if batch_size is not None else config.batch_size),
            ent_coef=float(ent_coef if ent_coef is not None else config.ent_coef),
        )
    model = ALGORITHMS[algo_name](**kwargs)

    behaviour = BehaviourLoggingCallback()
    callbacks = [
        behaviour,
        CheckpointCallback(
            save_freq=max(1, checkpoint_every // max(1, workers)),
            save_path=str(CHECKPOINT_DIR),
            name_prefix=f"{algo_name.lower()}_{style}",
        ),
    ]
    if progress:
        callbacks.append(ConsoleProgressCallback(behaviour=behaviour))

    print(f"training {algo_name} on style {style!r}: {total_timesteps:,} timesteps, "
          f"{workers} envs, seed {run_seed}", flush=True)
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=log_name,
        progress_bar=False,
    )

    destination = Path(out_path) if out_path is not None else model_path(style, algo_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(destination)
    vec.close()

    summary = {
        "style": style,
        "algorithm": algo_name,
        "timesteps": total_timesteps,
        "n_envs": workers,
        "seed": run_seed,
        "model": str(destination.relative_to(REPO_ROOT)),
        "tensorboard": f"logs/{log_name}",
        **behaviour.summary(),
    }
    print(f"saved {destination.relative_to(REPO_ROOT)}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a deep RL agent on the arena.")
    parser.add_argument("--style", choices=(*STYLES, "both"), default="direct",
                        help="control scheme to train; 'both' trains direct then rotation")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--algo", choices=sorted(ALGORITHMS), default=None)
    parser.add_argument("--tag", default=None, help="name for the TensorBoard run directory")
    parser.add_argument("--quiet", action="store_true", help="suppress the progress line")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Direct first when training both: it has no rotation to learn, so it converges sooner and
    # proves the reward function works before the harder style is attempted.
    styles = list(STYLES) if args.style == "both" else [args.style]
    for style in styles:
        train(
            style,
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            seed=args.seed,
            algo=args.algo,
            tag=args.tag,
            progress=not args.quiet,
        )
    return 0


if __name__ == "__main__":
    # The guard is load-bearing on Windows: SubprocVecEnv re-imports this module in each worker.
    sys.exit(main())
