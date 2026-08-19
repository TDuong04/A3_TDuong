"""Part II training entry point — NOT YET IMPLEMENTED.

    python -m train.train_arena --style direct
    python -m train.train_arena --style rotation --timesteps 400000

Trains one SB3 agent per control style and saves to `models/ppo_<style>.zip`. That folder name is
required by the brief; do not rename it or gitignore it.

WINDOWS: everything that launches `SubprocVecEnv` must sit behind `if __name__ == "__main__":` or
the process forks recursively. Set `SDL_VIDEODRIVER=dummy` before pygame is imported anywhere in
the worker path, and construct envs with `render_mode=None` so no worker opens a window.

Wrap each env in `Monitor(env, info_keywords=("phase", "spawners_destroyed", "enemies_killed"))`
and log to `logs/<algo>_<style>_<timestamp>/`. Those TensorBoard event files ship inside the
submitted zip — they are the rubric's evidence of training.

Log behavioural metrics alongside reward via the callback in `train/callbacks.py`. Reward alone
cannot distinguish an agent that is progressing through phases from one that has found a way to
farm a shaping term, and that distinction is what rubric J and report row R6 are asking about. Set
this up before the first real run: retraining to recover a metric you forgot to log costs hours.

Hyperparameters come from `config/arena.yaml`, overridable per flag so `sweep-runner` can vary one
axis at a time. Defaults-only training is explicitly marked down (J3).
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("see module docstring for the contract")


if __name__ == "__main__":
    main()
