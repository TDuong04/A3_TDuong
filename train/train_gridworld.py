"""Part I training entry point — NOT YET IMPLEMENTED.

    python -m train.train_gridworld --level 0 --algo q
    python -m train.train_gridworld --level 1 --algo sarsa
    python -m train.train_gridworld --level 6 --intrinsic-sweep

Reads `config/gridworld.yaml`, merges any `level_overrides` for the chosen level, seeds via
`common.seeding.seed_everything`, runs the algorithm headless, then writes to `results/`:

  - the learned Q-table (so `eval/play_gridworld.py` can replay the policy without retraining),
  - a per-episode history CSV: return, steps, died, collected, epsilon,
  - a smoothed training curve PNG.

Never render during training. Rendering belongs to `eval/play_gridworld.py`.

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


def main() -> None:
    raise NotImplementedError("see module docstring for the contract")


if __name__ == "__main__":
    main()
