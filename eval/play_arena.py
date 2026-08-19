"""Visual playback of a trained deep RL agent — NOT YET IMPLEMENTED.

    python -m eval.play_arena --style rotation
    python -m eval.play_arena --style direct --episodes 5

Rubric row I4 requires an evaluation script that can visually run each trained agent, so this must
work for both styles from `models/`. Use `deterministic=True` at evaluation.

This is what the video records for Part II. It must show enemies spawning and moving, projectiles
and collisions, and at least one phase progression — so make sure the loaded agent can actually
clear phase 1 before recording, and keep phase 1 tuned to be clearable in 20-30 seconds.

Include the observation overlay toggle and a HUD (phase, health, action name, cumulative reward).
Add `--human` to play the same env yourself, which is both a creativity feature and the fastest way
to tell a broken environment from a badly trained agent.

Print a summary over `--episodes` runs: mean and std return, mean phase reached, spawners destroyed,
survival time. Those numbers are report row R6's control-set comparison — run both styles under the
same seeds and tabulate.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("see module docstring for the contract")


if __name__ == "__main__":
    main()
