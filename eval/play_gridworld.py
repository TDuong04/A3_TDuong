"""Visual playback of a trained tabular policy — NOT YET IMPLEMENTED.

    python -m eval.play_gridworld --level 1 --algo sarsa
    python -m eval.play_gridworld --level 1 --compare      # Q-learning vs SARSA side by side

Loads the Q-table from `results/`, opens the Pygame window and plays the greedy policy. This is the
script the video records for Part I, so it has to make three things obvious to a viewer:

  - the level's items and monsters behaving per the rules,
  - the agent following a consistent learned policy rather than acting randomly (rubric V), which
    the policy-arrow overlay demonstrates far better than watching a single rollout,
  - for level 1, the difference between the Q-learning and SARSA routes — the `--compare` mode
    running both at once is the strongest possible C3 evidence.

Keyboard: pause, single-step, speed up/down, reset, toggle policy arrows, toggle Q-heatmap, switch
level, and hand control to a human. Add `--record` to dump frames to `results/` for the video.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("see module docstring for the contract")


if __name__ == "__main__":
    main()
