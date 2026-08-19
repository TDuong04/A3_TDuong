"""Q-learning and SARSA — NOT YET IMPLEMENTED.

Both algorithms live in this one module and share `select_action` and the `LinearEpsilon` schedule
from `common.schedules`. That sharing is deliberate: rubric C2 requires SARSA to use the same
exploration schedule as Q-learning, and a marker can verify that by reading one function instead of
diffing two files.

Shared action selection (rubric B1, B4):

    def select_action(q_row, epsilon, rng):
        if rng.random() < epsilon:
            return rng.integers(N_ACTIONS)
        best = np.flatnonzero(q_row == q_row.max())   # ALL tied-best actions
        return rng.choice(best)                        # random tie-break

Never use `np.argmax` here. It silently returns the lowest index among ties, which biases the agent
toward UP on every unvisited state and is a B4 failure the rubric checks for by name.

The update rules differ in exactly one term, and that term is the whole point of the comparison:

    Q-learning (off-policy):  target = r + gamma * max_a' Q[s'][a']
    SARSA      (on-policy):   target = r + gamma * Q[s'][a']   where a' is the action ACTUALLY
                                                                taken next by the same epsilon-greedy
                                                                policy

    Q[s][a] += alpha * (target - Q[s][a])

On a terminal transition the bootstrap term is zero for both — `target = r`. Forgetting that is the
most common tabular RL bug and it quietly inflates the value of dying.

SARSA's loop shape differs from Q-learning's: it must choose `a'` before it can update, so the
action for the next step is selected at the end of the current one and carried forward.

Intrinsic reward (Task 5) is added by the caller, not baked in here: the agent maintains a
per-episode visit counter `n(s)`, computes `r_i = strength / sqrt(n(s) + 1)`, and passes
`r + r_i` as the reward for the update. Environment rewards stay unchanged, and the counter resets
every episode.

Each training run returns per-episode histories — return, steps, died, collected, epsilon — which
become the report's training curves. Also return the Q-table so the renderer can draw the policy.
"""

from __future__ import annotations


def q_learning(*args, **kwargs):
    raise NotImplementedError("see module docstring for the contract")


def sarsa(*args, **kwargs):
    raise NotImplementedError("see module docstring for the contract")
