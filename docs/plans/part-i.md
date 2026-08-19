# Part I implementation plan — classical RL in the gridworld

**Scope:** rubric rows A (2), B (2.5), C (3), D (3) and F (3) — **13.5 points directly**, plus the
Part I half of the video (5) and three of the report's six criteria.

**Tickets:** A3-001 through A3-007. **Status:** none started; A3-001 is the only unblocked root.

Everything below is a decision already made. Where a choice is load-bearing the reasoning is stated,
because reversing it later means rewriting the Q-table format and rerunning every experiment.

---

## Decisions settled before any code

### The state key is the one that can sink this

Position alone is **not** a Markov state on levels 2–6. Standing on tile (4,4) holding the key with
one apple left is a different situation from standing on (4,4) with nothing collected, and a table
indexed by position alone will average those into one meaningless number. The agent will then appear
to "almost" learn the chest levels, which reads exactly like a hyperparameter problem and can eat a
week.

Use:

```python
state = (row, col, has_key, collected_mask)   # collected_mask: int bitmask over the level's
                                              # collectible cells, one bit per cell, fixed order
```

Build the bit order once at level load (sorted collectible coordinates) so it is stable across
episodes. Level 0 has 3 apples → 8 masks; level 3 has ~4 collectibles → 16. State counts stay in the
low thousands, which is comfortably tabular.

### Q-table shape

`defaultdict(lambda: np.zeros(4, dtype=np.float64))` keyed by the state tuple. Not a dense array —
the reachable state set is sparse in the full product space, and a dict keeps level switching trivial.
Optimistic initialisation is **not** used; zeros plus epsilon-greedy is what the brief describes.

### The two update rules

```
Q-learning (off-policy):  target = r + gamma * max(Q[s2])      if not done else r
SARSA      (on-policy):   target = r + gamma * Q[s2][a2]       if not done else r
Q[s][a] += alpha * (target - Q[s][a])
```

**The terminal case matters.** Bootstrapping through a terminal transition inflates the value of
dying and is the single most common tabular RL bug. On death and on collecting the last item, the
target is `r` alone.

SARSA's loop shape differs: `a2` must be chosen *before* the update, so the next action is selected
at the end of each step and carried into the following one.

### Action selection — shared, not duplicated

```python
def select_action(q_row, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    best = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(best))
```

Both algorithms call this one function and both construct `LinearEpsilon` from the same config
block. Rubric C2 requires SARSA to use the same exploration schedule as Q-learning; sharing the code
makes that verifiable by reading one function rather than trusting two copies to stay in sync.

`np.argmax` is banned here. It returns the lowest index among ties, which biases every unvisited
state toward UP — a graded failure (B4) that also quietly degrades learning everywhere.

### Step ordering in the environment

Fixed by the brief; the order is not negotiable and the second death check is the one people miss.

1. Compute target cell. Rock or off-grid → **no movement, no penalty, no reward change.**
2. Target is fire or a monster → agent dies, episode ends.
3. Collect: apple `+1`; key `0` and sets `has_key`; chest `+2` **only if** `has_key`, otherwise the
   chest stays and pays nothing.
4. Monsters move: each independently at `p = 0.4`, uniformly among non-rock in-grid directions.
5. **A monster that moved onto the agent kills it.** The brief kills the player both when it enters
   a monster tile and when a monster enters its tile.
6. Done when all collectibles are obtained or the agent died.

### Intrinsic reward (level 6)

Lives in the training loop, not the environment. The agent keeps `visits[state]` for the current
episode, computes `r_i = strength / sqrt(n(s) + 1)`, and passes `r + r_i` into the update.
Environment rewards are untouched and the counter resets every episode. Run level 6 at strengths
`0.0` and `0.5` and plot both curves on one axis.

---

## Sequence

Each milestone has a gate. Do not start the next one until the gate passes — every one of these
gates catches a class of bug that is far more expensive to find later.

### M1 · Environment core — A3-001

Build `GridWorld` with the step ordering above. No pygame import in this file at all.

**Gate:** a random-policy harness runs 10,000 steps across all 7 levels with no crash; episodes
always terminate within `max_steps_per_episode`; summed per-event rewards equal the reported episode
return; on level 2 the chest pays nothing until the key is taken. Write these as tests.

### M2 · Renderer — A3-002

`GridRenderer` draws a `GridWorld`, animates the agent between cells, and supports pause,
single-step, speed, reset, level switch and human play.

**Gate:** play all 7 levels by hand. Rocks block, fire kills, the key gates the chest, monsters move
at roughly 40%. Ten minutes of manual play here will find bugs that would otherwise surface as a flat
learning curve three days from now.

This is 2 points that require no machine learning. Bank them.

### M3 · Q-learning on level 0 — A3-003

**Gate:** greedy policy from the start reaches all three apples by a shortest path; the training
curve rises and plateaus; the policy-arrow overlay shows a coherent field, not noise. Save the
curve and a screenshot of the arrows — both go in the report.

### M4 · SARSA and the level 1 comparison — A3-004

Run both algorithms on level 1 under identical seeds and identical schedules.

**Gate:** the two greedy policies visibly differ along the fire row — Q-learning hugging the edge,
SARSA detouring. Produce the side-by-side policy figure. If they do not differ, check that SARSA is
not accidentally using `max`, and that epsilon at convergence is high enough for the risk to matter
(with epsilon → 0 the two converge and the effect disappears, so keep `epsilon_end` at 0.05, not 0).

This figure is the strongest single piece of Part I evidence and belongs in both report and video.

### M5 · Levels 2–3, then monsters on 4–5 — A3-005, A3-006

Levels 2–3 exercise the state key from M1; if M1 was right this is mostly a config and episode-count
exercise. Levels 4–5 add stochastic transitions and need the longer runs already configured in
`config/gridworld.yaml`.

**Gate:** both algorithms complete all four levels with correct termination and reward accounting;
training curves saved for each. Expect noisier curves on 4–5 — that is the stochasticity, not a bug,
and the report should say so.

Note A3-006 carries **no rubric row of its own**. It is required by the brief and appears in the
video, but it should not displace M6.

### M6 · Intrinsic reward on level 6 — A3-007

**Gate:** two curves on one axis, with and without the bonus, plus one paragraph explaining the
difference. If the curves do not separate, the level is not sparse enough — deepen the maze rather
than inflating the strength, because a large strength drowns the environment reward and teaches
wandering.

---

## Effort and ordering across three people

M1 → M2 are sequential and on the critical path for everything else in Part I. M3 → M6 are
sequential in dependency but small once M1 is right.

The efficient split is one person on M1+M2 (environment and renderer) while a second starts the
algorithms against a stub env, and the third begins Part II's arena entities immediately — Part II is
larger and entirely independent until training. Do not put two people on Part I's critical path; it
does not parallelise.

Rough shape: M1 and M2 are the bulk of the work; M3 through M6 are incremental once the state key
and step ordering are correct.

---

## Risks

**The state key.** Highest-impact failure and the hardest to attribute. Mitigated by asserting in
tests that the same tile with and without the key produces different state keys.

**Silent tie-breaking bias.** Add a test that `select_action` with epsilon 0 over an all-equal Q-row
returns every action at least once across many calls.

**Terminal bootstrap.** Add a test that a terminal transition's target equals `r` exactly.

**Curves that cannot be reproduced.** Every run records its seed in the results filename. A curve
without a seed is not evidence and cannot go in the report.

**Evidence collected too late.** Save the figure at the moment it is generated. Regenerating a curve
later means retraining, and by then the config may have moved.

---

## Artifacts Part I must leave behind

These are what the report and video are assembled from, so they are part of "done", not follow-up:

- training curve per level per algorithm, seed in the filename
- level 1 side-by-side policy figure (Q-learning vs SARSA)
- level 6 intrinsic-reward comparison curve
- policy-arrow screenshots for levels 0 and 2
- saved Q-tables so `eval/play_gridworld.py` replays without retraining
- one paragraph each on the SARSA/Q difference and the intrinsic-reward effect
