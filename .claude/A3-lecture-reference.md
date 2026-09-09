# A3 Reference — Reinforcement Learning & Deep RL Agents (GAIT)

**Purpose:** Course-grounded reference for the Claude Code agent implementing Assignment 3.
Distilled from the RMIT GAIT lecture slides (Weeks 2–10) and cross-checked against the A3 spec + marking rubric.

**How to use this file:** This is _reference_ material, not the task list. It tells you what the
course taught, the exact formulas to implement, and where the slides disagree with the libraries
you must use. **Authority order when sources conflict: (1) the A3 rubric, (2) the A3 spec text,
(3) these slides.** The rubric wins. Read the "Implementation gotchas" section before writing any
environment or training code — several requirements are subtly different from what the slides show.

---

## 0. Where A3 sits in the course

The course is _AI techniques applied to games_ (technique mastery over game-dev polish). A3 is the
final, group (of 3) assignment and the only one on Reinforcement Learning. It has two parts:

- **Part I — Classical / tabular RL (20 pts):** Q-learning + SARSA in a _visually rendered Pygame_
  gridworld across 7 levels (0–6). You implement the algorithms yourself. No SB3, no Gym API required.
- **Part II — Deep RL (20 pts):** A real-time Pygame arena exposed through a Gym-style API, trained
  with Stable-Baselines3 (DQN or PPO), two control schemes, TensorBoard logging.

Report (10 marks) + Video demo (5 marks) + Creativity (5 marks) sit on top; these are inside the 40.

---

## 1. RL theory the implementation must honour (Weeks 8 & 10)

### The RL loop (W8 s12–13)

`observe state → choose action → environment transitions → receive reward → repeat until episode ends`.
Every environment (gridworld and arena) is a concrete instance of this loop.

### Core vocabulary, mapped to this assignment (W8 s16–19)

- **State (gridworld):** agent grid position + locations of apples/fire/keys/chests/monsters +
  which objectives are already completed. The "objectives completed" part matters: a state that
  ignores whether the key is held or a chest is opened is _not Markov_ and will learn wrong.
- **Action:** discrete set. Gridworld = {up, down, left, right}. Arena = two schemes (see §3).
- **Reward:** numeric desirability signal. See exact tables in §2 and §3.
- **Episode:** one run start→termination. Ends on goal reached / death / max-step limit; then reset.

### Return with discounting (W8 s20)

```
R_t = Σ_{k=0..∞} γ^k · r_{t+k}
```

γ (gamma) controls horizon: small γ = short-sighted, large γ = long-term planning.

### Value functions (W8 s34–35)

- `V(s)` = expected return from state s under policy π.
- `Q(s,a)` = expected return taking a in s then following π. Q directly ranks actions — this is what
  both tabular algorithms learn.

### On-policy vs off-policy — the idea that drives the whole Q-vs-SARSA comparison (W8 s24–33, 42)

- **Q-learning = off-policy.** Updates toward `max_a' Q(s',a')`: learns the _optimal_ policy
  regardless of the exploratory action actually taken. "Learns what it should do." → tends to take
  **risky shortcuts** near hazards.
- **SARSA = on-policy.** Updates toward `Q(s',a')` for the action _actually chosen_ next. Exploration
  is baked into the target. → learns **safer, more conservative** paths near fire/monsters.
- The lecturer explicitly says you will "observe these differences clearly in gridworld." The report
  and video are expected to _show_ SARSA hugging safer routes than Q-learning (rubric C3).

---

## 2. Part I — tabular RL exact specs

### Gridworld rules (A3 spec; corroborated by W8 s16–18) — DO NOT alter rewards/mechanics

| Element        | Behaviour                                                                        |
| -------------- | -------------------------------------------------------------------------------- |
| Moves          | up / down / left / right                                                         |
| Rock           | blocks movement — attempting to enter = no movement (stay)                       |
| Fire / Monster | stepping in = **immediate death** (episode ends)                                 |
| Apple          | **+1** reward                                                                    |
| Key            | **0** reward, but enables opening chests                                         |
| Chest          | **+2** reward (requires key)                                                     |
| Episode end    | all collectible rewards obtained **or** agent dies                               |
| Monsters       | after each agent action, **40% chance** to move; move = random from allowed dirs |

Grid ~10×10 or 12×12, simple Pygame shapes. **Must be visually rendered + animated + interactive —
console/text output is explicitly not permitted (rubric A1).** Multiple levels with different layouts.

### Level plan

- **Level 0:** apples on the right only → Q-learning target (Task 1).
- **Level 1:** SARSA target (Task 2).
- **Levels 2–3:** multiple apples + a key + a chest, both algorithms (Task 3).
- **Levels 4–5:** monsters, stochastic transitions, both algorithms (Task 4).
- **Level 6:** intrinsic reward (Task 5).

### Q-learning update (W8 s38)

```
Q(s,a) ← Q(s,a) + α [ r + γ · max_{a'} Q(s',a') − Q(s,a) ]
```

### SARSA update (W8 s41)

```
Q(s,a) ← Q(s,a) + α [ r + γ · Q(s',a') − Q(s,a) ]
```

where a' is the action actually selected in s' by the same ε-greedy policy (choose a' _before_ the
update). This is the S-A-R-S-A tuple (W8 s40).

### Exploration (W8 s43–44) + A3 spec requirements

- ε-greedy: with prob ε pick a uniformly random action, else the greedy (best-Q) action.
- **Linear** ε decay from `epsilonStart` to `epsilonEnd` (config-driven). Slides only say "reduced over
  time, never zero" — the _linear_ schedule and the start/end values are a hard spec requirement
  (rubric B3). Implement it as an explicit linear interpolation across episodes, not exponential.
- **Random tie-breaking** when multiple actions share the max Q (rubric B4). Do not always return the
  first argmax — sample uniformly among tied actions.

### Stochastic transitions (W8 s45)

Monster movement makes Levels 4–5 stochastic (same action → different outcomes). Both algorithms must
still work; the agent must learn to avoid monsters while completing objectives. Evidence required:
working monster movement + training curves for L4 and L5.

### Intrinsic reward — Level 6 (A3 spec; concept in W8 s46)

The slide gives only the _idea_ (reward visiting novel states when external reward is sparse). The
exact formula is spec-defined:

```
r_i = intrinsicRewardStrength / sqrt( n(s) + 1 )
total_reward = environment_reward + r_i
```

Requirements:

- Keep **all environment rewards unchanged**.
- Maintain a **per-episode** visit counter `n(s)` for each state (reset every episode).
- Use `total_reward` inside the Q-learning **or** SARSA update (the intrinsic term enters the update,
  not just the logs).
- Evidence: training curves _with vs without_ intrinsic reward + a short explanation of the improvement.

### Learning curves (W8 s47)

Upward trend = learning; flat = failed; high variance = unstable. Every "evidence" rubric item
(Tasks 4 & 5) wants a curve — log episode return per episode and plot it.

---

## 3. Part II — deep RL exact specs

### Arena requirements (A3 spec; W10 s64 "Mapping to the Pygame Arena")

Real-time, continuous/semi-continuous movement (NOT tile-grid feel — rubric G1). Must include:
controllable player ship with movement + shooting; enemy spawners that periodically spawn enemies;
enemies that navigate toward the player; player health; enemy health; projectile collisions; a
**phase system** where destroying all active spawners advances to the next difficulty. Episode ends on
player death or max time/step count. Window ~800×600 or 960×680, simple shapes, manageable enemy count.

### Gym-style API (W10 s55–56) — ⚠ SEE GOTCHA #1

Expose `reset()`, `step(action)`, `render()`. Slides show `step` returning `(obs, reward, done, info)`.
**For SB3 you must return the Gymnasium 5-tuple — see §4.**

### Observation design (W10 s53) — fixed-size numeric vector, ~10–30 floats, no pixels

Must include at least: player position; player velocity; player orientation (if relevant to the
control scheme); distance + relative direction to nearest enemy; distance + relative direction to
nearest spawner; player health; current phase. Keep it fixed-size — pad/clip nearest-object features
so the vector length never changes when counts change.

### Two control schemes (A3 spec) — separate model + separate eval script for each (rubric I)

**Scheme 1 — rotation + thrust:** `0` no-op, `1` thrust forward, `2` rotate left, `3` rotate right,
`4` shoot. (Orientation is relevant here → include it in the obs.)
**Scheme 2 — direct movement:** `0` no-op, `1` up, `2` down, `3` left, `4` right, `5` shoot.
Both are **discrete** action spaces (W10 s54) → DQN or PPO both valid.

### Reward shaping (W10 s57; A3 spec) — justify any extra shaping in the report

Positive for destroying enemies; **larger** positive for destroying spawners; positive for phase
progression; negative for taking damage; **strong** negative on death. Optional shaping terms
(e.g. survival bonus, aim/approach shaping) must be justified — poor shaping → unstable/unintended
behaviour. Reward design is itself graded.

### Training (W10 s58–63)

- Stable-Baselines3 with **DQN or PPO**. PPO is on-policy, DQN off-policy (mirrors §1).
- Network = small MLP with **at least one hidden layer**.
- **Train headless** (no render) for speed; render only during evaluation.
- Log to **TensorBoard** (episode reward, length, loss, entropy).
- **Tune hyperparameters meaningfully** — defaults-only loses marks (rubric J3). Show exploration
  (a small table/plot of runs).
- Typical budget 100k–600k timesteps. Save checkpoints; **save final models in a folder named
  `models/`**.
- Provide eval script(s) that load a saved model, disable exploration, and _visually_ play the arena.
  Never judge on one episode — run several.

---

## 4. Implementation gotchas (read before coding)

**GOTCHA #1 — Gym vs Gymnasium API (highest risk).** The slides and the A3 text both show the legacy
4-tuple `step → (obs, reward, done, info)` and `reset → obs`. Modern SB3 (v2+) requires the
**Gymnasium** API:

```python
obs, info = env.reset(seed=None)                       # 5-tuple ecosystem: reset returns (obs, info)
obs, reward, terminated, truncated, info = env.step(a) # NOT (obs, reward, done, info)
```

`terminated` = episode ended by the environment's own rules (death, all-clear).
`truncated` = ended by the time/step limit. Build the arena to the 5-tuple or SB3 will error. Subclass
`gymnasium.Env`, set `observation_space` (`Box`, fixed shape, float32) and `action_space` (`Discrete`),
and validate with `stable_baselines3.common.env_checker.check_env` before training.
(Part I gridworld does **not** need the Gym API — you drive its loop directly.)

**GOTCHA #2 — Linear ε decay, not exponential.** Rubric B3 says linear from `epsilonStart`→`epsilonEnd`.
Implement `eps = eps_start + (eps_end - eps_start) * min(1, episode / decay_episodes)`. Read the values
from the config file (confirm whether an instructor config exists before hardcoding).

**GOTCHA #3 — Random tie-breaking (B4).** `np.argmax` returns the first max. Instead collect all
indices equal to the max Q and `random.choice` among them.

**GOTCHA #4 — Intrinsic reward enters the update.** `n(s)` is per-episode and resets each episode; the
intrinsic term must be added to the reward used in the Q/SARSA update, and environment rewards must be
left untouched (F).

**GOTCHA #5 — Markov state in the gridworld.** Encode key-held / chest-opened into the state key, or
Q-learning will oscillate. The state must contain everything needed to decide the next action (W8 s16).

**GOTCHA #6 — Fixed-size observation.** Enemy/spawner counts change over an episode; the obs vector
length must not. Use "nearest-K" or nearest-1 relative features with fixed padding.

---

## 5. Visibility & creativity (the marks most teams leave on the table)

The recurring lesson from prior assignments: **on-screen visibility of algorithm internals is graded
evidence, not decoration.** Multiple rubric items and the video require _proof the agent follows a
learned policy, not random actions_ (B5, C3, video criterion). Make internals renderable:

- Gridworld overlays: per-cell greedy action arrows, Q-value / state-value heatmap, current ε, visit
  counts `n(s)` (for L6), live episode return, and a policy-rollout mode that replays the learned
  greedy policy with exploration off.
- Arena eval HUD: current phase, health, nearest-enemy/spawner vectors the agent actually sees, and
  the chosen action — so the grader can see the observation→action mapping in real time.

**Creativity (5 marks, 12.5% of the grade)** has no slide backing — it's open. Cheapest high-value
wins: the visibility overlays above (they double as creativity + evidence), a side-by-side
Q-learning-vs-SARSA replay on the same hazard level, an intrinsic-reward on/off comparison animation,
and a clean phase-progression escalation in the arena.

---

## 6. Rubric checklist (point values)

**Part I (20)**

- A — Gridworld rendered/animated/interactive + mechanics correct — **2**
- B — Task 1 Q-learning: ε-greedy, correct off-policy update, linear ε decay (config), random
  tie-break, learned shortest path on L0 — **2.5**
- C — Task 2 SARSA: correct on-policy update, same ε schedule, evidence it's more conservative than
  Q-learning — **3**
- D — Task 3: L2–3 with multiple apples/key/chest, both algorithms, correct termination + reward
  accounting — **3**
- F — Task 5 intrinsic reward: exact formula, env rewards unchanged, per-episode `n(s)`, used in
  update, curve comparison + explanation — **3**
- (Task 4 monster levels are assessed through the above criteria + evidence curves)

**Part II (20)**

- G — Arena: real-time animated, player move+shoot, spawners, enemies chase, collisions, health,
  phase system, episode end — **4.5**
- H — Gym API (reset/step/render) + fixed-size numeric obs with required features — **2.5**
- I — Both control schemes implemented, separate saved model each, eval script each — **4**
- J — Reward design + SB3 (DQN/PPO, ≥1 hidden layer, TensorBoard) + meaningful hyperparameter
  tuning — **3**

**Cross-cutting**

- Report (≤10 pages incl. images, no appendix): both environments, obs design, reward design,
  hyperparameter exploration, control-set comparison, training evidence, originality — **2.5**
- Video (≤10 min, all members present ≥1 part, gridworld + arena shown, learned behaviour for both
  control schemes, ≥1 phase progression) — **5**
- Creativity beyond expectations — **5**

**Submission:** repo zip (all gridworld + arena code, training scripts, `models/`, TensorBoard logs) +
report PDF containing all student numbers, contribution summary, and the video link.

---

## 7. Slide → requirement index (for citing in the report)

| Report / code claim                           | Slide source   |
| --------------------------------------------- | -------------- |
| RL loop, agent/env/state/action/reward        | W8 s12–19      |
| Discounted return / γ                         | W8 s20         |
| Q(s,a), V(s)                                  | W8 s34–35      |
| On-policy vs off-policy (why SARSA is safer)  | W8 s24–33, s42 |
| Q-learning update rule                        | W8 s38         |
| SARSA update rule                             | W8 s39–41      |
| ε-greedy exploration                          | W8 s43–44      |
| Stochastic environments (monsters)            | W8 s45         |
| Intrinsic reward (concept)                    | W8 s46         |
| Learning curves                               | W8 s47         |
| Function approximation / why tables fail      | W10 s51–52     |
| Observation vector                            | W10 s53        |
| Discrete vs continuous actions                | W10 s54        |
| Gym API (reset/step/render)                   | W10 s55–56     |
| Reward shaping (arena)                        | W10 s57        |
| Value-based vs policy-based; DQN; PPO; SB3    | W10 s58–60     |
| Training loop; TensorBoard; checkpoint + eval | W10 s61–63     |

> Note: Weeks 2 (steering), 3 (FSM), 5 (A\* pathfinding), 6 (MCTS) are prior-assignment techniques and
> are not required for A3. They only matter here if the creativity component reuses steering (e.g.
> enemy navigation toward the player is a Seek/Pursue behaviour from W2) — reuse is explicitly allowed.
