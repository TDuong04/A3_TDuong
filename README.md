# A3 — Reinforcement Learning and DL Agents

Games and Artificial Intelligence Techniques · RMIT · Group of 3 · 40% of course grade
**Due: 19 September 2026, 11:59 PM**

Part I is tabular Q-learning and SARSA in a Pygame gridworld across seven levels. Part II is a
real-time Pygame arena with two deep RL agents trained by Stable Baselines3, one per control scheme.

`A3_Brief.pdf` in this repo is the specification and the source of truth. The fixed rewards and
mechanics it defines are mirrored in `gridworld/constants.py` and `arena/constants.py` and guarded
by the test suite. **Where anything disagrees with the brief, the brief wins.**

## Team

| Member           | Student number | Owns |
| ---------------- | -------------- | ---- |
| Do Le Trang Hanh | s3977994       |      |
| Huynh Thai Duong | s3978955       |      |
| Tran Minh Nghia  | s4123236       |      |

Part II training, evaluation, the sweep and the report are shared work — Part II is roughly half the
marks and too large for one person. **This table is still unfilled and it is now blocking marks**
(ticket [A3-017](docs/tickets/A3-017-decide-ownership-and-fill-in-the.md)): the report must list
every student number and a contribution summary, and everyone must appear in the video presenting at
least one part.

## Setup

Python 3.11+. Verified on 3.14.6 (macOS) and 3.14.4 (Windows). From the repo root:

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

```powershell
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```

`pytest` should report all tests passing before you write anything. The suite guards the brief's
fixed rewards and mechanics — if it fails, something has drifted from the spec. Everything is
headless and quick except the sweep test, which trains a real model; `pytest -m "not slow"` skips it.

## Commands

```bash
# Part I — training
python -m train.train_gridworld --level 0 --algo q
python -m train.train_gridworld --level 1 --algo sarsa
python -m train.train_gridworld --level 1 --compare    # C3: both algorithms, one config,
                                                       # side-by-side policy figure + death rates
python -m train.train_gridworld --levels 0 1 2 3 4 5 --seeds 0 1 2  # D + monsters: order, rates
python -m train.train_gridworld --level 6 --intrinsic-sweep     # F: intrinsic reward, 0.0 vs 0.5

# Part I — playback (this is what the video records)
python -m eval.play_gridworld --level 0 --algo q       # play a trained policy, arrows overlaid
python -m eval.play_gridworld --level 4 --algo sarsa   # monsters, stochastic transitions
python -m eval.play_gridworld --level 1 --compare      # the same contrast animated, for the video
python -m gridworld.render                             # A1: play any level by hand, no agent

# Part II — training
python -m train.train_arena --style direct
python -m train.train_arena --style rotation
python -m train.sweep_arena --dry-run                  # J3: the sweep's run plan, no training
python -m train.sweep_arena                            # J3: explore, confirm on 3 seeds, retrain,
                                                       # then the deterministic head-to-head
python -m train.sweep_arena --report-only --promote    # re-measure the head-to-head on the models
                                                       # already on disk, without retraining

# Part II — playback
python -m eval.play_arena --style rotation
python -m eval.play_arena --style direct --human       # G: play the same env from the keyboard
python -m eval.play_arena --style direct --random      # the chance-level baseline; also what runs
                                                       # by itself when models/ is still empty
python -m eval.play_arena --style both --no-window     # R6: writes results/arena_eval/
tensorboard --logdir logs
```

Both trainers take `--seed`, and every artifact they write carries the seed in its filename, so any
number in `results/` can be reproduced from the command in the file that reports it.

## Controls

The project is marked partly from a live demo, so both windows are driveable from the keyboard.

**Gridworld** (`eval/play_gridworld.py`, `gridworld.render`)

| Key             | Does                                                                             |
| --------------- | -------------------------------------------------------------------------------- |
| `SPACE`         | pause / resume playback                                                          |
| `N`             | single-step the policy                                                           |
| `+` / `-`       | speed up / slow down                                                             |
| `R`             | reset the episode                                                                |
| `0`–`6`         | switch level — every level's trained table is loaded, not just the one requested |
| `P`             | greedy policy arrows                                                             |
| `Q` / `H`       | per-cell Q-value heatmap                                                         |
| arrows / `WASD` | take the move yourself; playback pauses so you and the policy cannot fight       |
| `ESC`           | quit                                                                             |

**Arena** (`eval/play_arena.py`)

| Key             | Does                                                          |
| --------------- | ------------------------------------------------------------- |
| `O` / `TAB`     | observation overlay — all 20 features, live                   |
| `V`             | policy overlay — action probabilities and `V(s)`              |
| arrows / `WASD` | human play: move (`direct`) or rotate and thrust (`rotation`) |
| `SPACE`         | human play: shoot                                             |
| `ESC`           | quit                                                          |

## What is visible on screen

Algorithm internals are rendered, not just logged — anything a marker needs to see has to be in the
window.

| Where                                       | Shows                                                                                                                                                                |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Gridworld HUD                               | level, steps, return, key held, items collected, speed                                                                                                               |
| Gridworld **Learner** panel                 | algorithm name, alpha, gamma, the epsilon schedule and the episode budget that trained the policy on screen — read from the run's own summary, never from the config |
| Gridworld overlays                          | greedy policy arrows (`P`), per-cell Q-value heatmap (`Q` / `H`)                                                                                                     |
| Arena HUD                                   | phase, health, score, step, current action, control style                                                                                                            |
| Arena **observation** overlay (`O` / `TAB`) | all 20 features live, plus lines to the nearest enemy and spawner and the ship-local heading                                                                         |
| Arena **policy** overlay (`V`)              | the action probability the network assigned to every action, the one it chose, and the critic's `V(s)`                                                               |

In `--compare` mode the two gridworld panels each carry their own Learner block, so "Q-learning and
SARSA share one exploration schedule" is something a marker reads off the screen rather than takes
on trust.

## Where the evidence lives

Every table below is generated by the command named inside it, against a stated commit and seed —
cite these in the report rather than re-deriving numbers by hand.

| File                                                                       | Covers                                                                                                                                        |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| [`results/levels0-5_seeds0-1-2.md`](results/levels0-5_seeds0-1-2.md)       | rows B/C/D: both algorithms on levels 0–5, 3 seeds — solved, greedy steps vs the BFS optimum, collection order, key-before-chest, death rates |
| [`results/comparison_level1_seed*.md`](results/)                           | row C3: the Q-learning vs SARSA contrast, with `compare_level1_seed*.png`                                                                     |
| [`results/intrinsic_level6_q.md`](results/intrinsic_level6_q.md)           | row F: the count-based bonus swept over five strengths and five seeds — an honest negative result                                             |
| [`results/arena_sweep/sweep_table.md`](results/arena_sweep/sweep_table.md) | row J3: the hyperparameter sweep, one axis at a time, confirmed across seeds, with a reward-hacking flag                                      |
| [`results/arena_eval/comparison.md`](results/arena_eval/comparison.md)     | row I: the two control schemes measured head-to-head on the same seeded arenas                                                                |
| [`results/arena_validation/validation.md`](results/arena_validation/validation.md) | row H: the observation vector stress-tested — per-feature ranges, dead/saturated features by name, determinism, termination and steps/second   |
| [`results/arena_eval/comparison_random.md`](results/arena_eval/comparison_random.md) | row I: the same 30 seeded arenas under a uniform random policy — the baseline that makes "clears 28 of 30" mean something                       |
| [`docs/evaluations/`](docs/evaluations/)                                   | dated whole-project audits against the rubric, each pinned to the commit it measured                                                          |

`results/` also holds the training curves, policy figures, per-run summaries, episode histories and
saved Q-tables that the playback scripts load.

## Layout

```
common/      config loading, seeding, the shared epsilon schedule
gridworld/   Part I — constants, levels, env, renderer, algorithms
arena/       Part II — constants, entities, env, observation, renderer, policy view,
             legacy API adapter
config/      gridworld.yaml, arena.yaml — all tunable parameters live here, none in code
train/       training entry points, the sweep and the TensorBoard callback
eval/        visual playback scripts (these are what the video records)
tests/       spec-drift guards
scripts/     sync_tickets.py — regenerates the board and mirrors tickets to GitHub
docs/        tickets/ (the backlog), plans/ (design notes), evaluations/ (rubric audits)
models/      trained models — REQUIRED NAME, ships in the zip, never gitignored
logs/        TensorBoard runs — ships in the zip. `ppo_*_shipped/` hold the run.json,
             monitor CSVs and events for the two models in models/
results/     training curves, Q-tables, the sweep tables, screenshots for the report
report/      report source and exported PDF
```

## Tickets

The backlog lives in [`docs/tickets/`](docs/tickets/INDEX.md) — one markdown file per ticket, with
`INDEX.md` as the board. Every remaining piece of work is already tracked there, covering all 40
rubric points, with dependencies recorded so you can see what is unblocked right now.

Priority is derived from points at risk, not preference: P0 blocks a rubric row worth 3 or more
points or blocks someone else's work, P1 is a smaller row or degraded evidence, P2 is quality, P3 is
optional.

Every ticket is mirrored to a [GitHub issue](https://github.com/TDuong04/A3_TDuong/issues), labelled
by priority, type and area, with dependencies cross-linked. The ticket files remain the source of
truth; the issues exist for whoever prefers the web UI.

To raise, update or close a ticket, ask the `ticket-bot` agent rather than editing files directly —
it allocates ids, checks for duplicates, keeps the log honest and regenerates the board. If you are
editing by hand, copy `TEMPLATE.md`, keep every frontmatter field, append to the ticket's `## Log`,
then run:

```bash
python scripts/sync_tickets.py            # regenerate the board
python scripts/sync_tickets.py --github   # also open issues for any new tickets
```

The script never re-creates an issue for a ticket that already has one, so it is safe to re-run.

Closing a ticket requires its acceptance criteria to actually be met. If they are not, say what is
outstanding and leave it open.

## Working rules

**Never edit `gridworld/constants.py` or `arena/constants.py`.** They encode values fixed by the
brief, the tests guard them, and altering rewards or mechanics is explicitly forbidden.

**No parameters in code.** Episodes, alpha, gamma, epsilon bounds and every arena hyperparameter
belong in `config/*.yaml`. Config-driven epsilon decay is a graded criterion.

**Never render inside `step()`.** Simulation files must not import pygame at all. Training runs
headless; only the eval scripts open a window.

**Run `pytest` before you push.**

**The environment is frozen.** Both agents in `models/` and every table in `results/` were measured
against the current mechanics. A mechanic or reward change now invalidates all of it and costs a
retrain plus a re-measure.

**No AI attribution in commits.** No `Co-Authored-By` trailers, no generated-with footers.

## Build order

The environment comes before the agent, always. A bug shipped into training costs an hour to
discover and looks exactly like a hyperparameter problem while you are discovering it.

1. ~~Gridworld env + renderer, playable by hand.~~ **Done** — rubric row A.
2. ~~Q-learning on level 0, then SARSA on level 1. Levels 2–3, monsters on 4–5, intrinsic reward on 6.~~ **Done** — rows B, C, D, F.
3. ~~Arena entities + renderer, playable by hand with the keyboard.~~ **Done** — row G.
4. ~~Wrap in the Gym API.~~ **Done** — row H. Run `env-validator` before any training.
5. ~~Train style 2 (direct) first, then style 1.~~ **Done** — row I, both models in `models/`.
6. ~~Hyperparameter sweep, then full-budget runs on the winner.~~ **Done** — row J3.
7. **Report and video — the substantive work left, and 7.5 points plus creativity's 5.** `report/`
   is still empty. Tickets [A3-014](docs/tickets/A3-014-write-the-report.md),
   [A3-015](docs/tickets/A3-015-record-and-edit-the-video-demonstration.md),
   [A3-017](docs/tickets/A3-017-decide-ownership-and-fill-in-the.md),
   [A3-018](docs/tickets/A3-018-ship-creativity-features-beyond-the-brief.md).
