# A3 — Reinforcement Learning and DL Agents

Games and Artificial Intelligence Techniques · RMIT · Group of 3 · 40% of course grade
**Due: 19 September 2026, 11:59 PM**

Part I is tabular Q-learning and SARSA in a Pygame gridworld across seven levels. Part II is a
real-time Pygame arena with two deep RL agents trained by Stable Baselines3, one per control scheme.

`A3_Brief.pdf` in this repo is the specification and the source of truth. The fixed rewards and
mechanics it defines are mirrored in `gridworld/constants.py` and `arena/constants.py` and guarded
by the test suite. **Where anything disagrees with the brief, the brief wins.**

## Team

| Member | Student number | Owns |
|--------|----------------|------|
| _TBD_ | _TBD_ | Part I — gridworld env, levels, renderer |
| _TBD_ | _TBD_ | Part I — Q-learning, SARSA, intrinsic reward, curves |
| _TBD_ | _TBD_ | Part II — arena entities, env, observation, renderer |

Part II training, evaluation, the sweep and the report are shared work — Part II is roughly half the
marks and too large for one person. Fill this table in early: the report must list every student
number and a contribution summary, and everyone must appear in the video presenting at least one
part.

## Setup

Python 3.11+ (verified on 3.14.4, Windows). From the repo root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```

`pytest` should report all tests passing before you write anything. The suite guards the brief's
fixed rewards and mechanics — if it fails, something has drifted from the spec.

## Commands

```powershell
# Part I
python -m train.train_gridworld --level 0 --algo q
python -m train.train_gridworld --level 1 --algo sarsa
python -m train.train_gridworld --level 1 --compare    # C3: both algorithms, one config,
                                                       # side-by-side policy figure + death rates
python -m train.train_gridworld --levels 2 3 4 5 --seeds 0 1 2   # D + monsters: order, rates
python -m train.train_gridworld --level 6 --intrinsic-sweep     # F: intrinsic reward, 0.0 vs 0.5
python -m eval.play_gridworld --level 1 --compare      # the same contrast animated, for the video
                                                       # (not yet implemented — see A3-015)

# Part II
python -m train.train_arena --style direct
python -m train.train_arena --style rotation
python -m eval.play_arena --style rotation
tensorboard --logdir logs
```

## Layout

```
common/      config loading, seeding, the shared epsilon schedule
gridworld/   Part I — constants, levels, env, renderer, algorithms
arena/       Part II — constants, entities, env, observation, renderer, legacy API adapter
config/      gridworld.yaml, arena.yaml — all tunable parameters live here, none in code
train/       training entry points and the TensorBoard callback
eval/        visual playback scripts (these are what the video records)
tests/       spec-drift guards
models/      trained models — REQUIRED NAME, ships in the zip, never gitignored
logs/        TensorBoard runs — ships in the zip
results/     training curves, Q-tables, screenshots for the report
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

```powershell
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

**Run `pytest` before you push.** It takes under a second.

**Freeze the environment before final training runs.** Any mechanic or reward change afterwards
invalidates every model and curve already collected.

**No AI attribution in commits.** No `Co-Authored-By` trailers, no generated-with footers.

## Build order

The environment comes before the agent, always. A bug shipped into training costs an hour to
discover and looks exactly like a hyperparameter problem while you are discovering it.

1. Gridworld env + renderer, playable by hand. Rubric row A, no ML required.
2. Q-learning on level 0, then SARSA on level 1. Levels 2–3, then monsters on 4–5, then intrinsic
   reward on 6.
3. Arena entities + renderer, playable by hand with the keyboard. Rubric row G, 4.5 points, still no
   ML. If it is not fun to play, the environment is usually broken.
4. Wrap in the Gym API. Run `env-validator` before any training.
5. Train style 2 (direct) first — it learns faster and proves the reward function works. Then style 1.
6. Hyperparameter sweep, then full-budget runs on the winner.
7. Report and video. These are 7.5 points plus creativity's 5 — start them before the final week.
