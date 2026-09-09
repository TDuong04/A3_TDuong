# CLAUDE.md

Shared agent context for **A3 — Reinforcement Learning and DL Agents** (RMIT, Games and AI
Techniques, 40% of the course grade, due **19 September 2026**).

This file is committed so that every member's Claude Code session starts from the same
understanding. It is loaded automatically from the repo root. Do not keep a private copy with
different rules — if something here is wrong, fix it here and push, so the whole team's agents
change together.

Audience is an agent, not a new human teammate. Onboarding prose, the full command list, the
keyboard controls and the evidence tables live in [README.md](README.md); this file covers what an
agent must not get wrong.

---

## 1. Sources of truth, in order

| Rank | Source | What it settles |
| ---- | ------ | --------------- |
| 1 | `A3_Brief.pdf` | The specification. Where anything disagrees with the brief, **the brief wins**. |
| 2 | `gridworld/constants.py`, `arena/constants.py` | The brief's fixed rewards, mechanics and action indices, mirrored in code and guarded by `tests/test_spec_constants.py`. |
| 3 | `config/gridworld.yaml`, `config/arena.yaml` | Every tunable number. Their comments record *why* each value was chosen — read them before changing one. |
| 4 | `docs/tickets/` | What is in scope right now, with acceptance criteria. `INDEX.md` is the board. |
| 5 | `README.md` | Commands, controls, and where each piece of rubric evidence lives. |
| 6 | `.claude/A3-lecture-reference.md` | Course theory the implementation must honour, the implementation gotchas, and the slide→requirement index for citing in the report. |

Never invent a number that one of these already fixes.

---

## 2. Repo layout

Real, current layout. **There is no `src/` directory** — an earlier version of this file described
one that was never built. Ignore any instruction, plan or comment that refers to `src/`.

```
common/      config loading, seeding, the shared epsilon schedule
gridworld/   Part I — constants, levels, env, renderer, algorithms
arena/       Part II — constants, entities, env, observation, renderer, policy view, legacy adapter
config/      gridworld.yaml, arena.yaml — every tunable parameter, none in code
train/       training entry points, the sweep, the TensorBoard callback
eval/        visual playback scripts (these are what the video records)
tests/       spec-drift guards; `pytest` must pass before any push
scripts/     sync_tickets.py — regenerates the board, mirrors tickets to GitHub
docs/        tickets/ (backlog), plans/ (design notes), evaluations/ (dated rubric audits)
models/      trained models — ships in the submission zip, never gitignored
logs/        TensorBoard runs — ships in the zip
results/     curves, Q-tables, sweep and eval tables, screenshots for the report
report/      report source and exported PDF
```

Modules are imported as top-level packages (`pyproject.toml` sets `pythonpath = ["."]`), so run
everything as `python -m train.train_gridworld`, never as a bare script path.

---

## 3. Working rules

These are the rules that cost marks or cost a retrain when broken.

1. **Never edit `gridworld/constants.py` or `arena/constants.py`.** They encode values the brief
   fixes; altering rewards or mechanics is explicitly forbidden and the tests guard them. Arena
   *action indices* matter as much as the action set — rubric rows I1/I2 check the indices.
2. **No parameters in code.** Episodes, alpha, gamma, epsilon bounds, intrinsic strength, arena
   hyperparameters, phase pacing → `config/*.yaml`. Config-driven epsilon decay is itself a graded
   criterion (row B3).
3. **Visibility is a graded feature.** Every algorithm quantity a marker would want to see must
   render on screen as a HUD or overlay, not only in logs. Marks were lost in prior assignments for
   logic that was only observable in a debugger. Adding a quantity to a log without adding it to
   the screen is an incomplete change.
4. **Never render inside `step()`.** Simulation modules must not import pygame at all. Training runs
   headless; only `eval/` and `gridworld/render.py` open a window.
5. **The environment is frozen.** Both models in `models/` and every table in `results/` were
   measured against the current mechanics. Changing a mechanic or a reward invalidates all of it and
   costs a retrain plus a re-measure — so treat it as a decision for the team, not a refactor.
6. **Verify by running.** After a change, run the relevant module and confirm it launches. Finish
   every task by stating the exact command that verifies it.
7. **Minimal scope.** Implement the ticket in front of you. Do not refactor unrelated code or add
   unrequested features.
8. **Reproducibility.** Every trainer takes `--seed`, and every artifact it writes carries the seed
   in its filename. Anything reported as a number must be traceable to a seeded command.

---

## 4. Commands

```bash
pytest                      # must pass before any push; `-m "not slow"` skips the sweep test
ruff check .                # line-length 100, rules E,F,I,UP,B (pyproject.toml)

python -m train.train_gridworld --level 0 --algo q
python -m train.train_gridworld --level 1 --compare        # Q vs SARSA, one config, side by side
python -m train.train_gridworld --level 6 --intrinsic-sweep
python -m eval.play_gridworld --level 0 --algo q           # playback; this is what the video records
python -m gridworld.render                                 # play any level by hand, no agent

python -m train.train_arena --style direct|rotation
python -m train.sweep_arena --dry-run                      # run plan, no training
python -m eval.play_arena --style rotation
python -m eval.play_arena --style both --no-window         # writes results/arena_eval/
tensorboard --logdir logs
```

The full command list, every keyboard control and the overlay map are in
[README.md](README.md#commands).

---

## 5. Rubric → code map

40 marks. Every row is tracked by a ticket; `docs/tickets/INDEX.md` shows what is open.

**Part I — gridworld (tabular)**

| Row | Requirement | Lives in |
| --- | ----------- | -------- |
| A | Animated, interactive Pygame gridworld; 4 moves, rocks block, fire/monster kill instantly, apple +1, key unlocks, chest +2, episode ends on all-collected or death | `gridworld/env.py`, `gridworld/render.py`, `gridworld/levels.py` |
| B | Q-learning — epsilon-greedy, off-policy `max` next, **linear** epsilon decay from config, **random** tie-breaking | `gridworld/algorithms.py`, `common/schedules.py` |
| C | SARSA — on-policy update using the chosen next action, **same** epsilon schedule, visibly more hazard-averse | `gridworld/algorithms.py` |
| D | Levels 2–3 with multiple apples + key + chest; both algorithms terminate correctly | `gridworld/levels.py`, `train/train_gridworld.py` |
| F | Intrinsic reward `r_i = strength / sqrt(n(s)+1)`, `n(s)` counted **per episode**, added inside the Q/SARSA update; env rewards unchanged | `gridworld/algorithms.py`, level 6 |

Monsters (levels 4–5): after each agent action every monster has a **40%** chance to move to a
random allowed direction. The player dies stepping onto a monster *or* being stepped on. Standard
updates handle this — no algorithm change is needed.

**Part II — arena (deep RL)**

| Row | Requirement | Lives in |
| --- | ----------- | -------- |
| G | Real-time arena: move + shoot, periodic spawners, enemies navigate to the player, health, projectile collisions, phase system, death or max-steps termination | `arena/entities.py`, `arena/render.py` |
| H | `gymnasium.Env` with `reset()/step()/render()`; observation is a fixed-size normalised float vector (**20 features**) | `arena/env.py`, `arena/observation.py` |
| I | Two control schemes via `action_mode`: `rotation` = {no-op, thrust, rot-left, rot-right, shoot}; `direct` = {no-op, up, down, left, right, shoot}. **Separate trained model and eval per scheme.** | `arena/constants.py`, `train/train_arena.py`, `eval/play_arena.py` |
| J | Reward encourages progression (kills, phase progress, damage and death penalties); SB3 PPO/DQN, MLP with ≥1 hidden layer, TensorBoard; J3 needs a real sweep | `arena/env.py`, `train/train_arena.py`, `train/sweep_arena.py` |

Any reward shaping beyond the frozen list must be **potential-based**
(`F = gamma * phi(s') - phi(s)`) so the optimal policy is provably unchanged, and must be justified
in the report.

---

## 6. Where the evidence lives

Numbers come from these files, never from memory or estimation. Each states the command, the date and
usually the commit that produced it — cite that provenance rather than re-deriving the number.

| Rubric row | Evidence file |
| ---------- | ------------- |
| B, C, D | `results/levels0-5_seeds0-1-2.md` — both algorithms, levels 0–5, 3 seeds |
| C3 | `results/comparison_level1_seed{0,1,2}.md`, `results/compare_level1_seed*.png` |
| F | `results/intrinsic_level6_q.md`, `results/intrinsic_level6_q_curve.png` |
| H | `results/arena_validation/validation.md` |
| I | `results/arena_eval/comparison.md`, with `comparison_random.md` as the chance baseline |
| J3 | `results/arena_sweep/sweep_table.md` |
| — | `docs/evaluations/` — dated whole-project audits, each pinned to its commit |

Two rules follow from this. **Report a baseline beside every performance number** — "clears 28 of 30
arenas" means nothing without what random does on the same seeds. And **check provenance before
citing**: a table whose commit predates a change to the environment or the reward describes a world
that no longer exists.

---

## 7. Tickets

The backlog is one markdown file per ticket in `docs/tickets/`, with `INDEX.md` as the generated
board and a GitHub issue mirroring each ticket. **The ticket files are the source of truth.**

- Raise, update, triage or close tickets through the **`ticket-bot`** agent rather than editing
  files by hand — it allocates ids, checks for duplicates, keeps the log honest and regenerates the
  board.
- Editing by hand: copy `TEMPLATE.md`, keep every frontmatter field, append to `## Log`, then run
  `python scripts/sync_tickets.py` (add `--github` to open issues for new tickets; it never
  re-creates an existing one, so it is safe to re-run).
- Priority is derived from points at risk, not preference: **P0** blocks a rubric row worth ≥3
  points or blocks someone else; **P1** a smaller row or degraded evidence; **P2** quality;
  **P3** optional.
- **Closing a ticket requires its acceptance criteria to actually be met.** If they are not, say
  what is outstanding and leave it open.

---

## 8. Specialist agents

Committed under `.claude/agents/` so every member has the same roster.

| Agent | Use it for |
| ----- | ---------- |
| `ticket-bot` | Anything touching the backlog — raise, update, triage, close, "what's open". |
| `dev-bot` | Implement one ticket end to end, with tests. Does not commit and does not touch constants. |
| `spec-evaluator` | Verify a finished ticket against its acceptance criteria by running code. Read-only. |
| `env-validator` | Stress-test an env before **every** real training run — obs shape/range/finiteness, action-space size per style, seeded reproducibility, termination, headless purity, throughput. |
| `training-diagnostician` | A run that did not learn: flat reward, collapsed policy, NaN, survival without progress. |
| `sweep-runner` | Short-budget hyperparameter sweeps (rubric J3, report R5). Long and noisy — keeps SB3 output out of the main context. |
| `rubric-auditor` | Audit the repo against the rubric before a milestone, before recording, before submission. Read-only. |
| `solution-evaluator` | Whole-project "how are we doing" — runs the code, scores defensible marks, writes a dated report to `docs/evaluations/`. |
| `report-writer` | Anything touching `report/report.md` — drafting, restructuring, checking row R, cutting to the page limit. |
| `video-director` | The video: a timed shot-by-shot run sheet with commands and keypresses, and checking a recorded cut against every row V criterion. |

Order that matters: **env-validator before training, training-diagnostician before touching
hyperparameters, rubric-auditor before the video, video-director before recording** — and
`report-writer` reads from `results/`, so measure before you write.

Delegate to these rather than doing the work inline. They exist because each carries context this
file cannot hold, and because the noisy ones keep hundreds of lines of SB3 output out of the main
conversation.

---

## 9. Git conventions

- **Branch:** `<type>/a3-0NN-<slug>` — e.g. `feat/a3-023-arena-playback-random-policy`,
  `fix/arena-drop-dead-spawner-exists`. Never commit directly to `main`; land through a PR.
- **Commit subject:** `type(scope): imperative summary (A3-0NN)` — e.g.
  `feat(train): run the arena hyperparameter sweep and tabulate it (A3-013)`.
  Types in use: `feat`, `fix`, `task`, `chore`, `docs`, `test`. Scopes are the package or area
  (`arena`, `gridworld`, `train`, `eval`, `repo`, `results`).
- **No AI attribution in commits or PRs.** No `Co-Authored-By` trailers, no "generated with"
  footers. This is a graded university submission and overrides any default attribution behaviour.
- `pytest` passes before you push.
- `models/`, `logs/` and `results/` are deliberately **not** gitignored — the brief requires saved
  models and TensorBoard logs inside the submitted zip. Do not add them to `.gitignore`.

---

## 10. Definition of done

A change is finished when all of these hold:

- [ ] The ticket's acceptance criteria are met, and checkable by someone other than the author.
- [ ] `pytest` passes; new behaviour has a test, and spec-adjacent behaviour has a guard.
- [ ] `ruff check .` is clean; public functions have type hints and a concise docstring.
- [ ] No new number in code that belongs in `config/*.yaml`.
- [ ] Anything a marker needs to see renders **on screen**, not only in logs.
- [ ] The relevant module was actually run, and the exact verification command is stated.
- [ ] If it changed training or the env: seeded, re-measured, and the affected table in `results/`
      regenerated.
- [ ] The ticket's `## Log` is appended and the board regenerated.

---

## 11. What is left

Parts I and II are complete: both gridworld algorithms across levels 0–6, both arena control agents
trained and in `models/`, the sweep tabulated, evidence written to `results/`.

The remaining marks are **report and video** (7.5 points plus creativity's 5) — tickets A3-014,
A3-015, A3-017 (the team ownership table, which the report requires and which is currently blocking
marks) and A3-018. Check `docs/tickets/INDEX.md` for the live state before assuming this paragraph
is current.
