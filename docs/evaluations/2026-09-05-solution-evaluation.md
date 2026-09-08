# Solution evaluation — `main` @ `d04e6b1`

**Date:** 2026-09-05
**Evaluated ref:** `main` = `origin/main` = `d04e6b17ba49fe1823a446ece58a319f5ca4d450`
("Merge pull request #41 from TDuong04/feat/a3-012-agents-on-main")
**Interpreter:** `.venv/bin/python` (macOS; `.venv/Scripts/python.exe` does not exist), Python 3.14.6
**Method note:** during this evaluation the working tree was switched out from under the run to
`feat/a3-013-hyperparameter-sweep` (`70b1740`) with 190 dirty paths. Every measurement below was
therefore re-taken against a pristine export of `main`
(`git archive main | tar -x -C <scratch>/main_tree`), never against the live checkout, and never by
checking out a branch. The only file written into the repo is this report.

---

## Verdict

**26.5 / 40 currently defensible. Not submission-ready: 13.5 points (R, V, creativity, part of J)
have no artifact on `main` at all.**

Everything that has been built is in unusually good shape — 640/640 tests pass, every brief-fixed
mechanic verified by executing transitions rather than reading constants, and every Part I result
reproduces bit-for-bit from its stated seed. The gap is entirely in deliverables that were never
produced: no report, no video, no hyperparameter-tuning evidence on `main`, and no packaged
creativity claim.

---

## Rubric table (ordered by points at risk)

| Row | Pts | Status | Defensible | Evidence | Gap |
|---|---|---|---|---|---|
| **V** video | 5 | **MISSING** | 0 | Playback tooling exists and runs: `eval/play_gridworld.py:1`, `eval/play_arena.py:1`; phase progression is reachable on camera (22/30 episodes, direct) | No recording anywhere in the repo. No team members named (`README.md:16-20` team table is `_TBD_`). A3-015 open. |
| **Creativity** | 5 | **PARTIAL** | 2.0 | Observation overlay `arena/render.py:16-21`; human-play mode in both envs (`eval/play_arena.py:18`, `eval/play_gridworld.py` `WASD` control); Q-value heatmap + policy arrows `gridworld/render.py:434,465`; side-by-side Q vs SARSA playback `eval/play_gridworld.py --compare`; BFS optimality oracle `gridworld/algorithms.py:617`; honest negative-result intrinsic study with Welch intervals `results/intrinsic_level6_q.md:60-95` | Nothing is *claimed* as creativity anywhere (no report), and A3-018 is still open. Judged live, this is worth arguing for; on disk today it is unpackaged. |
| **R** report | 2.5 | **MISSING** | 0 | — | `report/` contains only `.gitkeep`. No student numbers, no contribution summary, no video link. Part II has **no figures at all** in `results/` (only raw TB event files), so R5 (hyperparameter tables/plots) and R6 (control-set comparison) have no plot to cite. A3-014 open. |
| **J** deep RL + tuning | 3 | **PARTIAL** | 2.0 | Rewards: every step decomposes exactly into the five constants + step penalty (probe below); `arena/constants.py:51-56`. PPO `MlpPolicy`, `net_arch [64,64]`, two `Tanh` hidden layers confirmed by loading `models/ppo_direct.zip`. TensorBoard: `logs/ppo_direct_full/events.out.tfevents.1788610181...` and `logs/ppo_rotation_full/...`, 19 scalar tags each, steps 16384→409600, including `behaviour/phase_reached`, `behaviour/survival_rate`. | "Hyperparameters tuned beyond defaults **with evidence**" is unmet on `main`: both models were trained on the `config/arena.yaml` baseline (verified from the zips: `lr 3e-4, n_steps 2048, batch 512, gamma 0.99, ent_coef 0.005, seed 0`), and the sweep lives only on the unmerged `feat/a3-013-hyperparameter-sweep`. No `run.json` or `monitor/*.csv` on `main` to tie either model to a command. ~1 pt at risk. |
| **G** arena | 4.5 | **SATISFIED** | 4.5 | Real-time float physics at 60 Hz, `ACTION_REPEAT=3` (`arena/constants.py:61-64`); continuous player position `(480.0, 340.0)`, not a tile grid. Measured: spawners emit on the 3.0 s config cadence (first spawns at t = 3.0, 6.0 s), enemies home in (nearest enemy 0.1 px → contact), player health 5 → 0 by contact damage, bullets kill enemies and spawners, phase advances on last spawner (`arena/env.py:415`), episode ends on death or `MAX_EPISODE_STEPS` (`arena/env.py:198-201`). HUD + health bars + banner in `arena/render.py`. | — |
| **I** two control styles | 4 | **SATISFIED** | 4.0 | `Discrete(5)` = `NOOP, THRUST, ROTATE_LEFT, ROTATE_RIGHT, SHOOT`; `Discrete(6)` = `NOOP, UP, DOWN, LEFT, RIGHT, SHOOT`, indices verified by execution. `models/ppo_direct.zip`, `models/ppo_rotation.zip` present under the exact required name, plus 8 checkpoints each. `eval/play_arena.py` visually runs either (`--style direct|rotation|both`). Both beat random decisively (table below). | Rotation agent is weak in absolute terms (mean return −0.53 over 30 episodes). Defensible, but see finding M3. |
| **C** SARSA | 3 | **SATISFIED** | 3.0 | On-policy target with no `max` in the update (`gridworld/algorithms.py:322-323`); the shared schedule is the same function object as Q-learning's (`gridworld/algorithms.py:106`). Measured contrast: Q 11 steps via rows [7,8] vs SARSA 13 steps via rows [6,7,8]; death rate at the ε=0.05 it trained under: **11.4% vs 1.8%** over 500 rollouts each. `results/compare_level1_seed0.json`, `results/comparison_level1_seed0.md`, `results/compare_level1_seed0.png`. Holds on seed 1 too (11 vs 13). | — |
| **D** levels 2–3 | 3 | **SATISFIED** | 3.0 | `gridworld/levels.py:46,60`: multiple apples + key + chest. Both algorithms, 3 seeds each, all solved: level 2 = 25/25 steps optimal on all 6 runs; level 3 = 34/34 on 5 of 6 (q seed 1 = 36). Collection order recorded with `key before chest = yes` in every row (`results/levels2-5_seeds0-1-2.md`). Independently re-measured: 100% success, 0% death, 0% truncation over 30 greedy rollouts per table. | — |
| **F** intrinsic reward | 3 | **SATISFIED** | 3.0 | Formula verbatim `gridworld/algorithms.py:351-361`; per-episode counter cleared at reset `gridworld/algorithms.py:388,466`; env untouched — `r_i` only enters a local `shaped` variable and the logged return is env reward alone (`gridworld/algorithms.py:478-482`). Curves with vs without + written explanation: `results/intrinsic_level6_q_curve.png`, `results/intrinsic_level6_q.md`, 5 strengths × 5 seeds. | Interpretation risk only: `n(s)` is keyed on the state *arrived in* (`s'`), documented and justified at `gridworld/algorithms.py:441-455`. A strict reading of the brief could want the state being left. ≤0.5 pt. |
| **H** Gym API | 2.5 | **SATISFIED** | 2.5 | Gymnasium-native `ArenaEnv` passes `gymnasium.utils.env_checker.check_env` for both styles. `arena/legacy_api.py:31-47` gives the brief's literal `reset() -> obs`, `step(a) -> (obs, reward, done, info)`, `render()`. Observation = `Box(-1, 1, (21,), float32)`, never pixels; contains player pos, velocity, heading sin/cos, health, phase, nearest-enemy and nearest-spawner relative vectors and distances (`arena/observation.py:9-25`). Measured over 20 000 random steps: every feature stayed inside [−1, 1]. | — |
| **B** Q-learning | 2.5 | **SATISFIED** | 2.5 | Off-policy `max` target `gridworld/algorithms.py:229`; linear ε from config `common/schedules.py:31-38` + `config/gridworld.yaml:8-10`; random tie-break `gridworld/algorithms.py:130-131`; no executable `argmax` anywhere (grep: only docstrings and tests). Level 0 greedy = **17 steps = BFS optimum 17**, 3/3 collected, 100% over 30 rollouts. | — |
| **A** gridworld | 2 | **SATISFIED** | 2.0 | Pygame renderer with HUD, legend, arrows, heatmap (`gridworld/render.py`), interactive playback (pause/step/speed/level-switch/human play) `eval/play_gridworld.py`. Mechanics verified by executing transitions, not by reading `constants.py` (see below). | — |
| | **40** | | **26.5** | | |

---

## Brief compliance, verified by executing transitions

Probe: `/private/tmp/.../scratchpad/probe_grid.py`, `probe_grid2.py`, `probe_arena.py` (run against
the exported `main` tree).

| Brief rule | Result |
|---|---|
| Exactly 4 gridworld actions | `step(4)` → `ValueError: action must be in 0..3, got 4` |
| Rocks block: no displacement, no reward change | pos `(0,4)` → `(0,4)`, `r=0.0`, `done=False` |
| Fire kills instantly on entry | `died=True, done=True, r=0.0` on step 1 of level 1 |
| Apple `+1` | cumulative `1.0` on reaching `(2,8)`, `collected=1` |
| Key `0` but unlocks | `r_cum=0.0`, `has_key=True` |
| Chest inert without key | reached `(5,8)` without key: `r_cum=0.0`, `collected=0`; re-entering pays `0.0` |
| Chest `+2` with key | `r=2.0` on the arrival step, `collected` 2 → 3 |
| Episode ends when all collectibles taken | level 0: 17 steps, `terminated=True`, `collected=3`; `step()` afterwards raises `RuntimeError: step() called after the episode ended; call reset() first` |
| Monster move probability `0.4` | measured **16018 / 40000 = 0.4004** over 20 000 agent steps |
| Monster moving onto a stationary player kills | 315/3000 = **0.1050**, matching `0.4 × 1/4` for a 4-way-legal monster |
| Style 1 = `Discrete(5)`, exact indices | `['NOOP','THRUST','ROTATE_LEFT','ROTATE_RIGHT','SHOOT']` |
| Style 2 = `Discrete(6)`, exact indices | `['NOOP','UP','DOWN','LEFT','RIGHT','SHOOT']` |
| Observation 10–30 normalized floats, never pixels | `Box(-1.0, 1.0, (21,), float32)`, 1-D, all features in range over 20 000 steps |
| `total = env_reward + r_i`, env rewards untouched | `env.step()` return is unmodified; `r_i` is added only to the local TD target (`gridworld/algorithms.py:478-482`) |
| Arena reward decomposition | every step of 5 model-driven episodes satisfied `r == step_penalty + Σ(constants × events)` exactly; events seen: 94 enemy kills, 20 spawner kills, 8 phase advances, 23 damage, 4 deaths |

No brief-fixed mechanic has drifted.

---

## Test suite

```
SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest -p no:cacheprovider
640 passed in 19.00s
```

(run inside the exported `main` tree; the live checkout at `70b1740` also reports `640 passed`).

Green is necessary, not sufficient — but the constant guards do assert the brief's literals rather
than re-deriving them from the code: `tests/test_spec_constants.py:15-20` (`REWARD_APPLE == 1.0`,
`REWARD_KEY == 0.0`, `REWARD_CHEST == 2.0`, `MONSTER_MOVE_PROBABILITY == 0.4`),
`tests/test_spec_constants.py:38,43` (exact action index tuples),
`tests/test_spec_constants.py:58` (`10 <= OBS_DIM <= 30`).
`tests/test_gridworld_algorithms.py:307` parses the AST to prove `argmax` is never called.

---

## Measured performance — Part I

Greedy rollouts from each committed `.npz`, 30 rollouts per deterministic table and **300** per
stochastic one (levels 4–5), env seeds 1000+k, policy seeds 2000+k. `claim` is `greedy_steps` from
the committed `summary_*.json`.

| Artifact | claim steps | measured median | success % | death % | trunc % | mean return | BFS |
|---|---|---|---|---|---|---|---|
| L0 q s0 | 17 | 17 | 100.0 | 0.0 | 0.0 | 3.00 | 17 (**optimal**) |
| L1 q s0 | 11 | 11 | 100.0 | 0.0 | 0.0 | 1.00 | 11 (**optimal**) |
| L1 q s1 | 11 | 11 | 100.0 | 0.0 | 0.0 | 1.00 | 11 |
| L1 sarsa s0 | 13 | 13 | 100.0 | 0.0 | 0.0 | 1.00 | 11 (**safer, longer — row C evidence holds**) |
| L1 sarsa s1 | 13 | 13 | 100.0 | 0.0 | 0.0 | 1.00 | 11 |
| L2 q s0/s1/s2 | 25 | 25 | 100.0 | 0.0 | 0.0 | 5.00 | 25 |
| L2 sarsa s0/s1/s2 | 25 | 25 | 100.0 | 0.0 | 0.0 | 5.00 | 25 |
| L3 q s0 / s1 / s2 | 34 / 36 / 34 | 34 / 36 / 34 | 100.0 | 0.0 | 0.0 | 4.00 | 34 |
| L3 sarsa s0/s1/s2 | 34 | 34 | 100.0 | 0.0 | 0.0 | 4.00 | 34 |
| L4 q s0 / s1 / s2 | 22 | 22 | 77.7 / 79.0 / 77.7 | 22.3 / 21.0 / 22.3 | 0.0 | 2.61–2.63 | 21 |
| L4 sarsa s0/s1/s2 | 22 | 22 | 76.0–80.0 | 20.0–24.0 | 0.0 | 2.57–2.65 | 21 |
| L5 q s0 / s1 / s2 | **19** / 30 / 30 | 28 / 30 / 30 | 52.3 / 53.0 / 54.7 | 47.7 / 47.0 / 45.3 | 0.0 | 3.03–3.06 | 28 |
| L5 sarsa s0 / s1 / s2 | 30 / **23** / 30 | 30 / 30 / 30 | 51.0 / 53.3 / 54.3 | 49.0 / 46.7 / 45.7 | 0.0 | 3.01–3.05 | 28 |

Level 0 is optimal, not merely solved. Level 1 shows the required Q-vs-SARSA split. Truncation is
0% everywhere. Deaths on 4–5 are the monsters, not a broken policy (mean return 2.6/3 and 3.0/4).

**Reproducibility (spot checks, exact):**

- `python -m train.train_gridworld --level 0 --algo q --seed 0` → identical Q-table (`states` and
  `values` bitwise equal) and identical summary values; 1.5 s.
- `python -m train.train_gridworld --level 1 --compare --seed 0` → `q 11 steps rows [7,8]`,
  `sarsa 13 steps rows [6,7,8]`, `11.4%` vs `1.8%` death — comparison JSON identical to the
  committed one; 2.5 s.
- `python -m train.train_gridworld --level 4 --algo q --seed 0` (a *stochastic* level, 12 000
  episodes) → every summary field identical, Q-table identical; 3.3 s.

Full Part I regeneration is on the order of 3–5 minutes of CPU. There is no schedule risk here.

---

## Measured performance — Part II

30 episodes per row, seeds 0–29, same arenas for every policy, `deterministic=True`, using the
project's own `eval.play_arena.evaluate()` harness with the policy swapped.

| Style | Policy | mean return | sd | phase mean | phase best | phase cleared | spawners | kills | steps | survival |
|---|---|---|---|---|---|---|---|---|---|---|
| direct | **PPO (models/ppo_direct.zip)** | **+17.14** | 19.00 | 1.93 | 4 | 22/30 | 2.60 | 15.90 | 1011.1 | 13.3% |
| direct | random baseline | −13.15 | 0.82 | 1.00 | 1 | 0/30 | 0.00 | 1.67 | 232.1 | 0.0% |
| direct | no-op baseline | −14.18 | 0.12 | 1.00 | 1 | 0/30 | 0.00 | 0.00 | 168.4 | 0.0% |
| rotation | **PPO (models/ppo_rotation.zip)** | **−0.53** | 10.73 | 1.47 | 2 | 14/30 | 1.30 | 6.77 | 745.9 | 13.3% |
| rotation | random baseline | −14.30 | 1.46 | 1.00 | 1 | 0/30 | 0.07 | 0.17 | 230.0 | 0.0% |
| rotation | no-op baseline | −14.18 | 0.12 | 1.00 | 1 | 0/30 | 0.00 | 0.00 | 168.4 | 0.0% |

Both models clearly beat random on every behavioural metric, so both have learned. `direct` is a
good demo; `rotation` still averages a **negative** return and clears a phase in fewer than half of
episodes.

Reproduce: `SDL_VIDEODRIVER=dummy .venv/bin/python -m eval.play_arena --style both --episodes 20
--no-window` → direct `+16.25 ± 11.28`, 15/20 cleared; rotation `+0.22 ± 11.37`, 10/20 cleared.

**Throughput and schedule.** Single-env headless: **59 230 agent steps/s** (direct), **71 297**
(rotation), each step = 3 physics frames. Committed TB logs show `time/fps` 2401→2877 (direct) and
3932→2644 (rotation) at `n_envs=8`, so a full 400 000-step run costs **~2.5 minutes wall clock**.
Retraining both agents, or running the entire 14-run × 100 k sweep, is well under an hour. All
remaining schedule risk is report and video, not compute.

---

## Findings by severity

### HIGH

**H1 — 7.5 points (R + V) have no artifact of any kind on `main`.**
`report/` contains only `.gitkeep`; there is no video, and `README.md:16-20` still lists the team as
`_TBD_ | _TBD_`. The brief requires student numbers, a contribution summary and a video link inside
the report. Tickets A3-014, A3-015 and A3-017 are open and correctly so.
Repro: `git ls-tree -r --name-only main report` → `report/.gitkeep`.

**H2 — `.gitignore` contradicts itself and ignores every artifact directory.**
`.gitignore:27-29` states "models/ logs/ and results/ are deliberately NOT ignored… Do not add them
here", and `.gitignore:32-35` then adds `results/`, `docs/tickets/`, `logs/`, `models/`. The
existing artifacts survive only because they were force-added earlier; **any new curve, model, log
or ticket will be silently skipped by `git add`**. This is exactly how the Part II figures the
report needs get lost between now and submission.
Repro: `git check-ignore -v results/ logs/ models/`.

**H3 — J3 (tuning with evidence) is unmet on `main`; ~1 point at risk.**
Both shipped models were trained on the untuned baseline — read back from the zips:
`lr 3e-4, n_steps 2048, batch_size 512, gamma 0.99, ent_coef 0.005, net_arch [64,64], seed 0,
num_timesteps 409600`. Of those, only `batch_size` and `ent_coef` differ from SB3's PPO defaults,
and there is no comparison on `main` to justify them. `train/sweep_arena.py`, `tests/test_sweep_arena.py`
and the sweep logs exist **only on `feat/a3-013-hyperparameter-sweep`** (`70b1740`), unmerged.
Repro: `git diff --stat main 70b1740 -- train tests config`.

### MEDIUM

**M1 — The two shipped models have no provenance on `main`.**
`train/train_arena.py:262-276` writes `<run>/run.json`, `<run>/monitor/*.csv` and `<run>/tb_1/`,
but `main` tracks only two bare event files placed directly in `logs/ppo_direct_full/` and
`logs/ppo_rotation_full/` — no `run.json`, no monitor CSVs, no `tb_1/` layer. The logs cannot be
tied to a command, a config or a commit, which weakens both J and R5.
Repro: `git ls-tree -r --name-only main logs` → 3 paths.

**M2 — Part II has zero report-ready figures.**
`results/` holds 210 Part I artifacts and **no** arena artifact (`ls results | grep -iE 'arena|ppo|phase'`
→ nothing). R5 wants hyperparameter tables/plots and R6 wants a control-set comparison; today the
only Part II evidence is raw `tfevents` and a table printed to stdout by `eval/play_arena.py`, which
writes nothing to disk.

**M3 — The rotation agent's deterministic return is negative.**
−0.53 ± 10.73 over 30 seeds, 1.30 spawners destroyed, phase cleared in 14/30. It beats random
(−14.30) so row I stands, but the video rubric wants both control schemes shown clearing content,
and roughly half the rotation takes will not clear a phase. Its own TB curve ends at
`rollout/ep_rew_mean = 6.525` (stochastic, 8 envs) versus −0.53 measured deterministically — quote
the measured number in the report, not the curve.
Repro: `SDL_VIDEODRIVER=dummy .venv/bin/python -m eval.play_arena --style rotation --episodes 30 --no-window`.

**M4 — Board drift: A3-011 is marked open but is merged.**
`docs/tickets/A3-011-*.md:6` has `status: open` and `docs/tickets/INDEX.md:13` lists it under Open,
yet `train/train_arena.py`, `train/callbacks.py`, `tests/test_train_arena.py` and the TensorBoard
logs it asks for all landed on `main` in PR #40 (`4806832`) and demonstrably work (smoke run below).
A3-009 is *not* mis-stated — it is correctly `done`. A3-013 is correctly open (unmerged).
The board also claims "**6 open · 14 done**" while one of those six is complete.

**M5 — Two committed level-5 summaries quote an unrepresentative single rollout.**
`results/summary_level5_q_seed0.json` claims `greedy_steps: 19` and
`results/summary_level5_sarsa_seed1.json` claims `23`, while the same tables measure a median of
**30** steps over 300 rollouts. On a stochastic level one greedy rollout is a sample; quoting 19 in
the report would overstate the policy.

### LOW

**L1 — `results/summary_level0_q_seed0.json` is stale relative to the code that would write it
today.** It lacks the `solved`, `collection_order`, `collection_order_text` and `key_precedes_chest`
fields the current trainer emits (all shared fields are identical). Regenerating costs 1.5 s.

**L2 — `n(s)` keyed on the arrival state.** `gridworld/algorithms.py:441-455` scores `s'` rather
than `s`, with the alternative measured and rejected in the docstring. Defensible and documented,
but a strict marker reading of "visits to the current state" could differ. ≤0.5 pt.

**L3 — README/venv platform mismatch.** `README.md:31` gives PowerShell setup and "verified on
3.14.4, Windows"; this machine runs 3.14.6 on macOS with `.venv/bin/python`. Everything works, but
the documented commands are Windows-only in a repo that is currently being developed on macOS.

### Entry points confirmed working (all under `SDL_VIDEODRIVER=dummy`)

| Command | Result |
|---|---|
| `python -m train.train_gridworld --level 0 --algo q` | `verdict: OPTIMAL`, 1.5 s |
| `python -m train.train_gridworld --level 1 --compare` | `verdict: ROUTES DIFFER`, 2.5 s |
| `python -m train.train_gridworld --levels 2 3 --seeds 0 --episodes 500` | both solved, markdown + JSON written |
| `python -m train.train_gridworld --level 6 --intrinsic-sweep --episodes 300 --intrinsic-seeds 0 1` | curve + markdown + JSON written, 4.2 s |
| `python -m train.train_arena --style direct --timesteps 4096 --n-envs 2` | model saved in 0.6 s, TB dir created |
| `python -m eval.play_gridworld --level 0 --algo q --frames 60` | window opened, exit 0 |
| `python -m eval.play_gridworld --level 4 --algo sarsa --frames 60` | exit 0 |
| `python -m eval.play_gridworld --level 1 --compare --frames 60` | exit 0 |
| `python -m eval.play_arena --style rotation --episodes 1` | ran, phase cleared, summary printed |
| `python -m eval.play_arena --style both --episodes 20 --no-window` | comparison table printed |
| `tensorboard --logdir logs` | 2 event dirs, 19 scalar tags each, 400 k steps |

---

## Highest-value next actions, ranked by points per effort

1. **Fix `.gitignore` (2 minutes, protects everything below).** Delete lines 32, 34, 35 (`results/`,
   `logs/`, `models/`) — and decide deliberately about `docs/tickets/`. Until this is done every new
   artifact has to be `git add -f`-ed and one forgotten flag loses a figure.
2. **Fill in the team table in `README.md:16-20` (10 minutes, unblocks 2.5 pts).** Student numbers
   and contribution split are hard requirements of row R and cannot be produced by anyone else at
   the last minute. Closes A3-017.
3. **Merge the sweep branch, or re-run it on `main` (≈1 hour of wall clock, ~1 pt + R5's plots).**
   `feat/a3-013-hyperparameter-sweep` already contains `train/sweep_arena.py` and 100 k-step logs;
   at 2 700 fps the full grid is well under an hour. This is the only outstanding *compute* work.
4. **Write a Part II results dump to `results/` (1 hour, unblocks R5/R6).** `eval.play_arena.evaluate()`
   already returns everything R6 needs; it just never writes a JSON, a curve or a comparison figure.
   Mirror what `train_gridworld` does for Part I.
5. **Write the report (5 pts of dependency: R 2.5 + it is where creativity gets claimed).** Every
   number it needs for Part I exists and reproduces; Part II needs items 3 and 4 first.
6. **Record the video (5 pts).** Tooling is ready today. Record `direct` for the arena; for
   `rotation`, pre-select seeds that clear a phase (14/30 do) rather than rolling live.
7. **Retrain or tune `rotation` (optional, ~3 min per attempt).** Lifting it above a zero mean return
   would strengthen I, J and the video's control-scheme comparison.
8. **Housekeeping (5 minutes).** Close A3-011 on the board; regenerate
   `results/summary_level0_q_seed0.json`; stop quoting `greedy_steps` on levels 4–5 without the
   distribution beside it.
