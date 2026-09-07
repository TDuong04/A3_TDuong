# Solution evaluation — `main` @ `66316d1`

**Date:** 2026-09-06
**Evaluated ref:** `main` = `origin/main` = `66316d1f4a8f8236eb760380ac192c7593eccd66`
("Merge pull request #42 from TDuong04/feat/a3-013-hyperparameter-sweep")
**Interpreter:** `.venv/bin/python` (macOS; `.venv/Scripts/python.exe` does not exist), Python 3.14.6
**Working tree:** clean apart from untracked `docs/evaluations/`. Every measurement was taken
against the live checkout at `66316d1`. The only file written into the repo is this report; all
probes, screenshots and reproduction runs went to the session scratchpad via `--results-dir`,
`--log-dir` and `--models-dir` overrides.
**Prior report:** `docs/evaluations/2026-09-05-solution-evaluation.md` (`main` @ `d04e6b1`).

---

## Verdict

**29.5 / 40 currently defensible. Not submission-ready: 10.5 points (R, V and most of creativity)
still have no artifact of any kind, and `report/` is an empty directory 13 days before the due
date.**

Everything that has been built remains in unusually good shape — 658 tests pass, every brief-fixed
mechanic verified by executing transitions, every Part I result reproduces bit-for-bit from its
stated seed, and both arena agents beat random decisively. Since yesterday the `.gitignore` hazard
is closed and the hyperparameter sweep has landed on `main` with full provenance. Nothing regressed.

**Delta correction:** the prior report's headline of "26.5 / 40" does not match its own rubric
table, which sums to **28.5**. The like-for-like movement in the last day is therefore **+1.0**
(J 2.0 → 2.5, creativity 2.0 → 2.5), not +3.0. See finding **M6**.

| | Available | Defensible | Prior (recomputed) |
|---|---:|---:|---:|
| Part I (A, B, C, D, F) | 13.5 | **13.5** | 13.5 |
| Part II (G, H, I, J) | 14.0 | **13.5** | 13.0 |
| Cross-cutting (R, V, creativity) | 12.5 | **2.5** | 2.0 |
| **Total** | **40.0** | **29.5** | **28.5** |

---

## Rubric table (ordered by points at risk)

| Row | Pts | Status | Defensible | Evidence | Gap |
|---|---|---|---|---|---|
| **V** video | 5 | **MISSING** | 0 | Tooling runs today: `eval/play_gridworld.py:1`, `eval/play_arena.py:1`; phase progression reachable on camera (direct clears in 22/30 seeded episodes, rotation in 14/30) | No recording anywhere in the repo. `README.md:17-19` still lists the team as `_TBD_ \| _TBD_`. A3-015 and A3-017 open. **Unchanged from 2026-09-05.** |
| **Creativity** | 5 | **PARTIAL** | 2.5 | Observation overlay `arena/render.py:349,379` (screenshot below); human-play mode both envs (`eval/play_arena.py:168`, `eval/play_gridworld.py` WASD); Q-value heatmap with per-cell numerals + policy arrows `gridworld/render.py:434,465`; side-by-side Q-vs-SARSA playback `eval/play_gridworld.py --compare`; BFS optimality oracle `gridworld/algorithms.py:625`; honest negative-result intrinsic study `results/intrinsic_level6_q.md`; **new:** config-driven reward-hacking detector `train/sweep_arena.py` + `tests/test_sweep_arena.py:152,166` | Nothing is *claimed* as creativity anywhere — no report. A3-018 open, and its branch has diverged badly (finding **H2**). +0.5 vs yesterday for the reward-hacking detector. |
| **R** report | 2.5 | **MISSING** | 0 | — | `git ls-tree -r --name-only main report` → `report/.gitkeep`. No student numbers, no contribution summary, no video link. Part II still has **no figure** in `results/` — only `results/arena_sweep/sweep_table.md` and `sweep_runs.json`, so R6 (control-set comparison) has no plot to cite. A3-014 open. **Unchanged.** |
| **G** arena | 4.5 | **SATISFIED** | 4.5 | Verified by execution at seed 11: phase 1 opens with 2 spawners / 0 enemies / health 5 (matches `config/arena.yaml:40-44`); first enemy at agent step 59 = **2.95 s** against the configured `spawn_interval: 3.0`; nearest-enemy distance 411 → 0 px (enemies home); player health 5 → 0 by contact, `terminated=True` at step 193; spawner destroyed at step 407 from health 3; phase sequence `[1,2,3,4]` with spawner counts `2,3,4` at seed 24, ending `truncated=True` at `steps=2000` = `MAX_EPISODE_STEPS`. Continuous float positions `(480.665, 340.0) → (482.322, 340.0) → …`, not a tile grid. `ACTION_REPEAT=3` at 60 Hz (`arena/constants.py:61-64`) | — |
| **I** two control styles | 4 | **SATISFIED** | 4.0 | `Discrete(5)` = `['NOOP','THRUST','ROTATE_LEFT','ROTATE_RIGHT','SHOOT']`; `Discrete(6)` = `['NOOP','UP','DOWN','LEFT','RIGHT','SHOOT']`, indices read back by execution; out-of-range rejected (`ValueError: action 5 is outside Discrete(5) for control style 'rotation'`). `models/ppo_direct.zip` and `models/ppo_rotation.zip` under the exact required folder name. `eval/play_arena.py --style direct\|rotation\|both` runs either, windowed or headless. Both beat random on every metric (table below) | Rotation still averages **−0.53** deterministically. Defensible; see **M3**. |
| **C** SARSA | 3 | **SATISFIED** | 3.0 | AST scan of `gridworld/algorithms.py`: `sarsa` (line 268) contains **zero** `max`/`argmax` references; `q_learning` (line 202) has one at line 241. Same schedule by construction `gridworld/algorithms.py:110`; epsilon sequences read from the two committed history CSVs are **identical element-for-element** over all 8000 episodes. Measured contrast: Q 11 steps via rows [7,8] vs SARSA 13 steps via rows [6,7,8]; death rate at the ε=0.05 it trained under **11.4% vs 1.8%** over 500 rollouts. Figure `results/compare_level1_seed0.png` is publication-ready | Only seeds 0 and 1 have committed tables; `config/gridworld.yaml:31` claims the split also holds on seed 2 but no artifact proves it. |
| **D** levels 2–3 | 3 | **SATISFIED** | 3.0 | `gridworld/levels.py:46,60` multiple apples + key + chest. Re-measured over 30 greedy rollouts per table: level 2 = 25/25 steps optimal on all 6 runs, 100% success, 0% death, 0% truncation; level 3 = 34/34 on 5 of 6 (q seed 1 = 36). `key before chest = yes` in every row of `results/levels2-5_seeds0-1-2.md` | — |
| **F** intrinsic reward | 3 | **SATISFIED** | 3.0 | Formula exact to 1e-12 for strengths 0.5/1.0 at n=0..4 (`gridworld/algorithms.py:359`); `n(s)` sequence within an episode `[0,1,2,3]`, after `reset()` `[0,1]`; env untouched — the only rewards `env.step()` returned during an intrinsic run were `[0.0]`/`{0.0,1.0,2.0}`; at strength 0.0 the Q-table is **bitwise identical** to plain `q_learning` on the same seed. Curves + written explanation, 5 strengths × 5 seeds, `results/intrinsic_level6_q.md`, `results/intrinsic_level6_q_curve.png` | Interpretation risk only: `n(s)` keyed on the arrival state `s'`, documented and measured at `gridworld/algorithms.py:441-455`. ≤0.5 pt. |
| **H** Gym API | 2.5 | **SATISFIED** | 2.5 | `gymnasium.utils.env_checker.check_env` **PASS** for both styles. `Box(-1.0, 1.0, (21,), float32)`, 1-D, never pixels; over 20 000 random steps the observation range was exactly `[-1.0000, 1.0000]` with **0** out-of-space samples. `arena/legacy_api.py:31-47` gives the brief's literal `reset() -> obs` / `step(a) -> (obs, reward, done, info)` — verified: `reset() -> ndarray (21,)`, `step() -> tuple len 4 ['ndarray','float','bool','dict']` | `render_mode='rgb_array'` is rejected (`ValueError: unknown render_mode 'rgb_array'; expected one of ['human'] or None`), so there is no programmatic frame dump for the video. Cosmetic. |
| **B** Q-learning | 2.5 | **SATISFIED** | 2.5 | Off-policy `max` target `gridworld/algorithms.py:241`; linear ε from config — `max \|ε − linear\|` over the 6000-episode decay window is **0.0**, ε[0]=1.0, ε[6000]=0.05 (`common/schedules.py:31-38`, `config/gridworld.yaml:8-10`); random tie-break `gridworld/algorithms.py:130-131`; no executable `argmax` anywhere (AST, `tests/test_gridworld_algorithms.py:307`). Level 0 greedy = **17 steps = BFS optimum 17**, 3/3 collected, 100% over 30 rollouts. ε plotted on every training curve (`results/curve_level0_q_seed0.png`) | — |
| **J** deep RL + tuning | 3 | **PARTIAL** | 2.5 | Reward decomposition exact on **9155/9155** PPO-driven steps: `r == −0.01 + 1.0·enemies + 5.0·spawners + 10.0·phases − 0.5·damage − 10.0·death` (`arena/constants.py:51-56`); events seen: 96 enemy kills, 31 spawner kills, 10 phase advances, 23 damage, 3 deaths. PPO `MlpPolicy`, `Linear(21,64)-Tanh-Linear(64,64)-Tanh` read back from all three zips. TensorBoard: `logs/ppo_direct_full/`, `logs/ppo_rotation_full/` 19 scalar tags each 16384→409600, plus **14 sweep runs** each with `run.json`, 8 monitor CSVs, `train.log` and an 18-tag `tb_1/`. Sweep table `results/arena_sweep/sweep_table.md` | The tuning found a **regression** and the report-facing artifact does not say so — see **H1**. Both shipped models remain the untuned baseline with **no `run.json` or monitor CSV on `main`** — see **M1**. 0.5 pt at risk. |
| **A** gridworld | 2 | **SATISFIED** | 2.0 | Mechanics verified by executing transitions, not by reading `constants.py` (table below). Pygame renderer with HUD, legend, animated slides, per-cell Q numerals and greedy arrows (`gridworld/render.py:556,591,434,465`); interactive playback with pause / single-step / speed / level-switch / human play (`eval/play_gridworld.py`) | — |
| | **40** | | **29.5** | | |

---

## Brief compliance, verified by executing transitions

Probes: scratchpad `probe_grid.py`, `probe_grid2.py`, `probe_grid3.py`, `probe_arena.py`,
`probe_arena2.py`, `probe_g.py`, `probe_intrinsic.py`, all run with
`PYTHONPATH=<repo> SDL_VIDEODRIVER=dummy .venv/bin/python`.

| Brief rule | Result |
|---|---|
| Exactly 4 gridworld actions | `step(4)` → `ValueError: action must be in 0..3, got 4`; `step(-1)` → same message with `-1` |
| Rocks block: no displacement, no reward change | level 2, agent at `(0,4)` acting RIGHT into rock `(0,5)`: `pos=(0,4) r=0.0 done=False died=False collected=0 return=0.0` |
| Grid edge blocks identically | UP from row 0: `pos=(0,0) r=0.0 done=False` |
| Fire kills instantly on entry | level 1, `(9,1)` → fire `(8,1)`: `r=0.0 done=True died=True terminated=True` |
| Apple `+1` | level 2 apple `(3,2)`: `cum_r=1.0 collected=1` |
| Key `0` but unlocks | `cum_r_after_key=0.0 has_key=True collected=1` |
| Chest inert without the key | reached `(5,8)` with no key: `cum_r=0.0 collected=0 done=False`; re-entering pays `r=0.0` |
| Chest `+2` with the key | final arrival step `r=2.0`, `collected` 1 → 2 |
| Episode ends when all collectibles taken | level 0: `steps=17 terminated=True collected=3/3`; the next `step()` raises `RuntimeError: step() called after the episode ended; call reset() first` |
| Monster move probability `0.4` | level 4: **31559 / 79088 = 0.3990**; level 5: **30991 / 77684 = 0.3989** |
| Agent stepping onto a monster dies | `pos=(5,6) r=0.0 died=True terminated=True done=True` |
| Monster moving onto a stationary agent kills | stationary agent at `(0,0)`, monster at `(0,1)` with 3 legal moves: **2713 / 20000 = 0.1356** against the predicted `0.4 × 1/3 = 0.1333` |
| Style 1 = `Discrete(5)`, exact indices | `['NOOP','THRUST','ROTATE_LEFT','ROTATE_RIGHT','SHOOT']` |
| Style 2 = `Discrete(6)`, exact indices | `['NOOP','UP','DOWN','LEFT','RIGHT','SHOOT']` |
| Observation 10–30 normalized floats, never pixels | `Box(-1.0, 1.0, (21,), float32)`, `ndim=1`; 20 000 random steps → range `[-1.0000, 1.0000]`, 0 out-of-space |
| `r_i = strength / sqrt(n(s)+1)`, `n(s)` per-episode | exact to 1e-12 for n=0..4; `record` sequence `[0,1,2,3]`, after `reset()` `[0,1]` |
| `total = env_reward + r_i`, env rewards untouched | `env.step()` return unmodified (only `{0.0, 1.0, 2.0}` observed); `r_i` enters only the local `shaped` variable; logged return is env reward alone (`gridworld/algorithms.py:478-489`) |
| Arena reward decomposition | 9155/9155 model-driven steps decompose exactly into the six constants |

No brief-fixed mechanic has drifted since 2026-09-05.

---

## Test suite

```
SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest -p no:cacheprovider
658 passed in 41.81s
```

Up from `640 passed` yesterday; the 18 new tests are `tests/test_sweep_arena.py`. Two of them are
`@pytest.mark.slow` and **are not deselected by default** (`pyproject.toml` adds the marker but no
`-m "not slow"`), so the headline number includes them:
`SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest -m slow` → `2 passed, 656 deselected in 21.04s`.

Green is necessary, not sufficient. Spot-check of the constant guards: they assert the brief's
literals rather than re-deriving them from the code —
`tests/test_spec_constants.py:15-17` (`REWARD_APPLE == 1.0`, `REWARD_KEY == 0.0`,
`REWARD_CHEST == 2.0`), `:20` (`MONSTER_MOVE_PROBABILITY == 0.4`), `:23` (`N_ACTIONS == 4`),
`:38` and `:43` (exact action index tuples), `:58` (`10 <= OBS_DIM <= 30`).
`tests/test_gridworld_algorithms.py:307` parses the AST to prove `argmax` is never called.

---

## Measured performance — Part I

Greedy rollouts from each committed `.npz`: 30 rollouts per deterministic table, **300** per
stochastic one (levels 4–5). Env seeds `1000+k`, policy seeds `2000+k`. `claim` is `greedy_steps`
from the committed `summary_*.json`. Baseline for every row is the BFS optimum computed
independently by `gridworld/algorithms.optimal_collection_steps`.

| Artifact | claim | measured median | success % | death % | trunc % | mean return | BFS |
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
| L5 q s0 / s1 / s2 | **19** / 30 / 30 | 28 / 30 / 30 | 52.3 / 53.0 / 54.7 | 47.7 / 47.0 / 45.3 | 3.03–3.06 | 28 |
| L5 sarsa s0 / s1 / s2 | 30 / **23** / 30 | 30 / 30 / 30 | 51.0 / 53.3 / 54.3 | 49.0 / 46.7 / 45.7 | 3.01–3.05 | 28 |

Level 6 (row F), one greedy rollout per table, BFS optimum **54**:

| strength | greedy steps (seeds 0–4) | collected | truncated |
|---|---|---|---|
| 0.0 | 54, 54, 54, 54, 54 | 2/2 all seeds | 0/5 |
| 0.1 | 54, 54, 54, 54, 66 | 2/2 all seeds | 0/5 |
| 0.25 | 800, 54, 56, 66, 800 | 0,2,2,2,1 | 2/5 |
| 0.5 | 800 × 5 | 0,1,1,1,1 | **5/5** |
| 1.0 | 800 × 5 | 0,1,0,1,0 | **5/5** |

The baseline arm is **optimal** on all five seeds; the bonus makes it strictly worse. That is the
honest negative result `results/intrinsic_level6_q.md` already documents, and it is stronger
evidence for row F than a flattering curve would be.

Truncation is 0% everywhere on levels 0–5. Deaths on 4–5 are the monsters, not a broken policy
(mean return 2.6/3 and 3.0/4).

**Reproducibility (exact, this machine, today):**

- `.venv/bin/python -m train.train_gridworld --level 0 --algo q --seed 0 --results-dir <scratch>`
  → `verdict: OPTIMAL` in **1 s**; Q-table `states` and `values` **bitwise equal** to
  `results/qtable_level0_q_seed0.npz`; every shared summary field identical.
- `... --level 1 --compare --seed 0` → `q 11 steps rows [7,8]`, `sarsa 13 steps rows [6,7,8]`,
  `11.4%` vs `1.8%` death, `verdict: ROUTES DIFFER` in **2 s**; comparison JSON **identical** to
  `results/compare_level1_seed0.json`.
- `... --level 4 --algo q --seed 0` (stochastic, 12 000 episodes) → **3 s**; Q-table bitwise equal,
  summary identical field for field.

Full Part I regeneration is a few minutes of CPU. There is no schedule risk here.

---

## Measured performance — Part II

30 episodes per row, seeds 0–29, the same arenas for every policy, `deterministic=True`, through
`eval.play_arena.evaluate()` with the policy swapped. Random and no-op baselines sit beside each
model on identical seeds.

| Policy | Style | mean return | sd | phase mean | best | cleared | spawners | kills | steps | survival |
|---|---|---|---|---|---|---|---|---|---|---|
| **`models/ppo_direct.zip`** | direct | **+17.14** | 19.00 | 1.93 | 4 | **22/30** | 2.60 | 15.90 | 1011.1 | 13.3% |
| `models/ppo_direct_sweep.zip` | direct | **−4.16** | 10.05 | 1.00 | 1 | **0/30** | 0.13 | 13.07 | 615.5 | 6.7% |
| random baseline | direct | −12.85 | 1.44 | 1.00 | 1 | 0/30 | 0.03 | 1.90 | 241.8 | 0.0% |
| no-op baseline | direct | −14.18 | 0.12 | 1.00 | 1 | 0/30 | 0.00 | 0.00 | 168.4 | 0.0% |
| **`models/ppo_rotation.zip`** | rotation | **−0.53** | 10.73 | 1.47 | 2 | **14/30** | 1.30 | 6.77 | 745.9 | 13.3% |
| random baseline | rotation | −14.37 | 1.61 | 1.00 | 1 | 0/30 | 0.10 | 0.17 | 254.0 | 0.0% |
| no-op baseline | rotation | −14.18 | 0.12 | 1.00 | 1 | 0/30 | 0.00 | 0.00 | 168.4 | 0.0% |

Both shipped models clearly beat random on every behavioural metric, so both have learned.
The **tuned** model does not: `ppo_direct_sweep.zip` beats random on return and kills but clears
**zero** phases in 30 seeded episodes against the incumbent's 22.

Per-episode phases, for choosing video takes:

- direct, seeds 0–29: `[2,1,2,3,3,2,2,2,1,2,3,1,1,2,2,2,2,1,2,2,1,2,2,2,4,1,3,2,1,2]` — seed **24**
  reaches phase 4.
- rotation, seeds 0–29: seeds that clear phase 1 are `[0,2,3,8,10,11,13,16,17,19,21,23,25,27]`.

Reproduce: `SDL_VIDEODRIVER=dummy .venv/bin/python -m eval.play_arena --style both --episodes 20
--no-window` → direct `+16.25 +/- 11.28`, 15/20 cleared; rotation `+0.22 +/- 11.37`, 10/20 cleared.

**Throughput and schedule.** Single-env headless, random policy: **67 722 agent steps/s** =
203 165 physics frames/s. Committed sweep `run.json` files give the real cost of the whole J3
experiment on this machine:

| Stage | Runs | Steps each | Wall clock |
|---|---|---|---|
| explore | 9 | 100 000 | 7.1–9.2 s each |
| confirm | 4 | 100 000 | 6.9–7.9 s each |
| final | 1 | 400 000 | 28.5 s |
| **total** | **14** | **1.7 M** | **131.2 s = 2.2 min** |

Re-running the entire sweep at a **400 k** budget instead of 100 k would cost roughly **9 minutes**.
All remaining schedule risk is report and video, not compute.

---

## On-screen visibility (CLAUDE.md golden rule 3, and rubric V)

Assessed by rendering headless and reading the pixels, not by reading the draw code.

**Part I — strong.** `gridworld/render.py` puts on screen, simultaneously:

- HUD strip: `Level 1  steps 6  return +0.0  key no  items 0/1  6.0 st/s`, plus a right-aligned
  state word (`POLICY` / `PAUSED` / `SOLVED` / `DEAD` / `OUT OF STEPS`) — `gridworld/render.py:556-588`.
- Q-value heatmap with the **numeric** `max_a Q(s,a)` printed in every cell to 2 dp, colour-scaled
  over the values actually present — `gridworld/render.py:434-463`.
- Greedy policy arrows per cell for the current `(has_key, collected_mask)`, with genuine ties drawn
  as multiple arrows and all-equal (unvisited) rows drawn as nothing — `gridworld/render.py:465-486`.
  This is the correct honest behaviour for rubric B4 and it is visible.
- A controls/overlay legend panel naming every key binding and the on/off state of both overlays —
  `gridworld/render.py:591-637`.
- The ε schedule is plotted on a secondary axis of every training curve
  (`results/curve_level0_q_seed0.png`), with the BFS optimum drawn as a reference line on the
  steps panel.

**Part I gaps.** The live HUD never shows ε, α, γ or the algorithm name — the window title and the
stdout banner say `Level 0 — Q-LEARNING`, but a viewer of the recording sees only `POLICY`. There is
no live *training* visualiser at all: the decaying ε and the TD update are visible in the plots, not
on screen. For a demo graded on "algorithm internals visible on screen", adding the algorithm name,
ε and α/γ to the HUD strip is a few lines and would close the last gap.

**Part II — strong.** `arena/render.py` puts on screen:

- HUD: `PHASE 1 | HEALTH 5/5 | SCORE +1.49 | STEP 251 | ACTION SHOOT | STYLE direct` —
  `arena/render.py:323-347`. The `ACTION` field is what proves a learned policy is driving.
- The live 21-feature observation panel, labelled from `observation.describe()` so the labels cannot
  drift from the layout — `arena/render.py:379-399`. This is the single most persuasive frame in the
  project.
- Lines drawn from the ship to the nearest enemy and nearest spawner, which is exactly what features
  8–17 encode; spawner health bars; a spawner pulse that tracks the spawn timer; player hit flicker
  during invulnerability (`arena/render.py:254-263`); a phase banner latched for `BANNER_SECONDS` so
  a one-frame flag is actually visible (`arena/render.py:401-409`).

**Part II gaps.** No policy internals are rendered — no action-probability bars, no value estimate,
no per-step reward breakdown. A marker sees *what* the agent did but not *why*. `render_mode` accepts
only `'human'`, so there is no `rgb_array` path for automated capture. And the A3-018 acceptance
criteria for muzzle flash, explosion particles and screen shake are not met on `main`.

---

## Findings by severity

### HIGH

**H1 — The hyperparameter sweep found a regression, and the report-facing artifact does not say so.**
`results/arena_sweep/sweep_table.md` names `n_steps=512` the winner, states the full-budget retrain's
training means (`phase reached 1.290, episode reward 2.50`) and closes with "Compare the two under
`deterministic=True` first and promote only if the tuned model actually wins." That comparison has
been done — it lives in `docs/tickets/A3-013-...md:69-75` — and the tuned model loses badly. My own
independent measurement over 30 seeds is worse than the ticket's 5:
`models/ppo_direct_sweep.zip` = **−4.16 mean return, 0/30 phases cleared, 0.13 spawners** against
`models/ppo_direct.zip` = **+17.14, 22/30, 2.60**. The report will be written from the `results/`
artifact, not the ticket, so as it stands the report would cite a "winner" that is a large
regression. The head-to-head belongs in `sweep_table.md`.
Repro: `SDL_VIDEODRIVER=dummy PYTHONPATH=. .venv/bin/python -c "from eval.play_arena import evaluate; from stable_baselines3 import PPO; print(evaluate('direct', episodes=30, seed=0, agent=PPO.load('models/ppo_direct_sweep.zip', device='cpu'))['phase_cleared_episodes'])"` → `0`.

**H2 — `origin/A3-018` (creativity, 5 pts) is branched from a pre-Part-II base and would revert the
entire arena training pipeline if merged.** Its merge-base with `main` is `58202d0` (A3-020), which
predates A3-011, A3-012 and A3-013. `git diff --stat origin/main origin/A3-018` shows it **deleting**
`train/sweep_arena.py` (717 lines), `tests/test_sweep_arena.py` (324), `tests/test_train_arena.py`
(407), `tests/test_arena_eval.py` (169), `results/arena_sweep/*`, all three `models/*.zip` and every
checkpoint, while rewriting `train/train_arena.py` and `train/callbacks.py`. It does add
`tests/test_arena_creativity.py` (267 lines) and 402 lines of `tests/test_arena_render.py`. The
5-point creativity row is the largest single row still open, and the branch carrying it cannot be
merged as-is. It needs rebasing onto `66316d1` before anyone touches it.
Repro: `git merge-base origin/main origin/A3-018` → `58202d0`; `git diff --stat origin/main origin/A3-018 | tail -20`.

**H3 — 7.5 points (R + V) still have no artifact of any kind, 13 days from the due date.**
`report/` contains only `.gitkeep`; there is no video; `README.md:17-19` still lists the team as
`_TBD_ | _TBD_`. The brief requires student numbers, a contribution summary and a video link inside
the report, and A3-017 (fill in the team table) is a 10-minute chore that nobody else can do at the
last minute. **Unchanged from 2026-09-05.**
Repro: `git ls-tree -r --name-only main report` → `report/.gitkeep`.

### MEDIUM

**M1 — The two shipped models still have no provenance on `main`.**
`train/train_arena.py` demonstrably writes `<run>/run.json`, `<run>/monitor/worker_*.csv` and
`<run>/tb_1/` — a 3-second smoke run produced exactly those four paths. But `logs/ppo_direct_full/`
and `logs/ppo_rotation_full/` each contain **one bare event file** and nothing else, so neither
shipped model can be tied to a command, a config or a commit. Every one of the 14 sweep runs has a
`run.json`; the two models that actually ship do not. This weakens J and R5.
Repro: `find logs/ppo_direct_full logs/ppo_rotation_full -type f` → 2 paths.

**M2 — Part II has one table and zero figures.**
`ls results/*.png | grep -iE 'arena|ppo'` → nothing. `results/arena_sweep/` holds `sweep_table.md`
and `sweep_runs.json` only. R5 wants hyperparameter tables *or plots* (the table satisfies this) but
R6 wants a control-set comparison, and `eval.play_arena.evaluate()` returns everything R6 needs while
writing nothing to disk — the comparison exists only as stdout. Partially improved since yesterday
(the sweep table is new); the missing piece is now a comparison figure and a learning-curve export.

**M3 — The rotation agent's deterministic return is negative.**
−0.53 ± 10.73 over 30 seeds, 1.30 spawners destroyed, phase cleared in 14/30. It beats random
(−14.37) so row I stands, but the video rubric wants learned behaviour shown for *both* control
schemes and roughly half the rotation takes will not clear a phase. Its own TB curve ends at
`rollout/ep_rew_mean = 6.5251` (stochastic, 8 envs) against −0.53 measured deterministically — the
report must quote the measured number, not the curve. Pre-select from the 14 clearing seeds listed
above rather than rolling live.
Repro: `SDL_VIDEODRIVER=dummy .venv/bin/python -m eval.play_arena --style rotation --episodes 30 --no-window`.

**M4 — Board drift: A3-011 is still marked open but has been merged and works.**
`docs/tickets/A3-011-...md:6` has `status: open` and `docs/tickets/INDEX.md:13` lists it under Open,
yet `train/train_arena.py`, `train/callbacks.py` and `tests/test_train_arena.py` are on `main` and a
smoke run succeeds in 3 s. The board claims "**5 open · 15 done**" while one of those five is
complete. A3-013 has correctly moved to Done since yesterday. **Otherwise unchanged.**

**M5 — `docs/tickets/A3-013-...md:79,95` cites ticket `A3-022`, which does not exist.**
The highest tracked id is A3-020. The follow-up the sweep itself identified — "phase 1 clearable in
20-30 s is still open and is the more likely fix than more timesteps" — is therefore **untracked**,
along with the recommendation to raise `sweep.budget_timesteps`. Two concrete, cheap pieces of work
have no ticket.
Repro: `ls docs/tickets | grep -oE "A3-[0-9]+" | sort -u` → ends at `A3-020`.

**M6 — The 2026-09-05 report's headline total is arithmetically wrong.**
Its rubric table's `Defensible` column sums to **28.5**, not the **26.5** stated in the verdict and
the table's own total row. Anyone reading yesterday's report and today's would infer a +3.0 jump when
the real like-for-like movement is +1.0. Flagging it so the trend line stays honest.

**M7 — Two committed level-5 summaries quote an unrepresentative single rollout.**
`results/summary_level5_q_seed0.json` claims `greedy_steps: 19` and
`results/summary_level5_sarsa_seed1.json` claims `23`, while the same tables measure a median of
**30** over 300 rollouts. On a stochastic level one greedy rollout is a sample. **Unchanged.**

### LOW

**L1 — `.gitignore` still ignores two directories that already contain tracked files.**
The headline fix landed (`results/`, `logs/`, `models/` are no longer ignored — `git check-ignore -v
results/ logs/ models/` now returns nothing for all three, closing yesterday's H2). But
`.gitignore:36-37` keeps `logs/**/*.zip` and `models/checkpoints/`, and `models/checkpoints/` already
holds **16 tracked** zips out of 24 on disk. The 8 new `final_n_steps-512_s0_*` checkpoints are
ignored and untracked. The rule and the repository disagree; a future `git add models/checkpoints`
will silently skip.
Repro: `git status --porcelain --ignored models | head`.

**L2 — `results/summary_level0_q_seed0.json` is stale relative to the code that writes it.**
A fresh run emits four extra fields (`solved`, `collection_order`, `collection_order_text`,
`key_precedes_chest`); all shared fields are byte-identical. Regenerating costs 1 s. **Unchanged.**

**L3 — `n(s)` keyed on the arrival state.** `gridworld/algorithms.py:441-455` scores `s'` rather than
`s`, with the alternative implemented, measured and rejected in the docstring. Defensible and
documented; a strict reading could differ. ≤0.5 pt. **Unchanged.**

**L4 — README/platform mismatch.** `README.md:28` says "verified on 3.14.4, Windows" and every code
block is PowerShell; this machine runs Python 3.14.6 on macOS with `.venv/bin/python`, and the sweep
logs were produced on `Les-MacBook-Pro-2.local` while the shipped models were produced on `TDuong04`.
Everything works; the documented commands are Windows-only in a repo now being developed on both.
**Unchanged.**

**L5 — `models/` now holds three zips.** `ppo_direct.zip`, `ppo_rotation.zip` and
`ppo_direct_sweep.zip`. The row-I requirement (one model per control style) is met, and the third is
legitimate J3 evidence, but a marker opening `models/` sees an unexplained extra file whose only
explanation lives in a ticket. One line in the report, or a `models/README.md`, fixes it.

**L6 — The `slow` marker is declared but never applied at the suite level.** `pyproject.toml` adds
`markers = ["slow: ..."]` with no matching `-m "not slow"` in `addopts`, so `pytest` runs the two
model-training tests every time. Harmless today (21 s of the 41.81 s total) but it will grow.

### Entry points confirmed working (all under `SDL_VIDEODRIVER=dummy`)

| Command | Result |
|---|---|
| `python -m pytest` | `658 passed in 41.81s` |
| `python -m train.train_gridworld --level 0 --algo q --seed 0` | `verdict: OPTIMAL`, 1 s, 5 artifacts written |
| `python -m train.train_gridworld --level 1 --compare --seed 0` | `verdict: ROUTES DIFFER`, 2 s, JSON identical to committed |
| `python -m train.train_gridworld --level 4 --algo q --seed 0` | 3 s, Q-table bitwise identical to committed |
| `python -m train.train_arena --style direct --timesteps 8192 --n-envs 2` | model saved in 1.2 s; `run.json`, 2 monitor CSVs and `tb_1/` written |
| `python -m train.sweep_arena --dry-run` | prints `9 exploratory runs at 100,000 steps`, then `2 configs x 3 seeds, then one run at 400,000 steps`, exit 0 |
| `python -m eval.play_gridworld --level 0 --algo q --frames 60` | `Level 0 — Q-LEARNING`, exit 0 |
| `python -m eval.play_gridworld --level 4 --algo sarsa --frames 60` | `Level 4 — SARSA`, exit 0 |
| `python -m eval.play_gridworld --level 1 --compare --frames 60` | `Level 1 — Q-LEARNING vs SARSA`, exit 0 |
| `python -m eval.play_gridworld --level 6 --algo q --frames 60` | `Level 6 — Q-LEARNING`, exit 0 |
| `python -m eval.play_arena --style both --episodes 20 --no-window` | comparison table printed, exit 0 |
| `tensorboard --logdir logs` | 16 event directories: 2 full runs (19 tags each) + 14 sweep runs (18 tags each) |

---

## Delta since 2026-09-05 (`d04e6b1` → `66316d1`)

**Closed:**

- **H2 (gitignore contradiction)** — `results/`, `logs/` and `models/` are no longer ignored.
  `git check-ignore -v results/ logs/ models/` returns nothing. Residual narrower issue at **L1**.
- **H3 (J3 unmet on `main`)** — the sweep is merged: `train/sweep_arena.py` (717 lines),
  `tests/test_sweep_arena.py` (18 tests), 14 run directories with `run.json` + monitor CSVs +
  `train.log` + `tb_1/`, `results/arena_sweep/sweep_table.md` and `sweep_runs.json`,
  `models/ppo_direct_sweep.zip`. README documents both sweep commands.
- **Board:** A3-013 correctly moved to Done with an unusually honest Result section.

**Still open, unchanged:** H1 (R + V have no artifact; team table `_TBD_`) → now **H3**;
M1 (no provenance for the shipped models); M2 (Part II figures, partially improved);
M3 (rotation negative); M4 (A3-011 mis-stated on the board); M5 (level-5 summaries) → now **M7**;
L1 (stale level-0 summary) → now **L2**; L2 (`n(s)` on `s'`) → now **L3**;
L3 (README platform) → now **L4**.

**New this cycle:** H1 (sweep table omits the head-to-head that shows the tuned model is a
regression); H2 (`origin/A3-018` diverged from a pre-Part-II base); M5 (`A3-022` cited but never
raised); M6 (prior report's arithmetic); L1 (`models/checkpoints/` ignored yet tracked);
L5 (third model in `models/`); L6 (`slow` marker never excluded).

**Regressions:** none. 658/658 tests pass, every Part I number reproduces bit-for-bit, and both
shipped arena models measure exactly as they did yesterday (+17.14 / −0.53 over 30 seeds;
+16.25 / +0.22 over 20).

---

## Highest-value next actions, ranked by points per effort

1. **Fill in the team table in `README.md:17-19`** — 10 minutes, unblocks 2.5 points. Student
   numbers and the contribution split are hard requirements of row R and nobody else can supply
   them. Closes A3-017. *(Carried over from yesterday's #2, still not done.)*
2. **Add the head-to-head to `results/arena_sweep/sweep_table.md`** — 20 minutes, protects ~0.5 of
   J and prevents the report citing a regression as a win. Paste the 30-seed table from this
   report beside the sweep's own conclusion. Fixes **H1**.
3. **Re-run the sweep at a 400 k budget** — **~9 minutes of compute**, measured from the committed
   `run.json` wall clocks (the whole 14-run grid at 100 k took 131.2 s). The sweep's own conclusion
   is that "the 100 k budget does not separate the axes"; at 400 k it becomes a decisive result
   instead of a documented null one, and J moves from 2.5 to 3. Highest points-per-minute item on
   the board and it has no ticket (**M5**).
4. **Write a Part II results dump to `results/`** — 1 hour, unblocks R6. `eval.play_arena.evaluate()`
   already returns mean return, phase, spawners, kills, steps and survival; it just never writes a
   JSON or a figure. Mirror what `train_gridworld` does for Part I, and include the random baseline
   so the comparison has a reference.
5. **Rebase `origin/A3-018` onto `66316d1`** — before any creativity work continues. As it stands
   the branch deletes the entire Part II training pipeline. 5 points sit behind it. Fixes **H2**.
6. **Write the report** — 2.5 points directly, and it is the only place creativity gets *claimed*
   (a further 2.5 currently unscored). Every Part I number exists and reproduces; Part II needs
   items 2–4 first.
7. **Record the video** — 5 points, tooling ready today. Direct agent seed 24 reaches phase 4; for
   rotation, use one of seeds `[0,2,3,8,10,11,13,16,17,19,21,23,25,27]` rather than rolling live.
8. **Put ε, α, γ and the algorithm name on the gridworld HUD** — 30 minutes, no rubric row of its
   own but directly serves the "algorithm internals visible on screen" rule the video is graded
   against. `gridworld/render.py:563-571` already builds the HUD parts list.
9. **Copy `run.json` and the monitor CSVs for the two shipped models into `logs/`, or retrain them
   with the current pipeline** — 5 minutes, fixes **M1** and ties both models to a command.
10. **Housekeeping** — close A3-011 on the board; raise the missing A3-021/A3-022; regenerate
    `results/summary_level0_q_seed0.json`; stop quoting `greedy_steps` on levels 4–5 without the
    distribution beside it; decide whether `models/checkpoints/` should be ignored or tracked.
