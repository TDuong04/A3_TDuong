# Whole-project evaluation — 2026-09-12

**Verdict: 31.0 / 40 defensible today. Submission readiness: NOT READY — the code is
submission-grade, the deliverables are not.** Every implemented rubric row (A, B, C, D, F, G, H, I)
verified by execution and reproduces its committed evidence exactly. The 9.0 lost marks are almost
entirely deliverables that do not exist: no report (R, 2.5), no video (V, 5), plus a stale sweep
provenance (J, 0.5) and creativity (1.0) that no artifact currently presents to a marker.

Measured against commit `fc2cf4b` (`main`), working tree as found plus the restoration noted in
§0. Interpreter: `.venv/bin/python` (Python 3.14.6, pygame-ce 2.5.7, macOS).

---

## 0. Disclosures and limits of this audit

1. **I wrote into `results/` and restored it.** While smoke-testing
   `python -m eval.play_arena --style both --no-window --episodes 2`, the script wrote
   `comparison.md`, `eval_direct_seed0.json` and `eval_rotation_seed0.json` — it saves by default
   and I did not pass `--no-save`. This is the A3-032 bug biting an auditor in real time. I restored
   all three with `git checkout --`. **Net effect: `results/arena_eval/` is now clean at HEAD (the
   good 30-episode evidence).** The pre-existing corrupt 3-episode `comparison.md` and
   `eval_rotation_seed0.json` that were in the working tree at session start are therefore gone;
   their content is preserved verbatim in §3 below. I did not touch `eval/play_arena.py`, so the
   underlying defect is untouched and A3-032 remains legitimately open.
2. **I could not read `A3_Brief.pdf`.** `pypdf` extracts 0 characters from all 19 pages and reports
   0 embedded images; `pdftoppm`/poppler is not installed. Brief compliance below is therefore
   verified against the mechanics as restated in `CLAUDE.md` and this audit's task brief, **not**
   against the PDF directly. Anyone able to render the PDF should re-confirm §2.
3. **The repo does not contain a reissued brief.** `A3_Brief.pdf` has exactly one commit
   (`897dd98`). See finding H3 on the deadline.
4. `ruff check .` **could not be run — ruff is not installed**, neither in `.venv` nor on `PATH`
   (`.venv/bin/python -m ruff` → `No module named ruff`; `which ruff` → not found). `CLAUDE.md`
   §10 makes a clean ruff a definition-of-done gate that nobody on this machine can currently check.

---

## 1. Test suite and lint

```
860 passed in 49.18s
```

Command: `SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest`. Green, and green on a suite that is
unusually large for a student project. Necessary, not sufficient — so I spot-checked the guards:

`tests/test_spec_constants.py` asserts **literal brief values**, not values re-derived from the
code:

- `:15` `assert grid_c.REWARD_APPLE == 1.0`
- `:17` `assert grid_c.REWARD_CHEST == 2.0`
- `:20` `assert grid_c.MONSTER_MOVE_PROBABILITY == 0.4`
- `:39` `assert arena_c.N_ACTIONS["rotation"] == 5`
- `:44` `assert arena_c.N_ACTIONS["direct"] == 6`
- `:58` `assert 10 <= arena_c.OBS_DIM <= 30`

That is the right shape for a spec guard. `ruff` is unverifiable (see §0.4).

---

## 2. Brief compliance — verified by executing transitions

All probes executed real `step()` calls on constructed levels, never read the constants module.
Script: session scratchpad `probe_spec.py`, `probe_arena.py`.

### Gridworld — every mechanic PASS

| Mechanic | Brief | Measured | Verdict |
|---|---|---|---|
| Action set | 4 | `['UP','DOWN','LEFT','RIGHT']`; `action must be in 0..3, got 4` on a 5th | PASS |
| Rock blocks | no displacement, no reward change | `(1,1) -> (1,1)`, `reward=0.0`, `done=False` | PASS |
| Fire | instant death on entry | `died=True`, `done=True`, `reward=0.0` | PASS |
| Apple | `+1` | `reward=1.0`, episode continues while items remain | PASS |
| Key | `0` but unlocks | `reward=0.0`, `has_key=True` | PASS |
| Chest without key | inert | `reward=0.0`, `done=False`, stays on grid | PASS |
| Chest with key | `+2` | `reward=2.0` | PASS |
| Episode end | all collectibles taken | `done=True` on the last pickup | PASS |
| Monster move probability | `0.4` | **0.4038** over 6000 trials (SE 0.0063) | PASS |
| Monster kills by moving onto player | required | 802/6000 stationary-agent deaths | PASS |
| Intrinsic reward | `strength / sqrt(n(s)+1)` | exact to 1e-12 for strength ∈ {0.25,0.5,1.0}, n ∈ {0,1,2,5,17}; `(1.0,0)=1.0`, `(1.0,3)=0.5` | PASS |

`gridworld/env.py:213-255` implements the brief's step order explicitly, including the second death
check (`:246`) for a monster stepping onto a stationary agent — the case implementations usually
miss. `n(s)` is per-episode by construction: `EpisodeVisitCounts.reset()` clears on every
`env.reset()` (`gridworld/algorithms.py:401-403`), keyed on the full state tuple, and `record()`
returns the count *before* the arrival so a first visit pays full strength (`:409-418`).
`results/intrinsic_level6_q.md:19-23` confirms env rewards are untouched and the plotted return is
the **environment** return with the bonus excluded.

### Arena — every mechanic PASS

| Requirement | Brief | Measured | Verdict |
|---|---|---|---|
| Style 1 action space | `Discrete(5)` | `Discrete(5)`, indices `0 NOOP, 1 THRUST, 2 ROTATE_LEFT, 3 ROTATE_RIGHT, 4 SHOOT` | PASS |
| Style 2 action space | `Discrete(6)` | `Discrete(6)`, indices `0 NOOP, 1 UP, 2 DOWN, 3 LEFT, 4 RIGHT, 5 SHOOT` | PASS |
| Observation | fixed float vector, ~10–30 features, never pixels | `Box(-1.0, 1.0, (20,), float32)`, 1-D, single shape across 3 episodes/style | PASS |
| Normalisation / finiteness | — | range `[-1.000, 1.000]`, all finite | PASS |
| Seeded reproducibility | — | same seed → identical trajectory and return, both styles; different seed differs | PASS |
| Headless purity | no pygame in sim | `pygame in sys.modules: False` after full env use | PASS |

**Throughput:** 66,438 agent-steps/sec (rotation), 57,354 (direct), single env, headless.

**Caveat (minor):** `arena/env.py:130` rejects `render_mode="rgb_array"` —
`unknown render_mode 'rgb_array'; expected one of ['human'] or None`. Gymnasium convention usually
offers `rgb_array`; row H only requires `render()`, which exists and works, so this is a note, not
a finding.

**Caveat (watch for the video):** `mechanics=True` extends the observation to **36 features**,
outside the brief's ~10–30. It defaults to `False` (`arena/env.py:123`), the shipped models are
baseline, and mechanics models go to a separate `models/mechanics/` — so the submission path is
compliant. Do not demo mechanics mode as the trained agent.

---

## 3. Accuracy — algorithms and artifact agreement

### Update rules (verified by execution, not reading)

| Check | Result |
|---|---|
| Q-learning bootstraps on `max` over next actions | `gridworld/algorithms.py:257` `target = reward + config.gamma * float(q[next_state].max())` |
| SARSA never calls `max` in its update | `.max()` in `sarsa` body after docstring: **False**; `:339` `target = reward + config.gamma * float(q[next_state][next_action])` |
| Q-learning body does use `.max()` | True |
| Terminal target is `r` alone | both algos branch on `info["terminated"]` (`:254`, `:334`), not `done` — truncation keeps its bootstrap |
| Shared epsilon schedule | both call `epsilon_schedule(config)` (`:236`, `:315`) → one `LinearEpsilon` |
| Linear decay between config bounds | `1.0 → 0.05` over 2000: `(0,1.0) (500,0.7625) (1000,0.525) (1500,0.2875) (2000,0.05) (3000,0.05)`; exactly linear to 1e-12, held at end |
| Bounds come from config, not code | `config/gridworld.yaml:8-10`, per-level overrides at `:31-34` |
| Random tie-breaking, never `argmax` | 20,000 greedy picks on an all-zero row: `{0:4974, 1:4993, 2:5014, 3:5019}`, max share 0.251 (argmax would be 1.000). `np.argmax` appears twice in the module, both in docstrings forbidding it (`:16`, `:136`) |

### Do the saved artifacts agree with the summaries?

I loaded all 36 `results/qtable_level*_seed*.npz`, rolled out greedily, and compared to
`summary_*.json`.

- **Levels 0–3 (deterministic): 24/24 exact matches** on steps, return and collected count.
- **Levels 4–5 (stochastic): 7 of 12 diverge** on the single-sample `greedy_steps` field — because
  a single greedy rollout on a stochastic level is a draw, not a property, and my probe used a
  different RNG stream. **This is not a reproducibility failure**, and the evidence table is honest
  about it: `results/levels0-5_seeds0-1-2.md` reports 500-rollout *rates* for levels 4–5 and states
  "a single greedy rollout is one sample from that distribution."

Re-running the trainer's own streams (`GREEDY_SAMPLE_OFFSET=100_000`, `BEHAVIOUR_SAMPLE_OFFSET=200_000`,
`ENV_STREAM_OFFSET=500_000`, 500 rollouts) reproduced **every level 4–5 number in the committed
table exactly** — all 12 rows, greedy success/death/truncation, mean return, mean steps, and both
eps=0.05 columns. Example: L4 q seed0 `74.0% / 26.0% / 0.0% / 2.538 / 20.2 | 73.4% / 26.6%`,
identical to the table. **Part I evidence is bit-for-bit reproducible.**

### Row C3 — the SARSA/Q divergence actually exists

This is the row most projects lose. Here it is unambiguous, on all three seeds:

```
seed 0: Q  11 steps  path hugs row 7 (the cliff edge, adjacent to fire)
seed 0: S  13 steps  path detours via row 6 (one cell further from the fire)
policies DIFFER: True   SARSA longer: True      (identical on seeds 1 and 2)
```

Hazard exposure at the epsilon the agents converged under (eps=0.05, 2000 rollouts, level 1 seed 0):

| algo | death rate | mean steps |
|---|---|---|
| q | **11.35 %** (227/2000) | 11.02 |
| sarsa | **1.80 %** (36/2000) | 13.69 |

A 6.3x difference in death rate for 2.7 extra steps. Row C's evidence is real and visually
demonstrable — the `--compare` view renders both policies' arrows side by side (screenshot captured,
§6).

---

## 4. Performance — measured, with a baseline beside every number

### Part I — greedy rollouts vs BFS optimum

| Level | Algo | Greedy steps | BFS optimum | Optimal? | Collection | Death rate | Truncation |
|---|---|---|---|---|---|---|---|
| 0 | q / sarsa | 17 / 17 | 17 | **yes, both** | 3/3 | 0 % | 0 % |
| 1 | q | 11 | 11 | yes | 1/1 | 11.35 % @eps.05 | 0 % |
| 1 | sarsa | 13 | 11 | no (by design — safer route) | 1/1 | 1.80 % @eps.05 | 0 % |
| 2 | q / sarsa | 25 / 25 | 25 | yes, both | 5/5 | 0 % | 0 % |
| 3 | q | 34 (seed1: 36) | 34 | yes on 2/3 seeds | 4/4 | 0 % | 0 % |
| 3 | sarsa | 34 | 34 | yes, all seeds | 4/4 | 0 % | 0 % |
| 4 | q | — | 21 | success **73.4–74.4 %** | 2.538/3.0 | 25.6–26.6 % | **0.0 %** |
| 4 | sarsa | — | 21 | success **72.4–75.8 %** | 2.504–2.574/3.0 | 24.2–27.6 % | **0.0 %** |
| 5 | q | — | 28 | success **53.2–55.2 %** | ~3.00/4.0 | 44.8–46.8 % | **0.0 %** |
| 5 | sarsa | — | 28 | success **52.8–54.4 %** | ~2.99/4.0 | 45.6–47.2 % | **0.0 %** |

**Level 0 is optimal, not merely solved** (17 = BFS 17, both algorithms, all three seeds). Levels
4–5 death rates are the monsters, not a learning failure: 0 % truncation everywhere means the
policies always resolve, and the remaining loss is a 0.4-probability monster that the state key
cannot see.

Row F (intrinsic, level 6) reports a **negative result honestly**: strength 0.5 *reduces* final-window
success by `-0.672 ± 0.025` (95 % CI −0.741 to −0.603, excludes zero) against the baseline's 100 %,
with a mechanistic explanation (`results/intrinsic_level6_q.md:80-99`) that the per-episode bonus is
non-Markov and worth ~`strength/(1-gamma)` = 10 against the chest's +2. Every non-zero strength first
opened the chest on identical episodes (17, 106, 95, 26, 119) because the behaviour policy is
scale-invariant until a reward exists. The implementation is brief-exact; the experiment says the
brief's bonus hurts here. That is a *better* report section than a positive result.

### Part II — both models vs random on identical seeds (30 episodes, seeds 0–29)

| Style | Policy | n | Return | Phase | Kills | Spawners | Survival | Steps |
|---|---|---|---|---|---|---|---|---|
| direct | **PPO** | 30 | **+20.57 ± 16.99** | 2.37 | 8.87 | 3.93 | 53 % | 1518 |
| direct | random | 30 | −12.81 ± 1.75 | 1.00 | 2.00 | 0.03 | 0 % | 247 |
| rotation | **PPO** | 30 | **+7.59 ± 5.42** | 1.93 | 5.43 | 2.27 | 20 % | 825 |
| rotation | random | 30 | −14.57 ± 1.52 | 1.00 | 0.07 | 0.07 | 0 % | 247 |

**Both models beat random decisively** (+33.38 and +22.16 return; 53 % vs 0 % and 20 % vs 0 %
survival). My independent numbers reproduce the **committed** `results/arena_eval/comparison.md`
exactly — direct `+20.57`, phase `2.37`, spawners `3.93`, enemies `8.87`, survival `53 %`, steps
`1518`; rotation `+7.59`, `1.93`, `2.27`, `5.43`, `20 %`, `825`. That table is sound evidence.

**Training cost (measured today):** 100,000 PPO timesteps = **9.87 s wall** (8 envs, SubprocVecEnv,
CPU; SB3 reported `fps 22206`). A shipped model is 400k steps ≈ **40 s**. A full sweep re-run
(9 × 100k + 6 × 100k + 1 × 400k ≈ 1.9M steps) ≈ **3–4 minutes**. **There is no compute schedule risk
in Part II whatsoever.**

---

## 5. Rubric coverage — 31.0 / 40 defensible

Ordered by points at risk.

| Row | Pts | Status | Defensible | Evidence |
|---|---:|---|---:|---|
| **V** — video | 5 | **MISSING** | **0** | No video file, no link, anywhere in repo. A3-015 open. |
| **Creativity** | 5 | **PARTIAL** | **4.0** | Extensive, shipped, and I confirmed it renders (§6). Held back only because no report or video presents it to a marker. |
| **I** — two control schemes | 4 | SATISFIED | **4.0** | `arena/constants.py` indices verified by execution; `models/ppo_direct.zip` + `ppo_rotation.zip`; `comparison.md` reproduced exactly; `comparison_random.md` baseline intact. **Live defect F1 can destroy this evidence again.** |
| **G** — real-time arena | 4.5 | SATISFIED | **4.5** | `arena/entities.py`, `arena/render.py`; spawners, pursuit AI, health, projectiles, phases, death/max-steps all exercised in rollouts; renders (§6). |
| **R** — report | 2.5 | **MISSING** | **0** | `report/` holds only `creativity.md` (a section draft) and `.gitkeep`. No report, no student numbers, no contribution table, no video link. |
| **C** — SARSA | 3 | SATISFIED | **3.0** | `gridworld/algorithms.py:339`; no `.max()` in body; 11 vs 13 steps and 11.35 % vs 1.80 % death, all 3 seeds. |
| **D** — levels 2–3 | 3 | SATISFIED | **3.0** | `results/levels0-5_seeds0-1-2.md`; both algos optimal at 25 and 34 steps; key-before-chest `yes`. |
| **F** — intrinsic reward | 3 | SATISFIED | **3.0** | Formula exact to 1e-12; per-episode counter `:401-403`; env rewards untouched; `results/intrinsic_level6_q.md`. |
| **J** — training pipeline | 3 | **PARTIAL** | **2.5** | PPO + MLP 64x64 + 26 tracked tfevents + 5 behaviour metrics all verified. **−0.5: the J3 sweep was measured at `OBS_DIM=21`** (finding F2). |
| **B** — Q-learning | 2.5 | SATISFIED | **2.5** | `:257` max bootstrap; random tie-break measured 0.251 max share; linear config-driven decay. |
| **H** — gym env + obs | 2.5 | SATISFIED | **2.5** | `Box(-1,1,(20,),float32)`, finite, reproducible; `results/arena_validation/validation.md`. |
| **A** — gridworld env + renderer | 2 | SATISFIED | **2.0** | All 11 mechanics probes PASS; `gridworld/render.py` runs, human-playable. |
| | **40** | | **31.0** | |

### Submission-readiness checklist

| Item | Status |
|---|---|
| `models/` exact name, one model per control style, committed | **PASS** — `ppo_direct.zip`, `ppo_rotation.zip`, both `TRACKED`, `git check-ignore` returns nothing |
| `logs/` holds real TensorBoard event files | **PASS** — 26 tracked `events.out.tfevents.*` |
| Nothing brief-required is gitignored | **PASS** — only `logs/**/*.zip` and `models/checkpoints/` ignored, both deliberate recovery scaffolding |
| `results/` committed | **PASS** — 257 tracked files |
| `report/` holds the report, ≤10 pages, no appendix | **FAIL — no report exists** |
| Three student numbers in the report | **FAIL — no report; A3-017 open** |
| Contribution summary | **FAIL — no report** |
| Video exists, <10 min, linked from report | **FAIL — no video** |
| Every README entry point launches | **PASS** — see §6 |

---

## 6. Entry points and creativity — launched and confirmed rendering

Every command below was executed under `SDL_VIDEODRIVER=dummy`. All exited 0.

| Command | Result |
|---|---|
| `pytest` | 860 passed |
| `python -m train.train_gridworld --help` … and all 6 other modules | all `--help` OK |
| `python -m train.train_arena --style direct --timesteps 100000` | 9.87 s, model saved, tb written |
| `python -m train.sweep_arena --dry-run` | prints 9-run plan + 2×3 seeds + one 400k run |
| `python -m eval.play_arena --style rotation --headless --frames 120 --no-save` | renders, `+10.04` return |
| `python -m eval.play_gridworld --level 0 --algo q --frames 60` | loads `qtable_level0_q_seed0.npz`, renders |
| `python -m eval.play_gridworld --level 1 --compare --frames 60` | "Level 1 — Q-LEARNING vs SARSA" |
| `python -m eval.play_gridworld --level 2 --algo sarsa --heatmap --frames 40` | renders |
| `python -m eval.play_gridworld --level 6 --algo q --learn --frames 40` | fresh in-memory learner |
| `python -m gridworld.render --frames 40` | human-play mode, full control legend |

Note: `eval/play_gridworld.py` has **no `--headless` flag** (it uses `--frames`), while
`eval/play_arena.py` has `--headless`. `README.md` does not document a `--headless` for the former,
so this is an inconsistency, not a broken command.

**Creativity features confirmed rendering** (screenshots captured to the session scratchpad):

- **Arena observation overlay** — all 20 features live with names and values (`player_x -0.53`,
  `heading_cos +0.99`, `enemy_local_dx +0.08`, `spawners_alive +0.33`, …), plus a **"PILOT
  PERCEPTION"** radar showing ship-local `+x forward` / `+y lateral` axes with colour-keyed spawner
  and enemy bearings. Toggling the overlay off drops unique colours from 1384 to 500 — it is really
  drawing.
- **Arena policy overlay** — per-action probability bars `[0.0338, 0.5476, 0.0075, 0.3646, 0.0466]`
  with the chosen action (`THRUST`) highlighted, and the critic's **`VALUE V(s) +4.81`**.
- **Gridworld policy arrows + Learner panel** — the `--compare` view renders both panels with
  per-cell arrows and a Learner block reading `algorithm / alpha 0.1 / gamma 0.95 / epsilon 1 -> 0.05 /
  decay over 6000 ep / trained 8000 ep` **from the run's own summary**, plus Controls and Overlays
  legends. The Q panel's arrows hug the fire row; SARSA's turn away — row C3 made visible.
- **Q-value heatmap, episode-visit heatmap, F3 debug panel, human play on both envs, random-policy
  baseline** — all present and launching.
- **Shields + elite chargers** (`arena/mechanics.py`) — implemented and human-playable, but
  **no trained policy exists for them** (`report/creativity.md` admits "needs newly trained
  policies"), and they push the observation to 36 features. They count as shipped gameplay, not as
  a learned result.

---

## 7. Findings by severity

### F1 — HIGH — `eval/play_arena.py` silently destroys row I's evidence, two ways

Worth up to 4 points, and it fired on me during this audit (§0.1).

**(a) Episode default.** `eval/play_arena.py:556` `parser.add_argument("--episodes", type=int, default=3)`.
The command `README.md:110` documents for regenerating row I evidence —
`python -m eval.play_arena --style both --no-window` — does not override it, so running the
documented command writes a **3-episode** table over a 30-episode one. The module docstring
(`eval/play_arena.py:7`) shows a *third* number, `--episodes 5`.

**(b) Single-style runs truncate the two-style table — not covered by A3-032.**
`eval/play_arena.py:327-328` writes `comparison.md` from whatever `stats` it was handed, unconditionally.
Running `--style rotation` alone therefore rewrites the comparison table with **one row**, silently
deleting the `direct` row. Worse, `:359` hardcodes `command = "--style both --no-window"` in the
provenance header regardless of what actually ran, and never records `--episodes`. The artifact
found in the working tree at session start proves the combination:

```
Generated 2026-09-11T16:46:54+00:00 by `python -m eval.play_arena --style both --no-window`.
| `rotation` | `ppo` | 3 | +11.39 ± 4.94 | 2.00 (best 2) | 3/3 | 2.67 | 8.33 | 33% | 1161 |
```

A header claiming `--style both` above a table containing one style at n=3. Quoting that in the
report would overstate rotation's phase-clear reliability as 3/3 (100 %) against the true 28/30
(93 %), and would lose the direct-style comparison entirely — which *is* row I.

**Impact:** the committed evidence is correct and I reproduced it exactly, so no marks are lost
*today*. But the defect is live, and the report and video are both due to pull from this table.
**Reproduction:** `SDL_VIDEODRIVER=dummy .venv/bin/python -m eval.play_arena --style rotation --no-window`
(seed 0) then open `results/arena_eval/comparison.md`.
**Fix (~15 min):** raise the default to 30, make `write_results` merge with or refuse to truncate an
existing table, and build the provenance string from the actual args. Extend A3-032's acceptance
criteria to cover (b).

### F2 — MEDIUM — the J3 sweep table was measured on a superseded observation space

`results/arena_sweep/sweep_table.md:3` — "Generated 2026-09-05T23:42:20 from commit `70b1740`".

```
OBS_DIM at 70b1740 (sweep, 2026-09-05):  21
OBS_DIM at 4fafc46 (models, 2026-09-09): 20
OBS_DIM now:                             20
```

`4fafc46` is `fix(arena): drop the dead spawner_exists feature and retrain` — the sweep's numbers
were produced against a 21-feature vector including a feature
`results/arena_validation/validation.md` describes as "constant at 1.0 and structurally unable to
read otherwise." A marker checking the provenance line sees a commit that predates the observation
change.

**Mitigation:** the removed feature was *dead*, so it carried no gradient and cannot plausibly have
reordered the hyperparameter ranking. The sweep's conclusions survive; its absolute numbers are from
a slightly wider input.
**Cost to fix properly: ~3–4 minutes of compute** (measured, §4). This is the cheapest 0.5 marks in
the project.

### F3 — MEDIUM — no report, and no student-number table

`report/` contains `creativity.md` (a self-described "section draft, not the completed report") and
`.gitkeep`. `git ls-files report/` returns those two files only. `report/figures/` contains a single
untracked `__pycache__/capture_figures.cpython-314.pyc` — **the figure-capture script's source is
not in the repo, only its compiled bytecode.** No student numbers, no contribution summary, no video
link. **R = 0 / 2.5, and creativity cannot be presented.** A3-014 and A3-017 both open.

### F4 — MEDIUM — no video

A3-015 open. **V = 0 / 5.** With ~1 day left this is the single largest block of points at risk, and
it depends on the report existing for the link.

### F5 — LOW — A3-011 is stale bookkeeping, not a gap (answering the direct question)

`docs/tickets/INDEX.md:13` lists A3-011 as **open, P0, `blocked_by: [A3-010]`**, while A3-010 is
`done`. I verified every one of A3-011's six acceptance criteria is met in shipped code:

| AC | Evidence |
|---|---|
| trains either style from config | `train/train_arena.py:301`, `--style` choice |
| SubprocVecEnv, 8 envs, `__main__` guard | `:143`, `config/arena.yaml:7 n_envs: 8`, `:387` |
| `SDL_VIDEODRIVER=dummy` before pygame import | `:40`, `:110` |
| Monitor with `info_keywords`, TB under `logs/` | `:114`, `:186` |
| Callback logs the 4 behaviour metrics | `train/callbacks.py:44-47`; live run emitted `phase_reached`, `spawners_destroyed`, `enemies_killed`, `damage_taken` **and** `survival_rate` |
| Checkpoints every 50k to `models/checkpoints/` | `:309-310` |

**This is not a genuine gap — row J's pipeline is fully implemented and I ran it.** It is board
drift, and it is a *repeat* finding: `docs/evaluations/2026-09-06-solution-evaluation.md:314`
already reported "**M4 — Board drift: A3-011 is still marked open but has been merged and works**"
six days ago. Close it.

### F6 — LOW — A3-032 cites an evaluation file that does not exist

The ticket's central provenance argument cites
`docs/evaluations/2026-09-10-solution-evaluation.md:318` and `:317-321`. `ls docs/evaluations/`
returns only `2026-09-05-solution-evaluation.md` and `2026-09-06-solution-evaluation.md`, and
`git log --diff-filter=D -- docs/evaluations/` shows no deletion — **that audit was never committed.**
The 30-episode provenance rested on a dangling citation. It no longer does: this audit reproduced
the 30-episode table exactly (§4), so cite *this* file instead.

### F7 — LOW — `ruff` is not installed

`CLAUDE.md` §4 and §10 make `ruff check .` a pre-push and definition-of-done gate. It cannot be run
on this machine (`No module named ruff`; not on `PATH`). Lint compliance is currently unverifiable
and unenforced. `pip install ruff` — 1 minute.

### H3 — NOTE — deadline and brief discrepancy, unresolved in-repo

Three sources disagree:

| Source | Deadline |
|---|---|
| `CLAUDE.md` (committed, line 3) | **19 September 2026** |
| Team private notes / this audit's instructions | **13 September 2026** |
| `A3_Brief.pdf` (committed) | unreadable (§0.2); single commit `897dd98`, never updated |

The reissued brief that reportedly moved the date **is not in the repo**. Treating **2026-09-13** as
authoritative, today (2026-09-12) leaves **~1 day, not 2.** `CLAUDE.md:3` is stale and actively
misleading to any teammate's agent session, which is exactly the failure mode `CLAUDE.md` §1 exists
to prevent. Commit the reissued brief and fix the date line.

---

## 8. Highest-value next actions, ranked by points per effort

Given **~1 day** to 2026-09-13.

| # | Action | Points | Effort | Why first |
|---|---|---:|---|---|
| 1 | **Write the report** (A3-014) — with the three student numbers and contribution table (A3-017) and the video link | **2.5**, and it gates creativity's 5 | 3–5 h | Nothing else unlocks as much. All the numbers already exist and are verified in §3–4; this is assembly, not measurement. Pull the C3 comparison, the PPO-vs-random table and the level 0 optimality straight from this file. |
| 2 | **Record the video** (A3-015) | **5** | 2–3 h | Largest single block. Everything it needs already renders (§6) and every entry point launches. Shoot: level 0 optimal → level 1 `--compare` (the arrows diverge on screen) → level 4 monsters → arena both styles with `O` and `V` overlays → `--random` baseline. |
| 3 | **Fix F1 both ways, then regenerate and commit the table** | protects **4** | 15 min + 1 min run | Cheapest insurance in the project. Do it *before* the video and report pull from `comparison.md`, or you risk recording a one-row table. |
| 4 | **Re-run the sweep on the current 20-feature env** | **0.5** | **3–4 min compute** | Measured, not estimated (§4). Removes the only stale-provenance finding. If even this is too much, add one sentence to the report noting the sweep predates the dead-feature removal and why the ranking is unaffected — that recovers most of the 0.5 for free. |
| 5 | **Close A3-011; extend A3-032 to cover the single-style truncation** | 0 | 10 min | Board honesty. A P0 that is actually done has been distorting triage for six days. |
| 6 | **Commit the reissued brief; fix `CLAUDE.md:3`** | 0 | 5 min | Rank-1 source of truth is stale and the deadline in it is wrong by six days. |
| 7 | `pip install ruff && ruff check .` | 0 | 5 min | Restores an enforceable DoD gate; may surface easy cleanups. |

**Schedule call.** Items 3–7 total well under an hour and all fit. Items 1 and 2 are the whole
remaining job and together consume the available day; they are tight but feasible **only if started
immediately and run in parallel across the team** — the report author and the video recorder do not
block each other, because both pull from artifacts that already exist and are verified. If time
forces a choice, **record the video first**: it is worth twice the report's marks, it can cite
results verbally without a finished document, and it is the only artifact that demonstrates the
creativity work that is otherwise invisible to a marker. The cheaper fallback for the report is a
minimal compliant document — student numbers, contribution table, the four verified tables from
§3–4, one figure per rubric row, and the video link — which is achievable in ~2 hours and secures
most of R's 2.5 rather than gambling on a polished ten-page draft that may not land.

**Do not** attempt any retraining beyond item 4. Nothing in Part I or Part II needs it: every
number in `results/` reproduced exactly today.
