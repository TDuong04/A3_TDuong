---
id: A3-037
title: Fix arena RL agent performance regression after the difficulty rebalance
type: bug
status: done
priority: P0
rubric: I
points_at_risk: 4
area: arena
github: https://github.com/TDuong04/A3_TDuong/issues/73
owner: unassigned
blocks: []
blocked_by: []
created: 2026-09-13
updated: 2026-09-13
---

# A3-037 — Fix arena RL agent performance regression after the difficulty rebalance

## Context

Documented retroactively, following A3-034's precedent for already-shipped work verified directly
before the ticket existed.

Commit `94040bd` ("adjust indentation and formatting for clarity in arena.yaml") is a misleading
message for what it actually did: it rebalanced `config/arena.yaml`'s difficulty values (player
health/speed/rotation/shoot-cooldown, phase enemy speed/health/spawn-interval, `max_enemies`). The
team decided to keep the new values as the deliberate go-forward baseline rather than revert, but
the rebalance silently broke the trained agents' behaviour, observed by a teammate playing/watching
the agent: (1) rotation kept shooting into empty space instead of targeting remaining spawners once
enemies were cleared; (2) the agent got stuck against walls/corners; (3) generally low hit-rate
(not literally "misses only what's directly in front", per investigation below); (4) rotation spun
in one direction endlessly and never caught an enemy chasing it from behind.

## Root cause and fixes

- **Diagnosed against the shipped (pre-rebalance) rotation model re-run under the new physics**, via
  a controlled A/B script: corner-dwelling rose 0.9% -> 15.0% of steps, and "enemy crosses the
  ship's nose without the ship settling on it" rose sharply (one episode: 392 crossings) from the
  physics change alone, not the model.
- **Three real bugs fixed in `arena/env.py`'s optional potential-based shaping** (`_aim_potential`,
  `_safety_potential`; only active when `shaping.aim_strength`/`safety_strength` are nonzero, and
  the checked-in default is `0.0` so these ship inert):
  - `_aim_potential` tracked only the nearest living enemy and returned `0.0` once enemies were
    gone, giving no gradient toward remaining spawners. Now falls back to the nearest living
    spawner.
  - Its lead calculation used the target's velocity *relative to the player*, but `Bullet.__init__`
    gives bullets a fixed world-frame velocity that does not inherit player velocity — the old code
    aimed at a physically wrong point (measured 34 degrees of ship-motion-induced error). Now uses
    the target's own velocity.
  - The term was `cos(bearing error)`, stationary (zero first-order gradient) at exactly 180
    degrees — precisely where a chaser sits astern. Changed to a linear
    `1 - |bearing error| / pi` so every bearing has a real gradient.
  - `_safety_potential` changed from scaling by the full arena diagonal (which rewarded retreating
    into corners, the farthest points in a rectangle) to saturating at a new
    `shaping.safe_distance` config value (300px).
- **Swept shaping strengths at full budget, confirmed on 3 seeds**: shaping measured *net harmful*
  on every combination tried (unstable across seeds, collapsed spawner-kill counts by orbiting
  instead of engaging). Both models ship with shaping OFF (`0.0`); the bug fixes above are kept
  (correct, tested) but inert at `0.0`.
- **`total_timesteps` raised `400000` -> `1600000`** in `config/arena.yaml`: the rebalance also
  quadrupled mean episode length (longer-surviving player), which silently starved training —
  PPO was getting roughly a quarter of the episode-level learning signal it used to at the old
  budget.
- **`gamma` raised `0.99` -> `0.997`**: at `gamma=0.99` the effective horizon (~100 steps) no
  longer matched the now much longer episodes/sparser phase rewards (a `direct`-style phase reward
  now lands roughly every ~530 steps, discounted by `0.99^530 = 0.005` — nearly invisible). Swept
  at full budget on 3 additional seeds for `direct` (the style showing the largest effect):
  `gamma=0.997` won cleanly and consistently (return, survival, phases cleared all improved on
  every seed tried); `gamma=0.999` was worse (horizon too long relative to PPO's `n_steps=2048`
  rollout window). Adopted `0.997` for both styles.
- A teammate independently retuned `player.shoot_cooldown` (`0.15` -> `0.25`) and
  `player.rotation_speed` (`400` -> `300`) in the same config edit, fixing the "sprays constantly"
  and "spins too fast to track a target" symptoms respectively.
- Both models retrained (`models/ppo_direct.zip`, `models/ppo_rotation.zip`; ~112s/model on CPU at
  the new budget) and re-evaluated.

## Final measured results

30 deterministic episodes, current models in `models/`, vs. the pre-rebalance shipped numbers and a
random-policy baseline (`results/arena_eval/comparison.md`, `comparison_random.md`):

| Style | Metric | Pre-rebalance shipped | Post-fix (this ticket) | Random baseline |
|---|---|---:|---:|---:|
| rotation | return | +228.10 | +357.69 ± 10.65 | -20.81 ± 2.30 |
| rotation | phases cleared | 26/30 | 30/30 | 0/30 |
| rotation | survival | 47% | 97% | 0% |
| rotation | spawners destroyed | 27.40 | 47.30 | 0.13 |
| rotation | hit rate | 12.8% | 76.0% (measured: `hit_rate` in `eval_rotation_seed0.json`) | — |
| direct | return | +56.55 (older physics, not directly comparable) | +72.38 ± 18.50 | -12.16 ± 6.55 |
| direct | phases cleared | 29/30 | 30/30 | 2/30 |
| direct | survival | — | 80% | 3% |

Both styles now clear every phase on all 30 seeded episodes and beat their own pre-rebalance
numbers as well as the random baseline by a wide margin.

Priority derivation: **P0**. The rebalance broke an existing trained model's real evaluated
behaviour and (per A3-036's log, filed the day before) left the test suite failing against the new
`config/arena.yaml` — both are named directly in CLAUDE.md's P0 criteria ("blocks... or breaks the
test suite or an existing trained model"). Rubric row I (arena eval) is worth 4 points, so this is
also a >=3-point row on its own. Priority reflects the severity of the regression this closes.

## Acceptance criteria

- [x] Root cause of all four reported behaviours identified and attributed (physics rebalance
      interacting with pre-existing shaping bugs and a stale training budget/gamma), not guessed.
- [x] Three shaping bugs fixed in `arena/env.py` (`_aim_potential` spawner fallback + target's-own-
      velocity lead + linear bearing term; `_safety_potential` saturating at `shaping.safe_distance`
      instead of the arena diagonal), verified by reading the diff directly.
- [x] Shaping strengths re-swept at full budget on 3 seeds; confirmed net harmful; both models ship
      with `aim_strength`/`safety_strength` at `0.0` (verified in `config/arena.yaml`).
- [x] `total_timesteps` (`400000` -> `1600000`) and `gamma` (`0.99` -> `0.997`) updated in
      `config/arena.yaml`, gamma choice confirmed by a 3-seed sweep for `direct`.
- [x] Both models retrained and re-evaluated at 30 deterministic episodes/style;
      `results/arena_eval/comparison.md` + `comparison_random.md` + the four `eval_*.json` files
      regenerated and show the post-fix numbers beating both the pre-rebalance baseline and the
      random-policy baseline (verified by reading the regenerated files directly — table above).
- [x] `tests/test_arena_entities.py`, `test_arena_env.py`, `test_measure_pacing.py`,
      `test_train_arena.py` updated to match the new config values/timing; new
      `tests/test_arena_shaping_defects.py` (5 tests) pins the three shaping bug fixes and the
      safety-saturation fix.
- [x] `pytest -m "not slow"` passes (verified directly: ran the full suite, exit code 0, no
      failures).
- [x] `ruff check .` clean on every file touched — the only remaining findings anywhere in
      `arena/env.py` (4: two pre-existing long-comment lines, two pre-existing `lru_cache` ->
      `cache` suggestions) all predate this session per `git blame` (2026-09-10/09-12 authorship).
- [x] `arena/constants.py` untouched and no frozen reward value changed (verified: no diff on
      `arena/constants.py`; the shaping terms this ticket fixes are the opt-in, currently-zeroed
      terms added by A3-034, not the brief's fixed reward list).

## Notes

Files touched: `arena/env.py` (the three shaping bug fixes), `config/arena.yaml` (`gamma`,
`total_timesteps`, new `shaping.safe_distance` key, shaping doc comments rewritten, plus the
teammate's `shoot_cooldown`/`rotation_speed` edits), `models/ppo_direct.zip` + `models/ppo_rotation.zip`
(retrained), `results/arena_eval/comparison.md` + `comparison_random.md` + the four `eval_*.json`
files (regenerated evidence), `tests/test_arena_entities.py` + `tests/test_arena_env.py` +
`tests/test_measure_pacing.py` + `tests/test_train_arena.py` (updated for the new config/timing),
new `tests/test_arena_shaping_defects.py`, plus `logs/ppo_rotation_shipped/` and
`logs/ppo_direct_shipped/` (TensorBoard/monitor logs regenerated to match the retrained models).

As of this ticket's closure, all of the above sits **modified/untracked in the working tree, not
yet committed** (confirmed via `git status`) — this agent does not commit per its operating
instructions. The next commit covering this work should reference this ticket id.

Cross-reference: `A3-032` and `A3-035` (both still open) are independent, unrelated defects against
the same evidence file (`results/arena_eval/comparison.md`) — an `--episodes` default bug and an
evaluation run-to-run non-determinism bug, respectively. Neither is fixed by this ticket, and this
ticket's regenerated table should still be treated as one sample per A3-035 until that is resolved.

## Log

- 2026-09-13 — Created and closed retroactively. Work was investigated, fixed, retrained and
  verified directly prior to this ticket existing. Verified independently rather than taken on
  trust: read the `arena/env.py` diff for all three shaping fixes, read `config/arena.yaml`'s diff
  for `gamma`/`total_timesteps`/`safe_distance`/`shoot_cooldown`/`rotation_speed`, read the
  regenerated `results/arena_eval/comparison.md` and `comparison_random.md` and cross-checked
  `hit_rate` directly from `eval_rotation_seed0.json` (measured 76.0%, matching the reported
  hit-rate recovery), confirmed `tests/test_arena_shaping_defects.py` exists (5 tests) and ran
  `pytest -m "not slow"` directly (exit 0, all passing) and `ruff check .` on every touched file
  (4 pre-existing findings in `arena/env.py`, all predating this session per `git blame`), and
  confirmed `arena/constants.py` carries no diff. Closing as done on that basis.
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/73
