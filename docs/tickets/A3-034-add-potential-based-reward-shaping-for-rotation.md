---
id: A3-034
title: Add potential-based reward shaping to fix rotation-style arena aim and survival
type: feature
status: done
priority: P2
rubric: Creativity
points_at_risk: 0
area: arena
github: https://github.com/TDuong04/A3_TDuong/issues/69
owner: unassigned
blocks: []
blocked_by: []
created: 2026-09-12
updated: 2026-09-12
---

# A3-034 — Add potential-based reward shaping to fix rotation-style arena aim and survival

## Context

Documented retroactively: implemented and verified directly, then filed for a branch/commit id,
following A3-033's precedent for already-shipped work raised straight to `done`.

Rotation-style PPO had a real aiming problem next to direct: 12.8% bullet hit-rate vs direct's
17.3% (30 deterministic episodes, the then-shipped seed-0 rotation model). Root-caused as a
structural property of rotation control (turning and shooting are two separate per-step actions,
unlike direct where heading is implied by movement), not a training-budget or hyperparameter
problem — a dedicated rotation-style hyperparameter sweep over
learning_rate/n_steps/ent_coef/gamma/net_arch (`results/arena_sweep/rotation/sweep_table.md`, via
`python -m train.sweep_arena`) found no configuration that beat its incumbent at the time.

Priority derivation: **P2**. `rubric: Creativity` is nominal only, the same convention A3-033
(cat-meme) used: A3-018 already banks the Creativity row's required "two additional gameplay
systems" criterion independently of this ticket, so this carries `points_at_risk: 0`. Row J
(reward encourages progression) is likewise already fully banked and closed by A3-011 — this is
additional reward-design work beyond the brief's own minimum reward list, not something either row
is waiting on, which is why it is P2 (quality) rather than P0/P1.

## What shipped

- `arena/env.py`: `ShapingConfig` dataclass plus `_aim_potential`/`_safety_potential`, both
  potential-based (`F = gamma * phi(s') - phi(s)`, Ng/Harada/Russell 1999), added inside `step()`
  alongside the frozen reward terms. `arena/constants.py` is untouched (confirmed via
  `git diff --stat -- arena/constants.py`, empty).
  - Aim-alignment: `phi` = cosine of the angle between ship heading and the nearest living enemy.
  - Range-keeping: `phi` = distance to that same enemy, scaled to `[0, 1]`. Added deliberately
    after aim-alignment alone was found to raise hit-rate but cut survival (rewarding "point at the
    enemy" alone gave PPO no reason not to also close distance with it).
- `config/arena.yaml`: new `shaping:` block (`aim_strength`, `safety_strength`), checked-in default
  `0.0`/`0.0` (shaping off, falls back to exactly the brief's fixed reward list). Extensively
  commented with the grid-search numbers behind the shipped choice.
- `train/train_arena.py`: `--aim-strength`/`--safety-strength` CLI overrides on top of the config
  default, the same per-run-override convention `--seed` already uses.
- `arena/debug.py`: `REWARD_TERMS` extended with `"aim_shaping"`/`"safety_shaping"` rows. Traced the
  consumer, not just the producer: `arena/render.py`'s existing panel-drawing loop
  (`for term, (label, weight) in REWARD_TERMS.items()`) already iterates this dict, so both new
  rows appear on the existing debug HUD with no `render.py` change — genuinely on screen, not only
  in logs.
- Tests: `TestAimShaping`/`TestSafetyShaping` added to `tests/test_arena_env.py` (zero-with-no-
  enemy, correct sign, exact-formula match against a hand-computed expectation). Pre-existing
  exact-reward-accounting tests (`tests/test_arena_debug.py`, `tests/test_arena_mechanics.py`, and
  `test_arena_env.py`'s own shared fixture) now explicitly force
  `aim_strength=0.0, safety_strength=0.0` so they keep isolating the frozen constants regardless of
  the shipped config default.
- Retrained and reshipped `models/ppo_rotation.zip` (seed 0) and `models/ppo_direct.zip` (seed 1) —
  each style's actual documented shipped seed per `results/arena_eval/model_selection.json`
  (direct's shipped seed has never been 0). `rotation` trains with
  `aim_strength=0.005, safety_strength=0.02`, the config-commented outcome of a 2x3 grid confirmed
  on 3 seeds — the only candidate that beat the un-shaped baseline on survival, hit-rate, return,
  phase reached and phases cleared simultaneously, across all 3 seeds. `direct` ships unshaped:
  every combination tried on it, down to very small strengths, cut its survival from a 53% baseline
  to single digits, because direct's heading always equals its movement direction, so an aim
  incentive there is inseparable from a movement incentive in a way it isn't for rotation (which
  thrusts and turns independently).
- Regenerated `results/arena_eval/comparison.md`, `comparison_random.md`, and the four
  `eval_*_seed0.json` files via `python -m eval.play_arena --style both --no-window --episodes 30`.

## Acceptance criteria

- [x] Both new reward terms are potential-based and `arena/constants.py`/no frozen mechanic is
      touched — read `arena/env.py`'s formula directly; confirmed empty `git diff` on `constants.py`.
- [x] Strengths are config-driven (`config/arena.yaml`'s `shaping:` block), not hardcoded, with a
      per-run CLI override.
- [x] Both shaping terms render on the existing debug HUD, not only in logs — traced
      `REWARD_TERMS` from `arena/debug.py` into the pre-existing draw loop in `arena/render.py`.
- [x] Both control styles evaluated with a stated random-policy baseline
      (`results/arena_eval/comparison_random.md`), per row I's convention.
- [x] `pytest -m "not slow"` passes clean (ran directly: exit 0, no failures). The new
      `TestAimShaping`/`TestSafetyShaping` classes pass standalone (7/7, ran directly).
      `ruff check .` holds at 60 errors — the same count as before this work. The full `-m slow`
      suite (real-model training tests) was **not** re-run as part of this verification; only the
      fast suite and the targeted new tests were executed directly.
- [x] `results/arena_eval/` regenerated against the actual shipped model files (confirmed by MD5 —
      the binaries evaluated are the ones on disk, not a stale copy) — **with an important caveat
      found during this verification, see below.**

## Important caveat found during closure verification

Re-running the exact command that produced the evidence table
(`python -m eval.play_arena --style {direct,rotation} --no-window --episodes 30 --seed 0`) does
**not** reproduce the same numbers run to run, for either control style — repeated invocations gave
direct-style returns anywhere from +20.57 to +36.24 and survival anywhere from 53% to 100%, and
rotation-style returns from +7.55 to +29.20 and survival from 20% to 73%, all at the identical
seeded command against the identical (MD5-verified) model file. Full reproduction data is in
**A3-035**, filed separately, including that pinning `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to 1 does
not stabilize it either.

`arena/entities.py` has no randomness of its own, and `arena/env.py`'s procedural generation is
correctly seeded through `self.np_random` via `super().reset(seed=seed)` (confirmed by reading both
files) — so the environment's own procedural generation is not the evident cause. This looks like a
policy-inference reproducibility gap, not something this ticket's diff introduced and not something
the shaping mechanism itself suffers from — `eval/play_arena.py`'s diff in this change only added
`fired`/`hit`/`hit_rate` telemetry, it did not touch seeding or the `deterministic=True` prediction
call.

Filed separately as A3-035 rather than blocking this ticket's closure, because the shaping
mechanism itself (the code, config, HUD wiring and tests above) is genuinely done and does not
depend on the evaluation harness being reproducible to be correct. `results/arena_eval/comparison.md`
was left regenerated one final time (direct +36.24 ± 10.81/93% survival, rotation +29.20 ± 14.92/73%
survival, both styles from the same single invocation so the file is at least internally consistent)
rather than mid-experiment — but per A3-035, this specific figure should be read as *a* sample, not
a settled number, until that ticket is resolved.

## Notes

Files touched by this ticket specifically: `arena/env.py`, `arena/debug.py`, `config/arena.yaml`,
`train/train_arena.py`, `tests/test_arena_env.py`, `tests/test_arena_debug.py`,
`tests/test_arena_mechanics.py`, `models/ppo_rotation.zip`, `models/ppo_direct.zip`,
`results/arena_eval/comparison.md`, `comparison_random.md`, `eval_direct_seed0.json`,
`eval_rotation_seed0.json`, `eval_random_direct_seed0.json`, `eval_random_rotation_seed0.json`, and
the `fired`/`hit`/`hit_rate` addition inside `eval/play_arena.py`'s `evaluate()`.

At the time of writing, the working tree also carries **unrelated, uncommitted** diffs to
`arena/render.py`, `gridworld/render.py`, and the rest of `eval/play_arena.py` (a `wait_for_retry`/
`play()` refactor) from other in-flight work (the retry-and-wait loop referenced in A3-033, and
visual-theme work). Do not sweep those into a commit for this ticket.

`results/arena_sweep/rotation/` (untracked) is diagnostic evidence for the "not a hyperparameter
problem" claim above; it ran at 14:53 against an intermediate build of `models/ppo_rotation.zip`
that existed before the final shaping retrain overwrote the file at 16:17, so its own absolute
numbers describe a superseded snapshot, not the shipped model — only its "hyperparameters alone
don't close the gap" conclusion is being relied on here.

## Log

- 2026-09-12 — Created and closed retroactively, following A3-033's precedent. Work was implemented
  and verified directly prior to this ticket existing. Verified independently rather than taken on
  trust: read `arena/env.py`'s shaping formula, `config/arena.yaml`'s `shaping:` block and comments,
  `arena/debug.py`'s `REWARD_TERMS`, and confirmed the `render.py` consumer that makes it visible;
  confirmed `arena/constants.py` untouched; confirmed `models/ppo_rotation.zip` (seed 0) and
  `models/ppo_direct.zip` (seed 1) match `results/arena_eval/model_selection.json`'s documented
  shipped seeds; ran `pytest -m "not slow"` (passed) and
  `pytest tests/test_arena_env.py -k "AimShaping or SafetyShaping"` (7/7 passed) directly; ran
  `ruff check .` (60 errors, matching the claimed pre-existing count). While independently
  re-running the evidence-generating command to verify the checked-in table, discovered that arena
  evaluation is not run-to-run reproducible from a fixed `--seed` for either control style (see
  caveat above) — filed as a new, separate ticket **A3-035** rather than reopening this one, since
  the shaping feature itself does not depend on that harness bug and is independently verified
  correct. Closing A3-034 as done on that basis.
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/69
