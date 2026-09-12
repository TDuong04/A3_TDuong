---
id: A3-035
title: Fix arena evaluation not being reproducible run to run from a fixed seed
type: bug
status: open
priority: P1
rubric: I
points_at_risk: 4
area: eval
owner: unassigned
blocks: []
blocked_by: []
created: 2026-09-12
updated: 2026-09-12
---

# A3-035 — Fix arena evaluation not being reproducible run to run from a fixed seed

## Context

Found while independently verifying A3-034's closure (potential-based reward shaping for the
rotation control style). Not caused by A3-034's diff and not part of its scope — `eval/play_arena.py`'s
diff there only adds `fired`/`hit`/`hit_rate` telemetry fields inside `evaluate()`; it does not touch
seeding, `env.reset`, or the `deterministic=True` predict call.

Re-running the exact command that produced the checked-in `results/arena_eval/comparison.md`
(`python -m eval.play_arena --style <style> --no-window --episodes 30 --seed 0`) does **not**
reproduce the checked-in numbers, for either control style, on repeat invocation:

| Run | style | threads | return | survival | phase reached | steps mean |
|---|---|---|---|---|---|---|
| original (`comparison.md` as of A3-034's retrain) | direct | default | +20.57 ± 16.71 | 53% | 2.37 | 1518 |
| rerun 1 | direct | default | +29.87 ± 7.76 | 100% | 2.77 | 2000 |
| rerun 2 | direct | default | +24.23 ± 9.88 | 93% | 2.50 | 1961 |
| rerun 3 (`--style both`) | direct | default | +36.24 ± 10.81 | 93% | 2.77 | 1978 |
| original (`comparison.md` as of A3-034's retrain) | rotation | default | +10.89 ± 5.58 | 30% | 2.00 | 1007 |
| rerun 1 | rotation | default | +10.89 ± 5.58 | 30% | 2.00 | 1007 |
| rerun 2 | rotation | default | +27.27 ± 17.33 | 57% | 2.67 | 1561 |
| rerun 3 | rotation | `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` | -88.50 ± 106.10 | 50% | 2.73 | 13228 |
| rerun 4 | rotation | `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` | -119.11 ± 100.38 | 67% | 2.70 | 17137 |
| rerun 5 (`--style both`) | rotation | default | +29.20 ± 14.92 | 73% | 2.50 | 1763 |

`results/arena_eval/comparison.md` was left on rerun 3/rerun 5 above (the last, matched `--style
both` invocation) so the checked-in file is at least internally consistent between its two rows,
rather than mid-experiment — but per this ticket, that figure is itself just one more sample.

Ruled out as the cause:

- **Stale/changed model file.** `models/ppo_direct.zip`/`ppo_rotation.zip` had identical MD5 and
  mtime across every rerun above.
- **`--style both` vs single-style seeding.** `eval/play_arena.py`'s `main()` calls the same
  `evaluate(style, episodes, seed, algo, agent)` once per style regardless of whether `--style both`
  or a single style is requested — no shared cross-style RNG consumption.
- **Unseeded randomness in the simulation.** `arena/entities.py` has no `random`/`np.random` calls
  at all. `arena/env.py`'s own randomness (spawner placement, initial heading) goes through
  `self.np_random`, correctly seeded via `super().reset(seed=seed)` in `reset()` — both confirmed
  by reading the source directly, not inferred.
- **Non-deterministic policy sampling.** `eval/play_arena.py`'s `DETERMINISTIC = True` is set
  correctly and passed through to `agent.predict(obs, deterministic=DETERMINISTIC)`.
- **Simple multi-threaded BLAS reduction order.** Pinning `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` was
  tried twice, back to back, on the theory that PyTorch's CPU matmul reduction order varies by
  thread scheduling. It did **not** stabilize the result — reruns 3 and 4 above, both under
  identical pinning, disagree with each other (-88.50 vs -119.11 return, 50% vs 67% survival) by
  about as much as any pair of default-threaded runs does.

Both pinned runs' step counts (13228 and 17137, vs the usual ~1000-2000) and deeply negative returns
are themselves a clue, and the fact that *both* pinned runs show the same pattern makes it unlikely
to be a fluke of one run: `arena/env.py`'s `step()` has no wall-clock dependency (`time.time()`/
`clock.tick()` only appear in `render.py`, confirmed by grep), so a real-time-pacing bug is unlikely.
More probably, single-threading changes something about episode length distribution directly (e.g.
a small number of episodes running out to a much larger `max_episode_steps` after clearing all three
configured phases with nothing left to fight, accumulating step-penalty across a long idle tail) —
but that is an inference from this investigation, not a confirmed mechanism, and it does not explain
the default-threaded variance (reruns 1-2 above) on its own.

Distinct from **A3-032** (open, same evidence file, same rubric row): A3-032 is a diagnosed defect
where `--episodes` defaults to 3 instead of 30, so the *documented* command silently under-samples.
This ticket's numbers above all used an explicit, correct `--episodes 30 --seed 0` and still
disagree run to run — fixing A3-032's default does not fix this, and fixing this does not fix
A3-032's default/docstring/provenance mismatches. Both should land before
`results/arena_eval/comparison.md` is cited in the report or video.

Priority derivation: **P1**, the same derivation A3-032 already uses for this exact file: row I is
worth 4 points (≥3 — not P0 by the row-size clause on its own), but this is exactly the "defect
degrading evidence quality" CLAUDE.md's P1 rule names. Not marked `blocks: [A3-014, A3-015]` for the
same reason A3-032 isn't — neither is otherwise stalled on it, but both should pull from a
re-verified table once this is understood, not the currently-uncertain one.

## Acceptance criteria

- [ ] Root cause identified for why identically-seeded, identical-model, identical-episode-count
      invocations of `evaluate()` (`eval/play_arena.py`) produce different `return_mean`/
      `survival_rate`/`phase_mean` on separate process runs.
- [ ] Either genuine run-to-run determinism is restored (repeated invocations of the same seeded
      command yield identical reported metrics), or — if bit-exact determinism is not achievable
      end to end — the evaluation protocol is changed to make the variance honest: report mean ± std
      across several independent seeded runs, not a single run's numbers presented as exact.
- [ ] `results/arena_eval/comparison.md` and `comparison_random.md` (plus their four backing JSON
      files) regenerated under whichever protocol results, with the file itself stating which
      protocol produced it.
- [ ] Checked whether `train/sweep_arena.py`'s promote/head-to-head stage (rubric J3, already
      `done` via A3-013) is affected — it calls this same `evaluate()` with `deterministic=True` to
      decide "promote or keep the incumbent". If that decision turns out to be noisy too, flag
      A3-013 for re-verification rather than silently trusting its recorded verdict (not reopening
      A3-013 pre-emptively from this ticket).

## Notes

Files likely touched: `eval/play_arena.py` (`evaluate()`, and whatever inference-determinism knob
turns out to matter), possibly `arena/env.py` around the two `self.np_random.uniform(...)` call
sites, `train/sweep_arena.py` (shares the same `evaluate()` path).

## Log

- 2026-09-12 — Filed while independently verifying A3-034's closure. Confirmed the model files are
  byte-identical across runs (MD5), confirmed `arena/entities.py` has no randomness and
  `arena/env.py`'s randomness is seeded via `self.np_random`, and confirmed — with two back-to-back
  pinned runs, not one — that pinning `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to 1 does not stabilize
  it. A final `--style both` run left `results/arena_eval/comparison.md` internally consistent
  (both rows from one invocation) rather than mid-experiment, but it is itself just one more sample
  per the table above. Root cause not identified. Left open — not this agent's to fix.
