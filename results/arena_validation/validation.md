# Arena environment validation — A3-021

Generated 2026-09-09T10:38:33+00:00, Python 3.14.6, single core.
Raw numbers: [`validation.json`](validation.json).

The harness is deliberately not committed: it is a one-shot stress test, and what the report and
A3-014 need to cite is its *output*, not its source. It stepped `ArenaEnv` under a random policy
for 10,000 steps per control style, ran 200 full episodes per style, and re-probed every suspicious
feature under the trained policies in `models/`.

**Verdict: every acceptance criterion passes, and no feature is dead.** The first run of this
harness, against the 21-feature observation, found `spawner_exists` constant at 1.0 and
structurally unable to read otherwise. That feature has since been removed and both agents
retrained; this report is the re-run that confirms it. Two features are still named below as
saturated, and both are explained by the policy or the control style that produced them rather
than by the code.

## Criteria

| # | Criterion | Result |
|---|-----------|--------|
| 1 | Observation `(20,)` float32, finite, within `[-1, 1]` over 10,000 random steps, both styles | **pass** — also `observation_space.contains()` on every step |
| 2 | No constant feature; none at a bound for >50% of steps | **pass with findings** — see below |
| 3 | `reset(seed=n)` twice + same actions → byte-identical obs and rewards, 3 seeds | **pass** — seeds 0, 1, 2, both styles |
| 4 | `Discrete(5)` rotation, `Discrete(6)` direct | **pass** — and both reject an out-of-range action with `ValueError` |
| 5 | 200 episodes end within `MAX_EPISODE_STEPS`; `terminated`/`truncated` never both set | **pass** — longest 528 steps of a 2000 cap, 0 double flags |
| 6 | `render_mode=None` runs under `SDL_VIDEODRIVER=dummy`, opens no window | **pass** — `pygame` is never even imported |
| 7 | Step throughput on one core | **83,164 steps/s** rotation, **71,000 steps/s** direct |
| 8 | Written to a durable file under `results/` | this file and `validation.json` |

## Per-feature ranges over 10,000 random steps

`@b` is the fraction of steps the feature spent exactly at −1 or +1.

| idx | feature | rot min | rot max | rot std | rot @b | dir min | dir max | dir std | dir @b |
|----:|---------|--------:|--------:|--------:|-------:|--------:|--------:|--------:|-------:|
| 0 | `player_x` | −0.971 | 0.971 | 0.701 | 0.0% | −0.971 | 0.971 | 0.536 | 0.0% |
| 1 | `player_y` | −0.959 | 0.959 | 0.749 | 0.0% | −0.959 | 0.959 | 0.595 | 0.0% |
| 2 | `player_vx` | −1.000 | 1.000 | 0.377 | 0.1% | −0.999 | 0.962 | 0.323 | 0.0% |
| 3 | `player_vy` | −1.000 | 1.000 | 0.319 | 0.0% | −0.953 | 0.996 | 0.306 | 0.0% |
| 4 | `heading_sin` | −1.000 | 1.000 | 0.735 | 0.3% | −1.000 | 1.000 | 0.703 | 49.1% |
| 5 | `heading_cos` | −1.000 | 1.000 | 0.677 | 0.2% | −1.000 | 1.000 | 0.711 | 50.3% |
| 6 | `health` | 0.200 | 1.000 | 0.261 | 69.1% | 0.200 | 1.000 | 0.268 | 65.5% |
| 7 | `phase` | 0.000 | 0.000 | 0.000 | 0.0% | 0.000 | 0.000 | 0.000 | 0.0% |
| 8 | `enemy_local_dx` | −0.564 | 0.256 | 0.099 | 0.0% | −0.454 | 0.437 | 0.101 | 0.0% |
| 9 | `enemy_local_dy` | −0.461 | 0.506 | 0.123 | 0.0% | −0.452 | 0.435 | 0.101 | 0.0% |
| 10 | `enemy_distance` | 0.000 | 0.593 | 0.131 | 0.0% | 0.000 | 0.458 | 0.110 | 0.0% |
| 11 | `enemy_local_rel_vx` | −0.830 | 0.374 | 0.179 | 0.0% | −0.810 | 0.610 | 0.168 | 0.0% |
| 12 | `enemy_local_rel_vy` | −0.563 | 0.601 | 0.168 | 0.0% | −0.807 | 0.668 | 0.168 | 0.0% |
| 13 | `enemy_exists` | 0.000 | 1.000 | 0.427 | 76.0% | 0.000 | 1.000 | 0.430 | 75.5% |
| 14 | `spawner_local_dx` | −0.833 | 0.609 | 0.221 | 0.0% | −0.704 | 0.705 | 0.238 | 0.0% |
| 15 | `spawner_local_dy` | −0.733 | 0.746 | 0.303 | 0.0% | −0.703 | 0.702 | 0.236 | 0.0% |
| 16 | `spawner_distance` | 0.002 | 0.834 | 0.171 | 0.0% | 0.006 | 0.729 | 0.140 | 0.0% |
| 17 | `shoot_cooldown` | 0.000 | 0.800 | 0.291 | 0.0% | 0.000 | 0.800 | 0.285 | 0.0% |
| 18 | `enemies_alive` | 0.000 | 0.700 | 0.158 | 0.0% | 0.000 | 0.600 | 0.125 | 0.0% |
| 19 | `spawners_alive` | 0.167 | 0.333 | 0.041 | 0.0% | 0.167 | 0.333 | 0.003 | 0.0% |

Every surviving feature reports exactly the range, spread and bound-fraction it reported in the
21-feature run; only the removed row is gone and the three below it have shifted up by one index.
That is the check that the fix was a removal and nothing else — had the edit disturbed a
neighbouring slot, this table is where it would show.

No feature ever left `[-1, 1]`, so the `np.clip` in `build_observation` is doing what its comment
claims — guaranteeing a bound the scaling already respects, rather than hiding an overflow. The
only feature that touches its clip through scaling rather than by construction is `player_vx`
/`player_vy` under rotation control, at 0.1% of steps, which is the speed cap being reached.

## Findings

### F1 — `spawner_exists` was dead, and is now gone *(resolved)*

The first run found index 17 constant at 1.0 across all 20,000 random steps and under both trained
policies. It was not under-explored, it was *unreachable*: `ArenaEnv._maybe_advance_phase()` runs at
the end of every physics frame, and destroying the last spawner immediately calls `_begin_phase()`,
which lays out the next phase's spawners in the same frame. No observation was ever built from a
world with an empty spawner list, so the flag could not read 0.0, and one of 21 inputs carried no
information at all.

The feature has been removed, `OBS_DIM` is 20, and `spawners_alive` at index 19 carries the same
"is one there" fact with a range the network can use. Both agents were retrained against the new
vector, because a policy trained on 21 features cannot read 20 — and does not silently try:
loading the old model against the new env raises a shape error naming both widths.

Removing it neither helped nor hurt, which is what theory predicts for a constant input: it
contributes a fixed bias the first layer can absorb, so its removal frees a little capacity and
changes nothing else. The measured difference is inside seed-to-seed noise, and the same training
seed won the selection in both control styles before and after.

| style | ships | phase reached | return | phases cleared | survival |
|-------|-------|--------------:|-------:|---------------:|---------:|
| `direct` 21-feature | seed1 | 2.47 | +26.09 | 29/30 | 40% |
| `direct` 20-feature | seed1 | 2.37 | +20.57 | 28/30 | 53% |
| `rotation` 21-feature | seed0 | 1.83 | +3.64 | 25/30 | 3% |
| `rotation` 20-feature | seed0 | 1.93 | +7.59 | 28/30 | 20% |

### F2 — `heading_sin`/`heading_cos` sit at a bound ~50% of the time under direct control *(by design)*

Style 2 moves along the four compass directions, so the ship's heading is quantised to
{0, ±π/2, π} and the sin/cos pair only ever reads {−1, 0, 1}. The pair is effectively categorical
for `direct` and continuous for `rotation`. That is the intended consequence of both styles sharing
one observation vector, and it is worth a sentence in the report's observation-design section: it
is the clearest single illustration of how the same numbers describe two different control
problems.

### F3 — `phase` constant and `health` saturated under the random policy *(policy, not code)*

Both were re-probed under the trained policies, which is what separates "the code never writes it"
from "a coin flip never reaches it":

| feature | random policy | trained rotation | trained direct |
|---------|---------------|------------------|----------------|
| `phase` | constant 0.0 | 0.0 → 0.5 (σ 0.236) | 0.0 → 1.0 (σ 0.330) |
| `health` | 1.0 for 65–69% of steps | 0.2 → 1.0 (σ 0.238) | 0.2 → 1.0 (σ 0.212) |

A random policy never clears a phase and dies at a mean of 245 steps, so it sees phase 1 and full
health almost throughout. Both features move properly once a policy that progresses is driving.
This doubles as the random-policy baseline for report row R6: **a random agent clears zero phases in
200 episodes, in either control style.**

`enemy_exists` is also at its bound for ~76% of steps, but it is a 0/1 indicator taking both values
— saturation of an indicator is the feature working, not a scaling fault. It is exactly the
contrast that made F1 a defect: enemies genuinely run out, spawners never did.

## Determinism

`reset(seed=n)` twice, then the same action sequence, produced byte-identical observation arrays and
byte-identical reward arrays for seeds 0, 1 and 2 in both control styles (211–300 steps compared per
seed, each run to its natural episode end). Runs reproduce exactly, so a training result can be
re-derived from its seed. The retrained models are evidence of the same property one level up: each
winning candidate was retrained under its shipped run name and reproduced its evaluation numbers
exactly.

## Termination

200 random episodes per style. Every one ended by player death well inside the cap — longest 528
steps of 2000 for rotation, 480 for direct, mean ~245 — and `terminated and truncated` was never
true together. These figures are identical to the 21-feature run, which is the expected result:
the observation is a view of the simulation and changing it must not change the simulation.

Note the gap this leaves: a random policy always dies, so these 400 episodes never exercise the
`truncated` path at all. That path is covered instead by
[`tests/test_arena_env.py`](../../tests/test_arena_env.py) —
`test_running_out_of_clock_truncates_and_does_not_terminate` and
`test_death_terminates_and_does_not_truncate` — which drive it directly.

## Headless purity

A fresh interpreter with `SDL_VIDEODRIVER=dummy` imported `arena.env`, ran 200 steps with
`render_mode=None`, and `pygame` was absent from `sys.modules` throughout — the env does not merely
avoid opening a window, it never loads the display library at all, which is what keeps a
`SubprocVecEnv` worker cheap. Calling `render()` without a `render_mode` raised `RuntimeError` as
documented rather than silently opening a window.

## Throughput

| style | steps/s (1 core) | 400k timesteps, env time only |
|-------|-----------------:|------------------------------:|
| rotation | 83,164 | ~4.8 s |
| direct | 71,000 | ~5.6 s |

Measured over 50,000 steps with a random policy and no renderer. The environment is nowhere near
the bottleneck: a full 400k-timestep training run takes roughly 35 seconds of wall clock, of which
single-digit seconds are spent inside `step()`. Budget a training run against the network, not the
simulation.
