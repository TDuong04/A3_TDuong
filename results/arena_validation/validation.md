# Arena environment validation — A3-021

Generated 2026-09-07T11:05:27+00:00, Python 3.14.6, single core.
Raw numbers: [`validation.json`](validation.json).

The harness is deliberately not committed: it is a one-shot stress test, and what the report and
A3-014 need to cite is its *output*, not its source. It stepped `ArenaEnv` under a random policy
for 10,000 steps per control style, ran 200 full episodes per style, and re-probed every suspicious
feature under the trained policies in `models/`.

**Verdict: every acceptance criterion passes.** Two features are named below as dead or saturated.
One of them (`spawner_exists`) is a genuine defect in the observation and needs its own ticket
against `arena/observation.py`; the rest are explained by the policy that produced them, not by the
code.

## Criteria

| # | Criterion | Result |
|---|-----------|--------|
| 1 | Observation `(21,)` float32, finite, within `[-1, 1]` over 10,000 random steps, both styles | **pass** — also `observation_space.contains()` on every step |
| 2 | No constant feature; none at a bound for >50% of steps | **pass with findings** — see below |
| 3 | `reset(seed=n)` twice + same actions → byte-identical obs and rewards, 3 seeds | **pass** — seeds 0, 1, 2, both styles |
| 4 | `Discrete(5)` rotation, `Discrete(6)` direct | **pass** — and both reject an out-of-range action with `ValueError` |
| 5 | 200 episodes end within `MAX_EPISODE_STEPS`; `terminated`/`truncated` never both set | **pass** — longest 528 steps of a 2000 cap, 0 double flags |
| 6 | `render_mode=None` runs under `SDL_VIDEODRIVER=dummy`, opens no window | **pass** — `pygame` is never even imported |
| 7 | Step throughput on one core | **88,039 steps/s** rotation, **74,839 steps/s** direct |
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
| 17 | `spawner_exists` | 1.000 | 1.000 | 0.000 | 100.0% | 1.000 | 1.000 | 0.000 | 100.0% |
| 18 | `shoot_cooldown` | 0.000 | 0.800 | 0.291 | 0.0% | 0.000 | 0.800 | 0.285 | 0.0% |
| 19 | `enemies_alive` | 0.000 | 0.700 | 0.158 | 0.0% | 0.000 | 0.600 | 0.125 | 0.0% |
| 20 | `spawners_alive` | 0.167 | 0.333 | 0.041 | 0.0% | 0.167 | 0.333 | 0.003 | 0.0% |

No feature ever left `[-1, 1]`, so the `np.clip` in `build_observation` is doing what its comment
claims — guaranteeing a bound that the scaling already respects, rather than hiding an overflow.
The only feature that touches its clip through scaling rather than by construction is `player_vx`
/`player_vy` under rotation control, at 0.1% of steps, which is the speed cap being reached.

## Findings

### F1 — `spawner_exists` (index 17) is dead, and structurally so *(defect)*

Constant at 1.0 across all 20,000 random steps and across both trained policies. It is not
under-explored, it is *unreachable*: `ArenaEnv._maybe_advance_phase()` runs at the end of every
physics frame, and destroying the last spawner immediately calls `_begin_phase()`, which lays out
the next phase's spawners in the same frame. No observation is ever built from a world with an
empty spawner list, so index 17 can never read 0.0.

One of 21 inputs therefore carries zero information, and `spawner_local_dx/dy` and
`spawner_distance` never need the zeroing branch that the flag exists to signal. This is a bug
against `arena/observation.py` and should be raised as its own ticket — the fix is to drop the
feature (and `OBS_DIM` with it) or to make it meaningful, not to widen a tolerance. It does not
block training: a constant input is wasted capacity, not a wrong gradient, and both agents trained
through it.

### F2 — `heading_sin`/`heading_cos` sit at a bound ~50% of the time under direct control *(by design)*

Style 2 moves along the four compass directions, so the ship's heading is quantised to
{0, ±π/2, π} and the sin/cos pair only ever reads {−1, 0, 1}. The pair is effectively categorical
for `direct` and continuous for `rotation`. That is the intended consequence of both styles sharing
one observation vector, and it is worth a sentence in the report's observation-design section: it
is the clearest single illustration of how the same 21 numbers describe two different control
problems.

### F3 — `phase` constant and `health` saturated under the random policy *(policy, not code)*

Both were re-probed under the trained policies, which is what separates "the code never writes it"
from "a coin flip never reaches it":

| feature | random policy | trained rotation | trained direct |
|---------|---------------|------------------|----------------|
| `phase` | constant 0.0 | 0.0 → 0.5 (σ 0.237) | 0.0 → 1.0 (σ 0.350) |
| `health` | 1.0 for 65–69% of steps | 0.2 → 1.0 (σ 0.283) | 0.2 → 1.0 (σ 0.234) |

A random policy never clears a phase and dies at a mean of 245 steps, so it sees phase 1 and full
health almost throughout. Both features move properly once a policy that progresses is driving.
This doubles as the random-policy baseline for report row R6: **a random agent clears zero phases in
200 episodes, in either control style.**

`enemy_exists` is also at its bound for ~76% of steps, but it is a 0/1 indicator taking both values
— saturation of an indicator is the feature working, not a scaling fault.

## Determinism

`reset(seed=n)` twice, then the same action sequence, produced byte-identical observation arrays and
byte-identical reward arrays for seeds 0, 1 and 2 in both control styles (211–300 steps compared per
seed, each run to its natural episode end). Runs reproduce exactly, so a training result can be
re-derived from its seed.

## Termination

200 random episodes per style. Every one ended by player death well inside the cap — longest 528
steps of 2000 for rotation, 480 for direct, mean ~245 — and `terminated and truncated` was never
true together.

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
| rotation | 88,039 | ~4.5 s |
| direct | 74,839 | ~5.3 s |

Measured over 50,000 steps with a random policy and no renderer. The environment is nowhere near
the bottleneck for A3-011: at 8 envs, a 400k-timestep run spends single-digit seconds inside
`step()`, so wall-clock time is dominated by the policy update. Budget the training run against the
network, not the simulation.
