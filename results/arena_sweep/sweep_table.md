# Arena hyperparameter sweep (A3-013, rubric J3)

Generated 2026-09-05T23:42:20 from commit `70b1740` by `python -m train.sweep_arena`.

Control style `direct`, 100,000 timesteps per exploratory run, one axis varied at a time from the baseline below. Each row is the mean over that run's last 100 episodes. Ranked by mean phase reached, then by mean episode reward.

## Baseline

| Parameter | Value |
|-----------|-------|
| `algorithm` | PPO |
| `n_envs` | 8 |
| `learning_rate` | 0.0003 |
| `n_steps` | 2048 |
| `batch_size` | 512 |
| `gamma` | 0.99 |
| `ent_coef` | 0.005 |
| `net_arch` | 64x64 |
| `seed` | 0 |

## Stage 1 — one axis at a time

| Rank | Config | Phase reached | Ep reward | Spawners | Enemies | Damage | Ep len | Flag |
|-----:|--------|--------------:|----------:|---------:|--------:|-------:|-------:|------|
| 1 | `n_steps=512` | 1.040 | -11.15 | 0.17 | 3.07 | 5.00 | 297 |  |
| 2 | `gamma=0.95` | 1.030 | -10.04 | 0.26 | 4.51 | 5.00 | 365 |  |
| 3 | `learning_rate=0.001` | 1.030 | -10.20 | 0.26 | 5.96 | 5.00 | 526 |  |
| 4 | `net_arch=128x128` | 1.020 | -9.04 | 0.25 | 6.77 | 5.00 | 476 | ⚠ reward hacking |
| 5 | `baseline` | 1.010 | -12.14 | 0.11 | 2.67 | 5.00 | 296 |  |
| 6 | `ent_coef=0.0` | 1.000 | -12.21 | 0.06 | 3.23 | 5.00 | 324 |  |
| 7 | `n_steps=4096` | 1.000 | -12.46 | 0.08 | 2.36 | 5.00 | 272 |  |
| 8 | `ent_coef=0.01` | 1.000 | -12.84 | 0.01 | 2.59 | 5.00 | 298 |  |
| 9 | `learning_rate=0.0001` | 1.000 | -13.04 | 0.06 | 1.55 | 5.00 | 239 |  |

Configs where episode reward rose above the baseline while mean phase reached did not (gain ≤ 0.02) are flagged as suspected reward hacking: `net_arch=128x128`. Reward bought without phase progression is the shaping terms being farmed. The flag annotates the row rather than disqualifying it — ranking on phase before reward is what keeps a farmer off the top of the table.

## Stage 2 — top configs across seeds

The best 2 configs re-run on seeds 0, 1, 2. A one-seed win is not a result.

| Rank | Config | Seeds | Phase reached | Ep reward | Spawners | Damage |
|-----:|--------|-------|--------------:|----------:|---------:|-------:|
| 1 | `n_steps=512` | 0, 1, 2 | 1.027 ± 0.012 | -10.64 ± 1.26 | 0.18 | 5.00 |
| 2 | `gamma=0.95` | 0, 1, 2 | 1.020 ± 0.008 | -10.49 ± 0.34 | 0.20 | 5.00 |

## Winner

`n_steps=512` — mean phase reached 1.027, mean episode reward -10.64 across 3 seeds.

Retrained at the full budget of 400,000 timesteps and saved to `models/ppo_direct_sweep.zip` (30s). Final-run means over its last 100 episodes: phase reached 1.290, episode reward 2.50.

This is **not** promoted to `models/ppo_direct.zip` automatically — when the axes separate by less than their seed noise, the ranking above is not evidence that the tuned model will win. The head-to-head below settles it under `deterministic=True`.

## Promote or not — head-to-head

Both models over the same 30 seeded episodes of control style `direct`, acting with `deterministic=True`.

| Role | Model | Return | Phase reached | Phases cleared | Spawners |
|------|-------|-------:|--------------:|---------------:|---------:|
| challenger | `models/ppo_direct_sweep.zip` | -4.16 ± 10.05 | 1.00 | 0/30 | 0.13 |
| incumbent | `models/ppo_direct.zip` | +17.14 ± 19.00 | 1.93 | 22/30 | 2.60 |

**Verdict: keep the incumbent.**
