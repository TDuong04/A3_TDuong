# Arena hyperparameter sweep (A3-013, rubric J3)

Generated 2026-09-12T00:28:45 from commit `fc2cf4b` by `python -m train.sweep_arena`.

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
| 1 | `n_steps=512` | 1.100 | -4.87 | 0.52 | 10.19 | 5.00 | 616 |  |
| 2 | `learning_rate=0.001` | 1.070 | -8.61 | 0.34 | 7.02 | 5.00 | 553 |  |
| 3 | `baseline` | 1.050 | -9.23 | 0.27 | 6.34 | 5.00 | 492 |  |
| 4 | `gamma=0.95` | 1.040 | -10.01 | 0.20 | 5.57 | 5.00 | 448 |  |
| 5 | `net_arch=128x128` | 1.030 | -8.29 | 0.30 | 7.39 | 5.00 | 498 | ⚠ reward hacking |
| 6 | `n_steps=4096` | 1.030 | -10.91 | 0.17 | 5.09 | 5.00 | 465 |  |
| 7 | `ent_coef=0.0` | 1.010 | -9.79 | 0.21 | 6.40 | 5.00 | 484 |  |
| 8 | `ent_coef=0.01` | 1.010 | -10.57 | 0.16 | 5.71 | 5.00 | 468 |  |
| 9 | `learning_rate=0.0001` | 1.000 | -12.95 | 0.05 | 1.85 | 5.00 | 255 |  |

Configs where episode reward rose above the baseline while mean phase reached did not (gain ≤ 0.02) are flagged as suspected reward hacking: `net_arch=128x128`. Reward bought without phase progression is the shaping terms being farmed. The flag annotates the row rather than disqualifying it — ranking on phase before reward is what keeps a farmer off the top of the table.

## Stage 2 — top configs across seeds

The best 2 configs re-run on seeds 0, 1, 2. A one-seed win is not a result.

| Rank | Config | Seeds | Phase reached | Ep reward | Spawners | Damage |
|-----:|--------|-------|--------------:|----------:|---------:|-------:|
| 1 | `n_steps=512` | 0, 1, 2 | 1.060 ± 0.028 | -7.06 ± 2.09 | 0.40 | 5.00 |
| 2 | `learning_rate=0.001` | 0, 1, 2 | 1.057 ± 0.012 | -9.50 ± 0.84 | 0.29 | 4.99 |

## Winner

`n_steps=512` — mean phase reached 1.060, mean episode reward -7.06 across 3 seeds.

Retrained at the full budget of 400,000 timesteps and saved to `models/ppo_direct_sweep.zip` (30s). Final-run means over its last 100 episodes: phase reached 1.900, episode reward 15.26.

This is **not** promoted to `models/ppo_direct.zip` automatically — when the axes separate by less than their seed noise, the ranking above is not evidence that the tuned model will win. The head-to-head below settles it under `deterministic=True`.

## Promote or not — head-to-head

Both models over the same 30 seeded episodes of control style `direct`, acting with `deterministic=True`.

| Role | Model | Return | Phase reached | Phases cleared | Spawners |
|------|-------|-------:|--------------:|---------------:|---------:|
| challenger | `models/ppo_direct_sweep.zip` | +10.77 ± 16.15 | 1.27 | 8/30 | 1.00 |
| incumbent | `models/ppo_direct.zip` | +20.57 ± 16.71 | 2.37 | 28/30 | 3.93 |

**Verdict: keep the incumbent.**
