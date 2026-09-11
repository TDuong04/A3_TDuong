# Learning time-lapse: `direct`

Generated 2026-09-11T08:43:08+00:00 by `python -m eval.play_arena --style direct --timelapse --episodes 30 --seed 0`.

Every saved snapshot of the shipped run (`logs/ppo_direct_shipped/`), played deterministically on the same 30 seeded arenas (seeds 0-29), after a uniform random policy on those arenas as the floor. Replay any row on screen with `python -m eval.play_arena --style direct --model <file>`; the HUD shows the step count read from the network itself.

| Training steps | Model | Return | Phase reached | Phase 1 cleared | Enemies | Survival |
|---------------:|-------|-------:|--------------:|----------------:|--------:|---------:|
| 0 (random policy) | — | -12.85 ± 1.44 | 1.00 (best 1) | 0/30 | 1.90 | 0% |
| 50,000 | `ppo_direct_shipped_50000_steps.zip` | -17.29 ± 1.12 | 1.00 (best 1) | 0/30 | 0.20 | 0% |
| 100,000 | `ppo_direct_shipped_100000_steps.zip` | -18.15 ± 2.90 | 1.00 (best 1) | 0/30 | 0.67 | 27% |
| 150,000 | `ppo_direct_shipped_150000_steps.zip` | -6.59 ± 12.22 | 1.30 (best 2) | 9/30 | 5.70 | 53% |
| 200,000 | `ppo_direct_shipped_200000_steps.zip` | +10.49 ± 8.11 | 2.00 (best 3) | 27/30 | 10.50 | 17% |
| 250,000 | `ppo_direct_shipped_250000_steps.zip` | +15.92 ± 15.02 | 2.20 (best 3) | 26/30 | 8.40 | 53% |
| 300,000 | `ppo_direct_shipped_300000_steps.zip` | +18.62 ± 12.41 | 2.23 (best 3) | 30/30 | 13.07 | 30% |
| 350,000 | `ppo_direct_shipped_350000_steps.zip` | +22.40 ± 12.94 | 2.50 (best 3) | 28/30 | 9.27 | 77% |
| 400,000 | `ppo_direct_shipped_400000_steps.zip` | +23.19 ± 13.25 | 2.47 (best 3) | 30/30 | 10.07 | 63% |
| 409,600 | `ppo_direct.zip` | +20.57 ± 16.71 | 2.37 (best 3) | 28/30 | 8.87 | 53% |
