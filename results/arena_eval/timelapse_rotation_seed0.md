# Learning time-lapse: `rotation`

Generated 2026-09-11T08:43:26+00:00 by `python -m eval.play_arena --style rotation --timelapse --episodes 30 --seed 0`.

Every saved snapshot of the shipped run (`logs/ppo_rotation_shipped/`), played deterministically on the same 30 seeded arenas (seeds 0-29), after a uniform random policy on those arenas as the floor. Replay any row on screen with `python -m eval.play_arena --style rotation --model <file>`; the HUD shows the step count read from the network itself.

| Training steps | Model | Return | Phase reached | Phase 1 cleared | Enemies | Survival |
|---------------:|-------|-------:|--------------:|----------------:|--------:|---------:|
| 0 (random policy) | — | -14.37 ± 1.61 | 1.00 (best 1) | 0/30 | 0.17 | 0% |
| 50,000 | `ppo_rotation_shipped_50000_steps.zip` | -15.00 ± 0.95 | 1.00 (best 1) | 0/30 | 0.20 | 0% |
| 100,000 | `ppo_rotation_shipped_100000_steps.zip` | -12.28 ± 5.26 | 1.07 (best 2) | 2/30 | 0.87 | 0% |
| 150,000 | `ppo_rotation_shipped_150000_steps.zip` | -11.85 ± 5.13 | 1.10 (best 2) | 3/30 | 1.97 | 3% |
| 200,000 | `ppo_rotation_shipped_200000_steps.zip` | -10.26 ± 7.67 | 1.20 (best 2) | 6/30 | 2.93 | 0% |
| 250,000 | `ppo_rotation_shipped_250000_steps.zip` | +0.89 ± 8.16 | 1.67 (best 2) | 20/30 | 5.33 | 10% |
| 300,000 | `ppo_rotation_shipped_300000_steps.zip` | +2.69 ± 10.45 | 1.73 (best 3) | 21/30 | 5.10 | 20% |
| 350,000 | `ppo_rotation_shipped_350000_steps.zip` | +6.57 ± 5.59 | 1.93 (best 2) | 28/30 | 4.33 | 17% |
| 400,000 | `ppo_rotation_shipped_400000_steps.zip` | +8.67 ± 5.43 | 2.00 (best 3) | 29/30 | 5.40 | 13% |
| 409,600 | `ppo_rotation.zip` | +7.59 ± 5.33 | 1.93 (best 2) | 28/30 | 5.43 | 20% |
