# Arena control-scheme comparison

Generated 2026-09-12T10:01:12+00:00 by `python -m eval.play_arena --style both --no-window --episodes 30`.

Deterministic policy (`deterministic=True`), both styles evaluated on the same seed sequence so they meet the same arenas.

| Style | Policy | Episodes | Return | Phase reached | Phases cleared | Spawners | Enemies | Survival | Steps |
|-------|--------|---------:|-------:|--------------:|---------------:|---------:|--------:|---------:|------:|
| `direct` | `ppo` | 30 | +36.24 ± 10.81 | 2.77 (best 4) | 29/30 | 5.13 | 15.33 | 93% | 1978 |
| `rotation` | `ppo` | 30 | +29.20 ± 14.92 | 2.50 (best 3) | 29/30 | 4.87 | 12.83 | 73% | 1763 |
