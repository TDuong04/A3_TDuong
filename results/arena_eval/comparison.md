# Arena control-scheme comparison

Generated 2026-09-11T15:56:56+00:00 by `python -m eval.play_arena --style both --no-window`.

Deterministic policy (`deterministic=True`), both styles evaluated on the same seed sequence so they meet the same arenas.

| Style | Policy | Episodes | Return | Phase reached | Phases cleared | Spawners | Enemies | Survival | Steps |
|-------|--------|---------:|-------:|--------------:|---------------:|---------:|--------:|---------:|------:|
| `direct` | `ppo` | 3 | +18.29 ± 15.72 | 2.33 (best 3) | 3/3 | 3.33 | 12.00 | 33% | 1521 |
