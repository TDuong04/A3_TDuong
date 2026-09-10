# Arena control-scheme comparison

Generated 2026-09-09T10:38:03+00:00 by `python -m eval.play_arena --style both --no-window`.

Deterministic policy (`deterministic=True`), both styles evaluated on the same seed sequence so they meet the same arenas.

| Style | Policy | Episodes | Return | Phase reached | Phases cleared | Spawners | Enemies | Survival | Steps |
|-------|--------|---------:|-------:|--------------:|---------------:|---------:|--------:|---------:|------:|
| `direct` | `ppo` | 30 | +20.57 ± 16.71 | 2.37 (best 3) | 28/30 | 3.93 | 8.87 | 53% | 1518 |
| `rotation` | `ppo` | 30 | +7.59 ± 5.33 | 1.93 (best 2) | 28/30 | 2.27 | 5.43 | 20% | 825 |
