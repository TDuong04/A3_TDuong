# Arena control-scheme comparison

Generated 2026-09-12T10:11:38+00:00 by `python -m eval.play_arena --style both --no-window --episodes 30`.

Deterministic policy (`deterministic=True`), both styles evaluated on the same seed sequence so they meet the same arenas.

| Style | Policy | Episodes | Return | Phase reached | Phases cleared | Spawners | Enemies | Survival | Steps |
|-------|--------|---------:|-------:|--------------:|---------------:|---------:|--------:|---------:|------:|
| `direct` | `ppo` | 30 | +56.55 ± 14.40 | 2.73 (best 3) | 29/30 | 5.27 | 34.57 | 100% | 2000 |
| `rotation` | `ppo` | 30 | +228.10 ± 108.69 | 8.13 (best 12) | 26/30 | 27.40 | 45.13 | 47% | 1625 |
