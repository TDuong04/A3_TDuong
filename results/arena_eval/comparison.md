# Arena control-scheme comparison

Generated 2026-09-06T11:16:18+00:00 by `python -m eval.play_arena --style both --no-window`.

Deterministic policy (`deterministic=True`), both styles evaluated on the same seed sequence so they meet the same arenas.

| Style | Episodes | Return | Phase reached | Phases cleared | Spawners | Enemies | Survival | Steps |
|-------|---------:|-------:|--------------:|---------------:|---------:|--------:|---------:|------:|
| `direct` | 3 | +8.90 ± 13.49 | 1.67 (best 2) | 2/3 | 1.67 | 14.00 | 0% | 760 |
| `rotation` | 3 | +0.77 ± 9.86 | 1.67 (best 2) | 2/3 | 1.33 | 4.33 | 0% | 439 |
