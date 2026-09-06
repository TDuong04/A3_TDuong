# Arena control-scheme comparison

Generated 2026-09-06T11:52:30+00:00 by `python -m eval.play_arena --style both --no-window`.

Deterministic policy (`deterministic=True`), both styles evaluated on the same seed sequence so they meet the same arenas.

| Style | Episodes | Return | Phase reached | Phases cleared | Spawners | Enemies | Survival | Steps |
|-------|---------:|-------:|--------------:|---------------:|---------:|--------:|---------:|------:|
| `direct` | 30 | +26.09 ± 19.01 | 2.47 (best 4) | 29/30 | 4.57 | 10.87 | 40% | 1433 |
| `rotation` | 30 | +3.64 ± 6.97 | 1.83 (best 2) | 25/30 | 1.83 | 3.33 | 3% | 505 |
