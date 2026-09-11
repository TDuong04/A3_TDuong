# Arena control-scheme comparison

Generated 2026-09-11T08:42:30+00:00 by `python -m eval.play_arena --style both --no-window --mechanics --episodes 30 --seed 0`.

Deterministic policy (`deterministic=True`), both styles evaluated on the same seed sequence so they meet the same arenas.

Rules: shield pickups and elite chargers.

| Style | Policy | Episodes | Return | Phase reached | Phases cleared | Spawners | Enemies | Survival | Steps | Shields collected | Hits blocked | Elite kills |
|-------|--------|---------:|-------:|--------------:|---------------:|---------:|--------:|---------:|------:|------------------:|-------------:|------------:|
| `direct` | `ppo` | 30 | +29.59 ± 24.21 | 2.17 (best 3) | 28/30 | 2.90 | 25.83 | 10% | 1107 | 0.23 | 0.23 | 0.23 |
| `rotation` | `ppo` | 30 | +30.72 ± 21.06 | 2.77 (best 4) | 30/30 | 5.37 | 7.13 | 20% | 1054 | 1.77 | 1.67 | 0.80 |
