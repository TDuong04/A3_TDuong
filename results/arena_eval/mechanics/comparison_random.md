# Arena control-scheme comparison

Generated 2026-09-11T08:42:23+00:00 by `python -m eval.play_arena --style both --no-window --random --mechanics --episodes 30 --seed 0`.

**Random-policy baseline, not a trained result.** Actions are drawn uniformly from the action space. These are the numbers chance alone produces on the same seeded arenas, which is what makes the trained table above them mean anything.

Rules: shield pickups and elite chargers.

| Style | Policy | Episodes | Return | Phase reached | Phases cleared | Spawners | Enemies | Survival | Steps | Shields collected | Hits blocked | Elite kills |
|-------|--------|---------:|-------:|--------------:|---------------:|---------:|--------:|---------:|------:|------------------:|-------------:|------------:|
| `direct` | `random` | 30 | -12.85 ± 1.44 | 1.00 (best 1) | 0/30 | 0.03 | 1.90 | 0% | 242 | 0.00 | 0.00 | 0.00 |
| `rotation` | `random` | 30 | -14.34 ± 1.76 | 1.00 (best 1) | 0/30 | 0.10 | 0.13 | 0% | 248 | 0.07 | 0.07 | 0.00 |
