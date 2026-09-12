# Arena control-scheme comparison

Generated 2026-09-12T09:18:48+00:00 by `python -m eval.play_arena --style both --no-window --random --episodes 30`.

**Random-policy baseline, not a trained result.** Actions are drawn uniformly from the action space. These are the numbers chance alone produces on the same seeded arenas, which is what makes the trained table above them mean anything.

| Style | Policy | Episodes | Return | Phase reached | Phases cleared | Spawners | Enemies | Survival | Steps |
|-------|--------|---------:|-------:|--------------:|---------------:|---------:|--------:|---------:|------:|
| `direct` | `random` | 30 | -12.85 ± 1.44 | 1.00 (best 1) | 0/30 | 0.03 | 1.90 | 0% | 242 |
| `rotation` | `random` | 30 | -14.37 ± 1.61 | 1.00 (best 1) | 0/30 | 0.10 | 0.17 | 0% | 254 |
