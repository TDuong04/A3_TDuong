# Arena control-scheme comparison

Generated 2026-09-11T15:52:23+00:00 by `python -m eval.play_arena --style both --no-window --random`.

**Random-policy baseline, not a trained result.** Actions are drawn uniformly from the action space. These are the numbers chance alone produces on the same seeded arenas, which is what makes the trained table above them mean anything.

| Style | Policy | Episodes | Return | Phase reached | Phases cleared | Spawners | Enemies | Survival | Steps |
|-------|--------|---------:|-------:|--------------:|---------------:|---------:|--------:|---------:|------:|
| `direct` | `random` | 3 | -12.49 ± 1.30 | 1.00 (best 1) | 0/3 | 0.00 | 2.67 | 0% | 265 |
