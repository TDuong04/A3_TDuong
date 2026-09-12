# Arena control-scheme comparison

Generated 2026-09-12T22:50:20+00:00 by `python -m eval.play_arena --style direct --no-window --random --episodes 30`.

**Random-policy baseline, not a trained result.** Actions are drawn uniformly from the action space. These are the numbers chance alone produces on the same seeded arenas, which is what makes the trained table above them mean anything.

The `rotation` row below is carried over from an earlier run, not remeasured by the command above -- see its own `Episodes` column for the sample size it was actually measured at.

| Style | Policy | Episodes | Return | Phase reached | Phases cleared | Spawners | Enemies | Survival | Steps |
|-------|--------|---------:|-------:|--------------:|---------------:|---------:|--------:|---------:|------:|
| `direct` | `random` | 30 | -12.16 ± 6.55 | 1.07 (best 2) | 2/30 | 0.47 | 8.30 | 3% | 883 |
| `rotation` | `random` | 30 | -20.81 ± 2.30 | 1.00 (best 1) | 0/30 | 0.13 | 2.60 | 0% | 907 |
