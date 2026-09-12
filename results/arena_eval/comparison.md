# Arena control-scheme comparison

Generated 2026-09-12T22:52:17+00:00 by `python -m eval.play_arena --style direct --no-window --episodes 30`.

Deterministic policy (`deterministic=True`), both styles evaluated on the same seed sequence so they meet the same arenas.

The `rotation` row below is carried over from an earlier run, not remeasured by the command above -- see its own `Episodes` column for the sample size it was actually measured at.

| Style | Policy | Episodes | Return | Phase reached | Phases cleared | Spawners | Enemies | Survival | Steps |
|-------|--------|---------:|-------:|--------------:|---------------:|---------:|--------:|---------:|------:|
| `direct` | `ppo` | 30 | +72.38 ± 18.50 | 3.27 (best 4) | 30/30 | 7.70 | 35.73 | 80% | 1936 |
| `rotation` | `ppo` | 30 | +357.69 ± 10.65 | 13.20 (best 14) | 30/30 | 47.30 | 21.70 | 97% | 1999 |
