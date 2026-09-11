---
id: A3-032
title: Fix the low default episode count that corrupts the arena comparison table
type: bug
status: open
priority: P1
rubric: I
points_at_risk: 4
area: eval
github: https://github.com/TDuong04/A3_TDuong/issues/56
owner: unassigned
blocks: []
blocked_by: []
created: 2026-09-10
updated: 2026-09-10
---

# A3-032 — Fix the low default episode count that corrupts the arena comparison table

## Context

Found while independently verifying A3-031. Not caused by A3-031 and not part of its scope.

`eval/play_arena.py:529` (as committed on `main` at `6ee89b7`) hardcodes
`parser.add_argument("--episodes", type=int, default=3)`. The one command `README.md:82` documents
for regenerating rubric row I's evidence table does not override it —
`python -m eval.play_arena --style both --no-window     # R6: writes results/arena_eval/` — so
running it exactly as documented writes a 3-episode table. The script's own module docstring
(`eval/play_arena.py:7`) instead shows `--episodes 5 --no-window   # the comparison table`, a third,
different number. Neither documented command matches how the evidence file that ships in the
submission was actually produced.

`results/arena_eval/comparison.md` and `comparison_random.md` (the row I evidence CLAUDE.md names
directly, with `comparison_random.md` as the required chance baseline) are, and always have been,
measured at 30 episodes per style — every trustworthy reproduction in `docs/evaluations/` explicitly
appends `--episodes 30` (`docs/evaluations/2026-09-10-solution-evaluation.md:318`,
`2026-09-06-solution-evaluation.md:313`), so the 30 figure is tribal knowledge living in dated audit
files, not in either place a teammate would actually look before regenerating the table.

This is not hypothetical: this morning's audit
(`docs/evaluations/2026-09-10-solution-evaluation.md:317-321`) reproduced `comparison.md` and
`comparison_random.md` exactly at 30 episodes/style against commit `4fafc46`. As of this ticket, the
same two files plus their three backing JSON files sit **modified and uncommitted** in the working
tree at 3 episodes per style — consistent with someone running the documented command, unmodified,
sometime after that audit. A 3-episode sample changes the headline numbers enough to mislead: direct
return 20.57 ± 16.71 (n=30, committed) versus 29.11 ± 14.37 (n=3, working tree); phases cleared 28/30
versus 3/3. Quoting the small-sample table in the report or video would materially overstate how
reliably the direct-style agent clears phase 2.

Reproduction: on a clean checkout of `main` at `6ee89b7`, run
`SDL_VIDEODRIVER=dummy python -m eval.play_arena --style both --no-window` (seed defaults to 0) and
open the written `results/arena_eval/comparison.md` — its `Episodes` column reads `3`, not `30`.

Priority derivation: **P1**. Row I is worth 4 points (≥3), so this is not P0 by the row-size clause
— but CLAUDE.md's P1 rule also names "a defect degrading evidence quality (a curve, a screenshot, a
demo moment the video needs)" independent of row size, and `comparison.md` is exactly that: the
evidence row I cites, currently sitting degraded and uncommitted.

## Acceptance criteria

- [ ] Running the exact command `README.md:82` documents for row I
      (`python -m eval.play_arena --style both --no-window`) does not silently drop below the
      sample size currently backing the committed `results/arena_eval/comparison.md` (30
      episodes/style) — either raise the default, or make the README/docstring commands state the
      flag actually required to reach it.
- [ ] `eval/play_arena.py`'s own module docstring example for "the comparison table" (currently
      `--episodes 5`) agrees with whatever the real evidence-generation invocation is.
- [ ] The table's "Generated ... by" provenance line (`eval/play_arena.py:353-365`, currently a
      fixed string that never reflects `--episodes`) names the episode count actually used, so a
      future run that does shrink the sample cannot do so invisibly.
- [ ] `results/arena_eval/comparison.md`, `comparison_random.md` and their three backing JSON files
      are restored to (or regenerated at) at least 30 episodes per style and committed.

## Notes

Files likely touched: `eval/play_arena.py` (the `--episodes` default and/or the `command`/header
text built around lines 353-365), `README.md:82`, `results/arena_eval/*`. Prior art: A3-019 fixed a
different row-evidence artifact defect the same way — regenerate, verify by reopening the artifact,
commit. This table also feeds A3-014 (report) and A3-015 (video); not marked `blocked_by` since
neither is otherwise stalled on it, but both should pull from the corrected table, not the
in-progress 3-episode one.

## Log

- 2026-09-10 — created while independently verifying A3-031. Confirmed by reading
  `eval/play_arena.py:529` on `main` at `6ee89b7` (hardcoded `default=3`), by reading `README.md:82`
  and `eval/play_arena.py:7` (neither documents `--episodes 30`), and by diffing the current working
  tree's `results/arena_eval/comparison.md`/`comparison_random.md` and their JSON files against the
  30-episode versions this morning's audit reproduced exactly against `4fafc46`
  (`docs/evaluations/2026-09-10-solution-evaluation.md:317-321`). Left open: not this agent's to fix,
  and out of scope for A3-031.
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/56
