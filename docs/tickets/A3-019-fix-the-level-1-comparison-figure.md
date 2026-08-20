---
id: A3-019
title: Fix the level 1 comparison figure for report and video use
type: bug
status: done
priority: P1
rubric: C
points_at_risk: 3.0
area: gridworld
github: https://github.com/TDuong04/A3_TDuong/issues/19
owner: unassigned
blocks: [A3-014, A3-015]
blocked_by: []
created: 2026-08-20
updated: 2026-08-20
---

# A3-019 — Fix the level 1 comparison figure for report and video use

## Context

`results/compare_level1_seed0.png` is the primary evidence for rubric row C, and it goes into both
the report and the video. The A3-004 evaluation found two defects in the figure itself. The
underlying result is correct and verified — this is purely about the artifact communicating it.

**The occlusion is the serious one.** The greedy route is drawn at `zorder=5, linewidth=3.0` over
the policy arrows at `zorder=3`, so Q-learning's row-7 arrows are hidden underneath the route line.
Those arrows are the visual core of the claim: they run parallel to the fire, which is *why*
Q-learning is the risky policy. A marker looking at the left panel cannot see the thing the panel
exists to show. The evaluator had to confirm it from the Q-table instead. SARSA's panel is
unaffected, because its route leaves row 7 visible — so the defect silently weakens exactly one
half of a side-by-side comparison.

**Legibility.** The figure is 1800x960 px, 12.0x6.4 in at 150 dpi. Dropped into a 6.5-inch text
column it scales to 54%, taking panel titles from 10pt to 5.4pt, the caption from 9pt to 4.9pt and
tick labels from 7pt to 3.8pt. Legible at full size, marginal below about 80%. The report has a hard
10-page limit, so the figure will be under pressure to shrink.

## Acceptance criteria

- [x] Route line no longer hides the policy arrows — thin it, drop its alpha, or reorder zorder so
      both read at once. Verify by reopening the PNG and confirming Q-learning's row-7 arrows are
      visible.
- [x] Font sizes raised so the figure stays legible at 60% scale, or the figure re-proportioned for
      a full-width landscape placement
- [x] Regenerated for seeds 0 and 1, and the markdown regenerated alongside so prose and figure
      cannot drift

## Notes

Two prose corrections belong with this, both in `train/train_gridworld.py`'s generated markdown:

- "the shortest safe route runs along row 7" implies uniqueness. There are **two** 11-step optima —
  row 7 above the fire and row 9 below it.
- Describe SARSA as "13 steps via row 6", not "never adjacent to the fire". On seed 8 SARSA's route
  descends a column early and clips one fire-adjacent cell, raising its death rate to 4.2% against
  2.2-2.4% elsewhere. Still 2.6x safer than Q-learning, so the headline holds — but the strongest
  phrasing does not.

## Log

- 2026-08-20 — raised from the A3-004 evaluation; the result is verified, the artifact is not
  communicating it
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/19
- 2026-08-20 — fixed. The route is now a wide translucent corridor at zorder 2.0 with a 1px centre
  line at 2.5, both below the arrow field at zorder 3, so nothing the panel exists to show can be
  covered. Confirmed by reopening the regenerated PNG: Q-learning's row-7 arrows read clearly,
  pointing right, parallel to the fire, inside its own route band. Panel titles 10 -> 13pt, legend
  8 -> 11, caption 9 -> 11, suptitle 13 -> 16, tick labels 7 -> 10, which holds at the 60% scale a
  10-page report will force. Both prose corrections applied: the optimum is now described as two
  tied 11-step routes (row 7 above the fire, row 9 below), and a paragraph now states the SARSA
  result as route length plus rows used and explicitly retires the stronger "never near the fire"
  phrasing, since on some seeds its route clips one fire-adjacent cell. Regenerated for seeds 0 and
  1; both still report ROUTES DIFFER. Closed.
