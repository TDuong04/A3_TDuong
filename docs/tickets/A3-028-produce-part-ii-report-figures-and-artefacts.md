---
id: A3-028
title: Produce Part II report figures and artefacts
type: task
status: open
priority: P1
rubric: R
points_at_risk: 2.5
area: report
part: 2
github: https://github.com/TDuong04/A3_TDuong/issues/31
owner: member-3
blocks: [A3-014]
blocked_by: [A3-012]
created: 2026-08-29
updated: 2026-08-29
---

# A3-028 — Produce Part II report figures and artefacts

## Context

A rubric audit (2026-08-29) found zero arena artefacts anywhere in the repo: no training curve
exported as a figure, no eval screenshot, no phase-progression capture, no observation-overlay
figure for the observation-design section. A3-012 produces the control-set comparison *numbers*
and the TensorBoard logs, but nothing turns either into a committed figure or a report-ready
write-up — the same gap Part I already closed with `results/comparison_level1_seed0.md` and its
figures.

Without this, A3-014 has no Part II evidence to paste into the report, only numbers in a
TensorBoard UI nobody else can open during grading.

## Acceptance criteria

- [ ] Reward curve (ep_rew_mean) and at least one behavioural curve (e.g. mean phase reached, or
      episode length) exported from TensorBoard as PNGs into `results/`, one pair per control
      style (rotation, direct)
- [ ] A screenshot of the arena mid-combat showing enemies, projectiles and collisions
      simultaneously on screen, saved into `results/`
- [ ] A screenshot with the observation-debug overlay switched on, saved into `results/`, for the
      report's observation-design section
- [ ] A short markdown write-up in `results/` (e.g. `results/comparison_arena_seed0.md`), in the
      same style as `results/comparison_level1_seed0.md` — control-set comparison table, the seed
      used, and a short explanation of what the curves and screenshots show — so A3-014 can paste
      directly from it
- [ ] Every figure and claim in the write-up cites the seed and command that produced it

## Notes

Depends on A3-012's trained models and eval runs existing first. Uses `eval/play_arena.py`
(A3-023) to capture screenshots and the `--episodes` summary, and the TensorBoard logs under
`logs/` (A3-011) for the curve exports. Also cite A3-021's validation results file
(`results/` — see A3-021) for the observation-design section rather than re-deriving feature
ranges. Feeds A3-014 directly, the same way A3-024 feeds it for Part I.

## Log

- 2026-08-29 — created from a rubric audit finding: nothing on the board produces the Part II
  equivalent of Part I's `results/*.md` write-ups and figures; A3-012 covers numbers and
  TensorBoard logs only, not committed figures or screenshots.
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/31
