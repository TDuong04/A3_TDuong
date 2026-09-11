---
id: A3-018
title: Ship creativity features beyond the brief
type: feature
status: open
priority: P1
rubric: Creativity
points_at_risk: 5.0
area: arena
github: https://github.com/TDuong04/A3_TDuong/issues/18
owner: unassigned
blocks: []
blocked_by: [A3-014]
created: 2026-08-19
updated: 2026-09-11
---

# A3-018 — Ship creativity features beyond the brief

## Context

The Creativity row requires additions beyond the brief. Observation overlays and visual
polish support the demonstration but do not establish two additional gameplay systems.

## Acceptance criteria

- [x] Observation overlay shipped and toggleable (shared with A3-009; supporting evidence)
- [x] Two additional mechanics: collectible one-hit shields and elite chargers with a distinct
      pursuit/wind-up/charge/recovery attack pattern; enable with `--mechanics`
- [x] Existing `eval/play_arena.py --human` uses the same action space and mechanics as training
- [x] Muzzle flash, hit flicker, explosion particles and screen shake remain renderer-owned;
      seeded rendered/unrendered tests cover both baseline and new mechanics
- [ ] Final report contains a paragraph naming each item and explaining why it exceeds the brief
      (section draft available in `report/creativity.md`; integration belongs to A3-014)

## Notes

Purely visual effects must never touch simulation state, or evaluation stops matching training. Ideas beyond these are welcome — raise them as separate P3 tickets.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/18
- 2026-09-08 — Integrated creativity features with completed A3-009/A3-012 on main: retain HUD, phase, observation, policy and trained/human playback; add renderer-owned combat feedback and pilot-perception compass. See docs/arena-creativity.md for integration differences, controls and originality rationale.
- 2026-09-08 — Verification: 65 focused and 721 full-suite tests passed; both shipped PPO models reached phase 2 at seed 0; offscreen playback inspected for both styles. Desktop keyboard testing remains manual.

- 2026-09-10 — Restored the stricter two-systems criteria after review. Implemented shield
  pickups and elite chargers in optional mechanics mode for both human and agent play.
  Added 36-feature observations, training/evaluation integration, distinct artifact directories,
  and gameplay/invariance/model compatibility tests. See `docs/arena-mechanics.md`.
  Reopened pending final report integration; existing trained models remain baseline models.

- 2026-09-10 — Integrated latest main (`c65acb2`) on the A3-018 branch, preserving its
  retrained 20-feature PPO models. Mechanics mode now has 36 features. Kept A3-030
  debugging separate. Verification: 286 arena tests passed; both shipped baseline
  models loaded and rendered, and mechanics PPO smoke training/save/reload passed
  for both control styles. Full-suite rerun was interrupted; no full-suite result
  is claimed for this integration.

- 2026-09-11 — Branch `feat/a3-018-trained-mechanics-and-timelapse` (off A3-031 `6b0bfde`).
  Two additions, both measured on seeds 0-29 beside a random policy on the same arenas:
  - **Trained mechanics agents.** PPO per style on the shield/elite rules, 2M steps, seed 0,
    commit `6b0bfde` (`logs/ppo_*_mechanics/run.json`), in `models/mechanics/`. Phase 1
    cleared 28/30 (direct) and 30/30 (rotation) against 0/30 random. The rotation agent takes
    1.77 shields and absorbs 1.67 hits per episode, and kills 0.80 elites; the direct agent
    mostly ignores shields (0.23). Evidence: `results/arena_eval/mechanics/`.
  - **Learning time-lapse.** `eval.play_arena --model <file>` replays any checkpoint with its
    training-step count (read from `num_timesteps`) and filename in the HUD. `--timelapse`
    tabulates every 50k snapshot of the shipped runs: direct clears phase 1 on 0/30 at 100k,
    27/30 at 200k, 30/30 at 300k. The final rows reproduce `comparison.md` exactly (+20.57
    28/30, +7.59 28/30). A checkpoint run writes to `results/arena_eval/checkpoints/` and can
    never overwrite the row I table. `.gitignore` now tracks the 16 `*_shipped_*` snapshots the
    time-lapse plays. The comparison table's reproduce line now carries `--episodes N --seed S`,
    which also meets A3-032's third criterion (its other three remain open there).
  - `report/creativity.md` rewritten as the final section draft with these numbers and their
    limits (2M vs 400k budget, so no mechanics-vs-baseline claim; 36 features above the
    appendix's 10-30 guidance). The report criterion stays open until A3-014 integrates it.
  - Verification: full non-slow suite passed in the worktree after these changes, including 19
    new tests (checkpoint loading, stale 21-feature checkpoint refused, row I table protected,
    provenance and mechanics columns, time-lapse ordering and de-duplication, HUD label drawn
    and clear of the fields).
