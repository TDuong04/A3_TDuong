---
id: A3-012
title: Train both control agents and tabulate the comparison
type: feature
status: open
priority: P0
rubric: I
points_at_risk: 4.0
area: eval
part: 2
github: https://github.com/TDuong04/A3_TDuong/issues/12
owner: member-1
blocks: [A3-014, A3-015]
blocked_by: [A3-011, A3-022, A3-023]
created: 2026-08-19
updated: 2026-08-29
---

# A3-012 — Train both control agents and tabulate the comparison

## Context

Row I is a checklist row: two styles, two saved models, working eval scripts. None of it
depends on the agent being especially good, so it is 4 points of low-risk marks — but only if the
scripts actually run from a clean checkout.

Train direct movement first. It learns faster, so it proves the reward function works before the
harder control style is attempted.

## Acceptance criteria

- [ ] A trained model per style saved in `models/` under that exact folder name
- [ ] Each model loads in `eval/play_arena.py` (A3-023) and runs with `deterministic=True`
- [ ] Both agents clear phase 1 at least once over 5 seeded evaluation episodes
- [ ] The `--episodes 5` summary tabulated for both styles under the same seeds, ready for the
      report's control-set comparison
- [ ] TensorBoard curves for both runs committed under `logs/`

## Notes

The eval script itself is A3-023 and phase-1 tuning is A3-022 — both are prerequisites, so this
ticket is the training run and the comparison numbers only. If an agent still cannot clear phase 1,
re-tune in config rather than adding timesteps; the video requires a visible phase progression.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/12
- 2026-08-29 — eval-script scope moved to A3-023 and phase-1 tuning to A3-022; this ticket is now the training run and the seeded comparison only. Assigned to member-1
