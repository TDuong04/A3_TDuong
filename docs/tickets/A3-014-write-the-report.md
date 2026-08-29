---
id: A3-014
title: Write the report
type: task
status: open
priority: P0
rubric: R
points_at_risk: 2.5
area: report
part: both
github: https://github.com/TDuong04/A3_TDuong/issues/14
owner: member-3
blocks: [A3-025]
blocked_by: [A3-012, A3-024, A3-028]
created: 2026-08-19
updated: 2026-08-29
---

# A3-014 — Write the report

## Context

Maximum 10 pages including images, no appendix — anything past page 10 is not marked, so the
page budget is a hard design constraint, not a target.

Start it before the final week. Collect figures as they are generated; regenerating a curve later
means retraining.

## Acceptance criteria

- [ ] Both environments described; observation design explained feature by feature
- [ ] Reward design justified per term against `arena/constants.py:46-56` (enemy kill +1, spawner
      kill +5, phase advance +10, damage -0.5, death -10, per-step -0.01): why each weight was
      chosen relative to the others, and what specific reward-hacking behaviour each term guards
      against (e.g. why the step penalty is small enough not to punish exploration but large
      enough to stop the agent hiding, why death outweighs a full phase's kill reward)
- [ ] Hyperparameter exploration shown as a table or plots
- [ ] Control-set comparison with numbers from seeded evaluation runs
- [ ] Originality justification, all student numbers, contribution summary, video link
- [ ] Confirmed at 10 pages or fewer with images included

## Notes

Figures come from `results/`. Every claim needs a seed or a run behind it.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/14
- 2026-08-29 — blocked_by A3-007 dropped (done) and A3-024 added — the Part I sections are drafted there first. Assigned to member-3
- 2026-08-29 — fixed a stale acceptance criterion: it required justifying "why the shaping term is
  potential-based", but the audit found the arena reward in `arena/constants.py:46-56` is a
  weighted event reward, not a potential-based shaping term — the code never implemented the
  design that wording described. Replaced with a criterion requiring per-term justification of
  the actual weights and the reward-hacking each guards against.
- 2026-08-29 — blocked_by A3-028 added — new ticket producing the Part II figures and write-up
  this report needs, mirroring how A3-024 supplies the Part I half.
