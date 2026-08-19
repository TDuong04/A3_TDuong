---
id: A3-001
title: Implement the gridworld environment core
type: feature
status: done
priority: P0
rubric: A
points_at_risk: 2.0
area: gridworld
github: https://github.com/TDuong04/A3_TDuong/issues/1
owner: unassigned
blocks: [A3-002, A3-003]
blocked_by: []
created: 2026-08-19
updated: 2026-08-20
---

# A3-001 — Implement the gridworld environment core

## Context

The env is the foundation for every Part I ticket and nothing else can start without it.
Contract is in the `gridworld/env.py` docstring.

The state key is the decision that matters: position alone is not Markov on levels 2-6, because two
visits to the same tile differ by whether the key is held and which collectibles remain. Getting
this wrong makes the chest levels unlearnable in a way that looks like a hyperparameter problem.

## Acceptance criteria

- [ ] `step()` follows the brief's order: blocked move, then death check, then collect, then move monsters, then the second death check
- [ ] Rocks block with no displacement, no reward change and no penalty
- [ ] Chest pays +2 only while the key is held; otherwise it stays put and pays nothing
- [ ] Episode ends when all collectibles are obtained or the agent dies
- [ ] State key includes held-key and remaining collectibles
- [ ] No pygame import anywhere in this module
- [ ] `info` carries `died`, `collected` and `steps`

## Notes

Files: `gridworld/env.py`. Constants already frozen in `gridworld/constants.py`.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/1
- 2026-08-20 — verified and committed in 232877c; evaluator killed 10/12 mutants, two test gaps since closed and proven to bite
