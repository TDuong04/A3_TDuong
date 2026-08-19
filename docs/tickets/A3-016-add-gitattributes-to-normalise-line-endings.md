---
id: A3-016
title: Add .gitattributes to normalise line endings
type: chore
status: open
priority: P2
rubric: none
points_at_risk: 0.0
area: infra
github: https://github.com/TDuong04/A3_TDuong/issues/16
owner: unassigned
blocks: []
blocked_by: []
created: 2026-08-19
updated: 2026-08-19
---

# A3-016 — Add .gitattributes to normalise line endings

## Context

Git currently warns `LF will be replaced by CRLF` on every file. Harmless for one person, but
with three people on possibly different platforms it produces phantom whole-file diffs that hide
real changes in review.

## Acceptance criteria

- [ ] `.gitattributes` with `* text=auto eol=lf`
- [ ] Binary types (`*.pdf`, `*.zip`, `*.png`) marked binary
- [ ] Existing files renormalised in a single dedicated commit

## Notes

Do this before anyone else clones the repo.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/16
