---
id: A3-016
title: Add .gitattributes to normalise line endings
type: chore
status: done
priority: P2
rubric: none
points_at_risk: 0.0
area: infra
part: none
github: https://github.com/TDuong04/A3_TDuong/issues/16
owner: unassigned
blocks: []
blocked_by: []
created: 2026-08-19
updated: 2026-08-20
---

# A3-016 — Add .gitattributes to normalise line endings

## Context

Git currently warns `LF will be replaced by CRLF` on every file. Harmless for one person, but
with three people on possibly different platforms it produces phantom whole-file diffs that hide
real changes in review.

## Acceptance criteria

- [x] `.gitattributes` with `* text=auto eol=lf`
- [x] Binary types (`*.pdf`, `*.zip`, `*.png`) marked binary
- [x] Existing files renormalised in a single dedicated commit

## Notes

Do this before anyone else clones the repo.

## Log

- 2026-08-19 — created during project setup
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/16
- 2026-08-20 - added `.gitattributes` with `* text=auto eol=lf`, plus a binary list covering the
  brief's PDF, every image and audio type, `*.npz` Q-tables, `models/**` (SB3 saves are zip
  archives) and TensorBoard event files. The last two matter specifically because `models/` and
  `logs/` ship inside the submission zip and neither survives being treated as text.
- 2026-08-20 - the third criterion turned out to need no commit: `git add --renormalize .` produced
  an empty diff, so the index was already LF throughout and only the working copy was being
  converted on checkout by `core.autocrlf=true`. `eol=lf` now overrides that per-clone setting, so
  the next person to clone gets LF regardless of how their git is configured. Recorded rather than
  silently ticked, since "renormalised" and "nothing needed renormalising" are different facts.
