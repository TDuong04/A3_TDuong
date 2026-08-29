---
id: A3-025
title: Package the submission and verify a clean checkout
type: task
status: open
priority: P0
rubric: none
points_at_risk: 0
area: infra
github: https://github.com/TDuong04/A3_TDuong/issues/26
owner: member-3
blocked_by: [A3-012, A3-014, A3-015]
blocks: []
created: 2026-08-29
updated: 2026-08-29
---

# A3-025 — Package the submission and verify a clean checkout

## Context

Nothing on the board covers actually handing the work in, and every failure mode here is
self-inflicted and total: `models/` gitignored so the marker has no agents to run, a
`requirements.txt` that does not install, a README command that references a flag which was
renamed three commits ago. None of it costs marks in a rubric row — it costs the rows the missing
files were evidence for.

Do a full dry run from a fresh clone. Not a skim of the file list.

## Acceptance criteria

- [ ] `git clone` into an empty directory, fresh venv, `pip install -r requirements.txt` succeeds
      on a machine that is not the author's
- [ ] `pytest` passes from that clean checkout
- [ ] Every command in `README.md` runs as written, including both `eval` scripts
- [ ] `models/` and `logs/` are present in the zip and not gitignored; the models in the zip are
      the ones shown in the video
- [ ] `results/` figures cited by the report are all present
- [ ] Report PDF included, 10 pages or fewer, with the video link inside it
- [ ] Zip opened once after building and its contents listed against this checklist

## Notes

Check `.gitignore` for `models/` and `logs/` early — a model committed on submission day is a
model nobody has verified loads.

## Log

- 2026-08-29 — created; no ticket covered submission packaging
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/26
