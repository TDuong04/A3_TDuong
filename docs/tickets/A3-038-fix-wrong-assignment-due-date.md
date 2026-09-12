---
id: A3-038
title: Fix the wrong assignment due date baked into the repo
type: bug
status: done
priority: P0
rubric: none
points_at_risk: 0
area: infra
github: https://github.com/TDuong04/A3_TDuong/issues/74
owner: unassigned
blocks: []
blocked_by: []
created: 2026-09-13
updated: 2026-09-13
---

# A3-038 — Fix the wrong assignment due date baked into the repo

## Context

Documented retroactively, following A3-034's precedent for already-shipped work verified directly
before the ticket existed.

The committed `CLAUDE.md` and `A3_Brief.pdf` said the deadline was **19 September 2026**. The user
confirmed via Canvas tonight (2026-09-13) that the real, reissued deadline is **13 September 2026,
11:59 PM** — today. A reissued brief PDF with the correct date had actually already been found the
day before and left sitting in an unapplied git stash; the confusion was only fully resolved and
fixed tonight.

Fixed:

- `A3_Brief.pdf` replaced with the correct Sep-13 PDF (extracted from that stash) — confirmed by
  extracting text from the new file: `Due Date: September 13, 2026, 11:59 PM`.
- The hard-coded "19 September 2026" date corrected to "13 September 2026, 11:59 PM" in three
  places that each independently baked it in for their own scheduling/priority logic:
  `CLAUDE.md`, `.claude/agents/ticket-bot.md`, `.claude/agents/solution-evaluator.md`.
- `README.md` already had the correct date and needed no change.

Priority derivation: **P0**. `rubric: none` — this maps to no single rubric row, so it does not
qualify under the row-size clause of CLAUDE.md's own priority rubric. But it is exactly the kind of
error the P0 tier exists to catch by spirit rather than letter: an incorrect due date propagating
through the shared agent context and two agents' own scheduling logic risked every subsequent
priority/scheduling call in the project being made against the wrong deadline, with the brief's
late-penalty schedule (2 marks/day, 0 marks after 5 days) applying against the entire 40-point
submission if it had gone unnoticed until the real date passed. It is already fixed, so this ticket
records severity-of-what-was-corrected, not remaining risk.

## Acceptance criteria

- [x] `A3_Brief.pdf` is the correct, reissued Sep-13 version (verified: extracted text reads
      "Due Date: September 13, 2026, 11:59 PM").
- [x] `CLAUDE.md`'s hard-coded due date corrected to "13 September 2026, 11:59 PM" (verified via
      diff).
- [x] `.claude/agents/ticket-bot.md`'s hard-coded due date corrected (verified via diff).
- [x] `.claude/agents/solution-evaluator.md`'s hard-coded due date corrected (verified via diff).
- [x] `README.md` checked and confirmed already correct — no change needed.
- [x] No other hard-coded "19 September" reference found in the checked files.

## Notes

Files touched: `A3_Brief.pdf`, `CLAUDE.md`, `.claude/agents/ticket-bot.md`,
`.claude/agents/solution-evaluator.md`. As of this ticket's closure, all four sit
modified/untracked in the working tree, not yet committed (confirmed via `git status`) — this
agent does not commit per its operating instructions. The next commit covering this work should
reference this ticket id and, given the stakes, should land promptly.

## Log

- 2026-09-13 — Created and closed retroactively. Work was fixed and verified directly prior to this
  ticket existing. Verified independently rather than taken on trust: diffed `CLAUDE.md`,
  `.claude/agents/ticket-bot.md` and `.claude/agents/solution-evaluator.md` directly and confirmed
  each now reads "13 September 2026, 11:59 PM"; extracted text from the new `A3_Brief.pdf` directly
  and confirmed it states "Due Date: September 13, 2026, 11:59 PM"; grepped for remaining "19
  September" references in the four files and found none. Closing as done on that basis.
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/74
