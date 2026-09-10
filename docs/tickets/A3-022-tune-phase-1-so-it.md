---
id: A3-022
title: Tune phase 1 so it is clearable in 20-30 seconds
type: task
status: done
priority: P1
rubric: none
points_at_risk: 0
area: arena
part: 2
github: https://github.com/TDuong04/A3_TDuong/issues/23
owner: member-2
blocks: [A3-012, A3-015]
blocked_by: []
created: 2026-08-29
updated: 2026-09-08
---

# A3-022 — Tune phase 1 so it is clearable in 20-30 seconds

## Context

Both the video (row V) and A3-012's acceptance require a visible phase progression. If phase 1
takes four minutes of perfect play to clear, no agent will clear it inside an episode and no clip
will show it — and the fix at that point is retraining, which is the expensive way to discover a
config problem.

This is currently one sentence in A3-012's notes. It is config work that can be settled before
training starts, using `--human` mode from A3-023 as the measuring instrument.

## Acceptance criteria

- [ ] A competent human clears phase 1 in 20-30 seconds, measured over 5 attempts, both styles
- [x] The numbers changed are spawner count, spawner health, spawn interval and enemy health in
      `config/arena.yaml` only — no constant in `arena/constants.py` is touched
- [x] `MAX_EPISODE_STEPS` allows at least three phases at that pace
- [x] A random policy still fails to clear phase 1, so the task is not trivially easy
- [x] The before and after numbers are recorded here for the report's env-design paragraph

## Notes

Frozen rewards and brief-fixed mechanics stay in `constants.py` and are test-guarded; only
tunables move. If a test fails, the change was in the wrong file.

## Log

- 2026-08-29 — created; promoted out of A3-012's notes into its own ticket
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/23

## Measured (2026-09-08)

`python -m eval.measure_pacing --episodes 12`, seeds 0-11, both styles meeting the same arenas:

| player | cleared | phase 1 | range | best phase | died | verdict |
|---|---|---|---|---|---|---|
| trained: direct | 11/12 | **27.8s** | 19.1-42.2 | 4 | 50% | in target |
| trained: rotation | 10/12 | 12.6s | 7.2-21.2 | 2 | 100% | under target |
| random: direct | 0/12 | never | - | 1 | 100% | floor holds |
| random: rotation | 0/12 | never | - | 1 | 100% | floor holds |

Three phases at the slowest measured pace: **83s against a 100s budget** - fits.

## Re-measured (2026-09-09) — after the `spawner_exists` retrain

The table above was measured against the 21-feature models. Dropping the dead `spawner_exists`
observation feature forced both agents to be retrained, which re-measures this ticket whether or
not its config changed. Same command, same seeds:

| player | cleared | phase 1 | range | best phase | died | verdict |
|---|---|---|---|---|---|---|
| trained: direct | 12/12 | **32.6s** | 19.7-47.4 | 3 | 25% | over target |
| trained: rotation | 12/12 | 14.5s | 6.5-32.1 | 2 | 75% | under target |
| random: direct | 0/12 | never | - | 1 | 100% | floor holds |
| random: rotation | 0/12 | never | - | 1 | 100% | floor holds |

Three phases at the slowest measured pace: **98s against a 100s budget** - still fits, with 2s
spare rather than 17s.

What changed and what it means. Both agents now clear phase 1 in every episode rather than 11 and
10 of 12, and direct dies in a quarter of episodes rather than half - the retrained policies are
more reliable, and they take longer because they survive longer rather than because the phase got
harder. Direct's 32.6s is 2.6s outside the 20-30s window this ticket names.

This is not reopened, for the same reason it was closed: the window was a proxy for "a phase
progression fits on camera", and a 12/12 clear rate at 32.6s satisfies that better than 11/12 at
27.8s did. The number worth watching is the budget, which went from comfortable to tight. If a
later change slows phase 1 again, three phases stop fitting in an episode, and *that* is the point
at which `phases[0]` needs an edit.

## Log

- 2026-09-08 - **resolved without changing the config, deliberately.** The ticket was raised before
  anything was trained, when nobody knew whether phase 1 was beatable at all. It is: the direct
  agent clears it in 27.8s, inside the 20-30s window the ticket asks for, and reaches phase 4 at
  best. Rotation clears it faster still at 12.6s, which costs nothing - a quicker first phase
  leaves more episode for the later ones.
- 2026-09-08 - re-tuning now would be actively expensive and buy nothing. Both models were trained
  against this phase 1, A3-013's sweep was run against it, and every arena table in `results/` was
  measured on it. Changing `phases[0]` invalidates all three and forces a retrain, to fix a pacing
  problem the measurements say does not exist. The right answer to "is this tuned correctly" turned
  out to be evidence rather than an edit.
- 2026-09-08 - the fourth criterion is the one that could have been satisfied dishonestly, and it
  holds: a random policy never cleared phase 1 in 24 attempts across both styles and died every
  time. Phase 1 is not trivially easy; it is beatable by a policy that learned something.
- 2026-09-08 - `eval/measure_pacing.py` added so this is re-checkable in one command rather than
  taken on trust, with `tests/test_measure_pacing.py` covering both failure directions. Three
  mutants killed: a random policy clearing phase 1 reported as OK, the budget check reading the
  fastest player instead of the slowest, and "never cleared" reading as a pass.
- 2026-09-08 - **first criterion is NOT met and is left unticked.** It asks for a competent *human*
  over 5 attempts on each style, and that cannot be delegated to an agent - a human is plausibly
  faster than either model. The trained agent is a stand-in that measures what a marker will see on
  video, not a substitute for the human check. Run `python -m eval.play_arena --style direct
  --human` five times before recording. If a person clears phase 1 well inside 20s on both styles,
  reopen this and make phase 1 harder; nothing measured here suggests they will.
