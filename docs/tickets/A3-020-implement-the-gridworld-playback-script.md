---
id: A3-020
title: Implement the gridworld playback script
type: feature
status: done
priority: P0
rubric: V
points_at_risk: 5.0
area: eval
github: https://github.com/TDuong04/A3_TDuong/issues/20
owner: unassigned
blocks: [A3-015]
blocked_by: []
created: 2026-08-20
updated: 2026-08-20
---

# A3-020 — Implement the gridworld playback script

## Context

`eval/play_gridworld.py` is still a stub that raises `NotImplementedError`, and no open ticket
owned it — A3-004 named it under "Files likely touched" and was closed without it, so it fell
through the gap between a closed feature ticket and a video ticket that assumes it already exists.

This is the script the video records for Part I. The video rubric does not ask to see a training
curve; it asks to see the gridworld running in a Pygame window with a Q-learning or SARSA agent,
items and monsters behaving correctly, and **evidence the agent follows a learned policy rather
than acting randomly**. A single greedy rollout is weak evidence of that — an agent walking a
sensible line for eleven steps looks much like a lucky random one to a marker watching a video.
The policy-arrow overlay is what makes it undeniable, and `gridworld/render.py` already draws
everything needed.

Rubric row A1 is **not** at risk: `python -m gridworld.render --level N` already opens an
interactive Pygame window with keyboard control, human play, pause, single-step, speed control and
level switching. The gap is specifically playback of a *trained* policy, and the `--compare` mode
that animates the row C claim.

`README.md` currently documents `python -m eval.play_gridworld --level 1 --compare` as if it
works, annotated only as "not yet implemented". A marker who runs the documented command gets a
traceback.

## Acceptance criteria

- [x] `python -m eval.play_gridworld --level N --algo q|sarsa` loads the saved Q-table from
      `results/` and animates the greedy policy in a Pygame window
- [x] Policy-arrow overlay toggleable and ON by default, so "follows a learned policy" is visible
      rather than inferred — this is the single most valuable frame in the Part I footage
- [x] `--compare` runs Q-learning and SARSA on level 1 at once, so the route difference is visible
      in motion rather than as a static figure
- [x] Works on a level with monsters (4 or 5) with monster movement visibly stochastic
- [x] Fails with a clear message, not a traceback, when the requested Q-table has not been trained
- [x] `README.md`'s "not yet implemented" annotation removed once it is true

## Notes

`gridworld/render.py` already has `GridRenderer`, `PlaybackApp`, the overlay drawing and the
keyboard map. This ticket should be mostly wiring plus a CLI, not new rendering code. Q-tables are
saved as `results/qtable_level{N}_{algo}_seed{S}.npz`; `load_q_table` already reads them.

`--record`, mentioned in the stub docstring, is optional — screen capture software is a valid
substitute and cheaper than maintaining a frame dumper.

## Log

- 2026-08-20 — raised while auditing Part I readiness. Found by checking which rubric row each
  Part I artifact actually serves, rather than by a test failing: nothing fails, because nothing
  tests a script that raises `NotImplementedError` by design.
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/20
- 2026-08-20 - implemented. `eval/play_gridworld.py` loads the trained table for a level and
  algorithm, plays it greedily in a Pygame window with the policy arrows on by default, and
  restarts the episode a moment after it ends - the auto-restart is the difference between "here
  is a rollout" and "here is a policy", since a level-0 episode is eleven steps and under two
  seconds of footage. `--compare` runs Q-learning and SARSA as two panels in one window, stepping
  and restarting together so the two routes on screen are always the same episode number. All four
  documented modes were smoke-run headless under `SDL_VIDEODRIVER=dummy`.
- 2026-08-20 - the dev agent hit a session limit before writing any tests, so the suite was written
  in the main session: `tests/test_play_gridworld.py`, 19 tests, suite 566 -> 585 passed.
  Assertions read pixels rather than surface dimensions, following the lesson from
  `tests/test_gridworld_render.py`, where 63 tests once asserted only that a surface had the size
  the renderer said it would - a tautology that let six visual regressions ship green.
- 2026-08-20 - five mutants against the new tests. Three died at once: playback exploring instead
  of acting greedily, `--compare` pointed at the wrong level, and the arrow overlay failing to
  reach the screen. **Two survived and were real holes.** Forcing `show_arrows` to False whenever
  the flag was supplied passed, because the test only ever asserted the False case; it is now
  parametrised over both directions. And indexing the Q-table with `table[state]` instead of
  `table.get(state)` passed, because the test parked the agent on a cell the table already
  contained - it now asserts the state is genuinely absent before looking for growth. That second
  one matters beyond the test: the tables are `defaultdict`s, so indexing them during playback
  would add a zero row per unseen state per step, and the table being demonstrated on camera would
  quietly stop being the table that was trained. Both mutants die now.
- 2026-08-20 - `--record` descoped, as the ticket allowed: screen capture software is a valid
  substitute and cheaper than maintaining a frame dumper. `README.md`'s "not yet implemented"
  annotation is gone and two more example commands were added. Closed.
