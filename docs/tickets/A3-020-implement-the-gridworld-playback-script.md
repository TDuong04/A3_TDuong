---
id: A3-020
title: Implement the gridworld playback script
type: feature
status: done
priority: P0
rubric: V
points_at_risk: 5.0
area: eval
part: 1
github: https://github.com/TDuong04/A3_TDuong/issues/20
owner: member-3
blocks: [A3-015]
blocked_by: []
created: 2026-08-20
updated: 2026-08-29
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

- [ ] `python -m eval.play_gridworld --level N --algo q|sarsa` loads the saved Q-table from
      `results/` and animates the greedy policy in a Pygame window
- [ ] Policy-arrow overlay toggleable and ON by default, so "follows a learned policy" is visible
      rather than inferred — this is the single most valuable frame in the Part I footage
- [ ] `--compare` runs Q-learning and SARSA on level 1 at once, so the route difference is visible
      in motion rather than as a static figure
- [ ] Works on a level with monsters (4 or 5) with monster movement visibly stochastic
- [ ] Fails with a clear message, not a traceback, when the requested Q-table has not been trained
- [ ] `README.md`'s "not yet implemented" annotation removed once it is true

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
- 2026-08-29 — closed — eval/play_gridworld.py is implemented with --compare and the TAB overlay swap, covered by tests/test_gridworld_playback_cli.py
