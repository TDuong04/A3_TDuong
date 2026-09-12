---
id: A3-033
title: Launcher screen for Part I / Part II gameplay
type: feature
status: open
priority: P2
rubric: none
points_at_risk: 0
area: eval
github: https://github.com/TDuong04/A3_TDuong/issues/62
owner: unassigned
blocks: []
blocked_by: []
created: 2026-09-11
updated: 2026-09-12
---

# A3-033 — Launcher screen for Part I / Part II gameplay

## Context

Reaching either of the project's playback entry points currently requires typing a CLI command
(`python -m eval.play_gridworld`, `python -m eval.play_arena --style direct|rotation`). This
ticket adds a single retro-styled picker screen, in the same pixel/CRT visual language already
shipped for the arena (`ArenaRenderer.draw_title` and the surrounding pixel-downscale /
scanline / cabinet-bezel machinery in `arena/render.py`), that lets the user choose a part and
launch it without remembering flags.

This carries no rubric row — Parts I and II are already implemented and scoring, and the report
and video (row R, row V) are what remain per `CLAUDE.md` section 11. This is a presentation/
creativity nice-to-have, not blocking graded work, so it must stay small: one picker screen, not
a new subsystem. If it grows beyond that scope, split off the remainder as a follow-up ticket
rather than expanding this one.

Training is explicitly out of scope for this ticket and stays a CLI-only workflow
(`python -m train.train_arena`, `python -m train.train_gridworld`); it is not reachable from the
launcher.

Precedent for the interaction pattern already exists in `eval/`:
- `ArenaRenderer.draw_title` (`arena/render.py`) — the pixelated title-screen visuals to reuse
  for the picker's look.
- `wait_for_start` in `eval/play_arena.py` — a CLI-launched windowed "press key to continue"
  loop; the picker's menu loop should follow the same shape (poll pygame events, redraw, block
  until a choice is made).
- `wait_for_retry` (added on branch `feat/a3-031-pixelate-banners`, PR #61, not yet merged) —
  "PRESS R TO RETRY" after death/survival; same precedent for CLI-launched windowed choice
  screens living in `eval/`, not in the env/render simulation code. A3-033 should follow suit:
  the picker lives under `eval/`, never inside `arena/env.py`, `arena/render.py`'s simulation
  path, or `gridworld/env.py` — those must stay headless per `CLAUDE.md` section 3 rule 4
  ("never render inside `step()`").

## Acceptance criteria

- [ ] New picker screen (e.g. `eval/launcher.py`, runnable as `python -m eval.launcher`) opens a
      pygame window using the same pixel-downscale/scanline/CRT bezel treatment as
      `ArenaRenderer.draw_title`, with no simulation state and no imports from `arena/env.py`'s
      or `gridworld/env.py`'s `step()` path.
- [ ] Menu offers "PART I — GRIDWORLD" and "PART II — ARENA"; selecting Part II then prompts for
      control style (`direct` / `rotation`) matching the `--style` values `eval/play_arena.py`
      already accepts.
- [ ] Selecting a part's play/eval option launches the existing windowed session — equivalent to
      `python -m eval.play_gridworld` or `python -m eval.play_arena --style <style>` — either via
      in-process call or subprocess, and the picker window correctly hands off (closes or
      backgrounds) so the two windows do not fight for focus/input.
- [ ] `pytest` passes; a smoke test confirms `eval/launcher.py` imports cleanly and its
      command-construction for at least one play launch matches the documented CLI invocations,
      without actually opening a window in the test run.
- [ ] `ruff check .` is clean on the new file(s).
- [ ] Verified by running `python -m eval.launcher` and confirming the window opens and both
      parts' play options launch their existing eval scripts.
- [ ] Picking Part I now first asks for a level (0-6), then asks agent-playback vs human-play;
      agent playback launches `python -m eval.play_gridworld --level N`, human play launches
      `python -m gridworld.render --level N`.
- [ ] Picking Part II (after control style) now also asks agent-playback vs human-play; agent
      playback launches `python -m eval.play_arena --style <style>` (unchanged), human play
      launches `python -m eval.play_arena --style <style> --human`.
- [ ] Existing tests plus new tests covering the level menu and the agent/human menu for both
      parts; `pytest` and `ruff check .` stay clean.
- [ ] Re-verified by actually running `python -m eval.launcher` through both new menu layers for
      both parts.

## Notes

Files likely touched: new `eval/launcher.py` (or similar), possibly a small shared helper if
title-screen drawing logic is factored out of `arena/render.py` for reuse — do not duplicate the
CRT/pixel-downscale code wholesale if it can be imported instead. No changes to `arena/env.py`,
`gridworld/env.py`, `arena/constants.py`, or `gridworld/constants.py`.

Training (SB3 PPO/DQN for arena, Q-learning/SARSA for gridworld) is out of scope for this
ticket and is not reachable from the launcher; it remains a CLI-only workflow via
`python -m train.train_arena` / `python -m train.train_gridworld`.

## Log

- 2026-09-11 — created; scoped from conversation to a picker screen for Part I/II play + eval
  launch plus background-subprocess training with coarse status, reusing the arena title-screen
  visual style. No rubric row — presentation/creativity polish, deliberately kept small since
  report and video are the remaining graded work.
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/62
- 2026-09-12 — scope narrowed at the user's request: dropped "launch training" entirely (the
  background-subprocess launch, status reporting, and the design-constraint section tied to it).
  Reason: training stays a CLI-only workflow (`python -m train.train_arena`,
  `python -m train.train_gridworld`) and is out of scope for this ticket. Renamed from
  "Launcher screen for eval and training entry points" to "Launcher screen for Part I / Part II
  gameplay"; file renamed to match. Remaining scope: a retro-styled picker for Part I/II
  play/eval only.
- 2026-09-12 — implemented. New `eval/launcher.py` (`python -m eval.launcher`) draws the picker
  through `ArenaRenderer.begin_retro_frame` / `finish_retro_frame`, two small methods factored out
  of `draw_title` so the cabinet look is shared rather than copied; `draw_title` now calls them and
  is otherwise unchanged. The picker imports no env and no `step()` path. Selecting a part closes
  the picker window before running `python -m eval.play_gridworld` or
  `python -m eval.play_arena --style <style>` as a subprocess, then returns to the menu when that
  session ends. Training stays unreachable. Tests in `tests/test_launcher.py`; verified with
  `SDL_VIDEODRIVER=dummy python -m eval.launcher --frames 120` (exit 0) and with a real subprocess
  handoff of both commands.
- 2026-09-12 — follow-up scope added at the user's request, right after the first-pass
  implementation above landed. Two more choices join the same one-screen picker (still additive,
  not a new ticket): (1) Part I now asks for a level 0-6 before launching, since
  `eval.play_gridworld` and `gridworld.render` both already accept `--level` and `N_LEVELS` is 7;
  (2) both parts now ask agent-playback vs. human-play. Gridworld human play is a separate
  existing entry point, `python -m gridworld.render --level N`, not a flag on
  `eval.play_gridworld`. Arena human play reuses `eval.play_arena` with the existing `--human`
  flag. No new CLI surface is needed anywhere — this is purely launcher wiring to scripts that
  already accept these options.
