---
id: A3-039
title: Fix arena human control silently dropping held keys in rotation and direct styles
type: bug
status: done
priority: P1
rubric: G
points_at_risk: 4.5
area: eval
github: https://github.com/TDuong04/A3_TDuong/issues/76
owner: unassigned
blocks: []
blocked_by: []
created: 2026-09-13
updated: 2026-09-13
---

# A3-039 — Fix arena human control silently dropping held keys in rotation and direct styles

## Context

Documented retroactively, following A3-034/A3-037's precedent: work investigated, fixed and
verified before this ticket existed, verified again independently here before closing.

Reported: "the human mode in rotation is not working, the ship doesn't move as i want" playing
`python -m eval.play_arena --human --style rotation`.

Root cause, found by reproducing with a standalone script before touching any code, not guessed:
`human_action()` in `eval/play_arena.py` used a fixed if/return priority chain per control style.
For rotation: `SPACE > LEFT/A > RIGHT/D > UP/W`. Because each `if` returns immediately, holding
multiple keys together meant only the first-checked one was ever expressed — not degraded,
completely unreachable for as long as the higher-priority key stayed held. Holding UP (thrust)
together with LEFT or RIGHT — the ordinary way to fly a thrust-based ship, turning while
accelerating — never thrust at all for as long as both were held. Holding SPACE with any movement
key blocked all movement outright. Reproduced directly: `human_action(held('UP','LEFT'),
'rotation')` returned `ROTATE_LEFT` on every call, never `THRUST`.

Env-frozen action space (`Discrete(5)`/`Discrete(6)`, fixed by the brief, guarded by rubric rows
I1/I2) genuinely accepts only one discrete action per `step()` — same for the human and the
trained agent — so this is not fixable by changing the action space. The fix has to live entirely
in how held keys map to that one action.

Row G's evidence command in `README.md` is `python -m eval.play_arena --style direct --human`, and
the module docstring calls `--human` play both a creativity feature and the fastest way to tell a
broken environment from a badly trained agent ("if a person cannot clear phase 1 either, the
problem is not the policy") — so a human unable to fly the ship as intended degrades that
evidence/demo path without touching the environment mechanics, trained models, or any measured
number.

## Fix

`human_action()` gained an explicit `frame: int = 0` parameter. Instead of returning the first
matching key in priority order, it now collects every currently-held key's action into a list and
returns `held[frame % len(held)]` — round-robining across all held actions. The call site in
`play()` now passes the render loop's own `frames` counter (already incremented once per rendered
frame, ~60Hz, the same cadence the action is sampled at for both human and agent play), so a held
combination like UP+LEFT now alternates between THRUST and ROTATE_LEFT roughly every frame — both
genuinely expressed within a handful of frames, imperceptible as flicker to a human player, rather
than one permanently starving the other. A single held key is unaffected (same action returned
every frame, matching pre-fix behaviour exactly). The identical bug existed in `direct` style's
branch of the same function (e.g. holding UP+SPACE together used to block movement while shooting)
and was fixed the same way, even though the report was specific to rotation.

## Acceptance criteria

- [x] `human_action()` in `eval/play_arena.py` expresses every currently-held key's action across
      frames instead of a fixed priority chain silently dropping lower-priority keys — verified by
      reading the diff directly (`frame` parameter, `held` list, `held[frame % len(held)]`).
- [x] Fix applied to both `rotation` and `direct` branches — verified by reading the diff.
- [x] A single held key returns the same action on every frame — no regression (verified by
      `test_a_single_held_key_is_stable_across_frames` and by direct inspection of the diff: the
      `held` list has exactly one entry when one key is held, and `x % 1 == 0` always).
- [x] New regression coverage in `tests/test_arena_eval.py` (`TestHumanControlCombinedKeys`, 5
      tests): thrust+turn both expressed across frames, shoot+thrust both expressed, `direct`
      style also alternates, single-key stability, and `frame` defaulting to 0 so every
      pre-existing single-key caller is unaffected — all 5 read directly and confirmed present.
- [x] Live-simulated the exact reported scenario (UP+LEFT held across 6 synthetic frames):
      alternates THRUST/ROTATE_LEFT/THRUST/... instead of returning ROTATE_LEFT on every frame —
      re-run directly during this ticket's verification, not taken on trust.
- [x] `tests/test_arena_eval.py` passes in full — re-run directly: 48 tests collected, all pass.
- [x] `pytest -m "not slow"` passes with no regressions elsewhere — re-run directly, full suite
      green.
- [x] `ruff check eval/play_arena.py tests/test_arena_eval.py` — 9 findings, none inside
      `human_action`, its docstring, the new test class, or the `play()` call site — confirmed
      individually via `git blame` on all 9 flagged lines (215, 384, 587, 685, 718, 729, 731 in
      `eval/play_arena.py`; 39, 322 in `tests/test_arena_eval.py`), every one authored on or
      before 2026-09-12, none touching this session's diff.
- [x] Real CLI path smoke-tested headless: `SDL_VIDEODRIVER=dummy python -m eval.play_arena
      --human --style rotation --headless --frames 5 --seed 0` — re-run directly during
      verification, exit code 0, launches and prints the control banner with no error.
- [x] No change to `arena/constants.py`, no change to any reward or the action space itself —
      confirmed no diff on `arena/constants.py`; this is purely how held keys map to the one
      action the unchanged action space accepts.

All nine boxes were checked by re-running the verification myself during this ticket's closure,
not by trusting the report handed to me.

## Notes

Files touched: `eval/play_arena.py` (`human_action` signature/body and its call site inside
`play()`), `tests/test_arena_eval.py` (new `TestHumanControlCombinedKeys` class, 5 tests).

As of this ticket's closure this diff sits modified in the working tree, not yet committed
(confirmed via `git status`) — this agent does not commit. The next commit covering this work
should reference this ticket id.

The working tree also carries unrelated uncommitted diffs to `results/arena_eval/comparison.md`,
`eval_direct_seed0.json` and `eval_rotation_seed0.json` (a rerun of the 30-episode agent
comparison table, timestamped 2026-09-13T07:09, from the agent policy not human play) — out of
scope for this ticket and not part of its deliverable; flagged here only so whoever commits does
not conflate the two. Per A3-035 (open), arena comparison-table numbers are known to vary run to
run, so that rerun should not be cited as evidence on its own.

## Log

- 2026-09-13 — Created and closed. Work was investigated, fixed and verified prior to this ticket
  existing; independently re-verified rather than taken on trust before closing: read the full
  diff in both files, ran `tests/test_arena_eval.py` directly (48 tests, all pass) and
  `pytest -m "not slow"` (full suite green), ran `ruff check` on both touched files and traced all
  9 findings to pre-existing lines via `git blame` (none inside the new/changed code), re-ran the
  UP+LEFT round-robin simulation and the headless CLI smoke test myself. Priority: P1 — row G's
  documented evidence command (`--human`) is degraded, not blocked (the row's mechanics are
  demonstrated equally by agent play, which was never affected, and a single held key already
  worked pre-fix), matching CLAUDE.md's P1 example of "a defect degrading evidence quality... a
  demo moment the video needs" rather than the P0 bar of blocking a row outright, another
  person's work, the test suite, or a trained model.
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/76
