---
id: A3-036
title: Add opt-in auto-aim shooting for arena rotation control style
type: feature
status: wontfix
priority: P3
rubric: none
points_at_risk: 0
area: arena
github: https://github.com/TDuong04/A3_TDuong/issues/71
owner: unassigned
blocks: []
blocked_by: []
created: 2026-09-12
updated: 2026-09-13
---

## Context

Rotation-style shooting currently always fires along `self.heading`
(`arena/entities.py:Player.shoot()`), which is a large part of why rotation's baseline hit-rate
(12.8%) trails direct's (17.3%) — see the comments in `config/arena.yaml`'s `shaping:` block, which
already compensates for this with potential-based `aim_strength`/`safety_strength` reward shaping
computed in `arena/env.py:_aim_potential()`.

The user has approved a harder mechanic on top of that: give rotation an **opt-in** auto-aim mode
where the SHOOT action fires a lead-computed shot at the nearest living enemy (or nearest living
spawner if no enemies are alive), instead of firing along the ship's current heading. This is a
deliberate feature addition, not a fix to the shipped baseline — it must not touch or invalidate the
existing frozen rotation model or its evidence tables (CLAUDE.md ss3 rule 5).

This follows the exact pattern already established for the `mechanics:` opt-in bundle (shield
pickups / elite chargers): a constructor flag, a CLI flag on both `train/train_arena.py` and
`eval/play_arena.py`, and a separate `models/` subdirectory so training this variant can never
overwrite `models/<style>/...` for the shipped baseline (see `train/train_arena.py:297-300` and
`eval/play_arena.py:212-213` for the `mechanics` precedent).

Design decided with the user (do not re-litigate):

1. **Opt-in, rotation-only.** `ArenaEnv.__init__` gains `auto_aim: bool = False`. Raise a clear
   `ValueError` at construction if `auto_aim=True` and `control_style="direct"` — direct's heading
   already equals its movement direction (`Player.move()` snaps heading to travel direction), so
   auto-aim is meaningless there and would break the "identical physics" comparison the report
   relies on (arena/entities.py module docstring; report row R6).
2. **Only the fired bullet's direction changes.** `Player.shoot()` gains an optional
   `heading: float | None = None` param. When given, it overrides `self.heading` for the muzzle
   spawn-offset and the bullet's travel direction, but must **not** mutate `self.player.heading`
   itself — turning/thrust/rendering stay exactly as they are today.
3. **Targeting**, in `arena/env.py:_fire()` when `self.auto_aim` is true: use the existing
   `nearest_alive()` helper (`arena/observation.py`) — first over
   `[e for e in self.enemies if e.alive]`, falling back to
   `[s for s in self.spawners if s.alive]` when no enemies are alive. If both are empty, fire
   straight ahead (no heading override — same as today).
4. **Lead-based aim using the TARGET's own velocity, not player-relative.** Reuse the existing
   quadratic intercept solver `_lead_offset(dx, dy, rvx, rvy, bullet_speed)`
   (`arena/env.py:650`, used today by `_aim_potential` at line ~491-496). Note the trap:
   `_aim_potential` passes `rvx, rvy = enemy.vx - player.vx, enemy.vy - player.vy` because that use
   is only a reward-shaping heuristic and doesn't need to be physically exact. `Bullet.__init__`
   (`arena/entities.py`) gives a bullet a fixed world-frame velocity of
   `cos(heading)*bullet_speed, sin(heading)*bullet_speed` — it does **not** inherit the player's
   velocity. So the real auto-aim mechanic must compute lead using `rvx, rvy = enemy.vx, enemy.vy`
   (or `0, 0` for a `Spawner`, which has no `.vx`/`.vy`), **not** relative to the player. Compute
   `dx, dy = target.x - player.x, target.y - player.y`, get `lead_x, lead_y` from `_lead_offset`,
   then `heading = atan2(lead_y, lead_x)` and pass it as the `Player.shoot()` override.
5. **Shaping interaction (guidance only, not enforced in code):** when training the auto_aim
   variant, recommend zeroing `aim_strength`/`safety_strength` in `config/arena.yaml`'s `shaping:`
   block, since that shaping exists to teach manual aiming, which becomes moot once shots
   auto-target. Note this in the CLI help / README, do not hardcode it.

## Acceptance criteria

- [ ] `Player.shoot()` (`arena/entities.py`) accepts an optional `heading: float | None = None`
      override affecting muzzle offset + bullet travel direction, without mutating
      `self.heading`.
- [ ] `ArenaEnv.__init__` accepts `auto_aim: bool = False`; constructing with
      `auto_aim=True, control_style="direct"` raises a `ValueError` with a clear message.
- [ ] `_fire()` in rotation style with `self.auto_aim=True` fires a lead-computed shot at the
      nearest living enemy, falling back to nearest living spawner when no enemies are alive,
      using the target's own velocity (`0, 0` for spawners) for the lead calculation — not
      player-relative velocity.
- [ ] `--auto-aim` CLI flag added to `train/train_arena.py` and `eval/play_arena.py`; enabling it
      routes saved/loaded models to a separate directory (e.g. `models/auto_aim`, mirroring the
      `mechanics` precedent) so `models/<style>/...` (the shipped rotation baseline) is never
      written to or read from when `--auto-aim` is used, and vice versa.
- [ ] New tests (e.g. `tests/test_arena_auto_aim.py`):
  - a rotation shot at an off-heading stationary enemy travels toward the enemy, not along
    `self.heading`;
  - a shot at a moving enemy leads it (the shot direction differs from the naive
    non-lead bearing and the intercept lands, not trails);
  - falls back to nearest living spawner when no enemies are alive;
  - `ArenaEnv(control_style="direct", auto_aim=True)` raises `ValueError`;
  - the player's `self.heading` / rotation controls are numerically unaffected by firing with
    auto-aim on (i.e. `heading` before and after `_fire()` is unchanged).
- [ ] `pytest` passes; `ruff check .` is clean.
- [ ] No edits to `arena/constants.py` and no reward values changed.
- [ ] The shipped rotation model(s) under `models/` and every table under `results/` are
      unmodified (no retrain of the frozen baseline as a side effect of this ticket).

## Notes

Files likely touched: `arena/entities.py` (`Player.shoot`), `arena/env.py` (`__init__`, `_fire`,
reuse of `_lead_offset`/`nearest_alive`), `train/train_arena.py` (CLI flag + models-dir routing,
mirroring the existing `mechanics` flag at lines ~297-300, 320-322, 386-387), `eval/play_arena.py`
(CLI flag + models-dir routing, mirroring `mechanics` at lines ~107-221, 257, 291, 348-352, 431-432,
502), `config/arena.yaml` (doc comment only, no value changes), new test file under `tests/`.

Prior art / pattern to copy: the `mechanics: bool = False` opt-in bundle (shield pickups, elite
chargers) already threaded through `ArenaEnv`, `train/train_arena.py` and `eval/play_arena.py` with
its own `models/mechanics` subdirectory — auto-aim should follow the identical shape with
`models/auto_aim`.

No rubric row maps to this — Parts I and II are already scored and evidenced; this is scope beyond
the tracked backlog, added at the user's request as a creativity/polish feature, not a rubric
requirement. Priority is P3 under the points-at-risk rubric: it does not block a rubric row (rubric:
none, points_at_risk: 0), does not block another person's work, and does not touch report/video
tickets currently on the critical path (A3-014, A3-015, A3-017, A3-018). If someone commits to using
it for the row-J "creativity" discussion or the video, re-derive priority upward at that point.

## Log

- 2026-09-12 — created. Design approved by user in conversation; filed directly to backlog as new
  scope beyond the currently tracked report/video work. Priority set to P3 (rubric: none, no points
  at risk, does not block other tickets or teammates) per the points-at-risk priority rubric in
  CLAUDE.md ss7.
- mirrored to GitHub issue https://github.com/TDuong04/A3_TDuong/issues/71
- 2026-09-12 — implemented, left uncommitted for review. `Player.shoot(heading=...)` override
  (`arena/entities.py`), `ArenaEnv(auto_aim=...)` plus `_auto_aim_heading()` reusing `_lead_offset`
  with the *target's* own velocity (`arena/env.py`), `--auto-aim` on both entry points routing to
  `models/auto_aim/` and `results/arena_eval/auto_aim/`, shaping guidance noted in
  `config/arena.yaml` and README (no values changed). 23 new tests in
  `tests/test_arena_auto_aim.py`, all passing; `arena/constants.py`, `models/` and `results/`
  untouched by this work. Two findings for whoever picks this up next, both outside this ticket:
  (1) six pre-existing test failures in the tree, caused by tuned `config/arena.yaml` values
  changed in commit 94040bd ("adjust indentation and formatting") plus further uncommitted edits to
  `player.speed`/`rotation_speed`/`invulnerability_time` — the same 74 tests pass against
  `HEAD~1`'s arena.yaml; (2) no renderer signal that auto-aim is on (the bullets show it, the
  title screen does not), if it is ever used on camera.
- 2026-09-13 — **cancelled at the user's explicit request, after reviewing the implemented
  behaviour in detail.** Verbatim: "please undo the auto-aim, this is not what i expected." Not an
  acceptance-criteria failure — the mechanic worked exactly as designed and logged above; seeing it
  running live was what changed the decision. All code fully reverted to HEAD:
  `arena/entities.py`, `arena/env.py`, `config/arena.yaml`, `eval/play_arena.py` and
  `train/train_arena.py` restored via `git checkout --`, and `tests/test_arena_auto_aim.py`
  deleted. Confirmed clean: no diff against HEAD remains in those files and no `auto_aim`
  reference remains anywhere in the tree. Status set to `wontfix` rather than deleting the file —
  kept as the record of what was tried and why it was rejected, matching how every other closed
  ticket in this repo is kept rather than removed. No rubric or teammate impact (rubric: none,
  points_at_risk: 0, nothing was blocked by this).
- GitHub issue #71 closed (reason: not planned) with a comment pointing back to this log.
- 2026-09-13 — Cross-reference only, no field changes beyond `updated`: the actual problem this
  ticket was meant to address (rotation's poor aiming/navigation — wall/corner-sticking, endless
  spinning, low hit-rate) was root-caused and fixed properly in **A3-037**, filed and closed the
  same night, as three real bugs in the existing potential-based shaping (`arena/env.py`) plus a
  training-budget/gamma fix made necessary by the difficulty rebalance in `94040bd` — not by adding
  a new aiming mechanic. Confirms this ticket's `wontfix` was the right call: auto-aim was the wrong
  shape of fix and, being a new mechanic layered on top of manual aiming, would not have touched the
  navigation symptoms at all.
