---
id: A3-030
title: Add arena policy and reward debug
type: feature
status: done
priority: P1
rubric: Creativity
points_at_risk: 5.0
area: arena
owner: unassigned
blocks: []
blocked_by: []
created: 2026-09-10
updated: 2026-09-10
---

# A3-030 — Add arena policy and reward debug

## Context

Part II must show the model's action preferences, actual reward contributions and physics
without a debugger. Extend the existing HUD, observation overlay, policy panel and
`eval/play_arena.py`; do not create another entry point or change simulation through rendering.

This ticket file was absent in the checkout and is restored from the supplied specification.
P1 addresses degraded evidence for the Creativity row shared with A3-018; its 5 points
are counted once. The original A3-028 debug issue was renumbered to A3-030 to avoid a
collision with report-figure work.

The original prerequisite was A3-023. Its random-policy playback delta and focused tests
from commit 7895cc6 on `feat/a3-023-arena-playback-random-policy` are integrated here,
retaining the newer A3-018 renderer and effects. The existing A3-012 PPO models for both
control styles are present. There is no remaining implementation blocker in this checkout.
Provisional member-2 ownership still needs resolving through A3-017; no new assignment
has been made silently.

## Acceptance criteria

- [x] Dedicated unused key, distinct from O/TAB, included in the legend: F3 debug.
- [x] Loaded PPO action probabilities labelled for the active control style, plus V(s),
  captured from the exact observation used to select the executed action.
- [x] Honest random/no-model, human and unavailable-diagnostics labels; no invented
  uniform model probabilities. DQN uses Q-values and max Q(s,a), not a PPO V(s) label.
- [x] Reward decomposition shows each actual step contribution and cumulative episode
  total, using existing reward constants. Capture occurs at reward calculation sites.
- [x] Collision radii for player, enemies, spawners and bullets; spawner countdowns;
  enemy target lines.
- [x] Pause and single-step: P/N, preserving SPACE shooting and existing overlay controls.
- [x] Read-only rendering: extend A3-018's shared seeded non-mutation test with debug
  on/off, verifying identical step outputs and serialized simulation state.
- [x] Headless offscreen rendering under SDL_VIDEODRIVER=dummy, with no display window.

## Implementation notes

`arena.env` records fired reward terms at their source and returns their sum. Damage
is a per-accepted-event penalty, not a health-delta estimate; the step penalty fires once
per agent step across ACTION_REPEAT physics frames. Episode totals advance once at the
end of step(), and reset clears both totals and the latest reward snapshot.

`ArenaPlayback` in the existing eval module owns stepping and stores an immutable
`ArenaDebugSnapshot`. It pairs policy data from pre-action s with that action's completed
transition and reward snapshot. The renderer explicitly labels the policy as pre-step
and the visible world / observation as post-step s'. Rendering never predicts, probes,
steps, samples RNG or accumulates reward. Pausing retains the terminal snapshot; R clears
snapshots, totals, queued steps and renderer-owned episode effects.

Random fallback follows A3-023: empty models/ permits random playback; a populated
directory missing the requested model gives the training command and never trains on
demand. Random result filenames remain separate from trained evaluation artifacts.

## Demo

```bash
python -m eval.play_arena --style direct --debug --paused --no-save
python -m eval.play_arena --style rotation --random --debug --paused --no-save
python -m eval.play_arena --style direct --human --debug
SDL_VIDEODRIVER=dummy python -m eval.play_arena --style rotation --random --debug --headless --frames 4 --episodes 1 --no-save
```

Keys: F3 debug (Fn+F3 on some Macs), P pause/resume, N single step, R reset;
O/TAB observation, V policy, E visual effects, SPACE human shooting, ESC quit.

For A3-026's video shot list, hold a PPO transition with N and read the selected action's
probability, V(s), and reward contributions. Show a reward-bearing collision and the
cumulative per-term totals for A3-014's reward justification. Neither report nor video
content is regenerated here.

## Validation

- Focused arena debug, evaluation, policy, renderer and creativity tests: 96 passed.
- Representative paused PPO-direct and random-rotation frames rendered and visually
  inspected, including observation coexistence, random labels, collision circles,
  timers and policy / reward layout.
- Full pytest suite: 811 passed in 42.97 seconds, with local multiprocessing socket
  permissions needed by the existing arena training tests.
- PPO-direct, random-rotation and human-direct CLI demos passed offscreen frame-budget
  smoke checks. Trained and random checks used --no-save; no result artifacts changed.
- No remaining technical blockers. Changes are uncommitted and have not been pushed.
