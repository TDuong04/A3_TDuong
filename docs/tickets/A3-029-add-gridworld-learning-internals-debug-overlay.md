---
id: A3-029
title: Add gridworld learning internals debug overlay
type: feature
status: done
priority: P1
rubric: Creativity
points_at_risk: 5.0
area: gridworld
owner: unassigned
blocks: []
blocked_by: []
created: 2026-09-09
updated: 2026-09-09
---

# A3-029 — Add gridworld learning internals debug overlay

## Context

Part I learning-update evidence must be visible without a debugger. Extend the existing
policy arrows, Q heatmap, human play, panels, legend and comparison mode rather than
rebuilding them. The core evidence is Q-learning's off-policy maximum successor value
versus SARSA's on-policy value for the action actually chosen next.

P1 represents degraded evidence for the Creativity row already shared with A3-018;
its 5 points are counted once on the board. This is a beyond-brief mechanic candidate.
The supplied issue was renumbered from A3-027 to avoid an existing ticket collision.
This file was absent from this checkout and was restored from the supplied specification.

## Acceptance criteria

- [x] Dedicated unused debug key, listed in the on-screen legend.
- [x] Latest actual transition: state, action, environment reward, next state,
  chosen Q before/after, TD target/error, current epsilon, alpha and gamma.
- [x] Explicit labelled `max_a' Q(s',a')` for Q-learning versus
  `Q(s', a'_chosen)` for SARSA, including the actual chosen successor action.
- [x] Greedy/exploratory selection branch and the original epsilon roll.
- [x] Level-6 intrinsic reward: prior count, strength/sqrt(count+1), environment
  reward and shaped reward separately labelled.
- [x] Separately toggleable per-cell episode visit-count heatmap.
- [x] Comparison selection follows the displayed algorithm and its own update.
- [x] Existing SPACE/N pause/single-step; paused terminal frames remain readable.
- [x] Headless rendering under SDL_VIDEODRIVER=dummy across all seven levels.
- [x] Same-seed transitions and Q-tables identical with overlays on/off; rendering
  does not draw random numbers, insert Q rows, change counts or modify snapshots.

## Implementation and demo

- `gridworld.algorithms.LiveLearner` performs actual updates one step at a time and
  creates immutable `LearningUpdate` records at their source. `select_action` has
  optional tracing of its existing roll and branch, with unchanged RNG consumption.
- `--learn` opts into fresh in-memory Q-tables and starts paused. Existing frozen-table
  evaluation remains the default; no update is reconstructed from saved artefacts.
- This checkout's comparison was side by side, without TAB. Both panels remain the
  default; TAB cycles both → Q-learning → SARSA → both. Both learners step together.
- `I`: debug panel. `V`: episode visits. `SPACE`: run/pause. `N`: one update.
  `R`: next episode, retaining Q values and advancing epsilon. `0`–`6`: switch level,
  clear telemetry/counts and start that level's schedule, retaining its in-memory table.
- SARSA carries the exact selected successor action and roll. True termination
  suppresses bootstrapping; truncation keeps it. A human move clears the learning
  snapshot because no learning update occurred.
- Intrinsic counts refer to the full arrival state before this arrival is counted,
  consistent with existing training. The heatmap aggregates spatial cell occupancies,
  including the initial cell. Both reset each episode.

```bash
python -m eval.play_gridworld --learn --level 1 --compare --env-seed 0 --policy-seed 0
python -m eval.play_gridworld --learn --level 6 --algo sarsa --intrinsic-strength 0.5
SDL_VIDEODRIVER=dummy python -m eval.play_gridworld --learn --compare --frames 2
```

For the A3-026 video shot list, hold a nonterminal update with N, use TAB to show each
algorithm's labelled successor term, and read the TD target aloud. On level 6, enable V
and explain the prior full-state count and the separately labelled reward components.
A3-026 is not present in this checkout; no video file was edited.

## Validation

- 295 focused tests passed, including source-calculation checks, live-versus-existing
  trainer equivalence, overlay on/off equivalence across all levels and both algorithms,
  SARSA carry-forward, terminal/truncated targets, reset, human moves and comparison.
- A level-6 SARSA frame was rendered and visually inspected; custom cell sizes have
  layout bounds tests. CLI demo smoke tests use the dummy SDL driver.
- Full suite: 780 passed in 43.39 seconds. The initial sandboxed run had four
  arena multiprocessing socket permission failures; the rerun with local subprocess
  permissions passed without changing arena code. Test dependencies were installed
  into `/private/tmp/a3-029-venv`, leaving the project dependency files unchanged.
