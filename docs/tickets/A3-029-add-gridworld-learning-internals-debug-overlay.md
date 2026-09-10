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

## Independent verification (2026-09-10, rebased onto main)

Cherry-picked onto main at `3a018c8` rather than merged from its original branch, which sat 12
commits behind and carried three A3-018 creativity commits belonging to their own PR. A3-029's own
commit touches no arena file, so it has no dependency on that work - only `INDEX.md` conflicted,
and the board is generated, so it was regenerated rather than hand-merged.

Suite 741 -> 800 on this branch.

What was checked by running rather than by reading:

- `test_live_updates_match_existing_training` is the load-bearing test and it is stronger than it
  looks. It runs the real trainer and `LiveLearner` on the same seed, records every
  `(action, result)` transition from both, and asserts the transition logs are *identical* and the
  final Q-tables are equal - so the overlay is displaying the real algorithm, not a reimplementation
  that resembles it. It also pins the TD algebra and the intrinsic formula per step.
- The off-policy/on-policy distinction is structural, not cosmetic: a Q-learning update carries
  `next_action=None` because it bootstraps from the maximum, and a SARSA update carries the actual
  chosen successor action. That is the difference rubric row C is graded on, and it is now on screen.
- The read-only claim holds under mutation. Changing the overlay's Q-table read from `.get(state)`
  to `[state]` - this project's recurring `defaultdict` trap, which would insert a zero row per
  drawn cell - is caught by `test_overlays_are_read_only_and_render_all_levels` on both algorithms.
- `--learn` launches and lists `I` and `V` in the on-screen legend; frozen-table evaluation remains
  the default, so nothing about the existing playback path changed.

Not re-verified line by line: the remaining self-ticked criteria above are the author's, and the
panel's visual layout at various cell sizes has a test but has not been looked at by a human on
this branch.
