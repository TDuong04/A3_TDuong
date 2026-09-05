# A3-012 — train both control agents and write the evaluation scripts

Branch: `feat/a3-012-arena-agents` · Rubric row I, 4 points · Started 2026-09-05

## What this is, in plain terms

The assignment needs two AI agents that play the arena game — one that steers like a spaceship
(turn left, turn right, thrust forward) and one that moves like a top-down character (up, down,
left, right). Each has to be **trained**, **saved to disk**, and **playable on screen** so a marker
can watch it. That is rubric row I: two styles, two saved models, two working evaluation scripts.

The row does not require the agents to be *good*. It requires them to exist, load, and run. That
makes it four points of low-risk marks — but only if the scripts work on a fresh clone.

## Why this ticket is bigger than it looks

A3-012 says "train the agents". You cannot train an agent without a training pipeline, and you
cannot watch one without a renderer. Neither exists on `main`. So this branch has to deliver three
tickets, in order:

| Ticket | What it is | Why it blocks A3-012 |
|---|---|---|
| A3-009 | The arena renderer — drawing ships, enemies, HUD | The eval script has nothing to show without it |
| A3-011 | The training pipeline — SB3 + TensorBoard logging | Nothing produces a model without it |
| A3-012 | Train both agents, write the eval script | The ticket asked for |

Skipping straight to A3-012 is not possible. Doing them in one branch is cheaper than three
branches, because the renderer and the eval script are written against each other.

## Step 1 — the renderer (A3-009)

**Do not write a new one.** A teammate already built `arena/render.py` (411 lines) on branch
`docs/tickets`, and it is good work: it counts pixels in its tests instead of asserting surface
sizes, it has a `test_drawing_never_mutates_the_simulation` invariant, and its observation panel is
driven by `observation.describe()` so the labels cannot drift from the vector. Reimplementing it
would duplicate a teammate's work for the second time on this project.

Take it as-is, then fix the three things a review found:

1. **Phase off-by-one.** Their renderer draws `env.phase + 1`, assuming phase is 0-based. Main's env
   is 1-based. Merged unchanged, the HUD reads "PHASE 2" during phase 1 and the first banner reads
   "PHASE 3". That is wrong on camera, in exactly the phase-progression shot the video rubric
   demands.
2. **Three missing env attributes.** The renderer reads `env.episode_return`, `env.last_observation`
   and `env.phase_just_advanced`; main's env has none of them. Without them the first `draw()`
   raises `AttributeError`.
3. **A missing assertion.** Their HUD test checks the *label* "PHASE" is drawn but never the
   *number*, which is why the off-by-one would have survived. Add the value assertion.

## Step 2 — the training pipeline (A3-011)

`train/callbacks.py` and `train/train_arena.py`.

The important design point is **what gets logged**. Reward alone cannot tell an agent that is
genuinely progressing from one that has found a way to farm a shaping term — both show a rising
curve. So the callback records, alongside reward:

```
behaviour/phase_reached      behaviour/spawners_destroyed
behaviour/enemies_killed     behaviour/damage_taken
behaviour/episode_length
```

Reward rising while `phase_reached` stays flat is reward hacking, and no reward graph alone will
ever show it.

Windows specifics that are not optional: `SubprocVecEnv` must sit behind `if __name__ ==
"__main__":` or the process forks recursively, and `SDL_VIDEODRIVER=dummy` must be set before pygame
is imported in any worker so no training process opens a window.

Checkpoints every 50k timesteps to `models/checkpoints/`, so a crash does not cost the whole run.

## Step 3 — the evaluation script (A3-012)

`eval/play_arena.py`, with:

- `--style {rotation,direct}` loading the matching model from `models/`
- `deterministic=True` — the rubric asks for it, and a stochastic policy on camera looks like a
  broken one
- `--human` to play the same environment from the keyboard. This is both a creativity feature and
  the fastest way to tell a broken environment from a badly trained agent
- `--episodes N` printing mean/std return, phase reached, spawners destroyed and survival time.
  Those numbers are the report's control-scheme comparison, so both styles run under the same seeds

## Step 4 — actually train

Direct style first. It has no rotation to learn, so it converges faster and proves the reward
function works before the harder style is attempted. If either agent cannot clear phase 1, the fix
is to **tune phase 1 in the config, not to add timesteps** — the video needs a visible phase
progression, and a marker watching an agent grind for ten minutes is worse evidence than one
watching it clear a phase in twenty seconds.

## How I will know it worked

- `pytest` green, and the currently-skipped renderer test unskips itself
- Two files in `models/`, loadable from a clean checkout
- Both styles clear phase 1 at least once under evaluation
- TensorBoard event files under `logs/`
- `--human` playable

## What I am deliberately not doing

- Not tuning hyperparameters. That is A3-013, a separate row, and mixing it in makes both harder to
  review.
- Not touching `arena/constants.py` or `gridworld/constants.py` — brief-fixed values.
- Not re-litigating the observation vector. It is committed, evaluated, and identical to the one a
  teammate derived independently.
