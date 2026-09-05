# What was done on `feat/a3-012-arena-agents` — in plain terms

For teammates. No RL background assumed.

## The short version

The arena game now has **two AI players that learned to play it by themselves**, and you can watch
either of them on screen or take the controls yourself. Both are saved to disk in the `models/`
folder the assignment brief demands, and both are good enough to clear the first difficulty phase,
which is the thing the video has to show.

That closes three tickets: **A3-009**, **A3-011** and **A3-012** — 4.5 + 3 + 4 = **11.5 rubric
points** of Part II.

## Why three tickets and not one

A3-012 says "train the agents and write the eval script". But you cannot train an agent with no
training program, and you cannot watch one with nothing that draws the game. Neither existed. So the
work had to run in order:

1. **A3-009** — the thing that draws the arena on screen
2. **A3-011** — the thing that runs the training
3. **A3-012** — the agents themselves, and the script that plays them back

## 1. The renderer (A3-009)

This draws the ship, the enemies, the spawners, the bullets, the health bars, and a status bar
showing which phase you are in and what the agent is doing right now. There is also a toggle (press
`O`) that overlays **what the agent can actually see** — lines to the nearest enemy and spawner, and
a live readout of all 21 numbers the AI receives. That overlay is worth having on camera: it is the
difference between a marker watching a dot move around and a marker watching an agent react to
something specific.

**I did not write this.** Hanh had already built it on the `docs/tickets` branch, and it was good —
in particular its tests check the actual pixels on screen rather than just checking the window is
the right size, which is a mistake this project has made before. Rewriting it would have duplicated
a teammate's work for the second time on this project, so I took it as-is and fixed three things a
review found:

- **It showed the wrong phase number.** It assumed phases were counted from 0, but our environment
  counts from 1. Left alone, the screen would have read "PHASE 2" during phase 1, and the banner
  would have said "PHASE 3" when you entered phase 2 — wrong in exactly the shot the video rubric
  asks for. One-line fix.
- **It needed three pieces of information the environment did not expose** — the running score, the
  agent's last observation, and a flag saying a phase had just been cleared. Without them the very
  first frame crashed.
- **Its test checked the word "PHASE" appeared, but never checked the number.** That is why the
  off-by-one slipped through. Added a test for the value, and confirmed it fails against the old
  code.

## 2. The training program (A3-011)

`python -m train.train_arena --style direct` runs a training session. It uses Stable Baselines3
(the standard library for this), runs 8 copies of the game in parallel, and takes about
**two and a half minutes** for a full 400,000-step run.

The part worth explaining is **what gets recorded**. The obvious thing to log is the score. The
problem is that a rising score does not prove the agent is playing well — it might have found a way
to milk one small reward over and over without ever progressing. Those two look identical on a score
graph. So the training also records what the agent actually *did*: how far it got through the
phases, how many spawners it destroyed, how many enemies it killed, how much damage it took, and
how often it survived to the end. If the score climbs but the phase count stays flat, that is
cheating, and only these extra graphs will show it.

One bug worth knowing about: the survival-rate graph would have been **silently wrong**. The
training library only passes along information it has been explicitly told to keep, and nobody had
told the environment to report whether the agent died. It would have quietly recorded "survived
100% of the time" for an entire training run, with no error message. There is now a test that
checks every graph we log corresponds to something the game actually reports.

## 3. The agents and the playback script (A3-012)

Two agents trained, one per control scheme:

- **direct** — moves up/down/left/right, like a top-down shooter
- **rotation** — turns left, turns right, thrusts forward, like Asteroids

**Direct was trained first on purpose.** It is the easier of the two, so if something is wrong with
the game's reward design, it shows up there first and cheaply, before time is spent on the harder
one.

### How well they play

Measured over 5 games each, both playing the same 5 arena layouts:

| | direct | rotation |
|---|---|---|
| score | **+19.18** ± 16.36 | +2.79 ± 11.83 |
| phase reached | 2.20 average, 3 best | 1.60 average, 2 best |
| cleared a phase in | **4 of 5 games** | 3 of 5 games |
| enemies killed | 14.2 | 5.2 |

For scale, an agent doing nothing useful scores about **−12.6**. Both are clearly playing, not
flailing.

**Direct is much better than rotation, and that is expected, not a bug.** The rotation agent has to
learn to *aim* before it can learn to *shoot* — it spends a large part of its training budget on a
steering problem the direct agent simply does not have. This is worth a sentence in the report: the
comparison between the two control schemes is a graded item, and "one was harder to learn, here is
why" is a better answer than "one was worse".

### Watching them

```powershell
.venv\Scripts\python.exe -m eval.play_arena --style direct      # watch the AI play
.venv\Scripts\python.exe -m eval.play_arena --style rotation
.venv\Scripts\python.exe -m eval.play_arena --style direct --human   # play it yourself
.venv\Scripts\python.exe -m eval.play_arena --style both --episodes 5 --no-window   # just the numbers
```

`--human` matters for more than fun: if a person cannot clear phase 1 either, the problem is the
game, not the AI. It is the fastest way to tell those two apart.

## A mistake worth recording

The first version of the tests had a hole. I checked by deliberately breaking the code to see
whether the tests would notice — and one break slipped through: making every one of the 5 evaluation
games use the *same* arena instead of 5 different ones.

If that had shipped, the "± 16.36" in the table above would have been **structurally zero**, and the
report would have quoted a spread that was never actually measured. The tests now check each game
gets its own layout.

This is the second time on this project that writing a test was easy and making it actually *catch*
anything was the real work.

## Where the project stands

- **Part I: complete**, 13.5 / 13.5.
- **Part II: 11.5 of 14** now banked. Only the hyperparameter tuning row (A3-013, 3 points) is left.
- Remaining overall: tuning, the report, the video, and the creativity row.

## What I deliberately left alone

- **Hyperparameter tuning.** That is A3-013, a separate graded row. Mixing it in would make both
  harder to review, and the current settings already produce agents that clear phases.
- **The brief's fixed constants.** Untouched, as always.
- **Improving the rotation agent.** It meets the requirement. Making it better is a tuning job, and
  it belongs with A3-013.
