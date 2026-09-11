# A3-015 — Read-aloud narration script

Companion to `docs/plans/A3-015-video-run-sheet.md`. The run sheet tells the team what to do; this
tells the three presenters what to say. Scene numbers, presenters and durations match the run
sheet's shots exactly — the two documents are kept in sync on purpose, so a change to one number
is a change to both.

**Blockers:** before any of this is recorded, the working-tree hazards in the run sheet's §0 must
be resolved (the modified `models/ppo_direct.zip`, the modified `config/arena.yaml`, the corrupted
`results/arena_eval/comparison.md`, the missing `models/mechanics/`). This script assumes a clean
tree that matches HEAD. Don't re-derive those blockers here — go read §0.

---

## How to use this document (60 seconds)

Open this on a second screen while you record. Each scene below matches a numbered shot in the run
sheet and lasts exactly as long as that shot's budgeted duration — if a scene says 1:10, that's the
edited length, not a raw-take target.

**[DO]** lines are stage directions in brackets: the exact command, the exact keys, in the exact
order, and what to wait for on screen. Do these. Don't say them out loud.

**SAY:** lines are the actual words, each one attributed to whichever of you is presenting that
scene. Read them close to verbatim — they're written the way people actually talk, not written to
be summarised on the fly. Where a scene has two SAY beats, the first is short and goes *before* you
act (launching the command, pressing a key); the second is longer and goes *over* the action, while
something plays out on its own — an episode looping, a phase timer running down. Don't go silent
and just watch the window; that beat exists so you have something to say while it happens.

Every number in every SAY line is already measured and cited in the run sheet. Don't add one, round
one, or soften one to make a sentence flow better — if a line feels clunky because of a number,
say the sentence around the number differently, not the number itself.

If a take is running long, cut words and re-record the line — don't speed up your delivery to make
the clock. A rushed number reads as an apology; a missing one is worse.

## Saying the recurring terms — say it this way, every time

| Term | Say it like | Not like |
| --- | --- | --- |
| SARSA | one word, "SAR-suh" | spelled out, "S. A. R. S. A." |
| Q-learning | "cue-learning" (the letter Q) | "kuh-learning" |
| Q-table / Q-value | "cue-table" / "cue-value" | "queue-table" |
| epsilon-greedy | two words, "EP-suh-lon, GREE-dee" | "epsi-LON" or run together |
| alpha, gamma | plain English, "AL-fuh," "GAM-uh" | spelling out or over-enunciating |
| bootstrap(ping) | plain word, "BOOT-strap" — means plugging in an estimate of a future value, not "restart" | — |
| on-policy / off-policy | hyphenated, both halves stressed evenly | "on policy" as two unrelated words |
| PPO | three letters, "P — P — O" | as a word, "poh" |
| PyGame | "PY-game" | "pig-ame" or "pie-gaim" |
| TensorBoard | one compound word | "Tensor. Board." as two |

---

## Scene 1 — Title & team

**Duration:** 0:30 | **Presenters:** all three | **Words:** 66 (target ~66 at 130 wpm)

**[DO]** All three on camera together, or a quick sequential cut between three webcam shots over a
title card reading the assignment name and all three student numbers.

SAY (Duong): "Hey, I'm Duong."

SAY (Hanh): "I'm Hanh."

SAY (Nghia): "And I'm Nghia."

SAY (Duong): "This is our Games and AI Techniques assignment — reinforcement learning, in two
parts."

SAY (Hanh): "Part one's a gridworld where we hand-built Q-learning and SARSA ourselves. Part two's
a real-time arena where we trained a deep RL agent with PPO."

SAY (Nghia): "Everything you're about to watch is running live, on the actual models and code we
submitted — nothing here is staged."

---

## Scene 2 — Gridworld by hand

**Duration:** 0:20 | **Presenter:** Duong | **Words:** 45 (target ~44)

**[DO]** Terminal: `python -m gridworld.render --level 0`. Press `W`/`A`/`S`/`D` (or arrows) a
few times to move. Press `1`, then `4`, to switch levels live.

SAY (Duong): "So first thing — this is a real Pygame window, not a terminal print-out. I'm driving
it myself right now, W-A-S-D, one move at a time. And this whole thing's got seven different
levels — I can jump between them live, just by tapping a number key."

---

## Scene 3 — Level 0: Q-learning, a policy you can watch

**Duration:** 0:55 | **Presenter:** Duong | **Words:** 113 (target ~121)

**[DO]** Terminal: `python -m eval.play_gridworld --level 0 --algo q`. Policy arrows are already
on — let one loop play while you say the first beat. Press `P` once to toggle the arrows off, then
`P` again to bring them back. Press `Q` (or `H`) to turn on the Q-value heatmap for the second beat.

SAY (Duong — before, over the first loop): "This table learned through Q-learning — for every
square and direction, it keeps a running guess of how good that move is, and once it's trained, it
always takes its best guess. That's the greedy policy, and every arrow here is it."

SAY (Duong — over the loop, with the heatmap on): "Watch it walk the same route every loop:
seventeen steps, and that's the shortest possible path to all three apples — not luck, not a random
walk. That's the actual recipe it trained with, over here: alpha point one, gamma point nine five,
epsilon starting at one and decaying down to point oh five over its first two thousand episodes of
three thousand. That decay is why it explores early and commits late."

---

## Scene 4 — Level 1: Q-learning vs SARSA, one different assumption

**Duration:** 1:10 | **Presenter:** Duong | **Words:** 152 (target ~154)

**[DO]** Terminal: `python -m eval.play_gridworld --level 1 --compare`. Press `-` two or three
times to slow playback before hunting for the divergence point. Watch for the row where the two
ships' paths split — Q-learning's ship stays on the row against the fire wall, SARSA's ship sits
one row above it. Press `SPACE` to pause right there. Say the third beat over the paused frame,
then `SPACE` to resume.

SAY (Duong — before): "Same level, two learners, trained under identical conditions — same
learning rate, same discount, same exploration schedule. The only reason they end up different is
how each one guesses what happens after its next move."

SAY (Duong — second beat, while both panels are running): "Q-learning assumes it'll always make
the best possible choice from here on, even mid-exploration. So standing next to the fire feels
free to it, and it learns the tight route: eleven steps, right along the wall. SARSA's more honest
— it bases its guess on the move it's actually about to make next, randomness included, so it
knows a random step near the fire can kill it. It backs off a row: thirteen steps."

SAY (Duong — over the paused frame, at the split): "Watch the paths split, right here — Q-learning
stays low, SARSA climbs. Run each one five hundred times with some exploration left on: Q-learning
dies eleven point four percent of the time, SARSA dies one point eight percent. One extra row of
caution, from one different assumption about what happens next."

---

## Scene 5 — Level 5: monsters, a kill or a near miss, and the right order

**Duration:** 0:50 (5a ~0:20, 5b ~0:30) | **Presenter:** Duong | **Words:** 47 + 56 = 103 (target ~99)

### 5a — guaranteed kill or near miss, live

**[DO]** Terminal: `python -m gridworld.render --level 5`. Steer with `W`/`A`/`S`/`D` toward
whichever cell the monster is on *right now* — its position is stochastic, don't memorise a fixed
sequence, watch the screen and react. If it dodges, that's a near miss — narrate it as one. Press
`R` to reset once it's resolved.

SAY (Duong): "Monsters on this level move on their own — forty percent chance, after every move I
make, that one of them steps somewhere. Same move from me, different outcome depending on where
they wander — that's a stochastic transition. Walk into one and it's instant death, same as fire."

### 5b — trained agent, correct order, no risk taken

**[DO]** Terminal: `python -m eval.play_gridworld --level 5 --algo sarsa`. Policy arrows already
on. Let one full episode play (about five seconds).

SAY (Duong): "Now the trained policy, same level. Watch the key indicator — it switches on at step
seven, the moment it's standing on the key. It picks up an apple on the way, then opens the chest
at step twenty — key first, because the chest won't open without it. Last apple at thirty, and it
never once sets foot on either monster. Full clear — right order, no unnecessary risks."

---

## Scene 6 — Level 6: a negative result, stated plainly

**Duration:** 0:40 | **Presenter:** Duong | **Words:** 85 (target ~88)

**[DO]** Open `results/intrinsic_level6_q_curve.png` as a still image (`open
results/intrinsic_level6_q_curve.png` on macOS). Not a live pygame window for this scene.

SAY (Duong): "Here's a negative result — a clean one. We gave the agent a bonus for visiting new
cells, and it found the chest faster: about ninety-five episodes sooner. But over its final five
hundred episodes it only solves the level thirty-two point eight percent of the time, against a
hundred percent without the bonus. Turn exploration off and zero of five seeds solve it, versus
five of five. The bonus was worth more than the chest, so it learned to tour the level, not finish
it."

**Cut candidate:** if the edit is running long, this is the first scene to drop — see run sheet §2.

---

## Scene 7 — Creativity: the update, live

**Duration:** 0:45 | **Presenter:** Nghia | **Words:** 91 (target ~99)

**[DO]** Terminal: `python -m eval.play_gridworld --learn --level 1 --compare`. Opens paused, debug
panel already up. Press `N` once for the first real update — say the first beat over it. Press `N`
again for a second update as you finish the second beat. `R` resets to a fresh episode if needed.

SAY (Nghia — after the first `N`): "This panel shows the actual math, one update at a time — not
in a debugger, right here. Press N, and it runs one real step: the state, the move it made, the
reward, where it landed. Right there's epsilon, alpha, gamma, and whether that step rolled under
epsilon — exploring — or over it — greedy. That's its value before the update."

SAY (Nghia — after the second `N`): "Here's the one line that differs between the two: Q-learning
plugs in the best possible next move; SARSA plugs in whatever move it's actually about to take.
Same formula, one different number."

---

## Scene 8 — Part II intro

**Duration:** 0:25 | **Presenter:** Hanh | **Words:** 57 (target ~55)

**[DO]** Voiceover over a title card ("Part II — Arena, Deep RL"). No command yet — the cut lands
on Scene 9's window opening.

SAY (Hanh): "A lookup table doesn't work here. The arena runs in real time, positions are
continuous, there's a dozen enemies at once — too many states to ever list. So we use a small
neural network that estimates how good a state is, through a standard reset-step-render interface,
on a fixed twenty-number snapshot of the world instead of pixels."

---

## Scene 9 — Arena, direct control: what it sees, what it decides, what it's paid for

**Duration:** 1:15 | **Presenter:** Duong | **Words:** 156 (target ~165)

**[DO]** Terminal: `python -m eval.play_arena --style direct --no-save`. Observation and policy
overlays are already on — no key needed. The `PHASE 2` banner can land at any point in the take;
when it does, say the fourth beat in the moment, out of order if it has to be. Press `F3` once
during a kill for the reward panel, say the third beat, press `F3` again to close it before the
scene ends.

SAY (Duong — setup, before or just as it opens): "This is direct control — no rotating, just up,
down, left, right, and shoot. It's already cleared plenty of arenas, so let's just watch it work."

SAY (Duong — over the gameplay, the longest beat): "See these twenty numbers down the side? That's
everything the network knows right now — where it is, the nearest enemy and spawner, its health,
what phase we're in — with a line drawn from the ship to each target, so you see what it's looking
at. This panel's the network's actual output: a bar per action, and a number for how good this
moment is. Nothing here's a black box — observation in, decision out, live."

SAY (Duong — over the F3 panel): "Quick peek at what it's rewarded for: one point oh for a kill,
five for a spawner, ten for clearing a phase — minus zero point five for getting hit, minus ten if
it dies. That's exactly what it's trained to chase."

SAY (Duong — when the banner lands): "And there it is — phase two. Both spawners are down, so the
game just got harder."

---

## Scene 10 — Arena, rotation control: the second scheme

**Duration:** 0:35 | **Presenter:** Duong | **Words:** 71 (target ~77)

**[DO]** Terminal: `python -m eval.play_arena --style rotation --no-save`. Overlays already on —
no keys needed, just let it play.

SAY (Duong): "Second control scheme, same environment — but this one's a completely separate
trained model. Instead of moving directly, it thrusts forward and rotates, so there's five actions
instead of six: no strafing, just turn and go. Watch the policy panel relabel itself right there —
no-op, thrust, rotate left, rotate right, shoot. Different set of choices, but the same idea
underneath: a value for every option, and it always takes the best one."

---

## Scene 11 — Random baseline: the number that gives the others meaning

**Duration:** 0:20 | **Presenter:** Duong | **Words:** 48 (target ~44)

**[DO]** Terminal: `python -m eval.play_arena --style direct --random --no-save`. No keys.

SAY (Duong): "Now watch the same environment with no policy at all — just random button mashing.
It barely survives, it never gets close to clearing a phase, and its score just stays negative the
whole time. That's the zero point — everything we've shown you trained is being measured against
this."

---

## Scene 12 — Training and measurement rigor

**Duration:** 1:15 | **Presenter:** Hanh | **Words:** 161 (target ~165)

**[DO]** `tensorboard --logdir logs` — launched and indexed *before* recording; the scene cuts to
an already-loaded browser tab. Scroll the scalar view for the first beat. Cut to
`results/arena_sweep/sweep_table.md` on screen for the second beat. Cut to `git show
HEAD:results/arena_eval/comparison.md` and `git show HEAD:results/arena_eval/comparison_random.md`
shown as on-screen text/tables for the third beat.

SAY (Hanh — over TensorBoard): "Every run logged straight to TensorBoard — reward, episode length,
exploration, for every setup we tried."

SAY (Hanh — over the sweep table, the longest beat): "Here's how we tuned it. We took one baseline
and changed five hyperparameters, one at a time. The best change — a shorter rollout length, five
twelve instead of the default — held up on three seeds, at a phase score of one point oh two
seven. One other config looked great on reward alone — a bigger network — but its phase score
never moved, so we flagged it as reward hacking and threw it out. We even retrained the winner at
the full budget and ran it head to head against the model we already had. It lost — minus four
point one six against plus twenty-six point oh nine, and never cleared one phase in thirty tries.
So we kept what we had."

SAY (Hanh — over the comparison table): "Against doing nothing: direct control averages plus
twenty point six, clears a phase in twenty-eight of thirty. Random averages minus twelve point
eight, clears zero."

---

## Scene 13 — Creativity recap and close

**Duration:** 0:30 | **Presenter:** Nghia | **Words:** 68 (target ~66)

**[DO]** Text recap over the team card again. No live `--mechanics` clip — see run sheet §0/§6:
`models/mechanics/` doesn't exist, so that flag runs a random policy, not a trained one.

SAY (Nghia): "Beyond what the brief asked for, we built a few extra things: opt-in shield and
elite-enemy mechanics in the arena, the live update panel you just watched, and a reward breakdown
panel that shows exactly what the arena agent's being paid for, moment to moment. Everything you've
seen in this video is running on the actual models and the actual code we submitted — nothing's
staged. Thanks for watching."

---

## Total

13 scenes, 1,210 spoken words, **9:30** at the ~2.2 words/second pace this script is budgeted to —
against the 10:00 cap, that's **0:30 of slack**. Matches `A3-015-video-run-sheet.md` §2 exactly. If
a rehearsal shows the total running past 9:45, cut Scene 6 first (saves 0:40); it is the one scene
not required by the brief's checklist.

---

## Per-presenter cue sheets

Rehearse only your own block — scene numbers and running times below are cumulative *within this
list*, not the video's overall running total (see the full run sheet for where each scene actually
lands).

### Duong — 6:05 of material across 8 scenes

| Scene | Title | Duration | Running |
| --- | --- | ---: | ---: |
| 2 | Gridworld by hand | 0:20 | 0:20 |
| 3 | Level 0 — Q-learning | 0:55 | 1:15 |
| 4 | Level 1 — Q vs SARSA | 1:10 | 2:25 |
| 5 | Level 5 — monsters & ordering | 0:50 | 3:15 |
| 6 | Level 6 — intrinsic reward | 0:40 | 3:55 |
| 9 | Arena, direct | 1:15 | 5:10 |
| 10 | Arena, rotation | 0:35 | 5:45 |
| 11 | Arena, random baseline | 0:20 | 6:05 |

(Plus one line in Scene 1, ~0:07.)

### Hanh — 1:40 of material across 2 scenes

| Scene | Title | Duration | Running |
| --- | --- | ---: | ---: |
| 8 | Part II intro | 0:25 | 0:25 |
| 12 | Training & rigor | 1:15 | 1:40 |

(Plus one line in Scene 1, ~0:15.)

### Nghia — 1:15 of material across 2 scenes

| Scene | Title | Duration | Running |
| --- | --- | ---: | ---: |
| 7 | Creativity — live TD update | 0:45 | 0:45 |
| 13 | Creativity recap & close | 0:30 | 1:15 |

(Plus one line in Scene 1, ~0:05.)
