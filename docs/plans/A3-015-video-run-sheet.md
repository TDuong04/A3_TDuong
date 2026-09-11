# A3-015 — Video demonstration run sheet (rubric row V, 5 marks)

Produced by the `video-director` role. Verified against the repo at commit `2d75027` (branch
`main`) on 2026-09-11: every command below was launched with `SDL_VIDEODRIVER=dummy` and a short
`--frames` budget and confirmed to start cleanly before being written into this sheet. `pytest -m
"not slow"` passes on the current tree (355 passed).

**Companion document:** the read-aloud narration for every shot below is written out in full,
scene by scene, in `docs/plans/A3-015-video-script.md` — that is what the three presenters read
from while recording. The two documents are kept in sync: shot numbers, presenters and durations
match exactly, and the "Spoken line" in each shot below is the same words as that scene's script,
condensed to a single table cell.

**Read the blockers in §0 first.** Two of them (the dirty `models/ppo_direct.zip`, the corrupted
`results/arena_eval/comparison.md`) must be resolved before anyone presses record, or the video
will show a model or a number that is not the one in the submitted repo.

---

## 0. Pre-flight checklist (do this before any camera rolls)

### 0.1 Working-tree hazards — resolve first

- [ ] **`models/ppo_direct.zip` is modified relative to HEAD** (`git diff --stat -- models/` shows a
      181937→181941 byte change). The brief requires "the models shown must be the ones included in
      the repository." Before recording: either commit the intended retrain through the normal PR
      process, or `git checkout -- models/ppo_direct.zip` to restore the committed one. **Do not
      record against the currently modified file** — decide as a team which model is canonical, and
      confirm the arena eval numbers below still hold against whichever one you keep (re-run
      `python -m eval.play_arena --style direct --no-window` and diff the printed phase-reached
      figure against `git show HEAD:results/arena_eval/comparison.md`).
- [ ] **`config/arena.yaml` and `eval/play_arena.py` are modified relative to HEAD.** The diff in
      `config/arena.yaml` is mostly a 2-space→4-space re-indent, but it also changes
      `episode.max_enemies` from 40 to 70 — a real behaviour change. `eval/play_arena.py`'s diff adds
      an `ESC`-during-hold quit path (harmless, but still not what is committed). Run
      `git status --porcelain` immediately before recording and confirm it is clean (or matches
      exactly what will be committed and pushed before submission) — what is on screen must match
      the submitted zip.
- [ ] **`results/arena_eval/comparison.md` in the working tree is corrupted.** As of this check it
      holds a single-row, 3-episode `direct`-only table (return +18.29, phase 2.33) instead of
      HEAD's real 30-episode, two-style table (`direct` +20.57 ± 16.71, phase 2.37, 28/30 cleared;
      `rotation` +7.59 ± 5.33, phase 1.93, 28/30 cleared). This is ticket **A3-032**: the default
      `--episodes 3` silently overwrites the report table any time `play_arena` is run without
      `--no-save`. **Confirmed during this verification** — re-running the eval script for the
      dry-run checks below overwrote it again, twice, including a file (`comparison_random.md`)
      that had not previously been touched.
  - Fix for recording day: pass **`--no-save`** on every rehearsal or dry-run command in this sheet
    (added to the launch commands below). Never run `play_arena` without `--no-save` between now
    and submission unless you are deliberately regenerating the table with the full 30 episodes.
  - Whoever writes the numbers into narration or on-screen text must quote
    `git show HEAD:results/arena_eval/comparison.md`, not the working-tree file.
- [ ] `python -m pytest -m "not slow"` passes (355 passed at last check — re-run after resolving the
      above).
- [ ] `ruff check .` is clean (not part of the video, but a cheap final sanity check before
      recording from a tree you are about to claim matches the submission).

### 0.2 Recording mechanics

- [ ] Each presenter records their own segments on their own machine (screen capture + voiceover),
      stitched together in editing — this run sheet assumes that workflow, not a single continuous
      take with people swapping seats.
- [ ] Terminal font size ≥ 16pt; pygame windows are small (gridworld ~700–900px, arena 960×680 plus
      side panels ~1500px wide) — record at native resolution or higher, do not downscale below
      1080p, or the HUD numbers and overlay text a marker needs to read will blur out.
- [ ] Confirm `.venv/bin/python` (macOS/Linux) or `.venv\Scripts\python.exe` (Windows) is the
      interpreter actually on `PATH` / invoked, not a system Python — probe for it, do not assume.
- [ ] Audio check on every presenter's mic before their segment; narration is graded evidence too
      ("narrating 'it has learned' is not evidence" cuts both ways — the overlay has to be visible
      **and** the presenter has to say what it's showing).
- [ ] Dry-run every command in the table below once, off-camera, in order, the day before recording.
      Confirm the phase-progression take (Shot 9) and the level-1 divergence take (Shot 4) actually
      produce the moments described — both depend on details below that assume the HEAD model/config,
      not the current dirty tree.
- [ ] Record **raw takes longer than the final edited duration** for any shot whose key moment is
      timing-dependent (Shots 5 and 9) — see §5 Contingency. Cut down in editing; do not try to time
      a live take to the second.
- [ ] Rehearse from `docs/plans/A3-015-video-script.md`, not by improvising from this run sheet's
      "Spoken line" column — that column is a condensed reference, the script is what gets read.

---

## 1. Presenters

| Presenter | Student no. | Owns (from README "Team"/git log) | Segments in this run sheet |
| --- | --- | --- | --- |
| Huynh Thai Duong | s3978955 | Both environments end to end: gridworld core/renderer/levels/Q-learning/SARSA, arena entities/physics/Gym API/renderer/HUD, training + playback for both control schemes | Shots 2, 3, 4, 5, 6, 9, 10, 11 |
| Do Le Trang Hanh | s3977994 | Part II training rigor: TensorBoard pipeline, hyperparameter sweep + tabulation, env validation, the stale-observation-regression retrain, on-screen algorithm visibility, ticket board/audits | Shots 8, 12 |
| Tran Minh Nghia | s4123236 | Creativity: opt-in arena mechanics (shields, elite chargers), the gridworld learning-internals debug overlay, arena policy/reward/physics debugging tooling | Shots 7, 13 |

All three also appear together in Shot 1. Every member presents at least one part with their own
voice over their own screen — this satisfies the brief's "must present at least 1 part" literally,
not just by being on camera.

---

## 2. Timing budget

Final **edited** run time: **9 minutes 30 seconds**, against a 10:00 hard cap — **0:30 of slack**.
The narration was rewritten to lead with the plain-language idea before the technical term, sound
spoken rather than compact, and give every quoted number the half-sentence of context that makes it
meaningful — which costs more words than the original caption-style lines, so six of the thirteen
shots (1, 3, 4, 6, 7, 8) run longer than in the first draft of this sheet. The other seven fit
their original slot unchanged. If the edit is running long, cut **Shot 6** first (saves 0:40) — it
is the one shot not required by the brief's checklist (see §4).

| # | Shot | Presenter | Duration | Running total |
| - | ---- | --------- | -------: | -------------: |
| 1 | Title & team | All three | 0:30 | 0:30 |
| 2 | Gridworld by hand | Duong | 0:20 | 0:50 |
| 3 | Level 0 — Q-learning learned policy | Duong | 0:55 | 1:45 |
| 4 | Level 1 — Q vs SARSA divergence | Duong | 1:10 | 2:55 |
| 5 | Level 5 — monsters + item ordering + kill/near-miss | Duong | 0:50 | 3:45 |
| 6 | Level 6 — intrinsic reward, honest result | Duong | 0:40 | 4:25 |
| 7 | Creativity — gridworld learning-internals overlay | Nghia | 0:45 | 5:10 |
| 8 | Part II intro | Hanh | 0:25 | 5:35 |
| 9 | Arena, direct style — overlays + phase progression | Duong | 1:15 | 6:50 |
| 10 | Arena, rotation style — second control scheme | Duong | 0:35 | 7:25 |
| 11 | Arena, random baseline contrast | Duong | 0:20 | 7:45 |
| 12 | Training & rigor — TensorBoard, sweep, numbers table | Hanh | 1:15 | 9:00 |
| 13 | Creativity recap & close | Nghia | 0:30 | 9:30 |

---

## 3. Shot-by-shot run sheet

Every command is `python -m ...` from the repo root, using `.venv/bin/python` (or
`.venv\Scripts\python.exe` on Windows) as the interpreter — do **not** call `python3`/`python` from
outside the venv, and never invoke a script by its file path. The "Spoken line" in each shot is the
same wording as the matching scene in `docs/plans/A3-015-video-script.md`, joined into one
paragraph — where a shot has more than one beat there, the beats are separated by " / " below.

### Shot 1 — Title & team

| Field | Detail |
| --- | --- |
| Presenter | All three |
| Duration | 0:30 |
| Command | none — title card / all three on camera together |
| Keys | none |
| Visible | Team name, assignment title, all three faces/names with student numbers on screen at once |
| Spoken line | Duong: "Hey, I'm Duong." Hanh: "I'm Hanh." Nghia: "And I'm Nghia." Duong: "This is our Games and AI Techniques assignment — reinforcement learning, in two parts." Hanh: "Part one's a gridworld where we hand-built Q-learning and SARSA ourselves. Part two's a real-time arena where we trained a deep RL agent with PPO." Nghia: "Everything you're about to watch is running live, on the actual models and code we submitted — nothing here is staged." |
| Concept / rubric | Satisfies "all three members appear" — the one moment that must be unambiguous on its own, independent of each presenter's later segment |

### Shot 2 — Gridworld is genuinely interactive

| Field | Detail |
| --- | --- |
| Presenter | Duong |
| Duration | 0:20 |
| Command | `python -m gridworld.render --level 0` |
| Keys | `W`/`A`/`S`/`D` (or arrows) a few times to move by hand; `1`, then `4` to switch levels live |
| Visible | A real Pygame window (never a terminal); the agent moves on each keypress; top HUD bar reads `Level N   steps N   return +0.0   key no   items 0/3`; level swap redraws the grid instantly |
| Spoken line | "So first thing — this is a real Pygame window, not a terminal print-out. I'm driving it myself right now, W-A-S-D, one move at a time. And this whole thing's got seven different levels — I can jump between them live, just by tapping a number key." |
| Concept / rubric | RL loop — observe state, choose action, transition, reward. Rubric V: gridworld in a Pygame window, never a terminal; environment is genuinely interactive |

### Shot 3 — Level 0: Q-learning, clear evidence of a learned policy

| Field | Detail |
| --- | --- |
| Presenter | Duong |
| Duration | 0:55 |
| Command | `python -m eval.play_gridworld --level 0 --algo q` |
| Keys | Policy arrows are **on by default** — narrate over 1–2 loops; press `P` once to toggle them off then on (proves it's a live overlay, not baked art); press `Q` (or `H`) to add the Q-value heatmap; optionally `-` once if 6 steps/sec plays too fast to narrate over |
| Visible | Arrows over every cell pointing toward the three apples; heatmap colouring once `Q` is pressed; side "Learner" panel reading `algorithm Q-learning`, `alpha 0.1`, `gamma 0.95`, `epsilon 1 -> 0.05`, `decay over 2000 ep`, `trained 3000 ep` — read from that table's own training summary; agent walks the route in 17 steps every loop |
| Spoken line | "This table learned through Q-learning — for every square and direction, it keeps a running guess of how good that move is, and once it's trained, it always takes its best guess. That's the greedy policy, and every arrow here is it. / Watch it walk the same route every loop: seventeen steps, and that's the shortest possible path to all three apples — not luck, not a random walk. That's the actual recipe it trained with, over here: alpha point one, gamma point nine five, epsilon starting at one and decaying down to point oh five over its first two thousand episodes of three thousand. That decay is why it explores early and commits late." |
| Concept / rubric | Q(s,a) and the greedy policy; linear epsilon-greedy decay. Rubric V: Q-learning agent in Pygame; **clear evidence of a learned policy** — cite `results/summary_level0_q_seed0.json`: `greedy_steps: 17`, `bfs_optimum_steps: 17`, `optimal: true` |

### Shot 4 — Level 1: Q-learning vs SARSA, on-policy vs off-policy

| Field | Detail |
| --- | --- |
| Presenter | Duong |
| Duration | 1:10 |
| Command | `python -m eval.play_gridworld --level 1 --compare` |
| Keys | Press `-` two or three times to slow playback (11–13 step episodes at the default 6 steps/sec are over in ~2 seconds — too fast to narrate live); watch for the row where the two panels' paths split (SARSA detours one row further from the fire wall); `SPACE` to pause right at that row; narrate; `SPACE` to resume; `TAB` once to show the single-panel cycle view exists |
| Visible | Two side-by-side grids, same shared "Learner" panel (`alpha 0.1`, `gamma 0.95`, `epsilon 1 -> 0.05`, `decay over 6000 ep` — proving both algorithms trained under one identical schedule); Q-learning's ship hugs the row directly above the fire (11 steps); SARSA's ship stays one row back (13 steps) |
| Spoken line | "Same level, two learners, trained under identical conditions — same learning rate, same discount, same exploration schedule. The only reason they end up different is how each one guesses what happens after its next move. / Q-learning assumes it'll always make the best possible choice from here on, even mid-exploration. So standing next to the fire feels free to it, and it learns the tight route: eleven steps, right along the wall. SARSA's more honest — it bases its guess on the move it's actually about to make next, randomness included, so it knows a random step near the fire can kill it. It backs off a row: thirteen steps. / Watch the paths split, right here — Q-learning stays low, SARSA climbs. Run each one five hundred times with some exploration left on: Q-learning dies eleven point four percent of the time, SARSA dies one point eight percent. One extra row of caution, from one different assumption about what happens next." |
| Concept / rubric | On-policy vs off-policy; Q-learning's bootstrap term (the best possible next value) vs SARSA's (the value of the action actually chosen next). Rubric V: SARSA on screen; comparison evidence. Cite `results/comparison_level1_seed0.md`: Q 11 steps via rows [7,8] (BFS optimum), SARSA 13 steps via rows [6,7,8]; death rate at eps 0.05 over 500 rollouts — Q 11.4%, SARSA 1.8% |
| **Caveat** | Do **not** say "SARSA never goes near the fire" — see §6 do-not-say list; on some seeds its route clips one fire-adjacent cell |

### Shot 5 — Level 5: monsters, stochastic transitions, apple/key/chest ordering, a kill or near miss

Two parts, same command family, cut together.

**5a — guaranteed kill/near miss, presenter-controlled**

| Field | Detail |
| --- | --- |
| Command | `python -m gridworld.render --level 5` |
| Keys | `W`/`A`/`S`/`D` — **steer live toward whichever cell the monster is currently on**; its position is stochastic (40% move chance per turn), so do not memorise a fixed key sequence — watch the screen and walk into it. If it dodges, narrate the near miss instead of the kill. `R` resets after |
| Visible | Instant death on contact — HUD flips to `DEAD` in red, same instant-kill rule as fire |
| Spoken line | "Monsters on this level move on their own — forty percent chance, after every move I make, that one of them steps somewhere. Same move from me, different outcome depending on where they wander — that's a stochastic transition. Walk into one and it's instant death, same as fire." |
| Concept / rubric | Stochastic transitions. Rubric V: monster behaviour visible; a monster kill or near miss |

**5b — trained agent, correct item ordering, hazard avoidance**

| Field | Detail |
| --- | --- |
| Command | `python -m eval.play_gridworld --level 5 --algo sarsa` |
| Keys | Policy arrows already on; let one full episode play (~5 seconds at default speed; `-` once if narrating over it) |
| Visible | HUD `key no` flips to `key YES` only after the key cell; item counter climbs `1/4 → 2/4 → 3/4 → 4/4`; agent completes the level in 30 steps without dying, never entering either monster's cell |
| Spoken line | "Now the trained policy, same level. Watch the key indicator — it switches on at step seven, the moment it's standing on the key. It picks up an apple on the way, then opens the chest at step twenty — key first, because the chest won't open without it. Last apple at thirty, and it never once sets foot on either monster. Full clear — right order, no unnecessary risks." |
| Concept / rubric | Markov state — key-held / chest-opened must be part of the state; stochastic transitions. Rubric V: correct item and monster behaviour, apple/key/chest ordering. Cite `results/summary_level5_sarsa_seed0.json`: `key_precedes_chest: true`, `collection_order_text: "K(6,1)@7 -> A(8,2)@10 -> C(9,7)@20 -> A(2,8)@30"`, `greedy_died: false` |
| Note (not shown live) | The same file's Q-learning counterpart, `results/summary_level5_q_seed0.json`, records `greedy_died: true` on this seed — an honest, citable example of Q-learning's riskier off-policy behaviour costing it the level. Worth a spoken aside ("on this same level, our committed Q-learning table's recorded rollout actually dies — that's the off-policy risk-taking we just talked about") but do not stage it live: it depends on a monster-movement seed not fixed by this command, and a live death that fails to reproduce the citation is worse than not showing it |

### Shot 6 — Level 6: intrinsic reward, an honest negative result

| Field | Detail |
| --- | --- |
| Presenter | Duong |
| Duration | 0:40 |
| Command | Open `results/intrinsic_level6_q_curve.png` as a still image (e.g. `open results/intrinsic_level6_q_curve.png` on macOS) — **not** a live pygame window for this one |
| Keys | none |
| Visible | The plotted training curves for intrinsic strength 0 vs 0.5 |
| Spoken line | "Here's a negative result — a clean one. We gave the agent a bonus for visiting new cells, and it found the chest faster: about ninety-five episodes sooner. But over its final five hundred episodes it only solves the level thirty-two point eight percent of the time, against a hundred percent without the bonus. Turn exploration off and zero of five seeds solve it, versus five of five. The bonus was worth more than the chest, so it learned to tour the level, not finish it." |
| Concept / rubric | Intrinsic reward; the bonus outweighing the reward it's meant to help find. **Not required by the brief's video checklist** — included for the intrinsic-reward beat in the workable shape and because an honest negative result is stronger evidence of rigor than a cherry-picked win. **First cut if the edit runs long, saves 0:40** |
| **Caveat** | Never call this a success. State both numbers with their baseline every time (see §6) |

### Shot 7 — Creativity: the learning-internals debug overlay, live

| Field | Detail |
| --- | --- |
| Presenter | Nghia |
| Duration | 0:45 |
| Command | `python -m eval.play_gridworld --learn --level 1 --compare` |
| Keys | Session opens **paused, with the debug panel already up** (message: "live learning: SPACE to run, N to update"). Press `N` once for the first update, say the first beat over it; press `N` again for a second update, say the second beat; `R` starts a fresh learning episode if needed |
| Visible | "Latest learning update" panel: `s = ...`, `a = ...`, `r_env = ...`, `s' = ...`, `epsilon = ... alpha = ... gamma = ...`, `roll = ... < epsilon: exploratory branch` (or `>=`: greedy), `Q(s,a) before = ...`, the bootstrap term labelled `max_a' Q(s',a')` for the Q-learning panel vs `Q(s', a'_chosen)` for the SARSA panel, `TD target = ...`, `TD error = ...`, `Q_after = ...` |
| Spoken line | "This panel shows the actual math, one update at a time — not in a debugger, right here. Press N, and it runs one real step: the state, the move it made, the reward, where it landed. Right there's epsilon, alpha, gamma, and whether that step rolled under epsilon — exploring — or over it — greedy. That's its value before the update. / Here's the one line that differs between the two: Q-learning plugs in the best possible next move; SARSA plugs in whatever move it's actually about to take. Same formula, one different number." |
| Concept / rubric | The Q-learning and SARSA update rules, differing only in the bootstrap term; the epsilon-greedy exploratory/greedy branch. This is the CLAUDE.md "visibility first" rule made literal, and the strongest possible reinforcement of rubric V's "clear evidence of a learned policy, not random actions" — it shows the mechanism producing the policy, not just its result |

### Shot 8 — Part II intro

| Field | Detail |
| --- | --- |
| Presenter | Hanh |
| Duration | 0:25 |
| Command | none — voiceover over a title card ("Part II — Arena, Deep RL") |
| Keys | none |
| Visible | Title card only; the cut lands on Shot 9's window opening |
| Spoken line | "A lookup table doesn't work here. The arena runs in real time, positions are continuous, there's a dozen enemies at once — too many states to ever list. So we use a small neural network that estimates how good a state is, through a standard reset-step-render interface, on a fixed twenty-number snapshot of the world instead of pixels." |
| Concept / rubric | Why tabular RL fails at this scale / function approximation; the Gym-style API; the observation vector |

### Shot 9 — Arena, direct control, overlays and a phase progression

| Field | Detail |
| --- | --- |
| Presenter | Duong |
| Duration | 1:15 (final, edited — **record a longer raw take**, see §5) |
| Command | `python -m eval.play_arena --style direct --no-save` |
| Keys | Observation and policy overlays are **on by default** (`show_observation_overlay: true`, `show_policy_overlay: true` in `config/arena.yaml`) — no key needed to reveal them. Press `F3` once during a kill to open the reward-decomposition debug panel; `F3` again to close it before the segment ends. The `PHASE 2` banner can land at any point in the take — react to it in the moment, even out of the beat order written below |
| Visible | Enemies spawning from spawners and homing toward the player; projectiles; collisions; the health bar dropping on a hit; a `PHASE 2` banner (1.6s, centred) the moment both phase-1 spawners are destroyed; the OBSERVATION panel — all 20 live features, with lines drawn from the ship to the nearest enemy and nearest spawner; the POLICY panel — one bar per action with the chosen action highlighted, and the critic's `V(s)`; while F3 is open, the per-term reward table (`Enemy destroyed (+1)`, `Spawner destroyed (+5)`, `Phase advance (+10)`, `Damage event (-0.5)`, `Death (-10)`, `Step penalty (-0.01)`) against `step() reward = ...` and `Episode total = ...` |
| Spoken line | "This is direct control — no rotating, just up, down, left, right, and shoot. It's already cleared plenty of arenas, so let's just watch it work. / See these twenty numbers down the side? That's everything the network knows right now — where it is, the nearest enemy and spawner, its health, what phase we're in — with a line drawn from the ship to each target, so you see what it's looking at. This panel's the network's actual output: a bar per action, and a number for how good this moment is. Nothing here's a black box — observation in, decision out, live. / Quick peek at what it's rewarded for: one point oh for a kill, five for a spawner, ten for clearing a phase — minus zero point five for getting hit, minus ten if it dies. That's exactly what it's trained to chase. / And there it is — phase two. Both spawners are down, so the game just got harder." |
| Concept / rubric | The observation vector; actor-critic PPO, policy vs value output; reward design. Rubric V: trained agent controlling the player; enemies, projectiles, collisions all visible; **at least one phase progression on camera** |
| **Evidence for reliability** | Per `git show HEAD:results/arena_eval/eval_direct_seed0.json`, `env.reset(seed=0+episode)` means the default `--seed 0 --episodes 3` plays exactly the seeded episodes recorded at `phases: [2, 3, 3, ...]` — i.e. against the **committed** model, the first three default episodes reach phase 2, phase 3 and phase 3. This holds only if `models/ppo_direct.zip` has been restored to HEAD per §0.1 |

### Shot 10 — Arena, rotation control: the second control scheme

| Field | Detail |
| --- | --- |
| Presenter | Duong |
| Duration | 0:35 |
| Command | `python -m eval.play_arena --style rotation --no-save` |
| Keys | Overlays already on by default; no keys required — let it play |
| Visible | The ship rotates and thrusts instead of moving on four fixed axes; the POLICY panel's action labels are now `NOOP / THRUST / ROT-LEFT / ROT-RIGHT / SHOOT` (Discrete(5)) instead of direct movement's six actions; a `PHASE 2` banner |
| Spoken line | "Second control scheme, same environment — but this one's a completely separate trained model. Instead of moving directly, it thrusts forward and rotates, so there's five actions instead of six: no strafing, just turn and go. Watch the policy panel relabel itself right there — no-op, thrust, rotate left, rotate right, shoot. Different set of choices, but the same idea underneath: a value for every option, and it always takes the best one." |
| Concept / rubric | Discrete action-space design, two control schemes. Rubric V: **both control schemes demonstrated** |
| **Evidence for reliability** | Per `git show HEAD:results/arena_eval/eval_rotation_seed0.json`, the first five seeded episodes (`seed 0` default) all record `phases: [2, 2, 2, 2, 2]` — phase 2 is reached in every one of the first five default episodes against the committed rotation model |

### Shot 11 — Random baseline: the contrast that makes "learned" mean something

| Field | Detail |
| --- | --- |
| Presenter | Duong |
| Duration | 0:20 |
| Command | `python -m eval.play_arena --style direct --random --no-save` |
| Keys | none |
| Visible | The ship moves erratically, takes damage almost immediately, dies fast; HUD return goes and stays negative; no phase banner ever appears |
| Spoken line | "Now watch the same environment with no policy at all — just random button mashing. It barely survives, it never gets close to clearing a phase, and its score just stays negative the whole time. That's the zero point — everything we've shown you trained is being measured against this." |
| Concept / rubric | Rubric V: this is the shot the brief calls out as worth 15 seconds — "learned, not random" shown, not asserted. Cite `git show HEAD:results/arena_eval/comparison_random.md`: `direct` random −12.85 ± 1.44, phase 1.00, 0/30 cleared, 0% survival, vs the trained `direct` +20.57 ± 16.71, phase 2.37, 28/30 cleared, 53% survival |

### Shot 12 — Training and measurement rigor

| Field | Detail |
| --- | --- |
| Presenter | Hanh |
| Duration | 1:15 |
| Command | `tensorboard --logdir logs` — **launch and let it index before recording**, do not start it live on camera; the segment cuts to an already-loaded browser tab, then to `results/arena_sweep/sweep_table.md` and `git show HEAD:results/arena_eval/comparison.md` shown as on-screen text/tables |
| Keys | Scroll through the TensorBoard scalar tab (episode reward, episode length, entropy); no pygame keys |
| Visible | TensorBoard reward/entropy curves climbing across training; the sweep table — five axes varied one at a time from the baseline, winner `n_steps=512` confirmed on seeds 0/1/2 at phase 1.027 ± 0.012; a callout that the 128×128 network was flagged and rejected as suspected reward hacking (reward up, phase flat); the retrained challenger `ppo_direct_sweep.zip` losing head-to-head to the incumbent (−4.16 ± 10.05, phase 1.00, 0/30 cleared vs +26.09 ± 19.01, phase 2.47, 29/30 cleared) — and the decision to keep the incumbent; the full 30-episode HEAD comparison table beside its random baseline |
| Spoken line | "Every run logged straight to TensorBoard — reward, episode length, exploration, for every setup we tried. / Here's how we tuned it. We took one baseline and changed five hyperparameters, one at a time. The best change — a shorter rollout length, five twelve instead of the default — held up on three seeds, at a phase score of one point oh two seven. One other config looked great on reward alone — a bigger network — but its phase score never moved, so we flagged it as reward hacking and threw it out. We even retrained the winner at the full budget and ran it head to head against the model we already had. It lost — minus four point one six against plus twenty-six point oh nine, and never cleared one phase in thirty tries. So we kept what we had. / Against doing nothing: direct control averages plus twenty point six, clears a phase in twenty-eight of thirty. Random averages minus twelve point eight, clears zero." |
| Concept / rubric | Training loop, TensorBoard logging, checkpointing; meaningful hyperparameter tuning (rubric J3). Not a strict rubric-V checklist item, but requested in the workable shape and it is the strongest evidence in the video that a number was measured, not eyeballed — always state a baseline beside a performance number (CLAUDE.md §6) |

### Shot 13 — Creativity recap and close

| Field | Detail |
| --- | --- |
| Presenter | Nghia |
| Duration | 0:30 |
| Command | none — text recap over the team card again |
| Keys | none |
| Visible | Team names/numbers once more; a short on-screen list: opt-in shield/elite mechanics, the gridworld TD-update overlay just shown, the arena F3 reward-decomposition panel |
| Spoken line | "Beyond what the brief asked for, we built a few extra things: opt-in shield and elite-enemy mechanics in the arena, the live update panel you just watched, and a reward breakdown panel that shows exactly what the arena agent's being paid for, moment to moment. Everything you've seen in this video is running on the actual models and the actual code we submitted — nothing's staged. Thanks for watching." |
| Concept / rubric | Creativity (separate 5-mark row) — closes the video cleanly under the 10-minute cap |
| **Note** | Do **not** show a live `--mechanics` arena clip here. Verified during this pass: `models/mechanics/` does not exist, so `python -m eval.play_arena --style direct --mechanics` prints `"No trained models in models/mechanics -- running a random policy"` and runs a **random** policy. Showing that next to the "trained agent" claim would contradict rubric V's requirement that the agent shown is trained. If a trained mechanics model exists by recording day, a 10–15s clip can replace the text recap — otherwise, text/stills only |

---

## 4. Coverage table — every brief bullet against a named shot

| Brief requirement | Shot(s) | Satisfied by |
| --- | --- | --- |
| Under 10 minutes | Whole video | 9:30 edited total, 0:30 slack (§2) |
| All three members appear and each presents ≥1 part | 1 (appear), 2–6/9–11 (Duong), 8/12 (Hanh), 7/13 (Nghia) | Each has a dedicated, named, voiced segment — not just a cameo |
| Gridworld in a Pygame window, Q-learning or SARSA agent | 3 (Q-learning), 4 & 5b (SARSA) | Window confirmed via `SDL_VIDEODRIVER=dummy` launch test; never a terminal |
| Correct item/monster behaviour (apple, key, chest ordering; monster kill or near miss) | 5a (kill/near miss), 5b (ordering) | `key_precedes_chest: true`, `collection_order_text` cited from `results/summary_level5_sarsa_seed0.json` |
| Clear evidence of a learned policy, not random actions | 3 (arrows + heatmap + near-optimal rollout), reinforced by 7 (TD update mechanism) and 11 (random contrast) | `greedy_steps: 17` = `bfs_optimum_steps: 17` cited; overlay + heatmap on screen, not asserted |
| Arena: trained agent, enemies, projectiles, collisions, ≥1 phase progression | 9 | `PHASE 2` banner; reliability backed by `eval_direct_seed0.json`'s recorded per-episode phases `[2, 3, 3]` for the default seed/episode range |
| Both control schemes demonstrated | 9 (direct), 10 (rotation) | Two short clips, per the brief's own allowance |
| Uploaded unlisted, link in the report PDF | — (post-production) | Not a shot — a submission step. Add to the team's final checklist: upload unlisted to YouTube or RMIT OneDrive, paste the link into `report/report.md` before export |

---

## 5. Contingency notes

- **Shot 5a (monster kill/near miss) is live and reactive, not scripted.** The monster's position
  is stochastic — do not memorise a fixed key sequence. Record 2–3 attempts if needed; keep whichever
  one lands a clean kill or an unambiguous near miss (the ship passing within one cell of the
  monster counts — narrate it as such). This shot is safe to re-record independently of everything
  else in Shot 5; it does not touch the trained table used in 5b.
- **Shot 5b depends on `results/qtable_level5_sarsa_seed0.npz` existing and matching what's
  committed** — confirmed present in this pass. If the team retrains level 5 before submission,
  re-verify `greedy_died: false` and `key_precedes_chest: true` in the new summary before reusing
  this shot's narration numbers.
- **Shot 4's divergence moment is fast (11–13 steps at 6 steps/sec ≈ 2 seconds per loop).** Slow
  down with `-` *before* hunting for the pause point, not after — pressing `-` mid-episode does not
  retroactively slow the frames already shown. If the pause lands on the wrong row, `R` resets both
  panels together and the loop restarts.
- **Shot 9's phase progression is the single highest-risk moment in the video** — if it does not
  occur, the arena section is what gets cut per the brief's own remark on this. Mitigations, in
  order of preference:
  1. Confirm `models/ppo_direct.zip` is the committed HEAD version (§0.1) before recording — the
     `[2, 3, 3]` per-episode phase record only holds for that exact model.
  2. Record continuously through **at least the first default episode** (raw ~60–90 seconds is
     enough; phase 1 is deliberately paced by ticket A3-022 to clear in ~20–30 real seconds for a
     trained agent) rather than trying to catch the banner live and cut immediately.
  3. If episode 0 does not advance, let the default 3-episode run continue — episodes 1 and 2 are
     also recorded in `eval_direct_seed0.json` as reaching phase 3.
  4. If none of the default three episodes advance (e.g. because the model in `models/` differs
     from what produced that table), increase `--episodes` (`--episodes 6 --no-save`) rather than
     switching narration to claim a progression that is not on screen. **Re-record rather than cut
     away** — a missing phase progression is a lost mark regardless of how good the rest of the cut is.
  5. Rotation style (Shot 10) is the safer fallback for guaranteeing *a* phase progression — its
     first five default episodes all reach phase 2 per `eval_rotation_seed0.json` — but the brief's
     phase-progression bullet does not require it on both styles, so use direct for the primary
     claim and keep rotation in reserve only if direct's take fails outright.
- **Shot 6 is the first thing to cut** if the edit is running long — it is not on the brief's
  checklist, only in the workable shape. Cutting it costs 0:40 and no rubric-V coverage.
- **Independently re-recordable shots:** 1, 2, 5a, 6, 7, 8, 11, 13 have no dependency on another
  shot's state (each opens a fresh env/table). Shots 3, 4, 5b, 9, 10, 12 depend on the committed
  models/tables in §0.1 being the ones actually on disk when recorded — re-verify those first if
  re-recording any of them on a different day.

---

## 6. Do-not-say list — claims the evidence does not support

- **Do not say "SARSA never goes near the fire."** `results/comparison_level1_seed0.md` states this
  explicitly as a caveat: on some seeds SARSA's route clips one fire-adjacent cell and its death
  rate rises to ~4%. Say "SARSA keeps its distance / is more hazard-averse," backed by the death-rate
  numbers (Q 11.4% vs SARSA 1.8% at eps 0.05), not "never."
- **Do not call the level 6 intrinsic-reward result a success.** It is a measured negative result:
  strength 0.5 solves the level 32.8% of the time over the final 500 episodes vs the baseline's
  100%, and 0/5 seeds solve it greedily vs 5/5 for the baseline. State the trade-off (faster first
  discovery, worse final policy) and the reason (discounted intrinsic return ≈ 5× the chest's value),
  not "it worked."
- **Do not quote a performance number without its baseline.** Every arena number in this sheet is
  paired with the random-policy number on the same seeded arenas — keep that pairing in the
  narration, not just in the shot's internal notes.
- **Do not quote `results/arena_eval/comparison.md` as it currently sits in the working tree.** It
  is corrupted to a single-row, 3-episode table (ticket A3-032). Always cite
  `git show HEAD:results/arena_eval/comparison.md`, or re-generate it properly
  (`python -m eval.play_arena --style both --episodes 30 --no-window`) and diff against HEAD before
  trusting the regenerated file.
- **Do not present the `--mechanics` arena clip as trained behaviour.** `models/mechanics/` does not
  exist; that flag currently falls back to a random policy (confirmed in this pass). If shown at
  all, it must be labelled as untrained/creativity-only.
- **Do not claim "the models shown are the ones in the repo" while `models/ppo_direct.zip` is
  modified relative to HEAD.** Resolve §0.1 first.
- **Do not call the sweep's 128×128 config "reward hacking" as an unqualified fact beyond what the
  evidence shows.** `results/arena_sweep/sweep_table.md` flags it as *suspected* reward hacking —
  reward rising while phase-reached stayed flat. Shot 12's line ("we flagged it as reward hacking")
  attributes the call to the team's own flagging criterion, which is what actually happened — don't
  strengthen it to "proven" on camera.

---

## 7. Defects noticed in the playback scripts (reported, not fixed — out of this role's scope)

- **`eval/play_arena.py` / `--episodes` default of 3 silently overwrites
  `results/arena_eval/comparison*.md` and the per-style JSON files on every run that isn't
  `--no-save`**, including runs made purely to check a command launches. This is already tracked as
  ticket **A3-032** (P1, open). Confirmed reproducible during this verification pass: two additional
  files (`comparison_random.md`, `eval_random_direct_seed0.json`) were overwritten by dry-run
  launches that had no intention of writing results, and could not be restored from this role's
  permissions (`git checkout` on results files was blocked by the sandbox's command classifier).
  Recommend the fix raise the default `--episodes` for anything that writes to
  `results/arena_eval/`, or require `--no-save` be the default and `--save` opt-in.
- **`python -m eval.play_arena --style direct --mechanics` has no guard or warning beyond a printed
  line** ("No trained models in models/mechanics -- running a random policy") when asked to show
  "the trained deep RL agent" implicitly by a presenter who forgets to read stdout. Given the video
  rubric's explicit requirement that "the agent shown is trained," this is worth a louder on-screen
  (not just stdout) warning banner — currently the pygame window itself gives no visual indication
  the policy is random, only the HUD's unlabelled behaviour would tip off a careful viewer.

---

## Total time: 9:30 (cap 10:00, slack 0:30)

## Blockers to clear before recording

1. `models/ppo_direct.zip` modified vs HEAD — resolve per §0.1 before Shots 9/10/12.
2. `config/arena.yaml` and `eval/play_arena.py` modified vs HEAD — working tree must be clean (or
   exactly what gets committed) before recording, so what's on screen matches the submission.
3. `results/arena_eval/comparison.md` (and `comparison_random.md`) corrupted in the working tree —
   quote HEAD's version in narration/on-screen text; pass `--no-save` on every rehearsal command.
4. No trained model in `models/mechanics/` — Shot 13 must not show a live `--mechanics` clip as
   "trained," per §6.
