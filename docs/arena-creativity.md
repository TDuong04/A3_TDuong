# Arena creativity features (A3-018)

Three cohesive feature groups satisfy the ticket: perception, human control, and
combat feedback. The existing A3-021 renderer is reused. The missing A3-023 overlay
and the human-play portion of A3-012 are implemented here. A3-022's broader HUD,
health bars and phase banners, and A3-012's trained-model playback remain separate.

## Demo

From the repository root with requirements installed:

```sh
python -m eval.play_arena --human --style direct --seed 0
python -m eval.play_arena --human --style rotation --seed 0
python -m arena.render --style direct --seed 0
python -m arena.render --style rotation --seed 0 --headless --frames 1200
```

The renderer command defaults to a scripted demonstration, not a trained policy.
The evaluation entry point currently requires `--human`.

| Control | Action |
| --- | --- |
| WASD / arrows, direct | Move and face that direction |
| W / up, rotation | Thrust |
| A/D / left/right, rotation | Turn |
| Space | Shoot; takes priority over movement |
| O | Toggle perception overlay |
| E | Toggle all combat effects |
| P | Pause/resume |
| R | Reset to the selected seed; retains pause setting |
| Escape | Exit |

Human play sends exactly one discrete action to `ArenaEnv.step()` per training
decision interval. Simultaneous movement keys use up/down/left/right priority in
direct mode and thrust/left/right priority in rotation mode. There is no mouse aim.
Losing window focus pauses and clears held keys. An ended human episode waits for R.

## Perception and originality

O starts with `evaluation.show_observation_overlay` from `config/arena.yaml`.
World links select the nearest living enemy and spawner using observation helpers;
cyan +x follows the ship nose and green +y points clockwise on screen.

The **pilot-perception compass** plots actual normalized target offsets from
`env.observation()`, keyed by `describe()`. Its radius represents one arena diagonal,
with local +x right and +y down. Enemy dots and outlined spawner circles distinguish
target types. Missing targets show zero values without retaining old markers.

Report-ready sentence: “Our original pilot-perception compass connects world-space
target links to the agent's actual normalized ship-relative inputs, letting a human
pilot compare their decisions with the same perception under both control styles.”

## Visual isolation and verification

`arena/visuals.py` stores bounded muzzle flashes, pulsing hit rings, radial explosion
particles and damped screen shake. `arena/render.py` owns this state; the panel stays
fixed while the world shakes. Durations and limits live in `visual_feedback` YAML.
Effects use deterministic geometry and no random stream. `arena/play.py` handles
keys, pacing, and sampling after every environment step, including multiple steps
in one display frame. Simulation, rewards, observations and environment APIs are unchanged.

For a custom loop, call `renderer.observe(env)` after each step and
`renderer.advance(elapsed_seconds)` once per visual update, then `renderer.draw(env)`.
Drawing also samples state; repeated draws do not duplicate events. The initial
sample establishes a baseline, and resets or effect toggles clear old feedback.
Effects cannot reconstruct events that occurred before sampling began.

Validation: 104 focused tests and the full 621-test pytest suite passed. Checks
compare complete environment state and returned observations/rewards between visual
and nonvisual trajectories, including environment and global Python/NumPy RNG state.
Offscreen overlay and combat frames were visually inspected; CLI runs and synthetic
keyboard events cover both styles. A physical keyboard/desktop play session was
not performed.
