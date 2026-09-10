# A3-018 integrated with main

Integration base: `bf75cfc` (main fetched on 2026-09-08). The older A3-018 branch
replaced an unfinished renderer/evaluation stub; it must not be copied over the
completed A3-009 and A3-012 implementation now on main.

| Area | Integration decision |
| --- | --- |
| Renderer | Keep main's coordinate mapping, HUD, health bars, phase banners and window lifecycle. |
| Observation | Keep O/TAB and the full 20-feature panel; add a compact pilot-perception compass. |
| Policy | Keep V, action probabilities/Q-values, chosen action and critic value. |
| Human/evaluation | Reuse main's keyboard mapping, seeded episodes, trained model loading and result export. |
| Combat | Reuse A3-018's bounded renderer-owned flashes, hit rings, particles and shake. E toggles these added effects. |

No simulation, physics, reward, observation, training configuration, saved model or
result artifact is modified. Main's existing invulnerability flicker remains active
when E disables the added combat effects. The older separate `arena.play` application
is not needed; use the existing evaluation entry point for both humans and agents.

## Run

From the repository root, with requirements installed:

```sh
python -m eval.play_arena --human --style direct --seed 0
python -m eval.play_arena --human --style rotation --seed 0
python -m eval.play_arena --style direct --episodes 1 --no-save
python -m eval.play_arena --style rotation --episodes 1 --no-save
python -m eval.play_arena --style both --episodes 1 --no-window --no-save
```

WASD/arrows move in direct mode; W/up thrusts and A/D or left/right turn in rotation
mode. Space shoots and takes priority. Main's rotation input priority remains
shoot, left, right, thrust. O/TAB toggles observation plus compass, V toggles policy,
E toggles added effects, and Escape exits. Episodes restart through the existing
seeded evaluation loop. The earlier A3-018 P/R controls and `arena.render` CLI do
not apply to this integration.

## Originality and visual isolation

Report-ready sentence: “Our pilot-perception compass connects world-space target
links to the agent's actual normalized ship-relative inputs, allowing human
choices to be compared with the same perception in both control styles.”

The compass uses feature names from `describe()` and values from `env.observation()`;
local +x is right (forward) and +y is down (clockwise lateral). The circle radius
represents the arena diagonal. Enemy dots and outlined spawner markers disappear
when no living target exists. O hides the compass if it obstructs the playfield.

`arena/visuals.py` reads entity changes after steps. Repeated draws do not duplicate
events, resets clear old effects, event storage is capped, and particles use fixed
radial geometry without RNG. Main's renderer clock advances visual ages. Drawing
uses an arena-coordinate subsurface below the HUD; shake is clipped to that field.
The HUD, observation/policy panels, compass and phase banner remain stationary.
The first sample establishes a baseline; events before sampling cannot be recovered.
Durations and limits are in the renderer-only `visual_feedback` YAML block.

## Verification

65 focused tests and all 721 tests in the full pytest suite passed. Training tests
require multiprocessing sockets outside the workspace sandbox. One seeded headless
evaluation with each shipped PPO model reached phase 2 (seed 0).

Focused tests cover real combat in both styles, complete simulation and RNG
invariance, reset/toggle cleanup, compass alignment at nonzero headings, missing
and coincident targets, and stationary HUD/panels during shake. Existing renderer,
policy and human/evaluation tests remain unchanged. Actual saved PPO models were
loaded for 120 rendered steps per style and the resulting images inspected.
Physical keyboard play in a desktop window was not performed.
