# Originality and creativity

The brief asks for an arena with spawners, homing enemies and phases, agents trained on it, and
evidence that they learned (Part II, pp. 6–8; rubric G p. 16, creativity p. 18). We went beyond
that in three directions: new gameplay that the agents had to learn, making learning itself
visible, and putting the algorithm on screen rather than in a log.

| Beyond the brief | Why it goes further | Evidence |
| --- | --- | --- |
| **Shield pickups and elite chargers**, trained | Two new systems in the same env and action spaces. Destroyed spawners drop a 10 s shield that absorbs one hit. From phase 2 an elite runs a telegraphed pursuit → wind-up → charge → recovery cycle and must die before the phase ends. A PPO agent per style was trained on them. | Rotation agent: 30/30 phase-1 clears, 1.77 shields taken and 1.67 hits absorbed per episode, 0.80 elite kills; random: 0/30 (`results/arena_eval/mechanics/`) |
| **Learning time-lapse** | Every 50k-step snapshot of the shipped runs replays on screen, the HUD showing the step count read from the network itself. | Direct clears phase 1 in 0/30 arenas at 100k, 27/30 at 200k, 30/30 at 300k (`results/arena_eval/timelapse_direct_seed0.md`) |
| **Live learning inspector** (Part I) | Each real Q-learning or SARSA update shown as it happens: the ε roll, the TD target, `max_a' Q(s',a')` against SARSA's chosen `Q(s',a')`, and the intrinsic bonus. | `eval.play_gridworld --learn`, pinned by a test to the trainer's exact transitions |
| **Observation and policy overlays** | The 20 features the network sees, lines to its targets, its action probabilities and critic `V(s)`, live. | `O` and `V` in `eval.play_arena` |
| **Character art and combat feedback** | State-driven sprites, thrust flames, muzzle flash, explosions and screen shake, all in the renderer and never in the simulation. | Seeded rendered-versus-headless invariance tests |

The shield result was not designed in. Collecting an orb earns nothing, so the rotation agent
learned to take shields purely because they spare it the damage and death penalties. The direct
agent mostly ignores them (0.23 per episode). We read that as a strategy difference, since the
rotation agent destroys more spawners (5.37 against 2.90 per episode) and orbs drop where
spawners die, but we have not tested it. The time-lapse also showed that the direct agent learned
to survive before it learned to fight. At 100k steps it clears nothing yet outlives 27% of
episodes, and by 200k it is clearing phase 1 in 27 of 30.

*Claim limits: the mechanics agents trained for 2M steps against the baseline's 400k, so we do not
compare the two. Their 36-feature observation is above the appendix's suggested 10–30, which is
why the models graded for the two control schemes remain the 20-feature ones.*
