# A3-018: shield pickups and elite chargers

Enable both systems with `--mechanics`. Omitting the flag retains the baseline rules and
20-feature observations used by the shipped PPO models. The two control styles retain
exactly their existing action spaces; collection happens through movement.

## Play and train

```sh
python -m eval.play_arena --human --style direct --mechanics --seed 0
python -m eval.play_arena --human --style rotation --mechanics --seed 0
python -m train.train_arena --style direct --mechanics --timesteps 2000000 --checkpoint-freq 500000 --run-name ppo_direct_mechanics
python -m train.train_arena --style rotation --mechanics --timesteps 2000000 --checkpoint-freq 500000 --run-name ppo_rotation_mechanics
python -m eval.play_arena --style both --mechanics --no-window --episodes 30 --seed 0
python -m eval.play_arena --style both --mechanics --no-window --episodes 30 --seed 0 --random
```

Models go to `models/mechanics/<algo>_<style>.zip`; evaluation exports go to
`results/arena_eval/mechanics/`. Custom model/results directories receive the same
`mechanics/` subdirectory. If no mechanics models exist, agent playback explicitly falls
back to the existing random policy. Human play needs no model. A checkpoint whose action
or observation space disagrees with the environment is rejected before inference.
Training metadata records the mechanics flag, observation dimension, and resolved rules.

## Shield pickup

Destroying a spawner drops one cyan S orb at its position. It expires after ten simulation
seconds; its countdown is visible. Touching it grants a single shield charge, shown as a
cyan ring around the ship. An already shielded player leaves additional orbs available until
expiry. Orbs survive phase transitions, but reset clears all pickups and shield state.

The next positive damage event consumes the shield, leaves health unchanged, and grants the
normal 0.6-second invulnerability window. Overlapping enemies cannot consume the shield and
immediately damage the player in the same frame. No damage penalty or damage-taken count is
applied for a blocked hit. Collecting an orb gives no separate reward.

## Elite charger

One elite appears at the start of each phase from phase 2 onward. It has six health, a larger
collision radius, and the same contact damage as a regular enemy. It uses four states:

| State | Duration | Behavior |
| --- | --- | --- |
| Pursuit | 2 seconds | Approach at 65 pixels/second |
| Wind-up | 0.8 seconds | Stop and lock direction toward the player's current position |
| Charge | Up to 0.5 seconds | Move at 420 pixels/second along that fixed direction |
| Recovery | 1 second | Stop, giving the player time to counterattack |

The wind-up line previews the charge. Color, health bar, state label, and countdown identify
the elite independently of the O and E cosmetic toggles. Reaching a wall ends a charge early
and starts recovery. Normal projectiles damage it in every state. An elite kill receives
the existing enemy reward, without a new reward coefficient.

Mechanics mode requires both all spawners and the elite to be destroyed before advancing.
Baseline mode preserves the brief's spawner-only progression. Regular enemies persist through
phase changes. Spawning reserves one enemy-cap slot for the elite.

## Agent perception and determinism

Mechanics mode appends 16 features, producing a 36-value float32 vector: shield charge;
nearest pickup's local dx/dy, existence, and remaining lifetime; elite local dx/dy, health,
existence, four one-hot attack states, normalized time remaining, and locked direction in
ship-local coordinates. Absent targets have zeroed fields. The original nearest-enemy
relative-velocity features use the larger possible closing speed in mechanics mode.

All gameplay timers use FIXED_DT and spawn placement uses the environment's seeded RNG.
Rendering only reads these states. No new action or image observation is introduced.
The 20-feature models in `models/` remain the baseline agents that rubric row I grades; the
mechanics agents below are separate models in `models/mechanics/`.

## Trained agents

One PPO agent per control style was trained on the mechanics rules with the baseline
hyperparameters from `config/arena.yaml` (seed 0, 8 envs, `[64, 64]` MLP) for 2M timesteps,
at commit `6b0bfde`. Provenance, including the resolved mechanics rules, is in
`logs/ppo_{direct,rotation}_mechanics/run.json`; each run took under three minutes
(156 s and 165 s). Snapshots every 500k steps are in `models/mechanics/checkpoints/`.

Measured deterministically on seeds 0-29 by the two `eval.play_arena ... --mechanics` commands
above (`results/arena_eval/mechanics/comparison.md` and `comparison_random.md`):

| Style | Policy | Return | Phase 1 cleared | Phase reached | Shields collected | Hits blocked | Elite kills |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | PPO | +29.59 ± 24.21 | 28/30 | 2.17 (best 3) | 0.23 | 0.23 | 0.23 |
| direct | random | −12.85 ± 1.44 | 0/30 | 1.00 | 0.00 | 0.00 | 0.00 |
| rotation | PPO | +30.72 ± 21.06 | 30/30 | 2.77 (best 4) | 1.77 | 1.67 | 0.80 |
| rotation | random | −14.34 ± 1.76 | 0/30 | 1.00 | 0.07 | 0.07 | 0.00 |

Per-episode means. What the numbers support, and what they do not:

- Both agents learned the extended rules: every trained episode but two clears phase 1, which
  under these rules also requires killing the six-health elite from phase 2 on, against 0/30
  for chance on the same arenas.
- The rotation agent uses the shield: 1.77 collected and 1.67 hits absorbed per episode, so
  almost every orb it takes is spent on a hit it would otherwise have taken. Collecting earns no
  reward, so this is instrumental behaviour learned through the damage and death penalties.
- The direct agent mostly ignores orbs (0.23). A plausible reading, not a tested one: orbs drop
  where spawners die, and the rotation agent fights at spawner range (5.37 spawners destroyed per
  episode against 2.90), while the direct agent spends its time on enemies (25.83 kills).
- Not claimed: that mechanics play is better or worse than baseline play. These agents trained
  for five times the baseline's 400k-step budget, so no like-for-like comparison exists. The
  36-feature observation is also above the 10-30 the brief's feasibility appendix suggests,
  which is guidance rather than a requirement, and one more reason the graded row I models stay
  the 20-feature ones.

## Demonstration and evidence

Watch a trained agent play the extended rules, with the policy panel on (`V`):

```sh
python -m eval.play_arena --style rotation --mechanics --seed 0 --episodes 1 --no-save
```

By hand: destroy a spawner, collect its S orb, and touch an enemy to show a blocked hit. In
phase 2, show the elite's warning line, move sideways during wind-up, then shoot during recovery.
Clear the spawners while leaving the elite alive to demonstrate the phase gate, then kill it.
Use O/Tab for observation features, E for cosmetic feedback, V for policy details, and Escape to exit.

`tests/test_arena_mechanics.py` covers real spawner/projectile combat, collection and expiry,
non-stacking and shield damage accounting, elite direction locking and wall recovery,
phase gating, reset, normalized observations, both human control modes, rendered-versus-
unrendered state/RNG invariance, PPO training/save/load, and isolated evaluation artifacts.
Trained performance is measured above; physical keyboard play is still verified by hand only.

The report section draft is in `report/creativity.md`; integrate it into the final A3-014 report.
