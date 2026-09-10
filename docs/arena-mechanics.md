# A3-018: shield pickups and elite chargers

Enable both systems with `--mechanics`. Omitting the flag retains the baseline rules and
20-feature observations used by the shipped PPO models. The two control styles retain
exactly their existing action spaces; collection happens through movement.

## Play and train

```sh
python -m eval.play_arena --human --style direct --mechanics --seed 0
python -m eval.play_arena --human --style rotation --mechanics --seed 0
python -m train.train_arena --style direct --mechanics
python -m train.train_arena --style rotation --mechanics
python -m eval.play_arena --style both --mechanics --no-window
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
Full mechanics training is required for useful new policies; the existing 20-feature
checkpoints remain baseline artifacts. Short smoke-training tests demonstrate integration,
not learned skill, balance, or performance improvement.

## Demonstration and evidence

Destroy a spawner, collect its S orb, and touch an enemy to show a blocked hit. In phase 2,
show the elite's warning line, move sideways during wind-up, then shoot during recovery.
Clear the spawners while leaving the elite alive to demonstrate the phase gate, then kill it.
Use O/Tab for observation features, E for cosmetic feedback, V for policy details, and Escape to exit.

`tests/test_arena_mechanics.py` covers real spawner/projectile combat, collection and expiry,
non-stacking and shield damage accounting, elite direction locking and wall recovery,
phase gating, reset, normalized observations, both human control modes, rendered-versus-
unrendered state/RNG invariance, PPO training/save/load, and isolated evaluation artifacts.
Physical keyboard play and trained mechanics performance still require a separate demonstration.

The report section draft is in `report/creativity.md`; integrate it into the final A3-014 report.
