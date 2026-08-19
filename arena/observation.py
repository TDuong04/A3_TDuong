"""Observation vector — NOT YET IMPLEMENTED.

Fixed-size numeric vector of `OBS_DIM` (21) float32 values. No pixels, no screenshots — the brief
forbids them and the rubric checks for a numeric feature vector.

Layout:

    idx  feature                                          normalization
    0-1  player x, y                                      / arena size, centred to [-1, 1]
    2-3  player vx, vy                                    / max speed
    4-5  heading sin, cos                                 already [-1, 1]
    6    health                                           / max health
    7    phase                                            / max phase
    8-9  nearest enemy dx, dy IN SHIP-LOCAL FRAME         / arena diagonal
    10   nearest enemy distance                           / arena diagonal
    11-12 nearest enemy relative vx, vy                   / max enemy speed
    13   nearest enemy exists flag                        0.0 or 1.0
    14-15 nearest spawner dx, dy IN SHIP-LOCAL FRAME      / arena diagonal
    16   nearest spawner distance                         / arena diagonal
    17   nearest spawner exists flag                      0.0 or 1.0
    18   shoot cooldown remaining                         / cooldown duration
    19   enemies alive                                    / a fixed cap
    20   spawners alive                                   / a fixed cap

Three rules that decide whether this trains at all:

**Ship-local frame.** Rotate relative positions into the ship's frame with the standard 2D rotation
by `-heading`. The rotation-control agent needs "target is 30 degrees to my left", not a world
coordinate it would have to combine with its own heading internally. Both control styles use the
identical vector anyway, which is what makes the two agents comparable in the report.

**Angles as sin/cos, never radians.** A raw angle wraps discontinuously at 2*pi and the network
cannot fit across the seam.

**Zero the slots and clear the flag when no entity exists.** Never leave stale values from the
previous frame, and guard the distance division when an entity sits exactly on the player —
a single NaN propagates through the network and kills the run silently.

Everything lands in roughly [-1, 1] here, inside the env. That is deliberate: hand-normalizing
beats `VecNormalize`, which interacts badly with DQN's replay buffer and adds a statistics file
that must then be shipped and loaded alongside every model.

`describe()` should return the human-readable feature names in index order. The eval overlay and the
report's observation-design figure both read from it, so the labels never drift from the layout.
"""

from __future__ import annotations


def build_observation(*args, **kwargs):
    raise NotImplementedError("see module docstring for the contract")
