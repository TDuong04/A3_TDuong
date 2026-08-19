"""Arena renderer — NOT YET IMPLEMENTED.

Rubric row G (4.5 points, the largest single row in the assignment) is mostly satisfied here and in
`entities.py`, and none of it requires machine learning. Build and bank it before touching SB3.

Contract: `ArenaRenderer` owns the pygame window and draws the entity lists it is handed. It is
constructed only when `render_mode` is set, and nothing in the simulation path may call into it.

Draw clear shapes as the brief suggests: the ship as a triangle pointing along its heading, enemies
as circles, spawners as pulsing squares scaled by remaining health, bullets as short lines. Health
bars over the player and spawners. A HUD showing phase, health, score, step count and the current
action name — the HUD is what makes the video's "clear evidence the agent follows a learned policy"
legible to a marker.

The observation overlay (toggle with a key, `show_observation_overlay` in `config/arena.yaml`) draws
exactly what the agent sees: lines to the nearest enemy and nearest spawner, the ship-local heading
vector, and bars for health and cooldown. It costs an hour and pays three times — creativity marks,
the report's observation-design figure, and the most persuasive thirty seconds of the video.

Feedback effects are cheap creativity marks: muzzle flash, hit flicker, explosion particles, screen
shake on damage, a phase-transition banner. Keep them purely visual so they can never affect
simulation state, or evaluation stops matching training.
"""

from __future__ import annotations


class ArenaRenderer:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("see module docstring for the contract")
