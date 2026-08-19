"""Arena entities — NOT YET IMPLEMENTED.

Pure simulation: `Player`, `Enemy`, `Spawner`, `Bullet`. No pygame imports anywhere in this file.
Rendering reads these objects; it never drives them.

All motion integrates against `FIXED_DT` (1/60), never a wall-clock delta. Wall-clock physics makes
training and evaluation diverge silently and destroys reproducibility.

`Player` carries position, velocity, heading, health, shoot cooldown and an invulnerability timer
after being hit (without it a single enemy contact drains all health in a few frames and every
episode ends the same way). It exposes both control styles: `thrust`/`rotate` for style 1 and
`move(dx, dy)` for style 2, over identical physics so the two agents remain comparable.

`Enemy` navigates toward the player — plain seek is enough, the brief asks for navigation, not
sophistication. Carries health and contact damage.

`Spawner` is stationary with health, emits an enemy every `spawn_interval` seconds, and stops when
destroyed. Destroying every active spawner advances the phase.

`Bullet` travels along its heading and expires on impact or when it leaves the arena. Cap the live
bullet count; an uncapped list is the usual cause of the simulation slowing to a crawl deep into
training.

Collision detection is circle-vs-circle on radii. Keep it simple — the brief asks for simple
physics, and every extra millisecond in `step()` multiplies by 400,000.
"""

from __future__ import annotations


class Player:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("see module docstring for the contract")


class Enemy:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("see module docstring for the contract")


class Spawner:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("see module docstring for the contract")


class Bullet:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("see module docstring for the contract")
