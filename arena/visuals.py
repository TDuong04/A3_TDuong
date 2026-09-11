"""Renderer-owned perception graphics and bounded combat feedback.

Only reads environment state. Effects use deterministic geometry, never any RNG.
The application supplies elapsed visual time; physics remains fixed-step in ArenaEnv.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pygame

from common.config import load_yaml

from .observation import describe

if TYPE_CHECKING:
    from .env import ArenaEnv

ENEMY_LINK = (255, 139, 148)
SPAWNER_LINK = (193, 163, 255)
FORWARD = (101, 225, 241)
LATERAL = (157, 238, 155)
FLASH = (255, 244, 201)
TEXT = (222, 231, 245)
PANEL = (28, 37, 52)


@dataclass(frozen=True)
class FeedbackConfig:
    muzzle_seconds: float
    hit_seconds: float
    explosion_seconds: float
    shake_seconds: float
    shake_pixels: float
    particle_speed: float
    particle_count: int
    max_events: int

    @classmethod
    def from_yaml(cls):
        return cls(**load_yaml('arena')['visual_feedback'])


@dataclass
class Effect:
    kind: str
    position: tuple[float, float]
    heading: float
    duration: float
    age: float = 0.0


class CombatFeedback:
    def __init__(self, config: FeedbackConfig | None = None):
        self.config = config or FeedbackConfig.from_yaml()
        self.reset()

    def reset(self):
        self.events: list[Effect] = []
        self._scratch = None
        self._player = None
        self._step = -1
        self._health = {}
        self._bullets = set()
        self._cooldown = 0.0
        self.shake_remaining = 0.0

    def _emit(self, kind, position, duration, heading=0.0):
        self.events.append(Effect(kind, position, heading, duration))
        self.events = self.events[-self.config.max_events:]

    def observe(self, env: ArenaEnv):
        """Sample after a step; repeated drawing cannot duplicate an event.

        Retain previous entities for one sample to inspect deaths removed by the
        environment's collision cleanup. Disappearing living entities are not kills.

        Effects age in wall time, one `advance` per drawn frame, which assumes the caller draws
        every step. A caller that does not — the report's figure capture steps hundreds of times
        between frames — would otherwise show a collage of muzzle flashes and explosions from
        moments long past, pinned wherever the ship happened to be. Skipped simulation therefore
        discards what is on the effect layer: those marks describe a moment that has gone.
        """
        if self._player is not env.player or env.steps < self._step:
            self.reset()
            self._player = env.player
        elif env.steps == self._step:
            return
        else:
            if env.steps - self._step > 1:
                self.events.clear()
                self.shake_remaining = 0.0
            if (set(env.bullets) - self._bullets
                    or env.player.shoot_cooldown_remaining > self._cooldown + 1e-9):
                p = env.player
                muzzle = (p.x + math.cos(p.heading) * p.radius,
                          p.y + math.sin(p.heading) * p.radius)
                self._emit('muzzle', muzzle, self.config.muzzle_seconds, p.heading)
            for entity, health in self._health.items():
                if entity.health < health:
                    self._emit('hit', entity.position, self.config.hit_seconds)
                    if entity is env.player:
                        self.shake_remaining = self.config.shake_seconds
                    if entity.health == 0:
                        self._emit('explosion', entity.position, self.config.explosion_seconds)
        self._health = {e: e.health for e in [env.player, *env.enemies, *env.spawners]}
        self._bullets = set(env.bullets)
        self._cooldown = env.player.shoot_cooldown_remaining
        self._step = env.steps

    def advance(self, dt: float):
        dt = max(0.0, dt)
        for event in self.events:
            event.age += dt
        self.events = [e for e in self.events if e.age < e.duration]
        self.shake_remaining = max(0.0, self.shake_remaining - dt)

    @property
    def offset(self):
        if self.shake_remaining <= 0:
            return (0, 0)
        strength = self.config.shake_pixels * self.shake_remaining / self.config.shake_seconds
        elapsed = self.config.shake_seconds - self.shake_remaining
        return (round(strength * math.cos(elapsed * 91)),
                round(strength * math.sin(elapsed * 73)))

    def _layer(self, size):
        """A cached transparent scratch surface the size of the playfield."""
        if self._scratch is None or self._scratch.get_size() != size:
            self._scratch = pygame.Surface(size, pygame.SRCALPHA)
        return self._scratch

    def draw(self, surface):
        """Composite every live effect in one alpha-blended pass.

        Effects fade by *alpha*, not by scaling their colour toward black. Fading toward black
        only looks like a fade against the empty playfield: drawn over the ship or an enemy, the
        same pixels read as a dirty grey blob smeared across a sprite. Alpha keeps a fresh flash
        fully bright and lets an old one disappear into whatever is underneath it.
        """
        if not self.events:
            return
        layer = self._layer(surface.get_size())
        layer.fill((0, 0, 0, 0))
        for event in self.events:
            fraction = 1 - event.age / event.duration
            color = (*FLASH, round(255 * fraction))
            x, y = event.position
            if event.kind == 'muzzle':
                dx, dy = math.cos(event.heading), math.sin(event.heading)
                pygame.draw.polygon(layer, color, [
                    (x + 17 * dx, y + 17 * dy),
                    (x - 5 * dy, y + 5 * dx), (x + 5 * dy, y - 5 * dx),
                ])
            elif event.kind == 'hit':
                # Two short bright pulses read as hit flicker without changing entity colors.
                if int(event.age / event.duration * 4) % 2 == 0:
                    pygame.draw.circle(layer, color, (x, y), 19, width=3)
            else:
                distance = event.age * self.config.particle_speed
                for index in range(self.config.particle_count):
                    angle = index * math.tau / self.config.particle_count
                    position = (x + math.cos(angle) * distance, y + math.sin(angle) * distance)
                    pygame.draw.circle(layer, color, position, max(1, round(4 * fraction)))
        surface.blit(layer, (0, 0))


class PerceptionOverlay:
    """World target links plus a compass built from the actual normalized vector.

    Compass +x is right (forward), +y is down (clockwise lateral). Unlike a world
    minimap, this view turns with the ship: a target ahead always appears right.
    """
    def __init__(self):
        self._font = None

    @staticmethod
    def values(env):
        return {name: float(value) for name, value in zip(describe(env.mechanics), env.observation(), strict=True)}

    @staticmethod
    def axes(player):
        c, s = math.cos(player.heading), math.sin(player.heading)
        return ((player.x + 48 * c, player.y + 48 * s),
                (player.x - 48 * s, player.y + 48 * c))

    def _label(self, surface, text, position, color=TEXT):
        if not pygame.font.get_init():
            pygame.font.init()
        if self._font is None:
            self._font = pygame.font.Font(None, 20)
        surface.blit(self._font.render(text, True, color), position)

    def draw_compass(self, surface, env):
        """Compact ship-relative view; existing HUD and full feature panel stay visible."""
        values = self.values(env)
        x, y = 12, surface.get_height() - 152
        pygame.draw.rect(surface, PANEL, (x, y, 234, 140), border_radius=8)
        self._label(surface, 'PILOT PERCEPTION  [O]', (x + 10, y + 8), FORWARD)
        center, radius = (x + 58, y + 80), 40
        pygame.draw.circle(surface, (82, 100, 123), center, radius, width=1)
        pygame.draw.line(surface, FORWARD, (center[0] - radius, center[1]),
                         (center[0] + radius, center[1]))
        pygame.draw.line(surface, LATERAL, (center[0], center[1] - radius),
                         (center[0], center[1] + radius))
        self._label(surface, '+x forward', (x + 110, y + 38), FORWARD)
        self._label(surface, '+y lateral', (x + 110, y + 58), LATERAL)
        markers = []
        for row, (prefix, color) in enumerate((('spawner', SPAWNER_LINK), ('enemy', ENEMY_LINK))):
            exists = values['spawners_alive'] > 0 if prefix == 'spawner' else values['enemy_exists']
            self._label(surface, prefix if exists else prefix + ': none',
                        (x + 110, y + 82 + row * 20), color)
            if exists:
                point = (center[0] + radius * values[f'{prefix}_local_dx'],
                         center[1] + radius * values[f'{prefix}_local_dy'])
                pygame.draw.line(surface, color, center, point, width=2)
                markers.append((prefix, color, point))
        for prefix, color, point in markers:
            pygame.draw.circle(surface, color, point, 6 if prefix == 'spawner' else 3,
                               width=1 if prefix == 'spawner' else 0)
        self._label(surface, 'Ring = arena diagonal', (x + 10, y + 120))
