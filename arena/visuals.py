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

from .observation import describe, nearest_alive

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
        """
        if self._player is not env.player or env.steps < self._step:
            self.reset()
            self._player = env.player
        elif env.steps == self._step:
            return
        else:
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

    def draw(self, surface):
        for event in self.events:
            fraction = 1 - event.age / event.duration
            color = tuple(round(channel * fraction) for channel in FLASH)
            x, y = event.position
            if event.kind == 'muzzle':
                dx, dy = math.cos(event.heading), math.sin(event.heading)
                pygame.draw.polygon(surface, color, [
                    (x + 17 * dx, y + 17 * dy),
                    (x - 5 * dy, y + 5 * dx), (x + 5 * dy, y - 5 * dx),
                ])
            elif event.kind == 'hit':
                # Two short bright pulses read as hit flicker without changing entity colors.
                if int(event.age / event.duration * 4) % 2 == 0:
                    pygame.draw.circle(surface, color, (x, y), 19, width=3)
            else:
                distance = event.age * self.config.particle_speed
                for index in range(self.config.particle_count):
                    angle = index * math.tau / self.config.particle_count
                    position = (x + math.cos(angle) * distance, y + math.sin(angle) * distance)
                    pygame.draw.circle(surface, color, position, max(1, round(4 * fraction)))


class PerceptionOverlay:
    """World target links plus a compass built from the actual normalized vector.

    Compass +x is right (forward), +y is down (clockwise lateral). Unlike a world
    minimap, this view turns with the ship: a target ahead always appears right.
    """
    def __init__(self):
        self._font = None

    @staticmethod
    def values(env):
        return {name: float(value) for name, value in zip(describe(), env.observation(), strict=True)}

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

    def draw_world(self, surface, env):
        p = env.player
        for candidates, color in ((env.enemies, ENEMY_LINK), (env.spawners, SPAWNER_LINK)):
            target = nearest_alive(p, candidates)
            if target is not None:
                pygame.draw.line(surface, color, p.position, target.position, width=1)
                pygame.draw.circle(surface, color, target.position, target.radius + 5, width=1)
        for tip, color, label in zip(self.axes(p), (FORWARD, LATERAL), ('+x', '+y'), strict=True):
            pygame.draw.line(surface, color, p.position, tip, width=2)
            self._label(surface, label, tip, color)

    def draw_panel(self, surface, env):
        values = self.values(env)
        pygame.draw.rect(surface, PANEL, (12, 12, 324, 300), border_radius=8)
        self._label(surface, 'PILOT PERCEPTION   [O] hide', (24, 23), FORWARD)
        self._label(surface, 'Actual agent input / normalized [-1, 1]', (24, 45))
        center, radius = (90, 136), 56
        pygame.draw.circle(surface, (82, 100, 123), center, radius, width=1)
        pygame.draw.line(surface, FORWARD, (34, 136), (146, 136))
        pygame.draw.line(surface, LATERAL, (90, 80), (90, 192))
        self._label(surface, '+x forward', (155, 116), FORWARD)
        self._label(surface, '+y lateral', (155, 139), LATERAL)
        self._label(surface, 'Ring = arena diagonal', (155, 164))
        markers = []
        for prefix, color in (('spawner', SPAWNER_LINK), ('enemy', ENEMY_LINK)):
            if values[f'{prefix}_exists']:
                point = (center[0] + radius * values[f'{prefix}_local_dx'],
                         center[1] + radius * values[f'{prefix}_local_dy'])
                pygame.draw.line(surface, color, center, point, width=2)
                markers.append((prefix, color, point))
        # Draw markers after all rays: a farther target's ray must not hide a nearer one.
        for prefix, color, point in markers:
            pygame.draw.circle(surface, color, point, 6 if prefix == 'spawner' else 3,
                               width=1 if prefix == 'spawner' else 0)
        for row, (prefix, color) in enumerate((('enemy', ENEMY_LINK), ('spawner', SPAWNER_LINK))):
            if values[f'{prefix}_exists']:
                label = (f"{prefix}: x {values[f'{prefix}_local_dx']:+.2f}  "
                         f"y {values[f'{prefix}_local_dy']:+.2f}  "
                         f"d {values[f'{prefix}_distance']:.2f}")
            else:
                label = f'{prefix}: none  /  x 0.00  y 0.00  d 0.00'
            self._label(surface, label, (24, 205 + row * 23), color)
        self._label(surface, f"health {values['health']:.2f}   cooldown {values['shoot_cooldown']:.2f}",
                    (24, 255))
        self._label(surface, 'Same perception for both control styles', (24, 281))
