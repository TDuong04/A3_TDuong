"""Arena entities — the pure simulation behind Part II.

`Player`, `Enemy`, `Spawner`, `Bullet`. No pygame imports anywhere in this file. Rendering reads
these objects; it never drives them.

All motion integrates against `FIXED_DT` (1/60), never a wall-clock delta — which is why no
`update()` here accepts a `dt` argument at all. Wall-clock physics makes training and evaluation
diverge silently and destroys reproducibility.

`Player` carries position, velocity, heading, health, shoot cooldown and an invulnerability timer
after being hit (without it a single enemy contact drains all health in a few frames and every
episode ends the same way). It exposes both control styles: `thrust`/`rotate` for style 1 and
`move(dx, dy)` for style 2, over identical physics so the two agents remain comparable.

"Identical physics" is meant literally. Both styles only ever *choose a direction*; they then
accelerate along it at the same `thrust`, decay at the same `drag`, saturate at the same `speed`
cap and are integrated by the same `_integrate` call. Style 1 turns the heading gradually at
`rotation_speed`; style 2 snaps the heading to the commanded direction, because a directional
agent that could not aim would have no way to shoot anything. Nothing else differs, so report row
R6 compares two policies over one dynamical system rather than two games.

Drag is the fraction of velocity retained per *second* of coasting and is applied as `drag ** dt`.
Applying `drag` once per frame instead would give a terminal speed of `thrust * dt / (1 - drag)` =
67 px/s, slower than every enemy in `config/arena.yaml`, and the arena would be unplayable.

`Enemy` navigates toward the player — plain seek is enough, the brief asks for navigation, not
sophistication. Carries health and contact damage.

`Spawner` is stationary with health, emits an enemy every `spawn_interval` seconds of simulated
time, and stops when destroyed. Destroying every active spawner advances the phase.

`Bullet` travels along its heading and expires on impact or when it leaves the arena. The live
bullet count is capped by `entities.max_bullets`: an uncapped list is the usual cause of the
simulation slowing to a crawl deep into training, so `Player.shoot` is told how many bullets are
already alive and refuses to add another past the cap.

Collision detection is circle-vs-circle on radii (`Entity.collides_with`, touching counts as a
hit). Keep it simple — the brief asks for simple physics, and every extra millisecond in `step()`
multiplies by 400,000.

Every speed, cooldown, health, radius and interval comes from `config/arena.yaml`; there are no
tuned literals in this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from common.config import load_yaml

from .constants import ARENA_HEIGHT, ARENA_WIDTH, FIXED_DT

# Timers count down in FIXED_DT increments, so a deadline that lands exactly on a frame boundary
# (0.6 s = 36 frames) can miss it by ~1e-15 of accumulated float error and fire one frame late.
# Comparing against this tolerance keeps every interval on the frame it was meant to land on.
TIME_EPSILON = 1e-9

# Successive enemies from one spawner are placed a golden angle apart around it. Deterministic
# (no RNG to seed, so a replay matches exactly) and it never stacks two enemies on one point.
_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


# --- configuration ----------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerConfig:
    """The `player` block of `config/arena.yaml`. Shared by both control styles."""

    max_health: int
    speed: float
    thrust: float
    rotation_speed: float  # degrees/second, converted on use
    drag: float
    shoot_cooldown: float
    bullet_speed: float
    invulnerability_time: float
    radius: float

    @property
    def rotation_speed_radians(self) -> float:
        """Headings are stored in radians so `math.cos`/`sin` need no conversion per frame."""
        return math.radians(self.rotation_speed)

    @classmethod
    def from_yaml(cls, name: str = "arena", section: str = "player") -> PlayerConfig:
        return cls(**load_yaml(name)[section])


@dataclass(frozen=True)
class EntityConfig:
    """The `entities` block: geometry, damage and the live bullet cap."""

    enemy_radius: float
    spawner_radius: float
    bullet_radius: float
    bullet_damage: int
    bullet_lifetime: float
    max_bullets: int
    enemy_contact_damage: int

    @classmethod
    def from_yaml(cls, name: str = "arena", section: str = "entities") -> EntityConfig:
        return cls(**load_yaml(name)[section])


@dataclass(frozen=True)
class PhaseConfig:
    """One entry of the `phases` list: how hard the arena is once that phase begins."""

    spawners: int
    spawner_health: int
    enemy_health: int
    enemy_speed: float
    spawn_interval: float

    @classmethod
    def from_yaml(cls, index: int, name: str = "arena", section: str = "phases") -> PhaseConfig:
        phases: list[dict[str, Any]] = load_yaml(name)[section]
        if index < 0:
            raise IndexError(f"phase index must be >= 0, got {index}")
        # An agent that clears the last configured phase keeps playing the hardest one rather than
        # crashing the episode; the difficulty ladder simply stops climbing.
        return cls(**phases[min(index, len(phases) - 1)])


@lru_cache(maxsize=None)
def player_config() -> PlayerConfig:
    """The shipped player tunables. Cached: the YAML is read once per process, not per entity."""
    return PlayerConfig.from_yaml()


@lru_cache(maxsize=None)
def entity_config() -> EntityConfig:
    return EntityConfig.from_yaml()


@lru_cache(maxsize=None)
def phase_config(index: int) -> PhaseConfig:
    return PhaseConfig.from_yaml(index)


@lru_cache(maxsize=None)
def n_phases() -> int:
    """How many phases are configured, before the difficulty ladder stops climbing."""
    return len(load_yaml("arena")["phases"])


# --- base entities ----------------------------------------------------------------------------


class Entity:
    """A circle in the arena: position, collision radius and an alive flag."""

    def __init__(self, x: float, y: float, radius: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)
        self.alive = True

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)

    def distance_to(self, other: Entity) -> float:
        return math.hypot(other.x - self.x, other.y - self.y)

    def collides_with(self, other: Entity) -> bool:
        """Circle-vs-circle on radii. Exactly touching counts as a hit, so a shot that grazes the
        rim is never silently dropped by a floating-point comparison."""
        if not (self.alive and other.alive):
            return False
        return self.distance_to(other) <= self.radius + other.radius

    def kill(self) -> None:
        self.alive = False

    def _clamp_to_arena(self) -> tuple[bool, bool]:
        """Keep the circle fully inside the arena. Returns which axes were clamped."""
        hit_x = hit_y = False
        low_x, high_x = self.radius, ARENA_WIDTH - self.radius
        low_y, high_y = self.radius, ARENA_HEIGHT - self.radius
        if self.x < low_x:
            self.x, hit_x = low_x, True
        elif self.x > high_x:
            self.x, hit_x = high_x, True
        if self.y < low_y:
            self.y, hit_y = low_y, True
        elif self.y > high_y:
            self.y, hit_y = high_y, True
        return hit_x, hit_y


class DestructibleEntity(Entity):
    """An `Entity` with health. `take_damage` reports whether the damage actually landed."""

    def __init__(self, x: float, y: float, radius: float, health: int) -> None:
        super().__init__(x, y, radius)
        self.max_health = int(health)
        self.health = int(health)

    def take_damage(self, amount: int = 1) -> bool:
        """Apply `amount` damage. Returns True if it landed; check `.alive` for destruction.

        A corpse absorbs nothing, so an enemy already killed this frame cannot be billed twice by
        two bullets arriving in the same collision pass.
        """
        if not self.alive:
            return False
        self.health -= int(amount)
        if self.health <= 0:
            self.health = 0
            self.kill()
        return True

    @property
    def health_fraction(self) -> float:
        """For the renderer's health bars and for observation normalization."""
        return self.health / self.max_health if self.max_health > 0 else 0.0


# --- player -----------------------------------------------------------------------------------


class Player(DestructibleEntity):
    """The controllable ship. One physics model, two ways of steering it.

    Style 1 (rotation + thrust): `rotate(-1)` / `rotate(+1)` then `thrust()`.
    Style 2 (direct directional): `move(dx, dy)`.

    Both only choose a direction; `update()` does the integration for both.
    """

    def __init__(
        self,
        x: float,
        y: float,
        heading: float = 0.0,
        config: PlayerConfig | None = None,
        entities: EntityConfig | None = None,
    ) -> None:
        self.config = config if config is not None else player_config()
        self.entities = entities if entities is not None else entity_config()
        super().__init__(x, y, self.config.radius, self.config.max_health)

        self.vx = 0.0
        self.vy = 0.0
        self.heading = _wrap_angle(heading)  # radians, 0 points along +x
        self.shoot_cooldown_remaining = 0.0
        self.invulnerable_remaining = 0.0
        self.shield_charge = 0
        self.shield_blocks = 0

        # `drag` is per second of coasting; converting once here keeps the per-frame work to one
        # multiply and keeps the constant honest if FIXED_DT ever changes.
        self._drag_per_frame = self.config.drag**FIXED_DT

        self._rotation_input = 0.0
        self._accel_x = 0.0
        self._accel_y = 0.0
        self._pending_thrust = 0.0

    # --- control style 1: rotation and thrust ---------------------------------------------

    def rotate(self, direction: float) -> None:
        """Turn at `rotation_speed`. `direction` is -1 for left (counter-clockwise on screen),
        +1 for right. Applied by `update()`, so the turn is integrated over FIXED_DT like
        everything else."""
        self._rotation_input += _clamp(float(direction), -1.0, 1.0)

    def thrust(self, amount: float = 1.0) -> None:
        """Accelerate along the current heading. Resolved in `update()`, after the frame's
        rotation, so a turn-and-burn frame pushes in the direction the ship ends up facing."""
        self._pending_thrust += _clamp(float(amount), 0.0, 1.0)

    # --- control style 2: direct directional ----------------------------------------------

    def move(self, dx: float, dy: float) -> None:
        """Accelerate along `(dx, dy)` at the same `thrust` style 1 uses, and face that way.

        The heading snaps because style 2 has no rotate action; without this the directional agent
        would be locked to its spawn heading and could never aim a shot.
        """
        norm = math.hypot(dx, dy)
        if norm == 0.0:
            return  # a null direction is a no-op, not a divide by zero
        ux, uy = dx / norm, dy / norm
        self.heading = math.atan2(uy, ux)
        self._accel_x += self.config.thrust * ux
        self._accel_y += self.config.thrust * uy

    # --- shared physics --------------------------------------------------------------------

    def update(self) -> None:
        """Advance exactly one physics frame of FIXED_DT. Takes no `dt` by design."""
        if self._rotation_input:
            rate = self.config.rotation_speed_radians
            spin = _clamp(self._rotation_input, -1.0, 1.0)
            self.heading = _wrap_angle(self.heading + spin * rate * FIXED_DT)

        if self._pending_thrust:
            push = _clamp(self._pending_thrust, 0.0, 1.0) * self.config.thrust
            self._accel_x += push * math.cos(self.heading)
            self._accel_y += push * math.sin(self.heading)

        self._integrate(self._accel_x, self._accel_y)
        self._tick_timers()
        self._clear_inputs()

    def _integrate(self, ax: float, ay: float) -> None:
        """The one integrator both control styles share: accelerate, drag, cap, move, clamp."""
        self.vx = (self.vx + ax * FIXED_DT) * self._drag_per_frame
        self.vy = (self.vy + ay * FIXED_DT) * self._drag_per_frame

        speed = math.hypot(self.vx, self.vy)
        if speed > self.config.speed:
            scale = self.config.speed / speed
            self.vx *= scale
            self.vy *= scale

        self.x += self.vx * FIXED_DT
        self.y += self.vy * FIXED_DT

        # Walls stop the ship dead rather than letting it grind along at full speed off-screen.
        hit_x, hit_y = self._clamp_to_arena()
        if hit_x:
            self.vx = 0.0
        if hit_y:
            self.vy = 0.0

    def _tick_timers(self) -> None:
        self.shoot_cooldown_remaining = max(0.0, self.shoot_cooldown_remaining - FIXED_DT)
        self.invulnerable_remaining = max(0.0, self.invulnerable_remaining - FIXED_DT)

    def _clear_inputs(self) -> None:
        self._rotation_input = 0.0
        self._accel_x = 0.0
        self._accel_y = 0.0
        self._pending_thrust = 0.0

    # --- combat -----------------------------------------------------------------------------

    def shoot(self, live_bullets: int = 0) -> Bullet | None:
        """Fire along the heading, or return None if blocked.

        Blocked by the cooldown or by the live bullet cap. A shot refused by the cap does not
        consume the cooldown — the cap throttles the arena, it must not silently punish the agent
        for pressing fire.
        """
        if not self.alive:
            return None
        if self.shoot_cooldown_remaining > TIME_EPSILON:
            return None
        if live_bullets >= self.entities.max_bullets:
            return None

        self.shoot_cooldown_remaining = self.config.shoot_cooldown
        muzzle = self.radius + self.entities.bullet_radius
        return Bullet(
            self.x + math.cos(self.heading) * muzzle,
            self.y + math.sin(self.heading) * muzzle,
            heading=self.heading,
            speed=self.config.bullet_speed,
            config=self.entities,
        )

    @property
    def can_shoot(self) -> bool:
        return self.alive and self.shoot_cooldown_remaining <= TIME_EPSILON

    @property
    def is_invulnerable(self) -> bool:
        return self.invulnerable_remaining > TIME_EPSILON

    def take_damage(self, amount: int = 1) -> bool:
        """Apply damage unless the invulnerability window is still open.

        Returns True only when the hit landed, so the env pays the damage penalty once per real
        hit instead of once per frame of contact.
        """
        if self.is_invulnerable:
            return False
        if self.alive and amount > 0 and self.shield_charge:
            self.shield_charge = 0
            self.shield_blocks += 1
            self.invulnerable_remaining = self.config.invulnerability_time
            return False
        if not super().take_damage(amount):
            return False
        self.invulnerable_remaining = self.config.invulnerability_time
        return True

    # --- views for the renderer and the observation ----------------------------------------

    @property
    def velocity(self) -> tuple[float, float]:
        return (self.vx, self.vy)

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def heading_degrees(self) -> float:
        return math.degrees(self.heading)

    def __repr__(self) -> str:
        return (
            f"Player(pos=({self.x:.1f}, {self.y:.1f}), heading={self.heading_degrees:.0f}deg, "
            f"health={self.health}/{self.max_health})"
        )


# --- enemy ------------------------------------------------------------------------------------


class Enemy(DestructibleEntity):
    """Seeks the player at a constant speed and damages it on contact.

    Plain seek, no steering behaviours: the brief asks for navigation toward the player, and a
    cheap update matters when it runs 400,000 times per training run.
    """

    def __init__(
        self,
        x: float,
        y: float,
        health: int,
        speed: float,
        config: EntityConfig | None = None,
    ) -> None:
        cfg = config if config is not None else entity_config()
        super().__init__(x, y, cfg.enemy_radius, health)
        self.config = cfg
        self.speed = float(speed)
        self.contact_damage = int(cfg.enemy_contact_damage)
        self.vx = 0.0
        self.vy = 0.0

    @classmethod
    def from_phase(
        cls, x: float, y: float, phase: PhaseConfig, config: EntityConfig | None = None
    ) -> Enemy:
        """Health and speed both come from the phase, so difficulty stays in one place."""
        return cls(x, y, health=phase.enemy_health, speed=phase.enemy_speed, config=config)

    def update(self, target_x: float, target_y: float) -> None:
        """Advance one FIXED_DT frame straight toward the target."""
        if not self.alive:
            return
        dx, dy = target_x - self.x, target_y - self.y
        distance = math.hypot(dx, dy)
        if distance <= TIME_EPSILON:
            # Sitting exactly on the player: hold still rather than divide by zero and produce the
            # NaN that would propagate into the observation and kill the run.
            self.vx = self.vy = 0.0
            return
        self.vx = self.speed * dx / distance
        self.vy = self.speed * dy / distance
        self.x += self.vx * FIXED_DT
        self.y += self.vy * FIXED_DT
        self._clamp_to_arena()

    @property
    def velocity(self) -> tuple[float, float]:
        return (self.vx, self.vy)

    def __repr__(self) -> str:
        return (
            f"Enemy(pos=({self.x:.1f}, {self.y:.1f}), health={self.health}/{self.max_health}, "
            f"speed={self.speed:.0f})"
        )


# --- spawner ----------------------------------------------------------------------------------


class Spawner(DestructibleEntity):
    """Stationary, destructible, and emits an enemy every `spawn_interval` simulated seconds.

    Destroying every active spawner is what advances the phase, so a destroyed spawner must go
    completely silent: `update()` returns None forever afterwards, whatever its timer said.
    """

    def __init__(
        self,
        x: float,
        y: float,
        phase: PhaseConfig,
        config: EntityConfig | None = None,
    ) -> None:
        cfg = config if config is not None else entity_config()
        super().__init__(x, y, cfg.spawner_radius, phase.spawner_health)
        self.config = cfg
        self.phase = phase
        self.spawn_interval = float(phase.spawn_interval)
        # Starts full, so the first enemy arrives one whole interval after the phase begins and
        # the player is never spawned on top of.
        self.spawn_timer = self.spawn_interval
        self.enemies_spawned = 0

    def update(self) -> Enemy | None:
        """Advance one FIXED_DT frame; returns the enemy emitted this frame, or None."""
        if not self.alive:
            return None
        self.spawn_timer -= FIXED_DT
        if self.spawn_timer > TIME_EPSILON:
            return None
        # Add rather than reset: the cadence stays exact instead of drifting by the fraction of a
        # frame the timer overshot by.
        self.spawn_timer += self.spawn_interval
        return self._emit()

    def _emit(self) -> Enemy:
        """Place the new enemy just clear of the spawner rim, a golden angle on from the last."""
        angle = _GOLDEN_ANGLE * self.enemies_spawned
        offset = self.radius + self.config.enemy_radius
        enemy = Enemy.from_phase(
            self.x + math.cos(angle) * offset,
            self.y + math.sin(angle) * offset,
            self.phase,
            self.config,
        )
        enemy._clamp_to_arena()
        self.enemies_spawned += 1
        return enemy

    @property
    def time_to_next_spawn(self) -> float:
        return max(0.0, self.spawn_timer)

    def __repr__(self) -> str:
        return (
            f"Spawner(pos=({self.x:.1f}, {self.y:.1f}), health={self.health}/{self.max_health}, "
            f"alive={self.alive}, spawned={self.enemies_spawned})"
        )


# --- bullet -----------------------------------------------------------------------------------


class Bullet(Entity):
    """Travels along a fixed heading until it hits something, expires, or leaves the arena."""

    def __init__(
        self,
        x: float,
        y: float,
        heading: float,
        speed: float,
        config: EntityConfig | None = None,
    ) -> None:
        cfg = config if config is not None else entity_config()
        super().__init__(x, y, cfg.bullet_radius)
        self.config = cfg
        self.heading = _wrap_angle(heading)
        self.speed = float(speed)
        self.vx = math.cos(self.heading) * self.speed
        self.vy = math.sin(self.heading) * self.speed
        self.damage = int(cfg.bullet_damage)
        self.lifetime_remaining = float(cfg.bullet_lifetime)

    def update(self) -> None:
        """Advance one FIXED_DT frame and expire on age or on leaving the arena."""
        if not self.alive:
            return
        self.x += self.vx * FIXED_DT
        self.y += self.vy * FIXED_DT
        self.lifetime_remaining -= FIXED_DT
        if self.lifetime_remaining <= TIME_EPSILON or self.is_outside_arena:
            self.kill()

    @property
    def is_outside_arena(self) -> bool:
        return not (0.0 <= self.x <= ARENA_WIDTH and 0.0 <= self.y <= ARENA_HEIGHT)

    @property
    def velocity(self) -> tuple[float, float]:
        return (self.vx, self.vy)

    def __repr__(self) -> str:
        return f"Bullet(pos=({self.x:.1f}, {self.y:.1f}), alive={self.alive})"


# --- small helpers ----------------------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _wrap_angle(angle: float) -> float:
    """Fold an angle into [-pi, pi) so a long episode of turning never drifts to huge radians."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
