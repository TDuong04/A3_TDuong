"""Optional arena rules. All timers advance in fixed simulation frames, without RNG."""
from dataclasses import dataclass
import math

from common.config import load_yaml
from .constants import FIXED_DT
from .entities import Enemy, Entity, TIME_EPSILON


@dataclass(frozen=True)
class MechanicsConfig:
    pickup_lifetime: float
    pickup_radius: float
    elite_first_phase: int
    elite_health: int
    elite_radius: float
    elite_speed: float
    pursuit_seconds: float
    windup_seconds: float
    charge_seconds: float
    recovery_seconds: float
    charge_speed: float

    def __post_init__(self):
        if any(value <= 0 for value in vars(self).values()):
            raise ValueError("Mechanics configuration values must be positive")

    @classmethod
    def from_yaml(cls):
        return cls(**load_yaml("arena")["mechanics"])


class ShieldPickup(Entity):
    def __init__(self, x, y, config: MechanicsConfig):
        super().__init__(x, y, config.pickup_radius)
        self.remaining = self.lifetime = config.pickup_lifetime

    def update(self):
        self.remaining = max(0.0, self.remaining - FIXED_DT)
        if self.remaining <= TIME_EPSILON:
            self.kill()


class EliteCharger(Enemy):
    """Pursue, lock a direction while winding up, charge, then recover."""
    STATES = ("pursuit", "windup", "charge", "recovery")

    def __init__(self, x, y, config: MechanicsConfig):
        super().__init__(x, y, health=config.elite_health, speed=config.elite_speed)
        self.mechanics_config = config
        self.radius = config.elite_radius
        self.state = "pursuit"
        self.remaining = config.pursuit_seconds
        self.charge_dx, self.charge_dy = 0.0, 0.0

    @property
    def state_duration(self):
        return getattr(self.mechanics_config, self.state + "_seconds")

    def _enter(self, state):
        self.state = state
        self.remaining = self.state_duration
        self.vx = self.vy = 0.0

    def update(self, target_x, target_y):
        if not self.alive:
            return
        if self.state == "pursuit":
            super().update(target_x, target_y)
        elif self.state == "charge":
            self.vx = self.charge_dx * self.mechanics_config.charge_speed
            self.vy = self.charge_dy * self.mechanics_config.charge_speed
            self.x += self.vx * FIXED_DT
            self.y += self.vy * FIXED_DT
            if any(self._clamp_to_arena()):
                self._enter("recovery")
                return
        self.remaining = max(0.0, self.remaining - FIXED_DT)
        if self.remaining > TIME_EPSILON:
            return
        if self.state == "pursuit":
            dx, dy = target_x - self.x, target_y - self.y
            distance = math.hypot(dx, dy)
            self.charge_dx, self.charge_dy = ((dx / distance, dy / distance)
                                           if distance > TIME_EPSILON else (1.0, 0.0))
            self._enter("windup")
        elif self.state == "windup":
            self._enter("charge")
        elif self.state == "charge":
            self._enter("recovery")
        else:
            self.charge_dx = self.charge_dy = 0.0
            self._enter("pursuit")
