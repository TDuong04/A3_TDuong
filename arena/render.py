"""Core Arena rendering (A3-021), independent of simulation updates.

ArenaEnv.render() lazily constructs ArenaRenderer and calls draw(env). Windowed
renderers own one Pygame display; headless renderers own only an offscreen surface.
Drawing presents a frame but never steps the world, consumes its RNG, or throttles
execution. The application owns events and pacing.

Scripted demonstration or human play (no trained model):
    python -m arena.render --style direct --seed 0
    python -m arena.render --style rotation --seed 0
    python -m arena.render --style direct --seed 0 --headless --frames 1200

Add --human for keyboard play. O toggles perception, E toggles feedback.
Effects and their clocks belong to this renderer, never the environment.
The broader HUD and phase banners remain in A3-022.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pygame

from common.config import load_yaml

from .visuals import CombatFeedback, PerceptionOverlay

from .constants import (
    ACTION_REPEAT,
    ARENA_HEIGHT,
    ARENA_WIDTH,
    FIXED_DT,
    DirectAction,
    RotationAction,
)

if TYPE_CHECKING:
    from .env import ArenaEnv

COLOR_BACKGROUND = (20, 25, 36)
COLOR_BOUNDARY = (66, 81, 102)
COLOR_PLAYER = (95, 202, 255)
COLOR_PLAYER_DEAD = (112, 119, 130)
COLOR_ENEMY = (244, 99, 112)
COLOR_SPAWNER = (185, 132, 255)
COLOR_BULLET = (255, 222, 112)


class ArenaRenderer:
    """Draw world coordinates directly in pixels, with no simulation-side state.

    Pygame has one global display. A windowed renderer refuses to replace an
    existing window. Use headless=True for additional renderers or composition.
    close() is idempotent; drawing afterwards allocates a fresh surface.
    """

    def __init__(
        self, *, headless: bool = False, caption: str = "A3 Arena",
        show_observation_overlay: bool | None = None, effects: bool = True,
    ) -> None:
        self.headless = bool(headless)
        self.caption = caption
        self.surface: pygame.Surface | None = None
        self._owns_display = False
        self.show_observation_overlay = (
            bool(load_yaml("arena")["evaluation"]["show_observation_overlay"])
            if show_observation_overlay is None else bool(show_observation_overlay)
        )
        self.effects_enabled = bool(effects)
        self.feedback = CombatFeedback()
        self.overlay = PerceptionOverlay()
        self._world = pygame.Surface((ARENA_WIDTH, ARENA_HEIGHT))

    def _ensure_surface(self) -> pygame.Surface:
        if self.surface is None:
            size = (ARENA_WIDTH, ARENA_HEIGHT)
            if self.headless:
                self.surface = pygame.Surface(size)
            else:
                if pygame.display.get_surface() is not None:
                    raise RuntimeError("A Pygame window is already open; use headless=True")
                initialized_here = not pygame.display.get_init()
                try:
                    pygame.display.init()
                    self.surface = pygame.display.set_mode(size)
                    pygame.display.set_caption(self.caption)
                except pygame.error:
                    if initialized_here:
                        pygame.display.quit()
                    raise
                self._owns_display = True
        return self.surface

    def toggle_observation_overlay(self) -> bool:
        self.show_observation_overlay = not self.show_observation_overlay
        return self.show_observation_overlay

    def toggle_effects(self) -> bool:
        self.effects_enabled = not self.effects_enabled
        self.feedback.reset()
        return self.effects_enabled

    def advance(self, dt: float) -> None:
        """Advance visual time only. Call from the application, never from step()."""
        self.feedback.advance(dt)

    def observe(self, env: ArenaEnv) -> None:
        """Optional per-step sampling so multiple steps per frame don't hide events."""
        if self.effects_enabled:
            self.feedback.observe(env)

    def reset_visuals(self) -> None:
        self.feedback.reset()

    def draw(self, env: ArenaEnv) -> pygame.Surface:
        """Read the current entities, draw them, and present our window if any."""
        target_surface = self._ensure_surface()
        self.observe(env)
        surface = self._world
        surface.fill(COLOR_BACKGROUND)
        pygame.draw.rect(surface, COLOR_BOUNDARY, surface.get_rect(), width=2)

        if self.show_observation_overlay:
            self.overlay.draw_world(surface, env)

        for spawner in env.spawners:
            if spawner.alive:
                radius = spawner.radius
                box = pygame.Rect(0, 0, round(2 * radius), round(2 * radius))
                box.center = (round(spawner.x), round(spawner.y))
                pygame.draw.rect(surface, COLOR_SPAWNER, box)

        for enemy in env.enemies:
            if enemy.alive:
                pygame.draw.circle(surface, COLOR_ENEMY, enemy.position, enemy.radius)

        for bullet in env.bullets:
            if bullet.alive:
                dx, dy = math.cos(bullet.heading), math.sin(bullet.heading)
                tail = (bullet.x - 2 * bullet.radius * dx, bullet.y - 2 * bullet.radius * dy)
                tip = (bullet.x + bullet.radius * dx, bullet.y + bullet.radius * dy)
                pygame.draw.line(
                    surface, COLOR_BULLET, tail, tip, width=max(2, round(bullet.radius))
                )

        player = env.player
        cos_h, sin_h = math.cos(player.heading), math.sin(player.heading)
        # Local +x is the nose; +y is clockwise on screen, matching observation.py.
        corners = ((1.0, 0.0), (-0.7, 0.75), (-0.7, -0.75))
        vertices = [
            (
                player.x + player.radius * (x * cos_h - y * sin_h),
                player.y + player.radius * (x * sin_h + y * cos_h),
            )
            for x, y in corners
        ]
        color = COLOR_PLAYER if player.alive else COLOR_PLAYER_DEAD
        pygame.draw.polygon(surface, color, vertices)

        if self.effects_enabled:
            self.feedback.draw(surface)
        target_surface.fill(COLOR_BACKGROUND)
        target_surface.blit(surface, self.feedback.offset if self.effects_enabled else (0, 0))
        if self.show_observation_overlay:
            self.overlay.draw_panel(target_surface, env)
        if self._owns_display:
            pygame.display.flip()
        return target_surface

    def close(self) -> None:
        """Release our display only; do not shut down unrelated Pygame subsystems."""
        if self._owns_display and pygame.display.get_surface() is self.surface:
            pygame.display.quit()
        self.surface = None
        self._owns_display = False
        self.reset_visuals()


def scripted_action(env: ArenaEnv) -> int:
    """A deterministic demonstration controller, explicitly not a learned policy.

    Move briefly, then wait for spawns and aim at approaching enemies. All combat
    and movement occur through step(); the demo never injects entities or damage.
    """
    from .observation import nearest_alive

    rotation = env.control_style == "rotation"
    if env.steps < 10:
        return int(RotationAction.THRUST if rotation else DirectAction.RIGHT)
    target = nearest_alive(env.player, env.enemies)
    if target is None:
        return 0
    dx, dy = target.x - env.player.x, target.y - env.player.y
    if rotation:
        desired = math.atan2(dy, dx)
        error = (desired - env.player.heading + math.pi) % (2 * math.pi) - math.pi
        turn = env.player.config.rotation_speed_radians * FIXED_DT * ACTION_REPEAT
        if abs(error) > turn / 2:
            return int(RotationAction.ROTATE_RIGHT if error > 0 else RotationAction.ROTATE_LEFT)
        return int(RotationAction.SHOOT)
    # Align vertically, then face and fire horizontally. The enemy keeps seeking us.
    if abs(dy) > target.radius:
        return int(DirectAction.DOWN if dy > 0 else DirectAction.UP)
    desired = 0.0 if dx >= 0 else -math.pi
    error = (desired - env.player.heading + math.pi) % (2 * math.pi) - math.pi
    if abs(error) > 0.01:
        return int(DirectAction.RIGHT if dx >= 0 else DirectAction.LEFT)
    return int(DirectAction.SHOOT)


def main(argv: Sequence[str] | None = None) -> int:
    # Kept as the established demo entry point; events/pacing live in the app.
    from .play import main as play_main

    return play_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
