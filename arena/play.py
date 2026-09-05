"""Keyboard and scripted playback of the unmodified training environment.

Exactly one discrete action per decision, as during training: shooting takes
priority over movement; movement priority is up, down, left, right. Rotation
priority is thrust, left, right. No combined actions or mouse aiming are added.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence

import pygame

from .constants import ACTION_REPEAT, CONTROL_STYLES, FIXED_DT, FPS, DirectAction, RotationAction
from .env import ArenaEnv
from .render import ArenaRenderer, scripted_action

CONTROLS = (
    'WASD/arrows: move (rotation: W/up thrust, A/left and D/right turn). '
    'Space: shoot (takes priority). O: perception. E: effects. P: pause. R: reset. ESC: exit.'
)


def human_action(style: str, held: set[int]) -> int:
    if style not in CONTROL_STYLES:
        raise ValueError(f'Unknown control style: {style}')
    if pygame.K_SPACE in held:
        return int(RotationAction.SHOOT if style == 'rotation' else DirectAction.SHOOT)
    bindings = (
        [((pygame.K_w, pygame.K_UP), RotationAction.THRUST),
         ((pygame.K_a, pygame.K_LEFT), RotationAction.ROTATE_LEFT),
         ((pygame.K_d, pygame.K_RIGHT), RotationAction.ROTATE_RIGHT)]
        if style == 'rotation' else
        [((pygame.K_w, pygame.K_UP), DirectAction.UP),
         ((pygame.K_s, pygame.K_DOWN), DirectAction.DOWN),
         ((pygame.K_a, pygame.K_LEFT), DirectAction.LEFT),
         ((pygame.K_d, pygame.K_RIGHT), DirectAction.RIGHT)]
    )
    for keys, action in bindings:
        if held.intersection(keys):
            return int(action)
    return 0


class ArenaApp:
    def __init__(self, *, style='direct', seed=0, human=True, headless=False):
        self.env = ArenaEnv(control_style=style, seed=seed)
        self.renderer = ArenaRenderer(headless=headless, caption=f'A3 Arena — {style}')
        self.seed = seed
        self.human = human
        self.headless = headless
        self.held: set[int] = set()
        self.running = True
        self.paused = False
        self.done = False
        self.accumulator = 0.0
        self.renderer.observe(self.env)

    def reset(self):
        self.env.reset(seed=self.seed)
        self.renderer.reset_visuals()
        self.renderer.observe(self.env)
        self.done = False
        self.accumulator = 0.0
        self.held.clear()

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.WINDOWFOCUSLOST:
            self.held.clear()
            self.paused = True
            self.accumulator = 0.0
        elif event.type == pygame.KEYUP:
            self.held.discard(event.key)
        elif event.type == pygame.KEYDOWN:
            # Ignore repeated keydown events for toggles, but retain held actions.
            if event.key in self.held:
                return
            self.held.add(event.key)
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key == pygame.K_o:
                self.renderer.toggle_observation_overlay()
            elif event.key == pygame.K_e:
                self.renderer.toggle_effects()
            elif event.key == pygame.K_p:
                self.paused = not self.paused
                self.accumulator = 0.0
            elif event.key == pygame.K_r:
                self.reset()

    def update(self, elapsed):
        if self.paused:
            return
        self.renderer.advance(elapsed)
        if self.done:
            return
        self.accumulator += max(0.0, elapsed)
        interval = FIXED_DT * ACTION_REPEAT
        while self.accumulator + 1e-9 >= interval:
            self.accumulator -= interval
            action = (human_action(self.env.control_style, self.held) if self.human
                      else scripted_action(self.env))
            _, _, terminated, truncated, _ = self.env.step(action)
            self.renderer.observe(self.env)
            if terminated or truncated:
                self.done = True
                break

    def run(self, frames=None):
        clock = pygame.time.Clock()
        count = 0
        try:
            self.renderer.draw(self.env)
            while self.running and (frames is None or count < frames):
                elapsed = FIXED_DT if self.headless else min(clock.tick(FPS) / 1000, 0.25)
                if not self.headless:
                    for event in pygame.event.get():
                        self.handle_event(event)
                if not self.running:
                    break
                if self.done and not self.human:
                    self.reset()
                was_done = self.done
                self.update(elapsed)
                if self.done and not was_done:
                    print(f'Episode ended: {self.env.steps} steps. R resets; P pauses/resumes.')
                self.renderer.draw(self.env)
                if not self.headless:
                    mode = 'HUMAN' if self.human else 'SCRIPTED'
                    status = 'PAUSED / P resumes' if self.paused else 'EPISODE OVER / R resets' if self.done else mode
                    pygame.display.set_caption(f'A3 Arena — {self.env.control_style} — {status} — O perception / E effects')
                count += 1
            print(f'Rendered {count} frames; {self.env.enemies_killed} enemies destroyed, '
                  f'{self.env.damage_taken} damage taken.')
        finally:
            self.close()

    def close(self):
        self.renderer.close()
        self.env.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Arena human play or scripted demonstration')
    parser.add_argument('--style', choices=CONTROL_STYLES, default='direct')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--human', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--frames', type=int)
    args = parser.parse_args(argv)
    if args.frames is not None and args.frames < 1:
        parser.error('--frames must be positive')
    if args.headless and args.frames is None:
        parser.error('--headless requires --frames')
    print(('Human play on the training environment.' if args.human
           else 'Scripted demonstration, not a trained policy.') + '\n' + CONTROLS)
    ArenaApp(style=args.style, seed=args.seed, human=args.human, headless=args.headless).run(args.frames)
    return 0
