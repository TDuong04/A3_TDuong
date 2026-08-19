"""Pygame renderer for the gridworld — NOT YET IMPLEMENTED.

Rubric row A1 requires the gridworld to be visually rendered, animated and interactive. Console or
text output scores zero, so this module is worth 2 points on its own.

Contract:

`GridRenderer(cell_size, fps)` owns the pygame window and draws a `GridWorld` on demand. No
gameplay logic here, and nothing in `env.py` may import pygame.

Draw with simple shapes (the brief explicitly permits this): rocks as grey blocks, fire as red,
monsters as dark triangles, apples as green circles, the key as a yellow shape, the chest as a
brown box, the agent as a blue circle. Colour-code the agent when it is holding the key.

Animate the agent sliding between cells rather than teleporting — "animated" is in the rubric
wording, and it also makes the video far more legible.

Support two overlays, both toggled by keypress and both directly useful as report figures:
  - policy arrows: the greedy action per cell, which is how B5 ("learned a shortest-path policy")
    gets demonstrated;
  - a per-cell Q-value heatmap.

Interactivity requirement: keyboard control of playback (pause, step, speed, level switch, reset)
and a human-play mode. That satisfies "interactive" and is genuinely useful for debugging the
environment before any agent exists.
"""

from __future__ import annotations


class GridRenderer:
    def __init__(self, cell_size: int = 56, fps: int = 30) -> None:
        raise NotImplementedError("see module docstring for the contract")
