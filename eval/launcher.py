"""A retro cabinet picker for the two playable parts of the assignment.

    python -m eval.launcher

Reaching either playback entry point otherwise means remembering a CLI command and its flags
(`python -m eval.play_gridworld --level N`, `python -m gridworld.render --level N`,
`python -m eval.play_arena --style direct|rotation [--human]`). This screen is the front of the
cabinet: pick PART I or PART II, pick that part's level or control scheme, pick whether the agent
or you are driving, and the existing script is started for you. It is presentation polish (ticket
A3-033), not a rubric row, so it stays one screen and adds no capability of its own — every command
it builds is one the target script already accepted before this file existed.

*Human play is a different script in Part I and a flag in Part II*, because that is how the two
parts were already built: `eval/play_gridworld.py` is the policy viewer and `gridworld/render.py`
is the hand-played window (CLAUDE.md's command table lists them separately), while
`eval/play_arena.py` carries its own `--human`. The launcher follows that split rather than
inventing a uniform flag, which would mean new CLI surface for a presentation screen.

Training is deliberately not reachable from here. `python -m train.train_gridworld` and
`python -m train.train_arena` stay a CLI workflow: they are long, seeded, artifact-producing runs
whose output has to be traceable to the exact command that made it, which a menu click is not.

Design notes
------------

*Nothing in this file touches a simulation.* There is no `ArenaEnv`, no `GridWorld` and no `step()`
anywhere below — the picker only draws, reads keys, and starts a subprocess. That is the same rule
`wait_for_start` and `wait_for_retry` in `eval/play_arena.py` follow, and it is why the picker lives
in `eval/` rather than in either env or renderer.

*The cabinet look is imported, not re-implemented.* `ArenaRenderer.begin_retro_frame` /
`finish_retro_frame` are the same pixel-downscale, scanline and bezel pass `draw_title` uses, so
this screen cannot drift away from the game it launches into. Anything drawn between those two
calls is chunked with the starfield; anything after stays crisp, which is why the headings are
pixelated and the option text is not.

*The window hands off rather than sharing.* `launch()` closes the picker's window before starting
the child process, so only one pygame window exists at a time and the two never compete for focus
or for the keyboard. When the child exits, the picker draws itself again from a fresh surface.

*Keys are read as edges, not as state.* A menu is the one place in this project where held-key
state is the wrong model: holding DOWN for a tenth of a second would scroll past every option.
Each frame diffs the keys that are down against the previous frame's, and acts only on the ones
that just went down. The first frame of a menu is deliberately treated as "nothing is fresh", so
the ENTER that started a session cannot immediately re-select when that session ends and the menu
comes back.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pygame

from arena.render import COLOR_ACCENT, COLOR_TEXT, COLOR_TEXT_DIM, ArenaRenderer
from gridworld.levels import N_LEVELS

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The control styles `eval/play_arena.py --style` accepts, in the order the menu offers them.
STYLES = ("direct", "rotation")

#: Returned by a menu when the user asked to go back rather than to pick something.
BACK = "back"

#: Returned by the agent-or-human menu. Which script each one runs is `play_command`'s business.
AGENT = "agent"
HUMAN = "human"

# Layout geometry only — where the lines sit on a window whose size `ArenaRenderer` owns. These are
# not tunable parameters: there is no experiment whose outcome would change one, which is the test
# CLAUDE.md applies when deciding what belongs in `config/*.yaml`. Measured from the middle of the
# window rather than its top, the way `draw_title` places its own lines, so the block stays centred
# on a window whose height the renderer is free to change.
HEADING_OFFSET = -220
PROMPT_OFFSET = -140
DETAIL_OFFSET = -104
OPTIONS_OFFSET = -40
OPTION_SPACING = 78
HINT_OFFSET = 26
CURSOR_GAP = 26
FOOTER_GAP = 58
FOOTER_CLEARANCE = 40

# Below this row spacing a per-row hint would overlap the next row's label, so the hint for the
# selected row moves up to the detail line under the prompt instead. The level menu is what forces
# this: eight rows do not fit a 738-pixel window at the roomy spacing two or three rows get.
MIN_HINT_SPACING = 56

#: Seconds per on/off half-cycle of the selection cursor, matching the title screen's prompt blink.
BLINK_SECONDS = 0.45


@dataclass(frozen=True)
class MenuOption:
    """One selectable line: the value the caller gets back, plus what the screen shows for it."""

    value: str
    label: str
    hint: str


MAIN_MENU: tuple[MenuOption, ...] = (
    MenuOption("gridworld", "PART I — GRIDWORLD", "tabular Q-learning / SARSA — pick a level next"),
    MenuOption("arena", "PART II — ARENA", "deep RL — pick a control scheme next"),
)

STYLE_MENU: tuple[MenuOption, ...] = (
    MenuOption("direct", "DIRECT", "no-op, up, down, left, right, shoot"),
    MenuOption("rotation", "ROTATION", "no-op, thrust, rotate left, rotate right, shoot"),
    MenuOption(BACK, "BACK", "choose a different part"),
)

# One line per level, describing what the level is for rather than what it contains — a picker is
# read at a glance, and "which one shows SARSA being cautious" is the question being asked. Keyed by
# index so the menu still builds if `gridworld/levels.py` grows a level before this list does.
LEVEL_HINTS: dict[int, str] = {
    0: "open field, three apples — the Q-learning demo",
    1: "cliff walk: fire between the start and the apple",
    2: "apples plus the key/chest dependency",
    3: "rock maze, longer key-to-chest route",
    4: "monsters and fire — they move after you do",
    5: "two monsters guarding the key/chest route",
    6: "sparse and exploration-hostile — the intrinsic-reward level",
}


def _level_options() -> tuple[MenuOption, ...]:
    """One row per gridworld level, counted from `N_LEVELS` rather than from a literal 7.

    `gridworld/levels.py` owns how many levels exist; a picker that disagreed with it would offer a
    level the env cannot load, or hide one the report cites.
    """
    return tuple(
        MenuOption(str(level), f"LEVEL {level}", LEVEL_HINTS.get(level, "gridworld level"))
        for level in range(N_LEVELS)
    )


LEVEL_MENU: tuple[MenuOption, ...] = (
    *_level_options(),
    MenuOption(BACK, "BACK", "choose a different part"),
)

GRIDWORLD_MODE_MENU: tuple[MenuOption, ...] = (
    MenuOption(AGENT, "AGENT PLAYBACK", "watch the trained Q-learning / SARSA policy"),
    MenuOption(HUMAN, "HUMAN PLAY", "play the level yourself — arrow keys"),
    MenuOption(BACK, "BACK", "choose a different level"),
)

ARENA_MODE_MENU: tuple[MenuOption, ...] = (
    MenuOption(AGENT, "AGENT PLAYBACK", "watch the trained PPO policy"),
    MenuOption(HUMAN, "HUMAN PLAY", "fly the arena yourself — keyboard"),
    MenuOption(BACK, "BACK", "choose a different control scheme"),
)

FOOTER = "UP/DOWN or W/S to move   ENTER or SPACE to select   1-9 to jump   ESC to quit"
FOOTER_BACK = "   BACKSPACE to go back"

# Read as edges every frame; anything not listed here is left to `ArenaRenderer._pump_events`,
# which already owns window close and ESC.
KEYS_UP = (pygame.K_UP, pygame.K_w)
KEYS_DOWN = (pygame.K_DOWN, pygame.K_s)
KEYS_SELECT = (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE)
KEYS_BACK = (pygame.K_BACKSPACE,)
KEYS_DIGIT = (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5,
              pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9)
WATCHED_KEYS = (*KEYS_UP, *KEYS_DOWN, *KEYS_SELECT, *KEYS_BACK, *KEYS_DIGIT)


# --- what a choice launches --------------------------------------------------------------------


def play_command(
    part: str,
    style: str | None = None,
    *,
    level: int | None = None,
    human: bool = False,
) -> list[str]:
    """The argv for a part's existing script, as documented in that script's module docstring.

    Four commands come out of here, one per path through the menus::

        gridworld, agent   python -m eval.play_gridworld --level N
        gridworld, human   python -m gridworld.render --level N
        arena, agent       python -m eval.play_arena --style S
        arena, human       python -m eval.play_arena --style S --human

    Built as a list and run without a shell, and with `sys.executable` rather than a bare `python`,
    so the child lands in the same interpreter (and therefore the same virtualenv) the picker is
    running in. A picker that silently started a different Python than the one with pygame and SB3
    installed would look like a broken launcher rather than a missing dependency.

    `level` omitted leaves `--level` off entirely, which is what the menu-free call in the tests
    checks: the target script's own default then applies, rather than this file asserting one.
    """
    if part == "gridworld":
        # Human play is its own entry point, not a flag: `eval/play_gridworld.py` renders a policy
        # and has no keyboard control to switch into.
        command = [sys.executable, "-m", "gridworld.render" if human else "eval.play_gridworld"]
        if level is not None:
            if not 0 <= level < N_LEVELS:
                raise ValueError(f"level {level} out of range 0..{N_LEVELS - 1}")
            command += ["--level", str(level)]
        return command
    if part == "arena":
        if style not in STYLES:
            raise ValueError(f"unknown control style {style!r}; expected one of {STYLES}")
        command = [sys.executable, "-m", "eval.play_arena", "--style", style]
        if human:
            command.append("--human")
        return command
    raise ValueError(f"unknown part {part!r}; expected 'gridworld' or 'arena'")


def _run_command(command: list[str]) -> int:
    """Run a child session to completion from the repo root, so `python -m` resolves packages."""
    return subprocess.call(command, cwd=str(REPO_ROOT))


def launch(
    renderer: ArenaRenderer,
    command: list[str],
    *,
    runner: Callable[[list[str]], int] = _run_command,
) -> int:
    """Close the picker window, run the session, and report its exit code.

    Closing first is the whole handoff: `ArenaRenderer.close()` tears the display down, so the
    child's window is the only one on screen and the picker is not sitting behind it swallowing
    keypresses. `_ensure_surface` reopens a window on the next frame, which is what lets the menu
    come back after the session ends without the caller reconstructing anything.
    """
    renderer.close()
    print("launching: " + " ".join(command))
    return runner(command)


# --- the picker screen ---------------------------------------------------------------------------


def _has_back(options: Sequence[MenuOption]) -> bool:
    """Whether this menu has somewhere to go back to — what makes BACKSPACE mean anything."""
    return any(option.value == BACK for option in options)


def _row_spacing(rows: int, height: int) -> int:
    """Pixels between option rows: the roomy default, squeezed only when the rows would not fit.

    The level menu is eight rows tall where the part and style menus are two or three, so a fixed
    spacing would run the last level off the bottom of the window. Shrinking to fit keeps one
    `draw_menu` for every menu instead of a special case for the long one.
    """
    if rows <= 1:
        return OPTION_SPACING
    band = (height - FOOTER_GAP - FOOTER_CLEARANCE) - (height // 2 + OPTIONS_OFFSET)
    return max(1, min(OPTION_SPACING, band // (rows - 1)))


def draw_menu(
    renderer: ArenaRenderer,
    heading: str,
    prompt: str,
    options: Sequence[MenuOption],
    index: int,
    *,
    blink_on: bool = True,
) -> pygame.Surface:
    """Draw one frame of a menu and return the surface, so headless callers can inspect it."""
    surface, field = renderer.begin_retro_frame()
    mid_x, mid_y = field.width // 2, field.height // 2
    spacing = _row_spacing(len(options), field.height)
    per_row_hints = spacing >= MIN_HINT_SPACING

    # Before the retro pass, like the arena title: the heading is part of the cabinet, not a
    # readout, so it earns the chunky treatment. The option text stays after it and stays crisp.
    title = renderer.font_banner.render(heading, True, COLOR_ACCENT)
    surface.blit(title, title.get_rect(center=(mid_x, mid_y + HEADING_OFFSET)))
    renderer.finish_retro_frame(surface, field)

    subtitle = renderer.font_hud.render(prompt, True, COLOR_TEXT)
    surface.blit(subtitle, subtitle.get_rect(center=(mid_x, mid_y + PROMPT_OFFSET)))

    if not per_row_hints and options:
        # Crowded menu: only the row under the cursor explains itself, on its own line up top.
        detail = renderer.font_small.render(options[index].hint, True, COLOR_TEXT_DIM)
        surface.blit(detail, detail.get_rect(center=(mid_x, mid_y + DETAIL_OFFSET)))

    for row, option in enumerate(options):
        y = mid_y + OPTIONS_OFFSET + row * spacing
        selected = row == index
        label = renderer.font_hud.render(
            f"{row + 1}.  {option.label}", True, COLOR_ACCENT if selected else COLOR_TEXT
        )
        label_rect = label.get_rect(center=(mid_x, y))
        surface.blit(label, label_rect)
        if per_row_hints:
            hint = renderer.font_small.render(option.hint, True, COLOR_TEXT_DIM)
            surface.blit(hint, hint.get_rect(center=(mid_x, y + HINT_OFFSET)))
        if selected and blink_on:
            cursor = renderer.font_hud.render(">", True, COLOR_ACCENT)
            surface.blit(cursor, cursor.get_rect(midright=(label_rect.left - CURSOR_GAP, y)))

    text = FOOTER + (FOOTER_BACK if _has_back(options) else "")
    footer = renderer.font_small.render(text, True, COLOR_TEXT_DIM)
    surface.blit(footer, footer.get_rect(center=(mid_x, field.height - FOOTER_GAP)))

    renderer.present()
    return surface


def _keys_down(pressed: Sequence[bool]) -> frozenset[int]:
    """The watched keys currently held, as a set this frame can be diffed against the last one."""
    return frozenset(key for key in WATCHED_KEYS if pressed[key])


def choose(
    renderer: ArenaRenderer,
    heading: str,
    prompt: str,
    options: Sequence[MenuOption],
    *,
    max_frames: int | None = None,
) -> str | None:
    """Show a menu and block until something is picked. Returns the option's value, or None to quit.

    Same shape as `wait_for_start` in `eval/play_arena.py`: redraw every frame, poll, return only
    when the user has actually decided. None means the window was closed or ESC was pressed —
    `ArenaRenderer._pump_events` owns both of those and reports them through `should_close`.

    `max_frames` is the escape hatch that keeps this testable: a headless run has nobody to press a
    key, so the tests cap the frames instead of hanging, exactly as `play()` caps frames for report
    figures. It is never set by `main()`'s interactive path.
    """
    index = 0
    frames = 0
    blink_frames = max(1, round(renderer.config.fps * BLINK_SECONDS))
    # None, not an empty set: the first frame establishes what is already held (the ENTER that
    # launched the session that just ended, say) so none of it counts as a fresh press.
    previous: frozenset[int] | None = None

    while True:
        draw_menu(renderer, heading, prompt, options, index,
                  blink_on=(frames // blink_frames) % 2 == 0)
        frames += 1
        if renderer.should_close:
            return None
        if max_frames is not None and frames >= max_frames:
            return None

        # A headless picker has no keyboard to read and no display for pygame to read it from, so
        # it only ever draws; `max_frames` is what ends such a run. That is the same arrangement
        # `play()` uses to keep automated runs from waiting on a key nobody is there to press.
        down = frozenset() if renderer.headless else _keys_down(pygame.key.get_pressed())
        fresh = frozenset() if previous is None else down - previous
        previous = down

        if fresh & frozenset(KEYS_UP):
            index = (index - 1) % len(options)
        if fresh & frozenset(KEYS_DOWN):
            index = (index + 1) % len(options)
        if fresh & frozenset(KEYS_BACK) and _has_back(options):
            return BACK
        for row, key in enumerate(KEYS_DIGIT[:len(options)]):
            if key in fresh:
                return options[row].value
        if fresh & frozenset(KEYS_SELECT):
            return options[index].value


#: What a part's sub-menus resolved to: an argv to launch, `BACK`, or None to quit. Three outcomes
#: rather than two because "go back to the part menu" is not "quit the picker".
Choice = list[str] | str | None


def gridworld_command(
    renderer: ArenaRenderer, *, max_frames: int | None = None
) -> Choice:
    """PART I's two questions — which level, then who is driving.

    The loop is what makes BACK step back exactly one layer: backing out of agent-or-human returns
    to the level menu, and only backing out of the level menu returns to the part menu. A straight
    line through the two menus would have to throw the user all the way out to the part menu to
    undo a single keypress.
    """
    while True:
        picked = choose(renderer, "PART I", "SELECT A LEVEL", LEVEL_MENU, max_frames=max_frames)
        if picked is None or picked == BACK:
            return picked

        level = int(picked)
        mode = choose(renderer, f"PART I — LEVEL {level}", "WHO IS PLAYING?",
                      GRIDWORLD_MODE_MENU, max_frames=max_frames)
        if mode is None:
            return None
        if mode == BACK:
            continue
        return play_command("gridworld", level=level, human=mode == HUMAN)


def arena_command(
    renderer: ArenaRenderer, *, max_frames: int | None = None
) -> Choice:
    """PART II's two questions — which control scheme, then who is driving. BACK steps back one."""
    while True:
        style = choose(renderer, "PART II", "SELECT A CONTROL SCHEME", STYLE_MENU,
                       max_frames=max_frames)
        if style is None or style == BACK:
            return style

        mode = choose(renderer, f"PART II — {style.upper()}", "WHO IS PLAYING?",
                      ARENA_MODE_MENU, max_frames=max_frames)
        if mode is None:
            return None
        if mode == BACK:
            continue
        return play_command("arena", style, human=mode == HUMAN)


def run(
    renderer: ArenaRenderer,
    *,
    runner: Callable[[list[str]], int] = _run_command,
    max_frames: int | None = None,
) -> int:
    """The picker loop: choose, launch, and come back to the menu when the session ends.

    Returns a process exit code. The loop is what makes the screen a cabinet rather than a one-shot
    prompt — after a session ends you are back at the menu, which is also what a demo needs when
    both parts have to be shown one after the other.
    """
    try:
        while True:
            part = choose(renderer, "A3 ARCADE", "SELECT A PART", MAIN_MENU, max_frames=max_frames)
            if part is None:
                return 0

            sub_menus = arena_command if part == "arena" else gridworld_command
            command = sub_menus(renderer, max_frames=max_frames)
            if command is None:
                return 0
            if command == BACK:
                continue

            launch(renderer, command, runner=runner)
            # A session that was quit with ESC must not close the picker as well: the window that
            # reported it no longer exists, and the next menu frame opens a fresh one.
            renderer.should_close = False
    finally:
        renderer.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retro picker for the two playable parts: gridworld and arena.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="quit after N menu frames without selecting anything (headless smoke test)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m eval.launcher` — the picker. See the module docstring for what it will not do."""
    args = build_parser().parse_args(argv)
    renderer = ArenaRenderer(caption="A3 Arcade")
    try:
        return run(renderer, max_frames=args.frames)
    except KeyboardInterrupt:
        renderer.close()
        return 0


if __name__ == "__main__":
    sys.exit(main())
