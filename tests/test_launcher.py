"""Tests for the part picker in `eval/launcher.py`.

Three things are worth protecting here and nothing else is.

*The commands it builds.* The picker's only real output is an argv list handed to a subprocess, and
a typo in it turns the whole screen into a button that does nothing. Each command is checked
against the parser of the script it starts, so a flag that `eval/play_arena.py` stops accepting
fails here rather than at a demo.

*That it never blocks in a test.* A menu loop waits for a keypress by design. Every test below
either caps the frames or replaces `choose` outright, the same discipline
`tests/test_arena_eval.py::TestTitleScreenGate` applies to the title screen.

*That the window hands off before the session starts.* Two pygame windows fighting for focus is the
failure this launcher would most plausibly ship with, so the launch test asserts the picker's
surface is already gone at the moment the child command runs.

No window is opened by any of this: the dummy SDL driver is set before pygame is imported, and
every renderer here is a headless one drawing to an off-screen surface.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

import eval.launcher as launcher  # noqa: E402
import eval.play_arena as play_arena  # noqa: E402
import eval.play_gridworld as play_gridworld  # noqa: E402
import gridworld.render as gridworld_render  # noqa: E402
from arena.render import COLOR_BEZEL, ArenaRenderer  # noqa: E402
from gridworld.levels import N_LEVELS  # noqa: E402


@pytest.fixture
def renderer():
    """An off-screen picker renderer, torn down however the test ends."""
    made = ArenaRenderer(headless=True)
    yield made
    made.close()


def count_color(surface, color) -> int:
    pixels = pygame.surfarray.array3d(surface)
    return int(((pixels[..., 0] == color[0])
                & (pixels[..., 1] == color[1])
                & (pixels[..., 2] == color[2])).sum())


# --- the commands it would run ------------------------------------------------------------------


def test_part_one_launches_the_documented_gridworld_playback_command():
    assert launcher.play_command("gridworld") == [sys.executable, "-m", "eval.play_gridworld"]


@pytest.mark.parametrize("style", ["direct", "rotation"])
def test_part_two_launches_the_documented_arena_playback_command(style):
    assert launcher.play_command("arena", style) == [
        sys.executable, "-m", "eval.play_arena", "--style", style,
    ]


@pytest.mark.parametrize("style", ["direct", "rotation"])
def test_the_arena_command_is_accepted_by_the_script_it_starts(style):
    """The real guard against drift: parse the built argv with `play_arena`'s own parser, so a
    renamed or removed flag fails here instead of at the cabinet."""
    argv = launcher.play_command("arena", style)[3:]
    assert play_arena.build_parser().parse_args(argv).style == style


def test_the_gridworld_command_is_accepted_by_the_script_it_starts():
    args = play_gridworld.build_parser().parse_args(launcher.play_command("gridworld")[3:])
    assert args.level is None and args.algo == "q", "defaults changed under the launcher"


def test_the_launcher_offers_exactly_the_styles_play_arena_accepts():
    offered = tuple(option.value for option in launcher.STYLE_MENU if option.value != launcher.BACK)
    assert offered == tuple(sorted(play_arena.STYLES))
    assert set(offered) == set(launcher.STYLES)


def test_an_unknown_part_or_style_is_refused_rather_than_silently_launched():
    with pytest.raises(ValueError):
        launcher.play_command("training")
    with pytest.raises(ValueError):
        launcher.play_command("arena", "keyboard")


# --- the level and agent/human commands -----------------------------------------------------------


@pytest.mark.parametrize("level", range(N_LEVELS))
def test_agent_playback_carries_the_picked_level(level):
    assert launcher.play_command("gridworld", level=level) == [
        sys.executable, "-m", "eval.play_gridworld", "--level", str(level),
    ]


@pytest.mark.parametrize("level", range(N_LEVELS))
def test_human_play_starts_the_hand_played_gridworld_script(level):
    """Human play is `gridworld.render`, not a flag on the policy viewer — there is no such flag."""
    assert launcher.play_command("gridworld", level=level, human=True) == [
        sys.executable, "-m", "gridworld.render", "--level", str(level),
    ]


@pytest.mark.parametrize("style", ["direct", "rotation"])
def test_arena_human_play_adds_the_flag_play_arena_already_has(style):
    assert launcher.play_command("arena", style, human=True) == [
        sys.executable, "-m", "eval.play_arena", "--style", style, "--human",
    ]


@pytest.mark.parametrize("level", range(N_LEVELS))
def test_the_agent_playback_command_is_accepted_by_the_script_it_starts(level):
    args = play_gridworld.build_parser().parse_args(
        launcher.play_command("gridworld", level=level)[3:]
    )
    assert args.level == level


@pytest.mark.parametrize("level", range(N_LEVELS))
def test_the_human_play_command_is_accepted_by_the_script_it_starts(level):
    """Same drift guard as the other three, against `gridworld/render.py`'s own parser."""
    args = gridworld_render.build_parser().parse_args(
        launcher.play_command("gridworld", level=level, human=True)[3:]
    )
    assert args.level == level and args.frames is None


@pytest.mark.parametrize("style", ["direct", "rotation"])
def test_the_arena_human_command_is_accepted_by_the_script_it_starts(style):
    args = play_arena.build_parser().parse_args(
        launcher.play_command("arena", style, human=True)[3:]
    )
    assert args.style == style and args.human is True


def test_a_level_outside_the_levels_that_exist_is_refused():
    with pytest.raises(ValueError):
        launcher.play_command("gridworld", level=N_LEVELS)
    with pytest.raises(ValueError):
        launcher.play_command("gridworld", level=-1, human=True)


def test_none_of_the_four_commands_can_start_training():
    """All four menu paths, checked against the module they would run."""
    commands = [
        launcher.play_command("gridworld", level=0),
        launcher.play_command("gridworld", level=0, human=True),
        launcher.play_command("arena", "direct"),
        launcher.play_command("arena", "direct", human=True),
    ]
    modules = [command[2] for command in commands]
    assert modules == [
        "eval.play_gridworld", "gridworld.render", "eval.play_arena", "eval.play_arena",
    ]
    assert not any(module.startswith("train.") for module in modules)


def test_no_menu_option_can_start_training():
    """Training is out of scope for this screen by decision, not by omission (ticket A3-033): it is
    a seeded, artifact-producing CLI workflow whose provenance a menu click would blur."""
    for option in launcher.MAIN_MENU:
        style = "direct" if option.value == "arena" else None
        module = launcher.play_command(option.value, style)[2]
        assert module.startswith("eval."), f"{option.label} starts {module}, not a playback script"


# --- what the menu shows ------------------------------------------------------------------------


def test_the_main_menu_offers_part_one_and_part_two():
    labels = [option.label for option in launcher.MAIN_MENU]
    assert labels[0].startswith("PART I ") and "GRIDWORLD" in labels[0]
    assert labels[1].startswith("PART II ") and "ARENA" in labels[1]
    assert [option.value for option in launcher.MAIN_MENU] == ["gridworld", "arena"]


def test_the_level_menu_offers_every_level_the_gridworld_has():
    """Seven rows because `gridworld/levels.py` says seven, not because 7 is typed here."""
    playable = [option for option in launcher.LEVEL_MENU if option.value != launcher.BACK]
    assert len(playable) == N_LEVELS == 7
    assert [option.value for option in playable] == [str(level) for level in range(N_LEVELS)]
    assert [option.label for option in playable] == [f"LEVEL {level}" for level in range(N_LEVELS)]
    assert launcher.LEVEL_MENU[-1].value == launcher.BACK, "no way back to the part menu"


@pytest.mark.parametrize("menu", [launcher.GRIDWORLD_MODE_MENU, launcher.ARENA_MODE_MENU])
def test_each_agent_or_human_menu_offers_both_and_a_way_back(menu):
    assert [option.value for option in menu] == [launcher.AGENT, launcher.HUMAN, launcher.BACK]
    assert menu[0].label == "AGENT PLAYBACK" and menu[1].label == "HUMAN PLAY"


def test_the_long_level_menu_still_fits_inside_the_window(renderer):
    """Eight rows at the part menu's spacing would run off a 738-pixel window; they must not."""
    rows = len(launcher.LEVEL_MENU)
    height = renderer.surface_size[1]
    spacing = launcher._row_spacing(rows, height)
    last_row = height // 2 + launcher.OPTIONS_OFFSET + (rows - 1) * spacing
    assert last_row < height - launcher.FOOTER_GAP
    assert launcher._row_spacing(len(launcher.MAIN_MENU), height) == launcher.OPTION_SPACING


def test_the_level_menu_draws_every_level_and_the_selected_level_s_hint(renderer):
    class RecordingFont:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.drawn: list[str] = []

        def render(self, text, *args, **kwargs):
            self.drawn.append(text)
            return self.wrapped.render(text, *args, **kwargs)

    renderer.font_hud = RecordingFont(renderer.font_hud)
    renderer.font_small = RecordingFont(renderer.font_small)
    launcher.draw_menu(renderer, "PART I", "SELECT A LEVEL", launcher.LEVEL_MENU, 6)

    for level in range(N_LEVELS):
        assert any(f"LEVEL {level}" in line for line in renderer.font_hud.drawn)
    assert launcher.LEVEL_HINTS[6] in renderer.font_small.drawn, "selected level explains itself"


def test_the_menu_is_drawn_with_the_arena_cabinet_treatment(renderer):
    """Same bezel as `draw_title`, because it is literally the same pass — `begin_retro_frame` and
    `finish_retro_frame` are shared, so this fails if the picker ever grows its own pipeline."""
    surface = launcher.draw_menu(renderer, "A3 ARCADE", "SELECT A PART", launcher.MAIN_MENU, 0)
    assert surface.get_size() == renderer.surface_size
    assert count_color(surface, COLOR_BEZEL) > 0, "no cabinet bezel drawn"


def test_the_menu_draws_every_option_and_marks_the_selected_one(renderer):
    class RecordingFont:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.drawn: list[str] = []

        def render(self, text, *args, **kwargs):
            self.drawn.append(text)
            return self.wrapped.render(text, *args, **kwargs)

    renderer.font_hud = RecordingFont(renderer.font_hud)
    launcher.draw_menu(renderer, "PART II", "SELECT A CONTROL SCHEME", launcher.STYLE_MENU, 1,
                       blink_on=True)
    drawn = renderer.font_hud.drawn
    assert any("DIRECT" in line for line in drawn)
    assert any("ROTATION" in line for line in drawn)
    assert ">" in drawn, "the selected row has no cursor"


# --- the loop, without a keyboard ----------------------------------------------------------------


def test_the_menu_does_not_block_when_nobody_is_there_to_press_a_key(renderer):
    """The headless counterpart of the title-screen gate: capped frames must end the loop, and the
    absence of a choice must read as 'quit', never as a selection nobody made."""
    assert launcher.choose(renderer, "A3 ARCADE", "SELECT A PART", launcher.MAIN_MENU,
                           max_frames=3) is None


def test_quitting_the_main_menu_launches_nothing(renderer, monkeypatch):
    started: list[list[str]] = []
    monkeypatch.setattr(launcher, "choose", lambda *_a, **_k: None)
    assert launcher.run(renderer, runner=lambda command: started.append(command) or 0) == 0
    assert started == []


def test_choosing_part_two_asks_for_a_style_and_then_launches_it(renderer, monkeypatch):
    answers = iter(["arena", "rotation", launcher.AGENT, None])
    prompts: list[str] = []

    def fake_choose(_renderer, _heading, prompt, _options, **_kwargs):
        prompts.append(prompt)
        return next(answers)

    started: list[list[str]] = []
    monkeypatch.setattr(launcher, "choose", fake_choose)
    launcher.run(renderer, runner=lambda command: started.append(command) or 0)

    assert "CONTROL SCHEME" in prompts[1], "part II must ask for a control style"
    assert started == [[sys.executable, "-m", "eval.play_arena", "--style", "rotation"]]


def test_going_back_from_the_style_menu_returns_to_the_part_menu(renderer, monkeypatch):
    answers = iter(["arena", launcher.BACK, "gridworld", "3", launcher.AGENT, None])
    started: list[list[str]] = []
    monkeypatch.setattr(launcher, "choose", lambda *_a, **_k: next(answers))
    launcher.run(renderer, runner=lambda command: started.append(command) or 0)
    assert started == [[sys.executable, "-m", "eval.play_gridworld", "--level", "3"]]


def scripted_choose(monkeypatch, answers):
    """Replace `choose` with a script of answers, recording the prompts and options it was shown.

    The menu layers are only visible from the outside as a sequence of questions, so the flow tests
    below assert on that sequence rather than on pixels.
    """
    asked: list[tuple[str, tuple[str, ...]]] = []
    remaining = iter(answers)

    def fake_choose(_renderer, _heading, prompt, options, **_kwargs):
        asked.append((prompt, tuple(option.value for option in options)))
        return next(remaining)

    monkeypatch.setattr(launcher, "choose", fake_choose)
    return asked


def test_choosing_part_one_asks_for_a_level_then_for_agent_or_human(renderer, monkeypatch):
    asked = scripted_choose(monkeypatch, ["gridworld", "4", launcher.AGENT, None])
    started: list[list[str]] = []
    launcher.run(renderer, runner=lambda command: started.append(command) or 0)

    assert asked[1][0] == "SELECT A LEVEL"
    assert asked[1][1] == tuple(option.value for option in launcher.LEVEL_MENU)
    assert len([value for value in asked[1][1] if value != launcher.BACK]) == N_LEVELS
    assert asked[2][0] == "WHO IS PLAYING?"
    assert started == [[sys.executable, "-m", "eval.play_gridworld", "--level", "4"]]


def test_part_one_human_play_launches_the_hand_played_window(renderer, monkeypatch):
    scripted_choose(monkeypatch, ["gridworld", "2", launcher.HUMAN, None])
    started: list[list[str]] = []
    launcher.run(renderer, runner=lambda command: started.append(command) or 0)
    assert started == [[sys.executable, "-m", "gridworld.render", "--level", "2"]]


def test_part_two_asks_for_agent_or_human_after_the_style(renderer, monkeypatch):
    asked = scripted_choose(monkeypatch, ["arena", "direct", launcher.HUMAN, None])
    started: list[list[str]] = []
    launcher.run(renderer, runner=lambda command: started.append(command) or 0)

    assert asked[2][0] == "WHO IS PLAYING?"
    assert asked[2][1] == (launcher.AGENT, launcher.HUMAN, launcher.BACK)
    assert started == [[sys.executable, "-m", "eval.play_arena", "--style", "direct", "--human"]]


def test_going_back_from_the_level_menu_returns_to_the_part_menu(renderer, monkeypatch):
    asked = scripted_choose(monkeypatch, ["gridworld", launcher.BACK, "arena", launcher.BACK, None])
    started: list[list[str]] = []
    assert launcher.run(renderer, runner=lambda command: started.append(command) or 0) == 0

    assert [prompt for prompt, _ in asked] == [
        "SELECT A PART", "SELECT A LEVEL", "SELECT A PART", "SELECT A CONTROL SCHEME",
        "SELECT A PART",
    ]
    assert started == [], "backing out is not a launch"


def test_going_back_from_the_gridworld_mode_menu_returns_to_the_level_menu(renderer, monkeypatch):
    """One layer back, not two: the level menu, with the part menu never asked again."""
    asked = scripted_choose(monkeypatch, ["gridworld", "5", launcher.BACK, "1", launcher.AGENT,
                                          None])
    started: list[list[str]] = []
    launcher.run(renderer, runner=lambda command: started.append(command) or 0)

    assert [prompt for prompt, _ in asked[:5]] == [
        "SELECT A PART", "SELECT A LEVEL", "WHO IS PLAYING?", "SELECT A LEVEL", "WHO IS PLAYING?",
    ]
    assert started == [[sys.executable, "-m", "eval.play_gridworld", "--level", "1"]]


def test_going_back_from_the_arena_mode_menu_returns_to_the_style_menu(renderer, monkeypatch):
    asked = scripted_choose(monkeypatch, ["arena", "direct", launcher.BACK, "rotation",
                                          launcher.HUMAN, None])
    started: list[list[str]] = []
    launcher.run(renderer, runner=lambda command: started.append(command) or 0)

    assert [prompt for prompt, _ in asked[:5]] == [
        "SELECT A PART", "SELECT A CONTROL SCHEME", "WHO IS PLAYING?", "SELECT A CONTROL SCHEME",
        "WHO IS PLAYING?",
    ]
    assert started == [[sys.executable, "-m", "eval.play_arena", "--style", "rotation", "--human"]]


@pytest.mark.parametrize("answers", [
    ["gridworld", None],
    ["gridworld", "0", None],
    ["arena", "direct", None],
])
def test_quitting_any_new_menu_quits_the_picker_without_launching(renderer, monkeypatch, answers):
    """ESC or a closed window in a sub-menu must end the picker, not fall through to a launch."""
    scripted_choose(monkeypatch, answers)
    started: list[list[str]] = []
    assert launcher.run(renderer, runner=lambda command: started.append(command) or 0) == 0
    assert started == []


def test_the_picker_window_is_closed_before_the_session_starts(renderer):
    """The focus/input handoff, asserted at the only moment it can be: while the child runs."""
    renderer.begin_retro_frame()
    assert renderer.surface is not None

    surfaces: list[object] = []
    launcher.launch(renderer, ["echo"], runner=lambda _command: surfaces.append(renderer.surface))
    assert surfaces == [None], "the picker still owned a window while the session ran"


def test_the_menu_comes_back_after_a_session_ends(renderer, monkeypatch):
    """`launch` closing the window must not end the picker: the next menu frame reopens one."""
    answers = iter(["gridworld", "0", launcher.AGENT, None])
    monkeypatch.setattr(launcher, "choose", lambda *_a, **_k: next(answers))
    launcher.run(renderer, runner=lambda _command: 0)
    surface = launcher.draw_menu(renderer, "A3 ARCADE", "SELECT A PART", launcher.MAIN_MENU, 0)
    assert surface.get_size() == renderer.surface_size


def test_main_runs_and_exits_on_a_frame_cap(monkeypatch):
    """`python -m eval.launcher --frames N` is the headless smoke test; it must return 0, not hang.

    The runner is stubbed anyway so that no real playback session could start even if the capped
    loop somehow reached a selection.
    """
    monkeypatch.setattr(launcher, "_run_command", lambda _command: 0)
    monkeypatch.setattr(launcher, "ArenaRenderer", lambda **kwargs: ArenaRenderer(headless=True))
    assert launcher.main(["--frames", "2"]) == 0


# --- what it is not allowed to import -------------------------------------------------------------


def test_the_launcher_imports_no_simulation_module():
    """A picker that imported an env would put `step()` one refactor away from the render loop.
    Checked on the source, not on `sys.modules`, because other tests in this suite import both envs
    for their own reasons and would mask the real answer."""
    tree = ast.parse(Path(launcher.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not {name for name in imported if name.endswith((".env", ".entities"))}
    assert "arena.env" not in imported and "gridworld.env" not in imported
