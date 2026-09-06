"""Headless tests for `gridworld.render`.

Pygame normally needs a display, which a CI box and a marker's laptop may not have. Every test here
runs under `SDL_VIDEODRIVER=dummy` (set in the fixture below, before pygame touches the video
subsystem) and draws to an off-screen `pygame.Surface` via `GridRenderer(headless=True)`, so the
whole renderer is exercised without a window ever opening.

What these tests can and cannot do: they read the pixels the renderer actually produced, so they
prove that each tile type reaches the screen in its own colour, that the keyed agent looks different
from the unkeyed one, and that policy arrows appear only where the Q-table says something. They
cannot prove the picture is *pretty* — that is the plan's M2 gate, which is ten minutes of
`python -m gridworld.render` by hand.

Pixel probes read cell centres and count exact-match arrow pixels. Exact matching is safe because
every shape here is drawn with `pygame.draw`, which does not antialias; the only antialiased pixels
on the surface are font glyphs, and no glyph colour reaches `COLOR_ARROW`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402  (must follow the driver assignment above)

from common.config import load_yaml  # noqa: E402
from gridworld import render as render_module  # noqa: E402
from gridworld.constants import Action, Tile  # noqa: E402
from gridworld.env import GridWorld  # noqa: E402
from gridworld.levels import N_LEVELS  # noqa: E402
from gridworld.render import (  # noqa: E402
    COLOR_AGENT,
    COLOR_APPLE,
    COLOR_ARROW,
    COLOR_CHEST,
    COLOR_FIRE_CORE,
    COLOR_FLOOR,
    COLOR_KEY,
    COLOR_MONSTER,
    COLOR_ROCK,
    HUMAN_KEYS,
    GridRenderer,
    PlaybackApp,
    RenderConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: `rendering` keys that must survive in `config/gridworld.yaml` for the block to mean anything.
RENDERING_KEYS = (
    "cell_size",
    "fps",
    "show_policy_arrows",
    "show_q_values",
    "animation_seconds",
    "steps_per_second",
)


@pytest.fixture
def renderer() -> GridRenderer:
    """An off-screen renderer. Small cells keep the surfaces cheap across 7 levels x 4 overlays."""
    made = GridRenderer(cell_size=24, headless=True)
    yield made
    made.close()


@pytest.fixture
def pixel_renderer() -> GridRenderer:
    """An off-screen renderer at the shipped 56px cell, where the pixel probes were calibrated.

    The 24px fixture is fine for smoke coverage, but at that size the chest's trim line lands on the
    cell centre and the shapes are only a handful of pixels across. The palette and arrow probes
    below sample the size the game is actually played at.
    """
    made = GridRenderer(cell_size=56, headless=True)
    yield made
    made.close()


def populated_q_table(env: GridWorld) -> dict[tuple, np.ndarray]:
    """A Q-table covering every non-rock cell for the env's current key/mask, with distinct values.

    Distinct values matter: a row where all four actions tie carries no policy, and the renderer
    deliberately draws nothing for it, so a uniform table would exercise none of the arrow code.
    """
    table: dict[tuple, np.ndarray] = {}
    for row in range(env.n_rows):
        for col in range(env.n_cols):
            values = np.array([row, col, -row, -col], dtype=np.float64) * 0.1
            table[(row, col, env.has_key, env.collected_mask)] = values
    return table


# --- pixel probes -------------------------------------------------------------------------------
#
# Written once here rather than inline per test: every assertion below that is worth anything is
# some variation on "read this rectangle back and count what is in it".

#: Tile character -> the label the assertions talk about. `START` and `MONSTER` never appear in
#: `env.grid` — `reset()` lifts both out of the layout — so neither needs an entry.
TILE_LABELS: dict[str, str] = {
    Tile.EMPTY: "floor",
    Tile.ROCK: "rock",
    Tile.FIRE: "fire",
    Tile.APPLE: "apple",
    Tile.KEY: "key",
    Tile.CHEST: "chest",
}

#: The colour each label must be found in at a cell centre. Taken from the renderer's own palette
#: rather than retyped, so a deliberate palette tweak moves both sides at once; what is under test
#: is the tile-to-colour *mapping*, not the hex values. For reference, at `cell_size=56` these
#: resolve to floor (46,50,62), rock (120,124,134), fire (244,158,66), apple (68,190,92),
#: key (238,206,66), chest (146,96,44), monster (36,22,46), agent (74,148,236).
EXPECTED_TILE_COLORS: dict[str, tuple[int, int, int]] = {
    "floor": COLOR_FLOOR,
    "rock": COLOR_ROCK,
    # The flame core sits over the red block, and the core is what a centre sample hits.
    "fire": COLOR_FIRE_CORE,
    "apple": COLOR_APPLE,
    "key": COLOR_KEY,
    "chest": COLOR_CHEST,
    "monster": COLOR_MONSTER,
    "agent": COLOR_AGENT,
}


def cell_pixels(renderer: GridRenderer, surface: pygame.Surface, row: int, col: int) -> np.ndarray:
    """The `(width, height, 3)` RGB block for one cell. A copy, so it survives the next redraw."""
    rect = renderer._cell_rect(row, col)
    return pygame.surfarray.array3d(surface)[rect.left : rect.right, rect.top : rect.bottom]


def grid_pixels(renderer: GridRenderer, surface: pygame.Surface, env: GridWorld) -> np.ndarray:
    """The playfield only. The HUD strip and the legend panel are text, not tiles."""
    cell = renderer.config.cell_size
    frame = pygame.surfarray.array3d(surface)
    top = renderer.hud_height
    return frame[0 : env.n_cols * cell, top : top + env.n_rows * cell]


def cell_center_color(
    renderer: GridRenderer, surface: pygame.Surface, row: int, col: int
) -> tuple[int, int, int]:
    """The single pixel at the middle of a cell — where every tile shape is centred."""
    center = renderer._cell_rect(row, col).center
    return tuple(int(channel) for channel in surface.get_at(center)[:3])


def arrow_mask(block: np.ndarray) -> np.ndarray:
    """Boolean map of the pixels drawn in the policy-arrow colour."""
    return np.all(block == np.array(COLOR_ARROW, dtype=block.dtype), axis=-1)


def count_arrow_pixels(block: np.ndarray) -> int:
    return int(arrow_mask(block).sum())


def sampled_tile_colors(
    renderer: GridRenderer, surface: pygame.Surface, env: GridWorld
) -> dict[str, tuple[int, int, int]]:
    """Cell-centre colour for one representative cell of every tile type the level contains.

    The agent and the monsters are drawn on top of whatever tile they stand on, so their cells are
    excluded from the static sweep and sampled separately under their own labels.
    """
    occupied = {env.agent_pos, *env.monster_positions}
    found: dict[str, tuple[int, int, int]] = {}
    for row in range(env.n_rows):
        for col in range(env.n_cols):
            if (row, col) in occupied:
                continue
            label = TILE_LABELS.get(env.tile_at(row, col))
            if label is not None and label not in found:
                found[label] = cell_center_color(renderer, surface, row, col)
    found["agent"] = cell_center_color(renderer, surface, *env.agent_pos)
    if env.monster_positions:
        found["monster"] = cell_center_color(renderer, surface, *env.monster_positions[0])
    return found


# --- surface allocation -------------------------------------------------------------------------


def test_headless_renderer_draws_without_a_display(renderer):
    env = GridWorld(level_index=0, seed=0)
    surface = renderer.draw(env)

    assert isinstance(surface, pygame.Surface)
    assert not pygame.display.get_init(), "a headless renderer must never open the video subsystem"


def test_surface_has_the_size_the_renderer_advertises(renderer):
    env = GridWorld(level_index=0, seed=0)
    surface = renderer.draw(env)

    expected = renderer.surface_size(env.n_rows, env.n_cols)
    assert surface.get_size() == expected
    # Grid plus the HUD strip and the legend panel, so the window is strictly larger than the grid.
    assert expected[0] == env.n_cols * renderer.config.cell_size + renderer.legend_width
    assert expected[1] == env.n_rows * renderer.config.cell_size + renderer.hud_height


@pytest.mark.parametrize("level", range(N_LEVELS))
def test_every_level_draws(renderer, level):
    """Smoke test: no level layout makes the renderer raise, and each one puts ink on the grid.

    The size check here is deliberately weak — it compares `_ensure_surface`'s output against the
    function that sized it, so on its own it proves only "did not raise". The real per-level
    assertions are the palette probes further down; this one exists to cover the drawing path for
    every layout, plus the floor-and-agent check that stops an empty surface passing as a level.
    """
    env = GridWorld(level_index=level, seed=0)
    surface = renderer.draw(env)

    assert surface.get_size() == renderer.surface_size(env.n_rows, env.n_cols)
    grid = grid_pixels(renderer, surface, env)
    assert np.all(grid == np.array(COLOR_FLOOR, dtype=grid.dtype), axis=-1).any(), "no floor drawn"
    assert cell_center_color(renderer, surface, *env.agent_pos) == COLOR_AGENT


@pytest.mark.parametrize("level", range(N_LEVELS))
@pytest.mark.parametrize("arrows", [False, True])
@pytest.mark.parametrize("heatmap", [False, True])
@pytest.mark.parametrize("table", ["empty", "populated"])
def test_every_level_draws_with_every_overlay_combination(renderer, level, arrows, heatmap, table):
    """Smoke test over the overlay matrix, with the one assertion the matrix can actually make.

    The size check proves nothing (see `test_every_level_draws`), so the assertion that matters here
    is the arrow-pixel count: arrows appear exactly when the toggle is on *and* the table has
    something to say. That holds for every level and independently of the heatmap, which is the only
    claim this combinatorial sweep is in a position to make. The arrows' direction, keying and
    tie-handling are pinned one at a time in the overlay-pixel section below.
    """
    env = GridWorld(level_index=level, seed=0)
    renderer.show_policy_arrows = arrows
    renderer.show_q_values = heatmap
    q_table = {} if table == "empty" else populated_q_table(env)

    surface = renderer.draw(env, q_table)

    assert surface.get_size() == renderer.surface_size(env.n_rows, env.n_cols)
    drawn = count_arrow_pixels(grid_pixels(renderer, surface, env))
    if arrows and table == "populated":
        assert drawn > 0, "the arrow overlay is on and the table has a policy, so arrows must show"
    else:
        assert drawn == 0, "no arrow may be drawn with the overlay off or with nothing learnt"


def test_drawing_a_defaultdict_q_table_does_not_grow_it(renderer):
    """The trained table is a `defaultdict`; indexing it per cell per frame would corrupt it."""
    from collections import defaultdict

    env = GridWorld(level_index=2, seed=0)
    table = defaultdict(lambda: np.zeros(4))
    table[env.state] = np.array([1.0, 0.0, 0.0, 0.0])
    renderer.show_policy_arrows = True
    renderer.show_q_values = True

    renderer.draw(env, table)

    assert len(table) == 1


def test_overlays_survive_a_mid_episode_state(renderer):
    """Arrows are keyed on the live `(has_key, collected_mask)`, which changes during an episode."""
    env = GridWorld(level_index=2, seed=0)
    for action in [Action.DOWN] * 8 + [Action.RIGHT]:  # down the left edge onto the key at (8,1)
        if env.done:
            break
        env.step(action)
    assert env.has_key, "the walk should have picked the key up"
    assert env.collected_mask, "and the mask should have moved off zero"

    renderer.show_policy_arrows = True
    renderer.show_q_values = True
    live = populated_q_table(env)
    surface = renderer.draw(env, live)

    assert surface.get_size() == renderer.surface_size(env.n_rows, env.n_cols)
    assert count_arrow_pixels(grid_pixels(renderer, surface, env)) > 0, (
        "the table is keyed on the mid-episode (has_key, mask), so the arrows must still be found"
    )
    stale = {(row, col, False, 0): values for (row, col, _, _), values in live.items()}
    assert count_arrow_pixels(grid_pixels(renderer, renderer.draw(env, stale), env)) == 0, (
        "a table keyed on the episode's opening state must draw nothing once the key is held"
    )


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(np.zeros(4), id="all-zero"),
        pytest.param(np.full(4, -3.0), id="all-negative"),
        pytest.param(np.array([-2.0, -1.0, -0.5, -0.25]), id="negative-spread"),
    ],
)
def test_heatmap_handles_degenerate_value_ranges(renderer, values):
    """An untrained or all-negative table must still shade, not divide by a zero span."""
    env = GridWorld(level_index=0, seed=0)
    table = {
        (row, col, env.has_key, env.collected_mask): values.copy()
        for row in range(env.n_rows)
        for col in range(env.n_cols)
    }
    renderer.show_q_values = True

    surface = renderer.draw(env, table)

    assert surface.get_size() == renderer.surface_size(env.n_rows, env.n_cols)
    # "Still shade" is the whole point: no plain floor may survive under a fully covered table.
    grid = grid_pixels(renderer, surface, env)
    assert not np.all(grid == np.array(COLOR_FLOOR, dtype=grid.dtype), axis=-1).any()


# --- what actually reaches the pixels -------------------------------------------------------------


@pytest.mark.parametrize("level", range(N_LEVELS))
def test_tile_colours_are_distinct_from_the_floor_and_from_each_other(pixel_renderer, level):
    """Every tile type the level contains must arrive on screen in its own colour.

    Two failure modes this is aimed at, both of which leave the renderer running happily: a tile
    drawn in the floor colour (invisible), and two different tiles sharing one shape (an apple and a
    chest that a marker cannot tell apart). Distinctness is the assertion that matters; the mapping
    check alongside it is what names the offender when it breaks.
    """
    env = GridWorld(level_index=level, seed=0)
    surface = pixel_renderer.draw(env)

    found = sampled_tile_colors(pixel_renderer, surface, env)
    assert "floor" in found and "agent" in found, "every level has open floor and a start tile"

    for label, color in found.items():
        assert color == EXPECTED_TILE_COLORS[label], f"{label} drawn in {color}"
        if label != "floor":
            assert color != found["floor"], f"{label} is indistinguishable from the floor"

    assert len(set(found.values())) == len(found), f"two tile types share a colour: {found}"


@pytest.mark.parametrize("level", range(N_LEVELS))
def test_the_keyed_agent_looks_different_from_the_unkeyed_agent(pixel_renderer, level):
    """A1 asks for the holding-the-key state to be visually distinct, on every level."""
    env = GridWorld(level_index=level, seed=0)
    row, col = env.agent_pos

    unkeyed = cell_pixels(pixel_renderer, pixel_renderer.draw(env), row, col)
    env.has_key = True
    keyed = cell_pixels(pixel_renderer, pixel_renderer.draw(env), row, col)

    assert not np.array_equal(unkeyed, keyed), "the keyed agent renders identically to the unkeyed"


# The arrow probes all target one empty cell of level 0, away from the agent at (0, 0), so nothing
# is drawn over the arrow afterwards.
ARROW_CELL = (5, 5)


def arrows_in_target_cell(renderer: GridRenderer, env: GridWorld, table) -> np.ndarray:
    surface = renderer.draw(env, table)
    return arrow_mask(cell_pixels(renderer, surface, *ARROW_CELL))


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([0.0, 0.0, 0.0, 0.0], id="all-zero"),
        pytest.param([-3.0, -3.0, -3.0, -3.0], id="all-equal-negative"),
        pytest.param([2.5, 2.5, 2.5, 2.5], id="all-equal-positive"),
    ],
)
def test_an_all_equal_q_row_draws_no_arrow(pixel_renderer, values):
    """An untouched state carries no policy; `argmax` of it would fabricate the B4 UP bias."""
    env = GridWorld(level_index=0, seed=0)
    pixel_renderer.show_policy_arrows = True
    table = {(*ARROW_CELL, env.has_key, env.collected_mask): np.array(values)}

    assert int(arrows_in_target_cell(pixel_renderer, env, table).sum()) == 0


def test_a_clear_argmax_draws_one_arrow_pointing_that_way(pixel_renderer):
    """The counterpart to the test above: a real preference is drawn, and drawn where it points."""
    env = GridWorld(level_index=0, seed=0)
    pixel_renderer.show_policy_arrows = True
    key = (*ARROW_CELL, env.has_key, env.collected_mask)

    up = arrows_in_target_cell(pixel_renderer, env, {key: np.array([1.0, 0.0, 0.0, 0.0])})
    down = arrows_in_target_cell(pixel_renderer, env, {key: np.array([0.0, 1.0, 0.0, 0.0])})

    assert up.sum() > 0 and down.sum() > 0
    assert not np.array_equal(up, down), "UP and DOWN must not draw the same arrow"
    half = up.shape[1] // 2
    assert up[:, :half].sum() > up[:, half:].sum(), "the UP arrow leans into the top of the cell"
    assert down[:, half:].sum() > down[:, :half].sum(), "and DOWN into the bottom"


def test_a_two_way_tie_draws_both_arrows(pixel_renderer):
    """A genuine tie is honest information; collapsing it to one arrow would hide it."""
    env = GridWorld(level_index=0, seed=0)
    pixel_renderer.show_policy_arrows = True
    key = (*ARROW_CELL, env.has_key, env.collected_mask)

    up = arrows_in_target_cell(pixel_renderer, env, {key: np.array([1.0, 0.0, 0.0, 0.0])})
    down = arrows_in_target_cell(pixel_renderer, env, {key: np.array([0.0, 1.0, 0.0, 0.0])})
    tie = arrows_in_target_cell(pixel_renderer, env, {key: np.array([1.0, 1.0, 0.0, 0.0])})

    assert tie.sum() > max(up.sum(), down.sum()), "a tie draws more ink than either arrow alone"
    half = tie.shape[1] // 2
    assert tie[:, :half].sum() > 0 and tie[:, half:].sum() > 0, "both tied arrows must be present"


#: Same cell, three different live states, three different greedy actions. A renderer that looked
#: the state up with a hardcoded `(False, 0)` would draw the first row's arrow in all three.
LIVE_STATE_TABLE = {
    (4, 4, False, 0): np.array([1.0, 0.0, 0.0, 0.0]),  # UP
    (4, 4, False, 1): np.array([0.0, 1.0, 0.0, 0.0]),  # DOWN
    (4, 4, True, 0): np.array([0.0, 0.0, 0.0, 1.0]),  # RIGHT
}


def test_arrows_follow_the_live_key_and_collected_mask(pixel_renderer):
    """Overlays are keyed on the CURRENT `(has_key, collected_mask)`, not on the episode's start."""
    env = GridWorld(level_index=0, seed=0)
    pixel_renderer.show_policy_arrows = True

    drawn = {}
    for has_key, mask in [(False, 0), (False, 1), (True, 0)]:
        env.has_key = has_key
        env.collected_mask = mask
        surface = pixel_renderer.draw(env, LIVE_STATE_TABLE)
        drawn[(has_key, mask)] = arrow_mask(cell_pixels(pixel_renderer, surface, 4, 4))

    for state, mask_image in drawn.items():
        assert mask_image.sum() > 0, f"state {state} is in the table and must draw its arrow"
    states = list(drawn)
    for first, second in [(0, 1), (0, 2), (1, 2)]:
        assert not np.array_equal(drawn[states[first]], drawn[states[second]]), (
            f"{states[first]} and {states[second]} drew one arrow: the live state is ignored"
        )


def test_a_state_the_table_has_never_seen_draws_no_arrow(pixel_renderer):
    """Unvisited states are skipped, not invented — which is what the `defaultdict` care buys."""
    env = GridWorld(level_index=0, seed=0)
    pixel_renderer.show_policy_arrows = True
    env.has_key = True
    env.collected_mask = 5  # a (key, mask) pair that `LIVE_STATE_TABLE` does not contain

    surface = pixel_renderer.draw(env, LIVE_STATE_TABLE)

    assert count_arrow_pixels(cell_pixels(pixel_renderer, surface, 4, 4)) == 0
    assert count_arrow_pixels(grid_pixels(pixel_renderer, surface, env)) == 0


# --- overlay toggles ----------------------------------------------------------------------------


@pytest.mark.parametrize("arrows, heatmap", [(False, True), (True, False)])
def test_the_renderer_takes_its_defaults_from_the_config_file(monkeypatch, arrows, heatmap):
    """Patch the `rendering` block to unmistakable values and check the renderer follows them.

    Asserting `renderer.show_policy_arrows is RenderConfig.from_yaml().show_policy_arrows` would be
    circular — both sides come from the same call — and would still pass with the YAML block deleted
    and the dataclass literals quietly taking over. Feeding in values that match no default, both
    ways round so neither boolean can be right by luck, is the only version of this test with teeth.
    """
    block = {
        "cell_size": 37,
        "fps": 11,
        "show_policy_arrows": arrows,
        "show_q_values": heatmap,
        "animation_seconds": 0.44,
        "steps_per_second": 3.25,
    }
    monkeypatch.setattr(render_module, "load_yaml", lambda name: {"rendering": block})

    config = RenderConfig.from_yaml()
    assert (config.cell_size, config.fps) == (37, 11)
    assert (config.animation_seconds, config.steps_per_second) == (0.44, 3.25)

    made = GridRenderer(headless=True)
    try:
        assert made.show_policy_arrows is arrows
        assert made.show_q_values is heatmap
        assert made.config.cell_size == 37
        assert made.config.fps == 11
        assert made._anim_duration == pytest.approx(0.44)
    finally:
        made.close()

    override = GridRenderer(cell_size=24, fps=90, headless=True)
    try:
        assert override.config.cell_size == 24, "an explicit argument wins over the config value"
        assert override.config.fps == 90
        assert override.show_policy_arrows is arrows, "but only the arguments that were passed"
    finally:
        override.close()


def test_the_shipped_config_still_carries_every_rendering_key():
    """The patched test above cannot notice a key going missing from the real file; this can."""
    block = load_yaml("gridworld")["rendering"]

    missing = [key for key in RENDERING_KEYS if key not in block]
    assert not missing, f"config/gridworld.yaml lost rendering keys: {missing}"
    assert RenderConfig.from_yaml() == RenderConfig(**{key: block[key] for key in RENDERING_KEYS})


def test_toggles_flip_the_overlays(renderer):
    arrows = renderer.show_policy_arrows
    heatmap = renderer.show_q_values

    assert renderer.toggle_policy_arrows() is (not arrows)
    assert renderer.toggle_q_values() is (not heatmap)
    assert renderer.toggle_policy_arrows() is arrows
    assert renderer.toggle_q_values() is heatmap


# --- animation ----------------------------------------------------------------------------------


def test_agent_interpolates_between_cells_rather_than_teleporting(renderer):
    env = GridWorld(level_index=0, seed=0)
    renderer.sync(env, animate=False)
    start = env.agent_pos
    env.step(Action.DOWN)
    renderer.sync(env)

    assert renderer.animation_progress == 0.0
    midway = renderer._interpolated(start, env.agent_pos)
    assert midway == pytest.approx(tuple(float(v) for v in start)), "the slide starts where it was"

    renderer.advance(renderer._anim_duration / 2)
    midway = renderer._interpolated(start, env.agent_pos)
    assert 0.0 < renderer.animation_progress < 1.0
    assert start[0] < midway[0] < env.agent_pos[0], "the agent is between the two cells"

    renderer.advance(renderer._anim_duration)
    assert renderer.animation_progress == 1.0
    assert renderer._interpolated(start, env.agent_pos) == pytest.approx(
        tuple(float(v) for v in env.agent_pos)
    )


def test_reset_and_level_switch_snap_instead_of_sliding(renderer):
    env = GridWorld(level_index=0, seed=0)
    env.step(Action.DOWN)
    renderer.sync(env)
    renderer.sync(env, animate=False)

    assert renderer.animation_progress == 1.0


# --- the interactive app ------------------------------------------------------------------------


def key(code: int) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYDOWN, key=code)


@pytest.fixture
def app() -> PlaybackApp:
    made = PlaybackApp(level=0, seed=0, headless=True, cell_size=24)
    yield made
    made.renderer.close()


def test_human_play_moves_the_agent_with_wasd_and_arrows(app):
    start = app.env.agent_pos
    app.handle_event(key(pygame.K_DOWN))
    after_arrow = app.env.agent_pos
    app.handle_event(key(pygame.K_d))
    after_wasd = app.env.agent_pos

    assert app.mode == "human"
    assert after_arrow == (start[0] + 1, start[1])
    assert after_wasd == (after_arrow[0], after_arrow[1] + 1)
    assert set(HUMAN_KEYS.values()) == set(Action), "all four actions must be reachable by hand"


def test_pause_toggles_and_single_step_advances_a_policy(app):
    app.policy = lambda env: int(Action.DOWN)
    app.paused = False
    assert app.status()["paused"] is False

    app.handle_event(key(pygame.K_SPACE))
    assert app.paused is True

    before = app.env.steps
    app.handle_event(key(pygame.K_n))
    assert app.env.steps == before + 1
    assert app.paused is True, "single-step must leave playback paused"


def test_speed_keys_change_the_step_rate_and_are_clamped(app):
    baseline = app.steps_per_second
    app.handle_event(key(pygame.K_EQUALS))
    assert app.steps_per_second > baseline
    app.handle_event(key(pygame.K_MINUS))
    assert app.steps_per_second == pytest.approx(baseline)

    for _ in range(50):
        app.handle_event(key(pygame.K_MINUS))
    assert app.steps_per_second >= 0.5
    for _ in range(100):
        app.handle_event(key(pygame.K_EQUALS))
    assert app.steps_per_second <= 60.0


def test_reset_key_restarts_the_episode(app):
    app.handle_event(key(pygame.K_DOWN))
    app.handle_event(key(pygame.K_DOWN))
    assert app.env.steps == 2

    app.handle_event(key(pygame.K_r))

    assert app.env.steps == 0
    assert app.env.episode_return == 0.0
    assert app.env.done is False


@pytest.mark.parametrize("level", range(N_LEVELS))
def test_number_keys_switch_level(app, level):
    app.handle_event(key(getattr(pygame, f"K_{level}")))

    assert app.env.level_index == level
    surface = app.draw()
    assert surface.get_size() == app.renderer.surface_size(app.env.n_rows, app.env.n_cols)


def test_overlay_keys_toggle_both_overlays(app):
    arrows = app.renderer.show_policy_arrows
    heat = app.renderer.show_q_values

    app.handle_event(key(pygame.K_p))
    app.handle_event(key(pygame.K_q))
    assert app.renderer.show_policy_arrows is (not arrows)
    assert app.renderer.show_q_values is (not heat)

    app.handle_event(key(pygame.K_h))
    assert app.renderer.show_q_values is heat, "H is an alias for Q"


def test_escape_stops_the_loop(app):
    app.handle_event(key(pygame.K_ESCAPE))

    assert app.running is False


def test_policy_playback_advances_on_its_own_clock(app):
    app.policy = lambda env: int(Action.RIGHT)
    app.paused = False
    app.steps_per_second = 10.0

    app.update(0.51)  # half a second at 10 steps/second

    assert app.env.steps == 5


def test_stepping_a_finished_episode_is_refused_not_crashed(app):
    app.policy = lambda env: int(Action.DOWN)
    app.paused = False
    while not app.env.done:
        app.policy_step()

    app.policy_step()
    app.handle_event(key(pygame.K_UP))
    app.update(1.0)

    assert app.env.done is True
    assert "reset" in app.message


def test_run_loop_survives_a_headless_frame_budget(app):
    app.run(max_frames=3)

    assert app.running is True, "the frame budget, not a quit event, ended the loop"


# --- the dependency direction --------------------------------------------------------------------


#: Run in a fresh interpreter: this test process has already imported pygame, so `sys.modules` here
#: can say nothing about what `gridworld.env` drags in.
IMPORT_PROBE = textwrap.dedent(
    """
    import sys

    import gridworld.env  # noqa: F401

    leaked = sorted(
        name
        for name in sys.modules
        if name == "pygame"
        or name.startswith("pygame.")
        or name == "render"
        or name.endswith(".render")
    )
    # Tagged because pygame prints a version banner to stdout the moment it is imported, which is
    # exactly the situation this probe reports on.
    print("LEAKED " + " ".join(leaked))
    sys.exit(1 if leaked else 0)
    """
)


def test_env_does_not_import_the_renderer():
    """Rendering depends on the env, never the reverse — otherwise training pulls in pygame.

    A regex over `env.py` only sees that file's own import lines, so any transitive route (env ->
    some helper -> render -> pygame) walks straight past it. Importing the module for real in a
    clean interpreter and reading `sys.modules` back is the check that cannot be routed around.
    """
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=120,
    )

    tag = "LEAKED "
    leaked = [line[len(tag) :] for line in result.stdout.splitlines() if line.startswith(tag)]
    assert result.returncode == 0, (
        f"importing gridworld.env pulled in: {'; '.join(leaked) or result.stdout}\n{result.stderr}"
    )


# --- the learner panel (rubric B/C visibility) -------------------------------------------------


class _RecordingFont:
    """A font that remembers the strings drawn through it."""

    def __init__(self, font) -> None:
        self.font, self.drawn = font, []

    def render(self, text, *args, **kwargs):
        self.drawn.append(text)
        return self.font.render(text, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.font, name)


FULL_SUMMARY = {
    "level": 1, "algo": "sarsa", "seed": 0, "episodes": 8000, "alpha": 0.1, "gamma": 0.95,
    "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay_episodes": 6000,
}


class TestLearnerInfo:
    def test_a_summary_becomes_the_lines_the_panel_draws(self):
        rows = dict(render_module.LearnerInfo.from_summary(FULL_SUMMARY).lines())
        assert rows["algorithm"] == "SARSA"
        assert rows["alpha"] == "0.1"
        assert rows["gamma"] == "0.95"
        assert rows["epsilon"] == "1 -> 0.05"
        assert rows["decay over"] == "6000 ep"

    def test_the_algorithm_key_is_spelled_out_for_the_viewer(self):
        """`q` on screen would not tell a marker which algorithm produced the policy."""
        rows = dict(render_module.LearnerInfo.from_summary({**FULL_SUMMARY, "algo": "q"}).lines())
        assert rows["algorithm"] == "Q-learning"

    def test_missing_fields_are_skipped_rather_than_invented(self):
        """A table whose training summary predates this panel still plays back; it just says less.
        Printing a default here would put a schedule on screen that the table never saw."""
        rows = dict(render_module.LearnerInfo(algorithm="q").lines())
        assert rows == {"algorithm": "Q-learning"}

    def test_an_empty_info_draws_nothing_at_all(self):
        assert render_module.LearnerInfo().lines() == ()

    def test_the_intrinsic_strength_is_shown_when_there_is_one(self):
        """Rubric row F is about the bonus, so its strength belongs on screen for level 6."""
        rows = dict(render_module.LearnerInfo(algorithm="q", intrinsic_strength=0.25).lines())
        assert rows["intrinsic"] == "0.25"


class TestLearnerPanel:
    def test_the_hyperparameters_reach_the_screen(self, renderer):
        """CLAUDE.md's visibility rule: epsilon, alpha and gamma must be legible in the window,
        not only in a log the marker never opens."""
        env = GridWorld(level_index=1, seed=0)
        renderer.font_small = _RecordingFont(renderer.font_small)
        renderer.font_title = _RecordingFont(renderer.font_title)
        info = render_module.LearnerInfo.from_summary(FULL_SUMMARY)
        renderer.draw(env, None, {"learner": info})

        drawn = renderer.font_small.drawn + renderer.font_title.drawn
        assert "Learner" in drawn
        for text in ("SARSA", "alpha", "0.1", "gamma", "0.95", "1 -> 0.05"):
            assert text in drawn, f"{text!r} never reached the screen"

    def test_the_panel_is_absent_without_a_learner(self, renderer):
        env = GridWorld(level_index=1, seed=0)
        renderer.font_title = _RecordingFont(renderer.font_title)
        renderer.draw(env, None, {})
        assert "Learner" not in renderer.font_title.drawn

    def test_the_controls_legend_survives_the_new_block(self, renderer):
        """The learner block is drawn above the controls; neither may push the other off."""
        env = GridWorld(level_index=1, seed=0)
        renderer.font_title = _RecordingFont(renderer.font_title)
        renderer.draw(env, None, {"learner": render_module.LearnerInfo.from_summary(FULL_SUMMARY)})
        assert {"Learner", "Controls", "Overlays"} <= set(renderer.font_title.drawn)

    def test_a_non_learner_status_value_is_ignored(self, renderer):
        """`status` is a loose mapping from the app; a stray value must not crash the window."""
        env = GridWorld(level_index=0, seed=0)
        renderer.draw(env, None, {"learner": "sarsa"})
