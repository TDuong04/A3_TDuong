from __future__ import annotations

import pytest

from gridworld.constants import LEGAL_TILES, Tile
from gridworld.levels import N_LEVELS, load_level, parse_level


@pytest.mark.parametrize("index", range(N_LEVELS))
def test_every_level_parses(index):
    grid = load_level(index)
    assert grid, "level parsed to an empty grid"
    assert len({len(row) for row in grid}) == 1, "rows are ragged"
    assert {tile for row in grid for tile in row} <= LEGAL_TILES


@pytest.mark.parametrize("index", range(N_LEVELS))
def test_exactly_one_start(index):
    grid = load_level(index)
    assert sum(row.count(Tile.START) for row in grid) == 1


@pytest.mark.parametrize("index", range(N_LEVELS))
def test_level_has_at_least_one_collectible(index):
    """An episode ends when all collectibles are obtained; a level with none can never end well."""
    grid = load_level(index)
    tiles = {tile for row in grid for tile in row}
    assert tiles & {Tile.APPLE, Tile.CHEST}


@pytest.mark.parametrize("index", (2, 3, 5, 6))
def test_chest_levels_provide_a_key(index):
    """A chest is only openable while holding a key, so a chest without a key is unwinnable."""
    grid = load_level(index)
    tiles = {tile for row in grid for tile in row}
    if Tile.CHEST in tiles:
        assert Tile.KEY in tiles, f"level {index} has a chest but no key"


def test_level_0_has_apples_only_on_the_right():
    """The brief specifies Level 0 contains only apples, on the right side of the map."""
    grid = load_level(0)
    width = len(grid[0])
    tiles = {tile for row in grid for tile in row}
    assert tiles <= {Tile.EMPTY, Tile.START, Tile.APPLE}, "level 0 must contain only apples"
    for r, row in enumerate(grid):
        for c, tile in enumerate(row):
            if tile == Tile.APPLE:
                assert c >= width // 2, f"apple at ({r},{c}) is not on the right half"


def test_ragged_level_is_rejected():
    with pytest.raises(ValueError):
        parse_level("S..\n....\n")


def test_missing_start_is_rejected():
    with pytest.raises(ValueError):
        parse_level("...\n.A.\n...\n")
