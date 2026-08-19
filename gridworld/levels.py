"""Level layouts.

Seven 10x10 levels, one per task in the brief. Grid edges act as walls, so no border of rocks is
drawn. Row 0 is the top row.

Legend: `.` empty  `#` rock  `F` fire  `M` monster  `A` apple  `K` key  `C` chest  `S` start
"""

from __future__ import annotations

from .constants import LEGAL_TILES, Tile

# Level 0 — Task 1 (Q-learning). Apples on the right side only, open field, no hazards.
# The optimal policy is a clean shortest path, which makes B5 easy to demonstrate.
LEVEL_0 = """\
S.........
..........
........A.
..........
..........
.......A..
..........
........A.
..........
..........
"""

# Level 1 — Task 2 (SARSA). A cliff-walk: the short route runs along the row directly above a
# wall of fire, the safe route detours upward. Q-learning learns the risky edge path; SARSA,
# accounting for its own exploration, keeps its distance. That contrast is the C3 evidence.
LEVEL_1 = """\
..........
..........
..........
..........
..........
..........
..........
..........
SFFFFFFFFA
..........
"""

# Level 2 — Task 3. Multiple apples plus the key/chest dependency: the chest is worth +2 but is
# only openable after the key has been collected, so the optimal route is order-dependent.
LEVEL_2 = """\
S....#....
.....#..A.
.....#....
..A..#....
.....#....
.....#..C.
..#####...
..........
.K.....A..
..........
"""

# Level 3 — Task 3, harder. Rock maze with a longer key-to-chest dependency.
LEVEL_3 = """\
S.#.......
..#.###.#.
..#...#.#.
....#.#.#A
.####.#...
....#.###.
.A#.#....K
..#.####.#
..#....#..
.....#.C..
"""

# Level 4 — Task 4. Monsters that move after each agent action with 40% probability, plus fire.
LEVEL_4 = """\
S.........
..........
...M......
..........
.....FF...
..A..FF..A
..........
......M...
..........
........A.
"""

# Level 5 — Task 4, harder. Two monsters guarding the key/chest route.
LEVEL_5 = """\
S....#....
.....#....
..M..#..A.
.....#....
..####....
.......M..
.K........
....###...
..A..#....
.....#.C..
"""

# Level 6 — Task 5 (intrinsic reward). Deliberately sparse and exploration-hostile: a single
# distant chest behind a long corridor, with the key at the far end of a dead-end branch. An
# epsilon-greedy agent rarely stumbles onto the reward, so the visit-count intrinsic bonus should
# produce a visibly better learning curve. That comparison is the F-row evidence.
LEVEL_6 = """\
S#........
.#.######.
.#.#....#.
.#.#.##.#.
.#.#.#K.#.
.#.#.####.
.#.#......
.#.######.
.#........
.........C
"""

LEVELS: tuple[str, ...] = (
    LEVEL_0,
    LEVEL_1,
    LEVEL_2,
    LEVEL_3,
    LEVEL_4,
    LEVEL_5,
    LEVEL_6,
)

N_LEVELS = len(LEVELS)


def parse_level(level: str) -> list[list[str]]:
    """Parse a level string into a mutable grid of tile characters."""
    grid = [list(row) for row in level.strip().splitlines()]
    widths = {len(row) for row in grid}
    if len(widths) != 1:
        raise ValueError(f"level rows have inconsistent widths: {sorted(widths)}")
    illegal = {tile for row in grid for tile in row} - LEGAL_TILES
    if illegal:
        raise ValueError(f"level contains illegal tiles: {sorted(illegal)}")
    starts = sum(row.count(Tile.START) for row in grid)
    if starts != 1:
        raise ValueError(f"level must contain exactly one start tile, found {starts}")
    return grid


def load_level(index: int) -> list[list[str]]:
    """Parse level `index` (0-6)."""
    if not 0 <= index < N_LEVELS:
        raise IndexError(f"level {index} out of range 0..{N_LEVELS - 1}")
    return parse_level(LEVELS[index])
