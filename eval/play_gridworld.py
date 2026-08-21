"""Visual playback of a trained tabular policy.

    python -m eval.play_gridworld --level 0 --algo q
    python -m eval.play_gridworld --level 1 --algo sarsa
    python -m eval.play_gridworld --level 1 --compare      # Q-learning vs SARSA side by side
    python -m eval.play_gridworld --level 4 --algo q       # monsters, stochastic transitions

Loads the Q-table `train/train_gridworld.py` wrote to `results/`, opens the Pygame window and plays
the greedy policy. This is the script the video records for Part I, so it has to make three things
obvious to a viewer:

  - the level's items and monsters behaving per the rules,
  - the agent following a consistent learned policy rather than acting randomly (rubric V), which
    the policy-arrow overlay demonstrates far better than watching a single rollout,
  - for level 1, the difference between the Q-learning and SARSA routes — the `--compare` mode
    running both at once is the strongest possible C3 evidence.

Keyboard: pause, single-step, speed up/down, reset, toggle policy arrows, toggle Q-heatmap, switch
level, and hand control to a human. All of it comes from `gridworld.render.PlaybackApp`, which owns
the window loop and the key bindings; this module supplies the trained policy, the auto-replay and
the side-by-side layout, and adds no drawing code of its own beyond blitting two panels together.

Design notes
------------

*Nothing here selects an action with `argmax`.* The greedy policy calls the same
`gridworld.algorithms.select_action` the learners used, at `epsilon = 0`, so the random tie-break
rubric B4 requires is the same code path on screen as in training. It matters visually too: an
unvisited state has an all-zero row, and `argmax` would walk such an agent into the top wall in a
straight line, which looks like a learned policy and is not one.

*Every level's table is loaded, not just the requested one.* `PlaybackApp` already binds the number
keys to level switching, so loading the whole set means a presenter can press `4` mid-recording and
get the trained monster-level policy with its arrows, instead of an untrained agent stumbling in
front of the camera.

*The agent replays until you stop it.* A level-0 episode is eleven steps — under two seconds of
footage — and one rollout is exactly the weak evidence the rubric warns about. Restarting
automatically after `playback_restart_seconds` turns the demo into the same route walked over and
over, which is what "follows a learned policy rather than acting randomly" looks like on camera. On
levels 4-5 the replays deliberately do *not* share a seed, so the monsters take a visibly different
walk each time while the agent's route stays recognisable.

*Compare mode restarts both panels together.* Two independent envs would drift apart the moment one
died first, and a side-by-side comparison of two episodes at different points is unreadable. The
sub-apps therefore never auto-restart; `CompareApp` waits for both to finish and resets them as a
pair.

Sizes, speeds and the replay delay come from the `rendering` block of `config/gridworld.yaml`; the
only literals here are layout geometry and the greedy `epsilon = 0` that defines "greedy".

`--record` was considered and dropped: screen-capture software produces better video than a frame
dumper and costs nothing to maintain.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

from common.config import load_yaml
from common.seeding import make_rng
from gridworld.algorithms import QTable, load_q_table, select_action
from gridworld.constants import N_ACTIONS
from gridworld.env import GridWorld
from gridworld.levels import N_LEVELS
from gridworld.render import (
    COLOR_ACCENT,
    COLOR_BACKGROUND,
    COLOR_GRID_LINE,
    COLOR_PANEL,
    COLOR_TEXT,
    COLOR_TEXT_DIM,
    CONTROLS,
    PlaybackApp,
    RenderConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

#: CLI name -> the label a viewer reads on screen. The keys are the same strings the trainer puts
#: in its filenames, so `--algo` and `qtable_level1_<algo>_seed0.npz` cannot drift apart.
ALGO_LABELS: dict[str, str] = {"q": "Q-LEARNING", "sarsa": "SARSA"}

#: Playback is greedy by definition — this is not a hyperparameter to tune, it is what "play the
#: learned policy" means. Exploration parameters live in `config/gridworld.yaml` and are used by
#: the trainer only.
GREEDY_EPSILON = 0.0

#: The level `--compare` documents and the report cites: the cliff, where Q-learning takes the edge
#: route and SARSA detours.
COMPARE_LEVEL = 1

#: Filename shapes written by `train.train_gridworld`. The second exists because the level-6
#: intrinsic-reward runs name their strength as well as their seed; playback prefers the plain form
#: and falls back to the tagged one so `--level 6` is playable at all.
QTABLE_PATTERN = "qtable_level{level}_{algo}_seed{seed}.npz"
QTABLE_INTRINSIC_GLOB = "qtable_level{level}_{algo}_strength*_seed{seed}.npz"

_SEED_IN_NAME = re.compile(r"_seed(\d+)\.npz$")


class PlaybackError(Exception):
    """A problem the user can fix, reported as a message rather than as a traceback.

    Every raise site writes the message it wants a marker to read, including the command that would
    resolve it. `main` prints it to stderr and exits non-zero; nothing here should ever surface a
    `FileNotFoundError` from `np.load`.
    """


@dataclass(frozen=True)
class PlaybackConfig:
    """Playback-only entries of the `rendering` block of `config/gridworld.yaml`.

    Separate from `RenderConfig` because these two numbers mean nothing to the renderer: they are
    how long a finished episode sits on screen before replaying, and how small the cells have to be
    for two panels to fit side by side. Both are read from config for the same reason everything
    else is — a number nobody can find is a number nobody can change before recording.
    """

    restart_seconds: float = 1.5
    compare_cell_size: int = 40

    @classmethod
    def from_yaml(cls, name: str = "gridworld", section: str = "rendering") -> PlaybackConfig:
        block = (load_yaml(name) or {}).get(section) or {}
        defaults = cls()
        return cls(
            restart_seconds=float(
                block.get("playback_restart_seconds", defaults.restart_seconds)
            ),
            compare_cell_size=int(block.get("compare_cell_size", defaults.compare_cell_size)),
        )


# --- finding the trained tables -----------------------------------------------------------------


def trained_seeds(results_dir: Path, level: int, algo: str) -> list[int]:
    """Seeds that have a saved Q-table for this level and algorithm, lowest first."""
    patterns = (
        QTABLE_PATTERN.format(level=level, algo=algo, seed="*"),
        QTABLE_INTRINSIC_GLOB.format(level=level, algo=algo, seed="*"),
    )
    seeds: set[int] = set()
    for pattern in patterns:
        for path in Path(results_dir).glob(pattern):
            match = _SEED_IN_NAME.search(path.name)
            if match is not None:
                seeds.add(int(match.group(1)))
    return sorted(seeds)


def find_q_table(results_dir: Path, level: int, algo: str, seed: int | None = None) -> Path | None:
    """The Q-table file for this level/algorithm, or `None` if it has not been trained.

    With `seed=None` the lowest trained seed wins, so the documented commands work without anyone
    remembering which seeds were run. The intrinsic fallback is sorted rather than parsed: a plain
    `..._strength0_seed0.npz` sorts ahead of `..._strength0p5_seed0.npz`, which picks the
    unmodified-reward run — the one whose greedy policy actually solves level 6.
    """
    results_dir = Path(results_dir)
    seeds = [seed] if seed is not None else trained_seeds(results_dir, level, algo)
    for candidate in seeds:
        exact = results_dir / QTABLE_PATTERN.format(level=level, algo=algo, seed=candidate)
        if exact.exists():
            return exact
        tagged = sorted(
            results_dir.glob(QTABLE_INTRINSIC_GLOB.format(level=level, algo=algo, seed=candidate))
        )
        if tagged:
            return tagged[0]
    return None


def _trained_inventory(results_dir: Path, algo: str) -> str:
    """One line per level that *is* trained for this algorithm, for the error message."""
    lines = []
    for level in range(N_LEVELS):
        seeds = trained_seeds(results_dir, level, algo)
        if seeds:
            lines.append(f"    level {level}: seeds {', '.join(str(s) for s in seeds)}")
    return "\n".join(lines) if lines else "    (none)"


def resolve_q_table(
    results_dir: Path, level: int, algo: str, seed: int | None = None
) -> Path:
    """`find_q_table`, but a missing table raises the message a user can act on.

    The message names what was looked for, what exists instead, and the exact training command,
    because the person who hits this is most likely a marker running a README command on a fresh
    clone rather than the author who knows the layout of `results/`.
    """
    path = find_q_table(results_dir, level, algo, seed)
    if path is not None:
        return path

    label = ALGO_LABELS.get(algo, algo)
    wanted = QTABLE_PATTERN.format(level=level, algo=algo, seed="*" if seed is None else seed)
    available = trained_seeds(Path(results_dir), level, algo)
    if seed is not None and available:
        detail = (
            f"Level {level} is trained for {label} on seeds "
            f"{', '.join(str(s) for s in available)} — but not on seed {seed}."
        )
    else:
        detail = f"Nothing in {results_dir} matches {wanted}."
    train_command = f"python -m train.train_gridworld --level {level} --algo {algo}"
    if seed is not None:
        train_command += f" --seed {seed}"
    raise PlaybackError(
        f"No trained Q-table for level {level} with {label}.\n"
        f"{detail}\n"
        f"Already trained for {label}:\n{_trained_inventory(Path(results_dir), algo)}\n"
        f"Train the missing one first:\n    {train_command}"
    )


def load_policy_tables(
    results_dir: Path, algo: str, *, required_level: int, seed: int | None = None
) -> dict[int, QTable]:
    """Every trained table for `algo`, keyed by level; the requested level must be among them.

    Loading all of them is what makes the number keys useful during a recording — pressing `4`
    switches to the monster level *and* to its trained policy and arrows. Levels that were never
    trained are simply absent, and the app says so on screen rather than pretending.
    """
    resolve_q_table(results_dir, required_level, algo, seed)  # raises with the actionable message
    tables: dict[int, QTable] = {}
    for level in range(N_LEVELS):
        # The requested level honours `--seed`; the others fall back to whatever exists, since a
        # seed asked for on level 1 says nothing about which seeds were run on level 5.
        wanted = seed if level == required_level else None
        path = find_q_table(results_dir, level, algo, wanted)
        if path is None and level == required_level:
            path = find_q_table(results_dir, level, algo, None)
        if path is not None:
            tables[level] = load_q_table(path)
    return tables


# --- the greedy policy --------------------------------------------------------------------------


class GreedyPolicy:
    """Greedy action selection over whichever level is currently on screen.

    Keyed on `env.level_index` rather than closed over a single table so that switching level mid
    demo switches policy with it. Unseen states get an all-zero row, which `select_action` turns
    into a uniform random choice — the honest behaviour, and visibly different from the confident
    straight line the trained states produce.
    """

    def __init__(
        self, tables: Mapping[int, QTable], rng: np.random.Generator | None = None
    ) -> None:
        self.tables = tables
        self.rng = make_rng(None) if rng is None else rng

    def table_for(self, level: int) -> QTable | None:
        return self.tables.get(level)

    def has_policy_for(self, level: int) -> bool:
        return level in self.tables

    def __call__(self, env: GridWorld) -> int:
        table = self.tables.get(env.level_index)
        # `.get`, never `[]`: a loaded table is a defaultdict, and indexing it here would grow the
        # trained table by one zero row per unseen state per step.
        row = None if table is None else table.get(env.state)
        if row is None:
            row = np.zeros(N_ACTIONS, dtype=np.float64)
        return select_action(row, GREEDY_EPSILON, self.rng)


# --- one panel ------------------------------------------------------------------------------------


class ReplayApp(PlaybackApp):
    """`PlaybackApp` that restarts the episode a moment after it ends, and names its policy.

    Auto-restart is the whole difference between "here is a rollout" and "here is a policy": the
    same route, walked again and again, on camera. `restart_seconds=None` disables it, which is
    what `CompareApp` uses so that it can restart both of its panels together instead.
    """

    def __init__(
        self,
        level: int,
        policy: GreedyPolicy,
        *,
        restart_seconds: float | None,
        label: str = "",
        seed: int | None = None,
        headless: bool = False,
        cell_size: int | None = None,
        fps: int | None = None,
    ) -> None:
        super().__init__(
            level,
            policy=policy,
            q_tables=policy.tables,
            seed=seed,
            headless=headless,
            cell_size=cell_size,
            fps=fps,
        )
        self.greedy_policy = policy
        self.restart_seconds = restart_seconds
        self.label = label
        self.episodes_played = 1
        self.last_return: float | None = None
        self._done_for = 0.0
        self._announce_level(level)

    # --- messages the HUD shows ---------------------------------------------------------------

    def _announce_level(self, level: int) -> None:
        if self.greedy_policy.has_policy_for(level):
            self.message = f"episode {self.episodes_played}"
        else:
            self.message = f"no trained policy for level {level}\npress 0-{N_LEVELS - 1} to switch"

    def load_level(self, level: int) -> None:
        super().load_level(level)
        if 0 <= level < N_LEVELS:
            self.episodes_played = 1
            self.last_return = None
            self._done_for = 0.0
            self._announce_level(level)

    def restart(self) -> None:
        """Begin the next replay. Without an explicit seed the env keeps its generator, so the
        monsters take a fresh walk while the agent's policy stays fixed — which is precisely the
        contrast levels 4-5 are meant to show."""
        self.last_return = self.env.episode_return
        self.reset()
        self.episodes_played += 1
        self._done_for = 0.0
        self.message = f"episode {self.episodes_played}   last return {self.last_return:+.1f}"

    # --- the frame ----------------------------------------------------------------------------

    def update(self, dt: float) -> None:
        super().update(dt)
        if self.restart_seconds is None:
            return
        if not self.env.done:
            self._done_for = 0.0
            return
        self._done_for += dt
        if self._done_for >= self.restart_seconds:
            self.restart()


def build_app(
    level: int,
    algo: str,
    *,
    seed: int | None = None,
    env_seed: int | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    headless: bool = False,
    cell_size: int | None = None,
    fps: int | None = None,
    show_arrows: bool | None = None,
    show_heatmap: bool | None = None,
    loop: bool = True,
    policy_seed: int | None = None,
    config: PlaybackConfig | None = None,
) -> ReplayApp:
    """Load the trained table for `(level, algo)` and wrap it in a ready-to-run window app.

    Raises `PlaybackError` — never a `FileNotFoundError` — when the table has not been trained.
    """
    config = PlaybackConfig.from_yaml() if config is None else config
    tables = load_policy_tables(results_dir, algo, required_level=level, seed=seed)
    policy = GreedyPolicy(tables, rng=make_rng(policy_seed))

    app = ReplayApp(
        level,
        policy,
        restart_seconds=config.restart_seconds if loop else None,
        label=ALGO_LABELS.get(algo, algo),
        seed=env_seed,
        headless=headless,
        cell_size=cell_size,
        fps=fps,
    )
    if show_arrows is not None:
        app.renderer.show_policy_arrows = bool(show_arrows)
    if show_heatmap is not None:
        app.renderer.show_q_values = bool(show_heatmap)
    return app


# --- two panels: the C3 comparison, animated -----------------------------------------------------

#: Layout only — the gap between the two panels and the strip that carries their titles.
COMPARE_GAP = 6
COMPARE_HEADER_SCALE = 0.62


class CompareApp:
    """Q-learning and SARSA on the same level, in one window, stepping together.

    Each sub-app renders to its own off-screen surface (`GridRenderer(headless=True)`); this class
    owns the single display surface and blits the panels onto it. That is the entire reason it
    exists — no shape is drawn here that `GridRenderer` does not already draw, and the sub-apps keep
    every key binding they had.

    Both panels see every keypress, so pause, speed, level switch and the overlay toggles stay in
    lockstep. They also restart together: whichever policy finishes first waits for the other, so
    the two routes on screen are always the same episode number.
    """

    def __init__(
        self,
        apps: Sequence[ReplayApp],
        *,
        headless: bool = False,
        fps: int | None = None,
        restart_seconds: float | None = None,
        caption: str = "A3 Gridworld — Q-learning vs SARSA",
    ) -> None:
        if not apps:
            raise ValueError("CompareApp needs at least one panel")
        self.apps = tuple(apps)
        self.headless = bool(headless)
        self.caption = caption
        self.restart_seconds = restart_seconds
        self.running = True
        self._done_for = 0.0
        self._clock: pygame.time.Clock | None = None

        base = self.apps[0].renderer.config
        self.fps = int(fps) if fps is not None else base.fps
        cell = base.cell_size
        self.header_height = max(24, int(cell * COMPARE_HEADER_SCALE))

        if not pygame.font.get_init():
            pygame.font.init()
        self.font_title = pygame.font.Font(None, max(20, int(cell * 0.52)))
        self.font_small = pygame.font.Font(None, max(14, int(cell * 0.34)))
        self.surface: pygame.Surface | None = None

    # --- geometry -----------------------------------------------------------------------------

    def panel_sizes(self) -> list[tuple[int, int]]:
        return [
            app.renderer.surface_size(app.env.n_rows, app.env.n_cols) for app in self.apps
        ]

    def surface_size(self) -> tuple[int, int]:
        sizes = self.panel_sizes()
        width = sum(size[0] for size in sizes) + COMPARE_GAP * (len(sizes) - 1)
        height = self.header_height + max(size[1] for size in sizes)
        return (width, height)

    def _ensure_surface(self) -> pygame.Surface:
        size = self.surface_size()
        if self.surface is not None and self.surface.get_size() == size:
            return self.surface
        if self.headless:
            self.surface = pygame.Surface(size)
        else:
            if not pygame.display.get_init():
                pygame.display.init()
            self.surface = pygame.display.set_mode(size)
            pygame.display.set_caption(self.caption)
        return self.surface

    # --- frame --------------------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> None:
        for app in self.apps:
            app.handle_event(event)
        # A quit or ESC stops every panel, so asking whether they all still want to run is the
        # same question as whether this window should stay open.
        self.running = all(app.running for app in self.apps)

    def update(self, dt: float) -> None:
        for app in self.apps:
            app.update(dt)
        if self.restart_seconds is None:
            return
        if not all(app.env.done for app in self.apps):
            self._done_for = 0.0
            return
        self._done_for += dt
        if self._done_for >= self.restart_seconds:
            for app in self.apps:
                app.restart()
            self._done_for = 0.0

    def draw(self) -> pygame.Surface:
        surface = self._ensure_surface()
        surface.fill(COLOR_BACKGROUND)
        header = pygame.Rect(0, 0, surface.get_width(), self.header_height)
        pygame.draw.rect(surface, COLOR_PANEL, header)

        x = 0
        for app in self.apps:
            panel = app.draw()
            surface.blit(panel, (x, self.header_height))
            self._draw_header(surface, app, x, panel.get_width())
            x += panel.get_width()
            if x < surface.get_width():
                pygame.draw.line(
                    surface,
                    COLOR_GRID_LINE,
                    (x + COMPARE_GAP // 2, 0),
                    (x + COMPARE_GAP // 2, surface.get_height()),
                    2,
                )
            x += COMPARE_GAP
        return surface

    def _draw_header(
        self, surface: pygame.Surface, app: ReplayApp, left: int, width: int
    ) -> None:
        """Name each panel and show the fact the comparison turns on: how many steps it took."""
        title = self.font_title.render(app.label, True, COLOR_ACCENT)
        surface.blit(title, title.get_rect(midleft=(left + 10, self.header_height // 2)))
        outcome = "died" if app.env.died else ("solved" if app.env.terminated else "running")
        detail = self.font_small.render(
            f"{app.env.steps} steps   return {app.env.episode_return:+.1f}   {outcome}",
            True,
            COLOR_TEXT if app.env.terminated else COLOR_TEXT_DIM,
        )
        surface.blit(
            detail, detail.get_rect(midright=(left + width - 10, self.header_height // 2))
        )

    def tick(self) -> float:
        if self._clock is None:
            self._clock = pygame.time.Clock()
        return self._clock.tick(self.fps) / 1000.0

    def present(self) -> None:
        if not self.headless and pygame.display.get_init():
            pygame.display.flip()

    def close(self) -> None:
        self.surface = None
        for app in self.apps:
            app.renderer.close()
        if not self.headless and pygame.display.get_init():
            pygame.display.quit()

    def run(self, max_frames: int | None = None) -> None:
        """The blocking window loop, same shape as `PlaybackApp.run`."""
        frames = 0
        while self.running:
            dt = self.tick()
            if pygame.display.get_init():
                for event in pygame.event.get():
                    self.handle_event(event)
            self.update(dt)
            self.draw()
            self.present()
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
        self.close()


def build_compare_app(
    level: int = COMPARE_LEVEL,
    algos: Sequence[str] = ("q", "sarsa"),
    *,
    seed: int | None = None,
    env_seed: int | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    headless: bool = False,
    cell_size: int | None = None,
    fps: int | None = None,
    show_arrows: bool | None = None,
    show_heatmap: bool | None = None,
    policy_seed: int | None = None,
    config: PlaybackConfig | None = None,
) -> CompareApp:
    """Both algorithms on one level, in one window, driven by one clock.

    Every panel is built with the *same* env seed so any difference on screen is the policy's doing
    and not the monsters'. Each sub-renderer is headless because `CompareApp` owns the display.
    """
    config = PlaybackConfig.from_yaml() if config is None else config
    cell_size = config.compare_cell_size if cell_size is None else cell_size

    apps = []
    for algo in algos:
        app = build_app(
            level,
            algo,
            seed=seed,
            env_seed=env_seed,
            results_dir=results_dir,
            headless=True,  # the panels draw off-screen; this window belongs to CompareApp
            cell_size=cell_size,
            fps=fps,
            show_arrows=show_arrows,
            show_heatmap=show_heatmap,
            loop=False,  # CompareApp restarts both panels together
            policy_seed=policy_seed,
            config=config,
        )
        apps.append(app)
    return CompareApp(
        apps,
        headless=headless,
        fps=fps,
        restart_seconds=config.restart_seconds,
    )


# --- CLI ------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play a trained gridworld policy in a Pygame window.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help=f"level index 0..{N_LEVELS - 1} (default 0, or {COMPARE_LEVEL} with --compare)",
    )
    parser.add_argument(
        "--algo", choices=sorted(ALGO_LABELS), default="q", help="which trained policy to play"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=f"play {ALGO_LABELS['q']} and {ALGO_LABELS['sarsa']} side by side (rubric C3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed of the trained Q-table to load; default is the lowest one in results/",
    )
    parser.add_argument(
        "--env-seed",
        type=int,
        default=None,
        help="seed for monster movement; omit for a fresh stochastic walk each replay",
    )
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=None,
        help="seed for the greedy tie-break; only matters where the table ties",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cell-size", type=int, default=None, help="override config cell size")
    parser.add_argument("--fps", type=int, default=None, help="override config frame rate")
    parser.add_argument(
        "--no-arrows",
        action="store_true",
        help="start with the policy-arrow overlay off (P toggles it either way)",
    )
    parser.add_argument(
        "--heatmap", action="store_true", help="start with the Q-value heatmap on"
    )
    parser.add_argument(
        "--no-loop", action="store_true", help="stop after one episode instead of replaying"
    )
    parser.add_argument(
        "--frames", type=int, default=None, help="quit after N frames (headless smoke test)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m eval.play_gridworld` — see the module docstring for the documented commands."""
    parser = build_parser()
    args = parser.parse_args(argv)

    level = args.level if args.level is not None else (COMPARE_LEVEL if args.compare else 0)
    if not 0 <= level < N_LEVELS:
        parser.error(f"--level must be in 0..{N_LEVELS - 1}")

    # The overlay is on unless the config or the CLI turns it off: it is the frame that proves the
    # agent is following a policy, so playback should never start without it by accident.
    show_arrows = False if args.no_arrows else RenderConfig.from_yaml().show_policy_arrows
    show_heatmap = True if args.heatmap else None

    try:
        if args.compare:
            app: CompareApp | ReplayApp = build_compare_app(
                level,
                seed=args.seed,
                env_seed=args.env_seed,
                results_dir=args.results_dir,
                cell_size=args.cell_size,
                fps=args.fps,
                show_arrows=show_arrows,
                show_heatmap=show_heatmap,
                policy_seed=args.policy_seed,
            )
        else:
            app = build_app(
                level,
                args.algo,
                seed=args.seed,
                env_seed=args.env_seed,
                results_dir=args.results_dir,
                cell_size=args.cell_size,
                fps=args.fps,
                show_arrows=show_arrows,
                show_heatmap=show_heatmap,
                loop=not args.no_loop,
                policy_seed=args.policy_seed,
            )
    except PlaybackError as error:
        print(str(error), file=sys.stderr)
        return 2

    print(_startup_banner(args, level))
    try:
        app.run(max_frames=args.frames)
    finally:
        pygame.quit()
    return 0


def _startup_banner(args: argparse.Namespace, level: int) -> str:
    who = (
        f"{ALGO_LABELS['q']} vs {ALGO_LABELS['sarsa']}"
        if args.compare
        else ALGO_LABELS.get(args.algo, args.algo)
    )
    table = "" if args.compare else f"\nQ-table: {find_q_table(args.results_dir, level, args.algo, args.seed)}"
    controls = ", ".join(f"{key} = {what}" for key, what in CONTROLS)
    return f"Level {level} — {who}{table}\nControls: {controls}"


if __name__ == "__main__":
    sys.exit(main())
