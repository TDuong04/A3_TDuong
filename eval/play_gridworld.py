"""Visual playback of a trained tabular policy.

    python -m eval.play_gridworld --level 1 --algo sarsa
    python -m eval.play_gridworld --level 1 --compare      # Q-learning vs SARSA, TAB to swap
    python -m eval.play_gridworld --level 4                # monsters; leave --env-seed unset

Loads the Q-table from `results/`, opens the Pygame window and plays the greedy policy. This is the
script the video records for Part I, so it has to make three things obvious to a viewer:

  - the level's items and monsters behaving per the rules,
  - the agent following a consistent learned policy rather than acting randomly (rubric V), which
    the policy-arrow overlay demonstrates far better than watching a single rollout,
  - for level 1, the difference between the Q-learning and SARSA routes — the `--compare` mode
    running both at once is the strongest possible C3 evidence.

Keyboard: pause, single-step, speed up/down, reset, toggle policy arrows, toggle Q-heatmap, switch
level, and hand control to a human. All of that lives in `gridworld.render.PlaybackApp`; this module
is the CLI, the table loading and the policy that reads them, and nothing else. Frame capture for
the video is left to screen recording software — cheaper than maintaining a frame dumper, and it
records the audio commentary too.

Implementation notes
--------------------

*The acting policy follows the overlay on screen.* `PlaybackApp` already cycles its overlays with
TAB; `ActiveTablePolicy` reads whichever table is selected at the moment it is asked for an action,
so under `--compare` one keypress swaps both the arrows and the route being walked. Comparing two
routes only means something if the arrows drawn over them belong to the same algorithm.

*A missing table is a user error, not a bug.* Nothing here is trained on demand — a video session
starting a 16,000-episode run by accident is worse than an error message — so an absent `.npz`
exits with the `train.train_gridworld` command that would produce it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # `python eval/play_gridworld.py` as well as `-m`
    sys.path.insert(0, str(REPO_ROOT))

import pygame  # noqa: E402  (imported for `quit()`; the renderer owns every draw call)

from common.seeding import make_rng  # noqa: E402
from gridworld.algorithms import QTable, load_q_table, make_q_table, select_action  # noqa: E402
from gridworld.env import GridWorld  # noqa: E402
from gridworld.levels import N_LEVELS  # noqa: E402
from gridworld.render import CONTROLS, PlaybackApp, algo_label  # noqa: E402

#: The algorithms this script can play back, in the order `--compare` cycles them.
ALGOS: tuple[str, ...] = ("q", "sarsa")

#: Level 1 is the cliff level the C3 comparison is written about, so `--compare` defaults there.
COMPARE_LEVEL = 1

DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

#: A greedy rollout is epsilon 0 by definition — the policy under test, not a tuned number. Anything
#: else is passed explicitly with `--epsilon` to show the behaviour policy instead.
GREEDY_EPSILON = 0.0

#: How many already-trained filenames the "not trained" message lists. Level 6 alone has 25, and an
#: error message that fills the terminal stops being read.
MAX_SUGGESTIONS = 6


class MissingQTableError(FileNotFoundError):
    """A requested Q-table has not been trained. Carries the message the CLI prints verbatim."""


def q_table_path(level: int, algo: str, seed: int, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    """Where `train.train_gridworld` writes the table for one (level, algorithm, seed) run."""
    return Path(results_dir) / f"qtable_level{level}_{algo}_seed{seed}.npz"


def _trained_alternatives(level: int, algo: str, results_dir: Path) -> list[str]:
    """Filenames that exist for this level and algorithm, so the error can suggest a seed.

    Level 6's tables carry an intrinsic-strength tag rather than a plain seed, which is exactly the
    case where a bare "file not found" would send someone hunting through the training script.
    """
    pattern = f"qtable_level{level}_{algo}_*.npz"
    return sorted(path.name for path in Path(results_dir).glob(pattern))


def load_level_tables(
    level: int, algo: str, seed: int, results_dir: Path = DEFAULT_RESULTS_DIR
) -> dict[int, QTable]:
    """The requested level's table, plus every other level trained under the same algo and seed.

    Loading the neighbours costs milliseconds and keeps the number keys useful: switching level
    mid-demo to show that levels 2-5 also work should not drop the arrows off the screen.
    """
    wanted = q_table_path(level, algo, seed, results_dir)
    if not wanted.is_file():
        raise MissingQTableError(_missing_message(level, algo, seed, results_dir))

    tables: dict[int, QTable] = {}
    for other in range(N_LEVELS):
        path = q_table_path(other, algo, seed, results_dir)
        if path.is_file():
            tables[other] = load_q_table(path)
    return tables


def _missing_message(level: int, algo: str, seed: int, results_dir: Path) -> str:
    """One message: what is missing, how to make it, and what was found instead."""
    path = q_table_path(level, algo, seed, results_dir)
    lines = [
        f"no trained Q-table for level {level} ({algo_label(algo)}, seed {seed}).",
        f"  expected: {path}",
        f"  train it: python -m train.train_gridworld --level {level} --algo {algo} --seed {seed}",
    ]
    found = _trained_alternatives(level, algo, results_dir)
    if found:
        shown = ", ".join(found[:MAX_SUGGESTIONS])
        if len(found) > MAX_SUGGESTIONS:
            shown += f", ... ({len(found)} files)"
        lines.append(f"  already trained for this level: {shown}")
    return "\n".join(lines)


class ActiveTablePolicy:
    """Epsilon-greedy over whichever algorithm's table the window currently has selected.

    Bound to the app after construction because the app owns the overlay selection. That indirection
    is the whole of `--compare`: TAB swaps the arrows and the acting policy together, so the two
    routes are walked by the tables they are drawn from.
    """

    def __init__(self, epsilon: float = GREEDY_EPSILON, rng: np.random.Generator | None = None):
        self.epsilon = float(epsilon)
        self.rng = make_rng(None) if rng is None else rng
        self.app: PlaybackApp | None = None
        self._fallback = make_q_table()

    def bind(self, app: PlaybackApp) -> None:
        self.app = app

    def __call__(self, env: GridWorld) -> int:
        table = self.app.q_table if self.app is not None else None
        if table is None:
            # Only reachable by switching to a level this seed never trained. An all-zero row makes
            # every action tie, so the agent walks randomly — and the HUD says so rather than
            # passing a random walk off as a learned policy.
            table = self._fallback
            if self.app is not None:
                self.app.message = f"no Q-table for level {env.level_index} — moving at random"
        return select_action(table[env.state], self.epsilon, self.rng)


def build_app(
    level: int,
    algos: Sequence[str],
    *,
    seed: int = 0,
    env_seed: int | None = None,
    epsilon: float = GREEDY_EPSILON,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    show_arrows: bool = True,
    headless: bool = False,
    cell_size: int | None = None,
    fps: int | None = None,
    policy_rng: np.random.Generator | None = None,
    app_class: type[PlaybackApp] = PlaybackApp,
) -> PlaybackApp:
    """Wire the loaded tables into a `PlaybackApp` that plays them.

    Raises `MissingQTableError` when the requested level has not been trained.

    `algos` holds one name normally and both under `--compare`; the first is the one on screen when
    the window opens.
    """
    overlays = {algo: load_level_tables(level, algo, seed, results_dir) for algo in algos}
    policy = ActiveTablePolicy(epsilon=epsilon, rng=policy_rng)
    app = app_class(
        level=level,
        seed=env_seed,
        policy=policy,
        overlays=overlays,
        algo=algos[0],
        epsilon=epsilon,
        headless=headless,
        cell_size=cell_size,
        fps=fps,
    )
    policy.bind(app)
    if app.renderer.show_policy_arrows != show_arrows:
        app.renderer.toggle_policy_arrows()
    return app


class ComparePlaybackApp(PlaybackApp):
    """`PlaybackApp` that restarts the episode when TAB swaps algorithm.

    Half a Q-learning route followed by half a SARSA one is not a comparison. Restarting means the
    viewer sees each algorithm's route from the same start state, back to back, in one window.
    """

    def cycle_overlay(self) -> str | None:
        before = self.algo
        algo = super().cycle_overlay()
        if algo != before:
            self.reset()
            self.message = f"overlay: {algo_label(algo)} — episode restarted"
        return algo


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI for the playback script. `--level` defaults to 1 under `--compare`, else 0."""
    parser = argparse.ArgumentParser(
        description="Animate a trained gridworld policy in a Pygame window (Part I video)."
    )
    parser.add_argument(
        "--level", type=int, default=None, help=f"level index 0..{N_LEVELS - 1} (default 0)"
    )
    parser.add_argument("--algo", choices=ALGOS, default="q", help="which trained policy to play")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="load both algorithms; TAB swaps arrows and route (defaults to level 1)",
    )
    parser.add_argument("--seed", type=int, default=0, help="training seed naming the Q-table file")
    parser.add_argument(
        "--env-seed",
        type=int,
        default=None,
        help="seed for monster movement; omit for a fresh draw each run",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=GREEDY_EPSILON,
        help="exploration rate of the played policy (0 = greedy)",
    )
    parser.add_argument(
        "--no-arrows", action="store_true", help="start with the policy-arrow overlay off"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="where Q-tables were saved"
    )
    parser.add_argument("--cell-size", type=int, default=None, help="override config cell size")
    parser.add_argument("--fps", type=int, default=None, help="override config frame rate")
    parser.add_argument(
        "--frames", type=int, default=None, help="quit after N frames (headless smoke test)"
    )
    args = parser.parse_args(argv)

    if args.level is None:
        args.level = COMPARE_LEVEL if args.compare else 0
    if not 0 <= args.level < N_LEVELS:
        parser.error(f"--level must be in 0..{N_LEVELS - 1}")
    if not 0.0 <= args.epsilon <= 1.0:
        parser.error("--epsilon must be in 0..1")
    args.algos = list(ALGOS) if args.compare else [args.algo]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: 0 on a clean exit, 2 when a requested Q-table has not been trained."""
    args = parse_args(argv)
    app_class = ComparePlaybackApp if args.compare else PlaybackApp
    try:
        app = build_app(
            args.level,
            args.algos,
            seed=args.seed,
            env_seed=args.env_seed,
            epsilon=args.epsilon,
            results_dir=args.results_dir,
            show_arrows=not args.no_arrows,
            cell_size=args.cell_size,
            fps=args.fps,
            app_class=app_class,
        )
    except MissingQTableError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"Playing level {args.level}: " + ", ".join(algo_label(a) for a in args.algos))
    print("Controls: " + ", ".join(f"{key} = {what}" for key, what in CONTROLS))
    try:
        app.run(max_frames=args.frames)
    finally:
        pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
