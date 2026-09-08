"""Visual playback of a trained deep RL agent, and a human-play mode for the same arena.

    python -m eval.play_arena --style rotation
    python -m eval.play_arena --style direct --episodes 5
    python -m eval.play_arena --style direct --human
    python -m eval.play_arena --style both --episodes 5 --no-window   # the comparison table

Rubric row I4 asks for an evaluation script that can visually run each trained agent, so this works
for both styles from `models/`. Actions are taken with `deterministic=True`: a stochastic policy on
camera looks like a broken one, and the rubric asks for the deterministic policy anyway.

This is what the video records for Part II. It has to show enemies spawning and moving, projectiles
and collisions, and at least one phase progression -- so the summary below reports how often a phase
was actually cleared, which is the thing to check *before* recording rather than after.

`--human` plays the same environment from the keyboard, through the same `env.step()` the agent
uses. It is a creativity feature, and it is also the fastest way to tell a broken environment from a
badly trained agent: if a person cannot clear phase 1 either, the problem is not the policy.

Input handling: the renderer pumps the event queue for window-level events (quit, overlay toggle),
and human control reads `pygame.key.get_pressed()` instead of the queue. Held keys are the right
model for a real-time game anyway -- you hold thrust, you do not tap it -- and reading key *state*
rather than key *events* means the two never compete for the same queue.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from arena.constants import DirectAction, RotationAction
from arena.env import ArenaEnv
from arena.policy_view import probe

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
#: Where `--save` writes. The report reads these instead of anyone retyping a terminal scroll.
RESULTS_DIR = REPO_ROOT / "results" / "arena_eval"
STYLES = ("direct", "rotation")

#: Evaluation is deterministic. The brief asks for it, and a policy that samples on camera reads as
#: an agent that has not learned anything.
DETERMINISTIC = True


class ArenaEvalError(Exception):
    """A problem the user can fix, printed as a message rather than raised as a traceback."""


def model_path(style: str, algo: str = "ppo") -> Path:
    return MODELS_DIR / f"{algo}_{style}.zip"


def resolve_model(style: str, algo: str = "ppo", models_dir: Path | None = None) -> Path:
    """The saved model for a style, or an error naming the command that would produce it."""
    directory = Path(models_dir) if models_dir is not None else MODELS_DIR
    path = directory / f"{algo}_{style}.zip"
    if path.exists():
        return path
    existing = sorted(p.name for p in directory.glob("*.zip")) if directory.exists() else []
    raise ArenaEvalError(
        f"No trained model for control style {style!r}.\n"
        f"Looked for {path}.\n"
        f"Models present: {', '.join(existing) if existing else '(none)'}\n"
        f"Train it first:\n    python -m train.train_arena --style {style}"
    )


def load_agent(style: str, algo: str = "ppo", models_dir: Path | None = None):
    """Load the SB3 policy. Imported lazily so `--human` needs no torch and no model on disk."""
    from stable_baselines3 import DQN, PPO

    path = resolve_model(style, algo, models_dir)
    loader = {"ppo": PPO, "dqn": DQN}[algo]
    return loader.load(path, device="cpu")


# --- headless evaluation --------------------------------------------------------------------------


def evaluate(
    style: str,
    *,
    episodes: int = 5,
    seed: int = 0,
    algo: str = "ppo",
    models_dir: Path | None = None,
    agent: Any | None = None,
) -> dict[str, Any]:
    """Run `episodes` seeded episodes with no window and return the numbers the report needs.

    Seeded from a fixed base so both control styles meet the same sequence of arenas. Report row R6
    compares the two, and a comparison across different layouts compares the layouts as much as the
    agents.
    """
    if style not in STYLES:
        raise ValueError(f"unknown control style {style!r}; expected one of {STYLES}")
    agent = load_agent(style, algo, models_dir) if agent is None else agent

    env = ArenaEnv(control_style=style, render_mode=None)
    returns: list[float] = []
    phases: list[int] = []
    spawners: list[int] = []
    kills: list[int] = []
    steps: list[int] = []
    survived: list[bool] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        info: dict[str, Any] = {}
        while not done:
            action, _ = agent.predict(obs, deterministic=DETERMINISTIC)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        returns.append(float(env.episode_reward))
        phases.append(int(info.get("phase", 1)))
        spawners.append(int(info.get("spawners_destroyed", 0)))
        kills.append(int(info.get("enemies_killed", 0)))
        steps.append(int(env.steps))
        survived.append(not terminated)
    env.close()

    def mean(values: list[Any]) -> float:
        return float(statistics.fmean(values)) if values else 0.0

    return {
        "style": style,
        "episodes": episodes,
        "seed": seed,
        "return_mean": mean(returns),
        "return_std": float(statistics.pstdev(returns)) if len(returns) > 1 else 0.0,
        "phase_mean": mean(phases),
        "phase_max": max(phases) if phases else 0,
        # The number the video depends on: a phase progression has to be visible at least once.
        "phase_cleared_episodes": sum(1 for p in phases if p > 1),
        "spawners_mean": mean(spawners),
        "enemies_mean": mean(kills),
        "steps_mean": mean(steps),
        "survival_rate": mean([1.0 if s else 0.0 for s in survived]),
        "returns": returns,
        "phases": phases,
    }


def format_summary(stats: dict[str, Any]) -> str:
    """One block per style, in the shape report row R6 tabulates."""
    return "\n".join(
        (
            f"style              {stats['style']}",
            f"episodes           {stats['episodes']} (seeds {stats['seed']}"
            f"-{stats['seed'] + stats['episodes'] - 1})",
            f"return             {stats['return_mean']:+.2f} +/- {stats['return_std']:.2f}",
            f"phase reached      {stats['phase_mean']:.2f} mean, {stats['phase_max']} best",
            f"phase cleared in   {stats['phase_cleared_episodes']} of {stats['episodes']} episodes",
            f"spawners destroyed {stats['spawners_mean']:.2f}",
            f"enemies killed     {stats['enemies_mean']:.2f}",
            f"survival           {stats['survival_rate']:.0%} "
            f"({stats['steps_mean']:.0f} steps mean)",
        )
    )


# --- persistence ---------------------------------------------------------------------------


def _reportable(path: Path) -> str:
    """Repo-relative path, so the JSON reads the same on any machine."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_results(
    stats: list[dict[str, Any]], results_dir: Path | None = None
) -> dict[str, Path]:
    """Persist evaluation numbers to `results/arena_eval/`, returning what was written.

    `evaluate()` measures everything report row R6 tabulates and then used to drop it on the floor:
    the numbers existed only in a terminal scroll, so every one of them reached the report by hand.
    Hand-copied numbers are how a report ends up disagreeing with its own artifacts, which is worse
    than having no table at all. One JSON per style keeps the raw episode lists; the markdown is
    the comparison table itself, ready to lift.
    """
    directory = Path(results_dir) if results_dir is not None else RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(UTC).isoformat(timespec="seconds")

    written: dict[str, Path] = {}
    for entry in stats:
        path = directory / f"eval_{entry['style']}_seed{entry['seed']}.json"
        path.write_text(json.dumps({**entry, "generated": generated}, indent=2) + "\n")
        written[entry["style"]] = path

    table = directory / "comparison.md"
    table.write_text(format_comparison_markdown(stats, generated))
    written["comparison"] = table
    return written


def format_comparison_markdown(stats: list[dict[str, Any]], generated: str) -> str:
    """The control-scheme comparison as a markdown table, in the shape report row R6 wants.

    Ranked by nothing: with two control styles the interesting thing is the pair, not a winner, and
    sorting a two-row table implies a verdict the seeds may not support.
    """
    header = (
        f"# Arena control-scheme comparison\n\n"
        f"Generated {generated} by `python -m eval.play_arena --style both --no-window`.\n\n"
        f"Deterministic policy (`deterministic={DETERMINISTIC}`), both styles evaluated on the "
        f"same seed sequence so they meet the same arenas.\n\n"
    )
    columns = (
        "| Style | Episodes | Return | Phase reached | Phases cleared | Spawners | Enemies "
        "| Survival | Steps |\n"
        "|-------|---------:|-------:|--------------:|---------------:|---------:|--------:"
        "|---------:|------:|\n"
    )
    rows = "".join(
        f"| `{entry['style']}` | {entry['episodes']} "
        f"| {entry['return_mean']:+.2f} ± {entry['return_std']:.2f} "
        f"| {entry['phase_mean']:.2f} (best {entry['phase_max']}) "
        f"| {entry['phase_cleared_episodes']}/{entry['episodes']} "
        f"| {entry['spawners_mean']:.2f} | {entry['enemies_mean']:.2f} "
        f"| {entry['survival_rate']:.0%} | {entry['steps_mean']:.0f} |\n"
        for entry in stats
    )
    return header + columns + rows


# --- human control ----------------------------------------------------------------------------


def human_action(keys, style: str) -> int:
    """Map the held keys to an action index for this control style.

    Held state rather than key events: you hold thrust in a real-time game, you do not tap it, and
    reading state leaves the event queue entirely to the renderer.
    """
    import pygame

    if style == "rotation":
        if keys[pygame.K_SPACE]:
            return int(RotationAction.SHOOT)
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            return int(RotationAction.ROTATE_LEFT)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            return int(RotationAction.ROTATE_RIGHT)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            return int(RotationAction.THRUST)
        return int(RotationAction.NOOP)

    if keys[pygame.K_SPACE]:
        return int(DirectAction.SHOOT)
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        return int(DirectAction.UP)
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        return int(DirectAction.DOWN)
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        return int(DirectAction.LEFT)
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        return int(DirectAction.RIGHT)
    return int(DirectAction.NOOP)


# --- the window -------------------------------------------------------------------------------


def play(
    style: str,
    *,
    episodes: int = 3,
    seed: int = 0,
    human: bool = False,
    algo: str = "ppo",
    models_dir: Path | None = None,
    headless: bool = False,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Run the env in a window, driven by the agent or by the keyboard.

    Returns the same summary shape `evaluate` does, so a recorded session and a headless run are
    directly comparable.
    """
    import pygame

    from arena.render import ArenaRenderer

    agent = None if human else load_agent(style, algo, models_dir)
    env = ArenaEnv(control_style=style, render_mode="human")
    renderer = ArenaRenderer(headless=headless)

    returns: list[float] = []
    phases: list[int] = []
    frames = 0
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            info: dict[str, Any] = {}
            while not done:
                # Human keys are read after this frame's draw, never before: draw() is what lazily
                # calls pygame.display.init(), and pygame.key.get_pressed() raises "video system
                # not initialized" if it runs on a display that has never been opened. The agent
                # branch has no such dependency, so it probes before stepping — the panel must
                # describe the observation that produced the action, not the one the step is about
                # to produce.
                if human:
                    view, action = None, None
                else:
                    predicted, _ = agent.predict(obs, deterministic=DETERMINISTIC)
                    action = int(predicted)
                    view = probe(agent, obs, style, action, algo=algo)

                renderer.draw(env, view)
                frames += 1
                if renderer.should_close:
                    raise KeyboardInterrupt
                if max_frames is not None and frames >= max_frames:
                    raise KeyboardInterrupt

                if human:
                    action = human_action(pygame.key.get_pressed(), style)

                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            renderer.draw(env)  # the final frame: the death or the clock running out
            returns.append(float(env.episode_reward))
            phases.append(int(info.get("phase", 1)))
    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()
        env.close()

    return {
        "style": style,
        "episodes": len(returns),
        "seed": seed,
        "returns": returns,
        "phases": phases,
        "frames": frames,
        "human": human,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visually run a trained arena agent, or play the arena yourself."
    )
    parser.add_argument("--style", choices=(*STYLES, "both"), default="direct")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--human", action="store_true", help="play from the keyboard instead")
    parser.add_argument("--algo", choices=("ppo", "dqn"), default="ppo")
    parser.add_argument("--no-window", action="store_true",
                        help="skip the window and print the summary only")
    parser.add_argument("--no-save", action="store_true",
                        help=f"do not write the evaluation numbers to {_reportable(RESULTS_DIR)}")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="override where the evaluation numbers are written")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    styles = list(STYLES) if args.style == "both" else [args.style]

    if args.human:
        if args.style == "both":
            print("--human plays one style at a time; pick --style direct or rotation",
                  file=sys.stderr)
            return 2
        controls = ("WASD / arrows to steer, SPACE to shoot" if args.style == "rotation"
                    else "WASD / arrows to move, SPACE to shoot")
        print(f"human play, style {args.style}: {controls}. O toggles observation, V policy, E effects, ESC quits.")
        play(args.style, episodes=args.episodes, seed=args.seed, human=True)
        return 0

    measured: list[dict[str, Any]] = []
    try:
        for style in styles:
            if not args.no_window:
                play(style, episodes=args.episodes, seed=args.seed, algo=args.algo)
            stats = evaluate(style, episodes=args.episodes, seed=args.seed, algo=args.algo)
            measured.append(stats)
            print()
            print(format_summary(stats))
    except ArenaEvalError as error:
        print(str(error), file=sys.stderr)
        return 1

    if measured and not args.no_save:
        written = write_results(measured, args.results_dir)
        print()
        for name, path in written.items():
            print(f"wrote {name:<12} {_reportable(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
