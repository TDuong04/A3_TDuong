"""Visual playback of a trained deep RL agent, and a human-play mode for the same arena.

    python -m eval.play_arena --style rotation
    python -m eval.play_arena --style direct --episodes 5
    python -m eval.play_arena --style direct --human
    python -m eval.play_arena --style direct --random    # before anything has been trained
    python -m eval.play_arena --style both --episodes 5 --no-window   # the comparison table

Rubric row I4 asks for an evaluation script that can visually run each trained agent, so this works
for both styles from `models/`. Actions are taken with `deterministic=True`: a stochastic policy on
camera looks like a broken one, and the rubric asks for the deterministic policy anyway.

This is what the video records for Part II. It has to show enemies spawning and moving, projectiles
and collisions, and at least one phase progression -- so the summary below reports how often a phase
was actually cleared, which is the thing to check *before* recording rather than after.

With `models/` empty the script runs a random policy instead of refusing to start, so it is
demonstrable before the first training run finishes -- and `--random` asks for that deliberately,
as the baseline the trained agents are measured against. A random run is labelled as one everywhere
it is written down, and never lands in the files the report's trained tables live in: a table that
looks like a result and is not is worse than no table.

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

from arena.constants import N_ACTIONS, DirectAction, RotationAction
from arena.env import ArenaEnv
from arena.debug import ArenaDebugSnapshot
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


class RandomPolicy:
    """A uniform random policy wearing SB3's `predict` signature.

    Two jobs. It makes this script runnable before any model exists, which is what stops the whole
    team queueing behind one training run to find out whether the playback loop works. And it is
    the baseline every trained number is quoted against: "clears phase 1 in 4 of 5 episodes" means
    nothing until someone knows what chance alone manages.

    Seeded, so a random run reproduces exactly like a trained one -- a baseline nobody can re-derive
    is not a baseline. `deterministic` is accepted and ignored: there is nothing here to be
    deterministic about, and rejecting the argument would break the interchangeability that is the
    entire point of matching the signature.
    """

    #: Recorded in every summary, filename and table this policy produces.
    name = "random"

    def __init__(self, n_actions: int, seed: int = 0) -> None:
        self.n_actions = int(n_actions)
        self._rng = np.random.default_rng(seed)

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.int64, None]:
        """An action index drawn uniformly, in the `(action, state)` shape SB3 returns."""
        return np.int64(self._rng.integers(self.n_actions)), None


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


def has_trained_models(models_dir: Path | None = None) -> bool:
    """Whether anything at all has been trained yet, in any style or algorithm.

    Top-level `*.zip` only: `models/checkpoints/` fills up during a run that has not finished and
    produced a final model yet, and treating a mid-run checkpoint as "trained" would send someone
    to a policy that is still moving.
    """
    directory = Path(models_dir) if models_dir is not None else MODELS_DIR
    return directory.exists() and any(directory.glob("*.zip"))


def load_policy(
    style: str,
    algo: str = "ppo",
    models_dir: Path | None = None,
    *,
    random: bool = False,
    seed: int = 0,
) -> tuple[Any, str]:
    """The policy to run, and the name to record it under.

    Two behaviours that read as contradictory and are not. With `models/` empty -- nothing trained
    yet, in any style -- this falls back to a random policy so the script is demonstrable before
    the first training run lands. With `models/` populated but *this* style missing, it raises:
    that is a typo or a half-finished run, and substituting random actions there would quietly
    produce a table indistinguishable from a trained result. Neither path ever trains on demand;
    both name the command that would.
    """
    if random:
        return RandomPolicy(N_ACTIONS[style], seed=seed), RandomPolicy.name
    if not has_trained_models(models_dir):
        directory = Path(models_dir) if models_dir is not None else MODELS_DIR
        print(
            f"No trained models in {_reportable(directory)} -- running a random policy.\n"
            f"Train one with:\n    python -m train.train_arena --style {style}",
            file=sys.stderr,
        )
        return RandomPolicy(N_ACTIONS[style], seed=seed), RandomPolicy.name
    return load_agent(style, algo, models_dir), algo


# --- headless evaluation --------------------------------------------------------------------------


def evaluate(
    style: str,
    *,
    episodes: int = 5,
    seed: int = 0,
    algo: str = "ppo",
    models_dir: Path | None = None,
    agent: Any | None = None,
    random_policy: bool = False,
) -> dict[str, Any]:
    """Run `episodes` seeded episodes with no window and return the numbers the report needs.

    Seeded from a fixed base so both control styles meet the same sequence of arenas. Report row R6
    compares the two, and a comparison across different layouts compares the layouts as much as the
    agents.
    """
    if style not in STYLES:
        raise ValueError(f"unknown control style {style!r}; expected one of {STYLES}")
    if agent is None:
        agent, policy = load_policy(style, algo, models_dir, random=random_policy, seed=seed)
    else:
        # An injected agent names itself if it can. That is how a `RandomPolicy` handed in
        # directly still labels its own output, rather than being filed under `ppo`.
        policy = str(getattr(agent, "name", algo))

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
        # Carried into every filename, summary and table below: a random baseline that reads as a
        # trained result is the one way this script can actively mislead the report.
        "policy": policy,
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
            f"policy             {stats.get('policy', 'ppo')}",
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
        path = directory / f"{_stem(entry)}.json"
        path.write_text(json.dumps({**entry, "generated": generated}, indent=2) + "\n")
        written[entry["style"]] = path

    # A random run never writes over the trained tables. `comparison.md` is what report row R6
    # cites, and one absent-minded `--random` overwriting it would put chance-level numbers under
    # a heading that claims a trained policy produced them.
    random_run = any(entry.get("policy") == RandomPolicy.name for entry in stats)
    table = directory / ("comparison_random.md" if random_run else "comparison.md")
    table.write_text(format_comparison_markdown(stats, generated))
    written["comparison"] = table
    return written


def _stem(entry: dict[str, Any]) -> str:
    """Filename stem for one style's numbers, keeping random runs off the trained filenames."""
    prefix = "eval_random" if entry.get("policy") == RandomPolicy.name else "eval"
    return f"{prefix}_{entry['style']}_seed{entry['seed']}"


def format_comparison_markdown(stats: list[dict[str, Any]], generated: str) -> str:
    """The control-scheme comparison as a markdown table, in the shape report row R6 wants.

    Ranked by nothing: with two control styles the interesting thing is the pair, not a winner, and
    sorting a two-row table implies a verdict the seeds may not support.
    """
    random_run = any(entry.get("policy") == RandomPolicy.name for entry in stats)
    if random_run:
        # Said before the table rather than after it: a heading is what gets screenshotted.
        policy_line = (
            "**Random-policy baseline, not a trained result.** Actions are drawn uniformly from "
            "the action space. These are the numbers chance alone produces on the same seeded "
            "arenas, which is what makes the trained table above them mean anything.\n\n"
        )
        command = "--style both --no-window --random"
    else:
        policy_line = (
            f"Deterministic policy (`deterministic={DETERMINISTIC}`), both styles evaluated on the "
            f"same seed sequence so they meet the same arenas.\n\n"
        )
        command = "--style both --no-window"
    header = (
        f"# Arena control-scheme comparison\n\n"
        f"Generated {generated} by `python -m eval.play_arena {command}`.\n\n"
        f"{policy_line}"
    )
    columns = (
        "| Style | Policy | Episodes | Return | Phase reached | Phases cleared | Spawners "
        "| Enemies | Survival | Steps |\n"
        "|-------|--------|---------:|-------:|--------------:|---------------:|---------:"
        "|--------:|---------:|------:|\n"
    )
    rows = "".join(
        f"| `{entry['style']}` | `{entry.get('policy', 'ppo')}` | {entry['episodes']} "
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


class ArenaPlayback:
    """Step controller shared by window playback and offscreen debugging tests.

    All policy inference happens once, immediately before the transition. Repeated
    draws and pauses reuse its immutable snapshot, including on the terminal frame.
    """

    def __init__(self, env: ArenaEnv, renderer: Any, agent: Any | None,
                 policy: str, algo: str = "ppo") -> None:
        self.env, self.renderer, self.agent = env, renderer, agent
        self.policy, self.algo = policy, algo
        self.snapshot: ArenaDebugSnapshot | None = None
        self.obs = env.last_observation.copy()
        self.done = False

    @property
    def policy_status(self) -> str:
        if self.policy == "human":
            return "HUMAN: no model controls actions"
        if self.policy == "random":
            return "RANDOM: uniform actions, no model"
        return f"{self.algo.upper()} model loaded; awaiting step"

    def reset(self, seed: int | None = None) -> None:
        self.obs, _ = self.env.reset(seed=seed)
        self.snapshot = None
        self.done = False
        self.renderer.step_requests = 0
        self.renderer.reset_requested = False
        self.renderer.reset_episode()

    def advance(self, human_action_source=None) -> bool:
        if self.renderer.reset_requested:
            self.reset()
            return False
        if self.done or (self.renderer.paused and not self.renderer.step_requests):
            return False
        if self.renderer.step_requests:
            self.renderer.step_requests -= 1
        before = self.obs.copy()
        view = None
        status = self.policy_status
        if self.policy == "human":
            action = 0 if human_action_source is None else int(human_action_source())
        else:
            predicted, _ = self.agent.predict(before, deterministic=DETERMINISTIC)
            action = int(predicted)
            if self.policy != "random":
                view = probe(self.agent, before, self.env.control_style, action, algo=self.algo)
                status = (f"{self.algo.upper()} model: policy from pre-step s" if view is not None
                          else f"{self.algo.upper()} model: diagnostics unavailable")
        self.obs, _, terminated, truncated, _ = self.env.step(action)
        self.done = terminated or truncated
        self.snapshot = ArenaDebugSnapshot(
            tuple(float(x) for x in before), tuple(float(x) for x in self.obs),
            view, status, self.env.latest_reward,
        )
        self.renderer.observe(self.env)
        return True

    def draw(self):
        view = None if self.snapshot is None else self.snapshot.policy
        return self.renderer.draw(
            self.env, view, debug_snapshot=self.snapshot,
            policy_status=self.policy_status if self.snapshot is None else self.snapshot.policy_status,
        )


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
    random_policy: bool = False,
    debug: bool = False,
    paused: bool = False,
) -> dict[str, Any]:
    """Run the env in a window, driven by the agent or by the keyboard.

    Returns the same summary shape `evaluate` does, so a recorded session and a headless run are
    directly comparable.
    """
    import pygame

    from arena.render import ArenaRenderer

    if human:
        agent, policy = None, "human"
    else:
        agent, policy = load_policy(style, algo, models_dir, random=random_policy, seed=seed)
    env = ArenaEnv(control_style=style, render_mode="human")
    renderer = ArenaRenderer(headless=headless)

    renderer.show_debug = debug
    renderer.paused = paused
    app = ArenaPlayback(env, renderer, agent, policy, algo)
    returns: list[float] = []
    phases: list[int] = []
    frames = 0
    try:
        for episode in range(episodes):
            app.reset(seed + episode)
            while True:
                app.draw()
                frames += 1
                if renderer.should_close or (max_frames is not None and frames >= max_frames):
                    raise KeyboardInterrupt
                # A paused ending remains visible. N never steps an ended environment;
                # R resets explicitly, or P resumes normal episode progression.
                if app.done and not renderer.paused and not renderer.reset_requested:
                    break
                app.advance(None if headless and human else
                            lambda: human_action(pygame.key.get_pressed(), style))
            returns.append(float(env.episode_reward))
            phases.append(env.phase)
    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()
        env.close()

    return {
        "style": style, "policy": policy, "episodes": len(returns), "seed": seed,
        "returns": returns, "phases": phases, "frames": frames, "human": human,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visually run a trained arena agent, or play the arena yourself."
    )
    parser.add_argument("--style", choices=(*STYLES, "both"), default="direct")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--human", action="store_true", help="play from the keyboard instead")
    parser.add_argument("--random", action="store_true",
                        help="run a uniform random policy: the baseline, and what runs anyway "
                             "when models/ is empty")
    parser.add_argument("--algo", choices=("ppo", "dqn"), default="ppo")
    parser.add_argument("--debug", action="store_true", help="start reward/physics debug on (F3)")
    parser.add_argument("--paused", action="store_true", help="start paused; N steps, P resumes")
    parser.add_argument("--headless", action="store_true", help="render offscreen without a window")
    parser.add_argument("--frames", type=int, default=None, help="limit playback frames for smoke tests")
    parser.add_argument("--models-dir", type=Path, default=None)
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
        print(f"human play, style {args.style}: {controls}. O toggles observation, V policy, E effects, F3 debug, P pause, N step, R reset, ESC quits.")
        play(args.style, episodes=args.episodes, seed=args.seed, human=True,
             debug=args.debug, paused=args.paused, headless=args.headless, max_frames=args.frames)
        return 0

    measured: list[dict[str, Any]] = []
    try:
        for style in styles:
            if not args.no_window:
                play(style, episodes=args.episodes, seed=args.seed, algo=args.algo,
                     random_policy=args.random, models_dir=args.models_dir,
                     debug=args.debug, paused=args.paused,
                     headless=args.headless, max_frames=args.frames)
            stats = evaluate(style, episodes=args.episodes, seed=args.seed, algo=args.algo,
                             random_policy=args.random, models_dir=args.models_dir)
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
