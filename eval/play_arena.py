"""Visual playback of a trained deep RL agent, and a human-play mode for the same arena.

    python -m eval.play_arena --style rotation
    python -m eval.play_arena --style direct --episodes 5
    python -m eval.play_arena --style direct --human
    python -m eval.play_arena --style direct --random    # before anything has been trained
    python -m eval.play_arena --style both --no-window   # the comparison table (30 episodes/style)

Rubric row I4 asks for an evaluation script that can visually run each trained agent, so this works
for both styles from `models/`. Actions are taken with `deterministic=True`: a stochastic policy on
camera looks like a broken one, and the rubric asks for the deterministic policy anyway.

The comparison table's default episode count is 30, not a quick-look number: `--episodes` used to
default to 3, which is what the command above actually ran as long as nobody remembered to type an
override that appeared only in this docstring and disagreed with it (5) -- see A3-032. The default
now matches the sample size the committed evidence in `results/arena_eval/` is measured at, so
running the command exactly as documented cannot silently under-sample it again.

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

In `--human` play, an episode's end (death or survival) freezes on that last frame — meme easter
egg included — until R is pressed to retry; an agent demo keeps its fixed `DEATH_HOLD_SECONDS`
beat instead, since it has to run unattended for recording.

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


def resolve_model(style: str, algo: str = "ppo", models_dir: Path | None = None,
                  *, mechanics: bool = False) -> Path:
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
        + (" --mechanics" if mechanics else "")
    )


def load_agent(style: str, algo: str = "ppo", models_dir: Path | None = None,
               *, mechanics: bool = False):
    """Load the SB3 policy. Imported lazily so `--human` needs no torch and no model on disk."""
    from stable_baselines3 import DQN, PPO

    path = resolve_model(style, algo, models_dir, mechanics=mechanics)
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
    mechanics: bool = False,
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
            f"Train one with:\n    python -m train.train_arena --style {style}"
            + (" --mechanics" if mechanics else ""),
            file=sys.stderr,
        )
        return RandomPolicy(N_ACTIONS[style], seed=seed), RandomPolicy.name
    if mechanics:
        return load_agent(style, algo, models_dir, mechanics=True), algo
    return load_agent(style, algo, models_dir), algo


# --- headless evaluation --------------------------------------------------------------------------


def validate_policy_space(agent, env):
    """Fail early instead of feeding an incompatible checkpoint a different feature vector."""
    observation_space = getattr(agent, "observation_space", None)
    action_space = getattr(agent, "action_space", None)
    if ((observation_space is not None and observation_space.shape != env.observation_space.shape)
            or (action_space is not None and action_space.n != env.action_space.n)):
        env.close()
        raise ArenaEvalError("Model spaces do not match this arena. Train with the same "
                             "control style and --mechanics setting used for playback.")



def evaluate(
    style: str,
    *,
    episodes: int = 5,
    seed: int = 0,
    algo: str = "ppo",
    models_dir: Path | None = None,
    agent: Any | None = None,
    random_policy: bool = False,
    mechanics: bool = False,
) -> dict[str, Any]:
    """Run `episodes` seeded episodes with no window and return the numbers the report needs.

    Seeded from a fixed base so both control styles meet the same sequence of arenas. Report row R6
    compares the two, and a comparison across different layouts compares the layouts as much as the
    agents.
    """
    if style not in STYLES:
        raise ValueError(f"unknown control style {style!r}; expected one of {STYLES}")
    if mechanics:
        models_dir = (Path(models_dir) if models_dir is not None else MODELS_DIR) / "mechanics"
    if agent is None:
        agent, policy = load_policy(style, algo, models_dir, random=random_policy, seed=seed, mechanics=mechanics)
    else:
        # An injected agent names itself if it can. That is how a `RandomPolicy` handed in
        # directly still labels its own output, rather than being filed under `ppo`.
        policy = str(getattr(agent, "name", algo))

    env = ArenaEnv(control_style=style, render_mode=None, mechanics=mechanics)
    validate_policy_space(agent, env)
    returns: list[float] = []
    phases: list[int] = []
    spawners: list[int] = []
    kills: list[int] = []
    fired: list[int] = []
    hit: list[int] = []
    steps: list[int] = []
    survived: list[bool] = []
    mechanic_metrics = {key: [] for key in ("pickups_collected", "shield_blocks", "elites_killed")}

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
        fired.append(int(info.get("bullets_fired", 0)))
        hit.append(int(info.get("bullets_hit", 0)))
        steps.append(int(env.steps))
        survived.append(not terminated)
        for key, values in mechanic_metrics.items():
            values.append(info.get(key, 0))
    env.close()

    def mean(values: list[Any]) -> float:
        return float(statistics.fmean(values)) if values else 0.0

    return {
        "mechanics": mechanics,
        "style": style,
        # Carried into every filename, summary and table below: a random baseline that reads as a
        # trained result is the one way this script can actively mislead the report.
        "policy": policy,
        "episodes": episodes,
        "seed": seed,
        **{key + "_mean": mean(values) for key, values in mechanic_metrics.items()},
        "return_mean": mean(returns),
        "return_std": float(statistics.pstdev(returns)) if len(returns) > 1 else 0.0,
        "phase_mean": mean(phases),
        "phase_max": max(phases) if phases else 0,
        # The number the video depends on: a phase progression has to be visible at least once.
        "phase_cleared_episodes": sum(1 for p in phases if p > 1),
        "spawners_mean": mean(spawners),
        "enemies_mean": mean(kills),
        # Aim quality: bullets landed on an enemy or a spawner, over bullets fired, across the
        # whole sample -- not an average of per-episode ratios, so an episode with few shots
        # cannot swing the number as much as one with many.
        "bullets_fired_mean": mean(fired),
        "bullets_hit_mean": mean(hit),
        "hit_rate": (sum(hit) / sum(fired)) if sum(fired) > 0 else 0.0,
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
            f"rules              {'shield + elite' if stats.get('mechanics') else 'baseline'}",
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


def _load_existing_entry(directory: Path, style: str, random_run: bool) -> dict[str, Any] | None:
    """The most recently written per-style stats for `style` already on disk, or `None`.

    `write_results` used to rebuild `comparison.md` from nothing but the entries it was just
    given, so re-running one style alone (say, spot-checking `rotation` after a retrain) silently
    deleted the other style's row -- even though its JSON was sitting untouched right next to it
    (A3-032). Reading that JSON back in is what lets a single-style rerun update its own row
    without erasing its neighbour's.
    """
    prefix = "eval_random" if random_run else "eval"
    candidates = sorted(directory.glob(f"{prefix}_{style}_seed*.json"))
    if not candidates:
        return None
    try:
        return json.loads(candidates[-1].read_text())
    except (OSError, ValueError):
        return None


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
    modes = {entry.get("mechanics", False) for entry in stats}
    if len(modes) > 1:
        raise ValueError("Compare control styles under the same mechanics setting")
    if True in modes:
        directory = directory / "mechanics"
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

    # Fill in any style this call did not itself measure from what is already on disk, so the
    # table always reflects every style that has ever been evaluated here, not just the last
    # command's --style argument.
    written_styles = {entry["style"] for entry in stats}
    table_entries = list(stats)
    carried_over: set[str] = set()
    for style in STYLES:
        if style in written_styles:
            continue
        existing = _load_existing_entry(directory, style, random_run)
        if existing is not None:
            table_entries.append(existing)
            carried_over.add(style)
    table_entries.sort(key=lambda entry: STYLES.index(entry["style"])
                       if entry["style"] in STYLES else len(STYLES))

    table.write_text(format_comparison_markdown(table_entries, generated, carried_over=carried_over))
    written["comparison"] = table
    return written


def _stem(entry: dict[str, Any]) -> str:
    """Filename stem for one style's numbers, keeping random runs off the trained filenames."""
    prefix = "eval_random" if entry.get("policy") == RandomPolicy.name else "eval"
    return f"{prefix}_{entry['style']}_seed{entry['seed']}"


def format_comparison_markdown(
    stats: list[dict[str, Any]], generated: str, *, carried_over: set[str] | None = None
) -> str:
    """The control-scheme comparison as a markdown table, in the shape report row R6 wants.

    Ranked by nothing: with two control styles the interesting thing is the pair, not a winner, and
    sorting a two-row table implies a verdict the seeds may not support.

    `carried_over` names styles in `stats` this call did not itself measure -- rows `write_results`
    filled in from an earlier run's JSON (A3-032) rather than silently dropping. The header states
    the episode count and command for the rows this call *did* measure; a carried-over row keeps
    whatever count it was actually measured with, visible in its own `Episodes` column, and is
    named explicitly rather than folded into a command line that did not produce it.
    """
    carried_over = carried_over or set()
    fresh = [entry for entry in stats if entry["style"] not in carried_over] or stats
    random_run = any(entry.get("policy") == RandomPolicy.name for entry in stats)
    episode_counts = sorted({entry["episodes"] for entry in fresh})
    # Named explicitly rather than left to the reader to notice in the table: the failure this
    # guards against (A3-032) is exactly a sample size nobody would think to check for.
    episodes_flag = f" --episodes {episode_counts[0]}" if len(episode_counts) == 1 else ""
    styles_flag = "both" if len(fresh) != 1 else fresh[0]["style"]
    if random_run:
        # Said before the table rather than after it: a heading is what gets screenshotted.
        policy_line = (
            "**Random-policy baseline, not a trained result.** Actions are drawn uniformly from "
            "the action space. These are the numbers chance alone produces on the same seeded "
            "arenas, which is what makes the trained table above them mean anything.\n\n"
        )
        command = f"--style {styles_flag} --no-window --random{episodes_flag}"
    else:
        policy_line = (
            f"Deterministic policy (`deterministic={DETERMINISTIC}`), both styles evaluated on the "
            f"same seed sequence so they meet the same arenas.\n\n"
        )
        command = f"--style {styles_flag} --no-window{episodes_flag}"
    if any(entry.get("mechanics") for entry in stats):
        command += " --mechanics"
        policy_line += "Rules: shield pickups and elite chargers.\n\n"
    carried_note = ""
    if carried_over:
        names = " and ".join(f"`{style}`" for style in sorted(carried_over))
        row_word = "row" if len(carried_over) == 1 else "rows"
        carried_note = (
            f"The {names} {row_word} below {'is' if len(carried_over) == 1 else 'are'} carried "
            f"over from an earlier run, not remeasured by the command above -- see its own "
            f"`Episodes` column for the sample size it was actually measured at.\n\n"
        )
    header = (
        f"# Arena control-scheme comparison\n\n"
        f"Generated {generated} by `python -m eval.play_arena {command}`.\n\n"
        f"{policy_line}"
        f"{carried_note}"
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


def wait_for_start(renderer, style: str, mechanics: bool = False) -> bool:
    """Show the title screen and block until SPACE/ENTER is pressed. Returns False on quit.

    This file already owns every other keypress (`human_action` above), so the title screen stays
    consistent with that: `arena/render.py` only draws, `play()` decides what a key means. The
    caller is responsible for only reaching this in a real windowed session — see `play()`, which
    skips it whenever the caller asked for headless rendering or a fixed frame count, so no
    automated evaluation or figure-capture run can hang waiting for a key nobody is there to press.
    """
    import pygame

    while True:
        renderer.draw_title(style, mechanics=mechanics)
        if renderer.should_close:
            return False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or keys[pygame.K_RETURN]:
            return True


def wait_for_retry(renderer, env, view) -> bool:
    """Freeze on the finished episode -- meme, banner and all -- until R (retry) is pressed or the
    window closes. Returns False to quit, mirroring `wait_for_start`.

    Human play only. `DEATH_HOLD_SECONDS`/`renderer.hold()` is a fixed beat meant for an agent demo
    that has to keep running unattended for recording; a human who just died gets to actually read
    the meme caption instead of it flashing by in a second. `env` and `view` are the finished
    episode's own state, unmodified -- this only keeps redrawing the same last frame with the retry
    prompt added. Reached only from a real windowed human session; see `play()`, which gates it the
    same way it gates `wait_for_start`, so no automated or frame-capped run can hang waiting for a
    key nobody is there to press.
    """
    import pygame

    while True:
        renderer.draw(env, view, show_retry_prompt=True)
        if renderer.should_close:
            return False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]:
            return True


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
    random_policy: bool = False,
    mechanics: bool = False,
) -> dict[str, Any]:
    """Run the env in a window, driven by the agent or by the keyboard.

    Returns the same summary shape `evaluate` does, so a recorded session and a headless run are
    directly comparable.
    """
    import pygame

    from arena.render import ArenaRenderer

    if mechanics:
        models_dir = (Path(models_dir) if models_dir is not None else MODELS_DIR) / "mechanics"
    if human:
        agent, policy = None, "human"
    else:
        agent, policy = load_policy(style, algo, models_dir, random=random_policy, seed=seed, mechanics=mechanics)
    env = ArenaEnv(control_style=style, render_mode="human", mechanics=mechanics)
    validate_policy_space(agent, env)
    renderer = ArenaRenderer(headless=headless)

    returns: list[float] = []
    phases: list[int] = []
    frames = 0
    try:
        # Only a real, unlimited windowed session gets a title screen: headless runs (the row-I
        # evidence table, every test in this project) and frame-capped runs (report figures) must
        # never wait on a key nobody is there to press, so both skip this by construction rather
        # than by an extra flag someone has to remember to pass.
        if not headless and max_frames is None:
            if not wait_for_start(renderer, style, mechanics=mechanics):
                raise KeyboardInterrupt
        # A real human session replays on R rather than stopping at `episodes`: an arcade "insert
        # coin" loop reads better than a scripted rep count nobody asked for. Everything else --
        # agent playback, headless runs, frame-capped report figures -- keeps the fixed-count
        # behaviour those callers actually rely on.
        interactive = human and not headless and max_frames is None
        episode = 0
        while interactive or episode < episodes:
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
                    # A random policy has no network to read, so the panel is left off rather
                    # than drawn empty. `probe` would swallow the AttributeError and return None
                    # anyway; asking it once per frame to fail is not how to express that.
                    view = (
                        None if policy == RandomPolicy.name
                        else probe(agent, obs, style, action, algo=algo)
                    )

                renderer.draw(env, view)
                frames += 1
                if renderer.should_close:
                    raise KeyboardInterrupt
                if max_frames is not None and frames >= max_frames:
                    raise KeyboardInterrupt

                if human:
                    action = 0 if headless else human_action(pygame.key.get_pressed(), style)

                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            # The final frame: the death or the clock running out. Held for a beat so the death
            # explosion actually plays; `max_frames` runs are capturing a fixed number of frames
            # and are left alone. The view is the one that produced the last action, so the policy
            # panel stays up through the hold instead of blinking off for the closing shot.
            renderer.draw(env, view)
            if max_frames is None:
                renderer.hold(env, view)
            returns.append(float(env.episode_reward))
            phases.append(int(info.get("phase", 1)))
            episode += 1
            if interactive and not wait_for_retry(renderer, env, view):
                break
    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()
        env.close()

    return {
        "mechanics": mechanics,
        "style": style,
        "policy": policy,
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
    #: Matches the sample size `results/arena_eval/comparison.md` is measured at (A3-032): the
    #: command this script's own docstring and README.md document for regenerating that evidence
    #: passes no --episodes override, so the default has to already be the number that belongs there.
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mechanics", action="store_true",
                        help="enable shields and elites; use models/mechanics")
    parser.add_argument("--headless", action="store_true", help="render offscreen")
    parser.add_argument("--frames", type=int, default=None, help="limit playback frames")
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--human", action="store_true", help="play from the keyboard instead")
    parser.add_argument("--random", action="store_true",
                        help="run a uniform random policy: the baseline, and what runs anyway "
                             "when models/ is empty")
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
        play(args.style, episodes=args.episodes, seed=args.seed, human=True,
             mechanics=args.mechanics, headless=args.headless, max_frames=args.frames)
        return 0

    measured: list[dict[str, Any]] = []
    try:
        for style in styles:
            if not args.no_window:
                play(style, episodes=args.episodes, seed=args.seed, algo=args.algo,
                     headless=args.headless, max_frames=args.frames,
                     random_policy=args.random, models_dir=args.models_dir, mechanics=args.mechanics)
            stats = evaluate(style, episodes=args.episodes, seed=args.seed, algo=args.algo,
                             random_policy=args.random, models_dir=args.models_dir, mechanics=args.mechanics)
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
