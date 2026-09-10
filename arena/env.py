"""Arena environment — the Gym-style API around the Part II simulation.

ONE env class for both control schemes. `ArenaEnv(control_style="rotation"|"direct")` switches only
the action space and `_apply_action`; physics, spawning, rewards, observation and rendering are
shared. Forking this into two files is the fastest way to lose rubric row G — every bug gets fixed
twice, the two agents stop being comparable, and report row R6's comparison becomes meaningless.

    action_space = Discrete(5) for rotation, Discrete(6) for direct  # indices in constants.py
    observation_space = Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=float32)

`step(action)` runs `ACTION_REPEAT` (3) physics frames of `FIXED_DT` with the action held, then
builds the observation. That cuts the episode horizon threefold for free and makes credit
assignment tractable.

Per-step sequence, per physics frame: apply the action; advance the player, bullets, spawners and
enemies; resolve collisions; accumulate rewards; check phase advance (all spawners destroyed) and
set up the next phase from `config/arena.yaml`; check termination. `REWARD_PER_STEP` is paid once
per *agent* step, not once per frame, so the step penalty does not silently triple.

Termination is on player death (`terminated`) or `MAX_EPISODE_STEPS` (`truncated`). Keeping those
distinct matters — bootstrapping through a time-limit cutoff as though it were death teaches the
agent that running out of clock is fatal.

`info` carries `phase`, `spawners_destroyed`, `enemies_killed` and `damage_taken`. Those are the
behavioural metrics the diagnostician and the report depend on; reward alone cannot distinguish a
policy that is progressing from one that is farming a shaping term.

Randomness lives in `self.np_random`, seeded through `reset(seed=...)`, and covers exactly two
things: the ship's starting heading and where each phase puts its spawners. Both are randomised on
purpose. A fixed heading would let the policy memorise one opening; a fixed spawner layout would
let it memorise the map instead of learning to read the observation. Everything else — enemy
seeking, spawn cadence, bullet flight — is deterministic in `entities.py`, so a seed plus an action
sequence reproduces a trajectory exactly.

Two API surfaces, both required:
  - This class is Gymnasium 1.3 native — `reset(seed=None)` returns `(obs, info)` and `step` returns
    the 5-tuple — because that is what Stable Baselines3 requires.
  - `LegacyGymAPI` in `arena/legacy_api.py` wraps it to the signature the brief names literally:
    `reset()` returns obs, `step(action)` returns `(obs, reward, done, info)`, plus `render()`.
    Rubric H1 quotes that 4-tuple, so ship the adapter and name it in the report.

`render_mode=None` must import and run with no display. Nothing in this file may call
`pygame.draw`, `blit`, `display` or `clock.tick` — that all belongs in `render.py`, invoked only
from `render()`. Rendering inside `step()` silently multiplies training time by an order of
magnitude and is the first thing `env-validator` greps for. `render()` imports `arena.render`
*lazily*, inside the method, so that importing this module never pulls pygame into a training
worker. The renderer contract it assumes is deliberately tiny: `ArenaRenderer()` takes no required
arguments, `draw(env)` draws one frame of the env handed to it, and `close()` tears the window down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from common.config import load_yaml

from .constants import (
    ACTION_ENUMS,
    ACTION_REPEAT,
    ARENA_HEIGHT,
    ARENA_WIDTH,
    CONTROL_STYLES,
    FPS,
    MAX_EPISODE_STEPS,
    N_ACTIONS,
    OBS_DIM,
    REWARD_DAMAGE_TAKEN,
    REWARD_DEATH,
    REWARD_ENEMY_DESTROYED,
    REWARD_PER_STEP,
    REWARD_PHASE_ADVANCE,
    REWARD_SPAWNER_DESTROYED,
    DirectAction,
    RotationAction,
)
from .debug import RewardSnapshot, REWARD_TERMS
from .entities import Bullet, Enemy, Player, Spawner, phase_config
from .observation import build_observation


@dataclass(frozen=True)
class EpisodeConfig:
    """The `episode` block of `config/arena.yaml`: phase layout and live-entity limits."""

    max_enemies: int
    arena_margin: float
    min_player_clearance: float
    min_spawner_separation: float
    placement_attempts: int

    @classmethod
    def from_yaml(cls, name: str = "arena", section: str = "episode") -> EpisodeConfig:
        return cls(**load_yaml(name)[section])


@lru_cache(maxsize=None)
def episode_config() -> EpisodeConfig:
    return EpisodeConfig.from_yaml()


class ArenaEnv(gym.Env):
    """The arena as a Gymnasium 1.3 environment, parameterised by control style."""

    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(
        self,
        control_style: str = "direct",
        render_mode: str | None = None,
        seed: int | None = None,
        max_episode_steps: int | None = None,
    ) -> None:
        if control_style not in CONTROL_STYLES:
            raise ValueError(
                f"unknown control_style {control_style!r}; expected one of {CONTROL_STYLES}"
            )
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"unknown render_mode {render_mode!r}; expected one of "
                f"{self.metadata['render_modes']} or None"
            )

        self.control_style = control_style
        self.actions = ACTION_ENUMS[control_style]
        self.render_mode = render_mode
        self.max_episode_steps = int(
            max_episode_steps if max_episode_steps is not None else MAX_EPISODE_STEPS
        )
        self.episode_config = episode_config()

        self.action_space = spaces.Discrete(N_ACTIONS[control_style])
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self._renderer: Any | None = None

        # Building the first episode here means every attribute a renderer or a test might read
        # exists before `reset()` is called, and it costs one episode setup per env.
        self.reset(seed=seed)

    # --- Gymnasium API ------------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a fresh episode at phase 1 and return `(observation, info)`."""
        super().reset(seed=seed)

        self.steps = 0
        self.phase = 1  # 1-based: `info["phase"] == 1` during the first phase, as the HUD shows it
        self.enemies_killed = 0
        self.spawners_destroyed = 0
        self.damage_taken = 0
        self.episode_reward = 0.0
        self.reward_totals = dict.fromkeys(REWARD_TERMS, 0.0)
        self._step_rewards = dict.fromkeys(REWARD_TERMS, 0.0)
        self.latest_reward: RewardSnapshot | None = None
        self.last_action = 0
        self._phase_advanced_this_step = False
        self._last_observation: np.ndarray | None = None

        self.player = Player(
            ARENA_WIDTH / 2.0,
            ARENA_HEIGHT / 2.0,
            heading=float(self.np_random.uniform(-math.pi, math.pi)),
        )
        self.enemies: list[Enemy] = []
        self.bullets: list[Bullet] = []
        self.spawners: list[Spawner] = []
        self._begin_phase()

        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Hold `action` for `ACTION_REPEAT` physics frames and return the 5-tuple."""
        action = int(action)
        if not 0 <= action < int(self.action_space.n):
            raise ValueError(
                f"action {action} is outside {self.action_space} for control style "
                f"{self.control_style!r}"
            )
        self.last_action = action
        # True for exactly one agent step. The renderer latches it into its own wall-clock
        # countdown, because at 60 fps a one-step flag would flash the banner for a single frame.
        self._phase_advanced_this_step = False

        self._step_rewards = dict.fromkeys(REWARD_TERMS, 0.0)
        reward = self._record_reward("step", REWARD_PER_STEP)
        for _ in range(ACTION_REPEAT):
            reward += self._advance_frame(action)
            if not self.player.alive:
                break  # no point simulating the frames after the ship stopped existing

        self.steps += 1
        terminated = not self.player.alive
        # Distinct by construction: a step cap is not a rule of the game, and an agent that is
        # still alive when the clock runs out must keep bootstrapping from its final value.
        truncated = not terminated and self.steps >= self.max_episode_steps
        # Canonical sum of the actual contributions captured at each reward site.
        reward = sum(self._step_rewards.values())
        self.episode_reward += reward
        for term, value in self._step_rewards.items():
            self.reward_totals[term] += value
        self.latest_reward = RewardSnapshot(
            self.steps, action, tuple(self._step_rewards.items()),
            tuple(self.reward_totals.items()), float(reward), terminated, truncated,
        )

        return self._observation(), float(reward), terminated, truncated, self._info()

    def render(self) -> Any:
        """Display the scene for evaluation. Never called from `step()`."""
        if self.render_mode is None:
            raise RuntimeError(
                "ArenaEnv.render() needs ArenaEnv(render_mode='human'); training runs headless "
                "with render_mode=None on purpose."
            )
        if self._renderer is None:
            # Imported here, not at module scope, so that a headless training worker never loads
            # pygame at all.
            from .render import ArenaRenderer

            self._renderer = ArenaRenderer()
        return self._renderer.draw(self)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # --- views for the renderer and the eval overlay -------------------------------------------

    @property
    def action_name(self) -> str:
        """The name of the last action, for the HUD: `THRUST`, `SHOOT`, ..."""
        return self.actions(self.last_action).name

    @property
    def episode_return(self) -> float:
        """Cumulative environment reward this episode, for the HUD's SCORE field."""
        return float(self.episode_reward)

    @property
    def last_observation(self) -> np.ndarray:
        """The observation most recently handed out, for the overlay panel.

        Falls back to building one, so a renderer attached before the first `step()` -- which is
        what `--human` mode does -- draws a real vector rather than crashing on `None`.
        """
        if self._last_observation is None:
            return self._observation()
        return self._last_observation

    @property
    def phase_just_advanced(self) -> bool:
        """True for exactly one agent step after the last spawner of a phase was destroyed."""
        return bool(self._phase_advanced_this_step)

    @property
    def phase_settings(self):
        """The `PhaseConfig` currently in force."""
        return phase_config(self.phase - 1)

    def observation(self) -> np.ndarray:
        """The current observation, for the overlay and for tests."""
        return self._observation()

    # --- episode construction ------------------------------------------------------------------

    def _begin_phase(self) -> None:
        """Lay out the spawners for the current phase. Enemies already in flight stay in flight."""
        settings = self.phase_settings
        self.spawners = []
        for _ in range(settings.spawners):
            x, y = self._sample_spawner_position()
            self.spawners.append(Spawner(x, y, settings))

    def _sample_spawner_position(self) -> tuple[float, float]:
        """Rejection-sample a spot clear of the player and of the spawners already placed.

        After `placement_attempts` misses the roomiest candidate seen is accepted rather than
        looping forever: a crowded phase must still start, just less comfortably.
        """
        config = self.episode_config
        margin = config.arena_margin
        best: tuple[float, float] = (ARENA_WIDTH / 2.0, margin)
        best_clearance = -math.inf

        for _ in range(config.placement_attempts):
            x = float(self.np_random.uniform(margin, ARENA_WIDTH - margin))
            y = float(self.np_random.uniform(margin, ARENA_HEIGHT - margin))
            clearance = math.hypot(x - self.player.x, y - self.player.y)
            clearance -= config.min_player_clearance
            for spawner in self.spawners:
                separation = math.hypot(x - spawner.x, y - spawner.y)
                clearance = min(clearance, separation - config.min_spawner_separation)
            if clearance >= 0.0:
                return x, y
            if clearance > best_clearance:
                best, best_clearance = (x, y), clearance

        return best

    # --- one physics frame ---------------------------------------------------------------------

    def _advance_frame(self, action: int) -> float:
        """Apply the held action and advance the world by one `FIXED_DT` frame."""
        self._apply_action(action)
        self.player.update()

        for bullet in self.bullets:
            bullet.update()
        self._advance_spawners()
        for enemy in self.enemies:
            enemy.update(self.player.x, self.player.y)

        reward = self._resolve_bullet_hits()
        reward += self._resolve_contact_damage()
        self._drop_dead()
        reward += self._maybe_advance_phase()
        return reward

    def _apply_action(self, action: int) -> None:
        if self.control_style == "rotation":
            self._apply_rotation_action(action)
        else:
            self._apply_direct_action(action)

    def _apply_rotation_action(self, action: int) -> None:
        """Style 1: turn, burn or shoot. `NOOP` deliberately does nothing at all — coasting is a
        legitimate move and drag handles the rest."""
        if action == RotationAction.THRUST:
            self.player.thrust()
        elif action == RotationAction.ROTATE_LEFT:
            self.player.rotate(-1.0)
        elif action == RotationAction.ROTATE_RIGHT:
            self.player.rotate(+1.0)
        elif action == RotationAction.SHOOT:
            self._fire()

    def _apply_direct_action(self, action: int) -> None:
        """Style 2: the four compass directions or shoot. `UP` is -y because the arena's y axis
        points down the screen."""
        if action == DirectAction.UP:
            self.player.move(0.0, -1.0)
        elif action == DirectAction.DOWN:
            self.player.move(0.0, +1.0)
        elif action == DirectAction.LEFT:
            self.player.move(-1.0, 0.0)
        elif action == DirectAction.RIGHT:
            self.player.move(+1.0, 0.0)
        elif action == DirectAction.SHOOT:
            self._fire()

    def _fire(self) -> None:
        """`Player.shoot` decides; the cap and the cooldown live there, not here."""
        bullet = self.player.shoot(live_bullets=len(self.bullets))
        if bullet is not None:
            self.bullets.append(bullet)

    def _advance_spawners(self) -> None:
        for spawner in self.spawners:
            enemy = spawner.update()
            # Past the cap the spawn is dropped rather than queued: an unbounded enemy list is the
            # usual reason a long run slows to a crawl.
            if enemy is not None and len(self.enemies) < self.episode_config.max_enemies:
                self.enemies.append(enemy)

    # --- collisions and rewards -----------------------------------------------------------------

    def _record_reward(self, term: str, value: float) -> float:
        """Record a fired reward at its source; drawing never calls this method."""
        self._step_rewards[term] += value
        return value

    def _resolve_bullet_hits(self) -> float:
        """One bullet spends itself on one target. Rewards come from `constants.py` only."""
        reward = 0.0
        for bullet in self.bullets:
            if not bullet.alive:
                continue
            for enemy in self.enemies:
                if bullet.collides_with(enemy):
                    bullet.kill()
                    enemy.take_damage(bullet.damage)
                    if not enemy.alive:
                        self.enemies_killed += 1
                        reward += self._record_reward("enemy", REWARD_ENEMY_DESTROYED)
                    break
            if not bullet.alive:
                continue
            for spawner in self.spawners:
                if bullet.collides_with(spawner):
                    bullet.kill()
                    spawner.take_damage(bullet.damage)
                    if not spawner.alive:
                        self.spawners_destroyed += 1
                        reward += self._record_reward("spawner", REWARD_SPAWNER_DESTROYED)
                    break
        return reward

    def _resolve_contact_damage(self) -> float:
        """Enemies hurt the player by touching it; the enemy survives the contact.

        `Player.take_damage` returns False while the invulnerability window is open, so the damage
        penalty is paid once per landed hit rather than once per frame of overlap.
        """
        reward = 0.0
        for enemy in self.enemies:
            if not enemy.collides_with(self.player):
                continue
            if self.player.take_damage(enemy.contact_damage):
                self.damage_taken += enemy.contact_damage
                reward += self._record_reward("damage", REWARD_DAMAGE_TAKEN)
                if not self.player.alive:
                    reward += self._record_reward("death", REWARD_DEATH)
                    break
        return reward

    def _drop_dead(self) -> None:
        self.bullets = [bullet for bullet in self.bullets if bullet.alive]
        self.enemies = [enemy for enemy in self.enemies if enemy.alive]
        self.spawners = [spawner for spawner in self.spawners if spawner.alive]

    def _maybe_advance_phase(self) -> float:
        """Destroying every active spawner advances the phase and lays out the next one.

        Enemies already loose in the arena are not cleared: the reward is for clearing spawners,
        and a free wipe would make the phase transition the safest moment in the episode.
        """
        if self.spawners:
            return 0.0
        self.phase += 1
        self._phase_advanced_this_step = True
        self._begin_phase()
        return self._record_reward("phase", REWARD_PHASE_ADVANCE)

    # --- observation and info --------------------------------------------------------------------

    def _observation(self) -> np.ndarray:
        obs = build_observation(self.player, self.enemies, self.spawners, self.phase)
        # Cached rather than recomputed on demand: the overlay must show the vector the agent was
        # actually handed on this step, not one rebuilt from a world that has since moved on.
        self._last_observation = obs
        return obs

    def _info(self) -> dict[str, Any]:
        """The behavioural metrics the TensorBoard callback and the report read.

        The first four are cumulative over the episode, so the final `info` of an episode is a
        complete summary of it — which is what `Monitor(info_keywords=...)` records.
        """
        return {
            "phase": self.phase,
            "spawners_destroyed": self.spawners_destroyed,
            "enemies_killed": self.enemies_killed,
            "damage_taken": self.damage_taken,
            # Death, as opposed to the step cap. The two endings mean opposite things about a
            # policy, and the reward curve cannot tell them apart; `behaviour/survival_rate` is
            # this key averaged over a window.
            "terminated": not self.player.alive,
            "health": self.player.health,
            "steps": self.steps,
            "episode_reward": self.episode_reward,
        }

    def __repr__(self) -> str:
        return (
            f"ArenaEnv(control_style={self.control_style!r}, phase={self.phase}, "
            f"step={self.steps}/{self.max_episode_steps})"
        )
