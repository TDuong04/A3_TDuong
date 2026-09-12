"""Arena environment — the Gym-style API around the Part II simulation.

ONE env class for both control schemes. `ArenaEnv(control_style="rotation"|"direct")` switches only
the action space and `_apply_action`; physics, spawning, rewards, observation and rendering are
shared. Forking this into two files is the fastest way to lose rubric row G — every bug gets fixed
twice, the two agents stop being comparable, and report row R6's comparison becomes meaningless.

    action_space = Discrete(5) for rotation, Discrete(6) for direct  # indices in constants.py
    observation_space = Box(-1.0, 1.0, shape=(20,), dtype=float32)  # baseline

`mechanics=True` enables shield pickups and elite chargers, extending the vector to 36
features. From phase 2, progression also requires defeating the elite. Baseline mode
preserves the rules and observation layout of existing checkpoints.

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
things: the ship's starting heading and where each phase puts its spawners (and elite when enabled). Both are randomised on
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
from dataclasses import dataclass, replace
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
    REWARD_DAMAGE_TAKEN,
    REWARD_DEATH,
    REWARD_ENEMY_DESTROYED,
    REWARD_PER_STEP,
    REWARD_PHASE_ADVANCE,
    REWARD_SPAWNER_DESTROYED,
    DirectAction,
    RotationAction,
)
from .debug import REWARD_TERMS, RewardSnapshot
from .entities import Bullet, Enemy, Player, Spawner, phase_config
from .mechanics import EliteCharger, MechanicsConfig, ShieldPickup
from .observation import (
    build_observation,
    describe,
    mechanic_observation,
    nearest_alive,
    observation_scales,
    to_ship_local,
)


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


@dataclass(frozen=True)
class ShapingConfig:
    """The `shaping` block of `config/arena.yaml` -- the reward terms beyond the brief's fixed
    list in `arena/constants.py`.

    Both potential-based only (`F = gamma * phi(s') - phi(s)`, Ng, Harada & Russell 1999): a term
    of this shape leaves the optimal policy provably unchanged for any strength, which is what lets
    it sit alongside the frozen list instead of silently redefining "the best policy". Wiring a
    strength number as a literal in `env.py` would be exactly the thing rule 2 forbids, so both
    read from config like every other tunable. `gamma` mirrors `training.gamma` so shaping
    discounts on the same horizon PPO does rather than a second, disagreeing one.

    `aim_strength` alone measurably improved hit rate but just as measurably cut survival (a 3-seed
    comparison at 0.01: hit rate 12.8%->16.8%, survival 20%->~8% mean) -- rewarding "point at the
    nearest enemy" gives no reason not to also close distance with it, and PPO found that reason on
    its own. `safety_strength` is the deliberate counterweight: same nearest enemy, but on
    *distance* instead of *angle*, so "track it" and "don't close in on it" are rewarded
    independently rather than trading off against each other in one term.
    """

    aim_strength: float
    safety_strength: float
    gamma: float

    @classmethod
    def from_yaml(cls, name: str = "arena") -> ShapingConfig:
        data = load_yaml(name)
        shaping = data["shaping"]
        return cls(
            aim_strength=float(shaping["aim_strength"]),
            safety_strength=float(shaping["safety_strength"]),
            gamma=float(data["training"]["gamma"]),
        )


@lru_cache(maxsize=None)
def shaping_config() -> ShapingConfig:
    return ShapingConfig.from_yaml()


class ArenaEnv(gym.Env):
    """The arena as a Gymnasium 1.3 environment, parameterised by control style."""

    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(
        self,
        control_style: str = "direct",
        render_mode: str | None = None,
        seed: int | None = None,
        max_episode_steps: int | None = None,
        mechanics: bool = False,
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
        self.mechanics = bool(mechanics)
        self.mechanics_config = MechanicsConfig.from_yaml() if self.mechanics else None
        self.actions = ACTION_ENUMS[control_style]
        self.render_mode = render_mode
        self.max_episode_steps = int(
            max_episode_steps if max_episode_steps is not None else MAX_EPISODE_STEPS
        )
        self.episode_config = episode_config()
        self.shaping = shaping_config()

        self.action_space = spaces.Discrete(N_ACTIONS[control_style])
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(describe(self.mechanics)),), dtype=np.float32
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
        # Aim-quality instrumentation for the arena sweep's rotation-vs-direct hit-rate
        # comparison (A3-013 follow-up): counted here rather than derived from reward,
        # because a bullet that expires unfired is a miss and the reward term does not
        # distinguish a miss from a bullet still in flight.
        self.bullets_fired = 0
        self.bullets_hit = 0
        self.damage_taken = 0
        self.pickups_collected = 0
        self.elites_killed = 0
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
        self.pickups: list[ShieldPickup] = []
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
        aim_before = self._aim_potential()
        safety_before = self._safety_potential()
        for _ in range(ACTION_REPEAT):
            reward += self._advance_frame(action)
            if not self.player.alive:
                break  # no point simulating the frames after the ship stopped existing

        self.steps += 1
        terminated = not self.player.alive
        # Distinct by construction: a step cap is not a rule of the game, and an agent that is
        # still alive when the clock runs out must keep bootstrapping from its final value.
        truncated = not terminated and self.steps >= self.max_episode_steps
        # One shaping term per *agent* step, matching REWARD_PER_STEP -- the MDP transition PPO
        # actually sees is (obs before this step, obs after ACTION_REPEAT frames), so phi is
        # evaluated at those two points, not once per physics frame. Two independent terms, not
        # one: aim rewards angle, safety rewards range, deliberately uncoupled (see ShapingConfig).
        self._record_reward(
            "aim_shaping",
            self.shaping.aim_strength * (self.shaping.gamma * self._aim_potential() - aim_before),
        )
        self._record_reward(
            "safety_shaping",
            self.shaping.safety_strength
            * (self.shaping.gamma * self._safety_potential() - safety_before),
        )
        # Canonical sum of the actual contributions captured at each reward site, so a site that
        # forgets to route through `_record_reward` is silently dropped rather than
        # double-counted -- the quieter of the two failure modes, and why the mutation test on
        # `_record_reward` exists.
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
        """True for exactly one agent step after all phase completion conditions are met."""
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
        if self.mechanics and self.phase >= self.mechanics_config.elite_first_phase:
            x, y = self._sample_spawner_position()
            self.enemies.append(EliteCharger(x, y, self.mechanics_config))

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
        self._advance_pickups()
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
            self.bullets_fired += 1

    def _advance_spawners(self) -> None:
        for spawner in self.spawners:
            enemy = spawner.update()
            # Past the cap the spawn is dropped rather than queued: an unbounded enemy list is the
            # usual reason a long run slows to a crawl.
            # Reserve a slot for the next phase's elite.
            cap = self.episode_config.max_enemies - int(self.mechanics)
            if enemy is not None and len(self.enemies) < cap:
                self.enemies.append(enemy)

    # --- collisions and rewards -----------------------------------------------------------------

    def _record_reward(self, term: str, value: float) -> float:
        """Record a fired reward at its source; drawing never calls this method."""
        self._step_rewards[term] += value
        return value

    def _aim_potential(self) -> float:
        """`phi(s)` for the optional aim-alignment shaping term (`ShapingConfig`): the cosine of
        the angle between the ship's heading and the direction to the *lead point* -- where the
        nearest living enemy will be when a bullet fired this instant would reach it -- rather than
        its current position, in the same ship-local frame `observation.py` uses. +1.0 is pointed
        straight at the lead point, -1.0 pointed straight away, 0.0 when no enemy is alive or the
        ship itself is not -- `phi(terminal) = 0` is what keeps potential-based shaping from paying
        out across a death.

        Using the raw current position here taught exactly the miss it was meant to fix: PPO was
        rewarded for pointing at where the enemy *was*, so a converged policy had no incentive to
        lead a moving target at all. `_lead_offset` solves the intercept time against the enemy's
        instantaneous velocity (linear extrapolation, matching how `Enemy.update` itself only ever
        commits to one frame of heading); still a pure function of `s`, so the term stays
        potential-based and the optimal-policy-invariance argument is untouched.
        """
        if not self.player.alive:
            return 0.0
        enemy = nearest_alive(self.player, [e for e in self.enemies if e.alive])
        if enemy is None:
            return 0.0
        dx, dy = enemy.x - self.player.x, enemy.y - self.player.y
        rvx, rvy = enemy.vx - self.player.vx, enemy.vy - self.player.vy
        lead_x, lead_y = _lead_offset(dx, dy, rvx, rvy, self.player.config.bullet_speed)
        distance = math.hypot(lead_x, lead_y)
        if distance <= 1e-6:
            return 1.0  # standing on top of the lead point: no meaningful angle, treat as aligned
        local_x, _ = to_ship_local(self.player.heading, lead_x, lead_y)
        return local_x / distance

    def _safety_potential(self) -> float:
        """`phi(s)` for the optional distance-keeping shaping term (`ShapingConfig`): distance to
        the same nearest living enemy `_aim_potential` tracks, scaled to [0, 1] by the arena
        diagonal. 1.0 is as far apart as the arena allows, 0.0 is touching. No enemy alive scores
        1.0 too -- nothing is closing on the ship, which is the same "as safe as maximally far
        away" a very distant enemy already approaches continuously. `phi(terminal) = 0`, same
        convention and same reason as `_aim_potential`.

        Deliberately independent of `_aim_potential`: that term is about angle, this one is about
        range, so "stay pointed at it" and "don't close the distance" are two separate incentives
        rather than one term quietly trading them against each other.
        """
        if not self.player.alive:
            return 0.0
        enemy = nearest_alive(self.player, [e for e in self.enemies if e.alive])
        if enemy is None:
            return 1.0
        distance = math.hypot(enemy.x - self.player.x, enemy.y - self.player.y)
        return min(1.0, distance / observation_scales().diagonal)

    def _advance_pickups(self) -> None:
        for pickup in self.pickups:
            pickup.update()
            if pickup.collides_with(self.player) and not self.player.shield_charge:
                self.player.shield_charge = 1
                self.pickups_collected += 1
                pickup.kill()
        self.pickups = [pickup for pickup in self.pickups if pickup.alive]

    def _resolve_bullet_hits(self) -> float:
        """One bullet spends itself on one target. Rewards come from `constants.py` only."""
        reward = 0.0
        for bullet in self.bullets:
            if not bullet.alive:
                continue
            for enemy in self.enemies:
                if bullet.collides_with(enemy):
                    bullet.kill()
                    self.bullets_hit += 1
                    enemy.take_damage(bullet.damage)
                    if not enemy.alive:
                        self.enemies_killed += 1
                        if isinstance(enemy, EliteCharger):
                            self.elites_killed += 1
                        reward += self._record_reward("enemy", REWARD_ENEMY_DESTROYED)
                    break
            if not bullet.alive:
                continue
            for spawner in self.spawners:
                if bullet.collides_with(spawner):
                    bullet.kill()
                    self.bullets_hit += 1
                    spawner.take_damage(bullet.damage)
                    if not spawner.alive:
                        self.spawners_destroyed += 1
                        if self.mechanics:
                            self.pickups.append(ShieldPickup(spawner.x, spawner.y,
                                                             self.mechanics_config))
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
        """Clearing spawners (and the elite in mechanics mode) advances the phase.

        Enemies already loose in the arena are not cleared: the reward is for clearing spawners,
        and a free wipe would make the phase transition the safest moment in the episode.
        """
        if self.spawners or any(isinstance(e, EliteCharger) and e.alive for e in self.enemies):
            return 0.0
        self.phase += 1
        self._phase_advanced_this_step = True
        self._begin_phase()
        return self._record_reward("phase", REWARD_PHASE_ADVANCE)

    # --- observation and info --------------------------------------------------------------------

    def _observation(self) -> np.ndarray:
        scales = observation_scales()
        if self.mechanics:
            scales = replace(scales, max_relative_speed=max(
                scales.max_relative_speed,
                self.mechanics_config.charge_speed + self.player.config.speed))
        obs = build_observation(self.player, self.enemies, self.spawners, self.phase, scales)
        if self.mechanics:
            obs = np.concatenate((obs, mechanic_observation(self.player, self.pickups, self.enemies)))
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
            "mechanics": self.mechanics,
            "pickups_collected": self.pickups_collected,
            "shield_blocks": self.player.shield_blocks,
            "elites_killed": self.elites_killed,
            "phase": self.phase,
            "spawners_destroyed": self.spawners_destroyed,
            "enemies_killed": self.enemies_killed,
            "bullets_fired": self.bullets_fired,
            "bullets_hit": self.bullets_hit,
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


def _lead_offset(
    dx: float, dy: float, rvx: float, rvy: float, bullet_speed: float
) -> tuple[float, float]:
    """The firing-solution offset to a target at relative position `(dx, dy)` moving at relative
    velocity `(rvx, rvy)`, for a bullet travelling at `bullet_speed`.

    Solves `|(dx, dy) + (rvx, rvy) * t| == bullet_speed * t` for the smallest positive intercept
    time `t` -- a quadratic in `t` -- and returns the target's extrapolated offset at that time.
    Falls back to the current offset (`t = 0`) when no positive root exists: the target and the
    ship are already colocated, or the target is outrunning what the bullet could ever catch, and
    "aim at where it is" is the only sensible fallback in that case anyway.
    """
    a = rvx * rvx + rvy * rvy - bullet_speed * bullet_speed
    b = 2.0 * (dx * rvx + dy * rvy)
    c = dx * dx + dy * dy

    t = 0.0
    if abs(a) > 1e-9:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            sqrt_discriminant = math.sqrt(discriminant)
            roots = ((-b + sqrt_discriminant) / (2.0 * a), (-b - sqrt_discriminant) / (2.0 * a))
            candidates = [root for root in roots if root > 0.0]
            if candidates:
                t = min(candidates)
    elif abs(b) > 1e-9:
        candidate = -c / b
        if candidate > 0.0:
            t = candidate

    return dx + rvx * t, dy + rvy * t
