"""_common.py: Shared infrastructure for environment-agnostic PCZ/PPO algorithm variants.

Provides:
    - ``_znorm``: Z-normalization helper.
    - ``ComponentRolloutBuffer``: RolloutBuffer with per-component reward storage.
    - ``collect_rollouts``: Rollout collection with component-reward extraction.
    - ``log_normalization_diagnostics``: Pre/post normalization scale logging.
    - ``PopArtMixin``: Adaptive value head rescaling.
    - ``MultiComponentCartPole``: CartPole wrapper for examples.
    - ``run_cartpole_example``: Standard training + evaluation runner.
"""

import inspect
import logging

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: z-normalise array with safe std
# ---------------------------------------------------------------------------


def _znorm(
    arr: np.ndarray,
    axis: int | None = None,
    eps: float = 1e-8,
    min_std: float | None = None,
) -> np.ndarray:
    """Z-normalise *arr* along *axis*.

    When ``std < threshold`` the result is mean-centred only (no variance
    scaling) to avoid amplifying noise for constant or ultra-sparse signals.

    Args:
        arr: Input array.
        axis: Axis along which to compute mean/std.
        eps: Near-zero guard (used when *min_std* is not set).
        min_std: Optional minimum standard deviation threshold.  When set,
            components with ``std < min_std`` are mean-centred but not
            variance-scaled.  Use this for sparse reward components (e.g.
            bonuses that fire on <5 % of timesteps).  If *None*, falls
            back to *eps*.
    """
    threshold = min_std if min_std is not None else eps
    mean = arr.mean(axis=axis, keepdims=True)
    std = arr.std(axis=axis, keepdims=True)
    safe_std = np.where(std > threshold, std, 1.0)
    result = np.where(std > threshold, (arr - mean) / safe_std, arr - mean)
    return np.clip(result, -3.0, 3.0)


def _init_component_weights(
    n_components: int,
    component_weights: list[float] | None = None,
) -> np.ndarray:
    """Validate and convert component weights to a numpy array.

    Returns an array of shape ``(n_components,)`` with dtype float32.
    Defaults to all-ones (equal weights) when *component_weights* is None.
    """
    if component_weights is not None:
        if len(component_weights) != n_components:
            raise ValueError(f"Expected {n_components} component weights, got {len(component_weights)}")
        return np.array(component_weights, dtype=np.float32)
    return np.ones(n_components, dtype=np.float32)


def _weighted_component_sum(
    arr: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted sum across the component axis (axis=2).

    Args:
        arr: shape ``(n_steps, n_envs, n_components)``
        weights: shape ``(n_components,)``

    Returns:
        shape ``(n_steps, n_envs)``
    """
    return (arr * weights[np.newaxis, np.newaxis, :]).sum(axis=2)


# ---------------------------------------------------------------------------
# Rollout buffer with per-component reward storage and bootstrap tracking
# ---------------------------------------------------------------------------


class ComponentRolloutBuffer(RolloutBuffer):
    """RolloutBuffer extended with per-component reward storage and bootstrap tracking.

    Stores per-component rewards alongside the standard scalar reward so that
    normalization methods can operate on individual components before GAE.
    Also tracks timeout bootstrap adjustments so they survive reward overwriting.
    """

    component_rewards: np.ndarray
    timeout_bootstrap: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        n_reward_components: int = 1,
    ):
        self.n_reward_components = n_reward_components
        # Running stats for per-component normalisation (running / vecnorm variants)
        self._running_mean = np.zeros(n_reward_components, dtype=np.float64)
        self._running_var = np.ones(n_reward_components, dtype=np.float64)
        self._running_count = 1e-4
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )

    def reset(self) -> None:
        super().reset()
        self.component_rewards = np.zeros(
            (self.buffer_size, self.n_envs, self.n_reward_components),
            dtype=np.float32,
        )
        self.timeout_bootstrap = np.zeros(
            (self.buffer_size, self.n_envs),
            dtype=np.float32,
        )
        self._component_pos = 0

    def add_component_rewards(self, components: np.ndarray) -> None:
        """Store per-component rewards for current step.

        Args:
            components: shape ``(n_envs, n_reward_components)``
        """
        self.component_rewards[self._component_pos] = components
        self._component_pos += 1

    def _reapply_bootstrap(self) -> None:
        """Re-apply timeout bootstrap adjustments after overwriting self.rewards."""
        self.rewards += self.timeout_bootstrap

    def _update_running_stats(self) -> None:
        """Update running mean/var from current buffer using Welford's algorithm."""
        flat = self.component_rewards.reshape(-1, self.n_reward_components)
        batch_mean = flat.mean(axis=0)
        batch_var = flat.var(axis=0)
        batch_count = flat.shape[0]

        delta = batch_mean - self._running_mean
        total_count = self._running_count + batch_count
        self._running_mean += delta * batch_count / total_count
        m_a = self._running_var * self._running_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self._running_count * batch_count / total_count
        self._running_var = m2 / total_count
        self._running_count = total_count


# ---------------------------------------------------------------------------
# Rollout collection with component-reward extraction
# ---------------------------------------------------------------------------


def _check_vecnormalize_reward(env: VecEnv) -> None:
    """Warn once if VecNormalize is wrapping the env with reward normalization."""
    current = env
    while current is not None:
        # SB3 VecNormalize has a norm_reward attribute
        if hasattr(current, "norm_reward") and getattr(current, "norm_reward", False):
            logger.warning(
                "VecNormalize with norm_reward=True detected. "
                "This algorithm handles reward normalization internally — "
                "set norm_reward=False on VecNormalize to avoid double normalization."
            )
            return
        current = getattr(current, "venv", None)


def collect_rollouts(
    algo: PPO,
    env: VecEnv,
    callback: BaseCallback,
    rollout_buffer: ComponentRolloutBuffer,
    n_rollout_steps: int,
    component_names: list[str],
    *,
    bootstrap_timeout: bool = True,
) -> bool | tuple[bool, th.Tensor, np.ndarray]:
    """Collect rollouts while extracting per-component rewards from ``info``.

    Returns *False* if the callback cancels, otherwise ``(True, last_values, dones)``.
    """
    assert algo._last_obs is not None, "No previous observation was provided"
    algo.policy.set_training_mode(False)

    # One-time check for VecNormalize double-normalization
    if not getattr(algo, "_vecnorm_checked", False):
        _check_vecnormalize_reward(env)
        algo._vecnorm_checked = True  # type: ignore[attr-defined]

    n_steps = 0
    rollout_buffer.reset()

    if algo.use_sde:
        algo.policy.reset_noise(env.num_envs)

    callback.on_rollout_start()

    while n_steps < n_rollout_steps:
        if algo.use_sde and algo.sde_sample_freq > 0 and n_steps % algo.sde_sample_freq == 0:
            algo.policy.reset_noise(env.num_envs)

        with th.no_grad():
            obs_tensor = obs_as_tensor(algo._last_obs, algo.device)
            actions, values, log_probs = algo.policy(obs_tensor)
        actions = actions.cpu().numpy()

        clipped_actions = actions
        if isinstance(algo.action_space, spaces.Box):
            if algo.policy.squash_output:
                clipped_actions = algo.policy.unscale_action(clipped_actions)
            else:
                clipped_actions = np.clip(actions, algo.action_space.low, algo.action_space.high)

        new_obs, rewards, dones, infos = env.step(clipped_actions)

        # ── Extract per-component rewards from info dicts ──────────────
        n_components = len(component_names)
        comp_rewards = np.zeros((env.num_envs, n_components), dtype=np.float32)
        for env_idx, info in enumerate(infos):
            components = info.get("reward_components", {})
            if not components and not getattr(algo, "_missing_components_warned", False):
                logger.warning(
                    "info['reward_components'] is empty or missing for env %d. "
                    "Component rewards will default to 0.0. Verify your "
                    "environment returns info['reward_components'] from step().",
                    env_idx,
                )
                algo._missing_components_warned = True  # type: ignore[attr-defined]
            for comp_idx, name in enumerate(component_names):
                val = components.get(name, None)
                if val is None:
                    if not getattr(algo, "_missing_key_warned", False):
                        logger.warning(
                            "Component '%s' not found in info['reward_components'] "
                            "(env %d, step %d). Defaulting to 0.0. Available keys: %s",
                            name,
                            env_idx,
                            n_steps,
                            list(components.keys()),
                        )
                        algo._missing_key_warned = True  # type: ignore[attr-defined]
                    val = 0.0
                comp_rewards[env_idx, comp_idx] = val
        rollout_buffer.add_component_rewards(comp_rewards)
        # ───────────────────────────────────────────────────────────────

        algo.num_timesteps += env.num_envs

        callback.update_locals(locals())
        if not callback.on_step():
            return False

        algo._update_info_buffer(infos, dones)
        n_steps += 1

        if isinstance(algo.action_space, spaces.Discrete):
            actions = actions.reshape(-1, 1)

        # Handle timeout by bootstrapping with V(s)
        if bootstrap_timeout:
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = algo.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = algo.policy.predict_values(terminal_obs)[0]
                    bootstrap_adj = algo.gamma * terminal_value
                    rewards[idx] += bootstrap_adj
                    rollout_buffer.timeout_bootstrap[n_steps - 1, idx] = float(bootstrap_adj)

        rollout_buffer.add(
            algo._last_obs,
            actions,
            rewards,
            algo._last_episode_starts,
            values,
            log_probs,
        )
        algo._last_obs = new_obs
        algo._last_episode_starts = dones

    with th.no_grad():
        values = algo.policy.predict_values(obs_as_tensor(new_obs, algo.device))

    # Log per-component reward statistics via SB3 logger (captured by MLflow KVWriter)
    _log_component_stats(algo, rollout_buffer, component_names)

    return True, values, dones


def _log_component_stats(
    algo: PPO,
    buf: ComponentRolloutBuffer,
    component_names: list[str],
) -> None:
    """Log per-component reward statistics to SB3's logger.

    Records mean, std, and sparsity (fraction of zeros) for each reward
    component. These are captured automatically by _MLflowKVWriter if
    MLflow is active.
    """
    for i, name in enumerate(component_names):
        comp = buf.component_rewards[: buf._component_pos, :, i]
        if comp.size == 0:
            continue
        algo.logger.record(f"reward_components/{name}_mean", float(comp.mean()))
        algo.logger.record(f"reward_components/{name}_std", float(comp.std()))
        sparsity = float((comp == 0.0).mean())
        algo.logger.record(f"reward_components/{name}_sparsity", sparsity)


def log_normalization_diagnostics(
    algo: PPO,
    pre_rewards: np.ndarray,
    post_rewards: np.ndarray,
) -> None:
    """Log pre/post normalization reward scale diagnostics.

    Helps detect normalization failures (e.g. collapsed variance, extreme
    scaling). Call this in each algorithm's ``collect_rollouts`` after
    applying normalization.

    Args:
        algo: The algorithm instance (for ``algo.logger``).
        pre_rewards: Rewards before normalization, shape ``(n_steps, n_envs)``.
        post_rewards: Rewards after normalization, shape ``(n_steps, n_envs)``.
    """
    algo.logger.record("norm_diag/pre_reward_mean", float(pre_rewards.mean()))
    algo.logger.record("norm_diag/pre_reward_std", float(pre_rewards.std()))
    algo.logger.record("norm_diag/post_reward_mean", float(post_rewards.mean()))
    algo.logger.record("norm_diag/post_reward_std", float(post_rewards.std()))
    pre_std = pre_rewards.std()
    if pre_std > 1e-8:
        algo.logger.record("norm_diag/scale_ratio", float(post_rewards.std() / pre_std))


# ---------------------------------------------------------------------------
# PopArt mixin: shared adaptive value head rescaling logic
# ---------------------------------------------------------------------------


class PopArtMixin:
    """Mixin providing PopArt adaptive value head rescaling.

    Maintains running mean/std of value targets and rescales the value head's
    last linear layer to preserve outputs when stats change.
    """

    _popart_mean: float
    _popart_var: float
    _popart_count: float

    def _init_popart(self) -> None:
        self._popart_mean = 0.0
        self._popart_var = 1.0
        self._popart_count = 1e-4

    def _update_popart_stats(self, returns: np.ndarray) -> tuple[float, float]:
        """Update running mean/var and return old values for rescaling."""
        old_mean, old_var = self._popart_mean, self._popart_var

        batch_mean = returns.mean()
        batch_var = returns.var()
        batch_count = returns.size

        delta = batch_mean - self._popart_mean
        total = self._popart_count + batch_count
        self._popart_mean += delta * batch_count / total
        m_a = self._popart_var * self._popart_count
        m_b = batch_var * batch_count
        self._popart_var = (m_a + m_b + delta**2 * self._popart_count * batch_count / total) / total
        self._popart_count = total

        return old_mean, old_var

    def _rescale_value_head(self, old_mean: float, old_var: float) -> None:
        """Rescale value head weights/bias to preserve outputs after stat update."""
        old_std = max(np.sqrt(old_var), 1e-8)
        new_std = max(np.sqrt(self._popart_var), 1e-8)

        value_net = self.policy.value_net  # type: ignore[attr-defined]
        if hasattr(value_net, "weight"):
            last_layer = value_net
        else:
            layers = [m for m in value_net.modules() if isinstance(m, th.nn.Linear)]
            if not layers:
                return
            last_layer = layers[-1]

        with th.no_grad():
            last_layer.weight.mul_(old_std / new_std)
            last_layer.bias.mul_(old_std / new_std)
            last_layer.bias.add_((old_mean - self._popart_mean) / new_std)

    def _apply_popart_to_returns(self, buf: ComponentRolloutBuffer) -> None:
        """Update PopArt stats, rescale value head, normalise returns."""
        old_mean, old_var = self._update_popart_stats(buf.returns)
        self._rescale_value_head(old_mean, old_var)
        new_std = max(np.sqrt(self._popart_var), 1e-8)
        buf.returns = (buf.returns - self._popart_mean) / new_std


# ---------------------------------------------------------------------------
# CartPole wrapper and example runner for algorithm demos
# ---------------------------------------------------------------------------


class MultiComponentCartPole(gym.Wrapper):
    """Wraps CartPole to provide multi-component rewards via info dict.

    Splits the scalar +1 reward into two synthetic components:
    - ``balance``: reward for keeping the pole upright (angle-based).
    - ``center``: reward for staying near the center (position-based).
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cart_pos, _, pole_angle, _ = obs

        balance = max(0.0, 1.0 - abs(pole_angle) / 0.2095)
        center = max(0.0, 1.0 - abs(cart_pos) / 2.4)

        info["reward_components"] = {
            "balance": balance,
            "center": center,
        }
        return obs, reward, terminated, truncated, info


RESOURCE_COMPONENT_NAMES = ["death_reward", "gold", "diamond"]


class MultiComponentResource(gym.Wrapper):
    """Wraps mo-gymnasium's ``resource-gathering-v0`` to provide per-component rewards.

    Converts the 3D reward vector from mo-gymnasium into our standard
    ``info["reward_components"]`` dict contract.

    Reward components (from mo-gymnasium's reward vector):
    - ``death_reward``: reward for avoiding death (negative when dead).
    - ``gold``: reward for collecting gold.
    - ``diamond``: reward for collecting diamonds.
    """

    def __init__(self, env: gym.Env, weights: np.ndarray | None = None):
        super().__init__(env)
        self._weights = weights if weights is not None else np.ones(len(RESOURCE_COMPONENT_NAMES), dtype=np.float32)
        # Override reward space to scalar (SB3 expects scalar rewards)
        self.reward_space = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        reward = np.asarray(reward, dtype=np.float32)
        info["reward_components"] = {
            "death_reward": float(reward[0]),
            "gold": float(reward[1]),
            "diamond": float(reward[2]),
        }

        scalar_reward = float(np.sum(reward * self._weights))

        return obs, scalar_reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info["reward_components"] = {name: 0.0 for name in RESOURCE_COMPONENT_NAMES}
        return obs, info


MARIO_COMPONENT_NAMES = ["x_pos", "time", "death", "coin", "score"]


class MultiComponentMario(gym.Env):
    """Gymnasium-compatible wrapper for gym-super-mario-bros.

    Wraps the NES Super Mario Bros environment (old gym API) to provide:
    - Gymnasium 5-tuple step API
    - Per-component reward decomposition via info["reward_components"]
    - Observation preprocessing (grayscale + resize to 84x84x1)
    - JoypadSpace action reduction (SIMPLE_MOVEMENT by default)
    - Optional live rendering via render_mode="human"

    Reward components (derived from info dict deltas):
    - x_pos: horizontal position change (forward progress)
    - time: time remaining change (negative = penalty)
    - death: -15.0 when a life is lost, 0.0 otherwise
    - coin: coin count change (scaled by 10.0)
    - score: game score change (scaled by 0.01)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Valid ROM modes: vanilla (full RGB), downsample, pixel, rectangle
    VALID_ROM_MODES = ("vanilla", "downsample", "pixel", "rectangle")

    def __init__(
        self,
        rom_mode: str = "vanilla",
        render_mode: str | None = None,
        actions: list | None = None,
        max_episode_steps: int = 2000,
    ):
        super().__init__()
        import os

        if render_mode != "human" and not os.environ.get("DISPLAY"):
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
        from nes_py.wrappers import JoypadSpace

        if rom_mode not in self.VALID_ROM_MODES:
            raise ValueError(f"Invalid rom_mode '{rom_mode}'. Choose from: {', '.join(self.VALID_ROM_MODES)}")

        # Create the raw NES env with the specified ROM mode
        self._nes_env = SuperMarioBrosEnv(rom_mode=rom_mode)
        # Apply JoypadSpace for action reduction
        if actions is None:
            from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

            actions = SIMPLE_MOVEMENT
        self._env = JoypadSpace(self._nes_env, actions)

        self.render_mode = render_mode
        self._render_failed = False
        self._pygame_screen = None  # Lazy-init pygame display for human mode
        self._pygame_clock = None
        self._max_episode_steps = max_episode_steps
        self._step_count = 0
        # Observation: 84x84x1 grayscale
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 1),
            dtype=np.uint8,
        )
        # Action space: Discrete from JoypadSpace
        self.action_space = spaces.Discrete(len(actions))

        # State tracking for reward component computation
        self._prev_info = None
        self._prev_life = None

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        """Downscale and grayscale: (240,256,3) -> (84,84,1)."""
        gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        h, w = gray.shape
        indices_h = (np.arange(84) * (h / 84)).astype(int)
        indices_w = (np.arange(84) * (w / 84)).astype(int)
        downscaled = gray[np.ix_(indices_h, indices_w)]
        return downscaled[:, :, np.newaxis]

    def _compute_components(self, info: dict) -> dict:
        """Compute reward components from info dict deltas."""
        prev = self._prev_info or info
        x_pos_delta = float(info.get("x_pos", 0) - prev.get("x_pos", 0))
        time_delta = float(info.get("time", 0) - prev.get("time", 0))
        # Death: life decreased
        death = -15.0 if info.get("life", 3) < prev.get("life", 3) else 0.0
        # Coins: delta scaled
        coin_delta = float(info.get("coins", 0) - prev.get("coins", 0)) * 10.0
        # Score: delta scaled down
        score_delta = float(info.get("score", 0) - prev.get("score", 0)) * 0.01
        return {
            "x_pos": x_pos_delta,
            "time": time_delta,
            "death": death,
            "coin": coin_delta,
            "score": score_delta,
        }

    def reset(self, *, seed=None, options=None):
        self._env.reset()
        # Step with NOOP to get initial info dict
        obs2, _, done, info = self._env.step(0)
        if done:
            obs2 = self._env.reset()
            _, _, _, info = self._env.step(0)
        self._prev_info = info
        self._prev_life = info.get("life", 2)
        self._step_count = 0
        processed = self._preprocess_obs(obs2)
        self._try_render()
        return processed, {"reward_components": {k: 0.0 for k in MARIO_COMPONENT_NAMES}}

    def step(self, action):
        obs, _reward, done, info = self._env.step(action)
        self._step_count += 1

        # Compute component rewards from info dict deltas
        components = self._compute_components(info)
        self._prev_info = info.copy()

        # Use component sum as scalar reward
        scalar_reward = sum(components.values())
        info["reward_components"] = components

        # Terminate on any life loss (standard Mario RL practice)
        # Each life is treated as a separate episode for faster learning
        current_life = info.get("life", self._prev_life)
        life_lost = current_life < self._prev_life
        self._prev_life = current_life

        terminated = done or life_lost
        # Truncate if max episode steps reached
        truncated = (not terminated) and (self._step_count >= self._max_episode_steps)

        if truncated:
            info["TimeLimit.truncated"] = True

        processed = self._preprocess_obs(obs)

        self._try_render()

        return processed, scalar_reward, terminated, truncated, info

    def _try_render(self):
        """Render the current NES frame via pygame (same backend as CartPole).

        Uses rgb_array from the NES env and blits it to a pygame window,
        avoiding nes-py's pyglet renderer which requires OpenGL GLU.
        """
        if self._render_failed or self.render_mode != "human":
            return
        try:
            frame = self._nes_env.render(mode="rgb_array")
            if frame is None:
                return
            import pygame

            if self._pygame_screen is None:
                pygame.init()
                pygame.display.init()
                h, w = frame.shape[:2]
                scale = 2  # 240x256 → 480x512 for visibility
                self._pygame_screen = pygame.display.set_mode((w * scale, h * scale))
                pygame.display.set_caption("Super Mario Bros")
                self._pygame_clock = pygame.time.Clock()
            # Blit frame to screen
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            scaled = pygame.transform.scale(surf, self._pygame_screen.get_size())
            self._pygame_screen.blit(scaled, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self._pygame_clock.tick(self.metadata["render_fps"])
        except Exception as e:
            self._render_failed = True
            logger.warning("Rendering disabled: %s", e)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._nes_env.render(mode="rgb_array")
        elif self.render_mode == "human":
            self._try_render()

    def close(self):
        if self._pygame_screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self._pygame_screen = None
        self._env.close()


HALFCHEETAH_COMPONENT_NAMES = ["run", "ctrl_cost"]


class MultiComponentHalfCheetah(gym.Wrapper):
    """Wraps Gymnasium's ``HalfCheetah-v4`` to provide per-component rewards.

    Extracts reward components from the info dict:
    - ``run``: Forward velocity reward (dense, ~0.1-1.0).
    - ``ctrl_cost``: Control cost penalty (dense, negative ~-0.1-0.3).

    Scale imbalance: ~1.5x — low, expect GDPO ≈ PPO (negative control).
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        info["reward_components"] = {
            "run": float(info["reward_run"]),
            "ctrl_cost": float(info["reward_ctrl"]),
        }

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info["reward_components"] = {name: 0.0 for name in HALFCHEETAH_COMPONENT_NAMES}
        return obs, info


HALFCHEETAH_K4_COMPONENT_NAMES = ["velocity", "ctrl_j01", "ctrl_j23", "ctrl_j45"]


class MultiComponentHalfCheetahK4(gym.Wrapper):
    """HalfCheetah K=4: velocity + 3 ctrl-cost groups (2 joints each).

    CRITICAL: Total reward = sum of K=4 components = K=2 total.
    ctrl_j01 + ctrl_j23 + ctrl_j45 = ctrl_cost (K=2).
    K-scaling invariant preserved.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        action = np.asarray(action, dtype=np.float32)

        velocity = float(info["reward_run"])
        ctrl_j01 = -0.1 * float(np.sum(np.square(action[0:2])))
        ctrl_j23 = -0.1 * float(np.sum(np.square(action[2:4])))
        ctrl_j45 = -0.1 * float(np.sum(np.square(action[4:6])))

        info["reward_components"] = {
            "velocity": velocity,
            "ctrl_j01": ctrl_j01,
            "ctrl_j23": ctrl_j23,
            "ctrl_j45": ctrl_j45,
        }

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info["reward_components"] = {name: 0.0 for name in HALFCHEETAH_K4_COMPONENT_NAMES}
        return obs, info


HALFCHEETAH_K6_COMPONENT_NAMES = [
    "velocity",
    "ctrl_j0",
    "ctrl_j1",
    "ctrl_j2",
    "ctrl_j3",
    "ctrl_j45",
]


class MultiComponentHalfCheetahK6(gym.Wrapper):
    """HalfCheetah K=6: velocity + 5 ctrl-cost groups (4 individual + 1 pair).

    CRITICAL: Total reward = sum of K=6 = sum of K=4 = sum of K=2.
    All ctrl components sum to the original ctrl_cost.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        action = np.asarray(action, dtype=np.float32)

        velocity = float(info["reward_run"])

        info["reward_components"] = {
            "velocity": velocity,
            "ctrl_j0": -0.1 * float(action[0] ** 2),
            "ctrl_j1": -0.1 * float(action[1] ** 2),
            "ctrl_j2": -0.1 * float(action[2] ** 2),
            "ctrl_j3": -0.1 * float(action[3] ** 2),
            "ctrl_j45": -0.1 * float(np.sum(np.square(action[4:6]))),
        }

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info["reward_components"] = {name: 0.0 for name in HALFCHEETAH_K6_COMPONENT_NAMES}
        return obs, info


HALFCHEETAH_K8_COMPONENT_NAMES = [
    "velocity_fwd",
    "velocity_bwd",
    "ctrl_bhip",
    "ctrl_bknee",
    "ctrl_bfoot",
    "ctrl_fhip",
    "ctrl_fknee",
    "ctrl_ffoot",
]


class MultiComponentHalfCheetahK8(gym.Wrapper):
    """HalfCheetah K=8: velocity sign-split (fwd/bwd) + 6 per-joint ctrl costs.

    velocity_fwd = max(reward_run, 0), velocity_bwd = min(reward_run, 0).
    Their sum = reward_run. Per-joint ctrl costs (bhip, bknee, bfoot, fhip,
    fknee, ffoot) sum to ctrl_cost. Total preserved: sum(K=8) = original reward.

    Provides a second env family with K=8 for the K-scaling claim.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        action = np.asarray(action, dtype=np.float32)

        run = float(info["reward_run"])
        info["reward_components"] = {
            "velocity_fwd": max(run, 0.0),
            "velocity_bwd": min(run, 0.0),
            "ctrl_bhip": -0.1 * float(action[0] ** 2),
            "ctrl_bknee": -0.1 * float(action[1] ** 2),
            "ctrl_bfoot": -0.1 * float(action[2] ** 2),
            "ctrl_fhip": -0.1 * float(action[3] ** 2),
            "ctrl_fknee": -0.1 * float(action[4] ** 2),
            "ctrl_ffoot": -0.1 * float(action[5] ** 2),
        }

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info["reward_components"] = {name: 0.0 for name in HALFCHEETAH_K8_COMPONENT_NAMES}
        return obs, info


HUMANOID_COMPONENT_NAMES = ["forward", "alive", "ctrl_cost"]


class MultiComponentHumanoid(gym.Wrapper):
    """Wraps Gymnasium's ``Humanoid-v4`` to provide per-component rewards.

    Extracts reward components from the info dict that MuJoCo already provides:
    - ``forward``: Forward velocity reward (dense, tiny ~0.001-0.1).
    - ``alive``: Survival bonus (+5.0 per step while standing, dense, dominant).
    - ``ctrl_cost``: Control cost penalty (dense, small negative ~-0.1).

    Scale imbalance: alive/forward ≈ 500-5000x — ideal for testing GDPO.

    The scalar reward returned matches Gymnasium's native computation:
    ``reward = forward + alive + ctrl_cost``.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        info["reward_components"] = {
            "forward": float(info["forward_reward"]),
            "alive": float(info["reward_alive"]),
            "ctrl_cost": float(info["reward_quadctrl"]),
        }

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info["reward_components"] = {name: 0.0 for name in HUMANOID_COMPONENT_NAMES}
        return obs, info


LUNAR_COMPONENT_NAMES = ["landing", "shaping", "fuel_main", "fuel_side"]


class MultiComponentLunarLander(gym.Wrapper):
    """Wraps mo-gymnasium's ``mo-lunar-lander-v3`` to provide per-component rewards.

    Converts the 4D reward vector from mo-gymnasium into our standard
    ``info["reward_components"]`` dict contract.

    Reward components (from mo-gymnasium's reward vector):
    - ``landing``: +100 for success, -100 for crash (sparse, terminal only).
    - ``shaping``: Potential-based shaping reward (dense, high variance).
    - ``fuel_main``: Main engine fuel cost in [-1, 0] (dense, small scale).
    - ``fuel_side``: Side engine fuel cost in [-1, 0] (dense, small scale).

    The scalar reward returned to the agent is the weighted sum of components
    (by default, equal weights = simple sum).

    Args:
        env: A mo-gymnasium environment returning vector rewards.
        weights: Optional weight vector for scalarisation.  Default: all ones.
    """

    def __init__(self, env: gym.Env, weights: np.ndarray | None = None):
        super().__init__(env)
        self._weights = weights if weights is not None else np.ones(4, dtype=np.float32)
        # Override reward space to scalar (SB3 expects scalar rewards)
        self.reward_space = None  # Remove mo-gymnasium's vector reward space

    def step(self, action):
        obs, reward_vec, terminated, truncated, info = self.env.step(action)

        # reward_vec is np.ndarray of shape (4,) from mo-gymnasium
        reward_vec = np.asarray(reward_vec, dtype=np.float32)

        info["reward_components"] = {
            "landing": float(reward_vec[0]),
            "shaping": float(reward_vec[1]),
            "fuel_main": float(reward_vec[2]),
            "fuel_side": float(reward_vec[3]),
        }

        # Scalar reward: weighted sum of components
        scalar_reward = float(np.dot(reward_vec, self._weights))

        return obs, scalar_reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info["reward_components"] = {name: 0.0 for name in LUNAR_COMPONENT_NAMES}
        return obs, info


LUNAR_K2_COMPONENT_NAMES = ["landing", "dense"]
LUNAR_K6_COMPONENT_NAMES = [
    "landing",
    "distance",
    "velocity",
    "angle",
    "fuel_main",
    "fuel_side",
]
LUNAR_K8_COMPONENT_NAMES = [
    "landing",
    "distance",
    "velocity",
    "angle",
    "leg_left",
    "leg_right",
    "fuel_main",
    "fuel_side",
]


class KScalingLunarLanderK2(gym.Wrapper):
    """K=2 variant: landing (sparse) + dense (shaping+fuel merged).

    Used for K-scaling experiments. Same total reward as K=4, just coarser
    decomposition: 1 sparse + 1 dense aggregate.
    """

    def __init__(self, env: gym.Env, weights: np.ndarray | None = None):
        super().__init__(env)
        self._weights = weights if weights is not None else np.ones(4, dtype=np.float32)
        self.reward_space = None

    def step(self, action):
        obs, reward_vec, terminated, truncated, info = self.env.step(action)
        reward_vec = np.asarray(reward_vec, dtype=np.float32)

        # K=2: landing stays separate, merge shaping+fuel_main+fuel_side
        landing = float(reward_vec[0])
        dense = float(reward_vec[1] + reward_vec[2] + reward_vec[3])

        info["reward_components"] = {"landing": landing, "dense": dense}

        scalar_reward = float(np.dot(reward_vec, self._weights))
        return obs, scalar_reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info["reward_components"] = {name: 0.0 for name in LUNAR_K2_COMPONENT_NAMES}
        return obs, info


class KScalingLunarLanderK6(gym.Wrapper):
    """K=6 variant: decompose shaping into 3 sub-potentials (distance/velocity/angle).

    Intermediate between K=4 (1 aggregate shaping) and K=8 (5 sub-potentials).
    Legs contact is merged back into the distance potential.
    Components: landing (sparse), distance, velocity, angle (dense shaping),
    fuel_main, fuel_side (dense, small scale).
    """

    def __init__(self, env: gym.Env, weights: np.ndarray | None = None):
        super().__init__(env)
        self._weights = weights if weights is not None else np.ones(4, dtype=np.float32)
        self.reward_space = None
        self._prev_sub_potentials: np.ndarray | None = None

    def _compute_sub_potentials(self, obs: np.ndarray) -> np.ndarray:
        """Compute 3 sub-potentials from observation vector."""
        x, y, vx, vy, angle, _, left_leg, right_leg = obs[:8]
        return np.array(
            [
                -100.0 * np.sqrt(x * x + y * y) + 10.0 * (left_leg + right_leg),  # distance + legs
                -100.0 * np.sqrt(vx * vx + vy * vy),  # velocity
                -100.0 * np.abs(angle),  # angle
            ],
            dtype=np.float64,
        )

    def step(self, action):
        obs, reward_vec, terminated, truncated, info = self.env.step(action)
        reward_vec = np.asarray(reward_vec, dtype=np.float32)

        current_subs = self._compute_sub_potentials(obs)
        if self._prev_sub_potentials is not None:
            sub_deltas = current_subs - self._prev_sub_potentials
        else:
            sub_deltas = np.zeros(3, dtype=np.float64)
        self._prev_sub_potentials = current_subs

        info["reward_components"] = {
            "landing": float(reward_vec[0]),
            "distance": float(sub_deltas[0]),
            "velocity": float(sub_deltas[1]),
            "angle": float(sub_deltas[2]),
            "fuel_main": float(reward_vec[2]),
            "fuel_side": float(reward_vec[3]),
        }

        scalar_reward = float(np.dot(reward_vec, self._weights))
        return obs, scalar_reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_sub_potentials = self._compute_sub_potentials(obs)
        info["reward_components"] = {name: 0.0 for name in LUNAR_K6_COMPONENT_NAMES}
        return obs, info


class KScalingLunarLanderK8(gym.Wrapper):
    """K=8 variant: decompose shaping into 5 sub-potentials.

    Splits the shaping reward into distance, velocity, angle, leg_left,
    leg_right using temporal differences of sub-potentials computed from
    the observation vector. Same total reward as K=4.

    Sub-potentials (from gymnasium LunarLander source):
    - distance: -100 * sqrt(x² + y²)
    - velocity: -100 * sqrt(vx² + vy²)
    - angle: -100 * |angle|
    - leg_left: 10 * left_contact
    - leg_right: 10 * right_contact
    """

    def __init__(self, env: gym.Env, weights: np.ndarray | None = None):
        super().__init__(env)
        self._weights = weights if weights is not None else np.ones(4, dtype=np.float32)
        self.reward_space = None
        self._prev_sub_potentials: np.ndarray | None = None

    def _compute_sub_potentials(self, obs: np.ndarray) -> np.ndarray:
        """Compute 5 sub-potentials from observation vector."""
        x, y, vx, vy, angle, _, left_leg, right_leg = obs[:8]
        return np.array(
            [
                -100.0 * np.sqrt(x * x + y * y),  # distance
                -100.0 * np.sqrt(vx * vx + vy * vy),  # velocity
                -100.0 * np.abs(angle),  # angle
                10.0 * left_leg,  # leg_left
                10.0 * right_leg,  # leg_right
            ],
            dtype=np.float64,
        )

    def step(self, action):
        obs, reward_vec, terminated, truncated, info = self.env.step(action)
        reward_vec = np.asarray(reward_vec, dtype=np.float32)

        # Compute sub-potential temporal differences
        current_subs = self._compute_sub_potentials(obs)
        if self._prev_sub_potentials is not None:
            sub_deltas = current_subs - self._prev_sub_potentials
        else:
            sub_deltas = np.zeros(5, dtype=np.float64)
        self._prev_sub_potentials = current_subs

        info["reward_components"] = {
            "landing": float(reward_vec[0]),
            "distance": float(sub_deltas[0]),
            "velocity": float(sub_deltas[1]),
            "angle": float(sub_deltas[2]),
            "leg_left": float(sub_deltas[3]),
            "leg_right": float(sub_deltas[4]),
            "fuel_main": float(reward_vec[2]),
            "fuel_side": float(reward_vec[3]),
        }

        scalar_reward = float(np.dot(reward_vec, self._weights))
        return obs, scalar_reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_sub_potentials = self._compute_sub_potentials(obs)
        info["reward_components"] = {name: 0.0 for name in LUNAR_K8_COMPONENT_NAMES}
        return obs, info


BIPEDAL_COMPONENT_NAMES = ["shaping", "energy", "crash"]


class MultiComponentBipedalWalker(gym.Wrapper):
    """Wraps Gymnasium's ``BipedalWalker-v3`` to provide per-component rewards.

    Decomposes the scalar reward into three components:
    - ``shaping``: Forward progress + head angle penalty (dense, ~0.3/step).
    - ``energy``: Motor torque cost (dense, small negative ~-0.03/step).
    - ``crash``: Terminal crash penalty (sparse, -100 on game_over).

    Scale imbalance: crash/shaping ≈ 300x when it fires — genuine sparse/dense
    mismatch similar to LunarLander's landing bonus.

    The scalar reward returned matches Gymnasium's native computation.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._prev_shaping: float | None = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Reconstruct components from the BipedalWalker source code.
        # The env computes: reward = (shaping - prev_shaping) + energy + crash
        # We need to decompose this.
        unwrapped = self.env.unwrapped

        # Shaping: 130 * pos.x / SCALE - 5.0 * |hull_angle|
        hull = unwrapped.hull
        if hull is not None:
            from gymnasium.envs.box2d.bipedal_walker import SCALE

            current_shaping = 130 * hull.position[0] / SCALE - 5.0 * abs(obs[0])
            if self._prev_shaping is not None:
                shaping_reward = current_shaping - self._prev_shaping
            else:
                shaping_reward = 0.0
            self._prev_shaping = current_shaping
        else:
            shaping_reward = 0.0

        # Energy cost: -0.00035 * MOTORS_TORQUE * |action_i| for each joint
        from gymnasium.envs.box2d.bipedal_walker import MOTORS_TORQUE

        energy_cost = sum(-0.00035 * MOTORS_TORQUE * float(np.clip(np.abs(a), 0, 1)) for a in action)

        # Crash: -100 if terminated by game_over or pos.x < 0
        # The env sets reward = -100 and overwrites the other components on crash.
        if terminated and hull is not None:
            game_over = unwrapped.game_over
            pos_x = hull.position[0]
            if game_over or pos_x < 0:
                crash_penalty = -100.0
                # On crash, the env replaces the entire reward with -100
                # So shaping and energy from this step are zeroed out
                shaping_reward = 0.0
                energy_cost = 0.0
            else:
                crash_penalty = 0.0
        else:
            crash_penalty = 0.0

        info["reward_components"] = {
            "shaping": float(shaping_reward),
            "energy": float(energy_cost),
            "crash": float(crash_penalty),
        }

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # Sync with the env's internal prev_shaping set during reset()
        self._prev_shaping = self.env.unwrapped.prev_shaping
        info["reward_components"] = {name: 0.0 for name in BIPEDAL_COMPONENT_NAMES}
        return obs, info


class WeightedRewardWrapper(gym.Wrapper):
    """Re-weight the scalar reward using component weights.

    Sits on top of any MultiComponent wrapper.  Reads
    ``info["reward_components"]`` and computes::

        reward = sum(w_i * component_i)

    This gives standard PPO (which ignores per-component normalization)
    the same priority signal that PCZ-PPO gets from its component weights.

    Args:
        env: Environment that returns ``info["reward_components"]``.
        component_names: Ordered list of component names.
        weights: Weight for each component (same order as *component_names*).
    """

    def __init__(
        self,
        env: gym.Env,
        component_names: list[str],
        weights: list[float],
    ):
        super().__init__(env)
        assert len(component_names) == len(weights)
        self._component_names = component_names
        self._weights = {n: w for n, w in zip(component_names, weights)}

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        components = info.get("reward_components", {})
        weighted_reward = sum(
            self._weights.get(name, 1.0) * components.get(name, 0.0) for name in self._component_names
        )
        return obs, weighted_reward, terminated, truncated, info


DEFAULT_COMPONENT_NAMES = ["balance", "center"]


def run_cartpole_example(
    algo_class: type,
    algo_name: str,
    component_names: list[str] | None = None,
    total_timesteps: int = 10_000,
    **extra_kwargs,
) -> None:
    """Standard CartPole training + evaluation example."""
    if component_names is None:
        component_names = DEFAULT_COMPONENT_NAMES

    print(f"=== {algo_name} on CartPole ===")
    print(f"Components: {', '.join(component_names)}")
    print()

    vec_env = make_vec_env(
        lambda: MultiComponentCartPole(gym.make("CartPole-v1")),
        n_envs=2,
    )

    kwargs = dict(
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        device="cpu",
    )
    kwargs.update(extra_kwargs)

    # Add reward_component_names if the class constructor accepts it
    sig = inspect.signature(algo_class.__init__)
    if "reward_component_names" in sig.parameters:
        kwargs["reward_component_names"] = component_names

    model = algo_class("MlpPolicy", vec_env, **kwargs)
    model.learn(total_timesteps=total_timesteps)

    # Quick evaluation
    obs = vec_env.reset()
    total_reward = 0.0
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _done, _info = vec_env.step(action)
        total_reward += reward.sum()
    print(f"\nEval reward (200 steps, 2 envs): {total_reward:.1f}")
    vec_env.close()


# ---------------------------------------------------------------------------
# MO-Reacher wrapper
# ---------------------------------------------------------------------------


class MultiComponentReacher(gym.Wrapper):
    """Wraps mo-gymnasium's ``mo-reacher-v4`` to provide per-component rewards.

    Converts the 4D reward vector from mo-gymnasium into our standard
    ``info["reward_components"]`` dict contract.

    Reward components (from mo-gymnasium's reward vector):
    - ``target_1`` through ``target_4``: Distance-based reward to each of 4
      targets.  All dense, range [-1, 1].

    Args:
        env: A mo-gymnasium environment returning vector rewards.
        weights: Optional weight vector for scalarisation.  Default: all ones.
    """

    def __init__(self, env: gym.Env, weights: np.ndarray | None = None):
        super().__init__(env)
        self._weights = weights if weights is not None else np.ones(4, dtype=np.float32)
        self.reward_space = None

    def step(self, action):
        obs, reward_vec, terminated, truncated, info = self.env.step(action)
        reward_vec = np.asarray(reward_vec, dtype=np.float32)

        info["reward_components"] = {f"target_{i + 1}": float(reward_vec[i]) for i in range(4)}

        scalar_reward = float(np.dot(reward_vec, self._weights))
        return obs, scalar_reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info["reward_components"] = {f"target_{i + 1}": 0.0 for i in range(4)}
        return obs, info


# ---------------------------------------------------------------------------
# MountainCar K-scaling wrappers
# ---------------------------------------------------------------------------

MOUNTAINCAR_K4_COMPONENT_NAMES = ["time", "velocity", "height", "goal"]


class MultiComponentMountainCar(gym.Wrapper):
    """Wraps Gymnasium's ``MountainCar-v0`` with K=4 reward decomposition.

    Decomposes the sparse -1/step reward into 4 components with explicit
    sparse/dense mismatch for K-scaling validation on a second domain
    (different dynamics from LunarLander):

    - ``time``: -0.1 per step (dense, constant — background cost).
    - ``velocity``: 10 * |v| (dense, 0 to 0.7 — momentum signal).
    - ``height``: 10 * sin(3*pos) (dense, -10 to +10 — potential energy shaping).
    - ``goal``: +100 on reaching pos >= 0.5 (sparse, binary).

    Scale mismatch: goal/time = 1000x when it fires; goal fires ~once/episode
    while time/velocity/height fire every step. This is analogous to
    LunarLander's landing/shaping/fuel mismatch.

    The shaping rewards (velocity, height) make MountainCar learnable by
    PPO — standard MountainCar with only -1/step has no gradient signal.
    """

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)

        position, velocity = float(obs[0]), float(obs[1])

        # Decompose into 4 components
        time_penalty = -0.1
        velocity_shaping = 10.0 * abs(velocity)
        height_shaping = 10.0 * np.sin(3.0 * position)  # MountainCar's height
        goal_bonus = 100.0 if terminated and position >= 0.5 else 0.0

        info["reward_components"] = {
            "time": time_penalty,
            "velocity": velocity_shaping,
            "height": height_shaping,
            "goal": goal_bonus,
        }

        # Scalar reward = sum of components (replaces standard -1/step)
        scalar_reward = time_penalty + velocity_shaping + height_shaping + goal_bonus
        return obs, scalar_reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info["reward_components"] = {name: 0.0 for name in MOUNTAINCAR_K4_COMPONENT_NAMES}
        return obs, info


MOUNTAINCAR_K6_COMPONENT_NAMES = [
    "time",
    "vel_right",
    "vel_left",
    "height_up",
    "height_down",
    "goal",
]


class MultiComponentMountainCarK6(gym.Wrapper):
    """K=6 MountainCar: decompose K=4's velocity and height into signed halves.

    CRITICAL: Total reward = sum of K=6 components = sum of K=4 components.
    K-scaling invariant preserved:
    - vel_right + vel_left = 10*|v| (K=4's velocity)
    - height_up + height_down = 10*sin(3*pos) (K=4's height)
    """

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)

        position, velocity = float(obs[0]), float(obs[1])

        time_penalty = -0.1
        vel_right = 10.0 * max(0.0, velocity)
        vel_left = 10.0 * max(0.0, -velocity)
        height_val = 10.0 * np.sin(3.0 * position)
        height_up = max(0.0, height_val)
        height_down = min(0.0, height_val)
        goal_bonus = 100.0 if terminated and position >= 0.5 else 0.0

        info["reward_components"] = {
            "time": time_penalty,
            "vel_right": vel_right,
            "vel_left": vel_left,
            "height_up": height_up,
            "height_down": height_down,
            "goal": goal_bonus,
        }

        # Same total as K=4: time + |v|*10 + sin(3*pos)*10 + goal
        scalar_reward = time_penalty + vel_right + vel_left + height_up + height_down + goal_bonus
        return obs, scalar_reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info["reward_components"] = {name: 0.0 for name in MOUNTAINCAR_K6_COMPONENT_NAMES}
        return obs, info


MOUNTAINCAR_K2_COMPONENT_NAMES = ["dense", "goal"]


class KScalingMountainCarK2(gym.Wrapper):
    """K=2 MountainCar: dense (time+velocity+height merged) + goal (sparse).

    For K-scaling comparison: K=2 should show PPO ≈ PCZ (simple structure).
    """

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)

        position, velocity = float(obs[0]), float(obs[1])

        dense = -0.1 + 10.0 * abs(velocity) + 10.0 * np.sin(3.0 * position)
        goal_bonus = 100.0 if terminated and position >= 0.5 else 0.0

        info["reward_components"] = {"dense": dense, "goal": goal_bonus}

        scalar_reward = dense + goal_bonus
        return obs, scalar_reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info["reward_components"] = {name: 0.0 for name in MOUNTAINCAR_K2_COMPONENT_NAMES}
        return obs, info
