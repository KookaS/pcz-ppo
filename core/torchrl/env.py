"""Environment wrappers and builders for TorchRL training.

Provides gymnasium wrappers that embed per-component reward vectors into the
observation space (as ``obs["reward_vec"]``), enabling TorchRL's pipeline to
propagate component rewards through the TensorDict without relying on the
info dict.

Supports both mo-gymnasium environments (vector rewards) and standard
gymnasium environments with ``info["reward_components"]`` (GDPO contract).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Gymnasium wrappers (applied before TorchRL's GymWrapper)
# ---------------------------------------------------------------------------


class RewardVecWrapper(gym.Wrapper):
    """Wrap an mo-gymnasium env: embed vector reward in obs, return scalar.

    For environments that return a vector reward from ``step()``, this wrapper:
    1. Stores the raw vector in ``obs["reward_vec"]``
    2. Returns a weighted (or summed) scalar reward for TorchRL's value function

    Args:
        env: An mo-gymnasium environment returning vector rewards.
        reward_dim: Number of reward components.
        weights: Optional per-component weights for scalar reward.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_dim: int,
        weights: list[float] | None = None,
    ):
        super().__init__(env)
        self.reward_dim = reward_dim
        self._weights = np.array(weights, dtype=np.float32) if weights is not None else None

        # Extend obs space to include reward_vec
        obs_space = env.observation_space
        if isinstance(obs_space, spaces.Dict):
            new_spaces = dict(obs_space.spaces)
        else:
            new_spaces = {"observation": obs_space}
        new_spaces["reward_vec"] = spaces.Box(low=-np.inf, high=np.inf, shape=(reward_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict(new_spaces)

        # Remove mo-gymnasium's vector reward_space — GymWrapper reads it
        # and would expect a vector reward tensor if present.
        if hasattr(self.unwrapped, "reward_space"):
            del self.unwrapped.reward_space

    def step(self, action):
        obs, reward_vec, terminated, truncated, info = self.env.step(action)
        reward_vec = np.asarray(reward_vec, dtype=np.float32)

        if not isinstance(obs, dict):
            obs = {"observation": obs}
        obs["reward_vec"] = reward_vec

        if self._weights is not None:
            scalar = float(np.dot(reward_vec, self._weights))
        else:
            scalar = float(np.sum(reward_vec))

        return obs, scalar, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if not isinstance(obs, dict):
            obs = {"observation": obs}
        obs["reward_vec"] = np.zeros(self.reward_dim, dtype=np.float32)
        return obs, info


class ScalarComponentWrapper(gym.Wrapper):
    """Wrap a standard env with info["reward_components"] into Dict obs.

    For environments that follow the GDPO contract (scalar reward +
    ``info["reward_components"]`` dict), this wrapper embeds the component
    values into ``obs["reward_vec"]`` for TorchRL compatibility.

    Args:
        env: A gymnasium env returning ``info["reward_components"]``.
        component_names: Ordered list of component names.
        weights: Optional weights for overriding the scalar reward.
    """

    def __init__(
        self,
        env: gym.Env,
        component_names: list[str],
        weights: list[float] | None = None,
    ):
        super().__init__(env)
        self.component_names = component_names
        self._weights = np.array(weights, dtype=np.float32) if weights is not None else None
        n = len(component_names)

        obs_space = env.observation_space
        if isinstance(obs_space, spaces.Dict):
            new_spaces = dict(obs_space.spaces)
        else:
            new_spaces = {"observation": obs_space}
        new_spaces["reward_vec"] = spaces.Box(low=-np.inf, high=np.inf, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Dict(new_spaces)
        # Remove mo-gymnasium's vector reward_space so TorchRL treats
        # the reward as scalar (the components are in obs["reward_vec"])
        self.reward_space = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        components = info.get("reward_components", {})
        reward_vec = np.array(
            [components.get(n, 0.0) for n in self.component_names],
            dtype=np.float32,
        )

        if not isinstance(obs, dict):
            obs = {"observation": obs}
        obs["reward_vec"] = reward_vec

        if self._weights is not None:
            reward = float(np.dot(reward_vec, self._weights))

        return obs, float(reward), terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if not isinstance(obs, dict):
            obs = {"observation": obs}
        obs["reward_vec"] = np.zeros(len(self.component_names), dtype=np.float32)
        return obs, info


class DiscreteAdapter(gym.Wrapper):
    """Convert one-hot action tensors to int for gymnasium Discrete envs.

    TorchRL's ``ProbabilisticActor`` with ``OneHotCategorical`` outputs
    one-hot vectors.  This wrapper converts them to integer indices before
    passing to the underlying gymnasium environment.
    """

    def step(self, action):
        if isinstance(action, np.ndarray) and action.ndim >= 1 and action.shape[-1] > 1:
            action = int(np.argmax(action))
        elif torch.is_tensor(action):
            action = int(torch.argmax(action).item())
        return self.env.step(action)


class ContinuousAdapter(gym.Wrapper):
    """Squeeze batch dim + clip actions + detect NaN for continuous envs.

    Defenses against MuJoCo NaN instability:
    1. Squeeze TorchRL's extra batch dimension
    2. Clip actions to action_space bounds (prevents MuJoCo explosion)
    3. Replace NaN actions with zeros (prevents simulator crash)
    4. Detect NaN observations and auto-reset (prevents NaN propagation)
    """

    def step(self, action):
        if isinstance(action, np.ndarray) and action.ndim > 1:
            action = action.squeeze(0)
        elif torch.is_tensor(action):
            action = action.squeeze(0).detach().cpu().numpy()

        # Replace NaN/Inf actions with zeros (prevents MuJoCo crash)
        if np.any(~np.isfinite(action)):
            action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)

        # Clip actions to space bounds (critical for MuJoCo stability)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Detect NaN observations → force reset
        obs_array = obs["observation"] if isinstance(obs, dict) else obs
        if np.any(~np.isfinite(obs_array)):
            obs, info = self.env.reset()
            reward = 0.0
            terminated = True

        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------


def make_single_env(
    env_name: str,
    render_mode: str | None = None,
    weights: list[float] | None = None,
):
    """Create a single TorchRL-ready environment from ENV_REGISTRY.

    Returns a ``GymWrapper``'d environment with reward components embedded
    in the observation space as ``obs["reward_vec"]``.

    Args:
        env_name: Name from ENV_REGISTRY (e.g. "lunarlander", "cartpole").
        render_mode: "human" for live rendering, None for headless.
        weights: Optional per-component reward weights.
    """
    from torchrl.envs import GymWrapper

    from ..env_config import ENV_REGISTRY

    if env_name not in ENV_REGISTRY:
        available = ", ".join(sorted(ENV_REGISTRY.keys()))
        raise KeyError(f"Unknown environment '{env_name}'. Available: {available}")

    env_cfg = ENV_REGISTRY[env_name]
    make_kwargs = {}
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode

    # Create base gymnasium env with appropriate wrapper
    is_mo = env_cfg.use_mo_gym or env_cfg.env_id.startswith("mo-")

    if is_mo:
        # mo-gym env: use RewardVecWrapper directly on vector reward
        import mo_gymnasium as mo_gym

        base_env = mo_gym.make(env_cfg.env_id, **make_kwargs)
        base_env = RewardVecWrapper(base_env, len(env_cfg.reward_components), weights)
    elif env_cfg.wrapper_fn is not None:
        base_env = gym.make(env_cfg.env_id, **make_kwargs)
        base_env = env_cfg.wrapper_fn(base_env)
        base_env = ScalarComponentWrapper(base_env, env_cfg.reward_components, weights)
    else:
        raise ValueError(
            f"Environment '{env_name}' is not supported for TorchRL (no mo-gym vector reward and no component wrapper)"
        )

    # Apply max episode steps
    if env_cfg.max_episode_steps is not None:
        base_env = gym.wrappers.TimeLimit(base_env, max_episode_steps=env_cfg.max_episode_steps)

    # Action adapter: Discrete → int conversion, Continuous → squeeze batch dim
    unwrapped_action_space = base_env.unwrapped.action_space
    if isinstance(unwrapped_action_space, spaces.Discrete):
        base_env = DiscreteAdapter(base_env)
    elif isinstance(unwrapped_action_space, spaces.Box):
        base_env = ContinuousAdapter(base_env)

    # Wrap for TorchRL
    env = GymWrapper(base_env)
    return env


def build_env(
    env_name: str,
    num_envs: int,
    render_mode: str | None = None,
    weights: list[float] | None = None,
    clip_obs: float = 10.0,
):
    """Create a parallel TorchRL environment with stability defenses.

    Uses ``ParallelEnv`` for multi-process env execution, maximizing
    CPU utilization during rollout collection.

    Defensive layers (matching CleanRL/SB3 best practices):
    1. Action clipping + NaN replacement (in ContinuousAdapter)
    2. NaN observation detection + auto-reset (in ContinuousAdapter)
    3. Observation clipping to [-clip_obs, clip_obs]

    Args:
        env_name: Name from ENV_REGISTRY.
        num_envs: Number of parallel environment instances.
        render_mode: "human" for rendering (forces num_envs=1).
        weights: Optional per-component reward weights.
        clip_obs: Clip observations to [-clip, clip]. Default 10.0.
    """
    from torchrl.envs import ParallelEnv

    if render_mode == "human" and num_envs > 1:
        print(f"  Warning: rendering forces num_envs=1 (was {num_envs})")
        num_envs = 1

    if num_envs == 1:
        env = make_single_env(env_name, render_mode=render_mode, weights=weights)
    else:

        def env_fn():
            return make_single_env(env_name, render_mode=render_mode, weights=weights)

        env = ParallelEnv(num_envs, env_fn)

    # Observation clipping (matches CleanRL/SB3 best practices)
    # Critical for MuJoCo: prevents extreme observations from causing
    # policy output explosion → MuJoCo simulator NaN.
    # Combined with ContinuousAdapter's action clipping + NaN detection,
    # this creates a full defensive stack against NaN propagation.
    if clip_obs is not None and clip_obs > 0:
        from torchrl.envs import ClipTransform, TransformedEnv

        env = TransformedEnv(
            env,
            ClipTransform(in_keys=["observation"], low=-clip_obs, high=clip_obs),
        )

    return env
