"""env_config.py: Environment configuration registry.

Each supported environment is registered with its reward components, wrapper,
policy type, and normalization settings.  This allows train.py and compare.py
to be fully env-agnostic — all env-specific logic lives here.

To add a new environment:
    1. Write a wrapper (in _common.py or here) that returns
       info["reward_components"] from step().
    2. Register it in ENV_REGISTRY below.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym


@dataclass(frozen=True)
class EnvConfig:
    """Configuration for a supported environment."""

    env_id: str
    """Gymnasium environment ID."""

    reward_components: list[str]
    """Names of reward components returned in info["reward_components"]."""

    policy: str = "MlpPolicy"
    """SB3 policy class name (MlpPolicy or CnnPolicy)."""

    norm_obs: bool = True
    """Whether to normalize observations with VecNormalize."""

    wrapper_fn: Callable[[gym.Env], gym.Env] | None = None
    """Optional wrapper applied after gym.make().  Must produce
    info["reward_components"] from step()."""

    max_episode_steps: int | None = None
    """If set, wrap the env in gymnasium.wrappers.TimeLimit."""

    description: str = ""
    """Short description for CLI help."""

    reward_component_weights: list[float] | None = None
    """Default post-normalization weights for each reward component.

    Applied after per-component z-normalization and before summation.
    Weights express *priority* (e.g. landing=5.0, fuel=0.5) while z-norm
    handles *scale* equalisation.  ``None`` means equal weights (all 1.0).
    Length must match ``reward_components``."""

    use_mo_gym: bool = False
    """Whether to use mo_gymnasium.make() instead of gymnasium.make()."""

    extra: str = ""
    """uv extra required to install dependencies (e.g. 'mario')."""


def _make_cartpole(env: gym.Env) -> gym.Env:
    from .algorithms._common import MultiComponentCartPole

    return MultiComponentCartPole(env)


def _make_resource(env: gym.Env) -> gym.Env:
    from .algorithms._common import MultiComponentResource

    return MultiComponentResource(env)


def _make_halfcheetah(env: gym.Env) -> gym.Env:
    from .algorithms._common import MultiComponentHalfCheetah

    return MultiComponentHalfCheetah(env)


def _make_halfcheetah_k4(env: gym.Env) -> gym.Env:
    from .algorithms._common import MultiComponentHalfCheetahK4

    return MultiComponentHalfCheetahK4(env)


def _make_halfcheetah_k6(env: gym.Env) -> gym.Env:
    from .algorithms._common import MultiComponentHalfCheetahK6

    return MultiComponentHalfCheetahK6(env)


def _make_halfcheetah_k8(env: gym.Env) -> gym.Env:
    from .algorithms._common import MultiComponentHalfCheetahK8

    return MultiComponentHalfCheetahK8(env)


def _make_humanoid(env: gym.Env) -> gym.Env:
    from .algorithms._common import MultiComponentHumanoid

    return MultiComponentHumanoid(env)


def _make_bipedalwalker(env: gym.Env) -> gym.Env:
    from .algorithms._common import MultiComponentBipedalWalker

    return MultiComponentBipedalWalker(env)


def _make_reacher(env: gym.Env) -> gym.Env:
    from .algorithms._common import MultiComponentReacher

    return MultiComponentReacher(env)


def _make_lunarlander(env: gym.Env) -> gym.Env:
    from .algorithms._common import MultiComponentLunarLander

    return MultiComponentLunarLander(env)


def _make_mountaincar(env: gym.Env) -> gym.Env:
    from .algorithms._common import MultiComponentMountainCar

    return MultiComponentMountainCar(env)


def _make_mountaincar_k6(env: gym.Env) -> gym.Env:
    from .algorithms._common import MultiComponentMountainCarK6

    return MultiComponentMountainCarK6(env)


def _make_mountaincar_k2(env: gym.Env) -> gym.Env:
    from .algorithms._common import KScalingMountainCarK2

    return KScalingMountainCarK2(env)


def _make_lunarlander_k6(env: gym.Env) -> gym.Env:
    """Create K=6 LunarLander via mo_gym (close the passed-in gym env)."""
    env.close()
    import mo_gymnasium as mo_gym

    from .algorithms._common import KScalingLunarLanderK6

    mo_env = mo_gym.make("mo-lunar-lander-v3")
    mo_env.unwrapped.reward_space = None
    return KScalingLunarLanderK6(mo_env)


def _make_lunarlander_k2(env: gym.Env) -> gym.Env:
    """Create K=2 LunarLander via mo_gym (close the passed-in gym env)."""
    env.close()
    import mo_gymnasium as mo_gym

    from .algorithms._common import KScalingLunarLanderK2

    mo_env = mo_gym.make("mo-lunar-lander-v3")
    # Remove mo_gym's vector reward_space from unwrapped env so TorchRL
    # treats the reward as scalar (GymWrapper checks env.unwrapped.reward_space)
    mo_env.unwrapped.reward_space = None
    return KScalingLunarLanderK2(mo_env)


def _make_lunarlander_k8(env: gym.Env) -> gym.Env:
    """Create K=8 LunarLander via mo_gym (close the passed-in gym env)."""
    env.close()
    import mo_gymnasium as mo_gym

    from .algorithms._common import KScalingLunarLanderK8

    mo_env = mo_gym.make("mo-lunar-lander-v3")
    mo_env.unwrapped.reward_space = None
    return KScalingLunarLanderK8(mo_env)


_trading_worker_counter = [0]
_hetero_trading_counter = [0]


def _make_hetero_trading(k: int):
    """Factory for heterogeneous-scale trading env."""

    def _factory(env: gym.Env) -> gym.Env:
        env.close()
        import os

        from .envs.trading import HeteroScaleTradingEnv

        worker_id = _hetero_trading_counter[0]
        _hetero_trading_counter[0] += 1
        base = os.environ.get("PCZ_BASE_SEED")
        if base is not None:
            data_seed = (int(base) * 1_000_003 + worker_id) & 0x7FFFFFFF
        else:
            data_seed = (hash((os.getpid(), worker_id)) & 0x7FFFFFFF) % 1_000_003
        return HeteroScaleTradingEnv(k=k, data_seed=data_seed)

    return _factory


def _make_trading(k: int):
    """Create factory for K-component trading env.

    Each worker gets a distinct data_seed so parallel workers see different
    OU trajectories. The base seed is read from the ``PCZ_BASE_SEED``
    environment variable (set by ``core.train.main`` from ``args.seed``) —
    this makes the per-worker data_seed bit-reproducible across machines
    (previously keyed on ``os.getpid()`` which varies by machine, breaking
    reproducibility even with identical ``--seed``).  If the env var is
    unset (e.g. when the env is built by a helper script), the PID fallback
    is preserved for backward compatibility.
    """

    def _factory(env: gym.Env) -> gym.Env:
        env.close()
        import os

        from .envs.trading import MultiComponentTradingEnv

        worker_id = _trading_worker_counter[0]
        _trading_worker_counter[0] += 1
        base = os.environ.get("PCZ_BASE_SEED")
        if base is not None:
            data_seed = (int(base) * 1_000_003 + worker_id) & 0x7FFFFFFF
        else:
            data_seed = (hash((os.getpid(), worker_id)) & 0x7FFFFFFF) % 1_000_003
        return MultiComponentTradingEnv(k=k, data_seed=data_seed)

    return _factory


def _make_mario_env(
    rom_mode: str = "vanilla",
    render_mode: str | None = None,
    max_episode_steps: int | None = None,
) -> gym.Env:
    """Create Super Mario Bros with the MultiComponentMario wrapper.

    Uses gym-super-mario-bros directly with a gymnasium-compatible wrapper.
    Falls back to SDL_VIDEODRIVER=dummy when no display is available
    (headless training). When render_mode="human", SDL opens a live window.

    Args:
        rom_mode: ROM mode string: vanilla, downsample, pixel, or rectangle.
        render_mode: "human" for live visualization, None for headless.
        max_episode_steps: Override max episode steps (passed to wrapper).
    """
    from .algorithms._common import MultiComponentMario

    kwargs = dict(rom_mode=rom_mode, render_mode=render_mode)
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    return MultiComponentMario(**kwargs)


# ---------------------------------------------------------------------------
# Environment Registry
# ---------------------------------------------------------------------------

ENV_REGISTRY: dict[str, EnvConfig] = {
    "cartpole": EnvConfig(
        env_id="CartPole-v1",
        reward_components=["balance", "center"],
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_cartpole,
        description="CartPole with balance/center reward decomposition",
    ),
    "supermario": EnvConfig(
        env_id="SuperMarioBros-v0",
        reward_components=["x_pos", "time", "death", "coin", "score"],
        policy="CnnPolicy",
        norm_obs=False,  # Don't normalize pixel observations
        wrapper_fn=None,  # Uses custom factory (_make_mario_env)
        description="Super Mario Bros with 5-component reward decomposition",
        extra="mario",
    ),
    "resource": EnvConfig(
        env_id="resource-gathering-v0",
        reward_components=["death_reward", "gold", "diamond"],
        reward_component_weights=[1.0, 1.0, 1.0],
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_resource,
        description="Resource Gathering with 3-component reward (death/gold/diamond)",
        use_mo_gym=True,
    ),
    "halfcheetah": EnvConfig(
        env_id="HalfCheetah-v4",
        reward_components=["run", "ctrl_cost"],
        reward_component_weights=None,  # Equal weights (low scale imbalance)
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_halfcheetah,
        max_episode_steps=1000,
        description="MuJoCo HalfCheetah with 2-component reward (run/ctrl_cost)",
    ),
    "halfcheetah-k4": EnvConfig(
        env_id="HalfCheetah-v4",
        reward_components=["velocity", "ctrl_j01", "ctrl_j23", "ctrl_j45"],
        reward_component_weights=None,
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_halfcheetah_k4,
        max_episode_steps=1000,
        description="HalfCheetah K=4 (velocity + 3 ctrl groups) for K-scaling",
    ),
    "halfcheetah-k6": EnvConfig(
        env_id="HalfCheetah-v4",
        reward_components=["velocity", "ctrl_j0", "ctrl_j1", "ctrl_j2", "ctrl_j3", "ctrl_j45"],
        reward_component_weights=None,
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_halfcheetah_k6,
        max_episode_steps=1000,
        description="HalfCheetah K=6 (velocity + 5 per-joint ctrl groups) for K-scaling",
    ),
    "halfcheetah-k8": EnvConfig(
        env_id="HalfCheetah-v4",
        reward_components=[
            "velocity_fwd",
            "velocity_bwd",
            "ctrl_bhip",
            "ctrl_bknee",
            "ctrl_bfoot",
            "ctrl_fhip",
            "ctrl_fknee",
            "ctrl_ffoot",
        ],
        reward_component_weights=None,
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_halfcheetah_k8,
        max_episode_steps=1000,
        description="HalfCheetah K=8 (velocity sign-split + 6 per-joint ctrl) for K-scaling",
    ),
    "humanoid": EnvConfig(
        env_id="Humanoid-v4",
        reward_components=["forward", "alive", "ctrl_cost"],
        reward_component_weights=[5.0, 1.0, 0.5],
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_humanoid,
        max_episode_steps=1000,
        description="MuJoCo Humanoid with 3-component reward (forward/alive/ctrl_cost)",
    ),
    "bipedalwalker": EnvConfig(
        env_id="BipedalWalker-v3",
        reward_components=["shaping", "energy", "crash"],
        reward_component_weights=[3.0, 0.5, 5.0],
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_bipedalwalker,
        max_episode_steps=1600,
        description="BipedalWalker with 3-component reward (shaping/energy/crash)",
    ),
    "reacher": EnvConfig(
        env_id="mo-reacher-v4",
        reward_components=["target_1", "target_2", "target_3", "target_4"],
        reward_component_weights=None,  # Equal weights (all same scale)
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_reacher,
        max_episode_steps=50,
        description="MO-Reacher with 4-component reward (distance to 4 targets)",
        use_mo_gym=True,
    ),
    "lunarlander": EnvConfig(
        env_id="mo-lunar-lander-v3",
        reward_components=["landing", "shaping", "fuel_main", "fuel_side"],
        reward_component_weights=[5.0, 3.0, 0.5, 0.5],
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_lunarlander,
        max_episode_steps=1000,
        description="MO-LunarLander with 4-component reward (landing/shaping/fuel)",
    ),
    # MountainCar K-scaling variants (second domain for K-scaling validation)
    "mountaincar": EnvConfig(
        env_id="MountainCar-v0",
        reward_components=["time", "velocity", "height", "goal"],
        reward_component_weights=[0.5, 3.0, 3.0, 10.0],
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_mountaincar,
        max_episode_steps=200,
        description="MountainCar K=4 with sparse/dense mismatch (time/velocity/height/goal)",
    ),
    "mountaincar-k6": EnvConfig(
        env_id="MountainCar-v0",
        reward_components=["time", "vel_right", "vel_left", "height_up", "height_down", "goal"],
        reward_component_weights=[0.5, 3.0, 3.0, 3.0, 3.0, 10.0],
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_mountaincar_k6,
        max_episode_steps=200,
        description="MountainCar K=6 (4 dense shaping + sparse goal) for K-scaling",
    ),
    "mountaincar-k2": EnvConfig(
        env_id="MountainCar-v0",
        reward_components=["dense", "goal"],
        reward_component_weights=None,
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_mountaincar_k2,
        max_episode_steps=200,
        description="MountainCar K=2 (dense aggregate + goal) for K-scaling",
    ),
    # K-scaling variants of LunarLander (same total reward, different decomposition)
    # Use LunarLander-v3 (not mo-) to avoid TorchRL's mo-gym reward vector handling.
    # The wrapper_fn creates mo_gym internally and applies the K-scaling decomposition.
    "lunarlander-k6": EnvConfig(
        env_id="LunarLander-v3",
        reward_components=[
            "landing",
            "distance",
            "velocity",
            "angle",
            "fuel_main",
            "fuel_side",
        ],
        reward_component_weights=[10.0, 3.0, 1.0, 1.0, 0.5, 0.5],  # proportional to K=4/K=8
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_lunarlander_k6,
        max_episode_steps=1000,
        description="LunarLander K=6 (shaping split into distance/velocity/angle) for K-scaling",
    ),
    "lunarlander-k2": EnvConfig(
        env_id="LunarLander-v3",
        reward_components=["landing", "dense"],
        reward_component_weights=None,  # Equal weights for K-scaling experiment
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_lunarlander_k2,
        max_episode_steps=1000,
        description="LunarLander K=2 (landing + dense aggregate) for K-scaling",
    ),
    "lunarlander-k8": EnvConfig(
        env_id="LunarLander-v3",
        reward_components=[
            "landing",
            "distance",
            "velocity",
            "angle",
            "leg_left",
            "leg_right",
            "fuel_main",
            "fuel_side",
        ],
        reward_component_weights=None,  # Equal weights for K-scaling experiment
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_lunarlander_k8,
        max_episode_steps=1000,
        description="LunarLander K=8 (shaping split into 5 sub-potentials) for K-scaling",
    ),
    "trading-k2": EnvConfig(
        env_id="CartPole-v1",  # placeholder — _make_trading closes it and creates trading env
        reward_components=["pnl", "costs"],
        reward_component_weights=None,
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_trading(2),
        max_episode_steps=4000,
        description="Financial trading K=2 (PnL + costs)",
        extra="trading",
    ),
    "trading-k3-clean": EnvConfig(
        env_id="CartPole-v1",
        reward_components=["pnl_gain", "pnl_loss", "txn_cost"],
        reward_component_weights=None,
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_trading(3),
        max_episode_steps=4000,
        description="Financial trading K=3 clean (K=4 minus borrow_cost/residual)",
        extra="trading",
    ),
    "trading-k4": EnvConfig(
        env_id="CartPole-v1",
        reward_components=["pnl_gain", "pnl_loss", "txn_cost", "borrow_cost"],
        reward_component_weights=None,
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_trading(4),
        max_episode_steps=4000,
        description="Financial trading K=4 (gains/losses + fee types)",
        extra="trading",
    ),
    "trading-k6": EnvConfig(
        env_id="CartPole-v1",
        reward_components=[
            "entry_pnl",
            "hold_pnl",
            "txn_cost",
            "borrow_cost",
            "spread_proxy",
            "residual",
        ],
        reward_component_weights=None,
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_trading(6),
        max_episode_steps=4000,
        description="Financial trading K=6 (action-split PnL + cost types)",
        extra="trading",
    ),
    "trading-k8": EnvConfig(
        env_id="CartPole-v1",
        reward_components=[
            "entry_gain",
            "entry_loss",
            "hold_gain",
            "hold_loss",
            "txn_cost",
            "borrow_cost",
            "spread_proxy",
            "residual",
        ],
        reward_component_weights=None,
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_trading(8),
        max_episode_steps=4000,
        description="Financial trading K=8 (full action×direction + cost decomposition)",
        extra="trading",
    ),
    # Heterogeneous-scale variants: per-step PnL O(1) + VaR breach -50 + bankruptcy -1000
    "trading-hk2": EnvConfig(
        env_id="CartPole-v1",
        reward_components=["pnl_step", "risk_events"],
        reward_component_weights=None,
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_hetero_trading(2),
        max_episode_steps=4000,
        description="Hetero-scale trading K=2 (pnl_step + risk_events)",
        extra="trading",
    ),
    "trading-hk4": EnvConfig(
        env_id="CartPole-v1",
        reward_components=["pnl_step", "costs", "var_breach", "bankruptcy"],
        reward_component_weights=None,
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_hetero_trading(4),
        max_episode_steps=4000,
        description="Hetero-scale trading K=4 (pnl + costs + var_breach + bankruptcy)",
        extra="trading",
    ),
    "trading-hk6": EnvConfig(
        env_id="CartPole-v1",
        reward_components=["pnl_gain", "pnl_loss", "txn_cost", "borrow_cost", "var_breach", "bankruptcy"],
        reward_component_weights=None,
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_hetero_trading(6),
        max_episode_steps=4000,
        description="Hetero-scale trading K=6 (sign-split PnL + costs + risk events)",
        extra="trading",
    ),
    "trading-hk8": EnvConfig(
        env_id="CartPole-v1",
        reward_components=[
            "entry_gain",
            "entry_loss",
            "hold_gain",
            "hold_loss",
            "txn_cost",
            "borrow_cost",
            "var_breach",
            "bankruptcy",
        ],
        reward_component_weights=None,
        policy="MlpPolicy",
        norm_obs=True,
        wrapper_fn=_make_hetero_trading(8),
        max_episode_steps=4000,
        description="Hetero-scale trading K=8 (action×sign PnL + costs + risk events)",
        extra="trading",
    ),
}


def make_env_factory(
    env_id_or_name: str,
    *,
    rom_mode: str = "vanilla",
    render_mode: str | None = None,
    max_episode_steps: int | None = None,
) -> Callable[[], gym.Env]:
    """Return a callable that creates one environment instance.

    Accepts either a registry name ('cartpole') or a gymnasium env ID
    ('CartPole-v1').  Looks up the config and returns the appropriate factory.

    Args:
        env_id_or_name: Registry name or gymnasium env ID.
        rom_mode: ROM mode for Super Mario Bros (vanilla/downsample/pixel/rectangle).
        render_mode: "human" for live visualization, None for headless.
        max_episode_steps: Override max episode steps. If None, uses config default.
    """
    # Resolve by name first, then by env_id
    if env_id_or_name in ENV_REGISTRY:
        config = ENV_REGISTRY[env_id_or_name]
        name = env_id_or_name
    else:
        # Search by env_id
        matches = {k: v for k, v in ENV_REGISTRY.items() if v.env_id == env_id_or_name}
        if matches:
            name, config = next(iter(matches.items()))
        else:
            raise KeyError(
                f"Unknown environment '{env_id_or_name}'. Available: {', '.join(sorted(ENV_REGISTRY.keys()))}"
            )

    # Resolve effective max_episode_steps: CLI override > config > None
    effective_max_steps = max_episode_steps if max_episode_steps is not None else config.max_episode_steps

    if name == "supermario":
        return lambda: _make_mario_env(
            rom_mode=rom_mode,
            render_mode=render_mode,
            max_episode_steps=effective_max_steps,
        )

    def _factory() -> gym.Env:
        make_kwargs = {}
        if render_mode is not None:
            make_kwargs["render_mode"] = render_mode
        # mo-gymnasium envs need mo_gym to create or register with gymnasium
        if config.use_mo_gym or config.env_id.startswith("mo-"):
            import mo_gymnasium as mo_gym

            env = mo_gym.make(config.env_id, **make_kwargs)
        else:
            env = gym.make(config.env_id, **make_kwargs)
        if config.wrapper_fn is not None:
            env = config.wrapper_fn(env)
        if effective_max_steps is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=effective_max_steps)
        return env

    return _factory


def get_env_config(env_name: str) -> EnvConfig:
    """Look up environment config by name.  Raises KeyError if not found."""
    if env_name not in ENV_REGISTRY:
        available = ", ".join(sorted(ENV_REGISTRY.keys()))
        raise KeyError(f"Unknown environment '{env_name}'. Available: {available}")
    return ENV_REGISTRY[env_name]
