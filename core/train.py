"""train.py: Training entrypoint for PCZ-PPO algorithm comparison.

Trains one algorithm on one environment with MLflow logging, evaluation
callbacks, and model checkpointing.  Used standalone or invoked by
``compare.py`` for multi-algorithm grid comparisons.

Usage::

    # Single algorithm on CartPole (default)
    uv run python -m core.train --algorithm=pcz-ppo --total-timesteps=50000

    # Single algorithm on SuperMario
    uv run python -m core.train --algorithm=pcz-ppo --env=supermario

    # With MLflow
    uv run python -m core.train --algorithm=pcz-ppo \\
        --total-timesteps=100000 \\
        --mlflow-tracking-uri http://127.0.0.1:5050

    # Evaluation only
    uv run python -m core.train --eval-only \\
        --checkpoint=runs/best_model/best_model.zip
"""

import inspect
import json
import logging
import os
import resource
import sys
from datetime import datetime


def _enforce_memory_limit():
    """Set per-process memory limit to prevent OOM crashes.

    Uses RLIMIT_AS to cap virtual address space at 10GB. If exceeded,
    Python raises MemoryError (graceful) instead of triggering the
    Linux OOM killer (which crashes WSL). This limit propagates to
    all child processes and threads automatically.

    Confirmed to work through `uv run` (tested 2026-04-13).
    """
    max_bytes = int(os.environ.get("MEMORY_LIMIT_GB", "10")) * 1024**3
    try:
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, resource.RLIM_INFINITY))
    except (ValueError, OSError):
        pass  # Non-fatal — limit not supported on this platform


_enforce_memory_limit()
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
)

from . import ALGORITHM_REGISTRY
from .env_config import EnvConfig, make_env_factory
from .mlflow_utils import MLflowCallback, setup_mlflow, teardown_mlflow
from .parser import SELF_NORMALIZING_ALGORITHMS, parse_args

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def setup_environment(
    env_config: EnvConfig,
    n_envs: int,
    seed: int = 42,
    gamma: float = 0.99,
    norm_reward: bool = True,
    rom_mode: str = "vanilla",
    render_mode: str | None = None,
    max_episode_steps: int | None = None,
    weighted_reward: list[float] | None = None,
):
    """Create vectorized env with monitoring and optional normalization.

    When render_mode is set and n_envs > 1, n_envs is forced to 1 with a
    warning — SB3's DummyVecEnv requires all sub-envs to share the same
    render_mode, and rendering N windows simultaneously is impractical.

    Args:
        weighted_reward: If set, wraps each sub-env with WeightedRewardWrapper
            so the scalar reward is ``sum(w_i * component_i)``. Used to give
            standard PPO the same priority signal as PCZ-PPO's component weights.
    """
    set_random_seed(seed)

    if render_mode and n_envs > 1:
        logger.warning("Rendering requires n_envs=1 (was %d). Forcing n_envs=1.", n_envs)
        print(f"  Warning: --render forces n_envs=1 (was {n_envs})")
        n_envs = 1

    base_factory = make_env_factory(
        env_config.env_id,
        rom_mode=rom_mode,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )

    if weighted_reward is not None:
        from .algorithms._common import WeightedRewardWrapper

        component_names = env_config.reward_components

        def factory():
            env = base_factory()
            return WeightedRewardWrapper(env, component_names, weighted_reward)
    else:
        factory = base_factory

    vec_env = make_vec_env(factory, n_envs=n_envs)

    vec_env = VecNormalize(
        vec_env,
        norm_obs=env_config.norm_obs,
        norm_reward=norm_reward,
        clip_obs=10.0,
        clip_reward=10.0 if norm_reward else float("inf"),
        gamma=gamma,
        epsilon=1e-8,
    )
    return vec_env


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class SaveVecNormalizeCallback(BaseCallback):
    """Save VecNormalize stats alongside best model."""

    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        best_reward = getattr(self.parent, "best_mean_reward", None)
        print(f"  New best model at step {self.num_timesteps:,} | mean reward: {best_reward:.2f}")
        env = self.model.get_env()
        if isinstance(env, VecNormalize):
            env.save(os.path.join(self.save_path, "best_vecnormalize.pkl"))
        return True


class EntCoefScheduleCallback(BaseCallback):
    """Interpolate model.ent_coef from *start* to *end* over training.

    Supports linear and cosine annealing schedules.  Cosine stays higher for
    longer and decays smoothly at the end, reducing the sharp entropy drop
    that can trigger late-stage policy collapse.

    Updates once per rollout.
    """

    def __init__(
        self,
        start: float,
        end: float,
        total_timesteps: int,
        schedule_type: str = "linear",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.start = start
        self.end = end
        self.total_timesteps = total_timesteps
        self.schedule_type = schedule_type

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        if self.schedule_type == "cosine":
            import math

            self.model.ent_coef = self.end + 0.5 * (self.start - self.end) * (1 + math.cos(math.pi * progress))
        else:
            self.model.ent_coef = self.start + (self.end - self.start) * progress
        return True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_single(args) -> dict[str, float] | None:
    """Train a single algorithm. Returns final eval metrics or None."""
    algorithm = args.algorithm

    # Dispatch TorchRL algorithms to separate training path
    ModelClass = ALGORITHM_REGISTRY[algorithm]
    if getattr(ModelClass, "is_tabular", False):
        return _train_single_tabular(args)
    if getattr(ModelClass, "is_torchrl", False):
        return _train_single_torchrl(args)

    env_cfg = args.env_config
    components = env_cfg.reward_components
    is_self_norm = algorithm in SELF_NORMALIZING_ALGORITHMS

    # Resolve ent_coef: if not explicitly set, use 0.01 for self-normalizing
    # algorithms (PCZ variants) and 0.0 for standard PPO.
    ent_coef_schedule = None
    if getattr(args, "ent_coef_schedule", None):
        parts = args.ent_coef_schedule.split(":")
        if len(parts) != 2:
            print("  Error: --ent-coef-schedule must be 'start:end' (e.g. '0.1:0.01')")
            sys.exit(1)
        ent_coef_start, ent_coef_end = float(parts[0]), float(parts[1])
        ent_coef = ent_coef_start  # initial value
        ent_coef_schedule = (ent_coef_start, ent_coef_end)
    else:
        ent_coef = args.ent_coef
        if ent_coef is None:
            ent_coef = 0.01 if is_self_norm else 0.0

    # Resolve component weights: CLI override > env config > None (equal)
    component_weights = None
    if getattr(args, "reward_component_weights", None):
        component_weights = [float(w) for w in args.reward_component_weights.split(",")]
        if len(component_weights) != len(components):
            print(
                f"  Error: --reward-component-weights has {len(component_weights)} "
                f"values but env has {len(components)} components ({components})"
            )
            sys.exit(1)
    elif env_cfg.reward_component_weights is not None:
        component_weights = list(env_cfg.reward_component_weights)

    # Run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.mlflow_run_name or f"{algorithm}_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "best_model"), exist_ok=True)

    # Print setup
    print(f"\n{'=' * 60}")
    print(f"  Algorithm:   {algorithm}")
    print(f"  Environment: {args.env} ({env_cfg.env_id})")
    print(f"  Components:  {components}")
    print(f"  Policy:      {env_cfg.policy}")
    print(f"  Timesteps:   {args.total_timesteps:,}")
    print(f"  Envs:        {args.n_envs}")
    print(f"  Seed:        {args.seed}")
    print(f"  Device:      {args.device}")
    print(
        f"  Norm reward: {'No (self-normalizing)' if is_self_norm else 'No (--no-reward-norm)' if getattr(args, 'no_reward_norm', False) else 'VecNormalize'}"
    )
    print(f"  Norm obs:    {env_cfg.norm_obs}")
    if ent_coef_schedule:
        sched_type = getattr(args, "ent_coef_schedule_type", "linear")
        print(f"  Ent coef:    {ent_coef_schedule[0]} → {ent_coef_schedule[1]} ({sched_type} schedule)")
    else:
        print(f"  Ent coef:    {ent_coef}")
    # Only display weights if the algorithm supports them
    ModelClass = ALGORITHM_REGISTRY[algorithm]
    sig = inspect.signature(ModelClass.__init__)
    algo_accepts_weights = "component_weights" in sig.parameters

    # For non-self-normalizing algorithms (PPO), apply component weights
    # at the environment level via WeightedRewardWrapper so PPO sees a
    # weighted scalar reward: VecNormalize(sum(w_i * component_i)).
    # Self-normalizing algorithms (GDPO variants) get weights post-normalization
    # in their collect_rollouts override.
    use_weighted_reward = component_weights is not None and not is_self_norm

    if component_weights is not None:
        weight_str = ", ".join(f"{n}={w:.1f}" for n, w in zip(components, component_weights))
        if use_weighted_reward:
            print(f"  Weights:     {weight_str} (env-level weighted reward)")
        elif algo_accepts_weights:
            print(f"  Weights:     {weight_str} (post-normalization)")
    print(f"  Output:      {run_dir}")
    print(f"{'=' * 60}\n")

    # Environment
    render_mode = "human" if getattr(args, "render", False) else None
    rom_mode = getattr(args, "mario_rom_mode", "vanilla")
    max_episode_steps = getattr(args, "max_episode_steps", None)
    vec_env = setup_environment(
        env_config=env_cfg,
        n_envs=args.n_envs,
        seed=args.seed,
        gamma=args.gamma,
        norm_reward=not is_self_norm and not getattr(args, "no_reward_norm", False),
        rom_mode=rom_mode,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        weighted_reward=component_weights if use_weighted_reward else None,
    )

    # MLflow
    mlflow_params = {
        "algorithm": algorithm,
        "env": args.env,
        "env_id": env_cfg.env_id,
        "reward_components": ",".join(components),
        "policy": env_cfg.policy,
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "ent_coef": ent_coef,
        "seed": args.seed,
        "device": args.device,
    }
    if component_weights is not None:
        mlflow_params["component_weights"] = ",".join(f"{w:.2f}" for w in component_weights)
    if use_weighted_reward:
        mlflow_params["weighted_reward"] = "true"
    if ent_coef_schedule:
        mlflow_params["ent_coef_schedule"] = f"{ent_coef_schedule[0]}:{ent_coef_schedule[1]}"
        mlflow_params["ent_coef_schedule_type"] = getattr(args, "ent_coef_schedule_type", "linear")
    mlflow_active = False
    if not args.no_mlflow and args.mlflow_tracking_uri:
        experiment = args.mlflow_experiment_name or "pcz-ppo"
        mlflow_active = setup_mlflow(
            args.mlflow_tracking_uri,
            experiment,
            run_name,
            mlflow_params,
        )

    # Model
    ModelClass = ALGORITHM_REGISTRY[algorithm]
    model_kwargs = dict(
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=os.path.join(run_dir, "tensorboard"),
        device=args.device,
        seed=args.seed,
        verbose=1,
    )

    # Add reward_component_names and component_weights if the algo accepts them
    sig = inspect.signature(ModelClass.__init__)
    if "reward_component_names" in sig.parameters:
        model_kwargs["reward_component_names"] = components
    if "component_weights" in sig.parameters and component_weights is not None and not use_weighted_reward:
        model_kwargs["component_weights"] = component_weights

    model = ModelClass(env_cfg.policy, vec_env, **model_kwargs)

    # Eval environment (separate instance with same wrapping, no rendering)
    # Eval env always uses raw (unweighted) rewards so that all algorithms
    # are evaluated on the same scale regardless of training-time weighting.
    eval_env = setup_environment(
        env_config=env_cfg,
        n_envs=1,
        seed=args.seed + 1000,
        gamma=args.gamma,
        norm_reward=not is_self_norm and not getattr(args, "no_reward_norm", False),
        rom_mode=rom_mode,
        render_mode=None,
        max_episode_steps=max_episode_steps,
    )
    # Match VecTransposeImage wrapping if model added it (CNN policies)
    if is_vecenv_wrapped(model.get_env(), VecTransposeImage):
        eval_env = VecTransposeImage(eval_env)

    # Callbacks
    eval_freq = max(args.eval_freq_episodes * args.n_steps, args.n_steps)
    best_cb = SaveVecNormalizeCallback(
        save_path=os.path.join(run_dir, "best_model"),
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best_model"),
        log_path=os.path.join(run_dir, "evaluations"),
        eval_freq=eval_freq,
        deterministic=True,
        n_eval_episodes=args.n_eval_episodes,
        callback_on_new_best=best_cb,
        verbose=1,
    )

    # MLflow step-level logging callback
    callbacks = [eval_cb]
    if ent_coef_schedule:
        callbacks.append(
            EntCoefScheduleCallback(
                start=ent_coef_schedule[0],
                end=ent_coef_schedule[1],
                total_timesteps=args.total_timesteps,
                schedule_type=getattr(args, "ent_coef_schedule_type", "linear"),
                verbose=0,
            )
        )
    if mlflow_active:
        callbacks.append(MLflowCallback(verbose=0))

    # Train
    print(f"Starting {algorithm} training...")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name=f"{algorithm}_run",
            progress_bar=True,
        )
        print(f"\n  {algorithm} training completed.")
    except KeyboardInterrupt:
        print(f"\n  {algorithm} training interrupted.")
        model.save(os.path.join(run_dir, "interrupted_model.zip"))

    # Ensure a model is saved
    best_zip = os.path.join(run_dir, "best_model", "best_model.zip")
    if not os.path.exists(best_zip):
        model.save(best_zip)
        env = model.get_env()
        if isinstance(env, VecNormalize):
            env.save(os.path.join(run_dir, "best_model", "best_vecnormalize.pkl"))
        print(f"  Saved final model as best: {best_zip}")

    # Evaluate
    eval_metrics = None
    if not args.no_eval:
        print(f"\n  Evaluating {algorithm}...")
        # Use eval_env for final evaluation so the reward is always on
        # the raw (unweighted) scale, regardless of training-time weighting.
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
        )
        eval_metrics = {
            "eval/mean_reward": float(mean_reward),
            "eval/std_reward": float(std_reward),
        }
        print(f"  Eval: mean={mean_reward:.2f} +/- {std_reward:.2f}")

    # Save run metadata
    meta = {**mlflow_params, "eval": eval_metrics, "run_dir": run_dir}
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    teardown_mlflow(mlflow_active, run_dir, eval_metrics)
    vec_env.close()
    eval_env.close()

    return eval_metrics


def _train_single_tabular(args) -> dict[str, float] | None:
    """Train a tabular algorithm (e.g. Q-learning). Returns eval metrics."""
    algorithm = args.algorithm
    env_cfg = args.env_config
    components = env_cfg.reward_components

    # Resolve component weights
    component_weights = None
    if getattr(args, "reward_component_weights", None):
        component_weights = [float(w) for w in args.reward_component_weights.split(",")]
        if len(component_weights) != len(components):
            print(
                f"  Error: --reward-component-weights has {len(component_weights)} "
                f"values but env has {len(components)} components ({components})"
            )
            sys.exit(1)
    elif env_cfg.reward_component_weights is not None:
        component_weights = list(env_cfg.reward_component_weights)

    # Run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.mlflow_run_name or f"{algorithm}_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # MLflow params
    ModelClass = ALGORITHM_REGISTRY[algorithm]
    mlflow_params = {
        "algorithm": algorithm,
        "framework": "tabular",
        "env": args.env,
        "env_id": env_cfg.env_id,
        "reward_components": ",".join(components),
        "total_timesteps": args.total_timesteps,
        "gamma": args.gamma,
        "seed": args.seed,
    }
    if component_weights is not None:
        mlflow_params["component_weights"] = ",".join(f"{w:.2f}" for w in component_weights)

    mlflow_active = False
    if not args.no_mlflow and args.mlflow_tracking_uri:
        experiment = args.mlflow_experiment_name or "pcz-ppo"
        mlflow_active = setup_mlflow(
            args.mlflow_tracking_uri,
            experiment,
            run_name,
            mlflow_params,
        )

    # MLflow log callback
    log_fn = None
    if mlflow_active:
        import mlflow as _mlflow

        def _log_fn(metrics, step):
            try:
                finite_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and np.isfinite(v)}
                if finite_metrics:
                    _mlflow.log_metrics(finite_metrics, step=step)
            except Exception:
                pass

        log_fn = _log_fn

    # Create and train
    model = ModelClass(
        args.env,
        reward_component_names=components,
        component_weights=component_weights,
        gamma=args.gamma,
        seed=args.seed,
    )
    eval_metrics = model.learn(
        total_timesteps=args.total_timesteps,
        log_fn=log_fn,
    )

    # Post-training evaluation
    if not args.no_eval:
        print(f"\n  Evaluating {algorithm}...")
        eval_result = model.evaluate(n_episodes=args.n_eval_episodes)
        eval_metrics = eval_result
        print(f"  Eval: mean={eval_result['eval/mean_reward']:.2f} +/- {eval_result['eval/std_reward']:.2f}")

    # Save model
    model.save(os.path.join(run_dir, "best_model"))

    # Save metadata
    meta = {**mlflow_params, "eval": eval_metrics, "run_dir": run_dir}
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    teardown_mlflow(mlflow_active, run_dir, eval_metrics)

    return eval_metrics


def _train_single_torchrl(args) -> dict[str, float] | None:
    """Train a TorchRL algorithm. Returns final eval metrics or None."""
    # Config Regression Gate (G3): refuse to launch if TorchRLConfig defaults
    # drifted from the committed baseline unless the operator acknowledges via
    # --confirm-config-changes. Triggered before any heavy work happens.
    from .torchrl.config_gate import check as _config_gate_check

    _config_gate_check(confirm=args.confirm_config_changes)

    algorithm = args.algorithm
    env_cfg = args.env_config
    components = env_cfg.reward_components

    # Resolve component weights
    component_weights = None
    if getattr(args, "reward_component_weights", None):
        component_weights = [float(w) for w in args.reward_component_weights.split(",")]
        if len(component_weights) != len(components):
            print(
                f"  Error: --reward-component-weights has {len(component_weights)} "
                f"values but env has {len(components)} components ({components})"
            )
            sys.exit(1)
    elif env_cfg.reward_component_weights is not None:
        component_weights = list(env_cfg.reward_component_weights)

    # Resolve ent_coef and schedule
    ModelClass = ALGORITHM_REGISTRY[algorithm]
    is_self_norm = getattr(ModelClass, "is_self_normalizing", False)
    ent_coef_schedule = None
    if getattr(args, "ent_coef_schedule", None):
        parts = args.ent_coef_schedule.split(":")
        if len(parts) != 2:
            print("  Error: --ent-coef-schedule must be 'start:end' (e.g. '0.1:0.01')")
            sys.exit(1)
        ent_coef_start, ent_coef_end = float(parts[0]), float(parts[1])
        ent_coef = ent_coef_start
        ent_coef_schedule = (ent_coef_start, ent_coef_end)
    else:
        ent_coef = args.ent_coef
        if ent_coef is None:
            ent_coef = 0.01 if is_self_norm else 0.0

    # Run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.mlflow_run_name or f"{algorithm}_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Map CLI args to TorchRL config
    # SB3 n_steps is per-env; TorchRL frames_per_batch is total across envs
    frames_per_batch = args.n_steps * args.n_envs

    config_kwargs = dict(
        num_envs=args.n_envs,
        seed=args.seed,
        lr=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=0.2,
        entropy_coeff=ent_coef,
        entropy_coeff_schedule=ent_coef_schedule,
        entropy_coeff_schedule_type=getattr(args, "ent_coef_schedule_type", "linear"),
        max_grad_norm=0.5,
        frames_per_batch=frames_per_batch,
        total_frames=args.total_timesteps,
        num_epochs=args.n_epochs,
        minibatch_size=args.batch_size,
        device=args.device,
        checkpoint_dir=os.path.join(run_dir, "checkpoints"),
    )
    if getattr(args, "policy_gain", None) is not None:
        config_kwargs["policy_gain"] = args.policy_gain
    if getattr(args, "znorm_clip", None) is not None:
        config_kwargs["znorm_clip"] = args.znorm_clip
    if getattr(args, "adam_eps", None) is not None:
        config_kwargs["adam_eps"] = args.adam_eps
    if getattr(args, "lr_anneal", False):
        config_kwargs["lr_anneal"] = True
    if getattr(args, "activation", None) is not None:
        config_kwargs["activation"] = args.activation
    if getattr(args, "hidden_size", None) is not None:
        config_kwargs["hidden_size"] = args.hidden_size
    if getattr(args, "variance_floor", None) is not None:
        config_kwargs["variance_floor"] = args.variance_floor
    if getattr(args, "component_gating", False):
        config_kwargs["component_gating"] = True
        # Auto-set meaningful floor if user didn't specify one
        if getattr(args, "variance_floor", None) is None:
            config_kwargs["variance_floor"] = 0.01

    # Print schedule info
    if ent_coef_schedule:
        sched_type = getattr(args, "ent_coef_schedule_type", "linear")
        print(f"  Ent coef:    {ent_coef_schedule[0]} → {ent_coef_schedule[1]} ({sched_type} schedule)")
    else:
        print(f"  Ent coef:    {ent_coef}")

    # MLflow setup
    mlflow_params = {
        "algorithm": algorithm,
        "framework": "torchrl",
        "env": args.env,
        "env_id": env_cfg.env_id,
        "reward_components": ",".join(components),
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "frames_per_batch": frames_per_batch,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_epsilon": config_kwargs.get("clip_epsilon", 0.2),
        "num_epochs": config_kwargs.get("num_epochs", 8),
        "minibatch_size": config_kwargs.get("minibatch_size", 256),
        "max_grad_norm": config_kwargs.get("max_grad_norm", 0.5),
        "ent_coef": ent_coef,
        "seed": args.seed,
        "device": args.device,
        "hidden_size": config_kwargs.get("hidden_size", 64),
        "activation": config_kwargs.get("activation", "leaky_relu"),
        "adam_eps": config_kwargs.get("adam_eps", 1e-8),
        "lr_anneal": config_kwargs.get("lr_anneal", False),
        "normalize_advantage": config_kwargs.get("normalize_advantage", True),
    }
    if ent_coef_schedule:
        mlflow_params["ent_coef_schedule"] = f"{ent_coef_schedule[0]}:{ent_coef_schedule[1]}"
        mlflow_params["ent_coef_schedule_type"] = getattr(args, "ent_coef_schedule_type", "linear")
    if component_weights is not None:
        mlflow_params["component_weights"] = ",".join(f"{w:.2f}" for w in component_weights)

    mlflow_active = False
    if not args.no_mlflow and args.mlflow_tracking_uri:
        experiment = args.mlflow_experiment_name or "pcz-ppo"
        mlflow_active = setup_mlflow(
            args.mlflow_tracking_uri,
            experiment,
            run_name,
            mlflow_params,
        )

    # MLflow log callback
    log_fn = None
    if mlflow_active:
        import mlflow as _mlflow

        def _log_fn(metrics, step):
            try:
                finite_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and np.isfinite(v)}
                if finite_metrics:
                    _mlflow.log_metrics(finite_metrics, step=step)
            except Exception:
                pass

        log_fn = _log_fn

    # Create algorithm instance
    model = ModelClass(
        args.env,
        reward_component_names=components,
        component_weights=component_weights,
        **config_kwargs,
    )

    # Train
    render = getattr(args, "render", False)
    model.learn(
        total_frames=args.total_timesteps,
        render=render,
        log_fn=log_fn,
    )

    # Evaluate trained policy
    eval_metrics = None
    if not args.no_eval:
        from .torchrl.training import evaluate_policy as torchrl_evaluate

        print(f"\n  Evaluating {algorithm} (TorchRL)...")
        n_eval = getattr(args, "n_eval_episodes", 10)
        eval_metrics = torchrl_evaluate(
            model.policy,
            args.env,
            n_episodes=n_eval,
            seed=args.seed + 1000,
        )
        print(f"  Eval: mean={eval_metrics['eval/mean_reward']:.2f} +/- {eval_metrics['eval/std_reward']:.2f}")

    # Save run metadata
    meta = {**mlflow_params, "eval": eval_metrics, "run_dir": run_dir}
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    teardown_mlflow(mlflow_active, run_dir, eval_metrics)

    return eval_metrics


# ---------------------------------------------------------------------------
# Evaluation only
# ---------------------------------------------------------------------------


def evaluate_only(args):
    """Load a checkpoint and evaluate without training."""
    if not args.checkpoint:
        print("Error: --checkpoint is required for --eval-only")
        sys.exit(1)

    algorithm = args.algorithm
    env_cfg = args.env_config
    is_self_norm = algorithm in SELF_NORMALIZING_ALGORITHMS

    render_mode = "human" if getattr(args, "render", False) else None
    rom_mode = getattr(args, "mario_rom_mode", "vanilla")
    max_episode_steps = getattr(args, "max_episode_steps", None)
    vec_env = setup_environment(
        env_config=env_cfg,
        n_envs=1,
        seed=args.seed,
        gamma=args.gamma,
        norm_reward=not is_self_norm,
        rom_mode=rom_mode,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )

    # Load VecNormalize stats if available
    checkpoint_dir = str(Path(args.checkpoint).parent)
    vecnorm_path = os.path.join(checkpoint_dir, "best_vecnormalize.pkl")
    if os.path.exists(vecnorm_path) and isinstance(vec_env, VecNormalize):
        vec_env = VecNormalize.load(vecnorm_path, vec_env.venv)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"  Loaded VecNormalize stats from {vecnorm_path}")

    # SB3's .load() calls __init__ which may require reward_component_names.
    # Use the base PPO loader then set the env — avoids the constructor check.
    from stable_baselines3 import PPO

    model = PPO.load(args.checkpoint, env=vec_env, device=args.device)
    print(f"  Loaded model from {args.checkpoint} (eval mode, base PPO)")

    mean_reward, std_reward = evaluate_policy(
        model,
        vec_env,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )
    print(f"\n  Evaluation: mean={mean_reward:.2f} +/- {std_reward:.2f}")
    vec_env.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Expose the RL seed to env factories
    # before any env construction. The trading env reads PCZ_BASE_SEED to
    # derive per-worker data_seed deterministically across machines (previously
    # keyed on os.getpid() which varies by machine).
    os.environ["PCZ_BASE_SEED"] = str(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    if args.eval_only:
        evaluate_only(args)
    else:
        train_single(args)


if __name__ == "__main__":
    main()
