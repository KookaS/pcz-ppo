"""CLI entry point for TorchRL PPO/GRPO training.

Usage::

    cd /workspace

    # PPO on LunarLander (default)
    uv run python -m core.torchrl

    # GRPO on LunarLander
    uv run python -m core.torchrl --algo=grpo

    # With decoupled rendering (renderer runs as a subprocess)
    uv run python -m core.torchrl --render

    # Custom hyperparameters
    uv run python -m core.torchrl --algo=ppo --env=lunarlander \\
        --num-envs=8 --total-frames=500000 --hidden-size=128

    # CartPole
    uv run python -m core.torchrl --env=cartpole --total-frames=100000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import torch


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="TorchRL PPO/GRPO training on PCZ environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Algorithm
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "grpo"],
        help="Algorithm: ppo (with GAE) or grpo (critic-free). Default: ppo.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="GRPO trajectory group size. Default: 8.",
    )

    # Environment
    parser.add_argument(
        "--env",
        type=str,
        default="lunarlander",
        help="Environment name from ENV_REGISTRY. Default: lunarlander.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environments. Default: 8.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    parser.add_argument(
        "--reward-weights",
        type=str,
        default=None,
        help="Comma-separated component weights (e.g. '5.0,3.0,0.5,0.5').",
    )

    # Network
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="MLP hidden layer size. Default: 64.",
    )

    # Training
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coeff", type=float, default=0.5)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--frames-per-batch",
        type=int,
        default=8192,
        help="Frames collected per training iteration. Default: 8192.",
    )
    parser.add_argument(
        "--total-frames",
        type=int,
        default=8192 * 500,
        help="Total training frames. Default: ~4M.",
    )
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=256)

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )

    # Rendering (decoupled)
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Launch decoupled renderer as a subprocess.",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/torchrl",
        help="Checkpoint directory. Default: checkpoints/torchrl.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Check TorchRL availability
    try:
        import tensordict  # noqa: F401
        import torchrl  # noqa: F401
    except ImportError as e:
        print(f"TorchRL not installed: {e}\nInstall with: uv sync --extra torchrl")
        sys.exit(1)

    from ..env_config import ENV_REGISTRY
    from .checkpoint import save_checkpoint
    from .config import TorchRLConfig
    from .env import build_env
    from .models import build_models
    from .training import build_training, train_loop

    # Resolve environment config
    if args.env not in ENV_REGISTRY:
        available = ", ".join(sorted(ENV_REGISTRY.keys()))
        print(f"Unknown environment '{args.env}'. Available: {available}")
        sys.exit(1)

    env_cfg = ENV_REGISTRY[args.env]

    # Parse component weights
    component_weights = None
    if args.reward_weights:
        component_weights = [float(w) for w in args.reward_weights.split(",")]
        if len(component_weights) != len(env_cfg.reward_components):
            print(
                f"Error: --reward-weights has {len(component_weights)} values "
                f"but env has {len(env_cfg.reward_components)} components "
                f"({env_cfg.reward_components})"
            )
            sys.exit(1)
    elif env_cfg.reward_component_weights is not None:
        component_weights = list(env_cfg.reward_component_weights)

    # Build config
    cfg = TorchRLConfig(
        algo=args.algo,
        group_size=args.group_size,
        env_name=args.env,
        num_envs=args.num_envs,
        seed=args.seed,
        hidden_size=args.hidden_size,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coeff=args.value_coeff,
        entropy_coeff=args.entropy_coeff,
        max_grad_norm=args.max_grad_norm,
        frames_per_batch=args.frames_per_batch,
        total_frames=args.total_frames,
        num_epochs=args.num_epochs,
        minibatch_size=args.minibatch_size,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        env_id=env_cfg.env_id,
        reward_dim=len(env_cfg.reward_components),
        component_names=list(env_cfg.reward_components),
        component_weights=component_weights,
    )

    # Device
    device = torch.device(cfg.device)
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")

    torch.manual_seed(cfg.seed)

    # Print setup
    print(f"\n{'=' * 60}")
    print(f"  TorchRL {cfg.algo.upper()} Training")
    print(f"  Environment: {cfg.env_name} ({cfg.env_id})")
    print(f"  Components:  {cfg.component_names}")
    if cfg.component_weights:
        weight_str = ", ".join(f"{n}={w:.1f}" for n, w in zip(cfg.component_names, cfg.component_weights))
        print(f"  Weights:     {weight_str}")
    print(f"  Parallel:    {cfg.num_envs} envs")
    print(f"  Total frames: {cfg.total_frames:,}")
    print(f"  Batch size:  {cfg.frames_per_batch:,}")
    print(f"  Device:      {device}")
    print(f"  Render:      {'decoupled subprocess' if args.render else 'off'}")
    print(f"{'=' * 60}\n")

    # Build environment
    env = build_env(
        cfg.env_name,
        num_envs=cfg.num_envs,
        weights=cfg.component_weights,
    )

    # Build models
    policy, value = build_models(env, cfg, device)

    # Build training components
    loss_module, advantage_module, optimizer, collector = build_training(env, policy, value, cfg, device)

    # Checkpoint callback
    def _save_fn(pol, val, opt, step):
        save_checkpoint(pol, val, opt, step, cfg.checkpoint_dir)

    # Launch decoupled renderer subprocess
    render_proc = None
    if args.render:
        render_cmd = [
            sys.executable,
            "-m",
            "core.torchrl.render",
            f"--env={cfg.env_name}",
            f"--checkpoint-dir={cfg.checkpoint_dir}",
            f"--hidden-size={cfg.hidden_size}",
        ]
        print("  Launching renderer subprocess...")
        render_proc = subprocess.Popen(render_cmd)
        print(f"  Renderer PID: {render_proc.pid}")

    # Train
    try:
        t0 = time.time()
        total = train_loop(
            collector,
            loss_module,
            advantage_module,
            optimizer,
            cfg,
            policy,
            value,
            _save_fn,
        )
        elapsed = time.time() - t0

        print(f"\nTraining complete in {elapsed:.1f}s")
        print(f"Total frames: {total:,}")
        print(f"Avg FPS: {total / elapsed:.0f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        _save_fn(policy, value, optimizer, -1)
    finally:
        collector.shutdown()
        try:
            env.close()
        except RuntimeError:
            pass  # collector.shutdown() may have already closed env

        # Terminate renderer
        if render_proc is not None:
            print("  Stopping renderer...")
            render_proc.terminate()
            try:
                render_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                render_proc.kill()
                render_proc.wait(timeout=2)
            print("  Renderer stopped.")


if __name__ == "__main__":
    main()
