"""Decoupled renderer for TorchRL training.

Runs as a separate process from training, hot-reloading the latest policy
checkpoint at regular intervals.  This keeps rendering completely decoupled
from the training loop, maximizing training throughput.

Usage::

    # Standalone (while training runs in another terminal):
    cd /workspace
    uv run python -m core.torchrl.render \\
        --env=lunarlander --checkpoint-dir=checkpoints/torchrl

    # Automatically via --render flag in training:
    uv run python -m core.torchrl --algo=ppo --render
"""

from __future__ import annotations

import argparse
import os
import time


def run_renderer(
    env_name: str,
    checkpoint_dir: str,
    hidden_size: int = 64,
    reload_interval: float = 30.0,
):
    """Main rendering loop with checkpoint hot-reload.

    Args:
        env_name: Environment name from ENV_REGISTRY.
        checkpoint_dir: Directory containing ``latest.pt``.
        hidden_size: MLP hidden size (must match training).
        reload_interval: Seconds between checkpoint reload attempts.
    """
    from .checkpoint import load_checkpoint
    from .config import TorchRLConfig
    from .models import build_models

    checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")

    # Wait for first checkpoint
    print(f"[Render] Waiting for checkpoint at {checkpoint_path}...")
    while not os.path.exists(checkpoint_path):
        time.sleep(2.0)
    print("[Render] Checkpoint found. Starting rendering.")

    # Build env with rendering (includes obs clipping for MuJoCo stability)
    from .env import build_env

    env = build_env(env_name, num_envs=1, render_mode="human")

    # Build models in deterministic mode
    cfg = TorchRLConfig(env_name=env_name, hidden_size=hidden_size, num_envs=1)
    policy, _ = build_models(env, cfg, "cpu", deterministic=True)

    # Load initial checkpoint
    last_reload = 0.0
    last_step = -1

    td = env.reset()
    ep_reward = 0.0
    ep = 0

    try:
        while True:
            # Hot-reload checkpoint periodically
            now = time.time()
            if now - last_reload > reload_interval:
                try:
                    step = load_checkpoint(policy, checkpoint_path)
                    if step != last_step:
                        print(f"[Render] Loaded checkpoint at step {step:,}")
                        last_step = step
                except Exception:
                    pass
                last_reload = now

            # Step
            td = policy(td)
            td = env.step(td)

            r = td["next", "reward"].item()
            done = td["next", "done"].item()
            ep_reward += r

            if done:
                print(f"[Render] Episode {ep} | Return: {ep_reward:.2f}")
                td = env.reset()
                ep_reward = 0.0
                ep += 1
            else:
                td = td["next"]

    except KeyboardInterrupt:
        print("\n[Render] Stopped.")
    finally:
        env.close()


def main():
    """CLI entry point for standalone rendering."""
    parser = argparse.ArgumentParser(description="Decoupled TorchRL renderer (hot-reloads checkpoints)")
    parser.add_argument(
        "--env",
        type=str,
        default="lunarlander",
        help="Environment name from ENV_REGISTRY.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/torchrl",
        help="Directory containing latest.pt checkpoint.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="MLP hidden size (must match training config).",
    )
    parser.add_argument(
        "--reload-interval",
        type=float,
        default=30.0,
        help="Seconds between checkpoint reload attempts.",
    )
    args = parser.parse_args()

    run_renderer(
        env_name=args.env,
        checkpoint_dir=args.checkpoint_dir,
        hidden_size=args.hidden_size,
        reload_interval=args.reload_interval,
    )


if __name__ == "__main__":
    main()
