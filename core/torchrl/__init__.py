"""TorchRL-based training for PCZ-PPO experiments.

Standalone training package using TorchRL as the RL backend.
Supports PPO and GRPO algorithms with per-component reward tracking,
parallel environments, and decoupled rendering.

Usage::

    cd /workspace
    uv run python -m core.torchrl --algo=ppo --env=lunarlander
    uv run python -m core.torchrl --algo=grpo --env=lunarlander --render
    uv run python -m core.torchrl --total-frames=500000
"""

from .checkpoint import load_checkpoint, save_checkpoint
from .config import TorchRLConfig

__all__ = ["TorchRLConfig", "load_checkpoint", "save_checkpoint"]
