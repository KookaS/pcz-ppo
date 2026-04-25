"""Checkpoint save/load for TorchRL training.

Saves policy, value, and optimizer state dicts alongside the current
training step.  The renderer process uses ``load_checkpoint`` to hot-reload
the latest policy for decoupled visualization.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def save_checkpoint(
    policy,
    value,
    optimizer,
    step: int,
    checkpoint_dir: str = "checkpoints/torchrl",
):
    """Save training checkpoint to ``checkpoint_dir/latest.pt``."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(
        {
            "policy": policy.state_dict(),
            "value": value.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


def load_checkpoint(policy, path: str) -> int:
    """Load policy weights from a checkpoint. Returns the training step."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    policy.load_state_dict(checkpoint["policy"])
    return checkpoint["step"]
