"""Configuration dataclass for TorchRL experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TorchRLConfig:
    """Hyperparameters for TorchRL PPO/GRPO training."""

    # Algorithm
    algo: str = "ppo"  # "ppo" or "grpo"
    group_size: int = 8  # GRPO trajectory group size

    # Environment (resolved from ENV_REGISTRY)
    env_name: str = "lunarlander"
    num_envs: int = 8
    seed: int = 42

    # Network
    hidden_size: int = 64
    policy_gain: float = 0.01  # Orthogonal init gain for policy head
    znorm_clip: float | None = None  # Clip z-normed rewards to [-clip, clip]
    variance_floor: float = 1e-8  # Min variance for z-norm denominator
    component_gating: bool = False  # Zero-out near-constant components in z-norm

    # Training
    lr: float = 3e-4
    adam_eps: float = 1e-8  # PyTorch default; SB3 uses 1e-5 but it hurts GDPO
    lr_anneal: bool = False  # Linear LR annealing to 0
    activation: str = "leaky_relu"  # "leaky_relu" or "tanh"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    entropy_coeff_schedule: tuple | None = None  # (start, end) for annealing
    entropy_coeff_schedule_type: str = "linear"  # "linear" or "cosine"
    max_grad_norm: float = 0.5
    frames_per_batch: int = 8192
    total_frames: int = 8192 * 500  # ~4M
    num_epochs: int = 8
    minibatch_size: int = 256
    normalize_advantage: bool = True

    # Device
    device: str = "cpu"

    # Checkpointing
    checkpoint_dir: str = "checkpoints/torchrl"

    # Resolved from ENV_REGISTRY (set by __main__.py)
    env_id: str = ""
    reward_dim: int = 0
    component_names: list[str] = field(default_factory=list)
    component_weights: list[float] | None = None
