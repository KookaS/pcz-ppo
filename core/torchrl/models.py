"""Neural network models for TorchRL PPO/GRPO training.

Provides a 3-layer MLP backbone and a model builder that creates
TorchRL-compatible actor (ProbabilisticActor) and critic (ValueOperator).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _ortho_init(module: nn.Module, gain: float = math.sqrt(2)) -> None:
    """Apply orthogonal initialization matching SB3 defaults."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _make_activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    return nn.LeakyReLU()


class MLP(nn.Module):
    """3-layer MLP with configurable activations and orthogonal init."""

    def __init__(self, input_dim: int, hidden_size: int, activation: str = "leaky_relu"):
        super().__init__()
        # Gain for orthogonal init: sqrt(2) for ReLU-like, 5/3 for Tanh
        gain = 5.0 / 3.0 if activation == "tanh" else math.sqrt(2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            _make_activation(activation),
            nn.Linear(hidden_size, hidden_size),
            _make_activation(activation),
            nn.Linear(hidden_size, hidden_size),
            _make_activation(activation),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                _ortho_init(m, gain=gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x.float())


def _is_continuous(action_spec) -> bool:
    """Check if action spec represents a continuous (Box) action space."""
    from torchrl.data import Bounded

    if isinstance(action_spec, Bounded):
        return action_spec.dtype.is_floating_point
    # Fallback: check for .n attribute (discrete has it, continuous doesn't)
    return not (hasattr(action_spec, "n") or (hasattr(action_spec, "space") and hasattr(action_spec.space, "n")))


def build_models(env, cfg, device, deterministic=False):
    """Build actor (policy) and critic (value) networks.

    Supports both discrete (OneHotCategorical) and continuous (TanhNormal)
    action spaces.

    Args:
        env: TorchRL environment (single or parallel).
        cfg: TorchRLConfig with ``hidden_size``.
        device: Torch device.
        deterministic: If True, use argmax/mode action selection (for evaluation).

    Returns:
        (policy, value) tuple of TorchRL modules.
    """
    from tensordict.nn import InteractionType, TensorDictModule
    from torchrl.modules import ProbabilisticActor, ValueOperator

    # Extract specs from env
    td = env.reset()
    obs_dim = td["observation"].shape[-1]
    action_spec = env.action_spec
    activation = getattr(cfg, "activation", "leaky_relu")
    policy_gain = getattr(cfg, "policy_gain", 0.01)

    continuous = _is_continuous(action_spec)

    # TanhNormal (continuous) has no analytical mode/mean — use RANDOM even
    # for deterministic eval (trained policies have low variance ≈ deterministic).
    # Discrete (OneHotCategorical) supports MODE for argmax action selection.
    if deterministic and not continuous:
        interaction = InteractionType.MODE
    else:
        interaction = InteractionType.RANDOM

    if continuous:
        # Continuous action space (e.g. MuJoCo)
        from torchrl.modules import TanhNormal

        action_dim = action_spec.shape[-1]

        # Actor outputs 2*action_dim: [mean, log_std] for each action
        actor_head = nn.Linear(cfg.hidden_size, 2 * action_dim)
        _ortho_init(actor_head, gain=policy_gain)
        actor_net = nn.Sequential(
            MLP(obs_dim, cfg.hidden_size, activation=activation),
            actor_head,
        ).to(device)

        from torchrl.modules import NormalParamExtractor

        actor_module = TensorDictModule(
            nn.Sequential(actor_net, NormalParamExtractor()),
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )

        # Use single-env bounds (not batched) for the distribution —
        # ParallelEnv repeats bounds per env (e.g. [4, 17]) but training
        # mini-batches have different batch sizes (e.g. [64, 17]).
        low = action_spec.space.low
        high = action_spec.space.high
        if low.dim() > 1:
            low = low[0]
            high = high[0]

        policy = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            spec=action_spec,
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": low,
                "high": high,
            },
            return_log_prob=not deterministic,
            default_interaction_type=interaction,
        )
    else:
        # Discrete action space (e.g. CartPole, LunarLander)
        from torchrl.modules.distributions import OneHotCategorical

        if hasattr(action_spec, "space") and hasattr(action_spec.space, "n"):
            n_actions = action_spec.space.n
        elif hasattr(action_spec, "n"):
            n_actions = action_spec.n
        else:
            raise ValueError(f"Cannot determine n_actions from action_spec: {action_spec}")

        actor_head = nn.Linear(cfg.hidden_size, n_actions)
        _ortho_init(actor_head, gain=policy_gain)
        actor_net = nn.Sequential(
            MLP(obs_dim, cfg.hidden_size, activation=activation),
            actor_head,
        ).to(device)

        actor_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"])

        policy = ProbabilisticActor(
            module=actor_module,
            in_keys=["logits"],
            spec=action_spec,
            distribution_class=OneHotCategorical,
            return_log_prob=not deterministic,
            default_interaction_type=interaction,
        )

    # Critic (same for both discrete and continuous)
    value_head = nn.Linear(cfg.hidden_size, 1)
    _ortho_init(value_head, gain=1.0)
    value_net = nn.Sequential(
        MLP(obs_dim, cfg.hidden_size, activation=activation),
        value_head,
    ).to(device)

    value = ValueOperator(module=value_net, in_keys=["observation"])

    return policy, value
