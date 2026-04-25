"""torchrl/pcz_ppo_popart.py: PCZ-PPO-PopArt — GDPO z-norm + PopArt value rescaling.

Combines per-component reward z-normalization with PopArt adaptive value
head rescaling. PopArt tracks running mean/std of returns and rescales
the value network's output layer to compensate for changing magnitudes.
"""

import torch
import torch.nn as nn

from .._base import TorchRLAlgorithm
from .._norm import weighted_sum, znorm


class TorchRLPCZPPOPopArt(TorchRLAlgorithm):
    """TorchRL PCZ-PPO + PopArt: per-component z-norm + adaptive value rescaling."""

    is_self_normalizing = True
    _algo_type = "ppo"

    def __init__(self, *args, popart_beta: float = 3e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self._popart_beta = popart_beta
        self._popart_mu = 0.0
        self._popart_nu = 1.0  # E[x^2]
        self._popart_initialized = False

    def _get_value_head(self):
        """Get the last linear layer of the value network."""
        modules = list(self.value.module.children())
        for m in reversed(modules):
            if isinstance(m, nn.Linear):
                return m
            if isinstance(m, nn.Sequential):
                for sub in reversed(list(m.children())):
                    if isinstance(sub, nn.Linear):
                        return sub
        return None

    def _compute_advantages(self, batch, advantage_module):
        # Step 1: GDPO per-component z-norm
        reward_vec = batch["next", "reward_vec"]
        D = reward_vec.shape[-1]
        normalized = torch.zeros_like(reward_vec)
        for i in range(D):
            normalized[..., i] = znorm(reward_vec[..., i])
        batch["next", "reward"] = weighted_sum(normalized, self._component_weights)

        # Step 2: Standard GAE
        batch = advantage_module(batch)

        # Step 3: PopArt — update stats and renormalize
        if "advantage" in batch:
            returns = batch.get("value_target", None)
            if returns is None:
                # Approximate returns from advantage + value
                returns = batch["advantage"] + batch.get("state_value", torch.zeros_like(batch["advantage"]))

            # Update running stats
            flat_returns = returns.reshape(-1)
            new_mean = flat_returns.mean().item()
            new_nu = (flat_returns**2).mean().item()

            old_mu = self._popart_mu
            old_sigma = max((self._popart_nu - self._popart_mu**2) ** 0.5, 1e-6)

            beta = self._popart_beta
            self._popart_mu = (1 - beta) * self._popart_mu + beta * new_mean
            self._popart_nu = (1 - beta) * self._popart_nu + beta * new_nu
            new_sigma = max((self._popart_nu - self._popart_mu**2) ** 0.5, 1e-6)

            # Rescale value head weights
            head = self._get_value_head()
            if head is not None and self._popart_initialized:
                with torch.no_grad():
                    head.weight.mul_(old_sigma / new_sigma)
                    head.bias.copy_((old_sigma * head.bias + old_mu - self._popart_mu) / new_sigma)

            self._popart_initialized = True

            # Normalize targets for training
            batch["advantage"] = (returns - self._popart_mu) / new_sigma - batch.get(
                "state_value", torch.zeros_like(returns)
            )

        return batch


if __name__ == "__main__":
    model = TorchRLPCZPPOPopArt(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
