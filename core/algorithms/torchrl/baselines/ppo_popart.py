"""torchrl/ppo_popart.py: TorchRL PPO-PopArt — PopArt adaptive value rescaling.

Standard PPO with PopArt. No reward normalization — PopArt handles the
value-side scale adaptation by tracking running return statistics and
rescaling the value head.
"""

import torch
import torch.nn as nn

from .._base import TorchRLAlgorithm


class TorchRLPPOPopArt(TorchRLAlgorithm):
    """TorchRL PPO with PopArt adaptive value head rescaling."""

    is_self_normalizing = True
    _algo_type = "ppo"

    def __init__(self, *args, popart_beta: float = 3e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self._popart_beta = popart_beta
        self._popart_mu = 0.0
        self._popart_nu = 1.0
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
        # Standard GAE (no reward normalization)
        batch = advantage_module(batch)

        if "advantage" in batch:
            returns = batch.get("value_target", None)
            if returns is None:
                returns = batch["advantage"] + batch.get("state_value", torch.zeros_like(batch["advantage"]))

            flat_returns = returns.reshape(-1)
            new_mean = flat_returns.mean().item()
            new_nu = (flat_returns**2).mean().item()

            old_mu = self._popart_mu
            old_sigma = max((self._popart_nu - self._popart_mu**2) ** 0.5, 1e-6)

            beta = self._popart_beta
            self._popart_mu = (1 - beta) * self._popart_mu + beta * new_mean
            self._popart_nu = (1 - beta) * self._popart_nu + beta * new_nu
            new_sigma = max((self._popart_nu - self._popart_mu**2) ** 0.5, 1e-6)

            head = self._get_value_head()
            if head is not None and self._popart_initialized:
                with torch.no_grad():
                    head.weight.mul_(old_sigma / new_sigma)
                    head.bias.copy_((old_sigma * head.bias + old_mu - self._popart_mu) / new_sigma)

            self._popart_initialized = True
            batch["advantage"] = (returns - self._popart_mu) / new_sigma - batch.get(
                "state_value", torch.zeros_like(returns)
            )

        return batch


if __name__ == "__main__":
    model = TorchRLPPOPopArt(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
