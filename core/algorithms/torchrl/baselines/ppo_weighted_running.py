"""torchrl/ppo_weighted_running.py: PPO with running z-norm on weighted scalar.

Weighted sum of components → running EMA z-normalization → GAE.
NO per-component decomposition — tests whether running EMA on the aggregate
scalar is sufficient, or whether per-component z-norm adds value.

This is the most direct ablation of GDPO's decomposition claim:
- If this ≈ GDPO-running → decomposition unnecessary, running EMA is the key
- If this < GDPO-running → decomposition adds value beyond smoothing
"""

import torch

from .._base import TorchRLAlgorithm
from .._norm import weighted_sum


class TorchRLPPOWeightedRunning(TorchRLAlgorithm):
    """PPO with running EMA z-normalization on the weighted scalar reward."""

    is_self_normalizing = True
    _algo_type = "ppo"

    def __init__(self, *args, ema_alpha: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self._ema_alpha = ema_alpha
        self._running_mean = None
        self._running_var = None

    def _compute_advantages(self, batch, advantage_module):
        reward_vec = batch["next", "reward_vec"]

        # Step 1: Weighted sum FIRST (no per-component normalization)
        scalar = weighted_sum(reward_vec, self._component_weights)

        # Step 2: Running EMA z-normalization on the scalar
        if self._running_mean is None:
            self._running_mean = torch.tensor(0.0, device=scalar.device)
            self._running_var = torch.tensor(1.0, device=scalar.device)

        flat = scalar.reshape(-1)
        batch_mean = flat.mean()
        batch_var = flat.var()
        a = self._ema_alpha
        self._running_mean = (1 - a) * self._running_mean + a * batch_mean
        self._running_var = (1 - a) * self._running_var + a * batch_var

        safe_std = torch.sqrt(torch.clamp(self._running_var, min=1e-8))
        normalized = (scalar - self._running_mean) / safe_std

        batch["next", "reward"] = normalized
        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPPOWeightedRunning(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
