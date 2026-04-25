"""torchrl/pcz_ppo_quantile.py: PCZ-PPO-Quantile — distribution-free rank normalization.

Z-normalization assumes ~Gaussian, but landing reward is ~0 for 95% of
steps and ±100 for 5% (wildly non-Gaussian). Quantile normalization
(rank-transform → uniform → unit-Gaussian via inverse CDF) is
distribution-free and handles heavy-tailed/bimodal components naturally.

Per-component quantile transform applied AFTER running mean centering
but BEFORE weighted sum (weights still control priority).
"""

import torch

from .._base import TorchRLAlgorithm
from .._norm import weighted_sum


def quantile_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Rank-based quantile normalization: rank → uniform → standard normal.

    Per-column (last dim) operation. Output is N(0, 1)-distributed if input
    is non-degenerate.
    """
    flat_shape = x.shape
    flat = x.reshape(-1, flat_shape[-1])  # [B, D]
    B = flat.shape[0]
    if B < 2:
        return x  # too few samples
    # Per-column rank in [0, B-1]
    ranks = torch.argsort(torch.argsort(flat, dim=0), dim=0).float()
    # To uniform in (0, 1) — shift to (eps, 1-eps) to avoid infinite normal
    u = (ranks + 0.5) / B
    u = u.clamp(eps, 1 - eps)
    # Inverse standard normal CDF (probit)
    # torch has erfinv → φ⁻¹(u) = √2 · erfinv(2u - 1)
    z = torch.erfinv(2 * u - 1) * (2**0.5)
    return z.reshape(flat_shape)


class TorchRLPCZPPOQuantile(TorchRLAlgorithm):
    """TorchRL PCZ-PPO with per-component quantile (rank) normalization."""

    is_self_normalizing = True
    _algo_type = "ppo"

    def _compute_advantages(self, batch, advantage_module):
        reward_vec = batch["next", "reward_vec"]
        normalized = quantile_normalize(reward_vec)
        batch["next", "reward"] = weighted_sum(normalized, self._component_weights)
        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPCZPPOQuantile(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
