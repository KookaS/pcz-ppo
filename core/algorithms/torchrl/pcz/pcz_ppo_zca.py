"""torchrl/pcz_ppo_zca.py: PCZ-PPO-ZCA — Mahalanobis (ZCA) whitening.

Replaces independent per-component z-norm with full-covariance whitening:
r̃ = Σ^(-1/2) (r - μ).  Decorrelates components while preserving original
axes (unlike PCA rotation).  Manual weights operate on whitened, decorrelated
components — restoring the interpretation that nominal weight = effective
weight.

Trade-off: covariance estimation noise with small rollout buffers.  Uses
running EMA of covariance matrix (D×D) for stability.
"""

import torch

from .._base import TorchRLAlgorithm
from .._norm import weighted_sum


class TorchRLPCZPPOZca(TorchRLAlgorithm):
    """TorchRL PCZ-PPO with ZCA (Mahalanobis) whitening."""

    is_self_normalizing = True
    _algo_type = "ppo"

    def __init__(self, *args, ema_alpha: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self._ema_alpha = ema_alpha
        self._running_mean = None
        self._running_cov = None

    def _compute_advantages(self, batch, advantage_module):
        reward_vec = batch["next", "reward_vec"]
        D = reward_vec.shape[-1]
        device = reward_vec.device

        if self._running_mean is None:
            self._running_mean = torch.zeros(D, device=device)
            self._running_cov = torch.eye(D, device=device)

        flat = reward_vec.reshape(-1, D)
        a = self._ema_alpha
        batch_mean = flat.mean(dim=0)
        # Sample covariance (B, D, D)
        centered = flat - batch_mean
        batch_cov = centered.t() @ centered / max(flat.shape[0] - 1, 1)

        self._running_mean = (1 - a) * self._running_mean + a * batch_mean
        self._running_cov = (1 - a) * self._running_cov + a * batch_cov

        # ZCA whitening: Σ^(-1/2)
        var_floor = self.config.variance_floor
        # Symmetric eigendecomp for stable Σ^(-1/2)
        eigvals, eigvecs = torch.linalg.eigh(self._running_cov + var_floor * torch.eye(D, device=device))
        eigvals = torch.clamp(eigvals, min=var_floor)
        sigma_inv_sqrt = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.t()

        whitened = (reward_vec - self._running_mean) @ sigma_inv_sqrt.t()

        batch["next", "reward"] = weighted_sum(whitened, self._component_weights)
        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPCZPPOZca(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
