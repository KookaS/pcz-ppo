"""torchrl/pcz_ppo_rlw.py: PCZ-PPO-RLW — Random Loss Weighting baseline.

Per Lin et al. (TMLR 2021), sample weights from Dirichlet each minibatch.
Essential control: if RLW matches fancy weight-learning methods (Kendall,
GradNorm), then the weight sensitivity story is about optimization
landscape regularization, not weight values. Applied AFTER per-component
z-normalization, like other PCZ variants.
"""

import torch

from .._base import TorchRLAlgorithm


class TorchRLPCZPPORlw(TorchRLAlgorithm):
    """TorchRL PCZ-PPO with Dirichlet random loss weighting (RLW baseline)."""

    is_self_normalizing = True
    _algo_type = "ppo"

    def __init__(self, *args, ema_alpha: float = 0.01, dirichlet_alpha: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._ema_alpha = ema_alpha
        self._dirichlet_alpha = dirichlet_alpha
        self._running_mean = None
        self._running_var = None

    def _compute_advantages(self, batch, advantage_module):
        reward_vec = batch["next", "reward_vec"]
        D = reward_vec.shape[-1]
        device = reward_vec.device

        if self._running_mean is None:
            self._running_mean = torch.zeros(D, device=device)
            self._running_var = torch.ones(D, device=device)

        flat = reward_vec.reshape(-1, D)
        a = self._ema_alpha
        self._running_mean = (1 - a) * self._running_mean + a * flat.mean(dim=0)
        self._running_var = (1 - a) * self._running_var + a * flat.var(dim=0)

        var_floor = self.config.variance_floor
        safe_std = torch.sqrt(torch.clamp(self._running_var, min=var_floor))
        z = (reward_vec - self._running_mean) / safe_std

        # Sample per-batch random weights from Dirichlet(α=1, ..., α=1)
        # Sum to 1 by construction; symmetric prior — no component preferred.
        alpha = torch.full((D,), self._dirichlet_alpha, device=device)
        weights = torch.distributions.Dirichlet(alpha).sample()  # shape [D]

        batch["next", "reward"] = (z * weights).sum(dim=-1, keepdim=True)
        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPCZPPORlw(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
