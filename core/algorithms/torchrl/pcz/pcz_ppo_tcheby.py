"""torchrl/pcz_ppo_tcheby.py: PCZ-PPO-Tcheby — min-max (Tchebycheff) scalarization.

Replaces linear weighted sum with the minimum weighted component after
z-normalization:  r̃ = min_k (w_k · z_k).  The policy gradient is driven
by the *worst-performing* component each step, which is a classic MOP
robustness trick: reduces sensitivity to weight misspecification, at the
cost of slower average learning (each step only improves one component).
"""

import torch

from .._base import TorchRLAlgorithm


class TorchRLPCZPPOTcheby(TorchRLAlgorithm):
    """TorchRL PCZ-PPO with Tchebycheff (min-max) scalarization."""

    is_self_normalizing = True
    _algo_type = "ppo"

    def __init__(self, *args, ema_alpha: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self._ema_alpha = ema_alpha
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
        normalized = (reward_vec - self._running_mean) / safe_std
        clip = self.config.znorm_clip
        if clip is not None:
            normalized = torch.clamp(normalized, -clip, clip)

        weights = torch.tensor(self._component_weights, device=device, dtype=normalized.dtype)
        weighted = normalized * weights  # (..., D)
        scalar = weighted.min(dim=-1).values
        batch["next", "reward"] = scalar.unsqueeze(-1)
        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPCZPPOTcheby(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=20_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
