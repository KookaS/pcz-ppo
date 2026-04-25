"""torchrl/pcz_ppo_asym.py: PCZ-PPO-Asym — asymmetric positive/negative z-norm.

For reward components with asymmetric distributions (e.g. LunarLander
landing reward: bounded-below penalty for crashing, unbounded-above bonus
for landing), separate running stats for positive and negative values
produces cleaner normalization than a shared mean/std that splits the
difference.

Each component k maintains four EMAs: (mean⁺_k, var⁺_k, mean⁻_k, var⁻_k).
Positive samples are z-normalized against the positive stats; negative
against the negative stats.  Zero is a hard partition.
"""

import torch

from .._base import TorchRLAlgorithm
from .._norm import weighted_sum


class TorchRLPCZPPOAsym(TorchRLAlgorithm):
    """TorchRL PCZ-PPO with asymmetric (sign-split) per-component z-norm."""

    is_self_normalizing = True
    _algo_type = "ppo"

    def __init__(self, *args, ema_alpha: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self._ema_alpha = ema_alpha
        self._stats = None  # dict with 'pos_mean','pos_var','neg_mean','neg_var'

    def _compute_advantages(self, batch, advantage_module):
        reward_vec = batch["next", "reward_vec"]
        D = reward_vec.shape[-1]
        device = reward_vec.device

        if self._stats is None:
            self._stats = {
                "pos_mean": torch.zeros(D, device=device),
                "pos_var": torch.ones(D, device=device),
                "neg_mean": torch.zeros(D, device=device),
                "neg_var": torch.ones(D, device=device),
            }

        flat = reward_vec.reshape(-1, D)
        a = self._ema_alpha
        var_floor = self.config.variance_floor

        pos_mask = flat > 0
        neg_mask = flat < 0

        for _sign, mask, mean_key, var_key in (
            ("pos", pos_mask, "pos_mean", "pos_var"),
            ("neg", neg_mask, "neg_mean", "neg_var"),
        ):
            for k in range(D):
                vals = flat[:, k][mask[:, k]]
                if vals.numel() >= 2:
                    m = vals.mean()
                    v = vals.var()
                    self._stats[mean_key][k] = (1 - a) * self._stats[mean_key][k] + a * m
                    self._stats[var_key][k] = (1 - a) * self._stats[var_key][k] + a * v

        pos_std = torch.sqrt(torch.clamp(self._stats["pos_var"], min=var_floor))
        neg_std = torch.sqrt(torch.clamp(self._stats["neg_var"], min=var_floor))

        pos_norm = (reward_vec - self._stats["pos_mean"]) / pos_std
        neg_norm = (reward_vec - self._stats["neg_mean"]) / neg_std

        sign = torch.sign(reward_vec)
        normalized = torch.where(sign > 0, pos_norm, torch.where(sign < 0, neg_norm, torch.zeros_like(reward_vec)))

        clip = self.config.znorm_clip
        if clip is not None:
            normalized = torch.clamp(normalized, -clip, clip)

        batch["next", "reward"] = weighted_sum(normalized, self._component_weights)
        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPCZPPOAsym(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=20_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
