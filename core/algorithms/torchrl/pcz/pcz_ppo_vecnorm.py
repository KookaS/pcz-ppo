"""torchrl/pcz_ppo_vecnorm.py: PCZ-PPO-VecNorm — running std normalization.

Divides each component by its running standard deviation (no mean subtraction),
similar to VecNormalize's reward normalization. Running statistics persist
across batches.
"""

import torch

from .._base import TorchRLAlgorithm
from .._norm import weighted_sum


class TorchRLPCZPPOVecnorm(TorchRLAlgorithm):
    """TorchRL PCZ-PPO with VecNormalize-style per-component std normalization."""

    is_self_normalizing = True
    _algo_type = "ppo"

    def __init__(self, *args, ema_alpha: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self._ema_alpha = ema_alpha
        self._running_var = None

    def _compute_advantages(self, batch, advantage_module):
        reward_vec = batch["next", "reward_vec"]
        D = reward_vec.shape[-1]

        if self._running_var is None:
            self._running_var = torch.ones(D, device=reward_vec.device)

        flat = reward_vec.reshape(-1, D)
        batch_var = flat.var(dim=0)
        a = self._ema_alpha
        self._running_var = (1 - a) * self._running_var + a * batch_var

        var_floor = self.config.variance_floor
        safe_std = torch.sqrt(torch.clamp(self._running_var, min=var_floor))
        normalized = reward_vec / safe_std  # No mean subtraction

        if self.config.component_gating:
            gate = (self._running_var > var_floor).float()
            normalized = normalized * gate

        clip = self.config.znorm_clip
        if clip is not None:
            normalized = torch.clamp(normalized, -clip, clip)

        batch["next", "reward"] = weighted_sum(normalized, self._component_weights)
        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPCZPPOVecnorm(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
