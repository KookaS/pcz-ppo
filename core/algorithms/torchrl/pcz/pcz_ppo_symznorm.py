"""torchrl/pcz_ppo_symznorm.py: PCZ-PPO-SymZnorm — symlog then z-norm.

Distinct from A9 (symlog alone replaces z-norm): here we apply
symlog(x) = sign(x) · log(1 + |x|) as a magnitude compression *before*
per-component running z-normalization. Symlog dampens heavy-tailed
reward spikes (e.g. LunarLander landing ±100) so running stats are less
perturbed, while z-norm still restores per-component scale parity.
"""

import torch

from .._base import TorchRLAlgorithm
from .._norm import weighted_sum


class TorchRLPCZPPOSymZnorm(TorchRLAlgorithm):
    """TorchRL PCZ-PPO with symlog compression then per-component z-norm."""

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

        # Symlog compression
        compressed = torch.sign(reward_vec) * torch.log1p(reward_vec.abs())

        if self._running_mean is None:
            self._running_mean = torch.zeros(D, device=device)
            self._running_var = torch.ones(D, device=device)

        flat = compressed.reshape(-1, D)
        a = self._ema_alpha
        self._running_mean = (1 - a) * self._running_mean + a * flat.mean(dim=0)
        self._running_var = (1 - a) * self._running_var + a * flat.var(dim=0)

        var_floor = self.config.variance_floor
        safe_std = torch.sqrt(torch.clamp(self._running_var, min=var_floor))
        normalized = (compressed - self._running_mean) / safe_std
        clip = self.config.znorm_clip
        if clip is not None:
            normalized = torch.clamp(normalized, -clip, clip)

        batch["next", "reward"] = weighted_sum(normalized, self._component_weights)
        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPCZPPOSymZnorm(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=20_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
