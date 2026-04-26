"""torchrl/pcz_ppo_cosw.py: PCZ-PPO-CosW — cosine weight annealing.

Anneals per-component weights from equal (all 1.0) at training start to
the user-configured target weights at training end using a cosine
schedule.  Intended to test the "weight initialization matters" hypothesis
Hypothesis: if PCZ-PPO is sensitive to starting weights, a gentle ramp
from equal weights should stabilise early training.

At step 0:  w_eff = ones(D)
At step T:  w_eff = configured weights
In between: cosine-interpolated.
"""

import math

import torch

from .._base import TorchRLAlgorithm


class TorchRLPCZPPOCosW(TorchRLAlgorithm):
    """TorchRL PCZ-PPO with cosine weight annealing (equal -> target)."""

    is_self_normalizing = True
    _algo_type = "ppo"

    def __init__(self, *args, ema_alpha: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self._ema_alpha = ema_alpha
        self._running_mean = None
        self._running_var = None
        self._frames_seen = 0

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

        # Cosine annealing from equal to target over training
        self._frames_seen += flat.shape[0]
        progress = min(self._frames_seen / max(self.config.total_frames, 1), 1.0)
        anneal = 0.5 * (1.0 - math.cos(math.pi * progress))  # 0 -> 1, cosine

        target = torch.tensor(self._component_weights, device=device, dtype=normalized.dtype)
        equal = torch.ones_like(target)
        w_eff = equal + (target - equal) * anneal

        batch["next", "reward"] = (normalized * w_eff).sum(dim=-1, keepdim=True)
        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPCZPPOCosW(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=20_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
