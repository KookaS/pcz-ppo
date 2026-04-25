"""torchrl/pcz_ppo_running.py: PCZ-PPO-Running — running mean/std per component.

Normalizes each component using exponentially-tracked running mean and std.
Statistics persist across batches, providing more stable normalization
early in training.
"""

import torch

from .._base import TorchRLAlgorithm
from .._norm import weighted_sum


class TorchRLPCZPPORunning(TorchRLAlgorithm):
    """TorchRL PCZ-PPO with running mean/std normalization per component."""

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

        # Lazily initialize running stats
        if self._running_mean is None:
            self._running_mean = torch.zeros(D, device=reward_vec.device)
            self._running_var = torch.ones(D, device=reward_vec.device)

        # Update running stats with EMA
        flat = reward_vec.reshape(-1, D)
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0)
        a = self._ema_alpha
        self._running_mean = (1 - a) * self._running_mean + a * batch_mean
        self._running_var = (1 - a) * self._running_var + a * batch_var

        # Normalize with running stats
        var_floor = self.config.variance_floor
        safe_std = torch.sqrt(torch.clamp(self._running_var, min=var_floor))
        normalized = (reward_vec - self._running_mean) / safe_std

        # Component gating: zero-out near-constant components
        if self.config.component_gating:
            gate = (self._running_var > var_floor).float()
            normalized = normalized * gate

        # Safety clip: prevent catastrophic z-score spikes from rare events
        # (e.g. Humanoid death after long survival → z-score of 50+)
        # Default znorm_clip=None disables; 10.0 is wide enough to not
        # affect normal training but catches policy-destroying outliers.
        clip = self.config.znorm_clip
        if clip is not None:
            normalized = torch.clamp(normalized, -clip, clip)

        batch["next", "reward"] = weighted_sum(normalized, self._component_weights)
        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPCZPPORunning(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
