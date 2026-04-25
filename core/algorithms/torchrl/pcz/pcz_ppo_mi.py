"""torchrl/pcz_ppo_mi.py: PCZ-PPO-MI — variance-proxy MI weighting.

Scales user weights by a variance-based proxy for per-component mutual
information with the action:  w_k^eff = w_k · (running_var_k / mean_var),
applied *after* z-normalization (on the weighted sum).  Components that
carry little variance (and hence little MI with action) get shrunk
automatically, without the hard on/off of component-gating.
"""

import torch

from .._base import TorchRLAlgorithm


class TorchRLPCZPPOMI(TorchRLAlgorithm):
    """TorchRL PCZ-PPO with variance-proxy MI soft weighting."""

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
        safe_var = torch.clamp(self._running_var, min=var_floor)
        safe_std = torch.sqrt(safe_var)
        normalized = (reward_vec - self._running_mean) / safe_std
        clip = self.config.znorm_clip
        if clip is not None:
            normalized = torch.clamp(normalized, -clip, clip)

        # MI proxy: scale weights by relative variance (informative components
        # get more weight). Normalized so mean multiplier is 1.0 — preserves
        # overall reward scale.
        var_ratio = safe_var / safe_var.mean()
        user_weights = torch.tensor(self._component_weights, device=device, dtype=normalized.dtype)
        effective_weights = user_weights * var_ratio

        batch["next", "reward"] = (normalized * effective_weights).sum(dim=-1, keepdim=True)
        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPCZPPOMI(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=20_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
