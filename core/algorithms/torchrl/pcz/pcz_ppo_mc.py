"""torchrl/pcz_ppo_mc.py: PCZ-PPO-MC — per-component z-norm + PPO loss on MC returns, no critic.

The critic-free cell of the {critic, no-critic} x {aggregate-norm, PCZ} factorial.
A1 (``torchrl-pcz-ppo-running``) is the critic cell.  This variant keeps PCZ
normalization and PPO's clipped surrogate loss but replaces GAE+critic with
Monte-Carlo discounted returns and batch-level advantage whitening.

Differs from PCZ-GRPO (``torchrl-pcz-grpo``) in that advantages are per-timestep
MC returns whitened across the batch — not trajectory-level group-relative
normalization.
"""

import torch

from .._base import TorchRLAlgorithm
from .._norm import weighted_sum


def _compute_mc_returns(rewards_scalar: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute per-timestep discounted MC returns within each trajectory.

    Args:
        rewards_scalar: [N, T] scalar rewards (post-weighted-sum).
        dones: [N, T] episode-end flags (done | truncated).
        gamma: discount factor.

    Returns:
        [N, T] tensor of G_t = sum_{t'>=t} gamma^(t'-t) * r_t'.
    """
    N, T = rewards_scalar.shape
    returns = torch.zeros_like(rewards_scalar)
    running = torch.zeros(N, device=rewards_scalar.device)
    for t in range(T - 1, -1, -1):
        running = rewards_scalar[:, t] + gamma * running * (~dones[:, t]).float()
        returns[:, t] = running
    return returns


class TorchRLPCZPPOMC(TorchRLAlgorithm):
    """PCZ-PPO with MC returns (no critic, no GAE).

    1. Normalize reward_vec per-component with running EMA stats.
    2. Weighted-sum to scalar reward.
    3. Compute per-timestep discounted MC returns within trajectories.
    4. Whiten returns across the batch -> advantages.
    5. PPO clipped loss (no value loss, since `_algo_type = "grpo"` disables it).
    """

    is_self_normalizing = True
    _algo_type = "grpo"

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
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0)
        a = self._ema_alpha
        self._running_mean = (1 - a) * self._running_mean + a * batch_mean
        self._running_var = (1 - a) * self._running_var + a * batch_var

        var_floor = self.config.variance_floor
        safe_std = torch.sqrt(torch.clamp(self._running_var, min=var_floor))
        normalized = (reward_vec - self._running_mean) / safe_std

        if self.config.component_gating:
            gate = (self._running_var > var_floor).float()
            normalized = normalized * gate

        clip = self.config.znorm_clip
        if clip is not None:
            normalized = torch.clamp(normalized, -clip, clip)

        scalar_reward = weighted_sum(normalized, self._component_weights)
        batch["next", "reward"] = scalar_reward

        dones = batch["next", "done"]
        truncated = batch["next"].get("truncated", torch.zeros_like(dones))
        dones_any = dones | truncated
        if scalar_reward.ndim == 3:
            scalar_reward = scalar_reward.squeeze(-1)
        if dones_any.ndim == 3:
            dones_any = dones_any.squeeze(-1)

        returns = _compute_mc_returns(scalar_reward, dones_any, self.config.gamma)

        mean = returns.mean()
        std = returns.std() + 1e-8
        advantages = (returns - mean) / std

        batch.set("advantage", advantages.unsqueeze(-1))
        return batch


if __name__ == "__main__":
    model = TorchRLPCZPPOMC(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
