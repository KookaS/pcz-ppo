"""torchrl/pcz_grpo.py: TorchRL PCZ-GRPO — per-component z-norm + GRPO.

Critic-free variant that:
1. Segments trajectories per environment.
2. Computes per-component trajectory returns.
3. Z-normalizes each component's returns independently.
4. Computes weighted sum of normalized returns.
5. Applies GRPO group normalization on the combined scalar returns.

Usage::

    from core.algorithms.torchrl.pcz_grpo import TorchRLPCZGRPO

    model = TorchRLPCZGRPO(
        "lunarlander",
        reward_component_names=["landing", "shaping", "fuel_main", "fuel_side"],
        component_weights=[5.0, 3.0, 0.5, 0.5],
        group_size=8,
    )
    model.learn(total_frames=500_000)
"""

import torch

from .._base import TorchRLAlgorithm


def _znorm_torch(x, min_std=1e-8):
    """Z-normalize tensor: (x - mean) / std."""
    mean = x.mean()
    std = x.std()
    if std > min_std:
        return (x - mean) / std
    return x - mean


def compute_pcz_grpo_advantages(batch, group_size, component_weights):
    """PCZ-GRPO advantages: per-component z-norm + GRPO group normalization.

    Args:
        batch: TensorDict with shape [N, T].
        group_size: Number of trajectories per normalization group.
        component_weights: Per-component weights for combining returns.

    Returns:
        Batch with ``"advantage"`` key set.
    """
    rewards = batch["next", "reward_vec"]  # [N, T, D]
    dones = batch["next", "done"]
    truncated = batch["next"].get("truncated", torch.zeros_like(dones))
    dones = dones | truncated

    if rewards.ndim == 2:
        rewards = rewards.unsqueeze(-1)
    if dones.ndim == 3:
        dones = dones.squeeze(-1)

    N, T, D = rewards.shape
    device = rewards.device

    # 1. Trajectory segmentation and per-component returns
    traj_returns = []
    traj_ids_all = torch.empty((N, T), dtype=torch.long, device=device)
    offset = 0

    for n in range(N):
        r = rewards[n]  # [T, D]
        d = dones[n].to(torch.int64)  # [T]
        traj_ids = (torch.cumsum(d, dim=0) - 1).clamp(min=0)
        num_traj = int(traj_ids.max().item()) + 1

        returns = torch.zeros((num_traj, D), device=device)
        returns.index_add_(0, traj_ids, r)
        traj_returns.append(returns)

        traj_ids_all[n] = traj_ids + offset
        offset += num_traj

    traj_returns = torch.cat(traj_returns, dim=0)  # [num_traj_total, D]

    # 2. Z-normalize each component's returns across trajectories
    normalized = torch.zeros_like(traj_returns)
    for i in range(D):
        normalized[:, i] = _znorm_torch(traj_returns[:, i])

    # 3. Weighted sum
    weights = torch.tensor(component_weights, device=device, dtype=normalized.dtype)
    scalar_returns = (normalized * weights).sum(dim=1)  # [num_traj_total]

    # 4. GRPO group normalization
    num_traj_total = scalar_returns.shape[0]
    num_groups = num_traj_total // group_size
    if num_groups == 0:
        batch.set("advantage", torch.zeros((N, T, 1), device=device))
        return batch

    valid_traj = num_groups * group_size
    grouped = scalar_returns[:valid_traj].view(num_groups, group_size)

    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True) + 1e-8
    adv_per_traj = ((grouped - mean) / std).view(-1)

    # 5. Map advantages back to timesteps
    adv = torch.zeros((N, T), device=device)
    valid_mask = traj_ids_all < valid_traj
    adv[valid_mask] = adv_per_traj[traj_ids_all[valid_mask]]
    adv[~valid_mask] = 0.0

    batch.set("advantage", adv.unsqueeze(-1))
    return batch


class TorchRLPCZGRPO(TorchRLAlgorithm):
    """TorchRL PCZ-GRPO: per-component z-norm + GRPO group normalization.

    Critic-free variant that z-normalizes each component's trajectory returns
    independently, then applies GRPO group normalization on the weighted sum.
    """

    is_self_normalizing = True
    _algo_type = "grpo"

    def _compute_advantages(self, batch, advantage_module):
        """Per-component z-norm of trajectory returns + GRPO grouping."""
        return compute_pcz_grpo_advantages(batch, self.config.group_size, self._component_weights)


if __name__ == "__main__":
    model = TorchRLPCZGRPO(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
        group_size=8,
    )
    model.learn()
