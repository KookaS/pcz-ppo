"""torchrl/pcz_ppo.py: TorchRL PCZ-PPO — per-component z-norm before GAE.

Z-normalizes each reward component independently before computing the
weighted sum and running standard GAE advantage estimation.  This ensures
components with different magnitudes contribute equally to the policy
gradient.

Usage::

    from core.algorithms.torchrl.pcz_ppo import TorchRLPCZPPO

    model = TorchRLPCZPPO(
        "lunarlander",
        reward_component_names=["landing", "shaping", "fuel_main", "fuel_side"],
        component_weights=[5.0, 3.0, 0.5, 0.5],
    )
    model.learn(total_frames=500_000)
"""

import torch

from .._base import TorchRLAlgorithm


def _znorm_per_env(x, min_std=1e-8, clip_range=None):
    """Z-normalize tensor per environment (axis=0), matching SB3 semantics.

    For a tensor of shape [T, N] (timesteps × envs), computes mean/std
    along T for each env independently.  For flat tensors [T*N], falls
    back to global normalization.

    Args:
        clip_range: If set (e.g. 3.0), clip output to [-clip_range, clip_range].
    """
    if x.ndim >= 2:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        out = torch.where(std > min_std, (x - mean) / std, x - mean)
    else:
        mean = x.mean()
        std = x.std()
        if std > min_std:
            out = (x - mean) / std
        else:
            out = x - mean
    if clip_range is not None:
        out = out.clamp(-clip_range, clip_range)
    return out


class TorchRLPCZPPO(TorchRLAlgorithm):
    """TorchRL PCZ-PPO: per-component reward z-normalization before GAE.

    Each reward component is z-normalized independently per environment,
    then combined with component weights before standard GAE advantage
    estimation.  The per-env normalization matches SB3's
    ComponentRolloutBuffer semantics (axis=0 over buffer steps).
    """

    is_self_normalizing = True
    _algo_type = "ppo"

    def _compute_advantages(self, batch, advantage_module):
        """Z-normalize per-component rewards per env, then compute GAE."""
        reward_vec = batch["next", "reward_vec"]  # [T, N, D] or [T*N, D]
        D = reward_vec.shape[-1]

        # Z-normalize each component per environment
        clip_range = getattr(self.config, "znorm_clip", None)
        normalized = torch.zeros_like(reward_vec)
        for i in range(D):
            normalized[..., i] = _znorm_per_env(reward_vec[..., i], clip_range=clip_range)

        # Weighted sum → new scalar reward
        weights = torch.tensor(
            self._component_weights,
            device=reward_vec.device,
            dtype=reward_vec.dtype,
        )
        new_reward = (normalized * weights).sum(dim=-1, keepdim=True)

        # Replace reward in batch for GAE computation
        batch["next", "reward"] = new_reward

        # Standard GAE on normalized rewards
        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPCZPPO(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
