"""torchrl/pcz_ppo_global.py: PCZ-PPO-Global — global z-norm per component.

Z-normalizes each component globally (all envs + timesteps combined)
rather than per-env. Useful when per-env statistics are noisy.
"""

import torch

from .._base import TorchRLAlgorithm
from .._norm import weighted_sum, znorm


class TorchRLPCZPPOGlobal(TorchRLAlgorithm):
    """TorchRL PCZ-PPO with global z-normalization per component."""

    is_self_normalizing = True
    _algo_type = "ppo"

    def _compute_advantages(self, batch, advantage_module):
        reward_vec = batch["next", "reward_vec"]
        D = reward_vec.shape[-1]

        normalized = torch.zeros_like(reward_vec)
        for i in range(D):
            normalized[..., i] = znorm(reward_vec[..., i])

        batch["next", "reward"] = weighted_sum(normalized, self._component_weights)
        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPCZPPOGlobal(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
