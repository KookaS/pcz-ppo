"""torchrl/ppo_znorm.py: TorchRL PPO with scalar reward z-normalization.

Z-normalizes the scalar reward before GAE. Treats the aggregate reward
as a single signal.
"""

from .._base import TorchRLAlgorithm
from .._norm import znorm


class TorchRLPPOZnorm(TorchRLAlgorithm):
    """TorchRL PPO with z-normalization on the scalar reward before GAE."""

    is_self_normalizing = True
    _algo_type = "ppo"

    def _compute_advantages(self, batch, advantage_module):
        reward = batch["next", "reward"]
        batch["next", "reward"] = znorm(reward).reshape(reward.shape)
        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPPOZnorm(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
