"""torchrl/pcz_ppo_log.py: PCZ-PPO-Log — per-component log compression.

Applies sign(r) * log(1 + |r|) to each component before weighted summation.
Reduces dynamic range without requiring statistics; robust to outliers.
"""

from .._base import TorchRLAlgorithm
from .._norm import log_compress, weighted_sum


class TorchRLPCZPPOLog(TorchRLAlgorithm):
    """TorchRL PCZ-PPO with per-component log compression."""

    is_self_normalizing = True
    _algo_type = "ppo"

    def _compute_advantages(self, batch, advantage_module):
        reward_vec = batch["next", "reward_vec"]
        compressed = log_compress(reward_vec)
        batch["next", "reward"] = weighted_sum(compressed, self._component_weights)
        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPCZPPOLog(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
