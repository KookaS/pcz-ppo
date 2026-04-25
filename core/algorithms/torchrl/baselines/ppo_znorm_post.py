"""torchrl/ppo_znorm_post.py: TorchRL PPO with post-aggregation z-normalization.

Sums per-component rewards with weights, then z-normalizes the scalar.
This is the "aggregate-then-normalize" approach that GDPO improves upon.
"""

from .._base import TorchRLAlgorithm
from .._norm import weighted_sum, znorm


class TorchRLPPOZnormPost(TorchRLAlgorithm):
    """TorchRL PPO: weighted sum of components, then z-normalize scalar."""

    is_self_normalizing = True
    _algo_type = "ppo"

    def _compute_advantages(self, batch, advantage_module):
        reward_vec = batch["next", "reward_vec"]
        raw_sum = weighted_sum(reward_vec, self._component_weights)
        batch["next", "reward"] = znorm(raw_sum).reshape(raw_sum.shape)
        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPPOZnormPost(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
