"""torchrl/pcz_ppo_clip.py: PCZ-PPO-Clip — per-component reward clipping.

Clips each reward component to [-1, 1] before weighted summation.
Simple, crude normalization that bounds each component's contribution.
"""

from .._base import TorchRLAlgorithm
from .._norm import clip_norm, weighted_sum


class TorchRLPCZPPOClip(TorchRLAlgorithm):
    """TorchRL PCZ-PPO with per-component reward clipping to [-1, 1]."""

    is_self_normalizing = True
    _algo_type = "ppo"

    def _compute_advantages(self, batch, advantage_module):
        reward_vec = batch["next", "reward_vec"]
        clipped = clip_norm(reward_vec, clip=1.0)
        batch["next", "reward"] = weighted_sum(clipped, self._component_weights)
        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPCZPPOClip(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
