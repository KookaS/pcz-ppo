"""torchrl/ppo_no_norm.py: TorchRL PPO with no normalization.

No reward normalization, no advantage normalization. The most raw PPO baseline.
"""

from .._base import TorchRLAlgorithm


class TorchRLPPONoNorm(TorchRLAlgorithm):
    """TorchRL PPO with no reward or advantage normalization."""

    is_self_normalizing = True
    _algo_type = "ppo"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("normalize_advantage", False)
        super().__init__(*args, **kwargs)

    def _compute_advantages(self, batch, advantage_module):
        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPPONoNorm(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
