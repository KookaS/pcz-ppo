"""torchrl/ppo_adv_only.py: TorchRL PPO with advantage whitening only.

No reward normalization. Advantage normalization (per-minibatch whitening)
is enabled via normalize_advantage=True in ClipPPOLoss.
"""

from .._base import TorchRLAlgorithm


class TorchRLPPOAdvOnly(TorchRLAlgorithm):
    """TorchRL PPO with advantage whitening only (no reward normalization)."""

    is_self_normalizing = True
    _algo_type = "ppo"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("normalize_advantage", True)
        super().__init__(*args, **kwargs)

    def _compute_advantages(self, batch, advantage_module):
        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPPOAdvOnly(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
