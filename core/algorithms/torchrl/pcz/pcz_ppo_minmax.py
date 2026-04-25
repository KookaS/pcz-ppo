"""torchrl/pcz_ppo_minmax.py: PCZ-PPO-MinMax — per-component min-max scaling.

Scales each reward component to [0, 1] using min-max normalization within
each batch. Falls back to 0.5 when a component has zero range.
"""

from .._base import TorchRLAlgorithm
from .._norm import minmax_per_env, weighted_sum


class TorchRLPCZPPOMinmax(TorchRLAlgorithm):
    """TorchRL PCZ-PPO with per-component min-max scaling to [0, 1]."""

    is_self_normalizing = True
    _algo_type = "ppo"

    def _compute_advantages(self, batch, advantage_module):
        reward_vec = batch["next", "reward_vec"]
        # Apply min-max per component globally (flatten envs+timesteps)
        orig_shape = reward_vec.shape
        flat = reward_vec.reshape(-1, orig_shape[-1])
        normalized = minmax_per_env(flat, dim=0)
        normalized = normalized.reshape(orig_shape)

        batch["next", "reward"] = weighted_sum(normalized, self._component_weights)
        return advantage_module(batch)


if __name__ == "__main__":
    model = TorchRLPCZPPOMinmax(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
