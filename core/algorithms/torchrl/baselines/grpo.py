"""torchrl/grpo.py: TorchRL GRPO — critic-free with group normalization.

Group Relative Policy Optimization: segments trajectories, groups them
sequentially, and normalizes advantages within each group.  No value
function is learned.

Usage::

    from core.algorithms.torchrl.grpo import TorchRLGRPO

    model = TorchRLGRPO(
        "lunarlander",
        reward_component_names=["landing", "shaping", "fuel_main", "fuel_side"],
        group_size=8,
    )
    model.learn(total_frames=500_000)
"""

from .._base import TorchRLAlgorithm


class TorchRLGRPO(TorchRLAlgorithm):
    """TorchRL GRPO — critic-free with trajectory group normalization.

    Computes trajectory-level returns, groups them sequentially, and
    normalizes within each group.  No value function is trained.
    """

    is_self_normalizing = True
    _algo_type = "grpo"


if __name__ == "__main__":
    model = TorchRLGRPO(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
        group_size=8,
    )
    model.learn()
