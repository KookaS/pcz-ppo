"""torchrl/ppo.py: TorchRL vanilla PPO with GAE.

Standard PPO using Generalized Advantage Estimation — no custom reward
normalization.  Equivalent to the SB3 ``StandardPPO`` but using TorchRL
as the backend with parallel environments.

Usage::

    from core.algorithms.torchrl.ppo import TorchRLPPO

    model = TorchRLPPO(
        "lunarlander",
        reward_component_names=["landing", "shaping", "fuel_main", "fuel_side"],
    )
    model.learn(total_frames=500_000)
"""

from .._base import TorchRLAlgorithm


class TorchRLPPO(TorchRLAlgorithm):
    """TorchRL vanilla PPO with GAE.

    Uses standard GAE advantage estimation with no reward normalization.
    Reward components are tracked but not modified.
    """

    is_self_normalizing = False
    _algo_type = "ppo"


if __name__ == "__main__":
    model = TorchRLPPO(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
