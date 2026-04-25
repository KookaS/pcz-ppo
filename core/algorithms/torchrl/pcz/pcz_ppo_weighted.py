"""torchrl/pcz_ppo_weighted.py: PCZ-PPO-Weighted — per-component z-norm with weights.

Identical to PCZ-PPO but makes the component weights explicit in the name.
Both algorithms perform per-component z-normalization followed by weighted
summation.
"""

from .pcz_ppo import TorchRLPCZPPO


class TorchRLPCZPPOWeighted(TorchRLPCZPPO):
    """TorchRL PCZ-PPO with weighted per-component z-normalization.

    Functionally identical to TorchRLPCZPPO — the naming makes the
    presence of component weights explicit for comparison experiments.
    """

    pass


if __name__ == "__main__":
    model = TorchRLPCZPPOWeighted(
        "cartpole",
        reward_component_names=["balance", "center"],
        component_weights=[1.0, 2.0],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
