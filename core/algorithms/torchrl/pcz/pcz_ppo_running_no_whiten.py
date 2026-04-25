"""torchrl/pcz_ppo_running_no_whiten.py: A6 ablation — PCZ-PPO-Running without advantage whitening.

Same as PCZ-PPO-Running but with normalize_advantage=False.
Tests whether post-aggregation advantage whitening is a critical stabilizer
for per-component z-normalization.
"""

from .pcz_ppo_running import TorchRLPCZPPORunning


class TorchRLPCZPPORunningNoWhiten(TorchRLPCZPPORunning):
    """PCZ-PPO-Running with advantage whitening disabled."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("normalize_advantage", False)
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    model = TorchRLPCZPPORunningNoWhiten(
        "cartpole",
        reward_component_names=["balance", "center"],
        total_frames=50_000,
        frames_per_batch=2048,
        num_envs=2,
    )
    model.learn()
