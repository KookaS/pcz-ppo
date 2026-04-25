"""pcz_ppo_popart.py: PCZ-PPO-PopArt — GDPO z-norm + PopArt value rescaling.

Combines GDPO per-component reward z-normalization (ensuring equal signal
contribution) with PopArt adaptive value head rescaling (ensuring the critic
tracks changing return magnitudes). This is a hybrid that addresses both the
reward-side and value-side normalization challenges.

Usage::

    from core.algorithms.pcz_ppo_popart import PCZPPOPopArt

    model = PCZPPOPopArt("MlpPolicy", env, reward_component_names=["a", "b"])
    model.learn(total_timesteps=50_000)

Reference: https://github.com/NVlabs/GDPO
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from .._common import (
    ComponentRolloutBuffer,
    PopArtMixin,
    _init_component_weights,
    _weighted_component_sum,
    _znorm,
    collect_rollouts,
    log_normalization_diagnostics,
    run_cartpole_example,
)


class PCZPPOPopArt(PopArtMixin, PPO):
    """PCZ-PPO + PopArt: per-component reward z-norm + adaptive value rescaling.

    Args:
        reward_component_names: List of component names. Must be provided.
        All other args forwarded to ``stable_baselines3.PPO``.
    """

    is_self_normalizing = True

    def __init__(
        self,
        *args,
        reward_component_names: list[str] | None = None,
        component_weights: list[float] | None = None,
        **kwargs,
    ):
        if reward_component_names is None:
            raise ValueError("PCZPPOPopArt requires reward_component_names.")
        self._reward_component_names = list(reward_component_names)
        self._n_reward_components = len(self._reward_component_names)
        self._component_weights = _init_component_weights(self._n_reward_components, component_weights)
        super().__init__(*args, **kwargs)
        if self.clip_range_vf is not None:
            raise ValueError(
                "PCZPPOPopArt is incompatible with clip_range_vf. After PopArt "
                "rescales the value head, old_values and new values are in "
                "different scales, breaking value clipping. Use clip_range_vf=None."
            )
        self._init_popart()

    def _setup_model(self) -> None:
        super()._setup_model()
        self.rollout_buffer = ComponentRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            n_reward_components=self._n_reward_components,
        )

    def collect_rollouts(
        self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int
    ) -> bool:
        result = collect_rollouts(
            self,
            env,
            callback,
            self.rollout_buffer,
            n_rollout_steps,
            self._reward_component_names,
        )
        if result is False:
            return False

        _, last_values, dones = result
        buf = self.rollout_buffer

        # Step 1: GDPO per-component z-norm on rewards
        pre_rewards = buf.component_rewards.sum(axis=2).copy()
        normalized = np.zeros_like(buf.component_rewards)
        for i in range(self._n_reward_components):
            normalized[:, :, i] = _znorm(buf.component_rewards[:, :, i], axis=0)
        buf.rewards = _weighted_component_sum(normalized, self._component_weights)
        log_normalization_diagnostics(self, pre_rewards, buf.rewards)
        buf._reapply_bootstrap()

        # Step 2: Standard GAE on the PCZ-normalized rewards
        buf.compute_returns_and_advantage(last_values, dones)

        # Step 3: PopArt — update stats and rescale value head
        self._apply_popart_to_returns(buf)

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True


if __name__ == "__main__":
    run_cartpole_example(PCZPPOPopArt, "PCZ-PPO-PopArt")
