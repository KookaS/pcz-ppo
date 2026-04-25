"""pcz_ppo_log.py: PCZ-PPO-Log — per-component log compression.

Applies log compression ``sign(r) * log(1 + |r|)`` to each reward component
before summing. This reduces the dynamic range of rewards without requiring
statistics, making it robust to outliers.

Usage::

    from core.algorithms.pcz_ppo_log import PCZPPOLog

    model = PCZPPOLog("MlpPolicy", env, reward_component_names=["a", "b"])
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
    _init_component_weights,
    _weighted_component_sum,
    collect_rollouts,
    log_normalization_diagnostics,
    run_cartpole_example,
)


class PCZPPOLog(PPO):
    """PCZ-PPO with per-component log compression.

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
            raise ValueError("PCZPPOLog requires reward_component_names.")
        self._reward_component_names = list(reward_component_names)
        self._n_reward_components = len(self._reward_component_names)
        self._component_weights = _init_component_weights(self._n_reward_components, component_weights)
        super().__init__(*args, **kwargs)

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

        pre_rewards = buf.component_rewards.sum(axis=2).copy()
        compressed = np.sign(buf.component_rewards) * np.log1p(np.abs(buf.component_rewards))
        buf.rewards = _weighted_component_sum(compressed, self._component_weights)
        log_normalization_diagnostics(self, pre_rewards, buf.rewards)
        buf._reapply_bootstrap()
        buf.compute_returns_and_advantage(last_values, dones)

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True


if __name__ == "__main__":
    run_cartpole_example(PCZPPOLog, "PCZ-PPO-Log")
