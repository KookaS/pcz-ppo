"""pcz_ppo_weighted.py: PCZ-PPO-Weighted — per-component z-norm with tunable weights.

Like PCZ-PPO but allows different weights for each normalized component
before summation. By default all weights are 1.0 (equal contribution).

Usage::

    from core.algorithms.pcz_ppo_weighted import PCZPPOWeighted

    model = PCZPPOWeighted(
        "MlpPolicy", env,
        reward_component_names=["a", "b"],
        component_weights=[1.0, 2.0],
    )
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
    _znorm,
    collect_rollouts,
    log_normalization_diagnostics,
    run_cartpole_example,
)


class PCZPPOWeighted(PPO):
    """PCZ-PPO with weighted per-component z-normalization.

    Args:
        reward_component_names: List of component names. Must be provided.
        component_weights: Optional weights for each component. Defaults to
            equal weights (1.0 each).
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
            raise ValueError("PCZPPOWeighted requires reward_component_names.")
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
            self.rollout_buffer,  # type: ignore[arg-type]
            n_rollout_steps,
            self._reward_component_names,
        )
        if result is False:
            return False

        _, last_values, dones = result
        buf = self.rollout_buffer  # type: ignore[union-attr]

        # Per-component z-norm with weights
        pre_rewards = buf.component_rewards.sum(axis=2).copy()
        normalized = np.zeros_like(buf.component_rewards)
        for i in range(self._n_reward_components):
            normalized[:, :, i] = _znorm(buf.component_rewards[:, :, i], axis=0)

        buf.rewards = _weighted_component_sum(normalized, self._component_weights)
        log_normalization_diagnostics(self, pre_rewards, buf.rewards)
        buf._reapply_bootstrap()
        buf.compute_returns_and_advantage(last_values, dones)

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True


if __name__ == "__main__":
    run_cartpole_example(PCZPPOWeighted, "PCZ-PPO-Weighted")
