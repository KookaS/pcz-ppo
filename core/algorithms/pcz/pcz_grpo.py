"""pcz_grpo.py: PCZ-GRPO — critic-free multi-component RL with MC returns.

GDPO-GRPO (Group reward-Decoupled normalization with Group Relative Policy
Optimization) is a critic-free PPO variant that:

1. Computes Monte-Carlo discounted returns **per reward component**.
2. Z-normalizes each component's returns independently (per-env, across timesteps).
3. Sums the normalised returns and applies final batch whitening.

No value function is learned — the advantage is the normalised return itself,
following the GRPO formulation from DeepSeek-R1.

Requirements:
    - The environment must return ``info["reward_components"]``.

Usage::

    from core.algorithms.pcz_grpo import PCZGRPO

    model = PCZGRPO("MlpPolicy", env, reward_component_names=["a", "b"])
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


class PCZGRPO(PPO):
    """Critic-free PPO with GDPO per-component MC return z-normalization.

    No value function is learned (``vf_coef=0``). Advantages are computed as
    z-normalized discounted MC returns per reward component, summed across
    components with final batch whitening.

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
            raise ValueError(
                "PCZGRPO requires reward_component_names — a list of keys expected in info['reward_components']."
            )
        self._reward_component_names = list(reward_component_names)
        self._n_reward_components = len(self._reward_component_names)
        self._component_weights = _init_component_weights(self._n_reward_components, component_weights)
        kwargs["vf_coef"] = 0.0
        # Disable SB3's per-minibatch advantage normalization — this class
        # applies its own batch whitening after component summation.
        kwargs.setdefault("normalize_advantage", False)
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
            bootstrap_timeout=False,  # No critic → no bootstrap
        )
        if result is False:
            return False

        _, last_values, dones = result
        buf = self.rollout_buffer
        gamma = self.gamma

        # 1. Discounted Monte-Carlo returns per component
        component_returns = np.zeros_like(buf.component_rewards)
        for i in range(self._n_reward_components):
            last_return = np.zeros(self.n_envs, dtype=np.float32)
            for step in reversed(range(self.n_steps)):
                if step == self.n_steps - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32)
                else:
                    next_non_terminal = 1.0 - buf.episode_starts[step + 1]
                last_return = buf.component_rewards[step, :, i] + gamma * last_return * next_non_terminal
                component_returns[step, :, i] = last_return

        # 2. Z-normalize each component's returns per-env
        normalized_returns = np.zeros_like(component_returns)
        for i in range(self._n_reward_components):
            normalized_returns[:, :, i] = _znorm(component_returns[:, :, i], axis=0)

        # 3. Sum normalised components
        combined = _weighted_component_sum(normalized_returns, self._component_weights)
        pre_returns = component_returns.sum(axis=2)
        log_normalization_diagnostics(self, pre_returns, combined)

        # 4. Final batch whitening
        mean = combined.mean()
        std = combined.std()
        if std > 1e-8:
            buf.advantages = (combined - mean) / std
        else:
            buf.advantages = combined - mean

        # Returns = summed MC returns (not normalized). vf_coef=0 so value loss is zero.
        buf.returns = _weighted_component_sum(component_returns, self._component_weights)

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True


if __name__ == "__main__":
    run_cartpole_example(PCZGRPO, "PCZ-GRPO (Critic-Free)")
