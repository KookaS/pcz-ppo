"""ppo_znorm.py: PPO with z-normalization on the aggregate scalar reward.

Z-normalizes the scalar reward (from ``env.step()``) per-env across timesteps
before computing GAE. This is the simplest form of reward normalization ---
it treats the aggregate reward as a single signal and normalizes it.

Usage::

    from core.algorithms.ppo_znorm import PPOZnorm

    model = PPOZnorm("MlpPolicy", env, reward_component_names=["a", "b"])
    model.learn(total_timesteps=50_000)
"""

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


class PPOZnorm(PPO):
    """PPO with z-normalization on the aggregate scalar reward.

    The scalar reward stored by ``env.step()`` is z-normalized per-env across
    timesteps within each rollout buffer, then standard GAE is computed.

    Args:
        reward_component_names: List of component names expected in
            ``info["reward_components"]``. Must be provided.
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
                "PPOZnorm requires reward_component_names — a list of keys expected in info['reward_components']."
            )
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
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
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

        # Strip bootstrap before z-norm so it isn't attenuated, then re-apply
        buf.rewards -= buf.timeout_bootstrap
        # Apply component weights if provided (overwrite scalar with weighted sum)
        if self._component_weights is not None:
            buf.rewards = _weighted_component_sum(buf.component_rewards, self._component_weights)
        pre_rewards = buf.rewards.copy()
        buf.rewards = _znorm(buf.rewards, axis=0)
        log_normalization_diagnostics(self, pre_rewards, buf.rewards)
        buf._reapply_bootstrap()
        buf.compute_returns_and_advantage(last_values, dones)

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True


if __name__ == "__main__":
    run_cartpole_example(PPOZnorm, "PPO Z-Norm (Aggregate Scalar)")
