"""ppo_znorm_post.py: PPO with z-normalization after summing reward components.

Sums the raw per-component rewards into a scalar, then z-normalizes that
scalar per-env across timesteps. This is the "normalize-after-aggregate"
baseline that GDPO is designed to improve upon.

Usage::

    from core.algorithms.ppo_znorm_post import PPOZnormPost

    model = PPOZnormPost("MlpPolicy", env, reward_component_names=["a", "b"])
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


class PPOZnormPost(PPO):
    """PPO with z-normalization after summing reward components.

    Raw per-component rewards are summed, then the scalar sum is z-normalized
    per-env across timesteps. This is the GRPO-style "aggregate-then-normalize"
    approach that GDPO's "normalize-then-aggregate" is designed to improve upon.

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
                "PPOZnormPost requires reward_component_names — a list of keys expected in info['reward_components']."
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

        # Weighted sum of raw components, then z-norm the scalar
        raw_sum = _weighted_component_sum(buf.component_rewards, self._component_weights)
        pre_rewards = raw_sum.copy()
        buf.rewards = _znorm(raw_sum, axis=0)
        log_normalization_diagnostics(self, pre_rewards, buf.rewards)
        buf._reapply_bootstrap()
        buf.compute_returns_and_advantage(last_values, dones)

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True


if __name__ == "__main__":
    run_cartpole_example(PPOZnormPost, "PPO Z-Norm Post (Sum Then Normalize)")
