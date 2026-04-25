"""ppo_adv_only.py: PPO with advantage whitening only (no reward normalization).

Uses standard PPO advantage normalization (per-minibatch whitening) but does
not apply any reward normalization. This isolates the effect of advantage
normalization from reward normalization.

Usage::

    from core.algorithms.ppo_adv_only import PPOAdvOnly

    model = PPOAdvOnly("MlpPolicy", env, reward_component_names=["a", "b"])
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
    collect_rollouts,
    run_cartpole_example,
)


class PPOAdvOnly(PPO):
    """PPO with advantage whitening only (no reward normalization).

    Advantage normalization (per-minibatch mean/std whitening) is enabled,
    but no reward normalization is applied. This is the default SB3 PPO
    behavior when VecNormalize is not used.

    Args:
        reward_component_names: Optional list of component names expected in
            ``info["reward_components"]``. Accepted for API consistency.
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
        self._reward_component_names = list(reward_component_names) if reward_component_names else []
        self._n_reward_components = len(self._reward_component_names)
        self._component_weights = (
            _init_component_weights(self._n_reward_components, component_weights)
            if self._n_reward_components > 0
            else None
        )
        kwargs.setdefault("normalize_advantage", True)
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        if self._n_reward_components > 0:
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
        if not self._reward_component_names:
            return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

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
        # Apply component weights if provided (overwrite scalar with weighted sum)
        if self._component_weights is not None:
            buf.rewards -= buf.timeout_bootstrap
            buf.rewards = _weighted_component_sum(buf.component_rewards, self._component_weights)
            buf._reapply_bootstrap()
        buf.compute_returns_and_advantage(last_values, dones)

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True


if __name__ == "__main__":
    run_cartpole_example(PPOAdvOnly, "PPO Advantage-Only Normalization")
