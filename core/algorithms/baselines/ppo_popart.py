"""ppo_popart.py: PPO-PopArt — PPO with PopArt adaptive value head rescaling.

Standard PPO combined with PopArt, which adaptively rescales the value head's
last linear layer to track changing return magnitudes. No reward normalization
is applied — PopArt handles the value-side scale adaptation.

Usage::

    from core.algorithms.ppo_popart import PPOPopArt

    model = PPOPopArt("MlpPolicy", env, reward_component_names=["a", "b"])
    model.learn(total_timesteps=50_000)
"""

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from .._common import (
    ComponentRolloutBuffer,
    PopArtMixin,
    _init_component_weights,
    _weighted_component_sum,
    collect_rollouts,
    run_cartpole_example,
)


class PPOPopArt(PopArtMixin, PPO):
    """PPO with PopArt adaptive value head rescaling.

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
            raise ValueError("PPOPopArt requires reward_component_names.")
        self._reward_component_names = list(reward_component_names)
        self._n_reward_components = len(self._reward_component_names)
        self._component_weights = _init_component_weights(self._n_reward_components, component_weights)
        super().__init__(*args, **kwargs)
        if self.clip_range_vf is not None:
            raise ValueError(
                "PPOPopArt is incompatible with clip_range_vf. After PopArt "
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

        # Apply component weights if provided (overwrite scalar with weighted sum)
        if self._component_weights is not None:
            buf.rewards -= buf.timeout_bootstrap
            buf.rewards = _weighted_component_sum(buf.component_rewards, self._component_weights)
            buf._reapply_bootstrap()
        # Standard GAE on (possibly re-weighted) scalar rewards
        buf.compute_returns_and_advantage(last_values, dones)

        # PopArt: update stats and rescale value head
        self._apply_popart_to_returns(buf)

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True


if __name__ == "__main__":
    run_cartpole_example(PPOPopArt, "PPO-PopArt")
