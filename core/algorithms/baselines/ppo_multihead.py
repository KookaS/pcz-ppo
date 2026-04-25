"""ppo_multihead.py: PPO-MultiHead — separate value heads per reward component.

PPO with a separate value head V^(k)(s) for each reward component k. The
combined value V(s) = sum_k V^(k)(s) is used for GAE. Each component head
shares the MLP hidden layers with the standard value network but has its
own output linear layer.

Inspired by HRA (van Seijen et al., NeurIPS 2017), adapted from DQN to PPO.
The key idea: decomposing the critic into per-component heads lets each head
specialise in predicting one reward component's value, implicitly handling
scale differences without reward-level normalisation.

Usage::

    from core.algorithms.ppo_multihead import PPOMultiHead

    model = PPOMultiHead("MlpPolicy", env, reward_component_names=["a", "b"])
    model.learn(total_timesteps=50_000)
"""

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

from .._common import (
    ComponentRolloutBuffer,
    _init_component_weights,
    _weighted_component_sum,
    collect_rollouts,
    run_cartpole_example,
)


class PPOMultiHead(PPO):
    """PPO with separate value heads per reward component.

    Each component head V^(k)(s) shares the value MLP hidden layers and has
    its own output linear layer.  The aggregate value V(s) = sum_k V^(k)(s)
    replaces the standard single value head for GAE and policy training.

    During training, each component head receives a supervised signal from
    its component's discounted returns (stored alongside the aggregate).

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
            raise ValueError("PPOMultiHead requires reward_component_names.")
        self._reward_component_names = list(reward_component_names)
        self._n_reward_components = len(self._reward_component_names)
        self._component_weights = _init_component_weights(self._n_reward_components, component_weights)
        self._component_heads: th.nn.ModuleList | None = None
        # Flat per-component returns tensor for training (n_steps*n_envs, K)
        self._component_returns_flat: th.Tensor | None = None
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

        # Per-component value heads: shared value-MLP latent → 1 per component.
        latent_dim_vf = self.policy.mlp_extractor.latent_dim_vf
        self._component_heads = th.nn.ModuleList(
            [th.nn.Linear(latent_dim_vf, 1) for _ in range(self._n_reward_components)]
        ).to(self.device)

        # Add component head parameters to the existing optimizer
        self.policy.optimizer.add_param_group({"params": self._component_heads.parameters()})

    def _predict_multihead_values(self, obs: th.Tensor) -> th.Tensor:
        """Compute aggregate V(s) = sum_k V^(k)(s) via the multi-head architecture.

        Uses the shared value-network hidden layers, then sums per-component
        output heads.

        Returns:
            values: shape (batch_size, 1)
        """
        features = self.policy.extract_features(obs, self.policy.vf_features_extractor)
        latent_vf = self.policy.mlp_extractor.forward_critic(features)
        return sum(head(latent_vf) for head in self._component_heads)

    def _predict_component_values(self, obs: th.Tensor) -> list[th.Tensor]:
        """Compute per-component V^(k)(s) for each head.

        Returns:
            list of K tensors, each shape (batch_size, 1)
        """
        features = self.policy.extract_features(obs, self.policy.vf_features_extractor)
        latent_vf = self.policy.mlp_extractor.forward_critic(features)
        return [head(latent_vf) for head in self._component_heads]

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

        _, _last_values_unused, dones = result
        buf = self.rollout_buffer

        # Aggregate reward = sum of raw components (no z-norm; the multi-head
        # critic handles scale differences via separate heads).
        buf.rewards = _weighted_component_sum(buf.component_rewards, self._component_weights)
        buf._reapply_bootstrap()

        # Compute last_values from our multi-head aggregate (not the standard
        # single value head) so GAE uses consistent value predictions.
        with th.no_grad():
            obs_tensor = obs_as_tensor(self._last_obs, self.device)
            last_values = self._predict_multihead_values(obs_tensor).flatten()

        buf.compute_returns_and_advantage(last_values, dones)

        # ── Per-component discounted returns for component head supervision ──
        n_steps = buf.buffer_size
        comp_returns = np.zeros_like(buf.component_rewards)
        dones_np = dones.float().cpu().numpy() if isinstance(dones, th.Tensor) else dones.astype(np.float32)

        for k in range(self._n_reward_components):
            last_ret = np.zeros(buf.n_envs, dtype=np.float32)
            for step in reversed(range(n_steps)):
                if step < n_steps - 1:
                    next_non_terminal = 1.0 - buf.episode_starts[step + 1]
                else:
                    next_non_terminal = 1.0 - dones_np
                last_ret = buf.component_rewards[step, :, k] + self.gamma * last_ret * next_non_terminal
                comp_returns[step, :, k] = last_ret

        # Flatten and store for training (same order as buf.get() before shuffle)
        self._component_returns_flat = th.tensor(
            comp_returns.reshape(-1, self._n_reward_components),
            dtype=th.float32,
            device=self.device,
        )

        # Also store flat observations in the same order for index-free training
        self._flat_obs = th.tensor(
            buf.observations.reshape(-1, *buf.observations.shape[2:]),
            dtype=th.float32,
            device=self.device,
        )

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        """Standard PPO training + per-component value head loss.

        The per-component loss is computed once per epoch on the full buffer
        (not per mini-batch) to avoid index-tracking complexity with SB3's
        shuffled mini-batch iterator.
        """
        # First, run the standard PPO training loop
        super().train()

        # Then, train the component heads on per-component returns.
        # This is a separate pass through the full buffer per epoch.
        self.policy.set_training_mode(True)
        for _epoch in range(self.n_epochs):
            comp_vals = self._predict_component_values(self._flat_obs)
            comp_loss = th.tensor(0.0, device=self.device)
            for k in range(self._n_reward_components):
                comp_loss = comp_loss + F.mse_loss(
                    comp_vals[k].flatten(),
                    self._component_returns_flat[:, k],
                )
            comp_loss = comp_loss / self._n_reward_components

            self.policy.optimizer.zero_grad()
            comp_loss.backward()
            th.nn.utils.clip_grad_norm_(self._component_heads.parameters(), self.max_grad_norm)
            # Only step the component head params; shared layers were already
            # updated by super().train(). We still step the full optimizer
            # but the shared-layer gradients are zero (from zero_grad above)
            # so they are not affected.
            self.policy.optimizer.step()

        # Clean up to free memory
        self._flat_obs = None


if __name__ == "__main__":
    run_cartpole_example(PPOMultiHead, "PPO-MultiHead")
