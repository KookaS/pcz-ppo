"""qlearning.py: Tabular Q-Learning with multi-component reward scalarization.

Tabular Q-learning baseline for discrete-state environments. Uses epsilon-greedy
exploration with exponential decay. Rewards are scalarized via component weights
before Q-value updates (weighted sum of per-component rewards).

Only works with environments that have discrete, small state spaces (e.g.
resource-gathering). Continuous-state environments like CartPole or LunarLander
require function approximation and are not supported.

Usage::

    uv run python -m core.train --algorithm=qlearning --env=resource \\
        --total-timesteps=50000 --no-eval --no-mlflow

    # Or directly:
    uv run python -m core.algorithms.qlearning
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .._common import _init_component_weights


class TabularQLearning:
    """Tabular Q-Learning with multi-component reward scalarization.

    Stores a Q-table as ``{state_tuple: np.array(n_actions)}``.  Component
    rewards are scalarized with ``component_weights`` before the TD update.

    Args:
        env_name: Environment name from ``ENV_REGISTRY``.
        reward_component_names: List of reward component names.
        component_weights: Per-component scalarization weights.
        alpha: Learning rate for Q-value updates.
        gamma: Discount factor.
        epsilon_start: Initial exploration rate.
        epsilon_end: Minimum exploration rate.
        epsilon_decay: Multiplicative decay per episode.
        seed: Random seed.
    """

    is_self_normalizing = True
    is_tabular = True

    def __init__(
        self,
        env_name: str,
        *,
        reward_component_names: list[str],
        component_weights: list[float] | None = None,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.99995,
        seed: int = 42,
        **kwargs,
    ):
        self.env_name = env_name
        self._component_names = list(reward_component_names)
        self._n_components = len(self._component_names)
        self._weights = _init_component_weights(self._n_components, component_weights)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.seed = seed

        self.q_table: dict[tuple, np.ndarray] = defaultdict(self._zero_q)
        self._n_actions: int | None = None
        self._rng = np.random.default_rng(seed)

    def _zero_q(self) -> np.ndarray:
        n = self._n_actions if self._n_actions is not None else 4
        return np.zeros(n)

    def _scalarize(self, reward_components: dict[str, float]) -> float:
        total = 0.0
        for i, name in enumerate(self._component_names):
            total += self._weights[i] * reward_components.get(name, 0.0)
        return total

    def _make_env(self, render_mode: str | None = None):
        from ..env_config import make_env_factory

        factory = make_env_factory(self.env_name, render_mode=render_mode)
        return factory()

    def learn(
        self,
        total_timesteps: int = 50_000,
        log_fn=None,
        log_interval: int = 1000,
    ) -> dict[str, float]:
        """Train the Q-table using epsilon-greedy exploration.

        Args:
            total_timesteps: Total environment steps budget.
            log_fn: Optional ``(metrics_dict, step)`` callback for MLflow.
            log_interval: Log metrics every N steps.

        Returns:
            Dict with final eval metrics.
        """
        env = self._make_env()
        self._n_actions = env.action_space.n

        epsilon = self.epsilon_start
        total_steps = 0
        episode = 0
        episode_rewards = []
        episode_lengths = []

        print(f"\n{'=' * 60}")
        print("  Tabular Q-Learning")
        print(f"  Environment: {self.env_name}")
        print(f"  Components:  {self._component_names}")
        weight_str = ", ".join(f"{n}={w:.1f}" for n, w in zip(self._component_names, self._weights))
        print(f"  Weights:     {weight_str}")
        print(f"  Timesteps:   {total_timesteps:,}")
        print(f"  Alpha:       {self.alpha}")
        print(f"  Gamma:       {self.gamma}")
        print(f"  Epsilon:     {self.epsilon_start} -> {self.epsilon_end}")
        print(f"{'=' * 60}\n")

        while total_steps < total_timesteps:
            obs, info = env.reset()
            state = tuple(obs.flat) if hasattr(obs, "flat") else tuple(np.asarray(obs).flat)
            done = False
            truncated = False
            ep_reward = 0.0
            ep_steps = 0

            while not (done or truncated) and total_steps < total_timesteps:
                # Epsilon-greedy
                if self._rng.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = int(np.argmax(self.q_table[state]))

                next_obs, reward, done, truncated, info = env.step(action)
                next_state = tuple(next_obs.flat) if hasattr(next_obs, "flat") else tuple(np.asarray(next_obs).flat)

                # Scalarize reward from components
                components = info.get("reward_components", {})
                if components:
                    scalar_reward = self._scalarize(components)
                else:
                    scalar_reward = float(reward)

                # Q-learning update
                best_next_q = np.max(self.q_table[next_state])
                td_target = scalar_reward + self.gamma * best_next_q * (1.0 - float(done))
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.alpha * td_error

                state = next_state
                ep_reward += scalar_reward
                ep_steps += 1
                total_steps += 1

            # Decay epsilon per episode
            epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)
            episode += 1
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_steps)

            # Logging
            if total_steps % log_interval < ep_steps + 1:
                recent = episode_rewards[-100:]
                mean_r = np.mean(recent)
                metrics = {
                    "rollout/ep_rew_mean": float(mean_r),
                    "rollout/ep_len_mean": float(np.mean(episode_lengths[-100:])),
                    "rollout/epsilon": float(epsilon),
                    "rollout/q_table_size": len(self.q_table),
                }
                if log_fn is not None:
                    log_fn(metrics, total_steps)
                if episode % 500 == 0:
                    print(
                        f"  Step {total_steps:>7,} | Ep {episode:>5} | "
                        f"eps={epsilon:.4f} | mean_r={mean_r:.3f} | "
                        f"states={len(self.q_table)}"
                    )

        env.close()

        # Final stats
        if episode_rewards:
            recent = episode_rewards[-100:]
            print(f"\n  Training done: {episode} episodes, {total_steps:,} steps")
            print(f"  Final 100-ep mean reward: {np.mean(recent):.3f}")
            print(f"  Q-table size: {len(self.q_table)} states")

        return {
            "eval/mean_reward": float(np.mean(episode_rewards[-100:])) if episode_rewards else 0.0,
            "eval/std_reward": float(np.std(episode_rewards[-100:])) if episode_rewards else 0.0,
        }

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        state = tuple(obs.flat) if hasattr(obs, "flat") else tuple(np.asarray(obs).flat)
        action = int(np.argmax(self.q_table[state]))
        return action, None

    def evaluate(self, n_episodes: int = 10) -> dict[str, float]:
        """Run greedy evaluation episodes."""
        env = self._make_env()
        rewards = []
        for _ in range(n_episodes):
            obs, info = env.reset()
            state = tuple(obs.flat) if hasattr(obs, "flat") else tuple(np.asarray(obs).flat)
            done, truncated = False, False
            ep_reward = 0.0
            while not (done or truncated):
                action = int(np.argmax(self.q_table[state]))
                obs, reward, done, truncated, info = env.step(action)
                state = tuple(obs.flat) if hasattr(obs, "flat") else tuple(np.asarray(obs).flat)
                components = info.get("reward_components", {})
                ep_reward += self._scalarize(components) if components else float(reward)
            rewards.append(ep_reward)
        env.close()
        return {
            "eval/mean_reward": float(np.mean(rewards)),
            "eval/std_reward": float(np.std(rewards)),
        }

    def save(self, path: str) -> None:
        """Save Q-table to JSON."""
        serializable = {}
        for state, values in self.q_table.items():
            key = json.dumps([float(x) for x in state])
            serializable[key] = values.tolist()
        data = {
            "n_actions": int(self._n_actions),
            "env_name": self.env_name,
            "component_names": self._component_names,
            "component_weights": self._weights.tolist(),
            "q_table": serializable,
        }
        p = Path(path)
        if p.is_dir():
            p = p / "q_table.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data))
        print(f"  Q-table saved to {p} ({len(serializable)} states)")

    def load(self, path: str) -> None:
        """Load Q-table from JSON."""
        p = Path(path)
        if p.is_dir():
            p = p / "q_table.json"
        data = json.loads(p.read_text())
        self._n_actions = data["n_actions"]
        self.q_table = defaultdict(self._zero_q)
        for key, values in data["q_table"].items():
            state = tuple(json.loads(key))
            self.q_table[state] = np.array(values)
        print(f"  Q-table loaded from {p} ({len(self.q_table)} states)")


if __name__ == "__main__":
    print("=== Tabular Q-Learning: Resource Gathering smoke test ===\n")
    model = TabularQLearning(
        "resource",
        reward_component_names=["death_reward", "gold", "diamond"],
        component_weights=[0.2, 1.0, 0.5],
        seed=42,
    )
    metrics = model.learn(total_timesteps=5_000)
    print(f"\nSmoke test result: {metrics}")

    eval_metrics = model.evaluate(n_episodes=5)
    print(f"Eval result: {eval_metrics}")
    print("\nSmoke test passed!")
