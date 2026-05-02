"""random_agent.py: Random-action baseline for trading-kN envs.

Sanity-check baseline for the market-making project (S-MKT-1.2). The kill-risk
pre-registration for R-MKT-1 requires LQ-MPC to beat random-action by ≥20%
mean episode return; this agent provides that lower-bound reference.

Samples uniformly from ``env.action_space`` at every step. No learning, no
state, no observation use.

Dispatch note: ``is_tabular = True`` so train.py routes to
``_train_single_tabular`` (instantiate, learn, evaluate, save). The flag is a
misnomer — see backlog follow-up to rename to ``is_planner``.

Usage::

    uv run python -m core.algorithms.baselines.random_agent  # smoke test

    uv run python -m core.train --algorithm=random-action --env=trading-k4 \\
        --total-timesteps=50000
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class RandomAgent:
    """Random-action baseline.

    Args:
        env_name: Environment name from ENV_REGISTRY.
        reward_component_names: Names of reward components (unused; accepted
            for tabular-dispatch interface compatibility).
        component_weights: Unused; accepted for interface compatibility.
        gamma: Unused (no learning).
        seed: Random seed for the action sampler.
    """

    is_self_normalizing = False
    is_torchrl = False
    is_tabular = True

    def __init__(
        self,
        env_name: str,
        *,
        reward_component_names: list[str],
        component_weights: list[float] | None = None,
        gamma: float = 0.99,
        seed: int = 42,
        **kwargs,
    ):
        self.env_name = env_name
        self._component_names = list(reward_component_names)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)

    def _make_env(self, render_mode: str | None = None):
        from ...env_config import make_env_factory

        factory = make_env_factory(self.env_name, render_mode=render_mode)
        return factory()

    def act(self, action_space) -> int:
        return int(action_space.sample())

    def predict(self, obs, deterministic: bool = True) -> tuple[int, None]:
        return int(self._rng.integers(0, 3)), None

    def learn(
        self,
        total_timesteps: int = 50_000,
        log_fn=None,
        log_interval: int = 1000,
    ) -> dict[str, float]:
        env = self._make_env()
        env.action_space.seed(self.seed)

        total_steps = 0
        episode = 0
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []

        print(f"\n{'=' * 60}")
        print("  Random-action baseline")
        print(f"  Environment: {self.env_name}")
        print(f"  Components:  {self._component_names}")
        print(f"  Timesteps:   {total_timesteps:,}")
        print(f"{'=' * 60}\n")

        while total_steps < total_timesteps:
            _, _info = env.reset(seed=self.seed + episode)
            done = False
            truncated = False
            ep_reward = 0.0
            ep_steps = 0
            while not (done or truncated) and total_steps < total_timesteps:
                action = self.act(env.action_space)
                _, reward, done, truncated, _info = env.step(action)
                ep_reward += float(reward)
                ep_steps += 1
                total_steps += 1

            episode += 1
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_steps)

            if total_steps % log_interval < ep_steps + 1:
                recent = episode_rewards[-100:]
                metrics = {
                    "rollout/ep_rew_mean": float(np.mean(recent)),
                    "rollout/ep_len_mean": float(np.mean(episode_lengths[-100:])),
                }
                if log_fn is not None:
                    log_fn(metrics, total_steps)
                if episode % 50 == 0:
                    print(f"  Step {total_steps:>7,} | Ep {episode:>4} | mean_r={metrics['rollout/ep_rew_mean']:+.3f}")

        env.close()
        if episode_rewards:
            recent = episode_rewards[-100:]
            print(f"\n  Rollout done: {episode} episodes, {total_steps:,} steps")
            print(f"  Final 100-ep mean reward: {np.mean(recent):+.3f} ± {np.std(recent):.3f}")

        return {
            "eval/mean_reward": float(np.mean(episode_rewards[-100:])) if episode_rewards else 0.0,
            "eval/std_reward": float(np.std(episode_rewards[-100:])) if episode_rewards else 0.0,
        }

    def evaluate(self, n_episodes: int = 10) -> dict[str, float]:
        env = self._make_env()
        env.action_space.seed(self.seed + 10_000)
        rewards = []
        for ep in range(n_episodes):
            _, _info = env.reset(seed=self.seed + 10_000 + ep)
            done, truncated = False, False
            ep_reward = 0.0
            while not (done or truncated):
                action = self.act(env.action_space)
                _, reward, done, truncated, _info = env.step(action)
                ep_reward += float(reward)
            rewards.append(ep_reward)
        env.close()
        return {
            "eval/mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "eval/std_reward": float(np.std(rewards)) if rewards else 0.0,
        }

    def save(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        meta = {"agent": "RandomAgent", "env_name": self.env_name, "seed": self.seed}
        with open(path_obj / "random_agent.json", "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: str) -> None:
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Random-action smoke test")
    parser.add_argument("--env", default="trading-k4")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from ...env_config import get_env_config

    cfg = get_env_config(args.env)
    agent = RandomAgent(
        args.env,
        reward_component_names=cfg.reward_components,
        seed=args.seed,
    )

    env = agent._make_env()
    env.action_space.seed(args.seed)
    obs, info = env.reset(seed=args.seed)

    total_reward = 0.0
    component_totals = dict.fromkeys(cfg.reward_components, 0.0)
    n_steps = 0
    n_episodes = 0
    actions_taken = {0: 0, 1: 0, 2: 0}
    for _ in range(args.steps):
        action = agent.act(env.action_space)
        actions_taken[int(action)] += 1
        obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        for name, val in info.get("reward_components", {}).items():
            component_totals[name] = component_totals.get(name, 0.0) + float(val)
        n_steps += 1
        if done or truncated:
            obs, info = env.reset(seed=args.seed + n_episodes + 1)
            n_episodes += 1

    env.close()

    print(f"\nRandom-action smoke test on {args.env} ({args.steps} steps):")
    print(f"  episodes:        {n_episodes + 1}")
    print(f"  total reward:    {total_reward:+.4f}")
    print(f"  per-step reward: {total_reward / max(n_steps, 1):+.5f}")
    print(f"  action mix:      short={actions_taken[0]}  flat={actions_taken[1]}  long={actions_taken[2]}")
    print("  component breakdown:")
    for name, total in component_totals.items():
        print(f"    {name:15s}: {total:+.4f}")
