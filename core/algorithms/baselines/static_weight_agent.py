"""static_weight_agent.py: Equal-weight static-policy baseline for trading-kN.

Sanity-check baseline for the market-making project (S-MKT-1.2). The kill-risk
pre-registration for R-MKT-1 requires LQ-MPC to beat an "equal-weight static"
baseline by ≥20% mean episode return; this agent provides that comparator.

Maintains a rolling buffer of close prices and acts on the z-score of the
current close vs. that buffer:

    z = (close - rolling_mean) / rolling_std
    z >  +threshold  →  short  (mean-reversion: price high, expect to fall)
    z < -threshold  →  long   (mean-reversion: price low, expect to rise)
    else            →  flat

No optimisation, no model fitting beyond a running window. The "equal-weight"
naming refers to the kill-risk framing (no per-component-weight tuning). The
threshold is a single hyperparameter (default 0.5) that doesn't adapt online.

Dispatch note: ``is_tabular = True`` so train.py routes to
``_train_single_tabular``. The flag is a misnomer — see backlog follow-up.

Usage::

    uv run python -m core.algorithms.baselines.static_weight_agent  # smoke

    uv run python -m core.train --algorithm=static-weight --env=trading-k4 \\
        --total-timesteps=50000
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np


class StaticWeightAgent:
    """Equal-weight static z-score threshold baseline for trading-kN envs.

    Args:
        env_name: Environment name from ENV_REGISTRY.
        reward_component_names: Names of reward components (unused; accepted
            for tabular-dispatch interface compatibility).
        component_weights: Unused; accepted for interface compatibility.
        gamma: Unused.
        threshold: Z-score magnitude above which the agent takes a directional
            position. Default 0.5 — conservative mean-reversion entry.
        history_window: Rolling-buffer length for z-score computation.
            Default 200 (matches LQMPCAgent for fair comparison).
        warm_start_threshold: Min buffer length before agent acts; below this,
            holds flat. Default 30.
        seed: Random seed (unused; deterministic policy).
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
        threshold: float = 0.5,
        history_window: int = 200,
        warm_start_threshold: int = 30,
        seed: int = 42,
        **kwargs,
    ):
        self.env_name = env_name
        self._component_names = list(reward_component_names)
        self.threshold = float(threshold)
        self.history_window = int(history_window)
        self.warm_start_threshold = int(warm_start_threshold)
        self.seed = int(seed)

        self._price_buffer: deque[float] = deque(maxlen=self.history_window)
        self._last_close: float = 100.0

    def _make_env(self, render_mode: str | None = None):
        from ...env_config import make_env_factory

        factory = make_env_factory(self.env_name, render_mode=render_mode)
        return factory()

    def _zscore(self, x: float) -> float:
        if len(self._price_buffer) < self.warm_start_threshold:
            return 0.0
        arr = np.asarray(self._price_buffer, dtype=np.float64)
        mu = float(arr.mean())
        sd = float(arr.std(ddof=0))
        if sd < 1e-9:
            return 0.0
        return (x - mu) / sd

    def act(self, info: dict) -> int:
        close = float(info.get("data_close", self._last_close))
        self._last_close = close
        self._price_buffer.append(close)

        if len(self._price_buffer) < self.warm_start_threshold:
            return 1
        z = self._zscore(close)
        if z > self.threshold:
            return 0
        if z < -self.threshold:
            return 2
        return 1

    def predict(self, obs, deterministic: bool = True) -> tuple[int, None]:
        info = {"data_close": self._last_close}
        return self.act(info), None

    def learn(
        self,
        total_timesteps: int = 50_000,
        log_fn=None,
        log_interval: int = 1000,
    ) -> dict[str, float]:
        env = self._make_env()

        total_steps = 0
        episode = 0
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []

        print(f"\n{'=' * 60}")
        print("  Static-weight (z-score threshold) baseline")
        print(f"  Environment: {self.env_name}")
        print(f"  Components:  {self._component_names}")
        print(f"  Threshold:   {self.threshold}")
        print(f"  History:     {self.history_window} (warm-up {self.warm_start_threshold})")
        print(f"  Timesteps:   {total_timesteps:,}")
        print(f"{'=' * 60}\n")

        while total_steps < total_timesteps:
            _, info = env.reset(seed=self.seed + episode)
            self._price_buffer.clear()
            self._last_close = float(info.get("data_close", 100.0))
            done = False
            truncated = False
            ep_reward = 0.0
            ep_steps = 0
            while not (done or truncated) and total_steps < total_timesteps:
                action = self.act(info)
                _, reward, done, truncated, info = env.step(action)
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
        rewards = []
        for ep in range(n_episodes):
            _, info = env.reset(seed=self.seed + 10_000 + ep)
            self._price_buffer.clear()
            self._last_close = float(info.get("data_close", 100.0))
            done, truncated = False, False
            ep_reward = 0.0
            while not (done or truncated):
                action = self.act(info)
                _, reward, done, truncated, info = env.step(action)
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
        meta = {
            "agent": "StaticWeightAgent",
            "env_name": self.env_name,
            "threshold": self.threshold,
            "history_window": self.history_window,
            "component_names": self._component_names,
        }
        with open(path_obj / "static_weight_agent.json", "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: str) -> None:
        meta_path = Path(path) / "static_weight_agent.json"
        if not meta_path.exists():
            return
        with open(meta_path) as f:
            meta = json.load(f)
        self.threshold = float(meta.get("threshold", self.threshold))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="StaticWeightAgent smoke test")
    parser.add_argument("--env", default="trading-k4")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from ...env_config import get_env_config

    cfg = get_env_config(args.env)
    agent = StaticWeightAgent(
        args.env,
        reward_component_names=cfg.reward_components,
        threshold=args.threshold,
        seed=args.seed,
    )

    env = agent._make_env()
    obs, info = env.reset(seed=args.seed)
    agent._price_buffer.clear()
    agent._last_close = float(info.get("data_close", 100.0))

    total_reward = 0.0
    component_totals = dict.fromkeys(cfg.reward_components, 0.0)
    n_steps = 0
    n_episodes = 0
    actions_taken = {0: 0, 1: 0, 2: 0}
    for _ in range(args.steps):
        action = agent.act(info)
        actions_taken[int(action)] += 1
        obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        for name, val in info.get("reward_components", {}).items():
            component_totals[name] = component_totals.get(name, 0.0) + float(val)
        n_steps += 1
        if done or truncated:
            obs, info = env.reset(seed=args.seed + n_episodes + 1)
            agent._price_buffer.clear()
            agent._last_close = float(info.get("data_close", 100.0))
            n_episodes += 1

    env.close()

    print(f"\nStaticWeight smoke test on {args.env} ({args.steps} steps, thr={args.threshold}):")
    print(f"  episodes:        {n_episodes + 1}")
    print(f"  total reward:    {total_reward:+.4f}")
    print(f"  per-step reward: {total_reward / max(n_steps, 1):+.5f}")
    print(f"  action mix:      short={actions_taken[0]}  flat={actions_taken[1]}  long={actions_taken[2]}")
    print("  component breakdown:")
    for name, total in component_totals.items():
        print(f"    {name:15s}: {total:+.4f}")
