"""mpc_lq.py: Linear-Quadratic Model-Predictive-Control baseline for trading envs.

Classical MPC baseline for the market-making project (E-MKT-1). At each step,
fits a one-parameter AR(1) (a discrete-time Ornstein-Uhlenbeck) on a rolling
window of recent prices, forecasts the next H price changes, and solves a QP
over a continuous position trajectory. The first action of the optimal
trajectory is executed (receding horizon); the QP is warm-started across calls.

LQ formulation, per planning step at time t:

    minimize_p Σ_{k=0..H-1} [ -w_pnl Δx̂_{t+k} p_k
                              + w_txn (p_k - p_{k-1})²
                              + w_borrow p_k² ]
    subject to     -1 ≤ p_k ≤ 1            (position bounds)
                   p_{-1} = current_position

where:
    Δx̂_{t+k} = AR(1) k-step price-change forecast
    w_pnl, w_txn, w_borrow = component weights from EnvConfig

After solving, the continuous p_0* is rounded to the nearest discrete action
in {-1, 0, +1} via threshold 0.33.

Approximations / design choices:
- LQ uses **quadratic** transaction cost (p_k - p_{k-1})² as a stand-in for the
  env's actual **linear** |Δp| cost. This is the canonical LQ-MPC simplification
  and is exactly what kill-risk R-MKT-1 tests: if the LQ form is too coarse,
  R-MKT-1 fires and the project pivots to NMPC via do-mpc.
- For K=4 components (pnl_gain, pnl_loss, txn_cost, borrow_cost), the gain/loss
  weights are averaged into a single w_pnl. This is exact for default equal
  weights (None → all 1.0) and an approximation for asymmetric weights.
- AR(1) is fit by simple OLS on the rolling price buffer. The env's underlying
  process IS an AR(1) (theta=0.05, mu=100), so OLS recovers the true parameters
  asymptotically. The agent does NOT call any private env method — it only
  reads `info["data_close"]` from step() output (option (b) per the
  market-making backlog S-MKT-1.1 design decision).

Dispatch note: ``is_tabular = True`` is set so train.py routes to
``_train_single_tabular`` (which is shape-compatible — instantiate, learn,
evaluate, save). The flag is a misnomer (LQMPCAgent is not tabular); see
follow-up backlog item to rename the dispatch flag to ``is_planner``.

Usage::

    uv run python -m core.algorithms.mpc_lq          # smoke test (1000 steps)

    uv run python -m core.train --algorithm=mpc-lq --env=trading-k4 \\
        --total-timesteps=50000

    >>> from core import ALGORITHM_REGISTRY
    >>> agent = ALGORITHM_REGISTRY["mpc-lq"]("trading-k4",
    ...     reward_component_names=["pnl_gain", "pnl_loss", "txn_cost", "borrow_cost"],
    ...     horizon=20)
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np


class LQMPCAgent:
    """Linear-quadratic MPC baseline for trading-kN envs.

    Args:
        env_name: Environment name from ENV_REGISTRY.
        reward_component_names: Names of reward components.
        component_weights: Per-component weights (mapped to QP cost coefficients).
        horizon: Planning horizon H (default 20).
        history_window: Rolling window for AR(1) fitting (default 200).
        warm_start_threshold: Min history length before MPC engages; below this,
            agent acts flat (default 30).
        gamma: Unused (RL-discount factor; LQ-MPC doesn't use it). Accepted for
            interface compatibility with the tabular-dispatch path.
        seed: Random seed.
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
        horizon: int = 20,
        history_window: int = 200,
        warm_start_threshold: int = 30,
        gamma: float = 0.99,
        seed: int = 42,
        **kwargs,
    ):
        self.env_name = env_name
        self._component_names = list(reward_component_names)
        self._n_components = len(reward_component_names)
        if component_weights is None:
            self._weights = np.ones(self._n_components, dtype=np.float64)
        else:
            if len(component_weights) != self._n_components:
                raise ValueError(f"Expected {self._n_components} component weights, got {len(component_weights)}")
            self._weights = np.asarray(component_weights, dtype=np.float64)
        self.horizon = int(horizon)
        self.history_window = int(history_window)
        self.warm_start_threshold = int(warm_start_threshold)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)

        self._w_pnl, self._w_txn, self._w_borrow = self._map_weights()

        self._price_buffer: deque[float] = deque(maxlen=self.history_window)
        self._last_close: float = 100.0
        self._last_position: float = 0.0
        self._ar_phi: float = 0.95
        self._ar_mu: float = 100.0

        self._build_problem()

    def _map_weights(self) -> tuple[float, float, float]:
        """Map per-component weights to (w_pnl, w_txn, w_borrow) QP coefficients.

        For trading-k4 (pnl_gain, pnl_loss, txn_cost, borrow_cost), the gain/loss
        weights are averaged. For trading-k2 (pnl, costs), pnl weight is taken
        directly and costs weight is split between txn and borrow. For other K
        the agent falls back to mean-of-positives / mean-of-negatives heuristics
        documented inline.
        """
        names = self._component_names
        weights = self._weights

        wpnl_acc, wpnl_n = 0.0, 0
        wtxn_acc, wtxn_n = 0.0, 0
        wbor_acc, wbor_n = 0.0, 0

        for n, w in zip(names, weights):
            lower = n.lower()
            w_f64 = float(w)
            if "pnl" in lower or "gain" in lower or "loss" in lower:
                wpnl_acc += w_f64
                wpnl_n += 1
            elif "txn" in lower or "spread" in lower or lower == "costs":
                wtxn_acc += w_f64
                wtxn_n += 1
            elif "borrow" in lower or "residual" in lower or "var" in lower or "bankrupt" in lower:
                wbor_acc += w_f64
                wbor_n += 1
            else:
                wbor_acc += w_f64
                wbor_n += 1

        w_pnl = (wpnl_acc / wpnl_n) if wpnl_n else 1.0
        w_txn = (wtxn_acc / wtxn_n) if wtxn_n else 1.0
        w_borrow = (wbor_acc / wbor_n) if wbor_n else 1.0
        return float(w_pnl), float(w_txn), float(w_borrow)

    def _build_problem(self) -> None:
        """Construct cvxpy QP once. Parameters get re-bound each act()."""
        import cvxpy as cp

        H = self.horizon
        self._p = cp.Variable(H, name="position_traj")
        self._dx = cp.Parameter(H, name="dx_forecast")
        self._p_init = cp.Parameter(name="p_init")

        delta = cp.hstack([self._p[0] - self._p_init, self._p[1:] - self._p[:-1]])

        pnl_term = cp.sum(cp.multiply(self._dx, self._p))
        txn_term = cp.sum_squares(delta)
        hold_term = cp.sum_squares(self._p)

        objective = cp.Minimize(-self._w_pnl * pnl_term + self._w_txn * txn_term + self._w_borrow * hold_term)
        constraints = [self._p >= -1.0, self._p <= 1.0]
        self._problem = cp.Problem(objective, constraints)

    def _fit_ar1(self) -> None:
        """OLS fit of x_{t+1} = a + b*x_t on the rolling price buffer.

        Sets self._ar_phi (= b, persistence) and self._ar_mu (= a/(1-b), mean).
        Falls back to (phi=0.95, mu=last_close) if buffer is short or singular.
        """
        if len(self._price_buffer) < self.warm_start_threshold:
            return
        x = np.asarray(self._price_buffer, dtype=np.float64)
        x_t = x[:-1]
        x_t1 = x[1:]
        x_mean = x_t.mean()
        var = np.sum((x_t - x_mean) ** 2)
        if var < 1e-9:
            return
        cov = np.sum((x_t - x_mean) * (x_t1 - x_t1.mean()))
        b = float(cov / var)
        b = float(np.clip(b, 0.0, 0.999))
        a = float(x_t1.mean() - b * x_mean)
        denom = 1.0 - b
        mu = a / denom if abs(denom) > 1e-6 else x_mean
        self._ar_phi = b
        self._ar_mu = float(mu)

    def _forecast_dx(self, x_curr: float) -> np.ndarray:
        """K-step AR(1) deterministic forecast → array of price changes."""
        H = self.horizon
        phi, mu = self._ar_phi, self._ar_mu
        future = np.empty(H + 1, dtype=np.float64)
        future[0] = x_curr
        for k in range(1, H + 1):
            future[k] = mu + phi * (future[k - 1] - mu)
        return np.diff(future)

    def _solve(self, dx: np.ndarray, p_init: float) -> float:
        """Solve QP and return p_0* (continuous position in [-1, 1])."""
        import cvxpy as cp

        self._dx.value = dx.astype(np.float64)
        self._p_init.value = float(p_init)
        try:
            self._problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except cp.error.SolverError:
            return p_init
        if self._problem.status not in ("optimal", "optimal_inaccurate") or self._p.value is None:
            return p_init
        return float(self._p.value[0])

    @staticmethod
    def _continuous_to_discrete(p_continuous: float, threshold: float = 0.33) -> int:
        """Map continuous position p* ∈ [-1, 1] to Discrete(3) action index.

        Trading env action space: 0=short(-1), 1=flat(0), 2=long(+1).
        """
        if p_continuous < -threshold:
            return 0
        if p_continuous > threshold:
            return 2
        return 1

    def act(self, info: dict) -> int:
        """Compute the next discrete action given env info dict.

        Reads info["data_close"] and info["position"] (set by gym_trading_env).
        Falls back to flat (action=1) if the close price is unavailable.
        """
        close = float(info.get("data_close", self._last_close))
        position = float(info.get("position", self._last_position))
        self._last_close = close
        self._last_position = position
        self._price_buffer.append(close)

        if len(self._price_buffer) < self.warm_start_threshold:
            return 1

        self._fit_ar1()
        dx = self._forecast_dx(close)
        p_star = self._solve(dx, p_init=position)
        return self._continuous_to_discrete(p_star)

    def predict(self, obs, deterministic: bool = True) -> tuple[int, None]:
        """Stable-Baselines-style predict(). Uses cached _last_close because
        the obs vector contains z-score / SMA features rather than raw price.
        Inside learn()/evaluate() loops we have access to info; predict() is
        a fallback path for compare.py-style standalone use."""
        info = {"data_close": self._last_close, "position": self._last_position}
        return self.act(info), None

    def _make_env(self, render_mode: str | None = None):
        from ...env_config import make_env_factory

        factory = make_env_factory(self.env_name, render_mode=render_mode)
        return factory()

    def learn(
        self,
        total_timesteps: int = 50_000,
        log_fn=None,
        log_interval: int = 1000,
    ) -> dict[str, float]:
        """Roll out the MPC policy for total_timesteps env steps.

        Returns final eval metrics (mean reward over last 100 episodes).
        Note: MPC has no learnable parameters beyond the AR(1) coefficients
        (which adapt online from the price buffer), so "learn" here is a
        rollout, not a gradient-based training loop.
        """
        env = self._make_env()

        total_steps = 0
        episode = 0
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []

        print(f"\n{'=' * 60}")
        print("  LQ-MPC (Linear-Quadratic Model Predictive Control)")
        print(f"  Environment: {self.env_name}")
        print(f"  Components:  {self._component_names}")
        print(f"  Weights:     pnl={self._w_pnl:.2f}, txn={self._w_txn:.2f}, borrow={self._w_borrow:.2f}")
        print(f"  Horizon:     {self.horizon}")
        print(f"  History:     {self.history_window} (warm-up {self.warm_start_threshold})")
        print(f"  Timesteps:   {total_timesteps:,}")
        print(f"{'=' * 60}\n")

        while total_steps < total_timesteps:
            _, info = env.reset(seed=self.seed + episode)
            self._price_buffer.clear()
            self._last_close = float(info.get("data_close", 100.0))
            self._last_position = float(info.get("position", 0.0))

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
                    "mpc/ar_phi": float(self._ar_phi),
                    "mpc/ar_mu": float(self._ar_mu),
                }
                if log_fn is not None:
                    log_fn(metrics, total_steps)
                if episode % 50 == 0:
                    print(
                        f"  Step {total_steps:>7,} | Ep {episode:>4} | "
                        f"mean_r={metrics['rollout/ep_rew_mean']:+.3f} | "
                        f"phi={self._ar_phi:.3f} mu={self._ar_mu:.2f}"
                    )

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
        """Greedy rollout for n_episodes; returns mean/std episode return."""
        env = self._make_env()
        rewards = []
        for ep in range(n_episodes):
            _, info = env.reset(seed=self.seed + 10_000 + ep)
            self._price_buffer.clear()
            self._last_close = float(info.get("data_close", 100.0))
            self._last_position = float(info.get("position", 0.0))
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
        """Persist a small metadata blob (no learnable weights to save)."""
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        meta = {
            "agent": "LQMPCAgent",
            "env_name": self.env_name,
            "horizon": self.horizon,
            "history_window": self.history_window,
            "component_names": self._component_names,
            "component_weights": list(self._weights),
            "w_pnl": self._w_pnl,
            "w_txn": self._w_txn,
            "w_borrow": self._w_borrow,
            "ar_phi": self._ar_phi,
            "ar_mu": self._ar_mu,
        }
        with open(path_obj / "mpc_lq.json", "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: str) -> None:
        """No-op load (LQMPCAgent has no learned weights). Restores AR(1) seed
        if present so a saved/restored agent starts from the same forecast."""
        meta_path = Path(path) / "mpc_lq.json"
        if not meta_path.exists():
            return
        with open(meta_path) as f:
            meta = json.load(f)
        self._ar_phi = float(meta.get("ar_phi", self._ar_phi))
        self._ar_mu = float(meta.get("ar_mu", self._ar_mu))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LQ-MPC smoke test")
    parser.add_argument("--env", default="trading-k4")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from ...env_config import get_env_config

    cfg = get_env_config(args.env)
    agent = LQMPCAgent(
        args.env,
        reward_component_names=cfg.reward_components,
        component_weights=cfg.reward_component_weights,
        horizon=args.horizon,
        seed=args.seed,
    )

    env = agent._make_env()
    obs, info = env.reset(seed=args.seed)
    agent._price_buffer.clear()
    agent._last_close = float(info.get("data_close", 100.0))
    agent._last_position = 0.0

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
            agent._last_position = 0.0
            n_episodes += 1

    env.close()

    print(f"\nLQ-MPC smoke test on {args.env} ({args.steps} steps, H={args.horizon}):")
    print(f"  episodes:        {n_episodes + 1}")
    print(f"  total reward:    {total_reward:+.4f}")
    print(f"  per-step reward: {total_reward / max(n_steps, 1):+.5f}")
    print(f"  AR(1) state:     phi={agent._ar_phi:.4f} mu={agent._ar_mu:.2f}")
    print(f"  action mix:      short={actions_taken[0]}  flat={actions_taken[1]}  long={actions_taken[2]}")
    print("  component breakdown:")
    for name, total in component_totals.items():
        print(f"    {name:15s}: {total:+.4f}")
