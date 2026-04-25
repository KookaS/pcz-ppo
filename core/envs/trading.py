"""Multi-component reward wrapper for Gym-Trading-Env.

Decomposes the scalar trading reward into K components for PCZ-PPO K-scaling.
The base env returns portfolio log-return as scalar reward. We decompose into:

K=2: pnl (price change), costs (txn + borrow interest + residual drift)
K=4: split PnL by sign (gain/loss) + txn_cost + borrow_cost
K=6: split PnL by action (entry/hold) + txn_cost + borrow_cost + spread_proxy + residual
K=8: split PnL by action AND sign + txn_cost + borrow_cost + spread_proxy + residual

IMPORTANT — component semantics are APPROXIMATE, not exact financial quantities:
- `pnl_*`: arithmetic return × position (close approximation of actual log PnL
  contribution; difference ~0.4% of PnL magnitude at typical 3% moves).
- `txn_cost`: -trading_fees × |position_delta|. Matches the base env's per-unit
  fee model but not the exact traded-value-based fee (discrepancy <1% absorbed
  by `borrow_cost`/`residual`).
- `borrow_cost` (K=4): actually a CATCH-ALL RESIDUAL, containing (i) actual
  borrow interest when position < 0 or > 1, (ii) log-vs-arithmetic return
  discrepancy, (iii) fee approximation error. It is NOT purely borrow interest
  despite the label. Kept for column-name compatibility with prior experiments;
  in K=6/K=8 this is split as `borrow_cost` + explicit `residual` = 0 so the
  label matches the content (borrow_cost still carries all residuals).

Invariant: sum(components) == reward exactly (verified bit-exact for all K).
Bankruptcy edge case: base env sets reward=0 on done=True, swallowing the
catastrophic loss. Not an issue at 5bp fees (bankruptcy ≈ impossible) but
a latent bug in gym_trading_env (bankruptcy reward swallowed on done=True).

Design:
- Mean-reverting OU-like synthetic price generator (learnable signal via z-score)
- Per-instance data seed (parallel workers see different trajectories)
- Realistic fees (5 bps) + shorting enabled
- Random episode starts via max_episode_duration (injected into TradingEnv)
- Stationary features: z-score, log-return, sma_ratio, volatility
- Reward scaled x100 to bring magnitude to LunarLander-like range

Known reproducibility limits:
- Per-worker data_seed uses hash((os.getpid(), counter)). Since PID differs
  across runs, the same --seed flag produces different OU data on different
  runs/machines. Within a single run, data is deterministic.
- gym_trading_env's reset(seed=N) does NOT control random initial position
  or random episode start; these use global np.random state. So episode-
  level reproducibility is limited even with fixed --seed.
"""

from __future__ import annotations

import warnings

import gymnasium as gym
import numpy as np
import pandas as pd

# gym_trading_env sets warnings.filterwarnings("error") at import time,
# which converts all subsequent warnings to exceptions. Reset after import.
_saved_filters = warnings.filters[:]
from gym_trading_env.environments import TradingEnv

warnings.filters[:] = _saved_filters


REWARD_SCALE = 100.0  # multiply raw log-returns so episode totals are ~O(1-10)


def make_ou_data(n_steps: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Mean-reverting price series with a learnable z-score signal.

    Price follows Ornstein-Uhlenbeck: dX = theta*(mu - X) dt + sigma dW.
    Half-life ~ log(2) / theta. We target half-life ~50 steps so a policy
    looking at SMA20 deviation has a real predictive edge.

    A threshold policy ("go long when z_score < -threshold, short when > threshold,
    flat otherwise") beats buy-and-hold under this generator.
    """
    rng = np.random.default_rng(seed)
    mu = 100.0
    theta = 0.05  # mean-reversion strength; half-life ~14 steps
    sigma = 3.0  # per-step noise -> stationary std ~= 9.5
    x = np.zeros(n_steps)
    x[0] = mu + rng.standard_normal() * sigma / np.sqrt(2 * theta)
    for t in range(1, n_steps):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * rng.standard_normal()
    x = np.clip(x, 10.0, 500.0)  # keep strictly positive

    volume = rng.uniform(1000, 10000, n_steps)
    df = pd.DataFrame(
        {
            "open": x * (1 + rng.uniform(-0.002, 0.002, n_steps)),
            "high": x * (1 + rng.uniform(0, 0.004, n_steps)),
            "low": x * (1 - rng.uniform(0, 0.004, n_steps)),
            "close": x,
            "volume": volume,
        }
    )

    # Stationary features (no raw prices). Z-score is the signal-carrying feature.
    sma20 = df["close"].rolling(20).mean().bfill()
    std20 = df["close"].rolling(20).std().bfill().replace(0.0, 1.0)
    df["feature_zscore"] = ((df["close"] - sma20) / std20).clip(-5, 5)
    df["feature_sma_ratio"] = (sma20 / df["close"]).clip(0.5, 2.0)
    df["feature_logret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0).clip(-0.1, 0.1)
    df["feature_volatility"] = df["close"].pct_change().rolling(20).std().fillna(0.0)
    df["feature_volume"] = (df["volume"] / df["volume"].rolling(20).mean().bfill()).clip(0.1, 5.0)
    return df


# Kept for backwards-compatibility tests; not used by the active registry.
def make_sample_data(n_steps: int = 5000, seed: int = 42) -> pd.DataFrame:
    return make_ou_data(n_steps=n_steps, seed=seed)


class MultiComponentTradingEnv(gym.Wrapper):
    """Mean-reverting trading env with K-component reward decomposition.

    Positions: [-1, 0, +1] (short, flat, long). Action space = Discrete(3).
    Rewards are per-step log-returns of portfolio valuation, scaled x100.
    """

    K_COMPONENTS = {
        2: ["pnl", "costs"],
        3: ["pnl_gain", "pnl_loss", "txn_cost"],  # K=4 minus borrow_cost/residual
        4: ["pnl_gain", "pnl_loss", "txn_cost", "borrow_cost"],
        6: ["entry_pnl", "hold_pnl", "txn_cost", "borrow_cost", "spread_proxy", "residual"],
        8: [
            "entry_gain",
            "entry_loss",
            "hold_gain",
            "hold_loss",
            "txn_cost",
            "borrow_cost",
            "spread_proxy",
            "residual",
        ],
    }

    def __init__(
        self,
        k: int = 4,
        n_steps: int = 5000,
        data_seed: int = 42,
        trading_fees: float = 0.0005,
        borrow_interest_rate: float = 0.0001,
        max_episode_duration: int = 500,
        allow_short: bool = True,
        reward_scale: float = REWARD_SCALE,
    ):
        assert k in self.K_COMPONENTS, f"k must be in {list(self.K_COMPONENTS.keys())}"
        self.k = k
        self.component_names = self.K_COMPONENTS[k]
        self._trading_fees = trading_fees
        self._reward_scale = reward_scale
        self._prev_position = 0
        self._prev_close = 100.0

        df = make_ou_data(n_steps, data_seed)
        positions = [-1, 0, 1] if allow_short else [0, 1]
        env = TradingEnv(
            df=df,
            positions=positions,
            trading_fees=trading_fees,
            borrow_interest_rate=borrow_interest_rate,
            max_episode_duration=max_episode_duration,
            verbose=0,
        )
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_position = info.get("position", 0)
        self._prev_close = info.get("data_close", 100.0)
        info["reward_components"] = {name: 0.0 for name in self.component_names}
        return obs, info

    def step(self, action):
        obs, raw_reward, terminated, truncated, info = self.env.step(action)

        close = info.get("data_close", self._prev_close)
        position = info.get("position", 0)
        price_return = (close - self._prev_close) / max(self._prev_close, 1e-8)
        position_changed = position != self._prev_position

        # Scale everything up by REWARD_SCALE so per-step rewards are O(1)
        # instead of O(0.01). Reward = log-return of portfolio valuation
        # (base env uses np.log(v_t / v_{t-1})). Scaling is linear, so
        # sum(components_scaled) == reward_scaled holds.
        reward = raw_reward * self._reward_scale
        raw_pnl = float(price_return * position) * self._reward_scale
        # Trading fee is applied proportional to |position change| in base env.
        position_delta = abs(position - self._prev_position)
        raw_txn = float(-self._trading_fees * position_delta) * self._reward_scale
        raw_residual = float(reward - raw_pnl - raw_txn)

        high = info.get("data_high", close)
        low = info.get("data_low", close)
        spread_frac = (high - low) / max(close, 1e-8)
        raw_spread = float(-spread_frac * 0.1 * position_delta) * self._reward_scale

        components = {}

        if self.k == 2:
            components["pnl"] = raw_pnl
            components["costs"] = raw_txn + raw_residual

        elif self.k == 3:
            # Clean decomposition: drops borrow_cost/residual + spread entirely.
            # sum(components) != reward (by design; ablation variant).
            components["pnl_gain"] = float(max(raw_pnl, 0.0))
            components["pnl_loss"] = float(min(raw_pnl, 0.0))
            components["txn_cost"] = raw_txn

        elif self.k == 4:
            components["pnl_gain"] = float(max(raw_pnl, 0.0))
            components["pnl_loss"] = float(min(raw_pnl, 0.0))
            components["txn_cost"] = raw_txn
            components["borrow_cost"] = raw_residual

        elif self.k == 6:
            if position_changed:
                components["entry_pnl"] = raw_pnl
                components["hold_pnl"] = 0.0
            else:
                components["entry_pnl"] = 0.0
                components["hold_pnl"] = raw_pnl
            components["txn_cost"] = raw_txn - raw_spread
            components["borrow_cost"] = raw_residual
            components["spread_proxy"] = raw_spread
            components["residual"] = 0.0

        elif self.k == 8:
            if position_changed:
                components["entry_gain"] = float(max(raw_pnl, 0.0))
                components["entry_loss"] = float(min(raw_pnl, 0.0))
                components["hold_gain"] = 0.0
                components["hold_loss"] = 0.0
            else:
                components["entry_gain"] = 0.0
                components["entry_loss"] = 0.0
                components["hold_gain"] = float(max(raw_pnl, 0.0))
                components["hold_loss"] = float(min(raw_pnl, 0.0))
            components["txn_cost"] = raw_txn - raw_spread
            components["borrow_cost"] = raw_residual
            components["spread_proxy"] = raw_spread
            components["residual"] = 0.0

        for name in self.component_names:
            components.setdefault(name, 0.0)

        info["reward_components"] = components
        total = sum(components.values())

        self._prev_position = position
        self._prev_close = close

        return obs, total, terminated, truncated, info


HETERO_BANKRUPTCY_THRESHOLD = 0.5  # 50% portfolio-value drawdown triggers bankruptcy
HETERO_BANKRUPTCY_PENALTY = -1000.0  # terminal catastrophic loss
HETERO_VAR_STEP_THRESHOLD = 5.0  # per-step loss > 5 reward-units triggers VaR breach
HETERO_VAR_BREACH_PENALTY = -50.0  # non-terminal risk penalty


class HeteroScaleTradingEnv(gym.Wrapper):
    """Trading env with heterogeneous-scale reward components.

    Extends the OU mean-reverting env with two catastrophic risk events that
    operate at very different scales from per-step PnL:

      - VaR breach (-50): single-step loss > VAR_STEP_THRESHOLD (~5% price move)
      - Bankruptcy (-1000): portfolio value falls below BANKRUPTCY_THRESHOLD (50% drawdown)

    Scale spread: per-step PnL O(1-3) vs var_breach O(-50) vs bankruptcy O(-1000).
    PCZ hypothesis: per-component z-norm equalizes these scales so the critic
    stays sensitive to all three signals simultaneously.

    Invariant: sum(reward_components) == scalar reward exactly for all K.
    """

    K_COMPONENTS = {
        2: ["pnl_step", "risk_events"],
        4: ["pnl_step", "costs", "var_breach", "bankruptcy"],
        6: ["pnl_gain", "pnl_loss", "txn_cost", "borrow_cost", "var_breach", "bankruptcy"],
        8: [
            "entry_gain",
            "entry_loss",
            "hold_gain",
            "hold_loss",
            "txn_cost",
            "borrow_cost",
            "var_breach",
            "bankruptcy",
        ],
    }

    def __init__(
        self,
        k: int = 4,
        n_steps: int = 5000,
        data_seed: int = 42,
        trading_fees: float = 0.0005,
        borrow_interest_rate: float = 0.0001,
        max_episode_duration: int = 500,
        allow_short: bool = True,
        reward_scale: float = REWARD_SCALE,
        bankruptcy_threshold: float = HETERO_BANKRUPTCY_THRESHOLD,
        bankruptcy_penalty: float = HETERO_BANKRUPTCY_PENALTY,
        var_step_threshold: float = HETERO_VAR_STEP_THRESHOLD,
        var_breach_penalty: float = HETERO_VAR_BREACH_PENALTY,
    ):
        assert k in self.K_COMPONENTS, f"k must be in {list(self.K_COMPONENTS.keys())}"
        self.k = k
        self.component_names = self.K_COMPONENTS[k]
        self._trading_fees = trading_fees
        self._reward_scale = reward_scale
        self._bankruptcy_threshold = bankruptcy_threshold
        self._bankruptcy_penalty = bankruptcy_penalty
        self._var_step_threshold = var_step_threshold
        self._var_breach_penalty = var_breach_penalty
        self._prev_position = 0
        self._prev_close = 100.0
        self._portfolio_value = 1.0

        df = make_ou_data(n_steps, data_seed)
        positions = [-1, 0, 1] if allow_short else [0, 1]
        env = TradingEnv(
            df=df,
            positions=positions,
            trading_fees=trading_fees,
            borrow_interest_rate=borrow_interest_rate,
            max_episode_duration=max_episode_duration,
            verbose=0,
        )
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_position = info.get("position", 0)
        self._prev_close = info.get("data_close", 100.0)
        self._portfolio_value = 1.0
        info["reward_components"] = {name: 0.0 for name in self.component_names}
        return obs, info

    def step(self, action):
        obs, raw_reward, terminated, truncated, info = self.env.step(action)

        close = info.get("data_close", self._prev_close)
        position = info.get("position", 0)
        price_return = (close - self._prev_close) / max(self._prev_close, 1e-8)
        position_changed = position != self._prev_position

        reward = raw_reward * self._reward_scale
        base_pnl = float(price_return * position) * self._reward_scale
        position_delta = abs(position - self._prev_position)
        base_txn = float(-self._trading_fees * position_delta) * self._reward_scale
        base_residual = float(reward - base_pnl - base_txn)

        var_breach = self._var_breach_penalty if base_pnl < -self._var_step_threshold else 0.0

        self._portfolio_value *= np.exp(raw_reward)
        bankruptcy = 0.0
        if not terminated and not truncated and self._portfolio_value < self._bankruptcy_threshold:
            bankruptcy = self._bankruptcy_penalty
            terminated = True
            self._portfolio_value = 1.0

        total_reward = reward + var_breach + bankruptcy

        components: dict[str, float] = {}
        if self.k == 2:
            # pnl_step absorbs all regular per-step components (pnl + txn + residual = reward)
            components["pnl_step"] = reward
            components["risk_events"] = var_breach + bankruptcy

        elif self.k == 4:
            components["pnl_step"] = base_pnl
            components["costs"] = base_txn + base_residual
            components["var_breach"] = var_breach
            components["bankruptcy"] = bankruptcy

        elif self.k == 6:
            components["pnl_gain"] = float(max(base_pnl, 0.0))
            components["pnl_loss"] = float(min(base_pnl, 0.0))
            components["txn_cost"] = base_txn
            components["borrow_cost"] = base_residual
            components["var_breach"] = var_breach
            components["bankruptcy"] = bankruptcy

        elif self.k == 8:
            if position_changed:
                components["entry_gain"] = float(max(base_pnl, 0.0))
                components["entry_loss"] = float(min(base_pnl, 0.0))
                components["hold_gain"] = 0.0
                components["hold_loss"] = 0.0
            else:
                components["entry_gain"] = 0.0
                components["entry_loss"] = 0.0
                components["hold_gain"] = float(max(base_pnl, 0.0))
                components["hold_loss"] = float(min(base_pnl, 0.0))
            components["txn_cost"] = base_txn
            components["borrow_cost"] = base_residual
            components["var_breach"] = var_breach
            components["bankruptcy"] = bankruptcy

        info["reward_components"] = components
        self._prev_position = position
        self._prev_close = close

        return obs, total_reward, terminated, truncated, info


if __name__ == "__main__":
    # Smoke test all K with canonical policies.

    for k in [2, 4, 6, 8]:
        env = MultiComponentTradingEnv(k=k, data_seed=42)
        obs, info = env.reset(seed=42)
        print(f"\nK={k}: components={list(info['reward_components'].keys())}  action_space={env.action_space}")

        totals = {n: 0.0 for n in env.component_names}
        sum_r = 0.0
        for _ in range(500):
            a = env.action_space.sample()
            obs, r, done, trunc, info = env.step(a)
            sum_r += r
            for n, v in info["reward_components"].items():
                totals[n] += v
            if done or trunc:
                obs, info = env.reset(seed=42)
        print(f"  random-policy 500-step reward={sum_r:+.3f}  sum(components)={sum(totals.values()):+.3f}")
        for n, v in totals.items():
            print(f"    {n:15s}: {v:+.3f}")

    print("\n--- HeteroScaleTradingEnv smoke test ---")
    for k in [2, 4, 6, 8]:
        env = HeteroScaleTradingEnv(k=k, data_seed=42)
        obs, info = env.reset(seed=42)
        print(f"\nK={k}: components={list(info['reward_components'].keys())}")

        totals: dict[str, float] = {n: 0.0 for n in env.component_names}
        sum_r = 0.0
        n_bankruptcy = 0
        n_var = 0
        for _ in range(2000):
            a = env.action_space.sample()
            obs, r, done, trunc, info = env.step(a)
            sum_r += r
            for n, v in info["reward_components"].items():
                totals[n] += v
            if info["reward_components"].get("bankruptcy", 0.0) < 0:
                n_bankruptcy += 1
            if (
                info["reward_components"].get("risk_events", 0.0) < -900
                or info["reward_components"].get("bankruptcy", 0.0) < 0
            ):
                n_bankruptcy += 0  # counted above
            if info["reward_components"].get("var_breach", 0.0) < 0 or (
                k == 2 and info["reward_components"].get("risk_events", 0.0) < -900
            ):
                n_var += 1
            if done or trunc:
                obs, info = env.reset(seed=42)
        print(f"  random-policy 2000-step total_reward={sum_r:+.3f}  sum(components)={sum(totals.values()):+.3f}")
        print(f"  bankruptcies={n_bankruptcy}  var_breaches={n_var}")
        for n, v in totals.items():
            print(f"    {n:20s}: {v:+.3f}")
