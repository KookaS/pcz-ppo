# Environment Integration Guide

## Required Interface

Any Gymnasium-compatible environment can be used with PCZ-PPO algorithms, provided it exposes per-component rewards through the `info` dict.

### The Contract

The environment's `step()` method must return reward components in `info["reward_components"]`:

```python
def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)

    # Required: per-component breakdown
    info["reward_components"] = {
        "component_name_1": float_value_1,
        "component_name_2": float_value_2,
        # ...
    }

    # The scalar reward should equal the sum of components
    # (or approximately, allowing for floating-point)
    assert abs(reward - sum(info["reward_components"].values())) < 1e-6

    return obs, reward, terminated, truncated, info
```

### Rules

1. **Keys must match `reward_component_names`**: The algorithm constructor receives a list of component names. These must exactly match the keys in `info["reward_components"]`.

2. **Missing keys default to 0.0**: If a key from `reward_component_names` is not present in a given step's `info["reward_components"]`, it silently defaults to 0.0. This can mask integration bugs — always verify all components are being returned.

3. **Components should sum to the scalar reward**: The scalar `reward` returned by `step()` is what SB3 stores in the rollout buffer. PCZ variants overwrite this with normalized values before GAE, but the raw scalar is used for VecNormalize (if active), logging, and the `StandardPPO` baseline.

4. **Components should have different scales**: PCZ-PPO is most beneficial when components have different magnitudes (e.g., one in [-10, 0] and another in [0, 0.5]). If all components have similar scales, PCZ-PPO should perform similarly to aggregate z-norm (B4).

5. **VecEnv compatibility**: The `info` dict is per-environment in a `VecEnv`. SB3's vectorized environments return a list of info dicts. The rollout collection function iterates over all envs and extracts components from each.

## Built-in Environments

Built-in environments are configured in `core/env_config.py` via `ENV_REGISTRY`. Each entry specifies the env ID, reward components, policy type, obs normalization, and wrapper — so `train.py` and `compare.py` are fully env-agnostic. Use `--env=<name>` on the CLI. Run `uv run python -c "from core.env_config import ENV_REGISTRY; print(sorted(ENV_REGISTRY.keys()))"` for the current list.

| `--env` name | Gym ID | Components | Policy | Install |
|--------------|--------|------------|--------|---------|
| `cartpole` | `CartPole-v1` | `balance center` | MlpPolicy | `uv sync` |
| `bipedalwalker` | `BipedalWalker-v3` | `shaping energy crash` | MlpPolicy | `uv sync` |
| `halfcheetah` | `HalfCheetah-v4` | `run ctrl_cost` | MlpPolicy | `uv sync` (MuJoCo) |
| `humanoid` | `Humanoid-v4` | `forward alive ctrl_cost` | MlpPolicy | `uv sync` (MuJoCo) |
| `lunarlander` | `mo-lunar-lander-v3` | `landing shaping fuel_main fuel_side` | MlpPolicy | `uv sync` |
| `lunarlander-k2` | `LunarLander-v3` | `landing dense` | MlpPolicy | `uv sync` |
| `lunarlander-k8` | `LunarLander-v3` | `landing distance velocity angle leg_left leg_right fuel_main fuel_side` | MlpPolicy | `uv sync` |
| `resource` | `resource-gathering-v0` | `death_reward gold diamond` | MlpPolicy | `uv sync` |
| `supermario` | `SuperMarioBros-v0` | `x_pos time death coin score` | CnnPolicy | `uv sync --extra mario` |
| `trading-k2` | (synthetic) | `pnl costs` | MlpPolicy | `uv sync --extra trading` |
| `trading-k4` | (synthetic) | `pnl_gain pnl_loss txn_cost borrow_cost` | MlpPolicy | `uv sync --extra trading` |
| `trading-k6` | (synthetic) | `entry_pnl hold_pnl txn_cost borrow_cost spread_proxy residual` | MlpPolicy | `uv sync --extra trading` |
| `trading-k8` | (synthetic) | `entry_gain entry_loss hold_gain hold_loss txn_cost borrow_cost spread_proxy residual` | MlpPolicy | `uv sync --extra trading` |
| `trading-hk2` | (synthetic) | `pnl_step risk_events` | MlpPolicy | `uv sync --extra trading` |
| `trading-hk4` | (synthetic) | `pnl_step costs var_breach bankruptcy` | MlpPolicy | `uv sync --extra trading` |
| `trading-hk6` | (synthetic) | `pnl_gain pnl_loss txn_cost borrow_cost var_breach bankruptcy` | MlpPolicy | `uv sync --extra trading` |
| `trading-hk8` | (synthetic) | `entry_gain entry_loss hold_gain hold_loss txn_cost borrow_cost var_breach bankruptcy` | MlpPolicy | `uv sync --extra trading` |

### MO-LunarLander

Uses `mo-gymnasium`'s `mo-lunar-lander-v3`. The 4D reward vector is converted to our `info["reward_components"]` contract by `MultiComponentLunarLander` (in `_common.py`):

| Component | Source | Range | Character |
|-----------|--------|-------|-----------|
| `landing` | reward[0] | [-100, +100] | Sparse (terminal only) — +100 success, -100 crash |
| `shaping` | reward[1] | (-inf, +inf) | Dense, high variance — potential-based shaping |
| `fuel_main` | reward[2] | [-1, 0] | Dense, small scale — main engine fuel cost |
| `fuel_side` | reward[3] | [-1, 0] | Dense, small scale — side engine fuel cost |

This is an ideal testbed for PCZ: the shaping reward is orders of magnitude larger than fuel costs, so per-component normalization should significantly outperform aggregate normalization.

**Default component weights:** `landing=10.0, shaping=5.0, fuel_main=0.5, fuel_side=0.5`. These weights are applied *after* z-normalization to express that landing is the primary objective (20x more important than fuel efficiency). Without weights, z-normalization equalizes fuel penalties with landing/shaping, causing the agent to over-optimize for fuel efficiency at the expense of actually landing (entropy collapse to deterministic "never fire engines" policy).

To add a new environment, register an `EnvConfig` entry in `ENV_REGISTRY` and create a wrapper in `algorithms/_common.py` (or `env_config.py`) that produces `info["reward_components"]`.

## Example: Wrapping an Existing Environment

```python
import gymnasium as gym
import numpy as np

class MultiComponentWrapper(gym.Wrapper):
    """Wraps any environment to expose multi-component rewards."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Example: decompose a robotics reward
        # (these would come from your actual reward function)
        info["reward_components"] = {
            "forward_velocity": self._forward_reward,
            "healthy_bonus": self._healthy_reward,
            "control_cost": self._ctrl_cost,
            "contact_cost": self._contact_cost,
        }

        return obs, reward, terminated, truncated, info
```

## Example: Using with PCZ-PPO

```python
from stable_baselines3.common.env_util import make_vec_env
from core import PCZPPO

COMPONENTS = ["forward_velocity", "healthy_bonus", "control_cost", "contact_cost"]

env = make_vec_env(lambda: MultiComponentWrapper(gym.make("Humanoid-v4")), n_envs=4)

model = PCZPPO(
    "MlpPolicy", env,
    reward_component_names=COMPONENTS,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    verbose=1,
)
model.learn(total_timesteps=1_000_000)
```

## Choosing Components

Good reward decompositions have:

| Property | Example | Why it matters |
|----------|---------|---------------|
| **Different scales** | Health: [-10, 0], Bonus: [0, 0.5] | High-magnitude components dominate without normalization |
| **Different frequencies** | Dense per-step penalty vs sparse event bonus | Sparse components have noisy statistics at small buffer sizes |
| **Semantic distinctness** | Safety vs efficiency vs task completion | Each component should represent a genuinely different objective |

Poor decompositions (don't bother with PCZ-PPO):

- All components have similar scales → aggregate z-norm (B4) is sufficient
- Single scalar reward (no decomposition possible) → use standard PPO (B1)
- Components are just rescaled versions of the same signal → merge them

## MPC-Relevant Internals (`trading-kN` envs)

Audience: market-making / MPC baseline development. This section documents the
planning-time state that classical-control agents (LQ-MPC, NMPC) need from the
`trading-kN` envs, separate from the gym↔train contract above.

**Source code:** `core/envs/trading.py` (`MultiComponentTradingEnv`,
`HeteroScaleTradingEnv`); price generator in `make_ou_data()`.

### Price process

Mean-reverting Ornstein-Uhlenbeck (discrete-time):

    x_{t+1} - x_t = θ (μ - x_t) + σ ε_t,   ε ~ N(0, 1)

Parameters (hardcoded in `make_ou_data`):

- `μ = 100.0` (long-run mean)
- `θ = 0.05` (mean-reversion strength → half-life ≈ 14 steps)
- `σ = 3.0` (per-step Gaussian noise)
- Stationary std ≈ σ / √(2θ) ≈ 9.5; clipped to [10, 500]

Equivalent AR(1) form: `x_{t+1} = a + φ x_t + σ ε_t` with `φ = 1 - θ = 0.95`,
`a = θ μ = 5.0`. **MPC may fit (a, φ) by OLS on a rolling window of
`info["data_close"]`** — recovers the true generator asymptotically.

The OHLC data is pre-generated at env-init time (`n_steps = 5000` per worker)
with a deterministic per-worker `data_seed`. Within an episode (max 500 steps),
the price series is fully determined; only the random episode start position
varies per `reset()`.

### Action and position semantics

- Action space: `Discrete(3)` mapped to positions `[-1, 0, +1]` (short / flat / long).
- After `step(action)`, `info["position"]` holds the position now in force.
- After `step(action)`, `info["data_close"]` holds the close price for the
  current bar; `data_open / data_high / data_low / data_volume` also exposed.

### Fill model

`gym_trading_env` fills at `close` price (no slippage model, no queue
dynamics). Position changes apply at the bar boundary. **For MPC, this means
the fill model is implicit: planning over price *changes* `Δx̂_t` is exact;
no separate fill-rate-vs-aggressiveness curve is needed at this fidelity.**

### Cost model

- **Trading fees** (linear): `-trading_fees × |Δposition|` per step, where
  `trading_fees = 5e-4` (5 bps). Applied by base env, surfaces as
  `txn_cost` after the K-decomposition.
- **Borrow interest** (linear): `-borrow_interest_rate` per step when
  `position ∈ {-1}` or `position > 1` (default `1e-4` per step). For
  `K=4` this is folded into a `borrow_cost` *residual* that ALSO catches
  log-vs-arithmetic return discrepancy and fee approximation error
  (see `MultiComponentTradingEnv` docstring — explicit warning).
- **No slippage, no spread** in the realized PnL (the K=6/K=8 `spread_proxy`
  component is a *signal* derived from `(high - low) / close`, not an
  actual cost charged by the env).

### Reward scaling

All components are multiplied by `REWARD_SCALE = 100.0` so per-step rewards
are O(1) instead of O(0.01). Sum-invariant: `sum(reward_components) ==
scalar_reward` exactly for K=2/4/6/8 (K=3 is intentionally non-summing —
clean-decomposition ablation).

### How an LQ-MPC agent accesses this state

Per market-making backlog S-MKT-1.1, the agent uses **option (b) — fit AR(1)
from observations only, no env-interface change**. Concretely:

1. After each `env.step(action)`, read `info["data_close"]` and append to a
   rolling window (default 200 samples).
2. Once warm-up (≥30 samples), fit `x_{t+1} = a + b x_t` by OLS; clip
   `b ∈ [0, 0.999]` for stability; recover `μ = a / (1 - b)`.
3. Forecast `Δx̂_{t+k} = (μ - x_t)(b^{k+1} - b^k)` for `k = 0..H-1`.
4. Solve QP over continuous position trajectory `p ∈ [-1, 1]^H`; threshold
   `p_0*` to discrete action.

Reference implementation: `core/algorithms/baselines/mpc_lq.py` (`LQMPCAgent`).

### Known LQ approximations (relevant to kill-risk R-MKT-1)

- **Quadratic txn cost** `(p_t - p_{t-1})²` is the LQ stand-in for the env's
  **linear** `|Δp|` cost. The kill-risk pre-registration explicitly tests
  whether this approximation is too coarse; if R-MKT-1 fires, NMPC via
  `do-mpc` (with linear costs) is the documented pivot.
- **Symmetric PnL weighting**: `pnl_gain` and `pnl_loss` weights are averaged
  into a single `w_pnl`. Exact for default equal weights, approximate for
  asymmetric weights (none currently used).
- **Deterministic forecast**: noise term `σ ε_t` is dropped in planning. For
  certainty-equivalent LQ this is standard; chance-constrained / scenario-tree
  variants would re-introduce it.

## Verifying the Integration

Run this check before starting experiments:

```python
env = make_vec_env(lambda: YourWrappedEnv(), n_envs=1)
obs = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step([action])

    # Check components exist
    comps = info[0].get("reward_components", {})
    assert all(name in comps for name in COMPONENTS), f"Missing: {set(COMPONENTS) - set(comps.keys())}"

    # Check sum matches scalar
    comp_sum = sum(comps[k] for k in COMPONENTS)
    assert abs(reward[0] - comp_sum) < 1e-4, f"Mismatch: reward={reward[0]}, sum={comp_sum}"

print("Environment integration verified.")
```

## Visualization (MANDATORY)

**Every environment MUST have working visualization before running experiments.** This is verified during integration using X11 forwarding.

### Render Verification Checklist

For each new environment, verify all three render paths:

1. **Raw Gymnasium render** — opens a window via X11:
   ```bash
   uv run python -c "
   import gymnasium as gym
   env = gym.make('ENV_ID', render_mode='human')
   env.reset()
   for _ in range(100):
       obs, r, term, trunc, info = env.step(env.action_space.sample())
       if term or trunc: env.reset()
   env.close()
   print('Render: OK')
   "
   ```

2. **TorchRL single-env render** — via `make_single_env`:
   ```bash
   uv run python -c "
   from core.torchrl.env import make_single_env
   env = make_single_env('ENV_NAME', render_mode='human')
   td = env.reset()
   for _ in range(100):
       td['action'] = env.action_spec.rand()
       td = env.step(td)
       td = td['next'] if not td['next', 'done'].item() else env.reset()
   env.close()
   print('TorchRL render: OK')
   "
   ```

3. **Trained agent evaluation** — via decoupled renderer with a checkpoint:
   ```bash
   uv run python -m core.torchrl.render \
       --env=ENV_NAME \
       --checkpoint-dir=/workspace/runs/BEST_RUN/checkpoints
   ```

### Render Status by Environment

| Environment | `render_mode="human"` | Decoupled Renderer | Verified |
|-------------|:---------------------:|:------------------:|:--------:|
| `cartpole` | OK | OK | 2026-04-10 |
| `lunarlander` | OK | OK | 2026-04-10 |
| `halfcheetah` | OK (MuJoCo viewer) | OK | 2026-04-10 |
| `humanoid` | OK (MuJoCo viewer) | OK | 2026-04-10 |
| `resource` | OK | OK | 2026-04-10 |
| `supermario` | Requires X11 | Not tested | — |

### Minimum Training Durations

Short runs are useful for algorithm comparison but will NOT produce competent agents. Every algorithm+env must be validated at full duration at least once:

| Environment | Comparison Scale | Full Validation | Notes |
|-------------|-----------------|-----------------|-------|
| `cartpole` | 5k | 50k | Trivial — solves quickly |
| `lunarlander` | 200k-500k | 500k-1M | Discrete, 8-dim obs |
| `halfcheetah` | 500k | 2-3M | Continuous, needs limb coordination |
| `humanoid` | 500k | 2-5M | Continuous, 376-dim obs, hardest env |
| `resource` | 50k | 100k | Simple grid world |
| `supermario` | 500k | 1M+ | CNN policy, high dimensional |

A "competent agent" means it visually performs the task (walks, lands, gathers) — not just that training loss decreases.

### Prerequisites

- **X11 forwarding**: `DISPLAY` env var must be set (`:0` on host, forwarded in devcontainer)
- **MuJoCo**: Installed via `mujoco` package (included in base deps). MuJoCo viewer opens automatically for `halfcheetah`, `humanoid`.
- **Devcontainer**: `/tmp/.X11-unix` is bind-mounted from host, `DISPLAY` inherited

## Docker Training with Environments

When using the Docker training container, environment-specific dependencies are pre-installed:

- **CartPole**: Included in base image (part of `gymnasium`)
- **LunarLander**: Included in base image (`mo-gymnasium` + `gymnasium[box2d]` are core dependencies)
- **SuperMario**: Requires `[mario]` extra (`gym-super-mario-bros`, pins numpy<2.0). Requires X11 forwarding for rendering with `--render` — the container mounts `/tmp/.X11-unix` and inherits `DISPLAY` from the host.

To test environments inside the Docker container interactively:
```bash
docker compose -f docker-compose.train.yml run --rm --entrypoint bash train
# Inside container:
uv run python -c "from core.algorithms._common import MultiComponentCartPole; print('CartPole OK')"
```

New environment wrappers added to `_common.py` are automatically available in the container (source is bind-mounted).

## VecNormalize Usage

| Algorithm Category | `norm_reward` | `norm_obs` |
|-------------------|:---:|:---:|
| `StandardPPO` (B1) | `True` (standard practice) | `True` |
| All PCZ variants (A1-A5, D1-D3, S4) | **`False`** | `True` |
| All z-norm variants (B4, A5) | **`False`** | `True` |
| PPONoNorm (B2), PPOAdvOnly (B3) | `False` | `True` |
| PCZ-GRPO (C1) | **`False`** | `True` |
| PPOPopArt (C2) | **`False`** | `True` |
| PPOMultiHead (C5) | **`False`** | `True` |

**The rule**: If the algorithm handles its own reward normalization, set `norm_reward=False`. Only `StandardPPO` (B1) relies on external VecNormalize for reward normalization.

```python
from stable_baselines3.common.vec_env import VecNormalize

# For StandardPPO (B1):
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# For everything else:
env = VecNormalize(env, norm_obs=True, norm_reward=False)
# Or simply don't wrap with VecNormalize reward normalization
```
