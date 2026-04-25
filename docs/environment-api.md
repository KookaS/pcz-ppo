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
| `halfcheetah` | `HalfCheetah-v5` | `run ctrl_cost` | MlpPolicy | `uv sync` (MuJoCo) |
| `humanoid` | `Humanoid-v5` | `forward alive ctrl_cost` | MlpPolicy | `uv sync` (MuJoCo) |
| `lunarlander` | `mo-lunar-lander-v3` | `landing shaping fuel_main fuel_side` | MlpPolicy | `uv sync` |
| `lunarlander-k2` | `LunarLander-v3` | `landing dense` | MlpPolicy | `uv sync` |
| `lunarlander-k8` | `LunarLander-v3` | `landing distance velocity angle leg_left leg_right fuel_main fuel_side` | MlpPolicy | `uv sync` |
| `resource` | `resource-gathering-v0` | `death_reward gold diamond` | MlpPolicy | `uv sync` |
| `supermario` | `SuperMarioBros-v0` | `x_pos time death coin score` | CnnPolicy | `uv sync --extra mario` |
| `trading-k2` | (synthetic) | `pnl costs` | MlpPolicy | `uv sync --extra trading` |
| `trading-k4` | (synthetic) | `pnl_gain pnl_loss txn_cost borrow_cost` | MlpPolicy | `uv sync --extra trading` |
| `trading-k6` | (synthetic) | `entry_pnl hold_pnl txn_cost borrow_cost spread_proxy residual` | MlpPolicy | `uv sync --extra trading` |
| `trading-k8` | (synthetic) | `entry_gain entry_loss hold_gain hold_loss txn_cost borrow_cost spread_proxy residual` | MlpPolicy | `uv sync --extra trading` |

### MO-LunarLander

Uses `mo-gymnasium`'s `mo-lunar-lander-v3`. The 4D reward vector is converted to our `info["reward_components"]` contract by `MultiComponentLunarLander` (in `_common.py`):

| Component | Source | Range | Character |
|-----------|--------|-------|-----------|
| `landing` | reward[0] | [-100, +100] | Sparse (terminal only) — +100 success, -100 crash |
| `shaping` | reward[1] | (-inf, +inf) | Dense, high variance — potential-based shaping |
| `fuel_main` | reward[2] | [-1, 0] | Dense, small scale — main engine fuel cost |
| `fuel_side` | reward[3] | [-1, 0] | Dense, small scale — side engine fuel cost |

This is an ideal testbed for PCZ: the shaping reward is orders of magnitude larger than fuel costs, so per-component normalization should significantly outperform aggregate normalization.

**Default component weights:** `landing=5.0, shaping=3.0, fuel_main=0.5, fuel_side=0.5`. These weights are applied *after* z-normalization to express that landing is the primary objective (10x more important than fuel efficiency). Without weights, z-normalization equalizes fuel penalties with landing/shaping, causing the agent to over-optimize for fuel efficiency at the expense of actually landing (entropy collapse to deterministic "never fire engines" policy).

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
