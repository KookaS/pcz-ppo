# TorchRL — Primary RL Backend

## Overview

TorchRL is the **primary development framework** for PCZ-PPO. All new algorithms, environments, and experiments use TorchRL exclusively. The SB3 backend is frozen as a reference implementation.

TorchRL algorithm variants are registered in `ALGORITHM_REGISTRY` with a `torchrl-` prefix, working with `train.py`, `compare.py`, and MLflow. Run `uv run python -c "from core import ALGORITHM_REGISTRY; print(len([k for k,v in ALGORITHM_REGISTRY.items() if getattr(v, 'is_torchrl', False)]))"` for the current count.

Key TorchRL features:
- **Parallel environments** via `ParallelEnv` (subprocess-based)
- **Decoupled rendering** via a separate subprocess that hot-reloads checkpoints
- **GRPO advantage estimation** with trajectory-level group normalization
- **Native PyTorch pipeline** — tensors stay on device throughout (relevant for GPU)

## Architecture

```
core/
├── torchrl/                    # Standalone TorchRL infrastructure
│   ├── config.py               # TorchRLConfig dataclass
│   ├── env.py                  # RewardVecWrapper, ParallelEnv builder
│   ├── models.py               # 3-layer MLP, ProbabilisticActor, ValueOperator
│   ├── training.py             # Training loop, GRPO advantage, GAE
│   ├── checkpoint.py           # Save/load for hot-reload rendering
│   └── render.py               # Decoupled renderer (subprocess)
│
└── algorithms/torchrl/         # Algorithm classes (registered in ALGORITHM_REGISTRY)
    ├── _base.py                # TorchRLAlgorithm base class
    ├── _norm.py                # Shared normalization helpers
    ├── baselines/              # Baseline variants (torchrl-ppo, torchrl-grpo, torchrl-qlearning, ...)
    └── pcz/                    # PCZ variants (torchrl-pcz-ppo, torchrl-pcz-grpo, ...)
```

### How it differs from SB3

| Aspect | SB3 | TorchRL |
|--------|-----|---------|
| **Backend** | numpy + PyTorch (mixed) | Pure PyTorch (TensorDict) |
| **Env parallelism** | `DummyVecEnv` (in-process) | `ParallelEnv` (subprocess per env) |
| **Reward components** | `info["reward_components"]` dict | `obs["reward_vec"]` tensor in observation space |
| **Customization point** | Override `collect_rollouts()` | Override `_compute_advantages()` |
| **Rendering** | In-process (blocks training) | Decoupled subprocess (non-blocking) |
| **Value function** | SB3 `MlpPolicy` (2-layer ReLU) | Custom 3-layer MLP (LeakyReLU) |

## Performance: SB3 vs TorchRL

### Benchmark Results (CPU, devcontainer, no GPU)

**CartPole — 50k steps, pcz-ppo normalization:**

| n_envs | SB3 (fps) | TorchRL (fps) | SB3 advantage |
|--------|-----------|---------------|---------------|
| 1 | 847 | 523 | 1.6x faster |
| 2 | 1,074 | 407 | 2.6x faster |
| 4 | 979 | 528 | 1.9x faster |

**LunarLander — 50k steps, 4 envs, pcz-ppo normalization:**

| Framework | FPS | Wall time |
|-----------|-----|-----------|
| SB3 | 1,358 | 37s |
| TorchRL | 536 | 93s |
| **SB3 advantage** | **2.5x faster** | |

### Why SB3 is faster on CPU

1. **DummyVecEnv is in-process.** SB3's vectorized environments run in the same process — no subprocess spawn, no IPC serialization, no pickle overhead. For cheap environments like CartPole (~0.05ms per step), this matters enormously.

2. **numpy is faster than TensorDict for small data.** SB3's rollout buffer uses numpy arrays. TorchRL wraps everything in `TensorDict`, which adds allocation and dispatch overhead that dominates when the actual computation (env step, forward pass) is trivial.

3. **ParallelEnv subprocess overhead.** TorchRL's `ParallelEnv` creates one subprocess per environment. Each step requires: serialize action → send via pipe → deserialize in subprocess → step env → serialize obs/reward → send back → deserialize. For a 0.05ms CartPole step, this overhead (typically 1-5ms per round-trip) is 20-100x the actual computation.

4. **Startup cost.** TorchRL's collector, ParallelEnv, and TensorDict initialization take several seconds. SB3's setup is near-instant.

### GPU Speedup Analysis

**tl;dr: GPU doesn't help for our current environments.** Training is environment-bound, not compute-bound.

#### Profiled component costs (CPU)

| Component | Cost | Notes |
|-----------|------|-------|
| CartPole `env.step()` | 0.007 ms | Extremely cheap |
| LunarLander `env.step()` | 0.030 ms | Very cheap |
| SuperMario `env.step()` | ~5 ms | Expensive (rendering, NES emulator) |
| MLP forward (batch=4096) | 1.0 ms | 5,000 params — tiny network |
| MLP backward (batch=4096) | 2.4 ms | |
| CNN forward (batch=64, 84x84) | 10.4 ms | 1.7M params — NatureCNN |
| CNN backward (batch=64) | 28.9 ms | |

#### Where time is spent per PPO iteration

Using n_steps=2048, n_envs=4, n_epochs=10, batch_size=64:

| Environment | Env stepping | Compute (fwd+bwd) | Bottleneck |
|-------------|-------------|-------------------|------------|
| CartPole + MLP | 14 ms (29%) | 36 ms (71%) | Compute |
| LunarLander + MLP | 61 ms (63%) | 36 ms (37%) | **Env (CPU-bound)** |
| SuperMario + CNN | 10,240 ms (96%) | 403 ms (4%) | **Env (CPU-bound)** |

#### Estimated GPU speedup

Assuming conservative GPU acceleration (3x for small MLP, 20x for CNN):

| Environment | CPU total | GPU total | Speedup | Verdict |
|-------------|-----------|-----------|---------|---------|
| CartPole + MLP | 50 ms | 26 ms | **1.9x** | Marginal — network is tiny |
| LunarLander + MLP | 97 ms | 74 ms | **1.3x** | Negligible — env-bound |
| SuperMario + CNN | 10,643 ms | 10,260 ms | **1.0x** | No gain — 96% in env.step() |

**The fundamental limit: gymnasium environments run on CPU.** No matter how fast the GPU processes the network, the training loop waits for `env.step()` on the CPU. For CartPole/LunarLander, the env is so fast that MLP compute actually matters — but the MLP is so small (5K params) that GPU overhead (kernel launch, memory transfer) would eat any theoretical gain.

#### When GPU would actually help

1. **GPU-native environments** (Isaac Gym, Brax, EnvPool): env steps run on GPU, eliminating the CPU bottleneck entirely. Can achieve 100-1000x speedup. Not applicable to our gymnasium-based envs.

2. **Large networks** (transformers, large CNNs, >1M params): GPU throughput advantage becomes meaningful when forward/backward pass takes >10ms. Our MLP has 5K params — too small.

3. **Very large batch sizes** (>64K samples): GPU parallelism shines with large batches. Our typical batch is 8K samples — GPU is underutilized.

4. **Multi-GPU data parallelism**: Distributing PPO updates across GPUs. Only useful at scale with expensive compute per iteration.

### Practical recommendations for faster training

Instead of adding a GPU to the current setup, these approaches will actually improve throughput:

**Tier 1: Drop-in optimizations (3-20x, days of effort)**

| Approach | Expected gain | How |
|----------|--------------|-----|
| **More CPU envs (SB3)** | 1.5-2x | `--n-envs=8` with SB3's DummyVecEnv |
| **SubprocVecEnv for expensive envs** | 2-4x | For SuperMario: true process parallelism |
| **Larger n_steps** | 10-20% | Less rollout/update switching overhead |
| **Fewer n_epochs** | Proportional | 5 epochs vs 10 = ~40% faster per iteration |
| **EnvPool** | 3-20x | C++ env backend replacing gymnasium vectorization |

**Tier 2: GPU-native environments (100-1000x, weeks of effort)**

| Approach | Expected gain | Applicable envs |
|----------|--------------|----------------|
| **Gymnax (JAX)** | 100-1000x | CartPole, LunarLander (already exist in Gymnax) |
| **MuJoCo Playground (MJX)** | 100-1000x | CartPole, LunarLander (physics-based) |
| **CuLE (CUDA)** | 40-190M frames/hr | Atari only (no NES/SuperMario) |

These move the environment to the GPU, eliminating the CPU-GPU transfer bottleneck entirely.

**Tier 3: End-to-end JAX pipeline (1000-4000x, months of effort)**

| Approach | Expected gain | Trade-off |
|----------|--------------|-----------|
| **PureJaxRL + Gymnax** | 4000x | Requires full rewrite from PyTorch to JAX |

Compiles the entire training loop (env + policy + loss + optimizer) into a single GPU kernel. 2048 PPO agents train in half the time of 1 PyTorch agent.

For the full research report with all benchmarks and citations, see `~/Documents/GPU_RL_Environments_Research_20260319/`.

### Summary: When to use which

| Scenario | Recommended | Why |
|----------|-------------|-----|
| CartPole / LunarLander | **SB3 on CPU** | 2-3x faster, env-bound anyway |
| SuperMario (CNN) | **SB3 on CPU** | 96% env-bound, GPU won't help |
| Quick experiments | **SB3** | Faster iteration, mature ecosystem |
| GPU-native envs (Isaac Gym) | **TorchRL on GPU** | Entire pipeline stays on device |
| Large networks (>1M params) | **TorchRL on GPU** | Compute-bound, GPU throughput wins |
| Maximum throughput research | **PureJaxRL + Gymnax** | 1000-4000x, full JAX rewrite |
| Decoupled rendering | **TorchRL** | Non-blocking checkpoint hot-reload |
| GRPO (critic-free) | **TorchRL** | Native implementation |
| Research / custom losses | **TorchRL** | PyTorch-native, easier to modify |

## Algorithm Reference

TorchRL algorithms, grouped by normalization strategy:

### Baselines (no reward normalization)

| Registry Name | Description |
|---------------|-------------|
| `torchrl-ppo` | Vanilla PPO with GAE |
| `torchrl-ppo-no-norm` | No normalization, no advantage whitening |
| `torchrl-ppo-adv-only` | Advantage whitening only |

### Scalar reward normalization

| Registry Name | Description |
|---------------|-------------|
| `torchrl-ppo-znorm` | Z-normalize scalar reward before GAE |
| `torchrl-ppo-znorm-post` | Weighted sum of components, then z-normalize |
| `torchrl-ppo-weighted-running` | Weighted sum of components, then running EMA z-norm (ablation: tests if decomposition adds value beyond smoothing) |

### Per-component normalization (PCZ variants)

| Registry Name | Description |
|---------------|-------------|
| `torchrl-pcz-ppo` | Per-component z-norm (batch stats) |
| `torchrl-pcz-ppo-global` | Per-component z-norm (global stats across all envs) |
| `torchrl-pcz-ppo-running` | Per-component running mean/std (EMA) |
| `torchrl-pcz-ppo-weighted` | Same as pcz-ppo (explicit weight naming) |
| `torchrl-pcz-ppo-vecnorm` | Running std normalization (no mean subtraction) |
| `torchrl-pcz-ppo-clip` | Clip components to [-1, 1] |
| `torchrl-pcz-ppo-minmax` | Min-max scale to [0, 1] |
| `torchrl-pcz-ppo-log` | Log compression: sign(x) * log(1+\|x\|) |

### Critic-free (GRPO)

| Registry Name | Description |
|---------------|-------------|
| `torchrl-grpo` | GRPO: trajectory group normalization, no critic |
| `torchrl-pcz-grpo` | Per-component z-norm + GRPO |

### Complex variants

| Registry Name | Description |
|---------------|-------------|
| `torchrl-pcz-ppo-popart` | GDPO z-norm + PopArt value rescaling |
| `torchrl-ppo-popart` | PopArt value rescaling (no reward norm) |
| `torchrl-ppo-multihead` | Multi-head value function (one head per component) |

### Tabular

| Registry Name | Description |
|---------------|-------------|
| `torchrl-qlearning` | Tabular Q-Learning (TorchRL-tagged, no neural network) |

## Component Gating

For environments with near-constant reward components (e.g. Humanoid's alive=5.0), z-normalization creates noise. Use `--component-gating` to automatically detect and exclude these components:

```bash
uv run python -m core.train --algorithm=torchrl-pcz-ppo-running --env=humanoid \
    --component-gating --total-timesteps=500000
```

- `--component-gating`: Zero-out components with running variance below the floor
- `--variance-floor`: Minimum variance threshold (default: auto-set to 0.01 when gating enabled)

Gating is opt-in and environment-specific. Unnecessary for LunarLander (all components have genuine variance).

## Decoupled Rendering

Training and rendering run as separate processes. The training process saves checkpoints periodically; the renderer hot-reloads the latest checkpoint every 30 seconds.

```bash
# Training with decoupled rendering
uv run python -m core.train --algorithm=torchrl-pcz-ppo --env=lunarlander --render

# Or standalone (two terminals):
# Terminal 1: Train
uv run python -m core.torchrl --algo=ppo --env=lunarlander

# Terminal 2: Render (connects to running training via checkpoint)
uv run python -m core.torchrl.render --env=lunarlander --checkpoint-dir=checkpoints/torchrl
```

This keeps rendering overhead completely out of the training loop, maximizing training throughput.

## Usage Examples

### Via train.py (recommended)

```bash
# Single TorchRL algorithm
uv run python -m core.train --algorithm=torchrl-pcz-ppo --env=lunarlander \
    --total-timesteps=100000 --n-envs=4

# Compare SB3 vs TorchRL
uv run python -m core.compare \
    --algorithms pcz-ppo torchrl-pcz-ppo \
    --total-timesteps=100000 --n-envs=4

# With MLflow
uv run python -m core.train --algorithm=torchrl-pcz-ppo --env=lunarlander \
    --mlflow-tracking-uri http://127.0.0.1:5050
```

### Python API

```python
from core.algorithms.torchrl.pcz_ppo import TorchRLPCZPPO

model = TorchRLPCZPPO(
    "lunarlander",
    reward_component_names=["landing", "shaping", "fuel_main", "fuel_side"],
    component_weights=[5.0, 3.0, 0.5, 0.5],
    num_envs=8,
)
model.learn(total_frames=500_000, render=True)
```

### Standalone

```bash
uv run python -m core.torchrl --algo=ppo --env=lunarlander --total-frames=500000
uv run python -m core.torchrl --algo=grpo --env=lunarlander --render
```
