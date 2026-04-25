# Architecture and Data Flow

## Overview

All 17 algorithms share a common architecture built on Stable Baselines 3's PPO. Each algorithm is a PPO subclass that overrides `collect_rollouts()` to inject its normalization strategy between rollout collection and GAE advantage computation.

```
┌──────────────────────────────────────────────────────┐
│                    Training Loop                      │
│  (SB3 PPO.learn → calls collect_rollouts repeatedly) │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│            collect_rollouts() [_common.py]            │
│                                                      │
│  For each timestep in the rollout buffer:            │
│    1. policy(obs) → actions, values, log_probs       │
│    2. env.step(actions) → obs, rewards, dones, infos │
│    3. Extract info["reward_components"] → comp_rewards│
│    4. Store in ComponentRolloutBuffer                │
│    5. Handle timeout bootstrap (gamma * V(terminal)) │
│                                                      │
│  Returns: (True, last_values, dones)                 │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│       Algorithm-Specific Normalization                │
│       (each .py file overrides this section)          │
│                                                      │
│  Read:  buf.component_rewards  (T, E, K)             │
│  Write: buf.rewards            (T, E)                │
│                                                      │
│  Examples:                                           │
│    pcz_ppo.py:  per-component per-env z-norm → sum  │
│    ppo_znorm.py: z-norm the aggregate scalar         │
│    pcz_grpo.py: MC returns + per-comp z-norm        │
│                                                      │
│  Then: buf._reapply_bootstrap()                      │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│   buf.compute_returns_and_advantage(last_values, d)  │
│   (SB3 standard GAE computation on buf.rewards)      │
│                                                      │
│   Writes: buf.returns, buf.advantages                │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│              PPO.train()                              │
│                                                      │
│  For each epoch, for each minibatch:                 │
│    - Policy loss (clipped surrogate)                 │
│    - Value loss (MSE on returns)                     │
│    - Advantage whitening (if normalize_advantage)    │
│    - Entropy bonus                                   │
│    - Backprop + gradient clipping                    │
└──────────────────────────────────────────────────────┘
```

## Key Classes

### `ComponentRolloutBuffer` (`_common.py:48`)

Extends SB3's `RolloutBuffer` with:

| Attribute | Shape | Purpose |
|-----------|-------|---------|
| `component_rewards` | `(T, E, K)` | Per-component rewards per timestep per env |
| `timeout_bootstrap` | `(T, E)` | Bootstrap adjustments for truncated episodes |
| `_running_mean` | `(K,)` | Running mean per component (for A3, C4) |
| `_running_var` | `(K,)` | Running variance per component |
| `_running_count` | scalar | Sample count for Welford's algorithm |

Where `T` = buffer_size (n_steps), `E` = n_envs, `K` = n_reward_components.

**Lifecycle:**
1. `reset()` — zeros `component_rewards`, `timeout_bootstrap`, resets `_component_pos`. Running stats persist across resets.
2. `add_component_rewards(comp)` — stores `(E, K)` array at current position, advances counter.
3. `_update_running_stats()` — Welford parallel merge of buffer batch into running stats.
4. `_reapply_bootstrap()` — adds `timeout_bootstrap` back to `rewards` after overwriting.

### `collect_rollouts()` (`_common.py:132`)

Shared rollout collection function used by all algorithms. Key behavior:

- Steps the vectorized env and extracts `info["reward_components"]` at each step.
- Missing component keys default to `0.0`.
- Timeout bootstrap: when a truncated episode ends (`TimeLimit.truncated=True`), adds `gamma * V(terminal_obs)` to the reward and records the adjustment separately.
- Returns `(True, last_values, dones)` or `False` if callback cancels.

### `_znorm()` (`_common.py:35`)

```python
def _znorm(arr, axis=None, eps=1e-8, min_std=None):
    threshold = min_std if min_std is not None else eps
    mean = arr.mean(axis=axis, keepdims=True)
    std = arr.std(axis=axis, keepdims=True)
    safe_std = np.where(std > threshold, std, 1.0)
    return np.where(std > threshold, (arr - mean) / safe_std, arr - mean)
```

- `axis=0`: per-env normalization (across timesteps, separately for each env column). Used by A1, A4, D2.
- `axis=None`: global normalization (all envs + timesteps flattened). Used by A2.
- `min_std`: optional minimum standard deviation threshold for sparse components. When set, components with `std < min_std` are mean-centred only (no variance scaling). Use for ultra-sparse signals (bonuses firing <5% of steps). If `None`, falls back to `eps=1e-8`.
- When `std < threshold`: falls back to mean-centering only (no variance scaling). Prevents division by near-zero for constant or ultra-sparse components.

### Runtime Safety Checks (`_common.py`)

**VecNormalize warning**: `collect_rollouts()` checks on first call whether the env is wrapped with `VecNormalize(norm_reward=True)` and logs a WARNING if so. This prevents silent double-normalization.

**Missing component warning**: If `info["reward_components"]` is empty/missing or a specific component key is absent, a WARNING is logged on first occurrence. The value defaults to 0.0 for robustness.

### `PopArtMixin` (`_common.py:238`)

Shared mixin for adaptive value head rescaling (used by `ppo_popart.py` and `pcz_ppo_popart.py`):

1. `_update_popart_stats(returns)` — Welford update of running mean/var from batch returns.
2. `_rescale_value_head(old_mean, old_var)` — Rescales the value head's last linear layer weights and bias to preserve output when stats change:
   ```
   w_new = w_old * old_std / new_std
   b_new = (b_old * old_std + old_mean - new_mean) / new_std
   ```
3. `_apply_popart_to_returns(buf)` — Combined: update stats, rescale head, normalize returns.

Accesses `self.policy.value_net` which in SB3's `MlpPolicy` is `nn.Linear(latent_dim_vf, 1)`.

## Bootstrap Handling

Timeout bootstrapping is critical for environments with time limits. The pattern across all algorithms:

1. **During collection** (`_common.py:199-213`): When an episode truncates, `gamma * V(terminal_obs)` is added to `rewards[idx]` AND recorded in `timeout_bootstrap[step, idx]`.
2. **After normalization**: The algorithm overwrites `buf.rewards` with normalized values, then calls `buf._reapply_bootstrap()` to add the bootstrap adjustment back.
3. **Why this works**: The bootstrap value is in value-function scale and should not be re-normalized. Stripping it before normalization and re-adding after preserves its original magnitude.

**Exception**: `pcz_grpo.py` sets `bootstrap_timeout=False` because it has no critic (vf_coef=0), so there is no value function to bootstrap from.

**Exception**: `ppo_znorm.py` (B4) explicitly strips bootstrap before z-norm (`buf.rewards -= buf.timeout_bootstrap`) and re-adds after, because it normalizes the stored `buf.rewards` in-place rather than overwriting from `component_rewards`.

## Docker Training Infrastructure

### Container Architecture

```
Host / Devcontainer
  │
  ├── docker compose -f docker-compose.mlflow.yml up -d
  │     ├── mlflow-postgres  (Postgres :5432)
  │     ├── mlflow-minio     (MinIO :9000/:9001)
  │     ├── mlflow-create-bucket (init container)
  │     └── mlflow-server    (MLflow :5050)
  │
  └── docker compose -f docker-compose.train.yml run --rm train <cmd>
        └── rl-train container
              ├── Image: Dockerfile.train (Python 3.12-slim + uv + deps)
              ├── Network: mlflow-network (shared with MLflow stack)
              ├── Entrypoint: uv run python -m <module>
              └── Bind mounts:
                    ./core         → /app/core         (ro)  — source code
                    ./pyproject.toml → /app/pyproject.toml (ro)
                    ./uv.lock      → /app/uv.lock      (ro)
                    ./runs         → /app/runs         (rw)  — model checkpoints
                    ./data         → /app/data         (rw)  — MLflow local DB
                    /tmp/.X11-unix → /tmp/.X11-unix    (ro)  — X11 display
```

### Image Layers (Dockerfile.train)

1. **Base**: `python:3.12-slim` — minimal Python runtime
2. **System deps**: build-essential, git, X11/SDL2 client libs, OpenGL, ffmpeg
3. **User**: non-root `trainer` (UID 1000)
4. **uv**: installed via official installer
5. **Python deps**: `uv sync --extra mario` bakes all dependencies into the image
6. Source code is NOT copied — it's bind-mounted at runtime for live editing

### Networking

The training container joins `mlflow-network` (declared as `external: true`), which is created by `docker-compose.mlflow.yml`. This allows the training container to reach MLflow by container name:

- `MLFLOW_TRACKING_URI=http://mlflow-server:5050` (set automatically)
- MinIO: `http://mlflow-minio:9000` (for artifact storage)
- Postgres: `mlflow-postgres:5432` (MLflow backend store)

The MLflow stack must be running before the training container starts (the network must exist).

**Devcontainer direct training**: When running `uv run python -m core.train` directly inside the devcontainer (not via Docker training container), the devcontainer must be manually connected to `mlflow-network` (`docker network connect mlflow-network <devcontainer-name>`). Additionally, use the MLflow container's **IP address** (not DNS name) as the tracking URI — MLflow's DNS rebinding protection rejects requests with Docker DNS hostnames from outside its trusted network. Get the IP with `docker inspect mlflow-server --format '{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}'`.

### Host Path Translation

When Docker is invoked from inside the devcontainer, bind-mount paths must be host-absolute (the Docker daemon runs on the host, not inside the devcontainer). `HOST_PROJECT_DIR` is **auto-set** by `devcontainer.json` via `containerEnv: {"HOST_PROJECT_DIR": "${localWorkspaceFolder}"}`:

```
Devcontainer path:  /workspace/core
Host path:          ${HOST_PROJECT_DIR}/core  (auto-detected)
Docker mount:       ${HOST_PROJECT_DIR}/core:/app/core
```

### Compose Overlays

| File | What it adds |
|------|-------------|
| `docker-compose.train.yml` | Base: volumes, env vars, network (CPU mode) |
| `docker-compose.train.gpu.yml` | `deploy.resources.reservations.devices` for NVIDIA GPU |

### Environment Variables Injected

| Variable | Value | Source |
|----------|-------|--------|
| `MLFLOW_TRACKING_URI` | `http://mlflow-server:5050` | `docker-compose.train.yml` |
| `DISPLAY` | `${DISPLAY:-:0}` | Inherited from host (X11) |
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU overlay |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility,graphics` | GPU overlay |
| `PYTHONUNBUFFERED` | `1` | `docker-compose.train.yml` |

## Hardware & Training Performance

**Development machine:** HP Omen 25L — Intel i9-11900 (8C/16T, 2.5GHz), 16GB RAM (14GB available), NVIDIA RTX 3060 12GB VRAM, 1TB disk.

**Throughput benchmarks** (TorchRL, LunarLander, n_envs=4):

| Device | FPS | Notes |
|--------|-----|-------|
| **CPU** | ~1070 | Optimal for gymnasium-based envs |
| **GPU** | ~475 | 2x slower — CPU→GPU transfer overhead dominates |

**Rule:** Always use CPU for gymnasium envs (LunarLander, CartPole, BipedalWalker, Resource Gathering). GPU only for GPU-native envs (Isaac Lab, Brax) or when env step time exceeds transfer overhead.

**Memory per run:** ~900MB-2.5GB (LunarLander). Safe to run 2 concurrent runs, max 3 with monitoring.

**Wall-clock estimates** (LunarLander, n_envs=4, CPU): 500k steps ≈ 8 min, 1M steps ≈ 16 min.


## Resource Monitoring

The `/monitor` skill provides system-level observability for training workloads. Use it to identify whether training is CPU-bound, memory-bound, I/O-bound, or GPU-underutilized, and to find optimal `n_envs` / batch size / model size settings.

| Category | Tools | Use Case |
|----------|-------|----------|
| CPU | `lscpu`, `mpstat`, `pidstat`, `sar`, `vmstat` | Core saturation, per-process CPU |
| Memory | `free`, `pidstat -r`, `sar -r` | RSS growth, swap pressure |
| Disk I/O | `iostat`, `iotop`, `pidstat -d` | Data loading bottlenecks |
| GPU | `nvidia-smi`, `nvtop` | GPU utilization, memory, thermals |
| Python | `py-spy`, `memray`, `scalene` | Profiling hot loops, memory leaks |
| Tracing | `strace`, `ltrace`, `valgrind` | Syscall overhead, library calls |

Invoke via `/monitor <context>` (e.g., `/monitor watch CPU and memory during PCZ training`).

## PPO-MultiHead Special Architecture

`ppo_multihead.py` has a unique architecture not shared by other variants:

```
obs → features_extractor → mlp_extractor.forward_critic() → latent_vf
                                                              ├─→ head_1 → V^(1)(s)
                                                              ├─→ head_2 → V^(2)(s)
                                                              └─→ head_K → V^(K)(s)
                                                                  sum → V(s)
```

- Component heads are `nn.Linear(latent_dim_vf, 1)` — connected to the shared value MLP hidden layer output, NOT to raw features.
- Aggregate `V(s) = sum_k V^(k)(s)` is used for GAE advantage estimation.
- Per-component returns are computed via MC discounted returns and used to supervise each head separately.
- The standard SB3 single value head (`policy.value_net`) still exists but is not the primary value prediction — the multi-head aggregate replaces it for GAE.
