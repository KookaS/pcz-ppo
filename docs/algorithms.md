# Algorithm Reference

All SB3 algorithms (frozen), organized by category (Q-learning is registered but not documented below — it's a tabular method, not a PPO variant). Each entry documents: experiment ID, registry name, class, file, what it does, and how it differs from the others.

---

## Baselines (B1–B4)

### B1: Standard PPO
- **Registry**: `ppo` | **Class**: `StandardPPO` | **File**: `ppo.py`
- **What**: Standard SB3 PPO. No custom reward normalization. The scalar reward from `env.step()` goes directly to GAE.
- **Reward norm**: None internal. Expects external `VecNormalize(norm_reward=True)`.
- **Advantage norm**: SB3 default (`normalize_advantage=True`).
- **Use case**: Primary baseline. Pair with `VecNormalize` for fair comparison.
- **Note**: Accepts `reward_component_names` for API consistency but only logs them; does not use them for normalization.

### B2: PPO No Normalization
- **Registry**: `ppo-no-norm` | **Class**: `PPONoNorm` | **File**: `ppo_no_norm.py`
- **What**: PPO with ALL normalization disabled. Raw rewards, raw advantages.
- **Reward norm**: None.
- **Advantage norm**: `normalize_advantage=False`.
- **Use case**: Lower bound. Answers: "does any normalization help at all?"

### B3: PPO Advantage-Only
- **Registry**: `ppo-adv-only` | **Class**: `PPOAdvOnly` | **File**: `ppo_adv_only.py`
- **What**: PPO with advantage whitening but no reward normalization.
- **Reward norm**: None.
- **Advantage norm**: `normalize_advantage=True`.
- **Use case**: Isolates the effect of advantage whitening from reward normalization. This is the default SB3 PPO behavior when no VecNormalize is used.

### B4: PPO Aggregate Z-Norm
- **Registry**: `ppo-znorm` | **Class**: `PPOZnorm` | **File**: `ppo_znorm.py`
- **What**: Z-normalizes the aggregate scalar reward per-env across timesteps before GAE.
- **Reward norm**: `buf.rewards = _znorm(buf.rewards, axis=0)` — operates on the scalar from `env.step()`.
- **Advantage norm**: SB3 default (True).
- **Bootstrap**: Strips bootstrap before z-norm, re-adds after.
- **Use case**: Tests whether the benefit of PCZ-PPO comes from z-normalization itself or from the per-component decomposition. If A1 > B4, the decomposition matters.

---

## PCZ-PPO Ablations (A1–A5)

### A1: PCZ-PPO (Core Method)
- **Registry**: `pcz-ppo` | **Class**: `PCZPPO` | **File**: `pcz_ppo.py`
- **What**: Per-component z-normalization per-env across timesteps, then sum (equal weights), then standard GAE.
- **Normalization**: For each component k: `normalized[:,:,k] = _znorm(component_rewards[:,:,k], axis=0)`. Then `rewards = normalized.sum(axis=2)`.
- **Advantage norm**: SB3 default (True) — this is the critical "post-aggregation advantage whitening" stabilizer.
- **Key insight**: Each component is independently standardized to zero mean and unit variance before summing. This prevents high-magnitude components from dominating.
- **Use case**: The proposed method. Compare against B1, B4, A6.

### A2: PCZ-PPO Global
- **Registry**: `pcz-ppo-global` | **Class**: `PCZPPOGlobal` | **File**: `pcz_ppo_global.py`
- **What**: Same as A1 but z-normalizes globally (all envs + timesteps) rather than per-env.
- **Normalization**: `_znorm(component_rewards[:,:,k], axis=None)` — single mean/std across the entire `(T, E)` buffer.
- **Use case**: Tests whether per-env statistics matter. Per-env gives each environment its own normalization baseline; global pools all environments.

### A3: PCZ-PPO Running (Co-Primary)
- **Registry**: `pcz-ppo-running` | **Class**: `PCZPPORunning` | **File**: `pcz_ppo_running.py`
- **What**: Per-component normalization using running mean/std that persist across rollouts (Welford's algorithm).
- **Normalization**: `(component_rewards - running_mean) / sqrt(running_var)`. Running stats updated each rollout via `buf._update_running_stats()`.
- **Use case**: Co-primary variant. Running stats are smoother than batch stats, reducing critic non-stationarity. Better for small buffers. Trade-off: slower adaptation to changing reward distributions.
- **Stats persistence**: `_running_mean`, `_running_var`, `_running_count` are initialized in `__init__()`, NOT in `reset()` — they survive across rollouts.

### A4: PCZ-PPO Weighted
- **Registry**: `pcz-ppo-weighted` | **Class**: `PCZPPOWeighted` | **File**: `pcz_ppo_weighted.py`
- **What**: Per-component z-norm with tunable weights applied after normalization.
- **Normalization**: `_znorm(comp[:,:,k], axis=0)`, then `_weighted_component_sum(normalized, weights)`.
- **Weights**: Default `[1.0, 1.0, ...]` (equal). Pass `component_weights=[w1, w2, ...]` or set defaults in `EnvConfig.reward_component_weights`. CLI: `--reward-component-weights 5.0,3.0,0.5,0.5`.
- **Note**: All 17 algorithms now accept `component_weights`. `pcz-ppo-weighted` is kept for backward compatibility but is functionally identical to `pcz-ppo` when both use the same weights.

### A5: PPO Z-Norm Post (Sum Then Normalize)
- **Registry**: `ppo-znorm-post` | **Class**: `PPOZnormPost` | **File**: `ppo_znorm_post.py`
- **What**: Sums raw component rewards into a scalar, THEN z-normalizes the scalar.
- **Normalization**: `raw_sum = component_rewards.sum(axis=2)` → `_znorm(raw_sum, axis=0)`.
- **Key difference from A1**: A1 normalizes BEFORE summing, A5 normalizes AFTER. This is the "aggregate-then-normalize" approach that GDPO's "normalize-then-aggregate" is designed to improve upon. If A1 > A5, the order of operations matters.

---

## Alternative Normalizations (C1–C5)

### C1: PCZ-GRPO (Critic-Free)
- **Registry**: `grpo-pcz` | **Class**: `PCZGRPO` | **File**: `pcz_grpo.py`
- **What**: Critic-free GRPO with per-component z-normalization on Monte-Carlo returns.
- **Key differences from A1**:
  - `vf_coef=0` — critic exists in the SB3 architecture but contributes zero to the loss.
  - No GAE — computes discounted MC returns per component, then z-normalizes.
  - No timeout bootstrapping (`bootstrap_timeout=False`).
  - Final batch whitening after component summation.
- **Advantage norm**: `normalize_advantage=False` — this class applies its own batch whitening after component summation, so SB3's per-minibatch normalization is disabled to avoid redundancy.
- **Use case**: Tests whether the critic adds value over critic-free GRPO. If A1 > C1, the critic's temporal credit assignment helps.

### C2: PPO + PopArt
- **Registry**: `ppo-popart` | **Class**: `PPOPopArt` | **File**: `ppo_popart.py`
- **What**: PPO with PopArt adaptive value head rescaling (no per-component normalization).
- **Mechanism**: After GAE, updates running mean/std of returns, rescales the value head's last linear layer to preserve outputs, then normalizes returns for the value loss.
- **Use case**: Tests value-side normalization (PopArt) vs reward-side normalization (PCZ-PPO).
- **Constraint**: `clip_range_vf` must be `None` (raises `ValueError` otherwise) — PopArt rescaling makes old/new values incomparable for clipping.

### C4: PCZ-PPO VecNormalize-Style
- **Registry**: `pcz-ppo-vecnorm` | **Class**: `PCZPPOVecnorm` | **File**: `pcz_ppo_vecnorm.py`
- **What**: Per-component running std normalization WITHOUT mean subtraction (VecNormalize-style).
- **Normalization**: `component_rewards / sqrt(running_var)` — divides by running std only, no mean shift.
- **Key difference from A3**: A3 subtracts running mean (full z-norm). C4 only divides by std (VecNormalize-style, preserves sign/offset).
- **Use case**: Isolates whether mean-centering is necessary or whether variance scaling alone suffices.

### C5: PPO Multi-Head Critic
- **Registry**: `ppo-multihead` | **Class**: `PPOMultiHead` | **File**: `ppo_multihead.py`
- **What**: PPO with K separate value heads sharing hidden layers. V(s) = sum_k V^(k)(s).
- **Architecture**: Shared `mlp_extractor.forward_critic()` → K separate `nn.Linear(latent_dim_vf, 1)` output heads.
- **GAE**: Uses the aggregate V(s) from multi-head sum.
- **Critic loss**: Per-component MC returns supervise each head separately.
- **No reward normalization**: Raw component rewards are summed for the aggregate reward.
- **Use case**: Tests whether decomposing the critic is better than decomposing the reward normalization. If A1 ≈ C5, both approaches handle scale differences; PCZ-PPO is simpler (no architecture changes).

---

## Reward Transform Variants (D1–D3)

### D1: PCZ-PPO Clip
- **Registry**: `pcz-ppo-clip` | **Class**: `PCZPPOClip` | **File**: `pcz_ppo_clip.py`
- **What**: Clips each component to `[-1, 1]` before summing.
- **Use case**: Crude bounding. Does simple clipping suffice without full z-normalization?

### D2: PCZ-PPO MinMax
- **Registry**: `pcz-ppo-minmax` | **Class**: `PCZPPOMinmax` | **File**: `pcz_ppo_minmax.py`
- **What**: Min-max normalizes each component to `[0, 1]` per-env across timesteps.
- **Normalization**: `(comp - min) / (max - min)`. Zero-range fallback: 0.5.
- **Use case**: Tests whether distribution shape (uniform vs Gaussian) matters.

### D3: PCZ-PPO Log
- **Registry**: `pcz-ppo-log` | **Class**: `PCZPPOLog` | **File**: `pcz_ppo_log.py`
- **What**: Log-compresses each component: `sign(r) * log(1 + |r|)`.
- **Use case**: Compresses large values without requiring buffer statistics. No mean/std estimation needed.

---

## Hybrids (S4)

### S4: PCZ-PPO + PopArt
- **Registry**: `pcz-ppo-popart` | **Class**: `PCZPPOPopArt` | **File**: `pcz_ppo_popart.py`
- **What**: Combines per-component z-normalization (PCZ-PPO) with PopArt value head rescaling.
- **Pipeline**: Per-component z-norm → sum → GAE → PopArt rescale + returns normalization.
- **Use case**: Tests whether reward-side (GDPO) and value-side (PopArt) normalization are complementary.
- **Constraint**: `clip_range_vf` must be `None` (raises `ValueError` otherwise).

---

## Quick Comparison Table

| ID | Registry Name | Reward Norm | Adv Norm | Critic | Key Question |
|----|--------------|-------------|----------|--------|-------------|
| B1 | `ppo` | VecNormalize (external) | Yes | Yes | Standard baseline |
| B2 | `ppo-no-norm` | None | No | Yes | Does normalization matter? |
| B3 | `ppo-adv-only` | None | Yes | Yes | Is reward norm needed? |
| B4 | `ppo-znorm` | Aggregate z-norm | Yes | Yes | Is benefit from decomposition or just z-norm? |
| A1 | `pcz-ppo` | Per-comp z-norm | Yes | Yes | **Proposed method** |
| A2 | `pcz-ppo-global` | Per-comp global z-norm | Yes | Yes | Per-env vs global stats? |
| A3 | `pcz-ppo-running` | Per-comp running z-norm | Yes | Yes | Batch vs running stats? |
| A4 | `pcz-ppo-weighted` | Per-comp z-norm + weights | Yes | Yes | Do weights matter after z-norm? |
| A5 | `ppo-znorm-post` | Sum then z-norm | Yes | Yes | Does order of operations matter? |
| C1 | `grpo-pcz` | Per-comp z-norm (MC) | Yes | No | Does the critic add value? |
| C2 | `ppo-popart` | PopArt (value-side) | Yes | Yes | Value-side vs reward-side norm? |
| C4 | `pcz-ppo-vecnorm` | Per-comp running std | Yes | Yes | Is mean-centering needed? |
| C5 | `ppo-multihead` | None (multi-head critic) | Yes | Yes (K heads) | Decompose critic vs decompose rewards? |
| D1 | `pcz-ppo-clip` | Clip [-1, 1] | Yes | Yes | Does crude bounding suffice? |
| D2 | `pcz-ppo-minmax` | MinMax [0, 1] | Yes | Yes | Does distribution shape matter? |
| D3 | `pcz-ppo-log` | Log compression | Yes | Yes | Statistic-free alternative? |
| S4 | `pcz-ppo-popart` | Per-comp z-norm + PopArt | Yes | Yes | Are reward-side and value-side complementary? |
