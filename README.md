# PCZ-PPO: Per-Component Z-Normalization for PPO

When a reinforcement learning environment decomposes its reward into K heterogeneous components (landing bonus, fuel penalty, control cost, …), standard PPO normalizes the *aggregate* and lets the highest-magnitude component dominate the learning signal. **PCZ-PPO** applies z-normalization independently to each component *before* GAE, equalizing scales while preserving per-component priority through explicit weights.

**Paper result:** At 500k steps, PCZ-PPO's mean performance is approximately invariant to K while standard PPO degrades progressively. The growing PCZ/PPO gap is driven by PPO collapsing under reward heterogeneity, not by PCZ improving. All three K∈{4,6,8} comparisons survive Holm–Bonferroni correction at α=0.05.

**What this repo gives you:**
- A drop-in TorchRL PPO variant (`torchrl-pcz-ppo-running`) — per-component z-normalization as a ~60-line preprocessing wrapper on the reward buffer, no architectural changes.
- A systematic K-scaling sweep (K∈{2,4,6,8}) with pre-registered seeds, Holm–Bonferroni correction, BCa bootstrap CIs, IQM, and stochastic-dominance tests.
- A 20-variant ablation suite (ZCA, symlog→z-norm, asymmetric clip, GRPO, multi-head critic, PopArt, …).
- A **data-driven paper pipeline**: every number in the PDF is backed by a fragment in `paper/generated/` rendered from `data/results.csv` — hand-typed numbers are forbidden and caught by pre-commit.

## Install

```bash
git clone https://github.com/KookaS/pcz-ppo
cd pcz-ppo
uv sync --extra torchrl   # TorchRL + LunarLander/BipedalWalker/CartPole
uv sync --extra trading   # + financial trading environments
```

Requires Python ≥ 3.11 and [uv](https://github.com/astral-sh/uv).

## Quickstart

```bash
# Train PCZ-PPO on LunarLander K=4 (~10 min on CPU)
uv run python -m core.train \
    --algorithm torchrl-pcz-ppo-running \
    --env lunarlander \
    --total-timesteps 500000

# Compare PCZ-PPO vs PPO across 5 seeds
uv run python -m core.compare \
    --algorithms torchrl-pcz-ppo-running torchrl-ppo \
    --env lunarlander \
    --seeds 42 43 44 45 46 \
    --total-timesteps 500000

# K=6 with canonical weights
uv run python -m core.train \
    --algorithm torchrl-pcz-ppo-running \
    --env lunarlander-k6 \
    --reward-component-weights 10.0,3.0,1.0,1.0,0.5,0.5 \
    --total-timesteps 500000
```

## Supported Environments

| `--env` | K | Components |
|---------|---|------------|
| `lunarlander` | 4 | landing, shaping, fuel_main, fuel_side |
| `lunarlander-k2` | 2 | landing, dense aggregate |
| `lunarlander-k6` | 6 | intermediate split |
| `lunarlander-k8` | 8 | fine-grained shaping split |
| `bipedalwalker` | 3 | shaping, energy, crash |
| `cartpole` | 2 | balance, center |
| `halfcheetah` | 2 | run, ctrl_cost |
| `trading-k4` | 4 | pnl_gain, pnl_loss, txn_cost, borrow_cost |

All environments return `info["reward_components"]` as `{str: float}`. See `docs/environment-api.md` for the full table and component semantics.

## Reproduce the Paper

The committed `data/results.csv` and `data/metrics/*.parquet` files are sufficient to rebuild all figures without a live MLflow instance.

```bash
# Rebuild stale figures and all generated/*.tex fragments
uv run python paper/paper_build.py --build

# Also recompile the PDF (requires latexmk)
uv run python paper/paper_build.py --build --pdf

# Run paper tests (formatter, fragment registry, headline-number regression)
uv run pytest paper/tests/

# Check that committed fragments match current data (CI gate)
uv run python paper/render_claims.py --check
```

If you have an MLflow instance with your own runs, export first:

```bash
uv run python -m core.plot.export_results \
    --tracking-uri http://127.0.0.1:5050 \
    --output data/results.csv --append

uv run python -m core.plot.export_metrics \
    --tracking-uri http://127.0.0.1:5050 \
    --output-dir data/metrics --append
```

See `docs/REPRODUCE.md` for full step-by-step instructions including expected wall-clock times.

## Key Results (from paper)

At 500k steps on LunarLander (Holm–Bonferroni corrected, all three K survive at α=0.05):

| K | n | PCZ-PPO | PPO | Δ | Welch p (Holm) | Var ratio |
|---|---|---------|-----|---|-----------------|-----------|
| 4 | 15 | +159.8±34.6 | +123.5±40.6 | +36.3 | 0.042 (0.042) | 1.38× |
| 6 | 15 | +136.7±34.8 | +82.1±47.2  | +54.6 | 0.0008 (0.003)| 1.84× |
| 8 | 10 | +143.8±45.4 | +18.0±163.4 | +125.8 | 0.049 (0.049) | 12.9× |

At 4M steps on LunarLander K=8 with heterogeneous weights (10,1,1,1,1,1,0.5,0.5), n=10: PCZ-PPO +179.0±35.7 vs PPO +124.6±59.9 (Δ=+54.4, p=0.026, Cohen d=+1.10).

**Sharp usage constraint:** At K=8 with equal weights and 4M steps, PCZ-PPO collapses while PPO remains robust. Only with hand-designed, strictly-positive heterogeneous weights does PCZ-PPO win at high K / long horizon. See §Limitations in the paper.

## Citation

```bibtex
@software{charrez2026pczppo,
  title   = {{PCZ-PPO}: Per-Component Z-Normalization for {PPO} with Heterogeneous Reward Components},
  author  = {Charrez, Olivier and Bussler, Maarten and Weber, Tobias},
  year    = {2026},
  url     = {https://github.com/KookaS/pcz-ppo},
  license = {Apache-2.0}
}
```

## License

Code: [Apache-2.0](LICENSE)
Paper text and figures (`paper/`): [CC-BY-4.0](paper/LICENSE)
