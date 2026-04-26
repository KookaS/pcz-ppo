# Reproducing the PCZ-PPO Paper

Step-by-step guide from a clean clone to a compiled PDF.

## Prerequisites

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) (`pip install uv` or `curl -Lsf https://astral.sh/uv/install.sh | sh`)
- LaTeX with latexmk (for PDF compilation): `apt-get install texlive-full latexmk` or equivalent
- ~4 GB disk space (dependencies + data)

Optional (for running new experiments):
- Docker + Docker Compose (MLflow stack)
- 16 GB RAM (for parallel training runs)

## Step 1: Clone and Install

```bash
git clone https://github.com/KookaS/pcz-ppo
cd pcz-ppo
uv sync --extra torchrl
```

Wall-clock: ~2–5 min depending on network.

## Step 2: Verify the Data Pipeline

The repository includes pre-committed experiment data:
- `artifacts/pcz-ppo/data/results.csv` — 1 row per run, final metrics + hyperparameters
- `artifacts/pcz-ppo/data/metrics/*.parquet` — full time-series (tracked via git-lfs)

Check that git-lfs has downloaded the parquet files:

```bash
git lfs pull
ls artifacts/pcz-ppo/data/metrics/*.parquet | wc -l   # should be > 100
```

## Step 3: Rebuild Figures and Paper Fragments

```bash
# Rebuild all stale stages (figures + generated/*.tex fragments)
uv run python artifacts/pcz-ppo/paper/paper_build.py --build

# Expected output: "Rebuilt N/7 stages" (or "All stages up-to-date")
```

The build DAG has 7 stages:
1. `fragments` — renders all `generated/*.tex` numerical claims from `results.csv`
2. `fig_ablation_bar` — ablation bar chart
3. `fig_cross_env` — cross-environment comparison figure
4. `fig_kscaling` — K-scaling results figure
5. `fig_learning_curves_4M` — 4M-step learning curves
6. `fig_sample_efficiency` — sample efficiency figure
7. `pdf` — compiles `pcz_ppo.pdf` via latexmk

Wall-clock for a full rebuild from scratch: ~3–5 min (figures) + ~30 s (PDF).

## Step 4: Compile the PDF

```bash
uv run python artifacts/pcz-ppo/paper/paper_build.py --build --pdf
```

Output: `artifacts/pcz-ppo/paper/pcz_ppo.pdf` (35 pages).

## Step 5: Run Paper Tests

```bash
uv run pytest artifacts/pcz-ppo/paper/tests/ -v
```

Tests cover:
- Fragment formatter correctness (mean±std, delta, p-value formatting)
- Registry invariants (every `\cnum{}` reference is registered, no orphan fragments)
- Atomicity (partial render leaves no stale fragments)
- Idempotency (re-running render produces identical output)
- Headline-number regression (K=4/6/8 headline stats match committed fragments)

## Step 6: Verify Claim Consistency

```bash
# Re-renders all fragments to a tempdir and diffs against committed files
uv run python artifacts/pcz-ppo/paper/render_claims.py --check

# Check for hand-typed numerical claims (should be none)
uv run python artifacts/pcz-ppo/paper/lint_hardcoded_numbers.py
```

Both should exit 0.

## Running New Experiments (Optional)

If you want to reproduce experiments from scratch rather than using the committed data:

### Start MLflow

```bash
docker compose --env-file .env.mlflow -f docker-compose.mlflow.yml up -d
# UI at http://localhost:5050
```

### Run Training

```bash
# Example: PCZ-PPO K=4 headline experiment (5 seeds, ~1h on CPU)
for seed in 42 43 44 45 46; do
    uv run python -m core.train \
        --algorithm torchrl-pcz-ppo-running \
        --env lunarlander \
        --seeds $seed \
        --total-timesteps 500000 \
        --reward-component-weights 10.0,5.0,0.5,0.5 \
        --ent-coef-schedule 0.1:0.01 \
        --mlflow-tracking-uri http://127.0.0.1:5050
done
```

See `docs/architecture.md` for the full environment catalog and training configuration reference.

### Export Data

```bash
uv run python -m core.plot.export_results \
    --tracking-uri http://127.0.0.1:5050 \
    --output artifacts/pcz-ppo/data/results.csv --append

uv run python -m core.plot.export_metrics \
    --tracking-uri http://127.0.0.1:5050 \
    --output-dir artifacts/pcz-ppo/data/metrics --append
```

Then re-run Step 3 to rebuild figures from the new data.

## Expected Disk Space

| Artifact | Size |
|----------|------|
| Python dependencies (`uv sync`) | ~2.5 GB |
| Parquet files (`data/metrics/`) | ~150 MB (200+ files × ~200–500 KB each) |
| `results.csv` | ~300 KB |
| Generated figures (PDF+PNG × 5) | ~10 MB |
| Compiled PDF | ~700 KB |

## Troubleshooting

**PDF compile fails:** Check `artifacts/pcz-ppo/paper/pcz_ppo.log` for LaTeX errors. Common cause: missing LaTeX package. Install `texlive-full` or the specific package reported.

**`paper_build.py --check` fails:** A figure or fragment is stale. Run `--build` to regenerate. If the check still fails after building, a figure script has a non-deterministic output (check for unseeded random calls).

**parquet file not found:** Run `git lfs pull`. If git-lfs is not installed, install it first: `git lfs install`.

**Smoke test an algorithm without MLflow:**
```bash
uv run python -m core.train \
    --algorithm torchrl-pcz-ppo-running \
    --env lunarlander \
    --total-timesteps 50000 \
    --no-mlflow
```
