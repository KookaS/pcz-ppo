"""Shared data loading for paper figure scripts.

All fig_*.py scripts import from here to query results.csv and parquet
metrics.  Never hardcode experiment numbers — read them from the data
pipeline.

Usage::

    from fig_data import load_results, query, load_parquet_curves

    df = load_results()
    pcz = query(df, algorithm="torchrl-pcz-ppo-running", env="lunarlander", total_timesteps=500000)
    # pcz = {"mean": ..., "std": ..., "seeds": ..., "runs": [...]}
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

# Resolve paths relative to this file (artifacts/pcz-ppo/paper/)
_PAPER_DIR = Path(__file__).parent
_DATA_DIR = _PAPER_DIR.parent / "data"
RESULTS_CSV = _DATA_DIR / "results.csv"
METRICS_DIR = _DATA_DIR / "metrics"


def load_results(path: str | Path | None = None) -> list[dict]:
    """Load results.csv as a list of dicts."""
    path = Path(path) if path else RESULTS_CSV
    with open(path) as f:
        return list(csv.DictReader(f))


def query(
    rows: list[dict],
    algorithm: str | None = None,
    env: str | None = None,
    total_timesteps: int | None = None,
    seed: int | None = None,
    weights: str | None = None,
    ent_coef_schedule: str | None = None,
    learning_rate: str | None = None,
) -> dict:
    """Filter results and compute aggregate stats.

    Args:
        weights: Component weights prefix filter (e.g. "10.00,5.00" to match
            "10.00,5.00,0.50,0.50"). Pass None to skip filtering.
        ent_coef_schedule: Entropy schedule filter (exact match). Pass
            ``"0.1:0.01"`` to restrict to canonical cosine-schedule runs and
            exclude fixed-entropy tuning-audit runs that share seeds.
        learning_rate: Learning rate exact-match filter (e.g. "0.0003"). Pass
            when HP-sweep runs at non-canonical LRs share seeds with
            canonical runs — prevents chrono-latest from displacing canonical
            seeds with non-canonical HP cells.

    Returns:
        {"mean": float, "std": float, "seeds": int, "runs": [filtered rows]}
    """
    filtered = rows
    if algorithm is not None:
        filtered = [r for r in filtered if r["algorithm"] == algorithm]
    if env is not None:
        filtered = [r for r in filtered if r["env"] == env]
    if total_timesteps is not None:
        filtered = [r for r in filtered if r["total_timesteps"] == str(total_timesteps)]
    if seed is not None:
        filtered = [r for r in filtered if r["seed"] == str(seed)]
    if weights is not None:
        filtered = [r for r in filtered if r.get("component_weights", "").startswith(weights)]
    if ent_coef_schedule is not None:
        filtered = [r for r in filtered if r.get("ent_coef_schedule", "") == ent_coef_schedule]
    if learning_rate is not None:
        filtered = [r for r in filtered if r.get("learning_rate", "") == learning_rate]

    if not filtered:
        return {"mean": 0.0, "std": 0.0, "seeds": 0, "runs": []}

    # Deduplicate by seed — chronologically-latest per unique seed.
    # This rule is applied uniformly across every (algorithm, env, ts, weights)
    # query in the paper.  It is intentionally neutral: it does not depend on
    # outcome (eval_mean) and cannot be tuned per-algorithm.  When two runs
    # share the same date, ties break on CSV row order (deterministic because
    # results.csv is written by export_results.py with stable ordering).  See
    # §A.3 "Seed selection rule" in the paper for the user-facing disclosure.
    filtered_sorted = sorted(
        [r for r in filtered if r.get("eval_mean")],
        key=lambda r: r.get("date", ""),
    )
    by_seed: dict[str, dict] = {}
    for r in filtered_sorted:
        by_seed[r["seed"]] = r

    evals = [float(r["eval_mean"]) for r in by_seed.values()]
    if not evals:
        return {"mean": 0.0, "std": 0.0, "seeds": 0, "runs": filtered}

    n = len(evals)
    # Sample SD (ddof=1) for inferential reporting — matches rliable and
    # standard practice in Agarwal et al. (NeurIPS 2021).  Single-seed
    # results have std=0 (no inference possible).
    ddof = 1 if n > 1 else 0
    return {
        "mean": round(float(np.mean(evals)), 1),
        "std": round(float(np.std(evals, ddof=ddof)), 1),
        "mean_raw": float(np.mean(evals)),
        "std_raw": float(np.std(evals, ddof=ddof)),
        "seeds": n,
        "runs": list(by_seed.values()),
    }


def load_parquet_curves(
    algorithm: str,
    env: str,
    total_timesteps: int,
    seeds: list[int] | None = None,
    metric: str = "rollout/reward_mean",
    metrics_dir: str | Path | None = None,
    weights: str | None = None,
    ent_coef_schedule: str | None = None,
    learning_rate: str | None = None,
) -> list[dict]:
    """Load time-series from parquet files for learning curve plots.

    Dedupes to one curve per (algorithm, env, timesteps, seed) by keeping the
    last matching row in results.csv — the same convention used by `query`.

    Args:
        weights: Component weights prefix filter (e.g. "10.00,5.00"). None skips filtering.
        ent_coef_schedule: Entropy schedule exact-match filter (e.g. "0.1:0.01").
            None skips filtering. Use "" to match runs with empty schedule.
        learning_rate: Exact-match filter (e.g. "0.0003"). None skips filtering.

    Returns list of {"seed": int, "steps": np.array, "values": np.array}
    """
    import pyarrow.parquet as pq

    metrics_dir = Path(metrics_dir) if metrics_dir else METRICS_DIR
    if not metrics_dir.is_dir():
        return []

    # First, find matching run_ids from results.csv
    rows = load_results()
    matching = [
        r
        for r in rows
        if r["algorithm"] == algorithm
        and r["env"] == env
        and r["total_timesteps"] == str(total_timesteps)
        and (seeds is None or int(r["seed"]) in seeds)
        and (weights is None or r.get("component_weights", "").startswith(weights))
        and (ent_coef_schedule is None or r.get("ent_coef_schedule", "") == ent_coef_schedule)
        and (learning_rate is None or r.get("learning_rate", "") == learning_rate)
    ]

    # Dedupe: one row per seed (keep last, matching `query` convention)
    by_seed: dict[str, dict] = {}
    for r in matching:
        by_seed[r["seed"]] = r
    matching = list(by_seed.values())

    curves = []
    for row in matching:
        run_id = row["run_id"]
        seed = int(row["seed"])
        # Find parquet file by run_id prefix
        for fname in os.listdir(metrics_dir):
            if fname.startswith(run_id) and fname.endswith(".parquet"):
                df = pq.read_table(str(metrics_dir / fname)).to_pandas()
                metric_df = df[df["metric"] == metric].sort_values("step")
                if not metric_df.empty:
                    curves.append(
                        {
                            "seed": seed,
                            "steps": metric_df["step"].values,
                            "values": metric_df["value"].values,
                        }
                    )
                break

    return curves
