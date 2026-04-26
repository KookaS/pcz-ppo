"""export_results.py: Export experiment summary to lightweight CSV.

Produces one row per MLflow run with key parameters, final eval statistics,
and per-component reward metrics.  This is the portable, version-controlled
experiment index — human-scannable at ~200-300 bytes/run.

The full metric time-series (for regenerating plots) can be exported
separately via ``export_metrics.py`` into parquet files.

Usage::

    # Export from MLflow
    python -m core.plot.export_results \
        --tracking-uri http://127.0.0.1:5050 \
        --output artifacts/pcz-ppo/data/results.csv

    # Append new runs (skip existing run_ids)
    python -m core.plot.export_results \
        --tracking-uri http://127.0.0.1:5050 \
        --output artifacts/pcz-ppo/data/results.csv \
        --append
"""

import argparse
import csv
import os
import sys
from datetime import UTC, datetime

from mlflow.tracking import MlflowClient

# Fixed columns — always present in every row
FIXED_COLUMNS = [
    "run_id",
    "date",
    "algorithm",
    "env",
    "seed",
    "total_timesteps",
    "n_envs",
    # Final eval
    "eval_mean",
    "eval_std",
    # Hyperparameters
    "learning_rate",
    "gamma",
    "gae_lambda",
    "clip_epsilon",
    "num_epochs",
    "minibatch_size",
    "max_grad_norm",
    "ent_coef",
    "ent_coef_schedule",
    "ent_coef_schedule_type",
    "frames_per_batch",
    "component_weights",
    "hidden_size",
    "activation",
    "adam_eps",
    "lr_anneal",
    "normalize_advantage",
    "device",
    "framework",
    "status",
]


def _load_existing(path: str) -> tuple[set[str], list[str]]:
    """Load run_ids and column headers from an existing CSV.

    Returns:
        (existing_ids, existing_columns)
    """
    if not os.path.exists(path):
        return set(), []
    ids = set()
    columns = []
    with open(path) as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        for row in reader:
            if "run_id" in row:
                ids.add(row["run_id"])
    return ids, columns


def _load_existing_rows(path: str) -> list[dict]:
    """Load every row of an existing CSV as list of dicts.

    Used when we need to rewrite due to new-column discovery but must
    preserve rows from OTHER experiments that are not in the current
    MLflow query. Prior behaviour dropped those rows silently — caused
    a 151-row regression during a trading-k6 export.
    """
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _discover_component_columns(runs, client: MlflowClient) -> list[str]:
    """Discover all reward_components/* metric keys across runs.

    Returns sorted list of component column names like
    ``["comp/landing_mean", "comp/landing_std", ...]``.
    """
    comp_keys = set()
    for run in runs:
        for key in run.data.metrics:
            if key.startswith("reward_components/"):
                # reward_components/landing_mean -> comp/landing_mean
                suffix = key.removeprefix("reward_components/")
                comp_keys.add(f"comp/{suffix}")
    return sorted(comp_keys)


def _get_param(params: dict, *keys, default="") -> str:
    """Try multiple param key names, return first found."""
    for k in keys:
        if k in params:
            return params[k]
    return default


def export_results(
    tracking_uri: str,
    output_path: str,
    experiment_name: str = "pcz-ppo",
    append: bool = False,
) -> int:
    """Export one-row-per-run summary to CSV.

    Args:
        tracking_uri: MLflow tracking URI.
        output_path: Output CSV path.
        experiment_name: MLflow experiment name.
        append: If True, skip runs already in the CSV.

    Returns:
        Number of new rows written.
    """
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Error: experiment '{experiment_name}' not found.")
        return 0

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time ASC"],
        max_results=5000,
    )
    if not runs:
        print(f"No runs in '{experiment_name}'.")
        return 0

    existing_ids, existing_columns = _load_existing(output_path) if append else (set(), [])
    new_runs = [r for r in runs if r.info.run_id[:12] not in existing_ids]

    if not new_runs:
        print(f"No new runs to export ({len(runs)} total, all already in CSV).")
        return 0

    # Discover dynamic component columns from the current experiment's runs.
    # CRITICAL: Union with existing CSV columns so cross-experiment columns
    # from older envs are preserved. Without this union, rewriting the CSV
    # after a new-column discovery silently drops `comp/*` columns belonging
    # to other experiments.
    comp_columns_this = _discover_component_columns(runs, client)
    existing_comp_cols = [c for c in existing_columns if c.startswith("comp/")]
    # Preserve order: existing comp columns first (stable), then new ones
    comp_columns = list(existing_comp_cols) + [c for c in comp_columns_this if c not in existing_comp_cols]
    all_columns = FIXED_COLUMNS + comp_columns

    # If appending, check if we need to rewrite (new columns discovered)
    needs_rewrite = False
    if append and os.path.exists(output_path) and existing_columns:
        new_cols = set(all_columns) - set(existing_columns)
        if new_cols:
            print(f"  New columns discovered: {new_cols} — rewriting CSV with full schema.")
            needs_rewrite = True

    if needs_rewrite:
        # Re-export this experiment's runs to get consistent columns, AND
        # preserve rows from other experiments (or older runs no longer in
        # MLflow) by carrying them over with empty values in the new cols.
        # Without the carry-over, cross-experiment rows and legacy rows get
        # silently deleted when a new export discovers additional comp/* columns.
        preserved_rows = [
            r for r in _load_existing_rows(output_path) if r.get("run_id") not in {ru.info.run_id[:12] for ru in runs}
        ]
        existing_ids = set()
        new_runs = runs
        mode = "w"
        write_header = True
    else:
        preserved_rows = []
        mode = "a" if append and os.path.exists(output_path) else "w"
        write_header = mode == "w"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rows_written = 0
    with open(output_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        # When rewriting, first carry over rows from other experiments /
        # older runs no longer in the current MLflow query (bug fix; see
        # `_load_existing_rows` docstring).
        for row in preserved_rows:
            writer.writerow(row)

        for run in new_runs:
            p = run.data.params
            m = run.data.metrics
            start_time = datetime.fromtimestamp(run.info.start_time / 1000, tz=UTC)

            row = {
                "run_id": run.info.run_id[:12],
                "date": start_time.strftime("%Y-%m-%d"),
                "algorithm": p.get("algorithm", ""),
                "env": p.get("env", ""),
                "seed": p.get("seed", ""),
                "total_timesteps": p.get("total_timesteps", ""),
                "n_envs": p.get("n_envs", ""),
                # Final eval
                "eval_mean": f"{m['eval/mean_reward']:.2f}" if "eval/mean_reward" in m else "",
                "eval_std": f"{m['eval/std_reward']:.2f}" if "eval/std_reward" in m else "",
                # Hyperparameters
                "learning_rate": p.get("learning_rate", ""),
                "gamma": p.get("gamma", ""),
                "gae_lambda": p.get("gae_lambda", ""),
                "clip_epsilon": p.get("clip_epsilon", ""),
                "num_epochs": _get_param(p, "num_epochs", "n_epochs"),
                "minibatch_size": _get_param(p, "minibatch_size", "mini_batch_size"),
                "max_grad_norm": p.get("max_grad_norm", ""),
                "ent_coef": p.get("ent_coef", ""),
                "ent_coef_schedule": p.get("ent_coef_schedule", ""),
                "ent_coef_schedule_type": p.get("ent_coef_schedule_type", ""),
                "frames_per_batch": p.get("frames_per_batch", ""),
                "component_weights": p.get("component_weights", ""),
                "hidden_size": p.get("hidden_size", ""),
                "activation": p.get("activation", ""),
                "adam_eps": p.get("adam_eps", ""),
                "lr_anneal": p.get("lr_anneal", ""),
                "normalize_advantage": p.get("normalize_advantage", ""),
                "device": p.get("device", ""),
                "framework": p.get("framework", ""),
                "status": run.info.status,
            }

            # Per-component final metrics (last logged value)
            for key in m:
                if key.startswith("reward_components/"):
                    suffix = key.removeprefix("reward_components/")
                    col = f"comp/{suffix}"
                    row[col] = f"{m[key]:.4f}"

            writer.writerow(row)
            rows_written += 1

    total = len(existing_ids) + rows_written + len(preserved_rows)
    print(f"Exported {rows_written} new runs ({total} total) to {output_path}")
    return rows_written


def main():
    parser = argparse.ArgumentParser(
        description="Export experiment summary (1 row/run) to CSV.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        required=True,
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="pcz-ppo",
        help="MLflow experiment name (default: pcz-ppo).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/pcz-ppo/data/results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new runs only (skip existing run_ids).",
    )
    args = parser.parse_args()

    count = export_results(
        args.tracking_uri,
        args.output,
        args.experiment_name,
        args.append,
    )
    if count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
