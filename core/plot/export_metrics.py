"""export_metrics.py: Export MLflow metric time-series to parquet.

Connects to an MLflow tracking server, pulls all metric histories for
runs in an experiment, and writes one parquet file per run into a
``metrics/`` directory.  Parquet is ~10x smaller than CSV for numeric
time-series and loads instantly into pandas/polars.

Each parquet file contains columns: ``metric``, ``step``, ``value``.
File naming: ``{run_id_short}_{algorithm}_{env}_s{seed}.parquet``.

Usage::

    cd /workspace

    # Export all runs from an experiment
    python -m core.plot.export_metrics \\
        --tracking-uri http://127.0.0.1:5050 \\
        --output-dir data/metrics

    # Append mode — skip runs that already have a parquet file
    python -m core.plot.export_metrics \\
        --tracking-uri http://127.0.0.1:5050 \\
        --output-dir data/metrics \\
        --append

    # Export specific metrics only
    python -m core.plot.export_metrics \\
        --tracking-uri http://127.0.0.1:5050 \\
        --output-dir data/metrics \\
        --metrics "rollout/reward_mean" "train/loss"

    # List available metrics
    python -m core.plot.export_metrics \\
        --tracking-uri http://127.0.0.1:5050 \\
        --list-metrics

    # Legacy CSV output (single file, all runs)
    python -m core.plot.export_metrics \\
        --tracking-uri http://127.0.0.1:5050 \\
        --output-dir data/metrics \\
        --format csv
"""

import argparse
import csv
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from mlflow.tracking import MlflowClient


def _run_filename(run, fmt: str = "parquet") -> str:
    """Build a descriptive filename for a run's metrics file."""
    p = run.data.params
    run_id = run.info.run_id[:12]
    algo = p.get("algorithm", "unknown").replace("/", "-")
    env = p.get("env", "unknown")
    seed = p.get("seed", "0")
    return f"{run_id}_{algo}_{env}_s{seed}.{fmt}"


def _existing_run_ids(output_dir: str, fmt: str) -> set[str]:
    """Scan output directory for already-exported run IDs."""
    if not os.path.isdir(output_dir):
        return set()
    ids = set()
    suffix = f".{fmt}"
    for fname in os.listdir(output_dir):
        if fname.endswith(suffix):
            # First 12 chars are the run_id
            ids.add(fname[:12])
    return ids


def export_experiment(
    tracking_uri: str,
    experiment_name: str,
    output_dir: str,
    metric_filters: list[str] | None = None,
    append: bool = False,
    fmt: str = "parquet",
) -> int:
    """Export metric histories for all runs in an experiment.

    Args:
        tracking_uri: MLflow tracking URI.
        experiment_name: Name of the experiment.
        output_dir: Directory for output files (one per run).
        metric_filters: If set, only export these metric keys.
        append: If True, skip runs already exported.
        fmt: Output format — "parquet" (default) or "csv".

    Returns:
        Number of runs exported.
    """
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Error: experiment '{experiment_name}' not found.")
        print("Available experiments:")
        for exp in client.search_experiments():
            print(f"  - {exp.name}")
        return 0

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time ASC"],
        max_results=5000,
    )
    if not runs:
        print(f"No runs found in experiment '{experiment_name}'.")
        return 0

    existing_ids = _existing_run_ids(output_dir, fmt) if append else set()
    new_runs = [r for r in runs if r.info.run_id[:12] not in existing_ids]

    if not new_runs:
        print(f"No new runs to export ({len(runs)} total, all already exported).")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(new_runs)} new run(s) in '{experiment_name}' (of {len(runs)} total)")

    runs_exported = 0
    for run in new_runs:
        run_id = run.info.run_id
        run_name = run.info.run_name or run_id[:8]

        metric_keys = list(run.data.metrics.keys())
        if metric_filters:
            metric_keys = [k for k in metric_keys if k in metric_filters]

        if not metric_keys:
            print(f"  {run_name}: no matching metrics, skipping")
            continue

        metrics_list = []
        steps_list = []
        values_list = []

        for key in sorted(metric_keys):
            history = client.get_metric_history(run_id, key)
            for point in history:
                metrics_list.append(key)
                steps_list.append(point.step)
                values_list.append(point.value)

        if not metrics_list:
            continue

        fname = _run_filename(run, fmt)
        fpath = os.path.join(output_dir, fname)

        if fmt == "parquet":
            table = pa.table(
                {
                    "metric": pa.array(metrics_list, type=pa.string()),
                    "step": pa.array(steps_list, type=pa.int64()),
                    "value": pa.array(values_list, type=pa.float64()),
                }
            )
            pq.write_table(table, fpath, compression="snappy")
        else:
            with open(fpath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "step", "value"])
                for m, s, v in zip(metrics_list, steps_list, values_list):
                    writer.writerow([m, s, v])

        n_points = len(metrics_list)
        n_metrics = len(set(metrics_list))
        print(f"  {run_name}: {n_metrics} metrics, {n_points} points -> {fname}")
        runs_exported += 1

    print(f"\nExported {runs_exported} runs to {output_dir}/")
    return runs_exported


def list_metrics(tracking_uri: str, experiment_name: str) -> None:
    """List all available metric keys across runs in an experiment."""
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Error: experiment '{experiment_name}' not found.")
        return

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    all_keys = set()
    for run in runs:
        all_keys.update(run.data.metrics.keys())

    print(f"Metrics in '{experiment_name}' ({len(all_keys)}):")
    for key in sorted(all_keys):
        print(f"  {key}")


def main():
    parser = argparse.ArgumentParser(
        description="Export MLflow metric time-series to parquet (one file per run).",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        required=True,
        help="MLflow tracking URI (e.g. http://127.0.0.1:5050).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="pcz-ppo",
        help="Name of the MLflow experiment (default: pcz-ppo).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/metrics",
        help="Output directory for metric files (default: data/metrics).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Filter: only export these metric keys.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Skip runs already exported (match by run_id in filenames).",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format: parquet (default) or csv.",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List available metrics and exit.",
    )
    args = parser.parse_args()

    if args.list_metrics:
        list_metrics(args.tracking_uri, args.experiment_name)
    else:
        count = export_experiment(
            args.tracking_uri,
            args.experiment_name,
            args.output_dir,
            args.metrics,
            args.append,
            args.format,
        )
        if count == 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
