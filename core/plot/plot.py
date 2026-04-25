"""plot.py: Generate comparison plots from MLflow-exported CSV data.

Reads CSV produced by ``export_metrics.py`` and generates multi-line
comparison plots with configurable smoothing, one plot per metric.
All runs are overlaid for easy visual comparison.

Usage::

    cd /workspace

    # Plot all metrics from exported CSV
    python -m core.plot.plot --csv metrics.csv --output-dir plots/

    # Plot specific metrics with heavy smoothing
    python -m core.plot.plot --csv metrics.csv \\
        --metrics "rollout/ep_rew_mean" "train/entropy_loss" \\
        --smoothing 75 --output-dir plots/

    # Grid layout with custom figure size
    python -m core.plot.plot --csv metrics.csv --grid --output-dir plots/
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
    }
)

# Default color palette — up to 17 algorithms
COLORS = [
    "#1f77b4",
    "#e07020",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
]

ALPHA_RAW = 0.15
ALPHA_SMOOTH = 1.0


def smoothing_to_ewm_span(smoothing: int) -> int | None:
    """Convert smoothing 0–100 to EWM span.

    0 = no smoothing (returns None), 100 = very heavy smoothing (span=200).
    """
    if smoothing <= 0:
        return None
    # Map 1–100 linearly to span 2–200
    return max(2, int(2 + (smoothing / 100) * 198))


def millions_formatter(x, _pos):
    """Format axis tick as e.g. 5K, 1M."""
    if x == 0:
        return "0"
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"


def plot_metric(
    df: pd.DataFrame,
    metric_name: str,
    smoothing: int = 50,
    ax: plt.Axes | None = None,
    show_legend: bool = True,
) -> plt.Axes:
    """Plot one metric with N runs overlaid.

    Args:
        df: DataFrame with columns [run_name, metric, step, value].
        metric_name: The metric key to plot.
        smoothing: 0–100 smoothing level.
        ax: Matplotlib axes to draw on (creates new figure if None).
        show_legend: Whether to show the legend.
    """
    metric_df = df[df["metric"] == metric_name].copy()
    if metric_df.empty:
        print(f"  Warning: no data for metric '{metric_name}'")
        if ax is None:
            _, ax = plt.subplots()
        ax.set_title(metric_name)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=14, color="#999")
        return ax

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    ewm_span = smoothing_to_ewm_span(smoothing)
    run_names = metric_df["run_name"].unique()

    for i, run_name in enumerate(sorted(run_names)):
        color = COLORS[i % len(COLORS)]
        run_data = metric_df[metric_df["run_name"] == run_name].sort_values("step")
        steps = run_data["step"].values
        values = run_data["value"].values

        # Raw data — faint background
        ax.plot(steps, values, color=color, alpha=ALPHA_RAW, linewidth=0.8, label="_nolegend_")

        # Smoothed line
        if ewm_span is not None:
            smoothed = pd.Series(values).ewm(span=ewm_span, min_periods=1).mean().values
        else:
            smoothed = values
        ax.plot(steps, smoothed, color=color, alpha=ALPHA_SMOOTH, linewidth=2.0, label=run_name)

    ax.set_xlabel("Step")
    ax.set_ylabel(metric_name.split("/")[-1].replace("_", " ").title())
    ax.set_title(metric_name)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(millions_formatter))

    if show_legend:
        ax.legend(loc="best", frameon=True, framealpha=0.8)

    return ax


def plot_all_metrics(
    csv_path: str,
    output_dir: str,
    metric_filters: list[str] | None = None,
    smoothing: int = 50,
    grid: bool = False,
    dpi: int = 150,
):
    """Generate plots for all (or filtered) metrics in a CSV.

    Args:
        csv_path: Path to CSV from export_metrics.py.
        output_dir: Directory to save PNG files.
        metric_filters: If set, only plot these metrics.
        smoothing: 0–100 smoothing level.
        grid: If True, put all metrics in one grid figure.
        dpi: Output resolution.
    """
    df = pd.read_csv(csv_path)

    # Validate columns
    required = {"run_name", "metric", "step", "value"}
    if not required.issubset(df.columns):
        print(f"Error: CSV must have columns {required}, got {set(df.columns)}")
        return

    all_metrics = sorted(df["metric"].unique())
    if metric_filters:
        all_metrics = [m for m in all_metrics if m in metric_filters]

    if not all_metrics:
        print("No metrics to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)
    n_runs = df["run_name"].nunique()

    print(f"\nPlotting {len(all_metrics)} metric(s) for {n_runs} run(s)")
    print(f"Smoothing: {smoothing}/100 (EWM span={smoothing_to_ewm_span(smoothing)})")

    if grid:
        # All metrics in one grid figure
        n = len(all_metrics)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, metric_name in enumerate(all_metrics):
            plot_metric(df, metric_name, smoothing=smoothing, ax=axes[i], show_legend=(i == 0))

        # Hide unused subplots
        for j in range(len(all_metrics), len(axes)):
            axes[j].set_visible(False)

        # Shared legend from first plot
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(6, len(labels)),
                frameon=False,
                bbox_to_anchor=(0.5, -0.02),
            )

        fig.suptitle(f"Training Metrics Comparison ({n_runs} runs)", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])

        out_path = os.path.join(output_dir, "grid_comparison.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved grid → {out_path}")
    else:
        # Individual plots
        for metric_name in all_metrics:
            fig, ax = plt.subplots(figsize=(12, 5))
            plot_metric(df, metric_name, smoothing=smoothing, ax=ax)
            fig.tight_layout()

            safe_name = metric_name.replace("/", "_").replace(" ", "_")
            out_path = os.path.join(output_dir, f"{safe_name}.png")
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from MLflow-exported CSV.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV from export_metrics.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save PNG plots (default: plots/).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Filter: only plot these metric keys.",
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=50,
        help="Smoothing level 0–100 (0=raw, 50=moderate, 100=heavy). Default: 50.",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Put all metrics in one grid figure instead of separate files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output resolution (default: 150).",
    )
    args = parser.parse_args()

    plot_all_metrics(
        csv_path=args.csv,
        output_dir=args.output_dir,
        metric_filters=args.metrics,
        smoothing=args.smoothing,
        grid=args.grid,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
