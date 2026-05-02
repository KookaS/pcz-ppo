"""component_correlation.py: Pairwise Pearson correlation heatmap for reward components.

Loads a single parquet file from ``artifacts/pcz-ppo/data/metrics/``, extracts the
per-component time-series (``reward_components/*_mean`` metrics), computes a pairwise
Pearson correlation matrix, and saves a labeled heatmap to
``artifacts/pcz-ppo/data/diagnostics/<run_name>_corr.{pdf,png}``.

Usage::

    uv run python -m core.plot.component_correlation <path-to-parquet>

    # Example
    uv run python -m core.plot.component_correlation \\
        artifacts/pcz-ppo/data/metrics/006f88c30b2c_torchrl-pcz-ppo-running_lunarlander-k6_s50.parquet
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_component_matrix(filepath: str) -> tuple[np.ndarray, list[str]]:
    """Extract reward-component time series from a parquet file.

    Filters rows where ``metric`` starts with ``reward_components/`` and ends
    with ``_mean``, aligns on the common step index (inner join), and returns
    a (T x K) float array alongside the K component names.

    Returns:
        matrix: float array of shape (T, K), rows are timesteps.
        names:  list of K component name strings (prefix/suffix stripped).

    Raises:
        ValueError: if fewer than 2 components are found.
    """
    df = pq.read_table(filepath).to_pandas()

    comp_metrics = sorted(
        m for m in df["metric"].unique() if m.startswith("reward_components/") and m.endswith("_mean")
    )

    if len(comp_metrics) < 2:
        raise ValueError(
            f"Need at least 2 reward-component metrics; found {len(comp_metrics)} "
            f"in {filepath}. Available metrics: {sorted(df['metric'].unique())}"
        )

    # Build a step-indexed dict of value arrays, then align on common steps.
    series: dict[str, dict[int, float]] = {}
    for metric in comp_metrics:
        subset = df[df["metric"] == metric].sort_values("step")
        series[metric] = dict(zip(subset["step"].tolist(), subset["value"].tolist()))

    # Inner join on steps present in all components.
    common_steps = sorted(set.intersection(*(set(s.keys()) for s in series.values())))

    if len(common_steps) < 2:
        raise ValueError(
            f"Fewer than 2 common timesteps across components — cannot compute "
            f"correlation.  Steps per component: "
            f"{ {m: len(v) for m, v in series.items()} }"
        )

    matrix = np.array(
        [[series[m][s] for m in comp_metrics] for s in common_steps],
        dtype=float,
    )  # (T, K)

    names = [m.removeprefix("reward_components/").removesuffix("_mean") for m in comp_metrics]

    return matrix, names


def compute_correlation(matrix: np.ndarray) -> np.ndarray:
    """Return the (K x K) Pearson correlation matrix for a (T x K) input."""
    # np.corrcoef expects (K, T) — variables as rows.
    return np.corrcoef(matrix.T)


def _run_name_from_path(filepath: str) -> str:
    """Derive a canonical run name from the parquet filename."""
    return os.path.splitext(os.path.basename(filepath))[0]


def plot_heatmap(
    corr: np.ndarray,
    names: list[str],
    run_name: str,
    output_dir: str,
    dpi: int = 150,
) -> tuple[str, str]:
    """Render the correlation matrix as a labeled heatmap.

    Saves both PDF and PNG to ``<output_dir>/<run_name>_corr.{pdf,png}``.

    Returns:
        (pdf_path, png_path)
    """
    k = len(names)
    fig_side = max(4.0, k * 0.9 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_side, fig_side))

    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="auto")

    # Axis labels
    ax.set_xticks(range(k))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(k))
    ax.set_yticklabels(names, fontsize=9)

    # Annotate each cell with the correlation value
    for i in range(k):
        for j in range(k):
            val = corr[i, j]
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=text_color)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=10)
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    ax.set_title(
        f"Reward-Component Correlation\n{run_name}",
        fontsize=11,
        pad=12,
    )

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{run_name}_corr.pdf")
    png_path = os.path.join(output_dir, f"{run_name}_corr.png")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return pdf_path, png_path


def component_correlation(
    parquet_path: str,
    output_dir: str | None = None,
    dpi: int = 150,
) -> tuple[str, str]:
    """End-to-end pipeline: load → correlate → plot.

    Args:
        parquet_path: Absolute or relative path to a parquet metrics file.
        output_dir:   Where to write outputs.  Defaults to
                      ``artifacts/pcz-ppo/data/diagnostics/`` relative to the
                      directory that contains the ``data/metrics/`` dir.
        dpi:          PNG resolution.

    Returns:
        (pdf_path, png_path)
    """
    parquet_path = os.path.abspath(parquet_path)

    if output_dir is None:
        # artifacts/pcz-ppo/data/metrics/<file> -> artifacts/pcz-ppo/data/diagnostics/
        metrics_dir = os.path.dirname(parquet_path)
        data_dir = os.path.dirname(metrics_dir)
        output_dir = os.path.join(data_dir, "diagnostics")

    matrix, names = load_component_matrix(parquet_path)
    corr = compute_correlation(matrix)
    run_name = _run_name_from_path(parquet_path)

    pdf_path, png_path = plot_heatmap(corr, names, run_name, output_dir, dpi=dpi)
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute pairwise Pearson correlation between reward-component time "
            "series and save a labeled heatmap as PDF and PNG."
        ),
    )
    parser.add_argument(
        "parquet",
        metavar="PATH",
        help="Path to a parquet metrics file (artifacts/pcz-ppo/data/metrics/*.parquet).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for output files.  Defaults to the diagnostics/ sibling "
            "of the metrics/ directory containing the parquet file."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG resolution (default: 150).",
    )
    args = parser.parse_args()

    try:
        pdf_path, png_path = component_correlation(
            parquet_path=args.parquet,
            output_dir=args.output_dir,
            dpi=args.dpi,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Saved PDF → {pdf_path}")
    print(f"Saved PNG → {png_path}")


if __name__ == "__main__":
    main()
