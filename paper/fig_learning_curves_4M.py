"""Generate 4M-step learning curves: K=4 plateau + K=8 weight-sensitivity panel.

Complement to ``fig_learning_curves.py`` (500k headline). At 4M steps:
  - K=4 with canonical weights (10,5,0.5,0.5): PCZ-PPO and PPO converge
    near the same asymptote (sample-efficiency framing, §5.3).
  - K=8 weight-sensitivity (negative control, §Limitations):
      * Heterogeneous weights (10,1,1,1,1,1,0.5,0.5) — PCZ wins.
      * Equal weights (env default, all 1.0) — PCZ collapses, PPO healthy.
      * Zero-vel weights (10,5,0,1,0.5,0.5,0.5,0.5) — PCZ collapses.
    Read together: PCZ-PPO at K=8/4M requires a strictly-positive heterogeneous
    weight set; equal or zero-weighted-component configurations are pathological.

Separate file from the 500k script because the weight/entropy configs differ
between 500k and 4M and a single parameterised script would be a pile of
if/else branches.

Usage:
    cd /workspace && uv run python artifacts/pcz-ppo/paper/fig_learning_curves_4M.py
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from fig_data import METRICS_DIR, load_results

# Declared inputs for paper_build.py (paths relative to this file's dir).
INPUTS = [
    "../data/results.csv",
    "../data/metrics/*_torchrl-pcz-ppo-running_lunarlander_*.parquet",
    "../data/metrics/*_torchrl-pcz-ppo-running_lunarlander-k8_*.parquet",
    "../data/metrics/*_torchrl-ppo_lunarlander_*.parquet",
    "../data/metrics/*_torchrl-ppo_lunarlander-k8_*.parquet",
]


def load_curves_exact(
    algorithm: str,
    env: str,
    total_timesteps: int,
    weights_exact: str,
    ent_coef_schedule: str = "0.1:0.01",
    metric: str = "rollout/reward_mean",
) -> list[dict]:
    """Load parquet curves with an EXACT weight-string match (not prefix).

    Needed to distinguish the equal-weights K=8 4M regime (component_weights
    column is empty string, env default = 1.0 each) from heterogeneous
    weight runs at the same env/timesteps. ``fig_data.load_parquet_curves``
    uses ``startswith`` and treats ``weights=""`` as "no filter", so it
    can't isolate the empty-weights cell.
    """
    rows = load_results()
    matching = [
        r
        for r in rows
        if r["algorithm"] == algorithm
        and r["env"] == env
        and r["total_timesteps"] == str(total_timesteps)
        and r.get("component_weights", "") == weights_exact
        and r.get("ent_coef_schedule", "") == ent_coef_schedule
        and r.get("eval_mean", "") != ""
    ]
    by_seed: dict[str, dict] = {}
    for r in matching:
        by_seed[r["seed"]] = r  # keep last (CSV is chrono-ordered)
    matching = list(by_seed.values())

    curves = []
    for row in matching:
        run_id = row["run_id"]
        seed = int(row["seed"])
        for fname in os.listdir(METRICS_DIR):
            if fname.startswith(run_id) and fname.endswith(".parquet"):
                df = pq.read_table(str(METRICS_DIR / fname)).to_pandas()
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


def plot_curve(ax, curves: list[dict], label: str, color: str, n_bins: int = 120):
    """Plot mean +/- std shading for a set of seed curves."""
    if not curves:
        ax.text(0.5, 0.5, f"No data for {label}", transform=ax.transAxes, ha="center", fontsize=10)
        return

    dfs = []
    for c in curves:
        dfs.append(pd.DataFrame({"step": c["steps"], "reward": c["values"], "seed": c["seed"]}))
    df = pd.concat(dfs, ignore_index=True)

    step_min, step_max = df["step"].min(), df["step"].max()
    bins = np.linspace(step_min, step_max, n_bins + 1)
    df["step_bin"] = pd.cut(df["step"], bins=bins, labels=False, include_lowest=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    grouped = df.groupby("step_bin")["reward"].agg(["mean", "std"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0)
    grouped["step"] = bin_centers[grouped["step_bin"].astype(int)]

    ax.plot(grouped["step"], grouped["mean"], color=color, label=f"{label} ({len(curves)}s)", linewidth=2)
    ax.fill_between(
        grouped["step"],
        grouped["mean"] - grouped["std"],
        grouped["mean"] + grouped["std"],
        alpha=0.2,
        color=color,
    )


PCZ_COLOR = "#2196F3"
PPO_COLOR = "#FF9800"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/pcz-ppo/paper/fig_learning_curves_4M.pdf")
    args = parser.parse_args()

    # Panel definitions: (env, weights_exact, title, schedule).  Titles are
    # bare configuration descriptors only — interpretation, section
    # references, and the win/collapse narrative live in the LaTeX caption,
    # not in the figure (so the figure stays compact and the prose stays
    # editable in one place).
    panels = [
        {
            "env": "lunarlander",
            "weights": "10.00,5.00,0.50,0.50",
            "title": "$K{=}4$, weights $(10,5,0.5,0.5)$",
        },
        {
            "env": "lunarlander-k8",
            "weights": "10.00,1.00,1.00,1.00,1.00,1.00,0.50,0.50",
            "title": "$K{=}8$, weights $(10,1,1,1,1,1,0.5,0.5)$",
        },
        {
            "env": "lunarlander-k8",
            "weights": "",
            "title": "$K{=}8$, equal weights (env default)",
        },
        {
            "env": "lunarlander-k8",
            "weights": "10.00,5.00,0.00,1.00,0.50,0.50,0.50,0.50",
            "title": "$K{=}8$, weights $(10,5,0,1,0.5,0.5,0.5,0.5)$",
        },
    ]

    _fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, panel in zip(axes, panels):
        for algo, label, color in [
            ("torchrl-pcz-ppo-running", "PCZ-PPO", PCZ_COLOR),
            ("torchrl-ppo", "PPO", PPO_COLOR),
        ]:
            curves = load_curves_exact(algo, panel["env"], 4_000_000, panel["weights"])
            plot_curve(ax, curves, label, color)
            print(
                f"  {panel['env']}/{algo}/weights='{panel['weights'] or '(empty)'}': "
                f"{len(curves)} seed curves loaded (4M)"
            )

        ax.set_xlabel("Training Steps", fontsize=11)
        ax.set_ylabel("Rollout Mean Reward", fontsize=11)
        ax.set_title(panel["title"], fontsize=11, fontweight="bold")
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.savefig(args.output.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
