"""Generate ablation bar chart: all algorithm variants on LunarLander 500k.

Reads from results.csv via fig_data. Shows mean +/- std for each variant.

Usage:
    cd /workspace && uv run python artifacts/pcz-ppo/paper/fig_ablation_bar.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fig_data import load_results, query

# Declared inputs for paper_build.py (paths relative to this file's dir).
INPUTS = [
    "../data/results.csv",
]
UNITS = "eval-mean-final"  # Layer 4: declared metric (rollout/eval/ratio/etc.)

# Script name and output name diverge here: the LaTeX cites
# ``fig_ablation.pdf`` from before the script was renamed to
# ``fig_ablation_bar.py``.  Declared explicitly so paper_build knows
# which files to hash-check.
OUTPUTS = [
    "fig_ablation.pdf",
    "fig_ablation.png",
]

# Algorithm registry: internal name -> display config
# Order doesn't matter — sorted by mean in the plot
VARIANTS = [
    {"algo": "torchrl-pcz-ppo-running", "label": "PCZ-PPO\n(running)", "color": "#2196F3"},
    {"algo": "torchrl-ppo-weighted-running", "label": "PPO (sum,\nrunning z-norm)", "color": "#64B5F6"},
    {"algo": "torchrl-ppo-znorm-post", "label": "PPO (sum,\nbatch z-norm)", "color": "#90CAF9"},
    {"algo": "torchrl-ppo", "label": "PPO\n(baseline)", "color": "#FF9800"},
    {"algo": "torchrl-ppo-znorm", "label": "PPO\nagg z-norm", "color": "#FFB74D"},
    {"algo": "torchrl-pcz-grpo", "label": "PCZ-GRPO\n(no critic)", "color": "#EF5350"},
    {"algo": "torchrl-ppo-popart", "label": "PopArt-PPO", "color": "#E57373"},
    {"algo": "torchrl-pcz-ppo-popart", "label": "PCZ-PPO\n+ PopArt", "color": "#F44336"},
    {"algo": "torchrl-ppo-no-norm", "label": "No norm", "color": "#BDBDBD"},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/pcz-ppo/paper/fig_ablation.pdf")
    parser.add_argument("--env", default="lunarlander")
    parser.add_argument("--timesteps", type=int, default=500000)
    args = parser.parse_args()

    rows = load_results()

    # Filter to primary weight config for lunarlander (10,5,0.5,0.5).
    # `learning_rate="0.0003"` mirrors render_claims.py canonical-LR guard so
    # ablation-bar means equal Table~\ref{tab:ablation} fragment values.
    weights = "10.00,5.00,0.50,0.50" if args.env == "lunarlander" else None
    learning_rate = "0.0003" if args.env == "lunarlander" else None

    names, means, stds, colors, seeds = [], [], [], [], []
    for v in VARIANTS:
        q = query(
            rows,
            algorithm=v["algo"],
            env=args.env,
            total_timesteps=args.timesteps,
            weights=weights,
            ent_coef_schedule="0.1:0.01",
            learning_rate=learning_rate,
        )
        if q["seeds"] == 0:
            print(f"  Skipping {v['algo']}: no data for {args.env}/{args.timesteps}")
            continue
        names.append(v["label"])
        means.append(q["mean"])
        stds.append(q["std"])
        colors.append(v["color"])
        seeds.append(q["seeds"])

    if not names:
        print("No data found. Check results.csv.")
        sys.exit(1)

    # Sort by mean descending
    order = np.argsort(means)[::-1]
    names = [names[i] for i in order]
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]
    colors = [colors[i] for i in order]
    seeds = [seeds[i] for i in order]

    _fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, color=colors, capsize=4, edgecolor="black", linewidth=0.5, alpha=0.85, width=0.7)

    for i, (m, s, n_seeds) in enumerate(zip(means, stds, seeds)):
        y_pos = m + s + 10 if m >= 0 else m - s - 15
        ax.text(
            i,
            y_pos,
            f"{m:+.0f}\u00b1{s:.0f} (n={n_seeds})",
            ha="center",
            va="bottom" if m >= 0 else "top",
            fontsize=7,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=25, ha="right")
    ax.set_ylabel("Eval Mean Reward", fontsize=12)
    ax.set_title(
        f"Algorithm Ablation \u2014 {args.env} {args.timesteps // 1000}k Steps", fontsize=13, fontweight="bold", pad=20
    )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.2, axis="y")
    # Symmetric clearance: top for positive-bar n labels above bar+std, bottom
    # for the most-negative bar's n label below bar-std.  Auto-derive from data
    # so adding new variants doesn't silently clip annotations.
    pad = max(stds) + 60
    ymax = max(means) + pad
    ymin = min(0.0, min(means) - pad)
    ax.set_ylim(bottom=ymin, top=ymax)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.savefig(args.output.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
