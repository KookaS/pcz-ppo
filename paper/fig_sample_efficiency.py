"""Generate sample efficiency figure: PCZ-PPO vs PPO across timestep budgets.

Reads from results.csv via fig_data.  Queries runs at each timestep budget.

Usage:
    uv run python paper/fig_sample_efficiency.py
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

TIMESTEPS = [100_000, 200_000, 500_000, 1_000_000]
TIMESTEP_LABELS = ["100k", "200k", "500k", "1M"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="paper/fig_sample_efficiency.pdf")
    parser.add_argument("--env", default="lunarlander")
    args = parser.parse_args()

    rows = load_results()

    # Filter to primary weight config for lunarlander
    weights = "10.00,5.00,0.50,0.50" if args.env == "lunarlander" else None

    pcz_mean, pcz_std, ppo_mean, ppo_std = [], [], [], []
    for ts in TIMESTEPS:
        pcz = query(
            rows,
            algorithm="torchrl-pcz-ppo-running",
            env=args.env,
            total_timesteps=ts,
            weights=weights,
            ent_coef_schedule="0.1:0.01",
        )
        ppo = query(
            rows,
            algorithm="torchrl-ppo",
            env=args.env,
            total_timesteps=ts,
            weights=weights,
            ent_coef_schedule="0.1:0.01",
        )
        pcz_mean.append(pcz["mean"])
        pcz_std.append(pcz["std"])
        ppo_mean.append(ppo["mean"])
        ppo_std.append(ppo["std"])
        print(
            f"  {ts // 1000}k: PCZ {pcz['mean']:.1f}+/-{pcz['std']:.1f} ({pcz['seeds']}s) | PPO {ppo['mean']:.1f}+/-{ppo['std']:.1f} ({ppo['seeds']}s)"
        )

    _fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(TIMESTEPS))

    # --- Left: Eval Mean ---
    ax = axes[0]
    ax.errorbar(
        x, pcz_mean, yerr=pcz_std, marker="o", color="#2196F3", label="PCZ-PPO", linewidth=2, markersize=8, capsize=5
    )
    ax.errorbar(
        x, ppo_mean, yerr=ppo_std, marker="s", color="#FF9800", label="PPO", linewidth=2, markersize=8, capsize=5
    )
    ax.fill_between(
        x, np.array(pcz_mean) - np.array(pcz_std), np.array(pcz_mean) + np.array(pcz_std), alpha=0.15, color="#2196F3"
    )
    ax.fill_between(
        x, np.array(ppo_mean) - np.array(ppo_std), np.array(ppo_mean) + np.array(ppo_std), alpha=0.15, color="#FF9800"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(TIMESTEP_LABELS)
    ax.set_xlabel("Training Timesteps", fontsize=12)
    ax.set_ylabel("Eval Mean Reward", fontsize=12)
    ax.set_title(f"Sample Efficiency \u2014 {args.env}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # --- Right: Gap and Variance ---
    ax2 = axes[1]
    gaps = [g - p for g, p in zip(pcz_mean, ppo_mean)]
    var_ratios = [(p / max(g, 0.1)) ** 2 for g, p in zip(pcz_std, ppo_std)]

    ax2.bar(x - 0.15, gaps, width=0.3, color="#4CAF50", alpha=0.8, label="PCZ - PPO (mean)")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, var_ratios, "D-", color="#9C27B0", linewidth=2, markersize=8, label="Variance ratio (PPO/PCZ)")

    ax2.set_xticks(x)
    ax2.set_xticklabels(TIMESTEP_LABELS)
    ax2.set_xlabel("Training Timesteps", fontsize=12)
    ax2.set_ylabel("Mean Reward Gap", fontsize=12, color="#4CAF50")
    ax2_twin.set_ylabel("Variance Ratio", fontsize=12, color="#9C27B0")
    ax2.set_title("PCZ-PPO Advantage Across Scales", fontsize=13, fontweight="bold")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.savefig(args.output.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
