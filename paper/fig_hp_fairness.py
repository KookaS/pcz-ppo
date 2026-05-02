"""Generate hyperparameter-fairness bar chart: PPO at its own best vs PCZ-PPO.

Three bars on LunarLander K=4, 500k:
  1. PPO canonical (cosine entropy schedule 0.1->0.01)         — paper headline
  2. PPO at its own best fixed-entropy cell (lr=3e-4, ent=0.0)  — HP audit
  3. PCZ-PPO at PPO's best cell (lr=3e-4, ent=0.0)              — HP audit

The point: even when PPO is given its own preferred tuning, PCZ-PPO meets or
exceeds it.  The canonical-config gap (n=15) is therefore not a tuning artifact.

Reads results.csv via fig_data.

Usage:
    cd /workspace && uv run python artifacts/pcz-ppo/paper/fig_hp_fairness.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fig_data import load_results, query

INPUTS = ["../data/results.csv"]
UNITS = "eval-mean-final"  # Layer 4: declared metric (rollout/eval/ratio/etc.)
OUTPUTS = ["fig_hp_fairness.pdf", "fig_hp_fairness.png"]

PCZ_COLOR = "#2196F3"
PPO_COLOR = "#FF9800"


def _grab_w7_cell(rows: list[dict], algo: str) -> tuple[float, float, int]:
    """PPO/PCZ at lr=3e-4, ent=0.0, no schedule, weights (10,5,0.5,0.5).  Mirrors paired HP audit."""
    matching = [
        r
        for r in rows
        if r.get("algorithm") == algo
        and r.get("env") == "lunarlander"
        and r.get("total_timesteps") == "500000"
        and r.get("learning_rate") == "0.0003"
        and r.get("ent_coef") == "0.0"
        and not r.get("ent_coef_schedule")
        and r.get("component_weights", "").startswith("10.00,5.00,0.50,0.50")
        and r.get("eval_mean")
    ]
    matching.sort(key=lambda r: r.get("date", ""))
    by_seed: dict[str, dict] = {}
    for r in matching:
        by_seed[r["seed"]] = r
    vals = [float(r["eval_mean"]) for r in by_seed.values()]
    if not vals:
        return 0.0, 0.0, 0
    arr = np.asarray(vals, dtype=float)
    ddof = 1 if len(arr) > 1 else 0
    return float(arr.mean()), float(arr.std(ddof=ddof)), len(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(Path(__file__).parent / "fig_hp_fairness.pdf"))
    args = parser.parse_args()

    rows = load_results()

    # Bar 1: PPO canonical (cosine schedule)
    ppo_canon = query(
        rows,
        algorithm="torchrl-ppo",
        env="lunarlander",
        total_timesteps=500_000,
        weights="10.00,5.00,0.50,0.50",
        ent_coef_schedule="0.1:0.01",
    )
    pcz_canon = query(
        rows,
        algorithm="torchrl-pcz-ppo-running",
        env="lunarlander",
        total_timesteps=500_000,
        weights="10.00,5.00,0.50,0.50",
        ent_coef_schedule="0.1:0.01",
    )

    # Bars 2 & 3: PPO and PCZ-PPO at PPO's best fixed-entropy cell (paired HP audit)
    ppo_best_m, ppo_best_s, ppo_best_n = _grab_w7_cell(rows, "torchrl-ppo")
    pcz_best_m, pcz_best_s, pcz_best_n = _grab_w7_cell(rows, "torchrl-pcz-ppo-running")

    bars = [
        {
            "label": "PPO\ncanonical\n(cosine sched.)",
            "mean": ppo_canon["mean"],
            "std": ppo_canon["std"],
            "n": ppo_canon["seeds"],
            "color": PPO_COLOR,
            "alpha": 0.75,
        },
        {
            "label": "PPO\nat own best\n(lr=3e-4, ent=0)",
            "mean": ppo_best_m,
            "std": ppo_best_s,
            "n": ppo_best_n,
            "color": PPO_COLOR,
            "alpha": 1.0,
            "edge": "black",
        },
        {
            "label": "PCZ-PPO\nat PPO's best\n(lr=3e-4, ent=0)",
            "mean": pcz_best_m,
            "std": pcz_best_s,
            "n": pcz_best_n,
            "color": PCZ_COLOR,
            "alpha": 1.0,
            "edge": "black",
        },
        {
            "label": "PCZ-PPO\ncanonical\n(cosine sched.)",
            "mean": pcz_canon["mean"],
            "std": pcz_canon["std"],
            "n": pcz_canon["seeds"],
            "color": PCZ_COLOR,
            "alpha": 0.75,
        },
    ]

    _, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(bars))
    means = [b["mean"] for b in bars]
    stds = [b["std"] for b in bars]

    for i, b in enumerate(bars):
        ax.bar(
            x[i],
            b["mean"],
            yerr=b["std"],
            color=b["color"],
            alpha=b["alpha"],
            capsize=4,
            edgecolor="black",
            linewidth=0.6,
            width=0.65,
        )
        ax.text(
            x[i],
            b["mean"] + b["std"] + 8,
            f"${b['mean']:+.1f}\\pm{b['std']:.1f}$\n($n{{=}}{b['n']}$)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([b["label"] for b in bars], fontsize=9)
    ax.set_ylabel("Eval Mean Reward", fontsize=11)
    ax.set_title(
        "Hyperparameter Fairness — PCZ-PPO beats PPO even at PPO's preferred tuning\n(LunarLander $K{=}4$, 500k steps, weights $(10,5,0.5,0.5)$)",
        fontsize=11,
        fontweight="bold",
    )
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(top=max(m + s for m, s in zip(means, stds)) + 60)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.savefig(args.output.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
