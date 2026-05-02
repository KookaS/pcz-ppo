"""Generate weight-sensitivity bar chart: K=8 / 4M cliff.

Single panel, LunarLander-K8 at 4M steps with cosine entropy schedule 0.1->0.01.
Four heterogeneity tiers ordered max-heterogeneous (left) to equal (right).

The visual story: PCZ-PPO wins at the most heterogeneous tier and *collapses to
near-zero* at every other tier — a cliff, not a slope.  Compare with PPO which
stays in the +125 to +214 range across all tiers.

The K=4 weight sensitivity (already in Table~\\ref{tab:weight_sensitivity})
shows a gentler version of the same pattern; we keep the K=8 panel here because
the cliff at long-horizon high-K is what the data actually shouts.

Reads results.csv via fig_data.

Usage:
    cd /workspace && uv run python artifacts/pcz-ppo/paper/fig_weight_sensitivity.py
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
OUTPUTS = ["fig_weight_sensitivity.pdf", "fig_weight_sensitivity.png"]

# K=8 weight tiers (LunarLander-K8 4M, cosine entropy)
TIERS = [
    {
        "label": "(10,1,1,1,1,1,0.5,0.5)",
        "tier": "Heterogeneous\n20$\\times$ max/min",
        "weights": "10.00,1.00,1.00,1.00,1.00,1.00,0.50,0.50",
        "exact": False,
    },
    {
        "label": "(5,3,1,1,1,1,0.5,0.5)",
        "tier": "Moderate\n10$\\times$",
        "weights": "5.00,3.00,1.00,1.00,1.00,1.00,0.50,0.50",
        "exact": False,
    },
    {
        "label": "(3,3,3,3,3,3,3,3)",
        "tier": "Flat",
        "weights": "3.00,3.00,3.00,3.00,3.00,3.00,3.00,3.00",
        "exact": False,
    },
    {"label": "(1,1,...,1)", "tier": "Equal\n(env default)", "weights": "", "exact": True},
]

PCZ_COLOR = "#2196F3"
PPO_COLOR = "#FF9800"


def _grab(rows, algo, weights, exact):
    if exact:
        matching = [
            r
            for r in rows
            if r["algorithm"] == algo
            and r["env"] == "lunarlander-k8"
            and r["total_timesteps"] == "4000000"
            and r.get("component_weights", "") == weights
            and r.get("ent_coef_schedule", "") == "0.1:0.01"
            and r.get("eval_mean")
        ]
        matching.sort(key=lambda r: r.get("date", ""))
        by_seed: dict = {}
        for r in matching:
            by_seed[r["seed"]] = r
        vals = [float(r["eval_mean"]) for r in by_seed.values()]
    else:
        q = query(
            rows,
            algorithm=algo,
            env="lunarlander-k8",
            total_timesteps=4_000_000,
            weights=weights,
            ent_coef_schedule="0.1:0.01",
        )
        vals = [float(r["eval_mean"]) for r in q["runs"]] if q["seeds"] > 0 else []
    if not vals:
        return 0.0, 0.0, 0
    arr = np.asarray(vals, dtype=float)
    ddof = 1 if len(arr) > 1 else 0
    return float(arr.mean()), float(arr.std(ddof=ddof)), len(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(Path(__file__).parent / "fig_weight_sensitivity.pdf"))
    args = parser.parse_args()

    rows = load_results()

    pcz_means, pcz_stds, pcz_n = [], [], []
    ppo_means, ppo_stds, ppo_n = [], [], []
    for tier in TIERS:
        m, s, n = _grab(rows, "torchrl-pcz-ppo-running", tier["weights"], tier["exact"])
        pcz_means.append(m)
        pcz_stds.append(s)
        pcz_n.append(n)
        m, s, n = _grab(rows, "torchrl-ppo", tier["weights"], tier["exact"])
        ppo_means.append(m)
        ppo_stds.append(s)
        ppo_n.append(n)

    _, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(TIERS))
    w = 0.4

    ax.bar(
        x - w / 2,
        pcz_means,
        w,
        yerr=pcz_stds,
        color=PCZ_COLOR,
        capsize=4,
        edgecolor="black",
        linewidth=0.5,
        label="PCZ-PPO",
        alpha=0.9,
        ecolor="#222",
    )
    ax.bar(
        x + w / 2,
        ppo_means,
        w,
        yerr=ppo_stds,
        color=PPO_COLOR,
        capsize=4,
        edgecolor="black",
        linewidth=0.5,
        label="PPO",
        alpha=0.9,
        ecolor="#222",
    )

    # Compute headroom for label placement
    ymax = max(max(pcz_means) + max(pcz_stds), max(ppo_means) + max(ppo_stds))
    ymin = min(min(pcz_means) - max(pcz_stds), min(ppo_means) - max(ppo_stds), 0)
    pad = (ymax - ymin) * 0.06

    # Labels above/below bars (fig_ablation style): never inside, never clipped.
    for i in range(len(TIERS)):
        for offset, m, s, n in [
            (-w / 2, pcz_means[i], pcz_stds[i], pcz_n[i]),
            (+w / 2, ppo_means[i], ppo_stds[i], ppo_n[i]),
        ]:
            if n == 0:
                continue
            if m >= 0:
                ypos = m + s + pad * 0.4
                va = "bottom"
            else:
                ypos = m - s - pad * 0.4
                va = "top"
            ax.text(
                i + offset,
                ypos,
                f"${m:+.0f}{{\\pm}}{s:.0f}$\n($n{{=}}{n}$)",
                ha="center",
                va=va,
                fontsize=9,
                fontweight="bold",
            )

    # Two-row x-axis labels: weights vector + heterogeneity tier name
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t['label']}\n{t['tier']}" for t in TIERS], fontsize=9)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_ylabel("Eval Mean Reward", fontsize=11)
    ax.set_title(
        "PCZ-PPO weight-sensitivity cliff at $K{=}8$ / 4M steps\n"
        "Wins only at heterogeneous-positive weights; collapses everywhere else",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.2, axis="y")

    # Add headroom for the labels above the tallest bar+std stack
    ax.set_ylim(bottom=ymin - pad * 1.5, top=ymax + pad * 2.5)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.savefig(args.output.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
