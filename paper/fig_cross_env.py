"""Generate cross-environment comparison figure.

Plots PCZ-PPO vs PPO across environments using min-max normalized reward so
scales are comparable. Raw absolute numbers live in Table~\\ref{tab:cross_env}.

Usage:
    uv run python paper/fig_cross_env.py
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

ENV_CONFIGS = [
    {"env": "lunarlander-k2", "K": 2, "mismatch": "Sparse/Dense", "display": "LL-K2"},
    {"env": "lunarlander", "K": 4, "mismatch": "Sparse/Dense", "display": "LL-K4"},
    {"env": "lunarlander-k6", "K": 6, "mismatch": "Sparse/Dense", "display": "LL-K6"},
    {"env": "lunarlander-k8", "K": 8, "mismatch": "Sparse/Dense", "display": "LL-K8"},
    {"env": "halfcheetah", "K": 2, "mismatch": "Dense/Dense", "display": "HalfCheetah"},
    {"env": "bipedalwalker", "K": 3, "mismatch": "Terminal sparse", "display": "BipedalWalker"},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="paper/fig_cross_env.pdf")
    parser.add_argument("--timesteps", type=int, default=500000)
    args = parser.parse_args()

    rows = load_results()

    # Canonical weight filters mirror render_claims.py
    ENV_WEIGHTS = {
        "lunarlander": "10.00,5.00,0.50,0.50",
        "lunarlander-k6": "10.00,3.00,1.00,1.00,0.50,0.50",
    }
    # Apply ent_coef filter only on LunarLander family (fixed-entropy tuning runs
    # collide there); other envs use cosine schedule consistently.
    LL_ENVS = {"lunarlander", "lunarlander-k2", "lunarlander-k6", "lunarlander-k8"}
    labels, pcz_m, pcz_s, ppo_m, ppo_s, k_values, ratios, seeds_str = [], [], [], [], [], [], [], []
    for cfg in ENV_CONFIGS:
        w = ENV_WEIGHTS.get(cfg["env"])
        ent = "0.1:0.01" if cfg["env"] in LL_ENVS else None
        pcz = query(
            rows,
            algorithm="torchrl-pcz-ppo-running",
            env=cfg["env"],
            total_timesteps=args.timesteps,
            weights=w,
            ent_coef_schedule=ent,
        )
        ppo = query(
            rows,
            algorithm="torchrl-ppo",
            env=cfg["env"],
            total_timesteps=args.timesteps,
            weights=w,
            ent_coef_schedule=ent,
        )
        if pcz["seeds"] == 0 or ppo["seeds"] == 0:
            print(f"  Skipping {cfg['env']}: missing data (PCZ={pcz['seeds']}s, PPO={ppo['seeds']}s)")
            continue
        labels.append(f"{cfg['display']}\n({cfg['mismatch']})")
        pcz_m.append(pcz["mean"])
        pcz_s.append(pcz["std"])
        ppo_m.append(ppo["mean"])
        ppo_s.append(ppo["std"])
        k_values.append(cfg["K"])
        # Ratio is PCZ/PPO; undefined when PPO<=0 so we report signed difference instead for those
        ratios.append(pcz["mean"] / ppo["mean"] if ppo["mean"] > 0 else float("nan"))
        seeds_str.append(f"{pcz['seeds']}s/{ppo['seeds']}s")
        print(
            f"  {cfg['env']}: PCZ {pcz['mean']:.1f}+/-{pcz['std']:.1f} ({pcz['seeds']}s) vs PPO {ppo['mean']:.1f}+/-{ppo['std']:.1f} ({ppo['seeds']}s)"
        )

    if not labels:
        print("No data found.")
        sys.exit(1)

    # --- Normalize: min-max within each env using {PCZ, PPO, 0} anchors so 0 reward is a shared baseline ---
    # Stds are scaled by the same range so error bars remain comparable within env.
    pcz_norm, ppo_norm, pcz_norm_err, ppo_norm_err = [], [], [], []
    for pm, om, ps, os in zip(pcz_m, ppo_m, pcz_s, ppo_s):
        lo = min(pm, om, 0.0)
        hi = max(pm, om)
        rng = hi - lo if hi > lo else 1.0
        pcz_norm.append((pm - lo) / rng)
        ppo_norm.append((om - lo) / rng)
        pcz_norm_err.append(ps / rng)
        ppo_norm_err.append(os / rng)

    _fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.4, 1]})
    x = np.arange(len(labels))
    width = 0.38

    # --- Left: normalized reward (std scaled by the same per-env range) ---
    ax = axes[0]
    ax.bar(
        x - width / 2,
        pcz_norm,
        width,
        yerr=pcz_norm_err,
        label="PCZ-PPO",
        color="#2196F3",
        alpha=0.9,
        edgecolor="black",
        linewidth=0.4,
        capsize=3,
        ecolor="#222",
    )
    ax.bar(
        x + width / 2,
        ppo_norm,
        width,
        yerr=ppo_norm_err,
        label="PPO",
        color="#FF9800",
        alpha=0.9,
        edgecolor="black",
        linewidth=0.4,
        capsize=3,
        ecolor="#222",
    )
    ylim_top = 1.25
    label_bbox = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#333", linewidth=0.4)
    for i, (pm, om) in enumerate(zip(pcz_m, ppo_m)):
        ax.text(
            i - width / 2,
            pcz_norm[i],
            f"{pm:.0f}",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            bbox=label_bbox,
            zorder=5,
        )
        ax.text(
            i + width / 2,
            ppo_norm[i],
            f"{om:.0f}",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            bbox=label_bbox,
            zorder=5,
        )
    for i, k in enumerate(k_values):
        ax.text(i, -0.06, f"K={k}", ha="center", fontsize=8, fontweight="bold", color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Normalized Eval Reward  (per-env min-max, 0 = PPO-or-zero floor)", fontsize=10)
    ax.set_title(f"Cross-Environment Comparison ({args.timesteps // 1000}k Steps)", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.12, ylim_top)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")

    # --- Right: PCZ/PPO ratio (where PPO > 0) ---
    ax2 = axes[1]
    valid = [(lbl, r) for lbl, r in zip(labels, ratios) if not np.isnan(r)]
    if valid:
        lbl_v, r_v = zip(*valid)
        xr = np.arange(len(lbl_v))
        colors = ["#2196F3" if r >= 1 else "#FF9800" for r in r_v]
        ax2.bar(xr, r_v, color=colors, alpha=0.85, edgecolor="black", linewidth=0.4)
        ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
        for i, r in enumerate(r_v):
            ax2.text(i, r + 0.05, f"{r:.2f}x", ha="center", fontsize=8, fontweight="bold")
        ax2.set_xticks(xr)
        ax2.set_xticklabels([lbl.split("\n")[0] for lbl in lbl_v], fontsize=8, rotation=20, ha="right")
    ax2.set_ylabel("PCZ-PPO / PPO (mean ratio)", fontsize=10)
    ax2.set_title("Relative Improvement", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.savefig(args.output.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
