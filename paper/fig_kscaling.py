"""Generate K-scaling figure: PCZ-PPO advantage grows with K.

Reads from results.csv via fig_data.  Maps lunarlander-k{N} env names
to K values automatically.

Usage:
    cd /workspace && uv run python artifacts/pcz-ppo/paper/fig_kscaling.py
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

# K -> env name mapping for LunarLander variants
K_ENV_MAP = {
    2: "lunarlander-k2",
    4: "lunarlander",
    6: "lunarlander-k6",
    8: "lunarlander-k8",
}


def main():
    parser = argparse.ArgumentParser(description="K-scaling figure")
    parser.add_argument("--output", default="artifacts/pcz-ppo/paper/fig_kscaling.pdf")
    parser.add_argument("--timesteps", type=int, default=500000)
    args = parser.parse_args()

    rows = load_results()
    k_values = sorted(K_ENV_MAP.keys())

    pcz_mean, pcz_std, pcz_seeds, pcz_std_raw = [], [], [], []
    ppo_mean, ppo_std, ppo_seeds, ppo_std_raw = [], [], [], []
    mh_mean, mh_std = [], []

    # Weight prefixes to filter out non-primary weight configurations
    # IMPORTANT: these must match the canonical filters used by
    # render_claims.py so that figure numbers equal fragment numbers.  K=6
    # MUST apply the weight filter (there is one accidental empty-weights
    # seed-42 row; see §A.3 Run-level deduplication rule).
    K_WEIGHTS = {
        2: None,  # K=2 uses lunarlander-k2 env, no ambiguity
        4: "10.00,5.00,0.50,0.50",  # K=4 primary
        6: "10.00,3.00,1.00,1.00,0.50,0.50",  # K=6 canonical (excludes empty-weights seed-42)
        8: None,  # K=8 uses lunarlander-k8 env, no ambiguity
    }

    for k in k_values:
        env = K_ENV_MAP[k]
        w = K_WEIGHTS.get(k)
        # ent_coef_schedule filter mirrors render_claims.py: isolates canonical
        # cosine-schedule runs from fixed-entropy tuning-audit runs
        pcz = query(
            rows,
            algorithm="torchrl-pcz-ppo-running",
            env=env,
            total_timesteps=args.timesteps,
            weights=w,
            ent_coef_schedule="0.1:0.01",
        )
        ppo = query(
            rows,
            algorithm="torchrl-ppo",
            env=env,
            total_timesteps=args.timesteps,
            weights=w,
            ent_coef_schedule="0.1:0.01",
        )
        mh = query(
            rows,
            algorithm="torchrl-ppo-multihead",
            env=env,
            total_timesteps=args.timesteps,
            weights=w,
            ent_coef_schedule="0.1:0.01",
        )

        pcz_mean.append(pcz["mean"])
        pcz_std.append(pcz["std"])
        pcz_std_raw.append(pcz["std_raw"])
        pcz_seeds.append(pcz["seeds"])
        ppo_mean.append(ppo["mean"])
        ppo_std.append(ppo["std"])
        ppo_std_raw.append(ppo["std_raw"])
        ppo_seeds.append(ppo["seeds"])
        mh_mean.append(mh["mean"] if mh["seeds"] > 0 else None)
        mh_std.append(mh["std"] if mh["seeds"] > 0 else None)

        print(
            f"  K={k}: PCZ {pcz['mean']:.1f}+/-{pcz['std']:.1f} ({pcz['seeds']}s) | PPO {ppo['mean']:.1f}+/-{ppo['std']:.1f} ({ppo['seeds']}s) | MH {mh['mean']:.1f} ({mh['seeds']}s)"
        )

    _fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})
    x = np.arange(len(k_values))

    # --- Left panel: Eval Mean ---
    ax = axes[0]
    width = 0.08
    ax.errorbar(
        x - width,
        pcz_mean,
        yerr=pcz_std,
        marker="o",
        color="#2196F3",
        label="PCZ-PPO",
        linewidth=2.5,
        markersize=10,
        capsize=6,
        capthick=1.5,
    )
    ax.errorbar(
        x + width,
        ppo_mean,
        yerr=ppo_std,
        marker="s",
        color="#FF9800",
        label="PPO",
        linewidth=2.5,
        markersize=10,
        capsize=6,
        capthick=1.5,
    )

    # Multi-head PPO intentionally omitted from the K-scaling plot: it has data
    # only at K=4 (n=10) and K=8 (n=3) — two points cannot show a trend, and
    # the K=4 value is already reported as row C5 in the ablation table.  The
    # query above is kept so the printed diagnostics still show MH coverage,
    # but the series is not drawn.

    ax.set_xlabel("K (Number of Reward Components)", fontsize=12)
    ax.set_ylabel("Eval Mean Reward", fontsize=12)
    ax.set_title("PCZ-PPO: Stable Across K", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # --- Right panel: Variance Ratio ---
    ax2 = axes[1]
    # Use raw (unrounded) stds so annotated values match the \cnum fragments
    var_ratios = [(p / max(g, 0.1)) ** 2 for g, p in zip(pcz_std_raw, ppo_std_raw)]

    ax2.bar(x, var_ratios, color="#4CAF50", alpha=0.8, width=0.5, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"K={k}" for k in k_values])
    ax2.set_ylabel("Variance Ratio (PPO / PCZ-PPO)", fontsize=12)
    ax2.set_title("PPO Variance Grows with K", fontsize=13, fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, axis="y")

    for i, vr in enumerate(var_ratios):
        ax2.text(i, vr * 1.15, f"{vr:.1f}x", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylim(top=ax2.get_ylim()[1] * 2.5)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.savefig(args.output.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
