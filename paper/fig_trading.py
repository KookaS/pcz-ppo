"""Generate synthetic-trading figure for App.D.3.

Two panels documenting the structural null at competence horizon:

  Left:  trading-k4 5M learning curves — PCZ-PPO and PPO at best HPs per
         algorithm (PPO: lr=1e-4 cosine 0.1->0.01; PCZ: lr=3e-4 fixed ent=0.01),
         seeds 42-51 (n=10). Oracle (+81.3) and 80%-of-oracle competence
         threshold (+65) marked. Visualises that both algorithms learn
         monotonically and plateau short of competence.

  Right: K-scaling at 5M — bar chart of final eval mean +- std for PCZ-PPO
         and PPO at K in {2,4,6,8}. K=4 at n=10 (Stage 2); K=2/6/8 at n=5
         (Stage 3 K-scaling extension). Oracle reference line.

Net visual story: trading is a structural null at competence; both algorithms
plateau below the 80%-of-oracle threshold; the K=6/8 directional gap is
visually consistent with the d~0.73 underpowered trend disclosed in
App.D.3 prose.

Reads from results.csv (final means) and parquet metrics (curves).

Usage:
    cd /workspace && uv run python artifacts/pcz-ppo/paper/fig_trading.py
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
from fig_data import METRICS_DIR, load_results, query

INPUTS = [
    "../data/results.csv",
    "../data/metrics/*_torchrl-pcz-ppo-running_trading-k*_*.parquet",
    "../data/metrics/*_torchrl-ppo_trading-k*_*.parquet",
]
UNITS = "rollout-per-step,eval-mean-final"  # Layer 4: declared metric (rollout/eval/ratio/etc.)
MIXED_UNITS_ACKNOWLEDGED = True  # Layer 4: figure mixes multiple metrics in one panel; caption explains the difference
OUTPUTS = ["fig_trading.pdf", "fig_trading.png"]

PCZ_COLOR = "#2196F3"
PPO_COLOR = "#FF9800"
ORACLE_COLOR = "#9C27B0"
ORACLE_VALUE = 81.3
COMPETENCE_THRESHOLD = 0.80 * ORACLE_VALUE  # 65.04


def load_curves_for_trading(
    algorithm: str,
    env: str,
    total_timesteps: int,
    learning_rate: str,
    ent_coef_schedule: str,
    ent_coef: str | None = None,
    seeds: list[int] | None = None,
    metric: str = "rollout/reward_mean",
) -> list[dict]:
    """Load parquet curves with HP filters specific to trading TV1.7.2 / TV1.7.3 runs.

    Trading runs use non-canonical HPs (per Stage 1 sweep): PPO lr=1e-4 cosine,
    PCZ lr=3e-4 fixed ent=0.01. The standard ``load_parquet_curves`` filter API
    handles ent_coef_schedule but not ent_coef, so we duplicate the loader here
    with the extra filter.
    """
    rows = load_results()
    matching = []
    for r in rows:
        if r["algorithm"] != algorithm:
            continue
        if r["env"] != env:
            continue
        if r["total_timesteps"] != str(total_timesteps):
            continue
        if r.get("learning_rate", "") != learning_rate:
            continue
        if r.get("ent_coef_schedule", "") != ent_coef_schedule:
            continue
        if ent_coef is not None and r.get("ent_coef", "") != ent_coef:
            continue
        if seeds is not None and int(r["seed"]) not in seeds:
            continue
        if r.get("eval_mean", "") in ("", "nan", None):
            continue
        matching.append(r)

    by_seed: dict[str, dict] = {}
    for r in sorted(matching, key=lambda x: x.get("date", "")):
        by_seed[r["seed"]] = r  # chrono-latest per seed
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


def plot_curves(ax, curves: list[dict], label: str, color: str, n_bins: int = 80):
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

    ax.plot(grouped["step"], grouped["mean"], color=color, label=f"{label} (n={len(curves)})", linewidth=2)
    ax.fill_between(
        grouped["step"],
        grouped["mean"] - grouped["std"],
        grouped["mean"] + grouped["std"],
        alpha=0.2,
        color=color,
    )


def panel_left_curves(ax) -> None:
    """trading-k4 5M learning curves at best HPs per algorithm.

    Plots ``rollout/reward_mean`` (per-step mean reward during training rollouts;
    raw/un-normalized). Different units from the right panel's episodic
    ``eval/mean_reward`` because the trading env has variable-length episodes
    (max 4000 steps, typically much shorter); per-episode eval is logged once
    post-training, so the time-series view requires per-step rollout. The
    visual story is identical: both algorithms learn monotonically, then
    plateau --- with PPO and PCZ-PPO indistinguishable throughout.
    """
    pcz_curves = load_curves_for_trading(
        "torchrl-pcz-ppo-running",
        "trading-k4",
        5_000_000,
        learning_rate="0.0003",
        ent_coef_schedule="",
        ent_coef="0.01",
        seeds=list(range(42, 52)),
    )
    ppo_curves = load_curves_for_trading(
        "torchrl-ppo",
        "trading-k4",
        5_000_000,
        learning_rate="0.0001",
        ent_coef_schedule="0.1:0.01",
        seeds=list(range(42, 52)),
    )

    plot_curves(ax, ppo_curves, "PPO (lr=1e-4, cosine)", PPO_COLOR)
    plot_curves(ax, pcz_curves, "PCZ-PPO (lr=3e-4, fixed ent=0.01)", PCZ_COLOR)

    ax.axhline(y=0, color="lightgray", linestyle="-", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Training steps", fontsize=11)
    ax.set_ylabel("Per-step rollout reward (training metric)", fontsize=11)
    ax.set_title(
        "Training — trading-k4 at 5M (rollout reward):\nboth algorithms learn and plateau, indistinguishable",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(left=0)


def panel_right_kscaling(ax) -> None:
    """K-scaling at 5M: bar chart with oracle line."""
    rows = load_results()
    K_values = [2, 4, 6, 8]
    pcz_means, pcz_stds, pcz_ns = [], [], []
    ppo_means, ppo_stds, ppo_ns = [], [], []

    for k in K_values:
        env = f"trading-k{k}"
        # PCZ at lr=3e-4 fixed ent=0.01
        pcz = query(
            rows,
            algorithm="torchrl-pcz-ppo-running",
            env=env,
            total_timesteps=5_000_000,
        )
        # PPO at lr=1e-4 cosine
        ppo = query(
            rows,
            algorithm="torchrl-ppo",
            env=env,
            total_timesteps=5_000_000,
        )
        pcz_means.append(pcz["mean"])
        pcz_stds.append(pcz["std"])
        pcz_ns.append(pcz["seeds"])
        ppo_means.append(ppo["mean"])
        ppo_stds.append(ppo["std"])
        ppo_ns.append(ppo["seeds"])

    x = np.arange(len(K_values))
    width = 0.35

    ax.bar(
        x - width / 2,
        ppo_means,
        width,
        yerr=ppo_stds,
        color=PPO_COLOR,
        alpha=0.85,
        capsize=4,
        edgecolor="black",
        linewidth=0.5,
        label="PPO",
    )
    ax.bar(
        x + width / 2,
        pcz_means,
        width,
        yerr=pcz_stds,
        color=PCZ_COLOR,
        alpha=0.85,
        capsize=4,
        edgecolor="black",
        linewidth=0.5,
        label="PCZ-PPO",
    )

    # Annotate n on each bar
    for i, (m, s, n) in enumerate(zip(ppo_means, ppo_stds, ppo_ns)):
        ax.text(x[i] - width / 2, m + s + 2, f"n={n}", ha="center", va="bottom", fontsize=8)
    for i, (m, s, n) in enumerate(zip(pcz_means, pcz_stds, pcz_ns)):
        ax.text(x[i] + width / 2, m + s + 2, f"n={n}", ha="center", va="bottom", fontsize=8)

    ax.axhline(
        y=ORACLE_VALUE,
        color=ORACLE_COLOR,
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"Oracle (+{ORACLE_VALUE:.1f})",
    )
    ax.axhline(y=0, color="lightgray", linestyle="-", linewidth=0.5, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in K_values])
    ax.set_xlabel("Number of reward components", fontsize=11)
    ax.set_ylabel("Episodic eval reward (final)", fontsize=11)
    ax.set_title(
        "Evaluation — K-scaling at 5M (final episodic eval):\nall p > 0.05; K=6/8 trend underpowered (d~0.73, n=5)",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.2, axis="y")
    # Adaptive top: ensure clearance above the tallest bar+std AND above the
    # oracle reference line, plus 15 units for the "n=N" annotation. Avoids
    # the K=4 PCZ "n=10" label from touching the bounding box (the previous
    # hardcoded ORACLE_VALUE + 25 left only ~2 units of clearance).
    max_top = max(m + s for m, s in zip(pcz_means + ppo_means, pcz_stds + ppo_stds))
    ax.set_ylim(bottom=0, top=max(max_top, ORACLE_VALUE) + 15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(Path(__file__).parent / "fig_trading.pdf"))
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})
    panel_left_curves(axes[0])
    panel_right_kscaling(axes[1])
    fig.suptitle(
        "Synthetic trading is a structural null at competence horizon",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.savefig(args.output.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
