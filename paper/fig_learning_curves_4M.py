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
UNITS = "rollout-per-step,eval-episodic"  # Layer 4: declared metric (rollout/eval/ratio/etc.)
MIXED_UNITS_ACKNOWLEDGED = True  # Layer 4: figure mixes multiple metrics in one panel; caption explains the difference


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

    ax.plot(grouped["step"], grouped["mean"], color=color, label=f"{label} (n={len(curves)})", linewidth=2)
    ax.fill_between(
        grouped["step"],
        grouped["mean"] - grouped["std"],
        grouped["mean"] + grouped["std"],
        alpha=0.2,
        color=color,
    )


PCZ_COLOR = "#2196F3"
PPO_COLOR = "#FF9800"


def eval_stats_exact(
    algorithm: str,
    env: str,
    total_timesteps: int,
    weights_exact: str,
    ent_coef_schedule: str = "0.1:0.01",
) -> tuple[float, float, int]:
    """Return (mean, std, n_seeds) of eval_mean for cells matching the exact weight string.

    Feeds the right column of the 4x2 figure layout — episodic eval-reward
    bars at 4M for each weight regime, paired row-wise with the rollout
    curves on the left column.
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
        and r.get("eval_mean", "") not in ("", "nan", None)
    ]
    by_seed: dict[str, dict] = {}
    for r in sorted(matching, key=lambda x: x.get("date", "")):
        by_seed[r["seed"]] = r
    evals = [float(r["eval_mean"]) for r in by_seed.values()]
    if not evals:
        return 0.0, 0.0, 0
    arr = np.asarray(evals)
    ddof = 1 if len(arr) > 1 else 0
    return float(arr.mean()), float(arr.std(ddof=ddof)), len(arr)


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

    # 4-row x 2-col layout: each row is one weight regime.  Left column
    # (3/4 width) = rollout-curve trajectories; right column (1/4 width)
    # = episodic eval bar at 4M.  Wide-aspect line plots make the 4M
    # x-axis legible, while the bar still answers "how good is the
    # converged deterministic policy?" without dominating the figure.
    _fig, axes = plt.subplots(4, 2, figsize=(14, 12), gridspec_kw={"width_ratios": [3, 1]})

    for row, panel in enumerate(panels):
        ax_top = axes[row, 0]
        ax_bot = axes[row, 1]

        # --- Left column: rollout curves (training metric) ---
        for algo, label, color in [
            ("torchrl-pcz-ppo-running", "PCZ-PPO", PCZ_COLOR),
            ("torchrl-ppo", "PPO", PPO_COLOR),
        ]:
            curves = load_curves_exact(algo, panel["env"], 4_000_000, panel["weights"])
            plot_curve(ax_top, curves, label, color)
            print(
                f"  {panel['env']}/{algo}/weights='{panel['weights'] or '(empty)'}': "
                f"{len(curves)} seed curves loaded (4M)"
            )

        ax_top.set_xlabel("Training Steps", fontsize=10)
        ax_top.set_ylabel("Per-step Train Reward", fontsize=10)
        ax_top.set_title(panel["title"], fontsize=10, fontweight="bold")
        ax_top.legend(fontsize=8, loc="upper left", framealpha=0.9)
        ax_top.grid(True, alpha=0.3)
        ax_top.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        # --- Bottom row: episodic eval bars (canonical RL metric) ---
        pcz_m, pcz_s, pcz_n = eval_stats_exact("torchrl-pcz-ppo-running", panel["env"], 4_000_000, panel["weights"])
        ppo_m, ppo_s, ppo_n = eval_stats_exact("torchrl-ppo", panel["env"], 4_000_000, panel["weights"])

        if pcz_n > 0 and ppo_n > 0:
            x_pos = [0, 1]
            means = [pcz_m, ppo_m]
            stds = [pcz_s, ppo_s]
            colors = [PCZ_COLOR, PPO_COLOR]
            labels = [f"PCZ-PPO\n(n={pcz_n})", f"PPO\n(n={ppo_n})"]
            ax_bot.bar(
                x_pos,
                means,
                yerr=stds,
                color=colors,
                alpha=0.85,
                capsize=6,
                edgecolor="black",
                linewidth=0.6,
            )
            # Annotate each bar with mean +/- std above the error-bar tip.
            # Compute a per-panel offset from the panel's data span so the
            # annotation sits clearly above the cap on every regime.
            top_tip = max(m + s for m, s in zip(means, stds))
            bot_tip = min(m - s for m, s in zip(means, stds))
            # Always include y=0 in the visible range — keeps the "0 baseline"
            # readable on all-positive panels and on signed panels.
            y_min_data = min(0.0, bot_tip)
            y_max_data = top_tip
            data_range = max(y_max_data - y_min_data, 1.0)
            ann_offset = 0.06 * data_range
            for xi, m, s in zip(x_pos, means, stds):
                ax_bot.text(
                    xi,
                    m + s + ann_offset,
                    f"${m:+.0f}{{\\pm}}{s:.0f}$",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            ax_bot.set_xticks(x_pos)
            ax_bot.set_xticklabels(labels, fontsize=9)
            ax_bot.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            # Headroom: leave another data-range fraction above the highest
            # annotation so its bbox stays well inside the axes (geometry-lint
            # enforces non-overflow at every text bbox).
            ax_bot.set_ylim(
                y_min_data - 0.05 * data_range,
                y_max_data + ann_offset + 0.18 * data_range,
            )
        else:
            ax_bot.text(0.5, 0.5, "No eval data", transform=ax_bot.transAxes, ha="center")
            ax_bot.set_xticks([])

        ax_bot.set_ylabel("Episodic Eval Reward", fontsize=10)
        ax_bot.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.savefig(args.output.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
