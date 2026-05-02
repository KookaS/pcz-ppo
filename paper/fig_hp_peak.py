"""Peak-vs-peak hyperparameter fairness figure.

Shows, for each weight tier on LunarLander K=4/500k, the best cell mean ± std
for PCZ-PPO and PPO across all (lr, ent_config) combinations evaluated in the
HP sweep.  The figure honestly answers: "even at its own best tuning,
does PPO match or exceed PCZ-PPO?"

Usage:
    cd /workspace && uv run python artifacts/pcz-ppo/paper/fig_hp_peak.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fig_data import load_results

INPUTS = ["../data/results.csv"]
UNITS = "hp-grid"  # Layer 4: declared metric (rollout/eval/ratio/etc.)
OUTPUTS = ["fig_hp_peak.pdf", "fig_hp_peak.png"]

PCZ_COLOR = "#2196F3"
PPO_COLOR = "#FF9800"

WEIGHT_TIERS = {
    "heterog\n(10,5,0.5,0.5)": "10.00,5.00,0.50,0.50",
    "moderate\n(5,3,1,1)": "5.00,3.00,1.00,1.00",
    "flat\n(3,3,3,3)": "3.00,3.00,3.00,3.00",
    "equal\n(1,1,1,1)": "1.00,1.00,1.00,1.00",
}

LRS = ["0.0001", "0.0003", "0.001"]

# (slug, ent_sched, ent_coef): ent_sched="" means fixed; ent_coef=None means any
ENT_CONFIGS = [
    ("cosine", "0.1:0.01", None),
    ("ent=0.0", "", "0.0"),
    ("ent=0.01", "", "0.01"),
]

ALGOS = [
    ("torchrl-pcz-ppo-running", "PCZ-PPO", PCZ_COLOR),
    ("torchrl-ppo", "PPO", PPO_COLOR),
]


def _best_cell(
    rows: list[dict],
    algo: str,
    w_prefix: str,
    min_n: int = 2,
) -> dict:
    """Find the best (lr, ent_config) cell for algo at the given weight tier.

    Returns {"mean", "std", "n", "lr", "ent", "n_cells"} or None if no data.
    """
    seeds_allowed = {"42", "43", "44"}
    by_cell: dict[tuple, dict[str, float]] = {}

    for r in rows:
        if r.get("algorithm") != algo:
            continue
        if r.get("env") != "lunarlander":
            continue
        if r.get("total_timesteps") != "500000":
            continue
        if r.get("seed") not in seeds_allowed:
            continue
        if not r.get("eval_mean"):
            continue
        if not r.get("component_weights", "").startswith(w_prefix):
            continue

        lr = r.get("learning_rate", "")
        ent_sched = r.get("ent_coef_schedule", "")
        ent_coef = r.get("ent_coef", "")

        # Match ent config
        matched_ent = None
        for slug, req_sched, req_coef in ENT_CONFIGS:
            if req_sched:
                if ent_sched == req_sched:
                    matched_ent = slug
                    break
            else:
                if not ent_sched and (req_coef is None or ent_coef == req_coef):
                    matched_ent = slug
                    break
        if matched_ent is None:
            continue

        cell_key = (lr, matched_ent)
        if cell_key not in by_cell:
            by_cell[cell_key] = {}

        seed = r["seed"]
        date = r.get("date", "")
        if seed not in by_cell[cell_key] or date > by_cell[cell_key].get(f"_d_{seed}", ""):
            by_cell[cell_key][seed] = float(r["eval_mean"])
            by_cell[cell_key][f"_d_{seed}"] = date

    # Collect valid cells (n >= min_n)
    valid = {}
    for (lr, ent), sd in by_cell.items():
        vals = [v for k, v in sd.items() if not k.startswith("_d_")]
        if len(vals) >= min_n:
            valid[(lr, ent)] = vals

    if not valid:
        return None

    # Pick cell with highest mean
    best_key = max(valid, key=lambda k: np.mean(valid[k]))
    best_vals = valid[best_key]
    arr = np.asarray(best_vals, dtype=float)
    ddof = 1 if len(arr) > 1 else 0

    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=ddof)),
        "n": len(arr),
        "lr": best_key[0],
        "ent": best_key[1],
        "n_cells": len(valid),
    }


def main():
    rows = load_results()

    tier_labels = list(WEIGHT_TIERS.keys())
    n_tiers = len(tier_labels)

    _fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(n_tiers)
    width = 0.35

    for i, (algo, label, color) in enumerate(ALGOS):
        offset = (i - 0.5) * width
        for j, (_tier_label, w_prefix) in enumerate(WEIGHT_TIERS.items()):
            cell = _best_cell(rows, algo, w_prefix)
            if cell is None:
                ax.bar(x[j] + offset, 0, width=width, color=color, alpha=0.4, edgecolor="black", linewidth=0.5)
                ax.text(x[j] + offset, 5, "no data", ha="center", va="bottom", fontsize=7, color="gray")
                continue

            ax.bar(
                x[j] + offset,
                cell["mean"],
                yerr=cell["std"],
                width=width,
                color=color,
                alpha=0.85,
                capsize=4,
                edgecolor="black",
                linewidth=0.6,
                label=label if j == 0 else None,
            )

            # Annotate with best HP cell
            lr_label = {"0.0001": "1e-4", "0.0003": "3e-4", "0.001": "1e-3"}.get(cell["lr"], cell["lr"])
            annot = f"${cell['mean']:+.0f}\\pm{cell['std']:.0f}$\n$n={cell['n']}$\n{lr_label},{cell['ent']}"
            ypos = cell["mean"] + cell["std"] + 10
            ax.text(x[j] + offset, max(ypos, 15), annot, ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels, fontsize=9)
    ax.set_ylabel("Best-cell Mean Reward (±1 SD)", fontsize=11)
    ax.set_title(
        "Peak-vs-Peak Hyperparameter Fairness — LunarLander $K{=}4$, 500k steps\n"
        "Best cell per weight tier (over lr $\\in\\{1\\mathrm{e}{-}4, 3\\mathrm{e}{-}4, 1\\mathrm{e}{-}3\\}$ "
        "$\\times$ ent $\\in\\{$cosine, 0.0, 0.01$\\}$)",
        fontsize=10,
        fontweight="bold",
    )
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.grid(True, alpha=0.2, axis="y")
    ax.legend(fontsize=10, loc="upper right")

    all_tops, all_bots = [], []
    for algo, _, _ in ALGOS:
        for _, w_prefix in WEIGHT_TIERS.items():
            cell = _best_cell(rows, algo, w_prefix)
            if cell:
                all_tops.append(cell["mean"] + cell["std"])
                all_bots.append(cell["mean"] - cell["std"])
    if all_tops:
        # ylim_bot must include the LOWEST error-bar bottom (mean-std), not just
        # the lowest bar top — otherwise error bars dipping below zero get
        # clipped at the panel edge (the negative-bar collapse case).
        ax.set_ylim(
            bottom=min(0, min(all_bots) - 20),
            top=max(all_tops) + 80,
        )

    out = Path(__file__).parent / "fig_hp_peak.pdf"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
