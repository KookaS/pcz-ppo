"""Entropy-coupling sweep figure: PCZ vs PPO across fixed entropy values.

Visualises Table~\\ref{tab:entropy_sweep} (LunarLander K=4, 500k, lr=3e-4,
canonical weights, no cosine schedule).  Demonstrates PCZ-PPO's wider
healthy-entropy range (0.0--0.05) versus PPO's narrower one (0.0--0.01).

Filters mirror render_claims._emit_rr6_entropy_sweep exactly so figure
points equal the corresponding rr6_*_stat fragments.
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
from fig_data import load_results

INPUTS = ["../data/results.csv"]
UNITS = "eval-mean-final"  # Layer 4: declared metric (rollout/eval/ratio/etc.)
OUTPUTS = ["fig_entropy_sweep.pdf", "fig_entropy_sweep.png"]

PCZ_COLOR = "#2196F3"
PPO_COLOR = "#FF9800"

ENT_VALUES = ["0.0", "0.01", "0.03", "0.05", "0.1"]


def _grab(rows: list[dict], algo: str, ent: str) -> tuple[float, float, int]:
    matching = [
        r
        for r in rows
        if r.get("algorithm") == algo
        and r.get("env") == "lunarlander"
        and r.get("total_timesteps") == "500000"
        and r.get("learning_rate") == "0.0003"
        and r.get("ent_coef") == ent
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
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "fig_entropy_sweep.pdf"),
    )
    args = parser.parse_args()

    rows = load_results()

    pcz_means, pcz_stds, pcz_ns = [], [], []
    ppo_means, ppo_stds, ppo_ns = [], [], []
    for ent in ENT_VALUES:
        m, s, n = _grab(rows, "torchrl-pcz-ppo-running", ent)
        pcz_means.append(m)
        pcz_stds.append(s)
        pcz_ns.append(n)
        m, s, n = _grab(rows, "torchrl-ppo", ent)
        ppo_means.append(m)
        ppo_stds.append(s)
        ppo_ns.append(n)
        print(
            f"  ent={ent}: PCZ {pcz_means[-1]:+.1f}+/-{pcz_stds[-1]:.1f} (n={pcz_ns[-1]}) | PPO {ppo_means[-1]:+.1f}+/-{ppo_stds[-1]:.1f} (n={ppo_ns[-1]})"
        )

    _, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(ENT_VALUES))
    w = 0.38

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

    ymax = max(max(pcz_means) + max(pcz_stds), max(ppo_means) + max(ppo_stds))
    ymin = min(min(pcz_means) - max(pcz_stds), min(ppo_means) - max(ppo_stds), 0)
    pad = (ymax - ymin) * 0.05

    for i in range(len(ENT_VALUES)):
        for offset, m, s, n in [
            (-w / 2, pcz_means[i], pcz_stds[i], pcz_ns[i]),
            (+w / 2, ppo_means[i], ppo_stds[i], ppo_ns[i]),
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
                fontsize=8,
                fontweight="bold",
            )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\mathrm{{ent}}{{=}}{e}$" for e in ENT_VALUES], fontsize=10)
    ax.set_ylabel("Eval Mean Reward", fontsize=11)
    ax.set_xlabel("Fixed entropy coefficient (no cosine schedule)", fontsize=11)
    ax.set_title(
        "Entropy-coupling sweep: PCZ-PPO has a wider healthy-entropy range\n"
        "(LunarLander $K{=}4$, 500k steps, $\\mathrm{lr}{=}3\\mathrm{e}{-}4$, canonical weights)",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.2, axis="y")
    # Generous symmetric padding (3x pad) so the two-line annotations
    # ("$+m \pm s$\n($n=...$)") at the most-extreme bars fit fully inside
    # the panel — at low pad multipliers the (n=...) line of the most
    # negative bar (e.g. ent=0.10) bleeds into the bottom bounding box.
    ax.set_ylim(bottom=ymin - pad * 3.0, top=ymax + pad * 3.0)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.savefig(args.output.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
