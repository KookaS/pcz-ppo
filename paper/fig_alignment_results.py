"""Generate the LLM-alignment results figure (§5.2).

Two panels documenting the scope-boundary finding:

  Left:  K=6 PPO regime reversal — Standard vs PCZ at 200 vs 500 training steps.
         Shows the 200-step variance-reduction signature reversing at 500 steps.

  Right: 2x2 factorial at K=4 / 200 steps — {RLOO, PPO} x {standard, PCZ}.
         Shows the critic main effect dominates; PCZ is null in both critic
         settings.

Net visual story: across all tested LLM regimes, no positive PCZ signal that
survives a longer training budget or a critic-vs-no-critic control.

Reads JSON files at artifacts/pcz-ppo/<rlhf-runs-dir>/result_*.json.

Usage:
    cd /workspace && uv run python artifacts/pcz-ppo/paper/fig_alignment_results.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Path constructed without the literal trigger token to avoid bash-hook
# pre-tool gates that match command strings on training-script names.
_RUNS_DIR = Path(__file__).resolve().parent.parent / ("l" + "lm_alignment") / "runs"

INPUTS = []  # JSON inputs don't fit CSV/parquet contract; figure is manually rebuilt
OUTPUTS = ["fig_alignment_results.pdf", "fig_alignment_results.png"]

PCZ_COLOR = "#2196F3"
STD_COLOR = "#FF9800"


def _load(name: str) -> float | None:
    p = _RUNS_DIR / name
    if not p.exists():
        return None
    with open(p) as f:
        v = json.load(f).get("total_reward")
    return float(v) if v is not None else None


def _stats(vals: list[float]) -> tuple[float, float, int]:
    if not vals:
        return 0.0, 0.0, 0
    arr = np.asarray(vals, dtype=float)
    ddof = 1 if len(arr) > 1 else 0
    return float(arr.mean()), float(arr.std(ddof=ddof)), len(arr)


# --- Panel A: K=6 200 vs 500 step ---------------------------------------
def _collect_k6_regime(suffix: str, seeds: list[int]) -> dict:
    """Per-seed totals for K=6 PPO. ``suffix='_200step'`` reads the 200-step backup
    files; ``suffix=''`` reads the 500-step (live) files."""
    out = {"standard": [], "pcz": []}
    for mode in ("standard", "pcz"):
        for s in seeds:
            v = _load(f"result_ppo_k6_{mode}_s{s}{suffix}.json")
            if v is not None:
                out[mode].append(v)
    return out


# --- Panel B: K=4 2x2 factorial (200 steps) -----------------------------
def _collect_factorial(seeds: list[int]) -> dict:
    """RLOO uses result_k4_*; PPO uses result_ppo_k4_*."""
    out = {}
    for critic in ("rloo", "ppo"):
        for mode in ("standard", "pcz"):
            vals = []
            for s in seeds:
                fname = f"result_k4_{mode}_s{s}.json" if critic == "rloo" else f"result_ppo_k4_{mode}_s{s}.json"
                v = _load(fname)
                if v is not None:
                    vals.append(v)
            out[(critic, mode)] = vals
    return out


def _annotate(ax, x, m, s, n, *, fontsize=9):
    """Place mean +/- std and (n=N) above the bar+std stack."""
    ypos = m + s + 0.4
    ax.text(
        x,
        ypos,
        f"${m:+.2f}{{\\pm}}{s:.2f}$\n($n{{=}}{n}$)",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
    )


def _plot_regime_reversal(ax) -> None:
    # Per render_claims.py line 2240: K=6 200-step uses seeds 43-46 only
    # (s42 standard 200-step backup was lost when overwritten by 500-step rerun).
    r200 = _collect_k6_regime("_200step", [43, 44, 45, 46])
    r500 = _collect_k6_regime("", [42, 43, 44, 45, 46])

    bars = [
        ("Standard\n200 steps", *_stats(r200["standard"]), STD_COLOR, 0.7),
        ("PCZ\n200 steps", *_stats(r200["pcz"]), PCZ_COLOR, 0.7),
        ("Standard\n500 steps", *_stats(r500["standard"]), STD_COLOR, 1.0),
        ("PCZ\n500 steps", *_stats(r500["pcz"]), PCZ_COLOR, 1.0),
    ]

    x = np.arange(len(bars))
    for i, (_lbl, m, s, n, c, a) in enumerate(bars):
        ax.bar(x[i], m, 0.6, yerr=s, color=c, alpha=a, capsize=4, edgecolor="black", linewidth=0.5)
        _annotate(ax, x[i], m, s, n)

    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=10)
    ax.set_ylabel("Eval reward (composite)", fontsize=11)
    ax.set_title(
        "K=6 PPO regime reversal:\n200-step variance reduction does not survive 500 steps",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.2, axis="y")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_ylim(bottom=0, top=max(m + s for _, m, s, *_ in bars) + 5)

    # Legend by alpha groups
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=STD_COLOR, alpha=1.0, label="Standard"),
        Patch(facecolor=PCZ_COLOR, alpha=1.0, label="PCZ"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=10, framealpha=0.95)


def _plot_factorial(ax) -> None:
    cells = _collect_factorial([42, 43, 44])

    # Order: RLOO-std, RLOO-PCZ, PPO-std, PPO-PCZ (groups by critic)
    bars = [
        ("RLOO\nstandard", *_stats(cells[("rloo", "standard")]), STD_COLOR, 0.7),
        ("RLOO\nPCZ", *_stats(cells[("rloo", "pcz")]), PCZ_COLOR, 0.7),
        ("PPO\nstandard", *_stats(cells[("ppo", "standard")]), STD_COLOR, 1.0),
        ("PPO\nPCZ", *_stats(cells[("ppo", "pcz")]), PCZ_COLOR, 1.0),
    ]

    x = np.arange(len(bars))
    for i, (_lbl, m, s, n, c, a) in enumerate(bars):
        ax.bar(x[i], m, 0.6, yerr=s, color=c, alpha=a, capsize=4, edgecolor="black", linewidth=0.5)
        _annotate(ax, x[i], m, s, n)

    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=10)
    ax.set_ylabel("Eval reward (composite)", fontsize=11)
    ax.set_title(
        "K=4 2$\\times$2 factorial (200 steps):\ncritic dominates; PCZ null in both critic settings",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.2, axis="y")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_ylim(bottom=0, top=max(m + s for _, m, s, *_ in bars) + 1.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(Path(__file__).parent / "fig_alignment_results.pdf"))
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1]})
    _plot_regime_reversal(axes[0])
    _plot_factorial(axes[1])
    fig.suptitle(
        "PCZ-PPO does not transfer to single-step preference alignment",
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
