"""Adversarial fixture: a figure script whose annotation extends past the
axis ylim, used to verify that ``lint_figure_geometry.py`` flags the
violation.

This file is intentionally bad.  The geometric lint should report:

    axes[0] text-in-bounds: text '...' at (..., ...) outside data bounds ...

If the lint stops catching this, the lint has regressed.

Not a real figure — not registered in paper_build's INPUTS-discovery.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUTS = []  # adversarial fixture; no real data dependencies
UNITS = "eval-mean-final"
OUTPUTS = ["fig_bad_geometry.pdf"]


def main():
    _fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar([0, 1, 2], [10, 20, 30], color="C0")
    ax.set_ylim(0, 35)
    ax.set_xlim(-0.5, 2.5)
    # Intentional violation: annotation at y=100, far above ylim_top=35.
    ax.text(1, 100, "WAY ABOVE THE PANEL", ha="center")
    out = Path(__file__).with_suffix(".pdf")
    plt.savefig(out)
    # Intentionally do NOT close the figure --- the test introspects the
    # live Figure via plt.get_fignums() after main() returns.


if __name__ == "__main__":
    main()
