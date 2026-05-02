"""Figure-integrity Layer 3: Geometric sanity checks.

Imports each ``fig_*.py`` module, runs its ``main()`` to render the figure,
then inspects every ``Axes`` for:

1. **Text-in-bounds**: every annotation drawn in ``ax.transData`` coordinates
   has its (x, y) position inside ``(xlim, ylim)``.  Catches the
   K=4-n=10-touching-boundary class of issue.

2. **Text-bbox-within-axis**: the rendered text bounding box does not extend
   significantly outside the axes' display bbox.  Catches annotations that
   bleed past the panel edge.

3. **Errorbar caps in ylim**: every errorbar's top/bottom cap is within
   ``ylim``.  Catches "std out of bounds".

Each violation is reported per-figure-per-axes-per-check.

Usage::

    uv run python artifacts/pcz-ppo/paper/lint_figure_geometry.py
    # exit 0 = OK, exit 1 = violations found

The lint imports each ``fig_*.py`` and calls its ``main()``.  This requires
the script's ``main()`` to be safe under repeated invocation (no global
state pollution).  All current scripts use ``matplotlib.use("Agg")`` and
``plt.tight_layout()``+``plt.savefig()``, which is compatible.

See ``FIGURE_INTEGRITY.md`` for known false-positive scenarios and design
trade-offs.
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER_DIR = Path(__file__).parent

# Allow margin (in display points) for text bbox extending past the axis
# edge: small slop is normal because matplotlib auto-positions axis labels
# slightly outside the data area.  This tolerance is for *data-area* text,
# which should be inside the axis bbox by at least this margin.
BBOX_TOLERANCE_PIXELS = 2.0

# Files to skip — shared loaders, not figure scripts.
NON_FIGURE_SCRIPTS = {"fig_data.py"}


def _figure_scripts() -> list[Path]:
    """Discover figure scripts in PAPER_DIR (anything ``fig_*.py`` except shared)."""
    return sorted(p for p in PAPER_DIR.glob("fig_*.py") if p.name not in NON_FIGURE_SCRIPTS)


def _import_and_run(script: Path) -> list[plt.Figure]:
    """Import the script and call its ``main()``; return all live Figures.

    Forces ``sys.argv`` to ``[script_name]`` so argparse sees no args (uses
    defaults).  Captures stdout to keep lint output clean.  Patches
    ``plt.savefig`` and ``Figure.savefig`` to no-op so the lint does NOT
    overwrite committed PDF/PNG outputs (matplotlib PDF generation embeds
    timestamps, so even a content-identical re-render produces a
    different byte stream — see pre-commit "files modified by this hook"
    failure mode).  We render in-memory only.
    """
    module_name = script.stem
    spec = importlib.util.spec_from_file_location(module_name, script)
    module = importlib.util.module_from_spec(spec)
    saved_argv = sys.argv
    sys.argv = [str(script)]
    plt.close("all")
    # Patch savefig to no-op for the duration of this lint.  Capture and
    # restore the originals so subsequent code (e.g. tests, paper_build)
    # is unaffected.
    saved_plt_savefig = plt.savefig
    saved_fig_savefig = matplotlib.figure.Figure.savefig

    def _noop(*args, **kwargs):
        return None

    plt.savefig = _noop
    matplotlib.figure.Figure.savefig = _noop
    try:
        with redirect_stdout(io.StringIO()):
            spec.loader.exec_module(module)
            if hasattr(module, "main"):
                module.main()
    finally:
        sys.argv = saved_argv
        plt.savefig = saved_plt_savefig
        matplotlib.figure.Figure.savefig = saved_fig_savefig
    fignums = plt.get_fignums()
    figs = [plt.figure(n) for n in fignums]
    for fig in figs:
        fig.canvas.draw()
    return figs


def _check_text_in_bounds(ax) -> list[str]:
    """Return list of violation strings for ax.texts outside data bounds."""
    violations = []
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for txt in ax.texts:
        # Only check text in data coords (transData). Axis labels live in
        # axes-relative or figure-relative transforms and should not be
        # bounded by data limits.
        if txt.get_transform() != ax.transData:
            continue
        x, y = txt.get_position()
        out_x = not (xlim[0] <= x <= xlim[1])
        out_y = not (ylim[0] <= y <= ylim[1])
        if out_x or out_y:
            content = txt.get_text()
            short = content if len(content) <= 40 else content[:37] + "..."
            violations.append(f"text '{short}' at ({x:.2f}, {y:.2f}) outside data bounds x{xlim} y{ylim}")
    return violations


def _check_text_bbox_within_axis(ax) -> list[str]:
    """Return list of violations for text bboxes extending past axis bbox."""
    violations = []
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer=renderer)
    for txt in ax.texts:
        if txt.get_transform() != ax.transData:
            continue
        try:
            tbox = txt.get_window_extent(renderer=renderer)
        except RuntimeError:
            continue
        # Allow a small tolerance — fonts often slightly overlap edges.
        below = ax_bbox.y0 - tbox.y0
        above = tbox.y1 - ax_bbox.y1
        left = ax_bbox.x0 - tbox.x0
        right = tbox.x1 - ax_bbox.x1
        worst = max(below, above, left, right)
        if worst > BBOX_TOLERANCE_PIXELS:
            content = txt.get_text()
            short = content if len(content) <= 40 else content[:37] + "..."
            side = "below" if below == worst else "above" if above == worst else "left" if left == worst else "right"
            violations.append(f"text '{short}' bbox extends {worst:.1f}px past axis {side} edge")
    return violations


def _check_errorbar_caps(ax) -> list[str]:
    """Return list of violations for errorbar caps outside ylim.

    ErrorbarContainer.error_lines may be a tuple of Line2D objects OR a
    tuple containing a single LineCollection, depending on matplotlib
    version. Handles both.
    """
    import numpy as np

    violations = []
    ylim = ax.get_ylim()
    for container in ax.containers:
        if not isinstance(container, matplotlib.container.ErrorbarContainer):
            continue
        if len(container) < 3 or not container[2]:
            continue
        error_lines = container[2]
        ymin_obs: float | None = None
        ymax_obs: float | None = None
        for el in error_lines:
            if hasattr(el, "get_segments"):
                # LineCollection: list of Nx2 arrays
                for seg in el.get_segments():
                    arr = np.asarray(seg)
                    if arr.size == 0:
                        continue
                    yvals = arr[:, 1]
                    if yvals.size == 0:
                        continue
                    yi = float(yvals.min())
                    ya = float(yvals.max())
                    ymin_obs = yi if ymin_obs is None else min(ymin_obs, yi)
                    ymax_obs = ya if ymax_obs is None else max(ymax_obs, ya)
            elif hasattr(el, "get_ydata"):
                ydata = el.get_ydata()
                if len(ydata) == 0:
                    continue
                yi = float(min(ydata))
                ya = float(max(ydata))
                ymin_obs = yi if ymin_obs is None else min(ymin_obs, yi)
                ymax_obs = ya if ymax_obs is None else max(ymax_obs, ya)
        if ymin_obs is None or ymax_obs is None:
            continue
        if ymin_obs < ylim[0] - 1e-6 or ymax_obs > ylim[1] + 1e-6:
            violations.append(f"errorbar y∈[{ymin_obs:.2f}, {ymax_obs:.2f}] outside ylim {ylim}")
    return violations


def _check_axes(ax, ax_idx: int) -> list[str]:
    out: list[str] = []
    for v in _check_text_in_bounds(ax):
        out.append(f"axes[{ax_idx}] text-in-bounds: {v}")
    for v in _check_text_bbox_within_axis(ax):
        out.append(f"axes[{ax_idx}] text-bbox: {v}")
    for v in _check_errorbar_caps(ax):
        out.append(f"axes[{ax_idx}] errorbar: {v}")
    return out


def lint() -> int:
    """Run lint on all figure scripts.  Return 0 OK / 1 violations / 2 error."""
    cwd = os.getcwd()
    os.chdir(PAPER_DIR.parent.parent.parent)  # workspace root, in case scripts use relative paths
    try:
        scripts = _figure_scripts()
        if not scripts:
            print("lint_figure_geometry: WARNING — no fig_*.py scripts found")
            return 0
        all_violations: list[tuple[str, list[str]]] = []
        for script in scripts:
            try:
                figs = _import_and_run(script)
            except Exception as e:
                all_violations.append((script.name, [f"FAILED to render: {type(e).__name__}: {e}"]))
                continue
            v: list[str] = []
            for fig in figs:
                for i, ax in enumerate(fig.axes):
                    v.extend(_check_axes(ax, i))
            if v:
                all_violations.append((script.name, v))
            plt.close("all")
    finally:
        os.chdir(cwd)
    if not all_violations:
        print(f"lint_figure_geometry: OK — {len(scripts)} figures, no geometric violations")
        return 0
    n_total = sum(len(v) for _, v in all_violations)
    print(f"lint_figure_geometry: FAIL — {n_total} geometric violation(s) across {len(all_violations)} figure(s)\n")
    print("Each violation indicates a text annotation, errorbar cap, or label")
    print("that renders outside the axis bounds (the K=4-n=10-touching-boundary class).\n")
    print("Fix: increase ylim/xlim padding (use generous adaptive computation,")
    print("e.g. `top = max(bar+std)+15` or `ylim = (ymin-3*pad, ymax+3*pad)`).\n")
    for script, violations in all_violations:
        print(f"  {script}:")
        for v in violations:
            print(f"    - {v}")
        print()
    return 1


if __name__ == "__main__":
    sys.exit(lint())
