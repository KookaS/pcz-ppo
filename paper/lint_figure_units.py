"""Figure-integrity Layer 4: Mixed-unit detector.

Each ``fig_*.py`` script should declare a module-level ``UNITS`` constant
identifying which reward metric (per-step rollout, episodic eval, etc.) the
figure plots.  The lint asserts:

1. Every ``fig_*.py`` declares ``UNITS``.
2. If the figure mixes metrics (``UNITS`` value contains a comma), the
   script must also have ``MIXED_UNITS_ACKNOWLEDGED = True`` and the
   corresponding figure caption in ``pcz_ppo.tex`` must contain the literal
   text ``different units`` or ``different metrics`` so the reader knows.
3. The caption metric language is consistent with the declared units (a
   ``UNITS = "eval-episodic"`` figure should not have a caption talking
   about "rollout"; a ``UNITS = "rollout-per-step"`` figure should not
   have a caption claiming the y-axis is "eval reward").

Recognised unit tokens (extend as needed):

* ``rollout-per-step`` — TorchRL's ``rollout/reward_mean`` (per-step,
  stochastic-policy training signal).
* ``eval-episodic`` — ``eval/mean_reward`` (per-episode total,
  deterministic policy).
* ``eval-mean-final`` — final eval mean (single point per run, used in
  bar charts).
* ``failure-rate`` — % of seeds with eval < threshold (Table 10 style).
* ``ratio`` — dimensionless ratio (PCZ/PPO mean ratio, variance ratio).
* ``normalized`` — per-env min-max-normalized reward.
* ``hp-grid`` — best-of-cell mean over an HP sweep.
* ``llm-composite`` — LLM-domain composite reward.

This is the lightest of the four layers (~15 lines per figure to add the
``UNITS`` declaration); the value is in catching the rollout-vs-eval class
of confusion that bit Figure 9 mid-conversation.

Usage::

    uv run python artifacts/pcz-ppo/paper/lint_figure_units.py
    # exit 0 = OK, exit 1 = violations
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

PAPER_TEX = Path(__file__).parent / "pcz_ppo.tex"
PAPER_DIR = Path(__file__).parent

NON_FIGURE_SCRIPTS = {"fig_data.py"}

# Recognised unit tokens (must match exactly, comma-separated when mixed).
KNOWN_UNITS = {
    "rollout-per-step",
    "eval-episodic",
    "eval-mean-final",
    "failure-rate",
    "ratio",
    "normalized",
    "hp-grid",
    "llm-composite",
}

# Caption keywords that suggest a specific metric. Used to cross-check that
# the caption's language matches the declared UNITS.
CAPTION_KEYWORDS = {
    "rollout-per-step": ["rollout", "per-step"],
    "eval-episodic": ["eval reward", "evaluation reward", "episodic"],
    "eval-mean-final": ["eval mean", "final eval"],
    "failure-rate": ["failure rate", "catastrophic"],
    "ratio": ["ratio"],
    "normalized": ["normalized", "min-max"],
    "hp-grid": ["best cell", "best-cell", "best of"],
    "llm-composite": ["composite reward", "preference reward"],
}


def _figure_scripts() -> list[Path]:
    return sorted(p for p in PAPER_DIR.glob("fig_*.py") if p.name not in NON_FIGURE_SCRIPTS)


def _parse_units_decl(script: Path) -> tuple[list[str] | None, bool]:
    """Parse ``UNITS`` and ``MIXED_UNITS_ACKNOWLEDGED`` from the script's AST.

    Returns ``(units_list, ack_flag)``. ``units_list`` is None if not declared.
    """
    tree = ast.parse(script.read_text())
    units: list[str] | None = None
    ack = False
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name):
                if tgt.id == "UNITS" and isinstance(node.value, ast.Constant):
                    raw = node.value.value
                    if isinstance(raw, str):
                        units = [u.strip() for u in raw.split(",")]
                elif tgt.id == "MIXED_UNITS_ACKNOWLEDGED" and isinstance(node.value, ast.Constant):
                    if node.value.value is True:
                        ack = True
    return units, ack


def _figure_label(script: Path) -> str | None:
    """Heuristic: look for ``\\label{fig:...}`` in pcz_ppo.tex referencing the script's
    PDF output. Returns the first matching label or None."""
    pdf_name = script.stem + ".pdf"
    tex = PAPER_TEX.read_text()
    # Find the \begin{figure}...\end{figure} block containing this PDF
    for m in re.finditer(r"\\begin\{figure\}.*?\\end\{figure\}", tex, re.DOTALL):
        block = m.group(0)
        if pdf_name in block or script.stem in block:
            label_match = re.search(r"\\label\{(fig:[^}]+)\}", block)
            if label_match:
                return label_match.group(1)
    return None


def _caption_for_label(label: str) -> str:
    """Extract the caption text for a given figure label from pcz_ppo.tex."""
    tex = PAPER_TEX.read_text()
    for m in re.finditer(r"\\begin\{figure\}.*?\\end\{figure\}", tex, re.DOTALL):
        block = m.group(0)
        if f"\\label{{{label}}}" not in block:
            continue
        cap_start = block.find("\\caption{")
        if cap_start == -1:
            return ""
        i = cap_start + len("\\caption{")
        depth = 1
        while i < len(block) and depth > 0:
            if block[i] == "{":
                depth += 1
            elif block[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        return block[cap_start + len("\\caption{") : i]
    return ""


def lint() -> int:
    scripts = _figure_scripts()
    if not scripts:
        print("lint_figure_units: WARNING — no fig_*.py scripts found")
        return 0

    violations: list[str] = []
    for script in scripts:
        units, ack = _parse_units_decl(script)
        if units is None:
            violations.append(f'{script.name}: missing UNITS declaration (add `UNITS = "..."` at module level)')
            continue
        unknown = [u for u in units if u not in KNOWN_UNITS]
        if unknown:
            violations.append(
                f"{script.name}: UNITS contains unknown token(s) {unknown}; "
                f"use one of {sorted(KNOWN_UNITS)} or extend KNOWN_UNITS in lint_figure_units.py"
            )
            continue
        is_mixed = len(units) > 1
        if is_mixed and not ack:
            violations.append(
                f"{script.name}: UNITS={units} mixes metrics, but MIXED_UNITS_ACKNOWLEDGED is not True. "
                f"Either consolidate to one unit or add `MIXED_UNITS_ACKNOWLEDGED = True` and "
                f"explain the mixing in the caption."
            )
            continue
        # Cross-check caption keywords
        label = _figure_label(script)
        if label is None:
            # Some scripts may not have a corresponding figure in pcz_ppo.tex
            # (e.g., if they were generated but not yet included). Skip.
            continue
        caption = _caption_for_label(label).lower()
        if is_mixed:
            if "different unit" not in caption and "different metric" not in caption:
                violations.append(
                    f"{script.name}: mixed UNITS={units} requires caption to contain "
                    f'"different units" or "different metrics" (currently in caption: nothing).'
                )
            continue

        # Single-unit: caption should mention an appropriate keyword if any
        # are listed for this unit. (This is a soft hint; we don't fail
        # if the caption uses synonyms, only if the caption uses keywords
        # for a *different* unit.)  Use word-boundary matching to avoid
        # spurious hits on substrings (e.g. "ratio" inside "configuration").
        def _word_in(kw: str, text: str) -> bool:
            return re.search(r"\b" + re.escape(kw) + r"\b", text) is not None

        for other_unit, keywords in CAPTION_KEYWORDS.items():
            if other_unit == units[0]:
                continue
            for kw in keywords:
                if _word_in(kw, caption):
                    own_kws = CAPTION_KEYWORDS.get(units[0], [])
                    if not any(_word_in(own_kw, caption) for own_kw in own_kws):
                        violations.append(
                            f"{script.name}: caption mentions '{kw}' (suggests {other_unit}) but "
                            f"UNITS={units}.  If the figure mixes metrics, set MIXED_UNITS_ACKNOWLEDGED=True."
                        )
                    break

    if not violations:
        print(f"lint_figure_units: OK — {len(scripts)} figures, all UNITS consistent with captions")
        return 0
    print(f"lint_figure_units: FAIL — {len(violations)} violation(s):\n")
    for v in violations:
        print(f"  - {v}")
    print()
    print('Fix: add `UNITS = "<token>"` at module level of each fig_*.py.')
    print(f"Recognised tokens: {sorted(KNOWN_UNITS)}")
    print("Mixed metrics (e.g. rollout + eval on same panel) require")
    print("`MIXED_UNITS_ACKNOWLEDGED = True` AND caption text containing")
    print('"different units" or "different metrics".')
    return 1


if __name__ == "__main__":
    sys.exit(lint())
