"""Figure-integrity Layer 1 safety net: winner-fragment coverage lint.

Catches the case where the paper references a Layer-1 data-derived fragment
(``\\cnum{X_winner}``, ``\\cnum{X_summary}``, ``\\cnum{X_direction}``,
``\\cnum{X_winner_verb}``, ``\\cnum{X_loser}``) but the prefix ``X`` does
not have a paired (PCZ vs PPO) emission in ``render_claims.py``.

Without this lint, missing winner fragments fail only at LaTeX compile
time (undefined ``\\cnum``), which is too late --- pre-commit should
catch it. With this lint, any drift between caption-text and
fragment-emitter is flagged immediately.

The lint is fast (parses the .tex file + queries the in-memory registry).

Usage::

    uv run python artifacts/pcz-ppo/paper/lint_winner_coverage.py
    # exit 0 = OK, exit 1 = missing fragments
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER_TEX = Path(__file__).parent / "pcz_ppo.tex"
PAPER_DIR = Path(__file__).parent

# The set of suffixes Layer 1 emits for every paired comparison.
WINNER_SUFFIXES = ("winner", "loser", "winner_verb", "direction", "summary")


def _referenced_winner_fragments(tex: str) -> set[str]:
    """Find every \\cnum{X_<suffix>} reference where suffix is a Layer-1 suffix."""
    suffix_re = "|".join(re.escape(s) for s in WINNER_SUFFIXES)
    pattern = re.compile(r"\\cnum\{([^}]+_(?:" + suffix_re + r"))\}")
    return set(pattern.findall(tex))


def _registry_fragment_names() -> set[str]:
    """Build the registry and return the set of all fragment names it produces.

    Imports ``render_claims`` and ``fig_data`` from the paper directory.
    Returns the set of fragment names that ``build_registry()`` would emit
    (independent of the ``used_filter`` --- we want to know what's
    *available*, not what's currently materialised).
    """
    sys.path.insert(0, str(PAPER_DIR))
    import fig_data
    import render_claims

    rows = fig_data.load_results()
    # Pass no filter so all registered fragments come back.
    reg = render_claims.build_registry(rows, used_filter=None)
    return set(reg.as_dict().keys())


def lint() -> int:
    if not PAPER_TEX.exists():
        print(f"lint_winner_coverage: ERROR — {PAPER_TEX} not found", file=sys.stderr)
        return 2
    tex = PAPER_TEX.read_text()
    referenced = _referenced_winner_fragments(tex)
    if not referenced:
        print("lint_winner_coverage: OK — no Layer-1 winner fragments referenced (no risk)")
        return 0
    available = _registry_fragment_names()
    missing = sorted(r for r in referenced if r not in available)
    if not missing:
        print(f"lint_winner_coverage: OK — all {len(referenced)} winner-fragment references resolve")
        return 0
    print(f"lint_winner_coverage: FAIL — {len(missing)} winner-fragment reference(s) have no emitter\n")
    print("Each \\cnum{X_winner|loser|direction|summary|winner_verb} must correspond")
    print("to a paired (PCZ, PPO) emit in render_claims.py.  Missing emits are")
    print("typically prefixes that use _emit_single() per algorithm rather than")
    print("_emit_pcz_ppo_pair() for the pair.  Fix by either:")
    print("  (a) adding a _emit_pcz_ppo_pair call for the prefix, OR")
    print("  (b) calling _emit_winner_claims explicitly with the relevant data.\n")
    for m in missing:
        # Suggest the prefix and suffix
        suffix = m.rsplit("_", 1)[-1]
        for ws in WINNER_SUFFIXES:
            if m.endswith("_" + ws):
                suffix = ws
                prefix = m.removesuffix("_" + ws)
                break
        else:
            prefix = m
        print(f"  - {m!r:50s}  (prefix='{prefix}', suffix='_{suffix}')")
    return 1


if __name__ == "__main__":
    sys.exit(lint())
