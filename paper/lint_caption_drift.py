"""Flag table caption seed-count drift vs fragment data.

Problem
-------
A table caption says ``3--5 seeds`` but the table body pulls
``\\cnum{foo_seeds}`` fragments that now resolve to 10.  Neither
``render_claims --check`` nor ``lint_hardcoded_numbers`` catches this:
captions are free-text prose, and ``3--5 seeds`` doesn't match the
existing lint's mean-pm-std / ratio patterns.

Detection
---------
1. Walk ``pcz_ppo.tex`` and split into ``\\begin{table}...\\end{table}`` blocks.
2. For each block, extract the caption text and the body.
3. Parse caption for seed claims:
     - ``N--M seeds``        -> range [N, M]
     - ``N seeds``           -> exact {N}
     - ``n{=}N per cell``    -> exact {N}
4. Collect every ``\\cnum{<name>_seeds}`` in the body.
5. Resolve each fragment by reading ``generated/<name>_seeds.tex``.
6. Fail if any fragment value is outside the caption's declared claim.

The caption lint complements the fragment renderer (which proves
fragments match data) and the hardcoded-numbers lint (which flags
hand-typed stats).  Together they close the mechanical-drift loop.

Usage
-----
    uv run python artifacts/pcz-ppo/paper/lint_caption_drift.py

Exit code 1 on any violation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PAPER = Path(__file__).parent / "pcz_ppo.tex"
GENERATED = Path(__file__).parent / "generated"

TABLE_BLOCK_RE = re.compile(
    r"\\begin\{table\}.*?\\end\{table\}",
    re.DOTALL,
)
CAPTION_RE = re.compile(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
CNUM_SEEDS_RE = re.compile(r"\\cnum\{([a-zA-Z0-9_]+_seeds)\}")

# Seed-claim patterns in caption text.  Order matters: the range pattern
# must be tried before the bare-int pattern, else "3--5 seeds" is parsed
# as the singleton "3".
RANGE_RE = re.compile(r"(\d+)\s*(?:--|-|\\textendash\s*|\u2013)\s*(\d+)\s*seeds?")
EXACT_SEEDS_RE = re.compile(r"(?<!\d)(\d+)\s*seeds?\b")
EXACT_N_EQ_RE = re.compile(r"n\s*\{?=\}?\s*(\d+)")

# Quantifier phrases that make a seed claim "per-cell" / "each" — when
# present, the caption value must match *every* body fragment; when
# absent we allow the caption's range to be a *cover* of body values
# (some rows may have fewer seeds).
PER_CELL_HINT_RE = re.compile(r"\bper\s+cell\b|\beach\s*\)?\s*$", re.IGNORECASE)


def _read_fragment_int(name: str) -> int | None:
    path = GENERATED / f"{name}.tex"
    if not path.exists():
        return None
    s = path.read_text().strip()
    try:
        return int(s)
    except ValueError:
        return None


def _parse_caption_claims(caption: str) -> list[tuple[str, tuple[int, ...]]]:
    """Return list of (kind, values) tuples.

    kind is one of "range", "exact", "n_eq", "fragment"; values is a tuple
    of ints (length 2 for range, length 1 for others).

    Caption-side \\cnum{*_seeds} references are resolved to their current
    integer value and added as "fragment" claims.  This lets a self-
    consistent caption like "$n{=}\\cnum{k4_pcz_seeds}$" cover a body
    fragment of the same name, even if no concrete integer is typed.
    """
    claims: list[tuple[str, tuple[int, ...]]] = []
    # Range first — consumes "3--5 seeds" so EXACT_SEEDS_RE won't re-match "5".
    masked = caption
    for m in RANGE_RE.finditer(caption):
        lo, hi = int(m.group(1)), int(m.group(2))
        claims.append(("range", (lo, hi)))
        masked = masked.replace(m.group(0), " ")
    # Mask out \cnum{...} before scanning bare-int patterns: otherwise
    # "n{=}\cnum{k4_pcz_seeds}" trips EXACT_N_EQ_RE on the "4" in "k4".
    masked_no_cnum = re.sub(r"\\cnum\{[^}]+\}", " ", masked)
    for m in EXACT_SEEDS_RE.finditer(masked_no_cnum):
        claims.append(("exact", (int(m.group(1)),)))
    for m in EXACT_N_EQ_RE.finditer(masked_no_cnum):
        claims.append(("n_eq", (int(m.group(1)),)))
    # Fragment-backed claims: resolve each \cnum{*_seeds} in caption.
    for m in CNUM_SEEDS_RE.finditer(caption):
        v = _read_fragment_int(m.group(1))
        if v is not None:
            claims.append(("fragment", (v,)))
    return claims


def _claim_covers(claim: tuple[str, tuple[int, ...]], value: int) -> bool:
    kind, vs = claim
    if kind == "range":
        lo, hi = vs
        return lo <= value <= hi
    return value == vs[0]


def _claim_describe(claim: tuple[str, tuple[int, ...]]) -> str:
    kind, vs = claim
    if kind == "range":
        return f"{vs[0]}--{vs[1]} seeds"
    if kind == "n_eq":
        return f"n={vs[0]}"
    if kind == "fragment":
        return f"\\cnum={vs[0]}"
    return f"{vs[0]} seeds"


def scan(paper: Path) -> list[tuple[int, str, str]]:
    """Return list of (table_start_lineno, fragment_name, message)."""
    text = paper.read_text()
    violations: list[tuple[int, str, str]] = []
    # Build cumulative line-offset index so we can report lineno per block.
    line_starts = [0]
    for line in text.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))

    def offset_to_lineno(ofs: int) -> int:
        # binary search would be faster but tables are few
        for i, start in enumerate(line_starts):
            if start > ofs:
                return i
        return len(line_starts)

    for block_match in TABLE_BLOCK_RE.finditer(text):
        block = block_match.group(0)
        lineno = offset_to_lineno(block_match.start())
        cap = CAPTION_RE.search(block)
        if not cap:
            continue
        caption = cap.group(1)
        claims = _parse_caption_claims(caption)
        if not claims:
            continue
        seed_refs = CNUM_SEEDS_RE.findall(block)
        if not seed_refs:
            continue
        body_values: dict[str, int] = {}
        for name in seed_refs:
            v = _read_fragment_int(name)
            if v is not None:
                body_values[name] = v
        if not body_values:
            continue
        # Violation iff NO caption claim covers a given body value.
        for name, value in body_values.items():
            if not any(_claim_covers(c, value) for c in claims):
                claim_descs = ", ".join(_claim_describe(c) for c in claims)
                violations.append(
                    (
                        lineno,
                        name,
                        f"caption claims [{claim_descs}] but \\cnum{{{name}}} resolves to {value}",
                    )
                )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paper", default=str(PAPER), help="Paper .tex path")
    args = parser.parse_args()

    violations = scan(Path(args.paper))
    if not violations:
        print("lint_caption_drift: OK — table captions match fragment seed counts")
        return 0
    print(f"lint_caption_drift: FAIL — {len(violations)} caption/body mismatch(es)\n")
    for lineno, _name, msg in violations:
        print(f"  L{lineno:4d}  {msg}")
    print("\nFix: update the caption's seed claim, or wire the caption number")
    print("to the fragment (e.g. replace '3--5 seeds' with 'up to \\cnum{foo_seeds} seeds').")
    return 1


if __name__ == "__main__":
    sys.exit(main())
