"""Flag hand-typed numerical claims in the paper LaTeX source.

Rationale
---------
``render_claims.py --check`` proves every ``\\cnum{...}`` fragment matches the
current data.  This linter solves the *other half*: numbers that were never
wired to a fragment, which the fragment-renderer cannot see.

Detection
---------
Scan ``pcz_ppo.tex`` (outside LaTeX comments) for tokens that look like
numerical claims.  Any match that isn't already inside an allow-listed
context (ALLOW_PATTERNS below) is a potential drift risk.

Patterns flagged:
  * ``<digits>.<digits> \\pm <digits>.<digits>`` — mean ± std not in \\cnum
  * ``<digits>.<digits>\\times`` — multiplicative ratio not in \\cnum
  * bare ``\\pm`` constructs in tables

Patterns explicitly NOT flagged (structural / hyperparameters / labels):
  * numbers inside ``\\cnum{...}`` (already data-driven)
  * hyperparameter prose in Experimental Setup (lr 3e-4, gamma 0.99, ...)
  * Cohen's d values, 95% CI intervals (future work: wire these too)
  * units ($p{=}0.047$, $K{=}8$, seed numbers 42-51)
  * numbers inside ``\\cite{...}`` / citation years

Usage
-----
    uv run python paper/lint_hardcoded_numbers.py

Exit code 1 if any unexpected hardcoded number found.

Use ``--allow <line:pattern>`` to add exceptions for historical/narrative
claims that are genuinely not tied to current data (e.g. "1.03×, 2 seeds"
describing a prior snapshot).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PAPER = Path(__file__).parent / "pcz_ppo.tex"

# Patterns to flag (must not appear hand-typed outside \cnum{})
FLAG_PATTERNS = [
    # "157.7 \pm 27.0" — reward stats
    (r"[-+]?\d+\.\d+\s*\\pm\s*\d+\.\d+", "mean±std"),
    # "1.41$\times$" or "1.41\times" — decimal ratio
    (r"\d+\.\d+\s*\\?\$?\\times\$?", "ratio×"),
    # Comparator + integer or decimal ratio (e.g. "roughly $3\times$", "more
    # than 2x", "nearly half"). These adjectives go stale silently if the
    # underlying number moves; either wire the adjective to a fragment via
    # render_claims.py:_emit_comparator() or allow-list with justification.
    # Matches the comparator phrase only — separate match from the ratio so
    # the row-context allow-list still works.
    (
        r"(?:roughly|approximately|about|nearly|almost|over|more than|"
        r"just under|just over|around|close to)\s*\$?\d+(?:\.\d+)?\s*\\?\$?\\times\$?",
        "comparator+ratio",
    ),
    # Verb-form comparators (these ARE the claim, not a hedge on a number).
    # Almost always need to be wired or allow-listed.
    (r"\b(?:halves|doubles|triples|quadruples)\b", "verb-comparator"),
]

# Allow-list: legitimately hand-typed numbers that are NOT data-derived.
# Each entry is matched against the full line; if the regex matches, the
# flagged match is ignored.  Use sparingly and justify each entry.
ALLOW_PATTERNS = [
    # Historical / narrative framing (describe prior-state, not current data)
    (r"1\.03\\times.*2 seeds", "historical: initial BW parity was 2-seed snapshot"),
    (r"originally 99% sparsity", "ratio describing domain property, not experiment"),
    # Hyperparameter prose
    (r"learning rate.*\\times 10\^\{-4\}", "hyperparameter: lr 3e-4"),
    # Cohen's d (not yet fragment-backed)
    (r"Cohen'?s \$?d\$?.{0,5}=", "Cohen's d — TODO: wire to fragment"),
    # 95% bootstrap CIs (not yet fragment-backed)
    (r"\[[-+]?\d+\.\d+,\s*[-+]?\d+\.\d+\]", "bootstrap CI — TODO: wire to fragment"),
    # siunitx-style \num{} would also be acceptable (future)
    (r"\\num\{", "siunitx \\num{} — acceptable"),
    # Appendix component-statistics table (derived from parquet; separate pipeline)
    (r"sigma/\|\\?mu\|", "appendix component stats — separate pipeline"),
    (r"texttt\{landing\}|texttt\{shaping\}|texttt\{fuel|texttt\{run\}", "component-stats row"),
    # Wallclock table (FPS numbers — separate benchmarking output, not in results.csv)
    (r"CartPole.*&.*\d+.*&.*\d+.*&.*\d+", "wallclock benchmark row — separate pipeline"),
    # "at 1M steps +183.7 \pm 28.1" is wrapped in \cnum already; but if grep
    # matches inside a line that also contains \cnum, accept it
    # (fall-through handled below).
    # Appendix LLM tables (different data pipeline)
    (r"LLM_|K=2 coarse-grained preliminary", "LLM appendix — separate pipeline"),
    # 1.0× placeholder = "no advantage" visual marker for envs at parity
    # (Reacher, Resource overview/appendix rows).  Not a query result.
    (r"neg(ative|\.\\? control).*1\.0\\times", "Reacher parity placeholder"),
    # (Resource parity placeholder removed with env.)
    (r"Reacher.*All dense.*Parity", "Reacher parity placeholder (appendix)"),
    # (Resource env removed — no valid data. Allow-list entry obsolete.)
    (r"\\texttt\{target_1|4 target distances", "Reacher appendix row"),
    # (A5 cross-seed SD/var ratio now wired to fragments, TODO resolved.)
    # Appendix component-stats σ/|μ| table (computed from parquet, not results.csv)
    (r"\\texttt\{energy\}.*0\.3", "appendix σ/|μ| row"),
    (r"\\texttt\{crash\}.*52\.6", "appendix σ/|μ| row"),
    (r"\\texttt\{ctrl\\_cost\}.*0\.3", "appendix σ/|μ| row"),
    # (Resource second-column allow-list entry obsolete — env removed.)
    # (Equal-weights ablation now wired to fragments abl_eqw_*; TODO resolved.)
    # SB3-vs-TorchRL hyperparameter variant (118.6 ± 26.8) — separate data pipeline
    (r"optimal.*LR annealing.*cosine entropy", "hyperparameter-fairness experiment — TODO wire"),
]

CNUM_RE = re.compile(r"\\cnum\{[^}]+\}")
COMMENT_RE = re.compile(r"(?<!\\)%")


def strip_comments(line: str) -> str:
    m = COMMENT_RE.search(line)
    return line[: m.start()] if m else line


def mask_cnum(line: str) -> str:
    """Replace every \\cnum{...} with a neutral token so regexes don't match inside."""
    return CNUM_RE.sub("<cnum>", line)


def line_allowed(line: str) -> str | None:
    for pat, reason in ALLOW_PATTERNS:
        if re.search(pat, line):
            return reason
    return None


def scan(path: Path) -> list[tuple[int, str, str, str]]:
    """Return list of (lineno, pattern_label, match_text, line) for violations."""
    violations: list[tuple[int, str, str, str]] = []
    text = path.read_text()
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = strip_comments(raw)
        if not line.strip():
            continue
        masked = mask_cnum(line)
        allow_reason = line_allowed(line)
        for pat, label in FLAG_PATTERNS:
            for m in re.finditer(pat, masked):
                if allow_reason:
                    continue
                violations.append((lineno, label, m.group(0), line.strip()))
    return violations


# --- Context / semantic-swap detection ----------------------------------
# Catches the failure mode where an author (or agent) copies a \cnum{}
# reference and forgets to change the context prefix — e.g. leaving
# \cnum{k4_ratio} inside the K=6 table row.
#
# Structural rule: when a table row begins with a literal context label,
# every \cnum on that row must use the matching prefix.

# Cross-context prefixes that are allowed in any row (derived quantities,
# cross-cutting metrics).  Add sparingly and justify.
CROSS_CONTEXT_PREFIXES = (
    "abl_",  # cross-row derived ablation stats (e.g. abl_decomp_delta = A1-A7)
)

ROW_CONTEXT_RULES = [
    # K-scaling table: rows start with "2 & ", "4 & ", "6 & ", "8 & "
    (
        re.compile(r"^\s*(\d+)\s*&"),
        lambda m: f"k{m.group(1)}_",
        "K-scaling row context",
    ),
    # Ablation + compute-matched tables: rows start with "A1 & ", "B2 & ", "C5 & ", "D1 & "
    (
        re.compile(r"^\s*([ABCD]\d+)\s*&"),
        lambda m: f"abl{m.group(1)}_",
        "Ablation row context",
    ),
]


def scan_row_context(path: Path) -> list[tuple[int, str, str, str]]:
    """Detect \\cnum references in a table row whose prefix doesn't match
    the row's context label.  Returns list of (lineno, msg, name, line).
    """
    text = path.read_text()
    violations: list[tuple[int, str, str, str]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = strip_comments(raw)
        names = re.findall(r"\\cnum\{([a-zA-Z0-9_\-]+)\}", line)
        if not names:
            continue
        for rx, prefix_fn, label in ROW_CONTEXT_RULES:
            m = rx.match(line)
            if not m:
                continue
            expected = prefix_fn(m)
            for name in names:
                if name.startswith(expected):
                    continue
                if any(name.startswith(p) for p in CROSS_CONTEXT_PREFIXES):
                    continue
                violations.append(
                    (
                        lineno,
                        f"{label}: row starts with {m.group(1)!r} "
                        f"but \\cnum references {name!r} "
                        f"(expected prefix {expected!r})",
                        name,
                        line.strip(),
                    )
                )
            break  # only one rule applies per row
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paper", default=str(PAPER), help="Paper .tex path")
    parser.add_argument("--quiet", action="store_true", help="Only print count")
    args = parser.parse_args()

    violations = scan(Path(args.paper))
    ctx_violations = scan_row_context(Path(args.paper))

    if not violations and not ctx_violations:
        print("lint_hardcoded_numbers: OK — no hardcoded numerical claims, no context mismatches")
        return 0
    if ctx_violations:
        print(f"lint_hardcoded_numbers: FAIL — {len(ctx_violations)} row-context mismatches\n")
        for lineno, msg, _name, line in ctx_violations:
            line_short = line if len(line) <= 120 else line[:120] + "…"
            print(f"  L{lineno:4d}  {msg}")
            print(f"        >>> {line_short}")
        if not violations:
            return 1
        print()

    if args.quiet:
        print(f"lint_hardcoded_numbers: FAIL — {len(violations)} hardcoded numbers")
        return 1

    print(f"lint_hardcoded_numbers: FAIL — {len(violations)} hardcoded numerical claims\n")
    print("Each of these should be wired through render_claims.py + \\cnum{...}")
    print("or explicitly allow-listed in ALLOW_PATTERNS with justification.\n")
    for lineno, label, match, line in violations:
        line_short = line if len(line) <= 120 else line[:120] + "…"
        print(f"  L{lineno:4d}  [{label:10s}]  {match!r}")
        print(f"        >>> {line_short}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
