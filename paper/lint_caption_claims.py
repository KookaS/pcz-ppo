"""Figure-integrity Layer 2: Caption ↔ data consistency lint.

Scans every ``\\caption{...}`` block in ``pcz_ppo.tex`` looking for hand-typed
direction-claim patterns ("PCZ wins", "PPO outperforms", "PCZ above PPO",
etc.) that must either reference a data-derived ``\\cnum{}`` fragment or be
explicitly allow-listed in ``ALLOW_PATTERNS``.

The motivation is that hand-typed claims silently drift when the underlying
data changes. ``render_claims.py`` Layer 1 emits ``_winner``, ``_summary``,
``_direction`` fragments for every paired (PCZ, PPO) cell --- captions
should reference these (e.g. ``\\cnum{w10_k4_4M_summary}``) instead of
hand-typing ``"PCZ-PPO outperforms PPO"``.

See ``FIGURE_INTEGRITY.md`` for the broader design and known limitations.

Usage::

    uv run python artifacts/pcz-ppo/paper/lint_caption_claims.py
    # exit 0 = OK, exit 1 = violations found

Algorithm:

1. Parse ``pcz_ppo.tex``, extract every figure caption (text between
   ``\\caption{...}`` matched-brace block, scoped to ``\\begin{figure}`` /
   ``\\end{figure}`` regions).
2. For each caption, scan for direction-claim regex matches.
3. For each match, check whether a ``\\cnum{}`` reference appears within
   ``JUSTIFICATION_WINDOW_CHARS`` of the match (before or after).
4. If no ``\\cnum{}`` is nearby, check ``ALLOW_PATTERNS`` for a documented
   exception; if neither, flag.

The default justification window is 80 characters --- enough to allow
``Δ=\\cnum{...}, p=\\cnum{...}`` adjacency to a phrase like "PCZ-PPO
outperforms PPO".
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER_TEX = Path(__file__).parent / "pcz_ppo.tex"

JUSTIFICATION_WINDOW_CHARS = 120

# Direction-claim patterns. Each is a regex that matches a hand-typed claim;
# (?i) makes them case-insensitive. Word boundaries on both sides to avoid
# matching inside larger words (e.g. "outperforms" inside a comment).
DIRECTION_PATTERNS = [
    re.compile(r"\b(?:PCZ-PPO|PCZ|PPO)\s+(?:wins?|outperforms?|beats?|loses?|underperforms?|fails?)\b", re.IGNORECASE),
    re.compile(r"\b(?:PCZ-PPO|PCZ|PPO)\s+(?:is\s+)?(?:above|below|ahead\s+of)\s+(?:PCZ-PPO|PCZ|PPO)\b", re.IGNORECASE),
    re.compile(r"\bPCZ-PPO\s+collapses?\b", re.IGNORECASE),
]

# Allow-list. Each entry is (pattern_text, reason). The pattern_text is a
# substring or short regex matched against the claim's surrounding ~80
# chars; if any allow-list entry matches, the violation is suppressed.
#
# Convention: only add entries that document a *legitimate* hand-typed
# claim. If the claim is just stale, fix it via \\cnum{} instead of
# allow-listing.
ALLOW_PATTERNS: list[tuple[str, str]] = [
    # Negative-control language in cross_env caption (K=2 PPO wins is the
    # pre-registered negative control; the data confirms it).
    (r"K=2.*PPO wins", "negative control: K=2 PPO wins is pre-registered, data-confirmed"),
    (r"PPO wins \(negative control", "negative control disclosure"),
    # K=8 weight-sensitivity figure: "PCZ-PPO collapses" at non-heterogeneous
    # weights is a domain finding (collapse threshold near zero), not a stale
    # data point.
    (r"collapses to near-zero", "weight-sensitivity domain language"),
    (r"PCZ-PPO collapses; PPO learns normally", "K=8 collapse caption, domain-language"),
    (r"PCZ-PPO collapses again", "K=8 collapse continuation"),
    # HP-fairness panel: "PCZ-PPO fails regardless of hyperparameter
    # configuration" is a multi-cell domain claim (cf. weight-sensitivity
    # cliff), not tied to one fragment.
    (r"fails regardless of hyperparameter", "multi-cell HP-fairness domain claim"),
    # Generic mention with explicit env or condition tag — these are
    # contextual descriptions tied to a fragment elsewhere in the caption.
    (r"only (?:by carefully|with hand-designed)", "weight-sensitivity domain disclosure"),
    # Failure-rate language uses 0% / 30% directly via \cnum{}; the
    # surrounding "incurs them regularly" is rhetorical, not a data claim.
    (r"incurs them regularly", "rhetorical framing for failure-rate claim"),
]


def _extract_captions(tex: str) -> list[tuple[int, str, str]]:
    """Extract (line_no, caption_text, surrounding_figure_label) tuples.

    Captions outside ``\\begin{figure}...\\end{figure}`` blocks are skipped
    (table captions are checked separately by ``lint_caption_drift.py``).
    """
    out: list[tuple[int, str, str]] = []
    # Find every \begin{figure}...\end{figure} block
    for fig_match in re.finditer(r"\\begin\{figure\}.*?\\end\{figure\}", tex, re.DOTALL):
        block = fig_match.group(0)
        block_start = fig_match.start()
        line_no = tex[:block_start].count("\n") + 1
        # Find the \caption{...} inside this block, with brace matching
        cap_start = block.find("\\caption{")
        if cap_start == -1:
            continue
        # Match braces from cap_start + len("\\caption")
        i = cap_start + len("\\caption{")
        depth = 1
        while i < len(block) and depth > 0:
            if block[i] == "{":
                depth += 1
            elif block[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            elif block[i] == "\\" and i + 1 < len(block):
                # Skip escaped braces
                i += 2
                continue
            i += 1
        caption_text = block[cap_start + len("\\caption{") : i]
        # Find the figure label
        label_match = re.search(r"\\label\{(fig:[^}]+)\}", block)
        label = label_match.group(1) if label_match else "(no label)"
        out.append((line_no, caption_text, label))
    return out


def _has_nearby_cnum(caption: str, span: tuple[int, int], window: int = JUSTIFICATION_WINDOW_CHARS) -> bool:
    """Check whether a \\cnum{} reference sits within ``window`` chars of the match span."""
    start, end = span
    pre = max(0, start - window)
    post = min(len(caption), end + window)
    snippet = caption[pre:post]
    return bool(re.search(r"\\cnum\{[^}]+\}", snippet))


def _is_allow_listed(snippet: str) -> str | None:
    """Return the reason if ``snippet`` matches an allow-list entry, else None."""
    for pattern, reason in ALLOW_PATTERNS:
        if re.search(pattern, snippet, re.IGNORECASE):
            return reason
    return None


def lint(tex_path: Path = PAPER_TEX) -> int:
    """Run the lint and return exit code (0 OK, 1 violations)."""
    if not tex_path.exists():
        print(f"lint_caption_claims: ERROR — {tex_path} not found", file=sys.stderr)
        return 2
    tex = tex_path.read_text()
    captions = _extract_captions(tex)
    if not captions:
        print("lint_caption_claims: WARNING — no figure captions found")
        return 0

    violations: list[tuple[str, int, str, str]] = []  # (label, line, claim, snippet)
    for line_no, cap, label in captions:
        for pat in DIRECTION_PATTERNS:
            for m in pat.finditer(cap):
                claim = m.group(0)
                if _has_nearby_cnum(cap, m.span()):
                    continue
                # Compute snippet for allow-list check
                start, end = m.span()
                pre = max(0, start - JUSTIFICATION_WINDOW_CHARS)
                post = min(len(cap), end + JUSTIFICATION_WINDOW_CHARS)
                snippet = cap[pre:post]
                if _is_allow_listed(snippet):
                    continue
                violations.append((label, line_no, claim, snippet))

    if not violations:
        print(f"lint_caption_claims: OK — {len(captions)} captions, no unjustified direction claims")
        return 0

    print(f"lint_caption_claims: FAIL — {len(violations)} unjustified direction claim(s) in figure captions\n")
    print("Each flagged claim must either (a) reference a data-derived")
    print(f"\\cnum{{}} fragment within ~{JUSTIFICATION_WINDOW_CHARS} chars, OR (b) be explicitly")
    print("allow-listed in ALLOW_PATTERNS with a stated reason.\n")
    for label, line, claim, snippet in violations:
        s = snippet.replace("\n", " ")
        if len(s) > 160:
            s = s[:160] + "…"
        print(f"  {label} (line ~{line}): {claim!r}")
        print(f"      ...{s}...")
        print()
    print("Fix: wrap the claim in \\cnum{<prefix>_winner}, \\cnum{<prefix>_summary},")
    print("or \\cnum{<prefix>_direction}, OR add a documented allow-list entry.")
    return 1


if __name__ == "__main__":
    sys.exit(lint())
