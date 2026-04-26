"""Paper build orchestrator with hash-checked DAG.

Extends the `render_claims.py --check` pattern from fragments to the
full paper build graph: figures, fragments, and (optionally) the PDF.

Design
------
Each ``fig_*.py`` declares its data dependencies at module level as a
literal list::

    INPUTS = [
        "../data/results.csv",
        "../data/metrics/*pcz-ppo-running_lunarlander_*.parquet",
    ]

Paths are glob patterns relative to the figure's own directory.  The
orchestrator AST-parses these (never imports the script) so it can build
the DAG without importing matplotlib etc.

Two implicit inputs apply to every figure: the script itself and
``fig_data.py`` (the shared loader).  A change to either invalidates
anything that depends on it.

Stages
------
* **fragments** — ``render_claims.py`` → ``generated/*.tex``.
  Inputs: ``results.csv`` + ``fig_data.py`` + ``render_claims.py``.
* **<figure>** — one per ``fig_*.py``.  Outputs ``fig_<name>.{pdf,png}``.
* **pdf** — ``pcz_ppo.tex`` + fragments + figures → ``pcz_ppo.pdf`` via
  ``latexmk``.  **Always included in ``--build``** so any upstream change
  (a fig regenerates, a fragment is re-rendered, ``pcz_ppo.tex`` is
  edited) cascades through to the compiled PDF — the dependency-tree
  guarantee the orchestrator promises.  **Excluded from ``--check`` by
  default** (use ``--pdf`` to opt in) because ``latexmk`` is slow and
  flaky; the pre-commit hook calls ``--check`` without ``--pdf``.

Lock file
---------
``paper_build.lock.json`` stores a SHA256 of each stage's inputs keyed
by stage name.  Committed to git.  ``--check`` recomputes and fails on
any mismatch (used by the pre-commit hook).  ``--build`` rebuilds stale
stages and updates the lock.

Agnosticity
-----------
The orchestrator is 90% project-agnostic.  Per-paper configuration lives
at the top of this file (``MAIN_TEX``, ``BIB``, ``DATA_DIR``,
``PDF_COMPILE_CMD``).  To port to another paper: copy this file, adjust
those four constants, and add ``INPUTS`` declarations to each figure.
A future ``paper_build.toml`` manifest would fully generalise this.

Usage
-----
::

    # Fast staleness check (used by pre-commit; skips PDF stage):
    uv run python artifacts/pcz-ppo/paper/paper_build.py --check

    # Full check including PDF:
    uv run python artifacts/pcz-ppo/paper/paper_build.py --check --pdf

    # Rebuild stale figures + fragments + PDF (the cascade — default):
    uv run python artifacts/pcz-ppo/paper/paper_build.py --build

    # Inspect the DAG (figures + fragments + PDF):
    uv run python artifacts/pcz-ppo/paper/paper_build.py --dag

    # Force every stage to rebuild regardless of lock state:
    uv run python artifacts/pcz-ppo/paper/paper_build.py --build --force

    # Rebuild figures + fragments only, skip PDF (rare; e.g. when
    # latexmk is broken and you just want to refresh the figs):
    uv run python artifacts/pcz-ppo/paper/paper_build.py --build --no-pdf
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── Per-paper configuration ──────────────────────────────────────────

PAPER_DIR = Path(__file__).resolve().parent


def _find_repo_root(start: Path) -> Path:
    """Walk up from start until we find a .git directory (repo root)."""
    for p in [start, *start.parents]:
        if (p / ".git").exists():
            return p
    return start.parents[2]  # fallback: artifacts/pcz-ppo/paper → repo root


REPO_ROOT = _find_repo_root(PAPER_DIR)
LOCKFILE = PAPER_DIR / "paper_build.lock.json"
GENERATED_DIR = PAPER_DIR / "generated"
MAIN_TEX = PAPER_DIR / "pcz_ppo.tex"
BIB = PAPER_DIR / "references.bib"
RENDER_CLAIMS = PAPER_DIR / "render_claims.py"
FIG_DATA = PAPER_DIR / "fig_data.py"

# Shared implicit inputs applied to every figure stage.
SHARED_IMPLICIT_INPUTS = ["fig_data.py"]

# Files matched by ``fig_*.py`` that are NOT figure scripts (shared
# libraries, data loaders, etc.).  Silently skipped during discovery.
NON_FIGURE_SCRIPTS = {"fig_data.py"}

# PDF build command (run with cwd=PAPER_DIR).
PDF_COMPILE_CMD = [
    "latexmk",
    "-pdf",
    "-interaction=nonstopmode",
    "-halt-on-error",
    "pcz_ppo.tex",
]


# ── Core types ───────────────────────────────────────────────────────


@dataclass
class Stage:
    name: str
    inputs: list[Path] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    cmd: list[str] = field(default_factory=list)
    cwd: Path = REPO_ROOT

    def input_hash(self) -> str:
        """SHA256 over each input's relative path + content."""
        h = hashlib.sha256()
        for p in self.inputs:
            try:
                rel = p.relative_to(REPO_ROOT)
            except ValueError:
                rel = p
            h.update(str(rel).encode())
            h.update(b"\0")
            if p.is_file():
                h.update(p.read_bytes())
            h.update(b"\0")
        return h.hexdigest()


# ── INPUTS parsing ───────────────────────────────────────────────────


def _parse_string_list(fig_path: Path, var_name: str) -> list[str] | None:
    """AST-parse ``<var_name> = [...]`` at module level.  Returns None if absent."""
    tree = ast.parse(fig_path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Name) and target.id == var_name):
                continue
            if not isinstance(node.value, (ast.List, ast.Tuple)):
                raise ValueError(f"{fig_path.name}: {var_name} must be a list/tuple literal")
            items: list[str] = []
            for elt in node.value.elts:
                if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
                    raise ValueError(f"{fig_path.name}: {var_name} elements must be string literals")
                items.append(elt.value)
            return items
    return None


def parse_inputs(fig_path: Path) -> list[str] | None:
    """AST-parse ``INPUTS = [...]`` at module level."""
    return _parse_string_list(fig_path, "INPUTS")


def parse_outputs(fig_path: Path) -> list[str] | None:
    """AST-parse optional ``OUTPUTS = [...]`` at module level.

    Returns None if not declared — caller falls back to the default
    convention ``fig_<name>.{pdf,png}``.
    """
    return _parse_string_list(fig_path, "OUTPUTS")


def _has_output_flag(fig_path: Path) -> bool:
    """Return True if the figure script declares an ``--output`` argparse flag."""
    return '"--output"' in fig_path.read_text()


def resolve_globs(base: Path, patterns: list[str]) -> list[Path]:
    """Resolve glob patterns relative to base.  Sorted, deduplicated, files only."""
    out: set[Path] = set()
    for pat in patterns:
        for match in base.glob(pat):
            if match.is_file():
                out.add(match.resolve())
    return sorted(out)


# ── DAG construction ─────────────────────────────────────────────────


def fragments_stage() -> Stage | None:
    if not RENDER_CLAIMS.exists():
        return None
    # render_claims reads results.csv via fig_data
    results_csv = REPO_ROOT / "artifacts/pcz-ppo/data/results.csv"
    inputs = [
        results_csv.resolve() if results_csv.exists() else results_csv,
        FIG_DATA.resolve(),
        RENDER_CLAIMS.resolve(),
    ]
    # Outputs: snapshot the current generated/ tree.  Used only for
    # output-existence check; the actual write set is determined by
    # render_claims.
    outputs = sorted(GENERATED_DIR.glob("*.tex")) if GENERATED_DIR.exists() else []
    return Stage(
        name="fragments",
        inputs=inputs,
        outputs=outputs,
        cmd=["uv", "run", "python", str(RENDER_CLAIMS.relative_to(REPO_ROOT))],
    )


def figure_stages() -> list[Stage]:
    stages: list[Stage] = []
    for fig in sorted(PAPER_DIR.glob("fig_*.py")):
        if fig.name in NON_FIGURE_SCRIPTS:
            continue
        patterns = parse_inputs(fig)
        if patterns is None:
            print(
                f"WARNING: {fig.name} has no INPUTS declaration; skipping (add INPUTS = [...] at module level)",
                file=sys.stderr,
            )
            continue
        data_inputs = resolve_globs(PAPER_DIR, patterns)
        implicit = [(PAPER_DIR / p).resolve() for p in SHARED_IMPLICIT_INPUTS if (PAPER_DIR / p).exists()]
        all_inputs = sorted({fig.resolve(), *implicit, *data_inputs})
        declared_outputs = parse_outputs(fig)
        if declared_outputs is not None:
            outputs = [(PAPER_DIR / o).resolve() for o in declared_outputs]
        else:
            outputs = [fig.with_suffix(".pdf"), fig.with_suffix(".png")]
        # Pass primary PDF output path explicitly so figure scripts don't rely on
        # a workspace-relative default (e.g. "artifacts/pcz-ppo/paper/fig_*.pdf").
        primary_pdf = next((o for o in outputs if o.suffix == ".pdf"), None)
        fig_cmd = ["uv", "run", "python", str(fig.relative_to(REPO_ROOT))]
        if primary_pdf is not None and _has_output_flag(fig):
            fig_cmd += ["--output", str(primary_pdf)]
        stages.append(
            Stage(
                name=fig.stem,
                inputs=all_inputs,
                outputs=outputs,
                cmd=fig_cmd,
            )
        )
    return stages


def pdf_stage() -> Stage:
    inputs = {MAIN_TEX.resolve()}
    if BIB.exists():
        inputs.add(BIB.resolve())
    if GENERATED_DIR.exists():
        inputs.update(p.resolve() for p in GENERATED_DIR.glob("*.tex"))
    inputs.update(p.resolve() for p in PAPER_DIR.glob("fig_*.pdf"))
    return Stage(
        name="pdf",
        inputs=sorted(inputs),
        outputs=[PAPER_DIR / "pcz_ppo.pdf"],
        cmd=PDF_COMPILE_CMD,
        cwd=PAPER_DIR,
    )


def build_graph(include_pdf: bool = False) -> list[Stage]:
    stages: list[Stage] = []
    frag = fragments_stage()
    if frag is not None:
        stages.append(frag)
    stages.extend(figure_stages())
    if include_pdf:
        stages.append(pdf_stage())
    return stages


# ── Lock file ────────────────────────────────────────────────────────


def load_lock() -> dict:
    if LOCKFILE.exists():
        return json.loads(LOCKFILE.read_text())
    return {}


def save_lock(lock: dict) -> None:
    LOCKFILE.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")


def is_stale(stage: Stage, lock: dict) -> tuple[bool, str]:
    """Return (stale, reason)."""
    entry = lock.get(stage.name)
    if entry is None:
        return True, "no lock entry"
    for out in stage.outputs:
        if not out.exists():
            return True, f"missing output {out.name}"
    current = stage.input_hash()
    if entry.get("input_hash") != current:
        return True, "input hash changed"
    return False, ""


# ── Commands ─────────────────────────────────────────────────────────


def cmd_check(stages: list[Stage], lock: dict) -> int:
    stale: list[tuple[Stage, str]] = []
    for s in stages:
        yes, reason = is_stale(s, lock)
        if yes:
            stale.append((s, reason))
    if not stale:
        print(f"OK: all {len(stages)} paper artifacts up to date.")
        return 0
    print("STALE paper artifacts detected:", file=sys.stderr)
    for s, reason in stale:
        print(f"  - {s.name:30s}  ({reason})", file=sys.stderr)
    print(
        "\nFix: uv run python artifacts/pcz-ppo/paper/paper_build.py --build",
        file=sys.stderr,
    )
    return 1


def cmd_build(stages: list[Stage], lock: dict) -> int:
    rebuilt = 0
    for s in stages:
        yes, reason = is_stale(s, lock)
        if not yes:
            print(f"[ok]      {s.name}")
            continue
        print(f"[rebuild] {s.name}  ({reason})")
        r = subprocess.run(s.cmd, cwd=s.cwd)
        if r.returncode != 0:
            print(f"FAILED: {s.name}", file=sys.stderr)
            return r.returncode
        # Re-discover outputs for stages with dynamic output sets.
        if s.name == "fragments":
            s.outputs = sorted(GENERATED_DIR.glob("*.tex"))
        if s.name == "pdf":
            # PDF inputs may now include freshly-built fragments/figures;
            # recompute hash post-build so the lock reflects what was
            # actually compiled together.
            s.inputs = pdf_stage().inputs
        lock[s.name] = {"input_hash": s.input_hash()}
        rebuilt += 1

    # Prune orphan lock entries for stages no longer in the DAG (e.g. a
    # ``fig_*.py`` that was deleted).  Skip pdf when it's temporarily
    # excluded by ``--no-pdf`` so the next plain ``--build`` doesn't have
    # to rebuild it from scratch.
    current_names = {s.name for s in stages}
    pruned: list[str] = []
    for orphan in [k for k in list(lock.keys()) if k not in current_names]:
        if orphan == "pdf":
            # Preserved across --no-pdf invocations so cascade still works
            # when the user re-enables PDF on the next build.
            continue
        del lock[orphan]
        pruned.append(orphan)
    if pruned:
        print(f"Pruned {len(pruned)} orphan lock entries: {', '.join(pruned)}")

    save_lock(lock)
    print(f"Rebuilt {rebuilt}/{len(stages)} stages.")
    return 0


def cmd_audit(strict: bool = False) -> int:
    """Print orphan figure scripts, bibliography entries, and fragments.

    Detects:
      * ``fig_*.py`` with no ``\\includegraphics{fig_NAME}`` in ``pcz_ppo.tex``.
        (PDF/PNG outputs of orphan scripts are also dead.)
      * ``.bib`` entries with no matching ``\\cite{KEY}`` in ``pcz_ppo.tex``.
        Multi-key cites like ``\\cite{a,b,c}`` are split correctly.
      * ``generated/*.tex`` fragments with no matching ``\\cnum{NAME}`` in
        ``pcz_ppo.tex``. ``render_claims.py --check`` also catches these via
        its tempdir diff; the audit surfaces them in a unified orphan report
        and serves as defense-in-depth if the renderer is bypassed.

    Skips intentionally:
      * Orphan ``\\label{}`` (false-positive prone — labels are sometimes
        referenced externally in slides or talks).
      * Dead prose / stale paragraphs (no simple automated detector).

    With ``strict=True`` returns non-zero on any orphan found (CI gate).
    Default is advisory (always returns 0) — orphans are a cleanup signal,
    not a correctness bug.
    """
    if not MAIN_TEX.exists():
        print(f"WARNING: {MAIN_TEX} not found; skipping audit", file=sys.stderr)
        return 0
    tex = MAIN_TEX.read_text()

    # ── Orphan figure scripts ────────────────────────────────────────
    # A script is dead iff *none* of its declared (or default-convention)
    # output basenames is referenced by an \includegraphics in the .tex.
    # Honor module-level ``OUTPUTS = [...]`` declarations so scripts like
    # ``fig_ablation_bar.py`` (which produces ``fig_ablation.pdf``) aren't
    # flagged as orphan.
    referenced_basenames = {
        m.group(1) for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^\}]+?)(?:\.pdf|\.png)?\}", tex)
    }
    fig_scripts = [s for s in sorted(PAPER_DIR.glob("fig_*.py")) if s.name not in NON_FIGURE_SCRIPTS]
    orphan_figs: list[str] = []
    for script in fig_scripts:
        declared_outputs = parse_outputs(script)
        if declared_outputs is not None:
            output_basenames = {Path(o).stem for o in declared_outputs}
        else:
            output_basenames = {script.stem}  # default convention
        if output_basenames.isdisjoint(referenced_basenames):
            orphan_figs.append(script.stem)

    # ── Orphan bibliography entries ──────────────────────────────────
    orphan_bibs: list[str] = []
    bib_keys: set[str] = set()
    if BIB.exists():
        bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", BIB.read_text()))
        # \cite, \citep, \citet, \citeauthor, \citeyear etc. all match \cite\w*
        cite_groups = re.findall(r"\\cite\w*(?:\[[^\]]*\])?\s*\{([^\}]+)\}", tex)
        cited_keys = {k.strip() for grp in cite_groups for k in grp.split(",")}
        orphan_bibs = sorted(bib_keys - cited_keys)

    # ── Orphan generated/*.tex fragments ─────────────────────────────
    # A fragment is dead iff no \cnum{name} in the .tex resolves to it.
    # render_claims.py also catches this in its tempdir diff (line 2003 of
    # render_claims.py:check), but auditing here gives a single orphan
    # inventory and protects against the case where a fragment is committed
    # by hand without re-running the renderer.
    orphan_frags: list[str] = []
    fragment_count = 0
    if GENERATED_DIR.exists():
        # strip LaTeX comments before scanning so commented-out \cnum{}
        # references don't keep dead fragments alive
        tex_no_comments = re.sub(r"(?<!\\)%[^\n]*", "", tex)
        cnum_refs = set(re.findall(r"\\cnum\{([a-zA-Z0-9_\-]+)\}", tex_no_comments))
        committed_frags = sorted(p.stem for p in GENERATED_DIR.glob("*.tex"))
        fragment_count = len(committed_frags)
        orphan_frags = [name for name in committed_frags if name not in cnum_refs]

    # ── Report ───────────────────────────────────────────────────────
    total = len(orphan_figs) + len(orphan_bibs) + len(orphan_frags)
    if total == 0:
        print(
            f"audit: clean — {len(fig_scripts)} figs, {len(bib_keys)} bib entries, "
            f"{fragment_count} fragments, all referenced."
        )
        return 0

    print(f"audit: {total} orphan(s) found:")
    if orphan_figs:
        print(f"\n  Orphan figure scripts ({len(orphan_figs)}) — produce dead PDF/PNG outputs:")
        for f in orphan_figs:
            print(f"    {f}.py  (output: {f}.{{pdf,png}})")
        print(f"  Fix: delete {' '.join(f + '.{py,pdf,png}' for f in orphan_figs)}")
    if orphan_bibs:
        print(f"\n  Orphan bibliography entries ({len(orphan_bibs)}) — never cited:")
        for k in orphan_bibs:
            print(f"    {k}")
        print(f"  Fix: remove these @entries from {BIB.name}")
    if orphan_frags:
        print(f"\n  Orphan generated fragments ({len(orphan_frags)}) — no \\cnum reference:")
        for name in orphan_frags:
            print(f"    generated/{name}.tex")
        print(
            "  Fix: re-run `uv run python artifacts/pcz-ppo/paper/render_claims.py` "
            "(it prunes orphans automatically) and commit the deletions."
        )
    return 1 if strict else 0


def cmd_dag(stages: list[Stage]) -> int:
    print(f"Paper build DAG ({len(stages)} stages)")
    print(f"  repo root: {REPO_ROOT}")
    print(f"  paper dir: {PAPER_DIR.relative_to(REPO_ROOT)}")
    print(f"  lock:      {LOCKFILE.relative_to(REPO_ROOT)}")
    print()
    for s in stages:
        print(f"[{s.name}]")
        print(f"  cmd: {' '.join(s.cmd)}")
        print(f"  inputs ({len(s.inputs)}):")
        for i in s.inputs[:6]:
            try:
                rel = i.relative_to(REPO_ROOT)
            except ValueError:
                rel = i
            print(f"    {rel}")
        if len(s.inputs) > 6:
            print(f"    ... +{len(s.inputs) - 6} more")
        print(f"  outputs ({len(s.outputs)}):")
        for o in s.outputs[:6]:
            try:
                rel = o.relative_to(REPO_ROOT)
            except ValueError:
                rel = o
            print(f"    {rel}")
        if len(s.outputs) > 6:
            print(f"    ... +{len(s.outputs) - 6} more")
        print()
    return 0


# ── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="Fail if any stage is stale.")
    ap.add_argument("--build", action="store_true", help="Rebuild stale stages; update lock.")
    ap.add_argument("--dag", action="store_true", help="Print the DAG and exit.")
    ap.add_argument(
        "--audit",
        action="store_true",
        help=(
            "Print orphan figure scripts (no \\includegraphics ref) and "
            "orphan .bib entries (no \\cite). Advisory; pair with --strict "
            "for non-zero exit on orphans."
        ),
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="With --audit: exit non-zero if any orphan is found (CI gate).",
    )
    ap.add_argument(
        "--pdf",
        action="store_true",
        help=(
            "With --check: include the PDF stage (off by default — pre-commit "
            "runs --check without --pdf for speed).  With --build/--dag the "
            "PDF stage is always included unless --no-pdf is set."
        ),
    )
    ap.add_argument(
        "--no-pdf",
        action="store_true",
        help=(
            "With --build: skip the PDF stage (figures + fragments only).  "
            "Use when latexmk is broken or you only want fast figure refresh."
        ),
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="With --build: rebuild every stage regardless of lock state.",
    )
    args = ap.parse_args()

    modes = sum([args.check, args.build, args.dag, args.audit])
    if modes != 1:
        ap.error("exactly one of --check / --build / --dag / --audit is required")
    if args.no_pdf and not args.build:
        ap.error("--no-pdf only meaningful with --build")
    if args.pdf and args.no_pdf:
        ap.error("--pdf and --no-pdf are mutually exclusive")
    if args.strict and not args.audit:
        ap.error("--strict only meaningful with --audit")

    # --audit doesn't need the build graph — fast path.
    if args.audit:
        return cmd_audit(strict=args.strict)

    # PDF stage inclusion policy (the dependency-tree cascade):
    #   --build: always include (cascade upstream changes through to PDF),
    #            unless --no-pdf is explicitly set.
    #   --dag:   always include (so users see the full graph).
    #   --check: only include when --pdf is set (pre-commit speed).
    if args.build:
        include_pdf = not args.no_pdf
    elif args.dag:
        include_pdf = True
    else:  # --check
        include_pdf = args.pdf

    stages = build_graph(include_pdf=include_pdf)
    lock = {} if args.force else load_lock()

    if args.dag:
        return cmd_dag(stages)
    if args.check:
        return cmd_check(stages, lock)
    if args.build:
        return cmd_build(stages, lock)
    return 2


if __name__ == "__main__":
    sys.exit(main())
