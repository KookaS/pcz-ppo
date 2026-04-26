#!/usr/bin/env python3
"""Reject prose / numeric-result fields in experiment manifest.json files.

Manifests are config-snapshot only (reproducibility metadata): CLI command,
algos, seeds, hyperparams, env, dates, MLflow IDs. Numeric results live in
``eval/eval_summary.json``; prose (verdict, interpretation) lives in
``project/pcz-ppo/journal.md``.

Forbidden keys (rejected at the top level of every manifest):
  verdict, interpretation, results, headline_metric, derived_ratios,
  coverage_notes

Exit 0 if all manifests are clean; exit 1 with a per-file report otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FORBIDDEN_KEYS = (
    "verdict",
    "interpretation",
    "results",
    "headline_metric",
    "derived_ratios",
    "coverage_notes",
)


def lint_manifest(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"{path}: invalid JSON ({e})"]
    if not isinstance(data, dict):
        return [f"{path}: top-level value must be a JSON object"]
    return [
        f"{path}: forbidden key {key!r} (move to eval/eval_summary.json or journal.md)"
        for key in FORBIDDEN_KEYS
        if key in data
    ]


def _discover_manifests() -> list[Path]:
    """Find experiment manifests under either layout.

    Internal repo: ``artifacts/pcz-ppo/experiments/<id>/manifest.json``.
    Public repo:   ``experiments/<id>/manifest.json``.
    """
    here = Path(__file__).resolve()
    for parents_up in (3, 1):
        # parents_up=3 → workspace root (artifacts/pcz-ppo/tools/lint_manifest.py)
        # parents_up=1 → public-repo root (tools/lint_manifest.py)
        if parents_up >= len(here.parents):
            continue
        repo_root = here.parents[parents_up]
        for sub in ("artifacts/pcz-ppo/experiments", "experiments"):
            candidate = repo_root / sub
            if candidate.is_dir():
                return sorted(candidate.glob("*/manifest.json"))
    return []


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        manifests = [Path(p) for p in argv[1:]]
    else:
        manifests = _discover_manifests()

    errors: list[str] = []
    for m in manifests:
        if m.name == "manifest.json":
            errors.extend(lint_manifest(m))

    if errors:
        print("lint_manifest: violations found:")
        for e in errors:
            print(f"  {e}")
        print(f"\nForbidden keys: {', '.join(FORBIDDEN_KEYS)}")
        return 1
    print(f"lint_manifest: OK ({len(manifests)} manifest(s) checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
