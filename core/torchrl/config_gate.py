"""Config Regression Gate (G3).

Snapshots TorchRLConfig defaults to ``config_defaults.json`` and detects drift
at training launch. Motivated by an adam_eps incident where a default change
silently contaminated 6+ experiments.

Workflow
--------
1.  ``snapshot_defaults()`` introspects ``TorchRLConfig`` and returns the
    static defaults (skips ``field(default_factory=...)`` fields, which are
    per-run).
2.  ``write_baseline()`` saves that snapshot to ``config_defaults.json``.
    Run once after the dataclass is intentionally edited.
3.  ``check()`` is invoked at training launch. It diffs the current
    snapshot against the JSON baseline. Drift → SystemExit(1) unless
    ``confirm=True`` (set by CLI flag ``--confirm-config-changes``), in
    which case it prints a warning and proceeds.

Why warn-not-block by default? A drift might be a deliberate change. The
gate makes the change *visible* and forces the operator to acknowledge it,
which is the actual missing safeguard.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from dataclasses import MISSING
from pathlib import Path
from typing import Any

from .config import TorchRLConfig

DEFAULTS_JSON = Path(__file__).parent / "config_defaults.json"


def snapshot_defaults() -> dict[str, Any]:
    """Return ``{field_name: default}`` for every TorchRLConfig field with a
    static default. Fields with ``default_factory`` (per-run state like
    ``component_names``) are skipped.

    Tuples are normalized to lists for JSON round-trip stability.
    """
    out: dict[str, Any] = {}
    for f in dataclasses.fields(TorchRLConfig):
        if f.default is MISSING:
            continue
        v = f.default
        if isinstance(v, tuple):
            v = list(v)
        out[f.name] = v
    return out


def write_baseline(path: Path = DEFAULTS_JSON) -> None:
    snap = snapshot_defaults()
    path.write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n")


def load_baseline(path: Path = DEFAULTS_JSON) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def diff_defaults(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Return ``{field: (baseline, current)}`` for fields that differ.

    Includes added (baseline has ``<missing>``) and removed (current has
    ``<missing>``) fields.
    """
    sentinel = "<missing>"
    diffs: dict[str, tuple[Any, Any]] = {}
    for k in sorted(set(current) | set(baseline)):
        b = baseline.get(k, sentinel)
        c = current.get(k, sentinel)
        if b != c:
            diffs[k] = (b, c)
    return diffs


def check(*, confirm: bool, baseline_path: Path = DEFAULTS_JSON) -> int:
    """Compare current TorchRLConfig defaults against ``baseline_path``.

    Returns:
      0 — no drift, or drift acknowledged via ``confirm=True``.
    Raises ``SystemExit(1)`` — drift detected and not acknowledged.
    """
    baseline = load_baseline(baseline_path)
    if baseline is None:
        print(
            f"[config-gate] No baseline at {baseline_path.name}. "
            "Run `python -m core.torchrl.config_gate --write-baseline` to create one.",
            file=sys.stderr,
        )
        return 0
    current = snapshot_defaults()
    diffs = diff_defaults(current, baseline)
    if not diffs:
        return 0

    print(
        f"[config-gate] WARNING: {len(diffs)} TorchRLConfig default(s) differ from baseline {baseline_path.name}:",
        file=sys.stderr,
    )
    for name, (was, now) in diffs.items():
        print(f"  - {name}: {was!r}  ->  {now!r}", file=sys.stderr)

    if confirm:
        print(
            "[config-gate] Proceeding under --confirm-config-changes. "
            "Update the baseline if this drift is intentional: "
            "`python -m core.torchrl.config_gate --write-baseline`",
            file=sys.stderr,
        )
        return 0
    print(
        "[config-gate] Refusing to launch training. Either:\n"
        "  (a) revert the unintended default change in core/torchrl/config.py, or\n"
        "  (b) re-run with --confirm-config-changes and update the baseline:\n"
        "      python -m core.torchrl.config_gate --write-baseline",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--write-baseline",
        action="store_true",
        help="Snapshot current TorchRLConfig defaults to config_defaults.json",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Diff current defaults vs baseline; non-zero exit on drift (no --confirm)",
    )
    args = p.parse_args()
    if args.write_baseline:
        write_baseline()
        print(f"Wrote {DEFAULTS_JSON}")
        return 0
    if args.check:
        return check(confirm=False)
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
