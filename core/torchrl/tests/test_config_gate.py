"""Tests for config_gate.py (G3 Config Regression Gate)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.torchrl import config_gate as cg


class TestSnapshotDefaults:
    def test_returns_known_fields(self):
        snap = cg.snapshot_defaults()
        # Spot-check a few stable fields
        assert snap["adam_eps"] == 1e-8
        assert snap["lr"] == 3e-4
        assert snap["clip_epsilon"] == 0.2
        assert snap["normalize_advantage"] is True

    def test_skips_default_factory_fields(self):
        # ``component_names: list[str] = field(default_factory=list)`` —
        # this MUST NOT appear in the snapshot (it's per-run state).
        snap = cg.snapshot_defaults()
        assert "component_names" not in snap

    def test_tuples_normalized_to_lists(self):
        # JSON round-trip of tuple fields would otherwise spuriously diff.
        # No current TorchRLConfig field has a tuple default, but we
        # still want the normalization to be in place.
        snap = cg.snapshot_defaults()
        for v in snap.values():
            assert not isinstance(v, tuple), "tuple leaked into snapshot"


class TestDiffDefaults:
    def test_no_drift_returns_empty(self):
        snap = cg.snapshot_defaults()
        assert cg.diff_defaults(snap, snap) == {}

    def test_changed_value_reported(self):
        snap = cg.snapshot_defaults()
        baseline = dict(snap)
        baseline["adam_eps"] = 1e-5
        diffs = cg.diff_defaults(snap, baseline)
        assert diffs == {"adam_eps": (1e-5, 1e-8)}

    def test_added_field_reported_as_missing(self):
        snap = dict(cg.snapshot_defaults())
        snap["new_field"] = 42
        baseline = cg.snapshot_defaults()
        diffs = cg.diff_defaults(snap, baseline)
        assert diffs["new_field"] == ("<missing>", 42)

    def test_removed_field_reported_as_missing(self):
        snap = dict(cg.snapshot_defaults())
        del snap["lr"]
        baseline = cg.snapshot_defaults()
        diffs = cg.diff_defaults(snap, baseline)
        assert diffs["lr"] == (3e-4, "<missing>")


class TestCheck:
    def test_clean_baseline_returns_zero(self, tmp_path: Path):
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(cg.snapshot_defaults(), indent=2))
        assert cg.check(confirm=False, baseline_path=path) == 0

    def test_drift_no_confirm_raises(self, tmp_path: Path):
        path = tmp_path / "baseline.json"
        baseline = dict(cg.snapshot_defaults())
        baseline["adam_eps"] = 1e-5  # the Cycle-9 contamination value
        path.write_text(json.dumps(baseline, indent=2))
        with pytest.raises(SystemExit) as exc:
            cg.check(confirm=False, baseline_path=path)
        assert exc.value.code == 1

    def test_drift_with_confirm_returns_zero(self, tmp_path: Path):
        path = tmp_path / "baseline.json"
        baseline = dict(cg.snapshot_defaults())
        baseline["adam_eps"] = 1e-5
        path.write_text(json.dumps(baseline, indent=2))
        assert cg.check(confirm=True, baseline_path=path) == 0

    def test_missing_baseline_returns_zero(self, tmp_path: Path):
        # No baseline present → emit guidance, return 0 (don't block first-run).
        path = tmp_path / "does_not_exist.json"
        assert cg.check(confirm=False, baseline_path=path) == 0


class TestCommittedBaselineMatchesSource:
    """The shipped config_defaults.json must match TorchRLConfig defaults.

    If this test fails, either:
      (a) revert the unintended config.py edit, or
      (b) `python -m core.torchrl.config_gate --write-baseline` and commit.
    """

    def test_baseline_in_sync(self):
        baseline = cg.load_baseline()
        assert baseline is not None, "config_defaults.json missing"
        diffs = cg.diff_defaults(cg.snapshot_defaults(), baseline)
        assert diffs == {}, (
            "TorchRLConfig defaults drifted from config_defaults.json. "
            "Re-snapshot via `python -m core.torchrl.config_gate --write-baseline`. "
            f"Drift: {diffs}"
        )
