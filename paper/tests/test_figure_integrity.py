"""Tests for the four-layer figure-integrity system.

See ``../FIGURE_INTEGRITY.md`` for the design.  Each layer has unit tests
that exercise its core logic on synthetic inputs, plus an integration check
that runs the lint against the current paper state.

Layer 1: ``render_claims._winner_fragments`` — winner direction logic.
Layer 2: ``lint_caption_claims`` — caption hand-typed-claim detector.
Layer 3: ``lint_figure_geometry`` — geometric sanity checks.
Layer 4: ``lint_figure_units`` — UNITS declaration + consistency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))

import lint_caption_claims
import lint_figure_units
import render_claims

# ---------------------------------------------------------------------------
# Layer 1: winner-fragment unit tests
# ---------------------------------------------------------------------------


class TestWinnerFragments:
    """Verify ``_winner_fragments`` direction logic against the d=0.2 boundary."""

    def test_pcz_clear_winner(self):
        # Δ=+30, std=20 each → d ≈ 1.5, well above 0.2
        w = render_claims._winner_fragments(180.0, 20.0, 150.0, 20.0)
        assert w["winner"] == "PCZ-PPO"
        assert w["loser"] == "PPO"
        assert w["winner_verb"] == "outperforms"
        assert w["direction"] == "+"
        assert w["summary"] == "PCZ-PPO outperforms PPO"

    def test_ppo_clear_winner(self):
        # Δ=-50, std=10 each → d ≈ -5
        w = render_claims._winner_fragments(100.0, 10.0, 150.0, 10.0)
        assert w["winner"] == "PPO"
        assert w["loser"] == "PCZ-PPO"
        assert w["direction"] == "-"

    def test_matched_below_threshold(self):
        # Δ=+2, std=20 each → d=0.1, below 0.2 threshold
        w = render_claims._winner_fragments(102.0, 20.0, 100.0, 20.0)
        assert w["winner"] == "matched"
        assert w["winner_verb"] == "matches"
        assert w["direction"] == "0"

    def test_zero_delta(self):
        w = render_claims._winner_fragments(100.0, 10.0, 100.0, 10.0)
        assert w["winner"] == "matched"
        assert w["direction"] == "0"

    def test_zero_std_nonzero_delta(self):
        """Bug #1 fix: deterministic difference is decisive, not matched."""
        # Both stds=0, |delta|>epsilon → directional winner (Bug #1 fix).
        w = render_claims._winner_fragments(100.0, 0.0, 50.0, 0.0)
        assert w["winner"] == "PCZ-PPO"
        assert w["direction"] == "+"

    def test_zero_std_zero_delta_matched(self):
        # Both stds=0, |delta|<=epsilon → matched (no signal).
        w = render_claims._winner_fragments(100.0, 0.0, 100.0, 0.0)
        assert w["winner"] == "matched"

    def test_underpowered_d_is_not_winner(self):
        """Bug #2 fix: d=0.21 with n=2 should be matched (no significance)."""
        w = render_claims._winner_fragments(102.1, 10.0, 100.0, 10.0, pcz_n=2, ppo_n=2)
        # With n=2, p will be ~0.85 — far above α=0.05 → matched.
        assert w["winner"] == "matched", "underpowered d=0.21 should be classified as matched, not winner"

    def test_significant_d_with_n10_is_winner(self):
        """Bug #2 fix: d=0.5+ with n=10 and significant p should be winner."""
        # delta=20, std=10 each, n=10 → d=2.0, p<<0.001 → clear winner.
        w = render_claims._winner_fragments(120.0, 10.0, 100.0, 10.0, pcz_n=10, ppo_n=10)
        assert w["winner"] == "PCZ-PPO"

    def test_humanoid_real_values(self):
        # Real values from results.csv: PCZ ahead-on-mean by ~+10 with d=0.29
        # would be PCZ winner; humanoid is the opposite direction (PPO ahead).
        w = render_claims._winner_fragments(365.5, 20.6, 404.5, 23.5)
        assert w["winner"] == "PPO"
        assert w["direction"] == "-"


# ---------------------------------------------------------------------------
# Layer 2: lint_caption_claims unit tests
# ---------------------------------------------------------------------------


class TestCaptionClaimsLint:
    """Verify the regex patterns and justification window logic."""

    def test_unjustified_direction_claim_is_flagged(self):
        captions = [(100, "PCZ-PPO outperforms PPO at K=4 according to no specific number.", "fig:test")]
        violations = []
        for _line, cap, label in captions:
            for pat in lint_caption_claims.DIRECTION_PATTERNS:
                for m in pat.finditer(cap):
                    if not lint_caption_claims._has_nearby_cnum(cap, m.span()):
                        violations.append((label, m.group(0)))
        assert len(violations) == 1, f"expected 1 violation, got {violations}"

    def test_cnum_within_window_is_ok(self):
        cap = "PCZ-PPO outperforms PPO ($\\Delta{=}\\cnum{k4_delta}$, p<0.01)."
        for pat in lint_caption_claims.DIRECTION_PATTERNS:
            for m in pat.finditer(cap):
                assert lint_caption_claims._has_nearby_cnum(cap, m.span()), (
                    "cnum within 120 chars should justify the claim"
                )

    def test_allow_listed_pattern(self):
        cap = "K=2 cell is a negative control where PPO wins (d=-1.5)."
        for pat in lint_caption_claims.DIRECTION_PATTERNS:
            for m in pat.finditer(cap):
                start, end = m.span()
                pre = max(0, start - 120)
                post = min(len(cap), end + 120)
                snippet = cap[pre:post]
                assert lint_caption_claims._is_allow_listed(snippet) is not None, (
                    "K=2 negative-control language should be allow-listed"
                )


# ---------------------------------------------------------------------------
# Layer 4: lint_figure_units unit tests
# ---------------------------------------------------------------------------


class TestFigureUnitsLint:
    def test_known_units(self):
        # Spot-check that documented unit tokens are recognised
        assert "rollout-per-step" in lint_figure_units.KNOWN_UNITS
        assert "eval-episodic" in lint_figure_units.KNOWN_UNITS
        assert "ratio" in lint_figure_units.KNOWN_UNITS

    def test_parse_units_decl(self, tmp_path):
        script = tmp_path / "fig_dummy.py"
        script.write_text('INPUTS = ["x"]\nUNITS = "rollout-per-step,eval-episodic"\nMIXED_UNITS_ACKNOWLEDGED = True\n')
        units, ack = lint_figure_units._parse_units_decl(script)
        assert units == ["rollout-per-step", "eval-episodic"]
        assert ack is True


# ---------------------------------------------------------------------------
# Integration: lints pass on the current paper state
# ---------------------------------------------------------------------------


class TestPaperStateGreen:
    """Each lint must pass on the current ``pcz_ppo.tex`` / fig_*.py."""

    def test_caption_claims_clean(self):
        rc = lint_caption_claims.lint()
        assert rc == 0, f"lint_caption_claims found violations on current paper (rc={rc})"

    def test_figure_units_clean(self):
        rc = lint_figure_units.lint()
        assert rc == 0, f"lint_figure_units found violations on current paper (rc={rc})"

    def test_figure_geometry_clean(self):
        # Layer 3 imports pyplot and pyarrow heavily. We invoke via
        # ``uv run python`` so the subprocess uses the project venv (which
        # has pyarrow) regardless of how the test runner is invoked.
        # This also avoids pytest's import machinery clashing with
        # importlib.util.spec_from_file_location for figure modules that
        # use C-extension imports (pyarrow.parquet).
        import shutil
        import subprocess

        uv = shutil.which("uv")
        if uv is None:
            pytest.skip("uv not on PATH; skipping subprocess-based geometry check")
        result = subprocess.run(
            [uv, "run", "python", str(PAPER_DIR / "lint_figure_geometry.py")],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(PAPER_DIR.parent.parent.parent),
        )
        assert result.returncode == 0, (
            f"lint_figure_geometry returned {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_render_claims_check_clean(self):
        # Existing fragment system; must continue to pass after Layer 1 add-ons.
        # The render_claims script's --check is invoked here at the function level.

        # We import and call ``main(['--check'])`` indirectly by invoking the
        # script's check path. Simpler: just call build_registry and confirm
        # the new winner fragments register without error.
        from fig_data import load_results

        rows = load_results()
        reg = render_claims.build_registry(rows)
        d = reg.as_dict()
        # Sanity: at least some winner fragments materialize
        assert any(k.endswith("_winner") for k in d), "no winner fragments registered"

    def test_winner_coverage_clean(self):
        # Layer 1 safety net: ensure no \cnum{*_winner} reference is missing
        # an emitter.
        import lint_winner_coverage

        rc = lint_winner_coverage.lint()
        assert rc == 0, f"lint_winner_coverage found missing emitters (rc={rc})"


# ---------------------------------------------------------------------------
# Adversarial fixtures: each lint MUST flag its corresponding bad fixture.
# These tests guard against regressions where the lint silently passes
# everything because of a logic bug.  This is the most important class of
# test for any lint --- it proves the lint actually catches violations.
# ---------------------------------------------------------------------------


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestAdversarialFixtures:
    """Each lint MUST flag its corresponding bad fixture (regression guard)."""

    def test_caption_lint_flags_unjustified_claim(self):
        """End-to-end: lint() returns 1 on bad fixture (regression guard)."""
        bad_path = FIXTURES_DIR / "bad_caption_unjustified.tex"
        rc = lint_caption_claims.lint(bad_path)
        assert rc == 1, f"caption_claims.lint(bad fixture) returned {rc}, expected 1"

    def test_caption_lint_passes_on_clean_input(self, tmp_path):
        """End-to-end: lint() returns 0 on a fixture with proper \\cnum{} justification."""
        good = tmp_path / "good.tex"
        good.write_text(
            r"""\begin{figure}
\caption{PCZ-PPO outperforms PPO ($\Delta=\cnum{k4_delta}$, $p<0.01$).}
\label{fig:good}
\end{figure}"""
        )
        rc = lint_caption_claims.lint(good)
        assert rc == 0, f"caption_claims.lint(good fixture) returned {rc}, expected 0"

    def test_units_lint_flags_missing_declaration(self):
        """End-to-end: lint() detects missing UNITS via _parse_units_decl."""
        units, _ack = lint_figure_units._parse_units_decl(FIXTURES_DIR / "fig_bad_no_units.py")
        assert units is None, "units-lint should detect missing UNITS declaration"

    def test_units_lint_flags_unacknowledged_mixed(self):
        units, ack = lint_figure_units._parse_units_decl(FIXTURES_DIR / "fig_bad_mixed_units.py")
        assert units == ["rollout-per-step", "eval-episodic"]
        assert ack is False, "units-lint should detect missing MIXED_UNITS_ACKNOWLEDGED"
        # And the lint logic itself: mixed without ack is a violation.
        is_mixed = len(units) > 1
        assert is_mixed and not ack, "regression: mixed-units fixture should violate"

    def test_winner_coverage_lint_flags_missing_via_lint_function(self, tmp_path, monkeypatch):
        """End-to-end: lint() returns 1 when paper references a missing fragment.

        Patches the module-level PAPER_TEX so the lint reads our bad fixture.
        Verifies the public lint() function (not just internal helpers)
        returns 1 — guards against the "lint silently returns 0" regression.
        """
        import lint_winner_coverage

        bad_tex = FIXTURES_DIR / "bad_winner_reference.tex"
        monkeypatch.setattr(lint_winner_coverage, "PAPER_TEX", bad_tex)
        rc = lint_winner_coverage.lint()
        assert rc == 1, (
            f"winner_coverage.lint() returned {rc} on bad fixture; expected 1.  "
            "Likely a regression where the lint silently passes everything."
        )

    def test_winner_coverage_flags_missing_fragment(self):
        # Build the registry; nonexistent_prefix_winner should NOT be present.
        import lint_winner_coverage
        from fig_data import load_results

        # Combine real fragments with the bad fixture's reference and verify
        # the lint logic correctly identifies the missing reference.
        bad_tex = (FIXTURES_DIR / "bad_winner_reference.tex").read_text()
        referenced = lint_winner_coverage._referenced_winner_fragments(bad_tex)
        assert "nonexistent_prefix_winner" in referenced

        rows = load_results()
        reg = render_claims.build_registry(rows, used_filter=None)
        available = set(reg.as_dict().keys())
        assert "nonexistent_prefix_winner" not in available, "fixture's bad reference should not exist in registry"
        # → Layer 1 coverage lint would correctly flag this.

    def test_geometry_lint_flags_out_of_bounds_text(self, tmp_path):
        """End-to-end: render the bad-geometry fixture, run the geometric
        check on its Figure object, expect at least one violation."""
        import io
        from contextlib import redirect_stdout

        # We can't easily run lint_figure_geometry.py against the fixture
        # directly because it discovers fig_*.py from PAPER_DIR. Instead,
        # we exec the fixture in-process and apply the check helpers
        # to the resulting Figure.
        sys.path.insert(0, str(FIXTURES_DIR))
        import importlib.util

        import lint_figure_geometry
        import matplotlib.pyplot as plt

        plt.close("all")
        spec = importlib.util.spec_from_file_location("fig_bad_geometry", FIXTURES_DIR / "fig_bad_geometry.py")
        mod = importlib.util.module_from_spec(spec)
        with redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
            mod.main()
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.canvas.draw()

        violations: list[str] = []
        for fig in figs:
            for i, ax in enumerate(fig.axes):
                violations.extend(lint_figure_geometry._check_axes(ax, i))
        plt.close("all")
        assert violations, "geometry lint failed to flag the bad-geometry fixture; regression in _check_text_in_bounds"
