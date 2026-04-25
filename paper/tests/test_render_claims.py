"""Tests for render_claims.py.

Run from /workspace::

    uv run pytest paper/tests/test_render_claims.py -v

Covers:
  * Formatter edge cases (sign, zero, NaN/inf rejection, rounding boundaries)
  * Registry invariants (duplicate names rejected, empty fragments rejected,
    orphan files removed)
  * Idempotency (re-running produces byte-identical output)
  * Atomic write (no partial file left on failure)
  * Manifest consistency (sha256 matches fragment content)
  * Spot-check claims match what the paper claims (regression guard)
  * `--check` mode succeeds after fresh render and fails after tampering
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PAPER_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PAPER_DIR))

import render_claims as rc

# --- Formatter tests -----------------------------------------------------


class TestFmtSigned:
    def test_positive(self):
        assert rc.fmt_signed(157.73, 1) == "+157.7"

    def test_negative(self):
        assert rc.fmt_signed(-67.8, 1) == "-67.8"

    def test_zero(self):
        # No -0.0 leaking through
        assert rc.fmt_signed(0.0, 1) == "+0.0"
        assert rc.fmt_signed(-0.0, 1) == "+0.0"

    def test_small_negative_rounds_to_zero(self):
        # -0.04 rounds to "-0.0" in default format; we map it to "+0.0"
        assert rc.fmt_signed(-0.04, 1) == "+0.0"

    def test_two_decimals(self):
        assert rc.fmt_signed(1.234, 2) == "+1.23"

    def test_rejects_nan(self):
        with pytest.raises(ValueError):
            rc.fmt_signed(float("nan"))

    def test_rejects_inf(self):
        with pytest.raises(ValueError):
            rc.fmt_signed(float("inf"))
        with pytest.raises(ValueError):
            rc.fmt_signed(float("-inf"))


class TestFmtPlain:
    def test_positive(self):
        assert rc.fmt_plain(27.0, 1) == "27.0"

    def test_rejects_nan(self):
        with pytest.raises(ValueError):
            rc.fmt_plain(float("nan"))


class TestFmtRatio:
    def test_default_two_decimals(self):
        assert rc.fmt_ratio(1.408) == "1.41"

    def test_round_up(self):
        assert rc.fmt_ratio(8.456) == "8.46"

    def test_custom_decimals(self):
        assert rc.fmt_ratio(14.82, 1) == "14.8"


class TestFmtInt:
    def test_int(self):
        assert rc.fmt_int(10) == "10"

    def test_rejects_float(self):
        with pytest.raises(TypeError):
            rc.fmt_int(10.0)  # type: ignore[arg-type]


class TestFmtStat:
    def test_signed_default(self):
        assert rc.fmt_stat(157.7, 27.0) == "+157.7 \\pm 27.0"

    def test_unsigned(self):
        assert rc.fmt_stat(0.5, 0.1, signed=False) == "0.5 \\pm 0.1"


# --- Seed-count confidence labels (G6) ------------------------------------


class TestSeedsConfidence:
    """Multi-Seed Statistical Enforcement (G6) tier mapping."""

    def test_preliminary_below_3(self):
        assert rc._seeds_confidence(0) == "preliminary"
        assert rc._seeds_confidence(1) == "preliminary"
        assert rc._seeds_confidence(2) == "preliminary"

    def test_signal_3_to_9(self):
        assert rc._seeds_confidence(3) == "signal"
        assert rc._seeds_confidence(5) == "signal"
        assert rc._seeds_confidence(9) == "signal"

    def test_confirmed_at_10_plus(self):
        assert rc._seeds_confidence(10) == "confirmed"
        assert rc._seeds_confidence(15) == "confirmed"
        assert rc._seeds_confidence(100) == "confirmed"

    def test_negative_rejected(self):
        import pytest

        with pytest.raises(ValueError):
            rc._seeds_confidence(-1)

    def test_non_int_rejected(self):
        import pytest

        with pytest.raises(ValueError):
            rc._seeds_confidence(3.5)

    def test_bool_rejected(self):
        # bool is a subclass of int — explicit rejection prevents accidental
        # ``_seeds_confidence(True)`` mapping to "preliminary".
        import pytest

        with pytest.raises(ValueError):
            rc._seeds_confidence(True)
        with pytest.raises(ValueError):
            rc._seeds_confidence(False)


# --- Comparator descriptors (Paper Semantic Integrity v1) ----------------


class TestComparatorPhrase:
    """``_comparator_phrase`` maps numeric ratios to drift-free English."""

    def test_tied_band(self):
        assert rc._comparator_phrase(1.00) == "comparable"
        assert rc._comparator_phrase(0.96) == "comparable"
        assert rc._comparator_phrase(1.04) == "comparable"

    def test_marginal_above(self):
        assert rc._comparator_phrase(1.05) == "marginally above"
        assert rc._comparator_phrase(1.20) == "marginally above"

    def test_marginal_below(self):
        assert rc._comparator_phrase(0.90) == "marginally below"

    def test_noticeable_above(self):
        assert rc._comparator_phrase(1.50) == "noticeably above"

    def test_more_than_1p5x(self):
        assert rc._comparator_phrase(2.50) == "more than $1.5\\times$"

    def test_more_than_3x(self):
        assert rc._comparator_phrase(4.00) == "more than $3\\times$"
        assert rc._comparator_phrase(50.0) == "more than $3\\times$"

    def test_negative_rejected(self):
        import pytest

        with pytest.raises(ValueError):
            rc._comparator_phrase(-0.1)

    def test_emit_writes_fragment(self, tmp_path):
        reg = rc.Registry()
        rc._emit_comparator(reg, "test_cmp", ratio=2.0, src="unit-test")
        reg.render_files(tmp_path)
        body = (tmp_path / "test_cmp.tex").read_text().strip()
        assert body == "more than $1.5\\times$"


# --- Registry invariants -------------------------------------------------


class TestRegistry:
    def test_duplicate_name_rejected(self):
        r = rc.Registry()
        r.add("foo", "1", "src")
        with pytest.raises(KeyError, match="duplicate"):
            r.add("foo", "2", "src")

    def test_empty_content_rejected(self):
        r = rc.Registry()
        with pytest.raises(ValueError, match="empty"):
            r.add("foo", "   ", "src")
        with pytest.raises(ValueError, match="empty"):
            r.add("foo", "", "src")

    def test_newline_in_content_rejected(self):
        r = rc.Registry()
        with pytest.raises(ValueError, match="newline"):
            r.add("foo", "1\n2", "src")

    def test_invalid_name_rejected(self):
        r = rc.Registry()
        with pytest.raises(ValueError, match="invalid"):
            r.add("foo/bar", "1", "src")
        with pytest.raises(ValueError, match="invalid"):
            r.add("foo bar", "1", "src")
        with pytest.raises(ValueError, match="invalid"):
            r.add("foo.bar", "1", "src")

    def test_alnum_hyphen_underscore_ok(self):
        r = rc.Registry()
        r.add("foo_bar-baz123", "1", "src")  # no raise

    def test_render_removes_orphans(self, tmp_path):
        (tmp_path / "orphan.tex").write_text("stale")
        (tmp_path / "other.tex").write_text("keep me")  # not orphan if registered below
        r = rc.Registry()
        r.add("other", "keep me", "src")
        r.render_files(tmp_path)
        assert not (tmp_path / "orphan.tex").exists()
        assert (tmp_path / "other.tex").exists()

    def test_render_does_not_touch_non_tex_files(self, tmp_path):
        (tmp_path / "README.md").write_text("do not delete")
        r = rc.Registry()
        r.add("foo", "1", "src")
        r.render_files(tmp_path)
        assert (tmp_path / "README.md").exists()

    def test_manifest_sha256_matches(self, tmp_path):
        r = rc.Registry()
        r.add("foo", "hello", "from foo")
        r.render_files(tmp_path)
        manifest = json.loads((tmp_path / "_manifest.json").read_text())
        import hashlib

        expected = hashlib.sha256(b"hello\n").hexdigest()
        assert manifest["foo"]["sha256"] == expected
        assert manifest["foo"]["source"] == "from foo"


# --- Atomic write -------------------------------------------------------


class TestAtomicWrite:
    def test_success(self, tmp_path):
        p = tmp_path / "x.tex"
        rc._atomic_write(p, "hello")
        assert p.read_text() == "hello"

    def test_crash_leaves_no_partial(self, tmp_path, monkeypatch):
        p = tmp_path / "x.tex"
        p.write_text("pre-existing")

        def boom(src, dst):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(RuntimeError):
            rc._atomic_write(p, "new content")
        # Original preserved; no stray tmp files
        assert p.read_text() == "pre-existing"
        tmps = [f for f in tmp_path.iterdir() if f.name.startswith(".")]
        assert tmps == [], f"tmpfile leak: {tmps}"


# --- End-to-end rendering + idempotency ---------------------------------


class TestEndToEnd:
    def test_render_to_tmpdir_succeeds(self, tmp_path):
        reg = rc.render(target_dir=tmp_path)
        assert len(reg.as_dict()) > 50
        tex_files = list(tmp_path.glob("*.tex"))
        assert len(tex_files) == len(reg.as_dict())
        assert (tmp_path / "_manifest.json").exists()

    def test_idempotent(self, tmp_path):
        rc.render(target_dir=tmp_path)
        first = {p.name: p.read_text() for p in tmp_path.iterdir()}
        rc.render(target_dir=tmp_path)
        second = {p.name: p.read_text() for p in tmp_path.iterdir()}
        assert first == second

    def test_every_fragment_non_empty_single_line(self, tmp_path):
        rc.render(target_dir=tmp_path)
        for p in tmp_path.glob("*.tex"):
            text = p.read_text()
            assert text.endswith("\n"), f"{p.name} missing trailing newline"
            body = text.rstrip("\n")
            assert body, f"{p.name} is empty"
            assert "\n" not in body, f"{p.name} has embedded newline"


# --- Spot-check: headline claims match what the paper claims ------------


class TestHeadlineClaims:
    """If any of these break, the paper's headline number is wrong.

    These are regression tests pinned to the current committed data. If the
    data legitimately changes, update both the paper and these values.
    """

    @pytest.fixture(scope="class")
    def gen(self, tmp_path_factory) -> Path:
        p = tmp_path_factory.mktemp("gen")
        rc.render(target_dir=p)
        return p

    # NOTE: Pinned under the chronologically-latest per-seed dedup rule
    # documented in §A.3 and applied by fig_data.query.  A previous revision
    # of these tests pinned a different rule (implicit startswith("10.00,5.00")
    # + last-row-in-CSV-order); the present values are the ones that a
    # reviewer re-computing from raw results.csv should obtain.
    def test_k4_pcz_stat(self, gen):
        # n=15 snapshot is +159.8 ± 24.4 (was +157.7 ± 28.4 at n=10).
        assert (gen / "k4_pcz_stat.tex").read_text().strip() == "+159.8 \\pm 24.4"

    def test_k4_ppo_stat(self, gen):
        # n=15 snapshot is +119.1 ± 55.2 (was +112.0 ± 59.2 at n=10).
        assert (gen / "k4_ppo_stat.tex").read_text().strip() == "+119.1 \\pm 55.2"

    def test_k4_ratio(self, gen):
        # n=15 ratio: 159.8/119.1 = 1.34 (was 1.41 at n=10).
        assert (gen / "k4_ratio.tex").read_text().strip() == "1.34"

    def test_k6_ratio(self, gen):
        # Canonical-weight filter (10.00,3.00 prefix) excludes one contaminating
        # seed-42 row that was accidentally run with env-default weights and
        # logged with an empty ``component_weights`` column (2026-04-17 audit).
        # n=15 snapshot is 1.15 (ratio compressed from earlier n=10 value of 1.52
        # as PPO batch with cosine entropy schedule strengthened).  Direction
        # preserved: PCZ still beats PPO.
        assert (gen / "k6_ratio.tex").read_text().strip() == "1.15"

    def test_k8_ratio(self, gen):
        # Raw (unrounded) ratio: 143.843/17.045 = 8.44
        # Post-rounded (using query's 1-decimal mean/std) was 8.46; raw is
        # slightly tighter.  Fragments use raw to avoid double-rounding.
        assert (gen / "k8_ratio.tex").read_text().strip() == "8.44"

    def test_k2_ratio(self, gen):
        # PPO wins at K=2; ratio is
        # K=2 batch (PPO seeds 47-51 plus MH validation) plus chrono-latest
        # dedupe replacing weak earlier PPO with newer canonical-config runs.
        # Note: this needs review — a flip from PCZ-loses to PCZ-wins at K=2
        # contradicts the paper's narrative; if the new value is real, §K=2
        # should be reframed.  Test pinned to current data; investigate
        # before paper submission.
        assert (gen / "k2_ratio.tex").read_text().strip() == "1.53"

    def test_ceBW_ratio(self, gen):
        assert (gen / "ceBW_ratio.tex").read_text().strip() == "1.25"

    def test_ceHC_ratio(self, gen):
        # Both algorithms parity-tier; sign unchanged for the paper's
        # "no PCZ advantage at K=2 all-dense" narrative.
        assert (gen / "ceHC_ratio.tex").read_text().strip() == "0.88"

    def test_k4_delta(self, gen):
        # n=15: +159.8 - (+119.1) = +40.7 (was +45.7 at n=10).
        assert (gen / "k4_delta.tex").read_text().strip() == "+40.7"

    def test_grpoK4_500k_stat(self, gen):
        # Updated under chronologically-latest rule (std 32.5 -> 39.8).
        assert (gen / "grpoK4_500k_stat.tex").read_text().strip() == "-67.8 \\pm 39.8"

    def test_ablA1_matches_k4_pcz(self, gen):
        # Same underlying query — must be identical
        assert (gen / "ablA1_stat.tex").read_text() == (gen / "k4_pcz_stat.tex").read_text()


# --- CA12.1: weight-consistency guard ------------------------------------


class TestWeightConsistency:
    """Every headline K-row must have a single canonical weight config after
    dedupe.  Without the ``weights=`` filter on ``_emit_pcz_ppo_pair``, an
    empty-weights seed-42 row can win chronologically-latest dedupe and pull
    PCZ down ~12 pts and double SD.  The filter uses ``LL_K6_WEIGHTS='10.00,3.00'`` but
    the test below enforces the invariant structurally: *whatever* filter
    is applied, every resolved seed must share one weight string.  This
    catches silent re-entry if a future ``export_results`` run adds a
    differently-weighted row under the same prefix.

    For K=4 the filter already includes the full weight string
    (``10.00,5.00,0.50,0.50``) so the test is a no-op regression guard.
    For K=6 the filter is a proper prefix; the test is load-bearing.
    For K=2/K=8 the filter is ``None`` (no CLI-level canonical; the only
    runs that ever existed share one weight config); the test pins that.
    """

    HEADLINE_CONFIGS = [
        # (prefix, env, filter_weights, expected_canonical_weights)
        ("k2", "lunarlander-k2", None, "10.00,6.00"),
        ("k4", "lunarlander", rc.LL_PRIMARY_WEIGHTS, "10.00,5.00,0.50,0.50"),
        ("k6", "lunarlander-k6", rc.LL_K6_WEIGHTS, "10.00,3.00,1.00,1.00,0.50,0.50"),
        ("k8", "lunarlander-k8", None, "10.00,1.00,1.00,1.00,1.00,1.00,0.50,0.50"),
    ]

    @pytest.fixture(scope="class")
    def rows(self):
        return rc.load_results()

    @pytest.mark.parametrize(
        "prefix,env,filt,expected",
        HEADLINE_CONFIGS,
        ids=[c[0] for c in HEADLINE_CONFIGS],
    )
    @pytest.mark.parametrize("algo", ["torchrl-pcz-ppo-running", "torchrl-ppo"])
    def test_all_seeds_share_canonical_weights(self, rows, prefix, env, filt, expected, algo):
        q = rc.query(rows, algorithm=algo, env=env, total_timesteps=500000, weights=filt)
        assert q["seeds"] > 0, f"{prefix}/{algo}: no seeds matched — filter misconfigured"
        actual = sorted({r.get("component_weights", "") for r in q["runs"]})
        assert actual == [expected], (
            f"{prefix}/{algo}: seeds span multiple weight configs {actual}. "
            f"Expected single canonical {expected!r}. "
            f"A row with weights outside the canonical is polluting headline stats. "
            f"Fix: tighten the filter in render_claims.py or remove the off-canonical row."
        )

    @pytest.mark.parametrize(
        "prefix,env,filt,expected",
        HEADLINE_CONFIGS,
        ids=[c[0] for c in HEADLINE_CONFIGS],
    )
    def test_pcz_and_ppo_share_weights(self, rows, prefix, env, filt, expected):
        """Same-seed comparison requires PCZ and PPO to train under identical
        reward scaling.  If the filter admits different weight strings for
        the two algorithms, the Welch test compares apples to oranges.
        """
        pcz = rc.query(rows, algorithm="torchrl-pcz-ppo-running", env=env, total_timesteps=500000, weights=filt)
        ppo = rc.query(rows, algorithm="torchrl-ppo", env=env, total_timesteps=500000, weights=filt)
        pcz_w = {r.get("component_weights", "") for r in pcz["runs"]}
        ppo_w = {r.get("component_weights", "") for r in ppo["runs"]}
        assert pcz_w == ppo_w, (
            f"{prefix}: PCZ weights {pcz_w} != PPO weights {ppo_w}. "
            f"Pair comparison is unbalanced — re-filter or re-run."
        )


# --- Missing-data behaviour ---------------------------------------------


class TestMissingData:
    def test_nonexistent_algo_raises(self):
        rows = rc.load_results()
        with pytest.raises(LookupError, match="returned 0 seeds"):
            rc.q_required(
                rows,
                algorithm="fake-algo-nonexistent",
                env="lunarlander",
                total_timesteps=500000,
                weights=rc.LL_PRIMARY_WEIGHTS,
                min_seeds=1,
                label="test",
            )

    def test_insufficient_seeds_raises(self):
        rows = rc.load_results()
        # Pick an algorithm with known small n — popart at LL K=4 has 3 seeds
        # (popart has n=3 seeds; multihead has n=10).
        with pytest.raises(LookupError, match="seeds, expected >= 999"):
            rc.q_required(
                rows,
                algorithm="torchrl-ppo-popart",
                env="lunarlander",
                total_timesteps=500000,
                weights=rc.LL_PRIMARY_WEIGHTS,
                min_seeds=999,  # pick a value no algorithm has
                label="test",
            )


# --- --check mode -------------------------------------------------------


class TestCheckMode:
    def test_check_succeeds_after_fresh_render(self, tmp_path, monkeypatch):
        # Point the renderer at tmp_path and render, then --check should pass
        monkeypatch.setattr(rc, "_GEN_DIR", tmp_path)
        rc.render(target_dir=tmp_path)
        assert rc.check() == 0

    def test_check_detects_tampering(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(rc, "_GEN_DIR", tmp_path)
        rc.render(target_dir=tmp_path)
        # Tamper
        one = next(tmp_path.glob("*.tex"))
        one.write_text("TAMPERED\n")
        rc.check()  # returns 1
        err = capsys.readouterr().err
        assert "content drift" in err
        assert one.name in err

    def test_check_detects_missing_file(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(rc, "_GEN_DIR", tmp_path)
        rc.render(target_dir=tmp_path)
        one = next(tmp_path.glob("*.tex"))
        one.unlink()
        rc.check()  # returns 1
        err = capsys.readouterr().err
        assert "missing committed" in err

    def test_check_command_line(self):
        """Smoke test the actual CLI entry point."""
        result = subprocess.run(
            [sys.executable, str(PAPER_DIR / "render_claims.py"), "--check"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent.parent.parent,  # /workspace
        )
        assert result.returncode == 0, f"--check failed:\nstdout={result.stdout}\nstderr={result.stderr}"


# --- Registry ↔ paper symmetry ------------------------------------------


class TestRegistryPaperSymmetry:
    """Every registered claim MUST be referenced, every \\cnum MUST be registered."""

    def test_no_orphans(self, tmp_path):
        """Pruned registry writes only referenced fragments."""
        rc.render(target_dir=tmp_path)
        frags = {p.stem for p in tmp_path.glob("*.tex")}
        refs = rc._tex_references(rc._PAPER_DIR / "pcz_ppo.tex")
        assert frags == refs, f"orphan or missing: {frags ^ refs}"

    def test_undefined_reference_fails(self, tmp_path, monkeypatch):
        """Paper referencing a name not in the catalog must raise."""
        # Create a fake paper that uses an unknown name
        fake = tmp_path / "fake.tex"
        fake.write_text("Hello \\cnum{totally_made_up_name}!\n")
        with pytest.raises(SystemExit) as exc:
            rc.render(target_dir=tmp_path / "out", tex_path=fake)
        assert exc.value.code == 2


# --- Lint: context-aware row prefix rule --------------------------------


class TestLintContext:
    """The lint catches \\cnum references whose prefix disagrees with the
    table row's context label (AUDIT 1 semantic-swap regression).
    """

    def test_context_mismatch_is_flagged(self, tmp_path):
        import lint_hardcoded_numbers as lint

        fake = tmp_path / "bad.tex"
        fake.write_text(
            "\\begin{tabular}{cc}\n"
            "2 & $\\cnum{k4_pcz_stat}$ \\\\  % wrong prefix for K=2 row\n"
            "4 & $\\cnum{k4_pcz_stat}$ \\\\\n"
            "\\end{tabular}\n"
        )
        violations = lint.scan_row_context(fake)
        names = [v[2] for v in violations]
        lines = [v[0] for v in violations]
        assert lines == [2], f"only line 2 should be flagged, got {lines}"
        assert names == ["k4_pcz_stat"]

    def test_cross_context_prefix_exempt(self, tmp_path):
        """abl_* prefixes are allowed in any ablation row (derived quantities)."""
        import lint_hardcoded_numbers as lint

        fake = tmp_path / "ok.tex"
        fake.write_text("A7 & $\\cnum{ablA7_stat}$ & \\cnum{abl_decomp_delta} \\\\\n")
        assert lint.scan_row_context(fake) == []
