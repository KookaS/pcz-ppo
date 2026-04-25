"""Tests for paper_build.cmd_audit (Paper Semantic Integrity v2)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make paper/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paper_build


@pytest.fixture
def fake_paper(tmp_path, monkeypatch):
    """Build a minimal paper directory with .tex, generated/, and .bib, then
    redirect ``paper_build``'s module-level paths into it. Returns the dir.
    """
    fake = tmp_path / "paper"
    fake.mkdir()
    (fake / "generated").mkdir()

    monkeypatch.setattr(paper_build, "PAPER_DIR", fake)
    monkeypatch.setattr(paper_build, "GENERATED_DIR", fake / "generated")
    monkeypatch.setattr(paper_build, "MAIN_TEX", fake / "pcz_ppo.tex")
    monkeypatch.setattr(paper_build, "BIB", fake / "references.bib")
    return fake


class TestOrphanFragmentDetection:
    def test_clean_state_returns_zero(self, fake_paper):
        # Two fragments, both referenced — no orphans.
        (fake_paper / "pcz_ppo.tex").write_text(r"\cnum{foo} and \cnum{bar}")
        (fake_paper / "references.bib").write_text("")
        for name in ("foo", "bar"):
            (fake_paper / "generated" / f"{name}.tex").write_text("1\n")
        assert paper_build.cmd_audit(strict=True) == 0

    def test_orphan_fragment_flagged_strict(self, fake_paper):
        (fake_paper / "pcz_ppo.tex").write_text(r"\cnum{foo}")
        (fake_paper / "references.bib").write_text("")
        # baz is committed but never referenced
        for name in ("foo", "baz"):
            (fake_paper / "generated" / f"{name}.tex").write_text("1\n")
        assert paper_build.cmd_audit(strict=True) == 1

    def test_orphan_fragment_advisory_default(self, fake_paper):
        # strict=False → return 0 even with orphans (still prints them).
        (fake_paper / "pcz_ppo.tex").write_text(r"\cnum{foo}")
        (fake_paper / "references.bib").write_text("")
        for name in ("foo", "baz"):
            (fake_paper / "generated" / f"{name}.tex").write_text("1\n")
        assert paper_build.cmd_audit(strict=False) == 0

    def test_commented_out_cnum_doesnt_keep_fragment_alive(self, fake_paper):
        # %\cnum{baz} is a comment — baz should still be flagged as orphan.
        (fake_paper / "pcz_ppo.tex").write_text(r"\cnum{foo} % \cnum{baz}")
        (fake_paper / "references.bib").write_text("")
        for name in ("foo", "baz"):
            (fake_paper / "generated" / f"{name}.tex").write_text("1\n")
        assert paper_build.cmd_audit(strict=True) == 1

    def test_escaped_percent_does_not_strip_real_reference(self, fake_paper):
        # `\%\cnum{baz}` — the % is escaped, so \cnum{baz} is real.
        (fake_paper / "pcz_ppo.tex").write_text(r"\cnum{foo} \%\cnum{baz}")
        (fake_paper / "references.bib").write_text("")
        for name in ("foo", "baz"):
            (fake_paper / "generated" / f"{name}.tex").write_text("1\n")
        # Both referenced — no orphans.
        assert paper_build.cmd_audit(strict=True) == 0

    def test_no_generated_dir(self, fake_paper):
        # Remove generated/ entirely — should not crash.
        (fake_paper / "generated").rmdir()
        (fake_paper / "pcz_ppo.tex").write_text("Hello world")
        (fake_paper / "references.bib").write_text("")
        assert paper_build.cmd_audit(strict=True) == 0

    def test_manifest_json_not_treated_as_fragment(self, fake_paper):
        # _manifest.json sits alongside *.tex but must be ignored.
        (fake_paper / "pcz_ppo.tex").write_text(r"\cnum{foo}")
        (fake_paper / "references.bib").write_text("")
        (fake_paper / "generated" / "foo.tex").write_text("1\n")
        (fake_paper / "generated" / "_manifest.json").write_text("{}")
        assert paper_build.cmd_audit(strict=True) == 0
