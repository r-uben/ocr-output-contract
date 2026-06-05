"""End-to-end tests: drive the fake engine, then assert with the harness.

These prove (a) a contract-conformant engine passes the conformance harness and
(b) the harness actually catches each class of violation it claims to. No real
PDF or network is used: inputs are tiny byte files standing in for documents.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ocr_output_contract import resolve_output_root
from ocr_output_contract.conformance import (
    ConformanceError,
    ExpectedDoc,
    assert_conforms,
)
from tests.fake_engine import FakeJob, run_engine


def _src(scan_root: Path, rel: str) -> Path:
    p = scan_root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"fake-pdf-bytes-" + rel.encode())
    return p


# --- happy paths -----------------------------------------------------------


def test_single_completed_doc_conforms(tmp_path: Path) -> None:
    scan_root = tmp_path / "inbox"
    scan_root.mkdir()
    src = _src(scan_root, "report.pdf")
    outcome = run_engine(scan_root, [FakeJob(src=src, pages=["p1", "p2", "p3"])])

    out_root = resolve_output_root(scan_root)
    assert_conforms(
        out_root,
        [ExpectedDoc(rel_key="report.pdf", pages=3, status="completed")],
        require_failures_nonzero_exit=outcome.exit_code != 0,
    )
    assert outcome.exit_code == 0


def test_batch_with_basename_collision_conforms(tmp_path: Path) -> None:
    scan_root = tmp_path / "inbox"
    scan_root.mkdir()
    a = _src(scan_root, "a/intro.pdf")
    b = _src(scan_root, "b/intro.pdf")
    run_engine(scan_root, [FakeJob(src=a, pages=["x"]), FakeJob(src=b, pages=["y", "z"])])

    out_root = resolve_output_root(scan_root)
    assert_conforms(
        out_root,
        [
            ExpectedDoc(rel_key="a/intro.pdf", pages=1),
            ExpectedDoc(rel_key="b/intro.pdf", pages=2),
        ],
    )
    # Both keyed distinctly in the root index — the collision is fixed.
    index = json.loads((out_root / "metadata.json").read_text())
    assert set(index["files"]) == {"a/intro.pdf", "b/intro.pdf"}


def test_doc_with_figures_conforms(tmp_path: Path) -> None:
    scan_root = tmp_path / "inbox"
    scan_root.mkdir()
    src = _src(scan_root, "withfig.pdf")
    run_engine(
        scan_root,
        [FakeJob(src=src, pages=["p1", "p2"], figures=[(1, 1, b"PNGDATA"), (2, 2, b"PNGDATA")])],
    )
    out_root = resolve_output_root(scan_root)
    assert_conforms(
        out_root,
        [ExpectedDoc(rel_key="withfig.pdf", pages=2, figures=[(1, 1), (2, 2)])],
    )


def test_failed_doc_records_status_failed(tmp_path: Path) -> None:
    scan_root = tmp_path / "inbox"
    scan_root.mkdir()
    ok = _src(scan_root, "ok.pdf")
    bad = _src(scan_root, "bad.pdf")
    outcome = run_engine(
        scan_root,
        [FakeJob(src=ok, pages=["good"]), FakeJob(src=bad, pages=[], fail=True, error="kaboom")],
    )
    assert outcome.exit_code == 1  # failure drives nonzero exit

    out_root = resolve_output_root(scan_root)
    assert_conforms(
        out_root,
        [
            ExpectedDoc(rel_key="ok.pdf", pages=1, status="completed"),
            ExpectedDoc(rel_key="bad.pdf", status="failed"),
        ],
        require_failures_nonzero_exit=outcome.exit_code != 0,
    )


def test_partial_doc_records_status_partial(tmp_path: Path) -> None:
    scan_root = tmp_path / "inbox"
    scan_root.mkdir()
    src = _src(scan_root, "partial.pdf")
    outcome = run_engine(
        scan_root,
        [FakeJob(src=src, pages=["p1", ""], page_errors={2: "page 2 failed"})],
    )
    assert outcome.exit_code == 1

    out_root = resolve_output_root(scan_root)
    assert_conforms(
        out_root,
        [ExpectedDoc(rel_key="partial.pdf", pages=2, status="partial")],
        require_failures_nonzero_exit=True,
    )


def test_output_dir_override_conforms(tmp_path: Path) -> None:
    scan_root = tmp_path / "inbox"
    scan_root.mkdir()
    src = _src(scan_root, "doc.pdf")
    override = tmp_path / "custom-out"
    run_engine(scan_root, [FakeJob(src=src, pages=["a"])], output_dir=override)
    assert_conforms(override, [ExpectedDoc(rel_key="doc.pdf", pages=1)])
    assert (override / "doc" / "doc.md").exists()


# --- the harness must catch violations ------------------------------------


def test_harness_catches_missing_root_metadata(tmp_path: Path) -> None:
    scan_root = tmp_path / "inbox"
    scan_root.mkdir()
    src = _src(scan_root, "doc.pdf")
    run_engine(scan_root, [FakeJob(src=src, pages=["a"])])
    out_root = resolve_output_root(scan_root)
    (out_root / "metadata.json").unlink()
    with pytest.raises(ConformanceError, match="root metadata.json missing"):
        assert_conforms(out_root, [ExpectedDoc(rel_key="doc.pdf", pages=1)])


def test_harness_catches_basename_keying(tmp_path: Path) -> None:
    """An engine that keyed the root index by basename must be flagged."""
    scan_root = tmp_path / "inbox"
    scan_root.mkdir()
    src = _src(scan_root, "a/intro.pdf")
    run_engine(scan_root, [FakeJob(src=src, pages=["x"])])
    out_root = resolve_output_root(scan_root)
    # Corrupt the index to be basename-keyed (the bug the canon forbids).
    index = json.loads((out_root / "metadata.json").read_text())
    index["files"] = {"intro.pdf": index["files"]["a/intro.pdf"]}
    (out_root / "metadata.json").write_text(json.dumps(index))
    with pytest.raises(ConformanceError, match="input-relative"):
        assert_conforms(out_root, [ExpectedDoc(rel_key="a/intro.pdf", pages=1)])


def test_harness_catches_yaml_frontmatter(tmp_path: Path) -> None:
    scan_root = tmp_path / "inbox"
    scan_root.mkdir()
    src = _src(scan_root, "doc.pdf")
    run_engine(scan_root, [FakeJob(src=src, pages=["a"])])
    out_root = resolve_output_root(scan_root)
    md = out_root / "doc" / "doc.md"
    md.write_text("---\nsource: x\n---\n" + md.read_text())
    with pytest.raises(ConformanceError, match="frontmatter"):
        assert_conforms(out_root, [ExpectedDoc(rel_key="doc.pdf", pages=1)])


def test_harness_catches_missing_page_markers(tmp_path: Path) -> None:
    scan_root = tmp_path / "inbox"
    scan_root.mkdir()
    src = _src(scan_root, "doc.pdf")
    run_engine(scan_root, [FakeJob(src=src, pages=["a"])])
    out_root = resolve_output_root(scan_root)
    md = out_root / "doc" / "doc.md"
    md.write_text("no page markers at all\n")
    with pytest.raises(ConformanceError, match="Page N"):
        assert_conforms(out_root, [ExpectedDoc(rel_key="doc.pdf", pages=1)])


def test_harness_catches_bad_figure_name(tmp_path: Path) -> None:
    scan_root = tmp_path / "inbox"
    scan_root.mkdir()
    src = _src(scan_root, "doc.pdf")
    run_engine(scan_root, [FakeJob(src=src, pages=["a"], figures=[(1, 1, b"x")])])
    out_root = resolve_output_root(scan_root)
    figs = out_root / "doc" / "figures"
    (figs / "page1_img1.jpeg").write_bytes(b"y")  # native-ext name (the bug)
    with pytest.raises(ConformanceError, match="naming"):
        assert_conforms(out_root, [ExpectedDoc(rel_key="doc.pdf", pages=1)])


def test_harness_catches_wrong_status(tmp_path: Path) -> None:
    scan_root = tmp_path / "inbox"
    scan_root.mkdir()
    src = _src(scan_root, "doc.pdf")
    run_engine(scan_root, [FakeJob(src=src, pages=["a"])])
    out_root = resolve_output_root(scan_root)
    with pytest.raises(ConformanceError, match="expected 'failed'"):
        assert_conforms(out_root, [ExpectedDoc(rel_key="doc.pdf", status="failed")])


def test_harness_catches_exit_policy_mismatch(tmp_path: Path) -> None:
    scan_root = tmp_path / "inbox"
    scan_root.mkdir()
    src = _src(scan_root, "doc.pdf")
    run_engine(scan_root, [FakeJob(src=src, pages=["a"])])
    out_root = resolve_output_root(scan_root)
    # Claim a nonzero exit while every doc is completed -> contradiction.
    with pytest.raises(ConformanceError, match="exit-policy"):
        assert_conforms(
            out_root,
            [ExpectedDoc(rel_key="doc.pdf", pages=1, status="completed")],
            require_failures_nonzero_exit=True,
        )


def test_harness_catches_naive_timestamp(tmp_path: Path) -> None:
    scan_root = tmp_path / "inbox"
    scan_root.mkdir()
    src = _src(scan_root, "doc.pdf")
    run_engine(scan_root, [FakeJob(src=src, pages=["a"])])
    out_root = resolve_output_root(scan_root)
    index = json.loads((out_root / "metadata.json").read_text())
    index["files"]["doc.pdf"]["timestamp"] = "2026-06-05T10:00:00"  # naive, no tz
    (out_root / "metadata.json").write_text(json.dumps(index))
    with pytest.raises(ConformanceError, match="UTC ISO-8601"):
        assert_conforms(out_root, [ExpectedDoc(rel_key="doc.pdf", pages=1)])
