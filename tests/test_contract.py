"""Unit tests for the contract's pure structural helpers."""

from __future__ import annotations

import re
from pathlib import Path

from ocr_output_contract import (
    DEFAULT_OUTPUT_DIRNAME,
    DocMetadata,
    RootIndex,
    RunOutcome,
    Status,
    assemble_pages,
    doc_dir_for,
    figure_filename,
    figure_markdown_link,
    figures_dir_for,
    is_truncated,
    is_within_output_root,
    iter_input_files,
    markdown_path_for,
    relative_key,
    resolve_output_root,
    run_fingerprint,
    sha256_checksum,
    split_native_pages,
    utc_timestamp,
    write_doc_metadata,
)

# --- output-root resolution ------------------------------------------------


def test_resolve_output_root_default_for_file(tmp_path: Path) -> None:
    f = tmp_path / "sub" / "doc.pdf"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    assert resolve_output_root(f) == f.parent / DEFAULT_OUTPUT_DIRNAME


def test_resolve_output_root_default_for_dir(tmp_path: Path) -> None:
    assert resolve_output_root(tmp_path) == tmp_path / DEFAULT_OUTPUT_DIRNAME


def test_resolve_output_root_override_verbatim(tmp_path: Path) -> None:
    override = tmp_path / "elsewhere"
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"x")
    assert resolve_output_root(f, override) == override


def test_output_root_never_required() -> None:
    # The signature must allow calling with only the input path.
    out = resolve_output_root(Path("/nope/doc.pdf"))
    assert out == Path("/nope") / DEFAULT_OUTPUT_DIRNAME


# --- input-relative keying (basename-collision fix) ------------------------


def test_relative_key_preserves_subtree(tmp_path: Path) -> None:
    root = tmp_path
    a = tmp_path / "a" / "intro.pdf"
    b = tmp_path / "b" / "intro.pdf"
    for p in (a, b):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    ka, kb = relative_key(a, root), relative_key(b, root)
    assert ka == "a/intro.pdf"
    assert kb == "b/intro.pdf"
    assert ka != kb  # the whole point: no collision


def test_relative_key_fallback_to_basename_when_outside(tmp_path: Path) -> None:
    outside = Path("/var/somewhere/file.pdf")
    assert relative_key(outside, tmp_path) == "file.pdf"


def test_doc_dir_and_md_path_mirror_subtree(tmp_path: Path) -> None:
    root = tmp_path / "ocr"
    doc_dir = doc_dir_for(root, "a/b/intro.pdf")
    assert doc_dir == root / "a" / "b" / "intro"
    assert markdown_path_for(doc_dir, "a/b/intro.pdf") == doc_dir / "intro.md"


# --- figure naming ---------------------------------------------------------


def test_figure_filename_normalises_to_png() -> None:
    assert figure_filename(2, 5) == "figure_2_page5.png"


def test_figures_dir_for(tmp_path: Path) -> None:
    assert figures_dir_for(tmp_path) == tmp_path / "figures"


def test_figure_markdown_link_resolves_relatively() -> None:
    link = figure_markdown_link(1, 3)
    assert link == "![Figure 1 (page 3)](./figures/figure_1_page3.png)"


# --- page assembly / splitting --------------------------------------------


def test_assemble_pages_uses_page_n_headers_no_frontmatter() -> None:
    body = assemble_pages(["first", "second"])
    assert body.startswith("## Page 1")
    assert "## Page 2" in body
    assert not body.startswith("---")  # no YAML frontmatter


def test_assemble_pages_empty() -> None:
    assert assemble_pages([]) == ""


def test_split_native_pages_roundtrips_with_assemble() -> None:
    pages = ["alpha text", "beta text", "gamma text"]
    blob = assemble_pages(pages)
    recovered = split_native_pages(blob)
    assert recovered == pages


def test_split_native_pages_preamble_folds_into_first() -> None:
    blob = "junk preamble\n\n## Page 1\n\nbody one"
    pages = split_native_pages(blob)
    assert pages == ["junk preamble\n\nbody one"]


def test_split_native_pages_no_markers_single_page() -> None:
    assert split_native_pages("no markers here") == ["no markers here"]


def test_split_native_pages_empty() -> None:
    assert split_native_pages("   ") == []


# --- truncation detection (engine-agnostic, duck-typed) --------------------


class _FakeFinish:
    def __init__(self, name: str) -> None:
        self.name = name


def test_is_truncated_on_finish_reason_enum() -> None:
    assert is_truncated(_FakeFinish("MAX_TOKENS"), parsed_pages=5, actual_pages=5)


def test_is_truncated_on_finish_reason_string() -> None:
    assert is_truncated("LENGTH", parsed_pages=1, actual_pages=1)


def test_is_truncated_on_page_shortfall() -> None:
    assert is_truncated(None, parsed_pages=3, actual_pages=10)


def test_not_truncated_when_complete() -> None:
    assert not is_truncated(_FakeFinish("STOP"), parsed_pages=10, actual_pages=10)


def test_not_truncated_when_page_count_unknown() -> None:
    assert not is_truncated(None, parsed_pages=2, actual_pages=0)


# --- primitives ------------------------------------------------------------


def test_sha256_checksum_prefix(tmp_path: Path) -> None:
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    cs = sha256_checksum(f)
    assert cs.startswith("sha256:")
    assert len(cs) == len("sha256:") + 64


def test_utc_timestamp_is_utc() -> None:
    ts = utc_timestamp()
    assert ts.endswith("+00:00") or ts.endswith("Z")
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", ts)


# --- metadata records ------------------------------------------------------


def test_doc_metadata_entry_keys() -> None:
    meta = DocMetadata(
        status=Status.COMPLETED,
        checksum="sha256:abc",
        model="m",
        backend="b",
        processing_time=1.234,
        timestamp=utc_timestamp(),
        output_path="/x.md",
        pages=3,
    )
    entry = meta.to_entry()
    assert entry["status"] == "completed"
    assert entry["processing_time"] == 1.23  # rounded
    assert "mode" not in entry  # only serialized when set
    assert "error" not in entry


def test_doc_metadata_mode_and_fallback_serialize() -> None:
    meta = DocMetadata(
        status=Status.COMPLETED,
        checksum="sha256:abc",
        model="m",
        backend="b",
        processing_time=1.0,
        timestamp=utc_timestamp(),
        output_path="/x.md",
        pages=1,
        mode="auto",
        fell_back_from_whole_pdf=True,
    )
    entry = meta.to_entry()
    assert entry["mode"] == "auto"
    assert entry["fell_back_from_whole_pdf"] is True


def test_doc_metadata_error_serializes() -> None:
    meta = DocMetadata(
        status=Status.FAILED,
        checksum="sha256:abc",
        model="m",
        backend="b",
        processing_time=0.0,
        timestamp=utc_timestamp(),
        output_path="",
        pages=0,
        error="boom",
    )
    assert meta.to_entry()["error"] == "boom"


def test_write_doc_metadata_is_self_describing(tmp_path: Path) -> None:
    import json

    meta = DocMetadata(
        status=Status.COMPLETED,
        checksum="sha256:abc",
        model="m",
        backend="b",
        processing_time=1.0,
        timestamp=utc_timestamp(),
        output_path="/x.md",
        pages=1,
    )
    path = write_doc_metadata(tmp_path, "a/intro.pdf", meta)
    data = json.loads(path.read_text())
    assert data["key"] == "a/intro.pdf"
    assert data["version"] == "1"
    assert data["status"] == "completed"


# --- root index ------------------------------------------------------------


def _completed_meta(output_path: str, **kw) -> DocMetadata:
    base = dict(
        status=Status.COMPLETED,
        checksum="sha256:abc",
        model="m",
        backend="b",
        processing_time=1.0,
        timestamp=utc_timestamp(),
        output_path=output_path,
        pages=1,
    )
    base.update(kw)
    return DocMetadata(**base)


def test_root_index_records_and_reloads(tmp_path: Path) -> None:
    # is_completed now requires the recorded output to still exist on disk.
    md = tmp_path / "a" / "intro" / "intro.md"
    md.parent.mkdir(parents=True)
    md.write_text("## Page 1\n\nx\n", encoding="utf-8")
    idx = RootIndex(tmp_path)
    idx.record("a/intro.pdf", _completed_meta("a/intro/intro.md"))
    # Re-open and confirm persisted + keyed by input-relative path.
    idx2 = RootIndex(tmp_path)
    assert idx2.get("a/intro.pdf") is not None
    assert idx2.is_completed("a/intro.pdf", "sha256:abc")
    assert not idx2.is_completed("a/intro.pdf", "sha256:different")


def test_is_completed_requires_output_on_disk(tmp_path: Path) -> None:
    md = tmp_path / "a" / "intro" / "intro.md"
    md.parent.mkdir(parents=True)
    md.write_text("x", encoding="utf-8")
    idx = RootIndex(tmp_path)
    idx.record("a/intro.pdf", _completed_meta("a/intro/intro.md"))
    assert idx.is_completed("a/intro.pdf", "sha256:abc")
    md.unlink()  # output deleted -> must reprocess
    assert not RootIndex(tmp_path).is_completed("a/intro.pdf", "sha256:abc")


def test_is_completed_invalidates_on_fingerprint_change(tmp_path: Path) -> None:
    md = tmp_path / "doc" / "doc.md"
    md.parent.mkdir(parents=True)
    md.write_text("x", encoding="utf-8")
    idx = RootIndex(tmp_path)
    idx.record("doc.pdf", _completed_meta("doc/doc.md", fingerprint="fp-v1"))
    assert idx.is_completed("doc.pdf", "sha256:abc", fingerprint="fp-v1")
    # Different run config (model/task/prompt) -> reprocess.
    assert not idx.is_completed("doc.pdf", "sha256:abc", fingerprint="fp-v2")


def test_root_index_tolerates_corruption(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text("{ not json", encoding="utf-8")
    idx = RootIndex(tmp_path)  # must not raise
    assert idx.files == {}


# --- exit-code policy ------------------------------------------------------


def test_run_outcome_clean_is_zero() -> None:
    o = RunOutcome()
    o.add(Status.COMPLETED)
    o.add(Status.COMPLETED, output_path="/a.md")
    assert o.exit_code == 0
    assert not o.has_failures
    assert o.outputs == ["/a.md"]


def test_run_outcome_failure_is_nonzero() -> None:
    o = RunOutcome()
    o.add(Status.COMPLETED)
    o.add(Status.FAILED, detail="b/x.pdf")
    assert o.exit_code == 1
    assert o.has_failures
    assert "b/x.pdf" in o.failures


def test_run_outcome_partial_is_nonzero() -> None:
    o = RunOutcome()
    o.add(Status.PARTIAL, detail="c/y.pdf")
    assert o.exit_code == 1


# --- keying collision (same stem, different extension) ---------------------


def test_doc_dir_disambiguates_same_stem_different_ext() -> None:
    root = Path("/out")
    pdf_dir = doc_dir_for(root, "a/foo.pdf")
    png_dir = doc_dir_for(root, "a/foo.png")
    assert pdf_dir != png_dir  # no on-disk collision
    assert pdf_dir == root / "a" / "foo"  # PDF keeps the clean canon folder
    assert png_dir == root / "a" / "foo_png"
    assert markdown_path_for(pdf_dir, "a/foo.pdf") != markdown_path_for(png_dir, "a/foo.png")
    # Metadata keys never collided (full relative path incl. extension):
    assert relative_key(Path("/s/a/foo.pdf"), Path("/s")) != relative_key(
        Path("/s/a/foo.png"), Path("/s")
    )


# --- output-root exclusion (resolved path, NOT by 'ocr' name) --------------


def test_is_within_output_root_uses_resolved_paths(tmp_path: Path) -> None:
    out = tmp_path / "ocr"
    (out / "doc").mkdir(parents=True)
    assert is_within_output_root(out, out)
    assert is_within_output_root(out / "doc" / "doc.md", out)
    assert not is_within_output_root(tmp_path / "input.pdf", out)


def test_iter_input_files_excludes_output_root(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "paper.pdf").write_bytes(b"x")
    out = tmp_path / "ocr"
    (out / "paper").mkdir(parents=True)
    (out / "paper" / "paper.md").write_text("## Page 1\n\nx", encoding="utf-8")
    found = {p.name for p in iter_input_files(tmp_path, out, {".pdf"})}
    assert found == {"paper.pdf"}  # the engine's own outputs are NOT re-ingested


def test_iter_input_files_does_not_exclude_by_ocr_name(tmp_path: Path) -> None:
    # A real input under a path component literally named 'ocr' (the user's own
    # .../toolkits/ocr/... tree) MUST still be discovered (the old name-match bug
    # would process ZERO files here).
    scan = tmp_path / "toolkits" / "ocr" / "papers"
    scan.mkdir(parents=True)
    (scan / "real.pdf").write_bytes(b"x")
    out = scan / "ocr"
    out.mkdir()
    found = {p.name for p in iter_input_files(scan, out, {".pdf"})}
    assert found == {"real.pdf"}


# --- run fingerprint -------------------------------------------------------


def test_run_fingerprint_stable_and_config_sensitive() -> None:
    a = run_fingerprint("gpt", "api")
    assert a == run_fingerprint("gpt", "api") and a.startswith("fp:")
    assert run_fingerprint("gpt", "api") != run_fingerprint("gpt2", "api")  # model
    assert run_fingerprint("gpt", "api") != run_fingerprint("gpt", "ollama")  # backend
    assert run_fingerprint("gpt", "api", task="t") != run_fingerprint("gpt", "api")  # task
    assert run_fingerprint("gpt", "api", task=None) != run_fingerprint("gpt", "api", task="")


# --- assemble_pages explicit numbering -------------------------------------


def test_assemble_pages_explicit_page_numbers() -> None:
    body = assemble_pages(["one", "two"], page_numbers=[3, 5])
    assert "## Page 3" in body and "## Page 5" in body
    assert "## Page 1" not in body
