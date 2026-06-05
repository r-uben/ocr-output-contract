"""A synthetic 'fake engine' that drives the contract exactly like a real one.

It stands in for any of the 7 OCR engine CLIs: it does NO real OCR, takes
per-page markdown text (and optional figure bytes) as input, and routes
everything through ``ocr_output_contract`` so the produced tree is, by
construction, canonical. The package's tests run the conformance harness against
this engine's output to prove the contract + harness agree.

This is the reference for how a consumer engine wires the library:
the bytes -> page-text boundary is the engine's job (here we just accept text);
EVERYTHING about where output goes comes from the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ocr_output_contract import (
    DocMetadata,
    RootIndex,
    RunOutcome,
    Status,
    assemble_pages,
    doc_dir_for,
    figure_filename,
    figures_dir_for,
    markdown_path_for,
    relative_key,
    resolve_output_root,
    sha256_checksum,
    utc_timestamp,
    write_doc_metadata,
)

MODEL = "fake-model-1"
BACKEND = "fake-engine"


@dataclass
class FakeJob:
    """One source file to 'process'.

    ``pages`` is the per-page markdown the engine 'recovered'. ``page_errors``
    (1-indexed page -> message) marks pages that failed: a job with some pages
    and some errors is PARTIAL; a job with no pages is FAILED. ``figures`` is a
    list of (figure_number, page_number, bytes) the engine extracted.
    """

    src: Path
    pages: list[str]
    page_errors: dict[int, str] = field(default_factory=dict)
    figures: list[tuple[int, int, bytes]] = field(default_factory=list)
    fail: bool = False
    error: str | None = None


def _status_for(job: FakeJob) -> Status:
    if job.fail or not job.pages:
        return Status.FAILED
    if job.page_errors:
        return Status.PARTIAL
    return Status.COMPLETED


def run_engine(
    scan_root: Path,
    jobs: list[FakeJob],
    output_dir: Path | None = None,
) -> RunOutcome:
    """Process ``jobs`` and emit canonical output. Returns the RunOutcome.

    Mirrors how a real engine's processor would use the contract end to end.
    """
    output_root = resolve_output_root(scan_root, output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    index = RootIndex(output_root)
    outcome = RunOutcome()

    for job in jobs:
        rel_key = relative_key(job.src, scan_root)
        doc_dir = doc_dir_for(output_root, rel_key)
        status = _status_for(job)
        checksum = sha256_checksum(job.src) if job.src.exists() else "sha256:" + "0" * 64

        if status is Status.FAILED:
            meta = DocMetadata(
                status=status,
                checksum=checksum,
                model=MODEL,
                backend=BACKEND,
                processing_time=0.01,
                timestamp=utc_timestamp(),
                output_path="",
                pages=0,
                error=job.error or "fake failure",
            )
            doc_dir.mkdir(parents=True, exist_ok=True)
            write_doc_metadata(doc_dir, rel_key, meta)
            index.record(rel_key, meta)
            outcome.add(status, detail=rel_key)
            continue

        # Write figures first so the markdown can reference resolvable paths.
        if job.figures:
            figures_dir = figures_dir_for(doc_dir)
            figures_dir.mkdir(parents=True, exist_ok=True)
            for fig_num, page_num, data in job.figures:
                (figures_dir / figure_filename(fig_num, page_num)).write_bytes(data)

        body = assemble_pages(job.pages)
        md_path = markdown_path_for(doc_dir, rel_key)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(body, encoding="utf-8")

        meta = DocMetadata(
            status=status,
            checksum=checksum,
            model=MODEL,
            backend=BACKEND,
            processing_time=0.02,
            timestamp=utc_timestamp(),
            output_path=str(md_path),
            pages=len(job.pages),
            error="; ".join(job.page_errors.values()) or None,
        )
        write_doc_metadata(doc_dir, rel_key, meta)
        index.record(rel_key, meta)
        outcome.add(status, detail=rel_key if status is not Status.COMPLETED else None,
                    output_path=str(md_path))

    return outcome
