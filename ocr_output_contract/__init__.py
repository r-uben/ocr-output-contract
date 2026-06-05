"""ocr-output-contract — the canonical OCR output contract for the engine family.

A zero-runtime-dependency library that owns *where OCR output goes and what
shape its metadata takes*. Every sibling OCR engine CLI imports it so they emit
a byte-identical output STRUCTURE — consistency by construction rather than by
convention.

The bytes -> page-text boundary stays per-engine: an engine produces a
``list[str]`` of per-page markdown (plus optional figure bytes) and routes them
through this library, which then owns output-root resolution, input-relative
keying, subtree-mirroring layout, ``## Page N`` page assembly, clean-body
enforcement, dual-level ``metadata.json`` writers, figure naming and the
uniform exit-code policy.

Import the whole public surface from the top-level package::

    from ocr_output_contract import (
        DocMetadata, RootIndex, RunOutcome, Status,
        assemble_pages, doc_dir_for, figure_filename, figure_markdown_link,
        figures_dir_for, markdown_path_for, relative_key, resolve_output_root,
        sha256_checksum, utc_timestamp, write_doc_metadata,
        is_truncated, split_native_pages,
    )

The reusable conformance harness lives in :mod:`ocr_output_contract.conformance`.
"""

from __future__ import annotations

from ocr_output_contract.contract import (
    DEFAULT_OUTPUT_DIRNAME,
    FIGURES_DIRNAME,
    METADATA_FILENAME,
    METADATA_VERSION,
    PAGE_MARKER_RE,
    TRUNCATION_FINISH_REASONS,
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
    markdown_path_for,
    relative_key,
    resolve_output_root,
    sha256_checksum,
    split_native_pages,
    utc_timestamp,
    write_doc_metadata,
)

__version__ = "0.1.0"

__all__ = [
    # version
    "__version__",
    # constants
    "DEFAULT_OUTPUT_DIRNAME",
    "FIGURES_DIRNAME",
    "METADATA_FILENAME",
    "METADATA_VERSION",
    "PAGE_MARKER_RE",
    "TRUNCATION_FINISH_REASONS",
    # status / records
    "Status",
    "DocMetadata",
    "RootIndex",
    "RunOutcome",
    # path / key computation
    "resolve_output_root",
    "relative_key",
    "doc_dir_for",
    "markdown_path_for",
    "figures_dir_for",
    "figure_filename",
    "figure_markdown_link",
    # page assembly / splitting / truncation
    "assemble_pages",
    "split_native_pages",
    "is_truncated",
    # primitives
    "sha256_checksum",
    "utc_timestamp",
    # metadata writers
    "write_doc_metadata",
]
