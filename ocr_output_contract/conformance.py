"""Reusable conformance harness for the canonical OCR output contract.

Every engine repo runs this against its OWN produced output to prove it conforms
to the family-wide contract — consistency verified, not just asserted in prose.

The entry point is :func:`assert_conforms`, which takes a produced ``output_root``
and a list of :class:`ExpectedDoc` descriptors (the input-relative key, expected
page count, and expected terminal status per document) and asserts every
canonical invariant:

* tree shape ``<output_root>/<rel/dir>/<stem>/<stem>.md`` for each doc
* ``## Page N`` page markers present (one per expected page)
* NO YAML frontmatter on the markdown body
* per-doc ``metadata.json`` AND a root ``metadata.json`` both exist, with the
  ratified keys/types, the root index keyed by input-relative path
* figure files (when present) follow ``figures/figure_<N>_page<P>.png``
* failures recorded as ``status="failed"`` (no silent drops)

It raises :class:`ConformanceError` (an ``AssertionError`` subclass) on the first
violation, with a message naming the document and the broken invariant, so it
plugs straight into a ``pytest`` test in a consumer repo.

Usage in a consumer repo (e.g. ``tests/test_conformance.py``)::

    from ocr_output_contract.conformance import ExpectedDoc, assert_conforms

    def test_engine_conforms(tmp_path):
        # ... run the engine, producing output under `out_root` ...
        assert_conforms(
            out_root,
            [ExpectedDoc(rel_key="a/intro.pdf", pages=2, status="completed")],
        )

The harness is import- and network-free: it only reads files the engine wrote.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ocr_output_contract.contract import (
    FIGURES_DIRNAME,
    METADATA_FILENAME,
    PAGE_MARKER_RE,
    Status,
    doc_dir_for,
    markdown_path_for,
)

#: Per-file metadata entry keys required by the canon (root index + per-doc).
REQUIRED_ENTRY_KEYS: dict[str, type | tuple[type, ...]] = {
    "status": str,
    "checksum": str,
    "model": str,
    "backend": str,
    "processing_time": (int, float),
    "timestamp": str,
    "output_path": str,
    "pages": int,
}

#: Valid status values.
_VALID_STATUSES = {s.value for s in Status}

#: A leading YAML frontmatter block (``---`` ... ``---`` at the very top).
_FRONTMATTER_RE = re.compile(r"\A﻿?---\s*\n.*?\n---\s*\n", re.DOTALL)

#: Canonical figure filename pattern: ``figure_<N>_page<P>.png``.
_FIGURE_NAME_RE = re.compile(r"^figure_\d+_page\d+\.png$")

#: A markdown inline image link ``![alt](target)``. Captures the raw target,
#: which may carry an optional "title" (``(path "title")``) stripped before use.
_IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

#: ISO-8601 UTC timestamp (``...+00:00`` or ``...Z``). Loose but rejects naive.
_UTC_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.*(\+00:00|Z)$")


class ConformanceError(AssertionError):
    """Raised when produced output violates the canonical contract."""


@dataclass
class ExpectedDoc:
    """A document the engine was asked to process, with its expected outcome.

    ``rel_key`` is the input-relative POSIX path (the same identity the contract
    keys metadata on). ``pages`` is the expected number of ``## Page N`` sections
    for a completed/partial doc (ignored for ``failed``). ``status`` is the
    terminal status the engine should have recorded.
    """

    rel_key: str
    pages: int = 1
    status: str = Status.COMPLETED.value
    #: Optional expected (figure_number, page_number) figure files under figures/.
    figures: list[tuple[int, int]] = field(default_factory=list)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConformanceError(message)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.exists(), f"{label} missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover - defensive
        raise ConformanceError(f"{label} is not valid JSON: {path} ({exc})") from exc
    _require(isinstance(data, dict), f"{label} is not a JSON object: {path}")
    return data


def _check_entry(entry: dict[str, Any], label: str, expected_status: str | None) -> None:
    """Validate a per-file metadata entry's keys, types and status."""
    for key, typ in REQUIRED_ENTRY_KEYS.items():
        _require(key in entry, f"{label}: missing required key '{key}'")
        _require(
            isinstance(entry[key], typ),
            f"{label}: key '{key}' has wrong type (got {type(entry[key]).__name__}, want {typ})",
        )
    _require(
        entry["status"] in _VALID_STATUSES,
        f"{label}: invalid status {entry['status']!r} (want one of {_VALID_STATUSES})",
    )
    _require(
        entry["checksum"].startswith("sha256:"),
        f"{label}: checksum must be 'sha256:...' (got {entry['checksum']!r})",
    )
    _require(
        bool(_UTC_TS_RE.search(entry["timestamp"])),
        f"{label}: timestamp must be UTC ISO-8601 (got {entry['timestamp']!r})",
    )
    if expected_status is not None:
        _require(
            entry["status"] == expected_status,
            f"{label}: status is {entry['status']!r}, expected {expected_status!r}",
        )


def _check_markdown_body(md_path: Path, label: str, expected_pages: int) -> None:
    """Assert the markdown body is clean (no frontmatter) with ``## Page N`` markers."""
    _require(md_path.exists(), f"{label}: markdown file missing: {md_path}")
    text = md_path.read_text(encoding="utf-8")
    _require(
        _FRONTMATTER_RE.match(text) is None,
        f"{label}: markdown body has YAML frontmatter (forbidden by contract): {md_path}",
    )
    markers = PAGE_MARKER_RE.findall(text)
    _require(
        len(markers) >= 1,
        f"{label}: markdown has no '## Page N' markers: {md_path}",
    )
    if expected_pages > 0:
        _require(
            len(markers) == expected_pages,
            f"{label}: found {len(markers)} '## Page N' markers, expected {expected_pages}",
        )


def _is_external_link(target: str) -> bool:
    """True for links the harness should not resolve on disk (URLs, data URIs)."""
    lowered = target.lower()
    return "://" in lowered or lowered.startswith(("http:", "https:", "data:", "mailto:", "#"))


def _check_inline_image_links(md_path: Path, label: str) -> None:
    """Assert every markdown ``![..](target)`` resolves to a file on disk.

    Local image targets are resolved relative to the ``.md``'s own directory and
    must exist; a dangling inline link (an engine that wrote a body referencing
    an image it never saved, or saved under a different name) FAILS conformance.
    This is the gap that let marker's dangling-inline-link bug through: the
    harness previously checked figure *filenames* but never the *targets* the
    markdown actually pointed at. External targets (http/https/data/anchors) are
    not resolved.
    """
    text = md_path.read_text(encoding="utf-8")
    base = md_path.parent
    for raw in _IMAGE_LINK_RE.findall(text):
        target = raw.strip()
        # Strip an optional markdown title: ![](path "Title") or ![](path 'Title').
        for quote in ('"', "'"):
            qpos = target.find(f" {quote}")
            if qpos != -1:
                target = target[:qpos].strip()
                break
        # Strip surrounding angle brackets: ![](<path with spaces>).
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1].strip()
        if not target or _is_external_link(target):
            continue
        # Drop any URL fragment/query on a local path.
        target = target.split("#", 1)[0].split("?", 1)[0]
        if not target:
            continue
        resolved = (base / target).resolve()
        _require(
            resolved.exists(),
            f"{label}: dangling inline image link {raw.strip()!r} -> {resolved} "
            f"does not exist (markdown references a figure that was never written)",
        )


def _check_figures(doc_dir: Path, expected: list[tuple[int, int]], label: str) -> None:
    """Assert every figure file follows ``figure_<N>_page<P>.png`` naming."""
    figures_dir = doc_dir / FIGURES_DIRNAME
    if not figures_dir.exists():
        _require(
            not expected,
            f"{label}: expected figures {expected} but no figures/ dir at {figures_dir}",
        )
        return
    for fig in figures_dir.iterdir():
        if fig.name == ".gitkeep":
            continue
        _require(
            bool(_FIGURE_NAME_RE.match(fig.name)),
            f"{label}: figure {fig.name!r} violates 'figure_<N>_page<P>.png' naming",
        )
    from ocr_output_contract.contract import figure_filename

    for fig_num, page_num in expected:
        want = figures_dir / figure_filename(fig_num, page_num)
        _require(want.exists(), f"{label}: expected figure missing: {want}")


def assert_conforms(
    output_root: Path | str,
    expected: list[ExpectedDoc],
    *,
    require_failures_nonzero_exit: bool | None = None,
) -> None:
    """Assert a produced ``output_root`` conforms to the canonical contract.

    Raises :class:`ConformanceError` on the first violation.

    Parameters
    ----------
    output_root:
        The directory the engine wrote into (the resolved output root).
    expected:
        One :class:`ExpectedDoc` per document the engine was asked to process.
    require_failures_nonzero_exit:
        If given, asserts the relationship between recorded failures and the
        contract's exit policy: ``True`` means the run *did* exit nonzero (so at
        least one doc must be non-``completed``); ``False`` means it exited zero
        (so every doc must be ``completed``). Pass the consumer's observed
        ``RunOutcome.exit_code != 0`` here to verify the exit-code wiring.
    """
    root = Path(output_root)
    _require(root.is_dir(), f"output root is not a directory: {root}")

    # --- root index -------------------------------------------------------
    root_index = _load_json(root / METADATA_FILENAME, "root metadata.json")
    _require("version" in root_index, "root metadata.json: missing 'version'")
    files = root_index.get("files")
    _require(
        isinstance(files, dict),
        "root metadata.json: 'files' must be an object keyed by input-relative path",
    )

    any_non_completed = False

    # --- per-document invariants -----------------------------------------
    for doc in expected:
        rel_key = doc.rel_key
        label = f"doc[{rel_key}]"
        doc_dir = doc_dir_for(root, rel_key)
        md_path = markdown_path_for(doc_dir, rel_key)

        # Root index must record this doc, keyed by input-RELATIVE path.
        _require(
            rel_key in files,
            f"{label}: root metadata.json has no entry keyed by input-relative "
            f"path {rel_key!r} (basename keying is a contract violation)",
        )
        _check_entry(files[rel_key], f"{label} root-index entry", doc.status)

        # Per-doc sidecar.
        sidecar = _load_json(doc_dir / METADATA_FILENAME, f"{label} per-doc metadata.json")
        _require(
            sidecar.get("key") == rel_key,
            f"{label}: per-doc metadata 'key' is {sidecar.get('key')!r}, "
            f"expected input-relative {rel_key!r}",
        )
        _check_entry(sidecar, f"{label} per-doc entry", doc.status)

        # The two levels must stay in sync on terminal status.
        _require(
            sidecar["status"] == files[rel_key]["status"],
            f"{label}: per-doc status {sidecar['status']!r} disagrees with "
            f"root-index status {files[rel_key]['status']!r}",
        )

        if doc.status != Status.COMPLETED.value:
            any_non_completed = True

        # Markdown + figure invariants only meaningful for non-failed docs.
        if doc.status == Status.FAILED.value:
            # Failure recorded — that is the invariant; no markdown is required.
            continue

        _check_markdown_body(md_path, label, doc.pages)
        _check_inline_image_links(md_path, label)
        _check_figures(doc_dir, doc.figures, label)

    # --- exit-policy relationship ----------------------------------------
    if require_failures_nonzero_exit is not None:
        if require_failures_nonzero_exit:
            _require(
                any_non_completed,
                "exit-policy: caller reported a nonzero exit but no expected doc "
                "is non-'completed' (the failure must be recorded)",
            )
        else:
            _require(
                not any_non_completed,
                "exit-policy: caller reported a zero exit but an expected doc is "
                "non-'completed' (failures must drive a nonzero exit)",
            )
