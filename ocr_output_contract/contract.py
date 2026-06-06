"""Canonical OCR output contract — pure, dependency-free output-shape logic.

This module implements the family-wide OCR output contract ratified in
``docs/plans/00-output-contract/DECISION.md``. It is intentionally free of any
engine-specific imports (no model SDKs, no PDF libraries) so it can be shared
verbatim by every sibling OCR engine CLI: structure is guaranteed identical
*by construction* because every engine emits through the same code.

Everything here is about *where bytes go and what shape metadata takes* — never
about *how OCR is performed*. The bytes -> page-text boundary stays engine-side:
each engine produces a ``list[str]`` of per-page markdown (and figure bytes) and
hands them to this library, which owns layout, page assembly, metadata, figure
naming and the exit-code policy.

Key guarantees:

* Default output root is ``<input-parent>/ocr/`` (``-o`` overrides verbatim;
  callers must never *require* ``-o``).
* Per-document layout mirrors the input subtree, keyed on the input-RELATIVE
  path (never the basename), fixing the family-wide basename-collision data-loss
  class: ``<root>/<rel/dir>/<stem>/<stem>.md``.
* All pages of a document live in one ``<stem>.md``, separated by ``## Page N``
  headers. The markdown body is clean — no YAML frontmatter; provenance lives
  only in the JSON sidecars.
* Metadata at BOTH levels: a per-document ``<stem>/metadata.json`` and a
  rolled-up root index ``metadata.json`` keyed by input-relative path. Failures
  are recorded (``status="failed"``). The two stay in sync (the root index is
  derived from / reconciled with the per-doc files).
* Figures are named ``figures/figure_<N>_page<P>.png`` under the doc folder.
* Exit policy is uniform across single-file and batch: nonzero if any
  file/page failed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: One-word default output-root folder, sitting next to the input. Mirrors the
#: existing ``Library/Papers/ocr/`` layout. NOT engine-specific.
DEFAULT_OUTPUT_DIRNAME = "ocr"

#: Schema version for the root index metadata document.
METADATA_VERSION = "1"

#: Filename for both the per-document sidecar and the rolled-up root index.
METADATA_FILENAME = "metadata.json"

#: Subfolder (under each doc folder) holding extracted figures.
FIGURES_DIRNAME = "figures"

#: Canonical ``sha256:``-prefixed sentinel recorded as the ``checksum`` of a
#: document whose input bytes could NOT be read (permission denied, deleted
#: mid-run, broken symlink). The schema requires a ``sha256:`` checksum even on
#: a failure record, so engines must record this (via :func:`failure_checksum`)
#: instead of ``None``/``""``. It can never equal a real digest, so a later
#: readable run gets a different checksum and reprocesses.
UNREADABLE_CHECKSUM = "sha256:" + "0" * 64

#: Matches a ``## Page N`` boundary header anywhere in a markdown blob. Used to
#: split a single-call OCR response back into per-page sections. Case-insensitive
#: and tolerant of leading whitespace / trailing text on the marker line.
PAGE_MARKER_RE = re.compile(r"^[ \t]*##[ \t]+Page[ \t]+(\d+)\b.*$", re.IGNORECASE | re.MULTILINE)

#: Finish-reason tokens (normalized to upper-case, non-alphanumerics stripped)
#: that indicate the model's output was cut off by a length/token limit. Used as
#: one of two truncation signals for auto-fallback. Membership-checked, so a
#: stray/unset finish_reason (e.g. a bare MagicMock) never matches.
TRUNCATION_FINISH_REASONS = {"MAXTOKENS", "MAX_TOKENS", "LENGTH", "MODELLENGTH"}


class Status(StrEnum):
    """Per-document / per-file processing status recorded in metadata.

    A ``StrEnum`` (Python 3.11+) so values serialize directly to JSON as plain
    strings; ``.value`` is used explicitly at every serialization site.
    """

    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


# ---------------------------------------------------------------------------
# Primitives: checksum, timestamp
# ---------------------------------------------------------------------------


def sha256_checksum(path: Path) -> str:
    """Return the ``sha256:<hexdigest>`` checksum of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def safe_checksum(path: Path) -> str | None:
    """Like :func:`sha256_checksum`, but return ``None`` instead of raising.

    Discovery can yield a file that is unreadable by the time it is processed
    (permission denied, deleted/replaced mid-run, a broken symlink that passed
    discovery). Engines should use this in the idempotency pre-check so an
    unreadable input is recorded as a per-file FAILURE and the batch CONTINUES,
    rather than an ``OSError`` propagating and aborting the whole run (the
    SYS-02 "one bad file aborts the batch" failure mode).
    """
    try:
        return sha256_checksum(path)
    except OSError:
        return None


def failure_checksum(path: Path) -> str:
    """Return a valid ``sha256:`` checksum for a FAILURE record.

    The real digest if the file is readable, else :data:`UNREADABLE_CHECKSUM`.
    Engines must use this when recording a ``status=failed`` doc whose input may
    be unreadable, so the failure entry still satisfies the metadata schema
    (``checksum`` starting with ``sha256:``) rather than writing ``None``/``""``
    and failing the conformance harness.
    """
    return safe_checksum(path) or UNREADABLE_CHECKSUM


def utc_timestamp() -> str:
    """Return the current time as a UTC ISO-8601 string (e.g. ``...+00:00``)."""
    return datetime.now(UTC).isoformat()


def run_fingerprint(
    model: str,
    backend: str,
    task: str | None = None,
    prompt: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Return a stable ``fp:<hexdigest>`` fingerprint of the run configuration.

    The fingerprint captures the parameters that change *what output a given
    input produces*: the model, the backend, an optional task selector, an
    optional prompt, and any engine-specific output-affecting flags passed via
    ``extra`` (e.g. ``{"raw": True, "dpi": 200, "analyze_figures": False,
    "pdf_mode": "auto"}``). It is stored in :class:`DocMetadata` and consulted
    by :meth:`RootIndex.is_completed` so that re-running an input under a
    different model / task / prompt / flag reprocesses it instead of silently
    reusing a cached result keyed only on the input checksum.

    Order- and whitespace-stable: fields are joined with a NUL separator and
    ``None`` is encoded distinctly from the empty string, so ``task=None`` and
    ``task=""`` do not collide. ``extra`` is serialized with sorted keys and
    ``repr`` values, so key order is irrelevant and ``True``/``1``/``"1"`` stay
    distinct. The long prompt is hashed verbatim, so prompt drift invalidates
    the cache without bloating metadata.
    """
    parts = [
        model or "",
        backend or "",
        "\x00" if task is None else task,
        "\x00" if prompt is None else prompt,
    ]
    if extra:
        # Canonical JSON: key order irrelevant; bool/int/str/None stay distinct;
        # rejects NaN/Inf. Pass RESOLVED effective flags (incl. falsy values).
        parts.append(json.dumps(extra, sort_keys=True, separators=(",", ":"), allow_nan=False))
    h = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()
    return f"fp:{h}"


# ---------------------------------------------------------------------------
# Path / key computation
# ---------------------------------------------------------------------------


def resolve_output_root(
    input_path: Path,
    output_dir: Path | None = None,
) -> Path:
    """Resolve the output root directory per the canon.

    * If ``output_dir`` is given, it is used verbatim (``-o`` override).
    * Otherwise the default is ``<input-parent>/ocr/`` for a single file, or
      ``<input>/ocr/`` for a directory input — one word, next to the input.

    The directory is NOT created here; that is the writer's job.
    """
    if output_dir is not None:
        return Path(output_dir)
    if input_path.is_dir():
        return input_path / DEFAULT_OUTPUT_DIRNAME
    return input_path.parent / DEFAULT_OUTPUT_DIRNAME


def is_within_output_root(path: Path, output_root: Path) -> bool:
    """True if ``path`` is the resolved output root itself or nested under it.

    This is the contract's required guard against *output self-ingestion*: when
    the default output root sits inside the scanned tree (``<input>/ocr/``), a
    naive recursive glob re-discovers the engine's own ``.md``/figure outputs as
    fresh inputs on the next run, recursively polluting the tree.

    Comparison is on the RESOLVED (absolute, symlink-collapsed) paths, so it
    targets the *real* output directory — never a path component that merely
    happens to be literally named ``ocr``. This is the fix for BOTH the
    glm/deepseek output self-ingestion class AND the gemini/mistral/nougat
    "exclude any component named 'ocr'" class (which would silently process ZERO
    files under a user's own ``.../toolkits/ocr/...`` tree).

    Engines MUST call this (directly, or via :func:`iter_input_files`) during
    input discovery to skip anything under the resolved output root.
    """
    try:
        resolved = Path(path).resolve()
        root = Path(output_root).resolve()
    except OSError:  # pragma: no cover - defensive (e.g. broken symlink loop)
        return False
    return resolved == root or root in resolved.parents


def iter_input_files(
    scan_root: Path,
    output_root: Path,
    suffixes: Iterable[str] | None = None,
) -> Iterator[Path]:
    """Walk ``scan_root`` recursively, yielding input files, excluding outputs.

    This is the REQUIRED discovery primitive for every engine. It:

    * recurses ``scan_root`` for files (so batch/nested inputs are found);
    * excludes anything at or under the RESOLVED ``output_root`` (via
      :func:`is_within_output_root`), pruning whole subtrees so the engine never
      re-ingests its own ``.md``/figure outputs and never descends into them;
    * optionally filters by suffix (case-insensitive, leading dot optional);
    * yields in sorted order for deterministic, reproducible batches.

    ``scan_root`` may be a single file (yielded if it matches and is not itself
    inside ``output_root``) or a directory. Suffixes like ``{".pdf", "png"}``
    are normalized to lower-case with a leading dot. Pass ``None`` to accept all
    files.

    Engines replace any hand-rolled ``rglob('*')`` / ``glob('**/*')`` discovery
    with this call so the output-root exclusion is uniform across the family.
    """
    scan = Path(scan_root)
    wanted = _normalize_suffixes(suffixes)

    if scan.is_file():
        if not is_within_output_root(scan, output_root) and _suffix_ok(scan, wanted):
            yield scan
        return

    if not scan.is_dir():
        return

    root_resolved = Path(output_root).resolve()
    matches: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(scan):
        here = Path(dirpath)
        # Prune the output-root subtree so we never descend into it.
        dirnames[:] = [
            d
            for d in dirnames
            if (here / d).resolve() != root_resolved and not _is_under(here / d, root_resolved)
        ]
        # If the current directory is itself the output root, skip its files too.
        if is_within_output_root(here, output_root):
            continue
        for name in filenames:
            fp = here / name
            if _suffix_ok(fp, wanted):
                matches.append(fp)
    yield from sorted(matches)


def _is_under(path: Path, resolved_root: Path) -> bool:
    """Resolved-path containment helper for :func:`iter_input_files` pruning."""
    try:
        resolved = Path(path).resolve()
    except OSError:  # pragma: no cover - defensive
        return False
    return resolved == resolved_root or resolved_root in resolved.parents


def _normalize_suffixes(suffixes: Iterable[str] | None) -> set[str] | None:
    """Normalize a suffix iterable to a lower-case, leading-dot set (or None)."""
    if suffixes is None:
        return None
    out: set[str] = set()
    for s in suffixes:
        s = s.strip().lower()
        if not s:
            continue
        out.add(s if s.startswith(".") else f".{s}")
    return out


def _suffix_ok(path: Path, wanted: set[str] | None) -> bool:
    if wanted is None:
        return True
    return path.suffix.lower() in wanted


def relative_key(file_path: Path, scan_root: Path) -> str:
    """Compute the canonical input-RELATIVE key for a source file.

    The key is the POSIX-style path of ``file_path`` relative to ``scan_root``.
    This is the identity used everywhere (metadata keys, subtree mirroring),
    NOT the basename — two ``intro.pdf`` files in different subdirectories get
    distinct keys, fixing the basename-collision data-loss class.

    ``scan_root`` is the directory that was scanned: the input directory for a
    batch, or the file's parent for a single-file run. If ``file_path`` is not
    under ``scan_root`` (which should not happen in normal use) the basename is
    used as a safe fallback.
    """
    fp = Path(file_path)
    root = Path(scan_root)
    try:
        rel = fp.resolve().relative_to(root.resolve())
    except ValueError:
        return fp.name
    return rel.as_posix()


def doc_dir_for(output_root: Path, rel_key: str) -> Path:
    """Return the per-document folder for an input-relative key.

    Layout: ``<root>/<rel/dir>/<stem>/`` — the input subtree is mirrored and a
    final folder named after the file stem holds the aggregated output. The
    relative directory portion is preserved so basenames in different
    subdirectories never collide.

    Same-stem / different-extension inputs in the SAME directory (e.g.
    ``a/foo.pdf`` and ``a/foo.png``) would otherwise collide on the ``<stem>``
    folder. PDFs (the canonical primary input) keep the clean ``<stem>`` folder;
    any other extension is disambiguated as ``<stem>_<ext>`` so the two never
    overwrite each other on disk. Metadata keys never collide regardless: they
    are the full input-relative path including extension (:func:`relative_key`).
    """
    rel = Path(rel_key)
    stem = rel.stem
    suffix = rel.suffix.lower().lstrip(".")
    folder = stem if suffix in ("", "pdf") else f"{stem}_{suffix}"
    return Path(output_root) / rel.parent / folder


def markdown_path_for(doc_dir: Path, rel_key: str) -> Path:
    """Return the aggregated markdown path ``<doc_dir>/<stem>.md``."""
    stem = Path(rel_key).stem
    return Path(doc_dir) / f"{stem}.md"


def figures_dir_for(doc_dir: Path) -> Path:
    """Return the figures subfolder for a document."""
    return Path(doc_dir) / FIGURES_DIRNAME


def figure_filename(figure_number: int, page_number: int) -> str:
    """Return the canonical figure filename ``figure_<N>_page<P>.png``.

    Figures are always normalised to PNG so links resolve and the family is
    uniform regardless of the embedded image's native format.
    """
    return f"figure_{figure_number}_page{page_number}.png"


def figure_markdown_link(figure_number: int, page_number: int) -> str:
    """Return a relative markdown image link that resolves from the ``.md``.

    The ``.md`` lives in the doc folder and figures live in ``figures/`` beside
    it, so a ``figures/<name>`` relative link resolves correctly.
    """
    name = figure_filename(figure_number, page_number)
    return f"![Figure {figure_number} (page {page_number})](./{FIGURES_DIRNAME}/{name})"


# ---------------------------------------------------------------------------
# Page assembly / splitting (engine-agnostic structural helpers)
# ---------------------------------------------------------------------------


def assemble_pages(pages: list[str], page_numbers: list[int] | None = None) -> str:
    """Join per-page markdown into one document body with ``## Page N`` headers.

    By default pages are labeled by 1-based list position (``## Page 1``,
    ``## Page 2``, ...). When ``page_numbers`` is given it supplies the EXPLICIT
    label for each page, so an engine running a ``--pages 3,5`` subset can
    preserve *source* page identity (``## Page 3``, ``## Page 5``) instead of
    silently reindexing to 1..N. ``page_numbers`` must be the same length as
    ``pages``.

    Each page's text is placed under its ``## Page N`` header and pages are
    separated by a blank line. The result is a clean markdown body with NO YAML
    frontmatter — provenance lives in the JSON sidecar only.
    """
    if page_numbers is not None and len(page_numbers) != len(pages):
        raise ValueError(
            f"page_numbers length ({len(page_numbers)}) must match pages length ({len(pages)})"
        )
    sections: list[str] = []
    for idx, text in enumerate(pages):
        number = page_numbers[idx] if page_numbers is not None else idx + 1
        body = (text or "").strip()
        sections.append(f"## Page {number}\n\n{body}")
    return "\n\n".join(sections) + ("\n" if sections else "")


def split_native_pages(markdown: str) -> list[str]:
    """Split a single-call markdown blob into per-page sections.

    Splits on ``## Page N`` boundary headers (the marker line itself is dropped,
    since :func:`assemble_pages` re-adds canonical headers). Any text appearing
    before the first marker (preamble the model may emit) is prepended to the
    first page so no content is lost.

    Falls back to a single page (the whole blob) when no markers are present, so
    the document is still emitted as valid contract output rather than discarded.
    Returns an empty list only for empty/whitespace-only input.

    Engine-agnostic: it operates only on the marker convention this contract
    defines, so any engine that emits a whole-document blob can recover pages.
    """
    text = (markdown or "").strip()
    if not text:
        return []

    matches = list(PAGE_MARKER_RE.finditer(markdown))
    if not matches:
        # Model ignored the marker instruction: keep everything as one page.
        return [text]

    pages: list[str] = []
    preamble = markdown[: matches[0].start()].strip()
    for i, m in enumerate(matches):
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[body_start:body_end].strip()
        if i == 0 and preamble:
            body = (preamble + "\n\n" + body).strip() if body else preamble
        pages.append(body)
    return pages


def _normalize_finish_reason(finish_reason: Any) -> str:
    """Normalize a finish_reason (enum, str, or None) to an upper-case token."""
    if finish_reason is None:
        return ""
    name = getattr(finish_reason, "name", None)
    raw = name if isinstance(name, str) else str(finish_reason)
    return re.sub(r"[^A-Za-z0-9]", "", raw).upper()


def is_truncated(
    finish_reason: Any,
    parsed_pages: int,
    actual_pages: int,
    recovered_page_numbers: list[int] | None = None,
) -> bool:
    """Return True if a whole-document response looks truncated (its tail dropped).

    Signals (no hardcoded page-count threshold):

    * **finish reason** — the model reports it stopped on a length/token limit
      (``MAX_TOKENS`` / ``LENGTH``). Unconditional truncation signal.
    * **missing tail** — when ``recovered_page_numbers`` is given (the numbers
      parsed from the ``## Page N`` markers, where ``N`` MUST be the *physical*
      PDF page number, not a ``1..k`` renumbering), truncation is inferred only
      when the END of the document is missing: ``max(valid recovered) <
      actual_pages``. Scattered internal gaps (legitimately BLANK pages a model
      omits) do NOT trigger — this fixes the costly false-positive where a blank
      cover/verso forced a needless full per-page re-OCR (~2x cost).

    A bare page-COUNT shortfall (``parsed_pages < actual_pages``) is deliberately
    NOT a truncation signal on its own: it cannot distinguish a dropped tail from
    blank pages. Pass ``recovered_page_numbers`` to enable the tail check.

    Caveat: genuinely blank *trailing* pages are indistinguishable from a dropped
    tail by page numbers alone; gate on an explicit end-of-document sentinel
    upstream if that matters. ``actual_pages <= 0`` disables the page-based
    signal. ``finish_reason`` is duck-typed (``.name`` str / plain string /
    ``None``), so no model SDK is required.
    """
    if _normalize_finish_reason(finish_reason) in TRUNCATION_FINISH_REASONS:
        return True
    if recovered_page_numbers is None or actual_pages <= 0:
        return False
    valid = {n for n in recovered_page_numbers if 1 <= n <= actual_pages}
    return max(valid, default=0) < actual_pages


# ---------------------------------------------------------------------------
# Metadata records
# ---------------------------------------------------------------------------


@dataclass
class DocMetadata:
    """Provenance for one source document, written to both metadata levels.

    Field set matches the ratified per-file schema:
    ``{status, checksum, model, backend, processing_time, timestamp,
    output_path, pages}``, plus engine-specific provenance fields that are only
    serialized when set: ``mode`` (the processing path actually used) and
    ``fell_back_from_whole_pdf`` (whether ``auto`` mode dropped from whole-PDF to
    page-by-page because the whole-PDF output was truncated).
    """

    status: Status
    checksum: str
    model: str
    backend: str
    processing_time: float
    timestamp: str
    output_path: str
    pages: int
    error: str | None = None
    #: Run-config fingerprint (model/backend/task/prompt) from :func:`run_fingerprint`.
    #: Lets :meth:`RootIndex.is_completed` invalidate when the run config changes.
    fingerprint: str | None = None
    #: Processing path actually used for this doc (engine-specific provenance).
    mode: str | None = None
    #: True when an auto run fell back from whole-PDF to per-page (non-silent).
    fell_back_from_whole_pdf: bool = False

    def to_entry(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dict (the per-file metadata entry)."""
        entry: dict[str, Any] = {
            "status": self.status.value,
            "checksum": self.checksum,
            "model": self.model,
            "backend": self.backend,
            "processing_time": round(self.processing_time, 2),
            "timestamp": self.timestamp,
            "output_path": self.output_path,
            "pages": self.pages,
        }
        if self.fingerprint is not None:
            entry["fingerprint"] = self.fingerprint
        if self.mode is not None:
            entry["mode"] = self.mode
            # Only meaningful alongside a mode; surfaced whenever a mode is known.
            entry["fell_back_from_whole_pdf"] = self.fell_back_from_whole_pdf
        if self.error is not None:
            entry["error"] = self.error
        return entry


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` as pretty JSON atomically (tmp file + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(str(tmp_path), str(path))


def write_doc_metadata(doc_dir: Path, rel_key: str, meta: DocMetadata) -> Path:
    """Write the per-document ``metadata.json`` sidecar beside its ``.md``.

    The per-doc sidecar describes that one document and additionally records its
    own input-relative ``key`` so it is self-describing for later reconciliation.
    """
    payload = {
        "version": METADATA_VERSION,
        "key": rel_key,
        **meta.to_entry(),
    }
    path = Path(doc_dir) / METADATA_FILENAME
    _atomic_write_json(path, payload)
    return path


class RootIndex:
    """The rolled-up root ``metadata.json`` index, keyed by input-relative path.

    Schema: ``{version, files: {<relpath>: <entry>}}``. Loads tolerantly (a
    corrupt or missing file starts fresh) and writes atomically. The root index
    is kept in sync with the per-doc sidecars: every per-doc write should be
    mirrored here via :meth:`record`.
    """

    def __init__(self, output_root: Path) -> None:
        self.output_root = Path(output_root)
        self.path = self.output_root / METADATA_FILENAME
        self._data: dict[str, Any] = {"version": METADATA_VERSION, "files": {}}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("files"), dict):
                    self._data = loaded
                    self._data.setdefault("version", METADATA_VERSION)
            except (json.JSONDecodeError, OSError):
                # Tolerant load: start fresh rather than crash on a bad index.
                self._data = {"version": METADATA_VERSION, "files": {}}

    @property
    def files(self) -> dict[str, Any]:
        return self._data.setdefault("files", {})

    def get(self, rel_key: str) -> dict[str, Any] | None:
        """Return the entry for an input-relative key, if present."""
        return self.files.get(rel_key)

    def is_completed(
        self,
        rel_key: str,
        checksum: str,
        *,
        fingerprint: str | None = None,
    ) -> bool:
        """True if ``rel_key`` may be safely skipped as already-processed.

        All of the following must hold for a skip (return True); otherwise the
        document needs (re)processing and this returns False:

        * an entry exists, recorded with ``status == "completed"``;
        * the recorded input ``checksum`` matches (input unchanged);
        * the recorded ``output_path`` still EXISTS on disk — a deleted output
          is never treated as completed, so it gets re-emitted (fixes the
          family-wide "index survives, .md deleted, doc silently skipped" bug);
        * if ``fingerprint`` is given, it matches the recorded run-config
          fingerprint — a re-run under a different model/task/prompt reprocesses
          instead of silently reusing the prior output. When the caller passes a
          fingerprint but the recorded entry has none (pre-0.1.1 index), the
          entry is treated as stale and reprocessed.

        ``output_path`` is resolved relative to the output root when stored as a
        relative path, so the check works whether the engine recorded an
        absolute or root-relative path.
        """
        entry = self.files.get(rel_key)
        if entry is None:
            return False
        if entry.get("status") != Status.COMPLETED.value:
            return False
        if entry.get("checksum") != checksum:
            return False
        if fingerprint is not None and entry.get("fingerprint") != fingerprint:
            return False
        return self._output_exists(entry.get("output_path"))

    def _output_exists(self, output_path: Any) -> bool:
        """True if the recorded markdown output still exists on disk."""
        if not output_path or not isinstance(output_path, str):
            return False
        p = Path(output_path)
        if not p.is_absolute():
            p = self.output_root / p
        return p.exists()

    def record(self, rel_key: str, meta: DocMetadata) -> None:
        """Record (or overwrite) a per-file entry and persist atomically."""
        self.files[rel_key] = meta.to_entry()
        self.save()

    def save(self) -> None:
        """Persist the index atomically."""
        _atomic_write_json(self.path, self._data)


# ---------------------------------------------------------------------------
# Exit-code policy
# ---------------------------------------------------------------------------


@dataclass
class RunOutcome:
    """Accumulates per-file/per-page outcomes to derive a uniform exit code.

    The policy is uniform across single-file and batch: the exit code is nonzero
    if ANY file or page failed (including partial documents). ``exit_code`` is 0
    only when everything that was attempted completed cleanly.
    """

    completed: int = 0
    failed: int = 0
    partial: int = 0
    failures: list[str] = field(default_factory=list)
    #: Absolute paths of markdown files written this run (for scripting / quiet).
    outputs: list[str] = field(default_factory=list)

    def add(
        self,
        status: Status,
        detail: str | None = None,
        output_path: str | None = None,
    ) -> None:
        if output_path:
            self.outputs.append(output_path)
        if status is Status.COMPLETED:
            self.completed += 1
        elif status is Status.PARTIAL:
            self.partial += 1
            if detail:
                self.failures.append(detail)
        else:
            self.failed += 1
            if detail:
                self.failures.append(detail)

    @property
    def has_failures(self) -> bool:
        return self.failed > 0 or self.partial > 0

    @property
    def exit_code(self) -> int:
        """0 if everything succeeded; 1 if any file/page failed or was partial."""
        return 1 if self.has_failures else 0
