# ocr-output-contract

The **canonical OCR output contract** for the OCR engine CLI family, plus a
**reusable conformance harness**. Every sibling engine CLI (gemini, mistral,
qwen, marker, nougat, glm, deepseek, socr) imports this library so they all emit
a **byte-identical output STRUCTURE** — consistency *by construction* rather than
by convention or copy-paste.

This library owns **where OCR output goes and what shape its metadata takes**. It
knows nothing about *how* OCR is performed: no model SDKs, no PDF parsers, no
network. It has **zero runtime dependencies**.

The contract is the one ratified in
`docs/plans/00-output-contract/DECISION.md` (the canon).

## The bytes -> page-text boundary

```
            ┌─────────────────────────┐        ┌────────────────────────────┐
  PDF /     │  ENGINE  (per-engine)   │        │  ocr-output-contract        │
  image  ─▶ │  bytes -> list[str]     │  ──▶   │  list[str] -> canonical tree │
  bytes     │  page text + fig bytes  │        │  + metadata + exit code      │
            └─────────────────────────┘        └────────────────────────────┘
```

Everything **left** of the boundary (rendering pages, calling a model, decoding
figure bytes) stays in each engine. Everything **right** of the boundary (output
root, layout, page assembly, metadata, figure naming, exit policy) comes from
this library and is therefore identical across all engines.

## The contract (what the structure looks like)

```
<input-parent>/ocr/                 # default root; `-o` overrides, never required
├── metadata.json                   # rolled-up index, keyed by INPUT-RELATIVE path
├── a/intro/                        # input subtree mirrored (no basename collision)
│   ├── intro.md                    # all pages, one file, `## Page N` headers, NO frontmatter
│   ├── metadata.json               # per-doc sidecar (self-describing: carries `key`)
│   └── figures/
│       └── figure_1_page2.png      # figure_<N>_page<P>.png, always normalised to PNG
└── b/intro/                        # same basename, distinct subtree -> distinct key
    ├── intro.md
    └── metadata.json
```

Invariants:

- **Output root.** Default `<input-parent>/ocr/` (one word, next to the input).
  `-o/--output-dir` overrides verbatim. No CLI may *require* `-o`.
- **Layout.** `<root>/<rel/dir>/<stem>/<stem>.md`, mirroring the input subtree,
  keyed on the **input-relative path** (not the basename).
- **Markdown body.** Clean content, pages under `## Page N` headers, **no YAML
  frontmatter** — all provenance lives in the JSON sidecars.
- **Metadata at both levels.** A per-doc `<stem>/metadata.json` and a rolled-up
  root `metadata.json`. Per-file entry:
  `{status, checksum:"sha256:…", model, backend, processing_time, timestamp:<UTC ISO-8601>, output_path, pages}`
  (plus optional `mode`, `fell_back_from_whole_pdf`, `error`). Root index adds
  `{version, files:{<relpath>: entry}}`. **Failures are recorded**
  (`status="failed"`). The two levels stay in sync.
- **Figures.** `figures/figure_<N>_page<P>.png`; links in the `.md` must resolve.
- **Exit codes.** Nonzero if *any* file/page fails; uniform across single-file
  and batch.

## Public API

Import everything from the top-level package:

```python
from ocr_output_contract import (
    # status + records
    Status, DocMetadata, RootIndex, RunOutcome,
    # path / key computation
    resolve_output_root, relative_key, doc_dir_for,
    markdown_path_for, figures_dir_for, figure_filename, figure_markdown_link,
    # page assembly / splitting / truncation (engine-agnostic structural helpers)
    assemble_pages, split_native_pages, is_truncated,
    # primitives
    sha256_checksum, utc_timestamp,
    # metadata writers
    write_doc_metadata,
    # constants
    DEFAULT_OUTPUT_DIRNAME, FIGURES_DIRNAME, METADATA_FILENAME,
    METADATA_VERSION, PAGE_MARKER_RE, TRUNCATION_FINISH_REASONS,
)
```

### Signatures

| Symbol | Signature / shape |
|---|---|
| `Status` | `StrEnum`: `COMPLETED`, `FAILED`, `PARTIAL` |
| `resolve_output_root` | `(input_path: Path, output_dir: Path \| None = None) -> Path` |
| `relative_key` | `(file_path: Path, scan_root: Path) -> str` (POSIX input-relative key) |
| `doc_dir_for` | `(output_root: Path, rel_key: str) -> Path` |
| `markdown_path_for` | `(doc_dir: Path, rel_key: str) -> Path` |
| `figures_dir_for` | `(doc_dir: Path) -> Path` |
| `figure_filename` | `(figure_number: int, page_number: int) -> str` |
| `figure_markdown_link` | `(figure_number: int, page_number: int) -> str` |
| `assemble_pages` | `(pages: list[str]) -> str` (joins under `## Page N`) |
| `split_native_pages` | `(markdown: str) -> list[str]` (inverse of `assemble_pages`) |
| `is_truncated` | `(finish_reason: Any, parsed_pages: int, actual_pages: int) -> bool` |
| `sha256_checksum` | `(path: Path) -> str` (returns `"sha256:<hex>"`) |
| `utc_timestamp` | `() -> str` (UTC ISO-8601) |
| `DocMetadata` | dataclass; `.to_entry() -> dict` |
| `write_doc_metadata` | `(doc_dir: Path, rel_key: str, meta: DocMetadata) -> Path` |
| `RootIndex` | `RootIndex(output_root)`; `.record(rel_key, meta)`, `.is_completed(rel_key, checksum)`, `.get(rel_key)`, `.save()` |
| `RunOutcome` | dataclass; `.add(status, detail=None, output_path=None)`, `.exit_code`, `.has_failures` |

## How an engine consumes it

The engine produces `list[str]` of per-page markdown (and optional figure bytes);
the library does the rest. End to end:

```python
from ocr_output_contract import (
    DocMetadata, RootIndex, RunOutcome, Status,
    assemble_pages, doc_dir_for, figure_filename, figures_dir_for,
    markdown_path_for, relative_key, resolve_output_root,
    sha256_checksum, utc_timestamp, write_doc_metadata,
)

def process(scan_root, sources, output_dir=None):
    out_root = resolve_output_root(scan_root, output_dir)   # never requires -o
    out_root.mkdir(parents=True, exist_ok=True)
    index = RootIndex(out_root)
    outcome = RunOutcome()

    for src in sources:
        rel_key = relative_key(src, scan_root)              # input-relative identity
        doc_dir = doc_dir_for(out_root, rel_key)

        try:
            pages = engine_ocr_to_page_text(src)            # <-- the only engine-specific call
        except Exception as exc:
            meta = DocMetadata(Status.FAILED, sha256_checksum(src), MODEL, BACKEND,
                               0.0, utc_timestamp(), "", 0, error=str(exc))
            doc_dir.mkdir(parents=True, exist_ok=True)
            write_doc_metadata(doc_dir, rel_key, meta)
            index.record(rel_key, meta)                     # failures MUST be recorded
            outcome.add(Status.FAILED, detail=rel_key)
            continue

        md_path = markdown_path_for(doc_dir, rel_key)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(assemble_pages(pages), encoding="utf-8")  # ## Page N, no frontmatter

        meta = DocMetadata(Status.COMPLETED, sha256_checksum(src), MODEL, BACKEND,
                           elapsed, utc_timestamp(), str(md_path), len(pages))
        write_doc_metadata(doc_dir, rel_key, meta)
        index.record(rel_key, meta)
        outcome.add(Status.COMPLETED, output_path=str(md_path))

    raise SystemExit(outcome.exit_code)                     # nonzero on any failure
```

Figures: write bytes to `figures_dir_for(doc_dir)/figure_filename(n, page)` (the
engine decodes/normalises bytes to PNG), and reference them via
`figure_markdown_link(n, page)` inside the page text before `assemble_pages`.

Whole-document single-call engines: recover pages with `split_native_pages(blob)`
and detect cut-off responses with `is_truncated(finish_reason, len(pages), actual_pages)`.

## Running the conformance suite in a consumer repo

`ocr_output_contract.conformance.assert_conforms` checks a produced output tree
against every canonical invariant. Add a test in any engine repo:

```python
# tests/test_conformance.py
from ocr_output_contract.conformance import ExpectedDoc, assert_conforms

def test_engine_conforms(tmp_path):
    scan_root = tmp_path / "inbox"; scan_root.mkdir()
    # ... write a tiny input, run YOUR engine producing output under out_root ...
    outcome = run_my_engine(scan_root)         # however your engine is invoked
    from ocr_output_contract import resolve_output_root
    out_root = resolve_output_root(scan_root)

    assert_conforms(
        out_root,
        [ExpectedDoc(rel_key="report.pdf", pages=3, status="completed")],
        require_failures_nonzero_exit=outcome.exit_code != 0,  # ties exit code to recorded status
    )
```

`assert_conforms(output_root, expected, *, require_failures_nonzero_exit=None)`
raises `ConformanceError` (an `AssertionError`) naming the document and the broken
invariant. `ExpectedDoc(rel_key, pages=1, status="completed", figures=[(n, p), ...])`
describes one document the engine was asked to process.

Run it:

```bash
uv run --no-sync python -m pytest -q
```

## Installation & version pinning

Until this is published to an index, consumers pin the git tag:

```toml
# pyproject.toml of a consuming engine
dependencies = [
    "ocr-output-contract @ git+https://github.com/r-uben/ocr-output-contract@v0.1.0",
]
```

Once on an index, prefer a compatible range:

```toml
dependencies = ["ocr-output-contract>=0.1,<0.2"]
```

**Semver:** **patch** = bug fix (no structural change); **minor** = additive
(new helper, new optional field) that keeps existing output byte-identical;
**major** = a change to the emitted contract (output shape, metadata schema,
keying). A major bump means every engine re-pins and re-runs conformance.

## Development

```bash
uv venv "$HOME/venvs/ocr-output-contract" --python 3.11
UV_PROJECT_ENVIRONMENT="$HOME/venvs/ocr-output-contract" uv pip install -e ".[dev]"
UV_PROJECT_ENVIRONMENT="$HOME/venvs/ocr-output-contract" uv run --no-sync python -m pytest -q
UV_PROJECT_ENVIRONMENT="$HOME/venvs/ocr-output-contract" uv run --no-sync ruff check .
```
