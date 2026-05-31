"""Table-aware extraction for PDF documents.

Detects tables in a PDF and rewrites the page text so that each table is
replaced by a delimited markdown block. Surrounding prose is preserved
verbatim, so downstream consumers (chunkers, indexers) can still work
with a single text stream while having unambiguous markers for table
boundaries.

Block format
------------

    @@@TABLE_BEGIN id="t1" page="3" title="TABELA I" rows="25" cols="6"@@@
    | ALTITUDE | PRESSAO | TOLERANCIA |
    | --- | --- | --- |
    | 0 | 1.013 | ±20 |
    ...
    @@@TABLE_END@@@

Marker design constraints

- ``@@@`` is improbable in aviation regulation prose.
- The marker payload (id/page/title/rows/cols) is metadata for the
  chunker; the chunker can rebuild a "table chunk" from these alone.
- Markdown (``|``-pipe rows + ``---`` separator) survives the existing
  ``TextCleaner`` whitespace and separator rules — see
  ``pipeline/text_cleaner.py`` for the audit.
- Detection requires real borders (PyMuPDF.find_tables, pdfplumber
  ``lines`` strategy). Borderless ``text``-strategy detection is
  intentionally avoided because the dry-run study showed it inflates
  false positives in linear prose.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from loguru import logger

# ---------------------------------------------------------------------------
# Markers — the chunker imports these constants instead of duplicating the
# raw strings, so they stay in sync.
# ---------------------------------------------------------------------------

TABLE_BEGIN_PREFIX = "@@@TABLE_BEGIN"
TABLE_END = "@@@TABLE_END@@@"

TABLE_BEGIN_RE = re.compile(
    r"@@@TABLE_BEGIN(?P<attrs>[^@\n]*)@@@",
)
TABLE_BLOCK_RE = re.compile(
    r"@@@TABLE_BEGIN(?P<attrs>[^@\n]*)@@@\n?(?P<body>.*?)\n?@@@TABLE_END@@@",
    re.DOTALL,
)

# Capture title in the lines preceding a detected table bbox.
_TITLE_RE = re.compile(
    r"(TABELA|Tabela|TAB\.|QUADRO|Quadro)\s+[IVXLC0-9]+",
    re.MULTILINE,
)

# Minimum table shape — below this we treat it as a 2-cell layout artifact.
_MIN_ROWS = 3
_MIN_COLS = 2

# Heuristic title-window: how many trailing lines above the bbox to scan
# when looking for "TABELA X" / "Tabela X" / "Quadro X".
_TITLE_WINDOW_LINES = 15


@dataclass
class _TableHit:
    """One detected table on a page (bbox in PDF coordinates + rows)."""
    bbox: Tuple[float, float, float, float]
    rows: List[List[Optional[str]]]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_text_with_tables(pdf_content: bytes, max_ctx_chars: int = 200) -> Optional[str]:
    """Extract page text with tables replaced by ``@@@TABLE_*@@@`` blocks.

    Args:
        pdf_content: Raw PDF bytes.
        max_ctx_chars: Max chars of trailing prose retained before the
            table marker to give the downstream chunker some narrative
            context (used for the title heuristic too). Set to 0 to
            disable context capture.

    Returns:
        The full document text, page-joined, or ``None`` if no PDF
        backend can open the bytes. Tables that fail extraction are
        kept as plain text — degradation is graceful.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not available; table-aware extraction skipped")
        return None

    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
    except Exception as exc:
        logger.warning(f"PyMuPDF failed to open PDF bytes: {exc}")
        return None

    plumb_pdf = _open_plumber_safely(pdf_content)
    plumb_pages = plumb_pdf.pages if plumb_pdf is not None else None

    table_id = 0
    out_pages: List[str] = []

    try:
        for page_no, page in enumerate(doc, start=1):
            tables = _detect_pymupdf(page)
            if not tables and plumb_pages is not None and page_no - 1 < len(plumb_pages):
                tables = _detect_pdfplumber(plumb_pages[page_no - 1])

            if not tables:
                out_pages.append(page.get_text() or "")
                continue

            tables.sort(key=lambda t: t.bbox[1])
            page_text_parts: List[str] = []
            cursor_y = page.rect.y0

            for hit in tables:
                table_id += 1
                _, y0, _, y1 = hit.bbox

                pre_clip = fitz.Rect(page.rect.x0, cursor_y, page.rect.x1, y0)
                pre_text = (page.get_text(clip=pre_clip) or "").rstrip()

                title = _find_title_above(pre_text, _TITLE_WINDOW_LINES)
                page_text_parts.append(pre_text)
                page_text_parts.append(
                    _render_block(
                        table_id=table_id,
                        page_no=page_no,
                        title=title,
                        rows=hit.rows,
                    )
                )
                cursor_y = y1

            tail_clip = fitz.Rect(page.rect.x0, cursor_y, page.rect.x1, page.rect.y1)
            tail_text = page.get_text(clip=tail_clip) or ""
            page_text_parts.append(tail_text)

            out_pages.append("\n".join(p for p in page_text_parts if p))
    finally:
        if plumb_pdf is not None:
            try:
                plumb_pdf.close()
            except Exception:
                pass
        doc.close()

    if table_id:
        logger.debug(
            f"extract_text_with_tables: {table_id} table(s) marked across {len(out_pages)} page(s)"
        )
    return "\n\n".join(p for p in out_pages if p)


# ---------------------------------------------------------------------------
# Detection — same shape filters as the validation dry-run
# ---------------------------------------------------------------------------


def _detect_pymupdf(page) -> List[_TableHit]:
    try:
        tabs = page.find_tables()
    except Exception as exc:
        logger.debug(f"PyMuPDF.find_tables failed on page: {exc}")
        return []
    out: List[_TableHit] = []
    for t in tabs.tables:
        try:
            rows = t.extract()
        except Exception:
            continue
        if not _shape_ok(rows):
            continue
        out.append(_TableHit(bbox=tuple(float(v) for v in t.bbox), rows=rows))
    return out


def _detect_pdfplumber(plumb_page) -> List[_TableHit]:
    settings = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
    try:
        found = plumb_page.find_tables(table_settings=settings)
    except Exception as exc:
        logger.debug(f"pdfplumber.find_tables failed: {exc}")
        return []
    out: List[_TableHit] = []
    for t in found:
        try:
            rows = t.extract()
        except Exception:
            continue
        if not _shape_ok(rows):
            continue
        out.append(_TableHit(bbox=tuple(float(v) for v in t.bbox), rows=rows))
    return out


def _open_plumber_safely(pdf_content: bytes):
    """Return an opened ``pdfplumber.PDF`` instance or ``None``.

    Caller is responsible for calling ``.close()`` (typically in a
    ``finally`` block) to release file handles.
    """
    try:
        import pdfplumber
    except ImportError:
        return None
    try:
        return pdfplumber.open(io.BytesIO(pdf_content))
    except Exception as exc:
        logger.debug(f"pdfplumber.open failed: {exc}")
        return None


def _shape_ok(rows: Sequence[Sequence[Optional[str]]]) -> bool:
    if not rows:
        return False
    nrows = len(rows)
    ncols = max((len(r) for r in rows), default=0)
    return nrows >= _MIN_ROWS and ncols >= _MIN_COLS


# ---------------------------------------------------------------------------
# Title heuristic + markdown rendering
# ---------------------------------------------------------------------------


def _find_title_above(pre_text: str, window_lines: int) -> str:
    """Search the last ``window_lines`` for a ``Tabela X`` / ``Quadro X``."""
    if not pre_text:
        return ""
    tail = pre_text.split("\n")[-window_lines:]
    blob = "\n".join(tail)
    m = _TITLE_RE.search(blob)
    return m.group(0) if m else ""


def _render_block(
    *,
    table_id: int,
    page_no: int,
    title: str,
    rows: Sequence[Sequence[Optional[str]]],
) -> str:
    md = _rows_to_markdown(rows)
    nrows = len(rows)
    ncols = max((len(r) for r in rows), default=0)
    safe_title = _escape_attr(title)
    header = (
        f'{TABLE_BEGIN_PREFIX} id="t{table_id}" page="{page_no}" '
        f'title="{safe_title}" rows="{nrows}" cols="{ncols}"@@@'
    )
    return f"{header}\n{md}\n{TABLE_END}"


def _rows_to_markdown(rows: Sequence[Sequence[Optional[str]]]) -> str:
    """Render a 2D row list as GFM markdown (``|``-pipe table)."""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm: List[List[str]] = []
    for r in rows:
        cells = [(c or "").replace("\n", " ").replace("|", "\\|").strip() for c in r]
        if len(cells) < width:
            cells.extend([""] * (width - len(cells)))
        norm.append(cells)
    head = "| " + " | ".join(norm[0]) + " |"
    sep = "| " + " | ".join(["---"] * width) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in norm[1:])
    return f"{head}\n{sep}\n{body}".rstrip()


def _escape_attr(value: str) -> str:
    """Quote-escape a string for use inside a ``@@@TABLE_BEGIN ... @@@`` attr."""
    return value.replace('"', "'").replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# Helpers used by the chunker — exposed so ``pipeline/chunking.py`` doesn't
# duplicate parser internals.
# ---------------------------------------------------------------------------


@dataclass
class TableBlock:
    """A parsed ``@@@TABLE_*@@@`` block, ready for the chunker to embed."""
    table_id: str
    page: Optional[int]
    title: str
    rows: int
    cols: int
    markdown: str
    span: Tuple[int, int]  # (start, end) offsets in the source text

    @property
    def column_headers_text(self) -> str:
        """Plain-text rendering of the first markdown row, for BM25/dense.

        Strips ``|`` pipes and ``---`` separator artifacts. Falls back to
        the raw first line if no header row is found. Used by the chunker
        to enrich the indexable text without altering the markdown body.
        """
        for line in self.markdown.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            if set(line.replace("|", "").replace(" ", "")) <= set("-:"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
            if cells:
                return " ".join(cells)
        return ""


def iter_table_blocks(text: str) -> Iterable[TableBlock]:
    """Yield every ``@@@TABLE_*@@@`` block found in ``text``."""
    for m in TABLE_BLOCK_RE.finditer(text):
        attrs = _parse_attrs(m.group("attrs"))
        try:
            page = int(attrs.get("page", "")) if attrs.get("page") else None
        except (TypeError, ValueError):
            page = None
        try:
            rows = int(attrs.get("rows", "0"))
            cols = int(attrs.get("cols", "0"))
        except (TypeError, ValueError):
            rows = cols = 0
        yield TableBlock(
            table_id=attrs.get("id", ""),
            page=page,
            title=attrs.get("title", ""),
            rows=rows,
            cols=cols,
            markdown=m.group("body").strip(),
            span=(m.start(), m.end()),
        )


_ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


def _parse_attrs(attrs_str: str) -> dict:
    """Parse ``id="t1" page="3" title="…" ...`` into a dict."""
    return {k: v for k, v in _ATTR_RE.findall(attrs_str)}
