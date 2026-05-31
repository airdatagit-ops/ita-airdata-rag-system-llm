"""Dry-run: validate alternatives 1 + 2 (table-aware extraction).

For each target PDF, build THREE text representations and measure how
many golden-set table cells survive each one:

  A. CURRENT — fitz.page.get_text()  (production cascade)
  B. PYMUPDF_MD — same get_text() but every detected table region is
     REPLACED by `Table.to_markdown()`.
  C. PLUMB_MD — same as B but using pdfplumber `lines` strategy and a
     hand-rolled markdown rendering.

Then simulate the production chunker by sliding a window over each
text (chars proxy for tokens) and report:

  - full-doc:        does the expected cell appear at all?
  - chunk-hit:       in how many chunks does it appear?
  - cell+context:    does any chunk contain the cell AND the table
                     label (a fragment from `expected_table_label`)?

Output: data/_table_study/dry_run_report.md and
        data/_table_study/dry_run_summary.csv
"""
from __future__ import annotations

import csv
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz
import pdfplumber

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "evaluation" / "golden_set_tables.csv"
OUT_DIR = ROOT / "data" / "_table_study"


# Map golden expected_doc_id -> path on disk (the PDFs we have)
DOC_TO_PATH = {
    "RBAC-43":    ROOT / "data/originals/anac/rbac-43.pdf",
    "RBAC-26":    ROOT / "data/originals/anac/rbac-26.pdf",
    "RBAC-105":   ROOT / "data/originals/anac/rbac-105.pdf",
    "ICA-100-12": ROOT / "data/originals/sislaer/ica-100-12.pdf",
    "ICA-100-37": ROOT / "data/originals/sislaer/ica-100-37.pdf",
}

# Production chunk size proxy (chars) — production is 270 tokens; ~1.5 chars/token
# for Portuguese gives ~400 chars; we use 1500 as a generous upper bound that
# still reflects the chunker's locality constraint.
CHUNK_CHARS = 1500
CHUNK_OVERLAP = 200


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s).strip().casefold()
    return s


def _has(needle: str, haystack: str) -> bool:
    n = _norm(needle)
    return bool(n) and (n in _norm(haystack))


def _label_token(label: str) -> str:
    """Pick a short, distinctive token from the table label for context check."""
    if not label:
        return ""
    m = re.search(r"(TABELA\s+[IVX0-9]+|Tabela\s+\d+)", label)
    if m:
        return m.group(0)
    parts = re.split(r"[—\-(]", label)
    return parts[0].strip()


def _split_chunks(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        chunks.append(text[i : i + size])
        if i + size >= n:
            break
        i += size - overlap
    return chunks


# ---------------------------------------------------------------------
# extraction strategies
# ---------------------------------------------------------------------


def _rows_to_md(rows: List[List[Optional[str]]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [
        [(c or "").replace("\n", " ").strip() for c in (r + [""] * (width - len(r)))]
        for r in rows
    ]
    head = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * width) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
    return f"{head}\n{sep}\n{body}"


def extract_current(pdf_path: Path) -> str:
    """Method A: replicate production cascade (fitz.get_text)."""
    parts: List[str] = []
    doc = fitz.open(pdf_path)
    for page in doc:
        parts.append(page.get_text() or "")
    doc.close()
    return "\n".join(parts)


def extract_pymupdf_md(pdf_path: Path) -> str:
    """Method B: get_text() per page, but each detected table region is
    replaced by `Table.to_markdown()` injected at the same logical spot.

    PyMuPDF tables have `.bbox`. We grab the surrounding text outside the
    bbox via `page.get_text(clip=...)` for above and below, then concat
    [above_text, table_md, below_text] for that page. For pages without
    tables we fall back to A.
    """
    parts: List[str] = []
    doc = fitz.open(pdf_path)
    for page in doc:
        try:
            tables = page.find_tables().tables
        except Exception:
            tables = []

        real_tables = []
        for t in tables:
            try:
                rows = t.extract()
            except Exception:
                continue
            nrows = len(rows)
            ncols = max((len(r) for r in rows), default=0)
            if nrows >= 3 and ncols >= 2:
                real_tables.append((t, rows))

        if not real_tables:
            parts.append(page.get_text() or "")
            continue

        page_rect = page.rect
        bands: List[Tuple[float, float, str]] = []
        last_y = page_rect.y0
        real_tables.sort(key=lambda tt: tt[0].bbox[1])
        for t, rows in real_tables:
            x0, y0, x1, y1 = t.bbox
            if y0 > last_y:
                clip = fitz.Rect(page_rect.x0, last_y, page_rect.x1, y0)
                bands.append((last_y, y0, page.get_text(clip=clip) or ""))
            md = _rows_to_md(rows)
            bands.append((y0, y1, f"\n[TABLE-MD]\n{md}\n[/TABLE-MD]\n"))
            last_y = y1
        if last_y < page_rect.y1:
            clip = fitz.Rect(page_rect.x0, last_y, page_rect.x1, page_rect.y1)
            bands.append((last_y, page_rect.y1, page.get_text(clip=clip) or ""))

        parts.append("\n".join(b[2] for b in bands))
    doc.close()
    return "\n".join(parts)


def extract_pdfplumber_md(pdf_path: Path) -> str:
    """Method C: pdfplumber lines strategy, table replaced by md.

    Same shape as B but using pdfplumber to find / extract the tables.
    """
    parts: List[str] = []
    settings = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            try:
                found = page.find_tables(table_settings=settings)
            except Exception:
                found = []

            real = []
            for t in found:
                try:
                    rows = t.extract()
                except Exception:
                    continue
                if len(rows) >= 3 and max((len(r) for r in rows), default=0) >= 2:
                    real.append((t, rows))

            if not real:
                parts.append(page.extract_text() or "")
                continue

            real.sort(key=lambda tt: tt[0].bbox[1])
            page_y0, page_y1 = float(page.bbox[1]), float(page.bbox[3])
            cursor = page_y0
            band_texts: List[str] = []
            for t, rows in real:
                tx0, ty0, tx1, ty1 = (float(v) for v in t.bbox)
                if ty0 > cursor:
                    above = page.within_bbox(
                        (float(page.bbox[0]), cursor, float(page.bbox[2]), ty0)
                    )
                    band_texts.append(above.extract_text() or "")
                band_texts.append(f"\n[TABLE-MD]\n{_rows_to_md(rows)}\n[/TABLE-MD]\n")
                cursor = ty1
            if cursor < page_y1:
                below = page.within_bbox(
                    (float(page.bbox[0]), cursor, float(page.bbox[2]), page_y1)
                )
                band_texts.append(below.extract_text() or "")
            parts.append("\n".join(band_texts))
    return "\n".join(parts)


# ---------------------------------------------------------------------
# golden set + measurement
# ---------------------------------------------------------------------


@dataclass
class Result:
    query_id: str
    doc_id: str
    cell: str
    label_token: str
    full_A: bool = False
    full_B: bool = False
    full_C: bool = False
    chunks_A: int = 0
    chunks_B: int = 0
    chunks_C: int = 0
    chunks_with_context_A: int = 0
    chunks_with_context_B: int = 0
    chunks_with_context_C: int = 0


def load_golden() -> List[dict]:
    rows = []
    with GOLDEN.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def evaluate_doc(doc_id: str, queries: List[dict], texts: Dict[str, str]) -> List[Result]:
    chunked = {k: _split_chunks(t) for k, t in texts.items()}
    results: List[Result] = []
    for q in queries:
        cell = q["expected_cell_value"]
        label = _label_token(q.get("expected_table_label", ""))
        r = Result(query_id=q["query_id"], doc_id=doc_id, cell=cell, label_token=label)
        for tag in ("A", "B", "C"):
            txt = texts[tag]
            cs = chunked[tag]
            full_hit = _has(cell, txt)
            setattr(r, f"full_{tag}", full_hit)
            n_cell = sum(1 for c in cs if _has(cell, c))
            setattr(r, f"chunks_{tag}", n_cell)
            n_ctx = sum(1 for c in cs if _has(cell, c) and (not label or _has(label, c)))
            setattr(r, f"chunks_with_context_{tag}", n_ctx)
        results.append(r)
    return results


# ---------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------


def write_summary(all_results: List[Result], by_doc_size: Dict[str, Dict[str, int]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "dry_run_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "query_id", "doc_id", "cell", "label_token",
            "full_A_current", "full_B_pymupdf_md", "full_C_plumb_md",
            "chunks_A", "chunks_B", "chunks_C",
            "chunks_with_context_A", "chunks_with_context_B", "chunks_with_context_C",
        ])
        for r in all_results:
            w.writerow([
                r.query_id, r.doc_id, r.cell, r.label_token,
                r.full_A, r.full_B, r.full_C,
                r.chunks_A, r.chunks_B, r.chunks_C,
                r.chunks_with_context_A, r.chunks_with_context_B, r.chunks_with_context_C,
            ])

    n = len(all_results)
    md = []
    md.append("# Dry-run: table-aware extraction (alternatives 1 + 2)")
    md.append("")
    md.append(f"Queries evaluated: **{n}** (over {len(DOC_TO_PATH)} PDFs)")
    md.append("")
    md.append("## Cell visibility — full document")
    md.append("")
    md.append("| Method | Cell appears in extracted text |")
    md.append("|---|---|")
    for tag, label in [
        ("A", "A — current production (fitz.get_text only)"),
        ("B", "B — PyMuPDF find_tables() → markdown"),
        ("C", "C — pdfplumber lines → markdown"),
    ]:
        n_hit = sum(1 for r in all_results if getattr(r, f"full_{tag}"))
        md.append(f"| {label} | {n_hit}/{n} = {100*n_hit/n:.1f}% |")
    md.append("")

    md.append("## Cell + context inside a single chunk (1500-char window)")
    md.append("")
    md.append("| Method | At least one chunk has cell + table label |")
    md.append("|---|---|")
    for tag, label in [
        ("A", "A — current"),
        ("B", "B — PyMuPDF MD"),
        ("C", "C — pdfplumber MD"),
    ]:
        n_hit = sum(1 for r in all_results if getattr(r, f"chunks_with_context_{tag}") > 0)
        md.append(f"| {label} | {n_hit}/{n} = {100*n_hit/n:.1f}% |")
    md.append("")

    md.append("## Per-document — cell+context hit rate")
    md.append("")
    md.append("| Doc | N | A_current | B_pymupdf | C_plumb | size_A→size_B (chars) |")
    md.append("|---|---|---|---|---|---|")
    by_doc: Dict[str, List[Result]] = {}
    for r in all_results:
        by_doc.setdefault(r.doc_id, []).append(r)
    for doc, rs in sorted(by_doc.items()):
        n_d = len(rs)
        a = sum(1 for r in rs if r.chunks_with_context_A > 0)
        b = sum(1 for r in rs if r.chunks_with_context_B > 0)
        c = sum(1 for r in rs if r.chunks_with_context_C > 0)
        sizes = by_doc_size[doc]
        md.append(
            f"| {doc} | {n_d} "
            f"| {a}/{n_d} | {b}/{n_d} | {c}/{n_d} "
            f"| {sizes['A']:,}→{sizes['B']:,} |"
        )
    md.append("")

    md.append("## Per-query")
    md.append("")
    md.append("| query_id | doc | cell | A | B | C | A_ctx | B_ctx | C_ctx |")
    md.append("|---|---|---|---|---|---|---|---|---|")

    def yn(b: bool) -> str:
        return "✓" if b else "✗"

    for r in all_results:
        md.append(
            f"| {r.query_id} | {r.doc_id} | `{r.cell}` "
            f"| {yn(r.full_A)} | {yn(r.full_B)} | {yn(r.full_C)} "
            f"| {r.chunks_with_context_A} | {r.chunks_with_context_B} | {r.chunks_with_context_C} |"
        )

    (OUT_DIR / "dry_run_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote:\n  - {csv_path.relative_to(ROOT)}\n  - {(OUT_DIR / 'dry_run_report.md').relative_to(ROOT)}")


# ---------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------


def main() -> int:
    queries = load_golden()
    by_doc: Dict[str, List[dict]] = {}
    for q in queries:
        by_doc.setdefault(q["expected_doc_id"], []).append(q)

    all_results: List[Result] = []
    by_doc_sizes: Dict[str, Dict[str, int]] = {}

    for doc_id, doc_queries in by_doc.items():
        path = DOC_TO_PATH.get(doc_id)
        if not path or not path.exists():
            print(f"[!] missing PDF for {doc_id} — skipping {len(doc_queries)} queries")
            continue
        print(f"=== {doc_id} ({path.name}) — extracting...")
        text_a = extract_current(path)
        print(f"   A current     : {len(text_a):>9,} chars")
        text_b = extract_pymupdf_md(path)
        print(f"   B pymupdf+md  : {len(text_b):>9,} chars")
        text_c = extract_pdfplumber_md(path)
        print(f"   C plumb+md    : {len(text_c):>9,} chars")
        by_doc_sizes[doc_id] = {"A": len(text_a), "B": len(text_b), "C": len(text_c)}

        # Persist sample texts for manual inspection
        sample_dir = OUT_DIR / "samples" / doc_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "A_current.txt").write_text(text_a, encoding="utf-8")
        (sample_dir / "B_pymupdf_md.txt").write_text(text_b, encoding="utf-8")
        (sample_dir / "C_plumb_md.txt").write_text(text_c, encoding="utf-8")

        results = evaluate_doc(doc_id, doc_queries, {"A": text_a, "B": text_b, "C": text_c})
        for r in results:
            print(
                f"   {r.query_id} cell={r.cell[:30]!r:30s} "
                f"full=A:{int(r.full_A)} B:{int(r.full_B)} C:{int(r.full_C)} "
                f"ctx=A:{r.chunks_with_context_A} B:{r.chunks_with_context_B} C:{r.chunks_with_context_C}"
            )
        all_results.extend(results)

    write_summary(all_results, by_doc_sizes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
