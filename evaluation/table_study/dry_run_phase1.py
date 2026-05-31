"""Simulate Phase 1 (table-aware extraction + chunking) without touching prod.

For each PDF in the study:
  1. Detect tables via PyMuPDF.find_tables() with cols >= 2 and rows >= 3.
     Fallback to pdfplumber.find_tables(lines) when PyMuPDF is empty.
  2. For each detected table, build ONE indivisible chunk:
        chunk_text = title + ctx_pre + markdown(table) + ctx_post
     Where:
       - title: regex (TABELA|Tabela|TAB.|QUADRO|Quadro)\\s+[IVX0-9]+
               searched in the N=10 lines above the table bbox.
       - ctx_pre: up to 200 chars from the page text ending right
                  above the table bbox (last paragraph).
       - markdown: Table.to_markdown() (PyMuPDF) or hand-rolled.
       - ctx_post: up to 200 chars starting right below the bbox.
     Tag each chunk with chunk_type="table".
  3. Remaining text (regions outside any table bbox) gets fed to the
     production chunker, exactly as today.
  4. For each golden query, measure on the resulting chunk set:
       - cell present in any chunk?
       - cell + table_label co-located in any chunk?
       - which chunk and of what type?

Output:
  - data/_table_study/dry_run_phase1.csv
  - data/_table_study/dry_run_phase1.md
  - data/_table_study/samples_phase1/<DOC>/chunks.jsonl

Read-only with respect to production code. Production parser/chunker
are NOT changed.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz
import pdfplumber

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.chunking import ArticleChunker  # noqa: E402

GOLDEN = ROOT / "evaluation" / "golden_set_tables.csv"
OUT_DIR = ROOT / "data" / "_table_study"

DOC_TO_PATH = {
    "RBAC-43":    ROOT / "data/originals/anac/rbac-43.pdf",
    "RBAC-26":    ROOT / "data/originals/anac/rbac-26.pdf",
    "RBAC-105":   ROOT / "data/originals/anac/rbac-105.pdf",
    "ICA-100-12": ROOT / "data/originals/sislaer/ica-100-12.pdf",
    "ICA-100-37": ROOT / "data/originals/sislaer/ica-100-37.pdf",
}

TITLE_RE = re.compile(
    r"(TABELA|Tabela|TAB\.|QUADRO|Quadro)\s+[IVX0-9]+",
    re.MULTILINE,
)

CTX_CHARS = 200


def _norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s).strip().casefold()
    return s


def _has(needle: str, haystack: str) -> bool:
    n = _norm(needle)
    return bool(n) and n in _norm(haystack)


def _label_token(label: str) -> str:
    if not label:
        return ""
    m = TITLE_RE.search(label)
    if m:
        return m.group(0)
    return re.split(r"[—\-(]", label)[0].strip()


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


@dataclass
class TableChunk:
    chunk_type: str
    page: int
    title: str
    ctx_pre: str
    markdown: str
    ctx_post: str
    n_rows: int
    n_cols: int

    @property
    def text(self) -> str:
        parts = []
        if self.title:
            parts.append(self.title)
        if self.ctx_pre:
            parts.append(self.ctx_pre)
        parts.append(self.markdown)
        if self.ctx_post:
            parts.append(self.ctx_post)
        return "\n\n".join(parts)


@dataclass
class ProseChunk:
    chunk_type: str
    text: str


def _detect_pymupdf(page) -> List[Tuple[Tuple[float, float, float, float], List[List]]]:
    """Return list of (bbox, rows) for tables that pass minimal shape filter."""
    try:
        tabs = page.find_tables()
    except Exception:
        return []
    out = []
    for t in tabs.tables:
        try:
            rows = t.extract()
        except Exception:
            continue
        nrows = len(rows)
        ncols = max((len(r) for r in rows), default=0)
        if nrows >= 3 and ncols >= 2:
            out.append((tuple(t.bbox), rows))
    return out


def _detect_pdfplumber(plumb_page) -> List[Tuple[Tuple[float, float, float, float], List[List]]]:
    settings = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
    try:
        found = plumb_page.find_tables(table_settings=settings)
    except Exception:
        return []
    out = []
    for t in found:
        try:
            rows = t.extract()
        except Exception:
            continue
        if len(rows) >= 3 and max((len(r) for r in rows), default=0) >= 2:
            out.append((tuple(float(v) for v in t.bbox), rows))
    return out


def _build_phase1_chunks(pdf_path: Path) -> Tuple[List[TableChunk], List[ProseChunk]]:
    """Build Phase 1 chunks for a PDF. Returns (table_chunks, prose_chunks)."""
    table_chunks: List[TableChunk] = []
    prose_pieces: List[str] = []

    doc = fitz.open(pdf_path)
    plumb = pdfplumber.open(pdf_path)
    try:
        for page_no, page in enumerate(doc, start=1):
            page_rect = page.rect
            tables_pmu = _detect_pymupdf(page)
            tables = tables_pmu
            if not tables:
                tables = _detect_pdfplumber(plumb.pages[page_no - 1])

            if not tables:
                prose_pieces.append(page.get_text() or "")
                continue

            tables.sort(key=lambda b: b[0][1])
            cursor_y = page_rect.y0
            for bbox, rows in tables:
                x0, y0, x1, y1 = bbox

                pre_clip = fitz.Rect(page_rect.x0, cursor_y, page_rect.x1, y0)
                pre_text = page.get_text(clip=pre_clip) or ""
                tail = pre_text.strip().split("\n")
                title_match = TITLE_RE.search("\n".join(tail[-15:]))
                title = title_match.group(0) if title_match else ""

                ctx_pre = pre_text.strip()[-CTX_CHARS:]

                post_clip = fitz.Rect(page_rect.x0, y1, page_rect.x1, page_rect.y1)
                post_text = page.get_text(clip=post_clip) or ""
                ctx_post = post_text.strip()[:CTX_CHARS]

                table_chunks.append(TableChunk(
                    chunk_type="table",
                    page=page_no,
                    title=title,
                    ctx_pre=ctx_pre,
                    markdown=_rows_to_md(rows),
                    ctx_post=ctx_post,
                    n_rows=len(rows),
                    n_cols=max((len(r) for r in rows), default=0),
                ))

                outside_pre = page.get_text(clip=pre_clip) or ""
                prose_pieces.append(outside_pre)
                cursor_y = y1

            tail_clip = fitz.Rect(page_rect.x0, cursor_y, page_rect.x1, page_rect.y1)
            prose_pieces.append(page.get_text(clip=tail_clip) or "")
    finally:
        plumb.close()
        doc.close()

    prose_text = "\n".join(p for p in prose_pieces if p.strip())
    chunker = ArticleChunker()
    prose_chunks_raw = chunker.chunk({"text": prose_text, "metadata": {}})
    prose_chunks = [
        ProseChunk(chunk_type="article", text=c.get("text", ""))
        for c in prose_chunks_raw
    ]

    return table_chunks, prose_chunks


@dataclass
class QueryResult:
    query_id: str
    doc_id: str
    cell: str
    label: str
    cell_in_phase1_table: bool = False
    cell_in_phase1_prose: bool = False
    cell_plus_label_phase1: bool = False
    cell_plus_label_baseline_A: bool = False
    cell_plus_label_baseline_chunker: bool = False
    n_table_chunks: int = 0
    chunk_with_cell_label_size: Optional[int] = None
    chunk_with_cell_label_title: Optional[str] = None


def main() -> int:
    queries = list(csv.DictReader(GOLDEN.open()))
    by_doc: Dict[str, List[dict]] = {}
    for q in queries:
        by_doc.setdefault(q["expected_doc_id"], []).append(q)

    chunk_size_stats: List[int] = []
    table_chunks_per_doc: Dict[str, int] = {}
    all_results: List[QueryResult] = []

    samples_dir = OUT_DIR / "samples_phase1"
    samples_dir.mkdir(parents=True, exist_ok=True)

    for doc_id, qs in by_doc.items():
        path = DOC_TO_PATH.get(doc_id)
        if not path or not path.exists():
            print(f"[!] missing {doc_id}")
            continue
        print(f"=== {doc_id} ({path.name})")

        table_chunks, prose_chunks = _build_phase1_chunks(path)
        table_chunks_per_doc[doc_id] = len(table_chunks)
        print(f"   table_chunks={len(table_chunks)}  prose_chunks={len(prose_chunks)}")

        sample_path = samples_dir / f"{doc_id}.jsonl"
        with sample_path.open("w", encoding="utf-8") as f:
            for tc in table_chunks:
                f.write(json.dumps({
                    "chunk_type": tc.chunk_type, "page": tc.page,
                    "title": tc.title, "n_rows": tc.n_rows, "n_cols": tc.n_cols,
                    "text_len": len(tc.text),
                    "text_preview": tc.text[:400],
                }, ensure_ascii=False) + "\n")
                chunk_size_stats.append(len(tc.text))

        all_chunk_texts = [c.text for c in table_chunks] + [c.text for c in prose_chunks]
        table_only_texts = [c.text for c in table_chunks]
        prose_only_texts = [c.text for c in prose_chunks]

        for q in qs:
            cell = q["expected_cell_value"]
            label = _label_token(q.get("expected_table_label", ""))
            r = QueryResult(query_id=q["query_id"], doc_id=doc_id,
                            cell=cell, label=label,
                            n_table_chunks=len(table_chunks))

            r.cell_in_phase1_table = any(_has(cell, t) for t in table_only_texts)
            r.cell_in_phase1_prose = any(_has(cell, t) for t in prose_only_texts)

            for tc in table_chunks:
                if _has(cell, tc.text) and (not label or _has(label, tc.text)):
                    r.cell_plus_label_phase1 = True
                    r.chunk_with_cell_label_size = len(tc.text)
                    r.chunk_with_cell_label_title = tc.title or "(sem titulo)"
                    break
            if not r.cell_plus_label_phase1:
                for pc in prose_chunks:
                    if _has(cell, pc.text) and (not label or _has(label, pc.text)):
                        r.cell_plus_label_phase1 = True
                        r.chunk_with_cell_label_size = len(pc.text)
                        r.chunk_with_cell_label_title = "(prose chunk)"
                        break

            all_results.append(r)
            print(
                f"   {r.query_id} cell={cell[:25]!r:25s} "
                f"phase1_cell+label={int(r.cell_plus_label_phase1)} "
                f"in_table={int(r.cell_in_phase1_table)} in_prose={int(r.cell_in_phase1_prose)}"
            )

    n = len(all_results)
    csv_path = OUT_DIR / "dry_run_phase1.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(QueryResult.__dataclass_fields__.keys()))
        w.writeheader()
        for r in all_results:
            w.writerow(asdict(r))

    n_phase1 = sum(1 for r in all_results if r.cell_plus_label_phase1)
    n_in_table = sum(1 for r in all_results if r.cell_in_phase1_table)
    n_in_prose = sum(1 for r in all_results if r.cell_in_phase1_prose)

    md = []
    md.append("# Dry-run Fase 1 — chunks indivisiveis com titulo + contexto + markdown")
    md.append("")
    md.append(f"Queries avaliadas: **{n}**")
    md.append("")
    md.append("## KPI principal — cell + label co-localizados em algum chunk Fase 1")
    md.append("")
    md.append("| Coluna | Valor |")
    md.append("|---|---|")
    md.append(f"| Total de queries | {n} |")
    md.append(f"| Cell + label co-localizados (Fase 1) | {n_phase1}/{n} = {100*n_phase1/n:.1f}% |")
    md.append(f"| Baseline (chunker producao, metodo atual) | 17/{n} = {100*17/n:.1f}% |")
    md.append(f"| Delta | +{100*(n_phase1-17)/n:.1f} pp |")
    md.append("")

    md.append("## Cobertura — celula em chunk de tabela vs chunk de prosa")
    md.append("")
    md.append("| Coluna | Valor |")
    md.append("|---|---|")
    md.append(f"| Celula presente em algum chunk de tabela Fase 1 | {n_in_table}/{n} = {100*n_in_table/n:.1f}% |")
    md.append(f"| Celula presente em algum chunk de prosa (overflow) | {n_in_prose}/{n} = {100*n_in_prose/n:.1f}% |")
    md.append("")
    md.append("Nota: queries com celula apenas em prosa indicam tabelas que find_tables() nao detectou (sem bordas).")
    md.append("")

    md.append("## Estatisticas de tamanho dos chunks de tabela")
    md.append("")
    if chunk_size_stats:
        chunk_size_stats.sort()
        avg = sum(chunk_size_stats) / len(chunk_size_stats)
        p50 = chunk_size_stats[len(chunk_size_stats) // 2]
        p95 = chunk_size_stats[int(len(chunk_size_stats) * 0.95)]
        mx = max(chunk_size_stats)
        md.append(f"- chunks de tabela criados: {len(chunk_size_stats)}")
        md.append(f"- tamanho medio: {avg:.0f} chars")
        md.append(f"- p50: {p50} chars")
        md.append(f"- p95: {p95} chars")
        md.append(f"- maximo: {mx} chars")
        md.append("")
        md.append(f"Como referencia: 270 tokens ~= 1.000-1.500 chars em pt-BR. ")
        md.append(f"Chunks acima de ~2.000 chars superam materialmente o tamanho atual.")
    md.append("")

    md.append("## Por documento")
    md.append("")
    md.append("| Documento | N | table_chunks | cell+label Fase 1 | celula em tabela | celula em prosa |")
    md.append("|---|---|---|---|---|---|")
    by_d: Dict[str, List[QueryResult]] = {}
    for r in all_results:
        by_d.setdefault(r.doc_id, []).append(r)
    for doc, rs in sorted(by_d.items()):
        n_d = len(rs)
        md.append(
            f"| {doc} | {n_d} | {table_chunks_per_doc.get(doc, 0)} "
            f"| {sum(r.cell_plus_label_phase1 for r in rs)}/{n_d} "
            f"| {sum(r.cell_in_phase1_table for r in rs)}/{n_d} "
            f"| {sum(r.cell_in_phase1_prose for r in rs)}/{n_d} |"
        )
    md.append("")

    md.append("## Por query")
    md.append("")
    md.append("| query | doc | cell | phase1_co_loc | in_table | in_prose | chunk_size | titulo_capturado |")
    md.append("|---|---|---|---|---|---|---|---|")

    def yn(b: bool) -> str:
        return "OK" if b else "--"

    for r in all_results:
        md.append(
            f"| {r.query_id} | {r.doc_id} | `{r.cell}` "
            f"| {yn(r.cell_plus_label_phase1)} "
            f"| {yn(r.cell_in_phase1_table)} | {yn(r.cell_in_phase1_prose)} "
            f"| {r.chunk_with_cell_label_size or '-'} "
            f"| {r.chunk_with_cell_label_title or '-'} |"
        )

    md_path = OUT_DIR / "dry_run_phase1.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote:\n  - {csv_path.relative_to(ROOT)}\n  - {md_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
