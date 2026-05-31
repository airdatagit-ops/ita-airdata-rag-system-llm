"""Scan PDFs for table candidates.

For each PDF, prints:
  - pages where word "TABELA" / "Tabela" appears with the heading line
  - PyMuPDF find_tables() (default — relies on visual lines)
  - pdfplumber extract_tables() with line strategy AND text strategy
  - filters out trivial 1-row "tables" (page header bars)

Pure read-only inspection; no production code touched.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber


ROOT = Path(__file__).resolve().parents[2]


def _short(s: str, n: int = 50) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _is_real_table(rows: int, cols: int) -> bool:
    """Filter out single-row 'tables' that are page headers/footers."""
    return rows >= 3 and cols >= 2


def _extract_first_row(rows):
    return [_short(c or "", 25) for c in (rows[0] if rows else [])[:6]]


def _scan_pymupdf(doc) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for i, page in enumerate(doc, start=1):
        try:
            tabs = page.find_tables()
        except Exception:
            continue
        for ti, t in enumerate(tabs.tables):
            try:
                rows = t.extract()
            except Exception:
                rows = []
            nrows = len(rows)
            ncols = max((len(r) for r in rows), default=0)
            if not _is_real_table(nrows, ncols):
                continue
            out.setdefault(i, []).append({
                "engine": "pymupdf",
                "rows": nrows, "cols": ncols,
                "head": _extract_first_row(rows),
            })
    return out


def _scan_pdfplumber(pdf_path: Path) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    settings_variants = [
        ("lines", {"vertical_strategy": "lines", "horizontal_strategy": "lines"}),
        ("text",  {"vertical_strategy": "text",  "horizontal_strategy": "text",
                   "snap_tolerance": 4, "join_tolerance": 4, "edge_min_length": 30}),
    ]
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            for label, settings in settings_variants:
                try:
                    rows_list = page.extract_tables(table_settings=settings)
                except Exception:
                    rows_list = []
                for rows in rows_list:
                    nrows = len(rows)
                    ncols = max((len(r) for r in rows), default=0)
                    if not _is_real_table(nrows, ncols):
                        continue
                    out.setdefault(i, []).append({
                        "engine": f"pdfplumber-{label}",
                        "rows": nrows, "cols": ncols,
                        "head": _extract_first_row(rows),
                    })
    return out


def scan(pdf_path: Path) -> None:
    if not pdf_path.exists():
        print(f"[!] missing: {pdf_path}")
        return

    print(f"\n=== {pdf_path.name} ({pdf_path.stat().st_size//1024} KB) ===")
    doc = fitz.open(pdf_path)
    n_pages = len(doc)

    table_heading_re = re.compile(r"^\s*(?:TABELA|Tabela|TAB\.|QUADRO|Quadro)\s+[IVX0-9]+", re.MULTILINE)
    headings: dict[int, list[str]] = {}
    for i, page in enumerate(doc, start=1):
        text = page.get_text() or ""
        hits = [m.group(0).strip() for m in table_heading_re.finditer(text)]
        if hits:
            headings[i] = hits
            for h in hits:
                idx = text.find(h)
                snippet = text[idx: idx + 200] if idx >= 0 else ""
                print(f"  p{i} heading: {_short(snippet, 120)}")

    pymu = _scan_pymupdf(doc)
    doc.close()
    plumb = _scan_pdfplumber(pdf_path)

    pages = sorted(set(pymu) | set(plumb))
    if not pages and not headings:
        print(f"  pages={n_pages} — no real tables and no 'TABELA'/'QUADRO' heading found.")
        return

    print(f"  pages={n_pages} — table-bearing pages:")
    for p in pages:
        print(f"   p{p}:")
        for ent in pymu.get(p, []) + plumb.get(p, []):
            print(f"     [{ent['engine']:18s}] rows={ent['rows']:>2} cols={ent['cols']} head={ent['head']}")


def main() -> int:
    targets = [
        # known-good candidates first
        "rbac-43.pdf",   # tolerâncias altímetro (Apêndice E)
        "rbac-39.pdf",   # marcas e identificação de aeronaves
        "rbac-65.pdf",   # licenciamento aeronautas
        "rbac-67.pdf",   # requisitos médicos
        "rbac-105.pdf",  # transporte de cargas perigosas
        # extra candidates
        "rbac-26.pdf", "rbac-34.pdf", "rbac-045.pdf",
        "rbac-60.pdf", "rbac-31.pdf", "rbac-35.pdf",
        "rbac-36.pdf", "rbac-133.pdf", "rbac-136.pdf",
        "rbac-142.pdf", "rbac-154.pdf", "rbac-183.pdf",
    ]
    base = ROOT / "data" / "originals" / "anac"
    for name in targets:
        p = base / name
        if p.exists():
            scan(p)

    sislaer = ROOT / "data" / "originals" / "sislaer"
    if sislaer.exists():
        for p in sorted(sislaer.glob("*.pdf")):
            scan(p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
