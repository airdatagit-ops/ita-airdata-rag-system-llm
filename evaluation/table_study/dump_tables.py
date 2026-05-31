"""Dump table markdown for selected (doc, page) targets.

Helper for building the table-bound golden set: prints each requested
table as markdown so the human can pick concrete cells to ask about.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parents[2]


# (doc_path_relative_to_repo_root, page_number_1based, [optional list of table indices to keep])
TARGETS = [
    ("data/originals/anac/rbac-43.pdf", 29, None),       # Tab I + Tab II
    ("data/originals/anac/rbac-43.pdf", 30, None),       # Tab III + Tab IV
    ("data/originals/anac/rbac-26.pdf", 3,  None),       # Tab 1 aplicabilidade
    ("data/originals/anac/rbac-105.pdf", 5,  None),      # Visibilidade VFR
    ("data/originals/sislaer/ica-100-12.pdf", 33, None), # Tab 1
    ("data/originals/sislaer/ica-100-12.pdf", 38, None), # Tab 2
    ("data/originals/sislaer/ica-100-12.pdf", 67, None), # Tab 7/8/9
    ("data/originals/sislaer/ica-100-37.pdf", 128, None), # nível transição
    ("data/originals/sislaer/ica-100-37.pdf", 129, None),
    ("data/originals/sislaer/ica-100-37.pdf", 235, None), # esteira turb
    ("data/originals/sislaer/ica-100-37.pdf", 276, None), # classe espaço aéreo
]


def to_markdown(rows):
    """Render a 2D list as markdown table."""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows]
    head = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * width) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
    return f"{head}\n{sep}\n{body}"


def main() -> int:
    out_dir = ROOT / "data" / "_table_study" / "tables_md"
    out_dir.mkdir(parents=True, exist_ok=True)

    for rel, page_no, _keep in TARGETS:
        pdf_path = ROOT / rel
        if not pdf_path.exists():
            print(f"[!] missing {pdf_path}")
            continue
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_no - 1]
            tables = page.extract_tables(table_settings={
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
            })

        if not tables:
            print(f"--- {pdf_path.name} p{page_no}: NO TABLE FOUND")
            continue

        out_file = out_dir / f"{pdf_path.stem}_p{page_no}.md"
        with out_file.open("w", encoding="utf-8") as f:
            for ti, t in enumerate(tables):
                f.write(f"## {pdf_path.name} — page {page_no} — table {ti}\n\n")
                f.write(to_markdown(t))
                f.write("\n\n")
        print(f"--- {pdf_path.name} p{page_no}: {len(tables)} table(s) -> {out_file.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
