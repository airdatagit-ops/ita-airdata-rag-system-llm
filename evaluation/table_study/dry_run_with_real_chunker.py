"""Re-run alternatives 1+2 using the PRODUCTION chunker (not a window).

Same A/B/C extraction methods as ``dry_run.py``, but the text gets fed
to ``pipeline.chunking.ArticleChunker`` exactly the way embedding does
in production. We then ask: in how many production-sized chunks does
the golden cell appear together with its table-label token?

This isolates the *chunking* contribution (vs *extraction*) — the
delta between A and B/C measures the upper bound a table-aware
extraction can deliver before we touch the chunker.
"""
from __future__ import annotations

import csv
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.chunking import ArticleChunker, ICAChunker  # noqa: E402

from evaluation.table_study.dry_run import (  # noqa: E402
    DOC_TO_PATH, GOLDEN, OUT_DIR,
    extract_current, extract_pymupdf_md, extract_pdfplumber_md,
    _norm, _has, _label_token,
)


def chunk_with_production(text: str, doc_id: str) -> List[str]:
    """Run the same chunker production uses for this doc type."""
    if doc_id.upper().startswith("ICA"):
        chunker = ICAChunker()
        article = {
            "doc_id": doc_id, "text": text,
            "article_num": "0", "article_text": text,
            "metadata": {},
        }
        chunks = chunker.chunk_document(article) if hasattr(chunker, "chunk_document") else []
        if not chunks:
            chunks = ICAChunker()._fallback_chunk(article, text)
    else:
        chunker = ArticleChunker()
        article = {"doc_id": doc_id, "text": text, "metadata": {}}
        chunks = chunker.chunk(article)
    return [c.get("text", "") for c in chunks]


def _try_chunk(text: str, doc_id: str) -> List[str]:
    """Robust wrapper — falls back to ArticleChunker if ICAChunker errors."""
    try:
        chunker = ArticleChunker()
        return [c["text"] for c in chunker.chunk({"text": text, "metadata": {}})]
    except Exception as exc:
        print(f"   [!] chunker error: {exc} — falling back to whole text")
        return [text]


def main() -> int:
    queries = list(csv.DictReader(GOLDEN.open()))
    by_doc: Dict[str, List[dict]] = {}
    for q in queries:
        by_doc.setdefault(q["expected_doc_id"], []).append(q)

    rows = []
    for doc_id, qs in by_doc.items():
        path = DOC_TO_PATH.get(doc_id)
        if not path or not path.exists():
            continue
        print(f"=== {doc_id} ({path.name})")

        text_a = extract_current(path)
        text_b = extract_pymupdf_md(path)
        text_c = extract_pdfplumber_md(path)

        chunks_a = _try_chunk(text_a, doc_id)
        chunks_b = _try_chunk(text_b, doc_id)
        chunks_c = _try_chunk(text_c, doc_id)
        print(f"   chunks  A={len(chunks_a)}  B={len(chunks_b)}  C={len(chunks_c)}")

        for q in qs:
            cell = q["expected_cell_value"]
            label = _label_token(q.get("expected_table_label", ""))
            r = {"query_id": q["query_id"], "doc_id": doc_id,
                 "cell": cell, "label": label}

            for tag, cs in (("A", chunks_a), ("B", chunks_b), ("C", chunks_c)):
                cell_only = sum(1 for c in cs if _has(cell, c))
                with_ctx = sum(1 for c in cs if _has(cell, c) and (not label or _has(label, c)))
                r[f"chunks_with_cell_{tag}"] = cell_only
                r[f"chunks_with_cell_AND_label_{tag}"] = with_ctx
            rows.append(r)
            print(
                f"   {r['query_id']} cell={cell[:25]!r:25s} "
                f"cell-only A:{r['chunks_with_cell_A']} B:{r['chunks_with_cell_B']} C:{r['chunks_with_cell_C']}  "
                f"+label A:{r['chunks_with_cell_AND_label_A']} B:{r['chunks_with_cell_AND_label_B']} C:{r['chunks_with_cell_AND_label_C']}"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "dry_run_real_chunker.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        if not rows:
            print("[!] no rows produced")
            return 1
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    md = ["# Dry-run with PRODUCTION chunker (alt. 1+2)"]
    md.append(f"\nQueries: **{n}**\n")
    md.append("## Cell + table-label co-occurrence in any production chunk\n")
    md.append("| Method | cell present | cell + label co-located |")
    md.append("|---|---|---|")
    for tag, label in (("A", "A — current"), ("B", "B — PyMuPDF MD"), ("C", "C — pdfplumber MD")):
        n_cell = sum(1 for r in rows if r[f"chunks_with_cell_{tag}"] > 0)
        n_ctx = sum(1 for r in rows if r[f"chunks_with_cell_AND_label_{tag}"] > 0)
        md.append(f"| {label} | {n_cell}/{n} = {100*n_cell/n:.1f}% | {n_ctx}/{n} = {100*n_ctx/n:.1f}% |")

    md.append("\n## Per-query (counts of chunks containing cell+label)\n")
    md.append("| query | doc | cell | A_cell | B_cell | C_cell | A_ctx | B_ctx | C_ctx |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['query_id']} | {r['doc_id']} | `{r['cell']}` "
            f"| {r['chunks_with_cell_A']} | {r['chunks_with_cell_B']} | {r['chunks_with_cell_C']} "
            f"| {r['chunks_with_cell_AND_label_A']} | {r['chunks_with_cell_AND_label_B']} | {r['chunks_with_cell_AND_label_C']} |"
        )

    out_md = OUT_DIR / "dry_run_real_chunker.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote:\n  - {out_csv.relative_to(ROOT)}\n  - {out_md.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
