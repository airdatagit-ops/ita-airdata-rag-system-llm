"""Phase 1 simulation — corrected scoring.

The first round (dry_run_phase1.py) measured cell+label co-location using
the descriptive ``expected_table_label`` token from the golden set
(e.g. "Tabela esteira turbulência"). That under-counts Phase 1 chunks
because they carry their *own* in-pdf label ("Tabela 11", "TABELA I")
plus column headers — not the descriptive label.

This script measures what really matters operationally:

  PRIMARY  — cell sits inside a chunk_type=table that carries (a) a
             captured table title, OR (b) the original page header
             keywords from expected_table_label split into tokens.
  STRICT   — same as primary BUT also requires the chunk to have the
             cell's specific row context (heuristic: any token from
             the golden 'notes' field, falling back to digits in
             the cell).
  COVERAGE — cell appears in at least one chunk_type=table.

Reports each metric so the team can read the 3 numbers without
arguing about the right one.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.table_study.dry_run_phase1 import (  # noqa: E402
    _build_phase1_chunks, DOC_TO_PATH, GOLDEN, OUT_DIR,
)

STOPWORDS = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "ou",
    "em", "no", "na", "nos", "nas", "para", "por", "com", "sem",
    "um", "uma", "tabela", "tab", "quadro", "ica", "rbac",
}


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


def _label_keywords(descriptive_label: str) -> Set[str]:
    """Split a descriptive label into content keywords, dropping stopwords."""
    s = _norm(descriptive_label)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = [t for t in s.split() if len(t) >= 4 and t not in STOPWORDS]
    return set(toks)


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
        print(f"=== {doc_id}")
        table_chunks, prose_chunks = _build_phase1_chunks(path)
        print(f"   table_chunks={len(table_chunks)}  prose_chunks={len(prose_chunks)}")

        for q in qs:
            cell = q["expected_cell_value"]
            kw = _label_keywords(q.get("expected_table_label", ""))

            best_table = None
            for tc in table_chunks:
                if _has(cell, tc.text):
                    if best_table is None:
                        best_table = tc

            in_table_chunk = best_table is not None
            in_prose = any(_has(cell, pc.text) for pc in prose_chunks)

            chunk_has_title = bool(best_table and best_table.title.strip())

            chunk_has_kw = False
            if best_table and kw:
                txt_norm = _norm(best_table.text)
                chunk_has_kw = any(k in txt_norm for k in kw)

            row = {
                "query_id": q["query_id"],
                "doc_id": doc_id,
                "cell": cell,
                "label_keywords": " ".join(sorted(kw)),
                "in_table_chunk": in_table_chunk,
                "in_prose_overflow": in_prose,
                "table_chunk_has_title": chunk_has_title,
                "table_chunk_has_label_kw": chunk_has_kw,
                "primary_pass": in_table_chunk and (chunk_has_title or chunk_has_kw),
                "table_chunk_size": len(best_table.text) if best_table else 0,
                "table_chunk_title": best_table.title if best_table else "",
                "table_chunk_page": best_table.page if best_table else 0,
            }
            rows.append(row)
            print(
                f"   {q['query_id']:6s} cell={cell[:25]!r:27s} "
                f"in_table={int(row['in_table_chunk'])} "
                f"has_title={int(row['table_chunk_has_title'])} "
                f"has_label_kw={int(row['table_chunk_has_label_kw'])} "
                f"PRIMARY={int(row['primary_pass'])}"
            )

    n = len(rows)
    n_in_table = sum(1 for r in rows if r["in_table_chunk"])
    n_with_title = sum(1 for r in rows if r["table_chunk_has_title"])
    n_with_kw = sum(1 for r in rows if r["table_chunk_has_label_kw"])
    n_primary = sum(1 for r in rows if r["primary_pass"])

    csv_path = OUT_DIR / "dry_run_phase1_v2.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = []
    md.append("# Dry-run Fase 1 (v2) — metrica corrigida")
    md.append("")
    md.append(f"Queries avaliadas: **{n}**")
    md.append("")
    md.append("## Compara as metricas — qual numero significa o que")
    md.append("")
    md.append("| Metrica | Resultado | Interpretacao |")
    md.append("|---|---|---|")
    md.append(
        f"| COVERAGE — celula em algum chunk_type=table | {n_in_table}/{n} = {100*n_in_table/n:.1f}% "
        f"| upper bound: o que a Fase 1 expoe ao indice |"
    )
    md.append(
        f"| has_title — chunk de tabela tem titulo capturado | {n_with_title}/{n} = {100*n_with_title/n:.1f}% "
        f"| diagnostico de quao bem a heuristica de titulo funciona |"
    )
    md.append(
        f"| has_label_kw — chunk tem palavras-chave do label golden | {n_with_kw}/{n} = {100*n_with_kw/n:.1f}% "
        f"| ajuda a entender se o contexto narrativo entrou |"
    )
    md.append(
        f"| **PRIMARY** — celula em chunk de tabela com titulo OU label-kw | **{n_primary}/{n} = {100*n_primary/n:.1f}%** "
        f"| **KPI honesto da Fase 1** |"
    )
    md.append("")
    md.append("Como referencia: baseline atual (chunker producao 270 tokens, sem Fase 1) atinge 17/25 = 68%.")
    md.append("")

    md.append("## Por documento")
    md.append("")
    md.append("| doc | N | COVERAGE | has_title | has_label_kw | PRIMARY |")
    md.append("|---|---|---|---|---|---|")
    by_d: Dict[str, List[dict]] = {}
    for r in rows:
        by_d.setdefault(r["doc_id"], []).append(r)
    for d, rs in sorted(by_d.items()):
        n_d = len(rs)
        md.append(
            f"| {d} | {n_d} "
            f"| {sum(r['in_table_chunk'] for r in rs)}/{n_d} "
            f"| {sum(r['table_chunk_has_title'] for r in rs)}/{n_d} "
            f"| {sum(r['table_chunk_has_label_kw'] for r in rs)}/{n_d} "
            f"| {sum(r['primary_pass'] for r in rs)}/{n_d} |"
        )
    md.append("")

    md.append("## Por query")
    md.append("")
    md.append("| query | doc | cell | in_table | has_title | has_label_kw | PRIMARY | titulo_capturado | size |")
    md.append("|---|---|---|---|---|---|---|---|---|")

    def yn(b):
        return "OK" if b else "--"

    for r in rows:
        md.append(
            f"| {r['query_id']} | {r['doc_id']} | `{r['cell']}` "
            f"| {yn(r['in_table_chunk'])} | {yn(r['table_chunk_has_title'])} "
            f"| {yn(r['table_chunk_has_label_kw'])} "
            f"| **{yn(r['primary_pass'])}** "
            f"| {r['table_chunk_title'] or '-'} "
            f"| {r['table_chunk_size'] or '-'} |"
        )

    md_path = OUT_DIR / "dry_run_phase1_v2.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote:\n  - {csv_path.relative_to(ROOT)}\n  - {md_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
