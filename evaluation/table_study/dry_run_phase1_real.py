"""End-to-end validation of the Phase 1 + Phase 2 implementation.

Runs the *real* parser ``extract_text_with_tables`` followed by the
*real* production chunker (``ICAChunker`` for ICAs/RBACs) and measures
the same metrics the dry-run simulator measured, but now over the
actual code that will run in production:

  * COVERAGE  — cell present in any ``chunk_type="table"`` chunk
  * has_title — chunk has a captured title in metadata
  * has_label_kw — chunk text contains keywords from the golden label
  * PRIMARY   — COVERAGE AND (has_title OR has_label_kw)

If COVERAGE/PRIMARY land near the simulator's 92% (data/_table_study/
dry_run_phase1_v2.md), the implementation reproduces the validated
behaviour. Anything materially below that means a bug was introduced.

Output:
  * data/_table_study/dry_run_phase1_real.csv
  * data/_table_study/dry_run_phase1_real.md
"""
from __future__ import annotations

import csv
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from parsers.table_extractor import extract_text_with_tables  # noqa: E402
from pipeline.chunking import ICAChunker  # noqa: E402

GOLDEN = ROOT / "evaluation" / "golden_set_tables.csv"
OUT_DIR = ROOT / "data" / "_table_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOC_TO_PATH = {
    "RBAC-43":    ROOT / "data/originals/anac/rbac-43.pdf",
    "RBAC-26":    ROOT / "data/originals/anac/rbac-26.pdf",
    "RBAC-105":   ROOT / "data/originals/anac/rbac-105.pdf",
    "ICA-100-12": ROOT / "data/originals/sislaer/ica-100-12.pdf",
    "ICA-100-37": ROOT / "data/originals/sislaer/ica-100-37.pdf",
}

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
    s = _norm(descriptive_label)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return {t for t in s.split() if len(t) >= 4 and t not in STOPWORDS}


@dataclass
class Row:
    query_id: str
    doc_id: str
    cell: str
    in_any_table_chunk: bool
    chunk_has_title: bool
    chunk_has_label_kw: bool
    primary_pass: bool
    chunk_text_size: int
    chunk_metadata_title: str
    chunk_table_id: str
    n_table_chunks_in_doc: int


def main() -> int:
    queries = list(csv.DictReader(GOLDEN.open()))
    by_doc: Dict[str, List[dict]] = {}
    for q in queries:
        by_doc.setdefault(q["expected_doc_id"], []).append(q)

    results: List[Row] = []

    for doc_id, qs in by_doc.items():
        path = DOC_TO_PATH.get(doc_id)
        if not path or not path.exists():
            print(f"[!] missing {doc_id}")
            continue
        print(f"=== {doc_id} ({path.name})")

        text = extract_text_with_tables(path.read_bytes())
        chunker = ICAChunker()
        chunks = chunker.chunk({
            "regulation_id": doc_id,
            "text": text,
            "metadata": {"doc_type": "ica" if doc_id.startswith("ICA") else "rbac"},
        })
        table_chunks = [c for c in chunks if c.get("chunk_type") == "table"]
        print(f"   chunks={len(chunks)} (table={len(table_chunks)})")

        for q in qs:
            cell = q["expected_cell_value"]
            kw = _label_keywords(q.get("expected_table_label", ""))

            # Score every table chunk that contains the cell, then pick
            # the one that ALSO matches title or label keywords. Same
            # criterion the dry_run_phase1_v2 simulator uses.
            best = None
            best_score = -1
            for tc in table_chunks:
                if not _has(cell, tc["text"]):
                    continue
                meta = tc.get("metadata") or {}
                tc_title = (meta.get("table_title") or "").strip()
                tc_norm = _norm(tc["text"])
                score = 0
                if tc_title:
                    score += 2
                if kw and any(k in tc_norm for k in kw):
                    score += 1
                if score > best_score:
                    best_score = score
                    best = tc
                    if score == 3:
                        break  # best possible match — short-circuit

            in_any = best is not None
            if best is not None:
                meta = best.get("metadata") or {}
                title = (meta.get("table_title") or "").strip()
                table_id = meta.get("table_id") or ""
                txt_norm = _norm(best["text"])
                has_title = bool(title)
                has_kw = any(k in txt_norm for k in kw) if kw else False
                size = len(best["text"])
            else:
                title = ""
                table_id = ""
                has_title = has_kw = False
                size = 0

            row = Row(
                query_id=q["query_id"],
                doc_id=doc_id,
                cell=cell,
                in_any_table_chunk=in_any,
                chunk_has_title=has_title,
                chunk_has_label_kw=has_kw,
                primary_pass=in_any and (has_title or has_kw),
                chunk_text_size=size,
                chunk_metadata_title=title,
                chunk_table_id=table_id,
                n_table_chunks_in_doc=len(table_chunks),
            )
            results.append(row)
            print(
                f"   {row.query_id:6s} cell={cell[:25]!r:27s} "
                f"in_table={int(row.in_any_table_chunk)} "
                f"title={int(row.chunk_has_title)} "
                f"kw={int(row.chunk_has_label_kw)} "
                f"PRIMARY={int(row.primary_pass)}"
            )

    n = len(results)
    csv_path = OUT_DIR / "dry_run_phase1_real.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(Row.__dataclass_fields__.keys()),
        )
        w.writeheader()
        for r in results:
            w.writerow(r.__dict__)

    n_in = sum(1 for r in results if r.in_any_table_chunk)
    n_t = sum(1 for r in results if r.chunk_has_title)
    n_k = sum(1 for r in results if r.chunk_has_label_kw)
    n_p = sum(1 for r in results if r.primary_pass)

    md = []
    md.append("# Validacao end-to-end — Fase 1 + Fase 2 implementadas")
    md.append("")
    md.append(f"Queries avaliadas: **{n}**")
    md.append("")
    md.append("Comparacao com a simulacao de Fase 1 (dry_run_phase1_v2.md):")
    md.append("")
    md.append("| Metrica | Simulacao (v2) | Implementacao real | Delta |")
    md.append("|---|---|---|---|")
    md.append(f"| COVERAGE | 23/25 = 92% | {n_in}/{n} = {100*n_in/n:.1f}% | {n_in-23:+d} |")
    md.append(f"| has_title | 19/25 = 76% | {n_t}/{n} = {100*n_t/n:.1f}% | {n_t-19:+d} |")
    md.append(f"| has_label_kw | 14/25 = 56% | {n_k}/{n} = {100*n_k/n:.1f}% | {n_k-14:+d} |")
    md.append(f"| **PRIMARY** | **23/25 = 92%** | **{n_p}/{n} = {100*n_p/n:.1f}%** | **{n_p-23:+d}** |")
    md.append("")
    md.append("Como referencia: baseline atual (sem Fase 1) atinge 17/25 = 68%.")
    md.append("")
    md.append("## Por documento")
    md.append("")
    md.append("| doc | N | COVERAGE | has_title | has_label_kw | PRIMARY |")
    md.append("|---|---|---|---|---|---|")
    by_d: Dict[str, List[Row]] = {}
    for r in results:
        by_d.setdefault(r.doc_id, []).append(r)
    for d, rs in sorted(by_d.items()):
        n_d = len(rs)
        md.append(
            f"| {d} | {n_d} "
            f"| {sum(r.in_any_table_chunk for r in rs)}/{n_d} "
            f"| {sum(r.chunk_has_title for r in rs)}/{n_d} "
            f"| {sum(r.chunk_has_label_kw for r in rs)}/{n_d} "
            f"| {sum(r.primary_pass for r in rs)}/{n_d} |"
        )
    md.append("")
    md.append("## Por query")
    md.append("")
    md.append("| query | doc | cell | in_table | title | kw | PRIMARY | titulo_capt | size |")
    md.append("|---|---|---|---|---|---|---|---|---|")

    def yn(b):
        return "OK" if b else "--"

    for r in results:
        md.append(
            f"| {r.query_id} | {r.doc_id} | `{r.cell}` "
            f"| {yn(r.in_any_table_chunk)} | {yn(r.chunk_has_title)} "
            f"| {yn(r.chunk_has_label_kw)} "
            f"| **{yn(r.primary_pass)}** "
            f"| {r.chunk_metadata_title or '-'} "
            f"| {r.chunk_text_size or '-'} |"
        )

    md_path = OUT_DIR / "dry_run_phase1_real.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote:\n  - {csv_path.relative_to(ROOT)}\n  - {md_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
