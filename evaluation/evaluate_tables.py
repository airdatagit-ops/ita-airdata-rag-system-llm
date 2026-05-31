"""Table-bound RAG evaluation.

Measures, per stage, whether the golden ``expected_cell_value`` survives
through the pipeline:

  - retrieval (any chunk in top-k from any sub-query)
  - evaluator-pass (chunks accepted by the cross-encoder)
  - generator-context (the text the LLM actually saw)
  - final-answer (the LLM output)

Plus a doc-hit metric per stage: did the chunk whose ``regulation_id``
matches the expected source make it through?

Reads ``evaluation/golden_set_tables.csv`` and writes:

  - ``evaluation/results/tables_<timestamp>.csv`` — one row per query
  - ``evaluation/results/tables_<timestamp>.md``  — aggregate summary

Usage:
    python -m evaluation.evaluate_tables
    python -m evaluation.evaluate_tables --no-generate --k 5

The ``--no-generate`` flag short-circuits the LLM (huge speedup) but
disables the ``answer_cell_hit`` column.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "evaluation" / "golden_set_tables.csv"
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"


# ---------------------------------------------------------------------
# Normalization & matching helpers
# ---------------------------------------------------------------------


def _normalize_text(s: str) -> str:
    """Collapse whitespace, casefold, strip accents — for forgiving cell matches."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s).strip().casefold()
    return s


def _cell_in_text(cell: str, text: str) -> bool:
    """Substring match with normalization. Empty cell never matches."""
    n_cell = _normalize_text(cell)
    if not n_cell:
        return False
    return n_cell in _normalize_text(text)


def _doc_match_token(expected_doc_id: str) -> str:
    """Build a substring token used to match retrieved regulation_id.

    The current normalizer in evaluate_retrieval has a bug for ``anac_rbac_*``
    chunks (it never strips the source-prefix). We sidestep it by matching
    the canonical token ``rbac-43`` / ``ica_100-12`` against the lowercase
    retrieved id.
    """
    e = expected_doc_id.lower().strip()
    e = e.replace(" ", "")
    if e.startswith("rbac-"):
        return e
    if e.startswith("ica-"):
        return "ica_" + e[len("ica-"):]
    return e


def _doc_in_retrieved(expected_doc_id: str, retrieved_id: str) -> bool:
    if not retrieved_id:
        return False
    token = _doc_match_token(expected_doc_id)
    return token in retrieved_id.lower()


# ---------------------------------------------------------------------
# Golden-set IO
# ---------------------------------------------------------------------


@dataclass
class GoldenItem:
    query_id: str
    query: str
    expected_doc_id: str
    expected_cell_value: str
    expected_table_label: str
    relevance: str
    category: str
    notes: str = ""


def load_golden(path: Path) -> List[GoldenItem]:
    if not path.exists():
        raise FileNotFoundError(path)
    items: List[GoldenItem] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            items.append(GoldenItem(
                query_id=row["query_id"].strip(),
                query=row["query"].strip(),
                expected_doc_id=row["expected_doc_id"].strip(),
                expected_cell_value=row["expected_cell_value"].strip(),
                expected_table_label=row.get("expected_table_label", "").strip(),
                relevance=row.get("relevance", "relevant").strip(),
                category=row.get("category", "table_bound").strip(),
                notes=row.get("notes", "").strip(),
            ))
    if not items:
        raise ValueError(f"Empty golden set: {path}")
    return items


# ---------------------------------------------------------------------
# Per-stage analysis
# ---------------------------------------------------------------------


@dataclass
class StageResult:
    """Snapshot of what each pipeline stage reveals for one query."""
    query_id: str = ""
    query: str = ""
    expected_doc_id: str = ""
    expected_cell_value: str = ""
    expected_table_label: str = ""

    n_subqueries: int = 0
    n_retrieved: int = 0
    n_evaluator: int = 0
    n_evaluator_accepted: int = 0
    n_generator: int = 0

    retrieval_doc_hit: bool = False
    retrieval_cell_hit: bool = False
    retrieval_doc_with_cell: int = 0     # how many retrieved chunks contained the cell
    retrieval_first_doc_rank: Optional[int] = None

    evaluator_doc_hit: bool = False
    evaluator_cell_hit: bool = False
    evaluator_top_score_for_doc: Optional[float] = None

    generator_doc_hit: bool = False
    generator_cell_hit: bool = False

    answer_cell_hit: Optional[bool] = None    # None when --no-generate
    answer_excerpt: str = ""

    total_time_ms: int = 0
    errors: str = ""


def _flatten_search_docs(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    seen_ids: set = set()
    flat: List[Dict[str, Any]] = []
    for sq, docs in (trace.get("search_documents_per_query") or {}).items():
        for d in docs or []:
            rid = d.get("regulation_id", "")
            key = (rid, d.get("text", "")[:60])
            if key in seen_ids:
                continue
            seen_ids.add(key)
            flat.append(d)
    return flat


def analyze(item: GoldenItem, response: Dict[str, Any], *, generated: bool) -> StageResult:
    trace = response.get("trace") or {}
    res = StageResult(
        query_id=item.query_id,
        query=item.query,
        expected_doc_id=item.expected_doc_id,
        expected_cell_value=item.expected_cell_value,
        expected_table_label=item.expected_table_label,
        total_time_ms=int(response.get("total_time_ms", 0)),
        errors="; ".join(trace.get("errors") or []),
    )

    res.n_subqueries = len(trace.get("rewritten_queries") or [])

    retrieved = _flatten_search_docs(trace)
    res.n_retrieved = len(retrieved)
    for rank, d in enumerate(retrieved):
        rid = d.get("regulation_id", "")
        text = d.get("text", "") or ""
        is_doc = _doc_in_retrieved(item.expected_doc_id, rid)
        has_cell = _cell_in_text(item.expected_cell_value, text)
        if is_doc:
            res.retrieval_doc_hit = True
            if res.retrieval_first_doc_rank is None:
                res.retrieval_first_doc_rank = rank
        if has_cell:
            res.retrieval_cell_hit = True
            res.retrieval_doc_with_cell += 1

    eval_scores = trace.get("evaluation_scores") or []
    res.n_evaluator = len(eval_scores)
    accepted = [e for e in eval_scores if e.get("accepted")]
    res.n_evaluator_accepted = len(accepted)
    for e in accepted:
        rid = e.get("regulation_id", "")
        eval_text = e.get("eval_text", "") or ""
        if _doc_in_retrieved(item.expected_doc_id, rid):
            res.evaluator_doc_hit = True
            score = e.get("score")
            if isinstance(score, (int, float)):
                if res.evaluator_top_score_for_doc is None or score > res.evaluator_top_score_for_doc:
                    res.evaluator_top_score_for_doc = float(score)
        if _cell_in_text(item.expected_cell_value, eval_text):
            res.evaluator_cell_hit = True

    gen_docs = trace.get("generator_documents") or []
    res.n_generator = len(gen_docs)
    for d in gen_docs:
        rid = d.get("regulation_id", "")
        text = d.get("text", "") or ""
        if _doc_in_retrieved(item.expected_doc_id, rid):
            res.generator_doc_hit = True
        if _cell_in_text(item.expected_cell_value, text):
            res.generator_cell_hit = True

    if generated:
        answer = response.get("answer", "") or ""
        res.answer_cell_hit = _cell_in_text(item.expected_cell_value, answer)
        ex = answer.replace("\n", " ").strip()
        res.answer_excerpt = (ex[:280] + "…") if len(ex) > 280 else ex
    else:
        res.answer_cell_hit = None

    return res


# ---------------------------------------------------------------------
# Aggregation & output
# ---------------------------------------------------------------------


def _rate(num: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{num}/{denom} = {100.0*num/denom:.1f}%"


def write_csv(results: List[StageResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(StageResult.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: ("" if v is None else v) for k, v in asdict(r).items()}
            w.writerow(row)


def write_markdown(results: List[StageResult], path: Path, generated: bool) -> None:
    n = len(results)
    n_doc_ret = sum(1 for r in results if r.retrieval_doc_hit)
    n_cell_ret = sum(1 for r in results if r.retrieval_cell_hit)
    n_doc_eval = sum(1 for r in results if r.evaluator_doc_hit)
    n_cell_eval = sum(1 for r in results if r.evaluator_cell_hit)
    n_doc_gen = sum(1 for r in results if r.generator_doc_hit)
    n_cell_gen = sum(1 for r in results if r.generator_cell_hit)
    n_ans = sum(1 for r in results if r.answer_cell_hit)

    out = []
    out.append(f"# Table-bound RAG baseline — {datetime.now().isoformat(timespec='seconds')}")
    out.append("")
    out.append(f"Queries: **{n}** — generation: **{'on' if generated else 'off'}**")
    out.append("")
    out.append("## Stage-by-stage hit rate")
    out.append("")
    out.append("| Stage | Doc hit | Cell hit |")
    out.append("|---|---|---|")
    out.append(f"| retrieval (top-k union sub-queries) | {_rate(n_doc_ret, n)} | {_rate(n_cell_ret, n)} |")
    out.append(f"| evaluator-accepted | {_rate(n_doc_eval, n)} | {_rate(n_cell_eval, n)} |")
    out.append(f"| generator context | {_rate(n_doc_gen, n)} | {_rate(n_cell_gen, n)} |")
    if generated:
        out.append(f"| final answer | n/a | {_rate(n_ans, n)} |")
    out.append("")

    by_doc: Dict[str, List[StageResult]] = {}
    for r in results:
        by_doc.setdefault(r.expected_doc_id, []).append(r)
    out.append("## Per-document breakdown")
    out.append("")
    out.append("| Document | N | retrieval_doc | retrieval_cell | eval_cell | gen_cell | answer_cell |")
    out.append("|---|---|---|---|---|---|---|")
    for doc, rs in sorted(by_doc.items()):
        n_d = len(rs)
        out.append(
            f"| {doc} | {n_d} "
            f"| {_rate(sum(r.retrieval_doc_hit for r in rs), n_d)} "
            f"| {_rate(sum(r.retrieval_cell_hit for r in rs), n_d)} "
            f"| {_rate(sum(r.evaluator_cell_hit for r in rs), n_d)} "
            f"| {_rate(sum(r.generator_cell_hit for r in rs), n_d)} "
            f"| {_rate(sum(bool(r.answer_cell_hit) for r in rs), n_d) if generated else 'n/a'} |"
        )
    out.append("")

    out.append("## Per-query")
    out.append("")
    out.append("| query_id | doc | retr | eval | gen | ans | first_doc_rank | accepted |")
    out.append("|---|---|---|---|---|---|---|---|")

    def yn(b: Optional[bool]) -> str:
        if b is None:
            return "—"
        return "✓" if b else "✗"

    for r in results:
        out.append(
            f"| {r.query_id} | {r.expected_doc_id} "
            f"| {yn(r.retrieval_cell_hit)} | {yn(r.evaluator_cell_hit)} "
            f"| {yn(r.generator_cell_hit)} | {yn(r.answer_cell_hit)} "
            f"| {r.retrieval_first_doc_rank if r.retrieval_first_doc_rank is not None else '—'} "
            f"| {r.n_evaluator_accepted}/{r.n_evaluator} |"
        )
    out.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    items = load_golden(GOLDEN_PATH)
    if args.sample:
        items = items[: args.sample]
    logger.info(f"Loaded {len(items)} table-bound queries from {GOLDEN_PATH.name}")

    from search.orchestrator.pipeline import RAGPipeline

    pipeline = RAGPipeline()

    results: List[StageResult] = []
    overall = time.time()
    for i, item in enumerate(items, 1):
        t0 = time.time()
        try:
            response = pipeline.query(
                item.query,
                limit=args.k,
                debug=True,
                return_sources=True,
                include_generation=not args.no_generate,
            )
        except Exception as exc:
            logger.error(f"[{item.query_id}] pipeline failed: {exc.__class__.__name__}: {exc}")
            response = {
                "answer": "",
                "sources": [],
                "total_time_ms": int((time.time() - t0) * 1000),
                "trace": {
                    "rewritten_queries": [],
                    "search_documents_per_query": {},
                    "evaluation_scores": [],
                    "generator_documents": [],
                    "errors": [f"{exc.__class__.__name__}: {exc}"],
                    "timings": {},
                },
            }
        r = analyze(item, response, generated=not args.no_generate)
        elapsed = int((time.time() - t0) * 1000)
        if not r.total_time_ms:
            r.total_time_ms = elapsed
        results.append(r)
        logger.info(
            f"[{item.query_id}] ({i}/{len(items)}) "
            f"retr_cell={r.retrieval_cell_hit} eval_cell={r.evaluator_cell_hit} "
            f"gen_cell={r.generator_cell_hit} ans_cell={r.answer_cell_hit} "
            f"({elapsed}ms)"
        )

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    suffix = "nogen" if args.no_generate else "full"
    csv_out = RESULTS_DIR / f"tables_{suffix}_{ts}.csv"
    md_out = RESULTS_DIR / f"tables_{suffix}_{ts}.md"
    write_csv(results, csv_out)
    write_markdown(results, md_out, generated=not args.no_generate)

    elapsed_total = time.time() - overall
    logger.success(
        f"Done in {elapsed_total:.1f}s. Wrote:\n"
        f"  - {csv_out.relative_to(REPO_ROOT)}\n"
        f"  - {md_out.relative_to(REPO_ROOT)}"
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Table-bound RAG baseline evaluator")
    p.add_argument("--k", type=int, default=5,
                   help="Documents per sub-query passed to the pipeline (default 5)")
    p.add_argument("--sample", type=int, default=None,
                   help="Limit to first N golden queries (smoke test)")
    p.add_argument("--no-generate", action="store_true",
                   help="Skip LLM generation (faster; answer_cell_hit unavailable)")
    args = p.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
