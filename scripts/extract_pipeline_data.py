"""RAG Pipeline data extractor.

Runs the full ``RAGPipeline`` on every query of an input CSV/XLSX and
produces a single ``.xlsx`` (sheet ``exploded``) with one row per
(query, sub-query, retrieved document). Each row carries the rewriter
output (sub-query text, facet, filters, sorts), the searcher output
(regulation_id, url, search score), the evaluator score and the
**exact enriched text** that the cross-encoder actually scored
(``evaluator_text``, plus ``_chars`` and ``_max_tokens`` for quick
filtering), a ``sent_to_generator`` flag, the **full text** the
document contributed to the generator context (only for docs the
generator actually saw), and the final answer.

Designed for offline analysis: every stage of the pipeline becomes a
column you can pivot/filter in Excel without re-running the LLM.

Input
-----
A ``.csv`` or ``.xlsx`` with a required ``query`` column. Other
columns (e.g. ``query_id``, ``expected_doc_id``, ``category``,
``notes``) are passed through into the output as ``input__<col>``.
If ``query_id`` is missing, an auto id (``Q001``, ``Q002``, …) is
generated.

Output
------
``data/pipeline_extracts/extract_<timestamp>.xlsx`` (override with
``--output``). Single sheet ``exploded``. Every input query produces
at least one row even when the pipeline returns 0 documents
(``errors`` column will indicate the failure).

Examples
--------
    # Full run (uses Ollama + Qdrant)
    python -m scripts.extract_pipeline_data \\
        --input evaluation/golden_set_gen_sample.csv

    # Limit to 3 queries, smaller K, skip the generator (much faster)
    python -m scripts.extract_pipeline_data \\
        --input evaluation/golden_set.csv \\
        --sample 3 --k 3 --no-generate

    # Custom output path (.xlsx)
    python -m scripts.extract_pipeline_data \\
        --input my_questions.xlsx \\
        --output data/pipeline_extracts/my_run.xlsx
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


REQUIRED_INPUT_COLUMN = "query"
INPUT_PASSTHROUGH_PREFIX = "input__"
DEFAULT_OUTPUT_DIR = Path("data/pipeline_extracts")
SHEET_NAME = "exploded"


# ---------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------


@dataclass
class InputRow:
    """One row from the input file."""
    query_id: str
    query: str
    extras: Dict[str, Any] = field(default_factory=dict)


def load_input(path: Path) -> List[InputRow]:
    """Load a ``.csv`` or ``.xlsx`` and return ``InputRow`` records.

    Raises ``ValueError`` if the file is missing the ``query`` column
    or contains zero usable rows.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str, engine="openpyxl")
        df = df.fillna("")
    else:
        raise ValueError(
            f"Unsupported input format '{suffix}'. Use .csv or .xlsx."
        )

    if REQUIRED_INPUT_COLUMN not in df.columns:
        raise ValueError(
            f"Input file is missing required column "
            f"'{REQUIRED_INPUT_COLUMN}'. Found columns: {list(df.columns)}"
        )

    rows: List[InputRow] = []
    for idx, raw in df.iterrows():
        query = str(raw.get(REQUIRED_INPUT_COLUMN, "")).strip()
        if not query:
            continue
        explicit_id = str(raw.get("query_id", "")).strip()
        query_id = explicit_id or f"Q{idx + 1:03d}"
        extras = {
            col: ("" if raw[col] is None else str(raw[col]))
            for col in df.columns
            if col not in (REQUIRED_INPUT_COLUMN, "query_id")
        }
        rows.append(InputRow(query_id=query_id, query=query, extras=extras))

    if not rows:
        raise ValueError(f"No usable rows found in {path}")

    return rows


# ---------------------------------------------------------------------
# Trace -> exploded rows
# ---------------------------------------------------------------------


def _filters_to_compact(filters: List[Dict[str, Any]]) -> str:
    """Render filter list as a compact JSON string for Excel cells."""
    if not filters:
        return ""
    return json.dumps(filters, ensure_ascii=False, separators=(",", ":"))


def _sorts_to_compact(sorts: List[Dict[str, Any]]) -> str:
    if not sorts:
        return ""
    return json.dumps(sorts, ensure_ascii=False, separators=(",", ":"))


def _index_evaluator_scores(
    evaluation_scores: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Map regulation_id -> {score, accepted, eval_text, eval_max_tokens}.

    The trace lists every doc that reached the evaluator (accepted +
    discarded), so a regulation_id missing from the map means it was
    filtered out before that stage (or the evaluator was disabled).

    ``eval_text`` is the exact enriched/truncated text the cross-encoder
    saw — handy for debugging "why did this doc score X?" without re-
    running the model.
    """
    by_id: Dict[str, Dict[str, Any]] = {}
    for entry in evaluation_scores or []:
        rid = entry.get("regulation_id")
        if not rid:
            continue
        by_id[rid] = {
            "score": entry.get("score"),
            "accepted": bool(entry.get("accepted", False)),
            "eval_text": entry.get("eval_text", "") or "",
            "eval_max_tokens": entry.get("eval_max_tokens", 0) or 0,
        }
    return by_id


def _index_generator_docs(
    generator_documents: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Map regulation_id -> generator-context payload (text + flags)."""
    by_id: Dict[str, Dict[str, Any]] = {}
    for entry in generator_documents or []:
        rid = entry.get("regulation_id")
        if not rid:
            continue
        by_id[rid] = {
            "text": entry.get("text", ""),
            "char_count": entry.get("char_count", 0),
            "truncated": bool(entry.get("truncated", False)),
        }
    return by_id


def _base_row(
    input_row: InputRow,
    response: Dict[str, Any],
    trace: Dict[str, Any],
    *,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the per-query identification fields shared by every row."""
    timings = (trace.get("timings") or {}) if trace else {}
    base: Dict[str, Any] = {
        "query_id": input_row.query_id,
        "query": input_row.query,
    }
    for k, v in input_row.extras.items():
        base[f"{INPUT_PASSTHROUGH_PREFIX}{k}"] = v

    base.update({
        "total_time_ms": (
            timings.get("total_ms")
            or response.get("total_time_ms", 0)
        ),
        "rewriter_ms": timings.get("rewriter_ms", 0),
        "searcher_ms": timings.get("searcher_ms", 0),
        "evaluator_ms": timings.get("evaluator_ms", 0),
        "generator_ms": timings.get("generator_ms", 0),
        "errors": "; ".join(trace.get("errors") or []) if trace else "",
        "final_answer": response.get("answer", ""),
        "generator_model": (trace or {}).get("generator_model", ""),
        "generator_grounded_only": (trace or {}).get(
            "generator_grounded_only", ""
        ),
    })
    if extra_fields:
        base.update(extra_fields)
    return base


def explode_response(
    input_row: InputRow,
    response: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert a single pipeline response into 1..N exploded rows.

    Guarantees:
    - Every input query yields at least one row (placeholder when the
      pipeline returned no docs / no sub-queries).
    - One row per (sub-query, retrieved document) when docs exist.
    - ``sent_to_generator=True`` only for docs whose ``regulation_id``
      appears in ``trace.generator_documents``.
    """
    trace = response.get("trace") or {}
    rewritten = trace.get("rewritten_queries") or []
    docs_per_query = trace.get("search_documents_per_query") or {}
    eval_by_id = _index_evaluator_scores(trace.get("evaluation_scores") or [])
    gen_by_id = _index_generator_docs(trace.get("generator_documents") or [])

    if not rewritten:
        return [_base_row(input_row, response, trace, extra_fields={
            "subquery_idx": 0,
            "subquery_text": input_row.query,
            "facet_type": "",
            "filters_json": "",
            "sorts_json": "",
            "doc_rank_in_subquery": None,
            "regulation_id": "",
            "doc_url": "",
            "doc_type": "",
            "doc_number": "",
            "doc_title": "",
            "search_score": None,
            "evaluator_score": None,
            "evaluator_accepted": None,
            "evaluator_text_chars": 0,
            "evaluator_max_tokens": 0,
            "evaluator_text": "",
            "sent_to_generator": False,
            "generator_text_chars": 0,
            "generator_truncated": False,
            "generator_text": "",
        })]

    rows: List[Dict[str, Any]] = []
    for sq_idx, sq in enumerate(rewritten):
        sq_text = sq.get("text", "")
        sq_filters = sq.get("filters") or []
        sq_sorts = sq.get("sorts") or []
        sq_facet = sq.get("facet_type", "")

        sq_docs = docs_per_query.get(sq_text) or []

        if not sq_docs:
            rows.append(_base_row(input_row, response, trace, extra_fields={
                "subquery_idx": sq_idx,
                "subquery_text": sq_text,
                "facet_type": sq_facet,
                "filters_json": _filters_to_compact(sq_filters),
                "sorts_json": _sorts_to_compact(sq_sorts),
                "doc_rank_in_subquery": None,
                "regulation_id": "",
                "doc_url": "",
                "doc_type": "",
                "doc_number": "",
                "doc_title": "",
                "search_score": None,
                "evaluator_score": None,
                "evaluator_accepted": None,
                "evaluator_text_chars": 0,
                "evaluator_max_tokens": 0,
                "evaluator_text": "",
                "sent_to_generator": False,
                "generator_text_chars": 0,
                "generator_truncated": False,
                "generator_text": "",
            }))
            continue

        for rank, doc in enumerate(sq_docs):
            rid = doc.get("regulation_id", "")
            eval_entry = eval_by_id.get(rid) or {}
            gen_entry = gen_by_id.get(rid) or {}
            sent = bool(gen_entry)

            rows.append(_base_row(input_row, response, trace, extra_fields={
                "subquery_idx": sq_idx,
                "subquery_text": sq_text,
                "facet_type": sq_facet,
                "filters_json": _filters_to_compact(sq_filters),
                "sorts_json": _sorts_to_compact(sq_sorts),
                "doc_rank_in_subquery": rank,
                "regulation_id": rid,
                "doc_url": doc.get("url", ""),
                "doc_type": doc.get("type", ""),
                "doc_number": doc.get("number", ""),
                "doc_title": doc.get("title", ""),
                "search_score": doc.get("score"),
                "evaluator_score": eval_entry.get("score"),
                "evaluator_accepted": eval_entry.get("accepted"),
                "evaluator_text_chars": len(eval_entry.get("eval_text", "") or ""),
                "evaluator_max_tokens": eval_entry.get("eval_max_tokens", 0) or 0,
                "evaluator_text": eval_entry.get("eval_text", "") or "",
                "sent_to_generator": sent,
                "generator_text_chars": gen_entry.get("char_count", 0),
                "generator_truncated": gen_entry.get("truncated", False),
                "generator_text": gen_entry.get("text", "") if sent else "",
            }))

    return rows


# ---------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------


_PREFERRED_COLUMN_ORDER: List[str] = [
    "query_id",
    "query",
    "subquery_idx",
    "subquery_text",
    "facet_type",
    "filters_json",
    "sorts_json",
    "doc_rank_in_subquery",
    "regulation_id",
    "doc_url",
    "doc_type",
    "doc_number",
    "doc_title",
    "search_score",
    "evaluator_score",
    "evaluator_accepted",
    "evaluator_text_chars",
    "evaluator_max_tokens",
    "evaluator_text",
    "sent_to_generator",
    "generator_text_chars",
    "generator_truncated",
    "generator_text",
    "final_answer",
    "generator_model",
    "generator_grounded_only",
    "rewriter_ms",
    "searcher_ms",
    "evaluator_ms",
    "generator_ms",
    "total_time_ms",
    "errors",
]


def _order_columns(columns: List[str]) -> List[str]:
    """Return columns in a deterministic, human-friendly order.

    Preferred fields come first, ``input__*`` passthrough fields right
    after the identification block, and any unexpected fields at the
    end (so we never silently drop a column).
    """
    preferred = [c for c in _PREFERRED_COLUMN_ORDER if c in columns]
    passthrough = sorted(
        c for c in columns if c.startswith(INPUT_PASSTHROUGH_PREFIX)
    )
    seen = set(preferred) | set(passthrough)
    leftover = [c for c in columns if c not in seen]

    ordered: List[str] = []
    for col in preferred:
        ordered.append(col)
        if col == "query":
            ordered.extend(passthrough)
    if not passthrough or "query" not in preferred:
        ordered.extend(passthrough)
    ordered.extend(leftover)
    seen_final: set = set()
    deduped: List[str] = []
    for col in ordered:
        if col not in seen_final:
            deduped.append(col)
            seen_final.add(col)
    return deduped


def write_xlsx(rows: List[Dict[str, Any]], output_path: Path) -> Path:
    """Write *rows* to ``output_path`` as a single-sheet ``.xlsx``.

    Auto-sizes columns up to a sensible cap so the file stays
    readable without manual fiddling.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame([{c: None for c in _PREFERRED_COLUMN_ORDER}])

    df = df.reindex(columns=_order_columns(list(df.columns)))

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=SHEET_NAME)
        sheet = writer.sheets[SHEET_NAME]
        for col_idx, col_name in enumerate(df.columns, start=1):
            sample = [str(v) for v in df[col_name].head(200).tolist()]
            max_len = max((len(v) for v in sample), default=0)
            width = min(max(len(col_name) + 2, max_len + 2), 60)
            sheet.column_dimensions[
                sheet.cell(row=1, column=col_idx).column_letter
            ].width = width

    return output_path


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------


def run_extraction(
    input_path: Path,
    output_path: Path,
    *,
    k: int,
    sample: Optional[int] = None,
    include_generation: bool = True,
    pipeline_factory=None,
) -> Path:
    """End-to-end extraction. Returns the output ``.xlsx`` path.

    ``pipeline_factory`` is a thunk returning an object with the same
    ``.query(question, *, limit, debug, return_sources, include_generation)``
    contract as ``RAGPipeline`` — used by tests to inject a mock and
    avoid GPU/Ollama calls.
    """
    rows = load_input(input_path)
    if sample is not None and sample > 0:
        rows = rows[:sample]

    if pipeline_factory is None:
        from search.orchestrator.pipeline import RAGPipeline
        pipeline = RAGPipeline()
    else:
        pipeline = pipeline_factory()

    logger.info(
        f"Extractor: {len(rows)} queries from {input_path} "
        f"(k={k}, generate={include_generation})"
    )

    all_rows: List[Dict[str, Any]] = []
    overall_start = time.time()
    for i, row in enumerate(rows, 1):
        t0 = time.time()
        try:
            response = pipeline.query(
                row.query,
                limit=k,
                debug=True,
                return_sources=True,
                include_generation=include_generation,
            )
        except Exception as exc:
            logger.error(
                f"[{row.query_id}] pipeline failed ({exc.__class__.__name__}): {exc}"
            )
            response = {
                "answer": "",
                "sources": [],
                "search_time_ms": 0,
                "llm_time_ms": 0,
                "total_time_ms": int((time.time() - t0) * 1000),
                "trace": {
                    "original_query": row.query,
                    "rewritten_queries": [],
                    "search_documents_per_query": {},
                    "evaluation_scores": [],
                    "generator_documents": [],
                    "errors": [
                        f"Pipeline crashed: {exc.__class__.__name__}: {exc}"
                    ],
                    "timings": {},
                },
            }

        exploded = explode_response(row, response)
        all_rows.extend(exploded)
        logger.info(
            f"[{row.query_id}] ({i}/{len(rows)}) "
            f"rows={len(exploded)} wall={int((time.time() - t0) * 1000)}ms"
        )

    written = write_xlsx(all_rows, output_path)
    logger.info(
        f"Extraction done: {len(all_rows)} rows in {written} "
        f"(elapsed {time.time() - overall_start:.1f}s)"
    )
    return written


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _default_output_path() -> Path:
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"extract_{ts}.xlsx"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract per-stage RAG pipeline data into a single .xlsx",
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Input .csv or .xlsx (must have a 'query' column)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output .xlsx path "
                             "(default: data/pipeline_extracts/extract_<ts>.xlsx)")
    parser.add_argument("--k", type=int, default=5,
                        help="Documents per sub-query passed to the pipeline")
    parser.add_argument("--sample", type=int, default=None,
                        help="Limit to the first N input rows")
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip the LLM generation stage "
                             "(faster; final_answer will be empty)")

    args = parser.parse_args(argv)

    output_path = args.output or _default_output_path()
    try:
        run_extraction(
            input_path=args.input,
            output_path=output_path,
            k=args.k,
            sample=args.sample,
            include_generation=not args.no_generate,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 2
    except Exception as exc:
        logger.error(f"Extraction failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
