"""
RAG Retrieval Quality Evaluation Script.

Evaluates retrieval quality using a golden set of queries and expected documents.

Metrics computed:
- Precision@K: Proportion of retrieved docs that are relevant
- Recall@K: Proportion of relevant docs that were retrieved
- MRR (Mean Reciprocal Rank): Average of 1/rank of first relevant doc
- Hit Rate@K: Proportion of queries with at least one relevant doc in top K
- NDCG@K: Normalized Discounted Cumulative Gain

Usage:
    python -m evaluation.evaluate_retrieval
    python -m evaluation.evaluate_retrieval --k 5 --workers 4
"""

import argparse
import csv
import json
import math
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from models.gpu_client import create_embedding_model
from database.qdrant_manager import QdrantManager
from config import config

try:
    from models.embeddings import SparseEncoder
except ImportError:
    SparseEncoder = None


@dataclass
class GoldenSetItem:
    """A single item from the golden set."""
    query_id: str
    query: str
    expected_doc_id: str
    relevance: str
    category: str
    notes: str = ""


@dataclass
class QueryResult:
    """Result of evaluating a single query."""
    query_id: str
    query: str
    category: str
    expected_docs: List[Tuple[str, str]]
    retrieved_docs: List[Tuple[str, float]]
    relevant_found: List[str]
    moderate_found: List[str]
    first_relevant_rank: Optional[int]
    precision_at_k: float
    recall: float
    ndcg_at_k: float
    hit: bool
    rewriter_subqueries: int = 0
    docs_before_evaluator: int = 0
    docs_after_evaluator: int = 0
    rewriter_ms: int = 0
    searcher_ms: int = 0
    evaluator_ms: int = 0
    total_ms: int = 0


@dataclass
class EvaluationResult:
    """Overall evaluation results."""
    timestamp: str
    k: int
    total_queries: int
    elapsed_seconds: float
    retrieval_queries: int
    retrieval_hit_rate: float
    retrieval_mrr: float
    retrieval_ndcg: float
    retrieval_precision: float
    retrieval_recall: float
    coverage_queries: int
    coverage_correct_rate: float
    mode: str = "raw"
    avg_rewriter_subqueries: float = 0.0
    avg_evaluator_filter_rate: float = 0.0
    avg_rewriter_ms: float = 0.0
    avg_searcher_ms: float = 0.0
    avg_evaluator_ms: float = 0.0
    avg_total_ms: float = 0.0
    query_results: List[QueryResult] = field(default_factory=list)


_GOLDEN_ID_RE = re.compile(
    r'^(?P<type>[A-Za-z]+(?:\s+[A-Za-z]+)*)-(?P<rest>.+)$',
)
_VERSION_YEAR_RE = re.compile(r'/\d{4}(?=$|-art)')


def _normalize_id(raw_id: str) -> str:
    """Normalize a golden-set or regulation_id to canonical form.

    Converts ``ICA-96-1-art563`` -> ``ica_96-1-art563`` (type_number format).
    Handles both golden-set dash-separated IDs and already-canonical IDs.
    """
    m = _GOLDEN_ID_RE.match(raw_id)
    if m:
        dtype = re.sub(r'[^a-z0-9]', '', m.group('type').lower())
        rest = m.group('rest')
        return f"{dtype}_{rest}"
    return raw_id


def _strip_version(norm_id: str) -> str:
    """Strip /YYYY version suffix for version-agnostic matching."""
    return _VERSION_YEAR_RE.sub('', norm_id)


def _extract_doc_id(raw_id: str) -> str:
    """Normalize and strip the article suffix.

    ``ica_96-1/2025-art563`` -> ``ica_96-1/2025``
    ``ICA-96-1-art10``       -> ``ica_96-1``
    """
    return _normalize_id(raw_id).split("-art")[0]


def _is_doc_level_id(norm_id: str) -> bool:
    return "-art" not in norm_id


def _matches_expected(retrieved_id: str, expected_id: str) -> bool:
    """Match a retrieved chunk against an expected golden-set ID.

    Matching is version-agnostic: ``ica_96-1/2025-art563`` matches ``ICA-96-1-art563``.
    Doc-level expected IDs (no ``-art``) match any article of that document.
    """
    r = _strip_version(_normalize_id(retrieved_id))
    e = _strip_version(_normalize_id(expected_id))

    if _is_doc_level_id(e):
        return _extract_doc_id(r) == e
    return r == e


def _any_match(retrieved_id: str, expected_ids) -> bool:
    return any(_matches_expected(retrieved_id, eid) for eid in expected_ids)


def _compute_ndcg(
    retrieved_ids: List[str],
    relevant: List[str],
    moderate: List[str],
    k: int,
) -> float:
    """Compute NDCG@K with graded relevance (relevant=2, moderate=1)."""
    matched_relevant: set = set()
    matched_moderate: set = set()
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        matched_exp = _first_unmatched(doc_id, relevant, matched_relevant)
        if matched_exp:
            matched_relevant.add(matched_exp)
            dcg += 2.0 / math.log2(i + 2)
            continue
        matched_exp = _first_unmatched(doc_id, moderate, matched_moderate)
        if matched_exp:
            matched_moderate.add(matched_exp)
            dcg += 1.0 / math.log2(i + 2)

    ideal_rels = sorted(
        [2.0] * len(relevant) + [1.0] * len(moderate),
        reverse=True
    )[:k]

    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def _first_unmatched(
    retrieved_id: str,
    expected_ids: List[str],
    already_matched: set,
) -> Optional[str]:
    """Return the first expected ID that matches and hasn't been matched yet."""
    for eid in expected_ids:
        if eid not in already_matched and _matches_expected(retrieved_id, eid):
            return eid
    return None


class RetrievalEvaluator:
    """Evaluates RAG retrieval quality against a golden set."""

    def __init__(self, golden_set_path: str = "evaluation/golden_set.csv"):
        self.golden_set_path = Path(golden_set_path)
        self.golden_set: Dict[str, List[GoldenSetItem]] = defaultdict(list)
        self._load_golden_set()

    def _load_golden_set(self):
        """Load golden set from CSV."""
        if not self.golden_set_path.exists():
            raise FileNotFoundError(f"Golden set not found: {self.golden_set_path}")

        with open(self.golden_set_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = GoldenSetItem(
                    query_id=row['query_id'],
                    query=row['query'],
                    expected_doc_id=row['expected_doc_id'],
                    relevance=row['relevance'],
                    category=row['category'],
                    notes=row.get('notes', '')
                )
                self.golden_set[item.query_id].append(item)

        logger.info(f"Loaded {len(self.golden_set)} unique queries from golden set")

    def _select_query_ids(self, sample: Optional[int] = None) -> List[str]:
        """Return ordered query ids honouring an optional ``sample`` cap.

        The golden set has two query categories: ``coverage`` (negative
        signals — the system should NOT find a relevant doc) and the
        rest (retrieval queries). Sampling keeps ALL coverage queries
        and caps only the retrieval list, so latency-critical A/B runs
        still report a meaningful ``coverage_correct_rate``. Selection
        is deterministic (insertion order = CSV order) for
        reproducibility.
        """
        all_ids = list(self.golden_set.keys())
        if sample is None or sample <= 0:
            return all_ids
        retrieval_ids: List[str] = []
        coverage_ids: List[str] = []
        for qid in all_ids:
            if self.golden_set[qid][0].category == 'coverage':
                coverage_ids.append(qid)
            else:
                retrieval_ids.append(qid)
        return retrieval_ids[:sample] + coverage_ids

    def _get_expected_docs(self, query_id: str) -> Tuple[List[str], List[str]]:
        relevant = []
        moderate = []
        for item in self.golden_set[query_id]:
            if item.relevance == 'relevant':
                relevant.append(item.expected_doc_id)
            elif item.relevance == 'moderate':
                moderate.append(item.expected_doc_id)
        return relevant, moderate

    def _evaluate_single_query(
        self,
        query_id: str,
        k: int,
        query_embedding,
        qdrant_manager: QdrantManager
    ) -> QueryResult:
        """Evaluate a single query using a pre-computed embedding."""
        items = self.golden_set[query_id]
        query = items[0].query
        category = items[0].category

        relevant_expected, moderate_expected = self._get_expected_docs(query_id)
        relevant_set = set(relevant_expected)
        moderate_set = set(moderate_expected)

        search_kwargs = {"limit": k}
        if self._hybrid_mode or self._sparse_mode:
            if query_embedding.get("dense") is not None:
                search_kwargs["dense_vector"] = query_embedding["dense"].tolist()
            if query_embedding.get("sparse") is not None:
                search_kwargs["sparse_vector"] = query_embedding["sparse"]
        else:
            dense = query_embedding.get("dense")
            search_kwargs["query_vector"] = dense.tolist() if hasattr(dense, 'tolist') else dense
            search_kwargs["score_threshold"] = config.SEARCH_SCORE_THRESHOLD
        raw_results = qdrant_manager.search(**search_kwargs)
        retrieved_ids = [r.payload.get('regulation_id', '') for r in raw_results]
        retrieved_scores = [(r.payload.get('regulation_id', ''), r.score) for r in raw_results]

        if 'NOT_IN_DB' in relevant_set:
            return QueryResult(
                query_id=query_id, query=query, category=category,
                expected_docs=[(d, 'irrelevant') for d in relevant_expected],
                retrieved_docs=retrieved_scores,
                relevant_found=[], moderate_found=[],
                first_relevant_rank=None,
                precision_at_k=0.0, recall=0.0, ndcg_at_k=0.0, hit=False,
            )

        relevant_found = []
        moderate_found = []
        seen_relevant: set = set()
        seen_moderate: set = set()
        for d in retrieved_ids:
            exp = _first_unmatched(d, list(relevant_set), seen_relevant)
            if exp:
                relevant_found.append(d)
                seen_relevant.add(exp)
                continue
            exp = _first_unmatched(d, list(moderate_set), seen_moderate)
            if exp:
                moderate_found.append(d)
                seen_moderate.add(exp)

        first_relevant_rank = next(
            (i for i, d in enumerate(retrieved_ids, 1)
             if _any_match(d, relevant_set)),
            None
        )

        all_expected = relevant_set | moderate_set
        matched = len(seen_relevant) + len(seen_moderate)
        precision_at_k = matched / k if k > 0 else 0.0
        recall = matched / len(all_expected) if all_expected else 0.0
        ndcg = _compute_ndcg(
            retrieved_ids, relevant_expected, moderate_expected, k,
        )
        hit = bool(relevant_found or moderate_found)

        return QueryResult(
            query_id=query_id, query=query, category=category,
            expected_docs=[(d, 'relevant') for d in relevant_expected] +
                         [(d, 'moderate') for d in moderate_expected],
            retrieved_docs=retrieved_scores,
            relevant_found=relevant_found, moderate_found=moderate_found,
            first_relevant_rank=first_relevant_rank,
            precision_at_k=precision_at_k, recall=recall,
            ndcg_at_k=ndcg, hit=hit,
        )

    def _evaluate_single_query_pipeline(
        self,
        query_id: str,
        k: int,
        pipeline,
    ) -> QueryResult:
        """Evaluate a single query running the full RAG pipeline (no generation)."""
        items = self.golden_set[query_id]
        query = items[0].query
        category = items[0].category

        relevant_expected, moderate_expected = self._get_expected_docs(query_id)
        relevant_set = set(relevant_expected)
        moderate_set = set(moderate_expected)

        response = pipeline.query(
            query,
            limit=k,
            include_generation=False,
            return_sources=True,
            debug=True,
        )
        sources = response.get("sources", [])
        retrieved_ids = [s.get("regulation_id", "") for s in sources]
        retrieved_scores = [
            (s.get("regulation_id", ""), float(s.get("score") or 0.0))
            for s in sources
        ]

        trace = response.get("trace", {})
        timings = trace.get("timings", {}) or {}
        rewritten = trace.get("rewritten_queries", []) or []
        docs_before_eval = int(trace.get("documents_after_dedup") or 0)
        docs_after_eval = int(trace.get("documents_accepted") or len(sources))

        if 'NOT_IN_DB' in relevant_set:
            return QueryResult(
                query_id=query_id, query=query, category=category,
                expected_docs=[(d, 'irrelevant') for d in relevant_expected],
                retrieved_docs=retrieved_scores,
                relevant_found=[], moderate_found=[],
                first_relevant_rank=None,
                precision_at_k=0.0, recall=0.0, ndcg_at_k=0.0, hit=False,
                rewriter_subqueries=len(rewritten),
                docs_before_evaluator=docs_before_eval,
                docs_after_evaluator=docs_after_eval,
                rewriter_ms=int(timings.get("rewriter_ms") or 0),
                searcher_ms=int(timings.get("searcher_ms") or 0),
                evaluator_ms=int(timings.get("evaluator_ms") or 0),
                total_ms=int(timings.get("total_ms") or 0),
            )

        relevant_found: List[str] = []
        moderate_found: List[str] = []
        seen_relevant: set = set()
        seen_moderate: set = set()
        for d in retrieved_ids:
            exp = _first_unmatched(d, list(relevant_set), seen_relevant)
            if exp:
                relevant_found.append(d)
                seen_relevant.add(exp)
                continue
            exp = _first_unmatched(d, list(moderate_set), seen_moderate)
            if exp:
                moderate_found.append(d)
                seen_moderate.add(exp)

        first_relevant_rank = next(
            (i for i, d in enumerate(retrieved_ids, 1)
             if _any_match(d, relevant_set)),
            None,
        )

        all_expected = relevant_set | moderate_set
        matched = len(seen_relevant) + len(seen_moderate)
        precision_at_k = matched / k if k > 0 else 0.0
        recall = matched / len(all_expected) if all_expected else 0.0
        ndcg = _compute_ndcg(
            retrieved_ids, relevant_expected, moderate_expected, k,
        )
        hit = bool(relevant_found or moderate_found)

        return QueryResult(
            query_id=query_id, query=query, category=category,
            expected_docs=[(d, 'relevant') for d in relevant_expected]
                         + [(d, 'moderate') for d in moderate_expected],
            retrieved_docs=retrieved_scores,
            relevant_found=relevant_found,
            moderate_found=moderate_found,
            first_relevant_rank=first_relevant_rank,
            precision_at_k=precision_at_k, recall=recall,
            ndcg_at_k=ndcg, hit=hit,
            rewriter_subqueries=len(rewritten),
            docs_before_evaluator=docs_before_eval,
            docs_after_evaluator=docs_after_eval,
            rewriter_ms=int(timings.get("rewriter_ms") or 0),
            searcher_ms=int(timings.get("searcher_ms") or 0),
            evaluator_ms=int(timings.get("evaluator_ms") or 0),
            total_ms=int(timings.get("total_ms") or 0),
        )

    def evaluate(
        self,
        k: int = 5,
        workers: int = 1,
        search_mode: str = "auto",
        sample: Optional[int] = None,
    ) -> EvaluationResult:
        """
        Run full evaluation. Embeddings are batched, search parallelized.

        Args:
            k: Number of results to retrieve per query
            workers: Number of parallel workers for Qdrant search
            search_mode: "dense", "sparse", "hybrid", or "auto" (from config)
            sample: If set, cap retrieval queries to the first N (coverage
                queries are always kept). Useful for A/B runs.
        """
        start = time.time()
        logger.info(f"Starting evaluation: k={k}, workers={workers}, sample={sample}")

        query_ids = self._select_query_ids(sample)
        queries = [self.golden_set[qid][0].query for qid in query_ids]

        if search_mode == "auto":
            use_dense = getattr(config, 'SEARCH_DENSE_ENABLED', True)
            use_sparse = getattr(config, 'SEARCH_SPARSE_ENABLED', False)
        else:
            use_dense = search_mode in ("dense", "hybrid")
            use_sparse = search_mode in ("sparse", "hybrid")

        if use_sparse and SparseEncoder is None:
            logger.warning("SparseEncoder not available — falling back to dense-only")
            use_sparse = False
            use_dense = True

        self._sparse_mode = use_sparse and not use_dense
        self._hybrid_mode = use_dense and use_sparse

        mode_label = "hybrid" if self._hybrid_mode else ("sparse" if self._sparse_mode else "dense")
        logger.info(f"Encoding {len(queries)} queries in batch (mode={mode_label})...")

        dense_embeddings = None
        sparse_embeddings = None
        if use_dense:
            embed_model = create_embedding_model()
            dense_embeddings = embed_model.encode(queries)
        if use_sparse:
            sparse_model = SparseEncoder()
            sparse_embeddings = sparse_model.encode(queries)

        embeddings = []
        for i in range(len(queries)):
            embeddings.append({
                "dense": dense_embeddings[i] if dense_embeddings is not None else None,
                "sparse": sparse_embeddings[i] if sparse_embeddings is not None else None,
            })
        logger.info(f"Encoded {len(queries)} queries")

        qdrant = QdrantManager()

        if workers <= 1:
            query_results = [
                self._evaluate_single_query(qid, k, embeddings[i], qdrant)
                for i, qid in enumerate(query_ids)
            ]
        else:
            query_results = [None] * len(query_ids)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self._evaluate_single_query, qid, k, embeddings[i], qdrant
                    ): i
                    for i, qid in enumerate(query_ids)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    query_results[idx] = future.result()

        elapsed = time.time() - start

        retrieval_results = [r for r in query_results if r.category != 'coverage']
        coverage_results = [r for r in query_results if r.category == 'coverage']

        n = len(retrieval_results)
        if n > 0:
            hit_rate = sum(1 for r in retrieval_results if r.hit) / n
            precision = sum(r.precision_at_k for r in retrieval_results) / n
            recall = sum(r.recall for r in retrieval_results) / n
            ndcg = sum(r.ndcg_at_k for r in retrieval_results) / n
            mrr = sum(
                (1.0 / r.first_relevant_rank) if r.first_relevant_rank else 0.0
                for r in retrieval_results
            ) / n
        else:
            hit_rate = precision = recall = ndcg = mrr = 0.0

        coverage_correct = 0.0
        if coverage_results:
            coverage_correct = sum(1 for r in coverage_results if not r.hit) / len(coverage_results)

        return EvaluationResult(
            timestamp=datetime.now().isoformat(),
            k=k,
            total_queries=len(query_results),
            elapsed_seconds=elapsed,
            retrieval_queries=n,
            retrieval_hit_rate=hit_rate,
            retrieval_mrr=mrr,
            retrieval_ndcg=ndcg,
            retrieval_precision=precision,
            retrieval_recall=recall,
            coverage_queries=len(coverage_results),
            coverage_correct_rate=coverage_correct,
            mode="raw",
            query_results=query_results,
        )

    def evaluate_pipeline(
        self,
        k: int = 5,
        workers: int = 1,
        sample: Optional[int] = None,
    ) -> EvaluationResult:
        """Run end-to-end evaluation through ``RAGPipeline`` (no generation).

        Measures the documents the Generator would actually receive after
        rewriter + searcher + evaluator. This is the metric that reflects
        what the user effectively sees as ``sources`` in the API response.

        ``sample`` caps retrieval queries (coverage queries always run).
        """
        from search.orchestrator.pipeline import RAGPipeline

        start = time.time()
        logger.info(
            f"Starting pipeline evaluation: k={k}, workers={workers}, sample={sample}"
        )

        pipeline = RAGPipeline()

        query_ids = self._select_query_ids(sample)

        if workers <= 1:
            query_results = [
                self._evaluate_single_query_pipeline(qid, k, pipeline)
                for qid in query_ids
            ]
        else:
            query_results = [None] * len(query_ids)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self._evaluate_single_query_pipeline, qid, k, pipeline
                    ): i
                    for i, qid in enumerate(query_ids)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    query_results[idx] = future.result()

        elapsed = time.time() - start

        retrieval_results = [r for r in query_results if r.category != 'coverage']
        coverage_results = [r for r in query_results if r.category == 'coverage']

        n = len(retrieval_results)
        if n > 0:
            hit_rate = sum(1 for r in retrieval_results if r.hit) / n
            precision = sum(r.precision_at_k for r in retrieval_results) / n
            recall = sum(r.recall for r in retrieval_results) / n
            ndcg = sum(r.ndcg_at_k for r in retrieval_results) / n
            mrr = sum(
                (1.0 / r.first_relevant_rank) if r.first_relevant_rank else 0.0
                for r in retrieval_results
            ) / n
            avg_subqueries = sum(r.rewriter_subqueries for r in retrieval_results) / n
            filter_rates = [
                1.0 - (r.docs_after_evaluator / r.docs_before_evaluator)
                for r in retrieval_results if r.docs_before_evaluator > 0
            ]
            avg_filter_rate = sum(filter_rates) / len(filter_rates) if filter_rates else 0.0
            avg_rew = sum(r.rewriter_ms for r in retrieval_results) / n
            avg_sea = sum(r.searcher_ms for r in retrieval_results) / n
            avg_eva = sum(r.evaluator_ms for r in retrieval_results) / n
            avg_tot = sum(r.total_ms for r in retrieval_results) / n
        else:
            hit_rate = precision = recall = ndcg = mrr = 0.0
            avg_subqueries = avg_filter_rate = 0.0
            avg_rew = avg_sea = avg_eva = avg_tot = 0.0

        coverage_correct = 0.0
        if coverage_results:
            coverage_correct = sum(1 for r in coverage_results if not r.hit) / len(coverage_results)

        return EvaluationResult(
            timestamp=datetime.now().isoformat(),
            k=k,
            total_queries=len(query_results),
            elapsed_seconds=elapsed,
            retrieval_queries=n,
            retrieval_hit_rate=hit_rate,
            retrieval_mrr=mrr,
            retrieval_ndcg=ndcg,
            retrieval_precision=precision,
            retrieval_recall=recall,
            coverage_queries=len(coverage_results),
            coverage_correct_rate=coverage_correct,
            mode="pipeline",
            avg_rewriter_subqueries=avg_subqueries,
            avg_evaluator_filter_rate=avg_filter_rate,
            avg_rewriter_ms=avg_rew,
            avg_searcher_ms=avg_sea,
            avg_evaluator_ms=avg_eva,
            avg_total_ms=avg_tot,
            query_results=query_results,
        )

    def save_results(self, result: EvaluationResult, output_dir: str = "evaluation/results"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        ts = result.timestamp.replace(":", "").replace("-", "")[:15]

        summary_path = output_path / f"summary_{result.mode}_{ts}.csv"
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            metrics = [
                ('timestamp', result.timestamp),
                ('mode', result.mode),
                ('k', result.k),
                ('elapsed_seconds', f"{result.elapsed_seconds:.2f}"),
                ('total_queries', result.total_queries),
                ('retrieval_queries', result.retrieval_queries),
                ('retrieval_hit_rate', f"{result.retrieval_hit_rate:.4f}"),
                ('retrieval_mrr', f"{result.retrieval_mrr:.4f}"),
                ('retrieval_ndcg', f"{result.retrieval_ndcg:.4f}"),
                ('retrieval_precision', f"{result.retrieval_precision:.4f}"),
                ('retrieval_recall', f"{result.retrieval_recall:.4f}"),
                ('coverage_queries', result.coverage_queries),
                ('coverage_correct_rate', f"{result.coverage_correct_rate:.4f}"),
            ]
            if result.mode == "pipeline":
                metrics += [
                    ('avg_rewriter_subqueries', f"{result.avg_rewriter_subqueries:.2f}"),
                    ('avg_evaluator_filter_rate', f"{result.avg_evaluator_filter_rate:.4f}"),
                    ('avg_rewriter_ms', f"{result.avg_rewriter_ms:.0f}"),
                    ('avg_searcher_ms', f"{result.avg_searcher_ms:.0f}"),
                    ('avg_evaluator_ms', f"{result.avg_evaluator_ms:.0f}"),
                    ('avg_total_ms', f"{result.avg_total_ms:.0f}"),
                ]
            writer.writerows(metrics)

        logger.info(f"Summary saved: {summary_path}")

        details_path = output_path / f"details_{result.mode}_{ts}.csv"
        with open(details_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = [
                'query_id', 'query', 'category', 'expected_docs', 'retrieved_docs',
                'relevant_found', 'moderate_found', 'first_relevant_rank',
                'precision_at_k', 'recall', 'ndcg_at_k', 'hit',
            ]
            if result.mode == "pipeline":
                header += [
                    'rewriter_subqueries', 'docs_before_evaluator',
                    'docs_after_evaluator', 'rewriter_ms', 'searcher_ms',
                    'evaluator_ms', 'total_ms',
                ]
            writer.writerow(header)
            for qr in result.query_results:
                row = [
                    qr.query_id,
                    qr.query,
                    qr.category,
                    ';'.join([f"{d}:{r}" for d, r in qr.expected_docs]),
                    ';'.join([f"{d}:{s:.3f}" for d, s in qr.retrieved_docs[:5]]),
                    ';'.join(qr.relevant_found),
                    ';'.join(qr.moderate_found),
                    qr.first_relevant_rank or 'N/A',
                    f"{qr.precision_at_k:.4f}",
                    f"{qr.recall:.4f}",
                    f"{qr.ndcg_at_k:.4f}",
                    qr.hit,
                ]
                if result.mode == "pipeline":
                    row += [
                        qr.rewriter_subqueries,
                        qr.docs_before_evaluator,
                        qr.docs_after_evaluator,
                        qr.rewriter_ms,
                        qr.searcher_ms,
                        qr.evaluator_ms,
                        qr.total_ms,
                    ]
                writer.writerow(row)

        logger.info(f"Details saved: {details_path}")

        json_path = output_path / f"results_{result.mode}_{ts}.json"
        metrics_blob = {
            'total_queries': result.total_queries,
            'retrieval_queries': result.retrieval_queries,
            'retrieval_hit_rate': result.retrieval_hit_rate,
            'retrieval_mrr': result.retrieval_mrr,
            'retrieval_ndcg': result.retrieval_ndcg,
            'retrieval_precision': result.retrieval_precision,
            'retrieval_recall': result.retrieval_recall,
            'coverage_queries': result.coverage_queries,
            'coverage_correct_rate': result.coverage_correct_rate,
        }
        if result.mode == "pipeline":
            metrics_blob.update({
                'avg_rewriter_subqueries': result.avg_rewriter_subqueries,
                'avg_evaluator_filter_rate': result.avg_evaluator_filter_rate,
                'avg_rewriter_ms': result.avg_rewriter_ms,
                'avg_searcher_ms': result.avg_searcher_ms,
                'avg_evaluator_ms': result.avg_evaluator_ms,
                'avg_total_ms': result.avg_total_ms,
            })
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': result.timestamp,
                'mode': result.mode,
                'k': result.k,
                'elapsed_seconds': result.elapsed_seconds,
                'metrics': metrics_blob,
                'query_results': [
                    {
                        'query_id': qr.query_id,
                        'query': qr.query,
                        'category': qr.category,
                        'hit': qr.hit,
                        'precision': qr.precision_at_k,
                        'recall': qr.recall,
                        'ndcg': qr.ndcg_at_k,
                        'first_rank': qr.first_relevant_rank,
                        'expected': [d for d, _ in qr.expected_docs],
                        'retrieved_top3': [d for d, _ in qr.retrieved_docs[:3]],
                    }
                    for qr in result.query_results
                ]
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON saved: {json_path}")
        return summary_path, details_path, json_path


def print_report(result: EvaluationResult):
    """Print a formatted evaluation report."""
    k = result.k
    print("\n" + "=" * 70)
    print(f"  RAG RETRIEVAL QUALITY EVALUATION REPORT  ({result.mode} mode)")
    print("=" * 70)
    print(f"  Timestamp:    {result.timestamp}")
    print(f"  Mode:         {result.mode}")
    print(f"  K:            {k}")
    print(f"  Elapsed:      {result.elapsed_seconds:.2f}s")
    print(f"  Queries:      {result.total_queries}")
    print()

    print("  RETRIEVAL METRICS")
    print("  " + "-" * 40)
    print(f"    Queries:        {result.retrieval_queries}")
    print(f"    Hit Rate@{k}:     {result.retrieval_hit_rate:.1%}")
    print(f"    MRR:            {result.retrieval_mrr:.4f}")
    print(f"    NDCG@{k}:         {result.retrieval_ndcg:.4f}")
    print(f"    Precision@{k}:    {result.retrieval_precision:.4f}")
    print(f"    Recall:         {result.retrieval_recall:.4f}")
    print()

    if result.mode == "pipeline":
        print("  PIPELINE METRICS (end-to-end)")
        print("  " + "-" * 40)
        print(f"    Avg sub-queries:   {result.avg_rewriter_subqueries:.2f}")
        print(f"    Evaluator filter:  {result.avg_evaluator_filter_rate:.1%} of docs dropped")
        print(f"    Avg Rewriter:      {result.avg_rewriter_ms:.0f}ms")
        print(f"    Avg Searcher:      {result.avg_searcher_ms:.0f}ms")
        print(f"    Avg Evaluator:     {result.avg_evaluator_ms:.0f}ms")
        print(f"    Avg total/query:   {result.avg_total_ms:.0f}ms")
        print()

    print("  COVERAGE METRICS")
    print("  " + "-" * 40)
    print(f"    Queries:             {result.coverage_queries}")
    print(f"    Correct 'Not Found': {result.coverage_correct_rate:.1%}")
    print()

    hits = [qr for qr in result.query_results if qr.hit and qr.category != 'coverage']
    misses = [qr for qr in result.query_results if not qr.hit and qr.category != 'coverage']

    print(f"  ✅ HITS ({len(hits)}/{result.retrieval_queries})")
    print("  " + "-" * 40)
    for qr in sorted(hits, key=lambda x: x.first_relevant_rank or 99):
        rank_str = f"rank {qr.first_relevant_rank}" if qr.first_relevant_rank else "moderate only"
        ndcg_str = f"NDCG={qr.ndcg_at_k:.2f}"
        print(f"    [{qr.query_id}] {qr.query[:55]:<55} {rank_str:<15} {ndcg_str}")

    print(f"\n  ❌ MISSES ({len(misses)}/{result.retrieval_queries})")
    print("  " + "-" * 40)
    for qr in misses:
        expected = [d for d, _ in qr.expected_docs]
        retrieved = [d for d, _ in qr.retrieved_docs[:3]]
        print(f"    [{qr.query_id}] {qr.query[:55]}")
        print(f"         Expected:  {expected}")
        print(f"         Got:       {retrieved}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument('--k', type=int, default=5, help="Number of results to retrieve")
    parser.add_argument('--golden-set', type=str, default="evaluation/golden_set.csv",
                       help="Path to golden set CSV")
    parser.add_argument('--output', type=str, default="evaluation/results",
                       help="Output directory for results")
    parser.add_argument('--workers', type=int, default=4,
                       help="Number of parallel workers for search")
    parser.add_argument('--quiet', action='store_true', help="Suppress detailed output")
    parser.add_argument('--search-mode', type=str, default="auto",
                       choices=["auto", "dense", "sparse", "hybrid"],
                       help="Search mode: auto (from config), dense, sparse, or hybrid")
    parser.add_argument('--use-pipeline', action='store_true',
                       help="Run end-to-end through RAGPipeline (rewriter+searcher+"
                            "evaluator, no LLM generation). Reflects what the user"
                            " actually receives as sources. Slower; use workers=1"
                            " for sequential timing accuracy.")
    parser.add_argument('--sample', type=int, default=None,
                       help="Cap retrieval queries to the first N (deterministic, "
                            "CSV order). Coverage queries are always kept. Useful "
                            "for fast A/B runs.")

    args = parser.parse_args()

    try:
        evaluator = RetrievalEvaluator(args.golden_set)
        if args.use_pipeline:
            result = evaluator.evaluate_pipeline(
                k=args.k, workers=args.workers, sample=args.sample,
            )
        else:
            result = evaluator.evaluate(
                k=args.k, workers=args.workers, search_mode=args.search_mode,
                sample=args.sample,
            )

        if not args.quiet:
            print_report(result)

        evaluator.save_results(result, args.output)

        if result.retrieval_hit_rate < 0.5:
            logger.warning(f"Low hit rate: {result.retrieval_hit_rate:.1%}")

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
