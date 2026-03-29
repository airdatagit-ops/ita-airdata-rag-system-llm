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

import numpy as np
from loguru import logger

from models.embeddings import EmbeddingModel
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
    query_results: List[QueryResult] = field(default_factory=list)


_SOURCE_PREFIX_RE = re.compile(r'^(?:sislaer|decea|pdf)_')
_YEAR_SUFFIX_RE = re.compile(r'/\d{4}$')
_EXPECTED_ID_RE = re.compile(
    r'^(?P<type>[A-Za-z]+(?:\s+[A-Za-z]+)*)-(?P<number>.+?)(?:-art(?P<article>\d+)(?:-(?P<sub>\d+))?)?$',
)


def _normalize_id(regulation_id: str) -> str:
    """Strip source prefix (sislaer_, decea_, pdf_) so golden-set IDs match stored IDs."""
    return _SOURCE_PREFIX_RE.sub('', regulation_id)


def _extract_doc_id(regulation_id: str) -> str:
    """Extract document-level ID by stripping the article suffix.

    'decea_ICA-96-1-art563' -> 'ICA-96-1'
    'ICA-7-58-art2-0'       -> 'ICA-7-58'
    'ICA-7-58'              -> 'ICA-7-58'
    """
    return _normalize_id(regulation_id).split("-art")[0]


def _is_doc_level_id(doc_id: str) -> bool:
    """True when the expected ID has no article suffix."""
    return "-art" not in doc_id


def _canonical_base(doc_type: str, number: str) -> str:
    """Compute canonical base ID (same logic as compute_canonical_id)."""
    dtype = re.sub(r'[^a-z0-9]', '', doc_type.lower())
    num = re.sub(r'[^0-9a-z./-]', '', number.lower()).strip('-./') 
    return f'{dtype}_{num}' if dtype and num else ''


def _expected_canonical(expected_id: str) -> Tuple[str, Optional[str]]:
    """Convert golden-set ID to (canonical_base, article_number).

    'ICA-96-1-art563' -> ('ica_96-1', '563')
    'ICA-7-58'        -> ('ica_7-58', None)
    """
    m = _EXPECTED_ID_RE.match(expected_id)
    if not m:
        return ('', None)
    base = _canonical_base(m.group('type'), m.group('number'))
    return (base, m.group('article'))


def _matches_expected(
    retrieved_id: str,
    expected_id: str,
    canonical_map: Optional[Dict[str, str]] = None,
) -> bool:
    """Match a retrieved chunk against an expected ID.

    Strategy 1: regulation_id comparison (source prefix stripped).
    Strategy 2: canonical_id comparison (year stripped, article matched).
    """
    r = _normalize_id(retrieved_id)
    e = _normalize_id(expected_id)
    if _is_doc_level_id(e):
        if _extract_doc_id(r) == e:
            return True
    elif r == e:
        return True

    canonical_id = (canonical_map or {}).get(retrieved_id)
    if canonical_id:
        r_base = _YEAR_SUFFIX_RE.sub('', canonical_id)
        r_art_m = re.search(r'-art(\d+)', retrieved_id)
        r_art = r_art_m.group(1) if r_art_m else None

        e_base, e_art = _expected_canonical(expected_id)
        if e_base and r_base == e_base:
            if e_art is None:
                return True
            return r_art == e_art

    return False


def _any_match(
    retrieved_id: str,
    expected_ids,
    canonical_map: Optional[Dict[str, str]] = None,
) -> bool:
    """True if retrieved_id matches ANY of the expected IDs."""
    return any(
        _matches_expected(retrieved_id, eid, canonical_map)
        for eid in expected_ids
    )


def _compute_ndcg(
    retrieved_ids: List[str],
    relevant: List[str],
    moderate: List[str],
    k: int,
    canonical_map: Optional[Dict[str, str]] = None,
) -> float:
    """Compute NDCG@K with graded relevance (relevant=2, moderate=1).

    Each expected doc is counted at most once (first matching chunk wins).
    """
    matched_relevant = set()
    matched_moderate = set()
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        matched_exp = _first_unmatched(doc_id, relevant, matched_relevant, canonical_map)
        if matched_exp:
            matched_relevant.add(matched_exp)
            dcg += 2.0 / math.log2(i + 2)
            continue
        matched_exp = _first_unmatched(doc_id, moderate, matched_moderate, canonical_map)
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
    canonical_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Return the first expected ID that matches and hasn't been matched yet."""
    for eid in expected_ids:
        if eid not in already_matched and _matches_expected(retrieved_id, eid, canonical_map):
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
        all_expected_set = relevant_set | moderate_set

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

        canonical_map: Dict[str, str] = {}
        for r in raw_results:
            reg_id = r.payload.get('regulation_id', '')
            meta = r.payload.get('metadata', {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            can_id = meta.get('canonical_id', '')
            if can_id:
                canonical_map[reg_id] = can_id

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
        seen_relevant = set()
        seen_moderate = set()
        for d in retrieved_ids:
            exp = _first_unmatched(d, list(relevant_set), seen_relevant, canonical_map)
            if exp:
                relevant_found.append(d)
                seen_relevant.add(exp)
                continue
            exp = _first_unmatched(d, list(moderate_set), seen_moderate, canonical_map)
            if exp:
                moderate_found.append(d)
                seen_moderate.add(exp)

        first_relevant_rank = next(
            (i for i, d in enumerate(retrieved_ids, 1)
             if _any_match(d, relevant_set, canonical_map)),
            None
        )

        all_expected = relevant_set | moderate_set
        matched = len(seen_relevant) + len(seen_moderate)
        precision_at_k = matched / k if k > 0 else 0.0
        recall = matched / len(all_expected) if all_expected else 0.0
        ndcg = _compute_ndcg(
            retrieved_ids, relevant_expected, moderate_expected, k, canonical_map,
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

    def evaluate(self, k: int = 5, workers: int = 1, search_mode: str = "auto") -> EvaluationResult:
        """
        Run full evaluation. Embeddings are batched, search parallelized.

        Args:
            k: Number of results to retrieve per query
            workers: Number of parallel workers for Qdrant search
            search_mode: "dense", "sparse", "hybrid", or "auto" (from config)
        """
        start = time.time()
        logger.info(f"Starting evaluation: k={k}, workers={workers}")

        query_ids = list(self.golden_set.keys())
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
            embed_model = EmbeddingModel()
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
            query_results=query_results,
        )

    def save_results(self, result: EvaluationResult, output_dir: str = "evaluation/results"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        ts = result.timestamp.replace(":", "").replace("-", "")[:15]

        summary_path = output_path / f"summary_{ts}.csv"
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            metrics = [
                ('timestamp', result.timestamp),
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
            writer.writerows(metrics)

        logger.info(f"Summary saved: {summary_path}")

        details_path = output_path / f"details_{ts}.csv"
        with open(details_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'query_id', 'query', 'category', 'expected_docs', 'retrieved_docs',
                'relevant_found', 'moderate_found', 'first_relevant_rank',
                'precision_at_k', 'recall', 'ndcg_at_k', 'hit'
            ])
            for qr in result.query_results:
                writer.writerow([
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
                    qr.hit
                ])

        logger.info(f"Details saved: {details_path}")

        json_path = output_path / f"results_{ts}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': result.timestamp,
                'k': result.k,
                'elapsed_seconds': result.elapsed_seconds,
                'metrics': {
                    'total_queries': result.total_queries,
                    'retrieval_queries': result.retrieval_queries,
                    'retrieval_hit_rate': result.retrieval_hit_rate,
                    'retrieval_mrr': result.retrieval_mrr,
                    'retrieval_ndcg': result.retrieval_ndcg,
                    'retrieval_precision': result.retrieval_precision,
                    'retrieval_recall': result.retrieval_recall,
                    'coverage_queries': result.coverage_queries,
                    'coverage_correct_rate': result.coverage_correct_rate,
                },
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
    print("  RAG RETRIEVAL QUALITY EVALUATION REPORT")
    print("=" * 70)
    print(f"  Timestamp:    {result.timestamp}")
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

    args = parser.parse_args()

    try:
        evaluator = RetrievalEvaluator(args.golden_set)
        result = evaluator.evaluate(k=args.k, workers=args.workers, search_mode=args.search_mode)

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
