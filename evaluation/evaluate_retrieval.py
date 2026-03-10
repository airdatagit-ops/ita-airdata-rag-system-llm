"""
RAG Retrieval Quality Evaluation Script.

This script evaluates the retrieval quality of the RAG system using a golden set
of queries and expected relevant documents.

Metrics computed:
- Precision@K: Proportion of retrieved docs that are relevant
- Recall@K: Proportion of relevant docs that were retrieved
- MRR (Mean Reciprocal Rank): Average of 1/rank of first relevant doc
- Hit Rate@K: Proportion of queries with at least one relevant doc in top K

Usage:
    python -m evaluation.evaluate_retrieval
    python -m evaluation.evaluate_retrieval --k 5 --output results/
"""

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from search.vector_search import VectorSearch


@dataclass
class GoldenSetItem:
    """A single item from the golden set."""
    query_id: str
    query: str
    expected_doc_id: str
    relevance: str  # relevant, moderate, irrelevant
    category: str
    notes: str = ""


@dataclass
class QueryResult:
    """Result of evaluating a single query."""
    query_id: str
    query: str
    expected_docs: List[Tuple[str, str]]  # (doc_id, relevance)
    retrieved_docs: List[Tuple[str, float]]  # (doc_id, score)
    relevant_found: List[str]
    moderate_found: List[str]
    first_relevant_rank: Optional[int]
    precision_at_k: float
    recall: float
    hit: bool


@dataclass
class EvaluationResult:
    """Overall evaluation results."""
    timestamp: str
    k: int
    total_queries: int
    unique_queries: int
    
    # Aggregate metrics
    mean_precision_at_k: float
    mean_recall: float
    mrr: float  # Mean Reciprocal Rank
    hit_rate: float
    
    # Coverage metrics
    coverage_queries: int
    coverage_hit_rate: float
    
    # Retrieval metrics
    retrieval_queries: int
    retrieval_hit_rate: float
    retrieval_mrr: float
    
    # Per-query results
    query_results: List[QueryResult] = field(default_factory=list)


class RetrievalEvaluator:
    """Evaluates RAG retrieval quality against a golden set."""
    
    def __init__(self, golden_set_path: str = "evaluation/golden_set.csv"):
        """
        Initialize evaluator.
        
        Args:
            golden_set_path: Path to the golden set CSV file
        """
        self.golden_set_path = Path(golden_set_path)
        self.golden_set: Dict[str, List[GoldenSetItem]] = defaultdict(list)
        self.vector_search = VectorSearch()
        
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
        """
        Get expected relevant and moderate docs for a query.
        
        Returns:
            Tuple of (relevant_doc_ids, moderate_doc_ids)
        """
        relevant = []
        moderate = []
        
        for item in self.golden_set[query_id]:
            if item.relevance == 'relevant':
                relevant.append(item.expected_doc_id)
            elif item.relevance == 'moderate':
                moderate.append(item.expected_doc_id)
        
        return relevant, moderate
    
    def _evaluate_query(self, query_id: str, k: int) -> QueryResult:
        """
        Evaluate a single query.
        
        Args:
            query_id: Query ID from golden set
            k: Number of results to retrieve
            
        Returns:
            QueryResult with metrics
        """
        items = self.golden_set[query_id]
        query = items[0].query
        category = items[0].category
        
        relevant_expected, moderate_expected = self._get_expected_docs(query_id)
        all_expected = relevant_expected + moderate_expected
        
        # Handle coverage tests (expected NOT_IN_DB)
        if 'NOT_IN_DB' in relevant_expected:
            # For coverage tests, we expect NO relevant results
            results = self.vector_search.search(query, limit=k)
            retrieved_ids = [r.get('regulation_id', '') for r in results]
            retrieved_scores = [(r.get('regulation_id', ''), r.get('score', 0)) for r in results]
            
            return QueryResult(
                query_id=query_id,
                query=query,
                expected_docs=[(d, 'relevant') for d in relevant_expected],
                retrieved_docs=retrieved_scores,
                relevant_found=[],
                moderate_found=[],
                first_relevant_rank=None,
                precision_at_k=0.0,
                recall=0.0,
                hit=False  # For coverage tests, we expect miss
            )
        
        # Normal retrieval test
        results = self.vector_search.search(query, limit=k)
        retrieved_ids = [r.get('regulation_id', '') for r in results]
        retrieved_scores = [(r.get('regulation_id', ''), r.get('score', 0)) for r in results]
        
        # Find relevant docs in retrieved
        relevant_found = [d for d in retrieved_ids if d in relevant_expected]
        moderate_found = [d for d in retrieved_ids if d in moderate_expected]
        
        # Calculate first relevant rank (1-indexed)
        first_relevant_rank = None
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_expected:
                first_relevant_rank = i
                break
        
        # Calculate metrics
        # Precision@K: relevant found / K
        relevant_in_retrieved = len([d for d in retrieved_ids if d in all_expected])
        precision_at_k = relevant_in_retrieved / k if k > 0 else 0.0
        
        # Recall: relevant found / total relevant expected
        total_relevant = len(all_expected)
        recall = relevant_in_retrieved / total_relevant if total_relevant > 0 else 0.0
        
        # Hit: at least one relevant doc found
        hit = len(relevant_found) > 0 or len(moderate_found) > 0
        
        return QueryResult(
            query_id=query_id,
            query=query,
            expected_docs=[(d, 'relevant') for d in relevant_expected] + 
                         [(d, 'moderate') for d in moderate_expected],
            retrieved_docs=retrieved_scores,
            relevant_found=relevant_found,
            moderate_found=moderate_found,
            first_relevant_rank=first_relevant_rank,
            precision_at_k=precision_at_k,
            recall=recall,
            hit=hit
        )
    
    def evaluate(self, k: int = 5) -> EvaluationResult:
        """
        Run full evaluation on golden set.
        
        Args:
            k: Number of results to retrieve per query
            
        Returns:
            EvaluationResult with all metrics
        """
        logger.info(f"Starting evaluation with k={k}")
        
        query_results = []
        retrieval_results = []
        coverage_results = []
        
        for query_id in self.golden_set.keys():
            result = self._evaluate_query(query_id, k)
            query_results.append(result)
            
            # Separate by category
            category = self.golden_set[query_id][0].category
            if category == 'coverage':
                coverage_results.append(result)
            else:
                retrieval_results.append(result)
        
        # Calculate aggregate metrics for retrieval queries
        if retrieval_results:
            mean_precision = sum(r.precision_at_k for r in retrieval_results) / len(retrieval_results)
            mean_recall = sum(r.recall for r in retrieval_results) / len(retrieval_results)
            hit_rate = sum(1 for r in retrieval_results if r.hit) / len(retrieval_results)
            
            # MRR
            reciprocal_ranks = []
            for r in retrieval_results:
                if r.first_relevant_rank:
                    reciprocal_ranks.append(1.0 / r.first_relevant_rank)
                else:
                    reciprocal_ranks.append(0.0)
            mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
            
            retrieval_hit_rate = hit_rate
            retrieval_mrr = mrr
        else:
            mean_precision = mean_recall = hit_rate = mrr = 0.0
            retrieval_hit_rate = retrieval_mrr = 0.0
        
        # Coverage metrics (expecting NOT to find)
        coverage_hit_rate = 0.0
        if coverage_results:
            # For coverage, hit means we DIDN'T find irrelevant content marked as relevant
            coverage_hit_rate = sum(1 for r in coverage_results if not r.hit) / len(coverage_results)
        
        return EvaluationResult(
            timestamp=datetime.now().isoformat(),
            k=k,
            total_queries=len(query_results),
            unique_queries=len(self.golden_set),
            mean_precision_at_k=mean_precision,
            mean_recall=mean_recall,
            mrr=mrr,
            hit_rate=hit_rate,
            coverage_queries=len(coverage_results),
            coverage_hit_rate=coverage_hit_rate,
            retrieval_queries=len(retrieval_results),
            retrieval_hit_rate=retrieval_hit_rate,
            retrieval_mrr=retrieval_mrr,
            query_results=query_results
        )
    
    def save_results(self, result: EvaluationResult, output_dir: str = "evaluation/results"):
        """
        Save evaluation results to CSV files.
        
        Args:
            result: EvaluationResult to save
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary_path = output_path / f"summary_{timestamp}.csv"
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['timestamp', result.timestamp])
            writer.writerow(['k', result.k])
            writer.writerow(['total_queries', result.total_queries])
            writer.writerow(['unique_queries', result.unique_queries])
            writer.writerow(['mean_precision_at_k', f"{result.mean_precision_at_k:.4f}"])
            writer.writerow(['mean_recall', f"{result.mean_recall:.4f}"])
            writer.writerow(['mrr', f"{result.mrr:.4f}"])
            writer.writerow(['hit_rate', f"{result.hit_rate:.4f}"])
            writer.writerow(['retrieval_queries', result.retrieval_queries])
            writer.writerow(['retrieval_hit_rate', f"{result.retrieval_hit_rate:.4f}"])
            writer.writerow(['retrieval_mrr', f"{result.retrieval_mrr:.4f}"])
            writer.writerow(['coverage_queries', result.coverage_queries])
            writer.writerow(['coverage_hit_rate', f"{result.coverage_hit_rate:.4f}"])
        
        logger.info(f"Summary saved to: {summary_path}")
        
        # Save detailed results
        details_path = output_path / f"details_{timestamp}.csv"
        with open(details_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'query_id', 'query', 'expected_docs', 'retrieved_docs',
                'relevant_found', 'moderate_found', 'first_relevant_rank',
                'precision_at_k', 'recall', 'hit'
            ])
            
            for qr in result.query_results:
                writer.writerow([
                    qr.query_id,
                    qr.query,
                    ';'.join([f"{d}:{r}" for d, r in qr.expected_docs]),
                    ';'.join([f"{d}:{s:.3f}" for d, s in qr.retrieved_docs[:5]]),
                    ';'.join(qr.relevant_found),
                    ';'.join(qr.moderate_found),
                    qr.first_relevant_rank or 'N/A',
                    f"{qr.precision_at_k:.4f}",
                    f"{qr.recall:.4f}",
                    qr.hit
                ])
        
        logger.info(f"Details saved to: {details_path}")
        
        # Save as JSON for programmatic access
        json_path = output_path / f"results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': result.timestamp,
                'k': result.k,
                'metrics': {
                    'total_queries': result.total_queries,
                    'unique_queries': result.unique_queries,
                    'mean_precision_at_k': result.mean_precision_at_k,
                    'mean_recall': result.mean_recall,
                    'mrr': result.mrr,
                    'hit_rate': result.hit_rate,
                    'retrieval_queries': result.retrieval_queries,
                    'retrieval_hit_rate': result.retrieval_hit_rate,
                    'retrieval_mrr': result.retrieval_mrr,
                    'coverage_queries': result.coverage_queries,
                    'coverage_hit_rate': result.coverage_hit_rate,
                },
                'query_results': [
                    {
                        'query_id': qr.query_id,
                        'query': qr.query,
                        'hit': qr.hit,
                        'precision': qr.precision_at_k,
                        'recall': qr.recall,
                        'first_rank': qr.first_relevant_rank,
                    }
                    for qr in result.query_results
                ]
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON saved to: {json_path}")
        
        return summary_path, details_path, json_path


def print_report(result: EvaluationResult):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("RAG RETRIEVAL QUALITY EVALUATION REPORT")
    print("=" * 60)
    print(f"Timestamp: {result.timestamp}")
    print(f"K (results per query): {result.k}")
    print(f"Total queries evaluated: {result.total_queries}")
    print()
    
    print("RETRIEVAL METRICS (queries expecting results)")
    print("-" * 40)
    print(f"  Queries:           {result.retrieval_queries}")
    print(f"  Hit Rate@{result.k}:        {result.retrieval_hit_rate:.1%}")
    print(f"  MRR:               {result.retrieval_mrr:.4f}")
    print(f"  Mean Precision@{result.k}: {result.mean_precision_at_k:.4f}")
    print(f"  Mean Recall:       {result.mean_recall:.4f}")
    print()
    
    print("COVERAGE METRICS (queries expecting NO results)")
    print("-" * 40)
    print(f"  Queries:           {result.coverage_queries}")
    print(f"  Correct 'Not Found': {result.coverage_hit_rate:.1%}")
    print()
    
    print("PER-QUERY RESULTS")
    print("-" * 40)
    
    # Group by hit/miss
    hits = [qr for qr in result.query_results if qr.hit]
    misses = [qr for qr in result.query_results if not qr.hit and 'NOT_IN_DB' not in str(qr.expected_docs)]
    
    print(f"\n✅ HITS ({len(hits)}):")
    for qr in hits[:10]:
        rank_str = f"rank {qr.first_relevant_rank}" if qr.first_relevant_rank else "moderate"
        print(f"  [{qr.query_id}] {qr.query[:50]}... → {rank_str}")
    
    print(f"\n❌ MISSES ({len(misses)}):")
    for qr in misses:
        expected = [d for d, r in qr.expected_docs]
        retrieved = [d for d, s in qr.retrieved_docs[:3]]
        print(f"  [{qr.query_id}] {qr.query[:50]}...")
        print(f"       Expected: {expected}")
        print(f"       Got:      {retrieved}")
    
    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument('--k', type=int, default=5, help="Number of results to retrieve")
    parser.add_argument('--golden-set', type=str, default="evaluation/golden_set.csv",
                       help="Path to golden set CSV")
    parser.add_argument('--output', type=str, default="evaluation/results",
                       help="Output directory for results")
    parser.add_argument('--quiet', action='store_true', help="Suppress detailed output")
    
    args = parser.parse_args()
    
    try:
        evaluator = RetrievalEvaluator(args.golden_set)
        result = evaluator.evaluate(k=args.k)
        
        if not args.quiet:
            print_report(result)
        
        # Save results
        evaluator.save_results(result, args.output)
        
        # Exit with success/failure based on hit rate
        if result.retrieval_hit_rate < 0.5:
            logger.warning(f"Low hit rate: {result.retrieval_hit_rate:.1%}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
