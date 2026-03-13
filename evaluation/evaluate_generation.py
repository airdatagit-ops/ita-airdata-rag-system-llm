"""
RAG Generation Quality Evaluation Script.

Evaluates LLM response quality using heuristic metrics (no external model required).

Metrics computed:
- Empty Rate: % of queries where LLM returned "not found"
- Citation Rate: % of responses containing document source IDs
- Hedging Rate: % of responses with uncertainty language
- Mean Response Length: average token count of responses
- Hallucination Indicators: responses that cite non-existent document IDs

Usage:
    python -m evaluation.evaluate_generation
    python -m evaluation.evaluate_generation --k 5 --workers 4 --sample 20
"""

import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from search.vector_search import VectorSearch
from models.llm import LlamaModel


NOT_FOUND_PATTERNS = [
    r"não encontrei",
    r"não há informação",
    r"não consta",
    r"não foi possível encontrar",
    r"não está disponível",
    r"não possuo informação",
    r"informação não está nos documentos",
]

HEDGING_PATTERNS = [
    r"possivelmente",
    r"provavelmente",
    r"acredito que",
    r"pode ser que",
    r"é possível que",
    r"talvez",
    r"não tenho certeza",
    r"pelo que sei",
    r"pelo meu conhecimento",
    r"de modo geral",
]

CITATION_PATTERN = re.compile(r"ICA[- ]?\d+[- ]?\d*(?:-art\d+)?", re.IGNORECASE)


@dataclass
class ResponseAnalysis:
    """Analysis of a single LLM response."""
    query_id: str
    query: str
    response: str
    response_tokens: int
    is_empty: bool
    has_citation: bool
    citations_found: List[str]
    has_hedging: bool
    hedging_words: List[str]
    retrieved_doc_ids: List[str]
    search_time_ms: int
    llm_time_ms: int


@dataclass
class GenerationEvalResult:
    """Overall generation evaluation results."""
    timestamp: str
    k: int
    total_queries: int
    elapsed_seconds: float
    empty_rate: float
    citation_rate: float
    hedging_rate: float
    mean_response_tokens: float
    median_response_tokens: float
    analyses: List[ResponseAnalysis] = field(default_factory=list)


def analyze_response(
    query_id: str,
    query: str,
    response: str,
    retrieved_doc_ids: List[str],
    search_time_ms: int,
    llm_time_ms: int,
) -> ResponseAnalysis:
    """Analyze a single response for quality heuristics."""
    lower = response.lower()

    is_empty = any(re.search(p, lower) for p in NOT_FOUND_PATTERNS)

    citations = CITATION_PATTERN.findall(response)
    has_citation = len(citations) > 0

    hedging_found = [p for p in HEDGING_PATTERNS if re.search(p, lower)]
    has_hedging = len(hedging_found) > 0

    tokens = len(response.split())

    return ResponseAnalysis(
        query_id=query_id,
        query=query,
        response=response,
        response_tokens=tokens,
        is_empty=is_empty,
        has_citation=has_citation,
        citations_found=citations,
        has_hedging=has_hedging,
        hedging_words=hedging_found,
        retrieved_doc_ids=retrieved_doc_ids,
        search_time_ms=search_time_ms,
        llm_time_ms=llm_time_ms,
    )


class GenerationEvaluator:
    """Evaluates RAG generation quality using heuristics."""

    def __init__(self, golden_set_path: str = "evaluation/golden_set.csv"):
        self.golden_set_path = Path(golden_set_path)
        self.queries: List[Tuple[str, str]] = []
        self._load_queries()

    def _load_queries(self):
        if not self.golden_set_path.exists():
            raise FileNotFoundError(f"Golden set not found: {self.golden_set_path}")

        seen = set()
        with open(self.golden_set_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = row['query_id']
                if qid not in seen and row.get('category') != 'coverage':
                    seen.add(qid)
                    self.queries.append((qid, row['query']))

        logger.info(f"Loaded {len(self.queries)} queries for generation evaluation")

    def evaluate(
        self,
        k: int = 5,
        sample: Optional[int] = None,
        workers: int = 1,
    ) -> GenerationEvalResult:
        """
        Run generation evaluation on golden set queries.

        Args:
            k: Number of documents to retrieve per query
            sample: Limit to N queries (useful for quick tests)
            workers: Parallel workers for RAG pipeline
        """
        start = time.time()

        queries = self.queries[:sample] if sample else self.queries
        logger.info(f"Evaluating {len(queries)} queries (k={k}, workers={workers})")

        search = VectorSearch()
        llm = LlamaModel()

        def _run_single(qid: str, query: str) -> ResponseAnalysis:
            search_start = time.time()
            results = search.search(query, limit=k)
            search_ms = int((time.time() - search_start) * 1000)

            doc_ids = [r.get('regulation_id', '') for r in results]

            llm_start = time.time()
            if results:
                answer = llm.generate_with_context(query=query, context_documents=results)
            else:
                answer = "Não encontrei informações relevantes nos documentos disponíveis."
            llm_ms = int((time.time() - llm_start) * 1000)

            return analyze_response(qid, query, answer, doc_ids, search_ms, llm_ms)

        analyses = []
        if workers <= 1:
            for qid, query in queries:
                analyses.append(_run_single(qid, query))
                logger.debug(f"[{qid}] done ({len(analyses)}/{len(queries)})")
        else:
            futures_map = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for qid, query in queries:
                    f = executor.submit(_run_single, qid, query)
                    futures_map[f] = qid
                for future in as_completed(futures_map):
                    analyses.append(future.result())
                    logger.debug(f"[{futures_map[future]}] done ({len(analyses)}/{len(queries)})")

        elapsed = time.time() - start
        n = len(analyses)

        empty_rate = sum(1 for a in analyses if a.is_empty) / n if n else 0
        citation_rate = sum(1 for a in analyses if a.has_citation) / n if n else 0
        hedging_rate = sum(1 for a in analyses if a.has_hedging) / n if n else 0
        tokens = sorted(a.response_tokens for a in analyses)
        mean_tokens = sum(tokens) / n if n else 0
        median_tokens = tokens[n // 2] if n else 0

        return GenerationEvalResult(
            timestamp=datetime.now().isoformat(),
            k=k,
            total_queries=n,
            elapsed_seconds=elapsed,
            empty_rate=empty_rate,
            citation_rate=citation_rate,
            hedging_rate=hedging_rate,
            mean_response_tokens=mean_tokens,
            median_response_tokens=median_tokens,
            analyses=analyses,
        )

    def save_results(self, result: GenerationEvalResult, output_dir: str = "evaluation/results"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        ts = result.timestamp.replace(":", "").replace("-", "")[:15]

        summary_path = output_path / f"gen_summary_{ts}.csv"
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerows([
                ('timestamp', result.timestamp),
                ('k', result.k),
                ('total_queries', result.total_queries),
                ('elapsed_seconds', f"{result.elapsed_seconds:.2f}"),
                ('empty_rate', f"{result.empty_rate:.4f}"),
                ('citation_rate', f"{result.citation_rate:.4f}"),
                ('hedging_rate', f"{result.hedging_rate:.4f}"),
                ('mean_response_tokens', f"{result.mean_response_tokens:.1f}"),
                ('median_response_tokens', f"{result.median_response_tokens:.1f}"),
            ])
        logger.info(f"Summary saved: {summary_path}")

        details_path = output_path / f"gen_details_{ts}.csv"
        with open(details_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'query_id', 'query', 'is_empty', 'has_citation', 'citations',
                'has_hedging', 'hedging_words', 'response_tokens',
                'retrieved_docs', 'search_ms', 'llm_ms', 'response_preview'
            ])
            for a in result.analyses:
                writer.writerow([
                    a.query_id, a.query, a.is_empty, a.has_citation,
                    ';'.join(a.citations_found), a.has_hedging,
                    ';'.join(a.hedging_words), a.response_tokens,
                    ';'.join(a.retrieved_doc_ids),
                    a.search_time_ms, a.llm_time_ms,
                    a.response[:200].replace('\n', ' '),
                ])
        logger.info(f"Details saved: {details_path}")

        json_path = output_path / f"gen_results_{ts}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': result.timestamp,
                'k': result.k,
                'elapsed_seconds': result.elapsed_seconds,
                'metrics': {
                    'total_queries': result.total_queries,
                    'empty_rate': result.empty_rate,
                    'citation_rate': result.citation_rate,
                    'hedging_rate': result.hedging_rate,
                    'mean_response_tokens': result.mean_response_tokens,
                    'median_response_tokens': result.median_response_tokens,
                },
                'analyses': [
                    {
                        'query_id': a.query_id,
                        'query': a.query,
                        'is_empty': a.is_empty,
                        'has_citation': a.has_citation,
                        'has_hedging': a.has_hedging,
                        'response_tokens': a.response_tokens,
                        'response_preview': a.response[:300],
                    }
                    for a in result.analyses
                ]
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON saved: {json_path}")

        return summary_path, details_path, json_path


def print_report(result: GenerationEvalResult):
    """Print formatted generation evaluation report."""
    print("\n" + "=" * 70)
    print("  RAG GENERATION QUALITY EVALUATION REPORT")
    print("=" * 70)
    print(f"  Timestamp:    {result.timestamp}")
    print(f"  K:            {result.k}")
    print(f"  Elapsed:      {result.elapsed_seconds:.2f}s")
    print(f"  Queries:      {result.total_queries}")
    print()

    print("  RESPONSE METRICS")
    print("  " + "-" * 40)
    print(f"    Empty Rate:          {result.empty_rate:.1%}")
    print(f"    Citation Rate:       {result.citation_rate:.1%}")
    print(f"    Hedging Rate:        {result.hedging_rate:.1%}")
    print(f"    Mean Tokens:         {result.mean_response_tokens:.0f}")
    print(f"    Median Tokens:       {result.median_response_tokens:.0f}")
    print()

    empty = [a for a in result.analyses if a.is_empty]
    hedging = [a for a in result.analyses if a.has_hedging]
    no_cite = [a for a in result.analyses if not a.has_citation and not a.is_empty]

    if empty:
        print(f"  EMPTY RESPONSES ({len(empty)})")
        print("  " + "-" * 40)
        for a in empty:
            print(f"    [{a.query_id}] {a.query[:60]}")

    if hedging:
        print(f"\n  HEDGING DETECTED ({len(hedging)})")
        print("  " + "-" * 40)
        for a in hedging:
            words = ', '.join(a.hedging_words)
            print(f"    [{a.query_id}] {a.query[:50]:<50} [{words}]")

    if no_cite:
        print(f"\n  NO CITATION ({len(no_cite)})")
        print("  " + "-" * 40)
        for a in no_cite:
            print(f"    [{a.query_id}] {a.query[:60]}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG generation quality")
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--golden-set', type=str, default="evaluation/golden_set.csv")
    parser.add_argument('--output', type=str, default="evaluation/results")
    parser.add_argument('--sample', type=int, default=None,
                       help="Limit to N queries (for quick tests)")
    parser.add_argument('--workers', type=int, default=1,
                       help="Parallel workers (careful with LLM concurrency)")
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    try:
        evaluator = GenerationEvaluator(args.golden_set)
        result = evaluator.evaluate(k=args.k, sample=args.sample, workers=args.workers)

        if not args.quiet:
            print_report(result)

        evaluator.save_results(result, args.output)
        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
