"""
RAG Generation Quality Evaluation Script.

Two evaluation modes:
- ``legacy`` (default): direct embedding + Qdrant + ``llm.generate_with_context``.
  Kept for backward-compatibility with historical reports.
- ``--use-pipeline``: runs the full ``RAGPipeline.query()`` (rewriter +
  searcher + evaluator + generator). This is what the user actually
  receives and SHOULD be used for production-grade comparisons.

Metrics computed:
- Heuristics (legacy): empty_rate, hedging_rate, response token stats.
- Citation grounding (programmatic, no LLM):
    citation_rate (presence), citations_grounded, citations_hallucinated,
    citation_grounding_rate (= grounded / total cited).
- LLM-as-judge (optional, ``--use-judge``):
    mean_faithfulness, mean_relevance, judge_unparseable_rate.

Usage:
    python -m evaluation.evaluate_generation                   # legacy heuristics
    python -m evaluation.evaluate_generation --use-pipeline    # end-to-end via RAGPipeline
    python -m evaluation.evaluate_generation --use-pipeline --use-judge
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
from typing import Dict, List, Optional

from loguru import logger

from models.gpu_client import create_embedding_model, create_llm
from database.qdrant_manager import QdrantManager
from config import config
from evaluation.evaluate_retrieval import (
    _matches_expected as _id_matches_expected,
)
from evaluation.llm_judge import LLMJudge

try:
    from models.embeddings import SparseEncoder
except ImportError:
    SparseEncoder = None


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

# Bracketed citations are the format the generator is instructed to use:
#   [ICA 100-40]  [MCA 56-5]  [Decreto 97.464]  [Lei 7.183]  [ICA 100-40-art10]
# We deliberately accept any common acronym/word followed by an
# alphanumeric reference; that catches future doc types without having
# to enumerate them. We strip surrounding brackets when extracting.
_BRACKET_CITATION = re.compile(
    r"\[([A-ZÁÉÍÓÚÂÊÔÃÕÇa-z][A-ZÁÉÍÓÚÂÊÔÃÕÇa-z.]{2,12}\s+[\d./-]+(?:-art\d+(?:-\d+)?)?)\]"
)
# Fallback for un-bracketed mentions (legacy heuristic; used only when no
# bracketed citation is present, to avoid double-counting).
CITATION_PATTERN = re.compile(
    r"(?:ICA|DCA|MCA|PCA|TCA|FCA|OCA|NSCA|RCA|ROCA|RICA|NPA|CIRCEA|BCA|BMA|IMA|RIMA|RMA)"
    r"[- ]?\d+[- ]?\d*(?:-art\d+)?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------
# Citation grounding (programmatic, no LLM)
# ---------------------------------------------------------------------


def extract_citations(response: str) -> List[str]:
    """Extract citations from a response.

    Prefers bracketed citations (the format the generator is instructed
    to use). Falls back to the legacy un-bracketed pattern when no
    bracketed citation is found, to avoid double-counting.
    """
    bracketed = [m.group(1).strip() for m in _BRACKET_CITATION.finditer(response)]
    if bracketed:
        return bracketed
    return CITATION_PATTERN.findall(response)


def compute_grounding(
    citations: List[str],
    retrieved_doc_ids: List[str],
) -> Dict[str, object]:
    """Check whether each citation is grounded in a retrieved doc.

    Returns ``{grounded, hallucinated, hallucinated_list}``.
    Matching is version-agnostic and accepts doc-level vs article-level
    citations (e.g. ``[ICA 100-40]`` matches any chunk of ica_100-40).
    """
    grounded = 0
    hallucinated_list: List[str] = []
    for cit in citations:
        cit_norm = cit.replace(" ", "-").replace(".", "")
        if any(_id_matches_expected(rid, cit_norm) for rid in retrieved_doc_ids):
            grounded += 1
        else:
            hallucinated_list.append(cit)
    return {
        "grounded": grounded,
        "hallucinated": len(hallucinated_list),
        "hallucinated_list": hallucinated_list,
    }


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
    citations_grounded: int = 0
    citations_hallucinated: int = 0
    hallucinated_citations: List[str] = field(default_factory=list)
    faithfulness_score: Optional[float] = None
    faithfulness_reasoning: str = ""
    relevance_score: Optional[float] = None
    relevance_reasoning: str = ""


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
    mode: str = "legacy"
    judge_used: bool = False
    judge_model: str = ""
    citation_grounding_rate: float = 0.0
    mean_citations_per_response: float = 0.0
    mean_faithfulness: Optional[float] = None
    mean_relevance: Optional[float] = None
    judge_unparseable_rate: float = 0.0
    analyses: List[ResponseAnalysis] = field(default_factory=list)


def analyze_response(
    query_id: str,
    query: str,
    response: str,
    retrieved_doc_ids: List[str],
    search_time_ms: int,
    llm_time_ms: int,
) -> ResponseAnalysis:
    """Analyze a single response for quality heuristics + citation grounding."""
    lower = response.lower()

    is_empty = any(re.search(p, lower) for p in NOT_FOUND_PATTERNS)
    citations = extract_citations(response)
    hedging_found = [p for p in HEDGING_PATTERNS if re.search(p, lower)]
    grounding = compute_grounding(citations, retrieved_doc_ids)

    return ResponseAnalysis(
        query_id=query_id,
        query=query,
        response=response,
        response_tokens=len(response.split()),
        is_empty=is_empty,
        has_citation=bool(citations),
        citations_found=citations,
        has_hedging=bool(hedging_found),
        hedging_words=hedging_found,
        retrieved_doc_ids=retrieved_doc_ids,
        search_time_ms=search_time_ms,
        llm_time_ms=llm_time_ms,
        citations_grounded=int(grounding["grounded"]),
        citations_hallucinated=int(grounding["hallucinated"]),
        hallucinated_citations=list(grounding["hallucinated_list"]),
    )


class GenerationEvaluator:
    """Evaluates RAG generation quality using heuristics."""

    def __init__(self, golden_set_path: str = "evaluation/golden_set.csv"):
        self.golden_set_path = Path(golden_set_path)
        self.queries: Dict[str, str] = {}
        self._load_queries()

    def _load_queries(self):
        if not self.golden_set_path.exists():
            raise FileNotFoundError(f"Golden set not found: {self.golden_set_path}")

        with open(self.golden_set_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = row['query_id']
                if qid not in self.queries and row.get('category') != 'coverage':
                    self.queries[qid] = row['query']

        logger.info(f"Loaded {len(self.queries)} queries for generation evaluation")

    def evaluate(self, k: int = 5, sample: Optional[int] = None) -> GenerationEvalResult:
        """
        Run generation evaluation. Search is batch-encoded and parallelized,
        LLM calls are sequential (Ollama serializes requests).

        Args:
            k: Number of documents to retrieve per query
            sample: Limit to N queries (useful for quick tests)
        """
        start = time.time()

        query_ids = list(self.queries.keys())[:sample]
        query_texts = [self.queries[qid] for qid in query_ids]
        logger.info(f"Evaluating {len(query_ids)} queries (k={k})")

        use_dense = getattr(config, 'SEARCH_DENSE_ENABLED', True)
        use_sparse = getattr(config, 'SEARCH_SPARSE_ENABLED', False)

        if use_sparse and SparseEncoder is None:
            logger.warning("SparseEncoder not available — falling back to dense-only")
            use_sparse = False
            use_dense = True

        hybrid_mode = use_dense and use_sparse
        sparse_mode = use_sparse and not use_dense
        mode_label = "hybrid" if hybrid_mode else ("sparse" if sparse_mode else "dense")

        embed_model = create_embedding_model()
        qdrant = QdrantManager()
        llm = create_llm()

        logger.info(f"Batch encoding {len(query_texts)} queries (mode={mode_label})...")
        dense_embeddings = embed_model.encode(query_texts) if use_dense else None

        sparse_embeddings = None
        if use_sparse:
            sparse_model = SparseEncoder()
            sparse_embeddings = sparse_model.encode(query_texts)

        analyses = []
        for i, qid in enumerate(query_ids):
            search_start = time.time()

            search_kwargs = {"limit": k}
            if hybrid_mode or sparse_mode:
                if dense_embeddings is not None:
                    search_kwargs["dense_vector"] = dense_embeddings[i].tolist()
                if sparse_embeddings is not None:
                    search_kwargs["sparse_vector"] = sparse_embeddings[i]
            else:
                search_kwargs["query_vector"] = dense_embeddings[i].tolist()
                search_kwargs["score_threshold"] = config.SEARCH_SCORE_THRESHOLD

            raw_results = qdrant.search(**search_kwargs)
            search_ms = int((time.time() - search_start) * 1000)

            doc_ids = [r.payload.get('regulation_id', '') for r in raw_results]
            context_docs = [
                {
                    'regulation_id': r.payload.get('regulation_id', ''),
                    'text': r.payload.get('text', ''),
                    'score': r.score,
                }
                for r in raw_results
            ]

            llm_start = time.time()
            if context_docs:
                answer = llm.generate_with_context(query=query_texts[i], context_documents=context_docs)
            else:
                answer = "Não encontrei informações relevantes nos documentos disponíveis."
            llm_ms = int((time.time() - llm_start) * 1000)

            analyses.append(analyze_response(qid, query_texts[i], answer, doc_ids, search_ms, llm_ms))
            logger.debug(f"[{qid}] done ({len(analyses)}/{len(query_ids)}) llm={llm_ms}ms")

        elapsed = time.time() - start
        return self._aggregate(analyses, k, elapsed, mode="legacy")

    def evaluate_pipeline(
        self,
        k: int = 5,
        sample: Optional[int] = None,
        judge: Optional[LLMJudge] = None,
        judge_workers: int = 1,
    ) -> GenerationEvalResult:
        """End-to-end evaluation through ``RAGPipeline`` (full prod path).

        Captures the actual sources the generator used (post-evaluator)
        and runs LLM-as-judge in a second pass when ``judge`` is given.
        """
        from search.orchestrator.pipeline import RAGPipeline

        start = time.time()
        query_ids = list(self.queries.keys())[:sample]
        logger.info(f"Pipeline generation eval: {len(query_ids)} queries (k={k})")

        pipeline = RAGPipeline()

        analyses: List[ResponseAnalysis] = []
        sources_by_qid: Dict[str, List[dict]] = {}
        for i, qid in enumerate(query_ids, 1):
            question = self.queries[qid]
            t0 = time.time()
            response = pipeline.query(
                question, limit=k, return_sources=True, debug=True,
            )
            wall_ms = int((time.time() - t0) * 1000)
            answer = response.get("answer") or ""
            sources = response.get("sources") or []
            doc_ids = [s.get("regulation_id", "") for s in sources]
            timings = (response.get("trace") or {}).get("timings") or {}
            search_ms = int(
                (timings.get("rewriter_ms") or 0)
                + (timings.get("searcher_ms") or 0)
                + (timings.get("evaluator_ms") or 0)
            ) or response.get("search_time_ms", 0)
            llm_ms = int(timings.get("generator_ms") or response.get("llm_time_ms", 0))

            analyses.append(analyze_response(
                qid, question, answer, doc_ids, search_ms, llm_ms,
            ))
            sources_by_qid[qid] = sources
            logger.info(
                f"[{qid}] done ({i}/{len(query_ids)}) "
                f"wall={wall_ms}ms gen={llm_ms}ms sources={len(sources)}"
            )

        if judge is not None:
            self._apply_judge(analyses, query_ids, judge, judge_workers, sources_by_qid)

        elapsed = time.time() - start
        return self._aggregate(
            analyses, k, elapsed,
            mode="pipeline",
            judge_used=judge is not None,
            judge_model=judge.model_name if judge else "",
        )

    def _apply_judge(
        self,
        analyses: List[ResponseAnalysis],
        query_ids: List[str],
        judge: LLMJudge,
        workers: int,
        sources_by_qid: Dict[str, List[dict]],
    ) -> None:
        """Run LLM-as-judge using sources already collected in the first pass.

        We deliberately reuse the ``sources`` returned by ``pipeline.query``
        instead of re-running the pipeline: a second search wastes GPU time
        and (with workers>1) creates contention with the judge LLM. Sources
        from the first pass are exactly what the generator saw, which is
        also what we want the judge to evaluate against.
        """
        logger.info(f"LLM-as-judge ({judge.model_name}) on {len(analyses)} responses")

        analysis_by_qid = {a.query_id: a for a in analyses}

        def judge_one(qid: str) -> None:
            a = analysis_by_qid[qid]
            sources = sources_by_qid.get(qid, [])
            t0 = time.time()
            faith = judge.judge_faithfulness(a.query, a.response, sources)
            rel = judge.judge_relevance(a.query, a.response)
            a.faithfulness_score = faith.score
            a.faithfulness_reasoning = faith.reasoning or (faith.parse_error or "")
            a.relevance_score = rel.score
            a.relevance_reasoning = rel.reasoning or (rel.parse_error or "")
            logger.info(
                f"[{qid}] judged in {int((time.time()-t0)*1000)}ms "
                f"faith={faith.raw_score} rel={rel.raw_score}"
            )

        if workers <= 1:
            for qid in query_ids:
                judge_one(qid)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(judge_one, qid) for qid in query_ids]
                for f in as_completed(futures):
                    f.result()

    def _aggregate(
        self,
        analyses: List[ResponseAnalysis],
        k: int,
        elapsed: float,
        *,
        mode: str = "legacy",
        judge_used: bool = False,
        judge_model: str = "",
    ) -> GenerationEvalResult:
        n = len(analyses)
        tokens = sorted(a.response_tokens for a in analyses)
        if not n:
            return GenerationEvalResult(
                timestamp=datetime.now().isoformat(), k=k,
                total_queries=0, elapsed_seconds=elapsed,
                empty_rate=0, citation_rate=0, hedging_rate=0,
                mean_response_tokens=0, median_response_tokens=0,
                mode=mode, judge_used=judge_used, judge_model=judge_model,
                analyses=analyses,
            )

        total_citations = sum(len(a.citations_found) for a in analyses)
        total_grounded = sum(a.citations_grounded for a in analyses)
        grounding_rate = (total_grounded / total_citations) if total_citations else 0.0

        faiths = [a.faithfulness_score for a in analyses if a.faithfulness_score is not None]
        rels = [a.relevance_score for a in analyses if a.relevance_score is not None]
        unparseable = (
            sum(
                1 for a in analyses
                if (judge_used and (a.faithfulness_score is None or a.relevance_score is None))
            ) / n if judge_used else 0.0
        )

        return GenerationEvalResult(
            timestamp=datetime.now().isoformat(),
            k=k,
            total_queries=n,
            elapsed_seconds=elapsed,
            empty_rate=sum(1 for a in analyses if a.is_empty) / n,
            citation_rate=sum(1 for a in analyses if a.has_citation) / n,
            hedging_rate=sum(1 for a in analyses if a.has_hedging) / n,
            mean_response_tokens=sum(tokens) / n,
            median_response_tokens=tokens[n // 2],
            mode=mode,
            judge_used=judge_used,
            judge_model=judge_model,
            citation_grounding_rate=grounding_rate,
            mean_citations_per_response=total_citations / n,
            mean_faithfulness=(sum(faiths) / len(faiths)) if faiths else None,
            mean_relevance=(sum(rels) / len(rels)) if rels else None,
            judge_unparseable_rate=unparseable,
            analyses=analyses,
        )

    def save_results(self, result: GenerationEvalResult, output_dir: str = "evaluation/results"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        ts = result.timestamp.replace(":", "").replace("-", "")[:15]

        summary_path = output_path / f"gen_summary_{result.mode}_{ts}.csv"
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            rows = [
                ('timestamp', result.timestamp),
                ('mode', result.mode),
                ('k', result.k),
                ('total_queries', result.total_queries),
                ('elapsed_seconds', f"{result.elapsed_seconds:.2f}"),
                ('empty_rate', f"{result.empty_rate:.4f}"),
                ('citation_rate', f"{result.citation_rate:.4f}"),
                ('citation_grounding_rate', f"{result.citation_grounding_rate:.4f}"),
                ('mean_citations_per_response', f"{result.mean_citations_per_response:.2f}"),
                ('hedging_rate', f"{result.hedging_rate:.4f}"),
                ('mean_response_tokens', f"{result.mean_response_tokens:.1f}"),
                ('median_response_tokens', f"{result.median_response_tokens:.1f}"),
            ]
            if result.judge_used:
                rows += [
                    ('judge_model', result.judge_model),
                    ('mean_faithfulness', f"{result.mean_faithfulness:.4f}"
                     if result.mean_faithfulness is not None else "N/A"),
                    ('mean_relevance', f"{result.mean_relevance:.4f}"
                     if result.mean_relevance is not None else "N/A"),
                    ('judge_unparseable_rate', f"{result.judge_unparseable_rate:.4f}"),
                ]
            writer.writerows(rows)
        logger.info(f"Summary saved: {summary_path}")

        details_path = output_path / f"gen_details_{result.mode}_{ts}.csv"
        with open(details_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = [
                'query_id', 'query', 'is_empty', 'has_citation', 'citations',
                'citations_grounded', 'citations_hallucinated',
                'hallucinated_citations',
                'has_hedging', 'hedging_words', 'response_tokens',
                'retrieved_docs', 'search_ms', 'llm_ms',
            ]
            if result.judge_used:
                header += [
                    'faithfulness', 'faithfulness_reasoning',
                    'relevance', 'relevance_reasoning',
                ]
            header.append('response_preview')
            writer.writerow(header)
            for a in result.analyses:
                row = [
                    a.query_id, a.query, a.is_empty, a.has_citation,
                    ';'.join(a.citations_found),
                    a.citations_grounded, a.citations_hallucinated,
                    ';'.join(a.hallucinated_citations),
                    a.has_hedging, ';'.join(a.hedging_words),
                    a.response_tokens, ';'.join(a.retrieved_doc_ids),
                    a.search_time_ms, a.llm_time_ms,
                ]
                if result.judge_used:
                    row += [
                        f"{a.faithfulness_score:.2f}" if a.faithfulness_score is not None else "N/A",
                        a.faithfulness_reasoning,
                        f"{a.relevance_score:.2f}" if a.relevance_score is not None else "N/A",
                        a.relevance_reasoning,
                    ]
                row.append(a.response[:200].replace('\n', ' '))
                writer.writerow(row)
        logger.info(f"Details saved: {details_path}")

        json_path = output_path / f"gen_results_{result.mode}_{ts}.json"
        metrics_blob = {
            'total_queries': result.total_queries,
            'empty_rate': result.empty_rate,
            'citation_rate': result.citation_rate,
            'citation_grounding_rate': result.citation_grounding_rate,
            'mean_citations_per_response': result.mean_citations_per_response,
            'hedging_rate': result.hedging_rate,
            'mean_response_tokens': result.mean_response_tokens,
            'median_response_tokens': result.median_response_tokens,
        }
        if result.judge_used:
            metrics_blob.update({
                'judge_model': result.judge_model,
                'mean_faithfulness': result.mean_faithfulness,
                'mean_relevance': result.mean_relevance,
                'judge_unparseable_rate': result.judge_unparseable_rate,
            })
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': result.timestamp,
                'mode': result.mode,
                'k': result.k,
                'elapsed_seconds': result.elapsed_seconds,
                'metrics': metrics_blob,
                'analyses': [
                    {
                        'query_id': a.query_id,
                        'query': a.query,
                        'is_empty': a.is_empty,
                        'has_citation': a.has_citation,
                        'citations_found': a.citations_found,
                        'citations_grounded': a.citations_grounded,
                        'citations_hallucinated': a.citations_hallucinated,
                        'hallucinated_citations': a.hallucinated_citations,
                        'has_hedging': a.has_hedging,
                        'response_tokens': a.response_tokens,
                        'llm_ms': a.llm_time_ms,
                        'search_ms': a.search_time_ms,
                        'retrieved_doc_ids': a.retrieved_doc_ids,
                        'faithfulness': a.faithfulness_score,
                        'relevance': a.relevance_score,
                        'response': a.response,
                    }
                    for a in result.analyses
                ],
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON saved: {json_path}")

        return summary_path, details_path, json_path


def print_report(result: GenerationEvalResult):
    """Print formatted generation evaluation report."""
    print("\n" + "=" * 70)
    print(f"  RAG GENERATION QUALITY EVALUATION REPORT  ({result.mode} mode)")
    print("=" * 70)
    print(f"  Timestamp:    {result.timestamp}")
    print(f"  Mode:         {result.mode}")
    print(f"  K:            {result.k}")
    print(f"  Elapsed:      {result.elapsed_seconds:.2f}s")
    print(f"  Queries:      {result.total_queries}")
    print()

    print("  RESPONSE METRICS")
    print("  " + "-" * 40)
    print(f"    Empty Rate:               {result.empty_rate:.1%}")
    print(f"    Citation Rate:            {result.citation_rate:.1%}")
    print(f"    Citation Grounding Rate:  {result.citation_grounding_rate:.1%}")
    print(f"    Mean Citations / Resp:    {result.mean_citations_per_response:.2f}")
    print(f"    Hedging Rate:             {result.hedging_rate:.1%}")
    print(f"    Mean Tokens:              {result.mean_response_tokens:.0f}")
    print(f"    Median Tokens:            {result.median_response_tokens:.0f}")
    print()

    if result.judge_used:
        print("  LLM-AS-JUDGE METRICS")
        print("  " + "-" * 40)
        print(f"    Judge Model:              {result.judge_model}")
        faith = (
            f"{result.mean_faithfulness:.3f}"
            if result.mean_faithfulness is not None else "N/A"
        )
        rel = (
            f"{result.mean_relevance:.3f}"
            if result.mean_relevance is not None else "N/A"
        )
        print(f"    Mean Faithfulness (0-1):  {faith}")
        print(f"    Mean Relevance    (0-1):  {rel}")
        print(f"    Unparseable Rate:         {result.judge_unparseable_rate:.1%}")
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


def _build_judge() -> LLMJudge:
    """Instantiate the configured LLM judge."""
    judge_model = config.JUDGE_MODEL or config.GENERATOR_MODEL or config.OLLAMA_MODEL
    if not judge_model:
        raise RuntimeError(
            "No JUDGE_MODEL / GENERATOR_MODEL / OLLAMA_MODEL configured."
        )
    judge_llm = create_llm(model_name=judge_model)
    return LLMJudge(
        llm=judge_llm,
        model_name=judge_model,
        generator_model=config.GENERATOR_MODEL or config.OLLAMA_MODEL,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG generation quality")
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--golden-set', type=str, default="evaluation/golden_set.csv")
    parser.add_argument('--output', type=str, default="evaluation/results")
    parser.add_argument('--sample', type=int, default=None,
                       help="Limit to N queries (for quick tests)")
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--use-pipeline', action='store_true',
                       help="Run end-to-end through RAGPipeline (rewriter+searcher+"
                            "evaluator+generator). Reflects what the user actually"
                            " receives. Recommended for real comparisons.")
    parser.add_argument('--use-judge', action='store_true',
                       help="Run LLM-as-judge for faithfulness + answer relevance."
                            " Requires --use-pipeline. Sets JUDGE_MODEL or falls back"
                            " to GENERATOR_MODEL (with self-preference warning).")
    parser.add_argument('--judge-workers', type=int, default=1,
                       help="Parallel workers for judge calls (sequential by default)")

    args = parser.parse_args()

    if args.use_judge and not args.use_pipeline:
        parser.error("--use-judge requires --use-pipeline")

    try:
        evaluator = GenerationEvaluator(args.golden_set)
        if args.use_pipeline:
            judge = _build_judge() if args.use_judge else None
            result = evaluator.evaluate_pipeline(
                k=args.k, sample=args.sample,
                judge=judge, judge_workers=args.judge_workers,
            )
        else:
            result = evaluator.evaluate(k=args.k, sample=args.sample)

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
