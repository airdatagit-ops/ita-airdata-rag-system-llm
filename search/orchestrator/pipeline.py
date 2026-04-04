"""RAG pipeline orchestrator.

Chains Rewriter -> Searcher -> Evaluator -> Generator with per-stage
error handling, timeouts, and an optional debug trace.
"""

from __future__ import annotations

import time
from typing import Dict, Generator, List, Optional

from loguru import logger

from config import config
from search.cache import CacheBackend, make_cache_key
from search.shared.exceptions import (
    EvaluatorError,
    GeneratorError,
    RewriterError,
    SearchBackendError,
    SearcherError,
)
from search.shared.schemas import (
    EvaluatedDocument,
    PipelineTrace,
    RewrittenQuery,
    StageTimings,
)
from search.rewriter import QueryRewriter
from search.searcher import DocumentSearcher
from search.evaluator import DocumentEvaluator
from search.generator import ResponseGenerator


class RAGPipeline:
    """Complete RAG pipeline: Rewrite -> Search -> Evaluate -> Generate.

    Accepts pre-built module instances via constructor (dependency
    injection).  When not provided, creates its own instances using
    the current configuration.
    """

    def __init__(
        self,
        rewriter: Optional[QueryRewriter] = None,
        searcher: Optional[DocumentSearcher] = None,
        evaluator: Optional[DocumentEvaluator] = None,
        generator: Optional[ResponseGenerator] = None,
        response_cache: Optional[CacheBackend] = None,
        *,
        search=None,
        llm=None,
    ):
        if search is not None or llm is not None:
            logger.debug(
                "Legacy search/llm arguments detected — wrapping in new modules"
            )

        self.rewriter = rewriter or QueryRewriter(
            llm=llm,
        )
        self.searcher = searcher or DocumentSearcher(
            vector_search=search,
        )
        self.evaluator = evaluator or DocumentEvaluator()
        self.generator = generator or ResponseGenerator(
            llm=llm,
        )
        self._response_cache = response_cache
        logger.info("RAGPipeline initialized (modular)")

    def query(
        self,
        question: str,
        *,
        date: Optional[str] = None,
        limit: int = 5,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        return_sources: bool = True,
        debug: Optional[bool] = None,
        grounded_only: Optional[bool] = None,
        max_queries: Optional[int] = None,
        evaluation_threshold: Optional[int] = None,
    ) -> Dict:
        """Answer a question using the full RAG pipeline.

        New optional parameters (all backward-compatible):
            debug: Include ``PipelineTrace`` in the response.
            grounded_only: Override generator groundedness.
            max_queries: Override rewriter max sub-queries.
            evaluation_threshold: Override evaluator threshold.
        """
        enable_debug = debug if debug is not None else config.PIPELINE_DEBUG

        use_cache = (
            self._response_cache is not None
            and history is None
            and not stream
            and not enable_debug
        )
        if use_cache:
            cache_key = make_cache_key("rag", question, str(date), str(limit))
            cached = self._response_cache.get(cache_key)
            if cached is not None:
                logger.info(f"RAG cache hit for: {question[:50]}...")
                return cached

        trace = PipelineTrace(original_query=question) if enable_debug else None
        timings = StageTimings() if enable_debug else None
        pipeline_start = time.time()

        # --------------------------------------------------------
        # 1. REWRITE
        # --------------------------------------------------------
        rewrite_start = time.time()
        try:
            rewritten = self.rewriter.rewrite(
                question, max_queries=max_queries,
            )
        except RewriterError as exc:
            logger.warning(f"Rewriter failed: {exc} — using original query")
            rewritten = [RewrittenQuery(text=question, facet_type="original")]
            if trace:
                trace.errors.append(f"Rewriter: {exc}")

        if timings:
            timings.rewriter_ms = int((time.time() - rewrite_start) * 1000)
        if trace:
            trace.rewritten_queries = rewritten

        # --------------------------------------------------------
        # 2. SEARCH
        # --------------------------------------------------------
        search_start = time.time()
        try:
            search_results = self.searcher.search(
                rewritten, limit=limit, date=date,
            )
        except SearchBackendError:
            logger.error("Search backend unavailable during RAG query")
            elapsed = int((time.time() - pipeline_start) * 1000)
            result = {
                "answer": "Serviço de busca temporariamente indisponível. Tente novamente.",
                "sources": [],
                "search_time_ms": elapsed,
                "llm_time_ms": 0,
                "total_time_ms": elapsed,
            }
            if trace:
                trace.errors.append("Searcher: backend unavailable")
                result["trace"] = trace.to_dict()
            return result
        except SearcherError as exc:
            logger.error(f"Searcher failed: {exc}")
            elapsed = int((time.time() - pipeline_start) * 1000)
            result = {
                "answer": "Erro ao buscar documentos. Tente novamente.",
                "sources": [],
                "search_time_ms": elapsed,
                "llm_time_ms": 0,
                "total_time_ms": elapsed,
            }
            if trace:
                trace.errors.append(f"Searcher: {exc}")
                result["trace"] = trace.to_dict()
            return result

        search_time = time.time() - search_start
        if timings:
            timings.searcher_ms = int(search_time * 1000)

        if trace:
            trace.search_results_per_query = search_results.results_per_query
            trace.total_documents_found = search_results.total_before_dedup
            trace.documents_after_dedup = search_results.total_after_dedup

        if not search_results.documents:
            elapsed = int((time.time() - pipeline_start) * 1000)
            result = {
                "answer": "Não encontrei informações relevantes nos documentos disponíveis.",
                "sources": [],
                "search_time_ms": int(search_time * 1000),
                "llm_time_ms": 0,
                "total_time_ms": elapsed,
            }
            if trace:
                timings.total_ms = elapsed
                trace.timings = timings
                result["trace"] = trace.to_dict()
            return result

        # --------------------------------------------------------
        # 3. EVALUATE
        # --------------------------------------------------------
        eval_start = time.time()
        query_texts = [q.text for q in rewritten]

        try:
            if len(query_texts) == 1:
                evaluated = self.evaluator.evaluate(
                    search_results.documents,
                    query_texts[0],
                    threshold=evaluation_threshold,
                )
            else:
                evaluated = self.evaluator.evaluate_multi_query(
                    search_results.documents,
                    query_texts,
                    threshold=evaluation_threshold,
                )
        except EvaluatorError as exc:
            logger.warning(f"Evaluator failed: {exc} — using all search results")
            evaluated = [
                EvaluatedDocument(
                    document=doc,
                    relevance_score=50.0,
                    query_text=question,
                )
                for doc in search_results.documents
            ]
            if trace:
                trace.errors.append(f"Evaluator: {exc}")

        if timings:
            timings.evaluator_ms = int((time.time() - eval_start) * 1000)

        effective_threshold = evaluation_threshold or config.EVALUATOR_THRESHOLD
        if trace:
            trace.evaluation_threshold = effective_threshold
            trace.evaluation_scores = [
                {
                    "regulation_id": ed.document.get("regulation_id", ""),
                    "score": ed.relevance_score,
                    "accepted": True,
                }
                for ed in evaluated
            ]
            discarded_ids = {
                d.get("regulation_id") for d in search_results.documents
            } - {ed.document.get("regulation_id") for ed in evaluated}
            for doc in search_results.documents:
                if doc.get("regulation_id") in discarded_ids:
                    trace.evaluation_scores.append({
                        "regulation_id": doc.get("regulation_id", ""),
                        "score": 0,
                        "accepted": False,
                    })
            trace.documents_accepted = len(evaluated)
            trace.documents_discarded = len(search_results.documents) - len(evaluated)

        if not evaluated:
            elapsed = int((time.time() - pipeline_start) * 1000)
            result = {
                "answer": "Não encontrei informações relevantes nos documentos disponíveis.",
                "sources": [],
                "search_time_ms": int(search_time * 1000),
                "llm_time_ms": 0,
                "total_time_ms": elapsed,
            }
            if trace:
                timings.total_ms = elapsed
                trace.timings = timings
                result["trace"] = trace.to_dict()
            return result

        sources = (
            [ed.document for ed in evaluated] if return_sources else []
        )

        # --------------------------------------------------------
        # 4. GENERATE
        # --------------------------------------------------------
        gen_start = time.time()

        if trace:
            trace.generator_model = self.generator.llm.model_name
            effective_grounded = (
                grounded_only if grounded_only is not None
                else self.generator.grounded_only
            )
            trace.generator_grounded_only = effective_grounded
            trace.generator_context_length = sum(
                len(ed.document.get("text", "")) for ed in evaluated
            )

        if stream:
            result = {
                "sources": sources,
                "search_time_ms": int(search_time * 1000),
                "answer_stream": self.generator.generate(
                    evaluated,
                    question,
                    rewritten_queries=[q.text for q in rewritten],
                    history=history,
                    grounded_only=grounded_only,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                ),
            }
            if trace:
                result["trace"] = trace.to_dict()
            return result

        try:
            answer = self.generator.generate(
                evaluated,
                question,
                rewritten_queries=[q.text for q in rewritten],
                history=history,
                grounded_only=grounded_only,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except GeneratorError as exc:
            logger.error(f"Generator failed: {exc}")
            answer = "Ocorreu um erro ao gerar a resposta. Tente novamente."
            if trace:
                trace.errors.append(f"Generator: {exc}")

        llm_time = time.time() - gen_start
        total_time = time.time() - pipeline_start

        if timings:
            timings.generator_ms = int(llm_time * 1000)
            timings.total_ms = int(total_time * 1000)

        if trace:
            trace.timings = timings

        response = {
            "answer": answer,
            "sources": sources,
            "search_time_ms": int(search_time * 1000),
            "llm_time_ms": int(llm_time * 1000),
            "total_time_ms": int(total_time * 1000),
        }

        if trace:
            response["trace"] = trace.to_dict()

        if use_cache:
            self._response_cache.set(cache_key, response)

        logger.info(f"RAG query completed in {total_time:.2f}s")
        return response
