"""RAG pipeline orchestrator.

Chains Rewriter -> Searcher -> Evaluator -> Generator with per-stage
error handling, timeouts, and an optional debug trace.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

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
from search.searcher.enrichment import enrich_documents
from search.evaluator import DocumentEvaluator
from search.generator import ResponseGenerator
from models.gpu_client import create_evaluator


class RAGPipeline:
    """Complete RAG pipeline: Rewrite -> Search -> Evaluate -> Generate.

    Accepts pre-built module instances via constructor (dependency
    injection).  When not provided, creates its own instances using
    the current configuration.

    The Rewriter and Evaluator can be disabled via config flags
    (``REWRITER_ENABLED``, ``EVALUATOR_ENABLED``) for CPU-constrained
    environments.  When disabled, the original query is passed straight
    to the Searcher and search results go directly to the Generator.
    """

    def __init__(
        self,
        rewriter: Optional[QueryRewriter] = None,
        searcher: Optional[DocumentSearcher] = None,
        evaluator: Optional[DocumentEvaluator] = None,
        generator: Optional[ResponseGenerator] = None,
        response_cache: Optional[CacheBackend] = None,
        *,
        rewriter_enabled: Optional[bool] = None,
        evaluator_enabled: Optional[bool] = None,
        search=None,
        llm=None,
    ):
        if search is not None or llm is not None:
            logger.debug(
                "Legacy search/llm arguments detected — wrapping in new modules"
            )

        self.rewriter_enabled = (
            rewriter_enabled if rewriter_enabled is not None
            else config.REWRITER_ENABLED
        )
        self.evaluator_enabled = (
            evaluator_enabled if evaluator_enabled is not None
            else config.EVALUATOR_ENABLED
        )

        self.rewriter = rewriter if self.rewriter_enabled else None
        if self.rewriter_enabled and rewriter is None:
            self.rewriter = QueryRewriter(llm=llm)

        self.searcher = searcher or DocumentSearcher(
            vector_search=search,
        )

        self.evaluator = evaluator if self.evaluator_enabled else None
        if self.evaluator_enabled and evaluator is None:
            self.evaluator = create_evaluator()

        self.generator = generator or ResponseGenerator(
            llm=llm,
        )
        self._response_cache = response_cache

        stages = ["Searcher", "Generator"]
        if self.rewriter_enabled:
            stages.insert(0, "Rewriter")
        if self.evaluator_enabled:
            stages.insert(-1, "Evaluator")
        logger.info(f"RAGPipeline initialized — active stages: {' → '.join(stages)}")

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
        include_generation: bool = True,
    ) -> Dict:
        """Answer a question using the full RAG pipeline.

        New optional parameters (all backward-compatible):
            debug: Include ``PipelineTrace`` in the response.
            grounded_only: Override generator groundedness.
            max_queries: Override rewriter max sub-queries.
            evaluation_threshold: Override evaluator threshold.
            include_generation: When False, return after the
                Evaluator stage (no LLM generation). Used by
                offline retrieval evaluators to measure the docs
                the Generator would actually receive without
                paying the generation cost.
        """
        enable_debug = debug if debug is not None else config.PIPELINE_DEBUG

        use_cache = (
            self._response_cache is not None
            and history is None
            and not stream
            and not enable_debug
            and include_generation
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
        if self.rewriter_enabled and self.rewriter is not None:
            try:
                rewritten = self.rewriter.rewrite(
                    question, max_queries=max_queries,
                )
            except RewriterError as exc:
                logger.warning(f"Rewriter failed: {exc} — using original query")
                rewritten = [RewrittenQuery(text=question, facet_type="original")]
                if trace:
                    trace.errors.append(f"Rewriter: {exc}")
        else:
            rewritten = [RewrittenQuery(text=question, facet_type="passthrough")]

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
                capture_per_query=trace is not None,
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
            trace.search_documents_per_query = {
                qtext: [
                    {
                        "regulation_id": doc.get("regulation_id", ""),
                        "url": (doc.get("metadata") or {}).get("url", ""),
                        "score": doc.get("score", 0.0),
                        "type": (doc.get("metadata") or {}).get("type", ""),
                        "number": (doc.get("metadata") or {}).get("number", ""),
                        "title": (doc.get("metadata") or {}).get("title", ""),
                    }
                    for doc in docs
                ]
                for qtext, docs in search_results.documents_per_query.items()
            }

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

        enrich_documents(search_results.documents, fields={"url", "title"})

        # --------------------------------------------------------
        # 3. EVALUATE
        # --------------------------------------------------------
        eval_start = time.time()

        if self.evaluator_enabled and self.evaluator is not None:
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
        else:
            evaluated = [
                EvaluatedDocument(
                    document=doc,
                    relevance_score=doc.get("score", 0) * 100,
                    query_text=question,
                )
                for doc in search_results.documents
            ]

        if timings:
            timings.evaluator_ms = int((time.time() - eval_start) * 1000)

        effective_threshold = evaluation_threshold or config.EVALUATOR_THRESHOLD
        if trace:
            trace.evaluation_threshold = effective_threshold if self.evaluator_enabled else 0
            trace.evaluation_scores = [
                {
                    "regulation_id": ed.document.get("regulation_id", ""),
                    "score": ed.relevance_score,
                    "accepted": True,
                    "url": (ed.document.get("metadata") or {}).get("url", ""),
                }
                for ed in evaluated
            ]
            if self.evaluator_enabled:
                discarded_ids = {
                    d.get("regulation_id") for d in search_results.documents
                } - {ed.document.get("regulation_id") for ed in evaluated}
                for doc in search_results.documents:
                    if doc.get("regulation_id") in discarded_ids:
                        trace.evaluation_scores.append({
                            "regulation_id": doc.get("regulation_id", ""),
                            "score": 0,
                            "accepted": False,
                            "url": (doc.get("metadata") or {}).get("url", ""),
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

        if not include_generation:
            elapsed = int((time.time() - pipeline_start) * 1000)
            if timings:
                timings.total_ms = elapsed
                trace.timings = timings
            response = {
                "answer": "",
                "sources": sources,
                "search_time_ms": int(search_time * 1000),
                "llm_time_ms": 0,
                "total_time_ms": elapsed,
            }
            if trace:
                response["trace"] = trace.to_dict()
            return response

        # --------------------------------------------------------
        # 4. GENERATE
        # --------------------------------------------------------
        gen_start = time.time()

        effective_grounded = (
            grounded_only if grounded_only is not None
            else self.generator.grounded_only
        )

        if trace:
            trace.generator_model = self.generator.llm.model_name
            trace.generator_grounded_only = effective_grounded
            from search.generator.prompts import build_generator_context
            gen_docs = self.generator.select_top_docs(evaluated)
            documents_for_ctx = [ed.document for ed in gen_docs]
            ctx = build_generator_context(documents_for_ctx)
            trace.generator_context_length = len(ctx)
            doc_char_limit = config.GENERATOR_MAX_DOC_CHARS
            trace.generator_documents = []
            for ed in gen_docs:
                doc = ed.document
                raw_text = doc.get("text", "") or ""
                sent_text = raw_text[:doc_char_limit] if doc_char_limit > 0 else raw_text
                trace.generator_documents.append({
                    "regulation_id": doc.get("regulation_id", ""),
                    "type": (doc.get("metadata") or {}).get("type", ""),
                    "number": (doc.get("metadata") or {}).get("number", ""),
                    "title": (doc.get("metadata") or {}).get("title", ""),
                    "score": ed.relevance_score,
                    "text": sent_text,
                    "char_count": len(sent_text),
                    "truncated": doc_char_limit > 0 and len(raw_text) > doc_char_limit,
                })

        if trace and timings:
            timings.total_ms = int((time.time() - pipeline_start) * 1000)
            trace.timings = timings

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
                result["_gen_start"] = time.time()
                result["_pipeline_start"] = pipeline_start
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
