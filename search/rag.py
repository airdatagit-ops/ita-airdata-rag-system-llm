"""RAG pipeline combining search and generation.

This is the single entry point for all search+generate flows (stateless
RAG, conversational chat with RAG, and streaming).  Non-RAG chat (pure
LLM conversation) bypasses this pipeline entirely.
"""

import time
from typing import Dict, Generator, List, Optional, Union

from loguru import logger

from search.vector_search import VectorSearch
from search.cache import CacheBackend, make_cache_key
from search.exceptions import SearchBackendError
from search.prompts import (
    SYSTEM_PROMPT,
    build_context_string,
    build_rag_prompt,
    build_chat_prompt,
    build_search_query,
)
from models.llm import LlamaModel


class RAGPipeline:
    """Complete RAG pipeline: retrieve + generate.

    Accepts pre-built instances via constructor (dependency injection).
    When not provided, creates its own instances.
    """

    def __init__(
        self,
        search: Optional[VectorSearch] = None,
        llm: Optional[LlamaModel] = None,
        response_cache: Optional[CacheBackend] = None,
    ):
        self.search = search or VectorSearch()
        self.llm = llm or LlamaModel()
        self._response_cache = response_cache
        logger.info("RAGPipeline initialized")

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
    ) -> Dict:
        """Answer a question using retrieval-augmented generation.

        Handles stateless RAG, conversational RAG (with history), and
        streaming.  The caller never needs to touch ``vector_search``,
        prompts, or ``llm`` directly.

        Args:
            question: User question (the current turn).
            date: Optional ISO date for temporal filtering.
            limit: Number of context documents to retrieve.
            history: Previous conversation messages (list of
                ``{"role": ..., "content": ...}`` dicts).  When provided,
                the search query is enriched with recent user messages and
                the prompt includes conversation history.
            stream: If ``True``, the returned dict contains an
                ``answer_stream`` generator instead of a complete
                ``answer`` string.
            temperature: LLM temperature override.
            max_tokens: LLM max-tokens override.
            return_sources: Include source documents in the response.

        Returns:
            Dict with ``answer`` (or ``answer_stream``), ``sources``, and
            timing information.
        """
        use_cache = (
            self._response_cache is not None
            and history is None
            and not stream
        )
        if use_cache:
            cache_key = make_cache_key("rag", question, str(date), str(limit))
            cached = self._response_cache.get(cache_key)
            if cached is not None:
                logger.info(f"RAG cache hit for: {question[:50]}...")
                return cached

        start_time = time.time()

        # 1. Enrich query with conversation context
        search_query = (
            build_search_query(question, history)
            if history
            else question
        )

        # 2. Retrieve
        try:
            if date:
                results = self.search.search_temporal(search_query, date, limit=limit)
            else:
                results = self.search.search(search_query, limit=limit)
        except SearchBackendError:
            logger.error("Search backend unavailable during RAG query")
            return {
                "answer": "Serviço de busca temporariamente indisponível. Tente novamente.",
                "sources": [],
                "search_time_ms": int((time.time() - start_time) * 1000),
                "llm_time_ms": 0,
                "total_time_ms": int((time.time() - start_time) * 1000),
            }

        search_time = time.time() - start_time

        if not results:
            return {
                "answer": "Não encontrei informações relevantes nos documentos disponíveis.",
                "sources": [],
                "search_time_ms": int(search_time * 1000),
                "llm_time_ms": 0,
                "total_time_ms": int(search_time * 1000),
            }

        # 3. Build prompt (conversational vs stateless)
        context_str = build_context_string(results)
        if history:
            prompt = build_chat_prompt(question, context_str, history)
        else:
            prompt = build_rag_prompt(question, context_str)

        sources = results if return_sources else []

        # 4. Generate
        if stream:
            return {
                "sources": sources,
                "search_time_ms": int(search_time * 1000),
                "answer_stream": self.llm.generate(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                ),
            }

        llm_start = time.time()
        answer = self.llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        llm_time = time.time() - llm_start
        total_time = time.time() - start_time

        response = {
            "answer": answer,
            "sources": sources,
            "search_time_ms": int(search_time * 1000),
            "llm_time_ms": int(llm_time * 1000),
            "total_time_ms": int(total_time * 1000),
        }

        if use_cache:
            self._response_cache.set(cache_key, response)

        logger.info(f"RAG query completed in {total_time:.2f}s")
        return response


if __name__ == "__main__":
    rag = RAGPipeline()
    result = rag.query("O que diz sobre licitações?")
    print(f"Answer: {result['answer']}")
