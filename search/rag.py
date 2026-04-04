"""RAG pipeline combining search and generation."""

import time
from typing import Dict, List, Optional

from loguru import logger

from search.vector_search import VectorSearch
from search.exceptions import SearchBackendError
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
    ):
        self.search = search or VectorSearch()
        self.llm = llm or LlamaModel()
        logger.info("RAGPipeline initialized")

    def query(
        self,
        question: str,
        date: Optional[str] = None,
        limit: int = 5,
        return_sources: bool = True
    ) -> Dict:
        """
        Answer question using RAG.

        Args:
            question: User question
            date: Optional date for temporal filtering
            limit: Number of context documents
            return_sources: Include source documents in response

        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()

        try:
            if date:
                results = self.search.search_temporal(question, date, limit=limit)
            else:
                results = self.search.search(question, limit=limit)
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
                "total_time_ms": int(search_time * 1000)
            }

        llm_start = time.time()
        answer = self.llm.generate_with_context(
            query=question,
            context_documents=results
        )
        llm_time = time.time() - llm_start

        total_time = time.time() - start_time

        response = {
            "answer": answer,
            "sources": results if return_sources else [],
            "search_time_ms": int(search_time * 1000),
            "llm_time_ms": int(llm_time * 1000),
            "total_time_ms": int(total_time * 1000)
        }

        logger.info(f"RAG query completed in {total_time:.2f}s")
        return response


if __name__ == "__main__":
    rag = RAGPipeline()
    result = rag.query("O que diz sobre licitações?")
    print(f"Answer: {result['answer']}")
