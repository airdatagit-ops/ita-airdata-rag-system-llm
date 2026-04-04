"""Query rewriter module.

Takes a raw user query and produces one or more optimised sub-queries
using an LLM, with optional filters and sorts for downstream search.
"""

from __future__ import annotations

import json
from typing import List, Optional

from loguru import logger

from config import config
from models.llm import LlamaModel
from search.shared.exceptions import RewriterError
from search.shared.schemas import (
    RewrittenQuery,
    SearchFilter,
    SearchSort,
    FilterOperator,
    SortOrder,
)
from search.shared.timeouts import with_timeout
from search.rewriter.prompts import REWRITER_SYSTEM_PROMPT, REWRITER_USER_PROMPT


class QueryRewriter:
    """Rewrites a user query into 1..N optimised sub-queries.

    Parameters
    ----------
    llm:
        LLM instance for rewriting.  When *None*, a dedicated instance
        is created using ``REWRITER_MODEL`` (falls back to the global
        ``OLLAMA_MODEL``).
    max_queries:
        Maximum number of sub-queries to generate.
    max_query_length:
        Maximum character length per rewritten query.
    timeout:
        Hard timeout (seconds) for the LLM call.
    """

    def __init__(
        self,
        llm: Optional[LlamaModel] = None,
        max_queries: Optional[int] = None,
        max_query_length: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        model_name = config.REWRITER_MODEL or config.OLLAMA_MODEL
        self.llm = llm or LlamaModel(model_name=model_name)
        self.max_queries = max_queries or config.REWRITER_MAX_QUERIES
        self.max_query_length = max_query_length or config.REWRITER_MAX_QUERY_LENGTH
        self.timeout = timeout or config.REWRITER_TIMEOUT
        self.temperature = config.REWRITER_TEMPERATURE
        logger.info(
            f"QueryRewriter initialized (model={model_name}, "
            f"max_queries={self.max_queries}, timeout={self.timeout}s)"
        )

    def rewrite(
        self,
        query: str,
        *,
        max_queries: Optional[int] = None,
    ) -> List[RewrittenQuery]:
        """Rewrite *query* into optimised sub-queries.

        On any failure (parse error, timeout, LLM error) the original
        query is returned as-is so the pipeline can continue.
        """
        effective_max = max_queries or self.max_queries

        system = REWRITER_SYSTEM_PROMPT.format(max_queries=effective_max)
        user = REWRITER_USER_PROMPT.format(query=query)

        def _call_llm() -> str:
            return self.llm.generate(
                prompt=user,
                system_prompt=system,
                temperature=self.temperature,
            )

        try:
            raw, timed_out = with_timeout(
                _call_llm,
                timeout_seconds=self.timeout,
                fallback_value="",
                stage_name="Rewriter",
            )
        except Exception as exc:
            logger.warning(f"Rewriter LLM call failed: {exc} — falling back to original query")
            return [self._fallback(query)]

        if timed_out or not raw:
            logger.warning("Rewriter timeout/empty — falling back to original query")
            return [self._fallback(query)]

        try:
            queries = self._parse_response(raw, effective_max)
        except RewriterError as exc:
            logger.warning(f"Rewriter parse error: {exc} — falling back to original query")
            return [self._fallback(query)]

        validated = self._validate_queries(queries)
        if not validated:
            return [self._fallback(query)]

        logger.info(f"Rewriter produced {len(validated)} sub-queries from original query")
        return validated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str, max_queries: int) -> List[RewrittenQuery]:
        """Parse the LLM JSON response into ``RewrittenQuery`` objects."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RewriterError(f"Invalid JSON from LLM: {exc}") from exc

        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            raise RewriterError(f"Expected JSON array, got {type(data).__name__}")

        queries: List[RewrittenQuery] = []
        for item in data[:max_queries]:
            if not isinstance(item, dict) or "text" not in item:
                continue
            filters = self._parse_filters(item.get("filters", []))
            sorts = self._parse_sorts(item.get("sorts", []))
            queries.append(
                RewrittenQuery(
                    text=str(item["text"]),
                    filters=filters,
                    sorts=sorts,
                    facet_type=str(item.get("facet_type", "general")),
                )
            )

        if not queries:
            raise RewriterError("LLM returned no valid queries")

        return queries

    @staticmethod
    def _parse_filters(raw_filters: list) -> List[SearchFilter]:
        parsed: List[SearchFilter] = []
        for f in raw_filters:
            if not isinstance(f, dict):
                continue
            try:
                parsed.append(
                    SearchFilter(
                        field=str(f.get("field", "")),
                        operator=FilterOperator(f.get("operator", "eq")),
                        value=f.get("value"),
                    )
                )
            except (ValueError, KeyError):
                continue
        return parsed

    @staticmethod
    def _parse_sorts(raw_sorts: list) -> List[SearchSort]:
        parsed: List[SearchSort] = []
        for s in raw_sorts:
            if not isinstance(s, dict):
                continue
            try:
                parsed.append(
                    SearchSort(
                        field=str(s.get("field", "")),
                        order=SortOrder(s.get("order", "desc")),
                    )
                )
            except (ValueError, KeyError):
                continue
        return parsed

    def _validate_queries(self, queries: List[RewrittenQuery]) -> List[RewrittenQuery]:
        """Enforce length limits and sanitise."""
        valid: List[RewrittenQuery] = []
        for q in queries:
            text = q.text.strip()
            if not text:
                continue
            if len(text) > self.max_query_length:
                text = text[: self.max_query_length]
                logger.debug(f"Truncated rewritten query to {self.max_query_length} chars")
            valid.append(q.model_copy(update={"text": text}))
        return valid

    @staticmethod
    def _fallback(query: str) -> RewrittenQuery:
        return RewrittenQuery(text=query, facet_type="original")
