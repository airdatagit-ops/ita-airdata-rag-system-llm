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
    ALLOWED_FILTER_FIELDS,
    FilterOperator,
    RewrittenQuery,
    SearchFilter,
    SearchSort,
    SortOrder,
    filter_registry,
)
from search.shared.timeouts import with_timeout
from search.rewriter.prompts import build_system_prompt, build_user_prompt


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

        system = build_system_prompt(effective_max)
        user = build_user_prompt(query)

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

    @staticmethod
    def _extract_json(raw: str) -> str:
        """Extract the first JSON array or object from LLM output.

        Handles trailing text, markdown fences, and other noise the
        small rewriter model may produce after the JSON payload.
        """
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        start = -1
        bracket = None
        for i, ch in enumerate(text):
            if ch in ("[", "{"):
                start = i
                bracket = ch
                break

        if start == -1:
            return text

        closing = "]" if bracket == "[" else "}"
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == bracket:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

        return text[start:]

    def _parse_response(self, raw: str, max_queries: int) -> List[RewrittenQuery]:
        """Parse the LLM JSON response into ``RewrittenQuery`` objects."""
        cleaned = self._extract_json(raw)

        try:
            data = json.loads(cleaned)
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
        accepted_types = filter_registry.type_values
        accepted_authorities = filter_registry.authority_values

        parsed: List[SearchFilter] = []
        for f in raw_filters:
            if not isinstance(f, dict):
                continue
            field = str(f.get("field", ""))
            if field not in ALLOWED_FILTER_FIELDS:
                logger.debug(f"Rewriter: discarding unknown filter field '{field}'")
                continue
            value = f.get("value")
            if value is None or value == "":
                continue
            if field == "metadata.type" and str(value) not in accepted_types:
                logger.debug(f"Rewriter: discarding invalid type value '{value}'")
                continue
            if field == "metadata.authority" and str(value) not in accepted_authorities:
                logger.debug(f"Rewriter: discarding invalid authority value '{value}'")
                continue
            try:
                parsed.append(
                    SearchFilter(
                        field=field,
                        operator=FilterOperator(f.get("operator", "eq")),
                        value=value,
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
        """Enforce length limits, discard empty/broken queries, deduplicate."""
        valid: List[RewrittenQuery] = []
        seen_texts: set[str] = set()
        for q in queries:
            text = q.text.strip()
            if not text or len(text) < 3:
                logger.debug("Rewriter: discarding empty/tiny query")
                continue
            if len(text) > self.max_query_length:
                text = text[: self.max_query_length]
                logger.debug(f"Truncated rewritten query to {self.max_query_length} chars")
            norm = text.lower()
            if norm in seen_texts:
                logger.debug(f"Rewriter: discarding duplicate query '{text[:50]}'")
                continue
            seen_texts.add(norm)
            valid.append(q.model_copy(update={"text": text}))
        return valid

    @staticmethod
    def _fallback(query: str) -> RewrittenQuery:
        return RewrittenQuery(text=query, facet_type="original")
