"""Document searcher module.

Takes rewritten queries from the Rewriter and executes vector searches
in parallel, merging and de-duplicating results.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from loguru import logger
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import config
from search.shared.exceptions import SearcherError, SearchBackendError
from search.shared.schemas import RewrittenQuery, SearchResults
from search.vector_search import VectorSearch
from search.searcher.filters import build_qdrant_filter, sort_documents


class DocumentSearcher:
    """Executes searches for a list of rewritten queries.

    Parameters
    ----------
    vector_search:
        Pre-built ``VectorSearch`` instance (injected).
    default_limit:
        Per-query result limit.
    """

    def __init__(
        self,
        vector_search: Optional[VectorSearch] = None,
        default_limit: Optional[int] = None,
    ):
        self.vector_search = vector_search or VectorSearch()
        self.default_limit = default_limit or config.SEARCH_TOP_K
        logger.info(f"DocumentSearcher initialized (limit={self.default_limit})")

    def search(
        self,
        queries: List[RewrittenQuery],
        *,
        limit: Optional[int] = None,
        date: Optional[str] = None,
        capture_per_query: bool = False,
    ) -> SearchResults:
        """Search for all *queries* in parallel and return merged results.

        When any sub-query carries a ``sort`` directive (e.g. temporal
        sort), the per-sub-query Qdrant fetch is widened by
        ``SEARCH_SORT_FETCH_MULTIPLIER`` so that the in-memory sort has
        a chance to reach genuinely older/newer chunks that fell outside
        the relevance top-K. The deduplicated, sorted list is then
        capped back to ``limit * len(queries)`` to keep the downstream
        evaluator load constant.

        Raises ``SearcherError`` if every single sub-query fails.
        """
        effective_limit = limit or self.default_limit
        has_sort = any(q.sorts for q in queries)
        fetch_limit = (
            effective_limit * config.SEARCH_SORT_FETCH_MULTIPLIER
            if has_sort
            else effective_limit
        )

        if len(queries) == 1:
            results_map = {
                queries[0].text: self._execute_single(queries[0], fetch_limit, date)
            }
        else:
            results_map = self._execute_parallel(queries, fetch_limit, date)

        all_docs: List[Dict] = []
        results_per_query: Dict[str, int] = {}

        for query_text, docs in results_map.items():
            results_per_query[query_text] = len(docs)
            all_docs.extend(docs)

        total_before = len(all_docs)
        deduped = self._deduplicate(all_docs)

        if has_sort:
            for q in queries:
                if q.sorts:
                    deduped = sort_documents(deduped, q.sorts)
                    cap = effective_limit * len(queries)
                    if len(deduped) > cap:
                        deduped = deduped[:cap]
                    break

        documents_per_query: Dict[str, List[Dict]] = (
            {q: list(docs) for q, docs in results_map.items()}
            if capture_per_query
            else {}
        )

        return SearchResults(
            documents=deduped,
            results_per_query=results_per_query,
            documents_per_query=documents_per_query,
            total_before_dedup=total_before,
            total_after_dedup=len(deduped),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _execute_single(
        self, query: RewrittenQuery, limit: int, date: Optional[str],
    ) -> List[Dict]:
        """Execute a single sub-query."""
        try:
            qdrant_filter = self._build_filter(query)

            if date:
                return self.vector_search.search_temporal(
                    query.text, date, limit=limit,
                )
            return self.vector_search.search(
                query.text, limit=limit, filters=qdrant_filter,
            )
        except SearchBackendError:
            raise
        except Exception as exc:
            logger.error(f"Search failed for query '{query.text[:50]}': {exc}")
            return []

    def _execute_parallel(
        self,
        queries: List[RewrittenQuery],
        limit: int,
        date: Optional[str],
    ) -> Dict[str, List[Dict]]:
        """Execute multiple sub-queries in parallel."""
        results: Dict[str, List[Dict]] = {}
        errors: List[str] = []

        with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as pool:
            future_to_query = {
                pool.submit(self._execute_single, q, limit, date): q
                for q in queries
            }

            for future in as_completed(future_to_query):
                q = future_to_query[future]
                try:
                    results[q.text] = future.result()
                except SearchBackendError:
                    raise
                except Exception as exc:
                    logger.error(f"Parallel search error for '{q.text[:50]}': {exc}")
                    errors.append(str(exc))
                    results[q.text] = []

        if all(len(docs) == 0 for docs in results.values()) and errors:
            raise SearcherError(
                f"All sub-queries failed: {'; '.join(errors)}"
            )

        return results

    @staticmethod
    def _build_filter(query: RewrittenQuery) -> Optional[Filter]:
        """Build Qdrant filter from rewritten-query filters, merged with
        the standard active/latest base filter."""
        base = Filter(must=[
            FieldCondition(key="status", match=MatchValue(value="active")),
            FieldCondition(key="is_latest", match=MatchValue(value=True)),
        ])

        if not query.filters:
            return base

        return build_qdrant_filter(query.filters, base_filter=base)

    @staticmethod
    def _deduplicate(documents: List[Dict]) -> List[Dict]:
        """Remove duplicate documents, keeping the one with the highest score."""
        seen: Dict[str, Dict] = {}

        for doc in documents:
            key = doc.get("regulation_id", "")
            text_snippet = (doc.get("text") or "")[:100]
            dedup_key = f"{key}:{text_snippet}"

            if dedup_key not in seen:
                seen[dedup_key] = doc
            else:
                existing_score = seen[dedup_key].get("score", 0) or 0
                new_score = doc.get("score", 0) or 0
                if new_score > existing_score:
                    seen[dedup_key] = doc

        return list(seen.values())
