"""Convert pipeline SearchFilter / SearchSort schemas into native Qdrant filters.

This module isolates the Qdrant-specific translation so the rest of the
pipeline only deals with the generic ``SearchFilter`` / ``SearchSort``
schemas defined in ``search.shared.schemas``.
"""

from __future__ import annotations

from typing import List, Optional

from loguru import logger
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    DatetimeRange,
)

from search.shared.schemas import FilterOperator, SearchFilter, SearchSort

_DATETIME_FIELDS = {"effective_date", "expiry_date"}

_ALLOWED_FILTER_FIELDS = {
    "regulation_id",
    "status",
    "is_latest",
    "metadata.category",
    "metadata.source",
    "metadata.authority",
    "metadata.number",
    "effective_date",
    "expiry_date",
    "canonical_id",
    "version_year",
}


def build_qdrant_filter(
    filters: List[SearchFilter],
    base_filter: Optional[Filter] = None,
) -> Optional[Filter]:
    """Build a Qdrant ``Filter`` from a list of ``SearchFilter`` objects.

    *base_filter* (if provided) is used as starting point — new
    conditions are appended to its ``must`` list.
    """
    if base_filter is not None:
        result = base_filter.model_copy(deep=True)
    else:
        result = Filter(must=[])

    for sf in filters:
        if sf.field not in _ALLOWED_FILTER_FIELDS:
            logger.debug(f"Ignoring unsupported filter field: {sf.field}")
            continue

        condition = _to_field_condition(sf)
        if condition is not None:
            result.must.append(condition)

    if not result.must and not getattr(result, "should", None):
        return None

    return result


def _to_field_condition(sf: SearchFilter) -> Optional[FieldCondition]:
    """Convert a single ``SearchFilter`` into a Qdrant ``FieldCondition``."""
    try:
        if sf.field in _DATETIME_FIELDS:
            return _datetime_condition(sf)
        return _scalar_condition(sf)
    except Exception as exc:
        logger.warning(f"Failed to convert filter {sf}: {exc}")
        return None


def _scalar_condition(sf: SearchFilter) -> Optional[FieldCondition]:
    op = sf.operator

    if op == FilterOperator.EQ:
        return FieldCondition(key=sf.field, match=MatchValue(value=sf.value))

    if op == FilterOperator.IN:
        if not isinstance(sf.value, list):
            return None
        return FieldCondition(key=sf.field, match=MatchValue(value=sf.value))

    range_kwargs = {}
    if op == FilterOperator.GT:
        range_kwargs["gt"] = sf.value
    elif op == FilterOperator.GTE:
        range_kwargs["gte"] = sf.value
    elif op == FilterOperator.LT:
        range_kwargs["lt"] = sf.value
    elif op == FilterOperator.LTE:
        range_kwargs["lte"] = sf.value
    elif op == FilterOperator.RANGE and isinstance(sf.value, dict):
        range_kwargs = {k: v for k, v in sf.value.items() if k in ("gt", "gte", "lt", "lte")}

    if range_kwargs:
        return FieldCondition(key=sf.field, range=Range(**range_kwargs))

    return None


def _datetime_condition(sf: SearchFilter) -> Optional[FieldCondition]:
    op = sf.operator
    val = sf.value

    range_kwargs = {}
    if op == FilterOperator.GTE:
        range_kwargs["gte"] = val
    elif op == FilterOperator.LTE:
        range_kwargs["lte"] = val
    elif op == FilterOperator.GT:
        range_kwargs["gt"] = val
    elif op == FilterOperator.LT:
        range_kwargs["lt"] = val
    elif op == FilterOperator.EQ:
        range_kwargs["gte"] = val
        range_kwargs["lte"] = val
    elif op == FilterOperator.RANGE and isinstance(val, dict):
        range_kwargs = {k: v for k, v in val.items() if k in ("gt", "gte", "lt", "lte")}

    if range_kwargs:
        return FieldCondition(key=sf.field, range=DatetimeRange(**range_kwargs))

    return None


def sort_documents(
    documents: list[dict],
    sorts: List[SearchSort],
) -> list[dict]:
    """In-memory sort of document dicts.

    Qdrant does not support arbitrary payload sorting on query results,
    so we apply sorts after retrieval.
    """
    for s in reversed(sorts):
        reverse = s.order.value == "desc"
        documents = sorted(
            documents,
            key=lambda d: d.get(s.field) or d.get("metadata", {}).get(s.field, ""),
            reverse=reverse,
        )
    return documents
