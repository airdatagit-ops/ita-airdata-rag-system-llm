"""Optional metadata enrichment from the SQLite document store.

When the Qdrant payload lacks fields the frontend needs (e.g. ``url``),
this module fills them in from the authoritative SQLite store using a
lightweight batch lookup.

The enrichment is *safe to skip* — the pipeline works without it.  It
only adds value when older indexed data is missing metadata that was
added to the ingestion pipeline later.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from loguru import logger


_ENRICHABLE_FIELDS = ("url", "title", "authority", "number")


def enrich_documents(
    documents: List[Dict[str, Any]],
    db_path: Optional[str] = None,
    *,
    fields: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Fill missing metadata on *documents* from the SQLite store.

    Only documents whose ``metadata`` dict is missing one of the target
    *fields* are looked up.  Returns the same list (mutated in-place).
    """
    if not documents:
        return documents

    target_fields = fields or set(_ENRICHABLE_FIELDS)

    needs_enrichment: Dict[str, List[Dict]] = {}
    for doc in documents:
        meta = doc.get("metadata") or {}
        reg_id = doc.get("regulation_id", "")
        if not reg_id:
            continue
        base_id = reg_id.split("-chunk-")[0].split("-art")[0]
        if any(not meta.get(f) for f in target_fields):
            needs_enrichment.setdefault(base_id, []).append(doc)

    if not needs_enrichment:
        return documents

    if db_path is None:
        try:
            from config import config
            db_path = config.STORE_DB_PATH
        except Exception:
            logger.debug("Enrichment skipped: config unavailable")
            return documents

    path = Path(db_path)
    if not path.exists():
        logger.debug(f"Enrichment skipped: DB not found at {path}")
        return documents

    try:
        conn = sqlite3.connect(str(path), timeout=3)
        conn.row_factory = sqlite3.Row
        try:
            placeholders = ",".join("?" * len(needs_enrichment))
            rows = conn.execute(
                f"SELECT doc_id, url, title, authority, number "
                f"FROM documents WHERE doc_id IN ({placeholders})",
                list(needs_enrichment.keys()),
            ).fetchall()

            enriched_count = 0
            for row in rows:
                doc_id = row["url"] and row["doc_id"]
                if not doc_id:
                    continue
                for doc in needs_enrichment.get(row["doc_id"], []):
                    meta = doc.setdefault("metadata", {})
                    changed = False
                    for field in target_fields:
                        if not meta.get(field) and row[field]:
                            meta[field] = row[field]
                            changed = True
                    if changed:
                        enriched_count += 1

            if enriched_count:
                logger.info(
                    f"Enrichment: filled metadata for {enriched_count} documents"
                )
        finally:
            conn.close()
    except Exception as exc:
        logger.warning(f"Enrichment failed (non-fatal): {exc}")

    return documents
