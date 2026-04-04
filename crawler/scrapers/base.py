"""
Abstract base class for all document scrapers.

Provides a unified async interface, shared utilities (save originals,
generate document IDs), and a default parallel fetch_all() using
asyncio.Semaphore + gather.

Concrete scrapers inherit from BaseScraper and implement:
  - source_name (property)
  - search(**kwargs) -> List[Dict]
  - fetch_document(doc, save_original) -> Optional[ScrapedDocument]
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import asyncio
from loguru import logger

_PROJECT_ROOT = Path(__file__).parent.parent.parent
ORIGINALS_DIR = _PROJECT_ROOT / "data" / "originals"

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def compute_canonical_id(doc_type: Optional[str], number: Optional[str]) -> Optional[str]:
    """Build a source-agnostic ID for cross-source deduplication.

    Strips the type prefix from *number* when present so that inputs from
    different sources produce the same canonical ID:

        ("ICA", "96-1")      -> "ica_96-1"   (SISLAER)
        ("ICA", "ICA96-1")   -> "ica_96-1"   (DECEA)
        ("ICA", "ICA-96-1")  -> "ica_96-1"   (DECEA slug)
        ("Lei", "8666")      -> "lei_8666"    (LexML)
    """
    if not doc_type or not number:
        return None
    dtype = re.sub(r"[^a-z0-9]", "", doc_type.lower())
    # Strip type prefix from number (DECEA passes "ICA96-1" as number)
    num_raw = number
    dtype_compact = doc_type.upper().replace(" ", "")
    num_upper = num_raw.upper().replace(" ", "")
    if num_upper.startswith(dtype_compact) and len(num_raw) > len(dtype_compact):
        num_raw = num_raw[len(dtype_compact):]
        # Remove leading dash left after stripping (ICA-96-1 -> 96-1)
        num_raw = num_raw.lstrip("-")
    num = re.sub(r"[^0-9a-z./-]", "", num_raw.lower()).strip("-./")
    return f"{dtype}_{num}" if dtype and num else None


def split_version_year(number: str | None) -> tuple[str, Optional[str]]:
    """Split a trailing ``/YYYY`` year suffix from a document number.

    Returns ``(number_without_year, year_str_or_None)``.
    Safely handles ``None`` / empty inputs.

        "96-1/2025"  -> ("96-1", "2025")
        "8666"       -> ("8666", None)
        "1082/GM3"   -> ("1082/GM3", None)  # not a 4-digit year
        None         -> ("", None)
    """
    if not number:
        return ("", None)
    if "/" in number:
        base, maybe_year = number.rsplit("/", 1)
        if len(maybe_year) == 4 and maybe_year.isdigit():
            return base, maybe_year
    return number, None


@dataclass
class ScrapedDocument:
    """Standardised output produced by every scraper."""

    doc_id: str
    source: str
    title: str
    content: str
    metadata: Dict = field(default_factory=dict)
    url: Optional[str] = None
    urn: Optional[str] = None
    doc_type: Optional[str] = None
    canonical_id: Optional[str] = None
    source_ref: Optional[str] = None
    status: Optional[str] = None
    number: Optional[str] = None
    authority: Optional[str] = None
    version_year: Optional[str] = None


class BaseScraper(ABC):
    """Async-first abstract scraper.

    Sync-only scrapers wrap blocking calls with ``asyncio.to_thread``.
    """

    # ── interface ────────────────────────────────────────────────

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique identifier for the source (e.g. ``'lexml'``, ``'decea'``)."""

    @abstractmethod
    async def search(self, *, limit: int = 100, **kwargs) -> List[Dict]:
        """Return document metadata dicts matching the given criteria."""

    @abstractmethod
    async def fetch_document(
        self, doc: Dict, save_original: bool = True
    ) -> Optional[ScrapedDocument]:
        """Fetch full content for a single document."""

    # ── default parallel fetch ──────────────────────────────────

    async def fetch_all(
        self,
        documents: List[Dict],
        concurrency: int = 10,
        save_original: bool = True,
    ) -> List[ScrapedDocument]:
        """Fetch all documents with controlled concurrency.

        Subclasses may override for custom behaviour.
        """
        n = len(documents)
        logger.info(f"[{self.source_name}] Fetching {n} documents (concurrency={concurrency})")

        sem = asyncio.Semaphore(concurrency)

        async def _fetch(idx: int, doc: Dict):
            async with sem:
                try:
                    return idx, await self.fetch_document(doc, save_original=save_original)
                except Exception as exc:
                    logger.error(f"[{self.source_name}] Error on {doc.get('title', '?')[:50]}: {exc}")
                    return idx, None

        tasks = [_fetch(i, d) for i, d in enumerate(documents)]
        raw = await asyncio.gather(*tasks)

        results: List[Optional[ScrapedDocument]] = [None] * n
        done = 0
        for idx, result in raw:
            results[idx] = result
            done += 1
            if done % 20 == 0 or done == n:
                logger.info(f"[{self.source_name}] Progress: {done}/{n}")

        return [r for r in results if r is not None]

    # ── shared helpers ──────────────────────────────────────────

    def make_doc_id(self, doc: Dict) -> str:
        """Generate a stable, filesystem-safe document ID.

        Override in subclass for source-specific logic.
        """
        title = doc.get("title", "unknown")
        return re.sub(r"[^a-zA-Z0-9._-]", "_", title)[:100]

    # ── save originals ──────────────────────────────────────────

    @staticmethod
    def save_original_file(
        content: bytes | str,
        *,
        folder_name: str,
        stem: str,
        extension: str = ".html",
        meta: Optional[Dict] = None,
    ) -> None:
        """Persist an original file (HTML or PDF) alongside its metadata JSON."""
        try:
            folder = ORIGINALS_DIR / folder_name
            folder.mkdir(parents=True, exist_ok=True)

            path = folder / f"{stem}{extension}"
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content, encoding="utf-8")

            if meta is not None:
                meta_payload = {**meta, "downloaded_at": datetime.now().isoformat()}
                (folder / f"{stem}_meta.json").write_text(
                    json.dumps(meta_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception as exc:
            logger.error(f"Error saving original ({stem}): {exc}")

    # ── async context-manager (optional) ────────────────────────

    async def __aenter__(self) -> "BaseScraper":
        return self

    async def __aexit__(self, *_) -> None:
        pass
