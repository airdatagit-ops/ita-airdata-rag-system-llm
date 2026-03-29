"""
Phase 1 -- Collect (Scrape) documents from all sources into the DocumentStore.

Uses the scraper registry so adding a new source requires only a new
BaseScraper subclass decorated with ``@register_scraper``.

Three collection modes control how re-runs behave:

  (default)   Skip documents that already exist in the store.
  --check     Re-download every document and recalculate its content hash.
  --force     Delete all documents of the requested source(s), then re-collect.

Usage:
    python -m scripts.collect
    python -m scripts.collect --sources lexml --limit 50
    python -m scripts.collect --sources pdf --pdf-dir ./data/pdfs
    python -m scripts.collect --force
"""

import argparse
import asyncio
from typing import Dict, List

from loguru import logger

from config import config
from crawler.scrapers import get_scraper, list_scrapers
from crawler.scrapers.base import BaseScraper, ScrapedDocument, compute_canonical_id
from parsers.temporal_extractor import TemporalExtractor
from pipeline.document_store import DocumentStore

_temporal_extractor = TemporalExtractor()


# -- helpers -----------------------------------------------------------------

def _extract_temporal(content: str, publication_date: str = None) -> Dict:
    temporal = _temporal_extractor.extract_dates(content, publication_date=publication_date)
    return {
        "effective_date": temporal.get("effective_date"),
        "expiry_date": temporal.get("expiry_date"),
        "status": "revoked" if temporal.get("is_revoked") else "active",
    }


def _publication_date(doc: ScrapedDocument) -> str | None:
    """Best-effort publication date from metadata (handles all sources).

    Note: raw ``publicacao`` is excluded because it can contain free text
    (e.g. "PUB BCA de 20/11/2019 página 016749").  The SISLAER scraper
    already extracts the date from it into ``publication_date``.
    """
    meta = doc.metadata or {}
    return (
        meta.get("date")
        or meta.get("publication_date")
        or meta.get("date_published")
        or meta.get("ato_publicacao")
        or meta.get("portaria_aprovacao")
    )


# -- generic collector -------------------------------------------------------

async def _collect_source(
    scraper: BaseScraper,
    store: DocumentStore,
    *,
    limit: int,
    concurrency: int,
    check: bool,
    force: bool,
    search_kwargs: Dict,
) -> Dict[str, int]:
    """Collect documents from a single scraper into the store."""
    source = scraper.source_name
    stats = {"inserted": 0, "updated": 0, "unchanged": 0, "errors": 0}

    if force:
        store.delete_by_source(source)

    # -- search
    async with scraper:
        search_limit = limit if limit > 0 else 10_000_000
        documents = await scraper.search(limit=search_limit, **search_kwargs)

        if not documents:
            logger.warning(f"[{source}] No documents found")
            return stats

        # -- skip existing (default mode)
        if not check and not force:
            before = len(documents)
            documents = [
                d for d in documents
                if not store.exists(scraper.make_doc_id(d))
            ]
            skipped = before - len(documents)
            if skipped:
                stats["unchanged"] = skipped
                logger.info(f"[{source}] Skipping {skipped} existing, {len(documents)} new")

        if not documents:
            logger.info(f"[{source}] All documents already collected")
            return stats

        # -- fetch in parallel
        results: List[ScrapedDocument] = await scraper.fetch_all(
            documents, concurrency=concurrency, save_original=True,
        )

    # -- persist
    for doc in results:
        try:
            temporal = _extract_temporal(doc.content, _publication_date(doc))
            # Scraper-provided status (e.g. SISLAER situacao) is authoritative
            if doc.status:
                temporal["status"] = doc.status

            number = doc.number or (doc.metadata or {}).get("number")
            authority = doc.authority or (doc.metadata or {}).get("authority")
            canonical = doc.canonical_id or compute_canonical_id(doc.doc_type, number)

            action = store.upsert_document(
                doc_id=doc.doc_id,
                source=doc.source,
                content=doc.content,
                metadata=doc.metadata,
                urn=doc.urn,
                url=doc.url,
                title=doc.title,
                doc_type=doc.doc_type,
                number=number,
                authority=authority,
                canonical_id=canonical,
                version_year=doc.version_year,
                **temporal,
            )
            stats[action] += 1
            if action != "unchanged":
                logger.info(f"[{source}] {action}: {doc.title[:60]}")

            for rel in (doc.metadata or {}).get("relations", []):
                store.upsert_relation(
                    source_doc_id=doc.doc_id,
                    target_ref=rel["target_ref"],
                    relation_type=rel["type"],
                )
        except Exception as exc:
            logger.error(f"[{source}] Error persisting {doc.doc_id}: {exc}")
            stats["errors"] += 1

    return stats


# -- LexML-specific: multi-keyword dedup search wrapper ----------------------

async def _collect_lexml(
    store: DocumentStore,
    *,
    limit: int,
    concurrency: int,
    keywords: str | None,
    check: bool,
    force: bool,
    federal_only: bool = True,
) -> Dict[str, int]:
    """LexML needs special handling: search per-keyword with dedup."""
    from crawler.scrapers.lexml_scraper import LexMLScraper

    kw_list = keywords.split(",") if keywords else config.lexml_keywords_list
    per_kw_limit = limit if limit > 0 else 10_000_000
    stats = {"inserted": 0, "updated": 0, "unchanged": 0, "errors": 0}

    if force:
        store.delete_by_source("lexml")

    scraper = LexMLScraper(skip_duplicates=False, concurrency=concurrency)
    async with scraper:
        seen_urns: set = set()
        documents: List[Dict] = []

        for kw in kw_list:
            kw_docs = await scraper.search(
                keywords=[kw], limit=per_kw_limit,
                doc_type="Legislação", federal_only=federal_only,
            )
            new = 0
            for doc in kw_docs:
                urn = doc.get("urn") or doc.get("url") or doc.get("title")
                if urn not in seen_urns:
                    seen_urns.add(urn)
                    documents.append(doc)
                    new += 1
            logger.info(f"[lexml] keyword '{kw}': {len(kw_docs)} found, {new} new (dedup)")

        logger.info(f"[lexml] Total unique documents to process: {len(documents)}")

        if not check and not force:
            before = len(documents)
            documents = [
                d for d in documents
                if not store.exists(scraper.make_doc_id(d))
            ]
            skipped = before - len(documents)
            if skipped:
                stats["unchanged"] = skipped
                logger.info(f"[lexml] Skipping {skipped} existing, {len(documents)} new")

        if not documents:
            logger.info("[lexml] All documents already collected")
            return stats

        results = await scraper.fetch_all(
            documents, concurrency=concurrency, save_original=True,
        )

    for doc in results:
        try:
            pub_date = _publication_date(doc)
            temporal = _extract_temporal(doc.content, pub_date)
            if doc.status:
                temporal["status"] = doc.status

            number = doc.number or (doc.metadata or {}).get("number")
            authority = doc.authority or (doc.metadata or {}).get("authority")
            canonical = doc.canonical_id or compute_canonical_id(doc.doc_type, number)

            action = store.upsert_document(
                doc_id=doc.doc_id,
                source=doc.source,
                content=doc.content,
                metadata=doc.metadata,
                urn=doc.urn,
                url=doc.url,
                title=doc.title,
                doc_type=doc.doc_type,
                number=number,
                authority=authority,
                canonical_id=canonical,
                version_year=doc.version_year,
                **temporal,
            )
            stats[action] += 1
            if action != "unchanged":
                logger.info(f"[lexml] {action}: {doc.title[:60]}")
        except Exception as exc:
            logger.error(f"[lexml] Error persisting {doc.doc_id}: {exc}")
            stats["errors"] += 1

    return stats


# -- orchestrator ------------------------------------------------------------

async def _run_async(args: argparse.Namespace) -> int:
    store = DocumentStore()
    sources = [s.strip().lower() for s in args.sources.split(",")]
    grand_stats: Dict[str, Dict[str, int]] = {}

    mode = "force" if args.force else ("check" if args.check else "default")
    logger.info(f"Starting collection: sources={sources}, limit={args.limit}, mode={mode}")

    for source in sources:
        if source == "lexml":
            stats = await _collect_lexml(
                store,
                limit=args.limit,
                concurrency=args.concurrency,
                keywords=args.keywords,
                check=args.check,
                force=args.force,
                federal_only=args.federal_only,
            )
        elif source in list_scrapers():
            search_kwargs: Dict = {}
            if source == "decea":
                types_list = [t.strip() for t in args.doc_types.split(",")]
                kw_list = [k.strip() for k in args.keywords.split(",")] if args.keywords else None
                search_kwargs = {"doc_types": types_list, "keywords": kw_list}
            elif source == "sislaer":
                types_list = [t.strip() for t in args.doc_types.split(",")]
                search_kwargs = {"doc_types": types_list}
            elif source == "pdf":
                search_kwargs = {"pdf_dir": args.pdf_dir}

            scraper = get_scraper(source, **_scraper_init_kwargs(source, args))
            stats = await _collect_source(
                scraper, store,
                limit=args.limit,
                concurrency=args.concurrency,
                check=args.check,
                force=args.force,
                search_kwargs=search_kwargs,
            )
        else:
            logger.warning(f"Unknown source: {source}. Available: {list_scrapers()}")
            continue

        grand_stats[source] = stats

    resolved = store.resolve_relations()
    if resolved:
        logger.info(f"Post-collection: resolved {resolved} document relations")

    store.compute_latest_versions()

    logger.info("=" * 60)
    logger.info("Collection Summary")
    logger.info("=" * 60)
    for src, st in grand_stats.items():
        logger.info(
            f"  {src}: inserted={st['inserted']}, updated={st['updated']}, "
            f"unchanged={st['unchanged']}, errors={st['errors']}"
        )
    logger.info(f"  Store totals: {store.stats()}")
    store.close()
    return 0


def _scraper_init_kwargs(source: str, args: argparse.Namespace) -> Dict:
    """Build constructor kwargs for a given scraper."""
    if source == "decea":
        return {"timeout": 30}
    if source == "pdf":
        return {"pdf_dir": args.pdf_dir}
    return {}


def run(args: argparse.Namespace) -> int:
    return asyncio.run(_run_async(args))


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 -- Collect documents into store")
    parser.add_argument(
        "--sources", type=str, default="sislaer,lexml",
        help="Comma-separated sources to collect (sislaer, lexml, decea, pdf)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max documents per source (0 = unlimited)")
    parser.add_argument("--concurrency", type=int, default=10, help="Parallel downloads")
    parser.add_argument(
        "--doc-types", type=str, default=config.SISLAER_DOC_TYPES,
        help="Document type filter (comma-separated, used by SISLAER and DECEA)",
    )
    parser.add_argument("--keywords", type=str, default=None, help="Custom keywords")
    parser.add_argument(
        "--federal-only", action=argparse.BooleanOptionalAction, default=True,
        help="Restrict LexML collection to federal legislation only (default: True)",
    )
    parser.add_argument(
        "--pdf-dir", type=str, default="./data/pdfs",
        help="Directory containing local PDF files (used with --sources pdf)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check", action="store_true",
        help="Re-download and verify content hash for all documents",
    )
    group.add_argument(
        "--force", action="store_true",
        help="Delete source documents from store and re-collect from scratch",
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
