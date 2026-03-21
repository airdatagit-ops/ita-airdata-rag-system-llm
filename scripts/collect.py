"""
Phase 1 – Collect (Scrape) documents from all sources into the DocumentStore.

Runs every registered scraper (LexML, DECEA, etc.) and persists documents
into the SQLite store.  Only documents whose content hash has changed are
actually written, so re-runs are fast and idempotent.

Usage:
    python -m scripts.collect                                         # full collection (all sources, no limit)
    python -m scripts.collect --sources lexml --limit 50              # only LexML, 50 docs
    python -m scripts.collect --sources pdf --pdf-dir ./data/pdfs     # local PDFs
    make collect                                                      # full collection
    make collect SOURCES=lexml LIMIT=50                               # LexML, 50 docs
    make collect SOURCES=lexml,decea,pdf                              # all sources incl. PDFs
"""

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger

from config import config
from pipeline.document_store import DocumentStore


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_doc_id(doc: Dict, source: str) -> str:
    """Derive a stable, filesystem-safe document ID."""
    if source == "lexml":
        urn = doc.get("urn", "")
        if urn:
            return re.sub(r"[^a-zA-Z0-9._-]", "_", urn)
        url = doc.get("url", "")
        if url:
            return re.sub(r"[^a-zA-Z0-9._-]", "_", url)[-100:]
    if source == "decea":
        slug = doc.get("slug", "")
        if slug:
            return f"decea_{slug}"
    title = doc.get("title", "unknown")
    return re.sub(r"[^a-zA-Z0-9._-]", "_", title)[:100]


def _extract_metadata(doc: Dict, source: str) -> Dict:
    """Pull source-specific fields into a metadata dict (everything except content)."""
    exclude = {"content"}
    return {k: v for k, v in doc.items() if k not in exclude}


# ── LexML collector ─────────────────────────────────────────────────────────

async def _collect_lexml(
    store: DocumentStore,
    limit: int,
    concurrency: int,
    keywords: str = None,
    force: bool = False,
) -> Dict[str, int]:
    from crawler.scrapers.lexml_scraper import LexMLScraper

    kw_list = keywords.split(",") if keywords else config.lexml_keywords_list
    search_limit = limit if limit > 0 else 10_000_000
    stats = {"inserted": 0, "updated": 0, "unchanged": 0, "errors": 0}

    async with LexMLScraper(skip_duplicates=False, concurrency=concurrency) as scraper:
        documents = await scraper.search(keywords=kw_list, limit=search_limit, doc_type="Legislação")
        logger.info(f"[lexml] Found {len(documents)} documents to process")

        sem = asyncio.Semaphore(concurrency)

        async def _process(doc: Dict) -> None:
            async with sem:
                try:
                    doc_id = _safe_doc_id(doc, "lexml")

                    if not force:
                        existing = store.get_document(doc_id)
                        if existing:
                            content = await scraper.get_document_text(doc, save_original=True)
                            if content:
                                new_hash = store.compute_content_hash(content)
                                if existing["content_hash"] == new_hash:
                                    stats["unchanged"] += 1
                                    return

                    content = await scraper.get_document_text(doc, save_original=True)
                    if not content or len(content.strip()) < 50:
                        logger.warning(f"[lexml] Skipping {doc.get('title', '?')[:50]}: no content")
                        stats["errors"] += 1
                        return

                    metadata = _extract_metadata(doc, "lexml")
                    action = store.upsert_document(
                        doc_id=doc_id,
                        source="lexml",
                        content=content,
                        metadata=metadata,
                        urn=doc.get("urn"),
                        url=doc.get("url"),
                        title=doc.get("title"),
                        doc_type=doc.get("doc_type"),
                    )
                    stats[action] += 1
                    if action != "unchanged":
                        logger.info(f"[lexml] {action}: {doc.get('title', doc_id)[:60]}")
                except Exception as exc:
                    logger.error(f"[lexml] Error processing {doc.get('title', '?')[:50]}: {exc}")
                    stats["errors"] += 1

        await asyncio.gather(*[_process(d) for d in documents])

    return stats


# ── DECEA collector ─────────────────────────────────────────────────────────

def _collect_decea(
    store: DocumentStore,
    limit: int,
    workers: int,
    doc_types: str = "ICA",
    keywords: str = None,
    force: bool = False,
) -> Dict[str, int]:
    from crawler.scrapers.decea_scraper import DECEAScraper

    types_list = [t.strip() for t in doc_types.split(",")]
    kw_list = [k.strip() for k in keywords.split(",")] if keywords else None
    stats = {"inserted": 0, "updated": 0, "unchanged": 0, "errors": 0}

    search_limit = limit if limit > 0 else 10_000_000
    scraper = DECEAScraper()
    documents = scraper.search(doc_types=types_list, keywords=kw_list, limit=search_limit)

    if not documents:
        logger.warning("[decea] No documents found")
        return stats

    logger.info(f"[decea] Found {len(documents)} documents to process")

    results = scraper.fetch_all(
        documents, workers=workers, extract_text=True, save_original=True,
    )

    for doc, text in results:
        try:
            content = text or doc.get("title", "")
            if not content or len(content.strip()) < 50:
                logger.warning(f"[decea] Skipping {doc.get('slug', '?')}: no content")
                stats["errors"] += 1
                continue

            doc_id = _safe_doc_id(doc, "decea")
            metadata = _extract_metadata(doc, "decea")

            action = store.upsert_document(
                doc_id=doc_id,
                source="decea",
                content=content,
                metadata=metadata,
                url=doc.get("source_url"),
                title=doc.get("title"),
                doc_type=doc.get("doc_type"),
            )
            stats[action] += 1
            if action != "unchanged":
                logger.info(f"[decea] {action}: {doc.get('slug', doc_id)}")
        except Exception as exc:
            logger.error(f"[decea] Error processing {doc.get('slug', '?')}: {exc}")
            stats["errors"] += 1

    return stats


# ── PDF (local) collector ────────────────────────────────────────────────────

def _collect_pdfs(
    store: DocumentStore,
    pdf_dir: str,
    recursive: bool = True,
    force: bool = False,
) -> Dict[str, int]:
    from parsers.pdf_parser import PDFParser

    stats = {"inserted": 0, "updated": 0, "unchanged": 0, "errors": 0}
    source_path = Path(pdf_dir)

    if not source_path.exists():
        logger.warning(f"[pdf] Directory not found: {source_path}")
        return stats

    pdf_paths = list(source_path.rglob("*.pdf") if recursive else source_path.glob("*.pdf"))
    if not pdf_paths:
        logger.warning(f"[pdf] No PDF files found in {source_path}")
        return stats

    logger.info(f"[pdf] Found {len(pdf_paths)} PDF files in {source_path}")

    parser = PDFParser()

    for pdf_path in pdf_paths:
        try:
            sections = parser.parse_pdf(str(pdf_path))
            if not sections:
                logger.warning(f"[pdf] No sections extracted from {pdf_path.name}")
                stats["errors"] += 1
                continue

            content = "\n\n".join(s.get("text", "") for s in sections)
            if len(content.strip()) < 50:
                logger.warning(f"[pdf] Skipping {pdf_path.name}: content too short")
                stats["errors"] += 1
                continue

            doc_id = f"pdf_{re.sub(r'[^a-zA-Z0-9._-]', '_', pdf_path.stem)}"

            title = sections[0].get("title") or pdf_path.stem
            metadata = {
                "filename": pdf_path.name,
                "path": str(pdf_path),
                "num_sections": len(sections),
            }

            action = store.upsert_document(
                doc_id=doc_id,
                source="pdf",
                content=content,
                metadata=metadata,
                title=title,
                doc_type="PDF",
            )
            stats[action] += 1
            if action != "unchanged":
                logger.info(f"[pdf] {action}: {pdf_path.name}")
        except Exception as exc:
            logger.error(f"[pdf] Error processing {pdf_path.name}: {exc}")
            stats["errors"] += 1

    return stats


# ── orchestrator ─────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    store = DocumentStore()
    sources = [s.strip().lower() for s in args.sources.split(",")]
    grand_stats: Dict[str, Dict[str, int]] = {}

    logger.info(f"Starting collection: sources={sources}, limit={args.limit}")

    for source in sources:
        if source == "lexml":
            stats = asyncio.run(
                _collect_lexml(
                    store,
                    limit=args.limit,
                    concurrency=args.concurrency,
                    keywords=args.keywords,
                    force=args.force,
                )
            )
            grand_stats["lexml"] = stats

        elif source == "decea":
            stats = _collect_decea(
                store,
                limit=args.limit,
                workers=args.workers,
                doc_types=args.doc_types,
                keywords=args.keywords,
                force=args.force,
            )
            grand_stats["decea"] = stats

        elif source == "pdf":
            stats = _collect_pdfs(
                store,
                pdf_dir=args.pdf_dir,
                force=args.force,
            )
            grand_stats["pdf"] = stats

        else:
            logger.warning(f"Unknown source: {source}")

    logger.info("=" * 60)
    logger.info("Collection Summary")
    logger.info("=" * 60)
    for src, st in grand_stats.items():
        logger.info(
            f"  {src}: inserted={st['inserted']}, updated={st['updated']}, "
            f"unchanged={st['unchanged']}, errors={st['errors']}"
        )
    logger.info(f"  Store totals: {store.stats()}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 – Collect documents into store")
    parser.add_argument(
        "--sources", type=str, default="lexml,decea",
        help="Comma-separated sources to collect (lexml, decea, pdf)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max documents per source (0 = unlimited)")
    parser.add_argument("--concurrency", type=int, default=5, help="Parallel downloads (LexML)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (DECEA)")
    parser.add_argument(
        "--doc-types", type=str, default="ICA,MCA,PCA,DCA,TCA,CIRCEA,NSCA,FCA",
        help="DECEA doc types (comma-separated, e.g. ICA,MCA,PCA)",
    )
    parser.add_argument("--keywords", type=str, default=None, help="Custom keywords")
    parser.add_argument(
        "--pdf-dir", type=str, default="./data/pdfs",
        help="Directory containing local PDF files (used with --sources pdf)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download even if document exists",
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
