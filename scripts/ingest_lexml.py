"""Script to ingest LexML documents via async web scraping."""

import argparse
import asyncio
import json
from pathlib import Path

from loguru import logger

from config import config
from crawler.scrapers.lexml_scraper import LexMLScraper
from pipeline.ingestion import IngestionPipeline

_DEFAULT_CONCURRENCY = 5


async def _download_one(
    scraper: LexMLScraper,
    doc: dict,
    doc_index: int,
    download_dir: Path,
) -> tuple[Path, str, dict] | None:
    """Download a single document. Returns (path, content, doc) or None on failure."""
    urn = doc.get("urn", "")
    if urn:
        safe_urn = urn.replace(":", "_").replace(";", "_").replace("/", "_")[-60:]
        json_path = download_dir / f"{safe_urn}.json"
    else:
        json_path = download_dir / f"doc_{doc_index}.json"

    content = await scraper.get_document_text(doc)
    if content:
        data = {**doc, "content": content, "doc_index": doc_index}
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return json_path, content, doc

    if await scraper.download_document(doc, str(json_path)):
        return json_path, None, doc

    return None


async def run(args: argparse.Namespace) -> int:
    keywords = args.keywords.split(",") if args.keywords else config.lexml_keywords_list
    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    json_paths: list[str] = []

    if not args.skip_download:
        async with LexMLScraper(skip_duplicates=not args.force_download) as scraper:
            documents = await scraper.search(
                keywords=keywords, limit=args.limit, doc_type="Legislação"
            )
            logger.info(f"Found {len(documents)} documents")

            stats = scraper.get_tracker_stats()
            if stats:
                logger.info(f"Tracker: {stats.get('total_documents', 0)} already tracked")

            # Filter known duplicates before dispatching parallel downloads
            to_download = []
            skipped = 0
            for i, doc in enumerate(documents):
                if not doc.get("url") and not doc.get("urn"):
                    logger.warning(f"Skipping doc {i}: no URL or URN")
                    continue
                if scraper.is_duplicate(doc):
                    skipped += 1
                    logger.debug(f"Duplicate: {doc.get('title', 'N/A')[:50]}")
                    continue
                to_download.append((i, doc))

            logger.info(f"Downloading {len(to_download)} documents, skipping {skipped} duplicates")

            sem = asyncio.Semaphore(args.concurrency)

            async def _bounded(i: int, doc: dict):
                async with sem:
                    return await _download_one(scraper, doc, i, download_dir)

            results = await asyncio.gather(
                *[_bounded(i, doc) for i, doc in to_download],
                return_exceptions=True,
            )

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Download error: {result}")
                    continue
                if result is None:
                    continue
                json_path, content, doc = result
                scraper.register_document(doc, content, str(json_path))
                json_paths.append(str(json_path))
                logger.info(f"Saved: {json_path.name}")

        logger.info(f"Downloaded {len(json_paths)} new documents, skipped {skipped} duplicates")
    else:
        json_paths = [str(p) for p in download_dir.glob("*.json")]
        logger.info(f"Found {len(json_paths)} existing JSON files")

    if not json_paths:
        logger.warning("No documents to ingest")
        return 1

    pipeline = IngestionPipeline()
    total = pipeline.ingest_json_documents(json_paths)
    logger.success(f"Ingested {total} chunks from {len(json_paths)} documents")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest LexML documents (async)")
    parser.add_argument("--keywords", type=str, help="Comma-separated keywords")
    parser.add_argument("--limit", type=int, default=100, help="Max documents to search")
    parser.add_argument("--download-dir", type=str, default="./data/lexml")
    parser.add_argument("--concurrency", type=int, default=_DEFAULT_CONCURRENCY,
                        help="Parallel downloads")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, ingest existing JSON files only")
    parser.add_argument("--force-download", action="store_true",
                        help="Download even if document already exists in tracker")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
