"""Script to ingest DECEA documents (ICA, MCA, etc.) via web scraping."""

import argparse
from pathlib import Path
from loguru import logger

from crawler.scrapers.decea_scraper import DECEAScraper
from pipeline.ingestion import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(description="Ingest DECEA documents (ICA, MCA, etc.)")
    parser.add_argument("--doc-types", type=str, default="ICA",
                        help="Comma-separated document types (e.g., ICA,MCA,PCA)")
    parser.add_argument("--slugs", type=str,
                        help="Comma-separated specific slugs (e.g., ICA-63-47,ICA-100-12)")
    parser.add_argument("--keywords", type=str,
                        help="Comma-separated keywords to filter by title")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max documents to fetch")
    parser.add_argument("--download-dir", type=str, default="./data/decea",
                        help="Directory to save downloaded documents")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use existing JSON files")
    parser.add_argument("--no-text", action="store_true",
                        help="Don't extract PDF text (just use description)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers for fetching documents")
    args = parser.parse_args()

    doc_types = args.doc_types.split(",") if args.doc_types else ["ICA"]
    slugs = args.slugs.split(",") if args.slugs else None
    keywords = args.keywords.split(",") if args.keywords else None

    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    json_paths = []

    if not args.skip_download:
        scraper = DECEAScraper()
        documents = scraper.search(
            doc_types=doc_types, keywords=keywords,
            limit=args.limit, slugs=slugs,
        )

        if not documents:
            logger.warning("No documents found")
            return 1

        results = scraper.fetch_all(
            documents, workers=args.workers,
            extract_text=not args.no_text, save_original=True,
        )

        for doc, text in results:
            content = text or doc.get("title", "")
            path = scraper.save_document_json(doc, content)
            json_paths.append(str(path))
            logger.info(f"Saved: {doc['slug']} ({len(content):,} chars)")
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


if __name__ == "__main__":
    exit(main())
