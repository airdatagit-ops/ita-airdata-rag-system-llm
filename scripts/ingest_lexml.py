"""Script to ingest LexML documents via web scraping."""

import argparse
import json
from pathlib import Path
from loguru import logger

from parsers.lexml_scraper import LexMLScraper
from parsers.document_tracker import get_tracker
from pipeline.ingestion import IngestionPipeline
from config import config


def main():
    parser = argparse.ArgumentParser(description="Ingest LexML documents")
    parser.add_argument("--keywords", type=str, help="Comma-separated keywords")
    parser.add_argument("--limit", type=int, default=100, help="Max documents")
    parser.add_argument("--download-dir", type=str, default="./data/lexml")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, use existing files")
    parser.add_argument("--force-download", action="store_true", help="Force download even if document already exists")
    parser.add_argument("--clear-tracker", action="store_true", help="Clear the document tracker before starting")
    args = parser.parse_args()

    # Parse keywords
    keywords = args.keywords.split(",") if args.keywords else config.lexml_keywords_list

    logger.info(f"Searching LexML for: {keywords}")

    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    # Handle tracker options
    if args.clear_tracker:
        tracker = get_tracker()
        tracker.clear()
        logger.info("Document tracker cleared")

    json_paths = []
    skipped_duplicates = 0

    if not args.skip_download:
        # Search and download
        scraper = LexMLScraper(skip_duplicates=not args.force_download)
        documents = scraper.search(keywords=keywords, limit=args.limit, doc_type="Legislação")

        logger.info(f"Found {len(documents)} documents")
        
        # Show tracker stats
        stats = scraper.get_tracker_stats()
        if stats:
            logger.info(f"Tracker: {stats.get('total_documents', 0)} documents already tracked")

        # Download document contents
        for i, doc in enumerate(documents):
            if not doc.get('url') and not doc.get('urn'):
                logger.warning(f"Skipping document {i}: no URL or URN")
                continue

            # Check for duplicates before downloading
            if scraper.is_duplicate(doc):
                skipped_duplicates += 1
                logger.debug(f"Skipping duplicate: {doc.get('title', 'N/A')[:50]}...")
                continue

            # Generate unique filename based on URN or URL hash
            doc_urn = doc.get('urn', '')
            if doc_urn:
                # Create filename from URN
                safe_urn = doc_urn.replace(':', '_').replace(';', '_').replace('/', '_')[-60:]
                json_path = download_dir / f"{safe_urn}.json"
            else:
                # Fallback to index-based naming
                json_path = download_dir / f"doc_{i}.json"
            
            # First try to get content directly
            content = scraper.get_document_text(doc)
            
            if content:
                # Save as JSON with metadata
                output_data = {
                    **doc,
                    'content': content,
                    'doc_index': i
                }
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                # Register in tracker
                scraper.register_document(doc, content, str(json_path))
                
                json_paths.append(str(json_path))
                logger.info(f"Saved document {i+1}/{len(documents)}: {doc.get('title', 'N/A')[:50]}...")
            else:
                # Try download method as fallback
                if scraper.download_document(doc, str(json_path)):
                    # Register in tracker
                    scraper.register_document(doc, file_path=str(json_path))
                    json_paths.append(str(json_path))

        logger.info(f"Downloaded {len(json_paths)} new documents, skipped {skipped_duplicates} duplicates")
    else:
        # Use existing JSON files
        json_paths = [str(p) for p in download_dir.glob("*.json")]
        logger.info(f"Found {len(json_paths)} existing JSON files")

    if not json_paths:
        logger.warning("No documents to ingest")
        return 1

    # Ingest using the modified pipeline
    pipeline = IngestionPipeline()
    total = pipeline.ingest_json_documents(json_paths)

    logger.success(f"✓ Ingested {total} chunks from {len(json_paths)} documents")
    return 0


if __name__ == "__main__":
    exit(main())
