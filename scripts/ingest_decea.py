"""Script to ingest DECEA documents (ICA, MCA, etc.) via web scraping."""

import argparse
import json
from pathlib import Path
from loguru import logger

from parsers.decea_scraper import DECEAScraper
from pipeline.ingestion import IngestionPipeline
from config import config


def main():
    parser = argparse.ArgumentParser(description="Ingest DECEA documents (ICA, MCA, etc.)")
    parser.add_argument("--doc-types", type=str, default="ICA", 
                        help="Comma-separated document types (e.g., ICA,MCA,PCA)")
    parser.add_argument("--slugs", type=str, 
                        help="Comma-separated specific slugs to fetch (e.g., ICA-63-47,ICA-100-12)")
    parser.add_argument("--keywords", type=str, 
                        help="Comma-separated keywords to filter by")
    parser.add_argument("--limit", type=int, default=100, 
                        help="Max documents to fetch")
    parser.add_argument("--download-dir", type=str, default="./data/decea",
                        help="Directory to save downloaded documents")
    parser.add_argument("--skip-download", action="store_true", 
                        help="Skip download, use existing files")
    parser.add_argument("--no-text", action="store_true",
                        help="Don't extract PDF text (just use description)")
    args = parser.parse_args()

    # Parse arguments
    doc_types = args.doc_types.split(",") if args.doc_types else ["ICA"]
    slugs = args.slugs.split(",") if args.slugs else None
    keywords = args.keywords.split(",") if args.keywords else None

    logger.info(f"DECEA Ingestion - Types: {doc_types}, Limit: {args.limit}")
    if slugs:
        logger.info(f"Specific slugs: {slugs}")

    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    json_paths = []

    if not args.skip_download:
        # Search and download
        scraper = DECEAScraper()
        
        documents = scraper.search(
            doc_types=doc_types,
            keywords=keywords,
            limit=args.limit,
            slugs=slugs
        )

        logger.info(f"Found {len(documents)} documents")

        if not documents:
            logger.warning("No documents found. Try specifying --slugs with known document IDs.")
            logger.info("Example: --slugs ICA-63-47,ICA-100-12,ICA-100-37")
            return 1

        # Download document contents
        for i, doc in enumerate(documents):
            slug = doc.get('slug', f'doc_{i}')
            safe_slug = slug.replace('/', '-')
            json_path = download_dir / f"{safe_slug}.json"
            
            # Get content (from PDF if possible, or just description)
            if args.no_text:
                content = doc.get('description', '')
            else:
                content = scraper.get_document_text(doc)
            
            if not content:
                content = doc.get('description', '')
            
            # Save as JSON with metadata
            output_data = {
                **doc,
                'content': content,
                'doc_index': i,
                'doc_type': 'ica'  # For document counter compatibility
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            json_paths.append(str(json_path))
            
            content_len = len(content) if content else 0
            logger.info(f"Saved {i+1}/{len(documents)}: {slug} ({content_len} chars)")

        logger.info(f"Downloaded {len(json_paths)} documents to {download_dir}")
    else:
        # Use existing JSON files
        json_paths = [str(p) for p in download_dir.glob("*.json")]
        logger.info(f"Found {len(json_paths)} existing JSON files")

    if not json_paths:
        logger.warning("No documents to ingest")
        return 1

    # Ingest using the pipeline
    pipeline = IngestionPipeline()
    total = pipeline.ingest_json_documents(json_paths)

    logger.success(f"✓ Ingested {total} chunks from {len(json_paths)} documents")
    
    # Show summary
    logger.info("\n" + "="*50)
    logger.info("Summary:")
    logger.info(f"  Documents found: {len(json_paths)}")
    logger.info(f"  Chunks created: {total}")
    logger.info(f"  Files saved to: {download_dir}")
    logger.info("="*50)
    
    return 0


if __name__ == "__main__":
    exit(main())
