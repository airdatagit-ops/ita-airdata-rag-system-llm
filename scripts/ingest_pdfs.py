"""Script to ingest PDF documents."""

import argparse
from pathlib import Path
from loguru import logger

from pipeline.ingestion import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF documents")
    parser.add_argument("--source", type=str, required=True, help="Directory with PDFs")
    parser.add_argument("--recursive", action="store_true", help="Search recursively")
    args = parser.parse_args()

    source_dir = Path(args.source)

    if not source_dir.exists():
        logger.error(f"Directory not found: {source_dir}")
        return 1

    # Find PDFs
    if args.recursive:
        pdf_paths = list(source_dir.rglob("*.pdf"))
    else:
        pdf_paths = list(source_dir.glob("*.pdf"))

    logger.info(f"Found {len(pdf_paths)} PDF files")

    if not pdf_paths:
        logger.warning("No PDF files found")
        return 0

    # Ingest
    pipeline = IngestionPipeline()
    total = pipeline.ingest_pdfs([str(p) for p in pdf_paths])

    logger.success(f"✓ Ingested {total} sections from {len(pdf_paths)} PDFs")
    return 0


if __name__ == "__main__":
    exit(main())
