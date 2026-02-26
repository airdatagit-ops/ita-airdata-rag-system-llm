#!/usr/bin/env python3
"""
Script to reset and rebuild the Qdrant vector database.

This script:
1. Optionally clears the document tracker
2. Deletes and recreates the Qdrant collection
3. Re-ingests all documents from the data folders

Usage:
    python -m scripts.reset_database --confirm
    python -m scripts.reset_database --confirm --clear-tracker
    python -m scripts.reset_database --confirm --only-lexml
    python -m scripts.reset_database --confirm --only-decea
"""

import argparse
from pathlib import Path
from loguru import logger

from database.qdrant_manager import QdrantManager
from parsers.document_tracker import get_tracker
from pipeline.ingestion import IngestionPipeline


DATA_DIR = Path(__file__).parent.parent / "data"
LEXML_DIR = DATA_DIR / "lexml"
DECEA_DIR = DATA_DIR / "decea"


def count_existing_documents():
    """Count existing JSON documents in data folders."""
    lexml_count = len(list(LEXML_DIR.glob("*.json"))) if LEXML_DIR.exists() else 0
    decea_count = len(list(DECEA_DIR.glob("*.json"))) if DECEA_DIR.exists() else 0
    return lexml_count, decea_count


def main():
    parser = argparse.ArgumentParser(
        description="Reset and rebuild the Qdrant vector database"
    )
    parser.add_argument(
        "--confirm", 
        action="store_true", 
        required=True,
        help="Confirm that you want to delete all vectors (required)"
    )
    parser.add_argument(
        "--clear-tracker", 
        action="store_true",
        help="Also clear the document tracker (allows re-downloading)"
    )
    parser.add_argument(
        "--only-lexml", 
        action="store_true",
        help="Only re-ingest LexML documents"
    )
    parser.add_argument(
        "--only-decea", 
        action="store_true",
        help="Only re-ingest DECEA documents"
    )
    parser.add_argument(
        "--skip-ingest", 
        action="store_true",
        help="Only reset the database, don't re-ingest documents"
    )
    args = parser.parse_args()

    if not args.confirm:
        logger.error("You must use --confirm to proceed with database reset")
        return 1

    # Show current state
    lexml_count, decea_count = count_existing_documents()
    logger.info(f"Found {lexml_count} LexML documents and {decea_count} DECEA documents")

    # Get tracker stats
    tracker = get_tracker()
    tracker_stats = tracker.get_stats()
    logger.info(f"Document tracker has {tracker_stats['total_documents']} tracked documents")

    # Clear tracker if requested
    if args.clear_tracker:
        tracker.clear()
        logger.warning("Document tracker cleared")

    # Reset Qdrant collection
    logger.warning("Recreating Qdrant collection (this will delete all vectors)...")
    db = QdrantManager()
    db.create_collection(recreate=True)
    logger.success("Qdrant collection recreated successfully")

    if args.skip_ingest:
        logger.info("Skipping re-ingestion as requested")
        return 0

    # Re-ingest documents
    pipeline = IngestionPipeline()
    total_chunks = 0

    # Determine which sources to ingest
    ingest_lexml = not args.only_decea
    ingest_decea = not args.only_lexml

    if ingest_lexml and lexml_count > 0:
        logger.info(f"Re-ingesting {lexml_count} LexML documents...")
        json_paths = [str(p) for p in LEXML_DIR.glob("*.json")]
        chunks = pipeline.ingest_json_documents(json_paths)
        total_chunks += chunks
        logger.success(f"Ingested {chunks} chunks from LexML documents")

    if ingest_decea and decea_count > 0:
        logger.info(f"Re-ingesting {decea_count} DECEA documents...")
        json_paths = [str(p) for p in DECEA_DIR.glob("*.json")]
        chunks = pipeline.ingest_json_documents(json_paths)
        total_chunks += chunks
        logger.success(f"Ingested {chunks} chunks from DECEA documents")

    # Show final stats
    logger.success(f"\n✓ Database reset complete!")
    logger.info(f"  Total chunks ingested: {total_chunks}")
    
    # Get collection info
    try:
        info = db.client.get_collection(db.collection_name)
        logger.info(f"  Vectors in collection: {info.points_count}")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    exit(main())
