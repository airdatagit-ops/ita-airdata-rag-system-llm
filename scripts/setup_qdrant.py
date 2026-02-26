"""Setup script for initializing Qdrant collection."""

from loguru import logger
from database.qdrant_manager import QdrantManager
from config import config


def main():
    """Initialize Qdrant collection with indexes."""
    logger.info("Setting up Qdrant collection...")

    manager = QdrantManager()

    # Create collection
    success = manager.create_collection(
        vector_size=config.EMBEDDING_DIMENSION,
        recreate=False  # Set to True to recreate
    )

    if success:
        logger.success("✓ Qdrant setup completed successfully!")

        # Get info
        info = manager.get_collection_info()
        logger.info(f"Collection: {manager.collection_name}")
        logger.info(f"Vectors: {info.get('vectors_count', 0)}")
        logger.info(f"Status: {info.get('status', 'unknown')}")
    else:
        logger.error("✗ Qdrant setup failed")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
