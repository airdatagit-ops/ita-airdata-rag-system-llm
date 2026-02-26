"""Test script to verify system functionality."""

from loguru import logger
from database.qdrant_manager import QdrantManager
from models.embeddings import EmbeddingModel
from models.llm import LlamaModel
from search.rag import RAGPipeline


def test_qdrant():
    """Test Qdrant connection."""
    try:
        manager = QdrantManager()
        info = manager.get_collection_info()
        logger.success(f"✓ Qdrant: {info.get('vectors_count', 0)} vectors")
        return True
    except Exception as e:
        logger.error(f"✗ Qdrant error: {e}")
        return False


def test_embeddings():
    """Test embedding model."""
    try:
        model = EmbeddingModel()
        emb = model.encode("test")
        logger.success(f"✓ Embeddings: {emb.shape}")
        return True
    except Exception as e:
        logger.error(f"✗ Embeddings error: {e}")
        return False


def test_llm():
    """Test LLM."""
    try:
        llm = LlamaModel()
        response = llm.generate("Diga olá", max_tokens=10)
        logger.success(f"✓ LLM: {response[:50]}...")
        return True
    except Exception as e:
        logger.error(f"✗ LLM error: {e}")
        return False


def test_rag():
    """Test RAG pipeline."""
    try:
        rag = RAGPipeline()
        result = rag.query("teste", limit=1)
        logger.success(f"✓ RAG: {result['total_time_ms']}ms")
        return True
    except Exception as e:
        logger.error(f"✗ RAG error: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Testing Aviation RAG System...")

    tests = [
        ("Qdrant", test_qdrant),
        ("Embeddings", test_embeddings),
        ("LLM", test_llm),
        ("RAG", test_rag)
    ]

    results = []
    for name, test_func in tests:
        logger.info(f"\nTesting {name}...")
        results.append((name, test_func()))

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{name:20s}: {status}")

    logger.info("=" * 50)
    logger.info(f"Total: {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())
