"""Pre-download all ML models required by the RAG pipeline.

Run this once before starting the server so that no model downloads
happen during request processing.

Usage:
    python -m scripts.download_models          # download all
    python -m scripts.download_models --skip-ollama  # skip Ollama pull
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from loguru import logger

from config import config


def download_embedding_model() -> None:
    """Download the dense embedding model (sentence-transformers)."""
    model_name = config.EMBEDDING_MODEL
    cache_dir = str(config.get_model_cache_path())

    logger.info(f"Downloading embedding model: {model_name}")
    start = time.time()

    from sentence_transformers import SentenceTransformer
    SentenceTransformer(model_name, cache_folder=cache_dir)

    logger.success(f"Embedding model ready ({time.time() - start:.1f}s)")


def download_cross_encoder() -> None:
    """Download the cross-encoder model used by the Evaluator."""
    if not config.EVALUATOR_ENABLED:
        logger.info("Evaluator disabled — skipping cross-encoder download")
        return

    model_name = config.CROSS_ENCODER_MODEL
    logger.info(f"Downloading cross-encoder model: {model_name}")
    start = time.time()

    from sentence_transformers import CrossEncoder
    CrossEncoder(model_name)

    logger.success(f"Cross-encoder model ready ({time.time() - start:.1f}s)")


def pull_ollama_models() -> None:
    """Pull Ollama LLM models used by the Rewriter and Generator."""
    models_to_pull: set[str] = set()

    models_to_pull.add(config.OLLAMA_MODEL)

    if config.REWRITER_ENABLED and config.REWRITER_MODEL:
        models_to_pull.add(config.REWRITER_MODEL)
    if config.GENERATOR_MODEL:
        models_to_pull.add(config.GENERATOR_MODEL)

    for model in sorted(models_to_pull):
        logger.info(f"Pulling Ollama model: {model}")
        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                timeout=600,
            )
            if result.returncode == 0:
                logger.success(f"Ollama model ready: {model}")
            else:
                logger.warning(f"ollama pull {model} failed (exit code {result.returncode})")
        except FileNotFoundError:
            logger.warning("ollama CLI not found — skipping Ollama model pulls")
            break
        except subprocess.TimeoutExpired:
            logger.warning(f"ollama pull {model} timed out after 600s")


def main():
    parser = argparse.ArgumentParser(description="Pre-download all ML models")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama model pulls")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding model")
    parser.add_argument("--skip-cross-encoder", action="store_true", help="Skip cross-encoder model")
    args = parser.parse_args()

    logger.info("=== Pre-downloading ML models ===")
    start = time.time()

    if not args.skip_embeddings:
        download_embedding_model()

    if not args.skip_cross_encoder:
        download_cross_encoder()

    if not args.skip_ollama:
        pull_ollama_models()

    logger.success(f"All models ready ({time.time() - start:.1f}s total)")


if __name__ == "__main__":
    main()
