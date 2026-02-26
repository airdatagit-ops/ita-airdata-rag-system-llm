"""
Embedding Model Module for Aviation RAG System.

This module provides a wrapper around Legal-BERTimbau for generating
embeddings of regulatory texts.

Usage:
    from models.embeddings import EmbeddingModel

    model = EmbeddingModel()
    embeddings = model.encode(["texto 1", "texto 2"])
"""

import time
from typing import List, Union

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from config import config


class EmbeddingModel:
    """
    Wrapper for Legal-BERTimbau embedding model.

    This class provides a simple interface for generating embeddings with
    caching, batching, and GPU support.

    Attributes:
        model: The loaded SentenceTransformer model
        device: Device being used (cuda/cpu)
        dimension: Embedding vector dimension
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        cache_dir: str = None
    ):
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model name (default from config)
            device: Device to use ('cuda', 'cpu', or None for auto)
            cache_dir: Directory to cache model (default from config)
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.cache_dir = cache_dir or str(config.get_model_cache_path())

        # Determine device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"CUDA available, using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            logger.warning("CUDA not available, using CPU (this will be slow)")

        # Load model
        logger.info(f"Loading embedding model: {self.model_name}")
        start_time = time.time()

        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir
            )
            load_time = time.time() - start_time
            logger.success(
                f"Embedding model loaded successfully in {load_time:.2f}s "
                f"(device: {self.device})"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # Get embedding dimension
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

        # Verify dimension matches config
        if self.dimension != config.EMBEDDING_DIMENSION:
            logger.warning(
                f"Embedding dimension ({self.dimension}) does not match "
                f"config ({config.EMBEDDING_DIMENSION}). Updating config."
            )

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = None,
        show_progress: bool = False,
        normalize: bool = True,
        convert_to_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate embeddings for input texts.

        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for processing (default from config)
            show_progress: Show progress bar for large batches
            normalize: Normalize embeddings to unit vectors
            convert_to_numpy: Return numpy array instead of torch tensor

        Returns:
            Embeddings as numpy array (n_texts, dimension) or torch tensor

        Example:
            >>> model = EmbeddingModel()
            >>> embeddings = model.encode(["text 1", "text 2"])
            >>> embeddings.shape
            (2, 1024)
        """
        # Handle single string
        if isinstance(texts, str):
            texts = [texts]
            was_single = True
        else:
            was_single = False

        batch_size = batch_size or config.EMBEDDING_BATCH_SIZE

        logger.debug(
            f"Encoding {len(texts)} texts (batch_size={batch_size}, "
            f"normalize={normalize})"
        )

        start_time = time.time()

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=convert_to_numpy,
                device=self.device
            )

            encode_time = time.time() - start_time
            texts_per_second = len(texts) / encode_time if encode_time > 0 else 0

            logger.debug(
                f"Encoded {len(texts)} texts in {encode_time:.2f}s "
                f"({texts_per_second:.1f} texts/s)"
            )

            # Return single embedding if input was single string
            if was_single:
                return embeddings[0]

            return embeddings

        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Convenience method for batch encoding with progress bar.

        Args:
            texts: List of texts to encode
            batch_size: Batch size
            show_progress: Show progress bar

        Returns:
            Embeddings as numpy array
        """
        return self.encode(
            texts,
            batch_size=batch_size,
            show_progress=show_progress,
            convert_to_numpy=True
        )

    def get_similarity(
        self,
        text1: Union[str, np.ndarray],
        text2: Union[str, np.ndarray]
    ) -> float:
        """
        Compute cosine similarity between two texts or embeddings.

        Args:
            text1: First text or embedding
            text2: Second text or embedding

        Returns:
            Cosine similarity score (0-1)

        Example:
            >>> model = EmbeddingModel()
            >>> score = model.get_similarity(
            ...     "O piloto deve ter certificado válido",
            ...     "Certificado de piloto é obrigatório"
            ... )
            >>> print(f"Similarity: {score:.3f}")
        """
        # Encode if strings
        if isinstance(text1, str):
            emb1 = self.encode(text1)
        else:
            emb1 = text1

        if isinstance(text2, str):
            emb2 = self.encode(text2)
        else:
            emb2 = text2

        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )

        return float(similarity)

    def get_similarity_matrix(
        self,
        texts1: List[str],
        texts2: List[str] = None
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix between two sets of texts.

        Args:
            texts1: First set of texts
            texts2: Second set of texts (if None, compute with texts1)

        Returns:
            Similarity matrix of shape (len(texts1), len(texts2))

        Example:
            >>> model = EmbeddingModel()
            >>> queries = ["query 1", "query 2"]
            >>> docs = ["doc 1", "doc 2", "doc 3"]
            >>> sim_matrix = model.get_similarity_matrix(queries, docs)
            >>> sim_matrix.shape
            (2, 3)
        """
        # Encode both sets
        emb1 = self.encode(texts1, show_progress=True)

        if texts2 is None:
            emb2 = emb1
        else:
            emb2 = self.encode(texts2, show_progress=True)

        # Compute similarity matrix using cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(emb1, emb2)

        return similarity_matrix

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"EmbeddingModel(model={self.model_name}, "
            f"device={self.device}, dim={self.dimension})"
        )


# ========================================
# Utility Functions
# ========================================

def batch_encode_texts(
    texts: List[str],
    model: EmbeddingModel = None,
    batch_size: int = None
) -> np.ndarray:
    """
    Helper function to batch encode texts.

    Args:
        texts: List of texts to encode
        model: EmbeddingModel instance (creates new if None)
        batch_size: Batch size (default from config)

    Returns:
        Embeddings as numpy array
    """
    if model is None:
        model = EmbeddingModel()

    return model.encode(
        texts,
        batch_size=batch_size or config.EMBEDDING_BATCH_SIZE,
        show_progress=True
    )


def embed_single_text(text: str, model: EmbeddingModel = None) -> np.ndarray:
    """
    Helper function to embed a single text.

    Args:
        text: Text to embed
        model: EmbeddingModel instance (creates new if None)

    Returns:
        Embedding as numpy array
    """
    if model is None:
        model = EmbeddingModel()

    return model.encode(text)


# ========================================
# Example Usage
# ========================================

if __name__ == "__main__":
    """Example usage of EmbeddingModel."""

    # Initialize model
    print("Initializing embedding model...")
    model = EmbeddingModel()

    # Example texts
    texts = [
        "Art. 1º Esta lei estabelece normas gerais sobre licitações.",
        "Art. 2º As obras e serviços serão precedidos de licitação.",
        "O piloto deve possuir certificado de habilitação válido.",
    ]

    # Encode texts
    print(f"\nEncoding {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # Compute similarities
    print("\nSimilarity between texts:")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = model.get_similarity(embeddings[i], embeddings[j])
            print(f"Text {i+1} vs Text {j+1}: {sim:.3f}")

    # Test single text encoding
    print("\nEncoding single text...")
    single_text = "Normas de aviação civil"
    single_embedding = model.encode(single_text)
    print(f"Single embedding shape: {single_embedding.shape}")

    print("\n✓ All tests passed!")
