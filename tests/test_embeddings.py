"""Tests for embeddings module."""

import pytest
import numpy as np
from models.embeddings import EmbeddingModel


@pytest.fixture
def model():
    """Create embedding model."""
    return EmbeddingModel()


def test_single_encoding(model):
    """Test encoding single text."""
    text = "Esta é uma lei sobre aviação"
    embedding = model.encode(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1024,)  # Legal-BERTimbau dimension


def test_batch_encoding(model):
    """Test batch encoding."""
    texts = ["Lei 1", "Lei 2", "Lei 3"]
    embeddings = model.encode(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 1024)


def test_similarity(model):
    """Test similarity computation."""
    text1 = "aviação civil"
    text2 = "aeronave comercial"

    sim = model.get_similarity(text1, text2)

    assert 0.0 <= sim <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
