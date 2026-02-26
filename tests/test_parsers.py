"""Tests for parsers module."""

import pytest
from parsers.temporal_extractor import TemporalExtractor


def test_temporal_extractor():
    """Test temporal date extraction."""
    extractor = TemporalExtractor()

    text = "Esta lei entra em vigor em 15/06/2023"
    dates = extractor.extract_dates(text, publication_date="2023-01-01")

    assert dates["effective_date"] == "2023-06-15"
    assert dates["is_revoked"] == False


def test_revocation_detection():
    """Test revocation detection."""
    extractor = TemporalExtractor()

    text = "Fica revogada a Lei nº 1234"
    dates = extractor.extract_dates(text)

    assert dates["is_revoked"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
