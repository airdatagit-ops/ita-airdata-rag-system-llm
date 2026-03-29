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


def test_revocation_detection_revoking_act():
    """A document that revokes ANOTHER law should NOT be marked as revoked itself."""
    extractor = TemporalExtractor()

    text = "Fica revogada a Lei nº 1234"
    dates = extractor.extract_dates(text)
    assert dates["is_revoked"] is False


def test_revocation_detection_revoked_by():
    """A document revoked by another regulation should be detected."""
    extractor = TemporalExtractor()

    text = "Esta instrução foi revogada pela Portaria nº 456/2024."
    dates = extractor.extract_dates(text)
    assert dates["is_revoked"] is True


def test_revocation_detection_perde_vigencia():
    extractor = TemporalExtractor()

    text = "Esta norma perde sua vigência em 01/01/2025."
    dates = extractor.extract_dates(text)
    assert dates["is_revoked"] is True


@pytest.mark.parametrize("text,expected", [
    ("entra em vigor em 1º de dezembro de 2021", "2021-12-01"),
    ("entra em vigor em 15 de janeiro de 2024", "2024-01-15"),
    ("entra em vigor a partir de 1° de março de 2020", "2020-03-01"),
    ("entra em vigor em 30 de AGOSTO de 2023", "2023-08-30"),
    ("entra em vigor em 15/06/2023", "2023-06-15"),
])
def test_effective_date_portuguese_spelled(text, expected):
    extractor = TemporalExtractor()
    dates = extractor.extract_dates(text)
    assert dates["effective_date"] == expected


def test_effective_date_na_data_publicacao():
    extractor = TemporalExtractor()
    text = "Esta Portaria entra em vigor na data de sua publicação."
    dates = extractor.extract_dates(text, publication_date="2021-12-09")
    assert dates["effective_date"] == "2021-12-09"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
