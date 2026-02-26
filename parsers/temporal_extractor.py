"""
Temporal Extractor Module for Aviation RAG System.

This module extracts temporal information from legal documents:
- Publication dates
- Effective dates (when regulation becomes valid)
- Expiry dates (when regulation is revoked)
- Version information

Usage:
    from parsers.temporal_extractor import TemporalExtractor

    extractor = TemporalExtractor()
    dates = extractor.extract_dates(document_text)
"""

import re
from datetime import datetime, timedelta
from typing import Dict, Optional, List

from dateutil import parser as date_parser
from loguru import logger

from config import config


class TemporalExtractor:
    """
    Extract temporal information from legal documents.

    This class uses regex patterns and heuristics to identify dates
    when regulations become effective and when they are revoked.
    """

    def __init__(self):
        """Initialize temporal extractor with regex patterns."""
        # Patterns for effective dates
        self.effective_patterns = [
            # "entra em vigor em 15/06/2023"
            r"entra(?:rá)?\s+em\s+vigor\s+(?:em|na data de|a partir de)?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            # "vigência a partir de 01/01/2024"
            r"vigência\s+a\s+partir\s+de\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            # "produzirá efeitos a partir de"
            r"produzirá\s+efeitos?\s+(?:a\s+partir\s+de)?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            # "passa a vigorar em"
            r"passa\s+a\s+vigorar\s+(?:em|na data de)?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            # "publicação" (will add default days)
            r"(?:após|da)\s+(?:sua\s+)?publicação",
        ]

        # Patterns for revocation dates
        self.revocation_patterns = [
            # "revoga a Lei 1234"
            r"revoga(?:da)?\s+(?:a|o)\s+(Lei|Decreto|Resolução|Portaria)\s+n?º?\s*(\d+)",
            # "fica revogado"
            r"(?:fica|são)\s+revogado?s?",
            # "perde vigência"
            r"perde(?:rá)?\s+(?:sua\s+)?vigência",
            # "deixa de vigorar"
            r"deixa(?:rá)?\s+de\s+vigorar",
        ]

        # Pattern for extracting dates
        self.date_pattern = r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})"

        logger.info("TemporalExtractor initialized")

    def extract_dates(
        self,
        text: str,
        publication_date: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Optional[str]]:
        """
        Extract temporal dates from document text.

        Args:
            text: Document text
            publication_date: Known publication date (ISO format)
            metadata: Additional metadata that might contain dates

        Returns:
            Dictionary with extracted dates:
            {
                "publication_date": "YYYY-MM-DD" or None,
                "effective_date": "YYYY-MM-DD" or None,
                "expiry_date": "YYYY-MM-DD" or None,
                "is_revoked": bool
            }

        Example:
            >>> extractor = TemporalExtractor()
            >>> text = "Esta lei entra em vigor em 15/06/2023"
            >>> dates = extractor.extract_dates(text)
            >>> dates["effective_date"]
            "2023-06-15"
        """
        result = {
            "publication_date": publication_date,
            "effective_date": None,
            "expiry_date": None,
            "is_revoked": False
        }

        # Extract effective date
        effective_date = self._extract_effective_date(text, publication_date)
        if effective_date:
            result["effective_date"] = effective_date

        # Check if document is revoked
        is_revoked = self._check_revocation(text)
        result["is_revoked"] = is_revoked

        # Extract revocation date if present
        if is_revoked:
            revocation_date = self._extract_revocation_date(text)
            if revocation_date:
                result["expiry_date"] = revocation_date

        # Fallback: if no effective date, use publication + default days
        if not result["effective_date"] and result["publication_date"]:
            result["effective_date"] = self._add_days_to_date(
                result["publication_date"],
                config.DEFAULT_EFFECTIVE_DAYS
            )
            logger.debug(
                f"No explicit effective date found, using publication + "
                f"{config.DEFAULT_EFFECTIVE_DAYS} days"
            )

        logger.debug(f"Extracted dates: {result}")
        return result

    def _extract_effective_date(
        self,
        text: str,
        publication_date: Optional[str] = None
    ) -> Optional[str]:
        """
        Extract effective date from text.

        Args:
            text: Document text
            publication_date: Publication date for fallback

        Returns:
            Effective date in ISO format or None
        """
        # Try each pattern
        for pattern in self.effective_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Check if pattern has a date group
                if match.groups():
                    date_str = match.group(1)
                    parsed_date = self._parse_date(date_str)
                    if parsed_date:
                        logger.debug(f"Found effective date: {parsed_date}")
                        return parsed_date
                else:
                    # Pattern like "após publicação" - use publication date
                    if publication_date:
                        effective = self._add_days_to_date(
                            publication_date,
                            config.DEFAULT_EFFECTIVE_DAYS
                        )
                        logger.debug(
                            f"Effective date from 'após publicação': {effective}"
                        )
                        return effective

        return None

    def _extract_revocation_date(self, text: str) -> Optional[str]:
        """
        Extract revocation/expiry date from text.

        Args:
            text: Document text

        Returns:
            Revocation date in ISO format or None
        """
        # Look for dates near revocation keywords
        for pattern in self.revocation_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Look for a date within 100 characters after the match
                context = text[match.end():match.end() + 100]
                date_match = re.search(self.date_pattern, context)
                if date_match:
                    date_str = date_match.group(0)
                    parsed_date = self._parse_date(date_str)
                    if parsed_date:
                        logger.debug(f"Found revocation date: {parsed_date}")
                        return parsed_date

        return None

    def _check_revocation(self, text: str) -> bool:
        """
        Check if document mentions being revoked.

        Args:
            text: Document text

        Returns:
            True if revoked, False otherwise
        """
        for pattern in self.revocation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.debug("Document appears to be revoked")
                return True
        return False

    def _parse_date(self, date_str: str) -> Optional[str]:
        """
        Parse date string to ISO format (YYYY-MM-DD).

        Args:
            date_str: Date string in various formats

        Returns:
            Date in ISO format or None if parsing fails
        """
        try:
            # Try parsing with dateutil (handles many formats)
            dt = date_parser.parse(date_str, dayfirst=True)
            return dt.strftime("%Y-%m-%d")
        except Exception as e:
            logger.warning(f"Could not parse date '{date_str}': {e}")
            return None

    def _add_days_to_date(self, date_str: str, days: int) -> str:
        """
        Add days to a date.

        Args:
            date_str: Date in ISO format
            days: Number of days to add

        Returns:
            New date in ISO format
        """
        dt = datetime.fromisoformat(date_str)
        new_dt = dt + timedelta(days=days)
        return new_dt.strftime("%Y-%m-%d")

    def extract_version_info(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract version/amendment information.

        Args:
            text: Document text

        Returns:
            Dictionary with version info:
            {
                "amends": "lei-1234",  # What this amends
                "amended_by": None      # What amended this (usually None for new docs)
            }
        """
        result = {
            "amends": None,
            "amended_by": None
        }

        # Pattern for amendments
        amendment_patterns = [
            r"altera\s+(?:a|o)\s+(Lei|Decreto|Resolução)\s+n?º?\s*(\d+)",
            r"modifica\s+(?:a|o)\s+(Lei|Decreto|Resolução)\s+n?º?\s*(\d+)",
            r"dá\s+nova\s+redação\s+(?:à|ao)\s+(Lei|Decreto|Resolução)\s+n?º?\s*(\d+)",
        ]

        for pattern in amendment_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doc_type = match.group(1).lower()
                doc_number = match.group(2)
                result["amends"] = f"{doc_type}-{doc_number}"
                logger.debug(f"Document amends: {result['amends']}")
                break

        return result

    def extract_all_dates_from_text(self, text: str) -> List[str]:
        """
        Extract all dates found in text.

        Args:
            text: Document text

        Returns:
            List of dates in ISO format
        """
        dates = []
        for match in re.finditer(self.date_pattern, text):
            date_str = match.group(0)
            parsed = self._parse_date(date_str)
            if parsed:
                dates.append(parsed)

        logger.debug(f"Found {len(dates)} dates in text")
        return dates


# ========================================
# Example Usage
# ========================================

if __name__ == "__main__":
    """Example usage of TemporalExtractor."""

    extractor = TemporalExtractor()

    # Example 1: Extract effective date
    print("=== Example 1: Effective Date ===")
    text1 = """
    Lei nº 12.345, de 15 de junho de 2023.

    Art. 1º Esta lei estabelece normas sobre aviação civil.

    Art. 50. Esta lei entra em vigor em 01/09/2023.
    """
    dates1 = extractor.extract_dates(text1, publication_date="2023-06-15")
    print(f"Publication: {dates1['publication_date']}")
    print(f"Effective: {dates1['effective_date']}")
    print(f"Revoked: {dates1['is_revoked']}")

    # Example 2: Document with revocation
    print("\n=== Example 2: Revoked Document ===")
    text2 = """
    Lei nº 54.321, de 10 de janeiro de 2024.

    Art. 1º Fica revogada a Lei nº 12.345.

    Art. 2º Esta lei entra em vigor na data de sua publicação.
    """
    dates2 = extractor.extract_dates(text2, publication_date="2024-01-10")
    print(f"Effective: {dates2['effective_date']}")
    print(f"Revoked: {dates2['is_revoked']}")

    # Example 3: Version information
    print("\n=== Example 3: Amendment ===")
    text3 = """
    Resolução nº 440, de 2023.

    Art. 1º Esta resolução altera a Lei nº 8.666.
    """
    version_info = extractor.extract_version_info(text3)
    print(f"Amends: {version_info['amends']}")

    # Example 4: Extract all dates
    print("\n=== Example 4: All Dates ===")
    text4 = """
    Publicado em 15/06/2023.
    Vigência a partir de 01/09/2023.
    Revogado em 10/01/2024.
    """
    all_dates = extractor.extract_all_dates_from_text(text4)
    print(f"All dates found: {all_dates}")

    print("\n✓ All examples completed!")
