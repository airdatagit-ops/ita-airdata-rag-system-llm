"""
LexML XML Parser for Aviation RAG System.

Parses LexML XML documents and extracts articles with metadata.
"""

from typing import Dict, List, Optional
from lxml import etree
from loguru import logger

from parsers.temporal_extractor import TemporalExtractor


class LexMLParser:
    """Parser for LexML XML documents."""

    def __init__(self):
        """Initialize LexML parser."""
        self.temporal_extractor = TemporalExtractor()
        logger.info("LexMLParser initialized")

    def parse_xml(self, xml_path: str) -> List[Dict]:
        """
        Parse LexML XML file and extract articles.

        Args:
            xml_path: Path to XML file

        Returns:
            List of article dictionaries
        """
        try:
            tree = etree.parse(xml_path)
            root = tree.getroot()

            # Extract document metadata
            metadata = self._extract_metadata(root)

            # Extract articles
            articles = self._extract_articles(root, metadata)

            logger.info(f"Parsed {len(articles)} articles from {xml_path}")
            return articles

        except Exception as e:
            logger.error(f"Error parsing XML {xml_path}: {e}")
            return []

    def _extract_metadata(self, root) -> Dict:
        """Extract document-level metadata."""
        metadata = {
            "urn": None,
            "lei_number": None,
            "lei_date": None,
            "publication_date": None,
            "title": None
        }

        # Try to find URN
        urn_elem = root.find('.//{*}Identificacao[@URN]')
        if urn_elem is not None:
            metadata["urn"] = urn_elem.get("URN")

            # Parse URN: urn:lex:br:federal:lei:1993-06-21;8666
            if metadata["urn"]:
                parts = metadata["urn"].split(':')
                if len(parts) >= 6:
                    date_and_num = parts[5].split(';')
                    metadata["lei_date"] = date_and_num[0]
                    metadata["lei_number"] = date_and_num[1] if len(date_and_num) > 1 else None

        # Find publication date
        date_elem = root.find('.//{*}Data')
        if date_elem is not None and date_elem.text:
            metadata["publication_date"] = date_elem.text

        return metadata

    def _extract_articles(self, root, doc_metadata: Dict) -> List[Dict]:
        """Extract all articles from document."""
        articles = []
        article_elements = root.xpath('//{*}Artigo')

        for art_elem in article_elements:
            article = self._parse_article(art_elem, doc_metadata)
            if article:
                articles.append(article)

        return articles

    def _parse_article(self, art_elem, doc_metadata: Dict) -> Optional[Dict]:
        """Parse individual article element."""
        try:
            article_id = art_elem.get('id', '')

            # Get article number from Rotulo
            rotulo_elem = art_elem.find('.//{*}Rotulo')
            rotulo = rotulo_elem.text if rotulo_elem is not None else ''

            # Get article title if exists
            titulo_elem = art_elem.find('.//{*}TituloDispositivo')
            titulo = titulo_elem.text if titulo_elem is not None else ''

            # Extract full text (caput + paragraphs)
            text_parts = []

            # Caput
            caput = art_elem.find('.//{*}Caput')
            if caput is not None:
                caput_text = ''.join(caput.itertext()).strip()
                if caput_text:
                    text_parts.append(('Caput', caput_text))

            # Paragraphs
            paragrafos = art_elem.findall('.//{*}Paragrafo')
            for para in paragrafos:
                para_rotulo_elem = para.find('.//{*}Rotulo')
                para_label = para_rotulo_elem.text if para_rotulo_elem is not None else 'Parágrafo'
                para_text = ''.join(para.itertext()).strip()
                if para_text:
                    text_parts.append((para_label, para_text))

            # Combine all text
            full_text = '\n\n'.join([f"{label}: {text}" for label, text in text_parts])

            # Extract temporal information
            temporal_info = self.temporal_extractor.extract_dates(
                full_text,
                publication_date=doc_metadata.get("publication_date")
            )

            # Build article dictionary
            article = {
                "regulation_id": f"{doc_metadata.get('lei_number', 'unknown')}-{article_id}",
                "article_id": article_id,
                "article_number": rotulo,
                "title": titulo,
                "text": full_text,
                "effective_date": temporal_info.get("effective_date"),
                "expiry_date": temporal_info.get("expiry_date"),
                "status": "superseded" if temporal_info.get("is_revoked") else "active",
                "metadata": {
                    **doc_metadata,
                    "article_id": article_id,
                    "article_number": rotulo,
                    "source": "lexml",
                    "chunk_type": "article"
                }
            }

            return article

        except Exception as e:
            logger.warning(f"Error parsing article: {e}")
            return None


if __name__ == "__main__":
    """Example usage."""
    parser = LexMLParser()
    # articles = parser.parse_xml("path/to/lei.xml")
    print("LexML Parser ready")
