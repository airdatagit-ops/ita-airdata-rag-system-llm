"""
PDF Parser for Aviation RAG System.

Parses PDF documents (ICAs) and extracts text and sections.
"""

from pathlib import Path
from typing import Dict, List, Optional
import re

from loguru import logger

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available, falling back to PyPDF2")

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

from config import config
from parsers.temporal_extractor import TemporalExtractor


class PDFParser:
    """Parser for PDF documents (ICAs, regulations)."""

    def __init__(self, enable_ocr: bool = None):
        """
        Initialize PDF parser.

        Args:
            enable_ocr: Enable OCR for scanned PDFs
        """
        self.enable_ocr = enable_ocr if enable_ocr is not None else config.ENABLE_OCR
        self.temporal_extractor = TemporalExtractor()

        if not PDFPLUMBER_AVAILABLE and not PYPDF2_AVAILABLE:
            raise ImportError("Neither pdfplumber nor PyPDF2 available. Install with: pip install pdfplumber PyPDF2")

        logger.info(f"PDFParser initialized (OCR: {self.enable_ocr})")

    def parse_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Parse PDF file and extract sections.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of section dictionaries
        """
        try:
            # Extract text
            text = self._extract_text(pdf_path)

            if not text or len(text.strip()) < 50:
                logger.warning(f"PDF {pdf_path} has very little text. May need OCR.")
                if self.enable_ocr:
                    text = self._extract_text_with_ocr(pdf_path)

            # Extract metadata from filename and text
            metadata = self._extract_metadata(pdf_path, text)

            # Split into sections
            sections = self._extract_sections(text, metadata)

            logger.info(f"Parsed {len(sections)} sections from {pdf_path}")
            return sections

        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path}: {e}")
            return []

    def _extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF."""
        if PDFPLUMBER_AVAILABLE:
            return self._extract_with_pdfplumber(pdf_path)
        elif PYPDF2_AVAILABLE:
            return self._extract_with_pypdf2(pdf_path)
        else:
            raise RuntimeError("No PDF library available")

    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber."""
        text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return '\n\n'.join(text)

    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2."""
        text = []
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return '\n\n'.join(text)

    def _extract_text_with_ocr(self, pdf_path: str) -> str:
        """Extract text using OCR (requires pytesseract or easyocr)."""
        logger.warning("OCR not yet implemented")
        return ""

    def _extract_metadata(self, pdf_path: str, text: str) -> Dict:
        """Extract metadata from PDF."""
        filename = Path(pdf_path).stem

        # Try to extract ICA number from filename
        # Example: ICA-100-12.pdf -> ICA 100-12
        ica_match = re.search(r'ICA[_-]?(\d+)[_-](\d+)', filename, re.IGNORECASE)

        metadata = {
            "source": "pdf",
            "filename": filename,
            "doc_type": "ica" if ica_match else "regulation",
            "doc_number": f"ICA {ica_match.group(1)}-{ica_match.group(2)}" if ica_match else filename
        }

        # Extract dates from text
        temporal_info = self.temporal_extractor.extract_dates(text)
        metadata.update(temporal_info)

        return metadata

    def _extract_sections(self, text: str, metadata: Dict) -> List[Dict]:
        """Extract sections from text."""
        sections = []

        # Pattern for numbered sections: 1.2.3 Title
        section_pattern = r'^(\d+(?:\.\d+)*)\s+([A-Z][^\n]+)$'

        # Split text into lines
        lines = text.split('\n')
        current_section = None
        current_text = []

        for line in lines:
            # Check if line is a section header
            match = re.match(section_pattern, line.strip())
            if match:
                # Save previous section
                if current_section:
                    sections.append(self._create_section(
                        current_section,
                        '\n'.join(current_text),
                        metadata
                    ))

                # Start new section
                current_section = {
                    "number": match.group(1),
                    "title": match.group(2).strip()
                }
                current_text = []
            else:
                # Add line to current section
                if line.strip():
                    current_text.append(line)

        # Add last section
        if current_section:
            sections.append(self._create_section(
                current_section,
                '\n'.join(current_text),
                metadata
            ))

        # If no sections found, treat whole document as one section
        if not sections:
            sections.append({
                "regulation_id": metadata.get("doc_number", "unknown"),
                "section_number": "1",
                "title": metadata.get("doc_number", "Document"),
                "text": text,
                "effective_date": metadata.get("effective_date"),
                "expiry_date": metadata.get("expiry_date"),
                "status": "active",
                "metadata": metadata
            })

        return sections

    def _create_section(self, section_info: Dict, text: str, metadata: Dict) -> Dict:
        """Create section dictionary."""
        return {
            "regulation_id": f"{metadata.get('doc_number', 'unknown')}-sec-{section_info['number']}",
            "section_number": section_info["number"],
            "title": section_info["title"],
            "text": f"Seção {section_info['number']}: {section_info['title']}\n\n{text}",
            "effective_date": metadata.get("effective_date"),
            "expiry_date": metadata.get("expiry_date"),
            "status": "active",
            "metadata": {
                **metadata,
                "section_number": section_info["number"],
                "chunk_type": "section"
            }
        }


if __name__ == "__main__":
    """Example usage."""
    parser = PDFParser()
    # sections = parser.parse_pdf("path/to/ica.pdf")
    print("PDF Parser ready")
