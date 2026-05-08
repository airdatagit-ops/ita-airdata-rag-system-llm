"""
PDF Parser for Aviation RAG System.

Parses PDF documents (ICAs) and extracts text and sections.
"""

from pathlib import Path
from typing import Dict, List
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


def extract_text_from_bytes(pdf_content: bytes) -> str | None:
    """Extract text from in-memory PDF bytes (docling | PyMuPDF -> pdfplumber -> OCR)."""
    # ── Docling path (quando ativado via config) ─────────────────────────────
    if config.PDF_EXTRACTION_BACKEND == 'docling':
        try:
            import io
            import tempfile
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions, PdfBackend

            backend_map = {
                'docling_parse': PdfBackend.DOCLING_PARSE,
                'pypdfium2': PdfBackend.PYPDFIUM2,
            }
            pdf_backend = backend_map.get(config.DOCLING_PDF_BACKEND, PdfBackend.DOCLING_PARSE)

            pipeline_options = PdfPipelineOptions(
                do_ocr=False,
                do_table_structure=config.DOCLING_DO_TABLE_STRUCTURE,
                force_backend_text=config.DOCLING_FORCE_BACKEND_TEXT,
            )
            if config.DOCLING_ARTIFACTS_PATH:
                from pathlib import Path as _Path
                pipeline_options.artifacts_path = _Path(config.DOCLING_ARTIFACTS_PATH)

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                        backend=pdf_backend,
                    )
                }
            )
            # Docling precisa de um path ou BytesIO; usar arquivo temporário
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                tmp.write(pdf_content)
                tmp.flush()
                result = converter.convert(tmp.name)
            text = result.document.export_to_markdown()
            if len(text.strip()) > 100:
                return text
        except Exception as e:
            logger.warning(f"Docling extraction failed, falling back to legacy: {e}")

    # ── Legacy path (padrão) ─────────────────────────────────────────────────
    try:
        import fitz
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        if len(text.strip()) > 100:
            return text
    except (ImportError, Exception):
        pass

    try:
        import pdfplumber
        import io
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        if len(text.strip()) > 100:
            return text
    except (ImportError, Exception):
        pass

    try:
        from parsers.ocr_processor import OCRProcessor
        ocr = OCRProcessor(use_gpu=False)
        text = ocr.extract_text_from_pdf_bytes(pdf_content, resolution=300)
        if text and len(text.strip()) > 50:
            return text
    except (ImportError, Exception):
        pass

    return None


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
            dl_doc = None

            # Quando docling está ativo, extrair tanto texto quanto o documento estruturado
            if config.PDF_EXTRACTION_BACKEND == 'docling':
                text, dl_doc = self._extract_with_docling(pdf_path)
                if not text or len(text.strip()) < 50:
                    logger.warning(f"Docling insufficient for {pdf_path}, falling back")
                    dl_doc = None
                    text = self._extract_text_legacy(pdf_path)
            else:
                text = self._extract_text(pdf_path)

            if not text or len(text.strip()) < 50:
                logger.warning(f"PDF {pdf_path} has very little text. May need OCR.")
                if self.enable_ocr:
                    text = self._extract_text_with_ocr(pdf_path)

            # Extract metadata from filename and text
            metadata = self._extract_metadata(pdf_path, text)

            # Serializar o DoclingDocument para JSON e guardar nos metadados (para chunking nativo)
            if dl_doc is not None and config.DOCLING_CHUNKING_ENABLED:
                try:
                    import json as _json
                    metadata['_docling_doc_json'] = _json.dumps(
                        dl_doc.export_to_dict(), ensure_ascii=False
                    )
                except Exception as e:
                    logger.warning(f"Failed to serialize DoclingDocument: {e}")

            # Split into sections
            sections = self._extract_sections(text, metadata)

            logger.info(f"Parsed {len(sections)} sections from {pdf_path}")
            return sections

        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path}: {e}")
            return []

    def _extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF (docling or legacy, depending on config)."""
        if config.PDF_EXTRACTION_BACKEND == 'docling':
            text, _ = self._extract_with_docling(pdf_path)
            if text and len(text.strip()) > 100:
                return text
            logger.warning(f"Docling extraction insufficient for {pdf_path}, falling back to legacy")

        # ── Legacy path ──────────────────────────────────────────────────────
        return self._extract_text_legacy(pdf_path)

    def _extract_text_legacy(self, pdf_path: str) -> str:
        """Extract text using legacy backends (pdfplumber/PyPDF2)."""
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

    def _extract_with_docling(self, pdf_path: str) -> tuple[str, object | None]:
        """
        Extract text (and optionally DoclingDocument) from PDF using docling.

        Returns:
            Tuple of (markdown_text, docling_document_or_None).
            docling_document_or_None is None if extraction fails.
        """
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions, PdfBackend

            backend_map = {
                'docling_parse': PdfBackend.DOCLING_PARSE,
                'pypdfium2': PdfBackend.PYPDFIUM2,
            }
            pdf_backend = backend_map.get(config.DOCLING_PDF_BACKEND, PdfBackend.DOCLING_PARSE)

            pipeline_options = PdfPipelineOptions(
                do_ocr=False,
                do_table_structure=config.DOCLING_DO_TABLE_STRUCTURE,
                force_backend_text=config.DOCLING_FORCE_BACKEND_TEXT,
            )
            if config.DOCLING_ARTIFACTS_PATH:
                from pathlib import Path as _Path
                pipeline_options.artifacts_path = _Path(config.DOCLING_ARTIFACTS_PATH)

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                        backend=pdf_backend,
                    )
                }
            )
            result = converter.convert(pdf_path)
            dl_doc = result.document
            markdown_text = dl_doc.export_to_markdown()
            logger.debug(f"Docling extracted {len(markdown_text)} chars from {pdf_path}")
            return markdown_text, dl_doc
        except Exception as e:
            logger.warning(f"Docling extraction failed for {pdf_path}: {e}")
            return "", None

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
