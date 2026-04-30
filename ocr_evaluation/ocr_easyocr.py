"""
OCR extraction using EasyOCR.

Converts PDF pages to images and runs EasyOCR with Portuguese + English models.

Usage:
    python -m ocr_evaluation.ocr_easyocr <path_to_pdf>
"""

import os
import sys
from pathlib import Path

import numpy as np
from loguru import logger

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.error("EasyOCR not available. Install with: pip install easyocr")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.error("pdf2image not available. Install with: pip install pdf2image")

# Singleton reader instance
_reader: "easyocr.Reader | None" = None


def _get_reader() -> "easyocr.Reader":
    """Get or create singleton EasyOCR Reader."""
    global _reader
    if _reader is None:
        use_gpu = os.environ.get("EASYOCR_USE_GPU", "0") == "1"
        logger.info(f"[EasyOCR] Inicializando Reader (GPU={use_gpu})...")
        _reader = easyocr.Reader(["pt", "en"], gpu=use_gpu)
        logger.info("[EasyOCR] Reader inicializado.")
    return _reader


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extrai texto de um PDF usando EasyOCR."""
    if not EASYOCR_AVAILABLE:
        logger.error("EasyOCR is not installed.")
        return ""
    if not PDF2IMAGE_AVAILABLE:
        logger.error("pdf2image is not installed.")
        return ""

    pdf_path = Path(pdf_path)
    logger.info(f"[EasyOCR] Processando {pdf_path.name}...")

    images = convert_from_path(str(pdf_path), dpi=250)
    logger.info(f"[EasyOCR] {len(images)} páginas convertidas em imagens (DPI=250)")

    reader = _get_reader()
    pages_text = []

    for i, img in enumerate(images, start=1):
        img_array = np.array(img)
        results = reader.readtext(img_array, detail=0, paragraph=True)
        page_text = "\n".join(results)
        pages_text.append(page_text)
        logger.debug(f"[EasyOCR] Página {i}/{len(images)}: {len(page_text)} caracteres")

    full_text = "\n\f\n".join(pages_text)
    logger.info(f"[EasyOCR] {pdf_path.name}: {len(full_text)} caracteres totais")
    return full_text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m ocr_evaluation.ocr_easyocr <caminho_do_pdf>")
        sys.exit(1)
    result = extract_text_from_pdf(Path(sys.argv[1]))
    print(result)
