"""
OCR extraction using docTR (Mindee).

Uses a Vision Transformer-based pipeline for end-to-end text detection + recognition.

Usage:
    python -m ocr_evaluation.ocr_doctr <path_to_pdf>
"""

import sys
from pathlib import Path

from loguru import logger

try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False
    logger.error("docTR not available. Install with: pip install \"python-doctr[torch]\" pdf2image")

# Singleton predictor instance
_predictor = None


def _get_predictor():
    """Get or create singleton docTR predictor."""
    global _predictor
    if _predictor is None:
        logger.info("[docTR] Inicializando predictor (db_resnet50 + crnn_vgg16_bn)...")
        _predictor = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)
        logger.info("[docTR] Predictor inicializado.")
    return _predictor


def _export_text(result) -> str:
    """Extract text from docTR result preserving page → block → line → word order."""
    pages_text = []

    exported = result.export()
    for page in exported["pages"]:
        lines_text = []
        for block in page["blocks"]:
            for line in block["lines"]:
                words = [word["value"] for word in line["words"]]
                lines_text.append(" ".join(words))
        pages_text.append("\n".join(lines_text))

    return "\n\f\n".join(pages_text)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extrai texto de um PDF usando docTR."""
    if not DOCTR_AVAILABLE:
        logger.error("docTR is not installed.")
        return ""

    pdf_path = Path(pdf_path)
    logger.info(f"[docTR] Processando {pdf_path.name}...")

    doc = DocumentFile.from_pdf(str(pdf_path))
    logger.info(f"[docTR] {len(doc)} páginas carregadas")

    predictor = _get_predictor()
    result = predictor(doc)

    full_text = _export_text(result)
    logger.info(f"[docTR] {pdf_path.name}: {len(full_text)} caracteres totais")
    return full_text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m ocr_evaluation.ocr_doctr <caminho_do_pdf>")
        sys.exit(1)
    result = extract_text_from_pdf(Path(sys.argv[1]))
    print(result)
