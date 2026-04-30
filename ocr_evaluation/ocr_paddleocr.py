"""
OCR extraction using PaddleOCR (PP-OCRv4).

Converts PDF pages to images and runs PaddleOCR with Portuguese language support.

Usage:
    python -m ocr_evaluation.ocr_paddleocr <path_to_pdf>
"""

import logging
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

# Suppress PaddleOCR verbose logs before import
logging.getLogger("ppocr").setLevel(logging.WARNING)

try:
    import paddle
    # Disable the PIR executor to avoid the bug:
    #   NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
    #   pir::ArrayAttribute<pir::DoubleAttribute>  (onednn_instruction.cc:116)
    # On x86 CPUs the PIR executor selects oneDNN ops that hit this unimplemented code path
    # in Paddle 3.3.x. os.environ does not work for this flag because Paddle reads it at C++
    # library load time; paddle.set_flags() is the only way to override it after import.
    paddle.set_flags({"FLAGS_enable_pir_in_executor": False})
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logger.error("PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.error("pdf2image not available. Install with: pip install pdf2image")

# Singleton OCR instance
_ocr_engine: "PaddleOCR | None" = None


def _get_ocr() -> "PaddleOCR":
    """Get or create singleton PaddleOCR engine."""
    global _ocr_engine
    if _ocr_engine is None:
        logger.info("[PaddleOCR] Inicializando engine...")
        # PaddleOCR 3.x removed show_log, use_angle_cls and use_gpu parameters.
        # GPU usage is now controlled via paddle environment configuration.
        _ocr_engine = PaddleOCR(lang="pt")
        logger.info("[PaddleOCR] Engine inicializado.")
    return _ocr_engine


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extrai texto de um PDF usando PaddleOCR."""
    if not PADDLE_AVAILABLE:
        logger.error("PaddleOCR is not installed.")
        return ""
    if not PDF2IMAGE_AVAILABLE:
        logger.error("pdf2image is not installed.")
        return ""

    pdf_path = Path(pdf_path)
    logger.info(f"[PaddleOCR] Processando {pdf_path.name}...")

    images = convert_from_path(str(pdf_path), dpi=300)
    logger.info(f"[PaddleOCR] {len(images)} páginas convertidas em imagens (DPI=300)")

    ocr = _get_ocr()
    pages_text = []

    for i, img in enumerate(images, start=1):
        img_array = np.array(img)
        # PaddleOCR 3.x: predict() returns a list of OCRResult objects (one per input image).
        # Each OCRResult has rec_texts (list[str]) and rec_scores (list[float]).
        results = ocr.predict(img_array)

        text_lines = []
        confidences = []

        for page_result in results:
            rec_texts = page_result.get("rec_texts", []) if page_result else []
            rec_scores = page_result.get("rec_scores", []) if page_result else []
            for text, score in zip(rec_texts, rec_scores):
                if isinstance(text, str):
                    text_lines.append(text)
                    confidences.append(float(score))

        page_text = "\n".join(text_lines)
        pages_text.append(page_text)

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        logger.debug(
            f"[PaddleOCR] Página {i}/{len(images)}: {len(page_text)} caracteres, "
            f"confiança média: {avg_conf:.3f}"
        )

    full_text = "\n\f\n".join(pages_text)
    logger.info(f"[PaddleOCR] {pdf_path.name}: {len(full_text)} caracteres totais")
    return full_text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m ocr_evaluation.ocr_paddleocr <caminho_do_pdf>")
        sys.exit(1)
    result = extract_text_from_pdf(Path(sys.argv[1]))
    print(result)
