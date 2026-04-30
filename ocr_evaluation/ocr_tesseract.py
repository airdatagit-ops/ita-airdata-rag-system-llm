"""
OCR extraction using Tesseract with image preprocessing.

Converts PDF pages to images and applies a preprocessing pipeline
(grayscale, binarization, denoising, deskew) before running Tesseract.

Usage:
    python -m ocr_evaluation.ocr_tesseract <path_to_pdf>
"""

import sys
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.error(
        "pytesseract not available. Install with: pip install pytesseract\n"
        "Also install Tesseract engine: sudo apt-get install tesseract-ocr tesseract-ocr-por"
    )

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.error("pdf2image not available. Install with: pip install pdf2image")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.error("OpenCV not available. Install with: pip install opencv-python-headless")


def _preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Apply preprocessing pipeline to a page image for better OCR accuracy.

    Pipeline:
        1. Convert to grayscale
        2. Otsu's binarization
        3. Light denoising
        4. Deskew (rotation correction)
        5. Upscale if too small
    """
    img = np.array(pil_image)

    # 1. Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # 2. Otsu's binarization
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Denoising
    denoised = cv2.fastNlMeansDenoising(binary, h=10)

    # 4. Deskew
    denoised = _deskew(denoised)

    # 5. Upscale if width < 2000px
    h, w = denoised.shape[:2]
    if w < 2000:
        scale = 2000 / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        denoised = cv2.resize(denoised, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return denoised


def _deskew(image: np.ndarray) -> np.ndarray:
    """Detect skew angle via minAreaRect on contours and rotate if > 0.5 degrees."""
    coords = np.column_stack(np.where(image < 128))
    if len(coords) < 50:
        return image

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # Normalize angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    logger.debug(f"  Deskew: rotated {angle:.2f}°")
    return rotated


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extrai texto de um PDF usando Tesseract OCR com preprocessing."""
    if not TESSERACT_AVAILABLE:
        logger.error("pytesseract is not installed.")
        return ""
    if not PDF2IMAGE_AVAILABLE:
        logger.error("pdf2image is not installed.")
        return ""
    if not CV2_AVAILABLE:
        logger.error("OpenCV is not installed.")
        return ""

    pdf_path = Path(pdf_path)
    logger.info(f"[Tesseract] Processando {pdf_path.name}...")

    images = convert_from_path(str(pdf_path), dpi=300)
    logger.info(f"[Tesseract] {len(images)} páginas convertidas em imagens (DPI=300)")

    pages_text = []
    for i, img in enumerate(images, start=1):
        logger.debug(f"[Tesseract] Página {i}/{len(images)}: preprocessing...")
        preprocessed = _preprocess_image(img)
        pil_preprocessed = Image.fromarray(preprocessed)

        text = pytesseract.image_to_string(pil_preprocessed, lang="por+eng", config="--oem 3 --psm 6")
        pages_text.append(text)
        logger.debug(f"[Tesseract] Página {i}: {len(text)} caracteres extraídos")

    full_text = "\n\f\n".join(pages_text)
    logger.info(f"[Tesseract] {pdf_path.name}: {len(full_text)} caracteres totais")
    return full_text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m ocr_evaluation.ocr_tesseract <caminho_do_pdf>")
        sys.exit(1)
    result = extract_text_from_pdf(Path(sys.argv[1]))
    print(result)
