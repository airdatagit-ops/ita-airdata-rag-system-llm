"""
OCR extraction using unstructured[pdf].

Uses unstructured's partition_pdf with OCR strategy for scanned PDFs.
Classifies elements by type (Title, NarrativeText, Table, etc.).

Usage:
    python -m ocr_evaluation.ocr_unstructured <path_to_pdf>
"""

import sys
from collections import Counter
from pathlib import Path

from loguru import logger

try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logger.error("unstructured not available. Install with: pip install \"unstructured[pdf]\"")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extrai texto de um PDF usando unstructured[pdf]."""
    if not UNSTRUCTURED_AVAILABLE:
        logger.error("unstructured[pdf] is not installed.")
        return ""

    pdf_path = Path(pdf_path)
    logger.info(f"[unstructured] Processando {pdf_path.name}...")

    elements = partition_pdf(filename=str(pdf_path), strategy="ocr_only", languages=["por"])

    logger.info(f"[unstructured] {len(elements)} elementos extraídos")

    # Build text preserving element sequence with double breaks between different types
    text_parts = []
    prev_type = None

    for element in elements:
        elem_type = type(element).__name__
        elem_text = element.text if hasattr(element, "text") else str(element)

        if not elem_text.strip():
            continue

        if prev_type is not None and elem_type != prev_type:
            text_parts.append("")  # double newline between different types

        text_parts.append(elem_text)
        prev_type = elem_type

    # Log element type counts
    type_counts = Counter(type(e).__name__ for e in elements)
    type_summary = ", ".join(f"{t}: {c}" for t, c in type_counts.most_common())
    logger.info(f"[unstructured] Tipos de elementos: {type_summary}")

    full_text = "\n".join(text_parts)
    logger.info(f"[unstructured] {pdf_path.name}: {len(full_text)} caracteres totais")
    return full_text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m ocr_evaluation.ocr_unstructured <caminho_do_pdf>")
        sys.exit(1)
    result = extract_text_from_pdf(Path(sys.argv[1]))
    print(result)
