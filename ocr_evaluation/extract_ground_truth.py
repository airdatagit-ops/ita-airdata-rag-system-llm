"""
Extract ground truth text from PDFs using pdfplumber.

Reads all PDFs from test_data/, extracts the digital text layer,
and saves results to correct_texts/ as .txt files.

Usage:
    python -m ocr_evaluation.extract_ground_truth
"""

from pathlib import Path

from loguru import logger

from ocr_evaluation._logging import configure_file_logging

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.error("pdfplumber not available. Install with: pip install pdfplumber")

BASE_DIR = Path(__file__).parent
TEST_DATA_DIR = BASE_DIR / "test_data"
CORRECT_TEXTS_DIR = BASE_DIR / "correct_texts"


def extract_ground_truth_from_pdf(pdf_path: Path) -> str | None:
    """
    Extract digital text from a PDF using pdfplumber.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text or None if extraction failed.
    """
    if not PDFPLUMBER_AVAILABLE:
        logger.error("pdfplumber is not installed. Cannot extract ground truth.")
        return None

    pages_text = []
    empty_pages = 0

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"  Processando {pdf_path.name}: {total_pages} páginas")

        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages_text.append(text)
            else:
                empty_pages += 1
                logger.warning(f"  Página {i}/{total_pages} de {pdf_path.name}: sem texto digital (possível scan)")
                pages_text.append("")

    if empty_pages > 0:
        logger.warning(f"  {pdf_path.name}: {empty_pages}/{total_pages} páginas sem texto digital")

    full_text = "\n\f\n".join(pages_text)
    return full_text


def main():
    """Extract ground truth for all PDFs in test_data/."""
    configure_file_logging()
    if not PDFPLUMBER_AVAILABLE:
        logger.error("pdfplumber is required. Install with: pip install pdfplumber")
        return

    CORRECT_TEXTS_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(TEST_DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"Nenhum PDF encontrado em {TEST_DATA_DIR}/")
        logger.info("Coloque os PDFs de teste nessa pasta e rode novamente.")
        return

    logger.info(f"Encontrados {len(pdf_files)} PDF(s) em {TEST_DATA_DIR}/")

    processed = 0
    empty_count = 0

    for pdf_path in pdf_files:
        text = extract_ground_truth_from_pdf(pdf_path)
        if text is None:
            continue

        output_path = CORRECT_TEXTS_DIR / f"{pdf_path.stem}.txt"
        output_path.write_text(text, encoding="utf-8")

        char_count = len(text)
        if char_count == 0 or text.strip() == "":
            empty_count += 1
            logger.warning(f"  {pdf_path.name}: texto extraído vazio!")
        else:
            logger.info(f"  {pdf_path.name}: {char_count} caracteres extraídos → {output_path.name}")

        processed += 1

    logger.info("=" * 60)
    logger.info("RESUMO DA EXTRAÇÃO DE GROUND TRUTH")
    logger.info("=" * 60)
    logger.info(f"PDFs processados: {processed}")
    logger.info(f"PDFs com texto vazio: {empty_count}")
    logger.info(f"Resultados salvos em: {CORRECT_TEXTS_DIR}/")


if __name__ == "__main__":
    main()
