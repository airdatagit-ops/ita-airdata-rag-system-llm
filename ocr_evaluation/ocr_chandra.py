"""
OCR extraction using Chandra OCR 2 (datalab-to/chandra-ocr-2).

Uses the HuggingFace inference method to run the model locally.
Requires: pip install chandra-ocr

Usage:
    python -m ocr_evaluation.ocr_chandra <path_to_pdf>
"""

import sys
from pathlib import Path

from loguru import logger

try:
    from chandra.model import InferenceManager
    from chandra.input import load_file
    from chandra.model.schema import BatchInputItem
    CHANDRA_AVAILABLE = True
except ImportError:
    CHANDRA_AVAILABLE = False
    logger.warning(
        "Chandra OCR 2 não disponível neste ambiente. "
        "Instale com: pip install chandra-ocr"
    )

# Singleton InferenceManager
_manager: "InferenceManager | None" = None


def _get_manager() -> "InferenceManager":
    """Get or create singleton InferenceManager using HuggingFace local inference."""
    global _manager
    if _manager is None:
        logger.info("[Chandra] Inicializando InferenceManager (method=hf)...")
        _manager = InferenceManager(method="hf")
        logger.info("[Chandra] InferenceManager inicializado.")
    return _manager


def extract_text_from_pdf(pdf_path: Path) -> str | None:
    """Extrai texto de um PDF usando Chandra OCR 2. Retorna None se não disponível."""
    if not CHANDRA_AVAILABLE:
        logger.warning("Chandra OCR 2 não disponível neste ambiente. Pulando...")
        return None

    pdf_path = Path(pdf_path)
    logger.info(f"[Chandra] Processando {pdf_path.name}...")

    try:
        images = load_file(str(pdf_path), config={})
        logger.info(f"[Chandra] {len(images)} páginas carregadas")

        manager = _get_manager()
        batch = [BatchInputItem(image=img) for img in images]
        results = manager.generate(batch)

        pages_text = []
        for i, result in enumerate(results, start=1):
            if result.error:
                logger.warning(f"[Chandra] Página {i}: erro na geração, usando texto vazio")
                pages_text.append("")
            else:
                pages_text.append(result.markdown or "")
            logger.debug(f"[Chandra] Página {i}/{len(images)}: {len(pages_text[-1])} caracteres")

        full_text = "\n\f\n".join(pages_text)
        logger.info(f"[Chandra] {pdf_path.name}: {len(full_text)} caracteres totais")
        return full_text

    except Exception as e:
        logger.error(f"[Chandra] Erro ao processar {pdf_path.name}: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m ocr_evaluation.ocr_chandra <caminho_do_pdf>")
        sys.exit(1)
    result = extract_text_from_pdf(Path(sys.argv[1]))
    if result is not None:
        print(result)
    else:
        print("Chandra OCR 2 não está disponível.", file=sys.stderr)
        sys.exit(1)
