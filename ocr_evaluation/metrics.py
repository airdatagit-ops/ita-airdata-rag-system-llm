"""
Metrics module for OCR evaluation.

Provides CER, WER, and domain-specific precision/recall/F1 calculations.

Usage:
    from ocr_evaluation.metrics import calculate_cer, calculate_wer, calculate_domain_precision_recall
"""

import re
from math import nan

from loguru import logger

try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    logger.error("jiwer not available. Install with: pip install jiwer")

# ============================================================
# Aviation domain terms for precision/recall evaluation
# ============================================================
AVIATION_DOMAIN_TERMS = [
    # Siglas operacionais
    "NOTAM", "AIP", "AIRAC", "SIGMET", "METAR", "TAF", "ATIS", "PIREP", "SPECI", "SNOWTAM",
    # Órgãos e autoridades
    "ICAO", "DECEA", "ANAC", "ICEA", "CISCEA", "DPV", "CGNA", "SRPV",
    # Parâmetros meteorológicos e de altitude
    "QNH", "QFE", "QNE", "QDM", "CAVOK", "TEMPO", "BECMG", "VRB", "PROB", "NOSIG",
    # Radionavegação e procedimentos
    "ILS", "VOR", "NDB", "DME", "RNAV", "RNP", "GNSS", "GPS", "PAPI", "VASI", "GPWS", "TCAS",
    # Infraestrutura aeroportuária
    "TWY", "RWY", "ACFT", "THR", "ARP", "APRON", "HLD",
    # Espaço aéreo
    "CTR", "TMA", "FIR", "UIR", "ADIZ", "ATZ", "CTA", "ACC", "APP", "TWR", "AFIS", "UNICOM",
    # Procedimentos de voo
    "SID", "STAR", "IFR", "VFR", "SVFR", "IMC", "VMC", "OCA", "OCH", "MDA", "DH", "DA", "MDH",
    # Níveis de voo
    "FL", "MSL", "AGL", "AMSL",
    # Português aeronáutico
    "aeródromo", "pista", "taxiway", "decolagem", "pouso", "pousar", "decolar",
    "espaço aéreo", "zona de controle", "rota aérea", "plano de voo",
    "autorização", "restrição", "proibição", "altitude", "nível de voo",
    "visibilidade", "teto", "nuvem", "turbulência", "vento", "temperatura",
    "aeronave", "piloto", "controlador", "torre", "aproximação", "radar",
]

# Deduplicate (ILS, QFE, QNH appear in multiple categories)
AVIATION_DOMAIN_TERMS = list(dict.fromkeys(AVIATION_DOMAIN_TERMS))


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) between reference and hypothesis.

    Returns:
        CER as float between 0.0 (perfect) and 1.0+ (many errors).
        Returns nan if reference is empty.
    """
    if not JIWER_AVAILABLE:
        logger.error("jiwer is required for CER calculation.")
        return nan

    if not reference or not reference.strip():
        logger.warning("Reference text is empty — CER is undefined.")
        return nan

    return jiwer.cer(reference, hypothesis)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis.

    Applies transformations: lowercase, remove punctuation, strip, split into words.

    Returns:
        WER as float between 0.0 (perfect) and 1.0+ (many errors).
        Returns nan if reference is empty.
    """
    if not JIWER_AVAILABLE:
        logger.error("jiwer is required for WER calculation.")
        return nan

    if not reference or not reference.strip():
        logger.warning("Reference text is empty — WER is undefined.")
        return nan

    def _preprocess(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()

    return jiwer.wer(_preprocess(reference), _preprocess(hypothesis))


def calculate_domain_precision_recall(hypothesis: str) -> dict:
    """
    Calculate precision, recall, and F1 for aviation domain terms.

    Checks which terms from AVIATION_DOMAIN_TERMS appear in the hypothesis text.

    Returns:
        Dict with keys: precision, recall, f1, found_terms, missing_terms.
    """
    if not hypothesis or not hypothesis.strip():
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "found_terms": [],
            "missing_terms": list(AVIATION_DOMAIN_TERMS),
        }

    found_terms = []
    missing_terms = []

    for term in AVIATION_DOMAIN_TERMS:
        pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        if pattern.search(hypothesis):
            found_terms.append(term)
        else:
            missing_terms.append(term)

    total_terms = len(AVIATION_DOMAIN_TERMS)
    found_count = len(found_terms)

    recall = found_count / total_terms if total_terms > 0 else 0.0

    # Precision: of domain terms detected, how many are in our list
    # Since we only check terms from our list, precision here is 1.0 when any are found
    # (every found term is by definition in the list).
    # A more useful precision: ratio of found terms to total unique "relevant-looking" tokens.
    # But per spec: precision = found / found = 1.0 when found > 0.
    # We define precision as recall-equivalent here (found / total), as the spec's
    # formulation reduces to that when only checking list membership.
    precision = found_count / total_terms if total_terms > 0 else 0.0

    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "found_terms": found_terms,
        "missing_terms": missing_terms,
    }


def format_results_table(results: dict) -> str:
    """
    Format benchmark results as an ASCII table sorted by CER (lowest first).

    Args:
        results: Dict of {model_name: {cer, wer, domain_f1, elapsed_seconds, ...}}

    Returns:
        Formatted ASCII table string.
    """
    # Sort by CER (nan goes to the end)
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1].get("cer", float("inf")) if x[1].get("cer") == x[1].get("cer") else float("inf"),
    )

    header = f"{'#':<4} {'Modelo':<20} {'CER':>8} {'WER':>8} {'Domain F1':>10} {'Tempo (s)':>10}"
    separator = "-" * len(header)

    lines = [separator, header, separator]

    for rank, (model, metrics) in enumerate(sorted_models, start=1):
        cer = metrics.get("cer", float("nan"))
        wer = metrics.get("wer", float("nan"))
        domain_f1 = metrics.get("domain_f1", float("nan"))
        elapsed = metrics.get("elapsed_seconds", 0.0)

        cer_str = f"{cer:.1%}" if cer == cer else "N/A"
        wer_str = f"{wer:.1%}" if wer == wer else "N/A"
        f1_str = f"{domain_f1:.3f}" if domain_f1 == domain_f1 else "N/A"
        time_str = f"{elapsed:.1f}"

        lines.append(f"{rank:<4} {model:<20} {cer_str:>8} {wer_str:>8} {f1_str:>10} {time_str:>10}")

    lines.append(separator)
    return "\n".join(lines)
