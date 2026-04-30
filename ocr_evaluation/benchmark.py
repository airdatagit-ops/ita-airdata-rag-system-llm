"""
OCR Benchmark orchestrator.

Runs all OCR models on test PDFs, calculates metrics (CER, WER, Domain F1),
and exports results to CSV/JSON.

Usage:
    python -m ocr_evaluation.benchmark
    python -m ocr_evaluation.benchmark --models tesseract,paddleocr,doctr
    python -m ocr_evaluation.benchmark --pdf ocr_evaluation/test_data/AIP_Brasil.pdf
    python -m ocr_evaluation.benchmark --input-dir /path/to/pdfs/
    python -m ocr_evaluation.benchmark --extract-only
    python -m ocr_evaluation.benchmark --extract-only --models tesseract,doctr
    python -m ocr_evaluation.benchmark --extract-only --input-dir ocr_evaluation/test_pdfs/
"""

import argparse
import csv
import importlib
import json
import re
import sys
import time
from datetime import datetime
from math import isnan
from pathlib import Path

from loguru import logger

from ocr_evaluation._logging import configure_file_logging
from ocr_evaluation.metrics import (
    calculate_cer,
    calculate_domain_precision_recall,
    calculate_wer,
    format_results_table,
)

BASE_DIR = Path(__file__).parent
TEST_DATA_DIR = BASE_DIR / "test_data"
CORRECT_TEXTS_DIR = BASE_DIR / "correct_texts"
DEFAULT_RESULTS_DIR = BASE_DIR / "results"

AVAILABLE_MODELS = {
    "tesseract": "ocr_evaluation.ocr_tesseract",
    "easyocr": "ocr_evaluation.ocr_easyocr",
    "doctr": "ocr_evaluation.ocr_doctr",
    "unstructured": "ocr_evaluation.ocr_unstructured",
    "chandra": "ocr_evaluation.ocr_chandra",
    "paddleocr": "ocr_evaluation.ocr_paddleocr",
}


def _is_paddle_runtime_incompatibility(error: Exception) -> bool:
    """Detect known Paddle runtime incompatibility for CPU oneDNN + PIR conversion."""
    message = str(error)
    patterns = (
        "ConvertPirAttribute2RuntimeAttribute not support",
        "pir::ArrayAttribute<pir::DoubleAttribute>",
        "onednn_instruction.cc:116",
    )
    return all(pattern in message for pattern in patterns)


def _normalize_text(text: str) -> str:
    """Normalize text for metric comparison: normalize newlines, collapse whitespace, strip."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _count_pages(text: str) -> int:
    """Count pages in text based on form-feed separators."""
    if not text:
        return 0
    return text.count("\f") + 1


def _discover_pdfs(pdf_arg: str | None, input_dir: str | None = None) -> list[Path]:
    """Find PDF files to process."""
    if pdf_arg:
        pdf_path = Path(pdf_arg)
        if not pdf_path.exists():
            logger.error(f"PDF não encontrado: {pdf_path}")
            return []
        return [pdf_path]

    search_dir = Path(input_dir) if input_dir else TEST_DATA_DIR
    if input_dir and not search_dir.is_dir():
        logger.error(f"Diretório de entrada não encontrado: {search_dir}")
        return []

    pdf_files = sorted(search_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"Nenhum PDF encontrado em {search_dir}/")
        if not input_dir:
            logger.info("Coloque os PDFs de teste em ocr_evaluation/test_data/ e rode novamente.")
        return []

    return pdf_files


def _load_ground_truth(pdf_path: Path) -> str | None:
    """Load ground truth text for a given PDF."""
    gt_path = CORRECT_TEXTS_DIR / f"{pdf_path.stem}.txt"
    if not gt_path.exists():
        logger.warning(f"Ground truth não encontrado para {pdf_path.name}. Rode extract_ground_truth.py primeiro.")
        return None
    return gt_path.read_text(encoding="utf-8")


def _run_single_model(model_name: str, module_path: str, pdf_path: Path) -> dict:
    """Run a single OCR model on a single PDF and return extraction result."""
    result = {
        "model": model_name,
        "pdf": pdf_path.name,
        "status": "OK",
        "text": "",
        "elapsed_seconds": 0.0,
    }

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        logger.error(f"[{model_name}] Não foi possível importar módulo: {e}")
        result["status"] = "IMPORT_ERROR"
        return result

    t_start = time.perf_counter()
    try:
        text = module.extract_text_from_pdf(pdf_path)
    except Exception as e:
        if model_name == "paddleocr" and _is_paddle_runtime_incompatibility(e):
            logger.warning(
                f"[{model_name}] Incompatibilidade de runtime detectada em {pdf_path.name}: {e}. "
                "Use versões de fallback (paddlepaddle<3.3, paddleocr<3.4)."
            )
            result["status"] = "SKIPPED_INCOMPATIBLE_RUNTIME"
        else:
            logger.error(f"[{model_name}] Erro ao processar {pdf_path.name}: {e}")
            result["status"] = "ERROR"
        result["elapsed_seconds"] = time.perf_counter() - t_start
        return result
    t_end = time.perf_counter()

    if text is None:
        result["status"] = "SKIPPED"
        result["elapsed_seconds"] = t_end - t_start
        return result

    result["text"] = text
    result["elapsed_seconds"] = t_end - t_start
    return result


def _calculate_metrics_for_result(extraction: dict, ground_truth: str | None) -> dict:
    """Calculate all metrics for a single extraction result."""
    text = extraction.get("text", "")
    norm_hyp = _normalize_text(text) if text else ""
    norm_ref = _normalize_text(ground_truth) if ground_truth else ""

    metrics = {
        "pdf": extraction["pdf"],
        "model": extraction["model"],
        "status": extraction["status"],
        "elapsed_seconds": extraction["elapsed_seconds"],
        "pages": _count_pages(text),
        "chars_extracted": len(text),
        "cer": float("nan"),
        "wer": float("nan"),
        "domain_recall": 0.0,
        "domain_precision": 0.0,
        "domain_f1": 0.0,
        "found_terms": [],
        "missing_terms": [],
    }

    if extraction["status"] not in ("OK",):
        return metrics

    # CER and WER require ground truth
    if norm_ref:
        metrics["cer"] = calculate_cer(norm_ref, norm_hyp)
        metrics["wer"] = calculate_wer(norm_ref, norm_hyp)

    # Domain terms (no ground truth needed)
    domain = calculate_domain_precision_recall(norm_hyp)
    metrics["domain_recall"] = domain["recall"]
    metrics["domain_precision"] = domain["precision"]
    metrics["domain_f1"] = domain["f1"]
    metrics["found_terms"] = domain["found_terms"]
    metrics["missing_terms"] = domain["missing_terms"]

    return metrics


def _aggregate_by_model(all_metrics: list[dict]) -> dict:
    """Aggregate metrics per model (mean over all PDFs)."""
    model_data: dict[str, list[dict]] = {}
    for m in all_metrics:
        model_data.setdefault(m["model"], []).append(m)

    aggregated = {}
    for model, entries in model_data.items():
        ok_entries = [e for e in entries if e["status"] == "OK"]
        if not ok_entries:
            aggregated[model] = {
                "cer": float("nan"),
                "wer": float("nan"),
                "domain_f1": float("nan"),
                "elapsed_seconds": sum(e["elapsed_seconds"] for e in entries),
                "status": entries[0]["status"] if entries else "UNKNOWN",
            }
            continue

        cers = [e["cer"] for e in ok_entries if e["cer"] == e["cer"]]  # filter nan
        wers = [e["wer"] for e in ok_entries if e["wer"] == e["wer"]]
        f1s = [e["domain_f1"] for e in ok_entries]

        aggregated[model] = {
            "cer": sum(cers) / len(cers) if cers else float("nan"),
            "wer": sum(wers) / len(wers) if wers else float("nan"),
            "domain_f1": sum(f1s) / len(f1s) if f1s else float("nan"),
            "elapsed_seconds": sum(e["elapsed_seconds"] for e in ok_entries),
            "pdfs_processed": len(ok_entries),
        }

    return aggregated


def _save_extracted_text(text: str, model_name: str, pdf_name: str, results_dir: Path):
    """Save OCR-extracted text to results directory."""
    model_dir = results_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_dir / f"{Path(pdf_name).stem}.txt"
    output_path.write_text(text, encoding="utf-8")


def _export_results(all_metrics: list[dict], aggregated: dict, results_dir: Path, timestamp: str):
    """Export detailed and summary results to CSV + JSON."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # Detail CSV
    detail_path = results_dir / f"details_{timestamp}.csv"
    detail_fields = [
        "pdf", "model", "status", "cer", "wer", "domain_precision", "domain_recall",
        "domain_f1", "elapsed_seconds", "pages", "chars_extracted",
    ]
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields, extrasaction="ignore")
        writer.writeheader()
        for m in all_metrics:
            row = {k: m.get(k, "") for k in detail_fields}
            # Convert nan to empty string for CSV
            for k in ("cer", "wer"):
                val = row.get(k)
                if isinstance(val, float) and isnan(val):
                    row[k] = ""
            writer.writerow(row)
    logger.info(f"Resultados detalhados salvos em: {detail_path}")

    # Summary CSV
    summary_path = results_dir / f"summary_{timestamp}.csv"
    summary_fields = ["model", "cer", "wer", "domain_f1", "elapsed_seconds", "pdfs_processed"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        for model, agg in aggregated.items():
            row = {"model": model, **{k: agg.get(k, "") for k in summary_fields if k != "model"}}
            for k in ("cer", "wer", "domain_f1"):
                val = row.get(k)
                if isinstance(val, float) and isnan(val):
                    row[k] = ""
            writer.writerow(row)
    logger.info(f"Resumo salvo em: {summary_path}")

    # Full JSON (includes term lists)
    full_path = results_dir / f"full_{timestamp}.json"
    json_safe = []
    for m in all_metrics:
        entry = {}
        for k, v in m.items():
            if isinstance(v, float) and isnan(v):
                entry[k] = None
            else:
                entry[k] = v
        json_safe.append(entry)
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "details": json_safe, "summary": {
            model: {k: (None if isinstance(v, float) and isnan(v) else v) for k, v in agg.items()}
            for model, agg in aggregated.items()
        }}, f, ensure_ascii=False, indent=2)
    logger.info(f"Resultados completos salvos em: {full_path}")


def _print_final_summary(aggregated: dict, pdf_count: int, timestamp: str):
    """Print the final ranking to the terminal."""
    print()
    print("=" * 60)
    print("BENCHMARK OCR — RESUMO FINAL")
    print("=" * 60)
    print(f"Modelos testados: {len(aggregated)}")
    print(f"PDFs processados: {pdf_count}")
    print(f"Timestamp: {timestamp}")
    print()
    print("RANKING POR CER (menor = melhor):")

    table = format_results_table(aggregated)
    print(table)
    print()
    print(f"Resultados detalhados salvos em: {DEFAULT_RESULTS_DIR}/")
    print("=" * 60)


def main():
    """Main benchmark orchestration."""
    configure_file_logging()
    parser = argparse.ArgumentParser(description="Benchmark de OCR para documentos normativos de aviação")
    parser.add_argument("--pdf", type=str, default=None, help="Caminho para um PDF específico")
    parser.add_argument(
        "--input-dir", type=str, default=None, dest="input_dir",
        help="Diretório contendo os PDFs de entrada (substitui o padrão ocr_evaluation/test_data/)",
    )
    parser.add_argument(
        "--models", type=str, default="all",
        help="Lista de modelos separados por vírgula (ex: tesseract,paddleocr,doctr) ou 'all'",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Pasta para salvar resultados")
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Pular extração e usar resultados já salvos",
    )
    parser.add_argument(
        "--extract-only", action="store_true",
        help="Apenas extrair texto dos PDFs e salvar em .txt, sem calcular métricas (não requer ground truth)",
    )
    args = parser.parse_args()

    results_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to run
    if args.models == "all":
        selected_models = dict(AVAILABLE_MODELS)
    else:
        selected_models = {}
        for name in args.models.split(","):
            name = name.strip().lower()
            if name in AVAILABLE_MODELS:
                selected_models[name] = AVAILABLE_MODELS[name]
            else:
                logger.warning(f"Modelo desconhecido: '{name}'. Modelos válidos: {list(AVAILABLE_MODELS.keys())}")

    if not selected_models:
        logger.error("Nenhum modelo válido selecionado.")
        sys.exit(1)

    # Discover PDFs
    pdf_files = _discover_pdfs(args.pdf, args.input_dir)
    if not pdf_files:
        sys.exit(1)

    logger.info(f"PDFs a processar: {len(pdf_files)}")
    logger.info(f"Modelos selecionados: {list(selected_models.keys())}")

    total_tasks = len(pdf_files) * len(selected_models)
    print()
    print("=" * 60)
    print(f"  PDFs encontrados : {len(pdf_files)}")
    print(f"  Modelos          : {len(selected_models)} ({', '.join(selected_models)})")
    print(f"  Tarefas totais   : {total_tasks}")
    print("=" * 60)
    print()

    # Try to import tqdm for progress bar
    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False

    _GREEN = "\033[92m"
    _RESET = "\033[0m"

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    all_metrics = []
    task_idx = 0

    for pdf_path in pdf_files:
        ground_truth = None if args.extract_only else _load_ground_truth(pdf_path)

        iterator = selected_models.items()
        if HAS_TQDM:
            iterator = tqdm(list(iterator), desc=f"{pdf_path.name}", leave=False)

        for model_name, module_path in iterator:
            task_idx += 1
            if not HAS_TQDM:
                logger.info(f"[{task_idx}/{total_tasks}] Processando {pdf_path.name} com {model_name}...")

            if args.skip_extraction:
                # Try to load previously extracted text
                saved_path = results_dir / model_name / f"{pdf_path.stem}.txt"
                if saved_path.exists():
                    extraction = {
                        "model": model_name,
                        "pdf": pdf_path.name,
                        "status": "OK",
                        "text": saved_path.read_text(encoding="utf-8"),
                        "elapsed_seconds": 0.0,
                    }
                else:
                    logger.warning(
                        f"Texto extraído não encontrado para {model_name}/{pdf_path.stem}. "
                        f"Rode sem --skip-extraction primeiro."
                    )
                    continue
            else:
                extraction = _run_single_model(model_name, module_path, pdf_path)

                # Save extracted text
                if extraction["status"] == "OK" and extraction["text"]:
                    _save_extracted_text(extraction["text"], model_name, pdf_path.name, results_dir)

            if args.extract_only:
                status = extraction["status"]
                elapsed = extraction["elapsed_seconds"]
                chars = len(extraction.get("text", ""))
                logger.info(
                    f"  {model_name} → {pdf_path.name}: status={status}, "
                    f"chars={chars}, Tempo={elapsed:.1f}s"
                )
                pct = task_idx / total_tasks * 100
                print(f"{_GREEN}  [{pct:5.1f}%] {model_name} → {pdf_path.name} concluído{_RESET}")
                continue

            # Calculate metrics
            metrics = _calculate_metrics_for_result(extraction, ground_truth)
            all_metrics.append(metrics)

            cer_str = f"{metrics['cer']:.1%}" if metrics["cer"] == metrics["cer"] else "N/A"
            wer_str = f"{metrics['wer']:.1%}" if metrics["wer"] == metrics["wer"] else "N/A"
            logger.info(
                f"  {model_name} → {pdf_path.name}: "
                f"CER={cer_str}, WER={wer_str}, Domain F1={metrics['domain_f1']:.3f}, "
                f"Tempo={metrics['elapsed_seconds']:.1f}s"
            )
            pct = task_idx / total_tasks * 100
            print(f"{_GREEN}  [{pct:5.1f}%] {model_name} → {pdf_path.name} concluído{_RESET}")

    if args.extract_only:
        print()
        print("=" * 60)
        print("EXTRAÇÃO OCR CONCLUÍDA")
        print("=" * 60)
        print(f"Modelos executados: {list(selected_models.keys())}")
        print(f"PDFs processados:   {len(pdf_files)}")
        print(f"Textos salvos em:   {results_dir}/")
        print("=" * 60)
        return

    if not all_metrics:
        logger.error("Nenhum resultado obtido. Verifique se os PDFs e modelos estão configurados corretamente.")
        sys.exit(1)

    # Aggregate and export
    aggregated = _aggregate_by_model(all_metrics)
    _export_results(all_metrics, aggregated, results_dir, timestamp)
    _print_final_summary(aggregated, len(pdf_files), timestamp)


if __name__ == "__main__":
    main()


# ============================================================
# DEPENDÊNCIAS
# ============================================================
# Tesseract:     pip install pytesseract pdf2image opencv-python-headless Pillow numpy
#                + sudo apt-get install tesseract-ocr tesseract-ocr-por
# EasyOCR:       pip install easyocr pdf2image
# docTR:         pip install "python-doctr[torch]" pdf2image
# unstructured:  pip install "unstructured[pdf]"
# Chandra:       pip install chandra-ocr  (se disponível; verificar repositório oficial)
# PaddleOCR:     pip install paddlepaddle paddleocr pdf2image
# Métricas:      pip install jiwer
# Progresso:     pip install tqdm  (opcional)
