"""
Validate and clean collected documents (DECEA, LexML, etc.), generating
quality reports and optionally saving cleaned versions to a versioned directory.

Usage:
    python -m scripts.validate_data
    python -m scripts.validate_data --data-dir data/decea --clean --output-dir data/cleaned/v1
    python -m scripts.validate_data --data-dir data/lexml --report-only
    python -m scripts.validate_data --data-dir data/lexml --clean
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.text_cleaner import TextCleaner, CleaningStats
from pipeline.quality import QualityValidator, QualityReport, QualityLevel


def load_documents(data_dir: str) -> list[dict]:
    """Load all JSON documents from a directory."""
    docs = []
    data_path = Path(data_dir)
    for fpath in sorted(data_path.glob("*.json")):
        with open(fpath, 'r', encoding='utf-8') as f:
            doc = json.load(f)
        doc['_source_path'] = str(fpath)
        doc['_filename'] = fpath.name
        docs.append(doc)
    return docs


def run_validation(
    data_dir: str,
    output_dir: str | None = None,
    do_clean: bool = False,
    report_only: bool = False,
) -> dict:
    """
    Validate all documents and optionally clean + save them.

    Returns:
        Summary dict with counts and reports
    """
    cleaner = TextCleaner()
    validator = QualityValidator()

    docs = load_documents(data_dir)
    logger.info(f"Loaded {len(docs)} documents from {data_dir}")

    reports: list[QualityReport] = []
    cleaning_stats_list: list[CleaningStats] = []
    level_counts: Counter = Counter()
    cleaned_docs: list[dict] = []

    for doc in docs:
        doc_id = doc.get('slug') or doc.get('urn') or doc.get('_filename', 'unknown')
        content = doc.get('content', '')

        # Validate BEFORE cleaning
        report_before = validator.validate(content, doc_id=f"{doc_id} (raw)")

        if do_clean and report_before.level != QualityLevel.GARBAGE:
            cleaned_text, stats = cleaner.clean(content, doc_id=doc_id)
            cleaning_stats_list.append(stats)

            report_after = validator.validate(cleaned_text, doc_id=doc_id)
            reports.append(report_after)
            level_counts[report_after.level] += 1

            if not report_only:
                cleaned_doc = doc.copy()
                cleaned_doc.pop('_source_path', None)
                cleaned_doc.pop('_filename', None)
                cleaned_doc['content'] = cleaned_text
                cleaned_doc['_quality'] = {
                    'level': report_after.level.value,
                    'alpha_ratio': round(report_after.alpha_ratio, 4),
                    'chars_before': stats.original_chars,
                    'chars_after': stats.cleaned_chars,
                    'reduction_pct': round(stats.reduction_pct, 2),
                }
                cleaned_docs.append((doc['_filename'], cleaned_doc))
        else:
            reports.append(report_before)
            level_counts[report_before.level] += 1

            if report_before.level == QualityLevel.GARBAGE:
                logger.warning(
                    f"SKIPPED {doc_id}: {report_before.reject_reason}"
                )

    # Print report
    print("\n" + "=" * 80)
    print("RELATÓRIO DE QUALIDADE DOS DOCUMENTOS")
    print("=" * 80)
    print(f"Diretório: {data_dir}")
    print(f"Total: {len(docs)} documentos")
    print(f"Data: {datetime.now().isoformat()}")
    print()

    for level in QualityLevel:
        count = level_counts.get(level, 0)
        label = level.value.upper()
        bar = "█" * count
        print(f"  {label:<8} {count:>3} {bar}")

    print()
    print("-" * 80)

    for level in QualityLevel:
        level_reports = [r for r in reports if r.level == level]
        if not level_reports:
            continue
        print(f"\n{'─'*3} {level.value.upper()} ({len(level_reports)}) {'─'*40}")
        for r in level_reports:
            print(r.summary())

    if cleaning_stats_list:
        total_before = sum(s.original_chars for s in cleaning_stats_list)
        total_after = sum(s.cleaned_chars for s in cleaning_stats_list)
        total_ctrl = sum(s.control_chars_removed for s in cleaning_stats_list)
        total_garbled = sum(s.garbled_lines_removed for s in cleaning_stats_list)
        total_hdrs = sum(s.headers_removed for s in cleaning_stats_list)
        total_pg = sum(s.page_numbers_removed for s in cleaning_stats_list)

        print(f"\n{'='*80}")
        print("RESUMO DA LIMPEZA")
        print(f"{'='*80}")
        total_disclaimers = sum(s.legal_disclaimers_removed for s in cleaning_stats_list)

        print(f"  Documentos limpos:        {len(cleaning_stats_list)}")
        print(f"  Chars antes:              {total_before:>12,}")
        print(f"  Chars depois:             {total_after:>12,}")
        print(f"  Redução:                  {total_before - total_after:>12,} "
              f"({(total_before - total_after) / total_before * 100:.1f}%)")
        print(f"  Control chars removidos:  {total_ctrl:>12,}")
        print(f"  Disclaimers removidos:    {total_disclaimers:>12,}")
        print(f"  Linhas garbled removidas: {total_garbled:>12,}")
        print(f"  Headers removidos:        {total_hdrs:>12,}")
        print(f"  Page numbers removidos:   {total_pg:>12,}")

    # Save cleaned docs
    if output_dir and cleaned_docs:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for filename, cleaned_doc in cleaned_docs:
            with open(out_path / filename, 'w', encoding='utf-8') as f:
                json.dump(cleaned_doc, f, ensure_ascii=False, indent=2)

        logger.success(f"Saved {len(cleaned_docs)} cleaned documents to {output_dir}")

        # Save quality report JSON
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'source_dir': data_dir,
            'output_dir': output_dir,
            'total_docs': len(docs),
            'counts': {level.value: level_counts.get(level, 0) for level in QualityLevel},
            'documents': [
                {
                    'doc_id': r.doc_id,
                    'level': r.level.value,
                    'total_chars': r.total_chars,
                    'alpha_ratio': round(r.alpha_ratio, 4),
                    'control_ratio': round(r.control_ratio, 4),
                    'reject_reason': r.reject_reason,
                    'warnings': r.warnings,
                }
                for r in reports
            ],
        }
        report_path = out_path / '_quality_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        logger.success(f"Quality report saved to {report_path}")

    return {
        'total': len(docs),
        'counts': dict(level_counts),
        'reports': reports,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate and optionally clean collected documents (DECEA, LexML, etc.)"
    )
    parser.add_argument(
        '--data-dir',
        default='data/decea',
        help='Directory with JSON documents to validate',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Directory to save cleaned documents (e.g. data/cleaned/v1)',
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Apply text cleaning before validation',
    )
    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Only print validation report, do not save files',
    )

    args = parser.parse_args()

    if args.clean and not args.output_dir and not args.report_only:
        args.output_dir = f"data/cleaned/v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Auto-generated output dir: {args.output_dir}")

    run_validation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        do_clean=args.clean,
        report_only=args.report_only,
    )


if __name__ == '__main__':
    main()
