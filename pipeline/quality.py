"""
Document quality validation for aviation regulation documents.

Classifies documents by quality level and provides metrics
to decide whether a document should be indexed.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger


class QualityLevel(str, Enum):
    """Quality classification for a document."""
    GARBAGE = "garbage"
    LOW = "low"
    MEDIUM = "medium"
    GOOD = "good"


@dataclass
class QualityReport:
    """Detailed quality metrics for a document."""
    doc_id: str
    total_chars: int = 0
    total_words: int = 0
    total_lines: int = 0
    empty_lines: int = 0

    alpha_count: int = 0
    digit_count: int = 0
    control_char_count: int = 0

    header_repetitions: int = 0
    page_number_count: int = 0
    noise_line_count: int = 0
    short_line_count: int = 0
    toc_dot_lines: int = 0

    level: QualityLevel = QualityLevel.GOOD
    reject_reason: Optional[str] = None
    warnings: list = field(default_factory=list)

    @property
    def alpha_ratio(self) -> float:
        return self.alpha_count / self.total_chars if self.total_chars > 0 else 0.0

    @property
    def control_ratio(self) -> float:
        return self.control_char_count / self.total_chars if self.total_chars > 0 else 0.0

    @property
    def noise_line_ratio(self) -> float:
        return self.noise_line_count / self.total_lines if self.total_lines > 0 else 0.0

    @property
    def empty_line_ratio(self) -> float:
        return self.empty_lines / self.total_lines if self.total_lines > 0 else 0.0

    @property
    def is_acceptable(self) -> bool:
        return self.level in (QualityLevel.MEDIUM, QualityLevel.GOOD)

    def summary(self) -> str:
        lines = [
            f"[{self.level.value.upper()}] {self.doc_id}",
            f"  chars={self.total_chars:,}  words={self.total_words:,}  "
            f"α={self.alpha_ratio:.1%}  ctrl={self.control_ratio:.1%}",
        ]
        if self.reject_reason:
            lines.append(f"  REJECTED: {self.reject_reason}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return '\n'.join(lines)


class QualityValidator:
    """
    Validates document quality using heuristic metrics.

    Thresholds:
        - GARBAGE: control_ratio > 10% OR alpha_ratio < 15%
        - LOW: alpha_ratio < 40% OR total_chars < 200
        - MEDIUM: alpha_ratio < 65%
        - GOOD: everything else
    """

    def __init__(
        self,
        min_chars: int = 200,
        min_alpha_ratio: float = 0.15,
        max_control_ratio: float = 0.10,
        garbage_alpha_threshold: float = 0.15,
        low_alpha_threshold: float = 0.40,
        medium_alpha_threshold: float = 0.65,
    ):
        self.min_chars = min_chars
        self.min_alpha_ratio = min_alpha_ratio
        self.max_control_ratio = max_control_ratio
        self.garbage_alpha_threshold = garbage_alpha_threshold
        self.low_alpha_threshold = low_alpha_threshold
        self.medium_alpha_threshold = medium_alpha_threshold

    def validate(self, text: str, doc_id: str = "") -> QualityReport:
        """
        Analyze text quality and return a detailed report.

        Args:
            text: The document text to validate
            doc_id: Identifier for logging

        Returns:
            QualityReport with metrics and classification
        """
        report = QualityReport(doc_id=doc_id)

        if not text:
            report.level = QualityLevel.GARBAGE
            report.reject_reason = "Empty content"
            return report

        report.total_chars = len(text)
        report.total_words = len(text.split())

        lines = text.split('\n')
        report.total_lines = len(lines)
        report.empty_lines = sum(1 for l in lines if not l.strip())

        self._count_chars(text, report)
        self._count_noise_patterns(text, lines, report)
        self._classify(report)

        return report

    def _count_chars(self, text: str, report: QualityReport) -> None:
        for ch in text:
            if ch.isalpha():
                report.alpha_count += 1
            elif ch.isdigit():
                report.digit_count += 1
            else:
                cat = unicodedata.category(ch)
                if cat.startswith('C') and ch not in '\n\r\t':
                    report.control_char_count += 1

    def _count_noise_patterns(
        self, text: str, lines: list[str], report: QualityReport
    ) -> None:
        report.header_repetitions = len(
            re.findall(r'MINISTÉRIO\s+DA\s+DEFESA', text, re.IGNORECASE)
        )
        report.page_number_count = len(
            re.findall(r'(?:^|\n)\s*\d{1,3}\s*/\s*\d{1,3}\s*(?:\n|$)', text)
        )
        report.noise_line_count = sum(
            1 for l in lines
            if l.strip() and not any(c.isalpha() for c in l)
        )
        report.short_line_count = sum(
            1 for l in lines if 0 < len(l.strip()) < 5
        )
        report.toc_dot_lines = sum(
            1 for l in lines if l.count('.') > 20
        )

    def _classify(self, report: QualityReport) -> None:
        if report.total_chars < self.min_chars:
            report.level = QualityLevel.LOW
            report.reject_reason = f"Content too short ({report.total_chars} chars)"
            return

        if report.control_ratio > self.max_control_ratio:
            report.level = QualityLevel.GARBAGE
            report.reject_reason = (
                f"Excessive control characters ({report.control_ratio:.1%}). "
                "Likely failed PDF extraction — needs OCR re-extraction."
            )
            return

        if report.alpha_ratio < self.garbage_alpha_threshold:
            report.level = QualityLevel.GARBAGE
            report.reject_reason = (
                f"Alpha ratio too low ({report.alpha_ratio:.1%}). "
                "Content is not readable text."
            )
            return

        if report.alpha_ratio < self.low_alpha_threshold:
            report.level = QualityLevel.LOW
            report.reject_reason = (
                f"Low text quality (α={report.alpha_ratio:.1%}). "
                "Mostly non-textual content (tables, codes, numbers)."
            )

        elif report.alpha_ratio < self.medium_alpha_threshold:
            report.level = QualityLevel.MEDIUM
        else:
            report.level = QualityLevel.GOOD

        # Warnings (informational, don't affect classification)
        if report.header_repetitions > 3:
            report.warnings.append(
                f"{report.header_repetitions} repeated institutional headers"
            )
        if report.page_number_count > 5:
            report.warnings.append(
                f"{report.page_number_count} embedded page numbers"
            )
        if report.noise_line_ratio > 0.3:
            report.warnings.append(
                f"{report.noise_line_ratio:.0%} of lines have no alphabetic content"
            )
        if report.toc_dot_lines > 10:
            report.warnings.append(
                f"{report.toc_dot_lines} table-of-contents lines"
            )
