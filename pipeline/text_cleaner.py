"""
Text cleaning pipeline for aviation regulation documents.

Applies heuristic-based transformations to remove noise from
PDF-extracted and web-scraped text before embedding and indexing.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger


_VOWELS = frozenset('aeiouáéíóúàâêôãõAEIOUÁÉÍÓÚÀÂÊÔÃÕ')


@dataclass
class CleaningStats:
    """Tracks what was removed/changed during cleaning."""
    original_chars: int = 0
    cleaned_chars: int = 0
    control_chars_removed: int = 0
    legal_disclaimers_removed: int = 0
    headers_removed: int = 0
    page_numbers_removed: int = 0
    garbled_lines_removed: int = 0
    excessive_whitespace_collapsed: int = 0
    unicode_normalized: bool = False

    @property
    def chars_removed(self) -> int:
        return self.original_chars - self.cleaned_chars

    @property
    def reduction_pct(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return (self.chars_removed / self.original_chars) * 100


# Boilerplate injected by Senado/LexML web portal into scraped content.
# Removed unconditionally (unlike institutional headers, which require repetition).
_LEGAL_DISCLAIMER_PATTERNS = [
    # "[Detalhes da Norma]" section label from the LexML portal
    re.compile(r'^\s*\[Detalhes\s+da\s+Norma\]\s*$', re.MULTILINE | re.IGNORECASE),
    # Senado/Planalto disclaimer line present in every HTML-scraped document
    re.compile(
        r'^[^\n]*Este\s+texto\s+não\s+substitui\s+o\s+original\s+publicado\s+no\s+Diário\s+Oficial[^\n]*$',
        re.MULTILINE | re.IGNORECASE,
    ),
]

# Patterns for institutional headers found across DECEA ICA documents
_HEADER_PATTERNS = [
    re.compile(
        r'^\s*MINISTÉRIO\s+DA\s+DEFESA\s*$',
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(
        r'^\s*COMANDO\s+DA\s+AERONÁUTICA\s*$',
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(
        r'^\s*DEPARTAMENTO\s+DE\s+CONTROLE\s+DO\s+ESPAÇO\s+AÉREO\s*$',
        re.MULTILINE | re.IGNORECASE,
    ),
]

# Page-level ICA identifier repeated on every page (e.g. "ICA 63-12/2021")
_ICA_PAGE_HEADER = re.compile(
    r'^\s*ICA\s+\d+[-–]\d+/\d{4}\s*$',
    re.MULTILINE,
)

# Page numbers in "N/M" format on their own line (e.g. "10/26")
_PAGE_NUMBER_LINE = re.compile(
    r'^\s*(\d{1,3})\s*/\s*(\d{1,3})\s*$',
    re.MULTILINE,
)

# Table-of-contents dotted lines (e.g. "CAPÍTULO I...............7")
_TOC_DOTS = re.compile(
    r'^.*\.{5,}\s*\d+\s*$',
    re.MULTILINE,
)

# Consecutive blank lines (3+)
_EXCESSIVE_NEWLINES = re.compile(r'\n{3,}')

# Multiple spaces
_EXCESSIVE_SPACES = re.compile(r'[ \t]{2,}')

# Lines that are just underscores/dashes (separators)
_SEPARATOR_LINE = re.compile(
    r'^\s*[_\-=]{5,}\s*$',
    re.MULTILINE,
)


class TextCleaner:
    """
    Cleans text from aviation regulation documents (PDF-extracted and web-scraped).

    The cleaning pipeline applies transformations in order:
    1. Unicode normalization (NFKC)
    2. Control character replacement (\\x03 → space, others removed)
    3. Legal portal disclaimers removal (LexML/Senado/Planalto boilerplate)
    4. Garbled text removal (ROT-3 / font-encoding artifacts)
    5. Header/footer removal (repeated institutional headers)
    6. Page number removal
    7. TOC dotted-line removal
    8. Separator line removal
    9. Smart quote/dash standardization
    10. Line-break hyphenation repair
    11. Whitespace normalization
    """

    def __init__(
        self,
        remove_legal_disclaimers: bool = True,
        remove_headers: bool = True,
        remove_page_numbers: bool = True,
        remove_toc_dots: bool = True,
        remove_separators: bool = True,
        remove_garbled: bool = True,
        fix_hyphenation: bool = True,
        normalize_unicode: bool = True,
        normalize_quotes: bool = True,
    ):
        self.remove_legal_disclaimers = remove_legal_disclaimers
        self.remove_headers = remove_headers
        self.remove_page_numbers = remove_page_numbers
        self.remove_toc_dots = remove_toc_dots
        self.remove_separators = remove_separators
        self.remove_garbled = remove_garbled
        self.fix_hyphenation = fix_hyphenation
        self.normalize_unicode = normalize_unicode
        self.normalize_quotes = normalize_quotes

    def clean(self, text: str, doc_id: str = "") -> tuple[str, CleaningStats]:
        """
        Apply the full cleaning pipeline to a text.

        Returns:
            Tuple of (cleaned_text, stats)
        """
        stats = CleaningStats(original_chars=len(text))

        if not text:
            return text, stats

        if self.normalize_unicode:
            text = self._normalize_unicode(text)
            stats.unicode_normalized = True

        text, ctrl_removed = self._remove_control_chars(text)
        stats.control_chars_removed = ctrl_removed

        if self.remove_legal_disclaimers:
            text, n = self._remove_legal_disclaimers(text)
            stats.legal_disclaimers_removed = n

        if self.remove_garbled:
            text, garbled_removed = self._remove_garbled_text(text)
            stats.garbled_lines_removed = garbled_removed

        if self.remove_headers:
            text, n = self._remove_headers(text)
            stats.headers_removed = n

        if self.remove_page_numbers:
            text, n = self._remove_page_numbers(text)
            stats.page_numbers_removed = n

        if self.remove_toc_dots:
            text = self._remove_toc_dots(text)

        if self.remove_separators:
            text = self._remove_separators(text)

        if self.normalize_quotes:
            text = self._normalize_quotes(text)

        if self.fix_hyphenation:
            text = self._fix_hyphenation(text)

        text = self._normalize_whitespace(text, stats)

        stats.cleaned_chars = len(text)

        if doc_id:
            logger.debug(
                f"Cleaned {doc_id}: {stats.original_chars} → {stats.cleaned_chars} chars "
                f"(-{stats.reduction_pct:.1f}%, ctrl={stats.control_chars_removed}, "
                f"disclaimers={stats.legal_disclaimers_removed}, "
                f"garbled={stats.garbled_lines_removed}, "
                f"hdrs={stats.headers_removed}, pgnums={stats.page_numbers_removed})"
            )

        return text, stats

    # ── Individual transformations ────────────────────────────────

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """NFKC normalization: unifies equivalent Unicode representations."""
        return unicodedata.normalize('NFKC', text)

    @staticmethod
    def _remove_control_chars(text: str) -> tuple[str, int]:
        """Remove control characters except newline, carriage return, tab.

        ETX (\\x03) is replaced with space instead of deleted, because some
        DECEA PDFs use it as a word separator in custom-encoded fonts.
        Dropping it would concatenate garbled words, making them harder to
        detect and worse for embeddings.
        """
        result = []
        removed = 0
        for ch in text:
            if ch == '\x03':
                result.append(' ')
                removed += 1
            else:
                cat = unicodedata.category(ch)
                if cat.startswith('C') and ch not in '\n\r\t':
                    removed += 1
                else:
                    result.append(ch)
        return ''.join(result), removed

    @staticmethod
    def _is_garbled_word(word: str) -> bool:
        """Check if a word looks like a font-encoding artifact (e.g. ROT-3).

        Uses a two-pronged heuristic:
        1. Extremely low vowel ratio (< 10%) is always suspicious.
        2. If applying ROT(-3) to ASCII letters *significantly increases*
           the vowel ratio, the word was likely encoded — original vowels
           (A,E,I,O,U) shift to consonant positions (D,H,L,R,X), but some
           original consonants (B,F,L,R,V) shift to vowel positions
           (E,I,O,U,Y), giving the encoded word a non-zero but still
           below-normal vowel ratio that a simple threshold would miss.
        """
        alpha_chars = [c for c in word if c.isalpha()]
        if len(alpha_chars) < 5:
            return False

        vowel_orig = sum(1 for c in alpha_chars if c in _VOWELS)
        ratio_orig = vowel_orig / len(alpha_chars)

        if ratio_orig < 0.10:
            return True

        decoded_vowels = 0
        for c in alpha_chars:
            if 'A' <= c <= 'Z':
                d = chr((ord(c) - ord('A') - 3) % 26 + ord('A'))
            elif 'a' <= c <= 'z':
                d = chr((ord(c) - ord('a') - 3) % 26 + ord('a'))
            else:
                d = c
            if d in _VOWELS:
                decoded_vowels += 1

        ratio_decoded = decoded_vowels / len(alpha_chars)

        return (ratio_decoded - ratio_orig) > 0.12 and ratio_decoded > 0.35

    @classmethod
    def _remove_garbled_text(cls, text: str) -> tuple[str, int]:
        """Detect and remove lines with font-encoding garbled text.

        Some DECEA PDFs use custom Type1/Type3 fonts without proper
        ToUnicode CMaps, producing ROT-3 shifted text (e.g. "TXH" instead
        of "QUE").  These lines are administrative boilerplate (portaria
        headers) that add noise to embeddings.

        A line is flagged when it contains >= 2 garbled words (4+ chars)
        AND they make up >= 50% of the line's 4+ char words.  Requiring
        two garbled words avoids false positives on legitimate English
        meteorological terms (HUNDRED, DRIZZLE, SHALLOW, etc.) which
        individually share vowel-pattern traits with ROT-3 text.
        """
        lines = text.split('\n')
        kept: list[str] = []
        removed = 0

        for line in lines:
            words = re.findall(r'[a-zA-ZÀ-ÿ]{3,}', line)
            candidate_words = [w for w in words if len(w) >= 5]

            if len(candidate_words) >= 2:
                garbled_count = sum(
                    1 for w in candidate_words if cls._is_garbled_word(w)
                )
                if garbled_count >= 2 and garbled_count / len(candidate_words) >= 0.5:
                    removed += 1
                    continue

            kept.append(line)

        return '\n'.join(kept), removed

    @staticmethod
    def _remove_legal_disclaimers(text: str) -> tuple[str, int]:
        """Remove boilerplate lines injected by legal portals (Senado, LexML, Planalto).

        Unlike institutional headers, these are always noise regardless of how
        many times they appear, so they are removed unconditionally.
        """
        total_removed = 0
        for pattern in _LEGAL_DISCLAIMER_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                text = pattern.sub('', text)
                total_removed += len(matches)
        return text, total_removed

    @staticmethod
    def _remove_headers(text: str) -> tuple[str, int]:
        """Remove repeated institutional headers."""
        total_removed = 0

        for pattern in _HEADER_PATTERNS:
            matches = pattern.findall(text)
            if len(matches) > 1:
                text = pattern.sub('', text)
                total_removed += len(matches)

        ica_matches = _ICA_PAGE_HEADER.findall(text)
        if len(ica_matches) > 1:
            text = _ICA_PAGE_HEADER.sub('', text)
            total_removed += len(ica_matches)

        return text, total_removed

    @staticmethod
    def _remove_page_numbers(text: str) -> tuple[str, int]:
        """Remove standalone page numbers like '10/26'."""

        def _is_page_number(match: re.Match) -> bool:
            num, total = int(match.group(1)), int(match.group(2))
            return 1 <= num <= total and total >= 2

        count = 0
        def _replacer(match: re.Match) -> str:
            nonlocal count
            if _is_page_number(match):
                count += 1
                return ''
            return match.group(0)

        text = _PAGE_NUMBER_LINE.sub(_replacer, text)
        return text, count

    @staticmethod
    def _remove_toc_dots(text: str) -> str:
        """Remove table-of-contents lines with dotted leaders."""
        return _TOC_DOTS.sub('', text)

    @staticmethod
    def _remove_separators(text: str) -> str:
        """Remove lines that are just dashes or underscores."""
        return _SEPARATOR_LINE.sub('', text)

    @staticmethod
    def _normalize_quotes(text: str) -> str:
        """Standardize curly quotes and dashes to their ASCII/standard forms."""
        replacements = {
            '\u201c': '"',  # "
            '\u201d': '"',  # "
            '\u2018': "'",  # '
            '\u2019': "'",  # '
            '\u00b3': '"',  # ³ (used as opening quote in some DECEA docs)
            '\u00b4': '"',  # ´ (used as closing quote in some DECEA docs)
            '\u00ad': '-',  # soft hyphen
            '\u2013': '–',  # en dash (keep as-is, it's valid)
            '\u2014': '—',  # em dash (keep as-is, it's valid)
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def _fix_hyphenation(text: str) -> str:
        """
        Rejoin words split by line-break hyphenation.

        Only joins when the second part starts with lowercase,
        avoiding false positives on compound words like "controlador-piloto"
        or acronyms like "ADS-B".
        """
        return re.sub(
            r'(\w{3,})-\s*\n\s*([a-záéíóúàâêôãõçü]\w{2,})',
            lambda m: m.group(1) + m.group(2),
            text,
        )

    @staticmethod
    def _normalize_whitespace(text: str, stats: CleaningStats) -> str:
        """Collapse excessive whitespace while preserving paragraph structure."""
        original_len = len(text)

        text = _EXCESSIVE_NEWLINES.sub('\n\n', text)
        text = _EXCESSIVE_SPACES.sub(' ', text)

        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)

        text = text.strip()

        stats.excessive_whitespace_collapsed = original_len - len(text)
        return text
