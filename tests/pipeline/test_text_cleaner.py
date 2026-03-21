"""Tests for pipeline.text_cleaner — TextCleaner and CleaningStats."""

import pytest

from pipeline.text_cleaner import TextCleaner, CleaningStats


# ---------------------------------------------------------------------------
# CleaningStats
# ---------------------------------------------------------------------------

class TestCleaningStats:

    def test_chars_removed_property(self):
        stats = CleaningStats(original_chars=1000, cleaned_chars=800)
        assert stats.chars_removed == 200

    def test_reduction_pct_property(self):
        stats = CleaningStats(original_chars=1000, cleaned_chars=750)
        assert stats.reduction_pct == pytest.approx(25.0)

    def test_reduction_pct_zero_when_no_original(self):
        stats = CleaningStats(original_chars=0, cleaned_chars=0)
        assert stats.reduction_pct == 0.0

    def test_default_fields_are_zero(self):
        stats = CleaningStats()
        assert stats.control_chars_removed == 0
        assert stats.legal_disclaimers_removed == 0
        assert stats.headers_removed == 0
        assert stats.page_numbers_removed == 0
        assert stats.garbled_lines_removed == 0


# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------

class TestNormalizeUnicode:

    def test_nfkc_ligature(self):
        # fi ligature (U+FB01) → plain fi
        cleaner = TextCleaner()
        text, _ = cleaner.clean("\ufb01le")
        assert text == "file"

    def test_fullwidth_to_ascii(self):
        cleaner = TextCleaner()
        text, _ = cleaner.clean("\uff21\uff22\uff23")  # Ａ Ｂ Ｃ → ABC
        assert text == "ABC"

    def test_disabled_unicode_normalization(self):
        cleaner = TextCleaner(normalize_unicode=False)
        original = "\ufb01le"
        text, _ = cleaner.clean(original)
        assert "\ufb01" in text


# ---------------------------------------------------------------------------
# Control character removal
# ---------------------------------------------------------------------------

class TestRemoveControlChars:

    def test_removes_bom(self):
        cleaner = TextCleaner()
        text, stats = cleaner.clean("before\ufeffafter")
        assert "\ufeff" not in text
        assert stats.control_chars_removed >= 1

    def test_replaces_etx_with_space(self):
        # ETX (\\x03) is used as word separator in some DECEA PDFs
        cleaner = TextCleaner()
        text, _ = cleaner.clean("word\x03word")
        assert "word word" in text

    def test_preserves_newlines_tabs_carriage_return(self):
        cleaner = TextCleaner()
        original = "line1\nline2\ttabbed\r\n"
        text, stats = cleaner.clean(original)
        assert "\n" in text
        assert "\t" in text

    def test_removes_null_byte(self):
        cleaner = TextCleaner()
        text, stats = cleaner.clean("abc\x00def")
        assert "\x00" not in text
        assert stats.control_chars_removed >= 1


# ---------------------------------------------------------------------------
# Legal disclaimer removal (LexML / Senado / Planalto boilerplate)
# ---------------------------------------------------------------------------

class TestRemoveLegalDisclaimers:

    def _content(self, body: str) -> str:
        """Wraps body with enough real text to pass the quality gate."""
        filler = (
            "Artigo primeiro estabelece os procedimentos aeronáuticos aplicáveis "
            "às aeronaves que operam no espaço aéreo brasileiro conforme normas vigentes. "
        ) * 5
        return body + "\n" + filler

    def test_removes_lexml_section_header(self):
        cleaner = TextCleaner()
        text, stats = cleaner.clean(self._content("[Detalhes da Norma]\n"))
        assert "[Detalhes da Norma]" not in text
        assert stats.legal_disclaimers_removed >= 1

    def test_removes_lexml_section_header_case_insensitive(self):
        cleaner = TextCleaner()
        text, stats = cleaner.clean(self._content("[DETALHES DA NORMA]\n"))
        assert "[DETALHES DA NORMA]" not in text
        assert stats.legal_disclaimers_removed >= 1

    def test_removes_senado_disclaimer(self):
        cleaner = TextCleaner()
        disclaimer = "Este texto não substitui o original publicado no Diário Oficial."
        text, stats = cleaner.clean(self._content(disclaimer + "\n"))
        assert "Este texto não substitui" not in text
        assert stats.legal_disclaimers_removed >= 1

    def test_removes_both_boilerplate_lines(self):
        cleaner = TextCleaner()
        boilerplate = (
            "[Detalhes da Norma]\n"
            "Este texto não substitui o original publicado no Diário Oficial.\n"
        )
        text, stats = cleaner.clean(self._content(boilerplate))
        assert "[Detalhes da Norma]" not in text
        assert "Este texto não substitui" not in text
        assert stats.legal_disclaimers_removed == 2

    def test_removes_disclaimer_with_surrounding_context(self):
        """Disclaimer embedded mid-content should still be removed."""
        cleaner = TextCleaner()
        filler = "Regulamento aeronáutico aplicável. " * 10
        content = filler + "\nEste texto não substitui o original publicado no Diário Oficial.\n" + filler
        text, stats = cleaner.clean(content)
        assert "Este texto não substitui" not in text
        assert stats.legal_disclaimers_removed >= 1

    def test_does_not_remove_when_disabled(self):
        cleaner = TextCleaner(remove_legal_disclaimers=False)
        disclaimer = "[Detalhes da Norma]\n"
        text, stats = cleaner.clean(self._content(disclaimer))
        assert "[Detalhes da Norma]" in text
        assert stats.legal_disclaimers_removed == 0

    def test_typical_lexml_content_cleaned(self):
        """End-to-end test with realistic LexML scraper output."""
        cleaner = TextCleaner()
        raw = (
            "[Detalhes da Norma]\n"
            "Este texto não substitui o original publicado no Diário Oficial.\n"
            "\n"
            "DECRETO N. 20.015 – DE 21 DE MAIO DE 1931\n"
            "Aprova o regulamento para a Aviação Naval.\n"
            "O Chefe do Governo Provisório, usando das atribuições que lhe confere "
            "o art. primeiro, resolve aprovar e mandar executar o regulamento para a "
            "Aviação Naval, que a este acompanha, assinado pelo ministro da Marinha.\n"
        )
        text, stats = cleaner.clean(raw)
        assert "[Detalhes da Norma]" not in text
        assert "Este texto não substitui" not in text
        assert "DECRETO N. 20.015" in text
        assert stats.legal_disclaimers_removed == 2


# ---------------------------------------------------------------------------
# Institutional header removal (DECEA)
# ---------------------------------------------------------------------------

class TestRemoveHeaders:

    def _decea_content(self, repeat: int = 3) -> str:
        header = "MINISTÉRIO DA DEFESA\n"
        body = (
            "Instrução do Comando da Aeronáutica estabelece procedimentos para "
            "operação de aeronaves no espaço aéreo brasileiro conforme normas vigentes. "
        ) * 4
        return (header + body + "\n") * repeat

    def test_removes_repeated_decea_header(self):
        cleaner = TextCleaner()
        text, stats = cleaner.clean(self._decea_content(repeat=3))
        assert stats.headers_removed > 0

    def test_preserves_single_occurrence(self):
        """A header that appears only once is legitimate content — keep it."""
        cleaner = TextCleaner()
        body = (
            "MINISTÉRIO DA DEFESA\n"
            "Instrução do Comando da Aeronáutica estabelece procedimentos para "
            "operação de aeronaves no espaço aéreo brasileiro conforme normas vigentes "
            "aplicáveis a todos os operadores certificados que atuam em território nacional. "
        ) * 2
        text, stats = cleaner.clean(body)
        # Only removes if repeated > 1 time, and here "MINISTÉRIO DA DEFESA" appears
        # exactly twice (once per repeat), which is > 1, so it IS removed
        # Use a unique non-repeating header to test preservation
        unique_body = (
            "MINISTÉRIO DA DEFESA\n"
            "Art 1 Esta instrução estabelece os procedimentos para operação de aeronaves "
            "no espaço aéreo brasileiro conforme as normas vigentes do Comando da Aeronáutica "
            "e regulamentações do DECEA aplicáveis a todos os operadores certificados. "
        )
        text2, stats2 = cleaner.clean(unique_body)
        assert stats2.headers_removed == 0

    def test_removes_repeated_ica_page_header(self):
        cleaner = TextCleaner()
        body = (
            "Instrução do Comando da Aeronáutica que estabelece procedimentos "
            "para operação de aeronaves no espaço aéreo brasileiro aplicáveis "
            "a todos os operadores e pilotos certificados em território nacional.\n"
        )
        # ICA identifier on its own line, repeated across pages
        content = ("ICA 63-12/2021\n" + body) * 3
        text, stats = cleaner.clean(content)
        assert stats.headers_removed > 0


# ---------------------------------------------------------------------------
# Page number removal
# ---------------------------------------------------------------------------

class TestRemovePageNumbers:

    def _with_body(self, page_line: str) -> str:
        body = (
            "Regulamento aeronáutico que estabelece normas operacionais para aeronaves "
            "que operam no espaço aéreo brasileiro conforme as regulamentações vigentes "
            "do Comando da Aeronáutica e do DECEA aplicáveis a todos os operadores. "
        )
        return body + "\n" + page_line + "\n" + body

    def test_removes_valid_page_number(self):
        cleaner = TextCleaner()
        text, stats = cleaner.clean(self._with_body("10/26"))
        assert stats.page_numbers_removed == 1

    def test_removes_page_number_at_start(self):
        cleaner = TextCleaner()
        text, stats = cleaner.clean(self._with_body("1/50"))
        assert stats.page_numbers_removed == 1

    def test_preserves_fractions(self):
        """N/M where N > M should NOT be removed (it's a fraction, not a page number)."""
        cleaner = TextCleaner()
        text, stats = cleaner.clean(self._with_body("3/2"))
        assert stats.page_numbers_removed == 0

    def test_preserves_single_total(self):
        """Single-page documents (1/1) should not be treated as page numbers."""
        cleaner = TextCleaner()
        text, stats = cleaner.clean(self._with_body("1/1"))
        # M must be >= 2 per the validation logic
        assert stats.page_numbers_removed == 0


# ---------------------------------------------------------------------------
# TOC dots removal
# ---------------------------------------------------------------------------

class TestRemoveTocDots:

    def test_removes_toc_dot_lines(self):
        cleaner = TextCleaner()
        body = (
            "Regulamento aeronáutico brasileiro para operação de aeronaves "
            "conforme normas do DECEA aplicáveis a todos os operadores certificados. "
        ) * 3
        toc_line = "CAPÍTULO I...................................7\n"
        content = body + "\n" + toc_line + "\n" + body
        text, _ = cleaner.clean(content)
        assert "CAPÍTULO I............" not in text

    def test_preserves_normal_dots(self):
        cleaner = TextCleaner()
        body = (
            "Regulamento aeronáutico que possui pontos finais no texto. "
            "Cada frase termina com um ponto. Isso é esperado. "
        ) * 5
        text, _ = cleaner.clean(body)
        assert "." in text


# ---------------------------------------------------------------------------
# Garbled text removal
# ---------------------------------------------------------------------------

class TestRemoveGarbledText:

    def test_removes_rot3_line(self):
        # "TXH" is the ROT-3 encoding of "QUE", "WRGRV" → "TODOS" etc.
        cleaner = TextCleaner()
        body = (
            "Instrução aeronáutica estabelece procedimentos para operação de aeronaves "
            "no espaço aéreo brasileiro conforme as normas vigentes aplicáveis a todos. "
        ) * 3
        garbled_line = "TXH WRGRV RV SLODWRV GHYHP FXPSULU DV QRUPDV\n"
        content = body + "\n" + garbled_line + "\n" + body
        text, stats = cleaner.clean(content)
        assert stats.garbled_lines_removed >= 1
        assert "TXH WRGRV" not in text

    def test_preserves_normal_portuguese_text(self):
        cleaner = TextCleaner()
        content = (
            "O regulamento estabelece os procedimentos para operação de aeronaves "
            "no espaço aéreo brasileiro conforme as normas vigentes do Comando da "
            "Aeronáutica aplicáveis a todos os operadores e pilotos certificados. "
        ) * 3
        _, stats = cleaner.clean(content)
        assert stats.garbled_lines_removed == 0

    def test_preserves_english_aviation_terms(self):
        """English terms like HUNDRED, DRIZZLE must not be flagged as garbled."""
        cleaner = TextCleaner()
        content = (
            "HUNDRED DRIZZLE SHALLOW BROKEN SCATTERED OVERCAST represent aviation "
            "weather terms used in aeronautical meteorology reports worldwide. "
            "Esses termos são utilizados nos relatórios meteorológicos aeronáuticos. "
        ) * 3
        _, stats = cleaner.clean(content)
        assert stats.garbled_lines_removed == 0


# ---------------------------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------------------------

class TestNormalizeWhitespace:

    def test_collapses_excessive_newlines(self):
        cleaner = TextCleaner()
        text, _ = cleaner.clean("line1\n\n\n\n\nline2")
        assert "\n\n\n" not in text
        assert "line1" in text and "line2" in text

    def test_collapses_excessive_spaces(self):
        cleaner = TextCleaner()
        text, _ = cleaner.clean("word   with    spaces")
        assert "word with spaces" in text

    def test_strips_leading_and_trailing(self):
        cleaner = TextCleaner()
        text, _ = cleaner.clean("   \n\nsome content here\n\n   ")
        assert text == text.strip()

    def test_strips_trailing_spaces_per_line(self):
        cleaner = TextCleaner()
        text, _ = cleaner.clean("line one   \nline two   \n")
        for line in text.split("\n"):
            assert line == line.rstrip()


# ---------------------------------------------------------------------------
# Hyphenation repair
# ---------------------------------------------------------------------------

class TestFixHyphenation:

    def test_rejoins_split_word(self):
        cleaner = TextCleaner()
        content = (
            "O regulamento estabelece que os procedimentos de opera-\nção devem "
            "ser seguidos por todos os operadores aeronáuticos certificados "
            "que atuam no espaço aéreo brasileiro conforme normas vigentes. "
        ) * 2
        text, _ = cleaner.clean(content)
        assert "operação" in text

    def test_preserves_compound_words(self):
        """Hyphenated compound like 'controlador-piloto' must NOT be joined."""
        cleaner = TextCleaner()
        content = (
            "O controlador-piloto coordena as operações aéreas no espaço aéreo "
            "brasileiro conforme os procedimentos estabelecidos pelo DECEA e "
            "aplicáveis a todos os voos realizados em território nacional. "
        ) * 3
        text, _ = cleaner.clean(content)
        assert "controlador-piloto" in text

    def test_disabled_hyphenation_fix(self):
        cleaner = TextCleaner(fix_hyphenation=False)
        content = (
            "O regulamento estabelece que os procedimentos de opera-\nção devem "
            "ser seguidos por todos os operadores aeronáuticos certificados. "
        ) * 2
        text, _ = cleaner.clean(content)
        assert "opera-" in text


# ---------------------------------------------------------------------------
# Quote normalization
# ---------------------------------------------------------------------------

class TestNormalizeQuotes:

    def test_normalizes_curly_double_quotes(self):
        cleaner = TextCleaner()
        text, _ = cleaner.clean('\u201cquote\u201d')
        assert '"quote"' in text

    def test_normalizes_curly_single_quotes(self):
        cleaner = TextCleaner()
        text, _ = cleaner.clean('\u2018quote\u2019')
        assert "'quote'" in text

    def test_removes_soft_hyphen(self):
        cleaner = TextCleaner()
        text, _ = cleaner.clean('word\u00adbreak')
        assert '\u00ad' not in text


# ---------------------------------------------------------------------------
# clean() — empty and edge cases
# ---------------------------------------------------------------------------

class TestCleanEdgeCases:

    def test_empty_string_returns_empty(self):
        cleaner = TextCleaner()
        text, stats = cleaner.clean("")
        assert text == ""
        assert stats.original_chars == 0
        assert stats.cleaned_chars == 0

    def test_stats_original_chars_set(self):
        cleaner = TextCleaner()
        content = "abc"
        _, stats = cleaner.clean(content)
        assert stats.original_chars == 3

    def test_returns_tuple(self):
        cleaner = TextCleaner()
        result = cleaner.clean("some text")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], CleaningStats)
