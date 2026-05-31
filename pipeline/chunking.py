"""Chunking module for splitting documents into optimal chunks.

Table-aware behaviour
---------------------

When the parser detects tables it rewrites the source text so each
table is wrapped in ``@@@TABLE_BEGIN ...@@@`` / ``@@@TABLE_END@@@``
markers (see ``parsers/table_extractor.py``). Both ``ArticleChunker``
and ``ICAChunker`` honour these markers via the same pre-processing
pipeline:

1. ``_split_table_blocks(text)`` returns alternating prose / table
   segments.
2. Each table segment becomes one indivisible chunk
   (``chunk_type="table"``) carrying title, page, the markdown body
   and a column-header prefix injected for BM25 / dense retrieval.
3. Prose segments go through the original article-aware logic.

Behaviour for documents *without* table markers is unchanged — the
pre-processor short-circuits when it finds zero ``@@@TABLE_BEGIN``
occurrences.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from loguru import logger
from config import config
from parsers.table_extractor import (
    TABLE_BEGIN_PREFIX,
    TableBlock,
    iter_table_blocks,
)


# Soft cap for table chunks. The Phase 1 dry-run measured p95=1472
# chars across 47 detected tables; we allow up to roughly 4x p95 so
# rare matrix tables (e.g. ICA 100-12 Tabela 7, 27x12) still survive
# as a single chunk. Anything bigger gets truncated below the marker
# with an explicit ``... (table truncated) ...`` footer to keep
# embedder/reranker token budgets predictable.
_TABLE_CHUNK_MAX_CHARS = 6_000


@dataclass
class _TableSegment:
    """A table block isolated from the surrounding prose."""
    block: TableBlock
    ctx_pre: str
    ctx_post: str


def _has_table_markers(text: str) -> bool:
    return TABLE_BEGIN_PREFIX in text


def _isolate_table_segments(
    text: str, max_ctx_chars: int = 200
) -> Tuple[List[str], List[_TableSegment]]:
    """Split ``text`` into prose pieces and table segments.

    Returns
    -------
    prose_parts
        Prose between (and around) the table blocks, in document order.
        Always one more entry than ``segments`` (sandwich layout).
    segments
        One ``_TableSegment`` per detected ``@@@TABLE_*@@@`` block.
    """
    blocks = list(iter_table_blocks(text))
    if not blocks:
        return [text], []

    prose_parts: List[str] = []
    segments: List[_TableSegment] = []
    cursor = 0
    for block in blocks:
        start, end = block.span
        prose_before = text[cursor:start]
        prose_parts.append(prose_before)
        ctx_pre = prose_before.rstrip()[-max_ctx_chars:] if max_ctx_chars else ""
        # ctx_post will be set on the next pass once we know the prose
        # segment that follows this table (filled in below).
        segments.append(_TableSegment(block=block, ctx_pre=ctx_pre, ctx_post=""))
        cursor = end
    prose_parts.append(text[cursor:])

    for i, seg in enumerate(segments):
        prose_after = prose_parts[i + 1]
        seg.ctx_post = prose_after.lstrip()[:max_ctx_chars] if max_ctx_chars else ""

    return prose_parts, segments


def _build_table_chunk_text(seg: _TableSegment) -> str:
    """Render the body text for a ``chunk_type="table"`` chunk.

    Layout (in order):

      1. Title line — ``Tabela X`` (or fallback) for narrative anchor.
      2. Column headers as plain text — Phase 2 enrichment for BM25
         and dense retrieval (the markdown body alone has noisy
         pipe characters that BM25 tokenisers split on, so we repeat
         the headers without pipes here).
      3. ctx_pre — last narrative paragraph above the table.
      4. Markdown body — preserves the structure for the LLM.
      5. ctx_post — first paragraph below the table.

    The ``@@@TABLE_*@@@`` markers themselves are stripped: they were
    just a transport mechanism between parser and chunker.
    """
    block = seg.block
    parts: List[str] = []
    title = (block.title or "").strip()
    if title:
        parts.append(title)

    headers = block.column_headers_text
    if headers:
        parts.append(f"Cabeçalhos: {headers}")

    if seg.ctx_pre:
        parts.append(seg.ctx_pre)

    body = block.markdown
    if len(body) > _TABLE_CHUNK_MAX_CHARS:
        body = body[:_TABLE_CHUNK_MAX_CHARS] + "\n... (table truncated) ..."
    parts.append(body)

    if seg.ctx_post:
        parts.append(seg.ctx_post)

    return "\n\n".join(p for p in parts if p)


def _make_table_chunk(article: Dict, seg: _TableSegment, chunk_idx: int) -> Dict:
    """Create one indivisible chunk for a detected table."""
    text = _build_table_chunk_text(seg)
    chunk = article.copy()
    chunk["text"] = text
    chunk["chunk_index"] = chunk_idx
    base_id = article.get("regulation_id", article.get("slug", "unknown"))
    table_id = seg.block.table_id or f"t{chunk_idx}"
    chunk["regulation_id"] = f"{base_id}-{table_id}"
    chunk["chunk_type"] = "table"

    metadata = dict(chunk.get("metadata") or {})
    metadata["chunk_type"] = "table"
    metadata["table_id"] = seg.block.table_id
    metadata["table_title"] = seg.block.title
    metadata["table_rows"] = seg.block.rows
    metadata["table_cols"] = seg.block.cols
    if seg.block.page is not None:
        metadata["table_page"] = seg.block.page
    chunk["metadata"] = metadata
    return chunk


def _strip_table_markers(text: str) -> str:
    """Remove ``@@@TABLE_*@@@`` lines from prose passed to legacy paths."""
    if TABLE_BEGIN_PREFIX not in text:
        return text
    cleaned = re.sub(
        r"@@@TABLE_BEGIN[^@\n]*@@@.*?@@@TABLE_END@@@",
        " ",
        text,
        flags=re.DOTALL,
    )
    return cleaned


class ArticleChunker:
    """Chunker that splits articles intelligently."""

    def __init__(self, max_tokens: int = None, overlap: int = None):
        self.max_tokens = max_tokens or config.CHUNK_MAX_TOKENS
        self.overlap = overlap or config.CHUNK_OVERLAP
        logger.info(f"ArticleChunker initialized (max_tokens={self.max_tokens})")

    def chunk(self, article: Dict) -> List[Dict]:
        """
        Chunk an article into smaller pieces if needed.

        Args:
            article: Article dictionary with 'text' field

        Returns:
            List of chunk dictionaries
        """
        text = article.get("text", "")

        # Phase 1: detected tables become indivisible chunks.
        if _has_table_markers(text):
            return self._chunk_with_tables(article, text)

        chunks = self._chunk_plain(article)
        logger.debug(f"Split article into {len(chunks)} chunks")
        return chunks

    def _chunk_with_tables(self, article: Dict, text: str) -> List[Dict]:
        """Phase 1 path — interleave table chunks with prose chunks.

        Each detected table becomes one ``chunk_type="table"`` chunk.
        Prose between tables is fed back through the legacy chunker so
        non-tabular content keeps its existing behaviour.
        """
        prose_parts, segments = _isolate_table_segments(text)
        out: List[Dict] = []
        for i, seg in enumerate(segments):
            prose = _strip_table_markers(prose_parts[i]).strip()
            if prose:
                out.extend(self._chunk_plain_prose(article, prose, len(out)))
            out.append(_make_table_chunk(article, seg, len(out)))
        tail = _strip_table_markers(prose_parts[-1]).strip()
        if tail:
            out.extend(self._chunk_plain_prose(article, tail, len(out)))
        logger.debug(
            f"Table-aware split: {sum(1 for c in out if c.get('chunk_type') == 'table')} "
            f"table chunks + {sum(1 for c in out if c.get('chunk_type') != 'table')} prose chunks"
        )
        return out

    def _chunk_plain_prose(self, article: Dict, text: str, start_idx: int) -> List[Dict]:
        """Run the legacy chunker on a prose segment and reindex chunks."""
        article_for_prose = {**article, "text": text}
        legacy_chunks = self._chunk_plain(article_for_prose)
        out = []
        for offset, chunk in enumerate(legacy_chunks):
            if chunk is article_for_prose:
                # Single-chunk fast path: rebuild the dict so we don't
                # mutate the caller's article.
                chunk = {**chunk}
            chunk["chunk_index"] = start_idx + offset
            base_id = article.get("regulation_id", "unknown")
            chunk["regulation_id"] = f"{base_id}-chunk-{start_idx + offset}"
            out.append(chunk)
        return out

    def _chunk_plain(self, article: Dict) -> List[Dict]:
        """Internal helper: run the original split logic without table support."""
        text = article["text"]
        estimated_tokens = len(text.split())
        if estimated_tokens <= self.max_tokens:
            return [article]
        segments = self._split_into_segments(text)
        chunks: List[Dict] = []
        current_chunk: List[str] = []
        current_size = 0

        for segment in segments:
            segment_size = len(segment.split())
            if segment_size > self.max_tokens:
                if current_chunk:
                    chunks.append(
                        self._create_chunk(article, "\n".join(current_chunk), len(chunks))
                    )
                    current_chunk = []
                    current_size = 0
                for sub in self._split_large_segment(segment, self.max_tokens):
                    chunks.append(self._create_chunk(article, sub, len(chunks)))
                continue

            if current_size + segment_size > self.max_tokens and current_chunk:
                chunks.append(
                    self._create_chunk(article, "\n".join(current_chunk), len(chunks))
                )
                if self.overlap > 0 and len(current_chunk) > 1:
                    overlap_segment = current_chunk[-1]
                    current_chunk = [overlap_segment, segment]
                    current_size = len(overlap_segment.split()) + segment_size
                else:
                    current_chunk = [segment]
                    current_size = segment_size
            else:
                current_chunk.append(segment)
                current_size += segment_size

        if current_chunk:
            chunks.append(self._create_chunk(article, "\n".join(current_chunk), len(chunks)))

        return chunks

    def _split_into_segments(self, text: str) -> List[str]:
        """
        Split text into meaningful segments.
        
        Tries multiple strategies:
        1. Split by double newlines (paragraphs)
        2. Split by single newlines
        3. Split by article markers (Art. 1º, Art. 2º, etc.)
        """
        # Try double newlines first
        segments = [s.strip() for s in text.split('\n\n') if s.strip()]
        
        if len(segments) > 1:
            return segments
        
        # Try single newlines
        segments = [s.strip() for s in text.split('\n') if s.strip()]
        
        if len(segments) > 10:  # Good enough segmentation
            return segments
        
        # Try splitting by article markers (common in Brazilian laws)
        # Pattern: Art. followed by number
        article_pattern = r'(?=\bArt\s*\.?\s*\d)'
        segments = re.split(article_pattern, text)
        segments = [s.strip() for s in segments if s.strip()]
        
        if len(segments) > 1:
            return segments
        
        # Fallback: split by sentences
        return self._split_by_sentences(text)

    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences."""
        # Brazilian Portuguese sentence endings
        # Handle abbreviations like Art., Nº, etc.
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ])'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_large_segment(self, text: str, max_tokens: int) -> List[str]:
        """Split a large segment into smaller chunks."""
        words = text.split()
        chunks = []
        current = []
        
        for word in words:
            current.append(word)
            if len(current) >= max_tokens:
                chunks.append(' '.join(current))
                # Keep some overlap
                current = current[-50:] if self.overlap > 0 else []
        
        if current:
            chunks.append(' '.join(current))
        
        return chunks

    def _create_chunk(self, article: Dict, text: str, chunk_idx: int) -> Dict:
        """Create chunk dictionary."""
        chunk = article.copy()
        chunk["text"] = text
        chunk["chunk_index"] = chunk_idx
        chunk["regulation_id"] = f"{article.get('regulation_id', 'unknown')}-chunk-{chunk_idx}"
        return chunk


class ICAChunker:
    """
    Specialized chunker for ICA (Instruções do Comando da Aeronáutica) documents.
    
    Splits documents by articles (Art. 1º, Art. 2º, etc.) and handles:
    - Artigos (Art.)
    - Parágrafos (§ or Parágrafo)
    - Incisos (I, II, III, etc.)
    - Alíneas (a), b), c), etc.)
    - Seções and Capítulos
    
    If an article is too large, it's split further while maintaining context.
    """
    
    # Regex patterns for Brazilian legal document structure
    ARTICLE_PATTERN = re.compile(
        r'(?:^|\n)\s*Art\.?\s*(\d+)[°º]?[\s\-\.]*',
        re.IGNORECASE | re.MULTILINE
    )
    
    SECTION_PATTERN = re.compile(
        r'(?:^|\n)\s*(?:SEÇÃO|CAPÍTULO|TÍTULO|ANEXO)\s+[IVXLC\d]+',
        re.IGNORECASE | re.MULTILINE
    )
    
    PARAGRAPH_PATTERN = re.compile(
        r'(?:^|\n)\s*(?:§\s*\d+[°º]?|Parágrafo\s+único)',
        re.IGNORECASE | re.MULTILINE
    )
    
    def __init__(self, max_tokens: int = None, overlap: int = None):
        """
        Initialize ICA chunker.
        
        Args:
            max_tokens: Maximum tokens per chunk (default: from config)
            overlap: Token overlap between chunks (default: from config)
        """
        self.max_tokens = max_tokens or config.CHUNK_MAX_TOKENS
        self.overlap = overlap or config.CHUNK_OVERLAP
        logger.info(f"ICAChunker initialized (max_tokens={self.max_tokens})")
    
    def chunk(self, article: Dict) -> List[Dict]:
        """
        Chunk an ICA document into article-based chunks.
        
        Args:
            article: Document dictionary with 'text' or 'content' field
            
        Returns:
            List of chunk dictionaries, one per article (or sub-article if too large)
        """
        text = article.get("text", "") or article.get("content", "")
        
        if not text or len(text.strip()) < 50:
            return [article]

        # Phase 1: detected tables become indivisible chunks. The
        # interleaving with the article-level logic happens here so
        # that an Article that contains a table still emits the
        # table as one indivisible chunk.
        if _has_table_markers(text):
            return self._chunk_with_tables_ica(article, text)

        # Extract document header (before first article)
        header = self._extract_header(text)
        
        # Split into articles
        articles = self._split_by_articles(text)
        
        if not articles:
            # No articles found, fall back to basic chunking
            logger.debug("No articles found, using fallback chunking")
            return self._fallback_chunk(article, text)
        
        chunks = []

        for art_num, art_text in articles:
            # Check if article is too large
            estimated_tokens = len(art_text.split())
            
            if estimated_tokens <= self.max_tokens:
                # Article fits in one chunk
                chunk = self._create_chunk(
                    article=article,
                    text=art_text,
                    article_num=art_num,
                    chunk_idx=len(chunks),
                    header=header if len(chunks) == 0 else None
                )
                chunks.append(chunk)
            else:
                # Article too large, split by paragraphs/incisos
                sub_chunks = self._split_large_article(art_text, art_num, self.max_tokens)
                for i, sub_text in enumerate(sub_chunks):
                    chunk = self._create_chunk(
                        article=article,
                        text=sub_text,
                        article_num=art_num,
                        chunk_idx=len(chunks),
                        sub_idx=i,
                        header=header if len(chunks) == 0 else None
                    )
                    chunks.append(chunk)
        
        logger.debug(f"Split ICA into {len(chunks)} chunks ({len(articles)} articles found)")
        return chunks

    def _chunk_with_tables_ica(self, article: Dict, text: str) -> List[Dict]:
        """Phase 1 path for ICA / RBAC documents.

        Each detected table becomes one indivisible ``chunk_type="table"``
        chunk. Prose between tables is fed through the regular ICA
        article-aware chunker so legal-structure splitting (Art./§/inciso)
        is preserved for non-tabular content.
        """
        prose_parts, segments = _isolate_table_segments(text)
        out: List[Dict] = []

        def _chunk_prose(prose: str) -> None:
            cleaned = _strip_table_markers(prose).strip()
            if not cleaned:
                return
            sub_article = {**article, "text": cleaned}
            sub_chunks = ICAChunker._chunk_articles_only(self, sub_article, cleaned)
            for chunk in sub_chunks:
                chunk["chunk_index"] = len(out)
                out.append(chunk)

        for i, seg in enumerate(segments):
            _chunk_prose(prose_parts[i])
            table_chunk = _make_table_chunk(article, seg, len(out))
            out.append(table_chunk)
        _chunk_prose(prose_parts[-1])

        logger.debug(
            f"ICA table-aware split: {sum(1 for c in out if c.get('chunk_type') == 'table')} "
            f"table chunks + {sum(1 for c in out if c.get('chunk_type') != 'table')} prose chunks"
        )
        return out

    def _chunk_articles_only(self, article: Dict, text: str) -> List[Dict]:
        """ICA chunking restricted to Article-aware logic, no table handling.

        Used by ``_chunk_with_tables_ica`` to chunk the prose segments
        without recursing into the table marker check.
        """
        header = self._extract_header(text)
        articles = self._split_by_articles(text)

        if not articles:
            return self._fallback_chunk(article, text)

        chunks: List[Dict] = []
        for art_num, art_text in articles:
            estimated_tokens = len(art_text.split())
            if estimated_tokens <= self.max_tokens:
                chunks.append(self._create_chunk(
                    article=article,
                    text=art_text,
                    article_num=art_num,
                    chunk_idx=len(chunks),
                    header=header if len(chunks) == 0 else None,
                ))
            else:
                for i, sub_text in enumerate(self._split_large_article(art_text, art_num, self.max_tokens)):
                    chunks.append(self._create_chunk(
                        article=article,
                        text=sub_text,
                        article_num=art_num,
                        chunk_idx=len(chunks),
                        sub_idx=i,
                        header=header if len(chunks) == 0 else None,
                    ))
        return chunks

    def _extract_header(self, text: str) -> str:
        """Extract document header (everything before Art. 1º)."""
        match = self.ARTICLE_PATTERN.search(text)
        if match:
            return text[:match.start()].strip()
        return ""
    
    def _split_by_articles(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text into articles.
        
        Returns:
            List of tuples (article_number, article_text)
        """
        articles = []
        
        # Find all article positions
        matches = list(self.ARTICLE_PATTERN.finditer(text))
        
        if not matches:
            return []
        
        for i, match in enumerate(matches):
            art_num = match.group(1)
            start = match.start()
            
            # End is start of next article or end of text
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text)
            
            art_text = text[start:end].strip()
            
            if art_text:
                articles.append((art_num, art_text))
        
        return articles
    
    def _split_large_article(self, text: str, art_num: str, max_tokens: int) -> List[str]:
        """
        Split a large article into smaller chunks.
        
        Tries to split by:
        1. Paragraphs (§)
        2. Incisos (I, II, III)
        3. Sentences
        """
        # Try splitting by paragraphs
        parts = self._split_by_paragraphs(text)
        
        if len(parts) > 1:
            return self._merge_parts_to_chunks(parts, max_tokens, f"Art. {art_num}")
        
        # Try splitting by incisos
        parts = self._split_by_incisos(text)
        
        if len(parts) > 1:
            return self._merge_parts_to_chunks(parts, max_tokens, f"Art. {art_num}")
        
        # Fallback: split by sentences or words
        return self._split_by_size(text, max_tokens)
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraph markers (§)."""
        # Split by § or "Parágrafo"
        parts = re.split(r'(?=\n\s*(?:§\s*\d+|Parágrafo))', text, flags=re.IGNORECASE)
        return [p.strip() for p in parts if p.strip()]
    
    def _split_by_incisos(self, text: str) -> List[str]:
        """Split text by incisos (I, II, III, etc.)."""
        # Roman numerals pattern
        pattern = r'(?=\n\s*[IVXLC]+\s*[\-–])'
        parts = re.split(pattern, text)
        return [p.strip() for p in parts if p.strip()]
    
    def _merge_parts_to_chunks(self, parts: List[str], max_tokens: int, context: str) -> List[str]:
        """Merge parts into chunks that fit max_tokens."""
        chunks = []
        current_chunk = []
        current_size = 0

        for part in parts:
            part_size = len(part.split())

            # Hard cap: a single part must never exceed max_tokens. Without this
            # guard, monolithic paragraphs (no §/inciso markers) leak through as
            # oversized chunks that silently truncate at the embedder/reranker.
            if part_size > max_tokens:
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    if not chunk_text.startswith(context):
                        chunk_text = f"{context} (continuação)\n{chunk_text}"
                    chunks.append(chunk_text)
                    current_chunk = []
                    current_size = 0
                for sub in self._split_by_size(part, max_tokens):
                    if not sub.startswith(context):
                        sub = f"{context} (continuação)\n{sub}"
                    chunks.append(sub)
                continue

            if current_size + part_size > max_tokens and current_chunk:
                # Save current chunk with context
                chunk_text = '\n'.join(current_chunk)
                if not chunk_text.startswith(context):
                    chunk_text = f"{context} (continuação)\n{chunk_text}"
                chunks.append(chunk_text)
                
                current_chunk = [part]
                current_size = part_size
            else:
                current_chunk.append(part)
                current_size += part_size
        
        # Add remaining
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if len(chunks) > 0 and not chunk_text.startswith(context):
                chunk_text = f"{context} (continuação)\n{chunk_text}"
            chunks.append(chunk_text)
        
        return chunks
    
    def _split_by_size(self, text: str, max_tokens: int) -> List[str]:
        """Split text by size when other methods fail."""
        words = text.split()
        chunks = []
        current = []
        
        for word in words:
            current.append(word)
            if len(current) >= max_tokens:
                chunks.append(' '.join(current))
                # Keep overlap
                current = current[-50:] if self.overlap > 0 else []
        
        if current:
            chunks.append(' '.join(current))
        
        return chunks
    
    def _fallback_chunk(self, article: Dict, text: str) -> List[Dict]:
        """Fallback chunking when no articles are found."""
        estimated_tokens = len(text.split())
        
        if estimated_tokens <= self.max_tokens:
            return [article]
        
        # Split by double newlines or sections
        parts = re.split(r'\n\n+', text)
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) <= 1:
            parts = text.split('\n')
            parts = [p.strip() for p in parts if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0

        def _flush_current() -> None:
            if not current_chunk:
                return
            chunk = article.copy()
            chunk["text"] = '\n'.join(current_chunk)
            chunk["chunk_index"] = len(chunks)
            chunk["regulation_id"] = f"{article.get('regulation_id', 'unknown')}-chunk-{len(chunks)}"
            chunks.append(chunk)

        for part in parts:
            part_size = len(part.split())

            # Hard cap: a single part must never exceed max_tokens. Without this
            # guard, free-form documents with no \n\n markers leak through as a
            # single oversized chunk (root cause of the historical 4K+ subword
            # outliers seen in the token audit).
            if part_size > self.max_tokens:
                _flush_current()
                current_chunk = []
                current_size = 0
                for sub_text in self._split_by_size(part, self.max_tokens):
                    chunk = article.copy()
                    chunk["text"] = sub_text
                    chunk["chunk_index"] = len(chunks)
                    chunk["regulation_id"] = f"{article.get('regulation_id', 'unknown')}-chunk-{len(chunks)}"
                    chunks.append(chunk)
                continue

            if current_size + part_size > self.max_tokens and current_chunk:
                _flush_current()
                current_chunk = [part]
                current_size = part_size
            else:
                current_chunk.append(part)
                current_size += part_size

        _flush_current()

        return chunks

    def _create_chunk(
        self,
        article: Dict,
        text: str,
        article_num: str,
        chunk_idx: int,
        sub_idx: int = None,
        header: str = None
    ) -> Dict:
        """
        Create a chunk dictionary.
        
        Args:
            article: Original document dictionary
            text: Chunk text content
            article_num: Article number (e.g., "1", "2")
            chunk_idx: Overall chunk index
            sub_idx: Sub-chunk index if article was split
            header: Document header to prepend (only for first chunk)
        """
        chunk = article.copy()
        
        # Prepend header for context if provided
        if header and len(header) < 500:
            chunk["text"] = f"{header}\n\n{text}"
        else:
            chunk["text"] = text
        
        chunk["chunk_index"] = chunk_idx
        chunk["article_number"] = article_num
        
        # Create unique ID
        base_id = article.get("regulation_id", article.get("slug", "unknown"))
        if sub_idx is not None:
            chunk["regulation_id"] = f"{base_id}-art{article_num}-{sub_idx}"
            chunk["chunk_type"] = "article_part"
        else:
            chunk["regulation_id"] = f"{base_id}-art{article_num}"
            chunk["chunk_type"] = "article"
        
        # Add metadata
        if "metadata" not in chunk:
            chunk["metadata"] = {}
        chunk["metadata"]["article_number"] = article_num
        chunk["metadata"]["chunk_type"] = chunk.get("chunk_type", "article")
        
        return chunk


_ICA_CHUNKER_TYPES = frozenset([
    'ica', 'mca', 'pca', 'nsca', 'dca', 'tca', 'fca', 'oca',
    'rca', 'roca', 'rica', 'rima', 'rma', 'npa', 'bca', 'bma',
    'ima', 'circea', 'decea',
    'rbac',  # Regulamentos Brasileiros da Aviação Civil (ANAC)
])

_ICA_TYPE_RE = re.compile(
    r'\b(?:' + '|'.join(sorted(_ICA_CHUNKER_TYPES, key=len, reverse=True)) + r')\b',
    re.IGNORECASE,
)


def get_chunker(doc_type: str = None) -> ArticleChunker:
    """
    Get appropriate chunker based on document type.
    
    Args:
        doc_type: Document type (e.g., 'ICA', 'lei', 'portaria')
        
    Returns:
        Appropriate chunker instance
    """
    if doc_type and _ICA_TYPE_RE.search(doc_type):
        return ICAChunker()
    
    return ArticleChunker()


if __name__ == "__main__":
    # Test both chunkers
    print("Testing ArticleChunker...")
    chunker = ArticleChunker()
    print(f"  max_tokens: {chunker.max_tokens}")
    
    print("\nTesting ICAChunker...")
    ica_chunker = ICAChunker()
    print(f"  max_tokens: {ica_chunker.max_tokens}")
    
    # Test with sample ICA text
    sample_text = """
    MINISTÉRIO DA DEFESA
    COMANDO DA AERONÁUTICA
    DEPARTAMENTO DE CONTROLE DO ESPAÇO AÉREO
    
    ICA 100-12 - Regras do Ar
    
    Art. 1º Esta Instrução estabelece as regras gerais de tráfego aéreo.
    
    § 1º As regras aplicam-se a todo o espaço aéreo brasileiro.
    § 2º Exceções podem ser autorizadas pelo órgão competente.
    
    Art. 2º Para efeitos desta Instrução, considera-se:
    
    I - aeronave: todo aparelho manobrável em voo;
    II - aeródromo: toda área destinada a pouso e decolagem;
    III - espaço aéreo: volume de ar sob jurisdição brasileira.
    
    Art. 3º O piloto em comando é responsável pela operação segura da aeronave.
    
    Parágrafo único. Em caso de emergência, o piloto pode desviar das regras.
    """
    
    test_doc = {
        "regulation_id": "ICA-100-12",
        "title": "Regras do Ar",
        "text": sample_text
    }
    
    chunks = ica_chunker.chunk(test_doc)
    print(f"\nSample ICA split into {len(chunks)} chunks:")
    for c in chunks:
        print(f"  - {c['regulation_id']}: {len(c['text'])} chars, Art. {c.get('article_number', 'N/A')}")
