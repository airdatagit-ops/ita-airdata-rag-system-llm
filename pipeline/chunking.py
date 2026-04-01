"""Chunking module for splitting documents into optimal chunks."""

import re
from typing import Dict, List, Optional, Tuple
from loguru import logger
from config import config


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
        estimated_tokens = len(text.split())

        # If article is small enough, return as single chunk
        if estimated_tokens <= self.max_tokens:
            return [article]

        # Try to split into meaningful segments
        segments = self._split_into_segments(text)
        
        chunks = []
        current_chunk = []
        current_size = 0

        for segment in segments:
            segment_size = len(segment.split())

            # If single segment is too large, split it further
            if segment_size > self.max_tokens:
                # Save current chunk first
                if current_chunk:
                    chunk = self._create_chunk(article, '\n'.join(current_chunk), len(chunks))
                    chunks.append(chunk)
                    current_chunk = []
                    current_size = 0
                
                # Split large segment by sentences
                sub_chunks = self._split_large_segment(segment, self.max_tokens)
                for sub in sub_chunks:
                    chunk = self._create_chunk(article, sub, len(chunks))
                    chunks.append(chunk)
                continue

            if current_size + segment_size > self.max_tokens and current_chunk:
                # Save current chunk
                chunk = self._create_chunk(article, '\n'.join(current_chunk), len(chunks))
                chunks.append(chunk)

                # Start new chunk with overlap if configured
                if self.overlap > 0 and len(current_chunk) > 1:
                    # Keep last segment for overlap
                    overlap_segment = current_chunk[-1]
                    current_chunk = [overlap_segment, segment]
                    current_size = len(overlap_segment.split()) + segment_size
                else:
                    current_chunk = [segment]
                    current_size = segment_size
            else:
                current_chunk.append(segment)
                current_size += segment_size

        # Add remaining
        if current_chunk:
            chunk = self._create_chunk(article, '\n'.join(current_chunk), len(chunks))
            chunks.append(chunk)

        logger.debug(f"Split article into {len(chunks)} chunks")
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
        
        # Extract document header (before first article)
        header = self._extract_header(text)
        
        # Split into articles
        articles = self._split_by_articles(text)
        
        if not articles:
            # No articles found, fall back to basic chunking
            logger.debug("No articles found, using fallback chunking")
            return self._fallback_chunk(article, text)
        
        chunks = []
        doc_id = article.get("regulation_id", article.get("slug", "unknown"))
        
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
        
        for part in parts:
            part_size = len(part.split())
            
            if current_size + part_size > self.max_tokens and current_chunk:
                chunk = article.copy()
                chunk["text"] = '\n'.join(current_chunk)
                chunk["chunk_index"] = len(chunks)
                chunk["regulation_id"] = f"{article.get('regulation_id', 'unknown')}-chunk-{len(chunks)}"
                chunks.append(chunk)
                
                current_chunk = [part]
                current_size = part_size
            else:
                current_chunk.append(part)
                current_size += part_size
        
        if current_chunk:
            chunk = article.copy()
            chunk["text"] = '\n'.join(current_chunk)
            chunk["chunk_index"] = len(chunks)
            chunk["regulation_id"] = f"{article.get('regulation_id', 'unknown')}-chunk-{len(chunks)}"
            chunks.append(chunk)
        
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
