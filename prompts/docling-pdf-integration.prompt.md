# Prompt: Integração do Docling para Extração e Chunking de PDFs

## Objetivo

Integrar a biblioteca **docling** (IBM, open-source) ao pipeline existente de extração e
chunkerização de PDFs com texto extraível (sem OCR), preservando **completamente** o
comportamento legado como padrão e adicionando variáveis de ambiente para ativar o novo
modo docling.

---

## Regras invioláveis

1. **O código original deve continuar funcionando por padrão.** Qualquer instalação que não
   definir `PDF_EXTRACTION_BACKEND=docling` deve se comportar exatamente como hoje.
2. **Nenhuma quebra de interface.** As assinaturas de funções e classes públicas existentes
   não devem mudar; novas somente podem ser adicionadas.
3. **Todas as mudanças devem ser commitadas separadamente** com mensagens descritivas no
   formato `type(scope): descrição` (ex.: `feat(parsers): add docling extraction backend`).

---

## Contexto arquitetural atual

### Pipeline de ingestão (ordem de execução)

```
scripts/collect.py
  └─► crawler/scrapers/pdf_scraper.py  (PDFScraper._parse_pdf_sync)
        └─► parsers/pdf_parser.py      (PDFParser.parse_pdf)
              ├─► _extract_text()      → pdfplumber → PyPDF2 (fallback)
              ├─► _extract_metadata()
              └─► _extract_sections()  → regex p/ seções numeradas
      ◄── ScrapedDocument(content=joined_sections_text, ...)
  └─► DocumentStore (SQLite, pipeline/document_store.py)

scripts/embed.py
  └─► DocumentStore.stream_all()
        └─► pipeline/text_cleaner.py   (TextCleaner.clean)
        └─► pipeline/chunking.py       (get_chunker(doc_type).chunk(article))
              ├─► ICAChunker           → p/ documentos ICA/DCA/FCA/MCA/NSCA etc.
              └─► ArticleChunker       → fallback genérico
        └─► models/embeddings.py       → gera vetores BGE-M3 (1024-d, cosine)
        └─► pipeline/embedding_store.py → persiste em Parquet

scripts/index.py
  └─► Parquet → Qdrant (upsert)
```

### Estrutura de dados que trafega entre componentes

- **`PDFParser.parse_pdf()`** retorna `List[Dict]` onde cada dict tem:  
  `regulation_id`, `section_number`, `title`, `text`, `effective_date`, `expiry_date`,
  `status`, `metadata`

- **`ScrapedDocument.content`** é a string única resultante de
  `"\n\n".join(s.get("text", "") for s in sections)`.

- **`chunker.chunk(article)`** recebe um `Dict` com campos mínimos `regulation_id`, `text`,
  `title`, `metadata` e retorna `List[Dict]` com os mesmos campos mais `chunk_index`.

- **`extract_text_from_bytes(pdf_content: bytes) -> str | None`** é uma função standalone
  em `parsers/pdf_parser.py` usada por outros módulos (ex.: API) para extração em memória.

---

## O que é o Docling e como ele funciona (para texto extraível)

### Instalação

```
docling>=2.7.0
```

### Uso básico para PDF com texto (sem OCR)

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, PdfBackend

pipeline_options = PdfPipelineOptions(
    do_ocr=False,                          # desabilita OCR (já é False por padrão)
    do_table_structure=True,               # extrai estrutura de tabelas
    force_backend_text=False,              # True = ignora layout model, usa texto nativo do PDF
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=PdfBackend.DOCLING_PARSE,  # ou PdfBackend.PYPDFIUM2 (mais simples/rápido)
        )
    }
)

result = converter.convert("caminho/para/arquivo.pdf")
dl_doc = result.document  # objeto DoclingDocument

# Exportações
markdown_text = dl_doc.export_to_markdown()    # texto com headings # ## ###
html_text     = dl_doc.export_to_html()        # HTML semântico com h1-h6, tabelas
doc_dict      = dl_doc.export_to_dict()        # JSON serializável (para persistência)

# Reconstrução a partir de dict (útil para persistir e recarregar)
from docling_core.types.doc import DoclingDocument
reconstructed_doc = DoclingDocument.model_validate(doc_dict)
```

### `PdfBackend` disponíveis

| Backend | Quando usar |
|---|---|
| `PdfBackend.DOCLING_PARSE` | Padrão. Melhor layout, multicoluna, tabelas complexas |
| `PdfBackend.PYPDFIUM2` | PDFs simples, mais rápido, menor consumo de memória |

### `force_backend_text=True`

Quando `True`, o docling ignora completamente o modelo de layout (LayoutModel neural) e
usa o texto nativo do PDF diretamente via backend. Muito mais rápido, mas perde a detecção
de hierarquia de headings por análise visual. Use para PDFs que já têm estrutura lógica
clara no texto (ex.: numeração de seções explícita como `1.2.3 Título`).

### Chunking nativo com `HybridChunker`

```python
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=270,          # alinhado a CHUNK_MAX_TOKENS
    merge_peers=True,        # mescla chunks consecutivos pequenos com mesmos headings
    repeat_table_header=True # repete header de tabela se ela for dividida em múltiplos chunks
)

chunks = list(chunker.chunk(dl_doc))

for chunk in chunks:
    chunk_text    = chunk.text           # texto do chunk
    headings      = chunk.meta.headings  # lista de headings-pai (contexto hierárquico)
    captions      = chunk.meta.captions  # para tabelas/figuras
    origin_page   = chunk.meta.origin.page_no  # número da página (se disponível)
```

Cada `chunk.meta.headings` é uma lista de strings com os headings pais, o que preserva
contexto hierárquico para RAG (ex.: `["Capítulo 3", "Seção 3.2"]`).

### Estrutura do `DoclingDocument`

- `dl_doc.texts` → lista de `TextItem` (parágrafos, títulos, equações...)
- `dl_doc.tables` → lista de `TableItem` (com estrutura linha×coluna)
- `dl_doc.pictures` → lista de `PictureItem`
- `dl_doc.body` → árvore hierárquica da leitura principal
- `dl_doc.furniture` → headers, footers (fora do body)

---

## Mudanças a implementar

### 1. `requirements.txt`

Adicionar ao final da seção de dependências de PDF (após `PyMuPDF>=1.24.0`):

```
docling>=2.7.0
```

**Commit:** `feat(deps): add docling>=2.7.0 to requirements`

---

### 2. `env.example`

Na seção `# PDF Parser Configuration`, **após** o bloco `OCR_LANGUAGE=por`, adicionar:

```bash
# ========================================
# Docling PDF Extraction (alternative backend)
# ========================================
# Backend de extração de PDF: 'legacy' usa pdfplumber/PyMuPDF/PyPDF2 (padrão)
# 'docling' usa a biblioteca docling para extração estruturada
PDF_EXTRACTION_BACKEND=legacy

# --- Opções Docling (somente quando PDF_EXTRACTION_BACKEND=docling) ---

# Backend interno do docling: 'docling_parse' (padrão, melhor para PDFs complexos)
# ou 'pypdfium2' (mais rápido, para PDFs simples)
DOCLING_PDF_BACKEND=docling_parse

# Extrair estrutura de tabelas (recomendado: true)
DOCLING_DO_TABLE_STRUCTURE=true

# Ignorar modelo de layout e usar texto nativo do PDF diretamente.
# true = mais rápido mas perde detecção de heading por análise visual
# false = usa LayoutModel para hierarquia mais precisa (recomendado para multicoluna)
DOCLING_FORCE_BACKEND_TEXT=false

# Caminho local para modelos docling pré-baixados (deixe vazio para auto-download)
# Útil em ambientes sem acesso à internet
# DOCLING_ARTIFACTS_PATH=./models_cache/docling

# Usar HybridChunker do docling em vez dos chunkers legados (ICAChunker/ArticleChunker).
# Requer PDF_EXTRACTION_BACKEND=docling. Produz chunks com metadados de heading-pai.
# false por padrão: docling extrai o texto e os chunkers legados são usados normalmente
DOCLING_CHUNKING_ENABLED=false
```

**Commit:** `feat(config): add DOCLING_* vars to env.example for docling PDF extraction mode`

---

### 3. `config.py` — classe `Settings`

Na seção `# PDF Parser Configuration`, logo após os campos `ENABLE_OCR` e `OCR_LANGUAGE`,
adicionar os novos campos:

```python
    # ========================================
    # Docling PDF Extraction (alternative backend)
    # ========================================
    # 'legacy' mantém comportamento atual; 'docling' ativa extração via docling
    PDF_EXTRACTION_BACKEND: str = getenv('PDF_EXTRACTION_BACKEND', 'legacy')

    # Backend interno do docling: 'docling_parse' ou 'pypdfium2'
    DOCLING_PDF_BACKEND: str = getenv('DOCLING_PDF_BACKEND', 'docling_parse')

    # Extrair estrutura de tabelas
    DOCLING_DO_TABLE_STRUCTURE: bool = getenv('DOCLING_DO_TABLE_STRUCTURE', 'true').lower() in ('true', '1', 'yes')

    # Bypassar modelo de layout, usar texto nativo do PDF diretamente
    DOCLING_FORCE_BACKEND_TEXT: bool = getenv('DOCLING_FORCE_BACKEND_TEXT', 'false').lower() in ('true', '1', 'yes')

    # Caminho para modelos pré-baixados do docling (None = auto-download)
    DOCLING_ARTIFACTS_PATH: Optional[str] = getenv('DOCLING_ARTIFACTS_PATH') or None

    # Usar HybridChunker do docling (requer PDF_EXTRACTION_BACKEND=docling)
    DOCLING_CHUNKING_ENABLED: bool = getenv('DOCLING_CHUNKING_ENABLED', 'false').lower() in ('true', '1', 'yes')
```

**Observação:** O import de `Optional` já existe no topo do arquivo (`from typing import List, Optional`).

**Commit:** `feat(config): add DOCLING_* settings fields to Settings class`

---

### 4. `parsers/pdf_parser.py` — extração com docling

#### 4a. Atualizar `extract_text_from_bytes()`

A função standalone deve tentar docling **antes** da cadeia fitz→pdfplumber→OCR quando
`PDF_EXTRACTION_BACKEND=docling`. Inserir o bloco abaixo **antes** do try/except do fitz:

```python
def extract_text_from_bytes(pdf_content: bytes) -> str | None:
    """Extract text from in-memory PDF bytes (docling | PyMuPDF -> pdfplumber -> OCR)."""
    # ── Docling path (quando ativado via config) ─────────────────────────────
    if config.PDF_EXTRACTION_BACKEND == 'docling':
        try:
            import io
            import tempfile
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions, PdfBackend

            backend_map = {
                'docling_parse': PdfBackend.DOCLING_PARSE,
                'pypdfium2': PdfBackend.PYPDFIUM2,
            }
            pdf_backend = backend_map.get(config.DOCLING_PDF_BACKEND, PdfBackend.DOCLING_PARSE)

            pipeline_options = PdfPipelineOptions(
                do_ocr=False,
                do_table_structure=config.DOCLING_DO_TABLE_STRUCTURE,
                force_backend_text=config.DOCLING_FORCE_BACKEND_TEXT,
            )
            if config.DOCLING_ARTIFACTS_PATH:
                from pathlib import Path as _Path
                pipeline_options.artifacts_path = _Path(config.DOCLING_ARTIFACTS_PATH)

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                        backend=pdf_backend,
                    )
                }
            )
            # Docling precisa de um path ou BytesIO; usar arquivo temporário
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                tmp.write(pdf_content)
                tmp.flush()
                result = converter.convert(tmp.name)
            text = result.document.export_to_markdown()
            if len(text.strip()) > 100:
                return text
        except Exception as e:
            logger.warning(f"Docling extraction failed, falling back to legacy: {e}")

    # ── Legacy path (padrão) ─────────────────────────────────────────────────
    try:
        import fitz
        # ... (código existente, sem alterações)
```

**Importante:** O resto da função (`fitz`, `pdfplumber`, `OCR`) permanece **exatamente** como
está. O bloco docling é inserido antes e, se falhar por qualquer motivo, a execução cai no
fallback legado normalmente.

#### 4b. Adicionar `_extract_with_docling()` à classe `PDFParser`

Adicionar o método abaixo **após** `_extract_with_pypdf2()` e **antes** de
`_extract_text_with_ocr()`:

```python
    def _extract_with_docling(self, pdf_path: str) -> tuple[str, object | None]:
        """
        Extract text (and optionally DoclingDocument) from PDF using docling.

        Returns:
            Tuple of (markdown_text, docling_document_or_None).
            docling_document_or_None is None if extraction fails.
        """
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions, PdfBackend

            backend_map = {
                'docling_parse': PdfBackend.DOCLING_PARSE,
                'pypdfium2': PdfBackend.PYPDFIUM2,
            }
            pdf_backend = backend_map.get(config.DOCLING_PDF_BACKEND, PdfBackend.DOCLING_PARSE)

            pipeline_options = PdfPipelineOptions(
                do_ocr=False,
                do_table_structure=config.DOCLING_DO_TABLE_STRUCTURE,
                force_backend_text=config.DOCLING_FORCE_BACKEND_TEXT,
            )
            if config.DOCLING_ARTIFACTS_PATH:
                from pathlib import Path as _Path
                pipeline_options.artifacts_path = _Path(config.DOCLING_ARTIFACTS_PATH)

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                        backend=pdf_backend,
                    )
                }
            )
            result = converter.convert(pdf_path)
            dl_doc = result.document
            markdown_text = dl_doc.export_to_markdown()
            logger.debug(f"Docling extracted {len(markdown_text)} chars from {pdf_path}")
            return markdown_text, dl_doc
        except Exception as e:
            logger.warning(f"Docling extraction failed for {pdf_path}: {e}")
            return "", None
```

#### 4c. Modificar `_extract_text()` para usar docling quando configurado

Substituir o método `_extract_text()` existente por:

```python
    def _extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF (docling or legacy, depending on config)."""
        if config.PDF_EXTRACTION_BACKEND == 'docling':
            text, _ = self._extract_with_docling(pdf_path)
            if text and len(text.strip()) > 100:
                return text
            logger.warning(f"Docling extraction insufficient for {pdf_path}, falling back to legacy")

        # ── Legacy path ──────────────────────────────────────────────────────
        if PDFPLUMBER_AVAILABLE:
            return self._extract_with_pdfplumber(pdf_path)
        elif PYPDF2_AVAILABLE:
            return self._extract_with_pypdf2(pdf_path)
        else:
            raise RuntimeError("No PDF library available")
```

#### 4d. Modificar `parse_pdf()` para armazenar o `DoclingDocument` no metadata

Modificar o método `parse_pdf()` para, quando no modo docling, armazenar o documento
serializado nos metadados de cada seção (necessário para o chunking nativo):

```python
    def parse_pdf(self, pdf_path: str) -> List[Dict]:
        """Parse PDF file and extract sections."""
        try:
            dl_doc = None

            # Quando docling está ativo, extrair tanto texto quanto o documento estruturado
            if config.PDF_EXTRACTION_BACKEND == 'docling':
                text, dl_doc = self._extract_with_docling(pdf_path)
                if not text or len(text.strip()) < 50:
                    logger.warning(f"Docling insufficient for {pdf_path}, falling back")
                    dl_doc = None
                    text = self._extract_text_legacy(pdf_path)
            else:
                text = self._extract_text(pdf_path)

            if not text or len(text.strip()) < 50:
                logger.warning(f"PDF {pdf_path} has very little text. May need OCR.")
                if self.enable_ocr:
                    text = self._extract_text_with_ocr(pdf_path)

            metadata = self._extract_metadata(pdf_path, text)

            # Serializar o DoclingDocument para JSON e guardar nos metadados (para chunking nativo)
            if dl_doc is not None and config.DOCLING_CHUNKING_ENABLED:
                try:
                    import json as _json
                    metadata['_docling_doc_json'] = _json.dumps(
                        dl_doc.export_to_dict(), ensure_ascii=False
                    )
                except Exception as e:
                    logger.warning(f"Failed to serialize DoclingDocument: {e}")

            sections = self._extract_sections(text, metadata)
            logger.info(f"Parsed {len(sections)} sections from {pdf_path}")
            return sections

        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path}: {e}")
            return []
```

**Adicionar também** o método `_extract_text_legacy()` como alias para o comportamento
atual (necessário para o fallback acima):

```python
    def _extract_text_legacy(self, pdf_path: str) -> str:
        """Extract text using legacy backends (pdfplumber/PyPDF2)."""
        if PDFPLUMBER_AVAILABLE:
            return self._extract_with_pdfplumber(pdf_path)
        elif PYPDF2_AVAILABLE:
            return self._extract_with_pypdf2(pdf_path)
        else:
            raise RuntimeError("No PDF library available")
```

**Commit:** `feat(parsers): add docling extraction path to PDFParser and extract_text_from_bytes`

---

### 5. `pipeline/chunking.py` — chunking nativo com `HybridChunker`

#### 5a. Adicionar classe `DoclingChunker`

Adicionar **antes** da função `get_chunker()` a nova classe:

```python
class DoclingChunker:
    """
    Chunker que usa o HybridChunker nativo do docling.

    Recebe um article dict com '_docling_doc_json' no metadata e usa
    HybridChunker (tokenization-aware) para produzir chunks com metadados
    hierárquicos (headings pais, captions, etc.).

    Fallback automático para ICAChunker/ArticleChunker se o DoclingDocument
    não estiver disponível.
    """

    def __init__(self, max_tokens: int = None, doc_type: str = None):
        self.max_tokens = max_tokens or config.CHUNK_MAX_TOKENS
        self.doc_type = doc_type or ''
        self._fallback = get_chunker(self.doc_type)
        logger.info(f"DoclingChunker initialized (max_tokens={self.max_tokens})")

    def chunk(self, article: Dict) -> List[Dict]:
        """
        Chunk via HybridChunker (docling) se possível, fallback para legado.

        O article deve conter 'metadata._docling_doc_json' (JSON string do
        DoclingDocument.export_to_dict()). Se ausente, usa chunker legado.
        """
        meta = article.get('metadata') or {}
        docling_json = meta.get('_docling_doc_json')

        if not docling_json:
            logger.debug("DoclingChunker: no docling doc in metadata, using legacy chunker")
            return self._fallback.chunk(article)

        try:
            import json as _json
            from docling_core.types.doc import DoclingDocument
            from docling.chunking import HybridChunker
            from transformers import AutoTokenizer

            doc_dict = _json.loads(docling_json)
            dl_doc = DoclingDocument.model_validate(doc_dict)

            tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
            chunker = HybridChunker(
                tokenizer=tokenizer,
                max_tokens=self.max_tokens,
                merge_peers=True,
                repeat_table_header=True,
            )

            chunks = []
            for i, chunk in enumerate(chunker.chunk(dl_doc)):
                chunk_dict = article.copy()

                # Enriquecer texto com headings-pai para maior contexto no embedding
                headings_prefix = ""
                if chunk.meta.headings:
                    headings_prefix = " > ".join(chunk.meta.headings) + "\n\n"

                chunk_dict["text"] = headings_prefix + chunk.text
                chunk_dict["chunk_index"] = i
                chunk_dict["regulation_id"] = (
                    f"{article.get('regulation_id', 'unknown')}-dchunk-{i}"
                )

                # Guardar metadados docling
                chunk_meta = dict(chunk_dict.get("metadata") or {})
                chunk_meta["chunk_type"] = "docling_hybrid"
                chunk_meta["docling_headings"] = chunk.meta.headings or []
                chunk_meta["docling_captions"] = [
                    str(c) for c in (chunk.meta.captions or [])
                ]
                # Remover o JSON pesado do metadata do chunk (já não é necessário)
                chunk_meta.pop("_docling_doc_json", None)
                chunk_dict["metadata"] = chunk_meta

                chunks.append(chunk_dict)

            logger.debug(
                f"DoclingChunker: {len(chunks)} chunks from HybridChunker "
                f"(doc: {article.get('regulation_id', '?')})"
            )
            return chunks if chunks else self._fallback.chunk(article)

        except Exception as e:
            logger.warning(
                f"DoclingChunker failed ({e}), falling back to legacy chunker"
            )
            return self._fallback.chunk(article)
```

#### 5b. Modificar `get_chunker()` para retornar `DoclingChunker` quando configurado

```python
def get_chunker(doc_type: str = None) -> ArticleChunker:
    """
    Get appropriate chunker based on document type and config.

    Returns DoclingChunker when DOCLING_CHUNKING_ENABLED=true and
    PDF_EXTRACTION_BACKEND=docling; otherwise returns legacy chunker.
    """
    if (
        config.PDF_EXTRACTION_BACKEND == 'docling'
        and config.DOCLING_CHUNKING_ENABLED
    ):
        return DoclingChunker(doc_type=doc_type)

    # ── Legacy path (padrão) ─────────────────────────────────────────────
    if doc_type and _ICA_TYPE_RE.search(doc_type):
        return ICAChunker()
    return ArticleChunker()
```

**Adicionar ao topo dos imports de `pipeline/chunking.py`**:

```python
from config import config
```

(já existe — verificar se está presente; se sim, não duplicar)

**Commit:** `feat(chunking): add DoclingChunker with HybridChunker and update get_chunker factory`

---

### 6. `crawler/scrapers/pdf_scraper.py` — sem mudanças necessárias

O `PDFScraper._parse_pdf_sync()` já chama `PDFParser().parse_pdf()`, que agora internamente
usa docling quando configurado. **Nenhuma mudança** é necessária neste arquivo.

---

### 7. `scripts/embed.py` — sem mudanças necessárias

A função `_chunk_document()` já chama `get_chunker(doc_type).chunk(article)`.
`get_chunker()` agora retorna `DoclingChunker` quando apropriado. **Nenhuma mudança**
é necessária neste arquivo.

---

## Diagrama do fluxo integrado

```
PDF_EXTRACTION_BACKEND=legacy (padrão):
  PDFParser.parse_pdf() → _extract_text() → pdfplumber/PyPDF2
  get_chunker() → ICAChunker | ArticleChunker     ← SEM MUDANÇA

PDF_EXTRACTION_BACKEND=docling, DOCLING_CHUNKING_ENABLED=false:
  PDFParser.parse_pdf() → _extract_with_docling() → markdown text
  get_chunker() → ICAChunker | ArticleChunker     ← melhor texto, mesmo chunker

PDF_EXTRACTION_BACKEND=docling, DOCLING_CHUNKING_ENABLED=true:
  PDFParser.parse_pdf() → _extract_with_docling() → markdown + serializa DoclingDocument em metadata
  get_chunker() → DoclingChunker → HybridChunker(DoclingDocument)  ← chunking estrutural nativo
```

---

## Comportamento de fallback em cada componente

| Componente | Falha em | Comportamento |
|---|---|---|
| `extract_text_from_bytes` | import docling, conversion error | cai no fitz → pdfplumber → OCR |
| `PDFParser._extract_with_docling` | qualquer exceção | retorna `("", None)` |
| `PDFParser._extract_text` | docling insuficiente (<100 chars) | usa pdfplumber/PyPDF2 |
| `DoclingChunker.chunk` | sem `_docling_doc_json` | usa `ICAChunker`/`ArticleChunker` |
| `DoclingChunker.chunk` | import, parse, chunking error | usa `ICAChunker`/`ArticleChunker` |

---

## Dependências novas (imports condicionais)

Todos os imports de `docling` devem ser feitos **dentro dos blocos try/except** de cada
método, nunca no topo do módulo. Isso garante que o sistema funcione mesmo sem docling
instalado (modo legacy).

Exceção: `DoclingChunker` em `pipeline/chunking.py` pode ter os imports no método `chunk()`
já que o fallback garante que o sistema funciona mesmo sem docling.

---

## Notas de implementação importantes

### Arquivo temporário em `extract_text_from_bytes`

O `DocumentConverter` do docling aceita um caminho de arquivo. Para o caso de bytes
em memória (`extract_text_from_bytes`), usar `tempfile.NamedTemporaryFile`. Preferir
`delete=True` e usar como context manager para garantir limpeza.

### `DOCLING_ARTIFACTS_PATH`

Quando definido, o docling não tentará baixar modelos da internet. Útil para ambientes
de produção air-gapped. Os modelos devem ser pré-baixados uma vez com:

```bash
# Executar uma vez em ambiente com internet
python -c "
from docling.document_converter import DocumentConverter
DocumentConverter()  # faz download dos modelos no cache padrão
"
# Copiar para DOCLING_ARTIFACTS_PATH
```

### Token counting no `DoclingChunker`

O `HybridChunker` usa o tokenizer do embedding model (BGE-M3) para contar tokens com
precisão real (subwords), ao contrário dos chunkers legados que contam palavras e aplicam
uma heurística (subword/word ratio ≈ 1.78). Isso significa que chunks do `DoclingChunker`
com `max_tokens=270` estarão mais próximos do limite real do modelo.

### Limpeza do `_docling_doc_json` no metadata

O JSON serializado do `DoclingDocument` pode ser grande (centenas de KB). Ele é necessário
somente no pipeline de embed. O `DoclingChunker` deve **remover** o campo
`_docling_doc_json` dos metadados de cada chunk gerado para não inflar o tamanho dos
registros no Qdrant e Parquet. Isso já está previsto no código de exemplo acima
(`chunk_meta.pop("_docling_doc_json", None)`).

### `export_to_markdown()` e os chunkers legados

Quando `DOCLING_CHUNKING_ENABLED=false` mas `PDF_EXTRACTION_BACKEND=docling`, o texto
extraído pelo docling está em formato Markdown com headings `## Seção 1.2` etc.
O regex de `_extract_sections()` em `pdf_parser.py` procura por padrão `^\d+\.\d+ Título$`
(seções numeradas). O markdown do docling preserva os números de seção nos headings quando
presentes no PDF. Se a extração de seções falhar (retornar lista vazia), o comportamento
atual já prevê fallback para documento único — isso é suficiente.

---

## Sequência de commits esperada

1. `feat(deps): add docling>=2.7.0 to requirements`
2. `feat(config): add DOCLING_* vars to env.example for docling PDF extraction mode`
3. `feat(config): add DOCLING_* settings fields to Settings class`
4. `feat(parsers): add docling extraction path to PDFParser and extract_text_from_bytes`
5. `feat(chunking): add DoclingChunker with HybridChunker and update get_chunker factory`

---

## Verificação pós-implementação

Após implementar todas as mudanças, verificar:

```bash
# 1. Modo legacy deve funcionar exatamente como antes
PDF_EXTRACTION_BACKEND=legacy python -c "
from parsers.pdf_parser import PDFParser
p = PDFParser()
print('PDFParser legacy OK')
"

# 2. Modo docling (extração) deve funcionar
PDF_EXTRACTION_BACKEND=docling DOCLING_CHUNKING_ENABLED=false python -c "
from parsers.pdf_parser import PDFParser
p = PDFParser()
print('PDFParser docling OK')
"

# 3. get_chunker deve retornar tipo correto
PDF_EXTRACTION_BACKEND=legacy python -c "
from pipeline.chunking import get_chunker
c = get_chunker('ICA')
print(type(c).__name__)  # ICAChunker
"

PDF_EXTRACTION_BACKEND=docling DOCLING_CHUNKING_ENABLED=true python -c "
from pipeline.chunking import get_chunker
c = get_chunker('ICA')
print(type(c).__name__)  # DoclingChunker
"

# 4. Testes existentes não devem quebrar
pytest tests/ -x -q
```
