# Prompt: Implementação do Pipeline de Benchmark de OCR

## Contexto do Projeto

Este repositório é um sistema de RAG (Retrieval-Augmented Generation) para documentos normativos da aviação civil brasileira (DECEA, AISWEB, AIP, NOTAM, etc.). O pipeline precisa extrair texto de PDFs, incluindo PDFs escaneados que não possuem camada de texto digital acessível.

O objetivo desta tarefa é criar um módulo completo de avaliação de modelos de OCR, localizado em uma pasta `ocr_evaluation/` na raiz do repositório. O módulo deve testar os seguintes modelos:

- **Tesseract OCR** (com preprocessing de imagem)
- **EasyOCR**
- **docTR** (Mindee)
- **unstructured[pdf]**
- **Chandra OCR 2**
- **PaddleOCR**

---

## Convenções do Projeto

- Python 3.12
- Linha máxima: 120 caracteres (conforme `pyproject.toml`)
- Logger: `from loguru import logger`
- Dependências opcionais devem ser importadas dentro de blocos `try/except ImportError` com mensagens de aviso claras
- Paths devem usar `pathlib.Path` em vez de `os.path`
- Nunca usar caminhos hardcoded — tudo relativo à pasta `ocr_evaluation/`

---

## Estrutura de Pastas a Criar

```
ocr_evaluation/
├── __init__.py
├── test_data/                    # PDFs de entrada (usuário coloca os arquivos aqui)
│   └── .gitkeep
├── correct_texts/                # Ground truth gerado pelo pdfplumber
│   └── .gitkeep
├── results/                      # JSONs e CSVs com resultados do benchmark
│   └── .gitkeep
├── extract_ground_truth.py       # Extrai texto digital via pdfplumber
├── ocr_tesseract.py              # OCR: Tesseract com preprocessing
├── ocr_easyocr.py                # OCR: EasyOCR
├── ocr_doctr.py                  # OCR: docTR (Mindee)
├── ocr_unstructured.py           # OCR: unstructured[pdf]
├── ocr_chandra.py                # OCR: Chandra OCR 2
├── ocr_paddleocr.py              # OCR: PaddleOCR
├── metrics.py                    # Funções de cálculo de CER, WER e Domain Precision/Recall
└── benchmark.py                  # Script orquestrador — roda todos os OCRs e compara métricas
```

---

## Arquivo 1: `ocr_evaluation/__init__.py`

Arquivo vazio ou com um comentário de módulo simples.

---

## Arquivo 2: `ocr_evaluation/extract_ground_truth.py`

**Propósito:** Script auxiliar que percorre todos os arquivos `.pdf` em `test_data/`, extrai o texto digital (não-OCR) usando `pdfplumber`, e salva o resultado em `correct_texts/` com o mesmo nome do PDF, mas extensão `.txt`.

**Lógica esperada:**

1. Iterar sobre todos os arquivos `*.pdf` em `ocr_evaluation/test_data/`
2. Para cada PDF, abrir com `pdfplumber.open()`
3. Iterar sobre todas as páginas (`pdf.pages`) e extrair o texto com `page.extract_text()` ou `page.extract_text(layout=True)`
4. Concatenar o texto de todas as páginas com separadores de página (`\n\f\n` ou `\n--- Página {n} ---\n`)
5. Salvar o resultado em `ocr_evaluation/correct_texts/{nome_do_pdf}.txt` com encoding UTF-8
6. Logar progresso: nome do arquivo, número de páginas, número de caracteres extraídos
7. Se uma página não tiver texto digital (página escaneada), logar um aviso e continuar
8. Exibir ao final um resumo: quantos PDFs processados, quantos com texto vazio

**Execute com:** `python -m ocr_evaluation.extract_ground_truth`

---

## Arquivo 3: `ocr_evaluation/ocr_tesseract.py`

**Propósito:** Extrair texto de PDFs via Tesseract OCR com um pipeline de preprocessing de imagem.

**Dependências:** `pytesseract`, `pdf2image`, `Pillow`, `opencv-python` (cv2), `numpy`

**Lógica esperada:**

1. Converter cada página do PDF em imagem usando `pdf2image.convert_from_path()` com DPI=300
2. Para cada imagem, aplicar o seguinte pipeline de preprocessing com OpenCV/Pillow **nesta ordem**:
   a. Converter para escala de cinza
   b. Aplicar limiarização adaptativa (Otsu's binarization): `cv2.threshold(..., cv2.THRESH_BINARY + cv2.THRESH_OTSU)`
   c. Remoção de ruído leve: `cv2.fastNlMeansDenoising`
   d. Deskew (correção de inclinação): detectar ângulo de inclinação via `cv2.minAreaRect` sobre contornos e rotacionar se o ângulo for maior que 0.5 graus
   e. Redimensionar para 300 DPI equivalente se a imagem estiver abaixo de 2000px de largura
3. Rodar `pytesseract.image_to_string(imagem, lang='por+eng', config='--oem 3 --psm 6')`
4. Concatenar resultado de todas as páginas
5. Retornar o texto completo

**Interface pública obrigatória:**
```python
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extrai texto de um PDF usando Tesseract OCR com preprocessing."""
    ...
```

**Execute com:** `python -m ocr_evaluation.ocr_tesseract <caminho_do_pdf>`
(quando executado diretamente, imprime o texto extraído no stdout)

---

## Arquivo 4: `ocr_evaluation/ocr_easyocr.py`

**Propósito:** Extrair texto de PDFs usando EasyOCR.

**Dependências:** `easyocr`, `pdf2image`, `numpy`

**Lógica esperada:**

1. Inicializar `easyocr.Reader(['pt', 'en'], gpu=False)` — usar padrão CPU para reprodutibilidade; permitir override via variável de ambiente `EASYOCR_USE_GPU=1`
2. Converter cada página do PDF em imagem com `pdf2image.convert_from_path()` com DPI=250
3. Para cada imagem, converter para array numpy e rodar `reader.readtext(img_array, detail=0, paragraph=True)`
4. Juntar os resultados em texto com quebras de linha
5. Concatenar todas as páginas com separador de página
6. O reader deve ser instanciado apenas uma vez (padrão singleton ou passado como argumento)

**Interface pública obrigatória:**
```python
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extrai texto de um PDF usando EasyOCR."""
    ...
```

**Execute com:** `python -m ocr_evaluation.ocr_easyocr <caminho_do_pdf>`

---

## Arquivo 5: `ocr_evaluation/ocr_doctr.py`

**Propósito:** Extrair texto de PDFs usando docTR da Mindee.

**Dependências:** `python-doctr[torch]` ou `python-doctr[tf]`, `pdf2image`

**Lógica esperada:**

1. Tentar importar `from doctr.io import DocumentFile` e `from doctr.models import ocr_predictor`; se não disponível, logar erro claro com instrução de instalação: `pip install "python-doctr[torch]"`
2. Instanciar o predictor com detecção e reconhecimento pré-treinados: `ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)`
3. Carregar o PDF diretamente com `DocumentFile.from_pdf(str(pdf_path))`
4. Rodar `result = model(doc)`
5. Exportar para JSON: `result.export()` e extrair o texto percorrendo a estrutura de blocos/linhas/palavras
6. Preservar a ordem de leitura (página → bloco → linha → palavra) ao montar o texto
7. Concatenar todas as páginas

**Interface pública obrigatória:**
```python
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extrai texto de um PDF usando docTR."""
    ...
```

**Execute com:** `python -m ocr_evaluation.ocr_doctr <caminho_do_pdf>`

---

## Arquivo 6: `ocr_evaluation/ocr_unstructured.py`

**Propósito:** Extrair texto de PDFs usando a biblioteca `unstructured` com estratégia OCR para PDFs escaneados.

**Dependências:** `unstructured[pdf]`, `pdf2image`, `pytesseract` (backend padrão do unstructured)

**Lógica esperada:**

1. Tentar importar `from unstructured.partition.pdf import partition_pdf`; se não disponível, logar erro com instrução: `pip install "unstructured[pdf]"`
2. Chamar `partition_pdf(filename=str(pdf_path), strategy="ocr_only", languages=["por"])` para forçar OCR mesmo em PDFs com texto digital (para fins de benchmark)
3. Iterar sobre os elementos retornados, que possuem tipos como `Title`, `NarrativeText`, `Table`, `ListItem`, etc.
4. Extrair o texto de cada elemento via `element.text`
5. Montar o texto final preservando a sequência dos elementos; adicionar quebra dupla entre elementos de tipos diferentes (ex: após um `Title`)
6. Logar a contagem de elementos por tipo ao final (ex: `Title: 5, NarrativeText: 32, Table: 8`)

**Interface pública obrigatória:**
```python
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extrai texto de um PDF usando unstructured[pdf]."""
    ...
```

**Execute com:** `python -m ocr_evaluation.ocr_unstructured <caminho_do_pdf>`

---

## Arquivo 7: `ocr_evaluation/ocr_chandra.py`

**Propósito:** Extrair texto de PDFs usando Chandra OCR 2.

**Dependências:** `chandra-ocr` (se disponível via pip)

**Lógica esperada:**

1. Tentar importar o pacote Chandra OCR 2; envolver em `try/except ImportError` com mensagem de erro clara e instrução de instalação
2. Se não disponível, a função `extract_text_from_pdf` deve retornar `None` e logar um aviso: `"Chandra OCR 2 não disponível neste ambiente. Pulando..."`
3. Se disponível, seguir a API documentada pelo pacote para processar PDFs em português
4. O script deve ser resiliente: se a importação falhar, o benchmark principal deve conseguir ignorar este modelo graciosamente

**Interface pública obrigatória:**
```python
def extract_text_from_pdf(pdf_path: Path) -> str | None:
    """Extrai texto de um PDF usando Chandra OCR 2. Retorna None se não disponível."""
    ...
```

**Execute com:** `python -m ocr_evaluation.ocr_chandra <caminho_do_pdf>`

---

## Arquivo 8: `ocr_evaluation/ocr_paddleocr.py`

**Propósito:** Extrair texto de PDFs usando PaddleOCR (PP-OCRv4).

**Dependências:** `paddlepaddle`, `paddleocr`, `pdf2image`, `numpy`, `Pillow`

**Lógica esperada:**

1. Tentar importar `from paddleocr import PaddleOCR`; se não disponível, logar erro com instrução de instalação
2. Instanciar com `PaddleOCR(use_angle_cls=True, lang='pt', use_gpu=False, show_log=False)` — GPU configurável via env `PADDLE_USE_GPU=1`
3. Suprimir os logs verbosos do PaddleOCR: `logging.getLogger('ppocr').setLevel(logging.WARNING)` antes de instanciar
4. Converter cada página do PDF em imagem com DPI=300 via `pdf2image`
5. Para cada imagem, converter para array numpy e rodar `ocr.ocr(img_array, cls=True)`
6. Extrair o texto da estrutura retornada: `result[0][i][1][0]` (texto) com `result[0][i][1][1]` (confiança)
7. Opcionalmente, logar a confiança média por página
8. Concatenar todas as páginas

**Interface pública obrigatória:**
```python
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extrai texto de um PDF usando PaddleOCR."""
    ...
```

**Execute com:** `python -m ocr_evaluation.ocr_paddleocr <caminho_do_pdf>`

---

## Arquivo 9: `ocr_evaluation/metrics.py`

**Propósito:** Módulo com as funções de cálculo das métricas de avaliação.

**Dependências:** `jiwer` (para CER e WER), sem outras dependências externas para as demais métricas.

### Função 1: `calculate_cer(reference: str, hypothesis: str) -> float`

- Usar `jiwer` com transformações de caractere:
  - `jiwer.cer(reference, hypothesis)`
- Retornar o CER como float entre 0.0 e 1.0 (0% = perfeito, 100% = sem acerto)
- Se `reference` estiver vazio, retornar `float('nan')` e logar aviso

### Função 2: `calculate_wer(reference: str, hypothesis: str) -> float`

- Usar `jiwer.wer(reference, hypothesis)` com as seguintes transformações aplicadas antes:
  - `jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])`
- Retornar o WER como float entre 0.0 e 1.0
- Se `reference` estiver vazio, retornar `float('nan')` e logar aviso

### Função 3: `calculate_domain_precision_recall(hypothesis: str) -> dict`

**Lista de termos do domínio da aviação** — o texto de hipótese não tem ground truth para isso, então meça presença no texto OCR em relação à lista completa abaixo. A lista deve estar definida como constante `AVIATION_DOMAIN_TERMS` no próprio módulo:

```
Siglas operacionais:
NOTAM, AIP, AIRAC, SIGMET, METAR, TAF, ATIS, PIREP, SPECI, SNOWTAM

Órgãos e autoridades:
ICAO, DECEA, ANAC, ICEA, CISCEA, DPV, CGNA, SRPV

Parâmetros meteorológicos e de altitude:
QNH, QFE, QNE, QDM, CAVOK, TEMPO, BECMG, VRB, PROB, NOSIG

Radionavegação e procedimentos:
ILS, VOR, NDB, DME, RNAV, RNP, GNSS, GPS, PAPI, VASI, GPWS, TCAS

Infraestrutura aeroportuária:
TWY, RWY, ACFT, THR, ARP, APRON, HLD, ILS

Espaço aéreo:
CTR, TMA, FIR, UIR, ADIZ, ATZ, CTA, ACC, APP, TWR, AFIS, UNICOM

Procedimentos de voo:
SID, STAR, IFR, VFR, SVFR, IMC, VMC, OCA, OCH, MDA, DH, DA, MDH

Níveis de voo:
FL, MSL, AGL, AMSL, QFE, QNH

Português aeronáutico:
aeródromo, pista, taxiway, decolagem, pouso, pousar, decolar,
espaço aéreo, zona de controle, rota aérea, plano de voo,
autorização, restrição, proibição, altitude, nível de voo,
visibilidade, teto, nuvem, turbulência, vento, temperatura,
aeronave, piloto, controlador, torre, aproximação, radar
```

**Lógica:**
1. Normalizar o texto de hipótese: lowercase, sem pontuação extra
2. Para cada termo da lista, verificar se ele aparece no texto (case-insensitive, palavra inteira com `\b`)
3. Calcular:
   - `precision` = (termos encontrados no texto) / (total de termos encontrados no texto que estão na lista) — ou seja, dos "tokens relevantes" extraídos, quantos estão na lista
   - `recall` = (termos da lista encontrados no texto) / (total de termos na lista)
   - `f1` = harmônica entre precision e recall
4. Retornar um dict: `{"precision": float, "recall": float, "f1": float, "found_terms": list[str], "missing_terms": list[str]}`

**Nota:** Para esta métrica, o "recall" mede quantos termos do domínio o OCR conseguiu preservar no texto extraído — extremamente relevante para RAG onde esses termos são as queries mais comuns.

### Função 4: `format_results_table(results: dict) -> str`

- Recebe o dict de resultados com estrutura `{model_name: {metric: valor}}`
- Retorna uma string formatada como tabela ASCII para exibição no terminal
- Ordenar modelos pelo CER (menor primeiro)

---

## Arquivo 10: `ocr_evaluation/benchmark.py`

**Propósito:** Script orquestrador principal. Executa todos os OCRs sobre os PDFs de teste, calcula as métricas e salva os resultados.

**Execute com:** `python -m ocr_evaluation.benchmark [--pdf <arquivo_específico>] [--models all|tesseract,easyocr,...]`

### Lógica completa esperada:

**1. Parse de argumentos (`argparse`):**
- `--pdf`: caminho para um PDF específico (opcional; se omitido, processa todos em `test_data/`)
- `--models`: lista separada por vírgula dos modelos a rodar (padrão: `all`)
  - Valores válidos: `tesseract`, `easyocr`, `doctr`, `unstructured`, `chandra`, `paddleocr`
- `--output-dir`: pasta para salvar resultados (padrão: `ocr_evaluation/results/`)
- `--skip-extraction`: flag booleana — se passada, pula a extração e usa resultados já salvos

**2. Descoberta de PDFs:**
- Listar todos os `*.pdf` em `ocr_evaluation/test_data/`
- Se nenhum PDF encontrado, exibir mensagem de ajuda clara e encerrar

**3. Verificação de Ground Truth:**
- Para cada PDF, verificar se existe o arquivo correspondente em `correct_texts/`
- Se não existir, logar aviso: `"Ground truth não encontrado para {pdf}. Rode extract_ground_truth.py primeiro."`
- PDFs sem ground truth são ignorados das métricas de CER/WER, mas ainda processados para salvar o output OCR

**4. Execução dos OCRs:**
- Para cada modelo selecionado, importar dinamicamente o módulo correspondente
- Usar `importlib.import_module(f"ocr_evaluation.ocr_{model_name}")`
- Chamar `module.extract_text_from_pdf(pdf_path)`
- Medir o tempo de execução com `time.perf_counter()`
- Salvar o texto extraído em `ocr_evaluation/results/{model_name}/{pdf_name}.txt`
- Se o modelo retornar `None` (ex: Chandra não disponível), marcar como `SKIPPED`
- Tratar exceções por modelo: se um OCR falhar em um PDF específico, logar o erro e continuar com os demais

**5. Cálculo de Métricas (por PDF, por modelo):**
- Carregar o ground truth do arquivo `correct_texts/{pdf_name}.txt`
- Calcular `CER`, `WER` e `domain_precision_recall` usando as funções de `metrics.py`
- Armazenar resultado em estrutura:
  ```python
  {
      "pdf": str,
      "model": str,
      "cer": float,
      "wer": float,
      "domain_recall": float,
      "domain_precision": float,
      "domain_f1": float,
      "found_terms": list[str],
      "missing_terms": list[str],
      "elapsed_seconds": float,
      "pages": int,
      "chars_extracted": int
  }
  ```

**6. Agregação de Resultados:**
- Calcular médias de CER, WER e Domain F1 por modelo (média sobre todos os PDFs)
- Exibir tabela resumo no terminal usando `metrics.format_results_table()`
- Exibir tabela detalhada com resultados por PDF

**7. Exportação de Resultados:**
- Salvar todos os resultados detalhados em `ocr_evaluation/results/details_{timestamp}.csv`
- Salvar o resumo agregado em `ocr_evaluation/results/summary_{timestamp}.csv`
- Salvar os resultados completos (incluindo listas de termos) em `ocr_evaluation/results/full_{timestamp}.json`
- O `timestamp` deve seguir o formato `%Y%m%dT%H%M%S`

**8. Exibição Final:**
- Imprimir no terminal:
  ```
  ============================================================
  BENCHMARK OCR — RESUMO FINAL
  ============================================================
  Modelos testados: 6
  PDFs processados: N
  Timestamp: YYYY-MM-DDTHH:MM:SS
  
  RANKING POR CER (menor = melhor):
  1. docTR          CER: 2.3%   WER: 4.1%   Domain F1: 0.91   Tempo: 45.2s
  2. PaddleOCR      CER: 3.1%   WER: 5.8%   Domain F1: 0.88   Tempo: 38.1s
  ...
  
  Resultados detalhados salvos em: ocr_evaluation/results/
  ============================================================
  ```

---

## Arquivo 11: `.gitkeep` nas subpastas

Criar arquivos `.gitkeep` vazios em `test_data/`, `correct_texts/` e `results/` para que as pastas sejam versionadas no git mesmo vazias.

---

## Dependências a Documentar

No final do script `benchmark.py`, adicionar um comentário de bloco `# DEPENDÊNCIAS` com o seguinte conteúdo de referência para instalação:

```
# Tesseract:     pip install pytesseract pdf2image opencv-python-headless Pillow numpy
#                + sudo apt-get install tesseract-ocr tesseract-ocr-por
# EasyOCR:       pip install easyocr pdf2image
# docTR:         pip install "python-doctr[torch]" pdf2image
# unstructured:  pip install "unstructured[pdf]"
# Chandra:       pip install chandra-ocr  (se disponível; verificar repositório oficial)
# PaddleOCR:     pip install paddlepaddle paddleocr pdf2image
# Métricas:      pip install jiwer
```

---

## Restrições e Regras de Implementação

1. **Nenhum modelo deve ser importado no escopo global do `benchmark.py`** — usar `importlib` para importação dinâmica
2. **Falhas de um modelo não devem interromper o benchmark** — usar `try/except` com logging
3. **Todos os textos devem ser normalizados** antes de cálculo de métricas: remover múltiplos espaços em branco, normalizar quebras de linha (`\r\n` → `\n`), strip leading/trailing whitespace por página
4. **O benchmark deve ser idempotente**: rodar duas vezes com `--skip-extraction` deve produzir exatamente o mesmo resultado
5. **Progresso deve ser exibido** usando `tqdm` se disponível, ou log simples `logger.info(f"[{i}/{total}] Processando {pdf_name}...")`
6. **Não usar caminhos absolutos** — todos os paths devem ser relativos ao diretório `ocr_evaluation/` resolvido dinamicamente via `Path(__file__).parent`
7. **Encoding UTF-8 explícito** em todas as operações de leitura/escrita de arquivos de texto

---

## Exemplo de Uso Esperado ao Final

```bash
# Passo 1: Colocar PDFs de teste na pasta
cp /path/to/documentos/*.pdf ocr_evaluation/test_data/

# Passo 2: Gerar ground truth a partir do texto digital dos PDFs
python -m ocr_evaluation.extract_ground_truth

# Passo 3: Rodar o benchmark completo
python -m ocr_evaluation.benchmark

# Ou rodar apenas modelos específicos
python -m ocr_evaluation.benchmark --models tesseract,paddleocr,doctr

# Ou testar em um PDF específico
python -m ocr_evaluation.benchmark --pdf ocr_evaluation/test_data/AIP_Brasil.pdf
```
