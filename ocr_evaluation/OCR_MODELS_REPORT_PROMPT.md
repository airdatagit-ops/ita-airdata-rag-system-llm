# Prompt: Relatório Técnico Comparativo dos Modelos de OCR

## Objetivo

Gere um relatório técnico detalhado e comparativo sobre os 6 modelos de OCR listados abaixo, que estão sendo avaliados em um sistema de RAG (Retrieval-Augmented Generation) para documentos normativos da aviação civil brasileira (DECEA, AISWEB, AIP, NOTAM, SISLAER, etc.).

O cenário-alvo é a extração de texto de PDFs escaneados contendo documentos regulatórios e operacionais da aviação, que possuem:
- Texto técnico denso com siglas e termos específicos (NOTAM, METAR, TAF, SIGMET, ILS, VOR, etc.)
- Tabelas com dados numéricos e coordenadas geográficas
- Layouts com múltiplas colunas e cabeçalhos hierárquicos
- PDFs com qualidade variável de digitalização (alguns com ruído, inclinação, baixo contraste)
- Texto em português com intercalação de termos em inglês e códigos ICAO
- Presença de caracteres especiais como °, ', ", /, coordenadas no formato DD°MM'SS"

---

## Modelos a Serem Descritos

Para **cada um** dos 6 modelos abaixo, escreva uma seção do relatório contendo as subseções indicadas.

### 1. Tesseract OCR

**Implementação utilizada:** `pytesseract` (binding Python) com pipeline de preprocessing via OpenCV.

**Configuração no projeto:**
- DPI de conversão: 300
- Idiomas: `por+eng`
- Engine mode: OEM 3 (LSTM)
- Page segmentation: PSM 6 (bloco uniforme de texto)
- Preprocessing: conversão para escala de cinza → binarização de Otsu → denoising (fastNlMeansDenoising, h=10) → correção de inclinação (deskew via minAreaRect) → upscale se largura < 2000px

**Subseções obrigatórias:**
- **O que é o modelo:** Origem, histórico (HP Labs → Google → open source), versão 4+ com LSTM. Explicar que é um motor OCR clássico que evoluiu para usar redes neurais.
- **Arquitetura:** Descrever o pipeline interno do Tesseract 4/5: detecção de layout (page segmentation) → reconhecimento de linhas via rede LSTM bidirecional → pós-processamento com dicionário. Mencionar a diferença entre o engine legado (tessedata) e o neural (tessdata_best).
- **Características interessantes:** Suporte a 100+ idiomas, customizável via treinamento fino (tesstrain), depende criticamente de preprocessing de imagem, é o backend padrão de muitas bibliotecas (ex: unstructured).
- **Pontos positivos para o cenário:** Rápido, leve, sem dependência de GPU, bom suporte ao português, configurável por PSM para diferentes layouts, preprocessing manual permite ajustar para qualidade de imagem específica.
- **Pontos negativos para o cenário:** Sensível a qualidade de imagem, layouts complexos (tabelas, múltiplas colunas) causam erros de ordem de leitura, não preserva estrutura do documento, siglas curtas podem ser confundidas com ruído, deskew e binarização nem sempre são suficientes para PDFs de baixa qualidade.

---

### 2. EasyOCR

**Implementação utilizada:** `easyocr.Reader` com suporte a português e inglês.

**Configuração no projeto:**
- DPI de conversão: 250
- Idiomas: `["pt", "en"]`
- GPU: desabilitada por padrão (configurável via env var `EASYOCR_USE_GPU`)
- Modo de saída: `detail=0, paragraph=True` (texto concatenado em parágrafos)

**Subseções obrigatórias:**
- **O que é o modelo:** Projeto open source (JaidedAI), baseado em deep learning end-to-end. Descrever que é uma solução "baterias incluídas" que combina detecção e reconhecimento em um único pacote pip.
- **Arquitetura:** Detector de texto: CRAFT (Character-Region Awareness for Text Detection). Reconhecedor: rede CRNN (Convolutional Recurrent Neural Network) com backbone ResNet + sequência BiLSTM + decoder CTC. Explicar o pipeline: detecção de regiões de texto → recorte e alinhamento → reconhecimento por região → agrupamento em parágrafos.
- **Características interessantes:** Suporte a 80+ idiomas out-of-the-box, API extremamente simples (3 linhas de código), modelos pré-treinados leves (~100MB), modo paragraph que agrupa texto automaticamente.
- **Pontos positivos para o cenário:** Fácil de instalar e usar, lida razoavelmente com textos multilíngue (pt+en), modo paragraph ajuda a reconstruir blocos de texto, não precisa de preprocessing manual.
- **Pontos negativos para o cenário:** DPI mais baixo (250) pode perder detalhes finos como caracteres de coordenadas, ordem de leitura em layouts complexos não é garantida, reconhecimento de tabelas é fraco, performance em CPU é lenta comparada a Tesseract, menos preciso em texto muito pequeno ou com muitas siglas curtas.

---

### 3. docTR (Mindee)

**Implementação utilizada:** `python-doctr[torch]` com predictor end-to-end.

**Configuração no projeto:**
- Conversão PDF→imagem: gerenciada internamente pelo docTR
- Arquitetura de detecção: `db_resnet50` (DBNet com backbone ResNet-50)
- Arquitetura de reconhecimento: `crnn_vgg16_bn` (CRNN com backbone VGG-16 + BatchNorm)
- Pesos: pré-treinados

**Subseções obrigatórias:**
- **O que é o modelo:** Biblioteca de OCR da empresa Mindee (startup francesa de processamento de documentos), open source, com foco em documentos estruturados. Descrever a filosofia de "document understanding" vs. simples OCR.
- **Arquitetura:** Dois estágios: (1) **Detecção** — DBNet (Differentiable Binarization Network) com backbone ResNet-50 que produz mapas de probabilidade de texto e mapas de threshold, permitindo binarização adaptativa aprendida. (2) **Reconhecimento** — CRNN com backbone VGG-16-BN que recebe recortes de texto e produz sequências de caracteres via decoder CTC. Explicar que a saída preserva hierarquia: página → bloco → linha → palavra, com coordenadas de bounding box.
- **Características interessantes:** Preserva a estrutura do documento (hierarquia página/bloco/linha/palavra), saída exportável para JSON com coordenadas, pipeline modular (pode trocar detector e reconhecedor), suporte a backend PyTorch e TensorFlow, detecção robusta a orientação.
- **Pontos positivos para o cenário:** Excelente preservação da ordem de leitura e estrutura, detecção robusta de blocos de texto em layouts complexos, saída estruturada permite reconstrução de tabelas, bom equilíbrio entre precisão e velocidade.
- **Pontos negativos para o cenário:** Modelos pré-treinados focados em inglês/francês — reconhecimento de caracteres acentuados em português pode ser inferior. Não tem modelo de linguagem integrado para correção de OCR. O backend PyTorch consome mais memória. Pode ter dificuldade com fontes muito pequenas dos cabeçalhos de AIP.

---

### 4. Unstructured

**Implementação utilizada:** `unstructured[pdf]` com estratégia `ocr_only`.

**Configuração no projeto:**
- Estratégia de partição: `"ocr_only"` (força OCR mesmo em PDFs com texto digital)
- Idioma: `["por"]`
- Backend OCR: Tesseract (padrão da biblioteca)
- Classifica elementos por tipo: Title, NarrativeText, Table, ListItem, etc.

**Subseções obrigatórias:**
- **O que é o modelo:** Biblioteca open source da empresa Unstructured.io, focada em ETL de documentos para LLMs e pipelines de RAG. Não é um motor OCR em si, mas um framework de particionamento que orquestra OCR + análise de layout + classificação de elementos.
- **Arquitetura:** Pipeline multi-estágio: (1) Detecção de layout via modelo de detecção de objetos (Detectron2 ou YOLOX) para identificar regiões semânticas (título, parágrafo, tabela, imagem). (2) OCR por região usando Tesseract ou PaddleOCR como backend. (3) Classificação de tipo do elemento (Title, NarrativeText, Table, etc.). (4) Ordenação de elementos na sequência de leitura. Explicar que a estratégia `"ocr_only"` bypassa a extração de texto digital e força OCR puro.
- **Características interessantes:** Saída semanticamente tipada (sabe distinguir título de parágrafo de tabela), pré-processamento focado em pipelines de RAG, integração nativa com LangChain e LlamaIndex, suporte a múltiplos formatos além de PDF (DOCX, HTML, imagens).
- **Pontos positivos para o cenário:** Classificação semântica de elementos é valiosa para chunking inteligente no RAG, identifica tabelas separadamente, preserva hierarquia do documento, facilita downstream processing.
- **Pontos negativos para o cenário:** Na estratégia `ocr_only`, usa Tesseract como backend — herda todas as limitações do Tesseract para reconhecimento. Adiciona overhead significativo de processamento para a camada de layout. Instalação pesada com muitas dependências. A classificação de elementos pode errar em documentos com layout não-padrão (ex: NOTAMs com formato telegráfico).

---

### 5. Chandra OCR 2

**Implementação utilizada:** `chandra-ocr` com inferência via HuggingFace local (`datalab-to/chandra-ocr-2`).

**Configuração no projeto:**
- Método de inferência: `"hf"` (modelo HuggingFace local)
- Processamento: batch (todas as páginas de uma vez via `BatchInputItem`)
- Formato de saída: Markdown
- Retorna `None` se indisponível (graceful degradation)

**Subseções obrigatórias:**
- **O que é o modelo:** Modelo de OCR baseado em vision-language model (VLM), desenvolvido pelo Datalab. É um modelo multimodal que utiliza um modelo de linguagem grande (LLM) para "ler" imagens de documentos, similar à abordagem de modelos como Nougat e GOT-OCR. É a abordagem mais moderna da lista, representando a fronteira do OCR baseado em IA generativa.
- **Arquitetura:** Abordagem encoder-decoder baseada em transformer: (1) **Encoder visual** que converte a imagem do documento em embeddings visuais (provavelmente baseado em ViT — Vision Transformer). (2) **Decoder de linguagem** (LLM) que autoregressivamente gera o texto extraído token a token, condicionado nos embeddings visuais. A saída é em formato Markdown, o que significa que o modelo não apenas reconhece caracteres, mas também infere a estrutura do documento (cabeçalhos, listas, tabelas em Markdown).
- **Características interessantes:** Saída em Markdown preservando estrutura do documento, não precisa de pipeline separado de detecção de texto, lida com layouts complexos de forma holística (o modelo "entende" o documento visualmente), pode inferir estrutura mesmo em documentos com layout degradado.
- **Pontos positivos para o cenário:** Saída Markdown é diretamente utilizável para chunking e indexação em RAG, compreensão holística do layout pode lidar melhor com documentos complexos como AIPs com tabelas e cabeçalhos, abordagem generativa pode ser mais robusta a ruído e degradação de imagem.
- **Pontos negativos para o cenário:** Modelo pesado — requer GPU com memória significativa para inferência em tempo razoável, pode "alucinar" texto que não existe no documento (problema inerente a modelos generativos), latência alta por página, suporte específico ao português pode ser limitado dependendo dos dados de treinamento, nem sempre disponível em todos os ambientes de instalação.

---

### 6. PaddleOCR (PP-OCRv4)

**Implementação utilizada:** `paddleocr` v3.x com API `predict()`.

**Configuração no projeto:**
- DPI de conversão: 300
- Idioma: `"pt"` (português)
- API: PaddleOCR 3.x (`predict()` ao invés do legado `ocr()`)
- Workaround: desabilita flag `FLAGS_enable_pir_in_executor` para compatibilidade com CPUs x86 + oneDNN no Paddle 3.3.x
- Saída: `OCRResult` com `rec_texts` e `rec_scores` (textos + confiança)

**Subseções obrigatórias:**
- **O que é o modelo:** Sistema de OCR desenvolvido pela Baidu (PaddlePaddle ecosystem), um dos mais populares globalmente com >40k stars no GitHub. PP-OCRv4 é a quarta geração do sistema ultra-leve da Baidu. Descrever que é um sistema de OCR completo otimizado para deployment em produção.
- **Arquitetura:** Pipeline de 3 estágios: (1) **Detecção de texto** — DB++ (Differentiable Binarization aprimorado) com backbone leve para localizar regiões de texto na imagem. (2) **Classificação de direção** — classificador de ângulo que detecta texto rotacionado 180° (use_angle_cls). (3) **Reconhecimento** — SVTR (Scene Text Recognition with Visual Transformer) otimizado, usando attention-based decoder para reconhecer caracteres em cada região detectada. O PP-OCRv4 introduz melhorias de knowledge distillation e data augmentation. Explicar a diferença entre o modelo server (mais preciso, mais pesado) e o modelo mobile (mais leve).
- **Características interessantes:** Extremamente otimizado para produção (modelos compactos <15MB para mobile), suporte a 80+ idiomas, oferece score de confiança por região, pipeline modularizável, suporte nativo a GPU e CPU, comunidade muito ativa com atualizações frequentes.
- **Pontos positivos para o cenário:** Bom equilíbrio entre precisão e velocidade, modelos leves que rodam bem em CPU, score de confiança permite filtrar extrações de baixa qualidade, suporte ao português, DPI alto (300) captura detalhes de siglas e caracteres especiais, otimizado para deployment.
- **Pontos negativos para o cenário:** Incompatibilidades do framework PaddlePaddle com certos ambientes (workaround PIR necessário), documentação centralizada em chinês, modelos pré-treinados focam em chinês/inglês — português é idioma secundário, não preserva estrutura de documento (saída é texto linear), tabelas complexas são extraídas como texto corrido.

---

## Formato do Relatório

O relatório final deve seguir a seguinte estrutura:

```markdown
# Relatório Técnico: Modelos de OCR para Documentos Normativos da Aviação Civil

## 1. Introdução
Breve descrição do contexto (sistema RAG + aviação civil), objetivo do benchmark,
e critérios de avaliação relevantes para o domínio.

## 2. Visão Geral dos Modelos
Tabela comparativa rápida com: nome, tipo de abordagem (clássico/deep learning/VLM),
ano de última versão, licença, necessidade de GPU.

## 3. Análise Detalhada por Modelo

### 3.1 Tesseract OCR
#### 3.1.1 Descrição do Modelo
#### 3.1.2 Arquitetura
#### 3.1.3 Características Relevantes
#### 3.1.4 Pontos Positivos para Documentos Normativos da Aviação
#### 3.1.5 Pontos Negativos e Limitações

### 3.2 EasyOCR
(mesma estrutura)

### 3.3 docTR (Mindee)
(mesma estrutura)

### 3.4 Unstructured
(mesma estrutura)

### 3.5 Chandra OCR 2
(mesma estrutura)

### 3.6 PaddleOCR (PP-OCRv4)
(mesma estrutura)

## 4. Análise Comparativa
### 4.1 Comparação de Abordagens Arquiteturais
Agrupar por paradigma: OCR clássico (Tesseract), deep learning end-to-end
(EasyOCR, docTR, PaddleOCR), framework de pipeline (Unstructured),
vision-language model (Chandra).

### 4.2 Trade-offs Relevantes para o Domínio
Discutir: precisão vs. velocidade, necessidade de GPU, preservação de estrutura,
robustez a documentos degradados, suporte ao português.

### 4.3 Recomendações por Cenário de Uso
- Melhor para produção em CPU limitada
- Melhor para máxima precisão com GPU disponível
- Melhor para preservação de estrutura/tabelas
- Melhor para pipeline de RAG

## 5. Conclusão
Síntese das descobertas e recomendações para o sistema RAG de documentos
normativos da aviação civil brasileira.
```

## Restrições

- O relatório deve ser técnico mas legível, adequado para um público de engenheiros de software e pesquisadores.
- Sempre que possível, cite os papers ou repositórios originais de cada modelo.
- Evite afirmações genéricas — seja específico sobre **por que** uma característica é positiva ou negativa para o cenário de documentos normativos da aviação.
- O relatório deve ter entre 3000 e 5000 palavras.
- Escreva em **português brasileiro**.
