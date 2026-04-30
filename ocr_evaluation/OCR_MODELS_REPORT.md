# Relatório Técnico: Modelos de OCR para Documentos Normativos da Aviação Civil

## 1. Introdução

Este relatório apresenta uma análise comparativa dos modelos de OCR (Optical Character Recognition) avaliados no contexto do sistema de RAG (Retrieval-Augmented Generation) para documentos normativos da aviação civil brasileira. O sistema processa documentos provenientes de fontes como DECEA, AISWEB, AIP, NOTAM e SISLAER, que possuem características particulares: texto técnico denso com siglas ICAO, tabelas com coordenadas geográficas, layouts de múltiplas colunas, e intercalação de português com termos em inglês.

O objetivo deste benchmark é identificar qual modelo de OCR oferece a melhor relação entre fidelidade na extração de texto e viabilidade operacional para alimentar o pipeline de RAG do projeto.

---

## 2. Visão Geral dos Modelos

| Modelo | Abordagem | Versão/Geração | Licença | GPU Necessária |
|---|---|---|---|---|
| Tesseract OCR | OCR clássico + LSTM | v4/v5 | Apache 2.0 | Não |
| EasyOCR | Deep learning end-to-end | v1.7+ | Apache 2.0 | Opcional |
| docTR (Mindee) | Deep learning end-to-end | v0.8+ | Apache 2.0 | Opcional |
| Unstructured | Framework de pipeline (Tesseract como backend) | v0.13+ | Apache 2.0 | Não |
| Chandra OCR 2 | Vision-Language Model (VLM) | v2 | Proprietária/Pesquisa | Sim (recomendada) |
| PaddleOCR | Deep learning end-to-end (PP-OCRv4) | v3.x | Apache 2.0 | Opcional |

---

## 3. Análise Detalhada por Modelo

### 3.1 Tesseract OCR

#### 3.1.1 Descrição do Modelo

O Tesseract é um dos motores de OCR mais antigos e conhecidos do mundo. Foi desenvolvido originalmente pela HP Labs nos anos 1980, adquirido e mantido pelo Google a partir de 2006, e hoje é um projeto open source ativo. A partir da versão 4, o Tesseract incorporou redes neurais LSTM para reconhecimento de texto, deixando de ser puramente baseado em regras e heurísticas. Continua sendo a referência de mercado para OCR open source e é o backend padrão de diversas bibliotecas de processamento de documentos.

#### 3.1.2 Arquitetura

O pipeline interno funciona em etapas sequenciais: primeiro, a **segmentação de página** (page segmentation) analisa o layout da imagem para identificar blocos de texto, colunas e linhas. Em seguida, cada linha de texto é processada por uma **rede LSTM bidirecional** que reconhece a sequência de caracteres. Por fim, um **pós-processamento com dicionário** corrige erros comuns de reconhecimento. O Tesseract oferece dois engines: o legado (tessedata), baseado em classificação de caracteres individuais, e o neural (tessdata\_best), que usa LSTM e é significativamente mais preciso.

No projeto, utilizamos o binding `pytesseract` com um pipeline de pré-processamento via OpenCV composto por seis etapas sequenciais:

1. **Conversão de PDF para imagem** via `pdf2image` a 300 DPI, produzindo uma imagem por página em formato PIL.
2. **Conversão para escala de cinza** via `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)`, eliminando a dimensão de cor desnecessária para o OCR.
3. **Binarização de Otsu** via `cv2.threshold` com flags `THRESH_BINARY + THRESH_OTSU`, que determina automaticamente o limiar ótimo de separação entre texto e fundo.
4. **Denoising leve** via `cv2.fastNlMeansDenoising` com parâmetro `h=10`, aplicado sobre a imagem binarizada para reduzir ruído residual sem borrar as bordas dos caracteres.
5. **Correção de inclinação (deskew)** via `cv2.minAreaRect` aplicado sobre os contornos escuros da imagem (pixels com valor abaixo de 128). O ângulo de rotação é detectado e normalizado; a rotação corretiva é aplicada somente se o ângulo absoluto for superior a 0,5 graus — caso contrário, a imagem é retornada sem alteração.
6. **Upscale condicional**: se a largura da imagem for inferior a 2000 pixels, ela é redimensionada para 2000 pixels de largura mantendo a proporção, via interpolação `INTER_CUBIC`.

Após o pré-processamento, o reconhecimento é realizado com `pytesseract.image_to_string` configurado com `lang="por+eng"` (português e inglês), engine LSTM (`--oem 3`) e modo de segmentação de página por bloco de texto uniforme (`--psm 6`).

#### 3.1.3 Características Relevantes

- Suporte a mais de 100 idiomas, incluindo bom suporte ao português
- Customizável via treinamento fino com o toolkit `tesstrain`
- Depende criticamente da qualidade do preprocessing de imagem
- É o backend padrão de muitas outras bibliotecas (por exemplo, o Unstructured o utiliza internamente)

#### 3.1.4 Pontos Positivos para Documentos Normativos da Aviação

Rápido e leve, roda inteiramente em CPU sem necessidade de GPU. O bom suporte ao português é relevante para os documentos normativos. A configuração por PSM permite adaptar a leitura para diferentes layouts. O preprocessing manual via OpenCV possibilita ajustar filtros para a qualidade específica dos PDFs do SISLAER.

#### 3.1.5 Pontos Negativos e Limitações

Sensível à qualidade da imagem de entrada — PDFs degradados com ruído ou baixo contraste produzem resultados ruins. Layouts complexos com tabelas e múltiplas colunas causam erros de ordem de leitura, o que é um problema direto para os documentos do AIP e do SISLAER, que frequentemente usam esses formatos. Siglas curtas de 2-3 letras (como FL, DH, DA), muito comuns na aviação, podem ser confundidas com ruído. Não preserva a estrutura do documento na saída.

---

### 3.2 EasyOCR

#### 3.2.1 Descrição do Modelo

EasyOCR é um projeto open source mantido pela JaidedAI, baseado inteiramente em deep learning. Sua proposta é ser uma solução "baterias incluídas": com poucas linhas de código, é possível extrair texto de imagens em mais de 80 idiomas. Combina detecção e reconhecimento de texto em um único pacote pip, sem necessidade de instalar dependências de sistema separadas.

#### 3.2.2 Arquitetura

Funciona em dois estágios. O **detector de texto** é baseado no CRAFT (Character-Region Awareness for Text Detection), uma rede que gera mapas de calor indicando regiões com caracteres e afinidade entre eles. A partir desses mapas, são extraídas bounding boxes de regiões de texto. O **reconhecedor** é uma rede CRNN (Convolutional Recurrent Neural Network) com backbone ResNet, seguida de camadas BiLSTM e decoder CTC, que converte cada recorte de texto em uma sequência de caracteres. O pipeline completo inclui recorte e alinhamento das regiões detectadas antes do reconhecimento, e opcionalmente agrupamento em parágrafos.

No projeto, utilizamos DPI 250, idiomas `["pt", "en"]`, GPU desabilitada por padrão, e modo `paragraph=True` para agrupamento automático do texto.

#### 3.2.3 Características Relevantes

- API extremamente simples (literalmente 3 linhas de código para uso básico)
- Modelos pré-treinados leves (~100 MB)
- Modo parágrafo que agrupa texto automaticamente
- Não exige preprocessing manual da imagem

#### 3.2.4 Pontos Positivos para Documentos Normativos da Aviação

Fácil de instalar e usar, o que agiliza a experimentação. O suporte bilíngue português-inglês é útil para documentos da aviação que misturam os dois idiomas. O modo paragraph facilita a reconstrução de blocos de texto para posterior chunking no RAG.

#### 3.2.5 Pontos Negativos e Limitações

O DPI mais baixo (250) pode causar perda de detalhes finos, como caracteres de coordenadas geográficas (°, ', "). A ordem de leitura em layouts complexos não é garantida. Reconhecimento de tabelas é fraco — a detecção do CRAFT não distingue colunas de tabelas de blocos de texto. Performance em CPU é significativamente mais lenta que o Tesseract. Siglas curtas e caracteres especiais da aviação tendem a ter taxas de erro mais altas.

---

### 3.3 docTR (Mindee)

#### 3.3.1 Descrição do Modelo

O docTR é uma biblioteca de OCR desenvolvida pela Mindee, uma startup francesa especializada em processamento inteligente de documentos. O diferencial do docTR é sua filosofia de "document understanding" em vez de simples reconhecimento de caracteres: ele busca compreender a estrutura do documento preservando hierarquia de página, bloco, linha e palavra, com coordenadas de bounding box para cada elemento.

#### 3.3.2 Arquitetura

Opera em dois estágios com modelos de deep learning independentes. A **detecção** usa a DBNet (Differentiable Binarization Network) com backbone ResNet-50, que produz mapas de probabilidade de texto e mapas de threshold aprendidos, permitindo uma binarização adaptativa mais precisa que métodos clássicos. O **reconhecimento** usa uma CRNN com backbone VGG-16 com BatchNorm, que recebe recortes de texto e os decodifica via CTC. A saída final é uma estrutura hierárquica (página → bloco → linha → palavra), exportável para JSON com coordenadas, o que permite reconstruir a posição espacial de cada elemento.

No projeto, usamos o backend PyTorch com detecção `db_resnet50` e reconhecimento `crnn_vgg16_bn`, ambos com pesos pré-treinados. A conversão de PDF para imagem é gerenciada internamente pelo docTR.

#### 3.3.3 Características Relevantes

- Saída estruturada com hierarquia completa (página/bloco/linha/palavra) e coordenadas
- Pipeline modular — é possível trocar os modelos de detecção e reconhecimento separadamente
- Suporte a backends PyTorch e TensorFlow
- Detecção robusta para orientações variadas de texto

#### 3.3.4 Pontos Positivos para Documentos Normativos da Aviação

A preservação da ordem de leitura e da estrutura do documento é valiosa para os PDFs do SISLAER e AIP, que possuem layouts com múltiplas seções e tabelas. A saída estruturada em blocos permite uma reconstrução mais fiel de tabelas com coordenadas e dados numéricos. Bom equilíbrio entre precisão e velocidade de processamento.

#### 3.3.5 Pontos Negativos e Limitações

Os modelos pré-treinados são focados em inglês e francês — o reconhecimento de caracteres acentuados em português (ç, ã, ô, é) pode ser inferior ao de modelos treinados especificamente para o idioma. Não possui modelo de linguagem integrado para correção contextual de erros de OCR. O backend PyTorch consome mais memória RAM. Fontes muito pequenas, comuns em cabeçalhos de tabelas dos documentos AIP, podem ser problemáticas.

---

### 3.4 Unstructured

#### 3.4.1 Descrição do Modelo

O Unstructured não é um motor de OCR propriamente dito, mas um framework de processamento e particionamento de documentos desenvolvido pela empresa Unstructured.io. Seu foco é o ETL (Extract, Transform, Load) de documentos para pipelines de LLMs e sistemas de RAG — exatamente o cenário deste projeto. Ele orquestra internamente um motor de OCR (Tesseract por padrão) combinado com análise de layout e classificação semântica dos elementos do documento.

#### 3.4.2 Arquitetura

Funciona como um pipeline multi-estágio: primeiro, um **modelo de detecção de layout** (baseado em Detectron2 ou YOLOX) identifica regiões semânticas na imagem (títulos, parágrafos, tabelas, imagens, listas). Em seguida, o **OCR** é aplicado região por região usando o Tesseract como backend. Depois, cada elemento é **classificado** por tipo (Title, NarrativeText, Table, ListItem, etc.). Por fim, os elementos são **ordenados** na sequência de leitura do documento. A estratégia `"ocr_only"` utilizada no benchmark força o OCR mesmo em PDFs que possuem camada de texto digital.

No projeto, utilizamos a estratégia `ocr_only` com idioma `["por"]` e o backend Tesseract padrão.

#### 3.4.3 Características Relevantes

- Saída semanticamente tipada: o resultado distingue título de parágrafo de tabela
- Integração nativa com LangChain e LlamaIndex para pipelines de RAG
- Suporte a múltiplos formatos de documento (PDF, DOCX, HTML, imagens)
- Focado especificamente em pré-processamento para LLMs

#### 3.4.4 Pontos Positivos para Documentos Normativos da Aviação

A classificação semântica dos elementos é extremamente valiosa para chunking inteligente no RAG — saber que um trecho é um "título" ou uma "tabela" permite estratégias de chunking mais sofisticadas. Identifica tabelas como elementos separados, o que é relevante para as tabelas de dados aeronáuticos. Projetado especificamente para o tipo de pipeline que o sistema implementa.

#### 3.4.5 Pontos Negativos e Limitações

Como usa o Tesseract internamente para o OCR propriamente dito, herda todas as limitações de reconhecimento do Tesseract. Adiciona overhead significativo de processamento para a camada de análise de layout. A instalação é pesada, com muitas dependências transitivas. A classificação de elementos pode errar em documentos com formato não-padrão, como NOTAMs com formato telegráfico ou tabelas com layouts incomuns do SISLAER.

---

### 3.5 Chandra OCR 2

#### 3.5.1 Descrição do Modelo

O Chandra OCR 2 representa a abordagem mais moderna desta avaliação. Desenvolvido pelo Datalab, é um modelo de OCR baseado em Vision-Language Model (VLM) — ou seja, utiliza um modelo de linguagem grande (LLM) para "ler" as imagens de documentos. Em vez de um pipeline tradicional de detecção + reconhecimento, o modelo recebe a imagem inteira e gera o texto autoregressivamente, token a token, de forma similar a como modelos como o GPT geram texto. Esta é a fronteira do OCR baseado em IA generativa, ocupando o mesmo espaço de modelos como Nougat (Meta) e GOT-OCR.

#### 3.5.2 Arquitetura

Utiliza uma abordagem encoder-decoder baseada em transformers. O **encoder visual**, provavelmente baseado em ViT (Vision Transformer), converte a imagem do documento em embeddings visuais. O **decoder de linguagem** (LLM) recebe esses embeddings e gera o texto autoregressivamente, condicionado no conteúdo visual. O diferencial é que a saída é em **formato Markdown**, o que significa que o modelo não apenas reconhece caracteres, mas também infere a estrutura semântica do documento — cabeçalhos, listas, tabelas.

No projeto, usamos inferência via HuggingFace local com o modelo `datalab-to/chandra-ocr-2`, processamento em batch via `BatchInputItem`, e graceful degradation (retorna `None` se indisponível).

#### 3.5.3 Características Relevantes

- Saída nativa em Markdown, preservando estrutura semântica do documento
- Não precisa de pipeline separado de detecção de texto — processa a imagem de forma holística
- Capaz de inferir estrutura mesmo em documentos com layout degradado
- Abordagem generativa que potencialmente entende o contexto do que está lendo

#### 3.5.4 Pontos Positivos para Documentos Normativos da Aviação

A saída em Markdown é diretamente utilizável para chunking e indexação no RAG, sem etapas intermediárias de reestruturação. A compreensão holística do layout pode lidar melhor com documentos complexos como AIPs e tabelas de dados SISLAER. A abordagem generativa pode ser mais robusta a ruído e degradação de imagem, pois o modelo pode "inferir" o texto provável a partir do contexto visual.

#### 3.5.5 Pontos Negativos e Limitações

Modelo pesado que requer GPU com memória significativa para inferência em tempo razoável — inviável para CPU em produção. Como todo modelo generativo, pode "alucinar" texto que não existe no documento, o que é particularmente perigoso para dados regulatórios da aviação onde precisão é crítica. Latência alta por página. O suporte ao português depende do treinamento e pode ser limitado. A disponibilidade de instalação é restrita e o modelo pode não funcionar em todos os ambientes.

---

### 3.6 PaddleOCR (PP-OCRv4)

#### 3.6.1 Descrição do Modelo

O PaddleOCR é o sistema de OCR desenvolvido pela Baidu dentro do ecossistema PaddlePaddle. Com mais de 40.000 estrelas no GitHub, é um dos projetos de OCR mais populares globalmente. O PP-OCRv4 é a quarta geração do sistema ultra-leve da Baidu, otimizado para deployment em produção com modelos compactos que funcionam em dispositivos com recursos limitados.

#### 3.6.2 Arquitetura

Funciona como um pipeline de 3 estágios. A **detecção de texto** usa o DB++ (versão aprimorada do Differentiable Binarization) com backbone leve, que localiza regiões de texto na imagem. A **classificação de direção** detecta texto rotacionado 180° para correção automática. O **reconhecimento** utiliza o SVTR (Scene Text Recognition with Visual Transformer) otimizado com decoder baseado em atenção. O PP-OCRv4 introduziu melhorias significativas via knowledge distillation e data augmentation. O sistema oferece variantes server (mais preciso) e mobile (mais leve, <15 MB).

No projeto, utilizamos a API v3.x com o método `predict()`, DPI 300, idioma português, e um workaround para desabilitar o flag `FLAGS_enable_pir_in_executor` necessário para compatibilidade com CPUs x86 + oneDNN no Paddle 3.3.x. A saída inclui textos reconhecidos (`rec_texts`) e scores de confiança (`rec_scores`).

#### 3.6.3 Características Relevantes

- Modelos extremamente compactos (versão mobile <15 MB)
- Score de confiança por região de texto reconhecido
- Suporte a 80+ idiomas
- Pipeline modularizável (detecção, classificação e reconhecimento podem ser trocados independentemente)
- Otimizado para deployment em produção

#### 3.6.4 Pontos Positivos para Documentos Normativos da Aviação

Bom equilíbrio entre precisão e velocidade. Modelos leves que rodam bem em CPU. Os scores de confiança permitem filtrar extrações de baixa qualidade — útil para identificar páginas problemáticas. O DPI alto (300) captura detalhes de siglas e caracteres especiais. Otimizado para ambiente de produção, o que é relevante para um sistema RAG que precisa processar documentos continuamente.

#### 3.6.5 Pontos Negativos e Limitações

O framework PaddlePaddle apresenta incompatibilidades com certos ambientes de execução — no projeto, foi necessário implementar um workaround para o PIR executor. A documentação é predominantemente em chinês, dificultando troubleshooting. Os modelos pré-treinados focam principalmente em chinês e inglês, com português como idioma secundário. Não preserva a estrutura do documento na saída (resultado é texto linear). Tabelas complexas são extraídas como texto corrido, perdendo a estrutura tabular.

---

## 4. Análise Comparativa

### 4.1 Comparação de Abordagens Arquiteturais

Os 6 modelos avaliados representam quatro paradigmas distintos de OCR:

**OCR Clássico + Neural (Tesseract):** Abordagem híbrida que combina segmentação de layout baseada em heurísticas com reconhecimento via LSTM. É a mais madura e estável, mas depende criticamente de preprocessing manual.

**Deep Learning End-to-End (EasyOCR, docTR, PaddleOCR):** Usam redes neurais tanto para detecção quanto reconhecimento. Diferem na escolha de arquiteturas (CRAFT vs. DBNet vs. DB++) e reconhecedores (CRNN vs. SVTR). São mais robustos a variações de imagem, mas exigem mais recursos de hardware.

**Framework de Pipeline (Unstructured):** Orquestra um motor OCR (Tesseract) com camadas adicionais de análise de layout e classificação semântica. Agrega valor para o downstream (RAG), mas herda as limitações do OCR subjacente.

**Vision-Language Model (Chandra OCR 2):** A abordagem mais disruptiva — trata OCR como uma tarefa de geração condicional. Potencialmente a mais precisa para documentos complexos, mas com o risco inerente de alucinações e o custo de GPU.

### 4.2 Trade-offs Relevantes para o Domínio

**Precisão vs. Velocidade:** Há um espectro claro. Tesseract é o mais rápido, seguido de PaddleOCR e docTR. EasyOCR e Unstructured são intermediários. Chandra é o mais lento, mas potencialmente o mais preciso em layouts complexos.

**CPU vs. GPU:** Tesseract, PaddleOCR e Unstructured funcionam bem em CPU. EasyOCR e docTR se beneficiam de GPU mas operam sem ela. Chandra essencialmente requer GPU.

**Preservação de Estrutura:** docTR e Unstructured preservam estrutura do documento (hierarquia de blocos e tipagem semântica respectivamente). Chandra gera Markdown estruturado. Os demais produzem texto linear.

**Suporte ao Português:** Tesseract tem o suporte mais maduro (pacote de idioma dedicado). PaddleOCR e EasyOCR possuem modelos para português mas como idioma secundário. docTR e Chandra dependem dos dados de treinamento, com foco primário em inglês.

### 4.3 Recomendações por Cenário de Uso

- **Melhor para produção em CPU limitada:** Tesseract (com preprocessing otimizado) ou PaddleOCR
- **Melhor para máxima precisão com GPU disponível:** Chandra OCR 2 ou docTR
- **Melhor para preservação de estrutura/tabelas:** docTR (hierarquia bloco/linha/palavra) ou Unstructured (tipagem semântica)
- **Melhor para pipeline de RAG:** Unstructured (integração nativa com LangChain/LlamaIndex, chunking semântico) ou Chandra (saída em Markdown)

---

## 5. O Benchmark Realizado

### 5.1 Metodologia

O benchmark foi executado sobre **4 documentos SISLAER** (Sistema de Legislação da Aeronáutica) em formato PDF escaneado, armazenados na pasta `ocr_evaluation/test_data/`. Para cada documento, uma versão de referência (ground truth) foi extraída previamente usando `pdfplumber` a partir da camada de texto digital dos PDFs — esse texto é considerado como a referência "correta" para comparação.

Cada modelo de OCR processa os PDFs de forma independente, convertendo as páginas em imagens e extraindo o texto via seu pipeline próprio. Os textos extraídos são normalizados (remoção de espaços excessivos, normalização de quebras de linha) antes da comparação com o ground truth.

### 5.2 Métricas Utilizadas

- **CER (Character Error Rate):** Taxa de erro por caractere, calculada via biblioteca `jiwer`. Mede a distância de edição normalizada entre o texto de referência e o texto extraído. Valores menores indicam melhor fidelidade — 0.0 é perfeito, 1.0 significa 100% de erro.

- **WER (Word Error Rate):** Taxa de erro por palavra, aplicada após normalização (lowercase, remoção de pontuação). Mede quantas palavras precisam ser inseridas, deletadas ou substituídas para transformar o texto extraído no texto de referência.

- **Domain Precision/Recall/F1:** Métrica específica do domínio que verifica a presença de **90+ termos técnicos da aviação** (siglas ICAO, órgãos reguladores, parâmetros meteorológicos, termos em português aeronáutico) no texto extraído. O recall mede quantos dos termos esperados o OCR conseguiu preservar — extremamente relevante para o RAG, onde esses termos são os tokens mais buscados nas queries dos usuários.

### 5.3 Modelos Testados e Configurações

| Modelo | DPI | Idiomas | Preprocessing | Observações |
|---|---|---|---|---|
| Tesseract | 300 | por+eng | Grayscale + Otsu + Denoising + Deskew | OEM 3, PSM 6 |
| EasyOCR | 250 | pt, en | Nenhum | CPU, paragraph=True |
| docTR | Padrão interno | — | Nenhum (DBNet + CRNN) | PyTorch backend |
| Unstructured | — | por | Nenhum (Tesseract interno) | strategy=ocr_only |
| Chandra OCR 2 | — | — | Nenhum (VLM) | HuggingFace local |
| PaddleOCR | 300 | pt | Nenhum (DB++ + SVTR) | API v3.x, predict() |

### 5.4 Ambiente de Execução

Os testes foram executados em CPU (sem GPU dedicada para OCR). O Chandra OCR 2 não foi avaliado com métricas neste ambiente por requerer GPU. O PaddleOCR apresentou erros na execução inicial, necessitando de workaround no flag PIR.

---

## 6. Resultados

### 6.1 Resumo Agregado

O benchmark processou **4 documentos SISLAER** com timestamp `20260423T094004`. A tabela abaixo apresenta os resultados médios por modelo, ordenados por CER crescente (menor = melhor):

| Ranking | Modelo | CER | WER | Domain F1 | Tempo Total (s) | PDFs Processados |
|---|---|---|---|---|---|---|
| 1 | Unstructured | 0.5298 | 3.1433 | 0.0264 | 356,6 | 4 |
| 2 | docTR | 0.5397 | 2.6867 | 0.0240 | 347,2 | 4 |
| 3 | EasyOCR | 0.5465 | 2.9447 | 0.0288 | 1490,4 | 4 |
| 4 | PaddleOCR | 0.5757 | 2.5858 | 0.0264 | 3008,6 | 4 |
| 5 | Tesseract | 0.7990 | 5.1397 | 0.0168 | 395,6 | 4 |
| — | Chandra OCR 2 | — | — | — | SKIPPED (sem GPU) | — |

**Observações:** Quatro modelos completaram o benchmark com CER entre 0.53 e 0.58, enquanto o Tesseract ficou significativamente abaixo com CER 0.80. O PaddleOCR obteve o menor WER (2.59) mas ao custo de um tempo de processamento ~8,7x maior que o docTR. O Chandra OCR 2 não pôde ser avaliado por requerer GPU, que não estava disponível no ambiente de testes.

### 6.2 Resultados Detalhados por Modelo

| PDF | Modelo | Status | CER | WER | Domain F1 | Tempo (s) | Páginas | Chars Extraídos |
|---|---|---|---|---|---|---|---|---|
| sislaer_47365.pdf | Tesseract | OK | 0.7573 | 2.3686 | 0.0096 | 49.80 | 12 | 12228 |
| sislaer_47365.pdf | EasyOCR | OK | 0.1218 | 0.4516 | 0.0192 | 190.10 | 12 | 11259 |
| sislaer_47365.pdf | docTR | OK | 0.1261 | 0.4199 | 0.0192 | 44.58 | 12 | 11269 |
| sislaer_47365.pdf | Unstructured | OK | 0.1104 | 0.4906 | 0.0192 | 37.88 | 1 | 11468 |
| sislaer_47365.pdf | Chandra OCR 2 | SKIPPED | — | — | — | 59.70 | 0 | 0 |
| sislaer_47365.pdf | PaddleOCR | OK | 0.1713 | 0.3670 | 0.0192 | 394.82 | 12 | 10441 |
| sislaer_47462.pdf | Tesseract | OK | 0.5051 | 0.9733 | 0.0096 | 139.12 | 31 | 44141 |
| sislaer_47462.pdf | EasyOCR | OK | 0.1390 | 0.2736 | 0.0288 | 527.73 | 31 | 49176 |
| sislaer_47462.pdf | docTR | OK | 0.1100 | 0.2019 | 0.0192 | 122.49 | 31 | 47418 |
| sislaer_47462.pdf | Unstructured | OK | 0.0845 | 0.2379 | 0.0288 | 135.24 | 1 | 47571 |
| sislaer_47462.pdf | Chandra OCR 2 | SKIPPED | — | — | — | 0.63 | 0 | 0 |
| sislaer_47462.pdf | PaddleOCR | OK | 0.2064 | 0.1702 | 0.0288 | 1042.55 | 31 | 46011 |
| sislaer_47534.pdf | Tesseract | OK | 0.9672 | 7.2961 | 0.0192 | 123.60 | 30 | 43258 |
| sislaer_47534.pdf | EasyOCR | OK | 0.9660 | 5.7796 | 0.0288 | 502.72 | 30 | 43929 |
| sislaer_47534.pdf | docTR | OK | 0.9653 | 5.2427 | 0.0192 | 118.93 | 30 | 43494 |
| sislaer_47534.pdf | Unstructured | OK | 0.9657 | 6.3981 | 0.0192 | 123.39 | 1 | 45600 |
| sislaer_47534.pdf | Chandra OCR 2 | SKIPPED | — | — | — | 1.22 | 0 | 0 |
| sislaer_47534.pdf | PaddleOCR | OK | 0.9662 | 5.0835 | 0.0192 | 1018.37 | 30 | 43081 |
| sislaer_47605.pdf | Tesseract | OK | 0.9664 | 9.9210 | 0.0288 | 83.05 | 17 | 19672 |
| sislaer_47605.pdf | EasyOCR | OK | 0.9592 | 5.2739 | 0.0385 | 269.85 | 17 | 20377 |
| sislaer_47605.pdf | docTR | OK | 0.9574 | 4.8824 | 0.0385 | 61.23 | 17 | 20426 |
| sislaer_47605.pdf | Unstructured | OK | 0.9585 | 5.4467 | 0.0385 | 60.07 | 1 | 20760 |
| sislaer_47605.pdf | Chandra OCR 2 | SKIPPED | — | — | — | 0.47 | 0 | 0 |
| sislaer_47605.pdf | PaddleOCR | OK | 0.9588 | 4.7224 | 0.0385 | 552.87 | 17 | 19921 |

**Observações:**

Os PDFs sislaer_47534.pdf e sislaer_47605.pdf apresentaram CER próximo a 0.96–0.97 em TODOS os modelos, indicando que esses documentos são de qualidade de imagem muito baixa e resistem a qualquer motor de OCR sem GPU.

No PDF sislaer_47365.pdf e sislaer_47462.pdf, os modelos Unstructured e docTR obtiveram resultados substancialmente melhores que o Tesseract.

### 6.3 Análise dos Termos de Domínio

O Domain F1 extremamente baixo em todos os modelos (0.017–0.029) reflete a dificuldade de extrair corretamente o vocabulário técnico aeronáutico dos documentos SISLAER. A lista de termos avaliados contém mais de 90 siglas e termos ICAO. Nos 4 PDFs testados:

- Praticamente nenhum modelo conseguiu encontrar mais de 3% dos termos esperados
- O único termo consistentemente extraído por todos os modelos foi **"DA"** — uma sigla de 2 letras que pode aparecer como artigo em português, o que provavelmente gera falsos positivos
- O Unstructured e o EasyOCR conseguiram extrair adicionalmente o termo **"aproximação"** (em português, e não a sigla ICAO correspondente) no PDF sislaer_47462.pdf
- O EasyOCR e PaddleOCR encontraram o termo **"FL"** (Flight Level) no sislaer_47534.pdf
- O Tesseract obteve Domain F1 = 0.0096 — o mais baixo — extraindo apenas "DA" em todos os documentos
- O Chandra OCR 2 obteve Domain F1 = 0.0 em todos os PDFs (nenhum caractere extraído)


### 6.4 Análise Qualitativa Manual dos Outputs

Paralelamente ao benchmark quantitativo, foi realizado um conferimento manual dos outputs de cada modelo em três documentos normativos da aeronáutica: **ICA 11-25 (2016)**, **ICA 205-47** e **ICA 11-138 (2016)**. Para cada documento, trechos representativos foram selecionados e os outputs de cada modelo foram analisados lado a lado com o documento original.

#### 6.4.1 Comportamento em Tabelas

As tabelas são um dos formatos mais desafiadores para OCR, pois exigem que o modelo compreenda não apenas os caracteres, mas a estrutura bidimensional do conteúdo.

**docTR** apresentou o melhor desempenho em tabelas simples: cada célula é colocada em uma linha própria, preservando a hierarquia linha a linha. Em tabelas com 3 colunas (como a tabela de setores/siglas/códigos da ICA 11-25), o resultado ficou muito próximo do original, com cada setor, sigla e código de tarefa claramente separados.

**PaddleOCR** também preservou bem a estrutura tabular, mas apresentou algumas inversões de ordem de colunas — em certos momentos, o código de tarefa apareceu antes da sigla na mesma linha, ao invés de seguir a ordem esquerda-direita do original.

**EasyOCR** apresentou o pior desempenho em tabelas: ao invés de processar célula por célula, o modelo colapsou colunas inteiras em uma única linha concatenada. Na tabela de setores da ICA 11-25 (pág. 37), por exemplo, todos os nomes de setores foram agrupados em uma linha, todos os códigos foram agrupados em outra linha separada, e todas as siglas em uma terceira — perdendo completamente a correspondência entre as linhas da tabela original.

**Unstructured** tentou representar a estrutura tabular usando o caractere pipe (`|`) como separador, mas o resultado foi inconsistente: algumas células ficaram corretas, outras foram truncadas ou misturadas. Em tabelas maiores, o Unstructured omitiu entradas inteiras.

**Tesseract** foi completamente ilegível em todos os trechos com tabelas testados, produzindo saídas com sequências de símbolos, letras soltas e caracteres sem sentido. Vale ressaltar, porém, que apesar do CER altíssimo nos documentos SISLAER do benchmark (0.799 em média), o Tesseract **não retorna 100% do documento ilegível**: em páginas com tipografia simples, texto corrido e boa qualidade de digitalização, o modelo consegue extrair partes do texto corretamente. O problema é que essas situações favoráveis não representam a maioria dos documentos testados.

#### 6.4.2 Comportamento com Textos em Níveis e Hierarquias

O teste com a página 28 da ICA 11-25, que apresenta palavras dispostas em múltiplos níveis de hierarquia visual (similar a um organograma textual), revelou diferenças importantes:

**docTR** identificou corretamente cada elemento como um bloco independente, colocando-os em linhas próprias e preservando a hierarquia. O resultado foi muito próximo do original.

**EasyOCR** mesclou elementos próximos em blocos agrupados, perdendo parte da separação hierárquica, embora o texto em si estivesse legível.

**PaddleOCR** produziu um resultado próximo ao original, mas com alguns itens fora de ordem em relação à sequência de leitura esperada.

**Unstructured** lidou razoavelmente bem com o texto principal, mas representou a tabela adjacente de forma parcialmente incorreta.

#### 6.4.3 Espaçamento entre Palavras

O espaçamento entre palavras é um indicativo de como o modelo interpreta espaços visuais no documento:

**EasyOCR** foi o modelo com maior incidência de espaçamento incorreto: em várias ocorrências, espaços entre palavras foram substituídos por underscores (ex: `COMANDO-GERAL DO_PESSOAL`, `LOCOMOÇÃO_`), e separadores de itens de sumário (como os pontilhados `...`) foram substituídos por espaços ou símbolos inconsistentes.

**PaddleOCR** utilizou múltiplos espaços consecutivos para representar espaçamentos maiores do documento original, o que é visualmente mais fiel mas pode interferir no pipeline de normalização de texto do RAG.

**docTR** produziu a saída mais limpa em termos de espaçamento, sem artefatos de underscores ou espaços duplos excessivos.

**Unstructured** utilizou em-dashes (—) como separadores em itens de sumário, o que é semanticamente adequado mas difere da representação original com pontilhados.

#### 6.4.4 Tratamento de Acentuação e Caracteres Especiais

**docTR** teve problemas pontuais com acentuação: escreveu "Açâo" ao invés de "Ação", "Gestâo" ao invés de "Gestão" — indicando que o modelo não foi treinado com dados suficientes em português para distinguir `â` de `ã`.

**EasyOCR** confundiu `º` (ordinal masculino) com `ª` (ordinal feminino) em vários lugares, e `O` (letra) com `0` (zero) em alguns códigos alfanuméricos.

**PaddleOCR** apresentou o padrão "ã0" ao invés de "ão" em algumas palavras (ex: "Açã0" ao invés de "Ação"), sugerindo confusão entre o caractere `o` e o dígito `0` após vogal nasal.

**Unstructured** apresentou a melhor fidelidade de acentuação dentre os modelos testados manualmente.

---

## 7. Conclusões

### 7.1 Síntese dos Resultados

O benchmark realizado sobre 4 documentos do tipo SISLAER, combinado com a análise qualitativa manual de outputs em 3 documentos normativos da aeronáutica (ICA 11-25, ICA 205-47 e ICA 11-138), permite uma avaliação abrangente dos 6 modelos de OCR testados.

**Tesseract** obteve o pior desempenho no benchmark quantitativo (CER 0.799, WER 5.14) e foi consistentemente ilegível nos documentos testados manualmente — tabelas, hierarquias de texto e sumários densos resultaram em saídas com caracteres sem sentido. Apesar disso, o Tesseract não retorna 100% de conteúdo ilegível: em páginas com tipografia simples e boa qualidade de digitalização, partes do texto são extraídas corretamente. Contudo, essa situação favorável não é a norma nos documentos normativos da aviação civil, que frequentemente utilizam layouts complexos com múltiplas colunas e tabelas.

**Chandra OCR 2** não pôde ser avaliado por requerer GPU, que não estava disponível no ambiente de testes. Pela natureza da abordagem (Vision-Language Model com saída em Markdown), é potencialmente o modelo mais preciso para documentos complexos, mas o risco de alucinações em dados regulatórios e a exigência de hardware especializado tornam sua adoção em produção condicionada a uma avaliação dedicada com GPU.

**EasyOCR** obteve CER intermediário (0.546) mas ao custo de um tempo de processamento alto (1490s para 4 PDFs). A análise qualitativa revelou problemas sérios de estruturação: colapso de colunas em tabelas, uso de underscores como separadores, e confusões entre caracteres similares (º/ª, O/0). Para o pipeline de RAG, onde a estrutura do texto impacta diretamente a qualidade dos chunks, esses artefatos são problemáticos.

**PaddleOCR** obteve o melhor WER (2.59) e boa qualidade visual nas análises manuais, preservando estrutura de tabelas razoavelmente bem. No entanto, o tempo de processamento de 3008s para apenas 4 PDFs (~12 minutos por PDF em CPU) é proibitivo para um sistema RAG que precisa processar documentos continuamente. Em ambiente com GPU disponível, o PaddleOCR se tornaria competitivo.

### 7.2 Modelos Recomendados

Com base no conjunto de evidências — benchmark quantitativo, análise qualitativa manual e viabilidade operacional — dois modelos se destacam como candidatos para uso no pipeline RAG:

**docTR (Mindee)** e **Unstructured** apresentaram os resultados mais equilibrados:

| Critério | docTR | Unstructured |
|---|---|---|
| CER médio | 0.5397 (2º lugar) | **0.5298 (1º lugar)** |
| WER médio | **2.6867 (melhor entre os 2)** | 3.1433 |
| Tempo total (4 PDFs) | **347,2s** | 356,6s |
| Preservação de tabelas | Boa (linha a linha) | Razoável (usa pipes, com erros) |
| Espaçamento | Limpo | Bom |
| Estrutura semântica | Hierarquia bloco/linha/palavra | Tipagem semântica (Título, Tabela, etc.) |
| Integração com RAG | Estrutura em blocos para chunking | Nativa com LangChain/LlamaIndex |
| Suporte ao português | Modelos focados em EN/FR | Usa Tesseract com pacote `por` |

**Recomendação para o pipeline RAG do projeto:**

**Se o objetivo prioritário é a integração com o pipeline de RAG** (chunking semântico, compatibilidade com LangChain/LlamaIndex, separação de tipos de conteúdo), o **Unstructured** é a escolha mais natural: além do melhor CER, sua tipagem semântica (Título, Parágrafo, Tabela) permite estratégias de chunking mais inteligentes que os demais modelos não oferecem nativamente.

**Se o objetivo prioritário é a fidelidade de extração do texto** (menor WER, melhor preservação de tabelas e estrutura de documentos), o **docTR** é o candidato preferencial: segundo melhor CER, menor WER, melhor tempo, e saída mais limpa nos testes manuais — especialmente em tabelas, onde superou todos os outros modelos.

Ambos os modelos rodam em CPU sem necessidade de GPU e têm tempos de processamento comparáveis (~87–89s por PDF em média). A escolha final depende de qual aspecto é mais crítico para o caso de uso: integração com LangChain ou fidelidade de extração.

#### TODO REMOVER O 7.3 DEPOIS
#### TODO Explicar que o Chandra OCR 2 é um caso de canhão para matar uma formiga, mas ainda vou precisar brincar um pouco com esse modelo.
### 7.3 Trabalhos Futuros

- Avaliar o **Chandra OCR 2** em ambiente com GPU dedicada, dado seu potencial para documentos com layouts complexos
- Investigar o **fine-tuning do docTR** com dados em português para corrigir erros de acentuação (â/ã, confusão em palavras como "Ação")
- Considerar uma **abordagem híbrida**: Unstructured para identificação semântica de elementos + docTR para reconhecimento de caracteres em tabelas
- Expandir o benchmark para incluir documentos das outras fontes do sistema (AIP, NOTAM, DECEA) que possuem características diferentes dos SISLAER

---

## Referências

- **Tesseract OCR:** https://github.com/tesseract-ocr/tesseract — Smith, R. (2007). "An Overview of the Tesseract OCR Engine." ICDAR.
- **EasyOCR:** https://github.com/JaidedAI/EasyOCR
- **CRAFT (detector EasyOCR):** Baek, Y. et al. (2019). "Character Region Awareness for Text Detection." CVPR.
- **docTR:** https://github.com/mindee/doctr
- **DBNet (detector docTR):** Liao, M. et al. (2020). "Real-time Scene Text Detection with Differentiable Binarization." AAAI.
- **Unstructured:** https://github.com/Unstructured-IO/unstructured
- **Chandra OCR 2:** https://huggingface.co/datalab-to/chandra-ocr-2
- **PaddleOCR:** https://github.com/PaddlePaddle/PaddleOCR — Du, Y. et al. (2020). "PP-OCR: A Practical Ultra Lightweight OCR System." arXiv:2009.09941.
- **PP-OCRv4:** Du, Y. et al. (2023). "PP-OCRv4: Universal Scene Text Recognition." arXiv.
- **jiwer (métricas):** https://github.com/jitsi/jiwer
