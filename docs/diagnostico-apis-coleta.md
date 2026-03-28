# Diagnóstico de Fontes de Coleta de Documentos

**Projeto:** ITA AirData RAG System  
**Data:** 28/03/2026  
**Escopo:** Avaliação de novas fontes de coleta (SISLAER, MCP APIBrasil) e comparação com as fontes atuais (DECEA Portal, LexML Brasil).

---

## 1. Fontes Atuais

### 1.1 DECEA Portal (`publicacoes.decea.mil.br`)

Portal de publicações do Departamento de Controle do Espaço Aéreo. Disponibiliza publicações oficiais do DECEA em formato PDF.

| Aspecto | Detalhe |
|---------|---------|
| **Tipo de acesso** | Web scraping (HTML + download de PDF) |
| **Tipos de documentos** | ICA, MCA, PCA, DCA, TCA, CIRCEA, NSCA, FCA |
| **Formato do conteúdo** | PDF (requer extração de texto, às vezes OCR) |
| **Metadados disponíveis** | Número, título, data de publicação, órgão de origem |
| **Limitações conhecidas** | Sem status de vigência, sem portaria de aprovação, sem relações entre documentos, conteúdo depende da qualidade do PDF |

### 1.2 LexML Brasil (`lexml.gov.br`)

Portal de legislação brasileira mantido pelo Senado Federal. Agrega normas de diversas esferas governamentais com links para textos completos no Senado e Planalto.

| Aspecto | Detalhe |
|---------|---------|
| **Tipo de acesso** | Web scraping (HTML assíncrono com rate limiting) |
| **Tipos de documentos** | Leis, Decretos, Portarias, Resoluções, RBAC |
| **Formato do conteúdo** | HTML (via Senado/Planalto) |
| **Metadados disponíveis** | Título, URN, autoridade, data, ementa, assuntos, localidade |
| **Limitações conhecidas** | Raramente contém publicações do COMAER (ICA, MCA, etc.), depende de sites de terceiros para conteúdo, sem status de vigência, sem relações entre documentos |

---

## 2. Fontes Avaliadas

### 2.1 SISLAER (`sislaer.fab.mil.br`)

#### O que é

O **SISLAER** (Sistema de Legislação Aeronáutica) é o portal oficial do **CENDOC** (Centro de Documentação do Comando da Aeronáutica) da FAB. É a **fonte autoritativa** para toda a legislação e normativa aeronáutica brasileira. Utiliza o sistema Sophia Biblioteca Web para gestão e consulta do acervo.

#### Tipo de acesso

**Web scraping via HTTP.** Não possui API REST documentada. O portal utiliza uma aplicação web com URLs previsíveis e estáveis, cujas páginas de detalhe já entregam o conteúdo dos documentos no HTML da resposta. A coleta é tecnicamente viável com controles adequados de taxa de requisição.

#### Cobertura de documentos

O SISLAER cobre **todos os tipos de publicações do Comando da Aeronáutica** e legislação correlata:

| Categoria | Tipos | Coberto hoje? |
|-----------|-------|:-------------:|
| **Publicações Convencionais** | DCA, FCA, ICA, MCA, NSCA, OCA, PCA, TCA | Sim (via DECEA) |
| **Publicações Não-Convencionais** | ACA, AVCA, OTCA, AIC, NOTAM, AIP-BRASIL, BCA | **Não** |
| **Publicações Regulamentares** | RCA, ROCA (Regulamentos), RICA (Regimentos Internos) | **Não** |
| **Legislação Aeronáutica** | Leis, Decretos, Portarias, Resoluções, Instruções Normativas, Medidas Provisórias | Parcial (via LexML) |
| **Tratados Internacionais** | Convenção de Chicago, acordos bilaterais, etc. | **Não** |

#### Metadados disponíveis por documento

| Metadado | Disponível no DECEA? | Disponível no LexML? | Disponível no SISLAER? |
|----------|:-------------------:|:-------------------:|:---------------------:|
| Título | Sim | Sim | Sim |
| Número | Sim | Sim | Sim |
| Tipo/Espécie | Sim | Sim (via URN) | Sim |
| Data de publicação | Sim | Sim | Sim |
| Autor/Autoridade (OM) | Parcial | Sim | Sim (500+ OMs catalogadas) |
| **Situação (vigente/revogado)** | Não | Não | **Sim** |
| **Portaria de aprovação** | Não | Não | **Sim** (data + texto da ementa) |
| **Ementa completa** | Não | Parcial | **Sim** |
| **Alterações (docs vinculados)** | Não | Não | **Sim** (links para documentos que alteram/são alterados) |
| **Correlações (docs relacionados)** | Não | Não | **Sim** (links para documentos correlatos) |
| **Natureza/Esfera** | Não | Não | **Sim** (Ostensiva, Reservada, etc.) |
| **BCA de referência** | Não | Não | **Sim** |
| URN padronizado | Não | Sim | Não |
| Assuntos/Subjects | Não | Sim | Não |

#### Formato do conteúdo

O conteúdo dos documentos está disponível frequentemente em **HTML/XHTML** diretamente no portal, o que produz texto de qualidade muito superior à extração de PDF. Quando disponível apenas em PDF, o download é possível para extração posterior.

#### Restrições de acesso

- Documentos com natureza **ostensiva** (maioria) são públicos e acessíveis
- Documentos classificados como sigilosos não disponibilizam o conteúdo integral (apenas título e metadados)

#### Avaliação geral

| Critério | Nota |
|----------|:----:|
| Cobertura de tipos de documentos | Excelente |
| Riqueza de metadados | Excelente |
| Qualidade do conteúdo textual | Excelente |
| Facilidade de coleta | Boa |
| Estabilidade/Disponibilidade | Boa |
| Existência de API estruturada | Não disponível |

---

### 2.2 MCP APIBrasil (`mcp.apibrasil.io`)

#### O que é

Plataforma de agregação de 120+ APIs brasileiras focada em **serviços comerciais e utilitários**. Oferece consultas a dados cadastrais, endereços, veículos, comunicação e serviços públicos.

#### APIs disponíveis

- CPF/CNPJ (Receita Federal, Serasa)
- CEP/Geolocalização (IBGE)
- Veículos (Placa, FIPE)
- Comunicação (WhatsApp, SMS)
- Correios (Rastreamento)
- Clima, Bancos, Feriados

#### Tipo de acesso

REST + GraphQL com autenticação por token. SDKs disponíveis para Python, Node.js, PHP, C#. Possui protocolo MCP nativo para integração com assistentes de IA.

#### Relevância para o projeto

**Nenhuma.** A plataforma não possui:
- APIs de legislação (aeronáutica ou geral)
- APIs de órgãos aeronáuticos (ANAC, DECEA, FAB)
- APIs de busca de documentos normativos ou regulatórios
- Qualquer funcionalidade relacionada a normativas, regulamentos ou publicações oficiais

#### Avaliação geral

| Critério | Nota |
|----------|:----:|
| Relevância para legislação aeronáutica | **Nenhuma** |

---

## 3. Análise Comparativa de Cobertura

### 3.1 Tipos de documentos por fonte

| Tipo | DECEA | LexML | SISLAER |
|------|:-----:|:-----:|:-------:|
| ICA | ✅ | ⚠️ raro | ✅ |
| MCA, PCA, DCA, TCA, CIRCEA, NSCA, FCA | ✅ | ❌ | ✅ |
| OCA | ❌ | ❌ | ✅ |
| RCA, ROCA (Regulamentos) | ❌ | ❌ | ✅ |
| RICA (Regimentos Internos) | ❌ | ❌ | ✅ |
| BCA (Boletins) | ❌ | ❌ | ✅ |
| ACA, AVCA (Avisos) | ❌ | ❌ | ✅ |
| OTCA (Ordens Técnicas) | ❌ | ❌ | ✅ |
| AIC, NOTAM, AIP-BRASIL | ❌ | ❌ | ✅ |
| Leis, Decretos | ❌ | ✅ | ✅ |
| Portarias | ❌ | ✅ | ✅ |
| Resoluções | ❌ | ✅ | ✅ |
| RBAC (ANAC) | ❌ | ⚠️ | ❌ |
| Tratados Internacionais | ❌ | ⚠️ | ✅ |

### 3.2 Qualidade de conteúdo

| Aspecto | DECEA | LexML | SISLAER |
|---------|:-----:|:-----:|:-------:|
| Formato primário | PDF | HTML | HTML / XHTML |
| Necessita OCR | Às vezes | Nunca | Raramente |
| Qualidade do texto extraído | Variável | Boa | Excelente |
| Auto-contido (sem dependência de terceiros) | Não (S3 AWS) | Não (Senado/Planalto) | Sim |

---

## 4. Ganhos e Perdas com SISLAER

### 4.1 Ganhos

| Ganho | Descrição |
|-------|-----------|
| **+10 tipos de documentos** | OCA, RCA, ROCA, RICA, BCA, ACA, AVCA, OTCA, AIC, NOTAM, AIP-BRASIL, Tratados Internacionais |
| **Status de vigência** | Saber se um documento está "Em vigor" ou "Revogado" — informação hoje inexistente |
| **Portaria de aprovação** | Rastreabilidade completa de quem aprovou cada publicação |
| **Grafo de relacionamentos** | Alterações e correlações entre documentos, permitindo navegação normativa |
| **Qualidade de texto superior** | Conteúdo em HTML elimina problemas de extração de PDF e OCR |
| **Fonte autoritativa única** | O SISLAER é O repositório oficial da FAB para legislação aeronáutica |

### 4.2 Perdas potenciais

| Perda | Impacto | Mitigação |
|-------|---------|-----------|
| URN padronizado (presente no LexML) | Baixo | Manter LexML como fonte complementar |
| RBAC/ANAC (se ausente no SISLAER) | Médio | LexML continua cobrindo esta lacuna |
| Documentos classificados | Baixo | Coletar apenas documentos ostensivos (pública maioria) |
| Legislação federal genérica | Baixo | LexML continua cobrindo legislação não-aeronáutica |

---

## 5. Estratégia Recomendada

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  FONTE PRIMÁRIA: SISLAER                                     │
│  ├─ Todas as publicações do COMAER (ICA, MCA, RCA, etc.)   │
│  ├─ Legislação aeronáutica (Leis, Decretos, Portarias)      │
│  ├─ Publicações especiais (BCA, AIC, NOTAM, etc.)           │
│  └─ Tratados internacionais de aviação civil                 │
│                                                              │
│  FONTE COMPLEMENTAR: LexML                                   │
│  ├─ Legislação federal geral não-aeronáutica                │
│  ├─ RBAC/ANAC quando não disponível no SISLAER              │
│  └─ Deduplicação automática evita sobreposição              │
│                                                              │
│  FALLBACK: DECEA Portal                                      │
│  └─ Mantido operacional para uso em caso de indisponibilidade│
│                                                              │
│  AUXILIAR: PDFs Locais                                       │
│  └─ Ingestão manual de documentos avulsos                    │
│                                                              │
│  DESCARTADO: MCP APIBrasil                                   │
│  └─ Sem relevância (nenhuma API de legislação)               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 5.1 Sobre substituição do DECEA Portal

O SISLAER é um **superconjunto** do DECEA Portal: contém todos os mesmos tipos de documentos com metadados significativamente mais ricos. O DECEA Portal pode ser mantido como fallback, mas deixa de ser fonte primária.

### 5.2 Sobre convivência com LexML

O LexML cobre legislação federal ampla que pode não estar integralmente no SISLAER. A recomendação é mantê-lo como **complementar**, com deduplicação automática para evitar documentos duplicados entre as fontes.

---

## 6. Riscos

| Risco | Probabilidade | Impacto | Mitigação |
|-------|:------------:|:-------:|-----------|
| Bloqueio por excesso de requisições | Média | Alto | Coleta com taxa conservadora e intervalos controlados |
| Mudança na estrutura do portal | Baixa | Alto | O SISLAER usa um produto comercial estável (Sophia) |
| Indisponibilidade temporária | Baixa | Médio | DECEA mantido como fallback; dados já coletados preservados |
| Documentos sigilosos inacessíveis | Certa | Baixo | Coletar apenas documentos ostensivos (maioria do acervo) |
| Documentos duplicados entre fontes | Certa | Médio | Requer tratamento de deduplicação cross-source |

---

## 7. Pontos de Atenção para Implementação

### 7.1 Deduplicação entre fontes

O sistema atual identifica documentos por IDs específicos de cada fonte (ex: `decea_ICA-100-12` vs `lexml_urn_lex_br_...`). Isso significa que **o mesmo documento coletado de fontes diferentes é armazenado como dois registros distintos**. O hash de conteúdo (`content_hash`) existe mas serve apenas para detectar atualizações do mesmo documento, não para deduplicar entre fontes.

Para a estratégia SISLAER + LexML funcionar sem duplicações, será necessário um mecanismo de deduplicação cross-source (por exemplo, normalização de identidade por número e tipo do documento, ou verificação de hash de conteúdo na inserção).

### 7.2 Buscabilidade dos novos metadados

Os novos campos do SISLAER (autoridade/OM, número do documento, portaria de aprovação, alterações, correlações) trazem valor operacional significativo. O armazenamento atual suporta campos genéricos em formato JSON, mas campos frequentemente consultados (como `authority` e `number`) teriam melhor performance e usabilidade como **colunas dedicadas e indexadas** no banco de dados. A estrutura de armazenamento pode precisar de ajuste para acomodar isso adequadamente.

---

## 8. Conclusão

O **SISLAER** representa a adição de maior valor para o sistema de coleta: é a fonte autoritativa oficial com cobertura 2.5x maior em tipos de documentos, metadados significativamente mais ricos (incluindo status de vigência e grafo de relacionamentos entre normas), e conteúdo textual de qualidade superior.

O **MCP APIBrasil** não possui relevância para o projeto e é descartado.

A estratégia recomendada — **SISLAER primário, LexML complementar, DECEA fallback** — maximiza cobertura sem perda de dados existentes.
