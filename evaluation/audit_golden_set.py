"""Audit the golden set to validate ground-truth quality.

For each non-coverage entry in ``evaluation/golden_set.csv``:
  * normalises the expected_doc_id using the same logic as
    ``evaluate_retrieval.py`` (version-agnostic, doc-vs-chunk).
  * checks whether the canonical document exists in ``data/store.db``
    (active + is_latest).
  * for article-level ids, queries Qdrant to confirm at least one
    chunk with the matching regulation_id is indexed.

Also produces overall distribution stats (category, doc_type, authority)
and flags queries whose phrasing implies a filter or sort intent so we
can later validate the rewriter on those.

Outputs a Markdown report at ``evaluation/results/golden_audit.md``.
"""

from __future__ import annotations

import csv
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import config  # noqa: E402

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue
except ImportError:  # pragma: no cover - hard-fail with helpful msg
    raise SystemExit(
        "qdrant-client is required. Install: pip install qdrant-client"
    )


GOLDEN_PATH = ROOT / "evaluation" / "golden_set.csv"
REPORT_PATH = ROOT / "evaluation" / "results" / "golden_audit.md"
STORE_DB = ROOT / "data" / "store.db"


# ----- ID normalization (mirrors evaluate_retrieval.py) ----------------

_GOLDEN_ID_RE = re.compile(
    r'^(?P<type>[A-Za-z][A-Za-z0-9.]*)[-_](?P<rest>.+)$'
)
_VERSION_YEAR_RE = re.compile(r'/\d{4}')


def normalize_id(raw_id: str) -> str:
    m = _GOLDEN_ID_RE.match(raw_id)
    if m:
        dtype = re.sub(r'[^a-z0-9]', '', m.group('type').lower())
        rest = m.group('rest')
        return f"{dtype}_{rest}"
    return raw_id


def strip_version(norm_id: str) -> str:
    return _VERSION_YEAR_RE.sub('', norm_id)


def is_doc_level(norm_id: str) -> bool:
    return "-art" not in norm_id


def split_article(norm_id: str) -> Tuple[str, Optional[str]]:
    """Split into (canonical_no_version, article_number_or_None)."""
    base = strip_version(norm_id)
    if "-art" in base:
        canonical, art = base.split("-art", 1)
        return canonical, art
    return base, None


# ----- Filter/sort intent heuristics -----------------------------------

_FILTER_KEYWORDS = {
    # type acronyms (the rewriter exposes these via FilterRegistry)
    "ICA": "type=ICA",
    "MCA": "type=MCA",
    "DCA": "type=DCA",
    "RCA": "type=RCA",
    "PCA": "type=PCA",
    "NSCA": "type=NSCA",
    "RICA": "type=RICA",
    "FCA": "type=FCA",
    # authorities
    "DECEA": "authority=DECEA",
    "ANAC": "authority=ANAC",
    "COMAER": "authority=COMAER",
    "FAB": "authority=FAB",
    "DCTA": "authority=DCTA",
    "DIRSA": "authority=DIRSA",
}

_SORT_PATTERNS = [
    (re.compile(r"\bmais\s+recente", re.IGNORECASE), "sort_by_date_desc"),
    (re.compile(r"\b(?:nov[ao]s?|atual|atualizad[ao]s?)\b", re.IGNORECASE), "sort_by_date_desc"),
    (re.compile(r"\b(?:antig[ao]s?|primeir[ao]s?)\b", re.IGNORECASE), "sort_by_date_asc"),
    (re.compile(r"\b(?:em\s+vigor|vigentes?)\b", re.IGNORECASE), "filter_active"),
]


def detect_filter_sort_intent(query: str) -> Dict[str, List[str]]:
    """Heuristic detection of filter/sort intents in query."""
    intents: Dict[str, List[str]] = {"filters": [], "sorts": []}

    upper_words = re.findall(r"\b[A-Z]{3,}\b", query)
    for word in upper_words:
        if word in _FILTER_KEYWORDS:
            intents["filters"].append(_FILTER_KEYWORDS[word])

    for pattern, label in _SORT_PATTERNS:
        if pattern.search(query):
            intents["sorts"].append(label)

    return intents


# ----- Audit core ------------------------------------------------------


@dataclass
class GoldenRow:
    query_id: str
    query: str
    expected_doc_id: str
    relevance: str
    category: str
    notes: str


@dataclass
class RowAudit:
    row: GoldenRow
    norm_id: str
    canonical_no_version: str
    article_number: Optional[str]
    canonical_in_store: bool = False
    canonical_doc_type: Optional[str] = None
    canonical_authority: Optional[str] = None
    canonical_is_latest: Optional[bool] = None
    article_in_qdrant: Optional[bool] = None
    qdrant_match_count: int = 0
    canonical_latest_chunks_in_qdrant: int = 0
    issue_flags: List[str] = field(default_factory=list)


def load_golden_set(path: Path) -> List[GoldenRow]:
    rows: List[GoldenRow] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(GoldenRow(
                query_id=r["query_id"],
                query=r["query"],
                expected_doc_id=r["expected_doc_id"],
                relevance=r["relevance"],
                category=r["category"],
                notes=r.get("notes", ""),
            ))
    return rows


def lookup_canonical(
    conn: sqlite3.Connection,
    canonical_no_version: str,
) -> Optional[Tuple[str, str, bool]]:
    """Return (doc_type, authority, is_latest) for the canonical id, or None."""
    cur = conn.execute(
        """
        SELECT doc_type, authority, is_latest
        FROM documents
        WHERE canonical_id = ?
        ORDER BY is_latest DESC, version_year DESC
        LIMIT 1
        """,
        (canonical_no_version,),
    )
    row = cur.fetchone()
    if not row:
        return None
    return row[0], row[1], bool(row[2])


def qdrant_chunk_exists(
    client: QdrantClient,
    canonical_no_version: str,
    article_number: str,
) -> int:
    """Count chunks in Qdrant matching canonical+article (version-agnostic, latest only)."""
    flt = Filter(must=[
        FieldCondition(
            key="canonical_id",
            match=MatchValue(value=canonical_no_version),
        ),
        FieldCondition(
            key="article_number",
            match=MatchValue(value=str(article_number)),
        ),
        FieldCondition(
            key="is_latest",
            match=MatchValue(value=True),
        ),
    ])
    result = client.count(
        collection_name=config.QDRANT_COLLECTION_NAME,
        count_filter=flt,
        exact=True,
    )
    return getattr(result, "count", 0)


def qdrant_canonical_latest_count(
    client: QdrantClient,
    canonical_no_version: str,
) -> int:
    """Count chunks in Qdrant for canonical_id with is_latest=true (any article)."""
    flt = Filter(must=[
        FieldCondition(
            key="canonical_id",
            match=MatchValue(value=canonical_no_version),
        ),
        FieldCondition(
            key="is_latest",
            match=MatchValue(value=True),
        ),
    ])
    result = client.count(
        collection_name=config.QDRANT_COLLECTION_NAME,
        count_filter=flt,
        exact=True,
    )
    return getattr(result, "count", 0)


def audit() -> Dict:
    rows = load_golden_set(GOLDEN_PATH)
    print(f"Loaded {len(rows)} golden rows ({len({r.query_id for r in rows})} unique queries)")

    conn = sqlite3.connect(STORE_DB)
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

    audits: List[RowAudit] = []
    for r in rows:
        if r.expected_doc_id == "NOT_IN_DB":
            continue

        norm = normalize_id(r.expected_doc_id)
        canonical_nv, art = split_article(norm)

        a = RowAudit(
            row=r, norm_id=norm,
            canonical_no_version=canonical_nv,
            article_number=art,
        )

        store_hit = lookup_canonical(conn, canonical_nv)
        if store_hit is None:
            a.issue_flags.append("canonical_missing_in_store_db")
        else:
            a.canonical_in_store = True
            a.canonical_doc_type = store_hit[0]
            a.canonical_authority = store_hit[1]
            a.canonical_is_latest = store_hit[2]
            if not store_hit[2]:
                a.issue_flags.append("canonical_not_latest")

        canonical_latest = qdrant_canonical_latest_count(client, canonical_nv)
        a.canonical_latest_chunks_in_qdrant = canonical_latest
        if canonical_latest == 0:
            a.issue_flags.append("canonical_latest_missing_in_qdrant")

        if art:
            cnt = qdrant_chunk_exists(client, canonical_nv, art)
            a.qdrant_match_count = cnt
            a.article_in_qdrant = cnt > 0
            if cnt == 0 and "canonical_latest_missing_in_qdrant" not in a.issue_flags:
                a.issue_flags.append("article_missing_in_qdrant")

        audits.append(a)

    coverage_rows = [r for r in rows if r.expected_doc_id == "NOT_IN_DB"]

    return {
        "rows": rows,
        "audits": audits,
        "coverage_rows": coverage_rows,
    }


# ----- Reporting -------------------------------------------------------


def render_report(result: Dict) -> str:
    rows: List[GoldenRow] = result["rows"]
    audits: List[RowAudit] = result["audits"]
    coverage_rows: List[GoldenRow] = result["coverage_rows"]

    queries: Dict[str, List[GoldenRow]] = defaultdict(list)
    for r in rows:
        queries[r.query_id].append(r)

    cat_counter: Counter = Counter(r.category for r in rows)
    rel_counter: Counter = Counter(r.relevance for r in rows)
    type_counter: Counter = Counter(
        a.canonical_doc_type for a in audits if a.canonical_doc_type
    )
    auth_counter: Counter = Counter(
        a.canonical_authority for a in audits if a.canonical_authority
    )

    issues: Dict[str, List[RowAudit]] = defaultdict(list)
    for a in audits:
        for flag in a.issue_flags:
            issues[flag].append(a)

    intent_hits: List[Tuple[str, str, Dict]] = []
    for qid, items in queries.items():
        intent = detect_filter_sort_intent(items[0].query)
        if intent["filters"] or intent["sorts"]:
            intent_hits.append((qid, items[0].query, intent))

    queries_by_doc: Counter = Counter()
    for r in rows:
        if r.expected_doc_id != "NOT_IN_DB":
            canonical_nv, _ = split_article(normalize_id(r.expected_doc_id))
            queries_by_doc[canonical_nv] += 1

    out: List[str] = []
    out.append("# Golden Set Audit\n")
    out.append(
        "_Auto-generated by `evaluation/audit_golden_set.py`._\n"
    )
    out.append("## 1. Resumo geral\n")
    out.append(f"- Linhas totais no CSV: **{len(rows)}**")
    out.append(f"- Queries únicas: **{len(queries)}**")
    out.append(f"- Linhas de retrieval (com expected_doc_id real): **{len(audits)}**")
    out.append(f"- Linhas de coverage (NOT_IN_DB): **{len(coverage_rows)}**\n")

    out.append("### 1.1 Distribuição por `category`")
    out.append("| category | count |")
    out.append("|---|---|")
    for k, v in cat_counter.most_common():
        out.append(f"| {k} | {v} |")
    out.append("")

    out.append("### 1.2 Distribuição por `relevance`")
    out.append("| relevance | count |")
    out.append("|---|---|")
    for k, v in rel_counter.most_common():
        out.append(f"| {k} | {v} |")
    out.append("")

    out.append("### 1.3 Distribuição por `doc_type` (do canonical resolvido)")
    out.append("| doc_type | count |")
    out.append("|---|---|")
    for k, v in type_counter.most_common():
        out.append(f"| {k} | {v} |")
    out.append("")

    out.append("### 1.4 Distribuição por `authority`")
    out.append("| authority | count |")
    out.append("|---|---|")
    for k, v in auth_counter.most_common():
        out.append(f"| {k} | {v} |")
    out.append("")

    out.append("### 1.5 Concentração de queries por documento\n")
    out.append("Top documentos com mais entradas no golden set (sinal de viés):\n")
    out.append("| canonical_id | linhas |")
    out.append("|---|---|")
    for k, v in queries_by_doc.most_common(10):
        out.append(f"| {k} | {v} |")
    out.append("")

    out.append("## 2. Problemas detectados\n")
    if not issues:
        out.append("_Nenhum problema detectado._\n")
    else:
        for flag, items in issues.items():
            out.append(f"### Flag: `{flag}` ({len(items)} ocorrência(s))\n")
            out.append("| query_id | query | expected_doc_id | norm_id | art | qdrant_count |")
            out.append("|---|---|---|---|---|---|")
            for a in items:
                q = a.row.query.replace("|", "\\|")[:80]
                out.append(
                    f"| {a.row.query_id} | {q} | {a.row.expected_doc_id} | "
                    f"{a.norm_id} | {a.article_number or '-'} | {a.qdrant_match_count} |"
                )
            out.append("")

    out.append("## 3. Queries com intent de filtro/ordenação\n")
    out.append(
        "Estas queries sugerem que o **rewriter deveria emitir filtros**"
        " (`metadata.type` ou `metadata.authority`) ou **sorts** (por data, etc.)."
        " Use-as para expandir o golden set com casos explícitos.\n"
    )
    if not intent_hits:
        out.append("_Nenhuma query com intent claro de filtro/sort detectada._\n")
    else:
        out.append("| query_id | query | filters_detectados | sorts_detectados |")
        out.append("|---|---|---|---|")
        for qid, q, intent in intent_hits:
            q_safe = q.replace("|", "\\|")[:90]
            out.append(
                f"| {qid} | {q_safe} | "
                f"{', '.join(intent['filters']) or '-'} | "
                f"{', '.join(intent['sorts']) or '-'} |"
            )
        out.append("")

    out.append("## 4. Recomendações\n")
    recs: List[str] = []
    if "canonical_missing_in_store_db" in issues:
        recs.append(
            "Corrigir/remover linhas com `canonical_missing_in_store_db`:"
            " ou o ID está errado no golden set ou o documento não foi indexado."
        )
    if "article_missing_in_qdrant" in issues:
        recs.append(
            "Investigar `article_missing_in_qdrant`: artigo não existe nos chunks."
            " Pode ser typo no número do artigo, artigo de versão antiga, ou"
            " falha de chunking. Validar manualmente cada caso."
        )
    if "canonical_not_latest" in issues:
        recs.append(
            "Linhas com `canonical_not_latest` podem ser intencionais (testando"
            " versão antiga) mas se não, atualizar para a versão vigente."
        )
    n_intent = len(intent_hits)
    if n_intent < 5:
        recs.append(
            f"Apenas **{n_intent}** queries com intent de filtro/sort foi detectada."
            " O golden set não exercita adequadamente o uso de filtros/ordenações"
            " do rewriter — adicionar pelo menos 5-10 queries explícitas com"
            " filtros (ex.: 'normas do DECEA sobre meteorologia') e sorts"
            " (ex.: 'normas mais recentes sobre drones')."
        )
    if max(queries_by_doc.values(), default=0) > 10:
        top_doc, top_n = queries_by_doc.most_common(1)[0]
        recs.append(
            f"Há concentração alta em `{top_doc}` ({top_n} linhas). Avaliar"
            " se isso introduz viés nas métricas agregadas."
        )
    if len(queries) < 100:
        recs.append(
            f"Tamanho do set ({len(queries)} queries únicas) é pequeno para"
            " medições estáveis. Plano: expandir para 120-150 (todo `expand_golden`)."
        )
    if not recs:
        recs.append("Nenhuma recomendação crítica. Set parece consistente.")
    for i, rec in enumerate(recs, 1):
        out.append(f"{i}. {rec}")
    out.append("")

    return "\n".join(out)


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = audit()
    report = render_report(result)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Report written to: {REPORT_PATH}")

    audits: List[RowAudit] = result["audits"]
    issue_total = sum(1 for a in audits if a.issue_flags)
    print(f"\nSummary: {issue_total}/{len(audits)} rows with issues.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
