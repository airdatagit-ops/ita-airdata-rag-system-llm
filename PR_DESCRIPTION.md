## Restructure Indexing Pipeline into 3 Independent Phases

### Summary

- Replace the monolithic scrape-chunk-embed-upsert pipeline with a 3-phase architecture where each phase is idempotent and only reprocesses documents whose content has changed (SHA256 hash detection)
- Introduce SQLite (data/store.db) as the central document registry and Parquet (data/embeddings/) for persistent embedding storage, eliminating expensive GPU recomputation on repeat runs
- Add make collect, make embed, make index, and make pipeline Makefile targets that can be run independently or in sequence, and can evolve into DAGs or microservices in the future
- Optimize collect phase with 3-flag semantics (default skip / --check / --force) for fast re-runs
- Add SQL analytics interface (CLI + datasette web UI) for exploring collected documents

### Architecture

Phase 1: COLLECT -- Web Scrapers (DECEA, LexML, local PDFs) store documents into SQLite with SHA256 hash detection

Phase 2: EMBED -- Read from SQLite, apply QualityValidator + TextCleaner, chunk with ArticleChunker/ICAChunker, generate embeddings with EmbeddingModel (dense, GPU) and SparseEncoder (BM25), write to Parquet + embedding_log in SQLite

Phase 3: INDEX -- Load embeddings from Parquet, bulk upsert into Qdrant with deferred HNSW indexing (I/O only, no model loaded)

### Collection Modes

| Flag | Makefile | Behavior | Use case |
|---|---|---|---|
| (none) | `make collect` | Skip docs that already exist in SQLite (no HTTP) | Day-to-day re-runs (near-instant) |
| `--check` | `make collect CHECK=1` | Re-download all docs and recalculate hash, update only if changed | Periodic verification of source changes |
| `--force` | `make collect FORCE=1` | Delete all docs of the source from SQLite, then re-collect from scratch | Rebuild a specific source completely |

### Why SQLite + Parquet

| Criterion | JSON files | SQLite | Parquet |
|---|---|---|---|
| Millions of docs | Bad (FS limits) | Excellent | Excellent |
| Portable (SCP to server) | Many files | Single file | Single file |
| Content hash lookup | O(n) scan | O(1) indexed | Not designed for |
| Bulk vector read | N/A | Slow (BLOB deser.) | Excellent (columnar) |
| No cloud dependency | Yes | Yes | Yes |

### Files Created

| File | Purpose |
|---|---|
| pipeline/document_store.py | SQLite document registry with persistent connection, content hash change detection, exists(), delete_by_source() |
| pipeline/embedding_store.py | Parquet-backed embedding storage with incremental update support |
| scripts/collect.py | Phase 1 orchestrator: per-keyword LexML search, all DECEA types, local PDFs, 3-flag semantics |
| scripts/embed.py | Phase 2 orchestrator (dense/sparse/hybrid flag, incremental) |
| scripts/index.py | Phase 3 orchestrator (Parquet to Qdrant bulk upsert) |
| scripts/query.py | Interactive SQL console with REPL, one-shot mode, preset queries (\tables, \schema, \sources, \types) |
| tests/pipeline/test_document_store.py | 22 tests for DocumentStore |
| tests/pipeline/test_embedding_store.py | 11 tests for EmbeddingStore |

### Files Removed (obsolete — replaced by 3-phase pipeline)

| File | Replacement |
|---|---|
| scripts/ingest_decea.py | `make collect SOURCES=decea` |
| scripts/ingest_lexml.py | `make collect SOURCES=lexml` |
| scripts/ingest_pdfs.py | `make collect SOURCES=pdf` |
| scripts/setup_qdrant.py | `make index RECREATE=1` |
| scripts/reset_database.py | `make collect FORCE=1` + `make index RECREATE=1` |

### Files Modified

| File | Change |
|---|---|
| Makefile | Add collect, embed, index, pipeline, query, explore targets; 3-flag support (CHECK, FORCE); remove legacy collect-decea/collect-lexml targets |
| config.py | Add STORE_DB_PATH, EMBEDDINGS_DIR, DEFAULT_EMBEDDING_MODE |
| env.example | Document new pipeline config variables |
| requirements.txt | Add pyarrow >= 14.0.0, datasette >= 0.64.0 |
| README.md | Update architecture, scripts, and Makefile sections; remove legacy script references |
| QUICKSTART.md | Update to use 3-phase pipeline commands |
| START_HERE.md | Update project structure to reflect new scripts |

### Usage

Full pipeline (first run ~15-30 min, subsequent: seconds if nothing changed):

    make pipeline

Individual phases:

    make collect                              # fast re-run (skip existing)
    make collect CHECK=1                      # verify source changes
    make collect FORCE=1                      # wipe + re-collect from scratch
    make collect SOURCES=lexml LIMIT=200      # only LexML, 200 docs
    make collect SOURCES=lexml,decea,pdf      # all sources including local PDFs
    make embed MODE=hybrid
    make index RECREATE=1

Analytics:

    make query                                           # interactive SQL console
    make query SQL="SELECT source, COUNT(*) FROM documents GROUP BY source"
    make explore                                         # datasette web UI

### Performance Expectations

| Phase | First run (10K docs) | Subsequent (default mode) |
|---|---|---|
| Collect | ~10-30 min (network bound) | Near-instant (skip existing, no HTTP) |
| Collect --check | ~10-30 min | ~10-30 min (re-downloads to verify hashes) |
| Embed (GPU) | ~5-15 min (1024-dim, batch 64) | Seconds (0 docs changed) |
| Embed (CPU) | ~2-4 hours | Seconds |
| Index | ~30-60 seconds | ~30-60 seconds (full reload) |

### Test Plan

- [x] All 208 tests pass (33 new + 175 existing, zero regressions)
- [x] DocumentStore: insert/update/unchanged detection, metadata storage, embedding tracking
- [x] EmbeddingStore: Parquet roundtrip, append, remove by doc_ids, multi-source loading
- [x] Existing scrapers, chunkers, embedding models, Qdrant manager untouched
- [x] LexML per-keyword search returns results (tested: 3029 docs collected)
- [x] DECEA all doc-types collection works (tested: 447 docs, 8 types)
- [x] SQL query tool works against live database
- [ ] End-to-end: make embed with GPU on university server
- [ ] End-to-end: make index with Qdrant running
