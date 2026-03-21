"""
Benchmark: Async LexML Scraper performance test.

Measures search + download times comparing sequential vs parallel strategies.

Run from project root:
    python scripts/benchmark_lexml.py --limit 20 --download 10
"""

import argparse
import asyncio
import time

from loguru import logger

from config import config
from crawler.scrapers.lexml_scraper import LexMLScraper


async def _benchmark_search(scraper: LexMLScraper, keywords: list, limit: int) -> tuple:
    """Returns (documents, elapsed_seconds)."""
    start = time.perf_counter()
    documents = await scraper.search(keywords=keywords, limit=limit, doc_type="Legislação")
    return documents, time.perf_counter() - start


async def _benchmark_sequential(scraper: LexMLScraper, documents: list, n: int) -> tuple:
    """Download documents one by one. Returns (success_count, elapsed_seconds)."""
    success = 0
    start = time.perf_counter()
    for doc in documents[:n]:
        if await scraper.get_document_text(doc):
            success += 1
    return success, time.perf_counter() - start


async def _benchmark_parallel(
    scraper: LexMLScraper, documents: list, n: int, concurrency: int = 5
) -> tuple:
    """Download documents in parallel. Returns (success_count, elapsed_seconds)."""
    sem = asyncio.Semaphore(concurrency)

    async def _one(doc):
        async with sem:
            return await scraper.get_document_text(doc)

    start = time.perf_counter()
    results = await asyncio.gather(
        *[_one(d) for d in documents[:n]], return_exceptions=True
    )
    elapsed = time.perf_counter() - start
    success = sum(1 for r in results if r and not isinstance(r, Exception))
    return success, elapsed


async def run_benchmark(keywords: list, limit: int, download_count: int) -> None:
    logger.info("=" * 60)
    logger.info("LexML Async Scraper Benchmark")
    logger.info("=" * 60)

    # Search (shared across both strategies)
    async with LexMLScraper(skip_duplicates=False, max_rate=1) as scraper:
        docs, search_time = await _benchmark_search(scraper, keywords, limit)
        logger.info(f"[Search] {len(docs)} docs in {search_time:.2f}s")

        if not docs:
            logger.warning("No documents found — cannot benchmark downloads.")
            return

        n = min(download_count, len(docs))

        # Sequential (1 req/s — simulates old synchronous behaviour)
        logger.info(f"\n[Sequential 1 req/s] Downloading {n} documents...")
        seq_ok, seq_time = await _benchmark_sequential(scraper, docs, n)
        logger.info(
            f"[Sequential] {seq_ok}/{n} OK in {seq_time:.2f}s "
            f"({seq_time / max(n, 1):.2f}s avg)"
        )

    # Parallel (5 req/s — default async mode)
    async with LexMLScraper(skip_duplicates=False, max_rate=5) as scraper:
        logger.info(f"\n[Parallel-5 @ 5 req/s] Downloading {n} documents...")
        par_ok, par_time = await _benchmark_parallel(scraper, docs, n, concurrency=5)
        logger.info(
            f"[Parallel-5] {par_ok}/{n} OK in {par_time:.2f}s "
            f"({par_time / max(n, 1):.2f}s avg)"
        )

    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"  Search ({len(docs)} docs):  {search_time:.2f}s")
    logger.info(f"  Sequential ({n} docs):      {seq_time:.2f}s")
    logger.info(f"  Parallel-5 ({n} docs):      {par_time:.2f}s")
    if seq_time > 0:
        speedup = seq_time / max(par_time, 0.001)
        logger.info(f"  Speedup:                    {speedup:.1f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark async LexML scraper")
    parser.add_argument("--keywords", type=str, default=None, help="Comma-separated keywords")
    parser.add_argument("--limit", type=int, default=20, help="Max docs to search")
    parser.add_argument("--download", type=int, default=10, help="Docs to download for benchmark")
    args = parser.parse_args()

    keywords = args.keywords.split(",") if args.keywords else config.lexml_keywords_list
    asyncio.run(run_benchmark(keywords, args.limit, args.download))


if __name__ == "__main__":
    main()
