"""
Script para coletar RBACs da ANAC.

Uso:
    python scripts/collect_anac_rbac.py
    python scripts/collect_anac_rbac.py --concurrency 3
    python scripts/collect_anac_rbac.py --dry-run  # lista RBACs sem baixar
"""

import argparse
import asyncio

from loguru import logger
from crawler.scrapers.anac_rbac_scraper import ANACRBACscraper


async def main(concurrency: int, dry_run: bool) -> None:
    async with ANACRBACscraper() as scraper:
        docs = await scraper.search()
        logger.info(f"Found {len(docs)} RBACs")

        if dry_run:
            for d in docs:
                print(f"  {d['rbac_id']:25s}  {d['url']}")
            return

        results = await scraper.fetch_all(docs, concurrency=concurrency)
        logger.success(f"Done: {len(results)}/{len(docs)} RBACs saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.concurrency, args.dry_run))
