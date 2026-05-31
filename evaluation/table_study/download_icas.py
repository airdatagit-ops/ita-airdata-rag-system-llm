"""One-shot helper to download a few SISLAER PDFs into data/originals/sislaer/.

Implements SISLAER's anti-bot two-step:
  1. GET  /acervo/detalhe/{id}              -> validation page w/ AntiForgeryToken
  2. wait 3s, POST /acervo/validaacessodetalhe with the token (Resultado=true)
  3. GET  /acervo/detalhe/{id} again        -> real detail page

Reuses the project's SISLAERScraper for the cookie session.
Run from repo root with venv active:

    python data/_table_study/download_icas.py
"""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bs4 import BeautifulSoup  # noqa: E402
from loguru import logger  # noqa: E402

from crawler.scrapers.sislaer_scraper import SISLAERScraper  # noqa: E402


TARGETS = [
    ("ica_100-12_2024", 48940),
    ("ica_100-37_2024", 48962),
]

OUT_DIR = ROOT / "data" / "originals" / "sislaer"


async def _solve_validation(scraper: SISLAERScraper, reg_id: int) -> str | None:
    """Run the two-step anti-bot validation. Returns the real detail HTML."""
    url = f"{scraper._base_url}/acervo/detalhe/{reg_id}"
    async with scraper._session.get(url) as resp:
        first = await resp.text()

    m = re.search(r"window\.AntiForgeryToken\s*=\s*'([^']+)'", first)
    if not m:
        logger.error(f"[{reg_id}] no AntiForgeryToken on validation page")
        return None
    token = m.group(1)

    await asyncio.sleep(3.2)

    headers = {
        "RequestVerificationToken": token,
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://www.sislaer.fab.mil.br",
        "Referer": url,
    }
    val_url = f"{scraper._base_url}/acervo/validaacessodetalhe"
    async with scraper._session.post(val_url, headers=headers) as r:
        body = await r.json(content_type=None)
        logger.debug(f"[{reg_id}] validaacesso => {body}")
        if not body.get("Resultado"):
            logger.error(f"[{reg_id}] validation rejected: {body}")
            return None

    async with scraper._session.get(url) as resp:
        html = await resp.text()
    if len(html) < 5000:
        logger.error(f"[{reg_id}] real detail HTML still too small ({len(html)} bytes)")
        return None
    return html


async def fetch_pdf(scraper: SISLAERScraper, reg_id: int) -> bytes | None:
    html = await _solve_validation(scraper, reg_id)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    candidates = list(soup.find_all("a", href=re.compile(r"Busca/Download")))
    if not candidates:
        candidates = list(soup.find_all("a", href=re.compile(r"\.pdf", re.I)))
    if not candidates:
        logger.warning(f"[{reg_id}] no download link on detail page")
        return None

    for a in candidates:
        pdf_url = a.get("href", "")
        if not pdf_url.startswith("http"):
            pdf_url = f"https://www.sislaer.fab.mil.br{pdf_url}"
        try:
            async with scraper._limiter:
                async with scraper._session.get(pdf_url) as resp:
                    if resp.status != 200:
                        logger.warning(f"[{reg_id}] download HTTP {resp.status} for {pdf_url}")
                        continue
                    pdf_bytes = await resp.read()
            if pdf_bytes[:4] == b"%PDF":
                return pdf_bytes
            logger.warning(f"[{reg_id}] download returned non-PDF bytes ({len(pdf_bytes)})")
        except Exception as exc:
            logger.error(f"[{reg_id}] download error: {exc}")
    return None


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    async with SISLAERScraper() as scraper:
        await scraper._ensure_search_session()
        for label, reg_id in TARGETS:
            out = OUT_DIR / f"{label}.pdf"
            if out.exists() and out.stat().st_size > 1000:
                logger.info(f"{label}: already exists ({out.stat().st_size//1024} KB)")
                continue
            pdf = await fetch_pdf(scraper, reg_id)
            if not pdf:
                logger.error(f"{label}: download failed")
                continue
            out.write_bytes(pdf)
            logger.success(f"{label}: saved {len(pdf)//1024} KB to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
