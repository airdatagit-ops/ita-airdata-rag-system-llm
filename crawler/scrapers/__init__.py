"""
Scraper registry — auto-discovers concrete scrapers at import time.

Usage:
    from crawler.scrapers import get_scraper, list_scrapers

    scraper = get_scraper("decea", timeout=30)
    all_names = list_scrapers()
"""

from __future__ import annotations

from typing import Dict, Type

from crawler.scrapers.base import BaseScraper, ScrapedDocument  # noqa: re-export

_REGISTRY: Dict[str, Type[BaseScraper]] = {}


def register_scraper(cls: Type[BaseScraper]) -> Type[BaseScraper]:
    """Class decorator that registers a scraper by its ``source_name``."""
    name = cls.source_name if isinstance(cls.source_name, str) else cls.source_name.fget(cls)
    _REGISTRY[name] = cls
    return cls


def get_scraper(name: str, **kwargs) -> BaseScraper:
    """Instantiate a registered scraper by source name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown scraper '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_scrapers() -> list[str]:
    """Return all registered scraper names."""
    return sorted(_REGISTRY)


def _auto_discover() -> None:
    """Import concrete scraper modules so their @register_scraper runs."""
    import importlib
    for mod_name in ("decea_scraper", "lexml_scraper", "pdf_scraper"):
        try:
            importlib.import_module(f"crawler.scrapers.{mod_name}")
        except ImportError:
            pass


_auto_discover()
