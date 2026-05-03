"""Shared fixtures for web tests.

The ``web/app`` package imports Pydantic settings at module import time,
which would fail when the root ``.env`` contains keys the Settings class
does not allow. We therefore load modules under ``web/app/`` directly via
``importlib`` to keep tests self-contained and fast.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def app_store_module() -> ModuleType:
    return _load_module("app_store", PROJECT_ROOT / "web" / "app" / "app_store.py")


@pytest.fixture()
def app_store(tmp_path, app_store_module):
    store = app_store_module.AppStore(tmp_path / "app.db")
    try:
        yield store
    finally:
        store.close()


@pytest.fixture(scope="session")
def backfill_module() -> ModuleType:
    return _load_module(
        "backfill_feedback_from_json",
        PROJECT_ROOT / "scripts" / "backfill_feedback_from_json.py",
    )
