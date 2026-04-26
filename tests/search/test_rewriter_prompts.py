"""Tests for the rewriter prompt selector (v1 / v2)."""

from __future__ import annotations

import os

import pytest

from search.rewriter.prompts import build_system_prompt, _build_v1, _build_v2


@pytest.fixture(autouse=True)
def _restore_env():
    """Ensure REWRITER_PROMPT_VERSION never leaks across tests."""
    saved = os.environ.get("REWRITER_PROMPT_VERSION")
    yield
    if saved is None:
        os.environ.pop("REWRITER_PROMPT_VERSION", None)
    else:
        os.environ["REWRITER_PROMPT_VERSION"] = saved


def test_default_is_v1():
    os.environ.pop("REWRITER_PROMPT_VERSION", None)
    p = build_system_prompt(3)
    assert p == _build_v1(3)


def test_unknown_version_falls_back_to_v1():
    os.environ["REWRITER_PROMPT_VERSION"] = "v99"
    assert build_system_prompt(3) == _build_v1(3)


def test_v2_selected_by_env():
    os.environ["REWRITER_PROMPT_VERSION"] = "v2"
    assert build_system_prompt(3) == _build_v2(3)


def test_v2_is_case_insensitive_and_trimmed():
    os.environ["REWRITER_PROMPT_VERSION"] = "  V2 "
    assert build_system_prompt(3) == _build_v2(3)


def test_v1_does_not_mention_sorts_or_effective_date():
    """Regression guard: v1 must remain the historical baseline."""
    p = _build_v1(3)
    assert "effective_date" not in p
    assert "sorts" not in p.lower() or 'sorts":[]' in p


def test_v2_teaches_effective_date_and_sorts():
    p = _build_v2(3)
    assert "effective_date" in p
    assert "SORTS" in p
    assert '"order":"desc"' in p
    assert "última" in p


def test_max_queries_is_substituted():
    p1 = _build_v1(5)
    p2 = _build_v2(5)
    assert "5 diverse search queries" in p1
    assert "5 diverse search queries" in p2


def test_v2_enforces_value_fidelity():
    """v2.2 guard: must explicitly forbid substituting unlisted values."""
    p = _build_v2(3)
    assert "VALUE FIDELITY" in p
    assert "do NOT substitute" in p.lower() or "DO NOT substitute" in p
    assert "RBAC" in p and "ANAC" in p


def test_v2_enforces_portuguese_only():
    """v2.2 guard: must explicitly forbid English sub-queries."""
    p = _build_v2(3)
    assert "Brazilian Portuguese" in p
    assert "NEVER emit a sub-query in English" in p


def test_v2_teaches_metadata_number_filter():
    """v2.3 guard: must list metadata.number as an allowed filter and
    require literal values copied from the user query."""
    p = _build_v2(3)
    assert "metadata.number" in p
    assert "NUMBER FILTER" in p
    assert "literal" in p.lower()
    assert "100-40" in p


def test_v2_teaches_simple_number_independent_of_type():
    """v2.3 patch: number filter MUST fire for simple numbers ("91", "7565")
    even when the type/authority word ("RBAC", "Lei") is not in the
    allowed list. Must include explicit examples for both shapes."""
    p = _build_v2(3)
    assert '"value":"91"' in p
    assert '"value":"7565"' in p
    assert "RBAC 91" in p
    assert "Lei 7565" in p
    assert "INDEPENDENT of rule 2" in p
