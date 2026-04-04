"""Tests for the timeout utility."""

import time

import pytest

from search.shared.timeouts import with_timeout


class TestWithTimeout:
    def test_fast_function_returns_result(self):
        result, timed_out = with_timeout(
            lambda: "ok",
            timeout_seconds=5,
            fallback_value="fallback",
            stage_name="test",
        )
        assert result == "ok"
        assert timed_out is False

    def test_slow_function_returns_fallback(self):
        def slow():
            time.sleep(10)
            return "late"

        result, timed_out = with_timeout(
            slow,
            timeout_seconds=1,
            fallback_value="fallback",
            stage_name="test",
        )
        assert result == "fallback"
        assert timed_out is True

    def test_exception_propagates(self):
        def failing():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            with_timeout(
                failing,
                timeout_seconds=5,
                fallback_value="fallback",
                stage_name="test",
            )

    def test_args_and_kwargs(self):
        def add(a, b, extra=0):
            return a + b + extra

        result, timed_out = with_timeout(
            add,
            timeout_seconds=5,
            fallback_value=0,
            stage_name="test",
            args=(1, 2),
            kwargs={"extra": 10},
        )
        assert result == 13
        assert timed_out is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
