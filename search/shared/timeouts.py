"""Timeout wrapper for LLM calls and other potentially slow operations.

Runs the target function in a thread and enforces a hard timeout.  On
expiry the caller receives the configured *fallback* value and the
timeout is recorded in the pipeline trace.
"""

from __future__ import annotations

import concurrent.futures
from typing import Callable, Optional, TypeVar

from loguru import logger


T = TypeVar("T")


def with_timeout(
    fn: Callable[..., T],
    timeout_seconds: int,
    fallback_value: T,
    *,
    stage_name: str = "unknown",
    args: tuple = (),
    kwargs: Optional[dict] = None,
) -> tuple[T, bool]:
    """Execute *fn* with a hard timeout.

    Returns:
        ``(result, timed_out)`` -- *timed_out* is ``True`` when the
        fallback was used.
    """
    kwargs = kwargs or {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result, False
        except concurrent.futures.TimeoutError:
            logger.warning(
                f"[{stage_name}] LLM call exceeded {timeout_seconds}s timeout — using fallback"
            )
            future.cancel()
            return fallback_value, True
        except Exception:
            raise
