"""
Shared file-logging configuration for the ocr_evaluation package.

Adds a rotating file sink to the loguru logger so every log message
(and unhandled exceptions) are persisted to disk.

Log files are written to:  ocr_evaluation/logs/ocr_evaluation_<timestamp>.log
"""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

_configured = False
LOG_DIR = Path(__file__).parent / "logs"


def configure_file_logging() -> None:
    """Add a rotating file sink to the shared loguru logger.

    Safe to call multiple times — only configures once per process.
    Also installs sys.excepthook so unhandled exceptions are captured
    and written to the log file before the process exits.
    """
    global _configured
    if _configured:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = LOG_DIR / f"ocr_evaluation_{timestamp}.log"

    logger.add(
        log_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        encoding="utf-8",
        enqueue=True,       # thread-safe writes
        rotation="50 MB",
        retention=10,       # keep last 10 log files
        backtrace=True,     # full traceback on exceptions
        diagnose=True,      # show variable values in tracebacks
    )

    # Capture unhandled exceptions so crashes are logged to file
    def _excepthook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.opt(exception=(exc_type, exc_value, exc_tb)).critical(
            "Uncaught exception — process is about to crash"
        )

    sys.excepthook = _excepthook

    logger.debug(f"File logging active → {log_path}")
    _configured = True
