"""Backfill feedback events from legacy chat_history JSON files.

Reads ``web/chat_history/*.json`` and imports every assistant message
rating as ``kind='star'`` events into ``data/app.db``.  Idempotent via the
``uq_fb_backfill_star`` unique index: re-running does not create
duplicates.

Usage:
    python scripts/backfill_feedback_from_json.py
    python scripts/backfill_feedback_from_json.py --history-dir web/chat_history --db data/app.db
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_app_store_module():
    """Load ``web/app/app_store.py`` without triggering the web.app package init."""
    path = PROJECT_ROOT / "web" / "app" / "app_store.py"
    spec = importlib.util.spec_from_file_location("app_store", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _coerce_ts(value) -> str:
    if not value:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )
    return str(value)


def backfill(history_dir: Path, db_path: Path) -> dict:
    app_store_mod = _load_app_store_module()
    store = app_store_mod.AppStore(db_path)
    try:
        files = sorted(history_dir.glob("*.json"))
        total_files = len(files)
        imported = 0
        skipped = 0
        for fp in files:
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover — defensive
                print(f"[skip] {fp.name}: {exc}", file=sys.stderr)
                continue
            session_id = data.get("session_id") or fp.stem
            model = data.get("model")
            for msg in data.get("messages", []):
                if msg.get("role") != "assistant":
                    continue
                message_id = msg.get("message_id")
                if not message_id:
                    continue
                ratings = msg.get("ratings") or {}
                ts = _coerce_ts(msg.get("timestamp"))
                for cat, val in ratings.items():
                    if cat not in app_store_mod.FEEDBACK_STAR_CATEGORIES:
                        continue
                    try:
                        ivalue = int(val)
                    except (TypeError, ValueError):
                        continue
                    if not (1 <= ivalue <= 5):
                        continue
                    try:
                        store.feedback.insert_event(
                            session_id=session_id,
                            message_id=message_id,
                            kind="star",
                            star_category=cat,
                            star_value=ivalue,
                            model_used=msg.get("model") or model,
                            used_rag=msg.get("use_rag"),
                            assistant_text=(msg.get("content") or "")[:2000],
                            sources_json=(
                                json.dumps(msg.get("sources"), ensure_ascii=False)
                                if msg.get("sources")
                                else None
                            ),
                            created_at=ts,
                        )
                        imported += 1
                    except Exception as exc:
                        msg_txt = str(exc).lower()
                        if "unique" in msg_txt:
                            skipped += 1
                        else:
                            print(
                                f"[warn] failed {fp.name} {message_id} {cat}: {exc}",
                                file=sys.stderr,
                            )
        return {"files": total_files, "imported": imported, "skipped_duplicates": skipped}
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill feedback from JSON history")
    parser.add_argument(
        "--history-dir",
        default=str(PROJECT_ROOT / "web" / "chat_history"),
        help="Directory with session JSON files (default: web/chat_history)",
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "data" / "app.db"),
        help="Target app.db path (default: data/app.db)",
    )
    args = parser.parse_args(argv)

    stats = backfill(Path(args.history_dir), Path(args.db))
    print(
        f"Backfill finished: files={stats['files']} "
        f"imported={stats['imported']} "
        f"skipped_duplicates={stats['skipped_duplicates']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
