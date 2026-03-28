"""
Interactive SQL console for the document store (SQLite).

Supports one-shot queries via --sql or an interactive REPL.
Opens the database in read-only mode to prevent accidental writes.

Usage:
    python -m scripts.query --sql "SELECT source, COUNT(*) n FROM documents GROUP BY source"
    python -m scripts.query          # interactive REPL
    make query SQL="SELECT ..."
    make query                       # interactive REPL
"""

import argparse
import readline  # noqa: F401 – enables arrow-key history in input()
import sqlite3
import sys
from pathlib import Path

from config import config

_STORE_PATH = Path(config.STORE_DB_PATH)

_HELP_TEXT = """
Commands:
  \\tables          List all tables
  \\schema [TABLE]  Show schema (all tables or a specific one)
  \\counts          Row counts per table
  \\sources         Documents grouped by source
  \\types           Documents grouped by source + doc_type
  \\help            Show this help
  \\q               Quit

Or type any SQL SELECT statement.
""".strip()

_PRESET_QUERIES = {
    "\\counts": """
        SELECT 'documents' AS tbl, COUNT(*) AS rows FROM documents
        UNION ALL
        SELECT 'embedding_log', COUNT(*) FROM embedding_log
    """,
    "\\sources": """
        SELECT source, COUNT(*) AS total,
               ROUND(AVG(LENGTH(content))) AS avg_chars,
               MIN(LENGTH(content)) AS min_chars,
               MAX(LENGTH(content)) AS max_chars
        FROM documents GROUP BY source ORDER BY total DESC
    """,
    "\\types": """
        SELECT source, doc_type, COUNT(*) AS total
        FROM documents
        GROUP BY source, doc_type
        ORDER BY source, total DESC
    """,
}


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        print(f"Error: database not found at {db_path}", file=sys.stderr)
        print("Run 'make collect' first to populate the store.", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _format_table(rows: list[sqlite3.Row]) -> str:
    if not rows:
        return "(0 rows)"

    headers = rows[0].keys()
    col_widths = [len(h) for h in headers]
    str_rows = []
    for row in rows:
        cells = [str(v) if v is not None else "NULL" for v in row]
        str_rows.append(cells)
        for i, cell in enumerate(cells):
            col_widths[i] = max(col_widths[i], len(cell))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    hdr = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"

    lines = [sep, hdr, sep]
    for cells in str_rows:
        lines.append("| " + " | ".join(c.ljust(w) for c, w in zip(cells, col_widths)) + " |")
    lines.append(sep)
    lines.append(f"({len(rows)} row{'s' if len(rows) != 1 else ''})")
    return "\n".join(lines)


def _run_query(conn: sqlite3.Connection, sql: str) -> None:
    try:
        rows = conn.execute(sql).fetchall()
        print(_format_table(rows))
    except sqlite3.Error as exc:
        print(f"SQL error: {exc}", file=sys.stderr)


def _handle_tables(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    for r in rows:
        print(f"  {r['name']}")


def _handle_schema(conn: sqlite3.Connection, table: str = None) -> None:
    if table:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type IN ('table','index') ORDER BY type, name"
        ).fetchall()

    for r in rows:
        if r["sql"]:
            print(r["sql"] + ";")
            print()


def _repl(conn: sqlite3.Connection) -> None:
    print(f"Connected to {_STORE_PATH} (read-only)")
    print("Type \\help for commands, \\q to quit.\n")

    while True:
        try:
            line = input("sql> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        cmd = line.lower().split()[0]

        if cmd in ("\\q", "\\quit", "exit", "quit"):
            break
        elif cmd == "\\help":
            print(_HELP_TEXT)
        elif cmd == "\\tables":
            _handle_tables(conn)
        elif cmd.startswith("\\schema"):
            parts = line.split(maxsplit=1)
            _handle_schema(conn, parts[1] if len(parts) > 1 else None)
        elif cmd in _PRESET_QUERIES:
            _run_query(conn, _PRESET_QUERIES[cmd])
        elif line.upper().startswith(("SELECT", "WITH", "PRAGMA", "EXPLAIN")):
            _run_query(conn, line)
        else:
            print("Only SELECT queries are allowed. Type \\help for commands.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the document store")
    parser.add_argument("--sql", type=str, help="SQL query to run (non-interactive)")
    parser.add_argument("--db", type=str, default=None, help="Path to SQLite database")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else _STORE_PATH
    conn = _connect(db_path)

    if args.sql:
        _run_query(conn, args.sql)
    else:
        _repl(conn)

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
