"""Request-scoped Agent trajectory persistence.

Only structured runtime metadata and bounded log text are stored. Prompts, model
responses, visible token chunks and credentials are deliberately excluded.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SENSITIVE_KEY = re.compile(r"(?:api[_-]?key|authorization|password|secret|access[_-]?token)", re.I)
_SECRET_VALUE = re.compile(
    r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+|(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{8,}"
)
_MAX_CONTENT_CHARS = 2000
_MAX_FIELD_CHARS = 1000


def get_trace_db_path() -> Path:
    configured = os.environ.get("AGENT_TRACE_DB")
    return Path(configured).expanduser() if configured else PROJECT_ROOT / "agent_traces.sqlite3"


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or get_trace_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def init_trace_store(db_path: Path | None = None) -> None:
    with _connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS trace_runs (
                trace_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                research_mode TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at REAL NOT NULL,
                finished_at REAL,
                latency_ms REAL,
                error TEXT NOT NULL DEFAULT '',
                final_chars INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS trace_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                event TEXT NOT NULL,
                node TEXT NOT NULL DEFAULT '',
                timestamp REAL NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(trace_id) REFERENCES trace_runs(trace_id)
            );
            CREATE INDEX IF NOT EXISTS idx_trace_runs_user_session
                ON trace_runs(user_id, session_id, started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_trace_events_trace
                ON trace_events(trace_id, id);
            """
        )


def _redact_text(value: Any, limit: int) -> str:
    text = str(value or "")
    text = _SECRET_VALUE.sub(lambda match: (match.group(1) if match.group(1) else "") + "[REDACTED]", text)
    return text[:limit]


def _safe_value(key: str, value: Any) -> Any:
    if _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact_text(value, _MAX_FIELD_CHARS)
    if isinstance(value, dict):
        return {str(k): _safe_value(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(key, item) for item in list(value)[:50]]
    return _redact_text(value, _MAX_FIELD_CHARS)


def create_trace_run(
    trace_id: str,
    session_id: str,
    user_id: str,
    research_mode: str,
    *,
    db_path: Path | None = None,
) -> None:
    init_trace_store(db_path)
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO trace_runs(
                trace_id, session_id, user_id, research_mode, status, started_at
            ) VALUES (?, ?, ?, ?, 'running', ?)
            """,
            (trace_id, session_id, user_id, research_mode, time.time()),
        )


def append_trace_event(payload: dict, *, db_path: Path | None = None) -> None:
    if not payload or payload.get("event") == "visible_token":
        return
    trace_id = str(payload.get("trace_id") or "")
    if not trace_id:
        return
    safe_payload = {
        str(key): _safe_value(str(key), value)
        for key, value in payload.items()
        if key not in {"content", "trace_id", "session_id", "timestamp", "node"}
    }
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO trace_events(trace_id, event, node, timestamp, content, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                trace_id,
                _redact_text(payload.get("event"), 80),
                _redact_text(payload.get("node"), 80),
                float(payload.get("timestamp") or time.time()),
                _redact_text(payload.get("content"), _MAX_CONTENT_CHARS),
                json.dumps(safe_payload, ensure_ascii=False, default=str),
            ),
        )


def finish_trace_run(
    trace_id: str,
    status: str,
    *,
    latency_ms: float | None = None,
    error: str = "",
    final_chars: int = 0,
    db_path: Path | None = None,
) -> None:
    with _connect(db_path) as connection:
        connection.execute(
            """
            UPDATE trace_runs
            SET status = ?, finished_at = ?, latency_ms = ?, error = ?, final_chars = ?
            WHERE trace_id = ?
            """,
            (
                status,
                time.time(),
                latency_ms,
                _redact_text(error, _MAX_FIELD_CHARS),
                max(0, int(final_chars or 0)),
                trace_id,
            ),
        )


def list_trace_runs(
    user_id: str,
    *,
    session_id: str = "",
    limit: int = 50,
    db_path: Path | None = None,
) -> list[dict]:
    init_trace_store(db_path)
    parameters: list[Any] = [user_id]
    query = "SELECT * FROM trace_runs WHERE user_id = ?"
    if session_id:
        query += " AND session_id = ?"
        parameters.append(session_id)
    query += " ORDER BY started_at DESC LIMIT ?"
    parameters.append(max(1, min(int(limit), 200)))
    with _connect(db_path) as connection:
        return [dict(row) for row in connection.execute(query, parameters).fetchall()]


def get_trace_run(trace_id: str, user_id: str, *, db_path: Path | None = None) -> dict | None:
    init_trace_store(db_path)
    with _connect(db_path) as connection:
        run = connection.execute(
            "SELECT * FROM trace_runs WHERE trace_id = ? AND user_id = ?",
            (trace_id, user_id),
        ).fetchone()
        if run is None:
            return None
        events = []
        for row in connection.execute(
            "SELECT event, node, timestamp, content, payload_json FROM trace_events WHERE trace_id = ? ORDER BY id",
            (trace_id,),
        ).fetchall():
            item = dict(row)
            try:
                item["fields"] = json.loads(item.pop("payload_json") or "{}")
            except json.JSONDecodeError:
                item["fields"] = {}
                item.pop("payload_json", None)
            events.append(item)
        result = dict(run)
        result["events"] = events
        return result
