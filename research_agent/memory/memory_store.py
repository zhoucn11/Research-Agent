import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.environ.get("AGENT_MEMORY_DB", PROJECT_ROOT / "agent_memory.sqlite3"))
DEFAULT_USER_ID = "default_user"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_memory_store() -> None:
    with _connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )
        session_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        if "user_id" not in session_columns:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default_user'"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_user_updated ON sessions(user_id, updated_at DESC)"
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS session_summaries (
                session_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS deleted_sessions (
                session_id TEXT PRIMARY KEY,
                deleted_at TEXT NOT NULL
            )"""
        )


def is_session_deleted(session_id: str) -> bool:
    init_memory_store()
    with _connect() as conn:
        row = conn.execute(
            "SELECT session_id FROM deleted_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return bool(row)


def create_session(title: str | None = None, user_id: str = DEFAULT_USER_ID) -> dict:
    init_memory_store()
    session_id = "session_" + uuid.uuid4().hex[:12]
    title = (title or "New research session").strip()[:80]
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions(session_id, title, created_at, updated_at, user_id) VALUES (?, ?, ?, ?, ?)",
            (session_id, title, now, now, user_id),
        )
    return {"session_id": session_id, "title": title, "created_at": now, "updated_at": now}


def ensure_session(session_id: str, title: str | None = None, user_id: str = DEFAULT_USER_ID) -> bool:
    init_memory_store()
    now = _now()
    with _connect() as conn:
        deleted = conn.execute(
            "SELECT session_id FROM deleted_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if deleted:
            return False

        row = conn.execute(
            "SELECT session_id, user_id FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row:
            if row["user_id"] != user_id:
                return False
            conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
            return True
        conn.execute(
            "INSERT INTO sessions(session_id, title, created_at, updated_at, user_id) VALUES (?, ?, ?, ?, ?)",
            (session_id, (title or "Research session")[:80], now, now, user_id),
        )
    return True


def list_sessions(limit: int = 50, user_id: str = DEFAULT_USER_ID) -> list[dict]:
    init_memory_store()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT session_id, title, created_at, updated_at FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def rename_session(session_id: str, title: str, user_id: str = DEFAULT_USER_ID) -> dict | None:
    init_memory_store()
    title = str(title or "").strip()[:80]
    if not title:
        return None
    now = _now()
    with _connect() as conn:
        row = conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
        if not row:
            return None
        conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
            (title, now, session_id),
        )
        updated = conn.execute(
            "SELECT session_id, title, created_at, updated_at FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return dict(updated) if updated else None


def delete_session(session_id: str, user_id: str = DEFAULT_USER_ID) -> bool:
    init_memory_store()
    now = _now()
    with _connect() as conn:
        row = conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
        if not row:
            return False
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.execute(
            """INSERT INTO deleted_sessions(session_id, deleted_at)
               VALUES (?, ?)
               ON CONFLICT(session_id) DO UPDATE SET deleted_at = excluded.deleted_at""",
            (session_id, now),
        )
    return True


def append_message(session_id: str, role: str, content: str, user_id: str = DEFAULT_USER_ID) -> None:
    if not ensure_session(session_id, user_id=user_id):
        return
    content = str(content or "").strip()
    if not content:
        return
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now),
        )
        if role == "user":
            title = content.replace("\n", " ")[:60]
            conn.execute(
                "UPDATE sessions SET title = CASE WHEN title IN ('Research session', 'New research session') THEN ? ELSE title END, updated_at = ? WHERE session_id = ?",
                (title, now, session_id),
            )
        else:
            conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))


def get_messages(session_id: str, limit: int = 200, user_id: str = DEFAULT_USER_ID) -> list[dict]:
    init_memory_store()
    if is_session_deleted(session_id):
        return []
    with _connect() as conn:
        owner = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
        if not owner:
            return []
        rows = conn.execute(
            """SELECT role, content, created_at
               FROM (
                   SELECT id, role, content, created_at
                   FROM messages
                   WHERE session_id = ?
                   ORDER BY id DESC
                   LIMIT ?
               ) recent
               ORDER BY id ASC""",
            (session_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def get_session_summary(session_id: str, user_id: str = DEFAULT_USER_ID) -> str:
    init_memory_store()
    if is_session_deleted(session_id):
        return ""
    with _connect() as conn:
        owner = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
        if not owner:
            return ""
        row = conn.execute(
            "SELECT summary FROM session_summaries WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return row["summary"] if row else ""


def update_session_summary(session_id: str, summary: str, user_id: str = DEFAULT_USER_ID) -> None:
    if not summary:
        return
    if is_session_deleted(session_id):
        return
    now = _now()
    with _connect() as conn:
        owner = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
        if not owner:
            return
        conn.execute(
            """INSERT INTO session_summaries(session_id, summary, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET summary = excluded.summary, updated_at = excluded.updated_at""",
            (session_id, summary, now),
        )


def get_user_profile(user_id: str = DEFAULT_USER_ID) -> dict:
    init_memory_store()
    with _connect() as conn:
        row = conn.execute("SELECT profile_json FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
    if not row:
        return {"preferences": [], "research_interests": [], "output_preferences": []}
    try:
        return json.loads(row["profile_json"])
    except json.JSONDecodeError:
        return {"preferences": [], "research_interests": [], "output_preferences": []}


def profile_to_text(profile: dict) -> str:
    parts = []
    for key, label in [
        ("preferences", "User preferences"),
        ("research_interests", "Research interests"),
        ("output_preferences", "Output preferences"),
    ]:
        values = profile.get(key) or []
        if values:
            parts.append(label + ": " + "; ".join(values[-8:]))
    return "\n".join(parts)


def _append_unique(profile: dict, key: str, value: str, limit: int = 30) -> None:
    value = value.strip()
    if not value:
        return
    values = profile.setdefault(key, [])
    if value not in values:
        values.append(value)
    del values[:-limit]


def update_user_profile_from_turn(user_message: str, assistant_message: str, user_id: str = DEFAULT_USER_ID) -> dict:
    profile = get_user_profile(user_id)
    text = user_message.strip()
    lowered = text.lower()

    session_scoped_markers = ["本会话", "当前会话", "这轮对话", "这个会话", "这次对话", "这一轮"]
    if any(marker in text for marker in session_scoped_markers):
        return profile

    preference_markers = ["以后", "下次", "记住", "我喜欢", "我希望", "回答时", "风格", "格式", "不要"]
    if any(marker in text for marker in preference_markers):
        _append_unique(profile, "preferences", text[:240])

    interest_markers = ["我主要研究", "我关注", "方向", "课题", "topic", "research"]
    if any(marker in text for marker in interest_markers) or any(k in lowered for k in ["rag", "agent", "mcp"]):
        _append_unique(profile, "research_interests", text[:180])

    output_markers = ["表格", "markdown", "简历", "面试", "综述", "对比", "列表"]
    if any(marker in text for marker in output_markers):
        _append_unique(profile, "output_preferences", text[:180])

    now = _now()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO user_profiles(user_id, profile_json, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET profile_json = excluded.profile_json, updated_at = excluded.updated_at""",
            (user_id, json.dumps(profile, ensure_ascii=False), now),
        )
    return profile


def remove_user_profile_entries_containing(keywords: list[str], user_id: str = DEFAULT_USER_ID) -> dict:
    profile = get_user_profile(user_id)
    keywords = [keyword for keyword in keywords if keyword]
    if not keywords:
        return profile

    changed = False
    for key in ("preferences", "research_interests", "output_preferences"):
        values = profile.get(key) or []
        filtered = [
            value for value in values
            if not any(keyword in value for keyword in keywords)
        ]
        if len(filtered) != len(values):
            profile[key] = filtered
            changed = True

    if changed:
        now = _now()
        with _connect() as conn:
            conn.execute(
                """INSERT INTO user_profiles(user_id, profile_json, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET profile_json = excluded.profile_json, updated_at = excluded.updated_at""",
                (user_id, json.dumps(profile, ensure_ascii=False), now),
            )
    return profile
