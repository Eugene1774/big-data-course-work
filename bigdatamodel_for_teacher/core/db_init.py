from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "course_logs.db"


def get_interaction_log_db_path() -> Path:
    return DB_PATH


def _apply_connection_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=10000;")


def init_interaction_log_db(db_path: str | Path | None = None) -> Path:
    resolved_path = Path(db_path) if db_path else DB_PATH
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(resolved_path, timeout=10.0) as conn:
        _apply_connection_pragmas(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interaction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                student_id TEXT,
                project_id TEXT,
                agent_role TEXT,
                user_input TEXT,
                agent_response TEXT,
                triggered_rules TEXT,
                next_step TEXT,
                session_id TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_interaction_logs_student_project_ts
            ON interaction_logs(student_id, project_id, timestamp)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_interaction_logs_session_ts
            ON interaction_logs(session_id, timestamp)
            """
        )
        conn.commit()
    return resolved_path


def _normalize_triggered_rules(triggered_rules: Any) -> List[Dict[str, Any]]:
    if triggered_rules is None:
        return []
    if isinstance(triggered_rules, str):
        text = triggered_rules.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return []
        return _normalize_triggered_rules(decoded)
    if isinstance(triggered_rules, dict):
        triggered_rules = [triggered_rules]
    if not isinstance(triggered_rules, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in triggered_rules:
        if isinstance(item, dict):
            normalized.append(dict(item))
            continue
        if isinstance(item, str):
            rule_id = item.strip().upper()
            if rule_id:
                normalized.append({"rule_id": rule_id})
    return normalized


def log_interaction_to_db(
    student_id: str,
    project_id: str,
    agent_role: str,
    user_input: str,
    agent_response: str,
    triggered_rules: Any,
    next_step: str,
    session_id: str,
    db_path: str | Path | None = None,
) -> int:
    resolved_path = init_interaction_log_db(db_path)
    payload = json.dumps(_normalize_triggered_rules(triggered_rules), ensure_ascii=False)

    with sqlite3.connect(resolved_path, timeout=10.0) as conn:
        _apply_connection_pragmas(conn)
        cursor = conn.execute(
            """
            INSERT INTO interaction_logs (
                student_id,
                project_id,
                agent_role,
                user_input,
                agent_response,
                triggered_rules,
                next_step,
                session_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(student_id or "").strip(),
                str(project_id or "").strip(),
                str(agent_role or "").strip(),
                str(user_input or "").strip(),
                str(agent_response or "").strip(),
                payload,
                str(next_step or "").strip(),
                str(session_id or "").strip(),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def fetch_session_dialogue(
    session_id: str,
    turn_limit: int = 3,
    db_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    normalized_session = str(session_id or "").strip()
    if not normalized_session:
        return []

    resolved_path = init_interaction_log_db(db_path)
    safe_limit = max(1, min(int(turn_limit), 50))
    with sqlite3.connect(resolved_path, timeout=10.0) as conn:
        _apply_connection_pragmas(conn)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT user_input, agent_response, timestamp, agent_role
            FROM interaction_logs
            WHERE session_id = ?
            ORDER BY timestamp ASC, id ASC
            LIMIT ?
            """,
            (normalized_session, safe_limit),
        ).fetchall()
    return [dict(row) for row in rows]


def log_interaction_to_db_async(
    student_id: str,
    project_id: str,
    agent_role: str,
    user_input: str,
    agent_response: str,
    triggered_rules: Any,
    next_step: str,
    session_id: str,
    db_path: str | Path | None = None,
) -> None:
    """Fire-and-forget version of log_interaction_to_db that runs in a background thread."""
    def _background_insert() -> None:
        try:
            log_interaction_to_db(
                student_id=student_id,
                project_id=project_id,
                agent_role=agent_role,
                user_input=user_input,
                agent_response=agent_response,
                triggered_rules=triggered_rules,
                next_step=next_step,
                session_id=session_id,
                db_path=db_path,
            )
        except Exception:
            pass
    threading.Thread(target=_background_insert, daemon=True).start()


init_interaction_log_db()
