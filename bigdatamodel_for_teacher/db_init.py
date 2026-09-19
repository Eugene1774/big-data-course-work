from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "app_state.db"


def get_db_path() -> Path:
    return DB_PATH


def _apply_connection_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        _apply_connection_pragmas(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interventions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_id TEXT NOT NULL,
                target_type TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'executed')),
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_interventions_target_status_ts
            ON interventions(target_id, target_type, status, timestamp)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_interventions_status_ts
            ON interventions(status, timestamp)
            """
        )
        conn.commit()


def insert_intervention(target_id: str, target_type: str, content: str) -> int:
    normalized_target_id = str(target_id or "").strip()
    normalized_target_type = str(target_type or "").strip()
    normalized_content = str(content or "").strip()
    if not normalized_target_id:
        raise ValueError("target_id is required.")
    if not normalized_target_type:
        raise ValueError("target_type is required.")
    if not normalized_content:
        raise ValueError("content is required.")

    timestamp = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        _apply_connection_pragmas(conn)
        cursor = conn.execute(
            """
            INSERT INTO interventions(target_id, target_type, content, status, timestamp)
            VALUES (?, ?, ?, 'pending', ?)
            """,
            (normalized_target_id, normalized_target_type, normalized_content, timestamp),
        )
        conn.commit()
        return int(cursor.lastrowid)


def fetch_pending_intervention(target_id: str, target_type: str = 'user') -> Dict[str, Any] | None:
    normalized_target_id = str(target_id or "").strip()
    if not normalized_target_id:
        return None

    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        _apply_connection_pragmas(conn)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT id, target_id, target_type, content, status, timestamp
            FROM interventions
            WHERE target_id = ? AND target_type = ? AND status = 'pending'
            ORDER BY timestamp ASC, id ASC
            LIMIT 1
            """,
            (normalized_target_id, target_type),
        ).fetchone()
    return dict(row) if row is not None else None


def mark_intervention_executed(intervention_id: int) -> bool:
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        _apply_connection_pragmas(conn)
        cursor = conn.execute(
            """
            UPDATE interventions
            SET status = 'executed'
            WHERE id = ? AND status = 'pending'
            """,
            (int(intervention_id),),
        )
        conn.commit()
        return cursor.rowcount > 0


def list_recent_interventions(target_id: str, target_type: str = 'user', limit: int = 5) -> List[Dict[str, Any]]:
    normalized_target_id = str(target_id or "").strip()
    if not normalized_target_id:
        return []

    safe_limit = max(1, min(int(limit), 20))
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        _apply_connection_pragmas(conn)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, target_id, target_type, content, status, timestamp
            FROM interventions
            WHERE target_id = ? AND target_type = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (normalized_target_id, target_type, safe_limit),
        ).fetchall()
    return [dict(row) for row in rows]


init_db()
