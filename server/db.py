import os
import sqlite3
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "meter.db")


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _conn()
    cur = conn.cursor()

    # Users
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user', 'coadmin', 'admin')),
        team INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Readings uploaded by users
    cur.execute("""
    CREATE TABLE IF NOT EXISTS readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        team INTEGER NOT NULL,
        meter_type TEXT NOT NULL,        -- 'earthing' or 'temp'
        label TEXT NOT NULL,             -- user label / site name etc
        value TEXT,                      -- numeric as text (to preserve '0.40')
        filename TEXT NOT NULL,
        ocr_json TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # Alerts for coadmin/admin
    cur.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        reading_id INTEGER NOT NULL,
        target_role TEXT NOT NULL CHECK(target_role IN ('coadmin', 'admin')),
        target_team INTEGER,              -- for coadmin alerts
        message TEXT NOT NULL,
        severity TEXT NOT NULL CHECK(severity IN ('low', 'high')),
        is_read INTEGER NOT NULL DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(reading_id) REFERENCES readings(id)
    )
    """)

    conn.commit()
    conn.close()


# -----------------------------
# Users
# -----------------------------
def create_user(username: str, password_hash: str, role: str, team: Optional[int]):
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users(username, password_hash, role, team) VALUES(?,?,?,?)",
        (username, password_hash, role, team),
    )
    conn.commit()
    conn.close()


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


# -----------------------------
# Readings
# -----------------------------
def insert_reading(*, user_id: int, team: int, meter_type: str, label: str, value: Optional[str], filename: str, ocr_json: str) -> int:
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO readings(user_id, team, meter_type, label, value, filename, ocr_json)
        VALUES(?,?,?,?,?,?,?)
        """,
        (user_id, team, meter_type, label, value, filename, ocr_json),
    )
    rid = cur.lastrowid
    conn.commit()
    conn.close()
    return int(rid)


def fetch_readings_all() -> List[Dict[str, Any]]:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT r.*, u.username
        FROM readings r
        JOIN users u ON u.id = r.user_id
        ORDER BY r.created_at DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return [dict(x) for x in rows]


def fetch_readings_by_team(team: int) -> List[Dict[str, Any]]:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT r.*, u.username
        FROM readings r
        JOIN users u ON u.id = r.user_id
        WHERE r.team = ?
        ORDER BY r.created_at DESC
    """, (team,))
    rows = cur.fetchall()
    conn.close()
    return [dict(x) for x in rows]


def fetch_readings_by_user(user_id: int) -> List[Dict[str, Any]]:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT r.*, u.username
        FROM readings r
        JOIN users u ON u.id = r.user_id
        WHERE r.user_id = ?
        ORDER BY r.created_at DESC
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()
    return [dict(x) for x in rows]


# -----------------------------
# Alerts
# -----------------------------
def create_alert(reading_id: int, target_role: str, target_team: Optional[int], message: str, severity: str):
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO alerts(reading_id, target_role, target_team, message, severity)
        VALUES(?,?,?,?,?)
        """,
        (reading_id, target_role, target_team, message, severity),
    )
    conn.commit()
    conn.close()


def fetch_alerts_for_admin(unread_only: bool = False) -> List[Dict[str, Any]]:
    conn = _conn()
    cur = conn.cursor()
    q = """
        SELECT a.*, r.team, r.label, r.meter_type, r.value, r.filename, r.created_at AS reading_time
        FROM alerts a
        JOIN readings r ON r.id = a.reading_id
        WHERE a.target_role='admin'
        {extra}
        ORDER BY a.created_at DESC
    """
    extra = "AND a.is_read=0" if unread_only else ""
    cur.execute(q.format(extra=extra))
    rows = cur.fetchall()
    conn.close()
    return [dict(x) for x in rows]


def fetch_alerts_for_coadmin(team: int, unread_only: bool = False) -> List[Dict[str, Any]]:
    conn = _conn()
    cur = conn.cursor()
    q = """
        SELECT a.*, r.team, r.label, r.meter_type, r.value, r.filename, r.created_at AS reading_time
        FROM alerts a
        JOIN readings r ON r.id = a.reading_id
        WHERE a.target_role='coadmin' AND a.target_team=?
        {extra}
        ORDER BY a.created_at DESC
    """
    extra = "AND a.is_read=0" if unread_only else ""
    cur.execute(q.format(extra=extra), (team,))
    rows = cur.fetchall()
    conn.close()
    return [dict(x) for x in rows]


def mark_alert_read(alert_id: int):
    conn = _conn()
    cur = conn.cursor()
    cur.execute("UPDATE alerts SET is_read=1 WHERE id=?", (alert_id,))
    conn.commit()
    conn.close()


def clear_alerts_admin():
    conn = _conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM alerts WHERE target_role='admin'")
    conn.commit()
    conn.close()


def count_unread_admin() -> int:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM alerts WHERE target_role='admin' AND is_read=0")
    row = cur.fetchone()
    conn.close()
    return int(row["c"])


def count_unread_coadmin(team: int) -> int:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM alerts WHERE target_role='coadmin' AND target_team=? AND is_read=0", (team,))
    row = cur.fetchone()
    conn.close()
    return int(row["c"])


def get_latest_reading_id_all() -> int:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM readings ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    return int(row["id"]) if row else 0
