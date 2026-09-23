"""SQLite-хранилище результатов анализа (jobs + passes)."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    video_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending|running|done|error
    fps REAL, width INTEGER, height INTEGER, n_frames INTEGER,
    error TEXT,
    created_at TEXT, finished_at TEXT,
    result_video_path TEXT          -- видео с отрисованными траекториями
);
CREATE TABLE IF NOT EXISTS passes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    release_frame INTEGER, catch_frame INTEGER,
    time_of_flight_s REAL, apex_height_m REAL, distance_m REAL,
    initial_speed_mps REAL, peak_speed_mps REAL,
    method TEXT, passer_bbox TEXT, catcher_bbox TEXT,
    trajectory_json TEXT
);
"""


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
    c = sqlite3.connect(settings.db_path)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript(SCHEMA)
        # мягкая миграция для старых БД: добавляем недостающие колонки
        cols = {r[1] for r in c.execute("PRAGMA table_info(jobs)")}
        if "result_video_path" not in cols:
            c.execute("ALTER TABLE jobs ADD COLUMN result_video_path TEXT")


def set_result_video(jid: str, path: str) -> None:
    with _conn() as c:
        c.execute("UPDATE jobs SET result_video_path=? WHERE id=?", (path, jid))


def create_job(video_path: str) -> str:
    jid = uuid.uuid4().hex[:12]
    with _conn() as c:
        c.execute("INSERT INTO jobs (id, video_path, status, created_at) VALUES (?,?,?,?)",
                  (jid, video_path, "pending", datetime.now(timezone.utc).isoformat()))
    return jid


def set_status(jid: str, status: str, meta: dict | None = None, error: str | None = None) -> None:
    fields, vals = ["status=?"], [status]
    if meta:
        for k in ("fps", "width", "height", "n_frames"):
            if k in meta:
                fields.append(f"{k}=?")
                vals.append(meta[k])
    if error:
        fields.append("error=?")
        vals.append(error)
    if status in ("done", "error"):
        fields.append("finished_at=?")
        vals.append(datetime.now(timezone.utc).isoformat())
    vals.append(jid)
    with _conn() as c:
        c.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id=?", vals)


def save_passes(jid: str, passes: list[dict]) -> None:
    with _conn() as c:
        c.executemany(
            """INSERT INTO passes (job_id, release_frame, catch_frame, time_of_flight_s,
               apex_height_m, distance_m, initial_speed_mps, peak_speed_mps, method,
               passer_bbox, catcher_bbox, trajectory_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(jid, p["release_frame"], p["catch_frame"], p["time_of_flight_s"],
              p["apex_height_m"], p["distance_m"], p["initial_speed_mps"],
              p["peak_speed_mps"], p["method"], json.dumps(p.get("passer_bbox")),
              json.dumps(p.get("catcher_bbox")), json.dumps(p.get("trajectory", [])))
             for p in passes])


def get_job(jid: str) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
    return dict(row) if row else None


def list_jobs(limit: int = 50) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_passes(jid: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM passes WHERE job_id=? ORDER BY release_frame", (jid,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["trajectory"] = json.loads(d.pop("trajectory_json") or "[]")
        d["passer_bbox"] = json.loads(d["passer_bbox"] or "null")
        d["catcher_bbox"] = json.loads(d["catcher_bbox"] or "null")
        out.append(d)
    return out
