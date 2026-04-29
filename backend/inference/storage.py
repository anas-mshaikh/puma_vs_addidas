from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def init_sqlite_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inference_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT,
                source_value TEXT,
                input_video_path TEXT,
                output_video_path TEXT,
                csv_path TEXT,
                model_path TEXT,
                confidence_threshold REAL,
                iou_threshold REAL,
                total_frames INTEGER,
                fps REAL,
                width INTEGER,
                height INTEGER,
                total_detections INTEGER,
                started_at TEXT,
                finished_at TEXT,
                status TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                frame_index INTEGER NOT NULL,
                timestamp_sec REAL NOT NULL,
                brand TEXT NOT NULL,
                confidence REAL NOT NULL,
                x1 INTEGER NOT NULL,
                y1 INTEGER NOT NULL,
                x2 INTEGER NOT NULL,
                y2 INTEGER NOT NULL,
                box_width INTEGER NOT NULL,
                box_height INTEGER NOT NULL,
                area INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES inference_runs(id)
            )
            """
        )
        conn.commit()


def create_inference_run(
    db_path: Path,
    *,
    source_type: str | None,
    source_value: str | None,
    input_video_path: str | None,
    model_path: str,
    confidence_threshold: float,
    iou_threshold: float,
    total_frames: int | None,
    fps: float | None,
    width: int | None,
    height: int | None,
    started_at: str | None = None,
    status: str = "running",
) -> int:
    init_sqlite_db(db_path)

    started_at = started_at or utc_now_iso()

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.execute(
            """
            INSERT INTO inference_runs (
                source_type,
                source_value,
                input_video_path,
                output_video_path,
                csv_path,
                model_path,
                confidence_threshold,
                iou_threshold,
                total_frames,
                fps,
                width,
                height,
                total_detections,
                started_at,
                finished_at,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_type,
                source_value,
                input_video_path,
                None,
                None,
                model_path,
                confidence_threshold,
                iou_threshold,
                total_frames,
                fps,
                width,
                height,
                0,
                started_at,
                None,
                status,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def add_detections(
    db_path: Path,
    *,
    run_id: int,
    detections: list[dict[str, object]],
) -> None:
    if not detections:
        return

    init_sqlite_db(db_path)

    rows = []
    for row in detections:
        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])
        rows.append(
            (
                run_id,
                int(row["frame_index"]),
                float(row["timestamp_sec"]),
                str(row["brand"]),
                float(row["confidence"]),
                x1,
                y1,
                x2,
                y2,
                int(row["box_width"]),
                int(row["box_height"]),
                int(abs((x2 - x1) * (y2 - y1))),
            )
        )

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executemany(
            """
            INSERT INTO detections (
                run_id,
                frame_index,
                timestamp_sec,
                brand,
                confidence,
                x1,
                y1,
                x2,
                y2,
                box_width,
                box_height,
                area
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def finalize_inference_run(
    db_path: Path,
    *,
    run_id: int,
    output_video_path: str,
    csv_path: str,
    total_detections: int,
    finished_at: str | None = None,
    status: str = "completed",
) -> None:
    init_sqlite_db(db_path)
    finished_at = finished_at or utc_now_iso()

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            UPDATE inference_runs
            SET output_video_path = ?,
                csv_path = ?,
                total_detections = ?,
                finished_at = ?,
                status = ?
            WHERE id = ?
            """,
            (
                output_video_path,
                csv_path,
                total_detections,
                finished_at,
                status,
                run_id,
            ),
        )
        conn.commit()
