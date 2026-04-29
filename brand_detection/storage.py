from __future__ import annotations

import re
import sqlite3
from pathlib import Path


def validate_sql_identifier(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(
            f"Invalid SQLite table name: {name!r}. Use letters, numbers, and underscores only."
        )


def store_detections_to_sqlite(
    db_path: Path,
    table_name: str,
    run_id: str,
    output_video_path: Path,
    output_csv_path: Path,
    detections: list[dict[str, object]],
) -> None:
    if not detections:
        return

    validate_sql_identifier(table_name)

    db_path.parent.mkdir(parents=True, exist_ok=True)

    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id text NOT NULL,
            frame_index integer NOT NULL,
            timestamp_sec real NOT NULL,
            brand text NOT NULL,
            confidence real NOT NULL,
            x1 integer NOT NULL,
            y1 integer NOT NULL,
            x2 integer NOT NULL,
            y2 integer NOT NULL,
            box_width integer NOT NULL,
            box_height integer NOT NULL,
            video_path text NOT NULL,
            output_video_path text NOT NULL,
            output_csv_path text NOT NULL,
            created_at text NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """

    insert_sql = f"""
        INSERT INTO {table_name} (
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
            video_path,
            output_video_path,
            output_csv_path
        ) VALUES (
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?
        )
    """

    with sqlite3.connect(db_path) as conn:
        conn.execute(create_table_sql)

        rows = []
        for row in detections:
            rows.append(
                (
                    run_id,
                    row["frame_index"],
                    row["timestamp_sec"],
                    row["brand"],
                    row["confidence"],
                    row["x1"],
                    row["y1"],
                    row["x2"],
                    row["y2"],
                    row["box_width"],
                    row["box_height"],
                    row["video_path"],
                    str(output_video_path),
                    str(output_csv_path),
                )
            )

        conn.executemany(insert_sql, rows)
        conn.commit()
