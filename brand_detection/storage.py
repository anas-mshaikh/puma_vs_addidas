from __future__ import annotations

import re
from pathlib import Path


def validate_sql_identifier(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(
            f"Invalid Postgres table name: {name!r}. Use letters, numbers, and underscores only."
        )


def store_detections_to_postgres(
    dsn: str,
    table_name: str,
    run_id: str,
    output_video_path: Path,
    output_csv_path: Path,
    detections: list[dict[str, object]],
) -> None:
    if not detections:
        return

    validate_sql_identifier(table_name)

    import psycopg

    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id bigserial PRIMARY KEY,
            run_id text NOT NULL,
            frame_index integer NOT NULL,
            timestamp_sec double precision NOT NULL,
            brand text NOT NULL,
            confidence double precision NOT NULL,
            x1 integer NOT NULL,
            y1 integer NOT NULL,
            x2 integer NOT NULL,
            y2 integer NOT NULL,
            box_width integer NOT NULL,
            box_height integer NOT NULL,
            video_path text NOT NULL,
            output_video_path text NOT NULL,
            output_csv_path text NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now()
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
            %(run_id)s,
            %(frame_index)s,
            %(timestamp_sec)s,
            %(brand)s,
            %(confidence)s,
            %(x1)s,
            %(y1)s,
            %(x2)s,
            %(y2)s,
            %(box_width)s,
            %(box_height)s,
            %(video_path)s,
            %(output_video_path)s,
            %(output_csv_path)s
        )
    """

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)

            enriched_rows = []
            for row in detections:
                enriched = dict(row)
                enriched["run_id"] = run_id
                enriched["output_video_path"] = str(output_video_path)
                enriched["output_csv_path"] = str(output_csv_path)
                enriched_rows.append(enriched)

            cur.executemany(insert_sql, enriched_rows)
