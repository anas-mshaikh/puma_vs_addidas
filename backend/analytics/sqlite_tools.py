from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from backend.analytics.sql_guard import (
    get_allowed_objects,
    quote_identifier,
    validate_readonly_sql,
)


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_database_schema(db_path: str) -> dict[str, Any]:
    conn = get_connection(db_path)

    try:
        rows = conn.execute(
            """
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()

        objects = []

        for row in rows:
            name = row["name"]
            obj_type = row["type"]

            columns = conn.execute(
                f"PRAGMA table_info({quote_identifier(name)})"
            ).fetchall()

            objects.append(
                {
                    "name": name,
                    "type": obj_type,
                    "columns": [
                        {
                            "name": col["name"],
                            "type": col["type"],
                        }
                        for col in columns
                    ],
                }
            )

        return {
            "database": str(Path(db_path).name),
            "objects": objects,
            "recommended_views": [
                "v_detections",
                "v_brand_summary",
                "v_first_appearance",
                "v_timeline_5s",
            ],
        }

    finally:
        conn.close()


def execute_readonly_sql(db_path: str, sql: str) -> dict[str, Any]:
    conn = get_connection(db_path)

    try:
        allowed_objects = get_allowed_objects(conn)
        safe_sql = validate_readonly_sql(sql, allowed_objects)

        # Basic query step guard.
        max_steps = 200_000
        steps = {"count": 0}

        def progress_handler():
            steps["count"] += 1
            if steps["count"] > max_steps:
                return 1
            return 0

        conn.set_progress_handler(progress_handler, 1000)

        try:
            df = pd.read_sql_query(safe_sql, conn)
        finally:
            conn.set_progress_handler(None, 0)

        return {
            "sql": safe_sql,
            "row_count": len(df),
            "columns": list(df.columns),
            "rows": df.head(100).to_dict(orient="records"),
        }

    finally:
        conn.close()


def format_tool_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)
