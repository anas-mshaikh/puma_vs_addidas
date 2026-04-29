from __future__ import annotations

import re
import sqlite3

import sqlglot
from sqlglot import exp


BANNED_SQL_KEYWORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "replace",
    "truncate",
    "attach",
    "detach",
    "pragma",
    "vacuum",
    "reindex",
}


def quote_identifier(identifier: str) -> str:
    safe = identifier.replace('"', '""')
    return f'"{safe}"'


def get_allowed_objects(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type IN ('table', 'view')
          AND name NOT LIKE 'sqlite_%'
        """
    ).fetchall()

    return {row[0] for row in rows}


def validate_readonly_sql(sql: str, allowed_objects: set[str]) -> str:
    sql = sql.strip().rstrip(";").strip()

    if not sql:
        raise ValueError("The model generated empty SQL.")

    if ";" in sql:
        raise ValueError("Only one SQL statement is allowed.")

    lowered = sql.lower()

    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ValueError("Only SELECT queries are allowed.")

    tokens = set(re.findall(r"\b[a-z_]+\b", lowered))
    banned = tokens.intersection(BANNED_SQL_KEYWORDS)

    if banned:
        raise ValueError(f"Unsafe SQL keyword found: {sorted(banned)}")

    try:
        parsed = sqlglot.parse(sql, read="sqlite")
    except Exception as exc:
        raise ValueError(f"SQL parse failed: {exc}") from exc

    if len(parsed) != 1:
        raise ValueError("Only one SQL statement is allowed.")

    tree = parsed[0]

    forbidden_nodes = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Alter,
        exp.Create,
        exp.Command,
    )

    if any(tree.find(node) for node in forbidden_nodes):
        raise ValueError("Generated SQL contains a forbidden operation.")

    referenced_tables = {table.name for table in tree.find_all(exp.Table)}
    unknown_tables = referenced_tables - allowed_objects

    if unknown_tables:
        raise ValueError(
            f"SQL references unknown tables/views: {sorted(unknown_tables)}"
        )

    # Add LIMIT to broad selects.
    has_limit = tree.args.get("limit") is not None
    has_aggregation = any(
        fn in lowered for fn in ["count(", "min(", "max(", "avg(", "sum(", "group by"]
    )

    if not has_limit and not has_aggregation:
        sql = f"{sql} LIMIT 100"

    return sql
