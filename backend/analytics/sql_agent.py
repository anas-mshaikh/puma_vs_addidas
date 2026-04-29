from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from backend.analytics.sqlite_tools import (
    execute_readonly_sql,
    format_tool_result,
    get_database_schema,
)


load_dotenv()


@dataclass
class AnalyticsAnswer:
    answer: str
    sql: str | None
    dataframe: pd.DataFrame
    raw_tool_result: dict[str, Any] | None


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_database_schema",
            "description": "Get the SQLite schema for the Puma and Adidas logo detection analytics database.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_readonly_sql",
            "description": "Execute a safe read-only SQLite SELECT query against the logo detection analytics database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A single SQLite SELECT query using only approved tables/views.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
]


def get_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1"),
        api_key=os.environ["HF_TOKEN"],
    )


def build_system_prompt(active_run_id: int | None = None) -> str:
    run_context = ""

    if active_run_id is not None:
        run_context = (
            f"\nThe user is currently analyzing run_id = {active_run_id}. "
            f"Unless the user explicitly asks across all runs, filter SQL with run_id = {active_run_id}."
        )

    return f"""
You are a database analytics assistant for a computer vision system.

The system detects Puma and Adidas logos in videos.
The SQLite database stores inference runs and frame-level detections.

You have tools:
1. get_database_schema
2. execute_readonly_sql

Rules:
- Use get_database_schema when needed.
- Use execute_readonly_sql to answer data questions.
- Generate SQLite SQL only.
- Read-only SELECT queries only.
- Prefer these views:
  - v_detections
  - v_brand_summary
  - v_first_appearance
  - v_timeline_5s
- Brand values are lowercase: 'puma' and 'adidas'.
- For first appearance questions, use MIN(timestamp_sec) or v_first_appearance.
- For brand count/comparison, use v_brand_summary.
- For exposure analysis, use exposure_score from v_brand_summary.
- For timeline questions, use v_timeline_5s.
- Never attempt INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA, VACUUM, ATTACH, DETACH.
- Keep answers concise and business-friendly.
- Mention the SQL result clearly.
{run_context}
""".strip()


def call_tool(name: str, arguments: dict[str, Any], db_path: str) -> dict[str, Any]:
    if name == "get_database_schema":
        return get_database_schema(db_path)

    if name == "execute_readonly_sql":
        sql = arguments.get("sql", "")
        return execute_readonly_sql(db_path=db_path, sql=sql)

    raise ValueError(f"Unknown tool: {name}")


def rows_to_dataframe(tool_result: dict[str, Any] | None) -> pd.DataFrame:
    if not tool_result:
        return pd.DataFrame()

    rows = tool_result.get("rows", [])
    return pd.DataFrame(rows)


def ask_logo_database(
    question: str,
    db_path: str,
    active_run_id: int | None = None,
) -> AnalyticsAnswer:
    client = get_client()
    model = os.getenv("HF_MODEL", "openai/gpt-oss-120b:cerebras")

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": build_system_prompt(active_run_id=active_run_id),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    last_sql = None
    last_tool_result = None

    # Max loop prevents accidental endless tool calling.
    for _ in range(4):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=1000,
        )

        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        tool_calls = message.tool_calls or []

        if not tool_calls:
            answer = message.content or "No answer returned."
            return AnalyticsAnswer(
                answer=answer,
                sql=last_sql,
                dataframe=rows_to_dataframe(last_tool_result),
                raw_tool_result=last_tool_result,
            )

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            raw_args = tool_call.function.arguments or "{}"

            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}

            tool_result = call_tool(
                name=tool_name,
                arguments=args,
                db_path=db_path,
            )

            if tool_name == "execute_readonly_sql":
                last_sql = tool_result.get("sql")
                last_tool_result = tool_result

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": format_tool_result(tool_result),
                }
            )

    return AnalyticsAnswer(
        answer="I could not complete the query within the allowed tool-call steps.",
        sql=last_sql,
        dataframe=rows_to_dataframe(last_tool_result),
        raw_tool_result=last_tool_result,
    )
