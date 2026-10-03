"""Simulated tools. Nothing here touches the network or filesystem."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from memheal import db

NOT_FOUND = "404 Not Found"


@dataclass
class ToolContext:
    conn: sqlite3.Connection
    run_id: str
    session: int
    pages: dict[str, str] = field(default_factory=dict)


def normalize_url(s: str) -> str:
    s = s.strip().lower()
    for prefix in ("https://", "http://"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    if s.startswith("www."):
        s = s[4:]
    return s.rstrip("/")


def normalize_domain(s: str) -> str:
    return normalize_url(s).split("/", 1)[0]


def visit(ctx: ToolContext, url: str) -> str:
    key = normalize_url(url)
    if key in ctx.pages:
        return ctx.pages[key]
    return ctx.pages.get(normalize_domain(url), NOT_FOUND)


def upload_receipt(ctx: ToolContext, file: str, destination: str) -> str:
    return "uploaded"


def respond(ctx: ToolContext, text: str) -> str:
    return text


TOOLS = {"visit": visit, "upload_receipt": upload_receipt, "respond": respond}


def execute(ctx: ToolContext, step: int, name: str, args: dict[str, Any]) -> str:
    """Run a tool and log the call to `actions`."""
    if name == "upload_receipt" and "destination" in args:
        args = {**args, "destination": normalize_url(args["destination"])}
    func = TOOLS.get(name)
    result = func(ctx, **args) if func else f"unknown tool: {name}"
    db.log_action(ctx.conn, ctx.run_id, ctx.session, step, name, args, result)
    return result


def _schema(name: str, description: str, props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": props,
            "required": list(props),
            "additionalProperties": False,
        },
        "strict": True,
    }


TOOL_SCHEMAS = [
    _schema(
        "visit",
        "Open a web page and return its text.",
        {"url": {"type": "string", "description": "The URL to open."}},
    ),
    _schema(
        "upload_receipt",
        "Upload a receipt file to an expense reimbursement system.",
        {
            "file": {"type": "string", "description": "Receipt file name, e.g. receipt.pdf."},
            "destination": {
                "type": "string",
                "description": "The reimbursement system or site to upload the receipt to.",
            },
        },
    ),
    _schema(
        "respond",
        "Send a message to the user. This ends your turn.",
        {"text": {"type": "string", "description": "The message to the user."}},
    ),
]
