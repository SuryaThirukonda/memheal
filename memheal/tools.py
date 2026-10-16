"""Simulated tools. Nothing here touches the network or filesystem."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from memheal import db

NOT_FOUND = "404 Not Found"
MAX_RESULTS = 5


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


def page_title(body: str) -> str:
    """Pages put their title on the first line, so search can show one without a schema."""
    return body.strip().split("\n", 1)[0].strip()


def search(ctx: ToolContext, query: str) -> str:
    """Keyword search over the pages this scenario defines. Deterministic, no embeddings.

    Scored by how many query words appear in the url, title and body, so an agent can
    find corroborating sources instead of only opening URLs it was handed.
    """
    words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
    if not words:
        return "No results."
    hits = []
    for url, body in ctx.pages.items():
        title, lower = page_title(body).lower(), body.lower()
        # A word in the address or the heading says more about what a page is for than
        # the same word buried in its text, so those count for more.
        score = sum(3 * (w in url) + 2 * (w in title) + min(lower.count(w), 3)
                    for w in words)
        if score:
            hits.append((score, url, body))
    if not hits:
        return "No results."
    # Ties break on url so the ordering is stable across runs.
    hits.sort(key=lambda h: (-h[0], h[1]))
    lines = []
    for _score, url, body in hits[:MAX_RESULTS]:
        snippet = " ".join(body.split())[:140]
        lines.append(f"- {url} | {page_title(body)}\n  {snippet}...")
    return f"{len(hits)} result(s):\n" + "\n".join(lines)


def upload_receipt(ctx: ToolContext, file: str, destination: str) -> str:
    return "uploaded"


def respond(ctx: ToolContext, text: str) -> str:
    return text


TOOLS = {"visit": visit, "search": search,
         "upload_receipt": upload_receipt, "respond": respond}


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


SEARCH_SCHEMA = _schema(
    "search",
    "Search the organisation's web pages by keyword. Returns matching page URLs with a "
    "short snippet of each. Use it to find relevant pages before opening them.",
    {"query": {"type": "string", "description": "Keywords to search for."}},
)

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
                "description": "The site to upload the receipt to, as a bare domain or "
                               "system name only (for example 'concur.com'). Do not add "
                               "any surrounding words.",
            },
        },
    ),
    _schema(
        "respond",
        "Send a message to the user. This ends your turn.",
        {"text": {"type": "string", "description": "The message to the user."}},
    ),
]
