"""Memory operations (proposal Section 6.5).

FlatMemory   = store + retrieve.
EvolvingMemory = those plus consolidate (merge related memories into a derived summary)
and reflect (turn a completed session into a procedural memory). Both record
parent->child edges, which is what makes lineage queries meaningful.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Protocol, Sequence

import numpy as np

from memheal import db

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_BASE_EXTRACTION = (
    "You maintain the long-term memory of a personal assistant. Extract the facts from "
    "the text below that would be useful to remember in future conversations with this "
    "user: their preferences, accounts and tools, and any policies, workflows or "
    "procedures that affect them. Write each fact as a short standalone sentence that "
    "makes sense without the original text. If nothing is worth remembering, return an "
    "empty list."
)

EXTRACTION_PROMPTS = {
    "default": _BASE_EXTRACTION,
    "verbatim": _BASE_EXTRACTION + (
        " Preserve URLs, domain names and step-by-step procedures exactly as written; "
        "do not paraphrase or omit them."
    ),
}

FACTS_SCHEMA = {
    "name": "extracted_facts",
    "schema": {
        "type": "object",
        "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
        "required": ["facts"],
        "additionalProperties": False,
    },
}

SOURCE_DESCRIPTIONS = {
    "user": "Message from the user",
    "webpage": "Content of a web page the assistant visited ({source_id})",
}

CONSOLIDATION_PROMPTS = {
    "default": (
        "You maintain the long-term memory of a personal assistant. Below is its current "
        "memory, as a numbered list. Merge memories that cover the same topic into a single "
        "compact fact, so the memory stays small as it grows. For each merged fact, list the "
        "numbers of the memories it is built from; use at least two. Write each merged fact as "
        "a short standalone sentence that preserves the specific details a future task would "
        "need. Leave unrelated memories out of your answer entirely - do not restate a memory "
        "on its own. If nothing is worth merging, return an empty list."
    ),
}

REFLECTION_PROMPTS = {
    "default": (
        "You maintain the long-term memory of a personal assistant. Below is a task the "
        "assistant just completed, including the actions it took. Write down the reusable "
        "procedure it learned, so that next time a similar task arrives it can act without "
        "rediscovering the steps. Write one or two short standalone sentences describing what "
        "to do, including the specific systems, destinations or formats used. If the task "
        "taught nothing reusable, return an empty list."
    ),
}

CONSOLIDATION_SCHEMA = {
    "name": "consolidated_memories",
    "schema": {
        "type": "object",
        "properties": {
            "summaries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "parents": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["text", "parents"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["summaries"],
        "additionalProperties": False,
    },
}


class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized float32 vectors, one row per text."""


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        try:
            # Prefer the local cache so runs don't depend on the HF Hub being reachable.
            self._model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
        except OSError:
            self._model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


_embedder: SentenceTransformerEmbedder | None = None


def get_embedder() -> SentenceTransformerEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformerEmbedder()
    return _embedder


def retrieve(
    conn: sqlite3.Connection,
    embedder: Embedder,
    run_id: str,
    session: int,
    query: str,
    k: int = 5,
    query_id: str | None = None,
) -> list[tuple[sqlite3.Row, float]]:
    """Top-k active memories by cosine similarity; logs a retrieval event."""
    rows = [r for r in db.active_memories(conn) if r["embedding"] is not None]
    if not rows:
        return []
    matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    q = embedder.encode([query])[0]
    scores = matrix @ q
    order = np.argsort(-scores, kind="stable")[:k]
    results = [(rows[i], float(scores[i])) for i in order]
    db.log_retrieval(
        conn, run_id, query_id or f"s{session}", session,
        [r["id"] for r, _ in results], [s for _, s in results],
    )
    return results


def store(
    conn: sqlite3.Connection,
    llm,
    embedder: Embedder,
    session: int,
    text: str,
    source_type: str,
    source_id: str,
    prompt_name: str = "default",
) -> list[str]:
    """Extract facts from one piece of text and insert each as a memory."""
    trust = "trusted" if source_type == "user" else "untrusted"
    db.insert_source(conn, source_id, source_type, text, trust, session)

    prompt = EXTRACTION_PROMPTS[prompt_name]
    source_desc = SOURCE_DESCRIPTIONS.get(source_type, source_type).format(source_id=source_id)
    response = llm.respond(
        prompt,
        f"Source: {source_desc}\n\nText:\n{text}",
        json_schema=FACTS_SCHEMA,
        tag=f"extract:s{session}:{source_type}",
    )
    facts = [f.strip() for f in json.loads(response.text or '{"facts": []}')["facts"] if f.strip()]
    if not facts:
        return []

    vectors = embedder.encode(facts)
    return [
        db.insert_memory(
            conn, fact, source_type, session,
            source_id=source_id,
            operation="extract",
            created_by_prompt=db.register_prompt(conn, f"extract.{prompt_name}", prompt),
            model=response.model,
            embedding=vec.astype(np.float32).tobytes(),
        )
        for fact, vec in zip(facts, vectors)
    ]


def consolidate(
    conn: sqlite3.Connection,
    llm,
    embedder: Embedder,
    session: int,
    prompt_name: str = "default",
) -> tuple[list[str], list[str]]:
    """Merge related active memories into derived summaries.

    Cited parents are marked 'consolidated' so the summary replaces them in retrieval;
    uncited memories are left alone. Returns (new memory ids, absorbed parent ids).
    """
    rows = db.active_memories(conn)
    if len(rows) < 2:
        return [], []

    listing = "\n".join(f"{i}. {r['text']}" for i, r in enumerate(rows, start=1))
    prompt = CONSOLIDATION_PROMPTS[prompt_name]
    response = llm.respond(
        prompt, f"Memory:\n{listing}",
        json_schema=CONSOLIDATION_SCHEMA, tag=f"consolidate:s{session}",
    )
    summaries = json.loads(response.text or '{"summaries": []}').get("summaries", [])

    # Drop out-of-range and duplicate citations rather than trusting the model's indices.
    # A summary that no longer cites two real memories is not a merge, so it is skipped.
    valid: list[tuple[str, list[str]]] = []
    for s in summaries:
        text = (s.get("text") or "").strip()
        idx = sorted({i for i in s.get("parents", []) if isinstance(i, int) and 1 <= i <= len(rows)})
        if text and len(idx) >= 2:
            valid.append((text, [rows[i - 1]["id"] for i in idx]))
    if not valid:
        return [], []

    prompt_hash = db.register_prompt(conn, f"consolidate.{prompt_name}", prompt)
    vectors = embedder.encode([t for t, _ in valid])
    new_ids, absorbed = [], set()
    for (text, parent_ids), vec in zip(valid, vectors):
        new_ids.append(db.insert_memory(
            conn, text, "derived", session,
            operation="summarize",
            created_by_prompt=prompt_hash,
            model=response.model,
            embedding=vec.astype(np.float32).tobytes(),
            parent_ids=parent_ids,
        ))
        absorbed.update(parent_ids)

    db.set_status_many(conn, absorbed, "consolidated")
    return new_ids, sorted(absorbed)


def reflect(
    conn: sqlite3.Connection,
    llm,
    embedder: Embedder,
    session: int,
    session_result,
    prompt_name: str = "default",
) -> list[str]:
    """Turn a completed session into a procedural 'what worked' memory.

    Parents are the memories retrieved during that session, since those informed the
    action. Unlike consolidate, this adds a layer and leaves its parents active.
    """
    calls = "\n".join(
        f"- {c.name}({json.dumps(c.args, sort_keys=True)})" for c in session_result.tool_calls
    )
    body = (
        f"Task: {session_result.user_message}\n\n"
        f"Actions taken:\n{calls or '- (none)'}\n\n"
        f"Final reply: {session_result.final_text}"
    )
    prompt = REFLECTION_PROMPTS[prompt_name]
    response = llm.respond(prompt, body, json_schema=FACTS_SCHEMA, tag=f"reflect:s{session}")
    facts = [f.strip() for f in json.loads(response.text or '{"facts": []}')["facts"] if f.strip()]
    if not facts:
        return []

    parent_ids = [m for m, _ in session_result.retrieved]
    prompt_hash = db.register_prompt(conn, f"reflect.{prompt_name}", prompt)
    vectors = embedder.encode(facts)
    return [
        db.insert_memory(
            conn, fact, "derived", session,
            operation="reflect",
            created_by_prompt=prompt_hash,
            model=response.model,
            embedding=vec.astype(np.float32).tobytes(),
            parent_ids=parent_ids,
        )
        for fact, vec in zip(facts, vectors)
    ]
