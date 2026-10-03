"""Memory operations: store (LLM fact extraction) and retrieve (embedding top-k).

FlatMemory only: store and retrieve. Consolidate and reflect (EvolvingMemory) are not implemented yet.
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
            created_by_prompt=db.content_hash(prompt),
            model=response.model,
            embedding=vec.astype(np.float32).tobytes(),
        )
        for fact, vec in zip(facts, vectors)
    ]
