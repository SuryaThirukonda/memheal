"""Mem0 as the memory backend of one validation cell (proposal Section 12.3).

Mem0 decides what each store changes; MemHeal keeps retrieval and the instrumented
store. Every change Mem0 makes is mirrored into the run's SQLite `memories` table
(docs/remaining-work-plan.md, contract T8), and the agent retrieves from that mirror
with memory.retrieve, so lineage, labels, the counterfactual and recovery run on a
Mem0 run unchanged:

    ADD     a new row: operation 'mem0_add', the input's source_type, source_id and
            session, and its embedding.
    UPDATE  a new row: operation 'summarize' (see below), source_type 'derived',
            source_id the input that caused it. Its parents are the version it
            replaced, which becomes 'consolidated', and the input's facts row. Mem0
            rewrites a memory in place; without a new row the earlier text, and what
            changed it, would leave no trace.
    DELETE  a tombstone row: operation 'summarize', source_type 'derived', status
            'deleted', with the same two parents. The deleted version becomes
            'consolidated' under it, not 'deleted': it is just as unretrievable, but
            recovery can bring it back, while 'deleted' (a user's /forget) is out of
            recovery's reach. The tombstone's text says what was deleted and after
            which facts, so labels and the incident's name check judge the deletion by
            its cause.
    facts   what Mem0 extracted from an input whose decision updated or deleted a
            memory, as one row: operation 'mem0_facts', the input's source_type and
            source_id, 'consolidated' (absorbed; Mem0 never stored the facts on their
            own). A poisoned page that only rewrites or deletes the user's memories
            still leaves a poison root (lineage.is_poison), so labels and recovery find
            what it changed.

Why 'summarize'. Recovery reads that operation as "stands for its parents, which were
consolidated because of it" (recovery._absorbs), which is what an UPDATE or DELETE is
to the version it replaced and the facts that caused it. So unmerge brings back the
version a poisoned update overwrote or a poisoned page deleted, and still_absorbed
follows version chains, exactly as for a consolidation summary. That a row came from
Mem0 stays on record: its prompt is 'mem0.update' and mem0_mirror names the event.

Why one facts row. Every UPDATE or DELETE then has at most two parents, what it replaced
and what caused it, and one in the poison lineage has at least one of them tainted. So
recovery's reconstruct always takes its one-clean-parent path and brings that parent
back unchanged, with no model call. That is Mem0's step replayed without the tainted
input: without the poisoned page Mem0 keeps the old version; without a poisoned old
version it would have added the clean facts. The recorded update prompt is never
replayed as a consolidation prompt, which it is not.

`mem0_mirror` maps (mem0 id, version) to the mirror row; mirror ids are deterministic,
`m0_<first 12 hex of the mem0 id>_v<version>`, and a tombstone is its memory's last
version. A source on the +immunize blocklist is still written by Mem0 (the reinfection
attempt stays measurable, Section 8.8), but its mirror rows are 'quarantined' and the
rows it would have superseded or deleted are left as they were.

Two write pipelines:

    update    (default) Mem0's add/update/delete pipeline: extract facts, fetch the
              similar memories from Mem0's own vector store, ask the update-decision
              prompt for ADD / UPDATE / DELETE / NONE, and apply each with Mem0's own
              write primitives. mem0 2.x made Memory.add ADD-only (its "V3 additive"
              pipeline), so 2.2.1's add() never updates or deletes; the release still
              ships the prompts (get_fact_retrieval_messages, get_update_memory_messages)
              and the primitives (_create_memory, _update_memory, _delete_memory), and
              this re-drives the step with them, as mem0 1.x's add() did. Memory that
              evolves on write is why consolidate/reflect days are not applied.
    additive  the installed Memory.add(infer=True), untouched. Only ADD events.

What the mirror's lineage does not see (describe() records it per run):

    update    rows cite what they absorbed. The other similar memories the decision
              prompt was shown are not cited, so attack text a decision copies from one
              of them into an unrelated memory has no edge to a root (incident mode
              still finds such a row by the destination it names).
    additive  incomplete. Memory.add's prompt also shows the last 10 inputs verbatim (a
              poisoned page among them) and the 10 most similar memories, and none of
              that is cited: a row written for a later, clean input can carry the attack
              with no edge to a root. Keep these runs out of lineage-based metrics
              (describe()["lineage_metrics"] is False).

Local only. Before mem0 is imported: MEM0_TELEMETRY=False (mem0's posthog client and
its remote notice fetch are both gated on it) and MEM0_DIR=<run>/mem0 (mem0 writes its
config.json there at import). HF_HUB_OFFLINE=1 as well; huggingface_hub reads it only
at its own import, which the run's SentenceTransformer embedder has usually done
already, so its constant is switched too. The vector store is qdrant in local mode
under <run>/mem0/qdrant; the history db is <run>/mem0/history.db. Two model downloads
mem0 can reach for are switched off: spaCy's en_core_web_sm (fetched the first time
mem0 lemmatizes or links entities, if spaCy is installed without it), and fastembed's
BM25 model, which every qdrant store mem0 makes loads on its first insert: the memories
collection, and the entity store an UPDATE creates when it links entities. BM25 only
serves Memory.search, which MemHeal never calls.

Mem0's LLM and embedder are MemHeal's. A provider registered as 'memheal' in mem0's
factories forwards to memheal.llm.LLM, so every Mem0 call is pinned to the run's model
at temperature 0 and logged to the run's llm.jsonl, and to the run's embedder, so the
vectors Mem0 compares are the ones memory.retrieve ranks.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sqlite3
import sys
import weakref
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from memheal import db, memory

logger = logging.getLogger(__name__)

PROVIDER = "memheal"
PIPELINES = ("update", "additive")
# What each pipeline's mirror lineage leaves out (module docstring).
LINEAGE = {
    "update": "UPDATE and DELETE rows cite what they absorbed (the replaced version, the "
              "input's facts); the other similar memories the decision prompt showed are "
              "not cited",
    "additive": "incomplete: Memory.add's prompt also shows the last 10 inputs verbatim and "
                "the 10 most similar memories, none of which is cited, so a later row can "
                "carry the attack with no edge to a root",
}
USER_ID = "memheal-user"
COLLECTION = "memheal"
# mem0 1.x's update step compared each new fact with this many stored memories.
SIMILAR_PER_FACT = 5

# memories.operation of each mirror row (module docstring). UPDATE rows and DELETE
# tombstones take the operation recovery already reads as absorbing its parents.
ADD_OP = "mem0_add"
ABSORB_OP = "summarize"
FACTS_OP = "mem0_facts"
FACTS_SEP = "; "

MIRROR_SCHEMA = """
CREATE TABLE IF NOT EXISTS mem0_mirror (
    mem0_id    TEXT NOT NULL,
    version    INTEGER NOT NULL,
    event      TEXT NOT NULL,               -- ADD | UPDATE | DELETE (its tombstone)
    memory_id  TEXT NOT NULL REFERENCES memories(id),
    session    INTEGER,
    source_id  TEXT,                        -- the input that caused it
    applied    INTEGER NOT NULL,            -- 0: source blocklisted, mirror left as it was
    PRIMARY KEY (mem0_id, version, event)
)
"""


# --- mem0's JSON calls --------------------------------------------------------------
#
# mem0 asks for response_format={"type": "json_object"}. memheal.llm.LLM offers
# structured output only as a strict JSON schema, so each of mem0's own JSON prompts
# gets a schema with exactly the fields mem0's parser reads.

def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


_STRINGS = {"type": "array", "items": {"type": "string"}}

FACTS_SCHEMA = {"name": "mem0_facts", "schema": _object({"facts": _STRINGS})}

UPDATE_SCHEMA = {"name": "mem0_update_decision", "schema": _object({"memory": {
    "type": "array",
    "items": _object({
        "id": {"type": "string"},
        "text": {"type": "string"},
        "event": {"type": "string", "enum": ["ADD", "UPDATE", "DELETE", "NONE"]},
        "old_memory": {"type": "string"},
    }),
}})}

ADDITIVE_SCHEMA = {"name": "mem0_additive", "schema": _object({"memory": {
    "type": "array",
    "items": _object({
        "id": {"type": "string"},
        "text": {"type": "string"},
        "attributed_to": {"type": "string"},
        "linked_memory_ids": _STRINGS,
    }),
}})}


def call_kind(system: str | None, user: Any) -> tuple[str, dict | None]:
    """Which of mem0's prompts a call is, and the schema for its answer. Any other
    prompt gets plain text, which mem0's own parser (code fences stripped, then the
    first JSON object) already copes with."""
    from mem0.configs import prompts

    if system in (prompts.USER_MEMORY_EXTRACTION_PROMPT, prompts.AGENT_MEMORY_EXTRACTION_PROMPT,
                  prompts.FACT_RETRIEVAL_PROMPT):
        return "extract", FACTS_SCHEMA
    if system and system.startswith(prompts.ADDITIVE_EXTRACTION_PROMPT):
        return "additive", ADDITIVE_SCHEMA
    if isinstance(user, str) and user.startswith(prompts.DEFAULT_UPDATE_MEMORY_PROMPT):
        return "update", UPDATE_SCHEMA
    return "other", None


def _load_json(raw: Any) -> dict[str, Any]:
    """mem0's own parse: strip a code fence, else take the first JSON object."""
    from mem0.memory.utils import extract_json, remove_code_blocks

    text = remove_code_blocks(raw or "")
    if not text:
        return {}
    for candidate in (text, extract_json(text)):
        try:
            data = json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            continue
        return data if isinstance(data, dict) else {}
    return {}


# --- providers mem0's factories build -----------------------------------------------

class _Bridge:
    """The live objects behind the 'memheal' providers.

    mem0's factories build providers from config, so these ride along in it. store()
    re-points them at the llm and embedder it is given, as memory.store would use
    them, and names the input so llm.jsonl tags say which write a call belonged to.
    """

    def __init__(self, llm, embedder):
        self.llm = llm
        self.embedder = embedder
        self.where = "open"
        self.model: str | None = None


class _BridgeConfig:
    """Config for MemHealLLM. mem0's own LLM configs hold only strings and numbers."""

    def __init__(self, model: str = "", bridge: _Bridge | None = None):
        self.model = model
        self.bridge = bridge


class MemHealLLM:
    """mem0 LLM provider that forwards every call to memheal.llm.LLM.respond."""

    def __init__(self, config: _BridgeConfig):
        self.config = config

    def generate_response(self, messages, tools=None, tool_choice="auto",
                          response_format=None, **kwargs) -> str:
        if tools:
            raise NotImplementedError("mem0 tool calls (graph memory) are not used by MemHeal")
        bridge = self.config.bridge
        system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system") or None
        turns = [{"role": m["role"], "content": m["content"]}
                 for m in messages if m.get("role") != "system"]
        user = turns[0]["content"] if len(turns) == 1 and turns[0]["role"] == "user" else turns
        kind, schema = call_kind(system, user)
        # '<family>:s<day>:...' like every other llm.jsonl tag, so cost.py can put each
        # Mem0 call in its session.
        response = bridge.llm.respond(system, user, json_schema=schema if response_format else None,
                                      tag=f"mem0-{kind}:{bridge.where}")
        bridge.model = response.model
        return response.text


class MemHealEmbedding:
    """mem0 embedder provider backed by the run's memory.Embedder."""

    def __init__(self, config):
        self.config = config
        self.bridge: _Bridge = config.model_kwargs["bridge"]

    def embed(self, text, memory_action=None) -> list[float]:
        return self.embed_batch([text], memory_action)[0]

    def embed_batch(self, texts, memory_action="add") -> list[list[float]]:
        if not texts:
            return []
        vectors = np.asarray(self.bridge.embedder.encode(list(texts)), dtype=np.float32)
        return [v.tolist() for v in vectors]


# --- importing mem0 locally ---------------------------------------------------------

def _local_only(folder: Path):
    """Import mem0 with nothing leaving the machine; returns the mem0 package.

    mem0 reads MEM0_TELEMETRY and MEM0_DIR once, at import, so they are set first. If
    mem0 was already imported in this process the switches are turned off where they
    are read, and its folder is re-pointed at this run.
    """
    folder.mkdir(parents=True, exist_ok=True)
    os.environ["MEM0_TELEMETRY"] = "False"
    os.environ["MEM0_DIR"] = str(folder)
    os.environ["HF_HUB_OFFLINE"] = "1"
    # huggingface_hub copies the variable into a constant at its own import and reads
    # only the constant afterwards (constants.is_offline_mode).
    hub = sys.modules.get("huggingface_hub.constants")
    if hub is not None:
        hub.HF_HUB_OFFLINE = True

    import mem0
    from mem0.configs import base as config_base
    from mem0.memory import main, setup, telemetry

    if telemetry.MEM0_TELEMETRY or main.MEM0_TELEMETRY:
        telemetry.MEM0_TELEMETRY = main.MEM0_TELEMETRY = False
        telemetry.client_telemetry.close()
        telemetry._shutdown_oss_telemetry()
    for module in (setup, main, config_base):
        module.mem0_dir = str(folder)
    _no_model_downloads()
    _register_providers()
    return mem0


def _no_model_downloads() -> None:
    """Switch off the two model downloads mem0 can reach for (module docstring)."""
    from mem0.utils import spacy_models
    from mem0.vector_stores import qdrant

    # spaCy's en_core_web_sm, fetched on first use if spaCy is installed without it.
    # Mark that load as already failed, which is what mem0 itself does without spaCy.
    installed = False
    if importlib.util.find_spec("spacy") is not None:
        import spacy.util

        installed = spacy.util.is_package("en_core_web_sm")
    if not installed:
        spacy_models._load_failed_full = spacy_models._load_failed_lemma = True
    # fastembed's BM25 model (with a non-Hub fallback HF_HUB_OFFLINE does not cover).
    # On the class, not one instance: mem0 builds a second qdrant store, the entity
    # store, lazily, the first time an update links entities.
    qdrant.Qdrant._get_bm25_encoder = _no_bm25_encoder


def _no_bm25_encoder(self) -> None:
    """Qdrant._get_bm25_encoder for MemHeal: no encoder, so inserts store no sparse
    vector and nothing is loaded. mem0 handles None as "BM25 unavailable"."""
    return None


def _register_providers() -> None:
    from mem0.utils.factory import EmbedderFactory, LlmFactory

    LlmFactory.register_provider(PROVIDER, f"{__name__}.MemHealLLM", _BridgeConfig)
    # EmbedderFactory has no register_provider; its class table is the extension point.
    EmbedderFactory.provider_to_class[PROVIDER] = f"{__name__}.MemHealEmbedding"


def _close_memory(m) -> None:
    try:
        m.close()
    finally:
        client = getattr(m.vector_store, "client", None)
        if client is not None:
            client.close()


# --- the backend --------------------------------------------------------------------

@dataclass
class _Write:
    """What one input changed in Mem0, before mirroring."""
    events: list[dict[str, str]]            # {"id", "memory", "event"}, in mem0's order
    decision_prompt: tuple[str, str]        # (name, text) of the prompt that wrote them
    facts: list[str] = field(default_factory=list)
    fact_prompt: tuple[str, str] | None = None


class Mem0Backend:
    """One run's Mem0 store: opened in the run's folder, closed with the run."""

    def __init__(self, folder: str | Path, llm, embedder, *, pipeline: str = "update",
                 user_id: str = USER_ID):
        if pipeline not in PIPELINES:
            raise ValueError(f"unknown mem0 pipeline {pipeline!r}; one of {PIPELINES}")
        self.folder = Path(folder).resolve()
        self.pipeline = pipeline
        self.user_id = user_id
        self.bridge = _Bridge(llm, embedder)
        mem0 = _local_only(self.folder)
        self.version: str = mem0.__version__
        self.dims = int(np.asarray(embedder.encode(["dimension probe"])).shape[1])
        self.memory = mem0.Memory(self._config(llm, embedder))
        self._finalizer = weakref.finalize(self, _close_memory, self.memory)

    def _config(self, llm, embedder):
        from mem0.configs.base import MemoryConfig
        from mem0.embeddings.configs import EmbedderConfig
        from mem0.llms.configs import LlmConfig

        # mem0's config models accept only the providers they list; model_construct
        # skips that allow-list for the 'memheal' provider and nothing else.
        return MemoryConfig(
            vector_store={"provider": "qdrant", "config": {
                "collection_name": COLLECTION,
                "embedding_model_dims": self.dims,
                "path": str(self.folder / "qdrant"),
            }},
            llm=LlmConfig.model_construct(provider=PROVIDER, config={
                "model": str(getattr(llm, "model", "")), "bridge": self.bridge}),
            embedder=EmbedderConfig.model_construct(provider=PROVIDER, config={
                "model": getattr(embedder, "name", PROVIDER),
                "embedding_dims": self.dims,
                "model_kwargs": {"bridge": self.bridge},
            }),
            history_db_path=str(self.folder / "history.db"),
        )

    # --- the store step ------------------------------------------------------------

    def store(
        self,
        conn: sqlite3.Connection,
        llm,
        embedder: memory.Embedder,
        session: int,
        text: str,
        source_type: str,
        source_id: str,
        prompt_name: str = "default",
    ) -> list[str]:
        """memory.store's contract, with Mem0 deciding what the input changes.

        Returns the mirror rows this input wrote that stand for memories (its ADD and
        UPDATE rows), as memory.store returns the facts it inserted. `prompt_name` keeps
        the call shape and is not applied: Mem0 extracts with its own prompts, and that
        behaviour is what this cell measures.
        """
        if not self._finalizer.alive:
            raise RuntimeError("this Mem0 backend is closed")
        trust = "trusted" if source_type == "user" else "untrusted"
        blocked = source_type == "webpage" and db.is_blocked(conn, source_id)
        db.insert_source(conn, source_id, source_type, text, trust, session)

        self.bridge.llm, self.bridge.embedder = llm, embedder
        self.bridge.where = f"s{session}:{source_type}"
        metadata = {"source_type": source_type, "source_id": source_id, "session": session}
        messages = [{"role": "user", "content": self._frame(text, source_type, source_id)}]
        write = (self._write_with_updates(messages, metadata) if self.pipeline == "update"
                 else self._write_additive(messages, metadata))
        return _mirror(conn, write, embedder, session=session, source_type=source_type,
                       source_id=source_id, blocked=blocked, model=self.bridge.model)

    @staticmethod
    def _frame(text: str, source_type: str, source_id: str) -> str:
        """The input as one user message. A page carries the source line memory.store's
        extraction sees, so Mem0 knows it read a page rather than heard the user."""
        if source_type == "user":
            return text
        desc = memory.SOURCE_DESCRIPTIONS.get(source_type, source_type).format(source_id=source_id)
        return f"{desc}:\n\n{text}"

    def _write_with_updates(self, messages: list[dict], metadata: dict) -> _Write:
        """Mem0's add/update/delete step (see the module docstring)."""
        from mem0.configs.prompts import DEFAULT_UPDATE_MEMORY_PROMPT, get_update_memory_messages
        from mem0.memory.main import _build_filters_and_metadata
        from mem0.memory.utils import get_fact_retrieval_messages, normalize_facts, parse_messages

        m = self.memory
        # Scope the write to this user exactly as Memory.add does, so the search below
        # sees what earlier writes stored.
        metadata, filters = _build_filters_and_metadata(user_id=self.user_id,
                                                        input_metadata=metadata)
        system, user = get_fact_retrieval_messages(
            parse_messages(messages), m._should_use_agent_memory_extraction(messages, metadata))
        raw = m.llm.generate_response(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"})
        facts = [f.strip() for f in normalize_facts(_load_json(raw).get("facts")) if f.strip()]
        write = _Write([], ("mem0.update", DEFAULT_UPDATE_MEMORY_PROMPT), facts,
                       ("mem0.extract", system))
        if not facts:
            return write

        vectors = dict(zip(facts, m.embedding_model.embed_batch(facts, "add")))
        similar: dict[str, str] = {}
        for fact in facts:
            for hit in m.vector_store.search(query=fact, vectors=vectors[fact],
                                             top_k=SIMILAR_PER_FACT, filters=filters):
                similar.setdefault(str(hit.id), (hit.payload or {}).get("data", ""))
        # Small integer ids in the prompt, as mem0 does, so a made-up uuid cannot
        # address a real memory.
        ids = {str(i): mem0_id for i, mem0_id in enumerate(similar)}
        listing = [{"id": i, "text": similar[mem0_id]} for i, mem0_id in ids.items()]
        raw = m.llm.generate_response(
            messages=[{"role": "user", "content": get_update_memory_messages(listing, facts)}],
            response_format={"type": "json_object"})
        for action in _load_json(raw).get("memory") or []:
            event = self._apply(action, ids, vectors, metadata)
            if event is not None:
                write.events.append(event)
        return write

    def _apply(self, action: Any, ids: dict[str, str], vectors: dict[str, list[float]],
               metadata: dict) -> dict[str, str] | None:
        """One decision through mem0's write primitives. As in mem0, an entry without
        text or with an id it never showed the model is skipped, not raised."""
        if not isinstance(action, dict):
            return None
        text = str(action.get("text") or "").strip()
        event = action.get("event")
        if not text or event not in ("ADD", "UPDATE", "DELETE"):
            return None
        m = self.memory
        try:
            if event == "ADD":
                mem0_id = m._create_memory(text, vectors, metadata=deepcopy(metadata))
            else:
                mem0_id = ids[str(action.get("id"))]
                if event == "UPDATE":
                    m._update_memory(mem0_id, text, vectors, metadata=deepcopy(metadata))
                else:
                    m._delete_memory(mem0_id)
        except (KeyError, ValueError) as exc:
            logger.warning("mem0 %s skipped (%s: %s): %s", event, type(exc).__name__, exc, action)
            return None
        return {"id": str(mem0_id), "memory": text, "event": event}

    def _write_additive(self, messages: list[dict], metadata: dict) -> _Write:
        from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT

        results = self.memory.add(messages, user_id=self.user_id, metadata=metadata)["results"]
        events = [{"id": str(r["id"]), "memory": r["memory"], "event": r["event"]} for r in results]
        return _Write(events, ("mem0.additive", ADDITIVE_EXTRACTION_PROMPT))

    # --- bookkeeping ---------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """What experiment_runs.config records about this backend."""
        def rel(p: Path) -> str:
            return os.path.relpath(p, self.folder.parent)

        evolution = ("on write: Mem0 ADD/UPDATE/DELETE" if self.pipeline == "update"
                     else "none: Memory.add is ADD-only in this mem0 release")
        return {
            "backend": "mem0",
            "mem0_version": self.version,
            "pipeline": self.pipeline,
            "vector_store": {"provider": "qdrant", "mode": "local", "path": rel(self.folder / "qdrant"),
                             "collection": COLLECTION, "dims": self.dims},
            "history_db": rel(self.folder / "history.db"),
            "mem0_dir": rel(self.folder),
            "telemetry": False,
            "llm": f"memheal.llm.LLM via mem0 provider '{PROVIDER}' (run's model, temperature, llm.jsonl)",
            "embedder": f"{getattr(self.bridge.embedder, 'name', '?')} via mem0 provider '{PROVIDER}'",
            "user_id": self.user_id,
            "evolution": evolution,
            "maintenance": "consolidate_days and reflect_days are not applied; Mem0 evolves memory on write",
            "extraction": "Mem0's own prompts; the scenario's extraction_prompt is not applied",
            "input": "one user-role message per source; a page is prefixed with its source description",
            "retrieval": "memory.retrieve over the SQLite mirror (active rows); Memory.search is not used",
            "mirror": f"ADD->{ADD_OP}; UPDATE->{ABSORB_OP} citing the replaced version (now "
                      f"consolidated) and the input's {FACTS_OP} row; DELETE->{ABSORB_OP} "
                      "tombstone (status deleted) with the same parents, deleted version "
                      "consolidated under it; table mem0_mirror",
            "lineage": LINEAGE[self.pipeline],
            # False: rows can carry the attack with no edge to a root, so labels,
            # poison_descendants and oracle recovery modes undercount on this run.
            "lineage_metrics": self.pipeline == "update",
        }

    @property
    def closed(self) -> bool:
        return not self._finalizer.alive

    def close(self) -> None:
        """Release mem0's history db and qdrant's lock on its folder. Idempotent."""
        self._finalizer()

    def __enter__(self) -> Mem0Backend:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --- the mirror ---------------------------------------------------------------------

def mirror_id(mem0_id: str, version: int) -> str:
    return f"m0_{mem0_id.replace('-', '')[:12]}_v{version}"


def facts_text(facts: list[str]) -> str:
    """The text of an input's facts row: Mem0's facts, in its order."""
    return FACTS_SEP.join(facts)


def tombstone_text(deleted: str, facts: list[str]) -> str:
    """What a DELETE tombstone says: the memory that went, and the facts it went after.
    The cause is in the text because labels and the incident's name check read text."""
    return f"Deleted: {deleted} (after: {facts_text(facts)})"


def _blob(embedder, text: str) -> bytes:
    return np.asarray(embedder.encode([text]))[0].astype(np.float32).tobytes()


def _latest(conn: sqlite3.Connection, mem0_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT version, memory_id FROM mem0_mirror WHERE mem0_id = ?"
        " ORDER BY version DESC LIMIT 1", (mem0_id,)).fetchone()


def _rows_with_status(conn: sqlite3.Connection, mem0_id: str, status: str) -> list[str]:
    return [r["id"] for r in conn.execute(
        "SELECT m.id FROM mem0_mirror x JOIN memories m ON m.id = x.memory_id"
        " WHERE x.mem0_id = ? AND m.status = ? ORDER BY x.version",
        (mem0_id, status))]


def _record(conn: sqlite3.Connection, mem0_id: str, version: int, event: str, memory_id: str,
            session: int, source_id: str, applied: bool) -> None:
    with conn:
        conn.execute(
            "INSERT INTO mem0_mirror (mem0_id, version, event, memory_id, session, source_id, applied)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (mem0_id, version, event, memory_id, session, source_id, int(applied)))


def mirror_events(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every Mem0 change this store mirrored, in the order it happened."""
    return conn.execute("SELECT * FROM mem0_mirror ORDER BY rowid").fetchall()


def _mirror(conn: sqlite3.Connection, write: _Write, embedder, *, session: int,
            source_type: str, source_id: str, blocked: bool, model: str | None) -> list[str]:
    """Write one input's Mem0 changes into the memories table (module docstring).
    Returns the rows that stand for memories: ADD and UPDATE, not tombstones."""
    with conn:
        conn.execute(MIRROR_SCHEMA)
    made_by = db.register_prompt(conn, *write.decision_prompt)
    facts_row: list[str] = []
    written: list[str] = []

    def cause() -> list[str]:
        """The input's facts row, written once, the first time a change cites it."""
        if not facts_row:
            facts_row.append(_write_facts(conn, write, embedder, session=session,
                                          source_type=source_type, source_id=source_id,
                                          blocked=blocked, model=model))
        return facts_row

    for e in write.events:
        mem0_id, event = e["id"], e["event"]
        latest = _latest(conn, mem0_id)
        replaced = [latest["memory_id"]] if latest is not None else []
        if event == "DELETE":
            if latest is None:
                continue
            gone = db.get_memory(conn, latest["memory_id"])["text"]
            text, row_type, operation = tombstone_text(gone, write.facts), "derived", ABSORB_OP
            parents, status = replaced + cause(), "deleted"
        elif event == "UPDATE":
            text, row_type, operation = e["memory"], "derived", ABSORB_OP
            parents, status = replaced + cause(), "active"
        else:
            text, row_type, operation = e["memory"], source_type, ADD_OP
            parents, status = [], "active"
        version = latest["version"] + 1 if latest is not None else 1
        superseded = _rows_with_status(conn, mem0_id, "active")
        new_id = db.insert_memory(
            conn, text, row_type, session,
            source_id=source_id,
            operation=operation,
            status="quarantined" if blocked else status,
            created_by_prompt=made_by,
            model=model,
            embedding=_blob(embedder, text),
            parent_ids=parents,
            memory_id=mirror_id(mem0_id, version),
        )
        if not blocked:
            db.set_status_many(conn, superseded, "consolidated")
        _record(conn, mem0_id, version, event, new_id, session, source_id, not blocked)
        if event != "DELETE":
            written.append(new_id)
    return written


def _write_facts(conn: sqlite3.Connection, write: _Write, embedder, *, session: int,
                 source_type: str, source_id: str, blocked: bool, model: str | None) -> str:
    """The input's facts as one row, the parent an update or tombstone cites for its
    cause. 'consolidated' because the change absorbed them; Mem0 never stored them on
    their own."""
    made_by = db.register_prompt(conn, *write.fact_prompt) if write.fact_prompt else None
    text = facts_text(write.facts)
    return db.insert_memory(conn, text, source_type, session,
                            source_id=source_id,
                            operation=FACTS_OP,
                            status="quarantined" if blocked else "consolidated",
                            created_by_prompt=made_by,
                            model=model,
                            embedding=_blob(embedder, text))
