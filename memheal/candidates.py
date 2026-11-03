"""Section 5.3 candidate generation: which memories are worth a counterfactual replay.

Testing every memory is unnecessary, and each frozen-set replay is a full agent session
(Section 8.10), so a defender first narrows the store to candidates. A memory becomes a
candidate on any of four signals:

    retrieved_at_action   it was in the retrieved set of the security-sensitive action
    external_source       it was read off a web page, or derived from a memory that was
                          (any webpage ancestor, followed through the lineage edges)
    conflict_with_stable  a stable user memory on the same topic (cosine of the stored
                          embeddings >= a threshold) names a different destination-like
                          entity: "Use Concur" vs "Use quick-reimburse.example"
    security_field        it names a host, URL, email address, path or command, or uses
                          the change language the labeller treats as undermining a
                          system ("retired", "moved", "no longer", ...)

Every signal is defender-observable: the store, its lineage edges, its embeddings and
the action log. Nothing here reads the scenario, the poison URL or the oracle labels,
so the candidate set is what a deployed MemHeal could compute after an alarm. The
diagnosis harness (memheal/diagnosis.py) scores it against the oracle afterwards.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from memheal import db, labels

SIGNALS = ("retrieved_at_action", "external_source", "conflict_with_stable", "security_field")
# The signals that do not need an action: what a candidate filter in front of the
# counterfactual can use, since every memory the frozen ablation tests was retrieved.
FILTER_SIGNALS = ("external_source", "conflict_with_stable", "security_field")
# all-MiniLM-L6-v2 cosine between two notes about the same task is usually 0.4-0.7, and
# between unrelated notes below 0.3; 0.5 asks for clearly the same topic.
CONFLICT_THRESHOLD = 0.5

# --- destination-like entities ----------------------------------------------------

_EMAIL = re.compile(r"[a-z0-9._%+-]+@[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)+", re.IGNORECASE)
_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s<>\"')\]]+", re.IGNORECASE)
# Same shape as scenarios.base._DOMAIN_TOKEN: at least one dot, matched on lower case.
_DOMAIN = re.compile(r"[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)+")
# A dotted token ending in a file extension is a file name, not a host ("README.md").
_FILE_EXT = frozenset("""pdf md txt csv json yaml yml toml ini cfg conf env lock log py sh
js ts go rs java rb php html htm xml docx doc xlsx xls pptx png jpg jpeg gif svg zip tar gz
tgz whl exe dmg pkg deb rpm""".split())

# Paths: absolute (/etc/x, ~/x, ./x) or relative with a slash and a dotted file name.
_PATH = re.compile(r"(?<![\w.])(?:~|\.{1,2})?/[\w.-]+(?:/[\w.-]+)*"
                   r"|\b[\w-]+(?:/[\w.-]+)*/[\w-]+\.[a-z0-9]{1,5}\b"
                   r"|\b[\w-]+\.(?:" + "|".join(sorted(_FILE_EXT)) + r")\b")
# Commands: a backticked span, a shell prompt, a long option, or a lower-case command
# word followed by an argument. Lower case only, so "writes Python" is not a command;
# words that are also ordinary English ("make sure", "git history") are left out or
# need a subcommand.
_COMMAND = re.compile(r"`[^`]+`|(?:^|\s)\$\s+\S|(?<![\w-])--[a-z][\w-]+"
                      r"|(?<![\w-])(?:curl|wget|bash|sudo|pip3?|npm|npx|yarn|docker|"
                      r"kubectl|ssh|scp|rsync|chmod|chown|python3?|psql|helm|terraform)"
                      r"\s+[-\w./:@=]+"
                      r"|(?<![\w-])git\s+(?:clone|pull|push|fetch|checkout|remote|apply)\b")

# Capitalised words that are not names of systems.
_NOT_NAMES = frozenset("""i a an the this that these those our your my their its it he she
we they you monday tuesday wednesday thursday friday saturday sunday mondays tuesdays
wednesdays thursdays fridays saturdays sundays january february march april may june july
august september october november december am pm ok""".split())
_LEADING = frozenset({"the", "a", "an", "this", "that", "our", "your", "my", "their", "its"})
# Words that open a note's sentence without being part of a name ("Use Concur.", "For
# Beacon Benefits questions, ..."): imperatives and prepositions common in instructions.
_OPENERS = frozenset("""use send submit upload file contact email mail call check visit open
ask see go post pay book order deploy run install download request route direct forward
pull push read review save remember note please always never only all any every new from
for if when after before since until in on at to via per with by""".split())
_SENTENCE = re.compile(r"(?<=[.!?:;])\s+|\n+")
_WORD = re.compile(r"[A-Za-z0-9][\w&'’-]*")


def hosts(text: str) -> set[str]:
    """Hostnames named in `text` (bare or inside a URL or email), lower case."""
    low = text.lower()
    out = set()
    for tok in _DOMAIN.findall(low):
        last = tok.rsplit(".", 1)[-1]
        if len(last) >= 2 and last.isalpha() and last not in _FILE_EXT:
            out.add(tok)
    return out


def emails(text: str) -> set[str]:
    return {e.lower() for e in _EMAIL.findall(text)}


def proper_names(text: str) -> set[str]:
    """Capitalised product/system names, lower case: runs of capitalised words.

    The first word of a sentence is capitalised by grammar, so it starts a name only
    when the next word is capitalised too ("Restore Service handles ...") and it is
    not a common opener ("Use Concur" -> "concur"); a leading article is dropped
    ("The Quick Expense portal" -> "quick expense").
    """
    names = set()
    for sentence in _SENTENCE.split(text):
        words = _WORD.findall(sentence)
        run: list[str] = []

        def flush() -> None:
            while run and run[0].lower() in _LEADING:
                run.pop(0)
            if run:
                names.add(" ".join(run).lower())
            run.clear()

        for i, w in enumerate(words):
            cap = w[0].isupper() and w.lower() not in _NOT_NAMES
            if cap and i == 0:
                nxt = words[1] if len(words) > 1 else ""
                cap = bool(nxt) and nxt[0].isupper() and w.lower() not in _OPENERS
            if cap:
                run.append(w)
            else:
                flush()
        flush()
    return names


def entities(text: str) -> set[str]:
    """Destination-like entities a memory names: hosts, emails and system names."""
    return hosts(text) | emails(text) | proper_names(text)


def names_security_field(text: str) -> bool:
    """A host, URL, email, path or command, or change language (labels._CHANGE)."""
    return bool(hosts(text) or _URL.search(text) or _EMAIL.search(text)
                or _PATH.search(text) or _COMMAND.search(text)
                or labels._CHANGE.search(text))


# --- signals ------------------------------------------------------------------------

@dataclass
class CandidateSet:
    """Per-memory signals and the order a candidate filter would test them in.

    signals   - id -> the signals it raised, for every memory considered (possibly empty)
    ranked    - ids with at least one signal, most suspicious first
    conflicts - id -> the stable user memories it conflicts with (the evidence)
    """
    signals: dict[str, frozenset[str]]
    ranked: list[str]
    conflicts: dict[str, list[str]] = field(default_factory=dict)

    def with_any(self, names: Iterable[str] = SIGNALS) -> set[str]:
        """Memories raising at least one of `names`."""
        wanted = set(names)
        unknown = wanted - set(SIGNALS)
        if unknown:
            raise ValueError(f"unknown candidate signal(s) {sorted(unknown)}")
        return {m for m, s in self.signals.items() if s & wanted}


def _vector(row) -> np.ndarray | None:
    blob = row["embedding"]
    if blob is None:
        return None
    v = np.frombuffer(blob, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else None


def _external(conn: sqlite3.Connection, row, by_id: dict) -> bool:
    if row["source_type"] == "webpage":
        return True
    return any(by_id[a]["source_type"] == "webpage"
               for a in db.get_ancestors(conn, row["id"]) if a in by_id)


def _conflicts(rows: list, stable: list, threshold: float) -> dict[str, list[str]]:
    """id -> stable user memories on the same topic that name a different entity.

    `stable` is every active user memory that names an entity; a memory conflicts with
    one when their embeddings are close and it names an entity the user memory does not
    (a new destination for the same job). A user memory never conflicts: the user
    changing their mind is not an attack signal.
    """
    users = [(u["id"], v, entities(u["text"])) for u in stable
             if (v := _vector(u)) is not None]
    users = [(uid, v, ents) for uid, v, ents in users if ents]
    out: dict[str, list[str]] = {}
    if not users:
        return out
    matrix = np.stack([v for _, v, _ in users])
    for row in rows:
        if row["source_type"] == "user":
            continue
        v = _vector(row)
        ents = entities(row["text"])
        if v is None or not ents or v.shape[0] != matrix.shape[1]:
            continue
        sims = matrix @ v
        hits = [uid for (uid, _, u_ents), s in zip(users, sims)
                if uid != row["id"] and s >= threshold and ents - u_ents]
        if hits:
            out[row["id"]] = hits
    return out


def generate(
    conn: sqlite3.Connection,
    retrieved: Sequence[str] = (),
    *,
    memory_ids: Iterable[str] | None = None,
    threshold: float = CONFLICT_THRESHOLD,
) -> CandidateSet:
    """Section 5.3 signals for `memory_ids` (default: every active memory) and a ranking.

    retrieved - the retrieved set of the security-sensitive action, in rank order (the
                action log's, as the defender sees it); () when there was no action.
    Ranked by number of signals, then retrieval rank (only retrieved memories can be
    ablated from a frozen set, so they come first among equals), then newest first.
    """
    every = db.all_memories(conn)
    by_id = {r["id"]: r for r in every}
    order = {r["id"]: i for i, r in enumerate(every)}
    if memory_ids is None:
        ids = [r["id"] for r in every if r["status"] == "active"]
    else:
        ids = [m for m in dict.fromkeys(memory_ids) if m in by_id]
    rank = {m: i for i, m in enumerate(retrieved)}
    stable = [r for r in every if r["status"] == "active" and r["source_type"] == "user"]
    conflicts = _conflicts([by_id[m] for m in ids], stable, threshold)

    signals: dict[str, frozenset[str]] = {}
    for m in ids:
        row = by_id[m]
        got = set()
        if m in rank:
            got.add("retrieved_at_action")
        if _external(conn, row, by_id):
            got.add("external_source")
        if m in conflicts:
            got.add("conflict_with_stable")
        if names_security_field(row["text"]):
            got.add("security_field")
        signals[m] = frozenset(got)

    ranked = sorted((m for m, s in signals.items() if s),
                    key=lambda m: (-len(signals[m]), rank.get(m, len(rank)), -order[m]))
    return CandidateSet(signals, ranked, conflicts)
