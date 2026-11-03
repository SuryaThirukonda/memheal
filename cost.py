"""What MemHeal costs to run (proposal Section 8.10): storage, latency and model tokens.

Section 8.10 claims normal operation is cheap because lineage metadata is small, and
that the expensive part, counterfactual replays, runs only when an incident is
investigated. This script measures both halves without calling a model or opening a
socket:

1. Storage, from finished run databases opened read-only. sqlite's dbstat gives the
   bytes of every table and index page-exactly (without dbstat the used pages are shared
   out by logical size instead, and the method is recorded). The file is split into
   memory text, embeddings, the other memory-row columns, provenance tables (edges,
   retrieval log, actions, sources, run config, prompts, recovery log, blocklist, and
   their indexes) and sqlite's own pages. The per-session snapshot copies are reported
   on their own: rollback and measurement use them, but a deployment could keep fewer.
2. Latency, on synthetic stores with the real schema: memory.retrieve end to end,
   including its retrieval_events write, at 10 to 10,000 memories with a constant query
   embedder, so the timing is the store and not the encoder; one real MiniLM query
   encode, only if the model loads from the local cache with the hub offline; lineage
   queries on chains and trees (the deepest and shallowest shape for a size), and on
   every memory of the real stores.
3. Tokens, from what every run already logged (llm.jsonl, recovery/*.llm.jsonl) and
   recover.py's records: calls, tokens and time per operation and per recovery
   configuration. Dollars appear only with --price-in and --price-out, both given by
   the user; nothing here knows a price.

Memory architectures are never pooled: every storage, log, lineage and recovery figure
is reported per memory mode (evolving, flat, ...), read from the run's own recorded
config, since Section 8.10 compares them and a pooled median describes neither.

    python cost.py
    python cost.py --runs '*_dense_*_evolving' --out results/cost
    python cost.py --runs '*_dense_*' --modes evolving flat
    python cost.py --skip-latency
    python cost.py --price-in <usd_per_1M_input> --price-out <usd_per_1M_output>

Outputs (under --out): cost_tables.md and cost.json, plus retrieval_latency.png and
lineage_latency.png when the latency benchmark runs (the default). The benchmark stores
are written to a temp folder inside --out (on disk, as the runs were, not tmpfs) and
deleted afterwards. Run the benchmark on an otherwise idle machine: the load average
before and after is recorded, and a busy machine is flagged in both outputs.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import platform
import random
import re
import sqlite3
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Hashable, Iterable, Sequence

import numpy as np

from memheal import db, memory

ROOT = Path(__file__).parent
RUNS = ROOT / "data" / "runs"
DEFAULT_OUT = ROOT / "results" / "cost"
DEFAULT_RECORDS = (ROOT / "results" / "recovery" / "records.jsonl",)
# The dense suite is the one every recovery experiment runs on, in both memory
# architectures (flat: experiment X2). run_demo names a folder
# `<n>_<condition>_<context>_dense_<category>` and appends `_<memory>` only when the
# memory is not flat, so a flat run has no suffix at all. The categories are spelled out
# (catalog.ATTACK_CATEGORIES) because the adaptive suite shares the `_dense_` infix, and
# ending each glob there leaves out the `_t<i>` and `_labels` variants.
DENSE_CATEGORIES = ("direct", "composed", "procedural")
DEFAULT_RUNS = tuple(f"*_dense_{c}{suffix}" for suffix in ("_evolving", "")
                     for c in DENSE_CATEGORIES)
# Memory modes run_demo writes into a folder name; a name without one is flat.
NAMED_MODES = ("evolving", "mem0")
# Report order; any other mode follows alphabetically.
MODE_ORDER = ("evolving", "flat", "mem0")

# --- storage categories ---------------------------------------------------------

MEMORY_TABLE = "memories"
PROVENANCE_TABLES = ("memory_edges", "retrieval_events", "actions", "sources",
                     "experiment_runs", "prompts", "recovery_events", "source_blocklist")
LINEAGE_TABLE = "memory_edges"
# content = memories.text, embedding = memories.embedding, memory_meta = every other
# memories column and the id index, provenance = the tables above with their indexes,
# other = a table this script does not know, sqlite = schema, sequence and free pages.
CATEGORIES = ("content", "embedding", "memory_meta", "provenance", "other", "sqlite")

# --- latency benchmark ------------------------------------------------------------

RETRIEVAL_SIZES = (10, 30, 100, 300, 1000, 3000, 10000)
LINEAGE_SIZES = (10, 30, 100, 300, 1000, 3000, 10000)
REPEATS = 30
WARMUP = 3
DIM = 384                                    # all-MiniLM-L6-v2
TEXT_BYTES = 90                              # a typical extracted fact (dense runs: ~90 B)
TREE_BRANCHING = 2
BENCH_QUERY = "Submit this quarter's restore test to the usual service."
BENCH_SEED = 0                               # the shuffled timing order, reproducible
# A 1-minute load average above this share of the cores means other work competed with
# the benchmark (Linux counts processes waiting on disk too, which the commit timings
# share); the timings are then flagged as measured on a busy machine.
BUSY_LOAD_PER_CORE = 0.25

# --- logs -------------------------------------------------------------------------

FAMILY_ORDER = ("agent", "extract", "consolidate", "reflect", "reconstruct")
# The log stamps a call when its response arrives, not when it was sent, so the time
# since the previous entry is the call plus whatever ran locally in between (embedding,
# tool execution, sqlite). A gap longer than this is a pause, not a call.
MAX_GAP_S = 120.0
_SESSION = re.compile(r"^[^:]+:s(\d+)\b")
_AGENT_STEP = re.compile(r"^agent:s(\d+):step(\d+)$")
_STORAGE_DELTA = re.compile(r"storage_delta=\+(\d+)")
_REPLAYS = re.compile(r"\breplays=(\d+)")

# The palette every MemHeal figure uses: categorical slots 1-4 in fixed order.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"


# --- small helpers ------------------------------------------------------------------

def select_runs(runs_dir: Path, patterns: Sequence[str],
                modes: Sequence[str] | None = None) -> list[Path]:
    """Run folders under runs_dir whose name matches any glob and that hold a run.db;
    with modes, only runs whose recorded memory mode is one of them."""
    if not runs_dir.is_dir():
        return []
    runs = [p for p in sorted(runs_dir.iterdir())
            if (p / "run.db").exists() and any(fnmatch.fnmatch(p.name, g) for g in patterns)]
    if modes is not None:
        runs = [p for p in runs if run_memory_mode(p) in modes]
    return runs


def _recorded_mode(conn: sqlite3.Connection) -> str | None:
    """agent.memory_mode from the run's own config row (every run since modes existed)."""
    row = conn.execute("SELECT config_json FROM experiment_runs ORDER BY rowid LIMIT 1"
                       ).fetchone()
    try:
        return (json.loads(row[0]).get("agent") or {}).get("memory_mode") if row else None
    except (TypeError, ValueError, AttributeError):
        return None


def run_memory_mode(run_dir: Path, conn: sqlite3.Connection | None = None) -> str:
    """The memory architecture a run used, as the run recorded it.

    The folder name cannot say it reliably: run_demo leaves the memory out of the name
    when it is flat, so `..._dense_composed` is a flat run, not a composed-memory one.
    The agent config in experiment_runs says it directly, then summary.json; only a run
    older than both falls back to the name, where no mode suffix means flat (runs from
    before EvolvingMemory were flat, and measure.py reads them the same way).
    """
    run_dir = Path(run_dir)
    mode = None
    try:
        if conn is not None:
            mode = _recorded_mode(conn)
        else:
            own = db.connect_readonly(run_dir / "run.db")
            try:
                mode = _recorded_mode(own)
            finally:
                own.close()
    except sqlite3.DatabaseError:              # no run.db, or not one this schema knows
        pass
    if not mode:
        try:
            mode = json.loads((run_dir / "summary.json").read_text()).get("memory_mode")
        except (OSError, ValueError, AttributeError):
            mode = None
    if not mode:
        parts = run_dir.name.split("_")
        mode = next((m for m in NAMED_MODES if m in parts), "flat")
    return mode


def mode_sort(mode: str) -> tuple:
    return (MODE_ORDER.index(mode) if mode in MODE_ORDER else len(MODE_ORDER), mode)


def run_modes(runs: Sequence[Path]) -> dict[str, str]:
    """Run folder name -> memory mode, read once for every section."""
    return {Path(p).name: run_memory_mode(p) for p in runs}


def describe(values: Iterable[float]) -> dict[str, float] | None:
    """Median, quartiles and range: the per-run spread Section 8.10 tables report."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    return {"n": len(vals), "median": float(med), "q1": float(q1), "q3": float(q3),
            "min": min(vals), "max": max(vals), "mean": float(np.mean(vals))}


def latency_ms(samples: Sequence[float]) -> dict[str, float]:
    """Median and p95 in milliseconds; p95 is what a user-facing budget has to cover."""
    ms = np.asarray(samples, dtype=float) * 1000
    return {"median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)),
            "samples": len(ms)}


def _rel(path: Path) -> str:
    """A path as the report should show it: relative to the repository when inside it."""
    try:
        return str(Path(path).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _pragma(conn: sqlite3.Connection, name: str) -> int:
    return int(conn.execute(f"PRAGMA {name}").fetchone()[0])


# --- 1. storage ---------------------------------------------------------------------

def owners(conn: sqlite3.Connection) -> dict[str, str]:
    """Every b-tree in the file mapped to the table it belongs to (an index to its table)."""
    out = {"sqlite_schema": "sqlite_schema", "sqlite_master": "sqlite_schema"}
    for name, tbl in conn.execute(
            "SELECT name, tbl_name FROM sqlite_master WHERE type IN ('table', 'index')"):
        out[name] = tbl
    return out


def dbstat_bytes(conn: sqlite3.Connection) -> dict[str, tuple[float, float]] | None:
    """(allocated, payload) bytes per b-tree from dbstat; None when sqlite lacks it.

    Allocated counts whole pages, overflow pages included, so the objects plus the
    free list add up to the file exactly.
    """
    try:
        rows = conn.execute(
            "SELECT name, SUM(pgsize), SUM(payload) FROM dbstat GROUP BY name").fetchall()
    except sqlite3.DatabaseError:
        return None
    return {name: (float(alloc or 0), float(payload or 0)) for name, alloc, payload in rows}


def estimated_bytes(conn: sqlite3.Connection) -> dict[str, tuple[float, float]]:
    """Without dbstat: the used pages (exact) shared out by each b-tree's logical size.

    Logical size is the summed byte length of every stored value (for an index, its
    columns plus an 8-byte rowid per entry). It ignores record headers and page fill,
    so the per-object figures are estimates; the total still matches the file.
    """
    page_size = _pragma(conn, "page_size")
    used = (_pragma(conn, "page_count") - _pragma(conn, "freelist_count")) * page_size
    logical: dict[str, float] = {"sqlite_schema": 0.0}
    for name, kind, tbl in conn.execute(
            "SELECT name, type, tbl_name FROM sqlite_master WHERE type IN ('table', 'index')"
    ).fetchall():
        if kind == "table":
            cols, per_row = [r[1] for r in conn.execute(f'PRAGMA table_info("{name}")')], 0
        else:
            cols = [r[2] for r in conn.execute(f'PRAGMA index_info("{name}")') if r[2]]
            per_row = 8
        expr = " + ".join(f'COALESCE(LENGTH(CAST("{c}" AS BLOB)), 0)' for c in cols) or "0"
        n, total = conn.execute(f'SELECT COUNT(*), TOTAL({expr}) FROM "{tbl}"').fetchone()
        logical[name] = float(total) + per_row * n
    # Every b-tree owns at least its root page; the remaining pages go by logical size.
    spare = max(used - page_size * len(logical), 0)
    weight = sum(logical.values()) or 1.0
    return {name: (page_size + spare * b / weight, b) for name, b in logical.items()}


def categorize(sizes: dict[str, tuple[float, float]], owner: dict[str, str],
               text_bytes: float, embedding_bytes: float,
               freelist_bytes: float) -> dict[str, float]:
    """Allocated bytes per category. The memories table's own pages are split by payload:
    text and embedding get their exact byte share of it, scaled up by the same page fill
    as the rest of the row, and the other columns get the remainder."""
    cats = dict.fromkeys(CATEGORIES, 0.0)
    for name, (alloc, payload) in sizes.items():
        table = owner.get(name, name)
        if name == MEMORY_TABLE:
            scale = alloc / payload if payload else 0.0
            cats["content"] += text_bytes * scale
            cats["embedding"] += embedding_bytes * scale
            cats["memory_meta"] += alloc - (text_bytes + embedding_bytes) * scale
        elif table == MEMORY_TABLE:
            cats["memory_meta"] += alloc
        elif table in PROVENANCE_TABLES:
            cats["provenance"] += alloc
        elif table.startswith("sqlite_"):
            cats["sqlite"] += alloc
        else:
            cats["other"] += alloc
    cats["sqlite"] += freelist_bytes
    return cats


def run_storage(run_dir: Path, use_dbstat: bool = True) -> dict[str, Any]:
    """Storage of one run's final store (run.db) and of its snapshot copies."""
    run_dir = Path(run_dir)
    conn = db.connect_readonly(run_dir / "run.db")
    try:
        page_size = _pragma(conn, "page_size")
        file_bytes = _pragma(conn, "page_count") * page_size
        freelist = _pragma(conn, "freelist_count") * page_size
        sizes = dbstat_bytes(conn) if use_dbstat else None
        method = "dbstat" if sizes is not None else "estimate"
        if sizes is None:
            sizes = estimated_bytes(conn)
        owner = owners(conn)
        n_mem, n_active, text_b, emb_b = conn.execute(
            "SELECT COUNT(*), TOTAL(status = 'active'), TOTAL(LENGTH(CAST(text AS BLOB))),"
            " TOTAL(LENGTH(embedding)) FROM memories").fetchone()
        n_edges = conn.execute(f"SELECT COUNT(*) FROM {LINEAGE_TABLE}").fetchone()[0]
        run = conn.execute("SELECT condition FROM experiment_runs ORDER BY rowid LIMIT 1"
                           ).fetchone()
        memory_mode = run_memory_mode(run_dir, conn)
    finally:
        conn.close()

    cats = categorize(sizes, owner, text_b, emb_b, freelist)
    # The same split over payload alone (the bytes of the stored values and record
    # headers): every b-tree holds at least one 4 KB page, so on stores this small the
    # allocated view is dominated by fixed per-table pages, while payload shows what grows
    # with the data.
    data = categorize({name: (payload, payload) for name, (_, payload) in sizes.items()},
                      owner, text_b, emb_b, 0.0)
    tables: dict[str, float] = defaultdict(float)
    table_payload: dict[str, float] = defaultdict(float)
    for name, (alloc, payload) in sizes.items():
        tables[owner.get(name, name)] += alloc
        table_payload[owner.get(name, name)] += payload
    payload_total = sum(data.values())
    snaps = sorted((run_dir / "snapshots").glob("*.db"))
    snap_bytes = sum(p.stat().st_size for p in snaps)
    n_mem = int(n_mem)
    return {
        "run_dir": run_dir.name,
        "memory_mode": memory_mode,
        "condition": run[0] if run else _condition_from_name(run_dir.name),
        "method": method,
        "file_bytes": file_bytes,
        "n_memories": n_mem,
        "n_active": int(n_active),
        "n_edges": int(n_edges),
        "text_payload_bytes": int(text_b),
        "embedding_payload_bytes": int(emb_b),
        "categories": {k: round(v, 1) for k, v in cats.items()},
        "payload_categories": {k: round(v, 1) for k, v in data.items()},
        "tables": {k: round(v, 1) for k, v in sorted(tables.items())},
        "table_payload": {k: round(v, 1) for k, v in sorted(table_payload.items())},
        # Every b-tree on its own, so an index is visible apart from the table it serves
        # (the tables above fold each index into its table).
        "btrees": {name: {"table": owner.get(name, name), "allocated": round(alloc, 1),
                          "payload": round(payload, 1)}
                   for name, (alloc, payload) in sorted(sizes.items())},
        "bytes_per_memory": file_bytes / n_mem if n_mem else None,
        "edges_per_memory": n_edges / n_mem if n_mem else None,
        "provenance_share": cats["provenance"] / file_bytes if file_bytes else None,
        # Upper bound: also count the memory row's own provenance columns (source,
        # operation, prompt hash, model, timestamps) and its id index.
        "provenance_share_with_meta":
            (cats["provenance"] + cats["memory_meta"]) / file_bytes if file_bytes else None,
        "provenance_payload_share": data["provenance"] / payload_total if payload_total else None,
        "lineage_bytes": tables.get(LINEAGE_TABLE, 0.0),
        "lineage_bytes_per_memory": tables.get(LINEAGE_TABLE, 0.0) / n_mem if n_mem else None,
        "lineage_payload_per_memory":
            table_payload.get(LINEAGE_TABLE, 0.0) / n_mem if n_mem else None,
        "lineage_payload_per_edge":
            table_payload.get(LINEAGE_TABLE, 0.0) / n_edges if n_edges else None,
        "n_snapshots": len(snaps),
        "snapshot_bytes": snap_bytes,
        "snapshot_ratio": snap_bytes / file_bytes if file_bytes else None,
    }


def _share(parts: dict[str, float], key: str) -> float | None:
    total = sum(parts.values())
    return parts[key] / total if total else None


def _condition_from_name(name: str) -> str:
    parts = name.split("_")
    return parts[1] if len(parts) > 1 else "unknown"


# (key, label, format): "n" = whole number, "x" = two decimals, "%" = share of the file.
STORAGE_METRICS = (
    ("file_bytes", "run.db bytes", "n"),
    ("n_memories", "memories (all statuses)", "n"),
    ("n_active", "active memories", "n"),
    ("bytes_per_memory", "file bytes per memory", "n"),
    ("edges_per_memory", "lineage edges per memory", "x"),
    ("lineage_bytes_per_memory", "lineage (edges + indexes), allocated bytes per memory", "n"),
    ("lineage_payload_per_memory", "lineage (edges + indexes), payload bytes per memory", "n"),
    ("lineage_payload_per_edge", "lineage payload bytes per edge", "n"),
    ("provenance_share", "provenance share of file (allocated pages)", "%"),
    ("provenance_share_with_meta", "same, incl. memory-row metadata", "%"),
    ("provenance_payload_share", "provenance share of payload (stored values)", "%"),
    ("snapshot_bytes", "snapshot copies, bytes per run", "n"),
    ("snapshot_ratio", "snapshot bytes / run.db bytes", "x"),
)


def is_index(name: str, table: str) -> bool:
    """An index b-tree (its own name differs from its table's); sqlite's schema, which
    dbstat may still call sqlite_master, is not."""
    return name != table and table != "sqlite_schema"


def mode_groups(rows: Sequence[dict]) -> dict[str, list[dict]]:
    """One group per memory mode, then per mode and condition. Modes are never pooled: a
    flat store has no lineage edges, so a pooled median describes neither architecture."""
    groups: dict[str, list[dict]] = {}
    for mode in sorted({r["memory_mode"] for r in rows}, key=mode_sort):
        rs = [r for r in rows if r["memory_mode"] == mode]
        groups[mode] = rs
        for cond in sorted({r["condition"] for r in rs}):
            groups[f"{mode}/{cond}"] = [r for r in rs if r["condition"] == cond]
    return groups


def storage_section(runs: Sequence[Path], use_dbstat: bool = True) -> dict[str, Any]:
    rows = [run_storage(p, use_dbstat) for p in runs]
    groups = mode_groups(rows)
    summary = {}
    for name, rs in groups.items():
        s = {m: describe(r[m] for r in rs) for m, _, _ in STORAGE_METRICS}
        s["share"] = {c: describe(r["categories"][c] / r["file_bytes"] for r in rs)
                      for c in CATEGORIES}
        s["payload_share"] = {c: describe(_share(r["payload_categories"], c) for r in rs)
                              for c in CATEGORIES}
        tables = sorted({t for r in rs for t in r["tables"]})
        s["tables"] = {t: describe(r["tables"].get(t, 0.0) for r in rs) for t in tables}
        s["table_payload"] = {t: describe(r["table_payload"].get(t, 0.0) for r in rs)
                              for t in tables}
        indexes = sorted({(n, b["table"]) for r in rs for n, b in r["btrees"].items()
                          if is_index(n, b["table"])})
        s["indexes"] = {n: {"table": t,
                            **{k: describe(r["btrees"].get(n, {}).get(k, 0.0) for r in rs)
                               for k in ("allocated", "payload")}}
                        for n, t in indexes}
        summary[name] = s
    return {"methods": sorted({r["method"] for r in rows}),
            "modes": sorted({r["memory_mode"] for r in rows}, key=mode_sort),
            "runs": rows, "summary": summary}


# --- 2. latency ---------------------------------------------------------------------

class ConstantEmbedder:
    """One fixed unit vector for every query, so a timed retrieve() measures the store
    (load, score, log) and not the sentence encoder, which is timed on its own."""

    name = "constant"

    def __init__(self, dim: int = DIM, seed: int = 0):
        v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
        self._v = (v / np.linalg.norm(v))[None, :]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.repeat(self._v, len(texts), axis=0)


def synthetic_store(path: Path, n: int, dim: int = DIM, seed: int = 0) -> sqlite3.Connection:
    """A real-schema store of n active memories with random unit float32 embeddings.

    Rows go in through one executemany: setup is not what is timed, and n separate
    commits would take longer than the benchmark itself at 10,000.
    """
    conn = db.connect(path)
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    filler = "x" * TEXT_BYTES
    rows = [(f"m_{i:08x}", f"synthetic fact {i} {filler}"[:TEXT_BYTES], i // 3 + 1,
             vecs[i].tobytes()) for i in range(n)]
    with conn:
        conn.executemany(
            "INSERT INTO memories (id, text, source_type, operation, session, status, model,"
            " embedding) VALUES (?, ?, 'user', 'extract', ?, 'active', 'synthetic', ?)", rows)
    return conn


def _timed(fn: Callable[[], Any], repeats: int, warmup: int = WARMUP) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def _interleaved(fns: dict[Hashable, Callable[[], Any]], repeats: int, warmup: int = WARMUP,
                 seed: int = BENCH_SEED) -> dict[Hashable, list[float]]:
    """Time every function `repeats` times, one call of each per round in a freshly
    shuffled order. Timing one size's calls back to back would let a burst of load from
    elsewhere on the machine land on that size alone and bend the curve; spread over the
    rounds, it lands on every size and part alike."""
    for fn in fns.values():
        for _ in range(warmup):
            fn()
    rng = random.Random(seed)
    order = list(fns)
    out: dict[Hashable, list[float]] = {key: [] for key in order}
    for _ in range(repeats):
        rng.shuffle(order)
        for key in order:
            t0 = time.perf_counter()
            fns[key]()
            out[key].append(time.perf_counter() - t0)
    return out


RETRIEVAL_PARTS = ("retrieve", "load", "score", "log")


def _retrieval_fns(conn: sqlite3.Connection, emb: ConstantEmbedder, n: int,
                   k: int) -> dict[str, Callable[[], Any]]:
    """memory.retrieve end to end, and its three parts on their own."""
    q = emb.encode([BENCH_QUERY])[0]

    def load():
        return np.stack([np.frombuffer(r["embedding"], dtype=np.float32)
                         for r in db.active_memories(conn)])

    matrix = load()
    ids = [f"m_{i:08x}" for i in range(min(k, n))]
    return {
        "retrieve": lambda: memory.retrieve(conn, emb, "bench", 1, BENCH_QUERY, k=k),
        "load": load,
        "score": lambda: np.argsort(-(matrix @ q), kind="stable")[:k],
        "log": lambda: db.log_retrieval(conn, "bench", "s1", 1, ids, [0.5] * len(ids)),
    }


def retrieval_latency(sizes: Sequence[int], repeats: int, workdir: Path, dim: int = DIM,
                      k: int = 5) -> list[dict[str, Any]]:
    """memory.retrieve end to end per store size, plus its three parts timed alone:
    loading the active rows into a matrix, scoring, and the retrieval_events write
    (the provenance a flat top-k store would not keep). Every store is built first and
    the calls are interleaved across sizes and parts (_interleaved)."""
    emb = ConstantEmbedder(dim)
    conns: dict[int, sqlite3.Connection] = {}
    paths = [Path(workdir) / f"retrieve_{n}.db" for n in sizes]
    try:
        fns: dict[Hashable, Callable[[], Any]] = {}
        for n, path in zip(sizes, paths):
            path.unlink(missing_ok=True)
            conns[n] = synthetic_store(path, n, dim)
            for part, fn in _retrieval_fns(conns[n], emb, n, k).items():
                fns[(n, part)] = fn
        samples = _interleaved(fns, repeats)
    finally:
        for conn in conns.values():
            conn.close()
        for path in paths:
            path.unlink(missing_ok=True)
    rows = []
    for n in sizes:
        row: dict[str, Any] = {"n": n}
        row.update({part: latency_ms(samples[(n, part)]) for part in RETRIEVAL_PARTS})
        row["parts_ms"] = sum(row[p]["median_ms"] for p in ("load", "score", "log"))
        row["log_share"] = row["log"]["median_ms"] / row["retrieve"]["median_ms"]
        rows.append(row)
    return rows


def encoder_latency(repeats: int, query: str = BENCH_QUERY) -> dict[str, Any]:
    """One real MiniLM query encode, from the local model cache only.

    The hub is forced offline first, and the model is loaded with local_files_only, so
    a machine without the cached model skips this (and says why) instead of
    downloading it. memory.SentenceTransformerEmbedder is not used because it falls
    back to the network.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(memory.EMBEDDING_MODEL, device="cpu", local_files_only=True)
    except Exception as exc:                 # missing package, missing cache, offline error
        return {"model": memory.EMBEDDING_MODEL,
                "skipped": f"{type(exc).__name__}: {exc}".splitlines()[0][:200]}
    samples = _timed(lambda: model.encode([query], normalize_embeddings=True,
                                          show_progress_bar=False), repeats)
    return {"model": memory.EMBEDDING_MODEL, **latency_ms(samples)}


def lineage_store(path: Path, shape: str, n: int,
                  branching: int = TREE_BRANCHING) -> tuple[sqlite3.Connection, str, str]:
    """n memories joined as a chain (each the child of the one before: depth n-1) or a
    balanced tree (node i's parent is (i-1)//branching: depth ~log n), the two extremes
    of depth for one size. Returns the connection, the root and the deepest node."""
    if shape not in ("chain", "tree"):
        raise ValueError(f"unknown shape {shape!r}")
    conn = db.connect(path)
    ids = [f"m_{i:08x}" for i in range(n)]
    parent = (lambda i: i - 1) if shape == "chain" else (lambda i: (i - 1) // branching)
    with conn:
        conn.executemany(
            "INSERT INTO memories (id, text, source_type, operation, session, status)"
            " VALUES (?, ?, 'derived', 'summarize', ?, 'active')",
            [(m, f"lineage node {i}", i + 1) for i, m in enumerate(ids)])
        conn.executemany("INSERT INTO memory_edges (parent_id, child_id) VALUES (?, ?)",
                         [(ids[parent(i)], ids[i]) for i in range(1, n)])
    return conn, ids[0], ids[-1]


def lineage_latency(sizes: Sequence[int], repeats: int, workdir: Path) -> list[dict[str, Any]]:
    """db.get_descendants from the root and db.get_ancestors from the deepest node, every
    shape and size built first and the queries interleaved (_interleaved)."""
    conns: list[sqlite3.Connection] = []
    paths: list[Path] = []
    fns: dict[Hashable, Callable[[], Any]] = {}
    found: dict[Hashable, int] = {}
    try:
        for shape in ("chain", "tree"):
            for n in sizes:
                path = Path(workdir) / f"lineage_{shape}_{n}.db"
                path.unlink(missing_ok=True)
                paths.append(path)
                conn, root, leaf = lineage_store(path, shape, n)
                conns.append(conn)
                for query, fn, start in (("descendants", db.get_descendants, root),
                                         ("ancestors", db.get_ancestors, leaf)):
                    key = (shape, n, query)
                    found[key] = len(fn(conn, start))
                    fns[key] = (lambda f=fn, c=conn, s=start: f(c, s))
        samples = _interleaved(fns, repeats)
    finally:
        for conn in conns:
            conn.close()
        for path in paths:
            path.unlink(missing_ok=True)
    return [{"shape": shape, "n": n, "query": query, "found": found[(shape, n, query)],
             **latency_ms(samples[(shape, n, query)])} for shape, n, query in fns]


def real_lineage_latency(runs: Sequence[Path]) -> dict[str, Any]:
    """Both lineage queries once on every memory of every selected run (read-only), the
    sizes recovery actually asks about."""
    samples: dict[str, list[float]] = {"descendants": [], "ancestors": []}
    found: dict[str, list[int]] = {"descendants": [], "ancestors": []}
    for run_dir in runs:
        conn = db.connect_readonly(Path(run_dir) / "run.db")
        try:
            for (mid,) in conn.execute("SELECT id FROM memories").fetchall():
                for query, fn in (("descendants", db.get_descendants),
                                  ("ancestors", db.get_ancestors)):
                    t0 = time.perf_counter()
                    got = fn(conn, mid)
                    samples[query].append(time.perf_counter() - t0)
                    found[query].append(len(got))
        finally:
            conn.close()
    return {q: {**latency_ms(s), "found": describe(found[q])}
            for q, s in samples.items() if s}


def loadavg() -> list[float] | None:
    """The 1, 5 and 15 minute load averages; None where the platform has none."""
    try:
        return [round(v, 2) for v in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def busy(loads: Iterable[list[float] | None], cores: int | None) -> bool:
    """Did other work compete with the benchmark? Any 1-minute load above the threshold."""
    limit = BUSY_LOAD_PER_CORE * (cores or 1)
    return any(load is not None and load[0] > limit for load in loads)


RAM_FILESYSTEMS = ("tmpfs", "ramfs")


def fs_from_mounts(path: str, mounts: str) -> str | None:
    """The filesystem type of the longest mount point holding path, from /proc/mounts
    text (`device mount_point type ...`, spaces in names escaped as \\040)."""
    best: tuple[str, str] | None = None
    for line in mounts.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        point, fstype = fields[1].replace("\\040", " "), fields[2]
        holds = path == point or path.startswith(point.rstrip("/") + "/")
        if holds and (best is None or len(point) > len(best[0])):
            best = (point, fstype)
    return best[1] if best else None


def filesystem(path: Path) -> str | None:
    """The filesystem the benchmark stores are written to; None off Linux. On a RAM
    filesystem (/tmp is often tmpfs) a commit skips the disk sync a real run pays, so
    the write and end-to-end timings would come out too low."""
    try:
        mounts = Path("/proc/mounts").read_text()
    except OSError:
        return None
    return fs_from_mounts(str(Path(path).resolve()), mounts)


def latency_section(runs: Sequence[Path], sizes: Sequence[int], lineage_sizes: Sequence[int],
                    repeats: int, encoder: bool, workdir: Path | None = None,
                    modes: dict[str, str] | None = None) -> dict[str, Any]:
    """workdir should be on the disk the runs used: on a RAM filesystem (/tmp is often
    tmpfs) the retrieval_events commit skips the fsync a real run pays.

    The load average is read before and after the timed parts, since a figure measured
    while other processes compete for the CPU or the disk is not the store's own cost.
    """
    load_before = loadavg()
    with tempfile.TemporaryDirectory(dir=workdir, prefix=".bench-") as tmp:
        tmp = Path(tmp)
        out = {
            "retrieval": retrieval_latency(sizes, repeats, tmp),
            "lineage_synthetic": lineage_latency(lineage_sizes, repeats, tmp),
        }
    # Per memory mode: a flat store has no edges, so its queries are trivially empty and
    # pooling them would hide what lineage costs where it exists.
    modes = modes if modes is not None else run_modes(runs)
    by_mode: dict[str, list[Path]] = defaultdict(list)
    for p in runs:
        by_mode[modes.get(Path(p).name) or run_memory_mode(p)].append(p)
    out["lineage_real"] = {m: real_lineage_latency(by_mode[m])
                           for m in sorted(by_mode, key=mode_sort)}
    out["encoder"] = (encoder_latency(repeats) if encoder
                      else {"model": memory.EMBEDDING_MODEL, "skipped": "--skip-encoder"})
    load_after = loadavg()
    cores = os.cpu_count()
    where = Path(workdir or tempfile.gettempdir())
    fs = filesystem(where)
    out["settings"] = {"repeats": repeats, "warmup": WARMUP, "dim": DIM, "k": 5,
                       "text_bytes": TEXT_BYTES, "tree_branching": TREE_BRANCHING,
                       "order": f"interleaved: each round times every size and part once, "
                                f"shuffled (seed {BENCH_SEED})",
                       "workdir": _rel(where), "workdir_fs": fs,
                       "ram_workdir": fs in RAM_FILESYSTEMS,
                       "loadavg_before": load_before, "loadavg_after": load_after,
                       "busy_load_threshold": round(BUSY_LOAD_PER_CORE * (cores or 1), 2),
                       "busy": busy((load_before, load_after), cores)}
    out["machine"] = machine()
    return out


def machine() -> dict[str, Any]:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"cpu": cpu, "cores": os.cpu_count(), "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version, "numpy": np.__version__,
            "platform": platform.platform()}


# --- 3. tokens from the logs ----------------------------------------------------------

@dataclass
class Call:
    """One logged model call."""
    source: str              # run | recovery
    config: str              # recovery configuration (its log's file name); '' for runs
    run_dir: str
    family: str              # agent | extract | consolidate | reflect | reconstruct | ...
    session: int | None
    input: int
    output: int
    cached: int
    gap: float | None        # seconds since the previous call in the same log
    memory_mode: str = ""    # the run's memory architecture
    concurrent: bool = False  # from a log several replays wrote at once: gap is None


def interleaved(tags: Iterable[str]) -> bool:
    """Do the agent calls in one log come from several sessions running at once?

    A session's steps are back-to-back calls with only local tool calls between them
    (agent.run_session), so in a log one thread wrote, every agent step k > 0 directly
    follows its own step k-1 among the agent calls. Concurrent counterfactual replays
    (recover.py --cf-workers > 1, all appending to one log) interleave theirs, and a
    step after anything else gives them away. Replays that all act at step 0 leave no
    such trace; the records' cf_workers catches those (concurrent_logs).
    """
    prev = None
    for tag in tags:
        m = _AGENT_STEP.match(tag or "")
        if not m:
            continue
        session, step = int(m.group(1)), int(m.group(2))
        if step > 0 and prev != (session, step - 1):
            return True
        prev = (session, step)
    return False


def read_calls(path: Path, *, source: str, run_dir: str, config: str = "",
               memory_mode: str = "", concurrent: bool = False) -> list[Call]:
    """Every call in one llm.jsonl. A torn line (a run killed mid-write) is skipped.

    The time since the previous line bounds a call's latency only when one thread wrote
    the log. When several replays wrote it at once (concurrent, or found interleaved),
    consecutive lines belong to different calls, the gaps shrink by about the number
    of workers, and no gap is reported.
    """
    entries = []
    with Path(path).open() as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    concurrent = concurrent or interleaved(e.get("tag") or "" for e in entries)
    calls: list[Call] = []
    prev_ts = None
    for entry in entries:
        tag = entry.get("tag") or ""
        ts = entry.get("ts")
        gap = ts - prev_ts if ts is not None and prev_ts is not None else None
        if concurrent or (gap is not None and not 0 < gap <= MAX_GAP_S):
            gap = None
        if ts is not None:
            prev_ts = ts
        usage = entry.get("usage") or {}
        m = _SESSION.match(tag)
        calls.append(Call(
            source=source, config=config, run_dir=run_dir,
            family=tag.split(":", 1)[0] or "untagged",
            session=int(m.group(1)) if m else None,
            input=int(usage.get("input_tokens") or 0),
            output=int(usage.get("output_tokens") or 0),
            cached=int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0),
            gap=gap, memory_mode=memory_mode, concurrent=concurrent,
        ))
    return calls


def concurrent_logs(records: Iterable[dict]) -> set[tuple[str, str]]:
    """(run folder, recover.py configuration tag) for every record whose counterfactual
    ran its replays concurrently (cf_workers > 1).

    recover.py names a configuration's log after config_tag, which leaves cf_workers
    out (the results do not depend on it), so the tag here is the configuration
    without its knob suffixes and a log is flagged when its name is that tag or starts
    with it plus '@' (see is_concurrent). That can also flag a workers=1 log of the
    same configuration, which only loses its time per call: never a wrong bound.
    """
    import recover                            # its own naming, not a copy of it
    out = set()
    for r in records:
        if (r.get("cf_workers") or 1) > 1:
            opts = recover.Options(verdicts=r.get("verdicts") or "oracle",
                                   modifiers=tuple(r.get("modifiers") or ()),
                                   source=r.get("source") or "patched")
            out.add((r.get("run_dir"), recover.config_tag(r.get("mode") or "none", opts)))
    return out


def is_concurrent(run_dir: str, stem: str, flagged: Collection[tuple[str, str]]) -> bool:
    return any(rd == run_dir and (stem == tag or stem.startswith(tag + "@"))
               for rd, tag in flagged)


def collect_calls(runs: Sequence[Path], modes: dict[str, str] | None = None,
                  concurrent: Collection[tuple[str, str]] = ()) -> list[Call]:
    calls: list[Call] = []
    for run_dir in runs:
        run_dir = Path(run_dir)
        mode = (modes or {}).get(run_dir.name) or run_memory_mode(run_dir)
        if (run_dir / "llm.jsonl").exists():
            calls += read_calls(run_dir / "llm.jsonl", source="run", run_dir=run_dir.name,
                                memory_mode=mode)
        for log in sorted((run_dir / "recovery").glob("*.llm.jsonl")):
            stem = log.name[: -len(".llm.jsonl")]
            calls += read_calls(log, source="recovery", run_dir=run_dir.name, config=stem,
                                memory_mode=mode,
                                concurrent=is_concurrent(run_dir.name, stem, concurrent))
    return calls


@dataclass(frozen=True)
class Prices:
    """USD per 1M tokens, supplied by the user on the command line. Cached input tokens
    are a subset of input tokens in the API's usage; without --price-cached they are
    charged at the full input price."""
    input: float
    output: float
    cached: float | None = None

    def usd(self, input_tokens: float, output_tokens: float, cached_tokens: float = 0) -> float:
        cached_price = self.input if self.cached is None else self.cached
        return ((input_tokens - cached_tokens) * self.input + cached_tokens * cached_price
                + output_tokens * self.output) / 1e6


def _family_sort(family: str) -> tuple:
    return (FAMILY_ORDER.index(family) if family in FAMILY_ORDER else len(FAMILY_ORDER), family)


def family_rows(calls: Sequence[Call], prices: Prices | None = None) -> list[dict[str, Any]]:
    """Calls, tokens and inter-call time per (log kind, memory mode, operation)."""
    groups: dict[tuple[str, str, str], list[Call]] = defaultdict(list)
    for c in calls:
        groups[(c.source, c.memory_mode, c.family)].append(c)
    rows = []
    for key in sorted(groups, key=lambda k: (k[0] != "run", mode_sort(k[1]), _family_sort(k[2]))):
        source, mode, family = key
        cs = groups[key]
        tin, tout, tcached = (sum(c.input for c in cs), sum(c.output for c in cs),
                              sum(c.cached for c in cs))
        gaps = [c.gap for c in cs if c.gap is not None]
        row = {"source": source, "memory_mode": mode, "family": family, "calls": len(cs),
               "runs": len({c.run_dir for c in cs}),
               "input_tokens": tin, "output_tokens": tout, "cached_tokens": tcached,
               "input_per_call": tin / len(cs), "output_per_call": tout / len(cs),
               "cached_share": tcached / tin if tin else 0.0,
               "concurrent_calls": sum(c.concurrent for c in cs),
               "gap_s": ({"median": float(np.median(gaps)),
                          "p95": float(np.percentile(gaps, 95)), "n": len(gaps)}
                         if gaps else None)}
        if prices:
            row["usd"] = prices.usd(tin, tout, tcached)
        rows.append(row)
    return rows


def run_rows(calls: Sequence[Call]) -> list[dict[str, Any]]:
    """Normal operation, per run: what the agent and its memory spent over every session."""
    by_run: dict[str, list[Call]] = defaultdict(list)
    for c in calls:
        if c.source == "run":
            by_run[c.run_dir].append(c)
    rows = []
    for run_dir, cs in sorted(by_run.items()):
        sessions = len({c.session for c in cs if c.session is not None}) or None
        tokens = sum(c.input + c.output for c in cs)
        row = {"run_dir": run_dir, "memory_mode": cs[0].memory_mode,
               "calls": len(cs), "sessions": sessions,
               "input_tokens": sum(c.input for c in cs),
               "output_tokens": sum(c.output for c in cs),
               "cached_tokens": sum(c.cached for c in cs),
               "tokens_per_session": tokens / sessions if sessions else None,
               "calls_per_session": len(cs) / sessions if sessions else None}
        for fam in {c.family for c in cs}:
            row[f"calls_{fam}"] = sum(c.family == fam for c in cs)
        rows.append(row)
    return rows


def recovery_log_rows(calls: Sequence[Call], prices: Prices | None = None) -> list[dict]:
    """Per recovery configuration file: everything its experiment spent, recovery plus
    the measurement replays and probes (records.jsonl has the recovery-only part)."""
    by_cfg: dict[tuple[str, str], list[Call]] = defaultdict(list)
    for c in calls:
        if c.source == "recovery":
            by_cfg[(c.memory_mode, c.config)].append(c)
    rows = []
    for (mode, cfg) in sorted(by_cfg, key=lambda k: (mode_sort(k[0]), k[1])):
        cs = by_cfg[(mode, cfg)]
        runs = len({c.run_dir for c in cs})
        tin, tout, tcached = (sum(c.input for c in cs), sum(c.output for c in cs),
                              sum(c.cached for c in cs))
        row = {"memory_mode": mode, "config": cfg, "runs": runs, "calls": len(cs),
               "calls_per_run": len(cs) / runs,
               "input_per_run": tin / runs, "output_per_run": tout / runs,
               "cached_share": tcached / tin if tin else 0.0,
               "concurrent_calls": sum(c.concurrent for c in cs)}
        if prices:
            row["usd_per_run"] = prices.usd(tin, tout, tcached) / runs
        rows.append(row)
    return rows


RUN_METRICS = ("calls", "sessions", "input_tokens", "output_tokens", "tokens_per_session",
               "calls_per_session")


def llm_section(runs: Sequence[Path], prices: Prices | None = None,
                modes: dict[str, str] | None = None,
                concurrent: Collection[tuple[str, str]] = ()) -> dict[str, Any]:
    """Normal-operation figures per memory mode (per_run_summary[mode]), never pooled."""
    calls = collect_calls(runs, modes, concurrent)
    per_run = run_rows(calls)
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in per_run:
        by_mode[r["memory_mode"]].append(r)
    order = sorted(by_mode, key=mode_sort)
    summary = {mode: {m: describe(r[m] for r in by_mode[mode]) for m in RUN_METRICS}
               for mode in order}
    out = {"families": family_rows(calls, prices),
           "run_families": {mode: sorted({c.family for c in calls
                                          if c.source == "run" and c.memory_mode == mode},
                                         key=_family_sort) for mode in order},
           "per_run": per_run, "per_run_summary": summary,
           "recovery_logs": recovery_log_rows(calls, prices),
           "gap_note": (f"time since the previous logged call in the same log, gaps over "
                        f"{MAX_GAP_S:.0f}s dropped; an upper bound on call latency only "
                        "where one thread wrote the log, so logs that concurrent replays "
                        "wrote (cf workers > 1) report none")}
    if prices and per_run:
        out["per_run_usd"] = {mode: describe(
            prices.usd(r["input_tokens"], r["output_tokens"], r["cached_tokens"])
            for r in by_mode[mode]) for mode in order}
    return out


# --- 3b. recovery cost from recover.py's records ---------------------------------------

def _storage_delta(rec: dict) -> int | None:
    m = _STORAGE_DELTA.search(rec.get("notes") or "")
    return int(m.group(1)) if m else None


def cf_replays(rec: dict) -> int | None:
    """Counterfactual replays one recovery ran: the T7 `cf_replays` field when the record
    has it, else the sum of every `replays=N` the incident rounds noted (+verify can run
    several)."""
    if rec.get("cf_replays") is not None:
        return int(rec["cf_replays"])
    found = _REPLAYS.findall(rec.get("notes") or "")
    return sum(int(n) for n in found) if found else None


def triggered(rec: dict) -> bool:
    """Was there an incident to investigate? A poisoned run whose logged action was
    judged harmful, or a clean run given a simulated false alarm. Counterfactual modes
    do nothing without one, so their per-incident cost is measured on these records."""
    if rec.get("condition") == "clean":
        return "simulated false alarm" in (rec.get("notes") or "")
    return bool(rec.get("incident"))


# recover.py's experiment knobs (T7) and the value each has in a record written before
# they existed or by a sweep that left it alone. k is absent: a record's k is the k its
# replays used, which is the run's own k unless the incident was observed fresh.
KNOBS = ("family", "k", "repeat", "depth", "cf_method", "cf_groups", "cf_workers", "observe")
KNOB_DEFAULTS = {"family": "recorded", "repeat": None, "depth": 0, "cf_method": "frozen",
                 "cf_groups": False, "cf_workers": 1, "observe": "logged"}


def knobs(rec: dict) -> tuple[tuple[str, Any], ...]:
    """The knobs a record departs from the defaults on, in KNOBS order: () for every
    record a default sweep wrote, old or new, so those still group and deduplicate
    together, while a k=3 and a k=10 measurement of one run stay two measurements."""
    out = []
    for name in KNOBS:
        value = rec.get(name)
        if value is None or (name in KNOB_DEFAULTS and value == KNOB_DEFAULTS[name]):
            continue
        if name == "k" and rec.get("observe") != "fresh":
            continue
        out.append((name, value))
    return tuple(out)


def cost_key(rec: dict) -> tuple:
    """What was done to the store, knobs included: measure.config_key plus knobs()."""
    import measure
    return (measure.config_key(rec), knobs(rec))


def cost_label(key: tuple) -> str:
    import measure
    base, kn = key
    return measure.config_label(base) + "".join(f" {k}={v}" for k, v in kn)


def record_key(rec: dict) -> tuple:
    """One measurement: measure.record_key with the knobs in the configuration."""
    return (rec.get("run_dir"), cost_key(rec), rec.get("checkpoint", 0),
            rec.get("condition", "poisoned"))


def load_records(path: Path) -> list[dict]:
    """measure.load_records (a repeated measurement counts once, the last one wins, in
    its original position; a torn line is skipped with a warning) with record_key above.

    The knobs are part of the key (as in measure.record_key), so a file holding several
    knob settings for one run (repeats, k, depth) keeps every setting; cost_key also
    groups the tables by them, which measure.py's tables do not.
    """
    rows = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"warning: {path}:{i} is not valid JSON, skipped")
    last = {record_key(r): i for i, r in enumerate(rows)}
    return [r for i, r in enumerate(rows) if last[record_key(r)] == i]


def load_checkpoint0(paths: Sequence[Path]) -> list[dict]:
    """Records (load_records), checkpoint 0 only: later reinfection checkpoints repeat
    the same recovery cost and would weigh it several times."""
    out = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        for rec in load_records(path):
            if rec.get("checkpoint", 0) == 0:
                out.append({**rec, "_records": path.name})
    return out


COST_METRICS = ("llm_calls", "input_tokens", "output_tokens", "seconds")


def recovery_rows(records: Sequence[dict], prices: Prices | None = None,
                  modes: dict[str, str] | None = None) -> list[dict]:
    """Recovery-only cost per memory mode, configuration (knobs included) and condition
    (a clean run is the false-alarm control). recover.py meters the recovery step and
    +verify rounds, not the replays that measure the result. Calls, tokens, time and
    replays are reported over the records with an incident to investigate (see
    triggered); `mean_calls_all` and `storage_delta` cover every record."""
    import measure
    modes = modes or {}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        mode = modes.get(r.get("run_dir"), "unknown")
        groups[(r["_records"], mode, cost_key(r), r.get("condition", "poisoned"))].append(r)
    order = measure.MODE_ORDER

    def sort_key(k):
        mode = k[2][0][0]
        return (k[0], mode_sort(k[1]), k[3] != "poisoned",
                order.index(mode) if mode in order else len(order), k[2])

    rows = []
    for key in sorted(groups, key=sort_key):
        rs = groups[key]
        hit = [r for r in rs if triggered(r)]
        row = {"records": key[0], "memory_mode": key[1], "config": cost_label(key[2]),
               "base": list(key[2][0]), "knobs": dict(key[2][1]),
               "condition": key[3], "runs": len(rs), "incidents": len(hit),
               "mean_calls_all": float(np.mean([r.get("llm_calls") or 0 for r in rs])),
               **{m: describe((r.get(m) or 0) for r in hit) for m in COST_METRICS},
               "cf_replays": describe(cf_replays(r) for r in hit),
               "storage_delta": describe(_storage_delta(r) for r in rs)}
        if any("cf_seconds" in r for r in hit):              # T7 records, when present
            row["cf_seconds"] = describe(r.get("cf_seconds") for r in hit)
        if prices:
            row["usd"] = describe(prices.usd(r.get("input_tokens") or 0,
                                             r.get("output_tokens") or 0) for r in hit)
        rows.append(row)
    return rows


# Each pair differs only in where the verdicts come from, so the paired difference per
# run is what the Section 5.4 counterfactual diagnosis itself costs.
DIAGNOSIS_PAIRS = (
    (("selective", "cf", (), "patched"), ("selective", "oracle", (), "patched")),
    (("unmerge", "cf", (), "patched"), ("unmerge", "oracle", (), "patched")),
)


# Knobs only a counterfactual has: an oracle record carries None for them.
CF_KNOBS = ("cf_method", "cf_groups", "cf_workers")


def diagnosis_rows(records: Sequence[dict], pairs=DIAGNOSIS_PAIRS,
                   modes: dict[str, str] | None = None) -> list[dict]:
    """Paired (cf minus oracle) cost per incident run: the counterfactual's own price.

    A cf record pairs with the oracle record of the same run and records file under the
    same knobs, the counterfactual's own options aside, so a k=3 counterfactual is never
    priced against a k=10 oracle. Rows are per memory mode and cf knob setting."""
    modes = modes or {}
    by = {(r["_records"], r["run_dir"], cost_key(r)): r for r in records
          if r.get("condition", "poisoned") == "poisoned" and triggered(r)}
    rows = []
    for cf, oracle in pairs:
        diffs: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for (src, run_dir, (base, kn)), r in by.items():
            shared = tuple((k, v) for k, v in kn if k not in CF_KNOBS)
            partner = by.get((src, run_dir, (oracle, shared)))
            if base != cf or partner is None:
                continue
            group = diffs[(modes.get(run_dir, "unknown"), kn, shared)]
            for m in COST_METRICS:
                group[m].append((r.get(m) or 0) - (partner.get(m) or 0))
        for (mode, kn, shared) in sorted(diffs, key=lambda g: (mode_sort(g[0]), g[1])):
            d = diffs[(mode, kn, shared)]
            rows.append({"memory_mode": mode, "cf": cost_label((cf, kn)),
                         "oracle": cost_label((oracle, shared)), "knobs": dict(kn),
                         "runs": len(d["seconds"]),
                         **{m: describe(v) for m, v in d.items()}})
    return rows


def recovery_section(paths: Sequence[Path], prices: Prices | None = None,
                     run_names: Iterable[str] | None = None,
                     modes: dict[str, str] | None = None) -> dict[str, Any]:
    """run_names, when given, keeps only those runs' records (an empty list keeps none),
    so the cost tables describe the same runs as the storage and log tables. modes maps
    a run folder to its memory mode; records of a run it lacks are 'unknown'."""
    records = load_checkpoint0(paths)
    if run_names is not None:
        keep = set(run_names)
        records = [r for r in records if r.get("run_dir") in keep]
    return {"records": [str(p) for p in paths], "n_records": len(records),
            "by_config": recovery_rows(records, prices, modes),
            "diagnosis": diagnosis_rows(records, modes=modes)}


# --- report -------------------------------------------------------------------------------

def md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _num(v: float | None, digits: int = 0) -> str:
    if v is None:
        return "-"
    return f"{v:,.{digits}f}"


def _spread(d: dict | None, digits: int = 0, scale: float = 1.0, unit: str = "") -> str:
    """median [q1, q3]"""
    if not d:
        return "-"
    f = lambda v: f"{v * scale:,.{digits}f}{unit}"
    return f"{f(d['median'])} [{f(d['q1'])}, {f(d['q3'])}]"


def _pct(d: dict | None, digits: int = 1) -> str:
    return _spread(d, digits, 100, "%")


def _ms(d: dict | None, key: str) -> str:
    if not d or key not in d:
        return "-"
    v = d[key]
    return f"{v:.3f}" if v < 1 else f"{v:.2f}" if v < 100 else f"{v:.0f}"


def storage_md(storage: dict) -> list[str]:
    md = ["## 1. Storage overhead (run.db, read-only)\n"]
    if not storage["runs"]:
        return md + ["No run databases matched.\n"]
    md.append(f"Method: {', '.join(storage['methods'])} (dbstat = page-exact bytes per table "
              "and index; estimate = used pages shared out by logical size). Cells are the "
              "median over runs [interquartile range]. Memory modes (read from each run's "
              "recorded config) are never pooled; each is also split by condition.\n")
    groups = list(storage["summary"])
    fmt = {"n": _spread, "x": lambda d: _spread(d, 2), "%": _pct}
    md.append(md_table(["metric", *[f"{g} (n={storage['summary'][g]['file_bytes']['n']})"
                                    for g in groups]],
                       [[label, *[fmt[kind](storage["summary"][g][m]) for g in groups]]
                        for m, label, kind in STORAGE_METRICS]))
    md.append("\nWhere the bytes of a run.db go, per memory mode. Allocated = whole pages, as "
              "on disk (the memories table's page slack is shared out in proportion to "
              "payload); payload = the stored values and record headers only. Every table "
              "and index holds at least one 4 KB page, so on stores this small the allocated "
              "view is dominated by fixed per-table pages; payload shows what grows with the "
              "data. `memory_meta` is every memories column other than text and embedding "
              "(id, status, source, operation, prompt hash, model, timestamps) plus the id "
              "index.\n")
    for mode in storage["modes"]:
        md += _storage_mode_md(storage, mode)
    return md


def _storage_mode_md(storage: dict, mode: str) -> list[str]:
    """The category, table and index breakdowns for one memory mode."""
    s = storage["summary"][mode]
    runs = [r for r in storage["runs"] if r["memory_mode"] == mode]
    md = [f"\n### {mode} memory (n={len(runs)})\n"]
    md.append(md_table(
        ["category", "share of file (allocated)", "allocated bytes per run",
         "share of payload", "payload bytes per run"],
        [[c, _pct(s["share"][c]), _spread(describe(r["categories"][c] for r in runs)),
          _pct(s["payload_share"][c]),
          _spread(describe(r["payload_categories"][c] for r in runs))] for c in CATEGORIES]))
    md.append(f"\nBytes per table, indexes included ({mode}, median [IQR]):\n")
    md.append(md_table(["table", "role", "allocated bytes per run", "share of file",
                        "payload bytes per run"],
                       [[t, _role(t), _spread(d),
                         _pct(describe(r["tables"].get(t, 0.0) / r["file_bytes"] for r in runs)),
                         _spread(s["table_payload"].get(t))]
                        for t, d in sorted(s["tables"].items(),
                                           key=lambda kv: -kv[1]["median"])]))
    if s["indexes"]:
        md.append(f"\nBytes per index ({mode}, median [IQR]; each is also counted in its "
                  "table's row above; `sqlite_autoindex_*` are the primary-key indexes "
                  "sqlite builds itself):\n")
        md.append(md_table(["index", "table", "allocated bytes per run",
                            "payload bytes per run"],
                           [[n, d["table"], _spread(d["allocated"]), _spread(d["payload"])]
                            for n, d in sorted(s["indexes"].items(),
                                               key=lambda kv: (kv[1]["table"], kv[0]))]))
    return md


def _role(table: str) -> str:
    if table == MEMORY_TABLE:
        return "memory"
    if table == LINEAGE_TABLE:
        return "provenance (lineage)"
    if table in PROVENANCE_TABLES:
        return "provenance"
    return "sqlite" if table.startswith("sqlite_") else "other"


def _load(load: list[float] | None) -> str:
    return " / ".join(f"{v:.2f}" for v in load) if load else "n/a"


def latency_md(lat: dict | None) -> list[str]:
    md = ["## 2. Latency (offline, synthetic stores)\n"]
    if not lat:
        return md + ["Skipped (--skip-latency).\n"]
    st, mach = lat["settings"], lat["machine"]
    md.append(f"Machine: {mach['cpu']}, {mach['cores']} cores, Python {mach['python']}, "
              f"SQLite {mach['sqlite']}, NumPy {mach['numpy']}. Load average (1 / 5 / 15 "
              f"min) {_load(st.get('loadavg_before'))} before the benchmark and "
              f"{_load(st.get('loadavg_after'))} after it. Each cell: median / p95 over "
              f"{st['repeats']} timed calls after {st['warmup']} warm-up calls, in "
              f"milliseconds. Stores are sqlite files with the real schema, {st['dim']}-d "
              f"random unit float32 embeddings and {st['text_bytes']}-byte texts, written "
              f"to a temp folder under {st['workdir']} and deleted afterwards; k = "
              f"{st['k']}; the query embedder returns a constant vector, so the encoder "
              "is not in these numbers. Calls are interleaved: each round times every "
              "store size and part once, in a shuffled order, so load from elsewhere on "
              "the machine falls on every size alike rather than on one.\n")
    if st.get("busy"):
        md.append(f"**Warning: the machine was busy** (a 1-minute load average above "
                  f"{st['busy_load_threshold']}, a quarter of the cores). Other work "
                  "competed with these timings, so they overstate the store's own cost and "
                  "should be re-measured on an idle machine.\n")
    if st.get("ram_workdir"):
        md.append(f"**Warning: the benchmark stores were on {st['workdir_fs']}**, a RAM "
                  "filesystem, so every retrieval_events commit skipped the disk sync a real "
                  "run pays: the write and end-to-end figures are too low. Re-run with "
                  "--workdir on the disk the runs use.\n")
    md.append(md_table(
        ["memories", "retrieve() end to end", "load active rows", "score + top-k",
         "retrieval_events write", "parts summed (medians)", "write share of median"],
        [[f"{r['n']:,}", *[f"{_ms(r[p], 'median_ms')} / {_ms(r[p], 'p95_ms')}"
                           for p in RETRIEVAL_PARTS],
          _ms(r, "parts_ms"), f"{100 * r['log_share']:.0f}%"] for r in lat["retrieval"]]))
    md.append("\nThe three parts are timed on their own, outside retrieve(); their medians "
              "need not add up to the end-to-end median, and this benchmark does not "
              "attribute the difference. The retrieval_events write is its own committed "
              "transaction (default rollback journal), so retrieve() pays one commit per "
              "call.\n")
    enc = lat["encoder"]
    if "skipped" in enc:
        md.append(f"\nReal query encode ({enc['model']}): skipped - {enc['skipped']}.\n")
    else:
        md.append(f"\nReal query encode ({enc['model']}, local cache, HF_HUB_OFFLINE=1, "
                  f"one query, CPU): median {_ms(enc, 'median_ms')} ms, p95 "
                  f"{_ms(enc, 'p95_ms')} ms over {enc['samples']} calls. A real retrieval costs "
                  "this plus the end-to-end figure above.\n")
    md.append(f"\nLineage queries on synthetic shapes (chain: depth n-1; balanced "
              f"{st['tree_branching']}-ary tree: depth ~log n). Descendants run from the "
              "root, ancestors from the deepest node; `found` is the result size.\n")
    md.append(md_table(["shape", "memories", "query", "found", "median ms", "p95 ms"],
                       [[r["shape"], f"{r['n']:,}", r["query"], f"{r['found']:,}",
                         _ms(r, "median_ms"), _ms(r, "p95_ms")]
                        for r in lat["lineage_synthetic"]]))
    real = lat.get("lineage_real") or {}
    rows = [[mode, q, d["samples"], _ms(d, "median_ms"), _ms(d, "p95_ms"), _spread(d["found"]),
             _num(d["found"]["max"])]
            for mode, queries in real.items() for q, d in queries.items()]
    if rows:
        md.append("\nLineage queries on the real stores (every memory of every selected "
                  "run, once each, per memory mode; a flat store has no edges, so its "
                  "results are empty):\n")
        md.append(md_table(["memory", "query", "calls", "median ms", "p95 ms",
                            "result size median [IQR]", "max"], rows))
    return md


def _gap(g: dict | None) -> str:
    return f"{g['median']:.2f} / {g['p95']:.2f}" if g else "-"


def llm_md(llm: dict, prices: Prices | None) -> list[str]:
    md = ["## 3. Model calls and tokens (from the logs)\n"]
    if not llm["families"]:
        return md + ["No llm.jsonl logs found for the selected runs.\n"]
    md.append("`run` = the runs' own llm.jsonl (normal operation); `recovery` = "
              "recovery/*.llm.jsonl, which holds recovery calls together with the replays "
              "and probes that measured each recovery. `memory` = the run's memory mode. "
              "Tokens are as the API reported them; cached tokens are a subset of input "
              f"tokens. Time per call is the {llm['gap_note']}.\n")
    header = ["log", "memory", "operation", "calls", "runs", "input tokens", "output tokens",
              "input / call", "output / call", "cached share", "time / call s (median / p95)"]
    if prices:
        header.append("USD (user prices)")
    rows = []
    for r in llm["families"]:
        row = [r["source"], r["memory_mode"], r["family"], f"{r['calls']:,}", r["runs"],
               f"{r['input_tokens']:,}", f"{r['output_tokens']:,}",
               _num(r["input_per_call"]), _num(r["output_per_call"], 1),
               f"{100 * r['cached_share']:.1f}%", _gap(r["gap_s"])]
        if prices:
            row.append(f"{r['usd']:,.2f}")
        rows.append(row)
    md.append(md_table(header, rows))
    n_concurrent = sum(r.get("concurrent_calls", 0) for r in llm["families"])
    if n_concurrent:
        md.append(f"\n{n_concurrent:,} of these calls come from logs that concurrent "
                  "counterfactual replays wrote together; they count toward calls and "
                  "tokens but give no time per call.\n")
    seen = "; ".join(f"{mode}: {', '.join(fams)}" for mode, fams in llm["run_families"].items())
    md.append(f"\nOperations seen in normal-operation logs ({seen}). Lineage edges, retrieval "
              "events and the action log are plain SQL writes, so MemHeal's provenance adds "
              "no model call to normal operation; every call above under `run` is one the "
              "agent or its memory makes anyway.\n")
    summary = llm["per_run_summary"]
    modes = list(summary)
    md.append("\nNormal operation per run, per memory mode (median [IQR] over runs):\n")
    rows = [[label, *[_spread(summary[mode][m], 1 if "per_session" in m else 0)
                      for mode in modes]] for m, label in (
        ("sessions", "sessions"), ("calls", "model calls"),
        ("calls_per_session", "calls per session"), ("input_tokens", "input tokens"),
        ("output_tokens", "output tokens"), ("tokens_per_session", "tokens per session"))]
    if prices and llm.get("per_run_usd"):
        rows.append(["USD per run (user prices)",
                     *[_spread(llm["per_run_usd"].get(mode), 3) for mode in modes]])
    md.append(md_table(["per run", *[f"{mode} (n={summary[mode]['calls']['n']})"
                                     for mode in modes]], rows))
    if llm["recovery_logs"]:
        md.append("\nRecovery experiment logs per configuration file (recovery plus its "
                  "measurement replays, probes and any reinfection sessions; mean per run "
                  "that has the file):\n")
        header = ["memory", "config file", "runs", "calls / run", "input / run",
                  "output / run", "cached share"]
        if prices:
            header.append("USD / run (user prices)")
        rows = []
        for r in llm["recovery_logs"]:
            row = [r["memory_mode"], r["config"] + (" (concurrent)" if r.get("concurrent_calls")
                                                    else ""),
                   r["runs"], _num(r["calls_per_run"], 1),
                   _num(r["input_per_run"]), _num(r["output_per_run"]),
                   f"{100 * r['cached_share']:.1f}%"]
            if prices:
                row.append(f"{r['usd_per_run']:.4f}")
            rows.append(row)
        md.append(md_table(header, rows))
    return md


def recovery_md(rec: dict, prices: Prices | None) -> list[str]:
    md = ["## 4. Recovery and diagnosis cost (recover.py records, checkpoint 0)\n"]
    if not rec["by_config"]:
        return md + ["No recovery records found for the selected runs.\n"]
    md.append(f"{rec['n_records']} records of the selected runs from "
              f"{', '.join(Path(p).name for p in rec['records'])} (a repeated measurement "
              "counts once, as in measure.py; records that differ in an experiment knob - "
              "family, k, repeat, depth, cf method, groups, workers, observe - are separate "
              "configurations, and the knobs away from their defaults follow the label). "
              "Calls, tokens and seconds cover the recovery step and any +verify rounds, "
              "not the replays that measured the result. `incidents` = poisoned runs whose "
              "logged action was judged harmful, or clean runs given a simulated false "
              "alarm (the control); the per-incident columns (median [IQR]) are over those "
              "records only, since counterfactual modes do nothing without one. `mean "
              "calls, all runs` and `rows added` (memories recovery wrote: "
              "reconstructions; quarantine deletes nothing) cover every record. `cf "
              "replays` = counterfactual replays.\n")
    extras = [x for x in ("cf_seconds",) if any(x in r for r in rec["by_config"])]
    multi = len({r["records"] for r in rec["by_config"]}) > 1
    header = [*(["records"] if multi else []), "memory", "configuration", "condition", "runs",
              "incidents", "mean calls, all runs", "LLM calls / incident",
              "input tokens / incident", "output tokens / incident", "seconds / incident",
              "cf replays / incident", *[f"{x} / incident" for x in extras], "rows added"]
    if prices:
        header.append("USD / incident (user prices, input at full price)")
    rows = []
    for r in rec["by_config"]:
        row = [*([r["records"]] if multi else []), r["memory_mode"], r["config"],
               r["condition"], r["runs"],
               r["incidents"], _num(r["mean_calls_all"], 1), _spread(r["llm_calls"]),
               _spread(r["input_tokens"]), _spread(r["output_tokens"]),
               _spread(r["seconds"], 2), _spread(r["cf_replays"]),
               *[_spread(r.get(x), 1) for x in extras], _spread(r["storage_delta"])]
        if prices:
            row.append(_spread(r.get("usd"), 4))
        rows.append(row)
    md.append(md_table(header, rows))
    if rec["diagnosis"]:
        md.append("\nCounterfactual diagnosis alone: per incident run, the paired difference "
                  "between the same recovery with counterfactual verdicts and with oracle "
                  "verdicts under the same knobs (median [IQR]).\n")
        md.append(md_table(
            ["memory", "difference", "runs", "LLM calls", "input tokens", "output tokens",
             "seconds"],
            [[d["memory_mode"], f"{d['cf']} - {d['oracle']}", d["runs"],
              _spread(d["llm_calls"]), _spread(d["input_tokens"]),
              _spread(d["output_tokens"]), _spread(d["seconds"], 2)]
             for d in rec["diagnosis"]]))
    return md


def _no_oracle(row: dict) -> bool:
    """The default incident[cf]+verify configuration on poisoned runs with incidents."""
    mode, verdicts, mods, _ = row["base"]
    return (mode == "incident" and verdicts == "cf" and list(mods) == ["verify"]
            and not row["knobs"] and row["condition"] == "poisoned" and row["incidents"])


def headline(report: dict) -> list[str]:
    """The few numbers Section 8.10's claim turns on, stated once at the top, per
    memory mode (the default knobs only, where recovery records have several)."""
    lines = []
    st = report["storage"]
    for mode in st.get("modes", []):
        s = st["summary"][mode]
        lines.append(
            f"- Storage ({mode}, {s['file_bytes']['n']} runs): a run.db is "
            f"{_spread(s['file_bytes'])} bytes over "
            f"{_spread(s['n_memories'])} memories ({_spread(s['bytes_per_memory'])} bytes per "
            f"memory). Provenance tables take {_pct(s['provenance_share'])} of the file's "
            f"pages ({_pct(s['provenance_share_with_meta'])} counting memory-row metadata), "
            f"mostly fixed one-page-per-table overhead; of the stored payload they are "
            f"{_pct(s['provenance_payload_share'])}. Lineage is "
            f"{_spread(s['edges_per_memory'], 2)} edges and "
            f"{_spread(s['lineage_payload_per_memory'])} payload bytes per memory "
            f"({_spread(s['lineage_bytes_per_memory'])} allocated). Per-session snapshot "
            f"copies add {_spread(s['snapshot_ratio'], 1)} times the run.db.")
    lat = report.get("latency")
    if lat:
        by_n = {r["n"]: r for r in lat["retrieval"]}
        pick = [n for n in (30, 1000, 10000) if n in by_n] or list(by_n)[:1]
        parts = [f"{_ms(by_n[n]['retrieve'], 'median_ms')} ms at {n:,}" for n in pick]
        caveats = {"busy": " Measured on a busy machine (see Section 2): re-measure before "
                           "quoting.",
                   "ram_workdir": " Measured on a RAM filesystem, without disk syncs (see "
                                  "Section 2): too low."}
        busy_note = "".join(note for flag, note in caveats.items()
                            if lat["settings"].get(flag))
        lines.append(f"- Retrieval (store side, median): {', '.join(parts)} memories; the "
                     f"retrieval_events write is part of that.{busy_note}")
        enc = lat["encoder"]
        if "skipped" not in enc:
            lines.append(f"- One MiniLM query encode: median {_ms(enc, 'median_ms')} ms.")
    for mode, s in report["llm"]["per_run_summary"].items():
        lines.append(f"- Normal operation ({mode}, {s['calls']['n']} runs): "
                     f"{_spread(s['calls'])} model calls per run over "
                     f"{_spread(s['sessions'])} sessions, {_spread(s['tokens_per_session'])} "
                     "tokens per session; provenance adds no model call.")
    rec = report["recovery"]
    first: dict[str, dict] = {}
    for d in rec.get("diagnosis", []):
        if not d["knobs"]:
            first.setdefault(d["memory_mode"], d)
    for mode, d in first.items():
        lines.append(f"- Counterfactual diagnosis ({mode}; {d['cf']} minus {d['oracle']}, "
                     f"per incident): +{_spread(d['llm_calls'])} calls, "
                     f"+{_spread(d['input_tokens'])} input tokens, "
                     f"+{_spread(d['seconds'], 1)} s.")
    for r in rec.get("by_config", []):
        if _no_oracle(r):
            lines.append(f"- No-oracle recovery ({r['memory_mode']}; {r['config']}, per "
                         f"incident, {r['incidents']} incidents): {_spread(r['llm_calls'])} "
                         f"calls, {_spread(r['seconds'], 1)} s, {_spread(r['cf_replays'])} "
                         "counterfactual replays.")
    return lines


def write_markdown(report: dict, prices: Prices | None, path: Path) -> None:
    modes = report["storage"].get("modes") or []
    md = ["# MemHeal cost (proposal Section 8.10)\n",
          f"Generated by `cost.py` from {len(report['storage']['runs'])} runs "
          f"({', '.join(report['args']['runs'])}"
          + (f"; memory modes {', '.join(report['args']['modes'])}"
             if report["args"].get("modes") else "")
          + f"), by memory mode: {', '.join(modes) or 'none'}. Offline: no model call, no "
          "network.\n"]
    if prices:
        md.append(f"Dollar figures use USER-SUPPLIED prices, not a quote: input "
                  f"{prices.input} USD / 1M tokens, output {prices.output} USD / 1M, cached "
                  f"input {prices.cached if prices.cached is not None else prices.input} "
                  "USD / 1M.\n")
    md += ["## Key numbers\n", *headline(report), ""]
    md += storage_md(report["storage"])
    md += latency_md(report.get("latency"))
    md += llm_md(report["llm"], prices)
    md += recovery_md(report["recovery"], prices)
    path.write_text("\n".join(md))


# --- figures -----------------------------------------------------------------------------

def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8, which="major")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, which="both")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _end_labels(ax, items: list[tuple[float, float, str]], min_ratio: float = 1.5) -> None:
    """Direct labels right of each series' last point, nudged apart in log-y so two
    series that end close together stay readable."""
    placed: list[float] = []
    for x, y, text in sorted(items, key=lambda t: t[1]):
        y_label = y
        if placed and y_label < placed[-1] * min_ratio:
            y_label = placed[-1] * min_ratio
        placed.append(y_label)
        ax.annotate(text, (x, y), xytext=(x * 1.25, y_label), fontsize=8, color=INK,
                    va="center", arrowprops={"arrowstyle": "-", "color": GRID, "lw": 0.8})
    if placed:                                  # keep nudged labels inside the axes
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, max(hi, placed[-1] * 1.3))


def active_ranges(storage_rows: Sequence[dict]) -> dict[str, tuple[int, int]]:
    """(fewest, most) active memories per memory mode over the real runs."""
    by_mode: dict[str, list[int]] = defaultdict(list)
    for r in storage_rows:
        by_mode[r["memory_mode"]].append(r["n_active"])
    return {m: (min(by_mode[m]), max(by_mode[m])) for m in sorted(by_mode, key=mode_sort)}


def plot_retrieval(lat: dict, real_ranges: dict[str, tuple[int, int]] | None,
                   out: Path) -> Path:
    """real_ranges: active memories per memory mode in the real runs; the band spans them
    all and its label names the modes it covers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = lat["retrieval"]
    xs = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    if real_ranges:
        lo = max(min(r[0] for r in real_ranges.values()), 1)
        hi = max(max(r[1] for r in real_ranges.values()), lo)
        ax.axvspan(lo, hi, color=GRID, alpha=0.55, lw=0, zorder=0)
        ax.text((lo * hi) ** 0.5, 0.98,
                f"active memories in the\nreal runs ({', '.join(real_ranges)})", ha="center",
                va="top", fontsize=7.5, color=MUTED, transform=ax.get_xaxis_transform())
    series = (("retrieve() end to end", "retrieve", PALETTE[0], "o"),
              ("retrieval_events write alone", "log", PALETTE[1], "s"))
    labels = []
    for label, key, colour, marker in series:
        med = [r[key]["median_ms"] for r in rows]
        p95 = [r[key]["p95_ms"] for r in rows]
        ax.fill_between(xs, med, p95, color=colour, alpha=0.15, lw=0)
        ax.plot(xs, med, color=colour, marker=marker, ms=6, lw=2, label=f"{label} (median)",
                mec=SURFACE, mew=1.5)
        labels.append((xs[-1], med[-1], label))
    enc = lat.get("encoder") or {}
    if "median_ms" in enc:
        # Stops at the last size, so its direct label sits clear of the line.
        ax.plot([xs[0] / 1.5, xs[-1]], [enc["median_ms"]] * 2, color=PALETTE[2], ls="--",
                lw=1.5, label="one real MiniLM query encode (median)")
        labels.append((xs[-1], enc["median_ms"], f"MiniLM encode {enc['median_ms']:.1f} ms"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xs[0] / 1.5, xs[-1] * 6)
    _end_labels(ax, labels)
    ax.set_xlabel("memories in the store (log scale)", color=MUTED, fontsize=9)
    ax.set_ylabel("milliseconds per call (log scale)", color=MUTED, fontsize=9)
    ax.set_title(f"Retrieval latency vs store size (median; band to p95; "
                 f"{lat['settings']['repeats']} calls each)", fontsize=10.5, color=INK,
                 loc="left")
    # Below the axes: the band label owns the top of the plot.
    ax.legend(fontsize=8, frameon=False, loc="upper center", labelcolor=INK, ncol=3,
              bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    path = out / "retrieval_latency.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_lineage(lat: dict, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = lat["lineage_synthetic"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    series = ((("chain", "descendants"), PALETTE[0], "o"), (("chain", "ancestors"), PALETTE[1], "s"),
              (("tree", "descendants"), PALETTE[2], "D"), (("tree", "ancestors"), PALETTE[3], "^"))
    labels = []
    xs_all = []
    for (shape, query), colour, marker in series:
        rs = [r for r in rows if r["shape"] == shape and r["query"] == query]
        if not rs:
            continue
        xs = [r["n"] for r in rs]
        xs_all += xs
        med = [r["median_ms"] for r in rs]
        label = f"{shape}: {query} of {'root' if query == 'descendants' else 'deepest node'}"
        ax.plot(xs, med, color=colour, marker=marker, ms=6, lw=2, label=label,
                mec=SURFACE, mew=1.5)
        labels.append((xs[-1], med[-1], f"{shape} {query}"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    if xs_all:
        ax.set_xlim(min(xs_all) / 1.5, max(xs_all) * 6)
    _end_labels(ax, labels)
    ax.set_xlabel("memories in the lineage (log scale)", color=MUTED, fontsize=9)
    ax.set_ylabel("milliseconds per query, median (log scale)", color=MUTED, fontsize=9)
    ax.set_title("Lineage query latency (recursive CTE over memory_edges)", fontsize=10.5,
                 color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    path = out / "lineage_latency.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# --- main ----------------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS),
                    help="globs over run-folder names (default: the dense suite's evolving "
                         "and flat runs; a flat run's name has no memory suffix)")
    ap.add_argument("--modes", nargs="+", default=None,
                    help="keep only runs whose recorded memory mode is one of these "
                         "(default: every mode the globs select; modes are always reported "
                         "apart)")
    ap.add_argument("--records", nargs="+", default=[str(p) for p in DEFAULT_RECORDS],
                    help="recover.py records files (missing files are skipped)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--sizes", nargs="+", type=int, default=list(RETRIEVAL_SIZES))
    ap.add_argument("--lineage-sizes", nargs="+", type=int, default=list(LINEAGE_SIZES))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--workdir", default=None,
                    help="where the synthetic benchmark stores are written and then deleted "
                         "(default: a temp folder inside --out)")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-encoder", action="store_true",
                    help="do not time the MiniLM encoder (it is never downloaded either way)")
    ap.add_argument("--no-dbstat", action="store_true",
                    help="use the page-count estimate even when sqlite has dbstat")
    ap.add_argument("--price-in", type=float, default=None,
                    help="USD per 1M input tokens (user-supplied; no default)")
    ap.add_argument("--price-out", type=float, default=None,
                    help="USD per 1M output tokens (user-supplied; no default)")
    ap.add_argument("--price-cached", type=float, default=None,
                    help="USD per 1M cached input tokens (default: the --price-in price)")
    args = ap.parse_args(argv)
    if (args.price_in is None) != (args.price_out is None):
        ap.error("--price-in and --price-out go together")
    if args.price_cached is not None and args.price_in is None:
        ap.error("--price-cached needs --price-in and --price-out")
    return args


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    # Before anything can import huggingface_hub: the encoder must come from the cache.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    prices = (Prices(args.price_in, args.price_out, args.price_cached)
              if args.price_in is not None else None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = select_runs(Path(args.runs_dir), args.runs, args.modes)
    modes = run_modes(runs)
    counts = defaultdict(int)
    for m in modes.values():
        counts[m] += 1
    print(f"{len(runs)} runs match {' '.join(args.runs)} under {args.runs_dir} ("
          + ", ".join(f"{m}: {counts[m]}" for m in sorted(counts, key=mode_sort)) + ")")

    report: dict[str, Any] = {
        "generated_by": "cost.py",
        "args": {"runs_dir": args.runs_dir, "runs": args.runs, "modes": args.modes,
                 "records": args.records, "sizes": args.sizes,
                 "lineage_sizes": args.lineage_sizes, "repeats": args.repeats},
        "prices_usd_per_1m": asdict(prices) if prices else None,
    }
    t0 = time.perf_counter()
    report["storage"] = storage_section(runs, use_dbstat=not args.no_dbstat)
    print(f"storage: {len(runs)} run.db files ({time.perf_counter() - t0:.1f}s)")
    # Always filtered to the selected runs: an empty selection keeps no records, so every
    # table describes the same runs.
    names = [p.name for p in runs]
    records = [Path(p) for p in args.records]
    keep = set(names)
    flagged = concurrent_logs(r for r in load_checkpoint0(records) if r.get("run_dir") in keep)
    t0 = time.perf_counter()
    report["llm"] = llm_section(runs, prices, modes, flagged)
    print(f"logs: {sum(r['calls'] for r in report['llm']['families']):,} calls "
          f"({time.perf_counter() - t0:.1f}s)")
    report["recovery"] = recovery_section(records, prices, run_names=names, modes=modes)
    print(f"records: {report['recovery']['n_records']} at checkpoint 0")
    report["latency"] = None
    if not args.skip_latency:
        t0 = time.perf_counter()
        report["latency"] = latency_section(runs, args.sizes, args.lineage_sizes, args.repeats,
                                            encoder=not args.skip_encoder,
                                            workdir=Path(args.workdir) if args.workdir else out,
                                            modes=modes)
        report["latency"]["active_memories"] = {
            m: list(r) for m, r in active_ranges(report["storage"]["runs"]).items()}
        st = report["latency"]["settings"]
        print(f"latency benchmark ({time.perf_counter() - t0:.1f}s; load average "
              f"{_load(st['loadavg_before'])} before, {_load(st['loadavg_after'])} after)")
        if st["busy"]:
            print(f"warning: 1-minute load average above {st['busy_load_threshold']}: the "
                  "machine was busy, so the latency figures include contention; re-measure "
                  "on an idle machine")
        if st["ram_workdir"]:
            print(f"warning: the benchmark stores were on {st['workdir_fs']} (RAM): commits "
                  "skipped the disk sync; re-run with --workdir on the runs' disk")

    (out / "cost.json").write_text(json.dumps(report, indent=1, default=str))
    write_markdown(report, prices, out / "cost_tables.md")
    made = [out / "cost.json", out / "cost_tables.md"]
    if report["latency"]:
        made.append(plot_retrieval(report["latency"],
                                   active_ranges(report["storage"]["runs"]) or None, out))
        made.append(plot_lineage(report["latency"], out))
    print("wrote " + ", ".join(str(p) for p in made))
    for line in headline(report):
        print(line)
    return report


if __name__ == "__main__":
    main()
