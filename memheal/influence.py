"""Frozen-set counterfactual influence for recovery (proposal Section 5.4).

Which memory made the agent act for the attacker? Section 5.4 answers it by ablation:
freeze the target session's original retrieved set R, replay the task once with R (the
baseline), then once per candidate m with R minus m, and score m by how often removing
it changes the security decision. The set is frozen rather than re-retrieved because
re-retrieval lets the next-nearest memory slide into the gap, and then a flip (or a
non-flip) says as much about the neighbour as about m.

This module only measures. It never quarantines anything (memheal/recovery.py does,
from these scores) and it never writes a memory: every replay runs with store=False, so
the store a recovery mode then edits is the same store that was measured.

It also never reads the scenario's ground truth beyond scoring: the decision is scored
with run_demo.score_session, which is what the `incident` mode is allowed to observe.

frozen_ablation's options beyond Section 5.4 are all off by default, so a default call
measures exactly what it always did:

workers           - replays run concurrently, each thread with its own connection to the
                    same database file. Replays wait on the LLM, not the CPU, and they
                    are most of a recovery's cost (Section 8.10).
stop_after_action - a replay ends at its first acting call, the only call it is scored on.
method            - "reretrieve" is Section 9.3's comparison with the frozen set: remove
                    m from (a throwaway copy of) the store and retrieve normally, so
                    whatever slides into the gap is part of the measurement, and shown.
groups            - Section 12.3's redundancy tests. Two memories that each carry the
                    attack hide each other from single removals: keep-only-one replays
                    ask whether a memory suffices on its own, and when no single removal
                    flips, pair and all-candidate removals find the smallest set that does.
"""

from __future__ import annotations

import itertools
import sqlite3
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from memheal import agent, db, tools

HARMFUL = "attacker"
METHODS = ("frozen", "reretrieve")

# A removal set "flips" the decision when at least this fraction of its replays no
# longer make it: the bar recovery.CF_THRESHOLD sets for a counterfactual verdict.
FLIP_THRESHOLD = 0.5

# Concurrent replays share one database file, and SQLite lets one connection write at a
# time, so a writer has to wait for another's commit rather than fail. Each commit is a
# handful of rows; a minute is far beyond any real wait.
BUSY_TIMEOUT_MS = 60_000


class _CallCounter:
    """Delegating wrapper that counts respond() calls.

    The real LLM keeps no call counter (only FakeLLM does), and recovery cost is
    reported per mode (Section 8.7), so the replays count their own calls. Concurrent
    replays share one counter, so the count is taken under a lock.
    """

    def __init__(self, llm):
        self._llm = llm
        self.calls = 0
        self._lock = threading.Lock()

    def respond(self, *args, **kwargs):
        with self._lock:
            self.calls += 1
        return self._llm.respond(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._llm, name)


class _SerialEmbedder:
    """Delegating wrapper that lets one encode() run at a time.

    A sentence-transformers model shares one tokenizer, which is not safe to call from
    several threads at once. Only re-retrieval replays encode (a frozen replay never
    retrieves), once each, in milliseconds next to an LLM call.
    """

    def __init__(self, embedder):
        self._embedder = embedder
        self._lock = threading.Lock()

    def encode(self, texts):
        with self._lock:
            return self._embedder.encode(texts)

    def __getattr__(self, name):
        return getattr(self._embedder, name)


def observed_action(conn: sqlite3.Connection, run_id: str, scenario, condition: str,
                    session: int) -> dict[str, Any]:
    """The decision a session made, as the defender sees it in the action log.

    For `incident` recovery this is the starting point: the observed destination is the
    only attack detail the defender gets, so it comes from the log, not the scenario.
    """
    from run_demo import score_session   # run_demo imports the whole harness; keep it lazy

    return score_session(conn, run_id, scenario, condition, session)


def original_retrieved(conn: sqlite3.Connection, session: int,
                       run_id: str | None = None) -> list[str]:
    """The memory ids the original run retrieved for `session`, in rank order.

    Only the agent's own query (query_id 's<session>') counts. Frozen replays log under
    'frozen:s<session>' and counterfactual run ids start with a 'cf' prefix; neither is
    the original retrieval. A snapshot that has been replayed into (for example by a
    task-family replay, which does re-retrieve) can hold several runs' 's<session>'
    events, so without an explicit run_id the earliest-logged run wins: that is the run
    the snapshot was taken from.
    """
    query_id = f"s{session}"
    if run_id is None:
        first = conn.execute(
            "SELECT run_id FROM retrieval_events WHERE session = ? AND query_id = ?"
            " AND run_id NOT LIKE 'cf%' ORDER BY rowid LIMIT 1",
            (session, query_id),
        ).fetchone()
        if first is None:
            return []
        run_id = first["run_id"]
    rows = conn.execute(
        "SELECT memory_id FROM retrieval_events WHERE run_id = ? AND session = ?"
        " AND query_id = ? ORDER BY rank, rowid",
        (run_id, session, query_id),
    ).fetchall()
    return [r["memory_id"] for r in rows]


def _prefix_in_use(conn: sqlite3.Connection, prefix: str) -> bool:
    pattern = prefix.replace("%", r"\%").replace("_", r"\_") + ":%"
    for table in ("actions", "retrieval_events"):
        hit = conn.execute(
            f"SELECT 1 FROM {table} WHERE run_id LIKE ? ESCAPE '\\' LIMIT 1", (pattern,)
        ).fetchone()
        if hit:
            return True
    return False


def _free_prefix(conn: sqlite3.Connection, run_prefix: str) -> str:
    """run_prefix, or run_prefix2, run_prefix3, ... if an earlier ablation used it.

    score_session reads the FIRST acting call under a run id, so two ablations on one
    store (the +verify loop runs incident again after escalation) sharing run ids
    would silently score the second with the first's actions.
    """
    prefix, k = run_prefix, 1
    while _prefix_in_use(conn, prefix):
        k += 1
        prefix = f"{run_prefix}{k}"
    return prefix


def _replay(conn, scenario, condition, task, frozen, llm, embedder, cfg, pages,
            run_id, *, stop_after_action: bool = False) -> dict[str, Any]:
    """One non-storing replay of the target task, scored like the original run.

    frozen=None retrieves normally (the re-retrieval ablation). Besides the score it
    returns the ids the replay rendered and its wall time, which frozen_ablation keeps
    out of the public baseline dict.
    """
    from run_demo import score_session

    started = time.perf_counter()
    session = scenario.target_session
    ctx = tools.ToolContext(conn, run_id, session, pages)
    # Passed only when set, so a default replay calls run_session exactly as before.
    extra = {"stop_after_action": True} if stop_after_action else {}
    ran = agent.run_session(ctx, llm, embedder, cfg, task,
                            frozen=None if frozen is None else list(frozen), store=False,
                            **extra)
    scored = score_session(conn, run_id, scenario, condition, session)
    return {"outcome": scored["outcome"], "destination": scored["destination"],
            "used_tool": scored["used_tool"],
            "retrieved": [m for m, _ in getattr(ran, "retrieved", ())],
            "seconds": time.perf_counter() - started}


def _public(got: dict[str, Any]) -> dict[str, Any]:
    return {k: got[k] for k in ("outcome", "destination", "used_tool")}


def _repeats(got: dict[str, Any], flagged: str | None) -> bool:
    """Did a replay make the decision being explained? Normally that is a harmful one;
    for a flagged decision (the clean control's simulated false alarm, which no judge
    calls harmful) it is acting on the flagged destination again."""
    if flagged is None:
        return got["outcome"] == HARMFUL
    d = got["destination"]
    return d is not None and tools.normalize_url(d) == tools.normalize_url(flagged)


# --- removal sets and where their replays run --------------------------------------

@dataclass
class _Setup:
    """What every replay of one ablation shares; only the removal set varies."""
    scenario: Any
    condition: str
    task: str
    retrieved: list[str]
    llm: Any
    embedder: Any
    cfg: agent.AgentConfig
    pages: dict[str, str]
    prefix: str
    flagged: str | None
    stop_after_action: bool
    method: str
    n_ambiguous: int

    def replay(self, conn: sqlite3.Connection, removed: tuple[str, ...],
               run_id: str) -> dict[str, Any]:
        # Frozen: R minus the removal set, in R's order. Re-retrieval: the store `conn`
        # was opened on already has the set quarantined, so retrieval cannot see it.
        frozen = (None if self.method == "reretrieve"
                  else [r for r in self.retrieved if r not in removed])
        got = _replay(conn, self.scenario, self.condition, self.task, frozen, self.llm,
                      self.embedder, self.cfg, self.pages, run_id,
                      stop_after_action=self.stop_after_action)
        got["repeated"] = _repeats(got, self.flagged)
        return got


@dataclass
class _Trial:
    """One removal set and its replays, under run ids {prefix}:{label}:{i}.

    expect_flip is the caller's prior for a single candidate: a first replay that
    disagrees with it is where one sample is least trustworthy, so it gets
    n_ambiguous replays in total.
    """
    label: str
    removed: tuple[str, ...]
    n: int
    expect_flip: bool | None = None
    replays: list[dict[str, Any]] = field(default_factory=list)
    ambiguous: bool = False

    @property
    def score(self) -> float:
        return sum(not r["repeated"] for r in self.replays) / len(self.replays)


def _run_trial(conn: sqlite3.Connection, setup: _Setup, trial: _Trial) -> None:
    def run(i: int) -> None:
        trial.replays.append(setup.replay(conn, trial.removed, f"{setup.prefix}:{trial.label}:{i}"))

    for i in range(trial.n):
        run(i)
    first_flipped = not trial.replays[0]["repeated"]
    trial.ambiguous = trial.expect_flip is not None and first_flipped != trial.expect_flip
    if trial.ambiguous:
        for i in range(len(trial.replays), max(trial.n, setup.n_ambiguous)):
            run(i)


def _database_file(conn: sqlite3.Connection) -> str | None:
    """The file behind conn's main database; None for an in-memory store, which no
    other connection can open, so its frozen replays stay on conn one at a time."""
    for row in conn.execute("PRAGMA database_list").fetchall():
        if row[1] == "main":
            return row[2] or None
    return None


class _Stores:
    """Where each trial's replays run.

    frozen, one worker - the caller's connection, exactly as before.
    frozen, workers>1  - a fresh connection per trial to the caller's database file, so
                         the replays' rows land in the store the caller then reads.
    reretrieve         - a throwaway copy of the store (sqlite backup API) with the
                         trial's removal set quarantined. The caller's store is never
                         edited, and every copy is deleted afterwards. Copies are
                         separate files, so these run concurrently even when the
                         caller's store is in memory.
    """

    def __init__(self, conn: sqlite3.Connection, method: str, workers: int):
        self.conn = conn
        self.method = method
        self.workers = workers
        self.path = _database_file(conn) if method == "frozen" and workers > 1 else None
        self._tmp: tempfile.TemporaryDirectory | None = None
        self._base: Path | None = None

    def __enter__(self) -> Self:
        if self.method == "reretrieve":
            self._tmp = tempfile.TemporaryDirectory(prefix="memheal-reretrieve-")
            self._base = Path(self._tmp.name) / "base.db"
            self.conn.commit()
            db.snapshot(self.conn, self._base)
        return self

    def __exit__(self, *exc) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()

    @property
    def concurrent(self) -> bool:
        return self.workers > 1 and (self.method == "reretrieve" or self.path is not None)

    @contextmanager
    def open(self, trial: _Trial) -> Iterator[sqlite3.Connection]:
        if self.method == "reretrieve":
            yield from self._copy_without(trial.removed)
        elif self.path is None:
            yield self.conn
        else:
            c = db.connect(self.path)
            c.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
            try:
                yield c
            finally:
                c.close()

    def _copy_without(self, removed: tuple[str, ...]) -> Iterator[sqlite3.Connection]:
        # Copied from the base snapshot rather than from the caller's connection, which
        # belongs to the caller's thread.
        path = Path(self._tmp.name) / f"{uuid.uuid4().hex}.db"
        src = sqlite3.connect(self._base)
        try:
            db.snapshot(src, path)
        finally:
            src.close()
        c = db.connect(path)
        try:
            db.set_status_many(c, removed, "quarantined")
            yield c
        finally:
            c.close()
            path.unlink(missing_ok=True)


def _run_trials(stores: _Stores, setup: _Setup, trials: list[_Trial]) -> None:
    """Run every trial's replays, in order or concurrently; results stay on the trials.

    Run ids do not depend on the order, so concurrent results are the sequential ones.
    """
    def job(trial: _Trial) -> None:
        with stores.open(trial) as c:
            _run_trial(c, setup, trial)

    if not stores.concurrent or not trials:
        for t in trials:
            job(t)
        return
    # Workers cannot see writes the caller has not committed, and would wait on its
    # lock. A sequential replay commits them anyway at its first logged action.
    stores.conn.commit()
    with ThreadPoolExecutor(max_workers=min(stores.workers, len(trials)),
                            thread_name_prefix="cf-replay") as pool:
        for _ in pool.map(job, trials):
            pass
    stores.conn.commit()   # no read transaction may pin a view from before the replays


def _plan(trials: list[_Trial], done: dict[frozenset, _Trial]) -> list[_Trial]:
    """The trials whose removal set has not been replayed yet; the rest reuse the one
    that was. With three candidates, keep-only-one of each is exactly one of the pairs;
    with two, the group is the pair; with one, keep-only is the baseline."""
    fresh = []
    for t in trials:
        key = frozenset(t.removed)
        if key not in done:
            done[key] = t
            fresh.append(t)
    return fresh


def _fields(trial: _Trial, setup: _Setup, baseline: dict[str, Any],
            **extra: Any) -> dict[str, Any]:
    out = {
        "score": trial.score,
        "outcomes": [r["outcome"] for r in trial.replays],
        "destinations": [r["destination"] for r in trial.replays],
        "n": len(trial.replays),
        **extra,
    }
    if setup.method == "reretrieve":
        # Retrieval is deterministic for a fixed store and query, so the first replay's
        # set is every replay's. Substitutes are what slid into the gap.
        got = trial.replays[0]["retrieved"]
        out["retrieved"] = got
        out["substitutes"] = [m for m in got if m not in baseline["retrieved"]]
    return out


def _score_entry(trial: _Trial, setup: _Setup, baseline: dict[str, Any]) -> dict[str, Any]:
    return _fields(trial, setup, baseline, ambiguous=trial.ambiguous)


def _set_entry(removed: tuple[str, ...], trial: _Trial, setup: _Setup,
               baseline: dict[str, Any]) -> dict[str, Any]:
    return {"removed": list(removed), **_fields(trial, setup, baseline)}


def _keep_entry(removed: tuple[str, ...], trial: _Trial, setup: _Setup,
                baseline: dict[str, Any]) -> dict[str, Any]:
    out = _set_entry(removed, trial, setup, baseline)
    out["repeat"] = 1.0 - out["score"]
    out["sufficient"] = out["repeat"] >= FLIP_THRESHOLD
    return out


def culprit_sets(result: dict[str, Any]) -> list[list[str]]:
    """Minimal removal sets that flip the decision.

    Flipping singletons when there are any; else flipping pairs (minimal, since no
    single member flips); else the all-candidates group if it flips; else none.
    Pairs and the group exist only when the ablation ran with groups=True.
    """
    singles = [[m] for m, e in result.get("scores", {}).items()
               if e["score"] >= FLIP_THRESHOLD]
    if singles:
        return singles
    pairs = [p["removed"] for p in result.get("pairs") or () if p["score"] >= FLIP_THRESHOLD]
    if pairs:
        return pairs
    group = result.get("group")
    return [group["removed"]] if group and group["score"] >= FLIP_THRESHOLD else []


def _partition(candidates, retrieved: list[str]) -> tuple[list[str], list[str]]:
    in_r = set(retrieved)
    seen: set[str] = set()
    tested: list[str] = []
    untested: list[str] = []
    for m in candidates:
        if m in seen:
            continue
        seen.add(m)
        (tested if m in in_r else untested).append(m)
    return tested, untested


def frozen_ablation(
    conn: sqlite3.Connection,
    scenario,
    condition: str,
    task: str,
    retrieved: list[str],
    candidates,
    llm,
    embedder,
    cfg: agent.AgentConfig,
    pages: dict[str, str],
    *,
    prior: dict[str, bool] | None = None,
    n: int = 1,
    n_ambiguous: int = 3,
    run_prefix: str = "cf",
    flagged: str | None = None,
    workers: int = 1,
    stop_after_action: bool = False,
    method: str = "frozen",
    groups: bool = False,
) -> dict[str, Any]:
    """Section 5.4 influence of each candidate on the target task's security decision.

    retrieved  - the frozen set R (usually original_retrieved for the target session).
    candidates - memories to ablate; ones not in R cannot have influenced this decision
                 through retrieval, so they are reported under 'untested', not scored 0.
    prior      - id -> True if the caller expects removing it to flip the decision (e.g.
                 its lineage verdict). A first replay that disagrees is the case where
                 one sample is least trustworthy, so it gets n_ambiguous replays in total.

    score = fraction of a candidate's replays whose outcome is no longer 'attacker'.
    If the baseline replay with all of R is not 'attacker', there is no harmful decision
    to explain, so no candidate is replayed and 'baseline_not_harmful' is True.
    flagged    - explain acting on this destination instead of an 'attacker' outcome:
                 what a detector that raised a false alarm on it would measure.

    workers    - >1 runs the replays concurrently, each thread on its own connection to
                 conn's database file (sequentially when conn is in memory). Results
                 equal workers=1 apart from timing.
    stop_after_action - end each replay at its first acting call (agent.run_session).
    method     - "frozen" (Section 5.4) or "reretrieve" (Section 9.3): the baseline
                 retrieves normally from conn's store, and each removal set is
                 quarantined in a throwaway copy before retrieving normally at cfg.k.
                 Candidates are still partitioned by R, so both methods test the same
                 memories; entries add 'retrieved' and 'substitutes' (what slid into the
                 gap), and the result adds 'baseline_retrieved'.
    groups     - also replay keep-only-one (m kept, the other tested candidates removed,
                 the rest of R kept) and, when no single removal flips, every pair of
                 tested candidates and all of them together. Adds 'keep_only' (id ->
                 entry with 'repeat' and 'sufficient'), 'pairs' (list) and 'group'.

    Always added: 'method', 'culprit_sets' (see culprit_sets()), 'seconds' (wall time)
    and 'replay_seconds' (baseline first, then each replay in the order planned).
    """
    if method not in METHODS:
        raise ValueError(f"unknown ablation method {method!r}; expected one of {METHODS}")
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    started = time.perf_counter()
    counted = _CallCounter(llm)
    prefix = _free_prefix(conn, run_prefix)
    prior = prior or {}
    retrieved = list(retrieved)
    tested, untested = _partition(candidates, retrieved)
    setup = _Setup(scenario, condition, task, retrieved, counted,
                   _SerialEmbedder(embedder) if workers > 1 else embedder, cfg, pages,
                   prefix, flagged, stop_after_action, method, n_ambiguous)

    baseline = setup.replay(conn, (), f"{prefix}:baseline:0")
    result: dict[str, Any] = {
        "baseline": _public(baseline),
        "baseline_not_harmful": not baseline["repeated"],
        "scores": {},
        "untested": untested,
        "run_prefix": prefix,
        "llm_calls": 0,
        "replays": 1,
        "method": method,
    }
    if method == "reretrieve":
        result["baseline_retrieved"] = baseline["retrieved"]
    if groups:
        result.update(keep_only={}, pairs=[], group=None)
    seconds = [baseline["seconds"]]

    if not result["baseline_not_harmful"] and tested:
        reps = max(n, 1)
        done: dict[frozenset, _Trial] = {frozenset(): _Trial("baseline", (), 1, replays=[baseline])}
        singles = [_Trial(m, (m,), reps, bool(prior[m]) if m in prior else None)
                   for m in tested]
        keeps = ({m: _Trial(f"keep:{m}", tuple(x for x in tested if x != m), reps)
                  for m in tested} if groups else {})
        ran = _plan(singles + list(keeps.values()), done)
        with _Stores(conn, method, workers) as stores:
            _run_trials(stores, setup, ran)
            for m in tested:
                result["scores"][m] = _score_entry(done[frozenset((m,))], setup, baseline)
            if groups:
                result["keep_only"] = {m: _keep_entry(t.removed, done[frozenset(t.removed)],
                                                      setup, baseline)
                                       for m, t in keeps.items()}
                if not culprit_sets(result):
                    pairs = [_Trial(f"pair:{a}+{b}", (a, b), reps)
                             for a, b in itertools.combinations(tested, 2)]
                    group = _Trial("group", tuple(tested), reps)
                    more = _plan(pairs + [group], done)
                    _run_trials(stores, setup, more)
                    ran += more
                    result["pairs"] = [_set_entry(p.removed, done[frozenset(p.removed)],
                                                  setup, baseline) for p in pairs]
                    result["group"] = _set_entry(group.removed, done[frozenset(group.removed)],
                                                 setup, baseline)
        result["replays"] += sum(len(t.replays) for t in ran)
        seconds += [r["seconds"] for t in ran for r in t.replays]

    result["culprit_sets"] = culprit_sets(result)
    result["llm_calls"] = counted.calls
    result["replay_seconds"] = [round(s, 3) for s in seconds]
    result["seconds"] = round(time.perf_counter() - started, 3)
    return result


def influence_report(result: dict[str, Any]) -> list[tuple[str, float]]:
    """(id, score) pairs, most influential first; ties keep the order they were tested."""
    scores = result.get("scores", {})
    return sorted(((m, s["score"]) for m, s in scores.items()), key=lambda p: -p[1])
