"""cost.py (proposal Section 8.10): storage, latency and token accounting.

Every store is a temp sqlite file built with the real schema, every log and records file
is synthetic, and the encoder is a stub, so these tests pin the arithmetic offline: no
model, no download, no socket.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import types

import numpy as np
import pytest

import cost
from memheal import db, memory

DIM = cost.DIM


@pytest.fixture
def no_network(monkeypatch):
    attempts = []

    def blocked(*args, **kwargs):
        attempts.append(args)
        raise OSError("network access blocked in test")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    return attempts


def _vec(i: int) -> bytes:
    v = np.random.default_rng(i).standard_normal(DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tobytes()


def _usage(tin: int, tout: int, cached: int = 0) -> dict:
    return {"input_tokens": tin, "output_tokens": tout,
            "input_tokens_details": {"cached_tokens": cached}}


def make_run(root, name, *, n_extract=6, condition="poisoned", snapshots=2, log=None,
             recovery_logs=None, memory_mode=None, lineage=True):
    """A finished run folder: run.db with memories, a consolidation summary (lineage),
    a retrieval, actions and the run row; snapshot copies; llm.jsonl.

    memory_mode goes into the run's recorded agent config, as run_demo writes it; None
    leaves it out, as the oldest runs did. lineage=False skips the summary and its edges,
    as a flat store has none."""
    run_dir = root / name
    (run_dir / "snapshots").mkdir(parents=True)
    conn = db.connect(run_dir / "run.db")
    agent_cfg = {"k": 5, **({"memory_mode": memory_mode} if memory_mode else {})}
    db.insert_run(conn, f"{name}-id", "case", condition, "fake-model", {"agent": agent_cfg})
    db.insert_source(conn, "s1:user", "user", "Use Concur for expenses.", "trusted", 1)
    prompt = db.register_prompt(conn, "extract.default", "extract the facts")
    ids = [db.insert_memory(conn, f"fact number {i} about the expense workflow", "user", 1,
                            source_id="s1:user", created_by_prompt=prompt, model="fake-model",
                            embedding=_vec(i)) for i in range(n_extract)]
    summary = ids[0]
    if lineage:
        summary = db.insert_memory(conn, "merged fact", "derived", 2, operation="summarize",
                                   embedding=_vec(99), parent_ids=ids[:3])
        db.set_status_many(conn, ids[:3], "consolidated")
    db.log_retrieval(conn, f"{name}-id", "s3", 3, [summary, ids[3]], [0.9, 0.5])
    db.log_action(conn, f"{name}-id", 3, 0, "respond", {"text": "ok"}, "ok")
    for s in range(snapshots):
        db.snapshot(conn, run_dir / "snapshots" / f"after_s{s:02d}.db")
    conn.close()
    if log is not None:
        (run_dir / "llm.jsonl").write_text("".join(json.dumps(e) + "\n" for e in log))
    for cfg, entries in (recovery_logs or {}).items():
        (run_dir / "recovery").mkdir(exist_ok=True)
        (run_dir / "recovery" / f"{cfg}.llm.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in entries))
    return run_dir


# --- storage --------------------------------------------------------------------------

def test_categorize_splits_the_memories_table_by_payload():
    sizes = {"memories": (1000.0, 800.0), "sqlite_autoindex_memories_1": (100.0, 50.0),
             "memory_edges": (200.0, 100.0), "idx_edges_child": (50.0, 20.0),
             "sqlite_schema": (100.0, 10.0), "mystery": (40.0, 1.0)}
    owner = {"memories": "memories", "sqlite_autoindex_memories_1": "memories",
             "memory_edges": "memory_edges", "idx_edges_child": "memory_edges",
             "sqlite_schema": "sqlite_schema", "mystery": "mystery"}
    cats = cost.categorize(sizes, owner, text_bytes=200, embedding_bytes=400, freelist_bytes=30)
    # Text and embedding get their payload share of the table's pages (1000/800 fill).
    assert cats["content"] == pytest.approx(250)
    assert cats["embedding"] == pytest.approx(500)
    assert cats["memory_meta"] == pytest.approx(250 + 100)     # rest of the row + id index
    assert cats["provenance"] == pytest.approx(250)
    assert cats["other"] == pytest.approx(40)
    assert cats["sqlite"] == pytest.approx(130)                # schema page + free list
    assert sum(cats.values()) == pytest.approx(1490 + 30)


@pytest.mark.parametrize("use_dbstat", [True, False])
def test_storage_categories_add_up_to_the_file(tmp_path, use_dbstat):
    run = make_run(tmp_path, "0001_poisoned_office_x_dense_direct_evolving")
    rs = cost.run_storage(run, use_dbstat=use_dbstat)

    assert rs["file_bytes"] == (run / "run.db").stat().st_size
    assert sum(rs["categories"].values()) == pytest.approx(rs["file_bytes"], abs=1)
    assert sum(rs["tables"].values()) <= rs["file_bytes"]
    if use_dbstat:
        conn = db.connect_readonly(run / "run.db")
        has_dbstat = cost.dbstat_bytes(conn) is not None
        conn.close()
        assert rs["method"] == ("dbstat" if has_dbstat else "estimate")
    else:
        assert rs["method"] == "estimate"
    assert rs["n_memories"] == 7 and rs["n_active"] == 4 and rs["n_edges"] == 3
    assert rs["edges_per_memory"] == pytest.approx(3 / 7)
    assert rs["embedding_payload_bytes"] == 7 * DIM * 4
    assert rs["condition"] == "poisoned" and rs["memory_mode"] == "evolving"
    for c in ("content", "embedding", "memory_meta", "provenance", "sqlite"):
        assert rs["categories"][c] > 0, c
    assert rs["categories"]["other"] == 0
    if rs["method"] == "dbstat":
        # Allocated pages can only hold more than the exact payload, never less.
        assert rs["categories"]["embedding"] >= rs["embedding_payload_bytes"]
        assert rs["categories"]["content"] >= rs["text_payload_bytes"]
    assert 0 < rs["provenance_share"] < rs["provenance_share_with_meta"] < 1
    # The payload view carries text and embeddings at their exact byte counts.
    data = rs["payload_categories"]
    assert data["embedding"] == pytest.approx(rs["embedding_payload_bytes"], abs=0.1)
    assert data["content"] == pytest.approx(rs["text_payload_bytes"], abs=0.1)
    assert data["memory_meta"] > 0 and data["provenance"] > 0
    assert 0 < rs["provenance_payload_share"] < 1
    assert rs["lineage_payload_per_edge"] > 0
    assert sum(rs["table_payload"].values()) == pytest.approx(sum(data.values()), abs=1)


def test_storage_never_writes_to_the_run(tmp_path):
    run = make_run(tmp_path, "0001_poisoned_office_x_dense_direct_evolving")
    before = hashlib.sha256((run / "run.db").read_bytes()).hexdigest()
    cost.run_storage(run)
    cost.run_storage(run, use_dbstat=False)
    assert hashlib.sha256((run / "run.db").read_bytes()).hexdigest() == before
    assert sorted(p.name for p in run.iterdir()) == ["run.db", "snapshots"]


def test_storage_counts_snapshots_and_lineage(tmp_path):
    run = make_run(tmp_path, "0001_poisoned_office_x_dense_direct_evolving", snapshots=3)
    rs = cost.run_storage(run)
    snaps = list((run / "snapshots").glob("*.db"))
    assert rs["n_snapshots"] == 3
    assert rs["snapshot_bytes"] == sum(p.stat().st_size for p in snaps)
    assert rs["snapshot_ratio"] == pytest.approx(rs["snapshot_bytes"] / rs["file_bytes"])
    assert rs["lineage_bytes"] == rs["tables"]["memory_edges"] > 0
    assert rs["lineage_bytes_per_memory"] == pytest.approx(rs["lineage_bytes"] / 7)


def test_storage_section_groups_by_mode_and_condition(tmp_path):
    a = make_run(tmp_path, "0001_poisoned_office_x_dense_direct_evolving")
    b = make_run(tmp_path, "0002_clean_office_x_dense_direct_evolving", condition="clean",
                 n_extract=8)
    st = cost.storage_section([a, b])
    assert list(st["summary"]) == ["evolving", "evolving/clean", "evolving/poisoned"]
    assert st["modes"] == ["evolving"]
    assert st["summary"]["evolving"]["n_memories"]["n"] == 2
    assert st["summary"]["evolving/clean"]["n_memories"]["median"] == 9
    assert st["summary"]["evolving/poisoned"]["n_memories"]["median"] == 7
    shares = st["summary"]["evolving"]["share"]
    assert sum(shares[c]["median"] for c in cost.CATEGORIES) == pytest.approx(1, abs=0.05)


def test_flat_runs_named_like_run_demo_are_selected_and_kept_apart(tmp_path):
    """run_demo adds no memory suffix for flat, so `..._dense_composed` is a flat run. It
    must be selected by default, labelled flat (not 'composed'), and never pooled with the
    evolving runs, whose lineage it does not have."""
    ev = make_run(tmp_path, "0001_poisoned_office_x_dense_direct_evolving",
                  memory_mode="evolving")
    flat = make_run(tmp_path, "1198_clean_travel_car_hire_dense_composed", condition="clean",
                    memory_mode="flat", lineage=False)
    # Not the dense suite's default selection: adaptive shares the infix; t1 and mem0.
    make_run(tmp_path, "0003_poisoned_office_x_dense_adaptive_evade_evolving",
             memory_mode="evolving")
    make_run(tmp_path, "0004_poisoned_office_x_dense_direct_evolving_t1", memory_mode="evolving")
    mem0 = make_run(tmp_path, "0005_poisoned_office_x_dense_procedural_mem0",
                    memory_mode="mem0")

    assert [p.name for p in cost.select_runs(tmp_path, cost.DEFAULT_RUNS)] == [ev.name,
                                                                                 flat.name]
    assert cost.run_memory_mode(flat) == "flat"
    assert cost.run_storage(flat)["memory_mode"] == "flat"
    assert cost.select_runs(tmp_path, ["*_dense_*"], modes=["flat"]) == [flat]
    assert mem0 not in cost.select_runs(tmp_path, ["*_dense_*"], modes=["evolving", "flat"])

    st = cost.storage_section([ev, flat])
    assert list(st["summary"]) == ["evolving", "evolving/poisoned", "flat", "flat/clean"]
    assert st["modes"] == ["evolving", "flat"]
    assert st["summary"]["flat"]["edges_per_memory"]["median"] == 0
    assert st["summary"]["evolving"]["edges_per_memory"]["median"] == pytest.approx(3 / 7)
    md = "\n".join(cost.storage_md(st))
    assert "### evolving memory (n=1)" in md and "### flat memory (n=1)" in md
    assert "composed" not in st["summary"]


def test_memory_mode_comes_from_the_run_not_its_name(tmp_path):
    # The recorded agent config wins over everything, the name included.
    a = make_run(tmp_path, "0001_clean_x_dense_composed_evolving", memory_mode="flat")
    assert cost.run_memory_mode(a) == "flat"
    # Without it, summary.json.
    b = make_run(tmp_path, "0002_clean_x_dense_composed")
    (b / "summary.json").write_text(json.dumps({"memory_mode": "evolving"}))
    assert cost.run_memory_mode(b) == "evolving"
    # A run older than both: a mode suffix in the name, and none means flat.
    assert cost.run_memory_mode(make_run(tmp_path, "0003_clean_x_dense_direct_evolving")) \
        == "evolving"
    assert cost.run_memory_mode(make_run(tmp_path, "0004_clean_x_dense_procedural")) == "flat"
    assert cost.run_modes([a, b]) == {a.name: "flat", b.name: "evolving"}


def test_describe_and_select_runs(tmp_path):
    d = cost.describe([4, None, 1, 3, 2])
    assert (d["n"], d["median"], d["q1"], d["q3"]) == (4, 2.5, 1.75, 3.25)
    assert cost.describe([]) is None and cost.describe([None]) is None

    make_run(tmp_path, "0001_poisoned_office_x_dense_direct_evolving")
    make_run(tmp_path, "0002_poisoned_office_x_direct_evolving")
    (tmp_path / "0003_empty_dense_direct_evolving").mkdir()        # no run.db
    names = [p.name for p in cost.select_runs(tmp_path, cost.DEFAULT_RUNS)]
    assert names == ["0001_poisoned_office_x_dense_direct_evolving"]
    assert cost.select_runs(tmp_path / "missing", ["*"]) == []


def test_storage_reports_every_index_on_its_own(tmp_path):
    run = make_run(tmp_path, "0001_poisoned_office_x_dense_direct_evolving")
    rs = cost.run_storage(run)
    bt = rs["btrees"]
    assert bt["idx_edges_child"]["table"] == "memory_edges"
    assert bt["idx_retrieval_query"]["table"] == "retrieval_events"
    assert bt["memories"]["table"] == "memories"
    # The per-b-tree bytes are the per-table bytes before folding.
    for table, total in rs["tables"].items():
        assert sum(b["allocated"] for b in bt.values() if b["table"] == table) == \
            pytest.approx(total, abs=1)

    st = cost.storage_section([run])
    idx = st["summary"]["evolving"]["indexes"]
    assert {"idx_edges_child", "idx_retrieval_query"} <= set(idx)
    assert not {"memories", "memory_edges", "sqlite_schema", "sqlite_sequence"} & set(idx)
    assert idx["idx_edges_child"]["allocated"]["median"] > 0
    md = "\n".join(cost.storage_md(st))
    assert "Bytes per index" in md and "| idx_edges_child | memory_edges |" in md


# --- logs -------------------------------------------------------------------------------

def test_read_calls_families_sessions_and_gaps(tmp_path):
    log = tmp_path / "llm.jsonl"
    entries = [
        {"ts": 100.0, "tag": "agent:s1:step0", "usage": _usage(100, 10, cached=40)},
        {"ts": 101.5, "tag": "extract:s1:user", "usage": _usage(50, 5)},
        {"ts": 400.0, "tag": "agent:s2:step0", "usage": _usage(120, 12)},   # after a pause
        {"ts": 402.0, "tag": "consolidate:s2", "usage": None},
    ]
    log.write_text("".join(json.dumps(e) + "\n" for e in entries) + '{"ts": 403, "tag": "refl')
    calls = cost.read_calls(log, source="run", run_dir="r1")

    assert [c.family for c in calls] == ["agent", "extract", "agent", "consolidate"]
    assert [c.session for c in calls] == [1, 1, 2, 2]
    # The first call has no predecessor; a gap past MAX_GAP_S is a pause, not a call.
    assert [c.gap for c in calls] == [None, pytest.approx(1.5), None, pytest.approx(2.0)]
    assert (calls[0].input, calls[0].output, calls[0].cached) == (100, 10, 40)
    assert (calls[3].input, calls[3].output) == (0, 0)


def test_llm_section_separates_runs_from_recovery(tmp_path):
    run_log = [
        {"ts": 10.0, "tag": "agent:s1:step0", "usage": _usage(900, 40, cached=100)},
        {"ts": 11.0, "tag": "agent:s1:step1", "usage": _usage(1000, 30)},
        {"ts": 12.0, "tag": "extract:s1:user", "usage": _usage(150, 40)},
        {"ts": 14.0, "tag": "agent:s2:step0", "usage": _usage(950, 20)},
        {"ts": 17.0, "tag": "consolidate:s2", "usage": _usage(700, 300)},
    ]
    rec_log = [
        {"ts": 50.0, "tag": "agent:s20:step0", "usage": _usage(800, 10)},
        {"ts": 51.0, "tag": "agent:s20:step0", "usage": _usage(800, 10)},
        {"ts": 52.0, "tag": "reconstruct:reflect:m_1", "usage": _usage(200, 50)},
    ]
    run = make_run(tmp_path, "0001_poisoned_office_x_dense_direct_evolving", log=run_log,
                   recovery_logs={"incident+verify": rec_log})
    llm = cost.llm_section([run])

    fams = {(r["source"], r["family"]): r for r in llm["families"]}
    assert [r["source"] for r in llm["families"]][:3] == ["run"] * 3      # runs first
    agent = fams[("run", "agent")]
    assert (agent["calls"], agent["input_tokens"], agent["output_tokens"]) == (3, 2850, 90)
    assert agent["cached_share"] == pytest.approx(100 / 2850)
    assert agent["gap_s"]["median"] == pytest.approx(1.5)         # gaps 1.0 and 2.0
    assert fams[("run", "consolidate")]["gap_s"]["median"] == pytest.approx(3.0)
    assert fams[("recovery", "reconstruct")]["calls"] == 1
    assert llm["run_families"] == {"evolving": ["agent", "extract", "consolidate"]}
    assert "usd" not in agent                                      # no prices, no dollars
    assert {r["memory_mode"] for r in llm["families"]} == {"evolving"}

    (per_run,) = llm["per_run"]
    assert (per_run["calls"], per_run["sessions"]) == (5, 2)
    assert per_run["tokens_per_session"] == pytest.approx((3700 + 430) / 2)
    assert per_run["memory_mode"] == "evolving"
    (cfg,) = llm["recovery_logs"]
    assert (cfg["config"], cfg["runs"], cfg["calls_per_run"]) == ("incident+verify", 1, 3)
    assert cfg["memory_mode"] == "evolving" and cfg["concurrent_calls"] == 0


def test_llm_figures_keep_memory_modes_apart(tmp_path):
    """Normal-operation calls and tokens per run, the headline included, are per memory
    mode: a pooled median would describe neither architecture."""
    def log(n_sessions, tin):
        return [{"ts": 10.0 + i, "tag": f"agent:s{i + 1}:step0", "usage": _usage(tin, 10)}
                for i in range(n_sessions)]

    ev = make_run(tmp_path, "0001_poisoned_a_dense_direct_evolving", memory_mode="evolving",
                  log=log(3, 1000))
    flat = make_run(tmp_path, "0002_poisoned_a_dense_direct", memory_mode="flat",
                    lineage=False, log=log(3, 400))
    modes = cost.run_modes([ev, flat])
    llm = cost.llm_section([ev, flat], modes=modes)

    s = llm["per_run_summary"]
    assert list(s) == ["evolving", "flat"]
    assert s["evolving"]["tokens_per_session"]["median"] == pytest.approx(1010)
    assert s["flat"]["tokens_per_session"]["median"] == pytest.approx(410)
    assert {(r["memory_mode"], r["family"]) for r in llm["families"]} == {
        ("evolving", "agent"), ("flat", "agent")}

    report = {"storage": cost.storage_section([ev, flat]), "llm": llm, "latency": None,
              "recovery": cost.recovery_section([], run_names=[ev.name, flat.name],
                                                modes=modes)}
    lines = cost.headline(report)
    assert any(line.startswith("- Normal operation (evolving, 1 runs): 3") for line in lines)
    assert any(line.startswith("- Normal operation (flat, 1 runs): 3") for line in lines)
    assert any(line.startswith("- Storage (flat, 1 runs)") for line in lines)
    assert not any("(all" in line for line in lines)
    md = "\n".join(cost.llm_md(llm, None))
    assert "| per run | evolving (n=1) | flat (n=1) |" in md


def test_concurrent_replay_logs_give_no_time_per_call(tmp_path):
    """Several replays appending to one log interleave, so the time since the previous
    line is not a latency bound; such logs keep their calls and tokens but no gap."""
    assert not cost.interleaved(["agent:s1:step0", "agent:s1:step1", "extract:s1:user",
                                 "agent:s2:step0", "agent:s2:step0", "agent:s2:step1"])
    assert cost.interleaved(["agent:s20:step0", "agent:s20:step0", "agent:s20:step1",
                             "agent:s20:step1"])
    assert cost.interleaved(["agent:s20:step0", "agent:s21:step1"])

    def entries(tags, spacing):
        return [{"ts": 10.0 + spacing * i, "tag": t, "usage": _usage(100, 5)}
                for i, t in enumerate(tags)]

    seq = entries(["agent:s20:step0", "agent:s20:step1", "agent:s20:step0"], 1.0)
    mixed = entries(["agent:s20:step0"] * 3 + ["agent:s20:step1"] * 3, 0.2)
    step0_only = entries(["agent:s20:step0"] * 6, 0.1)      # no trace in the log itself
    run = make_run(tmp_path, "0001_poisoned_a_dense_direct_evolving", recovery_logs={
        "selective-cf": seq, "incident+verify@groups": mixed, "incident@groups": step0_only})
    # The records name the configuration that ran with workers; the log name lacks them.
    flagged = cost.concurrent_logs([
        {**record(run.name, "incident", "cf"), "cf_workers": 6},
        {**record(run.name, "selective", "cf"), "cf_workers": 1},
    ])
    assert flagged == {(run.name, "incident")}

    by: dict = {}
    for c in cost.collect_calls([run], concurrent=flagged):
        by.setdefault(c.config, []).append(c)
    assert [c.gap for c in by["selective-cf"]] == [None, pytest.approx(1.0), pytest.approx(1.0)]
    assert not any(c.concurrent for c in by["selective-cf"])
    for cfg in ("incident+verify@groups", "incident@groups"):     # found / flagged
        assert all(c.gap is None and c.concurrent for c in by[cfg]), cfg

    llm = cost.llm_section([run], concurrent=flagged)
    logs = {r["config"]: r for r in llm["recovery_logs"]}
    assert logs["incident@groups"]["concurrent_calls"] == 6
    assert logs["selective-cf"]["concurrent_calls"] == 0
    (agent,) = [r for r in llm["families"] if r["source"] == "recovery"]
    assert agent["gap_s"]["n"] == 2 and agent["concurrent_calls"] == 12
    md = "\n".join(cost.llm_md(llm, None))
    assert "incident@groups (concurrent)" in md and "12 of these calls" in md


def test_prices_are_explicit_and_charge_cached_input():
    assert cost.Prices(1.0, 2.0).usd(1e6, 1e6) == pytest.approx(3.0)
    # Cached tokens are part of input; without a cached price they cost full input price.
    assert cost.Prices(1.0, 2.0).usd(1e6, 1e6, 4e5) == pytest.approx(3.0)
    assert cost.Prices(1.0, 2.0, cached=0.5).usd(1e6, 1e6, 4e5) == pytest.approx(2.8)
    rows = cost.family_rows([cost.Call("run", "", "r", "agent", 1, 1000, 100, 0, None)],
                            cost.Prices(1.0, 2.0))
    assert rows[0]["usd"] == pytest.approx((1000 + 200) / 1e6)


def test_price_flags_must_come_together():
    with pytest.raises(SystemExit):
        cost.parse_args(["--price-in", "1"])
    with pytest.raises(SystemExit):
        cost.parse_args(["--price-cached", "0.5"])
    assert cost.parse_args([]).price_in is None


# --- recovery records -------------------------------------------------------------------

def record(run_dir, mode, verdicts=None, *, modifiers=(), checkpoint=0, condition="poisoned",
           incident=True, calls=0, tin=0, tout=0, seconds=0.05, notes="storage_delta=+0 memories"):
    return {"run_dir": run_dir, "variant": "v", "domain": "office", "category": "direct",
            "condition": condition, "mode": mode, "verdicts": verdicts,
            "modifiers": list(modifiers), "source": "patched", "checkpoint": checkpoint,
            "task_outcomes": ["correct"], "task_destinations": [None], "asr": 0.0, "br": 1.0,
            "probes": [], "n_quarantined": 0, "n_restored": 0, "n_reconstructed": 0,
            "collateral": {}, "residual_attack": 0, "incident": incident,
            "root_was_consolidated": False, "verify_rounds": None, "llm_calls": calls,
            "input_tokens": tin, "output_tokens": tout, "seconds": seconds, "notes": notes}


def write_records(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return path


def recovery_fixture(tmp_path):
    A, B, C = "0001_poisoned_a", "0002_poisoned_b", "0003_clean_c"
    return write_records(tmp_path / "records.jsonl", [
        record(A, "selective", "oracle"),
        record(A, "selective", "cf", calls=99, seconds=99.0),            # superseded below
        record(A, "selective", "cf", calls=6, tin=6000, tout=300, seconds=5.0,
               notes="selective: replays=2; storage_delta=+0 memories"),
        record(A, "incident", "cf", modifiers=("verify",), calls=24, tin=24000, tout=1200,
               seconds=24.0, notes="incident: inferred_roots=1 replays=6; verify: "
               "incident->incident; incident: inferred_roots=1 replays=4; "
               "storage_delta=+1 memories"),
        record(A, "selective", "oracle", checkpoint=3, calls=50),          # reinfection
        record(B, "selective", "oracle", incident=False),
        record(B, "selective", "cf", incident=False, seconds=0.07),
        record(C, "incident", "cf", modifiers=("verify",), condition="clean", incident=False,
               calls=30, tin=30000, tout=1300, seconds=26.0,
               notes="control: simulated false alarm on x.example; incident: "
               "inferred_roots=1 replays=8; storage_delta=+0 memories"),
    ])


def test_recovery_rows_are_per_incident(tmp_path):
    rec = cost.recovery_section([recovery_fixture(tmp_path), tmp_path / "missing.jsonl"])
    rows = {(r["config"], r["condition"]): r for r in rec["by_config"]}

    cf = rows[("selective[cf] (patched)", "poisoned")]
    assert (cf["runs"], cf["incidents"]) == (2, 1)
    assert cf["mean_calls_all"] == pytest.approx(3.0)     # last record wins: (6 + 0) / 2
    assert cf["llm_calls"]["median"] == 6 and cf["seconds"]["median"] == 5.0
    assert cf["cf_replays"]["median"] == 2

    inc = rows[("incident[cf]+verify (patched)", "poisoned")]
    assert inc["cf_replays"]["median"] == 10            # both incident rounds
    assert inc["storage_delta"]["median"] == 1

    oracle = rows[("selective[oracle] (patched)", "poisoned")]
    assert oracle["runs"] == 2                           # checkpoint 3 left out
    assert oracle["llm_calls"]["max"] == 0 and oracle["cf_replays"] is None

    clean = rows[("incident[cf]+verify (patched)", "clean")]
    assert clean["incidents"] == 1 and clean["llm_calls"]["median"] == 30
    assert list(rows)[-1][1] == "clean"                  # poisoned rows first

    (diag,) = rec["diagnosis"]
    assert diag["runs"] == 1
    assert diag["llm_calls"]["median"] == 6
    assert diag["seconds"]["median"] == pytest.approx(5.0 - 0.05)


def test_recovery_section_keeps_only_selected_runs(tmp_path):
    rec = cost.recovery_section([recovery_fixture(tmp_path)], run_names=["0002_poisoned_b"],
                                modes={"0002_poisoned_b": "flat"})
    assert rec["n_records"] == 2
    assert {r["condition"] for r in rec["by_config"]} == {"poisoned"}
    assert {r["memory_mode"] for r in rec["by_config"]} == {"flat"}
    assert rec["diagnosis"] == []                        # b had no incident
    # An empty selection keeps no records, rather than switching the filter off.
    assert cost.recovery_section([recovery_fixture(tmp_path)], run_names=[])["n_records"] == 0


# The knob values a default T7 sweep writes, and the record of an oracle (no cf knobs).
DEFAULT_KNOBS = {"family": "recorded", "k": 5, "repeat": None, "depth": 0,
                 "cf_method": "frozen", "cf_groups": False, "cf_workers": 1,
                 "observe": "logged"}
ORACLE_KNOBS = {**DEFAULT_KNOBS, "cf_method": None, "cf_groups": None, "cf_workers": None}


def test_records_differing_in_a_knob_are_separate_measurements(tmp_path):
    """measure.py's key leaves the T7 knobs out; here a k=3 and a k=10 record of one run
    are two measurements, while a default re-run still supersedes an old record."""
    A = "0001_poisoned_a"
    fresh = {"observe": "fresh"}
    path = write_records(tmp_path / "k.jsonl", [
        record(A, "incident", "cf", calls=99),                        # old: no knob fields
        {**record(A, "incident", "cf", calls=20), **DEFAULT_KNOBS},   # same measurement, later
        {**record(A, "incident", "cf", calls=30), **DEFAULT_KNOBS, "k": 3, **fresh},
        {**record(A, "incident", "cf", calls=100), **DEFAULT_KNOBS, "k": 10, **fresh},
        {**record(A, "selective", "oracle"), **ORACLE_KNOBS},
        {**record(A, "selective", "oracle"), **ORACLE_KNOBS, "k": 3, **fresh},
        {**record(A, "selective", "cf", calls=7, seconds=4.0), **DEFAULT_KNOBS, "k": 3, **fresh},
        {**record(A, "unmerge", "oracle"), **ORACLE_KNOBS},
        {**record(A, "unmerge", "cf", calls=5, seconds=2.0), **DEFAULT_KNOBS, "cf_workers": 6,
         "cf_groups": True},
    ])
    assert [cost.knobs(r) for r in cost.load_records(path)][:3] == [
        (), (("k", 3), ("observe", "fresh")), (("k", 10), ("observe", "fresh"))]
    rec = cost.recovery_section([path])
    assert rec["n_records"] == 8                         # only the old incident record went

    rows = {r["config"]: r for r in rec["by_config"]}
    assert rows["incident[cf] (patched)"]["llm_calls"]["median"] == 20
    assert rows["incident[cf] (patched) k=3 observe=fresh"]["llm_calls"]["median"] == 30
    assert rows["incident[cf] (patched) k=10 observe=fresh"]["llm_calls"]["median"] == 100
    assert rows["incident[cf] (patched) k=3 observe=fresh"]["knobs"] == {"k": 3,
                                                                         "observe": "fresh"}
    assert "unmerge[cf] (patched) cf_groups=True cf_workers=6" in rows

    # Each counterfactual is priced against the oracle under the same knobs, its own
    # counterfactual options aside.
    diag = {d["cf"]: d for d in rec["diagnosis"]}
    assert set(diag) == {"selective[cf] (patched) k=3 observe=fresh",
                         "unmerge[cf] (patched) cf_groups=True cf_workers=6"}
    sel = diag["selective[cf] (patched) k=3 observe=fresh"]
    assert sel["oracle"] == "selective[oracle] (patched) k=3 observe=fresh"
    assert sel["llm_calls"]["median"] == 7
    assert diag["unmerge[cf] (patched) cf_groups=True cf_workers=6"]["oracle"] == \
        "unmerge[oracle] (patched)"
    md = "\n".join(cost.recovery_md(rec, None))
    assert "incident[cf] (patched) k=10 observe=fresh" in md


def test_cost_and_measure_share_one_knob_definition():
    """One knob table: a record that is a row of its own in measure.py's recovery tables
    is a row of its own here, under the same label."""
    import measure
    assert cost.knobs is measure.knobs and cost.CF_KNOBS is measure.CF_KNOBS
    assert not hasattr(cost, "KNOBS") and not hasattr(cost, "KNOB_DEFAULTS")
    A = "0001_poisoned_a"
    recs = [{**record(A, "incident", "cf"), **DEFAULT_KNOBS, "k": 3, "observe": "fresh",
             "cf_groups": True},
            {**record(A, "selective", "oracle"), **ORACLE_KNOBS, "repeat": 2},
            {**record(A, "none"), **ORACLE_KNOBS, "depth": 1, "observe": "fresh"}]
    for r in recs:
        assert cost.cost_label(cost.cost_key(r)) == measure.config_label(measure.setting_key(r))


def test_stop_after_action_is_a_cost_knob_from_the_field_or_the_note(tmp_path):
    """Replays cut at the first acting call make fewer calls: never pooled with full
    replays. Records written before the field existed carry recover.py's note."""
    A, B, C = "0001_poisoned_a", "0002_poisoned_b", "0003_poisoned_c"
    note = "replays stop at the first acting call; storage_delta=+0 memories"
    path = write_records(tmp_path / "s.jsonl", [
        {**record(A, "incident", "cf", calls=10), **DEFAULT_KNOBS, "stop_after_action": True},
        {**record(B, "incident", "cf", calls=12, notes=note), **DEFAULT_KNOBS},
        {**record(C, "incident", "cf", calls=40), **DEFAULT_KNOBS, "stop_after_action": False},
    ])
    rows = {r["config"]: r for r in cost.recovery_section([path])["by_config"]}
    assert set(rows) == {"incident[cf] (patched)",
                         "incident[cf] (patched) stop_after_action=True"}
    assert rows["incident[cf] (patched) stop_after_action=True"]["runs"] == 2
    assert rows["incident[cf] (patched)"]["llm_calls"]["median"] == 40


def test_cf_replays_prefers_the_record_field():
    assert cost.cf_replays({"cf_replays": 3, "notes": "replays=8"}) == 3
    assert cost.cf_replays({"notes": "incident: replays=8; incident: replays=2"}) == 10
    assert cost.cf_replays({"notes": "storage_delta=+0 memories"}) is None


def test_triggered():
    assert cost.triggered({"condition": "poisoned", "incident": True})
    assert not cost.triggered({"condition": "poisoned", "incident": False})
    assert cost.triggered({"condition": "clean", "notes": "control: simulated false alarm on a"})
    assert not cost.triggered({"condition": "clean", "notes": ""})


# --- latency ----------------------------------------------------------------------------

def test_synthetic_store_retrieves_and_logs(tmp_path):
    conn = cost.synthetic_store(tmp_path / "s.db", 50)
    emb = cost.ConstantEmbedder()
    results = memory.retrieve(conn, emb, "bench", 1, "anything", k=5)

    rows = db.active_memories(conn)
    assert len(rows) == 50
    matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    assert matrix.shape == (50, DIM)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5)
    scores = [s for _, s in results]
    assert len(results) == 5 and scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(float((matrix @ emb.encode(["q"])[0]).max()), abs=1e-6)
    assert len(db.get_retrievals(conn, "bench", "s1")) == 5    # the provenance write
    conn.close()


def test_retrieval_latency_rows(tmp_path):
    rows = cost.retrieval_latency([10, 40], repeats=3, workdir=tmp_path)
    assert [r["n"] for r in rows] == [10, 40]
    for r in rows:
        for part in ("retrieve", "load", "score", "log"):
            assert r[part]["samples"] == 3
            assert 0 < r[part]["median_ms"] <= r[part]["p95_ms"]
        assert r["log_share"] > 0
        assert r["parts_ms"] == pytest.approx(sum(r[p]["median_ms"]
                                                  for p in ("load", "score", "log")))
    assert list(tmp_path.iterdir()) == []                    # benchmark stores removed


def test_timings_interleave_sizes_and_parts():
    """One call of every (size, part) per round, in a reshuffled order, so a burst of
    load elsewhere cannot land on one size's samples alone."""
    calls = []
    fns = {key: (lambda key=key: calls.append(key)) for key in "abcd"}
    out = cost._interleaved(fns, repeats=5, warmup=1)
    assert all(len(v) == 5 for v in out.values())
    assert calls[:4] == list("abcd")                         # warm-up, once each
    rounds = [calls[4 + 4 * i: 8 + 4 * i] for i in range(5)]
    assert all(sorted(r) == list("abcd") for r in rounds)
    assert len({tuple(r) for r in rounds}) > 1               # reshuffled between rounds
    assert cost._interleaved(fns, repeats=5, warmup=0).keys() == out.keys()


def test_latency_records_the_load_and_flags_a_busy_machine(tmp_path, monkeypatch):
    run = make_run(tmp_path / "runs", "0001_poisoned_a_dense_direct_evolving")
    flat = make_run(tmp_path / "runs", "0002_poisoned_a_dense_direct", memory_mode="flat",
                    lineage=False)
    monkeypatch.setattr(cost.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(cost.os, "getloadavg", lambda: (9.5, 7.0, 6.0))
    lat = cost.latency_section([run, flat], [10], [5], repeats=2, encoder=False,
                               workdir=tmp_path)
    st = lat["settings"]
    assert st["loadavg_before"] == st["loadavg_after"] == [9.5, 7.0, 6.0]
    assert st["busy"] is True and st["busy_load_threshold"] == 4.0
    md = "\n".join(cost.latency_md(lat))
    assert "machine was busy" in md and "9.50 / 7.00 / 6.00" in md
    assert "page cache" not in md                            # an unmeasured explanation
    # Real-store lineage per memory mode: the flat store has no edges to follow.
    assert list(lat["lineage_real"]) == ["evolving", "flat"]
    assert lat["lineage_real"]["flat"]["ancestors"]["found"]["max"] == 0
    assert lat["lineage_real"]["evolving"]["ancestors"]["found"]["max"] == 3
    report = {"storage": {"modes": []}, "llm": {"per_run_summary": {}}, "latency": lat,
              "recovery": {}}
    assert any("busy machine" in line for line in cost.headline(report))

    monkeypatch.setattr(cost.os, "getloadavg", lambda: (0.5, 0.4, 0.3))
    lat = cost.latency_section([run], [10], [5], repeats=2, encoder=False, workdir=tmp_path)
    assert lat["settings"]["busy"] is False
    assert "machine was busy" not in "\n".join(cost.latency_md(lat))


def test_benchmark_on_a_ram_filesystem_is_flagged(tmp_path, monkeypatch):
    mounts = ("/dev/nvme0n1p5 / ext4 rw 0 0\n"
              "tmpfs /tmp tmpfs rw,nosuid 0 0\n"
              "/dev/sdb1 /mnt/my\\040disk ext4 rw 0 0\n")
    assert cost.fs_from_mounts("/home/u/repo/results/cost", mounts) == "ext4"
    assert cost.fs_from_mounts("/tmp/x/out", mounts) == "tmpfs"
    assert cost.fs_from_mounts("/tmpfoo", mounts) == "ext4"           # not under /tmp
    assert cost.fs_from_mounts("/mnt/my disk/out", mounts) == "ext4"
    assert cost.fs_from_mounts("/x", "") is None

    run = make_run(tmp_path / "runs", "0001_poisoned_a_dense_direct_evolving")
    for fs, flagged in (("tmpfs", True), ("ext4", False)):
        monkeypatch.setattr(cost, "filesystem", lambda path, fs=fs: fs)
        lat = cost.latency_section([run], [10], [5], repeats=1, encoder=False,
                                   workdir=tmp_path)
        assert lat["settings"]["workdir_fs"] == fs
        assert lat["settings"]["ram_workdir"] is flagged
        assert ("a RAM filesystem" in "\n".join(cost.latency_md(lat))) is flagged


def test_lineage_shapes_and_queries(tmp_path):
    conn, root, leaf = cost.lineage_store(tmp_path / "chain.db", "chain", 20)
    assert len(db.get_descendants(conn, root)) == 19
    assert len(db.get_ancestors(conn, leaf)) == 19
    conn.close()
    conn, root, leaf = cost.lineage_store(tmp_path / "tree.db", "tree", 15)
    assert len(db.get_descendants(conn, root)) == 14
    assert len(db.get_ancestors(conn, leaf)) == 3              # full binary tree, depth 3
    conn.close()
    with pytest.raises(ValueError):
        cost.lineage_store(tmp_path / "x.db", "star", 5)

    work = tmp_path / "w"
    work.mkdir()
    rows = cost.lineage_latency([7], repeats=2, workdir=work)
    found = {(r["shape"], r["query"]): r["found"] for r in rows}
    assert found == {("chain", "descendants"): 6, ("chain", "ancestors"): 6,
                     ("tree", "descendants"): 6, ("tree", "ancestors"): 2}
    assert all(r["n"] == 7 and r["samples"] == 2 for r in rows)


def test_real_lineage_latency_reads_every_memory(tmp_path):
    run = make_run(tmp_path, "0001_poisoned_office_x_dense_direct_evolving")
    real = cost.real_lineage_latency([run])
    assert real["descendants"]["samples"] == real["ancestors"]["samples"] == 7
    assert real["descendants"]["found"]["max"] == 1           # a parent of the summary
    assert real["ancestors"]["found"]["max"] == 3             # the summary's parents


def test_encoder_never_downloads(monkeypatch):
    seen = {}

    class NotCached:
        def __init__(self, name, **kwargs):
            seen.update(kwargs, name=name)
            raise OSError("model not in the local cache")

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=NotCached))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    out = cost.encoder_latency(repeats=2)

    assert out["skipped"].startswith("OSError: model not in the local cache")
    assert seen["local_files_only"] is True and seen["name"] == memory.EMBEDDING_MODEL
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_encoder_timed_when_cached(monkeypatch):
    class Cached:
        def __init__(self, name, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            return np.ones((len(texts), DIM), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=Cached))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    out = cost.encoder_latency(repeats=4)
    assert out["samples"] == 4 and out["median_ms"] >= 0 and "skipped" not in out


# --- the command ------------------------------------------------------------------------

def test_cli_end_to_end_offline(tmp_path, no_network, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    runs = tmp_path / "runs"
    log = [{"ts": 1.0 + i, "tag": f"agent:s{i}:step0", "usage": _usage(500, 20)}
           for i in range(1, 4)]
    a = make_run(runs, "0001_poisoned_a_dense_direct_evolving", log=log,
                 recovery_logs={"incident": log})
    make_run(runs, "0002_clean_b_dense_direct_evolving", condition="clean", log=log)
    make_run(runs, "0003_poisoned_c_direct_evolving", log=log)          # not selected
    f = make_run(runs, "0004_poisoned_a_dense_direct", memory_mode="flat", lineage=False,
                 log=log)                                               # flat: no suffix
    records = write_records(tmp_path / "records.jsonl", [
        record(a.name, "selective", "oracle"),
        record(a.name, "selective", "cf", calls=6, tin=6000, tout=300, seconds=5.0,
               notes="selective: replays=2; storage_delta=+0 memories"),
        record("0003_poisoned_c_direct_evolving", "selective", "oracle"),
        record(f.name, "selective", "oracle"),
    ])
    before = {p: p.read_bytes() for p in runs.rglob("*") if p.is_file()}
    out = tmp_path / "out"
    base = ["--runs-dir", str(runs), "--records", str(records), "--out", str(out)]

    report = cost.main(base + ["--sizes", "10", "20", "--lineage-sizes", "5", "--repeats", "2",
                               "--skip-encoder"])

    assert no_network == []
    assert {p: p.read_bytes() for p in runs.rglob("*") if p.is_file()} == before
    assert sorted(p.name for p in out.iterdir()) == [
        "cost.json", "cost_tables.md", "lineage_latency.png", "retrieval_latency.png"]
    assert [r["run_dir"] for r in report["storage"]["runs"]] == [
        "0001_poisoned_a_dense_direct_evolving", "0002_clean_b_dense_direct_evolving",
        "0004_poisoned_a_dense_direct"]
    assert report["storage"]["modes"] == ["evolving", "flat"]
    assert report["recovery"]["n_records"] == 3                       # run c filtered out
    assert {r["memory_mode"] for r in report["recovery"]["by_config"]} == {"evolving", "flat"}
    saved = json.loads((out / "cost.json").read_text())
    assert saved["prices_usd_per_1m"] is None
    assert saved["latency"]["encoder"]["skipped"] == "--skip-encoder"
    assert saved["latency"]["active_memories"] == {"evolving": [4, 4], "flat": [6, 6]}
    assert list(saved["latency"]["lineage_real"]) == ["evolving", "flat"]
    assert list(saved["llm"]["per_run_summary"]) == ["evolving", "flat"]
    md = (out / "cost_tables.md").read_text()
    for heading in ("## Key numbers", "## 1. Storage", "## 2. Latency", "## 3. Model calls",
                    "## 4. Recovery", "### evolving memory", "### flat memory"):
        assert heading in md
    assert "USD" not in md and "$" not in md

    # --modes narrows the selection by the recorded mode, not the name.
    report = cost.main(base + ["--runs", "*_dense_*", "--modes", "flat", "--skip-latency"])
    assert [r["run_dir"] for r in report["storage"]["runs"]] == ["0004_poisoned_a_dense_direct"]
    assert report["recovery"]["n_records"] == 1

    cost.main(base + ["--skip-latency", "--price-in", "1.5", "--price-out", "6"])
    md = (out / "cost_tables.md").read_text()
    assert "USER-SUPPLIED" in md and "USD" in md
    assert "Skipped (--skip-latency)" in md
    assert json.loads((out / "cost.json").read_text())["prices_usd_per_1m"]["input"] == 1.5


def test_cli_with_nothing_selected(tmp_path, no_network, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")              # main() forces these; restore
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    out = tmp_path / "out"
    report = cost.main(["--runs-dir", str(tmp_path / "none"), "--records",
                        str(tmp_path / "none.jsonl"), "--out", str(out), "--skip-latency"])
    assert report["storage"]["runs"] == [] and report["recovery"]["n_records"] == 0
    assert "No run databases matched" in (out / "cost_tables.md").read_text()

    # A records file with rows, but no run selected: the report must not describe runs
    # it was not generated from.
    runs = tmp_path / "runs"
    make_run(runs, "0001_poisoned_a_dense_direct_evolving")
    report = cost.main(["--runs-dir", str(runs), "--runs", "no_such_run_*", "--records",
                        str(recovery_fixture(tmp_path)), "--out", str(out), "--skip-latency"])
    assert report["storage"]["runs"] == []
    assert report["recovery"]["n_records"] == 0 and report["recovery"]["by_config"] == []
    assert cost.headline(report) == []
    md = (out / "cost_tables.md").read_text()
    assert "No recovery records found" in md and "incident[cf]" not in md
