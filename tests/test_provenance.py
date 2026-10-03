"""Provenance graph tests using the proposal's Section 2 running example.

M1: reimbursement portal = Concur        (user)
M2: receipt format = PDF                 (user)
M3: portal = quick-reimburse.example     (webpage)  <- poison root
M4: upload PDFs via quick-reimburse      (derived, parents M2, M3)
M5: travel receipts should be PDFs       (derived, parents M2, M4)
"""

import pytest

from memheal import db


@pytest.fixture
def graph(tmp_path):
    conn = db.connect(tmp_path / "prov.db")
    m = {}
    m[1] = db.insert_memory(conn, "Reimbursement portal is Concur.", "user", 1, memory_id="M1")
    m[2] = db.insert_memory(conn, "Receipts are PDFs.", "user", 4, memory_id="M2")
    m[3] = db.insert_memory(conn, "Reimbursement portal is quick-reimburse.example.", "webpage", 7, memory_id="M3")
    m[4] = db.insert_memory(
        conn, "Upload PDF receipts through quick-reimburse.example.", "derived", 10,
        operation="summarize", parent_ids=["M2", "M3"], memory_id="M4",
    )
    m[5] = db.insert_memory(
        conn, "Travel receipts should be PDFs.", "derived", 14,
        operation="summarize", parent_ids=["M2", "M4"], memory_id="M5",
    )
    yield conn
    conn.close()


def test_descendants(graph):
    assert db.get_descendants(graph, "M3") == {"M4", "M5"}
    assert db.get_descendants(graph, "M2") == {"M4", "M5"}
    assert db.get_descendants(graph, "M4") == {"M5"}
    assert db.get_descendants(graph, "M1") == set()
    assert db.get_descendants(graph, "M5") == set()


def test_ancestors(graph):
    assert db.get_ancestors(graph, "M5") == {"M2", "M3", "M4"}
    assert db.get_ancestors(graph, "M4") == {"M2", "M3"}
    assert db.get_ancestors(graph, "M1") == set()
    assert db.get_ancestors(graph, "M3") == set()


def test_diamond_is_deduplicated(graph):
    # M5 is reachable from M2 directly and via M4; the CTE must return it once.
    rows = graph.execute(
        """
        WITH RECURSIVE d(id) AS (
            SELECT child_id FROM memory_edges WHERE parent_id = 'M2'
            UNION
            SELECT e.child_id FROM memory_edges e JOIN d ON e.parent_id = d.id
        ) SELECT id FROM d
        """
    ).fetchall()
    assert sorted(r["id"] for r in rows) == ["M4", "M5"]


def test_duplicate_edges_ignored(graph):
    db.insert_edges(graph, "M4", ["M2", "M3"])
    n = graph.execute("SELECT COUNT(*) FROM memory_edges WHERE child_id = 'M4'").fetchone()[0]
    assert n == 2


def test_cycle_terminates(graph):
    # Not expected in practice, but a bad consolidation must not hang the query.
    db.insert_edges(graph, "M3", ["M5"])
    assert db.get_descendants(graph, "M3") == {"M4", "M5"}
    assert db.get_ancestors(graph, "M3") == {"M2", "M4", "M5"}


def test_snapshot_is_isolated(graph, tmp_path):
    snap_path = tmp_path / "snap.db"
    db.snapshot(graph, snap_path)
    db.insert_memory(graph, "Later memory.", "user", 21, memory_id="M6")

    snap = db.connect(snap_path)
    assert {r["id"] for r in db.all_memories(snap)} == {"M1", "M2", "M3", "M4", "M5"}
    assert db.get_descendants(snap, "M3") == {"M4", "M5"}
    assert db.get_memory(graph, "M6") is not None
    snap.close()
