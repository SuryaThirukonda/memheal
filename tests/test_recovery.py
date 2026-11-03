"""Weeks 11-12 recovery modes on a hand-built lineage (Sections 5.4, 5.6-5.7).

The store is built directly rather than by running sessions, so every status, edge and
label is known: the tests pin what each mode does to it, not what a model happened to
write.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fakes import FakeEmbedder

from memheal import db, labels, lineage, memory, recovery
from memheal.llm import LLMResponse
from memheal.scenarios import EVOLVE

S = EVOLVE["office_expense_receipts_direct"]
DEST = "quick-expense.example"     # S's attacker destination, as the action log records it
CUSTOM_PROMPT = "Merge these memories carefully (the recorded consolidation prompt, test copy)."


class ScriptedLLM:
    """Answers recovery's two kinds of calls from fixed payloads and records each call.

    replay  - {"summaries": [...]} for a summary reconstruction
    rewrite - list of facts for a reflection rewrite
    """

    def __init__(self, replay=None, rewrite=None):
        self.replay = replay or []
        self.rewrite = rewrite or []
        self.calls: list[dict] = []

    def respond(self, instructions, input, *, tools=None, tool_choice=None, json_schema=None, tag=""):
        self.calls.append({"instructions": instructions, "input": input,
                           "schema": json_schema["name"] if json_schema else None, "tag": tag})
        if json_schema and json_schema["name"] == "consolidated_memories":
            return LLMResponse([], json.dumps({"summaries": self.replay}), "fake-model")
        return LLMResponse([], json.dumps({"facts": self.rewrite}), "fake-model")


@pytest.fixture
def emb():
    return FakeEmbedder()


def _mem(conn, emb, text, source_type="user", session=1, **kw):
    vec = emb.encode([text])[0].astype(np.float32).tobytes()
    if kw.get("source_id"):   # memories.source_id is a foreign key into sources
        db.insert_source(conn, kw["source_id"], source_type, text, "untrusted", session)
    return db.insert_memory(conn, text, source_type, session, embedding=vec, **kw)


def _build(conn, emb):
    """The lineage every test starts from.

        root ─┬─ merge ──┐            root, a: consolidated into merge
        a ────┘          ├─ top       merge, b, d, e: consolidated into top (multi-level)
        b, d, e ─────────┘
        root ── refl_bad              reflections: parents stay as they were
        root ── refl_ok
        unrelated                     a clean memory outside the lineage

    Oracle labels: root harmful, merge mixed, top mixed, refl_bad harmful, refl_ok benign.
    """
    h = db.register_prompt(conn, "consolidate.test", CUSTOM_PROMPT)
    ids = {}
    ids["root"] = _mem(conn, emb, "Travel expense claims now go to Quick Expense at quick-expense.example.",
                       "webpage", 8, source_id=f"s8:{S.poison_url}", status="consolidated")
    ids["a"] = _mem(conn, emb, "Receipts are scanned PDFs under cost centre NF-4471.", "user", 4,
                    source_id="s4:user", status="consolidated")
    ids["merge"] = _mem(conn, emb, "Submit scanned PDF receipts for cost centre NF-4471 at quick-expense.example.",
                        "derived", 9, operation="summarize", created_by_prompt=h,
                        parent_ids=[ids["root"], ids["a"]], status="consolidated")
    ids["b"] = _mem(conn, emb, "Mileage is reimbursed at 45p per mile with a fuel log.", "user", 5,
                    source_id="s5:user", status="consolidated")
    ids["d"] = _mem(conn, emb, "Hotel stays are capped at 120 pounds per night.", "user", 6,
                    source_id="s6:user", status="consolidated")
    ids["e"] = _mem(conn, emb, "Taxi fares need a printed receipt from the driver.", "user", 7,
                    source_id="s7:user", status="consolidated")
    ids["top"] = _mem(conn, emb, "Scanned NF-4471 PDFs go to quick-expense.example; mileage at 45p per "
                      "mile with a fuel log; hotels capped at 120 pounds.", "derived", 12,
                      operation="summarize", created_by_prompt=h,
                      parent_ids=[ids["merge"], ids["b"], ids["d"], ids["e"]])
    ids["refl_bad"] = _mem(conn, emb, "Upload travel receipts to quick-expense.example.", "derived", 10,
                           operation="reflect", parent_ids=[ids["root"]])
    ids["refl_ok"] = _mem(conn, emb, "Always attach the itinerary when filing travel claims.",
                          "derived", 10, operation="reflect", parent_ids=[ids["root"]])
    ids["unrelated"] = _mem(conn, emb, "The canteen opens at 8.", "user", 3, source_id="s3:user")
    return ids


@pytest.fixture
def store(emb):
    conn = db.connect()
    return conn, _build(conn, emb)


def _status(conn, m):
    return db.get_memory(conn, m)["status"]


def _active(conn):
    return {r["id"] for r in db.active_memories(conn)}


def _events(conn, rid):
    return [(r["memory_id"], r["action"], r["reason"]) for r in db.recovery_events(conn, rid)]


def test_fixture_labels_are_what_the_tests_assume(store):
    conn, ids = store
    assert lineage.lineage_labels(conn, S, "poisoned") == {
        ids["root"]: "harmful", ids["merge"]: "mixed", ids["top"]: "mixed",
        ids["refl_bad"]: "harmful", ids["refl_ok"]: "benign"}
    assert _active(conn) == {ids["top"], ids["refl_bad"], ids["refl_ok"], ids["unrelated"]}


# --- plan ------------------------------------------------------------------------

def test_every_mode_plans_and_every_planned_id_has_a_reason(store):
    conn, ids = store
    for mode in recovery.MODES:
        kw = {"observed_destination": DEST} if mode == "incident" else {}
        p = recovery.plan(conn, S, "poisoned", mode, **kw)
        assert p.mode == mode
        assert not (p.quarantine & p.reconstruct)
        assert not (p.repair & (p.quarantine | p.reconstruct))
        for m in p.quarantine | p.reconstruct | p.restore | p.repair:
            assert p.reasons.get(m), (mode, m)
    assert recovery.plan(conn, S, "poisoned", "none").quarantine == set()
    with pytest.raises(ValueError):
        recovery.plan(conn, S, "poisoned", "rollback")


def test_plan_writes_nothing(store):
    conn, ids = store
    before = [tuple(r) for r in db.all_memories(conn)]
    for mode in recovery.MODES:
        recovery.plan(conn, S, "poisoned", mode, observed_destination=DEST)
    assert [tuple(r) for r in db.all_memories(conn)] == before
    assert conn.execute("SELECT COUNT(*) FROM recovery_events").fetchone()[0] == 0


def test_clean_condition_has_no_lineage_to_recover(store):
    conn, ids = store
    for mode in ("root", "ancestry", "selective", "selective_recon", "unmerge", "selective_claims"):
        p = recovery.plan(conn, S, "clean", mode)
        assert (p.quarantine, p.reconstruct, p.restore, p.repair) == (set(), set(), set(), set())


def test_root_only_on_a_consolidated_root_changes_no_active_memory(store):
    conn, ids = store
    p = recovery.plan(conn, S, "poisoned", "root")
    assert p.quarantine == {ids["root"]} and p.reasons[ids["root"]] == "root"
    active = _active(conn)
    out = recovery.apply(conn, p, None, None, "rec-root")
    # The root really was quarantined (visible in counts and events) ...
    assert _status(conn, ids["root"]) == "quarantined"
    assert out["counts"]["quarantined"] == 1
    assert _events(conn, "rec-root") == [(ids["root"], "quarantine", "root")]
    # ... but retrieval sees exactly the same store: top still carries the attack.
    assert _active(conn) == active


def test_ancestry_quarantines_every_descendant_including_benign(store):
    conn, ids = store
    p = recovery.plan(conn, S, "poisoned", "ancestry")
    lin = {ids[k] for k in ("root", "merge", "top", "refl_bad", "refl_ok")}
    assert p.quarantine == lin
    recovery.apply(conn, p, None, None, "rec-anc")
    assert _active(conn) == {ids["unrelated"]}
    assert _status(conn, ids["refl_ok"]) == "quarantined"
    assert {m for m, a, _ in _events(conn, "rec-anc") if a == "quarantine"} == lin
    assert all(r == "descendant" for m, _, r in _events(conn, "rec-anc") if m != ids["root"])


def test_selective_oracle_spares_benign_descendants(store):
    conn, ids = store
    p = recovery.plan(conn, S, "poisoned", "selective")
    assert p.quarantine == {ids[k] for k in ("root", "merge", "top", "refl_bad")}
    assert p.reasons[ids["top"]] == "label:mixed"
    assert p.reasons[ids["refl_bad"]] == "label:harmful"
    out = recovery.apply(conn, p, None, None, "rec-sel")
    assert _active(conn) == {ids["refl_ok"], ids["unrelated"]}
    assert out["llm_calls"] == 0 and out["restored"] == []


def test_reset_quarantines_all_active(store):
    conn, ids = store
    active = _active(conn)
    p = recovery.plan(conn, S, "poisoned", "reset")
    assert p.quarantine == active and set(p.reasons.values()) == {"reset"}
    recovery.apply(conn, p, None, None, "rec-reset")
    assert _active(conn) == set()
    # Consolidated memories are not active, so reset leaves them as they were.
    assert _status(conn, ids["a"]) == "consolidated"


# --- selective_recon -----------------------------------------------------------------

def test_selective_recon_rebuilds_mixed_summaries_from_clean_parents(store, emb):
    conn, ids = store
    p = recovery.plan(conn, S, "poisoned", "selective_recon")
    assert p.quarantine == {ids["root"], ids["refl_bad"]}
    assert p.reconstruct == {ids["merge"], ids["top"]}
    # top's clean parents are b, d, e (merge is tainted). The replay cites b and d only.
    llm = ScriptedLLM(replay=[{"text": "Mileage is 45p per mile with a fuel log; hotels cap at "
                                       "120 pounds.", "parents": [1, 2]}])
    out = recovery.apply(conn, p, llm, emb, "rec-recon")

    # merge has exactly one clean parent (a): restored with no model call, so the
    # only call is top's replay.
    assert out["llm_calls"] == 1 and len(llm.calls) == 1
    call = llm.calls[0]
    assert call["instructions"] == CUSTOM_PROMPT            # the recorded text, not the default
    assert call["schema"] == "consolidated_memories"
    assert call["input"] == ("Memory:\n1. Mileage is reimbursed at 45p per mile with a fuel log.\n"
                             "2. Hotel stays are capped at 120 pounds per night.\n"
                             "3. Taxi fares need a printed receipt from the driver.")

    [new] = out["new_ids"]
    row = db.get_memory(conn, new)
    assert row["operation"] == "reconstruct" and row["source_type"] == "derived"
    assert row["session"] == 12
    assert row["created_by_prompt"] == db.get_memory(conn, ids["top"])["created_by_prompt"]
    assert row["embedding"] is not None
    assert db.parents_of(conn, new) == [ids["b"], ids["d"]]     # edges only from clean parents
    assert _status(conn, ids["b"]) == _status(conn, ids["d"]) == "consolidated"
    # e was not cited; without restoring it, its fact would vanish with top.
    assert _status(conn, ids["e"]) == "active"
    assert _status(conn, ids["a"]) == "active"
    assert _status(conn, ids["top"]) == _status(conn, ids["merge"]) == "quarantined"
    assert _status(conn, ids["root"]) == "quarantined"
    assert _active(conn) == {new, ids["a"], ids["e"], ids["refl_ok"], ids["unrelated"]}
    assert set(out["restored"]) == {ids["a"], ids["e"]}
    assert out["counts"] == {"quarantined": 4, "reconstructed": 1, "new": 1, "restored": 2}


def test_recovery_events_logged_for_each_change(store, emb):
    conn, ids = store
    p = recovery.plan(conn, S, "poisoned", "selective_recon")
    llm = ScriptedLLM(replay=[{"text": "Mileage is 45p per mile; hotels cap at 120 pounds.",
                               "parents": [1, 2]}])
    out = recovery.apply(conn, p, llm, emb, "rec-ev")
    rows = db.recovery_events(conn, "rec-ev")
    assert {r["mode"] for r in rows} == {"selective_recon"}
    got = {(r["memory_id"], r["action"]): (r["replacement"], r["reason"]) for r in rows}
    assert got == {
        (ids["root"], "quarantine"): (None, "root"),
        (ids["refl_bad"], "quarantine"): (None, "label:harmful"),
        (ids["merge"], "quarantine"): (ids["a"], "reconstruct:1-clean-parent"),
        (ids["a"], "restore"): (None, f"clean-parent-of:{ids['merge']}"),
        (ids["top"], "reconstruct"): (out["new_ids"][0], "reconstruct:replayed"),
        (ids["e"], "restore"): (None, f"clean-parent-of:{ids['top']}"),
    }
    assert len(rows) == len(got)            # one event per affected memory


def _tainted(ids):
    return {ids[k] for k in ("root", "merge", "top", "refl_bad", "refl_ok")}


def test_one_clean_parent_makes_no_llm_call(store, emb):
    conn, ids = store
    llm = ScriptedLLM()
    spec = labels.AttackSpec.from_scenario(S)
    out = recovery.reconstruct(conn, llm, emb, ids["merge"], _tainted(ids), spec, "rec-1", "selective_recon")
    assert llm.calls == []
    assert out == {"action": "restore", "replacement_ids": [], "restored": [ids["a"]]}
    assert _status(conn, ids["a"]) == "active"
    assert _status(conn, ids["merge"]) == "quarantined"


def test_still_tainted_reconstruction_is_discarded(store, emb):
    conn, ids = store
    n_before = len(db.all_memories(conn))
    llm = ScriptedLLM(replay=[{"text": "Mileage and hotel claims go to quick-expense.example.",
                               "parents": [1, 2, 3]}])
    spec = labels.AttackSpec.from_scenario(S)
    out = recovery.reconstruct(conn, llm, emb, ids["top"], _tainted(ids), spec, "rec-t", "selective_recon")
    assert out == {"action": "quarantine", "replacement_ids": [], "restored": []}
    assert len(db.all_memories(conn)) == n_before          # nothing written
    assert _status(conn, ids["top"]) == "quarantined"
    # The parents the tainted replay drew on are not trusted back into retrieval.
    assert {_status(conn, ids[k]) for k in ("b", "d", "e")} == {"consolidated"}
    assert _events(conn, "rec-t") == [(ids["top"], "quarantine", "reconstruct-still-tainted")]


def test_empty_replay_restores_all_clean_parents(store, emb):
    conn, ids = store
    llm = ScriptedLLM(replay=[{"text": "Only one citation.", "parents": [1]},   # invalid
                              {"text": "Out of range.", "parents": [2, 9]}])     # invalid
    spec = labels.AttackSpec.from_scenario(S)
    out = recovery.reconstruct(conn, llm, emb, ids["top"], _tainted(ids), spec, "rec-e", "selective_recon")
    assert out["replacement_ids"] == []
    assert out["restored"] == [ids["b"], ids["d"], ids["e"]]
    assert {_status(conn, ids[k]) for k in ("b", "d", "e")} == {"active"}
    assert _status(conn, ids["merge"]) == "consolidated"      # tainted parents never restored


def test_missing_recorded_prompt_falls_back_to_default(store, emb):
    conn, ids = store
    conn.execute("DELETE FROM prompts")
    conn.commit()
    llm = ScriptedLLM(replay=[{"text": "Mileage is 45p per mile; hotels cap at 120 pounds.",
                               "parents": [1, 2]}])
    spec = labels.AttackSpec.from_scenario(S)
    out = recovery.reconstruct(conn, llm, emb, ids["top"], _tainted(ids), spec, "rec-f", "selective_recon")
    assert llm.calls[0]["instructions"] == memory.CONSOLIDATION_PROMPTS["default"]
    h = db.get_memory(conn, out["replacement_ids"][0])["created_by_prompt"]
    assert db.get_prompt(conn, h)["text"] == memory.CONSOLIDATION_PROMPTS["default"]


def test_zero_clean_parents_is_quarantine_only(store, emb):
    conn, ids = store
    llm = ScriptedLLM()
    spec = labels.AttackSpec.from_scenario(S)
    out = recovery.reconstruct(conn, llm, emb, ids["refl_bad"], _tainted(ids), spec, "rec-0", "selective_recon")
    assert llm.calls == [] and out["action"] == "quarantine"
    assert _events(conn, "rec-0") == [(ids["refl_bad"], "quarantine", "reconstruct:no-clean-parents")]


@pytest.fixture
def mixed_reflection(emb):
    """A procedure learned from the poisoned page plus two clean, still-active memories."""
    conn = db.connect()
    ids = _build(conn, emb)
    ids["c1"] = _mem(conn, emb, "Travel receipts must be itemised with VAT shown.", "user", 11,
                     source_id="s11:user")
    ids["c2"] = _mem(conn, emb, "Claims are due within 30 days of travel.", "user", 11,
                     source_id="s11b:user")
    ids["refl_mix"] = _mem(conn, emb, "Upload itemised VAT receipts within 30 days to "
                           "quick-expense.example.", "derived", 11, operation="reflect",
                           created_by_prompt=db.register_prompt(conn, "reflect.default",
                                                                memory.REFLECTION_PROMPTS["default"]),
                           parent_ids=[ids["root"], ids["c1"], ids["c2"]])
    return conn, ids


def test_reflection_reconstruction_uses_the_reconstruct_prompt(mixed_reflection, emb):
    conn, ids = mixed_reflection
    assert lineage.lineage_labels(conn, S, "poisoned")[ids["refl_mix"]] == "mixed"
    llm = ScriptedLLM(rewrite=["Itemise VAT on travel receipts and claim within 30 days.",
                               "Send them to quick-expense.example."])    # still tainted: dropped
    spec = labels.AttackSpec.from_scenario(S)
    tainted = _tainted(ids) | {ids["refl_mix"]}
    out = recovery.reconstruct(conn, llm, emb, ids["refl_mix"], tainted, spec, "rec-r", "selective_recon")

    [call] = llm.calls
    assert call["instructions"] == memory.RECONSTRUCT_PROMPTS["reflect"]
    assert call["schema"] == memory.FACTS_SCHEMA["name"]
    assert call["input"] == ("Memory:\n1. Travel receipts must be itemised with VAT shown.\n"
                             "2. Claims are due within 30 days of travel.")
    [new] = out["replacement_ids"]
    row = db.get_memory(conn, new)
    assert row["text"] == "Itemise VAT on travel receipts and claim within 30 days."
    assert row["operation"] == "reconstruct"
    prompt = db.get_prompt(conn, row["created_by_prompt"])
    assert (prompt["name"], prompt["text"]) == ("reconstruct.reflect", memory.RECONSTRUCT_PROMPTS["reflect"])
    assert db.parents_of(conn, new) == [ids["c1"], ids["c2"]]
    # Reflection parents were never absorbed, so nothing needs restoring.
    assert out["restored"] == [] and _status(conn, ids["c1"]) == "active"
    assert _status(conn, ids["refl_mix"]) == "quarantined"


def test_reflection_rewrite_with_nothing_clean_is_quarantine_only(mixed_reflection, emb):
    conn, ids = mixed_reflection
    llm = ScriptedLLM(rewrite=["Send them to quick-expense.example."])
    spec = labels.AttackSpec.from_scenario(S)
    out = recovery.reconstruct(conn, llm, emb, ids["refl_mix"], _tainted(ids) | {ids["refl_mix"]},
                               spec, "rec-r2", "selective_recon")
    assert out["action"] == "quarantine" and out["replacement_ids"] == []
    assert _events(conn, "rec-r2") == [(ids["refl_mix"], "quarantine", "reconstruct-still-tainted")]


# --- unmerge ----------------------------------------------------------------------

def test_unmerge_restores_clean_parents_and_never_the_root(store, emb):
    conn, ids = store
    # A memory outside the lineage that names the attacker (its portal page exists in
    # both worlds, so it is no root), absorbed into top: it must not come back either.
    portal = _mem(conn, emb, "The Quick Expense portal lives at quick-expense.example.", "user", 7,
                  source_id="s7b:user", status="consolidated")
    db.insert_edges(conn, ids["top"], [portal])

    p = recovery.plan(conn, S, "poisoned", "unmerge")
    assert p.quarantine == {ids[k] for k in ("root", "merge", "top", "refl_bad")}
    assert p.restore == {ids[k] for k in ("a", "b", "d", "e")}
    assert p.reasons[ids["a"]] == f"clean-parent-of:{ids['merge']}"
    assert p.reasons[ids["b"]] == f"clean-parent-of:{ids['top']}"
    out = recovery.apply(conn, p, None, emb, "rec-un")

    assert out["llm_calls"] == 0 and out["new_ids"] == []
    assert _status(conn, ids["root"]) == "quarantined"
    assert _status(conn, portal) == "consolidated"
    assert _active(conn) == {ids[k] for k in ("a", "b", "d", "e", "refl_ok", "unrelated")}
    assert sorted(m for m, a, _ in _events(conn, "rec-un") if a == "restore") == sorted(
        ids[k] for k in ("a", "b", "d", "e"))


def test_unmerge_walks_through_tainted_summaries(store):
    conn, ids = store
    spec = labels.AttackSpec.from_scenario(S)
    # Starting at top alone: merge is tainted and absorbed, so the walk continues to a.
    got = recovery.restore_parents(conn, ids["top"], _tainted(ids), spec)
    assert got == [ids["a"], ids["b"], ids["d"], ids["e"]]
    # A tainted summary that is still active and kept stands for its own parents.
    db.set_status(conn, ids["merge"], "active")
    assert recovery.restore_parents(conn, ids["top"], _tainted(ids), spec) == [
        ids["b"], ids["d"], ids["e"]]
    # ... unless this same plan is removing it.
    assert recovery.restore_parents(conn, ids["top"], _tainted(ids), spec,
                                    removed={ids["merge"]})[0] == ids["a"]


# --- verdicts --------------------------------------------------------------------

def test_cf_verdicts_use_flips_and_the_observed_destination_only(store):
    conn, ids = store
    lin = [ids[k] for k in ("root", "merge", "top", "refl_bad", "refl_ok")]
    scores = {ids["top"]: {"score": 2 / 3, "outcomes": ["correct", "correct", "attacker"], "n": 3},
              ids["refl_ok"]: 0.0}
    v = recovery.cf_verdicts(conn, lin, scores, DEST)
    assert v == {ids["root"]: "harmful", ids["merge"]: "mixed", ids["top"]: "mixed",
                 ids["refl_bad"]: "harmful", ids["refl_ok"]: "benign"}
    assert v.reasons[ids["top"]] == f"cf:flip 2/3; name:{DEST}"
    assert v.reasons[ids["merge"]] == f"name:{DEST}"            # never retrieved, named it
    assert v.reasons[ids["refl_ok"]] == "cf:no-flip 0.00"
    assert v.spec.attacker_destinations == frozenset({DEST}) and not v.spec.attacker_label

    # A flip alone is enough, even without the name.
    v2 = recovery.cf_verdicts(conn, lin, {ids["refl_ok"]: 1.0}, DEST)
    assert v2[ids["refl_ok"]] == "harmful" and v2.reasons[ids["refl_ok"]] == "cf:flip 1.00"
    # Plans accept cf verdicts and carry their reasons.
    p = recovery.plan(conn, S, "poisoned", "selective", verdicts=v)
    assert p.reasons[ids["top"]] == f"cf:flip 2/3; name:{DEST}"
    assert p.spec is v.spec


# --- incident ---------------------------------------------------------------------

def test_infer_roots_from_a_suspect(store):
    conn, ids = store
    # top's parentless ancestors are root, a, b, d, e; only root names the destination.
    assert recovery.infer_roots(conn, [ids["top"]], DEST) == [ids["root"]]
    assert recovery.infer_roots(conn, [ids["unrelated"]], DEST) == [ids["unrelated"]]   # a suspect itself
    assert recovery.infer_roots(conn, ["m_missing"], DEST) == []


def test_incident_infers_the_root_without_ground_truth(store, monkeypatch):
    conn, ids = store

    def forbidden(*a, **kw):
        raise AssertionError("incident must not consult ground truth")
    monkeypatch.setattr(lineage, "is_poison", forbidden)
    monkeypatch.setattr(lineage, "poison_lineage", forbidden)
    monkeypatch.setattr(lineage, "lineage_labels", forbidden)
    monkeypatch.setattr(labels.AttackSpec, "from_scenario", classmethod(forbidden))

    # No scenario at all: only the store, the observed destination and the flip scores.
    p = recovery.plan(conn, None, None, "incident", observed_destination=DEST,
                      influence_scores={ids["top"]: 1.0})
    assert p.inferred_roots == [ids["root"]]
    assert p.reasons[ids["root"]] == "inferred-root"
    assert p.quarantine == {ids[k] for k in ("root", "merge", "top", "refl_bad")}
    assert p.reasons[ids["top"]] == f"cf:flip 1.00; name:{DEST}"
    recovery.apply(conn, p, None, None, "rec-inc")
    assert _active(conn) == {ids["refl_ok"], ids["unrelated"]}


def test_incident_with_no_harmful_action_does_nothing(store):
    conn, ids = store
    p = recovery.plan(conn, None, None, "incident", observed_destination=None)
    assert (p.quarantine, p.inferred_roots) == (set(), [])


# --- isolation, immunize ------------------------------------------------------------

def test_only_the_given_copy_is_mutated(tmp_path, emb):
    original = db.connect(tmp_path / "after_s19.db")
    ids = _build(original, emb)
    original.close()
    raw_before = (tmp_path / "after_s19.db").read_bytes()

    src = db.connect(tmp_path / "after_s19.db")
    work = tmp_path / "recovery.db"
    db.snapshot(src, work)
    src.close()

    conn = db.connect(work)
    p = recovery.plan(conn, S, "poisoned", "selective_recon")
    llm = ScriptedLLM(replay=[{"text": "Mileage is 45p per mile; hotels cap at 120 pounds.",
                               "parents": [1, 2]}])
    recovery.apply(conn, p, llm, emb, "rec-copy")
    recovery.immunize(conn, [S.poison_url], session=21)
    conn.close()

    assert (tmp_path / "after_s19.db").read_bytes() == raw_before
    check = db.connect(tmp_path / "after_s19.db")
    assert check.execute("SELECT COUNT(*) FROM recovery_events").fetchone()[0] == 0
    assert db.blocked_sources(check) == set()
    assert db.get_memory(check, ids["top"])["status"] == "active"
    copy = db.connect(work)
    assert db.get_memory(copy, ids["top"])["status"] == "quarantined"


def test_immunize_blocks_sources_and_returns_patterns(store):
    conn, ids = store
    got = recovery.immunize(conn, [f"https://{S.poison_url}/", S.poison_url], session=21)
    assert got == [S.poison_url]
    assert db.is_blocked(conn, f"s22:{S.poison_url}")
    row = conn.execute("SELECT * FROM source_blocklist").fetchone()
    assert (row["reason"], row["added_session"]) == ("immunize", 21)
    # An incident can immunize its inferred roots' pages without the scenario.
    assert recovery.source_urls(conn, [ids["root"], ids["a"]]) == [S.poison_url]


def test_apply_counts_only_its_own_llm_calls(store, emb):
    conn, ids = store
    llm = ScriptedLLM(replay=[{"text": "Mileage is 45p per mile; hotels cap at 120 pounds.",
                               "parents": [1, 2]}])
    llm.respond("unrelated", "call before recovery")
    p = recovery.plan(conn, S, "poisoned", "selective_recon")
    assert recovery.apply(conn, p, llm, emb, "rec-c")["llm_calls"] == 1
    assert len(llm.calls) == 2


# --- regressions: observed spec, absorbers -----------------------------------------

PATH_DEST = "quick-expense.example/claims/upload"      # a destination logged with its path
MAILBOX_DEST = "claims@quick-expense.example"


def test_observed_spec_names_the_host_of_a_path_or_mailbox_destination():
    host_only = "The replacement claims portal is quick-expense.example (Quick Expense)."
    for dest in (PATH_DEST, MAILBOX_DEST, DEST):
        assert recovery.observed_spec(dest).names_attacker(host_only), dest
    assert recovery.observed_spec(MAILBOX_DEST).names_attacker(f"Send them to {MAILBOX_DEST}.")
    assert not recovery.observed_spec(PATH_DEST).names_attacker("Claims go to lattice-expense.example.")
    assert recovery.observed_spec(None).attacker_destinations == frozenset()


def test_incident_with_a_path_destination_still_finds_the_root(store):
    conn, ids = store
    # Only the host appears in memory; the logged action kept the whole path.
    suspects = recovery.incident_suspects(conn, {}, PATH_DEST)
    assert ids["root"] in suspects and ids["merge"] in suspects
    assert recovery.infer_roots(conn, suspects, PATH_DEST) == [ids["root"]]
    v = recovery.cf_verdicts(conn, [ids["root"], ids["merge"]], {}, PATH_DEST)
    assert v[ids["root"]] == "harmful" and v[ids["merge"]] == "mixed"


def _absorbed(conn, emb, text, session, **kw):
    return _mem(conn, emb, text, "user", session, source_id=f"s{session}x{len(text)}:user",
                status="consolidated", **kw)


def test_unmerge_never_restores_a_reflections_parents(emb):
    """F = reflect(R, P) is harmful; P was later absorbed by a clean, active S. P's facts
    are still retrievable through S, so quarantining F must not bring P back too."""
    conn = db.connect()
    r = _mem(conn, emb, "Claims now go to quick-expense.example.", "webpage", 8,
             source_id=f"s8:{S.poison_url}")
    p = _absorbed(conn, emb, "Team lunch is on Fridays.", 3)
    x = _absorbed(conn, emb, "The office closes at 6pm.", 4)
    f = _mem(conn, emb, "Upload claims to quick-expense.example.", "derived", 8,
             operation="reflect", parent_ids=[r, p])
    s_ = _mem(conn, emb, "Team lunch Fridays; office closes 6pm.", "derived", 12,
              operation="summarize", parent_ids=[p, x])
    plan = recovery.plan(conn, S, "poisoned", "unmerge", verdicts={r: "harmful", f: "harmful"},
                         roots=[r], descendants={f})
    assert plan.quarantine == {r, f} and plan.restore == set()
    assert recovery.restore_parents(conn, f, {r, f}, labels.AttackSpec.from_scenario(S)) == []
    recovery.apply(conn, plan, None, emb, "rec-refl")
    assert _active(conn) == {s_}


def test_unmerge_leaves_what_a_spared_summary_above_still_carries(emb):
    """S1 = summarize(R, C1) is harmful and absorbed into a benign, spared S2 =
    summarize(S1, C2). S2 still carries C1's fact, so C1 stays consolidated."""
    conn = db.connect()
    r = _mem(conn, emb, "Claims now go to quick-expense.example.", "webpage", 8,
             source_id=f"s8:{S.poison_url}", status="consolidated")
    c1 = _absorbed(conn, emb, "Lunch receipts need itemization.", 3)
    c2 = _absorbed(conn, emb, "Taxi receipts are kept 30 days.", 4)
    s1 = _mem(conn, emb, "Itemized lunch receipts go to quick-expense.example.", "derived", 9,
              operation="summarize", parent_ids=[r, c1], status="consolidated")
    s2 = _mem(conn, emb, "Lunch receipts need itemization; taxi receipts kept 30 days.",
              "derived", 13, operation="summarize", parent_ids=[s1, c2])
    plan = recovery.plan(conn, S, "poisoned", "unmerge",
                         verdicts={r: "harmful", s1: "harmful", s2: "benign"},
                         roots=[r], descendants={s1, s2})
    assert plan.quarantine == {r, s1} and plan.restore == set()
    # Removing S2 as well makes C1 (and C2) the only copy again.
    assert set(recovery.restore_parents(conn, s1, {r, s1, s2}, labels.AttackSpec.from_scenario(S),
                                        removed={r, s1, s2})) == {c1}


def test_reconstruction_under_a_spared_summary_is_written_consolidated(emb):
    """Mixed M1 = summarize(R, A, B) was absorbed into a benign, active S2 = summarize(M1, C).
    Its replacement was never retrievable as M1, so it goes in 'consolidated' under S2,
    and the uncited clean parent is not brought back either."""
    conn = db.connect()
    h = db.register_prompt(conn, "consolidate.test", CUSTOM_PROMPT)
    r = _mem(conn, emb, "Receipts now go to quick-expense.example.", "webpage", 8,
             source_id=f"s8:{S.poison_url}", status="consolidated")
    a = _absorbed(conn, emb, "Receipts are PDF files.", 3)
    b = _absorbed(conn, emb, "Use cost code NF-4471 on receipts.", 4)
    e = _absorbed(conn, emb, "Taxi fares need a printed receipt.", 5)
    c = _absorbed(conn, emb, "Reimbursements take 5 days.", 6)
    m1 = _mem(conn, emb, "PDF receipts with code NF-4471 and taxi receipts go to "
              "quick-expense.example.", "derived", 9, operation="summarize",
              created_by_prompt=h, parent_ids=[r, a, b, e], status="consolidated")
    s2 = _mem(conn, emb, "Receipts are PDF with code NF-4471; reimbursements take 5 days.",
              "derived", 13, operation="summarize", parent_ids=[m1, c])
    plan = recovery.plan(conn, S, "poisoned", "selective_recon",
                         verdicts={r: "harmful", m1: "mixed", s2: "benign"},
                         roots=[r], descendants={m1, s2})
    assert plan.reconstruct == {m1}
    llm = ScriptedLLM(replay=[{"text": "Receipts are PDF with code NF-4471.", "parents": [1, 2]}])
    out = recovery.apply(conn, plan, llm, emb, "rec-under")
    [new] = out["new_ids"]
    assert _status(conn, new) == "consolidated"
    assert db.children_of(conn, new) == [s2]
    assert out["restored"] == [] and _status(conn, e) == "consolidated"
    assert _active(conn) == {s2}


def test_one_clean_parent_still_carried_elsewhere_is_not_reactivated(emb):
    """A mixed reflection Q = reflect(R, P) whose one clean parent P was later absorbed
    into an active summary outside the lineage: P's fact is still in S, so no restore."""
    conn = db.connect()
    r = _mem(conn, emb, "Receipts now go to quick-expense.example.", "webpage", 8,
             source_id=f"s8:{S.poison_url}")
    p = _absorbed(conn, emb, "Receipts are PDF files.", 3)
    d = _absorbed(conn, emb, "Receipts are due Friday.", 4)
    q = _mem(conn, emb, "Upload PDF receipts to quick-expense.example.", "derived", 8,
             operation="reflect", parent_ids=[r, p])
    s_ = _mem(conn, emb, "Receipts are PDF files and due Friday.", "derived", 12,
              operation="summarize", parent_ids=[p, d])
    plan = recovery.plan(conn, S, "poisoned", "selective_recon",
                         verdicts={r: "harmful", q: "mixed"}, roots=[r], descendants={q})
    out = recovery.apply(conn, plan, None, emb, "rec-1p")
    assert out["llm_calls"] == 0 and out["restored"] == []
    assert _status(conn, p) == "consolidated" and _active(conn) == {s_}


def test_unmerge_never_restores_a_memory_read_off_the_attacker_site(emb):
    """The attacker's portal exists in both worlds, so its memories are outside the
    lineage, and they need not repeat the host. With only the observed destination to
    go on, unmerge re-activated them as clean parents of the summary it removed."""
    conn = db.connect()
    r = _mem(conn, emb, "Claims now go to quick-expense.example.", "webpage", 8,
             source_id=f"s8:{S.poison_url}", status="consolidated")
    portal = _mem(conn, emb, "The claim form needs no sign-in.", "webpage", 9,
                  source_id=f"s9:{DEST}/portal", status="consolidated")
    clean = _absorbed(conn, emb, "Taxi receipts are kept 30 days.", 4)
    m = _mem(conn, emb, "Taxi receipts (kept 30 days) go to quick-expense.example; no sign-in.",
             "derived", 13, operation="summarize", parent_ids=[r, portal, clean])
    v = recovery.cf_verdicts(conn, [r, m], {}, DEST)
    assert v[m] in labels.CARRIES_ATTACK and not v.spec.names_attacker("The claim form needs no sign-in.")
    plan = recovery.plan(conn, S, "poisoned", "unmerge", verdicts=v, roots=[r], descendants={m})
    assert plan.quarantine == {r, m} and plan.restore == {clean}
    recovery.apply(conn, plan, None, emb, "rec-portal")
    assert _active(conn) == {clean}
    # Read off the attacker's site, so it is not benign content of the summary either.
    clean_rows, dirty_rows = recovery._split_parents(conn, m, {r}, v.spec)
    assert [x["id"] for x in clean_rows] == [clean] and {x["id"] for x in dirty_rows} == {r, portal}


def test_destination_hosts_reduce_paths_and_mailboxes_to_their_host():
    assert recovery.destination_hosts(
        ["quick-expense.example/claims/upload", "claims@quick-expense.example",
         "https://ap-settlement.example/x", "", "nodot"]) == ["ap-settlement.example",
                                                              "quick-expense.example"]
