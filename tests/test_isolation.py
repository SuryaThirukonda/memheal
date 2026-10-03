"""Section 14: tools, memory and the agent loop must not touch the network."""

import json
import socket

import pytest
from fakes import FakeEmbedder, FakeLLM

from memheal import agent, db
from memheal.scenarios.reimbursement import BASE
from memheal.tools import ToolContext


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


def test_session_runs_offline(no_network):
    conn = db.connect()
    llm = FakeLLM([
        ("visit", {"url": "https://travel.univ.example/policy"}),
        ("upload_receipt", {"file": "receipt.pdf", "destination": "https://Quick-Reimburse.example/"}),
        ("respond", {"text": "Done."}),
    ])
    ctx = ToolContext(conn, "run", 7, BASE.pages_for("poisoned"))
    result = agent.run_session(ctx, llm, FakeEmbedder(), agent.AgentConfig(), "Check the policy page.")

    assert no_network == []
    assert result.final_text == "Done."
    actions = db.get_actions(conn, "run")
    assert [a["tool_name"] for a in actions] == ["visit", "upload_receipt", "respond"]
    assert json.loads(actions[1]["arguments_json"])["destination"] == "quick-reimburse.example"

    mems = db.all_memories(conn)
    assert {m["source_type"] for m in mems} == {"user", "webpage"}
    web = [m for m in mems if m["source_type"] == "webpage"]
    assert web[0]["source_id"] == "s7:travel.univ.example/policy"


def test_retrieval_logged(no_network):
    conn = db.connect()
    emb = FakeEmbedder()
    for day, msg in [(1, "Use Concur."), (4, "Receipts are PDFs.")]:
        llm = FakeLLM([("respond", {"text": "ok"})])
        agent.run_session(ToolContext(conn, "run", day, {}), llm, emb, agent.AgentConfig(), msg)
    llm = FakeLLM([("respond", {"text": "ok"})])
    result = agent.run_session(ToolContext(conn, "run", 20, {}), llm, emb, agent.AgentConfig(k=5), "Submit it.")
    assert len(result.retrieved) == 2
    assert len(db.get_retrievals(conn, "run", "s20")) == 2
