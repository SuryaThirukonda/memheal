"""The chatbot's memory must survive the process: close the DB, reopen it, recall."""

from fakes import FakeEmbedder, FakeLLM

from chat import ChatBot
from memheal import db


def test_memory_persists_across_restarts(tmp_path):
    path = tmp_path / "chat.db"
    emb = FakeEmbedder()

    conn = db.connect(path)
    bot = ChatBot(conn, FakeLLM([]), emb)
    assert bot.session == 1
    _, ids = bot.reply("My favorite color is green.")
    assert len(ids) == 1
    conn.close()

    # "Restart": new connection, new bot, nothing carried over in Python memory.
    conn = db.connect(path)
    bot = ChatBot(conn, FakeLLM([]), emb)
    assert bot.session == 2
    assert [r["text"] for r in db.active_memories(conn)] == ["My favorite color is green."]
    bot.reply("What is my favorite color?")
    (event,) = db.get_retrievals(conn, bot.run_id, "s2t1")
    assert event["memory_id"] == ids[0]

    assert db.set_status(conn, ids[0], "deleted")
    assert all(r["id"] != ids[0] for r in db.active_memories(conn))
    conn.close()
