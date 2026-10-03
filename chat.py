"""Interactive chatbot with persistent SQLite memory.

    uv run python chat.py                  # memory in data/chat.db
    uv run python chat.py --db mine.db     # a separate memory file

Each launch is one session. Every user message is distilled into facts with the
same extractor the experiments use and stored in the SQLite file, so the bot
remembers you after the program exits. Within a launch it also keeps the normal
conversation history.

Commands: /memories  /forget <id>  /help  /quit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from dotenv import load_dotenv

from memheal import db, memory
from memheal.llm import DEFAULT_MODEL, LLM

ROOT = Path(__file__).parent

SYSTEM_PROMPT = (
    "You are a helpful personal assistant with long-term memory. Below are memories "
    "from earlier conversations with this user; use them when they are relevant, and "
    "don't claim to remember things that aren't listed."
)


class ChatBot:
    def __init__(self, conn, llm, embedder, k: int = 5, extraction_prompt: str = "default"):
        self.conn = conn
        self.llm = llm
        self.embedder = embedder
        self.k = k
        self.extraction_prompt = extraction_prompt
        self.session = db.next_session(conn)
        self.run_id = f"chat-s{self.session}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.history: list[dict] = []
        db.insert_run(conn, self.run_id, "chat", "interactive", llm.model, {
            "llm": llm.params(),
            "k": k,
            "embedding_model": embedder.name,
            "system_prompt": SYSTEM_PROMPT,
            "extraction_prompt_text": memory.EXTRACTION_PROMPTS[extraction_prompt],
        })
        self.turn = 0

    def reply(self, user_text: str) -> tuple[str, list[str]]:
        """Answer one message, then store facts from it. Returns (reply, new memory ids)."""
        self.turn += 1
        retrieved = memory.retrieve(
            self.conn, self.embedder, self.run_id, self.session, user_text,
            k=self.k, query_id=f"s{self.session}t{self.turn}",
        )
        lines = [f"{i}. {row['text']}" for i, (row, _) in enumerate(retrieved, start=1)]
        instructions = SYSTEM_PROMPT + "\n\nMemories from earlier conversations:\n" + (
            "\n".join(lines) if lines else "(none)"
        )
        self.history.append({"role": "user", "content": user_text})
        response = self.llm.respond(instructions, self.history, tag=f"chat:s{self.session}t{self.turn}")
        self.history.append({"role": "assistant", "content": response.text})

        new_ids = memory.store(
            self.conn, self.llm, self.embedder, self.session, user_text, "user",
            f"s{self.session}t{self.turn}:user", prompt_name=self.extraction_prompt,
        )
        return response.text, new_ids


def print_memories(conn) -> None:
    rows = db.all_memories(conn)
    if not rows:
        print("  (no memories yet)")
    for r in rows:
        flag = "" if r["status"] == "active" else f"  [{r['status']}]"
        print(f"  {r['id']}  s{r['session']:<3} {r['source_type']:<7} {r['text']}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=str(ROOT / "data" / "chat.db"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--quiet", action="store_true", help="don't print memory writes")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    try:
        import readline  # noqa: F401  (arrow-key editing and history in input())
    except ImportError:
        pass

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = db.connect(db_path)
    llm = LLM(model=args.model, log_path=db_path.with_suffix(".llm.jsonl"))
    print("Loading embedding model...")
    bot = ChatBot(conn, llm, memory.get_embedder(), k=args.k)
    n_active = len(db.active_memories(conn))
    print(f"Session {bot.session} | {n_active} memories in {db_path} | /help for commands\n")

    try:
        while True:
            text = input("You: ").strip()
            if not text:
                continue
            if text in ("/quit", "/exit"):
                break
            if text == "/help":
                print("  /memories      list everything stored\n"
                      "  /forget <id>   stop using a memory (flagged, not deleted)\n"
                      "  /quit          exit (memories are already saved)")
                continue
            if text == "/memories":
                print_memories(conn)
                continue
            if text.startswith("/forget"):
                parts = text.split()
                ok = len(parts) == 2 and db.set_status(conn, parts[1], "deleted")
                print(f"  forgot {parts[1]}" if ok else "  usage: /forget <memory id from /memories>")
                continue

            answer, new_ids = bot.reply(text)
            print(f"Bot: {answer}")
            if new_ids and not args.quiet:
                for m in new_ids:
                    print(f"  [memory + {m}] {db.get_memory(conn, m)['text']}")
            print()
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        print(f"Saved. {len(db.active_memories(conn))} active memories in {db_path}")
        conn.close()


if __name__ == "__main__":
    main()
