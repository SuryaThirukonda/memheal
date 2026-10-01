import sqlite3
import uuid
from typing import List, Dict, Any


class MemoryStore:
    def __init__(self, db_path: str = "memheal.db"):
        self.path = db_path
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    session TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def add(self, text: str, source_type: str, session: str) -> str:
        mem_id = f"m_{uuid.uuid4().hex[:8]}"
        with self.conn:
            self.conn.execute(
                "INSERT INTO memories (id, text, source_type, session, status) VALUES (?, ?, ?, ?, ?)",
                (mem_id, text, source_type, session, "active"),
            )
        return mem_id

    def search(self, query: str = "") -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        if not query:
            cursor.execute(
                "SELECT id, text, source_type, session, status FROM memories WHERE status = 'active' ORDER BY created_at ASC"
            )
        else:
            cursor.execute(
                "SELECT id, text, source_type, session, status FROM memories WHERE status = 'active' AND text LIKE ? ORDER BY created_at ASC",
                (f"%{query}%",),
            )
        return [dict(row) for row in cursor.fetchall()]

    def set_status(self, memory_id: str, status: str) -> bool:
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE memories SET status = ? WHERE id = ?",
                (status, memory_id),
            )
            return cursor.rowcount > 0

    def counts(self) -> str:
        cursor = self.conn.cursor()
        cursor.execute("SELECT status, COUNT(*) FROM memories GROUP BY status")
        counts_dict = {row[0]: row[1] for row in cursor.fetchall()}
        active = counts_dict.get("active", 0)
        retracted = counts_dict.get("retracted", 0)
        return f"{active} active, {retracted} retracted"

    def close(self) -> None:
        if self.conn:
            self.conn.close()
