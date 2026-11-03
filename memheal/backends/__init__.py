"""Third-party memory backends (proposal Section 12.3).

A backend owns the write side of memory: on every store it decides what is added,
rewritten or removed. It mirrors each of those changes into the run's SQLite
`memories` table, and the agent keeps retrieving from that table with
memory.retrieve, so lineage, labels, the counterfactual and recovery read the same
store on a backend run as on a flat or evolving one.

Importing this package imports no backend library. Mem0 reads its telemetry switch
and its data folder once, at import, so it is imported only when a run opens it,
after the run's folder is known.
"""

from __future__ import annotations

#: memory_mode values whose store step belongs to a backend rather than memory.store.
BACKEND_MODES: tuple[str, ...] = ("mem0",)


def uses_backend(memory_mode: str) -> bool:
    return memory_mode in BACKEND_MODES
