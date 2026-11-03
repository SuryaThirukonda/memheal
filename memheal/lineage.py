"""Harness-side poison lineage: which memories are roots, what they became, and labels.

Ground truth stays out of the memory store (db.py's rule), so it is computed here from
the scenario. run_demo.py, measure.py and the Weeks 11-12 recovery code all need the
same answer to "which memories did the poisoned page plant", so it lives in one place.

Recovery modes that are allowed to know the root (root, ancestry, the oracle verdicts
for selective) use these functions; `incident` must not, since the whole point of that
mode is that the defender has only the observed harmful action.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import TYPE_CHECKING

from memheal import db
from memheal.labels import AttackSpec, label_lineage
from memheal.tools import normalize_url

if TYPE_CHECKING:  # importing the package builds the whole catalog; only the type is needed
    from memheal.scenarios.base import Scenario


# id(scenario) -> (scenario, urls). Scenarios hold dicts, so they are not hashable, and
# is_poison runs once per memory row; the scenario is kept so a recycled id cannot hit.
_POISON_URLS: dict[int, tuple[Scenario, frozenset[str]]] = {}


def poison_source_urls(scenario: Scenario) -> frozenset[str]:
    """Normalized urls whose memories are poison roots.

    The main poisoned page, plus every other page the poisoned world changes: one
    missing from the clean world (composed poison's notes page, which holds the
    destination) or with different text there. Some scenarios list their whole
    poisoned-world page set in poisoned_pages (expense_migration), including pages
    identical in both worlds; memories read from those are not poison, and counting
    them made benign announcements roots that recovery then quarantined and blocked.
    """
    hit = _POISON_URLS.get(id(scenario))
    if hit is not None and hit[0] is scenario:
        return hit[1]
    main = normalize_url(scenario.poison_url)
    clean = scenario.pages_for("clean") if "clean" in scenario.conditions else {}
    poisoned = (scenario.pages_for("poisoned") if "poisoned" in scenario.conditions
                else {normalize_url(k): v for k, v in scenario.poisoned_pages.items()})
    urls = {main}
    for u in scenario.poisoned_pages:
        u = normalize_url(u)
        if u not in clean or clean[u] != poisoned.get(u):
            urls.add(u)
    _POISON_URLS[id(scenario)] = (scenario, frozenset(urls))
    return frozenset(urls)


def is_poison(row, scenario: Scenario, condition: str) -> bool:
    """Harness-side ground truth: a memory extracted from a poisoned page.

    Matched on the source URL rather than the poison session, because an agent that
    has been told to watch that page may re-read it on a later day; those memories
    are just as poisoned as the first one.
    """
    if condition != "poisoned" or row["source_type"] != "webpage":
        return False
    # Every page the attacker changed, not only the main one: composed poison puts the
    # destination on a second page, and memories read from it are just as much roots.
    source = row["source_id"] or ""
    return any(source.endswith(f":{u}") for u in poison_source_urls(scenario))


def poison_lineage(conn: sqlite3.Connection, scenario: Scenario,
                   condition: str) -> tuple[list[str], set[str]]:
    """(roots in store order, every memory derived from any of them).

    Descendants exclude the roots themselves, so a caller can treat the two sets
    differently (Section 5.7 quarantines roots outright but judges descendants).
    """
    roots = [r["id"] for r in db.all_memories(conn) if is_poison(r, scenario, condition)]
    descendants: set[str] = set()
    for r in roots:
        descendants |= db.get_descendants(conn, r)
    return roots, descendants - set(roots)


def lineage_labels(conn: sqlite3.Connection, scenario: Scenario,
                   condition: str) -> dict[str, str]:
    """Section 7.8 oracle label for every root and descendant; empty outside 'poisoned'."""
    roots, descendants = poison_lineage(conn, scenario, condition)
    if not roots:
        return {}
    mems = {r["id"]: dict(r) for r in db.all_memories(conn)}
    parents: dict[str, list[str]] = defaultdict(list)
    for p_id, c_id in conn.execute("SELECT parent_id, child_id FROM memory_edges ORDER BY rowid"):
        parents[c_id].append(p_id)
    return label_lineage(mems, parents, roots, descendants, AttackSpec.from_scenario(scenario))


def unlinked_poison_day_reflections(conn: sqlite3.Connection, scenario: Scenario,
                                    condition: str) -> list[str]:
    """Poison-day reflections with no edge from any poison root, in store order.

    memory.reflect now also cites the memories its own session stored. Runs made before
    that change (folders below 0959) recorded only the retrieved set, so a procedure
    learned from the poisoned page has no edge to that page's memory, and the oracle
    lineage silently misses it. Only meaningful when the poison day did store a root.
    """
    roots, descendants = poison_lineage(conn, scenario, condition)
    rows = db.all_memories(conn)
    by_id = {r["id"]: r for r in rows}
    if not any(by_id[r]["session"] == scenario.poison_session for r in roots):
        return []
    linked = set(roots) | descendants
    return [r["id"] for r in rows
            if r["operation"] == "reflect" and r["session"] == scenario.poison_session
            and r["id"] not in linked]
