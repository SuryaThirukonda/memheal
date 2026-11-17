"""Weeks 11-12 recovery: plan and apply a cleanup of a poisoned memory store (Section 5.6-5.7).

Recovery always runs on a COPY of a finished run's last pre-target snapshot; the
harness (recover.py) makes the copy and replays the target task family afterwards.
Nothing here opens a file, so the only thing a call can mutate is the connection it
is handed.

A recovery is two steps. `plan` decides, from lineage and verdicts alone, which
memories to quarantine, which mixed ones to rebuild and which clean parents to bring
back; it writes nothing. `apply` carries the plan out in a fixed order (quarantine,
reconstruct, restore) and logs one recovery_event per memory it touched, so a
recovered copy says what was done to it and why.

Modes, from bluntest to most targeted:

    none             nothing (the replay baseline)
    reset            quarantine every active memory
    root             quarantine the known poison roots. Usually a no-op for retrieval,
                     because by the target day the roots are already 'consolidated'
                     into summaries that carry the attack on (Section 7.2). That is a
                     finding, so it is reported as is rather than patched over.
    ancestry         roots + every descendant, benign ones included
    selective        roots + descendants whose verdict is harmful or mixed
    selective_recon  roots + harmful descendants; mixed ones rebuilt from their clean
                     parents (Section 5.7)
    unmerge          selective, then re-activate the clean parents the quarantined
                     summaries had absorbed (lossless, no LLM calls)
    incident         the root is NOT known: start from the observed harmful action and
                     work back (Section 5.4). Uses only the store and the destination
                     the action log recorded, never the scenario's ground truth.
    selective_claims roots + harmful descendants; each mixed one split into atomic
                     claims and rewritten from the claims that do not carry the attack
                     (Section 12.3, memheal/claims.py). Under oracle verdicts a claim
                     that only undermines the legitimate system (the labeller's
                     'ambiguous', the attack's premise) is dropped too; cf verdicts know
                     no legitimate system, so there such a claim stays. Opt-in:
                     recover.py's `--mode all` keeps meaning the modes above.
    heal             incident's detection, unchanged, but what it flags is HEALED rather
                     than quarantined (memheal/heal.py): each flagged memory is split
                     into claims, a claim is poisoned if it names the observed
                     destination, if it asserts a change and comes from the roots' pages
                     (source taint), or if the claim-level counterfactual's delta
                     debugging finds it (for the culprits), and every flagged memory,
                     summaries included, keeps its benign claims in lineage order, roots
                     first. No ground truth. Opt-in, like selective_claims.
    heal_oracle      heal with claims labelled by the scenario's AttackSpec instead (the
                     upper bound on claim localisation; same detection, so the pair
                     differs only there). Opt-in.
    heal_rebuild     heal, except that a derived memory over healed parents is rebuilt
                     from them with its recorded prompt (Section 5.7's replay, guarded:
                     no input read off a root's page, no rebuilt claim its inputs do not
                     support) instead of filtered claim by claim. Opt-in.

`rollback` restores an earlier snapshot and re-runs history, so it lives in recover.py.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np

from memheal import claims, db, lineage, memory
from memheal.labels import CARRIES_ATTACK, AttackSpec, kept_benign_content
from memheal.tools import normalize_domain, normalize_url

if TYPE_CHECKING:  # importing the package builds the whole catalog; only the type is needed
    from memheal.scenarios.base import Scenario

MODES = ("none", "reset", "root", "ancestry", "selective", "selective_recon", "unmerge", "incident",
         "selective_claims", "heal", "heal_oracle", "heal_rebuild")
# Modes whose descendant decisions come from verdicts (oracle labels or counterfactuals).
VERDICT_MODES = frozenset({"selective", "selective_recon", "unmerge", "incident",
                           "selective_claims", "heal", "heal_oracle", "heal_rebuild"})
# Modes that heal what incident's detection flags (memheal/heal.py).
HEAL_MODES = ("heal", "heal_oracle", "heal_rebuild")
CF_THRESHOLD = 0.5
# The operation a claim-repaired memory is written with (memories.operation).
CLAIM_REPAIR = "claim_repair"
# The operations heal writes: a healed memory, and the temporary row a claim-level
# counterfactual replay renders in a culprit's place (never active, never retrievable).
HEAL_OP = "heal"
PROBE_OP = "cf_probe"

_SESSION_PREFIX = re.compile(r"^s\d+:")


@dataclass
class RecoveryPlan:
    """What a recovery will do. `quarantine`, `reconstruct`, `repair` and `heal` are
    disjoint: a memory being rebuilt, claim-repaired or healed is quarantined by that
    step itself, after its replacement exists."""
    mode: str
    quarantine: set[str] = field(default_factory=set)
    reconstruct: set[str] = field(default_factory=set)
    restore: set[str] = field(default_factory=set)
    # Mixed memories selective_claims rewrites from their attack-free claims.
    repair: set[str] = field(default_factory=set)
    # Everything a heal mode flags: what incident would quarantine, healed instead.
    heal: set[str] = field(default_factory=set)
    # The counterfactual's minimal flipping sets (heal's culprits, for the claim-level
    # counterfactual). Singletons when a single removal flips.
    culprit_sets: list[list[str]] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    inferred_roots: list[str] = field(default_factory=list)
    # Everything the plan considered tainted (roots, descendants, incident suspects).
    # Reconstruction must not treat any of it as a clean parent, even a spared one.
    lineage: set[str] = field(default_factory=set)
    # The attack description that "names the attacker" is judged against: the
    # scenario's for oracle runs, the observed destination's for cf and incident.
    spec: AttackSpec | None = None


class Verdicts(dict):
    """id -> label, plus why each label was given and the spec it was judged against.

    A plain dict to every caller that only wants labels; `plan` uses the extras so a
    counterfactual verdict shows up as 'cf:flip 2/3' rather than just 'label:harmful'.
    """

    def __init__(self, *a, spec: AttackSpec | None = None, **kw):
        super().__init__(*a, **kw)
        self.reasons: dict[str, str] = {}
        self.spec = spec


# --- verdicts ---------------------------------------------------------------------

def observed_terms(destination: str | None) -> frozenset[str]:
    """The strings a memory can use to name the observed destination.

    The action log keeps the whole normalized value: a path
    ('pkg-mirror-fast.example/install/corvus-auth/2.4.1') or a mailbox
    ('invoices@ap-settlement.example'). Memories mostly name only the host, and the
    word match needs the whole term, so the raw value alone matched nothing in those
    runs and incident inferred no roots. The host is still read off the logged action,
    so this adds no ground truth.
    """
    if not destination:
        return frozenset()
    # Lazy: importing the scenarios package builds the whole catalog.
    from memheal.scenarios.base import candidate_domains
    terms = {destination, normalize_url(destination), normalize_domain(destination),
             *candidate_domains(destination)}
    terms |= {t.rsplit("@", 1)[1] for t in terms if "@" in t}
    return frozenset(t for t in terms if t)


def observed_spec(destination: str | None) -> AttackSpec:
    """An AttackSpec built from nothing but the destination the harmful action used.

    This is all a defender has after an incident: no attacker system name, no list of
    attacker pages. Only memories that name the destination (or its host) count as
    naming the attacker.
    """
    return AttackSpec(attacker_destinations=observed_terms(destination))


def oracle_verdicts(conn: sqlite3.Connection, scenario: Scenario, condition: str) -> Verdicts:
    """Section 7.8 ground-truth labels (the upper bound for verdict-driven modes)."""
    out = Verdicts(lineage.lineage_labels(conn, scenario, condition),
                   spec=AttackSpec.from_scenario(scenario))
    for m, label in out.items():
        out.reasons[m] = f"label:{label}"
    return out


def _score(entry: Any) -> tuple[float | None, str]:
    """Normalize an influence score (a float, or frozen_ablation's {'score', 'outcomes',
    'n'} entry) to (score, 'k/n' text for the reason)."""
    if entry is None:
        return None, ""
    if isinstance(entry, Mapping):
        score = entry.get("score")
        if score is None:
            return None, ""
        n = int(entry.get("n") or len(entry.get("outcomes") or ()) or 1)
        return float(score), f"{round(float(score) * n)}/{n}"
    return float(entry), f"{float(entry):.2f}"


def _split_parents(conn: sqlite3.Connection, memory_id: str, tainted: set[str],
                   spec: AttackSpec) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
    """(clean parents, dirty parents) of one memory, as rows in citation order.

    Dirty = in the tainted set, or naming the attacker even though outside the lineage
    (the attacker's own portal page exists in both worlds, so it is never a root).
    """
    clean, dirty = [], []
    for p in db.parents_of(conn, memory_id):
        row = db.get_memory(conn, p)
        if row is None:
            continue
        (dirty if p in tainted or carries_attacker(row, spec) else clean).append(row)
    return clean, dirty


def carries_attacker(row: sqlite3.Row, spec: AttackSpec) -> bool:
    """Names the attacker, or was read off the attacker's own site.

    A page on the destination host is attacker-authored even where it never repeats
    the host ("Enrol Now handles benefits enrolment", from the portal). The portal
    exists in both worlds, so it is never a poison root and its memories sit outside
    the lineage; with only the observed destination to go on, the text check misses
    them and unmerge re-activated them as clean parents. The source url is in the
    store, so this needs nothing beyond the destination either.
    """
    if spec.names_attacker(row["text"]):
        return True
    src = row["source_id"] if row["source_type"] == "webpage" else None
    return bool(src) and spec.names_attacker(_SESSION_PREFIX.sub("", src))


def cf_verdicts(
    conn: sqlite3.Connection,
    lineage_ids: Iterable[str],
    influence_scores: Mapping[str, Any],
    observed_destination: str | None,
    threshold: float = CF_THRESHOLD,
) -> Verdicts:
    """Counterfactual verdicts (Section 5.4): no ground truth, only measured influence.

    A memory carries the attack if ablating it from the frozen retrieved set flipped the
    decision often enough (score >= threshold), or if its text names the destination the
    harmful action actually used. Most lineage memories were never retrieved by the
    target, so the name check is what reaches them. Harmful vs mixed then uses the same
    kept-benign-content test as the oracle labeller, against the observed-only spec.
    Everything else is benign.
    """
    spec = observed_spec(observed_destination)
    ids = [m for m in dict.fromkeys(lineage_ids)]
    tainted = set(ids)
    out = Verdicts(spec=spec)
    for m in ids:
        row = db.get_memory(conn, m)
        if row is None:
            continue
        score, frac = _score(influence_scores.get(m))
        flipped = score is not None and score >= threshold
        named = spec.names_attacker(row["text"])
        why = []
        if flipped:
            why.append(f"cf:flip {frac}")
        if named:
            why.append(f"name:{observed_destination}")
        if not why:
            out[m] = "benign"
            out.reasons[m] = "cf:untested" if score is None else f"cf:no-flip {frac}"
            continue
        clean, dirty = _split_parents(conn, m, tainted, spec)
        keeps = bool(clean) and kept_benign_content(
            row["text"], [r["text"] for r in clean], [r["text"] for r in dirty], spec)
        out[m] = "mixed" if keeps else "harmful"
        out.reasons[m] = "; ".join(why)
    return out


# --- incident: working back from the observed action --------------------------------

def incident_suspects(
    conn: sqlite3.Connection,
    influence_scores: Mapping[str, Any],
    observed_destination: str | None,
    threshold: float = CF_THRESHOLD,
) -> list[str]:
    """Memories the incident points at: counterfactual flips, plus every stored memory
    (whatever its status) naming the observed destination. A consolidated root is not
    retrievable but still names it, which is how the incident finds its way back.
    heal's probe rows are measurement scaffolding, not memories, and never count."""
    spec = observed_spec(observed_destination)
    flips = {m for m, e in influence_scores.items()
             if (s := _score(e)[0]) is not None and s >= threshold}
    return [r["id"] for r in db.all_memories(conn)
            if r["status"] != "deleted" and r["operation"] != PROBE_OP
            and (r["id"] in flips or (observed_destination and spec.names_attacker(r["text"])))]


def infer_roots(conn: sqlite3.Connection, suspects: Iterable[str],
                observed_destination: str | None) -> list[str]:
    """Parentless ancestors-or-self of the suspects that name the observed destination or
    are suspects themselves, in store order.

    A suspect summary cites the poisoned extraction and clean ones alike; only the
    ancestor that still names the destination is a root. Uses only the store and the
    destination, never is_poison or the scenario.
    """
    spec = observed_spec(observed_destination)
    suspects = {s for s in suspects if db.get_memory(conn, s) is not None}
    candidates: set[str] = set()
    for s in suspects:
        candidates |= {s} | db.get_ancestors(conn, s)
    roots = set()
    for c in candidates:
        if db.parents_of(conn, c):
            continue
        row = db.get_memory(conn, c)
        if row is not None and (c in suspects or spec.names_attacker(row["text"])):
            roots.add(c)
    return _store_order(conn, roots)


# --- planning ---------------------------------------------------------------------

def _store_order(conn: sqlite3.Connection, ids: Iterable[str]) -> list[str]:
    wanted = set(ids)
    return [r["id"] for r in db.all_memories(conn) if r["id"] in wanted]


def _absorbs(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    """Does this memory stand for its parents? A summary (or its rebuild) does: its
    parents were marked 'consolidated' because of it. A reflection cites its parents
    but leaves them active, so it absorbed nothing. A claim repair, or a healed memory,
    stands for what the memory it replaced stood for; that memory is its first parent
    (the edge written with it, before any absorber edge)."""
    original = _replaced_by(conn, row)
    if original is not None:
        return _absorbs(conn, original)
    if row["operation"] in (CLAIM_REPAIR, HEAL_OP):
        return False
    return row["operation"] in ("summarize", "reconstruct") and not _is_reflection(conn, row)


def _replaced_by(conn: sqlite3.Connection, row) -> sqlite3.Row | None:
    """The memory a claim repair or a healed memory replaced (its first parent), else None."""
    if row["operation"] not in (CLAIM_REPAIR, HEAL_OP):
        return None
    first = next(iter(db.parents_of(conn, row["id"])), None)
    return db.get_memory(conn, first) if first is not None else None


def _surviving_absorbers(conn: sqlite3.Connection, memory_id: str, gone: set[str]) -> list[str]:
    """Direct absorbing children of `memory_id` that this recovery leaves in place."""
    out = []
    for c in db.children_of(conn, memory_id):
        row = db.get_memory(conn, c)
        if (row is not None and c not in gone and row["status"] in ("active", "consolidated")
                and _absorbs(conn, row)):
            out.append(c)
    return out


def still_absorbed(conn: sqlite3.Connection, memory_id: str, gone: Iterable[str]) -> bool:
    """Is `memory_id`'s content still retrievable through an active summary that this
    recovery keeps? Follows absorbing edges down any number of levels (a summary of a
    summary), so a parent of a removed summary that a spared summary above it still
    carries is not brought back as a duplicate. `gone` = memories being removed."""
    gone = set(gone)
    seen: set[str] = set()
    stack = [memory_id]
    while stack:
        for c in db.children_of(conn, stack.pop()):
            if c in seen:
                continue
            seen.add(c)
            row = db.get_memory(conn, c)
            if row is None or not _absorbs(conn, row):
                continue
            if row["status"] == "active" and c not in gone:
                return True
            stack.append(c)
    return False


def restore_parents(
    conn: sqlite3.Connection,
    memory_id: str,
    tainted: set[str],
    spec: AttackSpec,
    *,
    removed: Iterable[str] = (),
) -> list[str]:
    """The clean 'consolidated' memories a quarantined summary had absorbed (a query;
    `apply` does the re-activating).

    Walks through parents that are tainted or quarantined summaries to their own
    parents, so a summary of a summary gives back the clean facts at every level. A
    tainted parent that stays active (a spared benign summary) still stands for what it
    absorbed, so the walk stops there; `removed` names memories this same plan is
    quarantining, which count as gone. A reflection absorbed nothing, so neither the
    start nor any memory walked through may be one: a reflection's parent that shows
    up 'consolidated' was absorbed by some other summary. And a parent some surviving
    active summary still carries is left alone, or it would be retrievable twice.
    Never returns a tainted memory or one naming the attacker or read off its site.
    """
    gone = set(removed) | {memory_id}
    start = db.get_memory(conn, memory_id)
    if start is None or not _absorbs(conn, start):
        return []
    out: list[str] = []
    seen: set[str] = set()

    def visit(p: str) -> None:
        if p in seen:
            return
        seen.add(p)
        row = db.get_memory(conn, p)
        if row is None:
            return
        if p in tainted or row["status"] == "quarantined":
            if (row["status"] != "active" or p in gone) and _absorbs(conn, row):
                for q in db.parents_of(conn, p):
                    visit(q)
            return
        if (row["status"] == "consolidated" and not carries_attacker(row, spec)
                and not still_absorbed(conn, p, gone)):
            out.append(p)

    for p in db.parents_of(conn, memory_id):
        visit(p)
    return out


def _empty_lineage(roots, descendants) -> bool:
    return not roots and not descendants


def plan(
    conn: sqlite3.Connection,
    scenario: Scenario | None,
    condition: str | None,
    mode: str,
    verdicts: Mapping[str, str] | None = None,
    roots: Iterable[str] | None = None,
    descendants: Iterable[str] | None = None,
    *,
    spec: AttackSpec | None = None,
    observed_destination: str | None = None,
    influence_scores: Mapping[str, Any] | None = None,
    threshold: float = CF_THRESHOLD,
    culprit_sets: Iterable[Iterable[str]] | None = None,
) -> RecoveryPlan:
    """Decide what `mode` would do to this store; writes nothing.

    roots/descendants default to the harness's poison lineage. `incident` never takes
    that default: it needs `observed_destination` (and the counterfactual
    `influence_scores`, if any were measured) and infers its own roots, so it can be
    called with scenario=None. heal does the same; heal_oracle also needs the scenario,
    for its claim labels. `culprit_sets` (heal only) are the counterfactual's minimal
    flipping sets; they default to the flipping singletons of `influence_scores`.
    `spec` defaults to the verdicts' own (cf), else the scenario's (oracle), else the
    observed destination's.
    """
    if mode not in MODES:
        raise ValueError(f"unknown recovery mode {mode!r}; expected one of {MODES}")
    p = RecoveryPlan(mode)
    if mode == "none":
        return p
    if mode == "reset":
        for r in db.active_memories(conn):
            p.quarantine.add(r["id"])
            p.reasons[r["id"]] = "reset"
        return p

    if mode == "incident":
        return _plan_incident(conn, p, verdicts, roots, descendants, spec,
                              observed_destination, influence_scores or {}, threshold)
    if mode in HEAL_MODES:
        return _plan_heal(conn, p, scenario, verdicts, roots, descendants, spec,
                          observed_destination, influence_scores or {}, threshold,
                          culprit_sets)

    if roots is None or descendants is None:
        lin_roots, lin_desc = lineage.poison_lineage(conn, scenario, condition)
        roots = lin_roots if roots is None else roots
        descendants = lin_desc if descendants is None else descendants
    roots = list(roots)
    descendants = set(descendants) - set(roots)
    p.lineage = set(roots) | descendants
    p.spec = spec or getattr(verdicts, "spec", None) or (
        AttackSpec.from_scenario(scenario) if scenario is not None
        else observed_spec(observed_destination))

    for r in roots:
        p.quarantine.add(r)
        p.reasons[r] = "root"
    if mode == "root" or _empty_lineage(roots, descendants):
        return p
    if mode == "ancestry":
        for d in descendants:
            p.quarantine.add(d)
            p.reasons[d] = "descendant"
        return p

    if verdicts is None:
        verdicts = oracle_verdicts(conn, scenario, condition)
    _select(p, descendants, verdicts, rebuild_mixed=(mode == "selective_recon"),
            repair_mixed=(mode == "selective_claims"))
    if mode == "unmerge":
        _plan_restore(conn, p)
    return p


def _select(p: RecoveryPlan, descendants: Iterable[str], verdicts: Mapping[str, str],
            *, rebuild_mixed: bool, repair_mixed: bool = False) -> None:
    """Verdict-driven descendant choices. Descendants without a verdict are spared:
    a verdict source that never looked at a memory has no grounds to remove it."""
    reasons = getattr(verdicts, "reasons", {})
    for d in descendants:
        label = verdicts.get(d)
        if label not in CARRIES_ATTACK:
            continue
        why = reasons.get(d) or f"label:{label}"
        if rebuild_mixed and label == "mixed":
            p.reconstruct.add(d)
            p.reasons[d] = f"{why}; reconstruct"
        elif repair_mixed and label == "mixed":
            p.repair.add(d)
            p.reasons[d] = f"{why}; claim-repair"
        else:
            p.quarantine.add(d)
            p.reasons[d] = why


def _plan_restore(conn: sqlite3.Connection, p: RecoveryPlan) -> None:
    tainted = p.lineage | p.quarantine
    for q in _store_order(conn, p.quarantine):
        for parent in restore_parents(conn, q, tainted, p.spec, removed=p.quarantine):
            if parent not in p.restore:
                p.restore.add(parent)
                p.reasons[parent] = f"clean-parent-of:{q}"


def _plan_incident(conn, p: RecoveryPlan, verdicts, roots, descendants, spec,
                   observed_destination, influence_scores, threshold) -> RecoveryPlan:
    """Root unknown: suspects from the incident, roots inferred from them, then selective.

    Suspects that do not trace back to an inferred root (a summary whose poisoned
    extraction was paraphrased past naming the destination) are judged too, so the
    memories the incident pointed at are never silently left behind.
    """
    p.spec = spec or getattr(verdicts, "spec", None) or observed_spec(observed_destination)
    suspects = incident_suspects(conn, influence_scores, observed_destination, threshold)
    roots = infer_roots(conn, suspects, observed_destination) if roots is None else list(roots)
    if descendants is None:
        descendants = set()
        for r in roots:
            descendants |= db.get_descendants(conn, r)
    p.inferred_roots = list(roots)
    p.lineage = set(roots) | set(descendants) | set(suspects)
    if not p.lineage:
        return p
    if verdicts is None:
        verdicts = cf_verdicts(conn, _store_order(conn, p.lineage), influence_scores,
                               observed_destination, threshold)
    for r in roots:
        p.quarantine.add(r)
        p.reasons[r] = "inferred-root"
    _select(p, p.lineage - set(roots), verdicts, rebuild_mixed=False)
    return p


def _plan_heal(conn, p: RecoveryPlan, scenario, verdicts, roots, descendants, spec,
               observed_destination, influence_scores, threshold, culprit_sets) -> RecoveryPlan:
    """Detection exactly as incident; everything incident would quarantine is healed.

    heal and heal_rebuild judge claims against the observed destination alone (p.spec
    stays incident's observed spec); they differ only in how memheal/heal.py heals what
    is flagged. heal_oracle swaps in the scenario's AttackSpec for the claim labels
    only: the flagged set is the same, so heal vs heal_oracle measures claim
    localisation and nothing else.
    """
    if p.mode == "heal_oracle" and scenario is None and spec is None:
        raise ValueError("heal_oracle labels claims with the scenario's AttackSpec; "
                         "pass the scenario")
    _plan_incident(conn, p, verdicts, roots, descendants, None if p.mode == "heal_oracle" else spec,
                   observed_destination, influence_scores, threshold)
    p.heal, p.quarantine = p.quarantine, set()
    if culprit_sets is None:
        culprit_sets = [[m] for m, e in influence_scores.items()
                        if (s := _score(e)[0]) is not None and s >= threshold]
    p.culprit_sets = [list(s) for s in culprit_sets]
    if p.mode == "heal_oracle":
        p.spec = spec or AttackSpec.from_scenario(scenario)
    return p


# --- applying ---------------------------------------------------------------------

class _CountingLLM:
    """Counts the respond() calls recovery makes, whatever the underlying client."""

    def __init__(self, llm):
        self._llm = llm
        self.calls = 0

    def respond(self, *a, **kw):
        if self._llm is None:
            raise RuntimeError("this recovery needs an LLM but none was given")
        self.calls += 1
        return self._llm.respond(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._llm, name)


def _quarantine(conn: sqlite3.Connection, memory_id: str) -> bool:
    """Quarantine unless already quarantined or deleted; True if the status changed."""
    row = db.get_memory(conn, memory_id)
    if row is None or row["status"] in ("quarantined", "deleted"):
        return False
    return db.set_status(conn, memory_id, "quarantined")


def _reactivate(conn: sqlite3.Connection, memory_id: str, gone: set[str] | None = None) -> bool:
    """consolidated -> active. With `gone`, only if no summary that survives the
    recovery still carries the memory; otherwise its facts would be retrievable twice."""
    row = db.get_memory(conn, memory_id)
    if row is None or row["status"] != "consolidated":
        return False
    if gone is not None and still_absorbed(conn, memory_id, gone):
        return False
    return db.set_status(conn, memory_id, "active")


def _embed(embedder, texts: list[str]) -> list[bytes]:
    return [v.astype(np.float32).tobytes() for v in embedder.encode(texts)]


def _is_reflection(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    if row["operation"] == "reflect":
        return True
    if row["operation"] == HEAL_OP:
        # A healed memory is whatever it healed; a rebuilt reflection keeps the
        # reflection rule if a later round rebuilds it again.
        original = _replaced_by(conn, row)
        return original is not None and _is_reflection(conn, original)
    # A reconstruct of a reflection keeps the reflection rule if it is rebuilt again.
    prompt = db.get_prompt(conn, row["created_by_prompt"]) if row["created_by_prompt"] else None
    return row["operation"] == "reconstruct" and prompt is not None \
        and prompt["name"].startswith("reconstruct.reflect")


def reconstruct(
    conn: sqlite3.Connection,
    llm,
    embedder: memory.Embedder,
    memory_id: str,
    tainted: set[str],
    spec: AttackSpec,
    recovery_id: str,
    mode: str,
    *,
    removed: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Rebuild one mixed memory from its clean parents only (Section 5.7).

    Always quarantines the memory itself and logs one event for it (action
    'reconstruct' when a replacement was written, else 'quarantine'). Returns
    {action: reconstruct|restore|quarantine, replacement_ids, restored}.

    `removed` is what the recovery takes out (apply passes its plan's quarantine and
    reconstruct sets); a summary outside it that stays active still carries what it
    absorbed, so a clean parent it carries is not re-activated and a replacement for a
    memory it absorbed is written 'consolidated' under it. Called on its own, with no
    plan, every tainted memory counts as removed.
    """
    row = db.get_memory(conn, memory_id)
    if row is None:
        return {"action": "quarantine", "replacement_ids": [], "restored": []}
    gone = (set(tainted) if removed is None else set(removed)) | {memory_id}
    clean, _ = _split_parents(conn, memory_id, tainted | {memory_id}, spec)
    clean_ids = [r["id"] for r in clean]

    new_ids: list[str] = []
    restored: list[str] = []
    if not clean:
        outcome = "reconstruct:no-clean-parents"
    elif len(clean) == 1:
        # One clean parent is the whole of the benign content; a replay would only
        # restate it. Bring it back as it was, with no model call.
        outcome = "reconstruct:1-clean-parent"
        if _reactivate(conn, clean_ids[0], gone):
            restored.append(clean_ids[0])
    elif _is_reflection(conn, row):
        new_ids, outcome = _rewrite_reflection(conn, llm, embedder, row, clean, spec, gone)
    else:
        new_ids, restored, outcome = _replay_summary(conn, llm, embedder, row, clean, spec, gone)

    _quarantine(conn, memory_id)
    replacement = ",".join(new_ids) or (",".join(restored) if outcome.endswith("1-clean-parent")
                                         else None)
    db.log_recovery(conn, recovery_id, mode, memory_id,
                    "reconstruct" if new_ids else "quarantine",
                    replacement=replacement, reason=outcome)
    for r in restored:
        db.log_recovery(conn, recovery_id, mode, r, "restore", reason=f"clean-parent-of:{memory_id}")
    action = "reconstruct" if new_ids else ("restore" if restored else "quarantine")
    return {"action": action, "replacement_ids": new_ids, "restored": restored}


def _insert_replacements(conn, row, texts: list[str], parents: list[list[str]], embedder,
                         prompt_hash: str, model: str, gone: set[str]) -> list[str]:
    """Write a rebuilt memory's replacements.

    A memory that was itself absorbed into a summary this recovery keeps was not
    retrievable before; its replacement goes in 'consolidated', with an edge to that
    summary, so recovery does not surface content the store already carries once.
    """
    absorbers = (_surviving_absorbers(conn, row["id"], gone)
                 if still_absorbed(conn, row["id"], gone) else [])
    status = "consolidated" if absorbers else "active"
    new_ids = []
    for text, cited, vec in zip(texts, parents, _embed(embedder, texts)):
        new = db.insert_memory(
            conn, text, "derived", row["session"],
            operation="reconstruct",
            status=status,
            created_by_prompt=prompt_hash,
            model=model,
            embedding=vec,
            parent_ids=cited,
        )
        for a in absorbers:
            db.insert_edges(conn, a, [new])
        new_ids.append(new)
    return new_ids


def _summary_outputs(conn, llm, row, inputs: list[sqlite3.Row], *,
                     tag: str) -> tuple[list[tuple[str, list[str]]], str, str | None]:
    """The recorded consolidation prompt replayed over `inputs` alone: each valid summary
    as (text, cited input ids), with the prompt's hash and the model that answered.

    The recorded text is what produced the memory, so replaying it holds the
    transformation fixed and changes only its input (Section 5.7). Same validity rule as
    memory.consolidate: real, distinct citations, at least two. Shared by reconstruct
    and heal, which differ only in what they do with the summaries.
    """
    recorded = db.get_prompt(conn, row["created_by_prompt"]) if row["created_by_prompt"] else None
    if recorded is not None:
        prompt, prompt_hash = recorded["text"], recorded["prompt_hash"]
    else:
        prompt = memory.CONSOLIDATION_PROMPTS["default"]
        prompt_hash = db.register_prompt(conn, "consolidate.default", prompt)

    listing = "\n".join(f"{i}. {r['text']}" for i, r in enumerate(inputs, start=1))
    response = llm.respond(prompt, f"Memory:\n{listing}",
                           json_schema=memory.CONSOLIDATION_SCHEMA, tag=tag)
    summaries = json.loads(response.text or '{"summaries": []}').get("summaries", [])
    out: list[tuple[str, list[str]]] = []
    for s in summaries:
        text = (s.get("text") or "").strip()
        idx = sorted({i for i in s.get("parents", []) if isinstance(i, int) and 1 <= i <= len(inputs)})
        if text and len(idx) >= 2:
            out.append((text, [inputs[i - 1]["id"] for i in idx]))
    return out, prompt_hash, response.model


def _replay_summary(conn, llm, embedder, row, clean: list[sqlite3.Row],
                    spec: AttackSpec, gone: set[str]) -> tuple[list[str], list[str], str]:
    """Replay the recorded consolidation prompt over just the clean parents.

    Returns (new ids, restored ids, outcome reason).
    """
    outputs, prompt_hash, model = _summary_outputs(conn, llm, row, clean,
                                                   tag=f"reconstruct:summarize:{row['id']}")
    kept: list[tuple[str, list[str]]] = []
    suspect: set[str] = set()
    for text, cited in outputs:
        if spec.names_attacker(text):
            # The attack came back from clean inputs alone. Discard the text, and do not
            # trust the parents it drew on back into retrieval either.
            suspect.update(cited)
            continue
        kept.append((text, cited))

    clean_ids = [r["id"] for r in clean]
    if not kept and suspect:
        restore = [c for c in clean_ids if c not in suspect]
        restored = [c for c in restore if _reactivate(conn, c, gone)]
        return [], restored, "reconstruct-still-tainted"
    if not kept:
        # Nothing valid came back; the clean parents are the benign content, keep them.
        return [], [c for c in clean_ids if _reactivate(conn, c, gone)], "reconstruct:empty-replay"

    new_ids = _insert_replacements(conn, row, [t for t, _ in kept], [c for _, c in kept],
                                   embedder, prompt_hash, model, gone)
    cited_all = {c for _, cited in kept for c in cited}
    # Mirrors consolidate: the new summary stands for what it cites.
    db.set_status_many(conn, [c for c in cited_all
                              if db.get_memory(conn, c)["status"] == "active"], "consolidated")
    # A clean parent the replay left out would otherwise vanish with the old summary.
    uncited = [c for c in clean_ids if c not in cited_all and c not in suspect]
    restored = [c for c in uncited if _reactivate(conn, c, gone)]
    return new_ids, restored, "reconstruct:replayed"


def _rewrite_reflection(conn, llm, embedder, row, clean: list[sqlite3.Row],
                        spec: AttackSpec, gone: set[str]) -> tuple[list[str], str]:
    """Rewrite a mixed procedure from its clean parents only.

    The original reflection prompt saw the session's actions, which are what the
    attacker steered, so replaying it would bring the taint back. Its parents are
    still active, so an empty answer just means quarantine.
    """
    facts, prompt_hash, model = _reflection_facts(conn, llm, clean,
                                                  tag=f"reconstruct:reflect:{row['id']}")
    tainted_facts = [f for f in facts if spec.names_attacker(f)]
    facts = [f for f in facts if not spec.names_attacker(f)]
    if not facts:
        return [], "reconstruct-still-tainted" if tainted_facts else "reconstruct:empty-replay"
    parent_ids = [r["id"] for r in clean]
    new_ids = _insert_replacements(conn, row, facts, [parent_ids] * len(facts), embedder,
                                   prompt_hash, model, gone)
    return new_ids, "reconstruct:reflect"


def _reflection_facts(conn, llm, inputs: list[sqlite3.Row], *,
                      tag: str) -> tuple[list[str], str, str | None]:
    """The reflection-rewrite prompt over `inputs` alone: the procedure facts they
    support, the prompt's hash and the model that answered. Shared by reconstruct and
    heal."""
    prompt = memory.RECONSTRUCT_PROMPTS["reflect"]
    prompt_hash = db.register_prompt(conn, "reconstruct.reflect", prompt)
    listing = "\n".join(f"{i}. {r['text']}" for i, r in enumerate(inputs, start=1))
    response = llm.respond(prompt, f"Memory:\n{listing}", json_schema=memory.FACTS_SCHEMA, tag=tag)
    facts = [f.strip() for f in json.loads(response.text or '{"facts": []}').get("facts", [])
             if f.strip()]
    return facts, prompt_hash, response.model


# --- claim repair (Section 12.3) ------------------------------------------------------

def claim_carries_attack(claim: str, row: sqlite3.Row, spec: AttackSpec) -> bool:
    """carries_attacker for one claim of `row`: it names the attacker, or the memory it
    came from was read off the attacker's own site (then every claim is theirs).

    One rule more than a whole memory gets: naming a destination's host without its
    top-level domain ("the quick-expense portal", "Quick Expense") counts. A split is a
    rewrite, and a rewrite can drop the '.example' that the whole-host match needs; the
    observed spec (cf) has no system name to catch "Quick Expense" otherwise.
    """
    return (carries_attacker({"text": claim, "source_type": row["source_type"],
                              "source_id": row["source_id"]}, spec)
            or claims.names_host_label(claim, spec.attacker_destinations))


def claim_is_ambiguous(claim: str, spec: AttackSpec) -> bool:
    """A claim that undermines the legitimate system ("Summit Registry requests are no
    longer processed") without naming the attacker: the labeller's 'ambiguous', and the
    premise the attack works from. Only a spec that knows the legitimate system (the
    oracle's) can tell; the observed spec never does."""
    return spec.undermines_legacy(claim)


def _standing_for(conn: sqlite3.Connection, memory_id: str, gone: set[str],
                  replaced: Mapping[str, str]) -> list[str]:
    """What carries an absorbed memory once this recovery is done: absorbing summaries
    it keeps, and the claim repairs of absorbing summaries it rewrote."""
    out = _surviving_absorbers(conn, memory_id, gone)
    for c in db.children_of(conn, memory_id):
        row = db.get_memory(conn, c)
        if c in replaced and row is not None and _absorbs(conn, row):
            out.append(replaced[c])
    return list(dict.fromkeys(out))


def repair_claims(
    conn: sqlite3.Connection,
    llm,
    embedder: memory.Embedder,
    memory_id: str,
    spec: AttackSpec,
    recovery_id: str,
    mode: str,
    *,
    removed: Iterable[str] | None = None,
    replaced: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Claim-level repair of one mixed memory (Section 12.3).

    The memory is split into atomic claims (claims.split_claims: the run's model, or the
    deterministic split when `llm` is None or answers badly). Each claim is judged by the
    rule a whole memory is judged by (claim_carries_attack against `spec`: the scenario's
    for oracle verdicts, the observed destination's for cf), and the claims that do not
    carry the attack are joined, unregenerated, into one new memory (operation
    'claim_repair', edge from the original). A claim that only undermines the legitimate
    system is dropped as well, and counted ('ambiguous=N' in the reason): selective and
    selective_recon lose it with the memory, so keeping it would compare unlike with
    like. Only the oracle spec knows the legitimate system, so under cf it stays.
    The original is always quarantined, with one recovery event. Nothing is written when
    no claim survives, nor when no claim carries the attack: the verdict says the memory
    does, so a split that cannot find where keeps nothing rather than everything, which
    is selective's answer.

    Status: a retrievable memory stays retrievable through its repair. A 'consolidated'
    one stays consolidated under whatever still carries it (a summary this recovery
    keeps, or the repair of one it rewrote: `replaced`, original -> repair); with nothing
    left carrying it, its repair is the only copy of its benign claims and goes in active.

    Returns {action: claim_repair|quarantine, replacement_ids, claims, kept, dropped,
    ambiguous, how}.
    """
    out: dict[str, Any] = {"action": "quarantine", "replacement_ids": [], "claims": [],
                           "kept": [], "dropped": [], "ambiguous": [], "how": None}
    row = db.get_memory(conn, memory_id)
    if row is None:
        return out
    if row["status"] in ("quarantined", "deleted"):
        # Not retrievable and carried by nothing: a repair would surface what the store
        # had already put away.
        db.log_recovery(conn, recovery_id, mode, memory_id, "quarantine",
                        reason=f"claim-repair:already-{row['status']}")
        return out
    gone = (set() if removed is None else set(removed)) | {memory_id}
    replaced = replaced or {}
    split = claims.split_claims(llm, row["text"], tag=f"claims:{memory_id}")
    kept, dropped = claims.partition(split.claims, lambda c: claim_carries_attack(c, row, spec))
    kept, ambiguous = claims.partition(kept, lambda c: claim_is_ambiguous(c, spec))
    text = claims.compose(kept)
    out.update(claims=split.claims, kept=kept, dropped=dropped, ambiguous=ambiguous,
               how=split.how)
    reason = f"claims:{split.how} kept={len(kept)}/{len(split.claims)}"
    if ambiguous:
        reason += f" ambiguous={len(ambiguous)}"
    if split.ungrounded:
        reason += f" ungrounded={len(split.ungrounded)}"

    new_id = None
    if not dropped:
        reason += "; no claim carries the attack"
    elif not kept:
        reason += "; no claim survives"
    elif claim_carries_attack(text, row, spec):  # joining adds no words; a last guard
        reason += "; still tainted"
    else:
        consolidated = row["status"] == "consolidated" and still_absorbed(conn, memory_id, gone)
        standing = _standing_for(conn, memory_id, gone, replaced) if consolidated else []
        prompt_hash = (db.register_prompt(conn, "claims.split", claims.CLAIMS_PROMPT)
                       if split.how == "llm" else None)
        new_id = db.insert_memory(
            conn, text, "derived", row["session"],
            operation=CLAIM_REPAIR,
            status="consolidated" if consolidated else "active",
            created_by_prompt=prompt_hash,
            model=split.model,
            embedding=_embed(embedder, [text])[0],
            parent_ids=[memory_id],
        )
        for a in standing:
            db.insert_edges(conn, a, [new_id])
        out.update(action=CLAIM_REPAIR, replacement_ids=[new_id])

    _quarantine(conn, memory_id)
    db.log_recovery(conn, recovery_id, mode, memory_id, CLAIM_REPAIR if new_id else "quarantine",
                    replacement=new_id, reason=reason)
    return out


def apply(
    conn: sqlite3.Connection,
    plan: RecoveryPlan,
    llm,
    embedder: memory.Embedder | None,
    recovery_id: str,
    spec: AttackSpec | None = None,
    *,
    heal=None,
) -> dict[str, Any]:
    """Carry out a plan: quarantine, then reconstruct, then claim-repair, then restore,
    then heal.

    That order matters. Reconstruction judges parents against the tainted set, so the
    quarantines must be in; restoration comes last so a parent a reconstruction just
    cited (and so re-absorbed) is not also put back on its own. A heal plan has nothing
    in the other sets; memheal/heal.py carries it out, with `heal` (a heal.HealContext:
    the harness's claim-level replays and the split cache) when the caller has one.
    Returns {counts, quarantined, new_ids, restored, claim_repaired, claim_ambiguous,
    claims_kept, claims_dropped, llm_calls}, plus 'heal' (heal.apply's summary) for the
    heal modes; llm_calls counts only the calls made here, claim_repaired lists the
    memories a claim repair replaced (their replacements are in new_ids),
    claim_ambiguous counts the claims those repairs dropped for undermining the
    legitimate system alone, and claims_kept/claims_dropped count the claims written
    into replacements and the claims judged to carry the attack (claim repair and heal).
    """
    spec = spec or plan.spec or observed_spec(None)
    counting = _CountingLLM(llm)
    quarantined: list[str] = []
    new_ids: list[str] = []
    restored: list[str] = []
    n_reconstructed = 0

    for m in _store_order(conn, plan.quarantine):
        if _quarantine(conn, m):
            quarantined.append(m)
        db.log_recovery(conn, recovery_id, plan.mode, m, "quarantine", reason=plan.reasons.get(m))

    tainted = plan.lineage | plan.quarantine | plan.reconstruct | plan.repair
    removed = plan.quarantine | plan.reconstruct | plan.repair
    for m in _store_order(conn, plan.reconstruct):
        row = db.get_memory(conn, m)
        before = row["status"] if row is not None else "deleted"
        out = reconstruct(conn, counting, embedder, m, tainted, spec, recovery_id, plan.mode,
                          removed=removed)
        if before not in ("quarantined", "deleted"):
            quarantined.append(m)
        new_ids += out["replacement_ids"]
        restored += [r for r in out["restored"] if r not in restored]
        n_reconstructed += bool(out["replacement_ids"])

    # Newest first: a summary is repaired before the memories it absorbed, so their
    # repairs can stay consolidated under its repair instead of surfacing twice.
    replaced: dict[str, str] = {}
    n_ambiguous = 0
    claims_kept = claims_dropped = 0
    for m in reversed(_store_order(conn, plan.repair)):
        row = db.get_memory(conn, m)
        before = row["status"] if row is not None else "deleted"
        out = repair_claims(conn, counting if llm is not None else None, embedder, m, spec,
                            recovery_id, plan.mode, removed=removed, replaced=replaced)
        if before not in ("quarantined", "deleted"):
            quarantined.append(m)
        claims_dropped += len(out["dropped"]) + len(out["ambiguous"])
        if out["replacement_ids"]:
            replaced[m] = out["replacement_ids"][0]
            new_ids += out["replacement_ids"]
            n_ambiguous += len(out["ambiguous"])
            claims_kept += len(out["kept"])

    for m in _store_order(conn, plan.restore):
        if m in plan.quarantine or m in tainted:
            continue
        if _reactivate(conn, m, removed):
            restored.append(m)
            db.log_recovery(conn, recovery_id, plan.mode, m, "restore", reason=plan.reasons.get(m))

    healed = None
    if plan.mode in HEAL_MODES:
        from memheal import heal as healing          # heal builds on this module
        healed = healing.apply(conn, plan, counting, embedder, recovery_id, spec,
                               context=heal, has_llm=llm is not None)
        quarantined += [m for m in healed["quarantined"] if m not in quarantined]
        new_ids += healed["new_ids"]
        restored += [r for r in healed["restored"] if r not in restored]
        claims_kept += healed["claims_kept"]
        claims_dropped += healed["claims_dropped"]

    out = {
        "counts": {"quarantined": len(quarantined), "reconstructed": n_reconstructed,
                   "new": len(new_ids), "restored": len(restored)},
        "quarantined": quarantined,
        "new_ids": new_ids,
        "restored": restored,
        "claim_repaired": list(replaced),
        "claim_ambiguous": n_ambiguous,
        "claims_kept": claims_kept,
        "claims_dropped": claims_dropped,
        "llm_calls": counting.calls,
    }
    if healed is not None:
        out["heal"] = healed
    return out


# --- immunize ---------------------------------------------------------------------

def immunize(conn: sqlite3.Connection, urls: Iterable[str], session: int | None = None) -> list[str]:
    """+immunize: block the poisoned sources so later reads of them are written
    'quarantined' (memory.store). Returns the normalized patterns, deduplicated."""
    patterns = [db.block_source(conn, u, reason="immunize", session=session) for u in urls]
    return list(dict.fromkeys(patterns))


def destination_hosts(destinations: Iterable[str]) -> list[str]:
    """Bare hosts of destinations as the log or the scenario names them: a path
    ('pkg-mirror-fast.example/install/x') or a mailbox ('invoices@ap-settlement.example')
    gives its host; a bare word with no dot is not a host and is skipped."""
    hosts = {normalize_domain(d.rsplit("@", 1)[-1]) for d in destinations if d}
    return sorted(h for h in hosts if "." in h)


def source_urls(conn: sqlite3.Connection, memory_ids: Iterable[str]) -> list[str]:
    """Webpage urls the given memories were extracted from, for immunizing an
    incident's inferred roots without consulting the scenario's poisoned pages."""
    out = []
    for m in memory_ids:
        row = db.get_memory(conn, m)
        if row is not None and row["source_type"] == "webpage" and row["source_id"]:
            out.append(_SESSION_PREFIX.sub("", row["source_id"]))
    return list(dict.fromkeys(out))
