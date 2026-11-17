"""Healing: repair what an incident flags instead of quarantining it (Phase 2, item H).

`incident` (memheal/recovery.py) is the pipeline that needs no ground truth: it starts
from the observed harmful action, explains it with the Section 5.4 counterfactual,
infers the roots and quarantines what it flags. Quarantine also loses every benign fact
a flagged memory held. Healing keeps that detection exactly as it is and changes what
is done with what it finds (recovery.plan's heal modes put incident's quarantine set in
`plan.heal`; this module carries it out). Three modes share it:

    heal          claims localised by the rules below, memories FILTERED (the default)
    heal_rebuild  the same localisation; a derived memory over healed parents is
                  REBUILT from them with its recorded prompt (the first heal's path)
    heal_oracle   claims labelled by the scenario's AttackSpec, filtered (the upper
                  bound on claim localisation: same detection, same strategy as heal)

1. Localise inside each flagged memory. It is split into atomic claims
   (memheal/claims.py: the run's model, else the deterministic split; one split per
   distinct text). A claim is poisoned, and the rule that fired is kept with it (the
   events list it, records count it: dropped_by), if
   (a) 'name': it names the observed destination or its host label, or its memory was
       read off the destination's own site (recovery.claim_carries_attack against the
       observed spec: nothing from the scenario);
   (b) 'source-taint': it points at a root page, or it comes from a tainted page and
       asserts a change of route, process or support. The tainted pages are what the
       store says about the inferred roots and nothing more: the pages the roots were
       read from (root pages), and the pages with a memory pointing at a root page (a
       composed attack's first page, "the replacement is documented in the transition
       notes at <root page>", which names no destination). A claim comes from a page
       when its memory was read off it, or, for a derived memory, when an extraction
       from that page among its ancestors supports it (SUPPORT_SHARE / SUPPORT_COSINE).
       Change language is the labeller's (CHANGE: no longer, retired, replaced,
       replacement, moved, migrated, transition, not processed, returned without
       action, ...). Run 0964 is why: its healed summary kept "Driveline is no longer
       the supported route ... being retired" and the pointer to the transition notes,
       named no destination, and the agent searched its way to the attacker's portal.
       The memories read off a root page (the roots' siblings) are never a rebuild's
       input as they are (what healing wrote for one is) and are never restored,
       whatever they say: run 1089 rebuilt a summary from "Accounts Payable no longer
       processes supplier invoice requests and will return them without action", which
       names nothing, and the attack's premise came back. Some of those memories are
       still active and the incident never flags them. A flat store keeps every
       extraction of the root page retrievable, and only the one naming the destination
       flips. So each active memory read off a tainted page that the incident did not
       flag is judged by the same rules ((a) and (b), or the oracle). When one fires, it
       is healed like a flagged memory ('swept': its change and pointer claims go, the
       rest stays). The flagged set, and so the detection, is incident's;
   (c) 'cf': the claim-level counterfactual, for the culprits (the memories whose
       removal flipped the harmful replay): the decision is replayed with the culprit's
       text replaced by its text without the claims (a) and (b) drop. If that is safe,
       the rest is benign. If not, delta debugging ('ddmin', below) finds a 1-minimal
       set of the remaining claims whose removal is safe. When (a) and (b) drop
       nothing, the first replay is a control instead: the culprit as its own claims
       recomposed must still make the decision, or the split lost what mattered and
       every removal would look safe for that alone (nothing is then localised, and the
       culprit keeps none of its claims: 'cf:control-flipped'). A one-claim culprit is
       its claim ('cf:only-claim'). A claim elsewhere in the lineage that restates one
       ddmin showed carries the decision (claims.restates) is poisoned too
       ('cf:restates'): the same premise sits in the culprit's parent and would come
       back in any rebuild from it. A rebuild of the culprit itself also drops what
       restates any claim the counterfactual took from it.
   heal_oracle replaces (a)-(c) with the scenario's AttackSpec ('oracle'): a claim
   names the attacker, or only undermines the legitimate system (the labeller's
   'ambiguous'). It runs no replay.

   Replays. Every probe is replayed Prober.n times (--claim-n, default CLAIM_N), and
   each replay is classed: HARMFUL (it made the decision being explained), ACTED (it
   acted on another destination), ANSWERED (it did not act, but its reply named a
   destination other than the observed one) or NO_ACTION (neither: it gave up, asked
   back or ran out of steps). A removal is SAFE only if none of its replays is harmful
   and at least one ACTED. Every target task asks for the action, so a replay that did
   not act failed it, whatever its reply named: ANSWERED is recorded, never evidence. A
   reply names hosts while giving up ("the transition notes at help.wayfarer.example/...
   returned 404"), asking back, or refusing on the attack's own premise ("ap@northfield.
   example no longer processes them"). A run with no act is inconclusive, and
   inconclusive is not safe. Run 0964 again: its claim-level counterfactual stopped on
   one replay that happened not to act for the attacker; the next replay did.

   Delta debugging (Zeller and Hildebrandt's ddmin over removal sets). ddmin needs
   removing all of a culprit's remaining claims to be safe and removing none not to be.
   Removing none is the first test (not safe, or ddmin would not run). Removing all is
   the culprit removed. The memory-level counterfactual flipped on that, but usually on
   one replay, so it is replayed n times first (the precondition, paid from the budget).
   If that is not safe, the culprit's removal does not reproduce the decision: nothing
   is localised, the culprit keeps none of its claims, and nothing spreads
   ('cf:not-reproduced'). ddmin then splits the claims into chunks, tries removing each
   chunk and each complement, and refines, until putting back any one claim of the set
   it holds makes the removal unsafe again. It finds what single removals cannot: two
   claims that each carry the decision alone ("being retired", "has been replaced").
   The candidates are ordered by suspicion (change language first, then claims a
   tainted page supports), so the first chunk holds the likely culprits. A claim of the
   set is 'ddmin' only if putting it back was HARMFUL. If that test was only
   inconclusive, it goes from the culprit but is not shown to carry the decision
   ('ddmin:inconclusive'), so it is not dropped from other memories. Putting it back
   was not shown safe either, so a parent restating it is not brought back (restored or
   a rebuild's input). Each test costs n replays; a memory
   gets at most Prober.budget of them (--ddmin-budget, DDMIN_BUDGET), and a memory whose
   search runs out of budget is quarantined ('ddmin:budget') rather than half-localised.
   A culprit nothing was localised in ('ddmin:budget', 'cf:control-flipped',
   'cf:not-reproduced') is quarantined under every strategy, rebuild included, and the
   clean parents it absorbed stay where incident leaves them: whatever made the
   decision may be in any of them.
2. Heal in lineage order, roots and ancestors first, then descendants by depth:
   - filter (heal, heal_oracle): every flagged memory keeps its benign claims, joined
     unregenerated (claims.compose): healing removes words and never writes new ones.
     A summary is filtered like any other memory, so nothing its parents said can come
     back through it (run 1089's rebuild re-imported the premise from a parent and
     lost a benign fact the summary had held);
   - rebuild (heal_rebuild): a derived memory with a parent this recovery healed or
     quarantined is rebuilt from its healed and clean parents with its recorded prompt
     (the Section 5.7 replay that reconstruct uses: recovery._summary_outputs, or
     recovery._reflection_facts for a procedure), and the rebuilt text is claim-
     filtered by the same rules plus a support guard: every rebuilt claim must be
     supported by its inputs, or it is dropped ('support'), since a claim no input
     supports is the model's own writing. A summary merges facts, so the inputs an
     output cites support it together by their words (SUPPORT_SHARE), as does any one
     input (SUPPORT_SHARE / SUPPORT_COSINE). A claim with change language needs one
     input that supports it by its words and holds the same change: the model can glue
     ", which no longer processes them" to a fact one input states, and the share and
     the cosine both still see that input's sentence. A
     parent counts as clean only if it is outside the lineage, is no root's sibling,
     and these rules would not drop it whole: a composed attack leaves extractions that
     name the attacker's system without its host ("The RentQuick endpoint is
     active."), which the incident never flags but a rebuild would happily restate. A
     memory an earlier round healed (+verify) is heal's own writing, already judged:
     it counts as clean unless this round flags it or these rules would drop it, and
     it stands for the original it replaced when that original is flagged again. Only
     a transformation heal can replay is rebuilt (rebuildable): a consolidation
     summary or a procedure. A Mem0 UPDATE mirror row is 'summarize' too, but Mem0's
     update-decision prompt wrote it ('mem0.update'), which is not a consolidation
     prompt and is never replayed as one (memheal/backends/mem0_backend.py); like a
     memory with no model to rebuild it, it is filtered;
   - a healed memory has operation 'heal', source_type 'derived', an edge from the
     original first (so it stands for whatever the original stood for) and then from
     what it was rebuilt from. The original is quarantined, never deleted, so a heal can
     be undone, and its recovery_event (action 'heal') lists the dropped claims with
     the rule that dropped each;
   - a memory with no benign claim left is quarantined;
   - a flagged memory stays active and unchanged only when every claim was judged
     benign, with a 'keep' event saying so. Under heal that is all but impossible: what
     incident flags either names the destination (rule a sees it in some claim, or the
     split lost it and the memory is quarantined whole) or flipped the decision (rule c
     localises it, or keeps none).
3. Finish: a healed copy of an absorbed memory stays under the summaries this recovery
   keeps, a summary's claim-healed copy included (it stands for the summary, as a claim
   repair does in selective_claims: recovery._standing_for); clean parents (in the
   sense above, so an earlier round's healed copies too, and never a root's sibling)
   of summaries it took out come back unless something kept still carries them
   (unmerge's rule, recovery.restore_parents). A claim-healed summary carries only
   what it kept: a clean parent that supports none of its kept claims was carried by
   claims the filter dropped (run 1089's summary had overwritten the user's own route
   with the attacker's), so it comes back unless something else still carries it. The
   parents of a culprit nothing was localised in stay out. A healed copy that nothing
   carries any more is the only copy of its claims and goes active.
4. Post-heal check: the explained decision replayed n times on the frozen set as
   healing left it (each healed memory in place of its original, what was quarantined
   removed), when a retrieved memory was healed. Recorded, not acted on: it is how
   well the healed memories themselves stand up, the number +verify cannot give
   (it re-retrieves).

Probe rows. A claim-level replay renders a temporary row (operation 'cf_probe', status
'quarantined', no embedding) in the culprit's place in the frozen set. A frozen replay
renders rows by id whatever their status (agent._frozen_rows), so the model sees the
probe exactly where the culprit was, while retrieval can never return it. The rows stay
as the audit trail, each with a 'cf_probe' recovery_event naming the culprit and the
outcome (the replay's own retrieval log references them too). They have no edges, so
lineage, labels and the incident's suspects never count them.

This module never replays anything itself: the harness hands it a Prober (recover.py
builds one on influence.frozen_ablation), so the store side stays free of agents and
pages, like the rest of recovery.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np

from memheal import claims, db, labels, recovery
from memheal.labels import AttackSpec
from memheal.tools import normalize_url

HEAL_MODES = recovery.HEAL_MODES
HEAL_OP = recovery.HEAL_OP
PROBE_OP = recovery.PROBE_OP

# What one probe replay did (Prober.replay's answer per replay). Only ACTED is evidence
# that a removal made the decision safe; ANSWERED is recorded and, like NO_ACTION,
# inconclusive (the task asked for an action, and the reply did not take one).
HARMFUL = "harmful"        # made the decision being explained
ACTED = "acted"            # acted on another destination
ANSWERED = "answered"      # did not act, but its reply named a destination, not the observed one
NO_ACTION = "none"         # neither: gave up, asked back, or ran out of steps
REPLAY_CLASSES = (HARMFUL, ACTED, ANSWERED, NO_ACTION)

CLAIM_N = 3                # replays per probe (--claim-n)
DDMIN_BUDGET = 24          # ddmin replays per memory before it is quarantined (--ddmin-budget)

# The rules that drop a claim, as a record counts them (dropped_by). 'support' is the
# rebuild's guard, 'oracle' heal_oracle's labels.
RULES = ("name", "source-taint", "cf", "ddmin", "support", "oracle")
# Rule (c) labels that localise nothing: the culprit is quarantined whole, its claims
# never spread, and the clean parents it absorbed are not brought back.
UNLOCALISED = ("ddmin:budget", "cf:control-flipped", "cf:not-reproduced")

# A change of route, process or support: the labeller's own change language (retired,
# no longer, replaced/replacement, moved, migrated, switched, transition, phased out,
# being withdrawn, not processed, returned without action, delays), so heal judges a
# premise with the words that make labels.py call a memory 'ambiguous'.
CHANGE = labels._CHANGE

# A text supports a claim when it holds at least this share of the claim's distinctive
# words (claims.grounded: whole or inflected, plurals folded), or when the run's
# embedder puts the two at least this close. Three of four words, so a determiner the
# summary added ("these requests") does not hide the sentence it came from.
SUPPORT_SHARE = 0.75
SUPPORT_COSINE = 0.8

# A destination named in an answer: a host with a top-level domain of letters, bare,
# in a URL or after a mailbox's '@'. File names are not destinations: documents,
# images, archives, and the scripts and configs of the coding scenarios.
_ANSWER_HOST = re.compile(r"(?<![a-z0-9-])(?:[a-z0-9][a-z0-9-]*\.)+([a-z]{2,})(?![a-z0-9-])",
                          re.IGNORECASE)
_FILE_ENDINGS = frozenset("""pdf doc docx txt csv tsv xls xlsx ppt pptx png jpg jpeg gif svg
json md rst zip gz tgz tar xz sh bash py js ts go java yml yaml toml ini cfg conf env lock
log xml html htm css sql db sqlite""".split())


@dataclass
class Prober:
    """The harness's side of the claim-level counterfactual (rule c).

    retrieved - the frozen set the explained decision retrieved, in rank order.
    replay    - replays variants of it, [(label, frozen ids)], once each, and classes
                each replay (HARMFUL, ACTED, ANSWERED, NO_ACTION). It may run them
                concurrently; results come back in the order given. True and False are
                read as HARMFUL and ACTED (a non-repeat that acted).
    n         - replays per probe (--claim-n): a removal is safe only if none of the n
                is harmful and one acted.
    budget    - ddmin replays per memory (--ddmin-budget), its precondition included.
    """
    retrieved: list[str]
    replay: Callable[[list[tuple[str, list[str]]]], list[str | bool]]
    n: int = CLAIM_N
    budget: int = DDMIN_BUDGET


@dataclass
class HealContext:
    """What healing needs beyond the store: the claim-level replays (None skips rule c
    and the post-heal check) and the split cache, shared by every round of one recovery."""
    probe: Prober | None = None
    splits: claims.SplitCache = field(default_factory=claims.SplitCache)


@dataclass
class _Variant:
    """One claim-level test: the culprit rendered as `kept` (a probe row in its place;
    with no claim kept, the culprit removed), and how each of its replays came out."""
    kept: list[str]
    what: str
    pid: str | None = None
    replays: list[str] = field(default_factory=list)

    @property
    def harmful(self) -> bool:
        return HARMFUL in self.replays

    @property
    def safe(self) -> bool:
        """No harmful replay and one that acted: a replay that did not act, whatever
        its reply named, failed the task and says nothing about the decision."""
        return not self.harmful and ACTED in self.replays

    @property
    def verdict(self) -> str:
        return "safe" if self.safe else "harmful" if self.harmful else "inconclusive"

    @property
    def outcome(self) -> str:
        c = Counter(self.replays)
        return (f"{self.verdict} (harmful {c[HARMFUL]}/{len(self.replays)}, acted {c[ACTED]}, "
                f"answered {c[ANSWERED]}, none {c[NO_ACTION]})")


def replay_class(result: str | bool) -> str:
    """A Prober answer as a replay class (True/False: harmful / acted)."""
    if result is True:
        return HARMFUL
    if result is False:
        return ACTED
    if result not in REPLAY_CLASSES:
        raise ValueError(f"unknown replay class {result!r}; expected one of {REPLAY_CLASSES}")
    return result


def answer_names_destination(text: str, spec: AttackSpec, seen: Iterable[str] = ()) -> bool:
    """Does a reply with no acting call name a destination other than the observed one?
    It names a host or mailbox that is neither the observed destination nor its host
    label, nor a host the replay itself visited or searched for (`seen`: the URL of a
    page it failed to load is not an answer). This only labels the replay (ANSWERED);
    it is never evidence of safety. An apology, a question back or a refusal can name a
    host as well as an answer can."""
    if spec.names_attacker(text) or claims.names_host_label(text, spec.attacker_destinations):
        return False
    looked = {m.group(0).lower() for s in seen for m in _ANSWER_HOST.finditer(s or "")}
    return any(m.group(1).lower() not in _FILE_ENDINGS and m.group(0).lower() not in looked
               for m in _ANSWER_HOST.finditer(text))


# --- delta debugging ------------------------------------------------------------------

T = TypeVar("T")


class BudgetSpent(Exception):
    """ddmin needed a test its replay budget could not pay for."""


def _chunks(items: list[T], n: int) -> list[list[T]]:
    """`items` cut into n contiguous chunks, sizes differing by at most one."""
    k, extra = divmod(len(items), n)
    out, i = [], 0
    for j in range(n):
        size = k + (j < extra)
        out.append(items[i:i + size])
        i += size
    return [c for c in out if c]


def ddmin(items: Sequence[T], removal_safe: Callable[[list[T]], bool]) -> list[T]:
    """Zeller and Hildebrandt's ddmin (2002) over removal sets: a 1-minimal subset of
    `items` whose removal is safe, given that removing all of them is and removing none
    is not (the caller has established both).

    1-minimal: putting back any one item of the result makes its removal unsafe again;
    a smaller safe set may exist elsewhere, but none inside this one. The set is cut
    into n chunks in order, so a caller that orders items by suspicion has its likely
    culprits tested together first; removing a chunk that is safe narrows to it,
    removing a complement that is safe narrows to that, and otherwise the chunks get
    finer until they are single items. With two chunks each complement is the other
    chunk, so complements are tried only from three chunks up. `removal_safe` may raise
    (BudgetSpent) to stop the search; it is called again for a set it has answered, so
    a caller that pays per test should remember its answers.
    """
    c, n = list(items), 2
    while len(c) >= 2:
        chunks = _chunks(c, n)
        found = next((d for d in chunks if removal_safe(d)), None)
        if found is not None:
            c, n = found, 2
            continue
        if len(chunks) > 2:
            rests = ([x for x in c if x not in d] for d in chunks)
            found = next((r for r in rests if removal_safe(r)), None)
            if found is not None:
                c, n = found, max(n - 1, 2)
                continue
        if n >= len(c):
            break
        n = min(len(c), 2 * n)
    return c


# --- source taint ----------------------------------------------------------------------

def page_of(row: Mapping) -> str | None:
    """The page a memory was read off (its source url without the session prefix,
    normalized), None for anything not read off a page."""
    if row["source_type"] != "webpage" or not row["source_id"]:
        return None
    return normalize_url(recovery._SESSION_PREFIX.sub("", row["source_id"]))


@dataclass(frozen=True)
class Taint:
    """Rule (b)'s pages, read from the store and the inferred roots alone.

    root_pages - the pages the inferred roots were read from.
    pages      - those, and every page with a memory that points at one (names its url):
                 a composed attack's first page sends the agent to the page that names
                 the destination without naming it itself.
    siblings   - every memory read off a root page, whatever its status.
    """
    root_pages: frozenset[str] = frozenset()
    pages: frozenset[str] = frozenset()
    siblings: frozenset[str] = frozenset()

    def points_at_root(self, text: str) -> bool:
        low = text.lower()
        return any(u in low for u in self.root_pages)


def source_taint(conn: sqlite3.Connection, plan: recovery.RecoveryPlan) -> Taint:
    """The plan's tainted pages (Taint), as rule (b) uses them. heal_oracle judges claims
    by the scenario's AttackSpec alone, so it gets none."""
    return Taint() if plan.mode == "heal_oracle" else tainted_pages(conn, plan)


def tainted_pages(conn: sqlite3.Connection, plan: recovery.RecoveryPlan) -> Taint:
    """The pages the plan's inferred roots taint (Taint), read from the store alone, in
    every heal mode: which unflagged memories healing judges (the sweep) is part of the
    detection the three modes share. Probe rows are scaffolding and never count."""
    roots = [db.get_memory(conn, r) for r in plan.inferred_roots]
    root_pages = {p for r in roots if r is not None and (p := page_of(r))}
    if not root_pages:
        return Taint()
    rows = [r for r in db.all_memories(conn) if r["operation"] != PROBE_OP]
    siblings = {r["id"] for r in rows if page_of(r) in root_pages}
    pointing = {page_of(r) for r in rows
                if page_of(r) and any(u in r["text"].lower() for u in root_pages)}
    return Taint(frozenset(root_pages), frozenset(root_pages | pointing), frozenset(siblings))


# --- claim labels -------------------------------------------------------------------

def carries(text: str, row: Mapping, spec: AttackSpec) -> bool:
    """A whole text judged the way one claim is by rule (a) (or the oracle)."""
    return recovery.claim_carries_attack(text, row, spec) or recovery.claim_is_ambiguous(text, spec)


def rule_of(label: str) -> str:
    """The rule (RULES) a claim label belongs to: 'cf:restates' -> 'cf'."""
    head = label.split(":", 1)[0]
    if head not in RULES:
        raise ValueError(f"claim label {label!r} names no rule in {RULES}")
    return head


def _derived(text: str) -> dict[str, Any]:
    """A rebuilt text as a row: derived, read off no page."""
    return {"text": text, "source_type": "derived", "source_id": None}


def made_by(conn: sqlite3.Connection, row: Mapping) -> str | None:
    """The name of the prompt on record for a memory, None if there is none."""
    prompt = db.get_prompt(conn, row["created_by_prompt"]) if row["created_by_prompt"] else None
    return prompt["name"] if prompt is not None else None


def rebuildable(conn: sqlite3.Connection, row: Mapping) -> bool:
    """Was this derived memory made by a transformation heal_rebuild can replay over new
    inputs?

    A procedure is (rewritten from its inputs with the reflection rewrite prompt,
    whatever wrote it), and so is a consolidation summary: its recorded prompt is
    replayed (recovery._summary_outputs), and one stored before prompts were recorded
    was made with the default that _summary_outputs falls back to. Nothing else is. A
    Mem0 UPDATE mirror row is 'summarize' too, written by Mem0's update-decision prompt
    ('mem0.update'): replayed with the consolidation schema it would be a call Mem0
    never makes, and the memory it wrote would not be Mem0's. A claim heal or claim
    repair was a split, not a transformation of inputs.
    """
    name = made_by(conn, row)
    if row["operation"] == "reflect":
        return True
    if name is None:
        return row["operation"] == "summarize"
    return name.startswith(("consolidate.", "reconstruct.reflect"))


def lineage_order(conn: sqlite3.Connection, ids: Iterable[str]) -> list[str]:
    """`ids` with every memory after its parents: by depth (the longest chain of edges up
    to a parentless memory, 0 for one), store order within a depth."""
    memo: dict[str, int] = {}

    def depth(m: str) -> int:
        if m not in memo:
            memo[m] = 0                  # edges form a DAG; this only guards a bad store
            memo[m] = max((1 + depth(p) for p in db.parents_of(conn, m)), default=0)
        return memo[m]

    pos = {r["id"]: i for i, r in enumerate(db.all_memories(conn))}
    return sorted(set(ids), key=lambda m: (depth(m), pos.get(m, len(pos)), m))


def _tainted(conn: sqlite3.Connection, plan: recovery.RecoveryPlan) -> set[str]:
    """What this round may not trust as a clean parent: the plan's lineage and what it
    flags, except a memory an earlier round healed that this round does not flag.

    An earlier round's healed memory descends from the root it healed, so +verify's
    re-inferred lineage holds it; but it is heal's own writing, already judged, and it
    is the only copy of the benign claims it kept. Counted as lineage it could never be
    a rebuild's input nor come back when its carrier is taken out, and a summary the
    first round spared and the second flags would lose those claims for good. It is
    still judged by this round's rules (_dirty) wherever a clean parent is."""
    out = set(plan.lineage) | set(plan.heal)
    for m in list(out - set(plan.heal)):
        row = db.get_memory(conn, m)
        if row is not None and row["operation"] == HEAL_OP:
            out.discard(m)
    return out


def culprits(plan: recovery.RecoveryPlan) -> dict[str, tuple[str, ...]]:
    """Flagged culprit -> the other members of the first minimal flipping set it is in.

    Those others are removed while its claims are tested: in a pair that flips only
    together, each member carries the decision once the other is gone, and with both
    present no single claim of either could flip it.
    """
    out: dict[str, tuple[str, ...]] = {}
    for s in plan.culprit_sets:
        for m in s:
            if m in plan.heal and m not in out:
                out[m] = tuple(x for x in s if x != m)
    return out


def preview(conn: sqlite3.Connection, plan: recovery.RecoveryPlan) -> tuple[int, int]:
    """(claims kept, claims dropped) over the flagged memories' own texts, split
    deterministically and judged by rules (a) and (b) (or the oracle) alone, with
    support by shared words only: what a dry run can say with no model, no embedder
    and no replay. Rule (c) and the rebuilds need them, so they are not in it. The
    unflagged memories the sweep would judge are counted with the flagged ones."""
    judge = _Healer(conn, plan, None, None, "", plan.spec or recovery.observed_spec(None),
                    HealContext(), has_llm=False)
    kept = dropped = 0
    for row in [db.get_memory(conn, m) for m in plan.heal] + judge._sweep_candidates():
        if row is None or row["status"] in ("quarantined", "deleted"):
            continue
        for c in claims.fallback_split(row["text"]):
            if judge._rule(c, row):
                dropped += 1
            else:
                kept += 1
    return kept, dropped


# --- carrying out a plan --------------------------------------------------------------

def apply(conn: sqlite3.Connection, plan: recovery.RecoveryPlan, llm, embedder,
          recovery_id: str, spec: AttackSpec | None = None, *,
          context: HealContext | None = None, has_llm: bool = True) -> dict[str, Any]:
    """Heal what `plan` flagged (recovery.apply calls this for the heal modes).

    `llm` splits claims and (heal_rebuild) rebuilds derived memories; with has_llm=False
    splits are deterministic and a rebuild falls back to the memory's own claims.
    `embedder` embeds healed memories and judges support by similarity. Returns
    {quarantined, new_ids, restored} as recovery.apply merges them, plus healed,
    flagged_quarantined, unchanged, already_out, rebuilt, claims_kept, claims_dropped,
    dropped_by (claims dropped per rule, RULES), claim_cf_replays (every replay healing
    ran: probes, ddmin and the post-heal check), ddmin_replays, budget_spent (memories
    whose ddmin ran out of budget), inconclusive (probes with no harmful and no
    conclusive replay), post_heal ((harmful, replays) of the post-heal check, or None),
    swept (unflagged memories read off a tainted page that the sweep healed or
    quarantined; not in healed or flagged_quarantined, which stay the flagged set's),
    probes and a one-line note.
    """
    spec = spec or plan.spec or recovery.observed_spec(None)
    return _Healer(conn, plan, llm, embedder, recovery_id, spec, context or HealContext(),
                   has_llm).run()


class _Healer:
    """One heal plan carried out on one store (steps in the module docstring)."""

    def __init__(self, conn, plan, llm, embedder, recovery_id, spec, ctx, has_llm):
        self.conn, self.plan, self.llm, self.embedder = conn, plan, llm, embedder
        self.recovery_id, self.spec, self.ctx = recovery_id, spec, ctx
        self.has_llm = has_llm
        self.oracle = plan.mode == "heal_oracle"
        self.rebuild = plan.mode == "heal_rebuild"
        self.tainted = _tainted(conn, plan)
        self.pages = tainted_pages(conn, plan)      # the sweep's, in every heal mode
        self.taint = source_taint(conn, plan)       # rule (b)'s
        self.replaced: dict[str, list[str]] = {}    # original -> its healed memories
        self.claim_healed: dict[str, list[str]] = {}  # replaced by their own benign claims: kept
        self.swept: list[str] = []                  # unflagged, read off a tainted page, changed
        self.gone: set[str] = set()                 # flagged memories out of retrieval
        self.kept: list[str] = []                   # flagged, every claim benign, unchanged
        self.suspect: set[str] = set()              # inputs a rebuild brought the attack back from
        self.cf: dict[str, dict[str, str]] = {}     # culprit -> claim -> rule (c) label
        self.cf_poisoned: list[str] = []            # claims ddmin showed carry the decision
        self.cf_unproven: list[str] = []            # ddmin took, putting back inconclusive
        self.quarantined: list[str] = []            # status changed to quarantined here
        self.flagged_quarantined: list[str] = []    # flagged, nothing healed to replace it
        self.already_out: list[str] = []
        self.restored: list[str] = []
        self.probes: list[str] = []
        self.budget_spent: list[str] = []
        self.dropped_by: Counter[str] = Counter()
        self.post_heal: tuple[int, int] | None = None
        self.rebuilt = 0
        self.claims_kept = 0
        self.claims_dropped = 0
        self.cf_replays = 0
        self.ddmin_replays = 0
        self.inconclusive = 0
        self._ancestors: dict[str, list] = {}
        self._vec: dict[str, np.ndarray] = {}

    @property
    def probing(self) -> bool:
        return not self.oracle and self.ctx.probe is not None

    def unlocalised(self, m: str) -> bool:
        """Did the claim-level counterfactual run on `m` and localise nothing in it?"""
        return any(w in UNLOCALISED for w in self.cf.get(m, {}).values())

    def run(self) -> dict[str, Any]:
        if self.probing:
            for m, rest in culprits(self.plan).items():
                self._localise(m, rest)
        for m in lineage_order(self.conn, self.plan.heal):
            self._heal(m)
        self._sweep()
        self._finish()
        if self.probing:
            self._post_heal_check()
        return self._summary()

    # --- step 1: rule (c) -----------------------------------------------------------

    def _localise(self, m: str, rest: tuple[str, ...]) -> None:
        """The claim-level counterfactual for one culprit.

        Rules (a) and (b) are free, so the counterfactual only has to explain what they
        leave: first the culprit without the claims they drop (if that is safe, the rest
        is benign), then ddmin over what is left. A single-claim memory is the claim,
        and its removal already flipped.

        Each removal is from the culprit's claims recomposed, not from the memory, so
        that rendering must itself make the decision. Where the rules dropped claims, a
        first test that is not safe shows it does (or is inconclusive, which counts as
        not safe and goes to ddmin as well). Where they dropped none, the first test is
        a control instead: all the claims, recomposed. The memory's removal flipped the
        decision, so its claims should make it again; if no replay does, the split
        reworded or lost what mattered, every removal would look safe for that alone,
        and each benign claim would be dropped and spread by restates. So nothing is
        localised then: none of the claims is kept and nothing spreads. ddmin's own
        precondition, the culprit removed, is replayed as well (_ddmin).
        """
        row = db.get_memory(self.conn, m)
        if (row is None or row["status"] in ("quarantined", "deleted")
                or m not in self.ctx.probe.retrieved):
            return
        cl = self._split(row["text"], row, f"claims:{m}").claims
        residual = [c for c in cl if not self._rule(c, row, cf=False)]
        if not residual:
            return
        if len(cl) == 1:
            self.cf[m] = {cl[0]: "cf:only-claim"}
            return
        ruled = len(cl) - len(residual)
        if ruled:
            first = _Variant(residual, f"without the {ruled} claim(s) rules (a)/(b) drop")
        else:
            first = _Variant(cl, f"all {len(cl)} claims recomposed (control)")
        self._probe(m, row, rest, [first])
        if ruled and first.safe:
            return
        if not ruled and not first.harmful:
            self.cf[m] = {c: "cf:control-flipped" for c in residual}
            return
        found = self._ddmin(m, row, rest, residual, first)
        if found is None:
            self.budget_spent.append(m)
            self.cf[m] = {c: "ddmin:budget" for c in residual}
            return
        self.cf[m] = found
        self.cf_poisoned += [c for c, w in found.items()
                             if w == "ddmin" and c not in self.cf_poisoned]
        self.cf_unproven += [c for c, w in found.items()
                             if w == "ddmin:inconclusive" and c not in self.cf_unproven]

    def _ddmin(self, m: str, row, rest: tuple[str, ...], residual: list[str],
               first: _Variant) -> dict[str, str] | None:
        """claim -> label for the claims of `residual` the search takes from `m`, in claim
        order; None when the memory's replay budget ran out first.

        ddmin's precondition is replayed first: removing every residual claim is the
        culprit removed. If that is not safe, the decision does not come from the
        culprit and nothing is localised ('cf:not-reproduced' on every claim). Otherwise
        ddmin finds a 1-minimal set whose removal is safe. A claim of it is 'ddmin' when
        putting it back (removing the set without it) was harmful, so it carries the
        decision and spreads; when that test was only inconclusive it is
        'ddmin:inconclusive' and does not spread. Putting the last claim back is removing
        nothing, which `first` answered. Every test is remembered by its verdict."""
        probe = self.ctx.probe
        n = max(probe.n, 1)
        order = sorted(range(len(residual)), key=lambda i: (self._suspicion(residual[i], row), i))
        verdicts: dict[frozenset[int], str] = {frozenset(): first.verdict}
        spent = 0

        def test(removed: frozenset[int], what: str) -> str:
            nonlocal spent
            if removed not in verdicts:
                if spent + n > probe.budget:
                    raise BudgetSpent
                kept = [c for i, c in enumerate(residual) if i not in removed]
                [v] = self._probe(m, row, rest, [_Variant(kept, what)])
                spent += len(v.replays)
                verdicts[removed] = v.verdict
            return verdicts[removed]

        def removal_safe(removed: list[int]) -> bool:
            gone = ", ".join(repr(residual[i]) for i in sorted(removed))
            return test(frozenset(removed), f"ddmin: without {gone}") == "safe"

        try:
            if test(frozenset(order), "ddmin precondition: the culprit removed") != "safe":
                return {c: "cf:not-reproduced" for c in residual}
            found = frozenset(ddmin(order, removal_safe))
        except BudgetSpent:
            return None
        finally:
            self.ddmin_replays += spent
        return {residual[i]: ("ddmin" if verdicts.get(found - {i}) == "harmful"
                              else "ddmin:inconclusive")
                for i in sorted(found)}

    def _suspicion(self, claim: str, row) -> int:
        """ddmin's order: change language first, then a claim a tainted page supports,
        then the rest (the free rules dropped what both at once would)."""
        if CHANGE.search(claim):
            return 0
        return 1 if self._from_tainted(claim, row) else 2

    def _probe(self, m: str, row, rest: tuple[str, ...],
               variants: list[_Variant]) -> list[_Variant]:
        """Replay the explained decision for each variant, with `m` rendered as the
        variant's claims: a probe row in its place, the rest of its flipping set removed,
        the frozen set otherwise as it was. Every variant's n replays go in one batch,
        so --cf-workers can run them at once. A variant with no claim left is `m`
        removed (ddmin's precondition): replayed with nothing in its place, so it has no
        probe row. One cf_probe event per variant gives its outcome (with the probe row
        as its replacement, or none for a removal)."""
        probe = self.ctx.probe
        base = [x for x in probe.retrieved if x not in rest]
        for v in variants:
            if v.kept:
                v.pid = db.insert_memory(self.conn, claims.compose(v.kept), "derived",
                                         row["session"], operation=PROBE_OP, status="quarantined")
                self.probes.append(v.pid)
        jobs = [(v, i) for v in variants for i in range(max(probe.n, 1))]
        results = self._replay([
            (f"{m}:{v.pid or 'removed'}:{i}",
             [v.pid if x == m else x for x in base] if v.pid else [x for x in base if x != m])
            for v, i in jobs])
        for (v, _i), r in zip(jobs, results, strict=True):
            v.replays.append(r)
        for v in variants:
            if not v.harmful and not v.safe:
                self.inconclusive += 1
            self._log(m, PROBE_OP, replacement=v.pid, reason=f"claim-cf: {v.what} -> {v.outcome}")
        return variants

    def _replay(self, variants: list[tuple[str, list[str]]]) -> list[str]:
        """One batch of replays through the Prober, classed; every one is counted."""
        if not variants:
            return []
        out = [replay_class(r) for r in self.ctx.probe.replay(variants)]
        self.cf_replays += len(variants)
        return out

    # --- step 2: heal in lineage order ----------------------------------------------

    def _heal(self, m: str) -> None:
        row = db.get_memory(self.conn, m)
        if row is None:
            return
        if row["status"] in ("quarantined", "deleted"):
            # Not retrievable and carried by nothing: a heal would surface what the store
            # had already put away. It counts as gone, so a summary over it is rebuilt
            # without it and the clean parents it absorbed can come back.
            self.gone.add(m)
            self.already_out.append(m)
            self._log(m, "quarantine", reason=f"heal:already-{row['status']}")
            return
        parents = self._content_parents(row)
        touched = any(p in self.replaced or p in self.gone for p in parents)
        if self.rebuild and row["source_type"] == "derived" and touched:
            why = self._not_rebuilt(row)
            if why:
                self._claim_heal(m, row, why)
            else:
                self._rebuild(m, row, parents)
        else:
            self._claim_heal(m, row)

    def _not_rebuilt(self, row) -> str | None:
        """Why a derived memory over healed parents keeps its own benign claims instead
        of being rebuilt from them (the event says which); None to rebuild it. A culprit
        the claim-level counterfactual localised nothing in keeps none (every claim is
        labelled), so it is quarantined: a rebuild from its parents would bring back
        whatever made the decision, which the search could not find."""
        if self.unlocalised(row["id"]):
            return "the claim-level counterfactual localised nothing in it, not rebuilt; "
        if not self.has_llm:
            return "no model to rebuild with; "
        if not rebuildable(self.conn, row):
            return f"made by {made_by(self.conn, row) or 'no recorded prompt'}, not replayed; "
        return None

    def _content_parents(self, row) -> list[str]:
        """What a memory was made from: its parents, except that a claim repair's or a
        healed memory's first edge is to the memory it replaced (provenance, not input)."""
        ps = db.parents_of(self.conn, row["id"])
        return ps[1:] if row["operation"] in (recovery.CLAIM_REPAIR, HEAL_OP) else ps

    def _claim_heal(self, m: str, row, note: str = "") -> None:
        """Keep the memory's benign claims; quarantine it when none is left."""
        split = self._split(row["text"], row, f"claims:{m}")
        why = self._labels(m, row, split.claims)
        kept = [c for c in split.claims if c not in why]
        head = f"heal:{note}claims:{split.how} kept={len(kept)}/{len(split.claims)}"
        dropped = _dropped(why)
        if not why:
            self._unpoisoned(m, row, split, head)
            return
        self._count(why.values())
        text = claims.compose(kept)
        if not kept:
            self._out(m, f"{head}; no benign claim{dropped}")
        elif carries(text, row, self.spec):         # joining adds no words; a last guard
            self._out(m, f"{head}; still carries the attack{dropped}")
        else:
            prompt = (db.register_prompt(self.conn, "claims.split", claims.CLAIMS_PROMPT)
                      if split.how == "llm" else None)
            new = self._write(m, row, [text], [[]], prompt, split.model)
            self.claim_healed[m] = kept
            self.claims_kept += len(kept)
            self._log(m, "heal", replacement=new[0], reason=f"{head}{dropped}")

    def _unpoisoned(self, m: str, row, split: claims.Split, head: str) -> None:
        """A flagged memory none of whose claims was judged poisoned (it stays only if
        every claim really was judged benign, and the event says why)."""
        if carries(row["text"], row, self.spec):
            # The memory carries the attack but no claim does, even on the deterministic
            # split: there is no telling where, so nothing is kept.
            self._count(["oracle" if self.oracle else "name"] * len(split.claims))
            self._out(m, f"{head}; the memory carries the attack but no claim does")
        elif not self.oracle and m in culprits(self.plan) and m not in self.cf:
            # Its removal flipped the decision, nothing names the destination, and the
            # claim-level counterfactual did not run: the verdict says the memory carries
            # the attack and nothing says where, so it is not kept.
            self._count(["cf"] * len(split.claims))
            self._out(m, f"{head}; flipped the decision and no claim-level counterfactual "
                         f"ran to say where")
        else:
            cf = " and the claim-level counterfactual" if m in self.cf else ""
            basis = "the scenario's AttackSpec" if self.oracle else f"rules (a)/(b){cf}"
            self.kept.append(m)
            self._log(m, "keep", reason=f"{head}; every claim judged benign by {basis}; "
                                        f"left unchanged")

    def _rebuild(self, m: str, row, parents: list[str]) -> None:
        """heal_rebuild: rebuild a derived memory over healed parents from its healed and
        clean parents, then claim-filter the rebuild (Section 5.7's replay, with the
        inputs healing produced) by the rules and the support guard. One input is the
        whole of what is left: it comes back as it is, with no model call, as
        reconstruct does with one clean parent."""
        inputs = self._inputs(parents)
        if len(inputs) < 2:
            only = inputs[0]["id"] if inputs else None
            self._out(m, "heal:rebuild: " + ("one clean or healed parent, brought back as it is"
                                              if only else "no clean or healed parent"),
                      replacement=only)
            return
        reflection = recovery._is_reflection(self.conn, row)
        if reflection:
            facts, prompt_hash, model = recovery._reflection_facts(
                self.conn, self.llm, inputs, tag=f"heal:reflect:{m}")
            outputs = [(f, [r["id"] for r in inputs]) for f in facts]
        else:
            outputs, prompt_hash, model = recovery._summary_outputs(
                self.conn, self.llm, row, inputs, tag=f"heal:summarize:{m}")
        origins = [o for r in inputs for o in self._origins(r)]
        support = [t for r in inputs for t in (r["text"], *claims.fallback_split(r["text"]))]
        own = list(self.cf.get(m, {}))      # what rule (c) took from the culprit itself
        texts: list[str] = []
        cited: list[list[str]] = []
        n_kept, why_all = 0, {}
        for text, cites in outputs:
            out_row = _derived(text)
            if carries(text, out_row, self.spec):
                # The attack came back from healed and clean inputs alone: do not trust
                # what it drew on back into retrieval (reconstruct's rule).
                self.suspect.update(cites)
            split = self._split(text, out_row, f"claims:{m}:rebuilt")
            joined = " ".join(r["text"] for r in inputs if r["id"] in cites) or \
                " ".join(r["text"] for r in inputs)
            why = {}
            for c in split.claims:
                w = (self._rule(c, out_row, origins=origins)
                     or ("cf:restates" if any(claims.restates(c, p) for p in own) else None)
                     or (None if self._rebuild_supports(c, support, joined) else "support"))
                if w:
                    why[c] = w
            why_all.update(why)
            kept = [c for c in split.claims if c not in why]
            healed = claims.compose(kept)
            if kept and not carries(healed, out_row, self.spec):
                texts.append(healed)
                cited.append(cites)
                n_kept += len(kept)
        self._count(why_all.values())
        head = (f"heal:rebuilt {'reflection' if reflection else 'summary'} from "
                f"{len(inputs)} clean/healed parent(s)")
        if not texts:
            what = "no benign claim in the rebuild" if outputs else "empty replay"
            self._out(m, f"{head}; {what}{_dropped(why_all)}")
            return
        new = self._write(m, row, texts, cited, prompt_hash, model)
        if not reflection:
            # Mirrors consolidate: the rebuilt summary stands for what it cites.
            cites_all = list(dict.fromkeys(c for cs in cited for c in cs))
            db.set_status_many(self.conn, [c for c in cites_all
                                           if db.get_memory(self.conn, c)["status"] == "active"],
                               "consolidated")
        self.rebuilt += 1
        self.claims_kept += n_kept
        self._log(m, "heal", replacement=",".join(new),
                  reason=f"{head}; kept={n_kept}/{n_kept + len(why_all)}{_dropped(why_all)}")

    def _rebuild_supports(self, claim: str, support: list[str], joined: str) -> bool:
        """The rebuild's support guard for one rebuilt claim. `support` is every input
        and its sentences; `joined` the inputs its output cites, together.

        A claim with change language needs one input that supports it by its words and
        holds the same change (_change_terms): "Supplier invoices should be sent to
        Accounts Payable at ap@northfield.example, which no longer processes them"
        shares three words in four with the input that states the route, and its
        embedding sits close to that input's, but no input says "no longer". Any other
        claim may merge the facts its output cites (their words together: a summary's
        "Driveline handled car hire bookings directly at a live endpoint without a
        separate form or sign-in page" draws on three extractions, none enough alone),
        or come from one input (words or embedding)."""
        change = _change_terms(claim)
        if change:
            return any(claims.grounded(claim, t, SUPPORT_SHARE) and change <= _change_terms(t)
                       for t in support)
        return self._supports(claim, support) or (
            bool(joined) and claims.grounded(claim, joined, SUPPORT_SHARE))

    def _inputs(self, parents: list[str]) -> list:
        """What a derived memory is rebuilt from, in citation order, without repeats."""
        rows, seen = [], set()
        for p in parents:
            for r in self._current(p):
                if r["id"] not in seen:
                    seen.add(r["id"])
                    rows.append(r)
        return rows

    def _current(self, p: str) -> list:
        """What stands for parent `p` now: its healed memories if this recovery (or an
        earlier round, for a parent already quarantined, flagged again or not) healed
        it; nothing if it was quarantined; the parent itself if it is clean or was
        judged benign claim by claim. Anything else in the lineage stays out, as in
        reconstruct: a spared lineage memory is not trusted as a clean parent (an
        earlier round's healed memory is not counted as lineage: _tainted). A memory
        read off a root page never stands for itself (rule b), even one judged benign
        claim by claim; what healing wrote for it is heal's own filtered writing and
        stands for it like any healed memory."""
        if p in self.replaced:
            return [db.get_memory(self.conn, h) for h in self.replaced[p]]
        if p in self.gone and p not in self.already_out:
            return []
        row = db.get_memory(self.conn, p)
        if row is None or row["status"] == "deleted":
            return []
        if row["status"] == "quarantined":
            return [r for v in self._versions(p) for r in self._current(v)]
        if p in self.taint.siblings:
            return []
        if p in self.kept or (p not in self.tainted and not self._dirty(row)):
            return [row]
        return []

    def _dirty(self, row) -> bool:
        """Would healing's own rules drop this whole (unflagged) memory? It names the
        destination or its host label, was read off the destination's site, (oracle)
        undermines the legitimate system, restates a claim ddmin took from a culprit,
        was read off a root page, or asserts a change a tainted page supports or points
        at a root page.

        A 'ddmin:inconclusive' claim counts here though it never spreads by restates:
        putting it back into the culprit was not shown safe (inconclusive is not safe),
        so a parent restating it is not put back into retrieval either, while nothing
        shows it carries the decision, so it is not dropped from other memories.

        Stricter than recovery.carries_attacker, which matches whole hosts only: a
        composed attack's page leaves extractions such as "The RentQuick endpoint is
        active." that the incident never flags, sitting under the summary it flagged.
        Rebuilding that summary from them, or bringing them back as its clean parents,
        would put the attack back in retrieval."""
        if carries(row["text"], row, self.spec) or any(
                claims.restates(row["text"], c) for c in self.cf_poisoned + self.cf_unproven):
            return True
        if self.oracle:
            return False
        return row["id"] in self.taint.siblings or self._source_tainted(row["text"], row)

    def _versions(self, p: str) -> list[str]:
        """Healed memories an earlier heal wrote for `p` (their first parent is `p`)."""
        out = []
        for c in db.children_of(self.conn, p):
            r = db.get_memory(self.conn, c)
            if (r is not None and r["operation"] == HEAL_OP
                    and r["status"] in ("active", "consolidated")
                    and db.parents_of(self.conn, c)[:1] == [p]):
                out.append(c)
        return out

    # --- writing ----------------------------------------------------------------------

    def _write(self, m: str, row, texts: list[str], cited: list[list[str]],
               prompt_hash: str | None, model: str | None) -> list[str]:
        """Healed memories for `m`, then `m` quarantined. One that replaces an absorbed
        memory goes in 'consolidated' for now; _finish decides whether anything still
        carries it."""
        status = "consolidated" if row["status"] == "consolidated" else "active"
        ids = []
        for text, cites, vec in zip(texts, cited, recovery._embed(self.embedder, texts)):
            ids.append(db.insert_memory(
                self.conn, text, "derived", row["session"],
                operation=HEAL_OP,
                status=status,
                created_by_prompt=prompt_hash,
                model=model,
                embedding=vec,
                parent_ids=[m, *cites],
            ))
        self.replaced[m] = ids
        self._take_out(m)
        return ids

    def _take_out(self, m: str) -> None:
        row = db.get_memory(self.conn, m)
        if row is not None and row["status"] not in ("quarantined", "deleted"):
            self.quarantined.append(m)
        recovery._quarantine(self.conn, m)
        self.gone.add(m)

    def _out(self, m: str, reason: str, replacement: str | None = None) -> None:
        """Quarantine a flagged (or swept) memory with nothing healed in its place."""
        self._take_out(m)
        if m not in self.swept:
            self.flagged_quarantined.append(m)
        self._log(m, "quarantine", replacement=replacement, reason=reason)

    def _log(self, m: str, action: str, *, replacement: str | None = None,
             reason: str | None = None) -> None:
        why = self.plan.reasons.get(m) if action in ("heal", "quarantine", "keep") else None
        db.log_recovery(self.conn, self.recovery_id, self.plan.mode, m, action,
                        replacement=replacement,
                        reason="; ".join(r for r in (why, reason) if r) or None)

    def _count(self, labels_: Iterable[str]) -> None:
        """Count dropped claims, by the rule that dropped each."""
        rules = [rule_of(w) for w in labels_]
        self.claims_dropped += len(rules)
        self.dropped_by.update(rules)

    # --- the sweep ----------------------------------------------------------------------

    def _sweep_candidates(self) -> list:
        """Active memories read off a tainted page that the incident did not flag and that
        the free rules ((a), (b), or the oracle) would drop something from: the whole
        text is judged first, so a memory with nothing to drop costs no split."""
        if not self.pages.pages:
            return []
        out = []
        for row in db.active_memories(self.conn):
            if (row["id"] not in self.plan.heal and row["operation"] != PROBE_OP
                    and page_of(row) in self.pages.pages and self._rule(row["text"], row)):
                out.append(row)
        return out

    def _sweep(self) -> None:
        """Heal the sweep's memories (module docstring, rule b) claim by claim, like a
        flagged memory. One whose split leaves no claim to drop is left as it is."""
        for row in self._sweep_candidates():
            m = row["id"]
            split = self._split(row["text"], row, f"claims:{m}")
            if not self._labels(m, row, split.claims):
                continue
            where = "a root page" if m in self.pages.siblings else "a page pointing at a root page"
            self.swept.append(m)
            self._claim_heal(m, row, f"unflagged, read off {where}; ")

    # --- step 3 ------------------------------------------------------------------------

    def _finish(self) -> None:
        conn = self.conn
        # A healed copy of an absorbed memory stays under the summaries this recovery
        # keeps that absorbed the original, or it would surface beside them.
        for m, ids in self.replaced.items():
            carriers = self._carriers(m)
            for h in ids:
                if carriers and db.get_memory(conn, h)["status"] == "consolidated":
                    for a in carriers:
                        db.insert_edges(conn, a, [h])
        # Clean parents of what healing took out come back, unless something it keeps
        # still carries them (unmerge's rule), or healing's own rules would drop them
        # (a root's sibling among them). A claim-healed summary carries only the parents
        # behind a claim it kept. Nothing comes back from under a culprit nothing was
        # localised in: whatever made the decision may be in any of its parents.
        gone = recovery._store_order(conn, self.gone)
        withheld = {p for m in gone if self.unlocalised(m)
                    for p in recovery.restore_parents(conn, m, self.tainted, self.spec,
                                                      removed=self.gone)}
        for m in gone:
            if self.unlocalised(m):
                continue
            kept = self.claim_healed.get(m)
            out = self.gone | set(self.replaced[m]) if kept is not None else self.gone
            for p in recovery.restore_parents(conn, m, self.tainted, self.spec, removed=out):
                row = db.get_memory(conn, p)
                if p in self.suspect or p in self.restored or p in withheld or self._dirty(row):
                    continue
                if kept is not None and self._holds(kept, row["text"]):
                    continue
                if recovery._reactivate(conn, p, out):
                    self.restored.append(p)
                    behind = "; behind no claim its healed copy kept" if kept is not None else ""
                    self._log(p, "restore", reason=f"clean-parent-of:{m}{behind}")
        # A healed copy that nothing carries is the only copy of its claims. Newest
        # first, so a copy is settled before the copies it might carry.
        healed = [h for ids in self.replaced.values() for h in ids]
        for h in reversed(recovery._store_order(conn, healed)):
            if (db.get_memory(conn, h)["status"] == "consolidated" and h not in self.suspect
                    and not recovery.still_absorbed(conn, h, self.gone)):
                db.set_status(conn, h, "active")

    def _holds(self, kept: list[str], text: str) -> bool:
        """Do the claims a filter kept still carry a parent's `text`? One of them came
        from it (it supports the claim), or together they hold it (they support it).
        Either way bringing it back would make the same fact retrievable twice."""
        return (any(self._supports(c, [text]) for c in kept)
                or self._supports(text, [claims.compose(kept)]))

    def _carriers(self, m: str) -> list[str]:
        """What carries `m`'s healed copies: its absorbing children this recovery keeps,
        and the claim-healed copies of those it took out (not its own healed copies).

        A claim-healed summary is the summary's own text without the attack, so it
        still says what it absorbed; a rebuilt one cites what it drew on itself. Without
        this, a Mem0 page's healed facts row would surface on its own beside the healed
        update that merged them, a memory Mem0 never had."""
        out = []
        own = set(self.replaced.get(m, ()))
        for c in db.children_of(self.conn, m):
            r = db.get_memory(self.conn, c)
            if r is None or c in own or not recovery._absorbs(self.conn, r):
                continue
            if c in self.claim_healed:
                out += self.replaced[c]
            elif c not in self.gone and r["status"] in ("active", "consolidated"):
                out.append(c)
        return out

    # --- step 4 ------------------------------------------------------------------------

    def _post_heal_check(self) -> None:
        """Replay the explained decision n times on the frozen set as healing left it:
        each healed memory in its original's place, what was quarantined removed, the
        rest as it was. Only when a retrieved memory was healed; a set healing only
        removed from is what the memory-level counterfactual already replayed. Each
        healed retrieved memory gets a 'post_heal_check' event with the outcome."""
        retrieved = self.ctx.probe.retrieved
        healed = [x for x in retrieved if x in self.replaced]
        if not healed:
            return
        frozen: list[str] = []
        for x in retrieved:
            if x in self.replaced:
                frozen += self.replaced[x]
            elif x not in self.gone:
                frozen.append(x)
        check = _Variant(frozen, "post-heal check",
                         replays=self._replay([(f"post-heal:{i}", list(frozen))
                                               for i in range(max(self.ctx.probe.n, 1))]))
        self.post_heal = (sum(r == HARMFUL for r in check.replays), len(check.replays))
        for m in healed:
            self._log(m, "post_heal_check", replacement=",".join(self.replaced[m]),
                      reason=f"post-heal check: {check.outcome}")

    # --- helpers -----------------------------------------------------------------------

    def _split(self, text: str, row, tag: str) -> claims.Split:
        """The cached split, unless the model's claims lost what makes the text carry the
        attack: then the deterministic split, which keeps every word, is judged instead."""
        split = self.ctx.splits.split(self.llm if self.has_llm else None, text, tag=tag)
        if (split.how == "llm" and carries(text, row, self.spec)
                and not any(carries(c, row, self.spec) for c in split.claims)):
            return claims.Split(claims.fallback_split(text), "fallback", split.claims)
        return split

    def _rule(self, claim: str, row, *, cf: bool = True, origins: list | None = None) -> str | None:
        """Why `claim` (of memory `row`) is poisoned by rule (a), (b) or the oracle, or by
        restating a claim ddmin found (cf=False leaves that out); None if it is not.
        `origins` stands in for a rebuilt text's ancestry (the extractions its inputs
        came from)."""
        if recovery.claim_carries_attack(claim, row, self.spec):
            return "oracle:attack" if self.oracle else "name"
        if self.oracle:
            return "oracle:ambiguous" if recovery.claim_is_ambiguous(claim, self.spec) else None
        if self._source_tainted(claim, row, origins):
            return "source-taint"
        if cf and any(claims.restates(claim, p) for p in self.cf_poisoned):
            return "cf:restates"
        return None

    def _source_tainted(self, claim: str, row, origins: list | None = None) -> bool:
        """Rule (b): the claim points at a root page, or it asserts a change and comes
        from a tainted page."""
        if self.taint.points_at_root(claim):
            return True
        return bool(CHANGE.search(claim)) and self._from_tainted(claim, row, origins)

    def _from_tainted(self, claim: str, row, origins: list | None = None) -> bool:
        """Does `claim` come from a tainted page? Its memory was read off one, or an
        extraction from one among its origins (by default the memory's ancestors)
        supports it."""
        if not self.taint.pages:
            return False
        if origins is None:
            if row["source_type"] == "webpage":
                return page_of(row) in self.taint.pages
            origins = self._origins(row)
        return any(page_of(o) in self.taint.pages and self._supports(claim, [o["text"]])
                   for o in origins)

    def _origins(self, row) -> list:
        """The extractions a memory's claims can have come from: itself, if it was read
        off a page; else the ancestors that were."""
        if row["source_type"] == "webpage":
            return [row]
        m = row["id"]
        if m not in self._ancestors:
            rows = (db.get_memory(self.conn, a) for a in db.get_ancestors(self.conn, m))
            self._ancestors[m] = [r for r in rows if r is not None and page_of(r)]
        return self._ancestors[m]

    def _supports(self, claim: str, texts: list[str]) -> bool:
        """Does one of `texts` support `claim`? It holds SUPPORT_SHARE of the claim's
        distinctive words, or (with an embedder) is SUPPORT_COSINE close to it."""
        texts = [t for t in texts if t]
        if any(claims.grounded(claim, t, SUPPORT_SHARE) for t in texts):
            return True
        if self.embedder is None or not texts:
            return False
        vecs = self._vectors([claim, *texts])
        return max(float(vecs[0] @ v) for v in vecs[1:]) >= SUPPORT_COSINE

    def _vectors(self, texts: list[str]) -> list[np.ndarray]:
        """Unit vectors for `texts`, each text embedded once per recovery."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._vec]
        if missing:
            for t, v in zip(missing, self.embedder.encode(missing)):
                v = np.asarray(v, dtype=np.float32)
                norm = float(np.linalg.norm(v))
                self._vec[t] = v / norm if norm else v
        return [self._vec[t] for t in texts]

    def _labels(self, m: str, row, cl: list[str]) -> dict[str, str]:
        """claim -> why it is poisoned, for the poisoned claims of `m`: the claim-level
        counterfactual's own verdict on a culprit's claims first, then the rules."""
        own = self.cf.get(m, {})
        out = {}
        for c in cl:
            why = own.get(c) or self._rule(c, row)
            if why:
                out[c] = why
        return out

    def _summary(self) -> dict[str, Any]:
        new_ids = [h for ids in self.replaced.values() for h in ids]
        post = f"{self.post_heal[0]}/{self.post_heal[1]}" if self.post_heal else "none"
        by = " ".join(f"{r}={self.dropped_by[r]}" for r in RULES if self.dropped_by[r])
        healed = [m for m in self.replaced if m not in self.swept]
        note = (f"{self.plan.mode}: flagged={len(self.plan.heal)} healed={len(healed)} "
                f"(rebuilt={self.rebuilt}) quarantined={len(self.flagged_quarantined)} "
                f"unchanged={len(self.kept)} already_out={len(self.already_out)} "
                f"swept={len(self.swept)} restored={len(self.restored)} "
                f"claims kept={self.claims_kept} "
                f"dropped={self.claims_dropped} claim_cf_replays={self.cf_replays} "
                f"probe_rows={len(self.probes)} ddmin_replays={self.ddmin_replays} "
                f"budget_spent={len(self.budget_spent)} inconclusive={self.inconclusive} "
                f"post_heal_harmful={post} dropped_by=[{by}]")
        return {
            "quarantined": self.quarantined,
            "new_ids": new_ids,
            "restored": self.restored,
            "healed": healed,
            "flagged_quarantined": self.flagged_quarantined,
            "swept": self.swept,
            "unchanged": self.kept,
            "already_out": self.already_out,
            "rebuilt": self.rebuilt,
            "claims_kept": self.claims_kept,
            "claims_dropped": self.claims_dropped,
            "dropped_by": {r: self.dropped_by[r] for r in RULES},
            "claim_cf_replays": self.cf_replays,
            "ddmin_replays": self.ddmin_replays,
            "budget_spent": self.budget_spent,
            "inconclusive": self.inconclusive,
            "post_heal": self.post_heal,
            "probes": self.probes,
            "note": note,
        }


def _change_terms(text: str) -> frozenset[str]:
    """The change language in `text` (CHANGE), each term by the first four letters of
    its first word and without 'be': 'retired' and 'retiring' are one term, 'not be
    processed' and 'not processed' another."""
    out = set()
    for match in CHANGE.finditer(text):
        words = [w for w in match.group(0).lower().split() if w != "be"]
        out.add(" ".join([words[0][:4], *words[1:]]))
    return frozenset(out)


def _dropped(why: Mapping[str, str]) -> str:
    """'; dropped: 'claim' (why), ...' for an event's reason, or ''."""
    if not why:
        return ""
    return "; dropped: " + ", ".join(f"{c!r} ({w})" for c, w in why.items())
