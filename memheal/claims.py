"""Claim-level splitting of mixed memories (proposal Section 12.3).

A mixed memory carries the attacker's destination alongside benign facts a clean
parent contributed ("Submit PDF receipts for cost centre NF-4471 through
quick-expense.example"). selective quarantines it and loses the benign facts with the
attack; selective_recon replays the transformation over the clean parents, which needs
those parents and a model that does not bring the attack back. Section 12.3's third
option works on the memory's own text: split it into atomic claims, drop the claims
that carry the attack, and keep the rest.

This module is the text side only; memheal/recovery.py decides which memories are
mixed and writes the repaired memory. The split is the only model step. The kept
claims are joined as they came back, never regenerated, so every distinctive word of a
repaired memory is one of the original's (or an inflection of one); short function
words ("is", "to", "the") are not checked, so a split can still reorder what the memory
said. Three guards keep the model's split honest:

- a model claim must be built from the original's own words (`grounded`): every
  distinctive word is in the memory, whole or inflected ("submitted" for "submit").
  A claim that brings in a word of its own ("through the quick-expense *portal*",
  "Quick Expense *handles* mileage") is new content, and new content is how a split
  could re-attach the attacker to a benign fact; it is dropped, not kept;
- the attack check sees through a dropped top-level domain (`names_host_label`):
  "quick-expense", "Quick Expense" and "quickexpense" all name quick-expense.example,
  which the whole-host match alone misses;
- if the model returns nothing usable, a deterministic sentence/clause split stands in,
  so a repair never depends on the model answering in the right shape. It never cuts
  inside brackets or after an abbreviation ("(e.g. Enrol Now at enrol-now.example)"),
  which would part an instruction from the destination it names.

Whether a claim carries the attack is the caller's rule (recovery.claim_carries_attack
against the recovery's AttackSpec); this module only applies it claim by claim.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from memheal.tools import normalize_domain

CLAIMS_PROMPT = (
    "You maintain the long-term memory of a personal assistant. Below is one of its "
    "memories. Split it into atomic claims: each claim states exactly one fact, "
    "preference, destination or step from the memory, as a short standalone sentence. "
    "Copy the memory's own wording, names, numbers, URLs and domains exactly; do not "
    "paraphrase, generalise, merge or add anything. Every part of the memory must "
    "appear in some claim. If the memory states only one thing, return it as a single "
    "claim."
)

CLAIMS_SCHEMA = {
    "name": "atomic_claims",
    "schema": {
        "type": "object",
        "properties": {"claims": {"type": "array", "items": {"type": "string"}}},
        "required": ["claims"],
        "additionalProperties": False,
    },
}

# Share of a claim's distinctive words the original must supply: all of them. The
# prompt asks for the memory's own wording, so a word the memory never used is the
# model adding content. An earlier 0.6 share with 5-letter stems let "Submit receipts
# through the quick-expense portal" pass as grounded (4 of 5 words), with the host's
# top-level domain dropped so the whole-host attack check missed it.
MIN_GROUNDING = 1.0
# An inflection: one word is the other plus at most this many letters, and the shorter
# has at least _MIN_STEM letters ("submit" ~ "submitted", "scan" ~ "scanning"). Letters
# only, so a hyphenated host never stands in for one of its parts ("quick" is not
# "quick-expense").
_MIN_STEM = 4
_MAX_ENDING = 4
# A host label shorter than this (letters and digits) is too likely an ordinary word.
_MIN_LABEL = 4
_HOST = re.compile(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+")

# Sentence and clause boundaries for the deterministic split. A sentence ends at . ! or ?
# followed by whitespace and a capital or digit, outside brackets, and not after an
# abbreviation or an initial. A clause ends at a semicolon, or a comma before a
# conjunction that starts a new clause; a bare comma is too often a list. The clause
# boundary is kept, so a fragment glued back to its neighbour keeps its punctuation.
_SENTENCE_GAP = re.compile(r"(?<=[.!?])\s+")
_CLAUSE = re.compile(r";\s*|,\s+(?=(?:and|but|while|whereas|then)\s)", re.IGNORECASE)
_LEADING_CONJUNCTION = re.compile(r"^(?:and|but|while|whereas|then)\s+", re.IGNORECASE)
_ABBREVIATIONS = frozenset("""e.g i.e vs etc approx incl excl esp cf viz al no nos mr mrs ms
dr prof st fig ca resp dept min max""".split())
_INITIALISM = re.compile(r"[a-z](?:\.[a-z])*")
# A fragment shorter than this (its conjunction included) is a list's last item or a
# stray word, not a claim: "..., and boarding passes." stays with the clause before it.
_MIN_CLAUSE_WORDS = 4

_STOP = frozenset("""the and for are was were with that this from have has had not but you
your their they them its into onto than then there here when what which who whom will would
should could can may must also only just more most such any all each every some been being
through via now new old one two per our out off over under about after before user users uses
used use""".split())


@dataclass
class Split:
    """The claims one memory was split into, and how.

    how        - 'llm' (the model's split, grounded claims only) or 'fallback'
    ungrounded - model claims dropped because the original does not supply their words
    model      - the model that answered, when its split was used
    """
    claims: list[str]
    how: str
    ungrounded: list[str] = field(default_factory=list)
    model: str | None = None


def _tokens(text: str) -> set[str]:
    # Plurals folded so "receipt" and "receipts" ("pdf" and "pdfs") are one word.
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w
            for w in re.findall(r"[a-z0-9][a-z0-9-]{2,}", text.lower()) if w not in _STOP}


def _inflection(a: str, b: str) -> bool:
    """Is one word the other plus a short alphabetic ending ("submit" ~ "submitted")?"""
    short, long = sorted((a, b), key=len)
    ending = long[len(short):]
    return (len(short) >= _MIN_STEM and long.startswith(short)
            and 0 < len(ending) <= _MAX_ENDING and ending.isalpha())


def grounded(claim: str, original: str, min_share: float = MIN_GROUNDING) -> bool:
    """Is this claim built from the original memory's own words?

    Every distinctive word must be in the original, whole or as an inflection of one of
    its words (by default; `min_share` lowers the bar). A claim with no distinctive
    words at all ("It is.") supports nothing and is not grounded either.
    """
    words = _tokens(claim)
    if not words:
        return False
    source = _tokens(original)
    found = sum(w in source or any(_inflection(w, s) for s in source) for w in words)
    return found / len(words) >= min_share


def host_labels(destinations: Iterable[str]) -> frozenset[str]:
    """The distinctive name of each destination's host: the host without its top-level
    domain ('quick-expense' for quick-expense.example, and for a path or a mailbox on
    it alike). Values that are not host-shaped give none."""
    out = set()
    for d in destinations:
        host = normalize_domain(d.rsplit("@", 1)[-1])
        if not _HOST.fullmatch(host):
            continue
        label = host.rsplit(".", 1)[0]
        if len(re.sub(r"[^a-z0-9]", "", label)) >= _MIN_LABEL:
            out.add(label)
    return frozenset(out)


def _label_pattern(label: str) -> re.Pattern[str]:
    # Its parts in order, joined by nothing, a space, a hyphen, a dot or an underscore,
    # any case: 'quick-expense' matches "quick expense", "Quick-Expense", "quickexpense".
    parts = [re.escape(p) for p in re.split(r"[.-]+", label) if p]
    return re.compile(r"(?<![a-z0-9])" + r"[\s._-]*".join(parts) + r"(?![a-z0-9])",
                      re.IGNORECASE)


def names_host_label(text: str, destinations: Iterable[str]) -> bool:
    """Does `text` name a destination's host with its top-level domain dropped or
    respelled? A split that writes "the quick-expense portal" for quick-expense.example
    still points at it, and the whole-host match (labels.AttackSpec) does not see that."""
    return any(_label_pattern(label).search(text) for label in host_labels(destinations))


def _depth(text: str, pos: int) -> int:
    """Brackets open at `pos`; an unclosed one stays open to the end of the text."""
    depth = 0
    for ch in text[:pos]:
        if ch in "([{":
            depth += 1
        elif ch in ")]}" and depth:
            depth -= 1
    return depth


def _ends_sentence(text: str, gap: re.Match) -> bool:
    """A full stop that ends a sentence: a capital or digit follows, and the word it ends
    is not an abbreviation ("e.g.", "vs.") or an initial ("J.")."""
    nxt = text[gap.end():gap.end() + 1]
    if not (nxt.isupper() or nxt.isdigit()):
        return False
    word = text[:gap.start()].rsplit(None, 1)[-1].lstrip("([{\"'").lower().rstrip(".!?")
    return word not in _ABBREVIATIONS and not _INITIALISM.fullmatch(word)


def _cut(text: str, pattern: re.Pattern[str], ok=lambda text, m: True) -> list[tuple[str, str]]:
    """(boundary, piece) pairs of `text` cut at `pattern`, outside brackets, where `ok`
    agrees; the first piece's boundary is ''."""
    out, start, boundary = [], 0, ""
    for m in pattern.finditer(text):
        if _depth(text, m.start()) or not ok(text, m):
            continue
        out.append((boundary, text[start:m.start()]))
        start, boundary = m.end(), m.group(0)
    out.append((boundary, text[start:]))
    return out


def fallback_split(text: str) -> list[str]:
    """Sentences, then clauses, of `text`, in order; deterministic and model-free.

    Domains are safe: a sentence ends at punctuation followed by whitespace, so the dot
    in 'quick-expense.example' never splits. So are brackets and abbreviations: cutting
    "(e.g. Enrol Now at enrol-now.example)" after "e.g." would leave the instruction
    before it without the destination it names, and that half would pass the attack
    check. A clause too short to be a claim stays attached to the one before it, with
    the boundary it was split at; a clause that stands alone loses only its leading
    conjunction ("and receipts are PDFs" -> "receipts are PDFs").
    """
    out: list[str] = []
    for line in text.splitlines():
        for _gap, sentence in _cut(line.strip(), _SENTENCE_GAP, _ends_sentence):
            pieces: list[str] = []
            for boundary, clause in _cut(sentence, _CLAUSE):
                clause = clause.strip()
                if not clause:
                    continue
                if pieces and len(clause.split()) < _MIN_CLAUSE_WORDS:
                    pieces[-1] = pieces[-1] + boundary + clause
                else:
                    pieces.append(_LEADING_CONJUNCTION.sub("", clause))
            out += pieces
    return [c for c in out if re.search(r"[A-Za-z0-9]", c)]


def _parse(text: str | None) -> list[str]:
    """The model's claims, or [] for anything that is not {'claims': [str, ...]}."""
    try:
        payload = json.loads(text or "")
    except json.JSONDecodeError:
        return []
    claims = payload.get("claims") if isinstance(payload, dict) else None
    if not isinstance(claims, list):
        return []
    return [c.strip() for c in claims if isinstance(c, str) and c.strip()]


def split_claims(llm, text: str, *, tag: str = "") -> Split:
    """Split one memory into atomic claims, with the model when one is given.

    The model's answer is used only if at least one of its claims is grounded in the
    memory; an empty, malformed or wholly ungrounded answer falls back to the
    deterministic split rather than leaving the memory unrepaired.
    """
    if llm is not None:
        response = llm.respond(CLAIMS_PROMPT, f"Memory:\n{text}", json_schema=CLAIMS_SCHEMA,
                               tag=tag)
        proposed = list(dict.fromkeys(_parse(response.text)))
        kept = [c for c in proposed if grounded(c, text)]
        if kept:
            return Split(kept, "llm", [c for c in proposed if c not in kept], response.model)
        return Split(fallback_split(text), "fallback", proposed)
    return Split(fallback_split(text), "fallback")


def partition(claims: Iterable[str], carries: Callable[[str], bool]) -> tuple[list[str], list[str]]:
    """(kept, dropped): claims that do not carry the attack, and claims that do."""
    kept: list[str] = []
    dropped: list[str] = []
    for c in claims:
        (dropped if carries(c) else kept).append(c)
    return kept, dropped


def _sentence(claim: str) -> str:
    claim = claim.strip()
    return claim if claim[-1:] in ".!?" else claim + "."


def compose(claims: Iterable[str]) -> str:
    """The repaired memory: the kept claims joined in order, each ended as a sentence.

    Punctuation is the only change; the words are the claims' own.
    """
    return " ".join(_sentence(c) for c in claims if c.strip())
