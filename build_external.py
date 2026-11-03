"""Build the LongMemEval- and MemBench-derived case files (proposal Sections 7.4, 7.5).

    uv run python build_external.py            # reads and writes data/external/

Reads the downloaded benchmark files and writes two compact case files that
`memheal.scenarios.external` turns into scenarios:

    data/external/cases_longmemeval.json   30 knowledge-update + 30 single-session-user /
                                           multi-session questions (oracle variant), the
                                           latter two taken in turn; multi-session runs
                                           out first, since most of its answers are
                                           computed totals that no session states
    data/external/cases_membench.json      20 factual (ThirdAgent/simple) + 20 reflective
                                           (FirstAgent highlevel even tids, ThirdAgent
                                           highlevel odd tids)

The downloads are untrusted data: they are parsed with `json` and nothing else, and the
loader script shipped with MemBench is never run. Selection is deterministic (sorted by
id, no sampling), so rebuilding gives the same cases.

A question is kept only if its answer can be scored the way recovery scores every benign
probe: a case-insensitive substring of the reply. So the expected string has to be short
and unambiguous (no alternatives, no parenthetical, no bare "yes" or small count that
any reply would contain), and - except for MemBench's reflective questions, whose answer
is an inference by design - it has to be stated in the history the agent is given, after
the per-session cap, in the turns the benchmark marks as evidence. Otherwise a failed
probe after recovery would say nothing about whether recovery lost the fact.

Reflective questions are multiple choice: their wrong options are kept as `distractors`,
so the probe can be scored as naming the right option and no other
(`memheal.scenarios.external.probe_correct`), not as containing it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

from memheal.scenarios.external import (SESSION_CHARS, cap, compose, kept_turns,
                                        leaked_hosts, localize_hosts)

ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "data" / "external"

LONGMEMEVAL_SRC = Path("longmemeval") / "longmemeval_oracle.json"
MEMBENCH_SIMPLE = Path("membench") / "ThirdAgent" / "simple.json"
MEMBENCH_HIGH = {"first": Path("membench") / "FirstAgent" / "highlevel.json",
                 "third": Path("membench") / "ThirdAgent" / "highlevel.json"}

UPDATE_TYPE = "knowledge-update"
OTHER_TYPES = ("single-session-user", "multi-session")
# Factual MemBench categories with one answer each. `hybrid` joins two questions (some
# half in Chinese) into one with a compound answer, which a substring check cannot score.
FACTUAL_KINDS = ("roles", "events", "items", "places")

MAX_WORDS = 4
NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty once twice".split())
PRONOUN_START = re.compile(r"(?:i|we|you|he|she|they)(?:'\w+)?\b", re.I)
AGGREGATE = re.compile(r"\b(?:total|more|less|fewer|difference|earlier|later|average|"
                       r"combined|altogether|how many|how much)\b", re.I)
# Words too common to tie a turn to the question's subject (see stated_in).
STOPWORDS = frozenset(
    "the and for are was were has have had does did what which who whom whose when where "
    "why how with from that this these those your you about into its his her their they "
    "them there been being any some someone something much many most more own owned like "
    "exactly kind might may can could would should will just also not but all one".split())


# --- answers a substring check can score ----------------------------------------

def canonical_answer(answer) -> str | None:
    """A short expected string for `answer`, or None if a substring check can't score it.

    Leading articles and hedges are dropped ("a yellow dress" is answered "the yellow
    dress", "about two hours" as "two hours"); thousands separators are the only commas
    allowed ("$400,000" is one value, "Boston, MA" is two ways of writing it). An answer
    phrased as the user's own sentence ("We've met up twice") is rejected: the reply
    says it in the second person.
    """
    a = str(answer).strip().rstrip(".").strip()
    if re.search(r"[();/]|,(?!\d{3})|\bor\b", a, re.I) or PRONOUN_START.match(a):
        return None
    a = re.sub(r"^(?:(?:a|an|the|about|around|approximately|roughly|nearly|almost)\s+)+", "",
               a, flags=re.I)
    if len(a.split()) > MAX_WORDS or too_weak(a):
        return None
    return a


def too_weak(expected: str) -> bool:
    """True for answers most replies contain by accident: yes/no, a small count."""
    low = expected.lower()
    return (len(low) < 3 or low in {"yes", "no", "none"} or low in NUMBER_WORDS
            or bool(re.fullmatch(r"\d{1,2}", low)))


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[^\W\d_]+", text.lower())
            if len(w) > 2 and w not in STOPWORDS}


def stated_in(expected: str, turns: list[tuple[str, bool]], question: str) -> bool:
    """The answer is in the history the agent sees - in an evidence turn, when the source
    flags any - and the question does not give it away.

    `turns` are the (text, is_evidence) turns kept after the cap and localisation.
    Evidence matters because MemBench plants the same value about someone else beside
    the target ("My dad is 29 years old." next to "My subordinate is 29 years young."):
    the answer turning up anywhere says nothing about how the history states the fact,
    and a correct recall ("29 years young") would fail the probe. An unflagged turn
    still counts when it is about the question's subject - it shares a content word
    with the question that the evidence also has - as when the target says "My brother
    flies planes for a living." and another turn "My brother's a commercial airline
    pilot".
    """
    low = expected.lower()
    if low in question.lower():
        return False
    evidence = [t for t, ev in turns if ev]
    if not evidence:
        return any(low in t.lower() for t, _ in turns)
    if any(low in t.lower() for t in evidence):
        return True
    anchors = content_words(question) & set().union(*map(content_words, evidence))
    return any(low in t.lower() and content_words(t) & anchors for t, ev in turns if not ev)


def unambiguous_choice(expected: str, choices: dict) -> bool:
    """No other option contains the answer or is contained in it ("Sour" vs "Sour and
    Salty"), so a reply naming another option cannot pass by substring."""
    low = expected.lower()
    others = [str(c).lower() for c in choices.values() if str(c).lower() != low]
    return all(low not in o and o not in low for o in others)


def date_of(stamp: str) -> str:
    """'2023/05/25 (Thu) 20:21' -> '2023/05/25'; "'2024-10-01 08:00' Tuesday" -> '2024-10-01'."""
    m = re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", str(stamp))
    return m.group(0) if m else str(stamp).strip(" '")


def localized(case: dict) -> dict:
    """The case with every hostname in its text rewritten to a reserved one."""
    out = dict(case, question=localize_hosts(case["question"]),
               expected=localize_hosts(case["expected"]))
    if "distractors" in case:
        out["distractors"] = [localize_hosts(d) for d in case["distractors"]]
    out["sessions"] = [dict(s, date=localize_hosts(s["date"]), text=localize_hosts(s["text"]))
                       for s in case["sessions"]]
    return out


def localized_turns(turns: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
    return [(localize_hosts(t), ev) for t, ev in turns]


# --- LongMemEval -----------------------------------------------------------------

def longmemeval_case(q: dict, limit: int = SESSION_CHARS) -> dict | None:
    """One oracle question -> a case, or None when it is not scorable by substring."""
    if q["question_id"].endswith("_abs"):        # abstention: the answer is "not stated"
        return None
    # A multi-session total or difference is computed, not stated; its value turning up
    # somewhere in the history (one item's price) is a coincidence, not the answer.
    if q["question_type"] == "multi-session" and AGGREGATE.search(q["question"]):
        return None
    expected = canonical_answer(q["answer"])
    if expected is None:
        return None
    dated = sorted(zip(q["haystack_dates"], q["haystack_session_ids"], q["haystack_sessions"]),
                   key=lambda t: t[0])
    sessions, kept = [], []
    for stamp, sid, turns in dated:
        user = [(t["content"], bool(t.get("has_answer"))) for t in turns if t["role"] == "user"]
        session_turns = kept_turns(user, limit)
        if session_turns:
            sessions.append({"id": sid, "date": date_of(stamp),
                             "text": "\n".join(t for t, _ in session_turns)})
            kept += session_turns
    case = localized({"source": "longmemeval", "id": q["question_id"],
                      "kind": q["question_type"], "question": q["question"],
                      "expected": expected, "answer": str(q["answer"]),
                      "question_date": q.get("question_date", ""), "sessions": sessions})
    if not case["sessions"] or not stated_in(case["expected"], localized_turns(kept),
                                             case["question"]):
        return None
    return case


def interleave(*groups: list) -> list:
    """a0 b0 a1 b1 ...: round-robin context assignment then spreads every group over
    every domain and attack category instead of giving each group a block of them."""
    out = []
    for i in range(max((len(g) for g in groups), default=0)):
        out += [g[i] for g in groups if i < len(g)]
    return out


def select_longmemeval(questions: list[dict], n_update: int = 30,
                       n_other: int = 30) -> list[dict]:
    """First scorable questions by id: n_update knowledge-update, and n_other taken in
    turn from single-session-user and multi-session (one fills in if the other runs out)."""
    by_type: dict[str, list[dict]] = {}
    for q in sorted(questions, key=lambda q: q["question_id"]):
        case = longmemeval_case(q)
        if case is not None:
            by_type.setdefault(q["question_type"], []).append(case)
    update = by_type.get(UPDATE_TYPE, [])[:n_update]
    other = interleave(*(by_type.get(t, []) for t in OTHER_TYPES))[:n_other]
    return interleave(update, other)


# --- MemBench --------------------------------------------------------------------

def _user_text(message: dict) -> str:
    """ThirdAgent messages are observations (`message`); FirstAgent ones are dialogue
    turns, of which only the user's side is kept (`user`, or `user_message`)."""
    for key in ("message", "user", "user_message"):
        if isinstance(message.get(key), str):
            return message[key]
    return ""


def pack(messages: list[tuple[str, str, bool]],
         limit: int = SESSION_CHARS) -> tuple[list[dict], list[tuple[str, bool]]]:
    """Consecutive (date, text, is_evidence) messages -> sessions of at most `limit`, and
    the (text, is_evidence) turns they keep.

    MemBench's factual trajectories have no session boundaries: a message per day
    would mean 9-15 sessions for under 1k characters. Packing in order keeps each
    trajectory to one or two sessions and never splits a message.
    """
    groups: list[list[tuple[str, str, bool]]] = []
    used = 0
    for date, text, ev in messages:
        text = cap(text, limit)
        if groups and used + 1 + len(text) <= limit:
            groups[-1].append((date, text, ev))
            used += 1 + len(text)
        else:
            groups.append([(date, text, ev)])
            used = len(text)
    sessions, kept = [], []
    for i, g in enumerate(groups, start=1):
        group_turns = kept_turns([(t, ev) for _d, t, ev in g], limit)
        sessions.append({"id": f"part{i}", "date": g[0][0],
                         "text": "\n".join(t for t, _ in group_turns)})
        kept += group_turns
    return sessions, kept


def membench_factual_case(kind: str, traj: dict, limit: int = SESSION_CHARS) -> dict | None:
    qa = traj["QA"]
    expected = canonical_answer(qa["answer"])
    if expected is None or not unambiguous_choice(expected, qa.get("choices", {})):
        return None
    targets = set(qa.get("target_step_id") or [])
    msgs = [(date_of(m.get("time", "")), _user_text(m), m.get("mid") in targets)
            for m in traj["message_list"]]
    sessions, kept = pack(msgs, limit)
    case = localized({"source": "membench", "id": f"ThirdAgent/simple/{kind}/{traj['tid']}",
                      "kind": f"fact-{kind}", "tid": traj["tid"], "memory": "factual",
                      "question": qa["question"], "expected": expected,
                      "answer": str(qa["answer"]), "sessions": sessions})
    if not stated_in(case["expected"], localized_turns(kept), case["question"]):
        return None
    return case


def membench_reflective_case(agent: str, kind: str, traj: dict,
                             limit: int = SESSION_CHARS) -> dict | None:
    """A preference question. The answer (a genre, a flavour) is never stated, so the
    options are put in the probe, as MemBench itself asks it, and the expected string is
    the right option's text. The other options are kept as `distractors`: the probe
    gives the answer away among them, so it is scored as naming it and no other."""
    qa = traj["QA"]
    choices = qa.get("choices") or {}
    truth = qa.get("ground_truth")
    expected = canonical_answer(choices.get(truth, qa["answer"]))
    if expected is None or not unambiguous_choice(expected, choices):
        return None
    distractors = [str(v).strip() for k, v in sorted(choices.items())
                   if k != truth and str(v).strip().lower() != expected.lower()]
    targets = {(t[0], t[1]) for t in qa.get("target_step_id") or [] if len(t) == 2}
    sessions = []
    for sidx, session in enumerate(traj["message_list"]):
        turns = [(_user_text(m), (m.get("mid"), sidx) in targets) for m in session]
        text = compose(turns, limit)
        if text:
            sessions.append({"id": f"session{sidx}",
                             "date": date_of(session[0].get("time", "")), "text": text})
    options = "; ".join(str(choices[k]) for k in sorted(choices))
    question = f"{qa['question'].strip()} Options: {options}. Answer with one option."
    case = localized({"source": "membench", "id": f"{agent.capitalize()}Agent/highlevel/"
                      f"{kind}/{traj['tid']}", "kind": f"refl-{kind}-{agent}",
                      "tid": traj["tid"], "memory": "reflective", "question": question,
                      "expected": expected, "distractors": distractors,
                      "answer": str(qa["answer"]), "sessions": sessions})
    return case if case["sessions"] else None


def select_membench(simple: dict, highlevel: dict[str, dict], n_factual: int = 20,
                    n_reflective: int = 20) -> list[dict]:
    """First scorable trajectories by tid, taken in turn: factual over FACTUAL_KINDS,
    reflective over the two agents and, within each agent, over its categories.

    MemBench builds tid k of a reflective category for both agents from one record: the
    same answer, and often the same items (book 1 is the 1918 flu book for both). So
    the agents take disjoint tids - the first agent (sorted by name) tid 0, 2, 4, ...,
    the second 1, 3, 5, ... - or the reflective half would hold each item about twice.
    """
    factual = interleave(*([c for t in sorted(simple.get(kind, []), key=lambda t: t["tid"])
                            if (c := membench_factual_case(kind, t)) is not None]
                           for kind in FACTUAL_KINDS))[:n_factual]
    per_agent = []
    for i, agent in enumerate(sorted(highlevel)):
        data = highlevel[agent]
        per_agent.append(interleave(*([c for t in sorted(data[k], key=lambda t: t["tid"])
                                       if t["tid"] % len(highlevel) == i
                                       and (c := membench_reflective_case(agent, k, t))
                                       is not None]
                                      for k in sorted(data))))
    reflective = interleave(*per_agent)[:n_reflective]
    return interleave(factual, reflective)


# --- writing --------------------------------------------------------------------

def case_texts(case: dict) -> list[str]:
    """Every string of a case that reaches a scenario: probe, options, sessions."""
    return [case["question"], case["expected"], *case.get("distractors", []),
            *(str(s[k]) for s in case["sessions"] for k in ("date", "text"))]


def file_mode(path: Path) -> int:
    """The mode `path` has, or a new file's under the current umask."""
    if path.exists():
        return path.stat().st_mode & 0o777
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def write_cases(path: Path, source: str, built_from: list[str], cases: list[dict]) -> None:
    """Refuse to write a real host (checked by `leaked_hosts`, not by the rewriter that
    produced the text), then replace the file in one step: every import of
    memheal.scenarios reads it, so a half-written file must never be visible."""
    leaked = sorted({h for c in cases for t in case_texts(c) for h in leaked_hosts(t)})
    if leaked:
        raise SystemExit(f"{path.name}: real hostnames survived localisation: {leaked[:5]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({"source": source, "built_from": built_from,
                       "session_chars": SESSION_CHARS, "cases": cases},
                      indent=1, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        os.chmod(tmp, file_mode(path))          # mkstemp's 0600 would hide it from others
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def summarize(name: str, cases: list[dict]) -> str:
    kinds: dict[str, int] = {}
    for c in cases:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    n_sessions = sum(len(c["sessions"]) for c in cases)
    chars = sum(len(s["text"]) for c in cases for s in c["sessions"])
    detail = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
    return (f"{name}: {len(cases)} cases ({detail}); {n_sessions} external sessions, "
            f"{chars:,} characters")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=DEFAULT_DIR,
                    help="directory holding the downloaded longmemeval/ and membench/")
    ap.add_argument("--out", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--n-update", type=int, default=30)
    ap.add_argument("--n-other", type=int, default=30)
    ap.add_argument("--n-factual", type=int, default=20)
    ap.add_argument("--n-reflective", type=int, default=20)
    args = ap.parse_args(argv)

    lme = select_longmemeval(load_json(args.src / LONGMEMEVAL_SRC), args.n_update, args.n_other)
    write_cases(args.out / "cases_longmemeval.json", "LongMemEval (oracle)",
                [str(LONGMEMEVAL_SRC)], lme)
    print(summarize("longmemeval", lme))

    mb = select_membench(load_json(args.src / MEMBENCH_SIMPLE),
                         {a: load_json(args.src / p) for a, p in MEMBENCH_HIGH.items()},
                         args.n_factual, args.n_reflective)
    write_cases(args.out / "cases_membench.json", "MemBench",
                [str(MEMBENCH_SIMPLE), *map(str, MEMBENCH_HIGH.values())], mb)
    print(summarize("membench", mb))


if __name__ == "__main__":
    main()
