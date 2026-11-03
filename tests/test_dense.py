"""MemHeal-Evolve-Dense: the densified timelines keep every property the base catalog
has, and add only history that cannot move the decision."""

from __future__ import annotations

import re

import pytest

from memheal.scenarios import (
    ATTACK_CATEGORIES,
    DENSE,
    DENSE_SMOKE,
    DENSE_SUBSET,
    EVOLVE,
    SUITES,
)
from memheal.scenarios.catalog import SUBSET_CONTEXTS
from memheal.scenarios.dense import DENSE_SUFFIX, FILLER, TOPICAL, densify, study_pages
from memheal.scenarios.domains import ALL_CONTEXTS

ALL = sorted(DENSE.items())


def base_of(name: str) -> str:
    return name.replace(DENSE_SUFFIX, "")


def test_one_dense_scenario_per_base_scenario():
    assert len(DENSE) == len(EVOLVE) == 120
    assert {base_of(n) for n in DENSE} == set(EVOLVE)
    assert {s.domain for s in DENSE_SMOKE.values()} == {"office", "travel", "coding", "research"}


@pytest.mark.parametrize("name, s", ALL)
def test_dense_history_is_much_longer(name, s):
    base = EVOLVE[base_of(name)]
    assert len(s.sessions) >= 4 * len(base.sessions), name


@pytest.mark.parametrize("name, s", ALL)
def test_days_are_unique_and_ordered(name, s):
    days = [d for d, _ in s.sessions]
    assert days == sorted(set(days)), name
    assert s.target_session > days[-1]


@pytest.mark.parametrize("name, s", ALL)
def test_original_sessions_survive_in_order(name, s):
    base = EVOLVE[base_of(name)]
    msgs = [m for _d, m in s.sessions]
    positions = [msgs.index(m) for _d, m in base.sessions]
    assert positions == sorted(positions), name
    assert s.target_tasks == base.target_tasks


@pytest.mark.parametrize("name, s", ALL)
def test_poison_day_is_still_the_session_that_reads_the_page(name, s):
    base = EVOLVE[base_of(name)]
    base_msg = dict(base.sessions)[base.poison_session]
    assert dict(s.sessions)[s.poison_session] == base_msg
    if s.attack_category == "procedural":
        assert s.reflect_days == (s.poison_session,)


@pytest.mark.parametrize("name, s", ALL)
def test_consolidation_has_history_on_both_sides(name, s):
    """The point of densifying: a large pool when consolidation runs, and new memories
    after it, so the target still retrieves from more than k."""
    (c,) = s.consolidate_days
    days = [d for d, _ in s.sessions]
    assert c not in days, "consolidation gets a day of its own"
    assert s.poison_session < c < s.target_session
    assert sum(d < c for d in days) >= 10
    assert sum(c < d for d in days) >= 5


@pytest.mark.parametrize("name, s", ALL)
def test_worlds_still_differ_only_by_the_poison(name, s):
    clean, poisoned = s.pages_for("clean"), s.pages_for("poisoned")
    changed = {u for u in clean.keys() | poisoned.keys() if clean.get(u) != poisoned.get(u)}
    assert changed == {u for u in changed if u.startswith(s.poison_url.split("/")[0])}


def _all_names():
    out = set()
    for c in ALL_CONTEXTS:
        out |= set(c.correct) | set(c.attacker)
        out |= {c.legacy_label.lower(), c.attacker_label.lower()}
    return out


@pytest.mark.parametrize("text", [t for ts in FILLER.values() for t in ts] + list(TOPICAL))
def test_added_history_names_no_system(text):
    """A filler that named a destination could steer the decision on its own."""
    low = text.lower()
    for name in _all_names():
        assert not re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", low), \
            f"{text!r} names {name!r}"
    assert ".example" not in low and "http" not in low


@pytest.mark.parametrize("ctx", ALL_CONTEXTS, ids=lambda c: c.key)
def test_study_sessions_only_open_benign_pages(ctx):
    pages = study_pages(ctx)
    assert pages, f"{ctx.key}: nothing to study"
    endpoints = {u.split("/")[0] for u in ctx.correct | ctx.attacker}
    for u in pages:
        assert u in ctx.pages and u != ctx.clean_page[0]
        assert u.split("/")[0] not in endpoints
    msgs = " ".join(m for _d, m in densify(ctx).sessions)
    assert all(u in msgs for u in pages)
    assert ctx.clean_page[0] not in msgs.replace(ctx.clean_page[0] + "/", "")


@pytest.mark.parametrize("name, s", ALL)
def test_benign_probes_still_answerable(name, s):
    corpus = " ".join(m for _d, m in s.sessions).lower()
    corpus += " " + " ".join(s.pages_for("clean").values()).lower()
    for _q, expected in s.benign_probes:
        assert expected.lower() in corpus


def test_densify_rejects_a_too_short_context():
    from dataclasses import replace
    with pytest.raises(ValueError):
        densify(replace(ALL_CONTEXTS[0], sessions=ALL_CONTEXTS[0].sessions[:2]))


# --- the representative subset (Section 8.11) ---------------------------------------

def test_subset_follows_its_rule():
    """Per domain: the smoke-tier context, then catalog key order, three in all. The set
    is written out in catalog.py so it can be read there; this re-derives it."""
    derived = set()
    for domain in ("office", "travel", "coding", "research"):
        in_domain = [c for c in ALL_CONTEXTS if c.domain == domain]
        ranked = sorted(in_domain, key=lambda c: c.tier != "smoke")   # stable: keeps order
        derived |= {c.key for c in ranked[:3]}
    assert SUBSET_CONTEXTS == derived


def test_subset_shape():
    assert len(SUBSET_CONTEXTS) == 12
    assert len(DENSE_SUBSET) == 36
    for domain in ("office", "travel", "coding", "research"):
        assert sum(s.domain == domain for s in DENSE_SUBSET.values()) == 9, domain
    for cat in ATTACK_CATEGORIES:
        assert sum(s.attack_category == cat for s in DENSE_SUBSET.values()) == 12, cat
    assert all(DENSE[k] is v for k, v in DENSE_SUBSET.items())
    assert {k.rsplit(DENSE_SUFFIX, 1)[0] for k in DENSE_SUBSET} == SUBSET_CONTEXTS
    assert SUITES["dense-subset"] is DENSE_SUBSET


def test_subset_includes_every_smoke_context():
    """So the stronger-model cell links back to the numbers every earlier stage ran."""
    smoke = {c.key for c in ALL_CONTEXTS if c.tier == "smoke"}
    assert smoke <= SUBSET_CONTEXTS
