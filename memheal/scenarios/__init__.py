"""Scenario registry.

ALL_VARIANTS holds the hand-built development cases (reimbursement, expense_migration)
plus the full MemHeal-Evolve catalog of 120 controlled scenarios (Section 7.1).
"""

from memheal.scenarios.adaptive import ADAPTIVE_CATEGORIES, build_adaptive
from memheal.scenarios.base import Scenario
from memheal.scenarios.catalog import ATTACK_CATEGORIES, SUBSET_CONTEXTS, build
from memheal.scenarios.dense import densify
from memheal.scenarios.domains import ALL_CONTEXTS
from memheal.scenarios.expense_migration import VARIANTS as _EXPENSE
from memheal.scenarios.external import LONGMEMEVAL, MEMBENCH
from memheal.scenarios.reimbursement import DEFAULT_VARIANT, VARIANTS as _REIMBURSEMENT

#: MemHeal-Evolve: 40 contexts x 3 attack categories.
EVOLVE: dict[str, Scenario] = build(ALL_CONTEXTS)

#: MemHeal-Evolve-Dense: the same contexts with ~4x the history, so retrieval selects.
DENSE: dict[str, Scenario] = build(densify(c) for c in ALL_CONTEXTS)

#: Adaptive poison on the dense timelines (Section 12.3): new attack categories only.
DENSE_ADAPTIVE: dict[str, Scenario] = build_adaptive(densify(c) for c in ALL_CONTEXTS)

#: The representative subset for the stronger-model and cross-model runs (Section 8.11).
DENSE_SUBSET: dict[str, Scenario] = {
    k: v for k, v in DENSE.items()
    if k.removesuffix(f"_{v.attack_category}").removesuffix("_dense") in SUBSET_CONTEXTS}

#: Everything addressable by `--variant`.
ALL_VARIANTS: dict[str, Scenario] = {**_REIMBURSEMENT, **_EXPENSE, **EVOLVE, **DENSE,
                                     **DENSE_ADAPTIVE, **LONGMEMEVAL, **MEMBENCH}

#: Named suites beyond the evolve/dense slices that run_demo.resolve_suite parses.
SUITES: dict[str, dict[str, Scenario]] = {
    "dense-adaptive": DENSE_ADAPTIVE,
    "dense-subset": DENSE_SUBSET,
    "longmemeval": LONGMEMEVAL,
    "membench": MEMBENCH,
}

#: The cheap early-testing subset (Section 12.1's "dataset v0.1" smoke set).
SMOKE: dict[str, Scenario] = {k: v for k, v in EVOLVE.items() if v.tier == "smoke"}


#: Dense counterpart of SMOKE.
DENSE_SMOKE: dict[str, Scenario] = {k: v for k, v in DENSE.items() if v.tier == "smoke"}


def by_domain(domain: str, suite: dict[str, Scenario] = EVOLVE) -> dict[str, Scenario]:
    return {k: v for k, v in suite.items() if v.domain == domain}


def by_category(category: str, suite: dict[str, Scenario] = EVOLVE) -> dict[str, Scenario]:
    return {k: v for k, v in suite.items() if v.attack_category == category}


__all__ = ["ADAPTIVE_CATEGORIES", "ALL_VARIANTS", "ATTACK_CATEGORIES", "DEFAULT_VARIANT", "DENSE",
           "DENSE_ADAPTIVE", "DENSE_SMOKE", "DENSE_SUBSET", "EVOLVE", "LONGMEMEVAL", "MEMBENCH",
           "SMOKE", "SUITES",
           "Scenario", "by_category", "by_domain"]
