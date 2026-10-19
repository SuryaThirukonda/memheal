"""Scenario registry.

ALL_VARIANTS holds the hand-built development cases (reimbursement, expense_migration)
plus the full MemHeal-Evolve catalog of 120 controlled scenarios (Section 7.1).
"""

from memheal.scenarios.base import Scenario
from memheal.scenarios.catalog import ATTACK_CATEGORIES, build
from memheal.scenarios.domains import ALL_CONTEXTS
from memheal.scenarios.expense_migration import VARIANTS as _EXPENSE
from memheal.scenarios.reimbursement import DEFAULT_VARIANT, VARIANTS as _REIMBURSEMENT

#: MemHeal-Evolve: 40 contexts x 3 attack categories.
EVOLVE: dict[str, Scenario] = build(ALL_CONTEXTS)

#: Everything addressable by `--variant`.
ALL_VARIANTS: dict[str, Scenario] = {**_REIMBURSEMENT, **_EXPENSE, **EVOLVE}

#: The cheap early-testing subset (Section 12.1's "dataset v0.1" smoke set).
SMOKE: dict[str, Scenario] = {k: v for k, v in EVOLVE.items() if v.tier == "smoke"}


def by_domain(domain: str) -> dict[str, Scenario]:
    return {k: v for k, v in EVOLVE.items() if v.domain == domain}


def by_category(category: str) -> dict[str, Scenario]:
    return {k: v for k, v in EVOLVE.items() if v.attack_category == category}


__all__ = ["ALL_VARIANTS", "ATTACK_CATEGORIES", "DEFAULT_VARIANT", "EVOLVE", "SMOKE",
           "Scenario", "by_category", "by_domain"]
