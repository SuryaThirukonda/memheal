"""Scenario registry. Import ALL_VARIANTS to reach every case by name."""

from memheal.scenarios.base import Scenario
from memheal.scenarios.expense_migration import VARIANTS as _EXPENSE
from memheal.scenarios.reimbursement import DEFAULT_VARIANT, VARIANTS as _REIMBURSEMENT

ALL_VARIANTS: dict[str, Scenario] = {**_REIMBURSEMENT, **_EXPENSE}

__all__ = ["ALL_VARIANTS", "DEFAULT_VARIANT", "Scenario"]
