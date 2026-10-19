"""Domain content for MemHeal-Evolve (Section 7.1): 10 contexts each."""

from memheal.scenarios.domains.coding import CONTEXTS as CODING
from memheal.scenarios.domains.office import CONTEXTS as OFFICE
from memheal.scenarios.domains.research import CONTEXTS as RESEARCH
from memheal.scenarios.domains.travel import CONTEXTS as TRAVEL

ALL_CONTEXTS = (*OFFICE, *TRAVEL, *CODING, *RESEARCH)

__all__ = ["ALL_CONTEXTS", "OFFICE", "TRAVEL", "CODING", "RESEARCH"]
