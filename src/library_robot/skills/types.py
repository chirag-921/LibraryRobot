"""Skill interface data types and failure codes.

No MuJoCo or simulation imports are allowed in this module so that
high-level planners and mock executors can run purely offline.
"""

from dataclasses import dataclass
from typing import Optional, Set

# Failure codes defined in project_phases_mobile_mujoco.md (Phase 10 / Section 7)
P = "P"  # Planning failure
N = "N"  # Navigation failure
X = "X"  # Extraction failure (hook slips, etc.)
G = "G"  # Grasp failure
L = "L"  # Placement failure
V = "V"  # Verification failure
S = "S"  # Simulation failure

# Descriptive aliases
FAILURE_PLANNING = P
FAILURE_NAVIGATION = N
FAILURE_EXTRACTION = X
FAILURE_GRASP = G
FAILURE_PLACE = L
FAILURE_VERIFICATION = V
FAILURE_SIMULATION = S

VALID_FAILURE_CODES: Set[str] = {P, N, X, G, L, V, S}


@dataclass
class SkillResult:
    """Standardized result returned by all robot skill primitives."""

    success: bool
    failure_code: Optional[str] = None
    message: str = ""
    duration_s: float = 0.0  # seconds; needs tuning

    def __post_init__(self) -> None:
        if not self.success and self.failure_code is None:
            # Default unassigned failure code to simulation/unknown error
            self.failure_code = S
        if self.failure_code is not None and self.failure_code not in VALID_FAILURE_CODES:
            raise ValueError(
                f"Invalid failure code '{self.failure_code}'. Must be one of {VALID_FAILURE_CODES}"
            )
