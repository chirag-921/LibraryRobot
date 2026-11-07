"""Skill primitive stubs for robot actions.

Exposes the standardized high-level skill interface:
- navigate(target)
- grasp(obj)
- place(obj, shelf)
- verify(expected)

These stubs return successful SkillResult instances and do not import MuJoCo.
"""

from typing import Any, Optional
from .types import SkillResult, P, N, G, L, V, S


# Provisional simulated durations for mock/stub actions
DEFAULT_NAVIGATE_DURATION_S = 3.5  # seconds; needs tuning
DEFAULT_GRASP_DURATION_S = 2.0     # seconds; needs tuning
DEFAULT_PLACE_DURATION_S = 2.5     # seconds; needs tuning
DEFAULT_VERIFY_DURATION_S = 0.5    # seconds; needs tuning


def navigate(target: Any, duration_s: float = DEFAULT_NAVIGATE_DURATION_S) -> SkillResult:
    """Stub for navigation skill to move robot to a table or shelf parking pose."""
    target_str = str(target)
    if not target_str:
        return SkillResult(
            success=False,
            failure_code=N,
            message="Navigation failed: empty target specified",
            duration_s=0.0,
        )
    return SkillResult(
        success=True,
        failure_code=None,
        message=f"Successfully navigated to target '{target_str}' (stub)",
        duration_s=duration_s,
    )


def grasp(obj: Any, duration_s: float = DEFAULT_GRASP_DURATION_S) -> SkillResult:
    """Stub for grasping an object (e.g., book on table or shelf)."""
    obj_str = str(obj)
    if not obj_str:
        return SkillResult(
            success=False,
            failure_code=G,
            message="Grasp failed: empty object specified",
            duration_s=0.0,
        )
    return SkillResult(
        success=True,
        failure_code=None,
        message=f"Successfully grasped object '{obj_str}' (stub)",
        duration_s=duration_s,
    )


def place(obj: Any, shelf: Any, duration_s: float = DEFAULT_PLACE_DURATION_S) -> SkillResult:
    """Stub for placing an object into a target shelf compartment."""
    obj_str = str(obj)
    shelf_str = str(shelf)
    if not obj_str or not shelf_str:
        return SkillResult(
            success=False,
            failure_code=L,
            message="Place failed: invalid object or target shelf specified",
            duration_s=0.0,
        )
    return SkillResult(
        success=True,
        failure_code=None,
        message=f"Successfully placed object '{obj_str}' into shelf '{shelf_str}' (stub)",
        duration_s=duration_s,
    )


def verify(expected: Optional[Any] = None, duration_s: float = DEFAULT_VERIFY_DURATION_S) -> SkillResult:
    """Stub for verification check (e.g. checking book placement)."""
    expected_desc = str(expected) if expected is not None else "state"
    return SkillResult(
        success=True,
        failure_code=None,
        message=f"Verification confirmed for {expected_desc} (stub)",
        duration_s=duration_s,
    )
