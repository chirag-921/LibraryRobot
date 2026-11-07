"""Mock executor for testing planners offline against the skill interface.

Executes an action sequence step-by-step using skill stubs or mock overrides,
logs each step, and immediately stops if any step fails.
"""

from typing import Any, Callable, Dict, List, Optional
from .types import SkillResult, P, S
from .stubs import navigate, grasp, place, verify


class MockExecutor:
    """Executes a planned sequence of skill actions without requiring physics."""

    def __init__(
        self,
        skill_handlers: Optional[Dict[str, Callable[..., SkillResult]]] = None,
        fail_on_step: Optional[int] = None,
        injected_failure_result: Optional[SkillResult] = None,
    ) -> None:
        """
        Parameters
        ----------
        skill_handlers : dict, optional
            Mapping of skill names to callable implementations. Defaults to stubs.
        fail_on_step : int, optional
            0-based step index to inject an intentional failure for testing.
        injected_failure_result : SkillResult, optional
            The failure result to return when fail_on_step is hit.
        """
        self.handlers: Dict[str, Callable[..., SkillResult]] = {
            "navigate": navigate,
            "grasp": grasp,
            "place": place,
            "verify": verify,
        }
        if skill_handlers:
            self.handlers.update(skill_handlers)

        self.fail_on_step = fail_on_step
        self.injected_failure_result = injected_failure_result or SkillResult(
            success=False,
            failure_code=S,
            message="Injected simulated failure",
            duration_s=0.1,  # seconds; needs tuning
        )
        self.log: List[Dict[str, Any]] = []

    def execute_action(self, action: Any) -> SkillResult:
        """Dispatches an action represented as a dictionary, tuple, or list."""
        if isinstance(action, dict):
            skill_name = action.get("skill") or action.get("name") or action.get("action")
            if not skill_name:
                return SkillResult(
                    success=False,
                    failure_code=P,
                    message="Missing skill identifier in action dict",
                    duration_s=0.0,
                )
            # Support kwargs or positional arguments
            if "args" in action and isinstance(action["args"], (list, tuple)):
                args = action["args"]
                kwargs = action.get("kwargs", {})
            else:
                args = []
                kwargs = {k: v for k, v in action.items() if k not in ("skill", "name", "action")}
        elif isinstance(action, (list, tuple)):
            if not action:
                return SkillResult(
                    success=False,
                    failure_code=P,
                    message="Empty action tuple/list",
                    duration_s=0.0,
                )
            skill_name = str(action[0])
            args = action[1:]
            kwargs = {}
        else:
            return SkillResult(
                success=False,
                failure_code=P,
                message=f"Unsupported action format: {type(action)}",
                duration_s=0.0,
            )

        handler = self.handlers.get(skill_name.lower())
        if handler is None:
            return SkillResult(
                success=False,
                failure_code=P,
                message=f"Unknown skill '{skill_name}'",
                duration_s=0.0,
            )

        try:
            return handler(*args, **kwargs)
        except Exception as exc:
            return SkillResult(
                success=False,
                failure_code=S,
                message=f"Exception raised executing skill '{skill_name}': {exc}",
                duration_s=0.0,
            )

    def run(self, actions: List[Any]) -> Dict[str, Any]:
        """Runs the entire action list sequentially.
        
        Halts immediately on the first step that reports success=False.
        """
        self.log.clear()
        total_duration_s = 0.0

        for step_idx, action in enumerate(actions):
            if self.fail_on_step is not None and step_idx == self.fail_on_step:
                result = self.injected_failure_result
            else:
                result = self.execute_action(action)

            total_duration_s += result.duration_s
            entry = {
                "step": step_idx,
                "action": action,
                "result": result,
            }
            self.log.append(entry)

            if not result.success:
                return {
                    "success": False,
                    "failed_step": step_idx,
                    "total_duration_s": total_duration_s,
                    "steps_executed": step_idx + 1,
                    "total_steps": len(actions),
                    "log": self.log,
                }

        return {
            "success": True,
            "failed_step": None,
            "total_duration_s": total_duration_s,
            "steps_executed": len(actions),
            "total_steps": len(actions),
            "log": self.log,
        }
