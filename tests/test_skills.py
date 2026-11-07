"""Offline pytest tests for the skill interface, stubs, and MockExecutor."""

import subprocess
import sys
import pytest

from library_robot.skills import (
    SkillResult,
    P,
    N,
    X,
    G,
    L,
    V,
    S,
    VALID_FAILURE_CODES,
    navigate,
    grasp,
    place,
    verify,
    MockExecutor,
)


def test_no_mujoco_imported():
    """Verify that importing library_robot.skills does not import MuJoCo."""
    code = (
        "import sys, library_robot.skills; "
        "assert 'mujoco' not in sys.modules, 'skills must not import MuJoCo'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Subprocess failed: {result.stderr}"


def test_failure_code_constants():
    """Verify expected failure code constants match the plan specification."""
    expected = {"P", "N", "X", "G", "L", "V", "S"}
    assert VALID_FAILURE_CODES == expected
    assert P == "P"
    assert N == "N"
    assert X == "X"
    assert G == "G"
    assert L == "L"
    assert V == "V"
    assert S == "S"


def test_skill_result_valid():
    """Verify SkillResult construction and fields."""
    res_success = SkillResult(success=True, message="Completed", duration_s=1.2)
    assert res_success.success is True
    assert res_success.failure_code is None
    assert res_success.message == "Completed"
    assert res_success.duration_s == 1.2

    # Failed result without explicit code defaults to S
    res_failed_default = SkillResult(success=False, message="Sim error")
    assert res_failed_default.success is False
    assert res_failed_default.failure_code == S

    # Failed result with explicit code
    res_nav_fail = SkillResult(success=False, failure_code=N, message="Path blocked")
    assert res_nav_fail.failure_code == N

    # Invalid code raises ValueError
    with pytest.raises(ValueError):
        SkillResult(success=False, failure_code="UNKNOWN_CODE")


def test_skill_stubs():
    """Verify standard stubs return SkillResults with valid fields."""
    nav_res = navigate("Table_A")
    assert nav_res.success is True
    assert "Table_A" in nav_res.message
    assert nav_res.duration_s > 0

    # Empty target should fail with N
    nav_fail = navigate("")
    assert nav_fail.success is False
    assert nav_fail.failure_code == N

    grasp_res = grasp("Book_1")
    assert grasp_res.success is True
    assert "Book_1" in grasp_res.message

    grasp_fail = grasp("")
    assert grasp_fail.success is False
    assert grasp_fail.failure_code == G

    place_res = place("Book_1", "A3")
    assert place_res.success is True
    assert "A3" in place_res.message

    place_fail = place("Book_1", "")
    assert place_fail.success is False
    assert place_fail.failure_code == L

    verify_res = verify({"book": "Book_1", "shelf": "A3"})
    assert verify_res.success is True
    assert verify_res.failure_code is None


def test_mock_executor_all_success():
    """Verify MockExecutor runs a multi-step plan to completion."""
    executor = MockExecutor()
    actions = [
        {"skill": "navigate", "target": "Table_A"},
        {"skill": "grasp", "obj": "Book_Math"},
        {"skill": "navigate", "target": "Shelf_Compartment_A3"},
        {"skill": "place", "obj": "Book_Math", "shelf": "A3"},
        {"skill": "verify", "expected": {"obj": "Book_Math", "shelf": "A3"}},
    ]

    summary = executor.run(actions)
    assert summary["success"] is True
    assert summary["failed_step"] is None
    assert summary["steps_executed"] == 5
    assert len(summary["log"]) == 5
    assert summary["total_duration_s"] > 0

    # Test tuple-based actions also execute properly
    tuple_actions = [
        ("navigate", "Table_B"),
        ("grasp", "Book_Physics"),
    ]
    summary_tuple = executor.run(tuple_actions)
    assert summary_tuple["success"] is True
    assert summary_tuple["steps_executed"] == 2


def test_mock_executor_stops_on_failing_step():
    """Verify that MockExecutor halts immediately when an action fails."""
    # Action 1 fails (step index 1)
    injected_failure = SkillResult(
        success=False,
        failure_code=G,
        message="Grasp failed: object dropped",
        duration_s=0.5,
    )
    executor = MockExecutor(fail_on_step=1, injected_failure_result=injected_failure)

    actions = [
        {"skill": "navigate", "target": "Table_A"},
        {"skill": "grasp", "obj": "Book_Math"},
        {"skill": "navigate", "target": "Shelf_A"},
        {"skill": "place", "obj": "Book_Math", "shelf": "A3"},
    ]

    summary = executor.run(actions)
    assert summary["success"] is False
    assert summary["failed_step"] == 1
    assert summary["steps_executed"] == 2  # Only step 0 and 1 were executed
    assert len(summary["log"]) == 2
    assert summary["log"][1]["result"].failure_code == G

    # Step 2 and Step 3 must NOT have been executed
    executed_skills = [item["action"]["skill"] for item in summary["log"]]
    assert executed_skills == ["navigate", "grasp"]
    assert "place" not in executed_skills


def test_mock_executor_unknown_skill():
    """Verify MockExecutor reports planning failure (P) on unknown skill."""
    executor = MockExecutor()
    summary = executor.run([{"skill": "fly_to_moon"}])
    assert summary["success"] is False
    assert summary["failed_step"] == 0
    assert summary["log"][0]["result"].failure_code == P
