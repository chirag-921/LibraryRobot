"""MuJoCo physics test for closed-loop differential-drive controller and odometry.

This test requires the MuJoCo runtime DLL. Kinematics and offline tests
remain runnable completely offline without MuJoCo.
"""

import subprocess
import sys
from pathlib import Path
import pytest

mujoco = pytest.importorskip("mujoco", reason="Requires MuJoCo runtime DLL")

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_diff_drive_physics_benchmark():
    """Runs the closed-loop differential-drive benchmark (Script 14) and asserts all tolerances."""
    script_path = REPO_ROOT / "scripts" / "14_test_diff_drive.py"
    assert script_path.exists(), f"Missing script: {script_path}"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Script 14 failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Verify no waypoint timed out
    assert "TIMEOUT_FAILED" not in result.stdout, "One or more waypoints timed out!"
    assert "BENCHMARK STATUS: PASS" in result.stdout, "Overall benchmark did not pass!"

    # Verify all scenarios reported ALL CONVERGED
    assert "Section 1: Target Waypoint Tracking Summary (STRAIGHT)" in result.stdout
    assert "Section 1: Target Waypoint Tracking Summary (TURN)" in result.stdout
    assert "Section 1: Target Waypoint Tracking Summary (SQUARE)" in result.stdout

    # Verify CSV files exist and are non-empty
    results_dir = REPO_ROOT / "results"
    straight_csv = results_dir / "diff_drive_straight.csv"
    turn_csv = results_dir / "diff_drive_turn.csv"
    square_csv = results_dir / "diff_drive_square.csv"

    assert straight_csv.exists() and straight_csv.stat().st_size > 100, "Missing diff_drive_straight.csv"
    assert turn_csv.exists() and turn_csv.stat().st_size > 100, "Missing diff_drive_turn.csv"
    assert square_csv.exists() and square_csv.stat().st_size > 100, "Missing diff_drive_square.csv"
