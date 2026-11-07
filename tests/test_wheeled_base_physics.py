"""MuJoCo physics test for the wheeled mobile base validation.

This test requires the MuJoCo runtime DLL. XML-generation and skill tests
remain runnable completely offline without MuJoCo.
"""

import subprocess
import sys
from pathlib import Path
import pytest

mujoco = pytest.importorskip("mujoco", reason="Requires MuJoCo runtime DLL")

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_wheeled_base_physics_validation():
    """Runs the full headless wheeled base validation suite (Script 13)."""
    script_path = REPO_ROOT / "scripts" / "13_validate_wheeled_base.py"
    assert script_path.exists(), f"Missing script: {script_path}"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Validation script failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "[PASS] Motor Direction Test" in result.stdout
    assert "[PASS] Straight Drive Test" in result.stdout
    assert "[PASS] In-Place Turn Test" in result.stdout
    assert "[PASS] Tip & Stability Test" in result.stdout

    # Verify result files were generated
    json_path = REPO_ROOT / "results" / "wheeled_base_validation.json"
    csv_path = REPO_ROOT / "results" / "wheeled_base_validation.csv"
    log_path = REPO_ROOT / "logs" / "wheeled_base_validation.txt"

    assert json_path.exists(), "Missing JSON results file"
    assert csv_path.exists(), "Missing CSV results file"
    assert log_path.exists(), "Missing text summary log file"
