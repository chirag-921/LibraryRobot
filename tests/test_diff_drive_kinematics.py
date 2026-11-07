"""Offline unit tests for differential-drive kinematics, acceleration limiting, and odometry.

These tests run completely offline and MUST NOT import MuJoCo.
"""

import math
import subprocess
import sys
import pytest

from library_robot.control.diff_drive import (
    wrap_angle,
    twist_to_wheel_speeds,
    wheel_speeds_to_twist,
    DiffDriveKinematics,
    check_arm_wheels_mutual_exclusion,
)
from library_robot.control.odometry import (
    DiffDriveOdometry,
    compare_odometry_to_ground_truth,
)


def test_no_mujoco_imported():
    """Verify that diff_drive and odometry modules do not import MuJoCo."""
    code = (
        "import sys; "
        "from library_robot.control.diff_drive import twist_to_wheel_speeds; "
        "from library_robot.control.odometry import DiffDriveOdometry; "
        "assert 'mujoco' not in sys.modules, 'control.diff_drive and odometry must not import MuJoCo'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Subprocess failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"


def test_wrap_angle():
    """Verify angle normalization to [-pi, pi]."""
    assert math.isclose(wrap_angle(0.0), 0.0)
    assert math.isclose(wrap_angle(math.pi), math.pi, abs_tol=1e-6)
    assert math.isclose(wrap_angle(3.0 * math.pi), math.pi, abs_tol=1e-6)
    assert math.isclose(wrap_angle(-3.0 * math.pi), -math.pi, abs_tol=1e-6)
    assert math.isclose(wrap_angle(math.pi / 2.0), math.pi / 2.0, abs_tol=1e-6)


def test_kinematics_pure_forward():
    """Verify pure forward motion produces equal positive wheel speeds."""
    r = 0.10
    L = 0.62
    v = 0.50
    w = 0.0

    w_l, w_r = twist_to_wheel_speeds(v, w, wheel_radius=r, wheel_separation=L)
    expected_speed = v / r  # 5.0 rad/s
    assert math.isclose(w_l, expected_speed)
    assert math.isclose(w_r, expected_speed)

    # Invert back to twist
    v_rec, w_rec = wheel_speeds_to_twist(w_l, w_r, wheel_radius=r, wheel_separation=L)
    assert math.isclose(v_rec, v)
    assert math.isclose(w_rec, w, abs_tol=1e-6)


def test_kinematics_pure_turn():
    """Verify pure CCW turn produces opposite wheel speeds."""
    r = 0.10
    L = 0.62
    v = 0.0
    w = 1.0

    w_l, w_r = twist_to_wheel_speeds(v, w, wheel_radius=r, wheel_separation=L)
    expected_differential = (w * L / 2.0) / r  # 3.1 rad/s
    assert math.isclose(w_l, -expected_differential)
    assert math.isclose(w_r, +expected_differential)

    v_rec, w_rec = wheel_speeds_to_twist(w_l, w_r, wheel_radius=r, wheel_separation=L)
    assert math.isclose(v_rec, v, abs_tol=1e-6)
    assert math.isclose(w_rec, w)


def test_kinematics_roundtrip():
    """Verify twist <-> wheel speed roundtrip across various inputs."""
    r = 0.10
    L = 0.62

    test_cases = [
        (0.2, 0.4),
        (-0.3, -0.5),
        (0.6, -0.8),
        (0.0, 0.0),
    ]
    for v_in, w_in in test_cases:
        wl, wr = twist_to_wheel_speeds(v_in, w_in, r, L)
        v_out, w_out = wheel_speeds_to_twist(wl, wr, r, L)
        assert math.isclose(v_in, v_out, abs_tol=1e-6)
        assert math.isclose(w_in, w_out, abs_tol=1e-6)


def test_invalid_kinematics_args():
    """Verify invalid geometric dimensions raise ValueError."""
    with pytest.raises(ValueError):
        twist_to_wheel_speeds(0.5, 0.0, wheel_radius=-0.1, wheel_separation=0.6)
    with pytest.raises(ValueError):
        twist_to_wheel_speeds(0.5, 0.0, wheel_radius=0.1, wheel_separation=0.0)


def test_acceleration_slew_limiting():
    """Verify acceleration limiter caps rate of velocity change."""
    dt = 0.01  # 10 ms
    max_a = 0.80  # m/s^2 -> max dv = 0.008 m/s per step
    max_alpha = 1.50  # rad/s^2 -> max dw = 0.015 rad/s per step

    kin = DiffDriveKinematics(
        max_linear_accel=max_a,
        max_angular_accel=max_alpha,
        max_linear_velocity=0.60,
        max_angular_velocity=1.20,
    )

    # Step command jump from 0 to 0.5 m/s
    v1, w1 = kin.limit_twist(target_v=0.50, target_omega=0.0, dt=dt)
    assert math.isclose(v1, max_a * dt, abs_tol=1e-6)
    assert math.isclose(w1, 0.0, abs_tol=1e-6)

    # Angular command jump from 0 to 1.0 rad/s
    v2, w2 = kin.limit_twist(target_v=0.50, target_omega=1.0, dt=dt)
    assert math.isclose(v2, 2.0 * max_a * dt, abs_tol=1e-6)
    assert math.isclose(w2, max_alpha * dt, abs_tol=1e-6)


def test_odometry_arc_integration():
    """Verify dead-reckoned odometry tracks straight and turning motion."""
    r = 0.10
    L = 0.62
    odom = DiffDriveOdometry(wheel_radius=r, wheel_separation=L)

    # Drive forward at 5 rad/s for 1 second (v = 0.5 m/s -> dist = 0.5 m)
    dt = 0.002
    steps = 500
    for _ in range(steps):
        odom.update(wheel_left_vel=5.0, wheel_right_vel=5.0, dt=dt)

    x, y, yaw = odom.get_pose()
    assert math.isclose(x, 0.50, abs_tol=1e-3)
    assert math.isclose(y, 0.0, abs_tol=1e-6)
    assert math.isclose(yaw, 0.0, abs_tol=1e-6)
    assert math.isclose(odom.distance_travelled, 0.50, abs_tol=1e-3)


def test_odometry_slip_calculation():
    """Verify slip is computed as odometry distance minus ground-truth distance."""
    odom_pose = (1.05, 0.02, 0.01)
    gt_pose = (1.00, 0.00, 0.00)
    s_odom = 1.08
    s_gt = 1.00

    metrics = compare_odometry_to_ground_truth(odom_pose, gt_pose, s_odom, s_gt)
    assert math.isclose(metrics["slip_distance_m"], 0.08, abs_tol=1e-6)
    assert math.isclose(metrics["slip_fraction"], 0.08 / 1.08, abs_tol=1e-6)
    assert math.isclose(metrics["position_error_m"], math.hypot(0.05, 0.02), abs_tol=1e-6)
    assert math.isclose(metrics["heading_error_rad"], 0.01, abs_tol=1e-6)


def test_mutual_exclusion_rule():
    """Verify mutual exclusion between wheel drive and arm motion."""
    # 1. Wheels active, arm stationary -> OK
    assert check_arm_wheels_mutual_exclusion(
        wheel_commands=(2.0, 2.0),
        arm_velocities={"joint1": 0.0, "joint2": 0.005},
        arm_vel_threshold=0.02,
    ) is True

    # 2. Wheels stopped, arm moving -> OK
    assert check_arm_wheels_mutual_exclusion(
        wheel_commands=(0.0, 0.0),
        arm_velocities={"joint1": 0.15, "joint2": 0.0},
        arm_vel_threshold=0.02,
    ) is True

    # 3. Both active -> VIOLATION
    assert check_arm_wheels_mutual_exclusion(
        wheel_commands=(2.0, 2.0),
        arm_velocities={"joint1": 0.15, "joint2": 0.0},
        arm_vel_threshold=0.02,
    ) is False
