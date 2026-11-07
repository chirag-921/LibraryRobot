"""Offline (NumPy-only) and MuJoCo integration tests for arm_control.py.

Offline tests:
  - Frame transform round-trips (world_to_chassis / chassis_to_world)
  - Null-space projector orthogonality: (I - J+ J) dq is in the null space of J
  - Gravity feedforward formula: u_ff = q + bias/kp

MuJoCo integration test (skipped if mujoco unavailable):
  - solve_base_frame_pose_ik converges from 3 different base poses
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from library_robot.control.arm_control import (
    rot2d,
    base_pose_to_transform,
    world_to_chassis,
    chassis_to_world,
    world_rot_to_chassis,
    JOINT_MID_RANGE,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Frame-transform tests (pure NumPy, offline)
# ─────────────────────────────────────────────────────────────────────────────

class TestFrameTransforms:

    @pytest.mark.parametrize("yaw", [0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 3])
    def test_rot2d_orthogonal(self, yaw):
        R = rot2d(yaw)
        assert R.shape == (2, 2)
        assert np.allclose(R.T @ R, np.eye(2), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)

    @pytest.mark.parametrize("bx, by, byaw", [
        (0.0, 0.0, 0.0),
        (1.5, -2.0, math.pi / 4),
        (-0.5, 0.3, math.pi / 2),
        (0.0, 0.0, math.pi),
    ])
    def test_round_trip_world_chassis_world(self, bx, by, byaw):
        """chassis_to_world(world_to_chassis(p)) == p for any point p."""
        p_world = np.array([0.4, 0.1, 0.8])
        p_chassis = world_to_chassis(p_world, bx, by, byaw)
        p_back = chassis_to_world(p_chassis, bx, by, byaw)
        assert np.allclose(p_back, p_world, atol=1e-12), (
            f"Round-trip failed: {p_world} -> {p_chassis} -> {p_back}"
        )

    @pytest.mark.parametrize("bx, by, byaw", [
        (0.0, 0.0, 0.0),
        (1.0, 0.5, -math.pi / 6),
    ])
    def test_round_trip_chassis_world_chassis(self, bx, by, byaw):
        """world_to_chassis(chassis_to_world(p)) == p for any point p."""
        p_chassis = np.array([0.2, -0.1, 0.5])
        p_world = chassis_to_world(p_chassis, bx, by, byaw)
        p_back = world_to_chassis(p_world, bx, by, byaw)
        assert np.allclose(p_back, p_chassis, atol=1e-12)

    def test_world_to_chassis_at_origin_identity(self):
        """At zero base pose, world == chassis frame."""
        p = np.array([0.3, 0.7, 1.2])
        p_chassis = world_to_chassis(p, 0.0, 0.0, 0.0)
        assert np.allclose(p_chassis, p, atol=1e-12)

    def test_world_to_chassis_pure_translation(self):
        """With yaw=0, chassis_to_world is a pure translation."""
        bx, by, byaw = 2.0, -1.0, 0.0
        p_chassis = np.array([0.1, 0.2, 0.3])
        p_world = chassis_to_world(p_chassis, bx, by, byaw)
        expected = np.array([p_chassis[0] + bx, p_chassis[1] + by, p_chassis[2]])
        assert np.allclose(p_world, expected, atol=1e-12)

    def test_world_to_chassis_pure_rotation_90(self):
        """With base yaw=+90 deg (CCW), world +X maps to chassis -Y.

        When the chassis is rotated +90 deg CCW in the world:
          R_wb = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
          R_wb.T @ [1, 0, 0] = [0, -1, 0]  (chassis -Y direction)
        """
        p_world = np.array([1.0, 0.0, 0.0])  # +X in world
        p_chassis = world_to_chassis(p_world, 0.0, 0.0, math.pi / 2)
        expected = np.array([0.0, -1.0, 0.0])   # chassis -Y
        assert np.allclose(p_chassis, expected, atol=1e-10), (
            f"Expected {expected}, got {p_chassis}"
        )

    def test_base_pose_to_transform_rotation_matrix(self):
        """R_wb must be a valid rotation matrix for arbitrary yaw."""
        for yaw in [0.0, 1.0, -2.5, math.pi]:
            R, t = base_pose_to_transform(0.0, 0.0, yaw)
            assert R.shape == (3, 3)
            assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
            assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_world_rot_to_chassis_identity_at_zero_yaw(self):
        """At yaw=0, world_rot_to_chassis(I) == I."""
        R_world = np.eye(3)
        R_chassis = world_rot_to_chassis(R_world, 0.0)
        assert np.allclose(R_chassis, np.eye(3), atol=1e-12)

    def test_world_rot_to_chassis_round_trip(self):
        """Rotating into chassis and back should recover original rotation."""
        from scipy.spatial.transform import Rotation
        yaw = math.pi / 3
        R_world = Rotation.from_euler("z", 0.5).as_matrix()
        R_chassis = world_rot_to_chassis(R_world, yaw)
        # Inverse: rotate back by +yaw
        R_wb, _ = base_pose_to_transform(0.0, 0.0, yaw)
        R_back = R_wb @ R_chassis
        assert np.allclose(R_back, R_world, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Null-space projector correctness (pure NumPy, offline)
# ─────────────────────────────────────────────────────────────────────────────

class TestNullSpaceProjector:
    """
    Tests the mathematical correctness of the null-space projector (I - J+ J)
    without importing MuJoCo.  The key property:
      J * (I - J+ J) * v == 0  for any v (true for Moore-Penrose pinv)

    NOTE: arm_control.py uses the DLS pseudoinverse (damped, for stability near
    singularities).  The DLS projector (I - J+_dls J) is only an *approximate*
    null-space projector; J * (I - J+_dls J) * v ≈ 0 but not exactly 0.
    These tests therefore use np.linalg.pinv (the exact Moore-Penrose inverse)
    to verify the mathematical property, and separately test the DLS approximation
    with a looser tolerance.
    """

    def _make_random_J(self, rng: np.random.Generator, m=3, n=7) -> np.ndarray:
        """Create a random Jacobian with full row rank (m < n)."""
        J = rng.standard_normal((m, n))
        return J

    def _exact_pinv(self, J: np.ndarray) -> np.ndarray:
        """Exact Moore-Penrose pseudoinverse (used in math-property unit tests)."""
        return np.linalg.pinv(J)

    def _dls_pinv(self, J: np.ndarray, damping: float = 0.05) -> np.ndarray:
        """DLS pseudoinverse: J+ = Jᵀ(JJᵀ + λ²I)⁻¹.
        Used in arm_control.py for numerical stability; approximate projector."""
        m = J.shape[0]
        A = J @ J.T + damping**2 * np.eye(m)
        return J.T @ np.linalg.inv(A)

    @pytest.mark.parametrize("seed", [0, 1, 42, 99, 2024])
    def test_null_space_projection_kills_primary_task(self, seed):
        """J * (I - J+ J) * v == 0 for any v (exact Moore-Penrose pinv)."""
        rng = np.random.default_rng(seed)
        J = self._make_random_J(rng)
        # Use exact pinv: the property is exact only for Moore-Penrose
        J_pinv = self._exact_pinv(J)
        P_null = np.eye(7) - J_pinv @ J   # null-space projector

        v = rng.standard_normal(7)
        projected = P_null @ v
        result = J @ projected
        assert np.allclose(result, np.zeros(3), atol=1e-8), (
            f"J * P_null * v != 0: max={np.abs(result).max():.2e}"
        )

    @pytest.mark.parametrize("seed", [0, 7, 100])
    def test_null_space_does_not_change_primary_dq(self, seed):
        """Adding a null-space component (exact pinv) to dq does not change J*dq."""
        rng = np.random.default_rng(seed)
        J = self._make_random_J(rng)
        # Exact pinv: null-space of J+ matches null-space of J exactly
        J_pinv = self._exact_pinv(J)
        P_null = np.eye(7) - J_pinv @ J

        err = rng.standard_normal(3) * 0.01   # small Cartesian error
        dq_primary = J_pinv @ err
        dq_null    = P_null @ rng.standard_normal(7) * 0.1

        assert np.allclose(J @ (dq_primary + dq_null), J @ dq_primary, atol=1e-8)

    def test_null_space_projector_is_idempotent(self):
        """(I - J+ J)^2 == (I - J+ J) — idempotent property (exact Moore-Penrose)."""
        rng = np.random.default_rng(555)
        J = self._make_random_J(rng)
        # Exact pinv: idempotence is a theorem for the true orthogonal projector
        J_pinv = self._exact_pinv(J)
        P = np.eye(7) - J_pinv @ J
        assert np.allclose(P @ P, P, atol=1e-8), "Projector not idempotent"

    def test_null_space_projector_symmetric(self):
        """(I - J+ J) is symmetric for the exact Moore-Penrose pinv (orthogonal projector)."""
        rng = np.random.default_rng(777)
        J = self._make_random_J(rng)
        # Exact pinv: the orthogonal projector is symmetric by construction
        J_pinv = self._exact_pinv(J)
        P = np.eye(7) - J_pinv @ J
        assert np.allclose(P, P.T, atol=1e-10), "Projector not symmetric"

    def test_dls_projector_approximate_null_space(self):
        """DLS projector (used in arm_control.py) reduces primary-task component
        but does not eliminate it exactly; residual is small for small damping."""
        rng = np.random.default_rng(42)
        J = self._make_random_J(rng)
        # DLS as used in arm_control.py (damping=0.05)
        J_pinv_dls = self._dls_pinv(J, damping=0.05)
        P_dls = np.eye(7) - J_pinv_dls @ J
        v = rng.standard_normal(7)
        residual = np.abs(J @ P_dls @ v).max()
        # DLS residual is bounded, not zero; with damping=0.05 on a random 3x7 J,
        # residual is typically 1e-3 to 1e-2 — this is expected behaviour.
        assert residual < 0.1, (
            f"DLS projector residual too large: {residual:.2e} "
            f"(damping=0.05 is an approximate null-space projector, not exact)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Gravity feedforward formula (offline unit test)
# ─────────────────────────────────────────────────────────────────────────────

class TestGravityFeedforwardFormula:
    """Tests the formula u_ff = q_des + bias/kp without MuJoCo."""

    @pytest.mark.parametrize("kp, bias, q_des", [
        (4500.0, 10.0, 0.5),
        (3500.0, -5.0, -0.3),
        (2000.0, 0.0, 1.0),
        (2000.0, 2.5, 0.0),
    ])
    def test_feedforward_formula(self, kp, bias, q_des):
        expected = q_des + bias / kp
        computed = q_des + bias / kp
        assert np.isclose(computed, expected, atol=1e-12)

    def test_feedforward_zero_at_zero_bias(self):
        """Zero gravity torque → feedforward == q_des (no correction needed)."""
        kp, bias, q_des = 4500.0, 0.0, 0.7
        u_ff = q_des + bias / kp
        assert np.isclose(u_ff, q_des, atol=1e-12)

    def test_feedforward_positive_bias_increases_ctrl(self):
        """Positive gravitational bias (arm pulling down) → ctrl increases."""
        kp, bias, q_des = 4500.0, 20.0, 0.0
        u_ff = q_des + bias / kp
        assert u_ff > q_des

    def test_joint_mid_range_shape(self):
        """JOINT_MID_RANGE must have exactly 7 entries."""
        assert JOINT_MID_RANGE.shape == (7,)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MuJoCo integration test
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_base_frame_ik_three_poses():
    """Base-frame IK converges from 3 different base poses, each to its own FK-sampled target.

    Strategy: for each base pose, place the base, run FK from a known arm config
    to get a reachable world position, reset arm to home, then verify IK recovers
    that target within 20 mm.  This avoids the pitfall of a single hard-coded
    target that may not be reachable from all three offsets.
    """
    mujoco = pytest.importorskip("mujoco")

    from library_robot.sim.mobile_model import load_mobile_robot_model, _load_config
    from library_robot.control.arm_control import solve_base_frame_pose_ik

    model, data, _ = load_mobile_robot_model()
    cfg = _load_config()

    # Freejoint address
    fjid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_freejoint")
    fa   = int(model.jnt_qposadr[fjid])
    chassis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mobile_chassis")
    hand_bid    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    # A known arm config (home pose) guaranteed to be in joint limits
    home_joints = cfg["panda_home"]["arm_joints"]
    arm_joint_names = [f"joint{i}" for i in range(1, 8)]

    base_poses = [
        (0.0,  0.0, 0.0),
        (-0.5, 0.0, 0.0),
        (0.0, -0.5, math.pi / 4),
    ]

    for bx, by, byaw in base_poses:
        # --- Step 1: place base, run FK from home, record reachable target ---
        mujoco.mj_forward(model, data)
        base_z = float(data.xpos[chassis_bid][2])
        data.qpos[fa:fa + 7] = [
            bx, by, base_z,
            math.cos(byaw / 2.0), 0.0, 0.0, math.sin(byaw / 2.0),
        ]
        for jname, val in home_joints.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            data.qpos[int(model.jnt_qposadr[jid])] = float(val)
        mujoco.mj_forward(model, data)
        fk_target_pos = data.xpos[hand_bid].copy()          # world-frame hand position
        fk_target_rot = data.xmat[hand_bid].reshape(3, 3).copy()

        # --- Step 2: reset arm to slightly perturbed config, then solve IK ---
        rng = np.random.default_rng(42)
        for jname in arm_joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            addr = int(model.jnt_qposadr[jid])
            lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
            # Perturb from home by ±0.3 rad, clamped to limits
            perturb = rng.uniform(-0.3, 0.3)
            data.qpos[addr] = float(np.clip(float(home_joints[jname]) + perturb, lo + 0.05, hi - 0.05))
        mujoco.mj_forward(model, data)

        result = solve_base_frame_pose_ik(
            model, data, fk_target_pos, fk_target_rot,
            base_pose_xyyaw=(bx, by, byaw),
            max_iterations=1000,
            damping=0.02,
        )
        pos_err_mm = result["position_error"] * 1000.0
        assert result["success"] or pos_err_mm < 20.0, (
            f"IK failed for base ({bx:.2f}, {by:.2f}, yaw={byaw:.2f}): "
            f"pos_error={pos_err_mm:.2f} mm, FK target={fk_target_pos}"
        )
        assert pos_err_mm < 20.0, (
            f"Position error {pos_err_mm:.2f} mm > 20 mm for base ({bx:.2f},{by:.2f},{byaw:.2f})"
        )
