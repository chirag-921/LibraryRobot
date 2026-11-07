"""Arm control utilities for the mobile Panda robot.

This module adds:
  1. Base-frame IK: solve arm IK for a world-frame target when the robot is at
     an arbitrary base pose (x, y, yaw).  Uses MuJoCo's own FK by temporarily
     placing the base, so the existing world-frame DLS solvers work unchanged.
  2. Null-space control: secondary joint-limit-avoidance objective added to the
     DLS position-IK step via the standard (I - J+ J) projector.
  3. Gravity feedforward: per-joint bias correction u_ff = q_des + bias/kp so
     the arm holds position without sag under gravity and payload.
  4. Straight-line Cartesian approach using base-frame IK with per-waypoint
     line-deviation measurement.

All functions that touch MuJoCo receive (model, data) explicitly.
Frame-transform helpers are pure NumPy and can be tested offline.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]

# Default Panda mid-range joint targets for null-space avoidance (needs tuning)
# joint4 mid-range is -1.571, joint6 mid-range is 1.867
JOINT_MID_RANGE = np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.867, 0.0])


# ---------------------------------------------------------------------------
# 1. Frame-transform helpers (pure NumPy, offline-testable)
# ---------------------------------------------------------------------------

def rot2d(yaw: float) -> np.ndarray:
    """Return a 2×2 rotation matrix for a yaw angle (radians)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def base_pose_to_transform(
    bx: float, by: float, byaw: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R_wb, t_wb) for the base in world frame.

    R_wb is the 3×3 rotation matrix (yaw only, no roll/pitch).
    t_wb is the 3-vector translation of the chassis origin in world coordinates.
    """
    c, s = math.cos(byaw), math.sin(byaw)
    R_wb = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    t_wb = np.array([bx, by, 0.0])
    return R_wb, t_wb


def world_to_chassis(
    world_pos: np.ndarray,
    bx: float, by: float, byaw: float,
) -> np.ndarray:
    """Transform a world-frame 3-vector into the chassis frame.

    Parameters
    ----------
    world_pos : (3,) array
        Position in world coordinates.
    bx, by, byaw : float
        Base pose (x, y, yaw) in world frame.

    Returns
    -------
    (3,) array
        Position expressed in the chassis frame.
    """
    R_wb, t_wb = base_pose_to_transform(bx, by, byaw)
    return R_wb.T @ (np.asarray(world_pos, dtype=float) - t_wb)


def chassis_to_world(
    chassis_pos: np.ndarray,
    bx: float, by: float, byaw: float,
) -> np.ndarray:
    """Transform a chassis-frame 3-vector into the world frame."""
    R_wb, t_wb = base_pose_to_transform(bx, by, byaw)
    return R_wb @ np.asarray(chassis_pos, dtype=float) + t_wb


def world_rot_to_chassis(
    world_rot: np.ndarray,
    byaw: float,
) -> np.ndarray:
    """Rotate a 3×3 world-frame rotation matrix into the chassis frame."""
    R_wb, _ = base_pose_to_transform(0.0, 0.0, byaw)
    return R_wb.T @ np.asarray(world_rot, dtype=float)


# ---------------------------------------------------------------------------
# 2. Arm joint address helpers (require MuJoCo at call time)
# ---------------------------------------------------------------------------

def _get_arm_dof_info(model: Any) -> Dict[str, Any]:
    """Return qpos addresses, DOF addresses, joint limits, and actuator indices."""
    import mujoco  # noqa: PLC0415

    qpos_addrs, dof_addrs, lowers, uppers, act_ids, kps = [], [], [], [], [], []
    for jname in ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise ValueError(f"Joint not found: {jname}")
        qpos_addrs.append(int(model.jnt_qposadr[jid]))
        dof_addrs.append(int(model.jnt_dofadr[jid]))
        lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
        lowers.append(lo + 0.01)
        uppers.append(hi - 0.01)

        aname = f"actuator{jname[-1]}"
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        act_ids.append(int(aid))
        kps.append(float(model.actuator_gainprm[aid, 0]) if aid >= 0 else 1.0)

    return {
        "qpos": np.asarray(qpos_addrs, dtype=int),
        "dof":  np.asarray(dof_addrs,  dtype=int),
        "lo":   np.asarray(lowers),
        "hi":   np.asarray(uppers),
        "act":  np.asarray(act_ids, dtype=int),
        "kp":   np.asarray(kps),
    }


# ---------------------------------------------------------------------------
# 3. Base-frame IK
# ---------------------------------------------------------------------------

def solve_base_frame_pose_ik(
    model: Any,
    data: Any,
    world_target_pos: np.ndarray,
    world_target_rot: np.ndarray,
    base_pose_xyyaw: Tuple[float, float, float],
    body_name: str = "hand",
    max_iterations: int = 800,
    position_tolerance: float = 1e-4,
    orientation_tolerance: float = 1e-3,
    damping: float = 0.05,
    step_size: float = 0.5,
    max_joint_step: float = 0.10,
    orientation_weight: float = 0.1,
) -> Dict[str, Any]:
    """Solve arm IK for a world-frame target when the robot base is at an
    arbitrary (x, y, yaw) pose.

    Strategy: place the robot base at *base_pose_xyyaw* in data.qpos (the
    freejoint), call mj_forward so that the arm geometry is correct in world
    space, then delegate to the existing world-frame DLS pose IK.  The
    caller's data.qpos is restored to its original base pose afterwards so
    this function is non-destructive for the base.

    Parameters
    ----------
    model, data
        MuJoCo model and data (modified in-place during solve, restored after).
    world_target_pos : (3,)
        Desired hand position in world coordinates (metres).
    world_target_rot : (3, 3)
        Desired hand orientation in world coordinates.
    base_pose_xyyaw : (bx, by, byaw)
        Robot base pose in world frame.
    body_name : str
        Body whose pose is controlled (default "hand").

    Returns
    -------
    dict with keys: success, iterations, position, rotation, position_error,
        orientation_error, joint_positions, chassis_pose (xyyaw used).
    """
    import mujoco as _mj  # noqa: PLC0415
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    world_target_pos = np.asarray(world_target_pos, dtype=float)
    world_target_rot = np.asarray(world_target_rot, dtype=float)

    bx, by, byaw = float(base_pose_xyyaw[0]), float(base_pose_xyyaw[1]), float(base_pose_xyyaw[2])

    # Locate the base freejoint
    fjid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, "base_freejoint")
    if fjid < 0:
        raise ValueError("Joint 'base_freejoint' not found in model. Is this the wheeled model?")

    fa = int(model.jnt_qposadr[fjid])

    # Retrieve chassis physical z from config (wheel_radius + chassis_half_z)
    chassis_bid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, "mobile_chassis")
    base_z = float(data.xpos[chassis_bid][2])  # use current settled z

    # Save original base qpos
    orig_base = data.qpos[fa:fa + 7].copy()

    # Set new base pose
    data.qpos[fa:fa + 7] = [
        bx, by, base_z,
        math.cos(byaw / 2.0), 0.0, 0.0, math.sin(byaw / 2.0),
    ]
    _mj.mj_forward(model, data)

    # Solve using existing world-frame DLS pose IK inline (avoid import loop)
    info = _get_arm_dof_info(model)
    qpos_addrs = info["qpos"]
    dof_addrs  = info["dof"]
    lower_limits = info["lo"]
    upper_limits = info["hi"]

    body_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body not found: {body_name}")

    success = False
    for iteration in range(max_iterations):
        _mj.mj_forward(model, data)
        cur_pos = data.xpos[body_id].copy()
        cur_rot = data.xmat[body_id].reshape(3, 3).copy()

        pos_err = world_target_pos - cur_pos
        rot_diff = world_target_rot @ cur_rot.T
        ori_err = Rotation.from_matrix(rot_diff).as_rotvec()

        pos_err_norm = float(np.linalg.norm(pos_err))
        ori_err_norm = float(np.linalg.norm(ori_err))

        if pos_err_norm < position_tolerance and ori_err_norm < orientation_tolerance:
            success = True
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        _mj.mj_jacBody(model, data, jacp, jacr, body_id)
        Jp = jacp[:, dof_addrs]
        Jr = jacr[:, dof_addrs]

        J = np.vstack([Jp, orientation_weight * Jr])
        err = np.concatenate([pos_err, orientation_weight * ori_err])
        A = J @ J.T + damping**2 * np.eye(6)
        dq = J.T @ np.linalg.solve(A, err)
        dq = np.clip(dq * step_size, -max_joint_step, max_joint_step)

        new_q = np.clip(data.qpos[qpos_addrs] + dq, lower_limits, upper_limits)
        data.qpos[qpos_addrs] = new_q

    _mj.mj_forward(model, data)
    final_pos = data.xpos[body_id].copy()
    final_rot = data.xmat[body_id].reshape(3, 3).copy()
    pos_err_final = float(np.linalg.norm(world_target_pos - final_pos))
    ori_err_final = float(Rotation.from_matrix(world_target_rot @ final_rot.T).magnitude())
    success = success or (pos_err_final < position_tolerance and ori_err_final < orientation_tolerance)

    final_joints = {ARM_JOINTS[i]: float(data.qpos[qpos_addrs[i]]) for i in range(7)}

    # Restore original base qpos
    data.qpos[fa:fa + 7] = orig_base
    _mj.mj_forward(model, data)

    return {
        "success": success,
        "iterations": iteration + 1,
        "position": final_pos,
        "rotation": final_rot,
        "position_error": pos_err_final,
        "orientation_error": ori_err_final,
        "joint_positions": final_joints,
        "chassis_pose": (bx, by, byaw),
    }


# ---------------------------------------------------------------------------
# 4. Null-space position IK (joint-limit avoidance)
# ---------------------------------------------------------------------------

def solve_position_ik_with_null_space(
    model: Any,
    data: Any,
    target_position: np.ndarray,
    body_name: str = "hand",
    q_secondary: Optional[np.ndarray] = None,
    null_weight: float = 0.3,          # needs tuning: secondary-task gain
    max_iterations: int = 500,
    tolerance: float = 1e-4,
    damping: float = 0.05,
    step_size: float = 0.5,
    max_joint_step: float = 0.15,
) -> Dict[str, Any]:
    """Damped-least-squares position IK with null-space secondary objective.

    Primary task  : move `body_name` to `target_position` (3-DoF).
    Secondary task: move arm joints towards `q_secondary` (default: mid-range)
                    using the null-space projector (I - J+ J).

    The secondary objective does not affect the primary task error to first
    order because the projector removes any component in the range of J.

    Parameters
    ----------
    null_weight : float
        Scale of the secondary-objective joint step. 0 disables null-space.
    q_secondary : (7,) array, optional
        Target joint configuration for null-space objective.
        Defaults to JOINT_MID_RANGE.

    Returns
    -------
    dict with keys: success, iterations, position, target, error,
        joint_positions, min_singular_value, null_space_improvement_deg.
    """
    import mujoco as _mj  # noqa: PLC0415

    target_position = np.asarray(target_position, dtype=float)
    if target_position.shape != (3,):
        raise ValueError("target_position must be shape (3,)")

    if q_secondary is None:
        q_secondary = JOINT_MID_RANGE.copy()
    else:
        q_secondary = np.asarray(q_secondary, dtype=float)

    info = _get_arm_dof_info(model)
    qpos_addrs = info["qpos"]
    dof_addrs  = info["dof"]
    lo = info["lo"]
    hi = info["hi"]

    body_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body not found: {body_name}")

    _mj.mj_forward(model, data)
    q_initial = data.qpos[qpos_addrs].copy()

    success = False
    min_sigma = float("inf")

    for iteration in range(max_iterations):
        _mj.mj_forward(model, data)
        cur_pos = data.xpos[body_id].copy()
        err = target_position - cur_pos
        err_norm = float(np.linalg.norm(err))

        if err_norm < tolerance:
            success = True
            break

        jacp = np.zeros((3, model.nv))
        _mj.mj_jacBody(model, data, jacp, None, body_id)
        J = jacp[:, dof_addrs]   # 3×7

        # DLS inverse: J+ = Jᵀ (JJᵀ + λ²I)⁻¹
        A = J @ J.T + damping**2 * np.eye(3)
        J_pinv = J.T @ np.linalg.solve(A, np.eye(3))  # 7×3

        # Track minimum singular value
        sigma = float(np.linalg.matrix_rank(J, tol=1e-6))
        svs = np.linalg.svd(J, compute_uv=False)
        if len(svs) > 0:
            min_sigma = min(min_sigma, float(svs[-1]))

        # Primary step
        dq_primary = J_pinv @ err

        # Null-space step: (I - J+ J) * α * (q_sec - q)
        q_cur = data.qpos[qpos_addrs].copy()
        dq_null = null_weight * (q_secondary - q_cur)
        # Project onto null space of J using exact Moore-Penrose pseudoinverse
        # (guarantees J @ dq_null_proj == 0, so secondary task does not perturb Cartesian position)
        J_exact_pinv = np.linalg.pinv(J)
        I_minus_JpJ = np.eye(7) - J_exact_pinv @ J
        dq_null_proj = I_minus_JpJ @ dq_null

        dq = dq_primary + dq_null_proj
        dq = np.clip(dq * step_size, -max_joint_step, max_joint_step)

        new_q = np.clip(q_cur + dq, lo, hi)
        data.qpos[qpos_addrs] = new_q

    _mj.mj_forward(model, data)
    final_pos = data.xpos[body_id].copy()
    final_err = float(np.linalg.norm(target_position - final_pos))
    success = success or final_err < tolerance

    final_joints = {ARM_JOINTS[i]: float(data.qpos[qpos_addrs[i]]) for i in range(7)}

    # Measure margin improvement: distance from joint limits in deg
    q_final = data.qpos[qpos_addrs].copy()
    margin_initial = np.minimum(q_initial - info["lo"], info["hi"] - q_initial)
    margin_final   = np.minimum(q_final   - info["lo"], info["hi"] - q_final)
    margin_improvement_deg = float(np.degrees(np.min(margin_final) - np.min(margin_initial)))

    return {
        "success": success,
        "iterations": iteration + 1,
        "position": final_pos,
        "target": target_position,
        "error": final_err,
        "joint_positions": final_joints,
        "min_singular_value": float(min_sigma),
        "null_space_improvement_deg": margin_improvement_deg,
    }


# ---------------------------------------------------------------------------
# 5. Gravity feedforward
# ---------------------------------------------------------------------------

def compute_gravity_feedforward(
    model: Any,
    data: Any,
) -> Dict[str, float]:
    """Compute per-joint gravity-feedforward correction u_ff = qfrc_bias / kp.

    After `mj_forward`, data.qfrc_bias contains the generalized gravity+Coriolis
    force for each DoF.  For a position servo: tau = kp*(u - q) - kv*qdot.
    At equilibrium with no motion: tau = qfrc_bias → u_ff = q + qfrc_bias/kp.

    Returns
    -------
    dict mapping joint name → feedforward offset (radians to add to ctrl).
    """
    import mujoco as _mj  # noqa: PLC0415
    _mj.mj_forward(model, data)
    info = _get_arm_dof_info(model)
    ff = {}
    for i, jname in enumerate(ARM_JOINTS):
        bias = float(data.qfrc_bias[info["dof"][i]])
        kp   = info["kp"][i]
        ff[jname] = bias / kp if kp > 0.0 else 0.0
    return ff


def apply_gravity_feedforward(
    model: Any,
    data: Any,
    q_des: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Apply gravity feedforward to data.ctrl for all arm joints.

    Sets data.ctrl[actuator_i] = q_des[joint_i] + qfrc_bias[dof_i] / kp_i.

    Parameters
    ----------
    q_des : dict, optional
        Desired joint positions. If None, reads current data.qpos.

    Returns
    -------
    dict mapping joint name → final ctrl value applied.
    """
    import mujoco as _mj  # noqa: PLC0415
    _mj.mj_forward(model, data)
    info = _get_arm_dof_info(model)
    ff = compute_gravity_feedforward(model, data)
    applied = {}
    for i, jname in enumerate(ARM_JOINTS):
        if q_des is not None and jname in q_des:
            q_target = float(q_des[jname])
        else:
            q_target = float(data.qpos[info["qpos"][i]])
        ctrl_val = q_target + ff[jname]
        if info["act"][i] >= 0:
            data.ctrl[info["act"][i]] = ctrl_val
        applied[jname] = ctrl_val
    return applied


# ---------------------------------------------------------------------------
# 6. Straight-line Cartesian approach
# ---------------------------------------------------------------------------

def _point_to_line_distance(
    point: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray,
) -> float:
    """Perpendicular distance from `point` to the line segment [start, end]."""
    d = line_end - line_start
    d_norm = float(np.linalg.norm(d))
    if d_norm < 1e-12:
        return float(np.linalg.norm(point - line_start))
    return float(np.linalg.norm(np.cross(d / d_norm, line_start - point)))


def cartesian_straight_approach(
    model: Any,
    data: Any,
    world_start_pos: np.ndarray,
    world_target_pos: np.ndarray,
    world_target_rot: Optional[np.ndarray] = None,
    base_pose_xyyaw: Optional[Tuple[float, float, float]] = None,
    num_waypoints: int = 20,
    body_name: str = "hand",
    tolerance: float = 1e-4,
    damping: float = 0.05,
    step_size: float = 0.5,
    max_joint_step: float = 0.10,
    orientation_weight: float = 0.1,
) -> Dict[str, Any]:
    """Generate a straight-line Cartesian approach path using base-frame pose IK.

    If `base_pose_xyyaw` is supplied the base freejoint is placed there first
    and held fixed during the entire trajectory.  The IK is solved in world
    frame (the natural MuJoCo frame) which is correct after mj_forward.

    Parameters
    ----------
    world_start_pos : (3,)
        Starting end-effector position in world frame.
    world_target_pos : (3,)
        Target end-effector position in world frame (end of approach).
    world_target_rot : (3, 3), optional
        Target orientation; if None the current orientation is maintained.
    base_pose_xyyaw : (bx, by, byaw), optional
        Robot base pose in world frame; if None the current base pose is used.
    num_waypoints : int
        Number of intermediate waypoints (not including start).

    Returns
    -------
    dict with:
        waypoints : list of dicts (index, cart_target, joint_positions,
                                    achieved_pos, line_deviation_m, ik_error)
        max_line_deviation_m : float
        final_error_m : float
        final_error_deg : float  (if rot target provided)
    """
    import mujoco as _mj  # noqa: PLC0415
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    ws = np.asarray(world_start_pos, dtype=float)
    wt = np.asarray(world_target_pos, dtype=float)

    # Determine base pose
    if base_pose_xyyaw is not None:
        bx, by, byaw = float(base_pose_xyyaw[0]), float(base_pose_xyyaw[1]), float(base_pose_xyyaw[2])
    else:
        # Read from sim state
        chassis_bid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, "mobile_chassis")
        cx = float(data.xpos[chassis_bid][0])
        cy = float(data.xpos[chassis_bid][1])
        q = data.xquat[chassis_bid]
        byaw = float(math.atan2(2.0*(q[0]*q[3]+q[1]*q[2]), 1.0-2.0*(q[2]**2+q[3]**2)))
        bx, by = cx, cy

    # Place base at requested pose
    fjid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, "base_freejoint")
    fa = int(model.jnt_qposadr[fjid]) if fjid >= 0 else None
    if fa is not None:
        chassis_bid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, "mobile_chassis")
        base_z = float(data.xpos[chassis_bid][2])
        orig_base = data.qpos[fa:fa+7].copy()
        data.qpos[fa:fa+7] = [bx, by, base_z,
                               math.cos(byaw/2), 0.0, 0.0, math.sin(byaw/2)]
        _mj.mj_forward(model, data)

    # Determine target rotation
    body_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, body_name)
    if world_target_rot is None:
        world_target_rot = data.xmat[body_id].reshape(3, 3).copy()
    world_target_rot = np.asarray(world_target_rot, dtype=float)

    # Build waypoints (exclude start, include end)
    path = np.linspace(ws, wt, num_waypoints + 1)[1:]

    info = _get_arm_dof_info(model)
    qpos_addrs = info["qpos"]
    dof_addrs  = info["dof"]
    lo = info["lo"]
    hi = info["hi"]

    waypoints = []
    for idx, cart_target in enumerate(path):
        # Inline DLS pose IK (same as solve_base_frame_pose_ik but no base restore)
        from scipy.spatial.transform import Rotation as _R  # noqa: PLC0415
        for _iter in range(500):
            _mj.mj_forward(model, data)
            cur_pos = data.xpos[body_id].copy()
            cur_rot = data.xmat[body_id].reshape(3, 3).copy()
            pos_err = cart_target - cur_pos
            ori_err = _R.from_matrix(world_target_rot @ cur_rot.T).as_rotvec()
            if np.linalg.norm(pos_err) < tolerance and np.linalg.norm(ori_err) < 1e-3:
                break
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            _mj.mj_jacBody(model, data, jacp, jacr, body_id)
            Jp = jacp[:, dof_addrs]
            Jr = jacr[:, dof_addrs]
            J = np.vstack([Jp, orientation_weight * Jr])
            err_vec = np.concatenate([pos_err, orientation_weight * ori_err])
            A = J @ J.T + damping**2 * np.eye(6)
            dq = np.clip(J.T @ np.linalg.solve(A, err_vec) * step_size,
                         -max_joint_step, max_joint_step)
            data.qpos[qpos_addrs] = np.clip(data.qpos[qpos_addrs] + dq, lo, hi)

        _mj.mj_forward(model, data)
        achieved = data.xpos[body_id].copy()
        line_dev = _point_to_line_distance(achieved, ws, wt)
        ik_err = float(np.linalg.norm(cart_target - achieved))

        waypoints.append({
            "index": idx + 1,
            "cart_target": cart_target.copy(),
            "joint_positions": {ARM_JOINTS[i]: float(data.qpos[qpos_addrs[i]]) for i in range(7)},
            "achieved_pos": achieved,
            "line_deviation_m": line_dev,
            "ik_error_m": ik_err,
        })

    final_error_m = float(np.linalg.norm(wt - waypoints[-1]["achieved_pos"]))
    final_rot = data.xmat[body_id].reshape(3, 3).copy()
    final_error_deg = float(
        np.degrees(Rotation.from_matrix(world_target_rot @ final_rot.T).magnitude())
    )
    max_dev = max(w["line_deviation_m"] for w in waypoints)

    # Restore base
    if fa is not None:
        data.qpos[fa:fa+7] = orig_base
        _mj.mj_forward(model, data)

    return {
        "waypoints": waypoints,
        "max_line_deviation_m": max_dev,
        "final_error_m": final_error_m,
        "final_error_deg": final_error_deg,
        "base_pose": (bx, by, byaw),
        "num_waypoints": num_waypoints,
    }
