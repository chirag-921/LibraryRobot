"""Script 16: Arm accuracy test on the mobile base.

Tests 10 random reachable end-effector poses, twice:
  - Run A: base at world origin (0, 0, 0)
  - Run B: base parked at a table parking pose from the library scene layout

For each pose reports:
  position_error_mm, orientation_error_deg, iterations, converged, min_sigma

Target: >= 9 of 10 poses within 2 cm position and 5 deg orientation.

Saves results to results/arm_accuracy_test.csv.
Logs full output to logs/script16_output.txt.
"""

from __future__ import annotations

import csv
import math
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR    = PROJECT_ROOT / "logs"

RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ── Tee output ───────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, path: Path):
        self._t = sys.stdout
        self._f = path.open("w", encoding="utf-8")
    def write(self, s: str):
        # Terminal may be cp1252 on Windows; encode with replace to avoid crash
        try:
            self._t.write(s)
        except UnicodeEncodeError:
            self._t.write(s.encode(self._t.encoding or "ascii", errors="replace").decode(self._t.encoding or "ascii", errors="replace"))
        self._f.write(s)
    def flush(self):
        self._t.flush(); self._f.flush()
    def close(self):
        self._f.close()

_log = _Tee(LOGS_DIR / "script16_output.txt")
sys.stdout = _log

# ── Imports ───────────────────────────────────────────────────────────────────

import mujoco
from scipy.spatial.transform import Rotation

from library_robot.sim.mobile_model import load_mobile_robot_model, _load_config
from library_robot.control.arm_control import (
    ARM_JOINTS,
    solve_base_frame_pose_ik,
    solve_position_ik_with_null_space,
    compute_gravity_feedforward,
    apply_gravity_feedforward,
    cartesian_straight_approach,
    _get_arm_dof_info,
    JOINT_MID_RANGE,
    world_to_chassis,
    chassis_to_world,
    world_rot_to_chassis,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _sep(char="=", n=72):
    print(char * n)

def _header(title: str):
    _sep()
    print(title)
    _sep()

def _sub(title: str):
    _sep("-", 72)
    print(title)
    _sep("-", 72)

def _load_robot():
    """Load wheeled robot model, place base at origin, arm at home."""
    model, data, _ = load_mobile_robot_model()
    cfg = _load_config()
    # Ensure arm at home pose
    for jname, target in cfg["panda_home"]["arm_joints"].items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        qadr = int(model.jnt_qposadr[jid])
        data.qpos[qadr] = float(target)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{jname[-1]}")
        data.ctrl[aid] = float(target)
    mujoco.mj_forward(model, data)
    return model, data, cfg

def _set_base_pose(model, data, bx: float, by: float, byaw: float):
    fjid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_freejoint")
    fa = int(model.jnt_qposadr[fjid])
    chassis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mobile_chassis")
    base_z = float(data.xpos[chassis_bid][2])
    data.qpos[fa:fa+7] = [bx, by, base_z,
                           math.cos(byaw/2), 0.0, 0.0, math.sin(byaw/2)]
    mujoco.mj_forward(model, data)

def _reset_arm_home(model, data, cfg):
    for jname, target in cfg["panda_home"]["arm_joints"].items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        qadr = int(model.jnt_qposadr[jid])
        data.qpos[qadr] = float(target)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{jname[-1]}")
        data.ctrl[aid] = float(target)
    mujoco.mj_forward(model, data)

def _sample_reachable_poses(model, data, cfg, n: int, rng: np.random.Generator, base_pose):
    """
    Sample n reachable arm targets by:
      1. Setting base at base_pose
      2. Drawing random joint configs within limits
      3. Computing FK → that world-frame (pos, rot) is the target
    """
    bx, by, byaw = base_pose
    _set_base_pose(model, data, bx, by, byaw)

    info = _get_arm_dof_info(model)
    qpos_addrs = info["qpos"]
    hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    poses = []
    while len(poses) < n:
        # Random joints uniformly in [lo+0.05, hi-0.05]
        lo = info["lo"] + 0.05
        hi = info["hi"] - 0.05
        q_rand = rng.uniform(lo, hi)
        data.qpos[qpos_addrs] = q_rand
        mujoco.mj_forward(model, data)
        pos = data.xpos[hand_bid].copy()
        rot = data.xmat[hand_bid].reshape(3, 3).copy()
        # Ensure above chassis floor (z > 0.3 m) and within reasonable reach
        if pos[2] > 0.30:
            poses.append((pos.copy(), rot.copy(), q_rand.copy()))

    # Restore home
    _reset_arm_home(model, data, cfg)
    _set_base_pose(model, data, bx, by, byaw)
    return poses

def _run_accuracy_test(model, data, cfg, poses, base_pose, run_label: str, rows: list):
    _sub(f"Accuracy Test — {run_label}  base=({base_pose[0]:.2f}, {base_pose[1]:.2f}, yaw={math.degrees(base_pose[2]):.1f}°)")
    bx, by, byaw = base_pose
    _set_base_pose(model, data, bx, by, byaw)

    info = _get_arm_dof_info(model)
    qpos_addrs = info["qpos"]
    hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    n_pass = 0
    print(f"{'#':>3}  {'pos_err_mm':>12}  {'ori_err_deg':>12}  {'iters':>6}  {'conv':>5}  {'min_sigma':>10}")
    print("-" * 65)

    for i, (tgt_pos, tgt_rot, q_true) in enumerate(poses):
        # Reset arm to home before each solve
        _reset_arm_home(model, data, cfg)
        _set_base_pose(model, data, bx, by, byaw)

        result = solve_base_frame_pose_ik(
            model, data,
            world_target_pos=tgt_pos,
            world_target_rot=tgt_rot,
            base_pose_xyyaw=base_pose,
            max_iterations=800,
            position_tolerance=1e-4,
            orientation_tolerance=1e-3,
        )

        # Measure min singular value via Jacobian at the solution
        # Temporarily set joints to solution
        fa_save = None
        fjid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_freejoint")
        if fjid >= 0:
            fa_save = int(model.jnt_qposadr[fjid])
            base_z = float(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mobile_chassis")][2])
            orig_base = data.qpos[fa_save:fa_save+7].copy()
            data.qpos[fa_save:fa_save+7] = [bx, by, base_z,
                                             math.cos(byaw/2), 0.0, 0.0, math.sin(byaw/2)]
        for j, jname in enumerate(ARM_JOINTS):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            data.qpos[int(model.jnt_qposadr[jid])] = result["joint_positions"][jname]
        mujoco.mj_forward(model, data)
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, None, hand_bid)
        J7 = jacp[:, info["dof"]]
        svs = np.linalg.svd(J7, compute_uv=False)
        min_sigma = float(svs[-1])
        # Restore
        if fa_save is not None:
            data.qpos[fa_save:fa_save+7] = orig_base
        mujoco.mj_forward(model, data)

        pos_err_mm  = result["position_error"] * 1000.0
        ori_err_deg = math.degrees(result["orientation_error"])
        converged   = result["success"]
        iters       = result["iterations"]

        pass_this   = pos_err_mm <= 20.0 and ori_err_deg <= 5.0
        if pass_this:
            n_pass += 1

        flag = "Y" if converged else "N"
        print(f"{i+1:>3}  {pos_err_mm:>12.3f}  {ori_err_deg:>12.4f}  {iters:>6}  {flag:>5}  {min_sigma:>10.4f}")

        rows.append({
            "run": run_label,
            "pose_index": i + 1,
            "base_x": bx,
            "base_y": by,
            "base_yaw_deg": math.degrees(byaw),
            "target_pos_x": tgt_pos[0],
            "target_pos_y": tgt_pos[1],
            "target_pos_z": tgt_pos[2],
            "pos_error_mm": pos_err_mm,
            "ori_error_deg": ori_err_deg,
            "iterations": iters,
            "converged": converged,
            "within_2cm_5deg": pass_this,
            "min_singular_value": min_sigma,
        })

    print("-" * 65)
    print(f"Poses within 2 cm & 5 deg: {n_pass}/10  ({'PASS' if n_pass >= 9 else 'FAIL — target is 9/10'})")
    return n_pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    _header("SCRIPT 16: ARM ACCURACY TEST ON MOBILE BASE")

    model, data, cfg = _load_robot()

    # ── 1. Base-frame IK demo with 3 poses ──────────────────────────────────
    _header("SECTION 1: BASE-FRAME IK — 3 BASE POSES, SAME WORLD TARGET")

    # Choose a world target that is definitely reachable from origin
    mujoco.mj_forward(model, data)
    hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    world_target_pos = np.array([0.40, 0.10, 0.80])
    world_target_rot = data.xmat[hand_bid].reshape(3, 3).copy()

    base_poses_demo = [
        (0.0,   0.0,   0.0,       "Origin (0, 0, yaw=0°)"),
        (0.15, -0.10,  0.20,      "Forward-right (0.15, -0.10, yaw=+11.5°)"),
        (-0.10, 0.10, -0.20,      "Back-left (-0.10, 0.10, yaw=-11.5°)"),
    ]

    print(f"World target position: {world_target_pos}")
    print(f"World target orientation: identity-ish (home rotation)\n")
    for bx, by, byaw, label in base_poses_demo:
        _reset_arm_home(model, data, cfg)
        result = solve_base_frame_pose_ik(
            model, data, world_target_pos, world_target_rot,
            base_pose_xyyaw=(bx, by, byaw),
        )
        # Express target in chassis frame for reference
        tgt_chassis = world_to_chassis(world_target_pos, bx, by, byaw)
        print(f"  Base: {label}")
        print(f"    Chassis-frame target:  {tgt_chassis.round(4)}")
        print(f"    IK converged:          {result['success']}")
        print(f"    Position error:        {result['position_error']*1000:.3f} mm")
        print(f"    Orientation error:     {math.degrees(result['orientation_error']):.4f} deg")
        print(f"    Iterations:            {result['iterations']}")
        print()

    # ── 2. Null-space control demo ───────────────────────────────────────────
    _header("SECTION 2: NULL-SPACE CONTROL — JOINT-LIMIT AVOIDANCE")

    _reset_arm_home(model, data, cfg)
    _set_base_pose(model, data, 0.0, 0.0, 0.0)

    # Choose a target near a joint-limit-stressing configuration
    test_target = np.array([0.55, 0.20, 0.70])

    print(f"Target position: {test_target}")
    print()

    # Solve WITHOUT null-space
    _reset_arm_home(model, data, cfg)
    _set_base_pose(model, data, 0.0, 0.0, 0.0)
    from library_robot.control.inverse_kinematics import solve_position_ik
    ik_data_no = mujoco.MjData(model)
    ik_data_no.qpos[:] = data.qpos
    mujoco.mj_forward(model, ik_data_no)
    res_no_ns = solve_position_ik(model, ik_data_no, test_target)
    info = _get_arm_dof_info(model)
    q_no_ns = np.array([res_no_ns["joint_positions"][j] for j in ARM_JOINTS])
    margin_no_ns_deg = float(np.degrees(np.min(
        np.minimum(q_no_ns - info["lo"], info["hi"] - q_no_ns)
    )))

    # Solve WITH null-space
    _reset_arm_home(model, data, cfg)
    _set_base_pose(model, data, 0.0, 0.0, 0.0)
    ik_data_ns = mujoco.MjData(model)
    ik_data_ns.qpos[:] = data.qpos
    mujoco.mj_forward(model, ik_data_ns)
    res_ns = solve_position_ik_with_null_space(model, ik_data_ns, test_target)
    q_ns = np.array([res_ns["joint_positions"][j] for j in ARM_JOINTS])
    margin_ns_deg = float(np.degrees(np.min(
        np.minimum(q_ns - info["lo"], info["hi"] - q_ns)
    )))

    print(f"  Without null-space:")
    print(f"    Position error:   {res_no_ns['error']*1000:.3f} mm")
    print(f"    Min joint margin: {margin_no_ns_deg:.2f} deg")
    print(f"    Joints (rad): {[round(res_no_ns['joint_positions'][j],4) for j in ARM_JOINTS]}")
    print()
    print(f"  With null-space (weight=0.3, target=mid-range):")
    print(f"    Position error:   {res_ns['error']*1000:.3f} mm  (primary task unchanged)")
    print(f"    Min joint margin: {margin_ns_deg:.2f} deg")
    print(f"    Joints (rad): {[round(res_ns['joint_positions'][j],4) for j in ARM_JOINTS]}")
    print()

    err_diff = abs(res_ns["error"] - res_no_ns["error"]) * 1000.0
    margin_improvement = margin_ns_deg - margin_no_ns_deg
    print(f"  Primary task error change: {err_diff:.4f} mm  (should be < 1 mm)")
    print(f"  Joint-limit margin change: {margin_improvement:+.2f} deg  (positive = improved)")

    # ── 3. Gravity feedforward demo ──────────────────────────────────────────
    _header("SECTION 3: GRAVITY FEEDFORWARD — SAG REDUCTION")

    # Place arm at extended pose + 2 kg payload on hand
    _reset_arm_home(model, data, cfg)
    _set_base_pose(model, data, 0.0, 0.0, 0.0)

    # Extended pose from script 13 (similar to test D/E)
    extended_joints = {
        "joint1": 0.0,
        "joint2": 0.85,
        "joint3": 0.0,
        "joint4": -0.30,
        "joint5": 0.0,
        "joint6": 1.05,
        "joint7": 0.0,
    }
    for jname, target in extended_joints.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        qadr = int(model.jnt_qposadr[jid])
        data.qpos[qadr] = float(target)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{jname[-1]}")
        data.ctrl[aid] = float(target)   # no feedforward

    # Attach 2 kg payload to hand
    hand_bid_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    orig_mass = float(model.body_mass[hand_bid_id])
    model.body_mass[hand_bid_id] = orig_mass + 2.0
    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)

    # Simulate 1 s without feedforward
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    info2 = _get_arm_dof_info(model)
    q_no_ff = data.qpos[info2["qpos"]].copy()
    sag_no_ff_deg = {}
    for i, jname in enumerate(ARM_JOINTS):
        sag_no_ff_deg[jname] = abs(math.degrees(q_no_ff[i] - float(extended_joints[jname])))

    print("WITHOUT gravity feedforward (2 kg payload, 1 s sim):")
    print(f"  {'Joint':<8}  {'Target(rad)':>12}  {'Actual(rad)':>12}  {'Sag(deg)':>10}")
    print("  " + "-"*50)
    for jname in ARM_JOINTS:
        t = extended_joints[jname]
        a = q_no_ff[ARM_JOINTS.index(jname)]
        print(f"  {jname:<8}  {t:>12.4f}  {a:>12.4f}  {sag_no_ff_deg[jname]:>10.4f}")
    max_sag_no = max(sag_no_ff_deg.values())
    print(f"  Max sag: {max_sag_no:.4f} deg")
    print()

    # Now apply feedforward
    # Reset to extended pose
    for jname, target in extended_joints.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        data.qpos[int(model.jnt_qposadr[jid])] = float(target)
        data.qvel[int(model.jnt_dofadr[jid])] = 0.0
    mujoco.mj_forward(model, data)
    # Apply feedforward
    ff = apply_gravity_feedforward(model, data, q_des=extended_joints)
    print("Gravity feedforward corrections (rad added to ctrl):")
    for jname in ARM_JOINTS:
        ff_val = ff[jname] - extended_joints[jname]
        print(f"  {jname}: +{ff_val:+.6f} rad  (ctrl={ff[jname]:.6f})")
    print()

    # Simulate 1 s with feedforward
    for _ in range(500):
        # Re-apply feedforward each step (since qfrc_bias changes with config)
        apply_gravity_feedforward(model, data, q_des=extended_joints)
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    q_with_ff = data.qpos[info2["qpos"]].copy()
    sag_with_ff_deg = {}
    for i, jname in enumerate(ARM_JOINTS):
        sag_with_ff_deg[jname] = abs(math.degrees(q_with_ff[i] - float(extended_joints[jname])))

    print("WITH gravity feedforward (2 kg payload, 1 s sim):")
    print(f"  {'Joint':<8}  {'Target(rad)':>12}  {'Actual(rad)':>12}  {'Sag(deg)':>10}")
    print("  " + "-"*50)
    for jname in ARM_JOINTS:
        t = extended_joints[jname]
        a = q_with_ff[ARM_JOINTS.index(jname)]
        print(f"  {jname:<8}  {t:>12.4f}  {a:>12.4f}  {sag_with_ff_deg[jname]:>10.4f}")
    max_sag_ff = max(sag_with_ff_deg.values())
    print(f"  Max sag: {max_sag_ff:.4f} deg  (target: < 0.1 deg)")
    print(f"  Reduction: {max_sag_no:.4f} deg -> {max_sag_ff:.4f} deg  ({(1-max_sag_ff/max_sag_no)*100:.1f}% improvement)")

    # Restore mass
    model.body_mass[hand_bid_id] = orig_mass
    mujoco.mj_setConst(model, data)

    # ── 4. Cartesian approach demo ───────────────────────────────────────────
    _header("SECTION 4: STRAIGHT-LINE CARTESIAN APPROACH (10 cm down)")

    model2, data2, cfg2 = _load_robot()
    _set_base_pose(model2, data2, 0.0, 0.0, 0.0)

    # Reset arm to home
    for jname, target in cfg2["panda_home"]["arm_joints"].items():
        jid = mujoco.mj_name2id(model2, mujoco.mjtObj.mjOBJ_JOINT, jname)
        data2.qpos[int(model2.jnt_qposadr[jid])] = float(target)
        aid = mujoco.mj_name2id(model2, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{jname[-1]}")
        data2.ctrl[aid] = float(target)
    mujoco.mj_forward(model2, data2)

    hand_bid2 = mujoco.mj_name2id(model2, mujoco.mjtObj.mjOBJ_BODY, "hand")
    start_pos = data2.xpos[hand_bid2].copy()
    target_pos = start_pos + np.array([0.0, 0.0, -0.10])  # 10 cm down

    print(f"Start position:  {start_pos.round(4)}")
    print(f"Target position: {target_pos.round(4)}")
    print(f"Approach vector: [0, 0, -0.10] m (straight down)")
    print()

    result_traj = cartesian_straight_approach(
        model2, data2,
        world_start_pos=start_pos,
        world_target_pos=target_pos,
        base_pose_xyyaw=(0.0, 0.0, 0.0),
        num_waypoints=20,
    )

    print(f"{'WP':>3}  {'Achieved Z':>12}  {'Line Dev (mm)':>14}  {'IK err (mm)':>12}")
    print("-"*48)
    for wp in result_traj["waypoints"]:
        print(f"{wp['index']:>3}  {wp['achieved_pos'][2]:>12.5f}  "
              f"{wp['line_deviation_m']*1000:>14.3f}  {wp['ik_error_m']*1000:>12.3f}")

    max_dev_mm = result_traj["max_line_deviation_m"] * 1000.0
    final_err_mm = result_traj["final_error_m"] * 1000.0
    print(f"\nMax line deviation: {max_dev_mm:.3f} mm  (target: < 5.0 mm)")
    print(f"Final position error: {final_err_mm:.3f} mm")
    print(f"Final orientation error: {result_traj['final_error_deg']:.4f} deg")
    if max_dev_mm < 5.0:
        print("PASS: straight-line approach within 5 mm tolerance")
    else:
        print(f"FAIL: max deviation {max_dev_mm:.3f} mm > 5.0 mm limit")

    # ── 5 & 6. Accuracy tests: origin and table parking pose ─────────────────
    _header("SECTION 5: 10-POSE ACCURACY TEST")

    rng = np.random.default_rng(seed=42)

    # --- Run A: base at origin ---
    base_A = (0.0, 0.0, 0.0)
    model_A, data_A, cfg_A = _load_robot()
    poses_A = _sample_reachable_poses(model_A, data_A, cfg_A, 10, rng, base_A)

    rows: list = []
    n_pass_A = _run_accuracy_test(model_A, data_A, cfg_A, poses_A, base_A,
                                   "A_origin", rows)

    # --- Run B: base parked in front of Table B from the library layout ---
    # From library_layout.yaml: Table B is at pos [-0.5, 0.0, 0.0], depth 0.4 m
    # Parking: robot nose 5 cm from near edge of table, facing +Y (yaw = pi/2)
    # Near edge of Table B at y = 0.0 - 0.4/2 = -0.20 m
    # Robot nose (front bumper) = robot_x + chassis_hx = robot_x + 0.34
    # So robot_x + 0.34 + 0.05 = -0.20 → robot_x = -0.59
    # Actually Table B in layout: pos=[-0.5, 0.0] facing default.
    # Let's use a sensible parking pose: base at (-0.59, 0.0, pi/2)
    # meaning the robot is to the left of Table B facing +X (towards the table)
    table_B_x = -0.5   # from library_layout.yaml
    parking_x = table_B_x - 0.34 - 0.05  # chassis front at 5 cm gap from table
    base_B = (parking_x, 0.0, 0.0)
    print(f"\nParking pose for Table B approach: ({parking_x:.3f}, 0.0, yaw=0°)")
    print(f"  (Table B center at x={table_B_x} m; chassis half-length=0.34 m; gap=0.05 m)\n")

    model_B, data_B, cfg_B = _load_robot()
    poses_B = _sample_reachable_poses(model_B, data_B, cfg_B, 10, rng, base_B)

    n_pass_B = _run_accuracy_test(model_B, data_B, cfg_B, poses_B, base_B,
                                   "B_table_parking", rows)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "arm_accuracy_test.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults saved to: {csv_path}")

    # ── Overall summary ───────────────────────────────────────────────────────
    _header("OVERALL SUMMARY")
    print(f"Run A (origin):        {n_pass_A}/10 within 2 cm & 5 deg  {'PASS' if n_pass_A >= 9 else 'FAIL'}")
    print(f"Run B (table parking): {n_pass_B}/10 within 2 cm & 5 deg  {'PASS' if n_pass_B >= 9 else 'FAIL'}")

    overall = (n_pass_A >= 9) and (n_pass_B >= 9)
    print(f"\nOverall: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        import traceback
        traceback.print_exc()
        rc = 1
    finally:
        sys.stdout = _log._t
        _log.close()
    sys.exit(rc)
