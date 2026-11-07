"""Headless validation suite for the wheeled mobile base in MuJoCo.

Executes four physical tests on the wheeled mobile robot:
  1. Motor Direction Test (forward drive and turning sign conventions)
  2. Straight-Line Open-Loop Drive Test (distance, lateral drift, heading drift)
  3. In-Place Open-Loop Turn Test (yaw rate, turn angle, position drift)
  4. Tip & Stability Test (extended arm + 0.5 kg payload under accel/brake/turn)

Saves results to:
  - results/wheeled_base_validation.json
  - results/wheeled_base_validation.csv
  - logs/wheeled_base_validation.txt
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mujoco
import numpy as np
import yaml

from library_robot.sim.mobile_model import load_mobile_robot_model


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "wheeled_validation.yaml"

if CONFIG_PATH.exists():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {}


def get_cfg(section: str, key: str, default: Any) -> Any:
    """Retrieve config parameter with fallback."""
    return CONFIG.get(section, {}).get(key, default)


def quat_to_euler(w: float, x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert quaternion [w, x, y, z] to roll, pitch, yaw in radians."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def get_base_pose(data: mujoco.MjData) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Return base (x, y, z) position and (roll, pitch, yaw) orientation."""
    pos = data.qpos[:3].copy()
    w, x, y, z = data.qpos[3:7]
    euler = quat_to_euler(w, x, y, z)
    return pos, euler


def check_wheel_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    wheel_geom_id: int,
    ground_geom_id: int,
) -> bool:
    """Check if specified wheel geom has an active collision contact with ground."""
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 == wheel_geom_id and c.geom2 == ground_geom_id) or (
            c.geom2 == wheel_geom_id and c.geom1 == ground_geom_id
        ):
            return True
    return False


def setup_fresh_model() -> Tuple[mujoco.MjModel, mujoco.MjData, int, int, int, int, int]:
    """Load wheeled model, reset to home, and return key object IDs."""
    model, data, base_type = load_mobile_robot_model(base_type="wheeled")
    assert base_type == "wheeled", "Expected wheeled base model"

    left_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_velocity")
    right_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_velocity")
    left_wheel_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_geom")
    right_wheel_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_wheel_geom")
    ground_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

    if any(idx < 0 for idx in [left_act, right_act, left_wheel_geom, right_wheel_geom, ground_geom]):
        raise RuntimeError("Missing required actuator or geom in wheeled model")

    return model, data, left_act, right_act, left_wheel_geom, right_wheel_geom, ground_geom


def settle_robot(model: mujoco.MjModel, data: mujoco.MjData, duration_s: float = 0.10) -> None:
    """Step physics with zero wheel command to let the base settle on floor."""
    steps = max(1, int(duration_s / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)


# ==============================================================================
# TEST A: MOTOR DIRECTION TEST
# ==============================================================================

def test_motor_direction() -> Dict[str, Any]:
    print("\n--- Test A: Motor Direction Test ---")
    model, data, left_act, right_act, _, _, _ = setup_fresh_model()
    settle_time_s = get_cfg("general", "settle_time_s", 0.10)
    wheel_speed = get_cfg("motor_direction", "wheel_speed", 2.0)
    duration_s = get_cfg("motor_direction", "duration_s", 1.0)
    min_forward_dist = get_cfg("motor_direction", "min_forward_dist_m", 0.05)
    min_turn_angle = get_cfg("motor_direction", "min_turn_angle_rad", 0.10)

    # 1. Forward motion test: equal positive commands (+wheel_speed)
    settle_robot(model, data, settle_time_s)
    p0, (roll0, pitch0, yaw0) = get_base_pose(data)

    steps = int(duration_s / model.opt.timestep)
    data.ctrl[left_act] = wheel_speed
    data.ctrl[right_act] = wheel_speed
    for _ in range(steps):
        mujoco.mj_step(model, data)

    p1, (_, _, _) = get_base_pose(data)
    fwd_displacement_x = p1[0] - p0[0]
    lateral_disp_y = p1[1] - p0[1]

    # 2. Counter-clockwise turn test: left = -wheel_speed, right = +wheel_speed
    model2, data2, left_act2, right_act2, _, _, _ = setup_fresh_model()
    settle_robot(model2, data2, settle_time_s)
    _, (_, _, yaw_start) = get_base_pose(data2)

    data2.ctrl[left_act2] = -wheel_speed
    data2.ctrl[right_act2] = +wheel_speed
    for _ in range(steps):
        mujoco.mj_step(model2, data2)

    _, (_, _, yaw_end) = get_base_pose(data2)
    turn_angle_ccw = yaw_end - yaw_start

    fwd_pass = bool(fwd_displacement_x > min_forward_dist and abs(lateral_disp_y) < 0.05)
    turn_pass = bool(turn_angle_ccw > min_turn_angle)
    passed = fwd_pass and turn_pass

    status = "PASS" if passed else "FAIL"
    print(f"  Forward displacement (dx): {fwd_displacement_x:.4f} m (threshold: > {min_forward_dist:.4f} m)")
    print(f"  Lateral displacement (dy): {lateral_disp_y:.6f} m")
    print(f"  CCW Turn angle (dyaw):     {turn_angle_ccw:.4f} rad ({math.degrees(turn_angle_ccw):.2f} deg) (threshold: > {min_turn_angle:.4f} rad)")
    print(f"  Sign convention:           Positive (+u, +u) -> +X forward; Differential (-u, +u) -> +Z counter-clockwise yaw")
    print(f"[{status}] Motor Direction Test")

    return {
        "test": "motor_direction",
        "passed": passed,
        "forward_displacement_x_m": float(fwd_displacement_x),
        "lateral_displacement_y_m": float(lateral_disp_y),
        "turn_angle_ccw_rad": float(turn_angle_ccw),
        "sign_convention": "+u,+u drives +X; -u,+u turns +yaw (CCW)",
    }


# ==============================================================================
# TEST B: STRAIGHT-LINE OPEN-LOOP DRIVE TEST
# ==============================================================================

def test_straight_drive() -> Dict[str, Any]:
    print("\n--- Test B: Straight Drive Test ---")
    model, data, left_act, right_act, _, _, _ = setup_fresh_model()
    settle_time_s = get_cfg("general", "settle_time_s", 0.10)
    wheel_speed = get_cfg("straight_drive", "wheel_speed", 3.0)
    duration_s = get_cfg("straight_drive", "duration_s", 2.0)
    min_dist = get_cfg("straight_drive", "min_forward_dist_m", 0.35)
    max_lat_drift = get_cfg("straight_drive", "max_lateral_drift_m", 0.03)
    max_heading_drift = get_cfg("straight_drive", "max_heading_drift_rad", 0.02)

    settle_robot(model, data, settle_time_s)
    p0, (_, _, yaw0) = get_base_pose(data)

    steps = int(duration_s / model.opt.timestep)
    data.ctrl[left_act] = wheel_speed
    data.ctrl[right_act] = wheel_speed
    for _ in range(steps):
        mujoco.mj_step(model, data)

    p1, (_, _, yaw1) = get_base_pose(data)
    dist_travelled = p1[0] - p0[0]
    lateral_drift = abs(p1[1] - p0[1])
    heading_drift = abs(yaw1 - yaw0)

    passed = bool(
        dist_travelled >= min_dist
        and lateral_drift <= max_lat_drift
        and heading_drift <= max_heading_drift
    )
    status = "PASS" if passed else "FAIL"

    print(f"  Distance travelled: {dist_travelled:.4f} m (threshold: >= {min_dist:.4f} m)")
    print(f"  Lateral drift:      {lateral_drift:.6f} m (threshold: <= {max_lat_drift:.4f} m)")
    print(f"  Heading drift:      {heading_drift:.6f} rad ({math.degrees(heading_drift):.4f} deg) (threshold: <= {max_heading_drift:.4f} rad)")
    print(f"[{status}] Straight Drive Test")

    return {
        "test": "straight_drive",
        "passed": passed,
        "distance_travelled_m": float(dist_travelled),
        "lateral_drift_m": float(lateral_drift),
        "heading_drift_rad": float(heading_drift),
    }


# ==============================================================================
# TEST C: IN-PLACE OPEN-LOOP TURN TEST
# ==============================================================================

def test_inplace_turn() -> Dict[str, Any]:
    print("\n--- Test C: In-Place Turn Test ---")
    model, data, left_act, right_act, _, _, _ = setup_fresh_model()
    settle_time_s = get_cfg("general", "settle_time_s", 0.10)
    wheel_speed = get_cfg("inplace_turn", "wheel_speed", 2.0)
    duration_s = get_cfg("inplace_turn", "duration_s", 2.0)
    min_turn_angle = get_cfg("inplace_turn", "min_turn_angle_rad", 0.80)
    max_pos_drift = get_cfg("inplace_turn", "max_position_drift_m", 0.02)

    settle_robot(model, data, settle_time_s)
    p0, (_, _, yaw0) = get_base_pose(data)

    steps = int(duration_s / model.opt.timestep)
    data.ctrl[left_act] = -wheel_speed
    data.ctrl[right_act] = +wheel_speed
    for _ in range(steps):
        mujoco.mj_step(model, data)

    p1, (_, _, yaw1) = get_base_pose(data)
    total_yaw = yaw1 - yaw0
    yaw_rate = total_yaw / duration_s
    pos_drift = math.hypot(p1[0] - p0[0], p1[1] - p0[1])

    passed = bool(total_yaw >= min_turn_angle and pos_drift <= max_pos_drift)
    status = "PASS" if passed else "FAIL"

    print(f"  Total turn angle:  {total_yaw:.4f} rad ({math.degrees(total_yaw):.2f} deg) (threshold: >= {min_turn_angle:.4f} rad)")
    print(f"  Average yaw rate:  {yaw_rate:.4f} rad/s ({math.degrees(yaw_rate):.2f} deg/s)")
    print(f"  Position drift:    {pos_drift:.6f} m (threshold: <= {max_pos_drift:.4f} m)")
    print(f"[{status}] In-Place Turn Test")

    return {
        "test": "inplace_turn",
        "passed": passed,
        "total_turn_rad": float(total_yaw),
        "yaw_rate_rad_s": float(yaw_rate),
        "position_drift_m": float(pos_drift),
    }


# ==============================================================================
# TEST D: TIP AND DYNAMIC STABILITY TEST
# ==============================================================================

def test_tip_and_stability() -> Dict[str, Any]:
    print("\n--- Test D: Tip & Stability Test ---")
    model, data, left_act, right_act, left_wheel_geom, right_wheel_geom, ground_geom = setup_fresh_model()

    payload_mass = get_cfg("tip_test", "payload_mass_kg", 0.50)
    extended_joints = get_cfg(
        "tip_test",
        "extended_arm_joints",
        {"joint1": 0.0, "joint2": 0.50, "joint4": -1.00, "joint6": 1.50},
    )
    drive_speed = get_cfg("tip_test", "drive_accel_speed", 4.0)
    drive_duration_s = get_cfg("tip_test", "drive_duration_s", 1.0)
    brake_duration_s = get_cfg("tip_test", "brake_duration_s", 0.5)
    turn_speed = get_cfg("tip_test", "turn_speed", 3.0)
    turn_duration_s = get_cfg("tip_test", "turn_duration_s", 1.0)
    post_brake_duration_s = get_cfg("tip_test", "post_turn_brake_duration_s", 0.5)
    max_pitch_thresh = get_cfg("tip_test", "max_pitch_rad", 0.10)
    max_roll_thresh = get_cfg("tip_test", "max_roll_rad", 0.10)

    # Attach book payload to hand body
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if hand_id < 0:
        raise RuntimeError("Hand body not found in model")
    model.body_mass[hand_id] += payload_mass
    mujoco.mj_setConst(model, data)

    # Position arm into extended forward-reaching posture
    for j_name, target_pos in extended_joints.items():
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        if j_id >= 0:
            data.qpos[model.jnt_qposadr[j_id]] = target_pos
            act_num = j_name.replace("joint", "")
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{act_num}")
            if act_id >= 0:
                data.ctrl[act_id] = target_pos

    mujoco.mj_forward(model, data)
    settle_robot(model, data, 0.10)

    # Hand position debug
    chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mobile_chassis")
    hand_reach = data.xpos[hand_id] - data.xpos[chassis_id]
    print(f"  Extended arm reach relative to chassis: X={hand_reach[0]:.3f} m, Z={hand_reach[2]:.3f} m")
    print(f"  Payload mass attached to gripper:       {payload_mass:.2f} kg")

    max_roll = 0.0
    max_pitch = 0.0
    all_contacts_maintained = True
    flipped = False

    def monitor_step():
        nonlocal max_roll, max_pitch, all_contacts_maintained, flipped
        mujoco.mj_step(model, data)
        _, (roll, pitch, _) = get_base_pose(data)
        max_roll = max(max_roll, abs(roll))
        max_pitch = max(max_pitch, abs(pitch))

        # Check for catastrophic flip (pitch or roll > 70 degrees / 1.2 rad)
        if abs(roll) > 1.2 or abs(pitch) > 1.2:
            flipped = True

        # Check wheel ground contact
        left_contact = check_wheel_contact(model, data, left_wheel_geom, ground_geom)
        right_contact = check_wheel_contact(model, data, right_wheel_geom, ground_geom)
        if not (left_contact and right_contact):
            all_contacts_maintained = False

    # 1. Drive forward aggressively
    data.ctrl[left_act] = drive_speed
    data.ctrl[right_act] = drive_speed
    for _ in range(int(drive_duration_s / model.opt.timestep)):
        monitor_step()

    # 2. Aggressive brake
    data.ctrl[left_act] = 0.0
    data.ctrl[right_act] = 0.0
    for _ in range(int(brake_duration_s / model.opt.timestep)):
        monitor_step()

    # 3. Aggressive in-place turn
    data.ctrl[left_act] = -turn_speed
    data.ctrl[right_act] = +turn_speed
    for _ in range(int(turn_duration_s / model.opt.timestep)):
        monitor_step()

    # 4. Final brake / settle
    data.ctrl[left_act] = 0.0
    data.ctrl[right_act] = 0.0
    for _ in range(int(post_brake_duration_s / model.opt.timestep)):
        monitor_step()

    passed = bool(
        not flipped
        and max_roll <= max_roll_thresh
        and max_pitch <= max_pitch_thresh
        and all_contacts_maintained
    )
    status = "PASS" if passed else "FAIL"

    print(f"  Max roll recorded:  {max_roll:.6f} rad ({math.degrees(max_roll):.3f} deg) (threshold: <= {max_roll_thresh:.4f} rad)")
    print(f"  Max pitch recorded: {max_pitch:.6f} rad ({math.degrees(max_pitch):.3f} deg) (threshold: <= {max_pitch_thresh:.4f} rad)")
    print(f"  Chassis flipped:    {flipped}")
    print(f"  Wheel floor contact: {'Maintained throughout' if all_contacts_maintained else 'Contact interrupted'}")
    print(f"[{status}] Tip & Stability Test")

    return {
        "test": "tip_and_stability",
        "passed": passed,
        "payload_mass_kg": float(payload_mass),
        "max_roll_rad": float(max_roll),
        "max_pitch_rad": float(max_pitch),
        "flipped": flipped,
        "all_wheel_contacts_maintained": all_contacts_maintained,
    }


def main():
    print("=" * 60)
    print("PHASE 3: WHEELED BASE PHYSICAL VALIDATION SUITE")
    print("=" * 60)

    results = [
        test_motor_direction(),
        test_straight_drive(),
        test_inplace_turn(),
        test_tip_and_stability(),
    ]

    all_passed = all(r["passed"] for r in results)

    # Save JSON results
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "wheeled_base_validation.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"overall_pass": all_passed, "tests": results}, f, indent=2)

    # Save CSV results
    csv_path = results_dir / "wheeled_base_validation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Test", "Status", "Details"])
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            details = "; ".join(f"{k}={v}" for k, v in r.items() if k not in ("test", "passed"))
            writer.writerow([r["test"], status, details])

    # Save Text Summary in logs/
    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "wheeled_base_validation.txt"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("PHASE 3: WHEELED BASE PHYSICAL VALIDATION SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Overall Status: {'PASS' if all_passed else 'FAIL'}\n\n")
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            f.write(f"[{status}] {r['test']}\n")
            for k, v in r.items():
                if k not in ("test", "passed"):
                    f.write(f"    {k}: {v}\n")
            f.write("\n")

    print("\n" + "=" * 60)
    print(f"OVERALL VALIDATION RESULT: {'ALL PASS' if all_passed else 'ONE OR MORE FAILED'}")
    print(f"Results saved to:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - {log_path}")
    print("=" * 60)

    if not all_passed:
        exit(1)


if __name__ == "__main__":
    main()
