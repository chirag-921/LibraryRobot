"""Gravity-compensation diagnostic for the wheeled mobile Panda robot.

This script answers two key questions:
  1. With panda_gravity_compensation=0 (new default), what is the contact-force
     balance vs. expected weight?  Is the missing weight due to the gravcomp flag?
  2. After explicitly setting arm_gravcomp=1 at runtime, does the contact sum
     change, and by how much?

For each scenario (arm_gravcomp=False, arm_gravcomp=True) the script:
  A. Loads the wheeled model.
  B. Optionally sets body_gravcomp=1 for all Panda links.
  C. Prints body_gravcomp and body_mass for key bodies.
  D. Settles at TUCKED pose for 1 s, then measures vertical contact force sum.
  E. Settles at EXTENDED pose for 1 s, then measures vertical contact force sum.
  F. Prints discrepancy vs. total_mass * 9.81 and flags if |delta| > 1 N.

Ends with a comparison table.
"""

import sys
from pathlib import Path

import numpy as np
import mujoco
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from library_robot.sim.mobile_model import (
    generate_mobile_models,
    load_mobile_robot_model,
)

# ---------------------------------------------------------------------------
# Body-name discovery: Menagerie uses link0-link7 / hand / left_finger /
# right_finger (without panda_ prefix).  Try both naming conventions.
# ---------------------------------------------------------------------------
ARM_BODY_CANDIDATES = [
    # panda_ prefix first (script-13 style), then bare names
    "panda_link0", "link0",
    "panda_link1", "link1",
    "panda_link2", "link2",
    "panda_link3", "link3",
    "panda_link4", "link4",
    "panda_link5", "link5",
    "panda_link6", "link6",
    "panda_link7", "link7",
    "panda_hand",  "hand",
    "right_finger",
    "left_finger",
]

BASE_BODY_NAMES = [
    "mobile_chassis",
    "left_wheel",
    "right_wheel",
    "front_ball_caster",
    "rear_ball_caster",
]


def discover_arm_bodies(model: mujoco.MjModel):
    """Return list of (name, body_id) for every unique arm body found."""
    seen_ids = set()
    result = []
    for name in ARM_BODY_CANDIDATES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0 and bid not in seen_ids:
            seen_ids.add(bid)
            result.append((name, bid))
    return result


def total_robot_mass(model: mujoco.MjModel):
    """Sum body_mass for all base + arm bodies; return (total, name_mass_pairs)."""
    pairs = []
    total = 0.0

    # Base bodies
    for name in BASE_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        m = float(model.body_mass[bid]) if bid >= 0 else 0.0
        pairs.append((name, bid, m))
        total += m

    # Arm bodies (discovered)
    for name, bid in discover_arm_bodies(model):
        m = float(model.body_mass[bid])
        pairs.append((name, bid, m))
        total += m

    return total, pairs


def measure_vertical_contact_force(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """
    Sum vertical (world-Z) contact forces for all contacts involving the ground geom.

    The contact frame is stored row-major in data.contact[i].frame (shape=9).
    Row 0 = contact normal direction (world frame).
    Force in contact frame: f_arr[0] is along the normal.
    World-frame force = frame.T @ f_contact[:3]
    We sum the Z-component of world-frame force for ground contacts.
    """
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    f_arr = np.zeros(6)
    total_z = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 != ground_id and c.geom2 != ground_id:
            continue
        mujoco.mj_contactForce(model, data, i, f_arr)
        # frame is 3x3, row-major → shape (9,)
        frame = np.array(c.frame).reshape(3, 3)
        # world_force = frame.T @ f_contact[:3]
        world_force = frame.T @ f_arr[:3]
        total_z += world_force[2]
    return float(total_z)


def settle_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_targets: dict,
    duration_s: float = 1.0,
):
    """Command arm joints to target_positions, then step for duration_s."""
    for j_name, target in joint_targets.items():
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        if j_id >= 0:
            data.qpos[model.jnt_qposadr[j_id]] = target
            act_num = j_name.replace("joint", "")
            act_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{act_num}"
            )
            if act_id >= 0:
                data.ctrl[act_id] = target

    # Zero wheel commands
    for aname in ("left_wheel_velocity", "right_wheel_velocity"):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        if aid >= 0:
            data.ctrl[aid] = 0.0

    mujoco.mj_forward(model, data)
    steps = max(1, int(duration_s / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)


TUCKED_JOINTS = {
    "joint1": 0.0,
    "joint2": -0.785398,
    "joint3": 0.0,
    "joint4": -2.356194,
    "joint5": 0.0,
    "joint6": 1.570796,
    "joint7": 0.785398,
}

EXTENDED_JOINTS = {
    "joint1": 0.0,
    "joint2": -0.3,
    "joint3": 0.0,
    "joint4": -2.0,
    "joint5": 0.0,
    "joint6": 1.9,
    "joint7": 0.785,
}


def run_scenario(arm_gravcomp_on: bool) -> dict:
    """
    Load model, optionally enable arm gravcomp, measure contact forces
    for tucked and extended poses.
    """
    label = "arm_gravcomp=TRUE" if arm_gravcomp_on else "arm_gravcomp=FALSE"
    print(f"\n{'='*70}")
    print(f"SCENARIO: {label}")
    print(f"{'='*70}")

    model, data, _ = load_mobile_robot_model(base_type="wheeled")

    # Optionally enable arm gravcomp
    arm_bodies = discover_arm_bodies(model)
    if arm_gravcomp_on:
        for name, bid in arm_bodies:
            model.body_gravcomp[bid] = 1.0
        mujoco.mj_setConst(model, data)

    # Print body_gravcomp for arm bodies
    print(f"\n[{label}] body_gravcomp for arm bodies:")
    for name, bid in arm_bodies:
        gc = float(model.body_gravcomp[bid])
        print(f"  {name:20s} (id={bid:3d})  gravcomp={gc:.1f}")

    # Compute total mass
    total_mass, mass_pairs = total_robot_mass(model)
    print(f"\n[{label}] body_mass breakdown:")
    for name, bid, m in mass_pairs:
        print(f"  {name:30s} (id={bid if bid >= 0 else 'N/A':>3})  mass={m:.4f} kg")
    print(f"  {'TOTAL':30s}         mass={total_mass:.4f} kg")
    expected_N = total_mass * 9.81
    print(f"  Expected weight (mass*9.81) = {expected_N:.4f} N")

    # ---- TUCKED pose ----
    print(f"\n[{label}] Settling TUCKED pose (1.0 s)...")
    settle_pose(model, data, TUCKED_JOINTS, duration_s=1.0)
    contact_z_tucked = measure_vertical_contact_force(model, data)
    discrepancy_tucked = expected_N - contact_z_tucked
    flag_t = " <<< |delta| > 1 N" if abs(discrepancy_tucked) > 1.0 else ""
    print(f"  Vertical contact force (TUCKED):   {contact_z_tucked:.4f} N")
    print(f"  Expected weight:                    {expected_N:.4f} N")
    print(f"  Discrepancy:                        {discrepancy_tucked:.4f} N{flag_t}")

    # ---- EXTENDED pose ----
    print(f"\n[{label}] Settling EXTENDED pose (1.0 s)...")
    # Reload fresh model for extended pose to avoid pose coupling
    model2, data2, _ = load_mobile_robot_model(base_type="wheeled")
    if arm_gravcomp_on:
        for name, bid in discover_arm_bodies(model2):
            model2.body_gravcomp[bid] = 1.0
        mujoco.mj_setConst(model2, data2)
    settle_pose(model2, data2, EXTENDED_JOINTS, duration_s=1.0)
    contact_z_extended = measure_vertical_contact_force(model2, data2)
    discrepancy_extended = expected_N - contact_z_extended
    flag_e = " <<< |delta| > 1 N" if abs(discrepancy_extended) > 1.0 else ""
    print(f"  Vertical contact force (EXTENDED): {contact_z_extended:.4f} N")
    print(f"  Expected weight:                    {expected_N:.4f} N")
    print(f"  Discrepancy:                        {discrepancy_extended:.4f} N{flag_e}")

    return {
        "label": label,
        "arm_gravcomp_on": arm_gravcomp_on,
        "total_mass_kg": total_mass,
        "expected_N": expected_N,
        "contact_z_tucked_N": contact_z_tucked,
        "discrepancy_tucked_N": discrepancy_tucked,
        "contact_z_extended_N": contact_z_extended,
        "discrepancy_extended_N": discrepancy_extended,
    }


def main():
    print("=" * 70)
    print("GRAVITY COMPENSATION DIAGNOSTIC: check_gravcomp.py")
    print("=" * 70)

    # Step A: Load config
    config_path = REPO_ROOT / "configs" / "robot.yaml"
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    gc_config = float(cfg.get("base", {}).get("panda_gravity_compensation", 0.0))
    print(f"\nConfig panda_gravity_compensation = {gc_config}")

    # Step B: Regenerate model XML
    print("\nRegenerating wheeled model XML...")
    generated = generate_mobile_models()
    for btype, path in generated.items():
        print(f"  Generated: {path}")

    # Step C: Run both scenarios
    result_off = run_scenario(arm_gravcomp_on=False)
    result_on  = run_scenario(arm_gravcomp_on=True)

    # Step D: Comparison table
    print("\n")
    print("=" * 90)
    print("COMPARISON TABLE: gravcomp_true vs gravcomp_false  |  tucked vs extended")
    print("=" * 90)
    hdr = (
        f"{'Scenario':<28} | {'Total mass (kg)':>15} | {'Expected (N)':>12} | "
        f"{'Contact (N) tucked':>18} | {'Delta tucked (N)':>16} | "
        f"{'Contact (N) ext':>15} | {'Delta ext (N)':>14}"
    )
    print(hdr)
    print("-" * 130)
    for r in [result_off, result_on]:
        flag_t = "!" if abs(r["discrepancy_tucked_N"]) > 1.0 else " "
        flag_e = "!" if abs(r["discrepancy_extended_N"]) > 1.0 else " "
        print(
            f"{r['label']:<28} | {r['total_mass_kg']:>15.4f} | {r['expected_N']:>12.4f} | "
            f"{r['contact_z_tucked_N']:>18.4f} | {r['discrepancy_tucked_N']:>15.4f}{flag_t} | "
            f"{r['contact_z_extended_N']:>15.4f} | {r['discrepancy_extended_N']:>13.4f}{flag_e}"
        )
    print("=" * 90)
    print("NOTE: '!' = |discrepancy| > 1 N (force balance error)")
    print("NOTE: Positive discrepancy => contact force LESS than weight (some mass levitated by gravcomp).")
    print("NOTE: gravcomp=TRUE arms are supported by joint actuators, not by contact => contact sum < weight.")


if __name__ == "__main__":
    main()
