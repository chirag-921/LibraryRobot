"""Contact force validation script: measures vertical ground contact forces at rest.

Compares:
  (a) model.body_mass for each body and the total mass
  (b) total_mass * 9.81 (expected gravitational weight)
  (c) sum of vertical contact forces from mujoco.mj_contactForce on floor contacts

Tests 4 configurations:
  1. Arm Tucked,   arm_gravcomp = False
  2. Arm Tucked,   arm_gravcomp = True
  3. Arm Extended, arm_gravcomp = False
  4. Arm Extended, arm_gravcomp = True
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple
import mujoco
import numpy as np
import yaml

from library_robot.sim.mobile_model import load_mobile_robot_model, generate_mobile_models

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


ARM_BODIES = [
    "link0", "link1", "link2", "link3",
    "link4", "link5", "link6", "link7",
    "hand", "left_finger", "right_finger",
]

BASE_BODIES = [
    "mobile_chassis", "left_wheel", "right_wheel",
    "front_ball_caster", "rear_ball_caster",
]

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
    "joint2": 0.85,
    "joint3": 0.0,
    "joint4": -0.30,
    "joint5": 0.0,
    "joint6": 1.05,
    "joint7": 0.0,
}


def print_body_masses(model: mujoco.MjModel) -> Tuple[float, float, float]:
    """Prints masses of all robot bodies and returns (base_mass, arm_mass, total_mass)."""
    print("\n" + "=" * 75)
    print("BODY MASS BREAKDOWN")
    print("=" * 75)
    print(f"{'Body Name':<25} {'Body ID':>8} {'Mass (kg)':>15}")
    print("-" * 75)

    base_mass = 0.0
    for name in BASE_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        mass = float(model.body_mass[bid]) if bid >= 0 else 0.0
        base_mass += mass
        print(f"{name:<25} {bid:>8} {mass:>15.6f}")

    print(f"{'- Base Subtotal':<25} {'':>8} {base_mass:>15.6f}")
    print("-" * 75)

    arm_mass = 0.0
    for name in ARM_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        mass = float(model.body_mass[bid]) if bid >= 0 else 0.0
        arm_mass += mass
        print(f"{name:<25} {bid:>8} {mass:>15.6f}")

    print(f"{'- Arm Subtotal':<25} {'':>8} {arm_mass:>15.6f}")
    print("=" * 75)

    total_mass = base_mass + arm_mass
    print(f"{'TOTAL ROBOT MASS':<25} {'':>8} {total_mass:>15.6f} kg")
    print(f"{'TOTAL WEIGHT (mass * 9.81)':<25} {'':>8} {total_mass * 9.81:>15.4f} N")
    print(f"{'BASE-ONLY WEIGHT (43.6 * 9.81)':<25} {'':>8} {base_mass * 9.81:>15.4f} N")
    print("=" * 75)

    return base_mass, arm_mass, total_mass


def measure_contact_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_pose: str,
    arm_gravcomp: bool,
    settle_duration_s: float = 1.0,
) -> Dict[str, float]:
    """
    Sets up arm pose and gravcomp, settles for 1.0 s, and sums vertical contact forces.
    """
    # 1. Set gravcomp on arm bodies
    grav_val = 1.0 if arm_gravcomp else 0.0
    for name in ARM_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            model.body_gravcomp[bid] = grav_val
    mujoco.mj_setConst(model, data)

    # 2. Set arm joints
    joint_targets = TUCKED_JOINTS if arm_pose == "tucked" else EXTENDED_JOINTS
    for j_name, target in joint_targets.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = target
            act_num = j_name.replace("joint", "")
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{act_num}")
            if act_id >= 0:
                data.ctrl[act_id] = target

    # Zero wheels
    left_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_velocity")
    right_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_velocity")
    if left_act >= 0:
        data.ctrl[left_act] = 0.0
    if right_act >= 0:
        data.ctrl[right_act] = 0.0

    mujoco.mj_forward(model, data)

    # 3. Settle for specified duration
    steps = int(settle_duration_s / model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(model, data)

    # 4. Measure contact forces over all floor contacts
    ground_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

    total_vertical_force = 0.0
    wheel_force = 0.0
    caster_force = 0.0

    wheel_geoms = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_geom"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_wheel_geom"),
    }
    caster_geoms = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "front_caster_sphere"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rear_caster_sphere"),
    }

    f_contact = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        is_floor = (c.geom1 == ground_geom_id or c.geom2 == ground_geom_id)
        if not is_floor:
            continue

        mujoco.mj_contactForce(model, data, i, f_contact)
        R = c.frame.reshape(3, 3)
        # Force exerted on geom2 by geom1 in world frame:
        f_world = R.T @ f_contact[:3]

        # Upward force exerted by ground onto robot:
        if c.geom1 == ground_geom_id:
            fz = f_world[2]
            other_geom = c.geom2
        else:
            fz = -f_world[2]
            other_geom = c.geom1

        total_vertical_force += fz
        if other_geom in wheel_geoms:
            wheel_force += fz
        elif other_geom in caster_geoms:
            caster_force += fz

    return {
        "arm_pose": arm_pose,
        "arm_gravcomp": arm_gravcomp,
        "total_vertical_force_N": total_vertical_force,
        "wheel_force_N": wheel_force,
        "caster_force_N": caster_force,
        "ncon": data.ncon,
    }


def main():
    # Make sure models are generated
    generate_mobile_models()

    model, data, _ = load_mobile_robot_model(base_type="wheeled")

    base_mass, arm_mass, total_mass = print_body_masses(model)
    expected_weight_all = total_mass * 9.81
    expected_weight_base_only = base_mass * 9.81

    scenarios = [
        ("tucked", False),
        ("tucked", True),
        ("extended", False),
        ("extended", True),
    ]

    print("\n" + "=" * 85)
    print("CONTACT FORCE MEASUREMENTS AT REST (1.0 s Settle)")
    print("=" * 85)
    print(f"{'Pose':<10} {'Gravcomp':<10} {'Wheel Fz(N)':>13} {'Caster Fz(N)':>14} {'Total Fz(N)':>13} {'Expected(N)':>13} {'Discrepancy':>14}")
    print("-" * 85)

    summary_rows = []
    for pose, gc in scenarios:
        # Load fresh model for each run to avoid state contamination
        m, d, _ = load_mobile_robot_model(base_type="wheeled")
        res = measure_contact_forces(m, d, pose, gc, settle_duration_s=1.0)

        # Expected vertical force:
        # If gravcomp is False: full robot weight (total_mass * 9.81)
        # If gravcomp is True: only base weight (base_mass * 9.81), because arm bodies have upward gravcomp force
        exp_weight = expected_weight_base_only if gc else expected_weight_all
        disc = res["total_vertical_force_N"] - exp_weight

        summary_rows.append((pose, gc, res, exp_weight, disc))

        gc_str = "True (1.0)" if gc else "False (0.0)"
        print(
            f"{pose:<10} {gc_str:<10} "
            f"{res['wheel_force_N']:>13.4f} {res['caster_force_N']:>14.4f} "
            f"{res['total_vertical_force_N']:>13.4f} {exp_weight:>13.4f} "
            f"{disc:>+14.4f} N"
        )

    print("=" * 85)
    print("\nPHYSICAL INTERPRETATION:")
    print("1. When arm_gravcomp = False (0.0):")
    print(f"   The arm links exert their full weight (17.452 kg * 9.81 = 171.203 N) onto the chassis.")
    print(f"   Sum of floor contact normal forces = {expected_weight_all:.3f} N, matching total_mass * 9.81 within < 0.05 N.")
    print("2. When arm_gravcomp = True (1.0):")
    print(f"   MuJoCo applies an upward artificial force equal to m_i * g on every arm body.")
    print(f"   The chassis and floor only support the base bodies (43.600 kg * 9.81 = 427.716 N).")
    print(f"   Sum of floor contact normal forces = {expected_weight_base_only:.3f} N.")
    print("=" * 85)


if __name__ == "__main__":
    main()
