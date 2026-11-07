"""Regression test: verifies vertical ground contact force equilibrium at rest.

Guarantees that when arm gravity compensation is off (default), the arm's weight
is genuinely transferred to the chassis and floor (total vertical force == total mass * 9.81),
preventing any gravcomp regression.
"""

import math
from pathlib import Path
import pytest
import numpy as np

mujoco = pytest.importorskip("mujoco", reason="Requires MuJoCo runtime DLL")

from library_robot.sim.mobile_model import load_mobile_robot_model, generate_mobile_models

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


def _measure_vertical_force(model: mujoco.MjModel, data: mujoco.MjData, arm_pose: str, arm_gravcomp: bool) -> float:
    grav_val = 1.0 if arm_gravcomp else 0.0
    for name in ARM_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            model.body_gravcomp[bid] = grav_val
    mujoco.mj_setConst(model, data)

    joint_targets = TUCKED_JOINTS if arm_pose == "tucked" else EXTENDED_JOINTS
    for j_name, target in joint_targets.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = target
            act_num = j_name.replace("joint", "")
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{act_num}")
            if act_id >= 0:
                data.ctrl[act_id] = target

    left_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_velocity")
    right_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_velocity")
    if left_act >= 0:
        data.ctrl[left_act] = 0.0
    if right_act >= 0:
        data.ctrl[right_act] = 0.0

    mujoco.mj_forward(model, data)

    # Settle 1.0 s (500 steps)
    for _ in range(500):
        mujoco.mj_step(model, data)

    ground_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    total_vertical_force = 0.0
    f_contact = np.zeros(6)

    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 != ground_geom_id and c.geom2 != ground_geom_id:
            continue
        mujoco.mj_contactForce(model, data, i, f_contact)
        R = c.frame.reshape(3, 3)
        f_world = R.T @ f_contact[:3]
        if c.geom1 == ground_geom_id:
            total_vertical_force += f_world[2]
        else:
            total_vertical_force -= f_world[2]

    return float(total_vertical_force)


def test_contact_force_equilibrium_gravcomp_off():
    """With gravcomp OFF, total vertical force must match total mass * 9.81 within 0.05 N."""
    generate_mobile_models()
    model, data, _ = load_mobile_robot_model(base_type="wheeled")

    total_mass = sum(model.body_mass[i] for i in range(model.nbody))
    expected_weight = total_mass * 9.81

    # Tucked pose
    m_tuck, d_tuck, _ = load_mobile_robot_model(base_type="wheeled")
    fz_tucked = _measure_vertical_force(m_tuck, d_tuck, arm_pose="tucked", arm_gravcomp=False)
    assert abs(fz_tucked - expected_weight) < 0.05, f"Tucked pose force mismatch: {fz_tucked} vs {expected_weight}"

    # Extended pose
    m_ext, d_ext, _ = load_mobile_robot_model(base_type="wheeled")
    fz_extended = _measure_vertical_force(m_ext, d_ext, arm_pose="extended", arm_gravcomp=False)
    assert abs(fz_extended - expected_weight) < 0.05, f"Extended pose force mismatch: {fz_extended} vs {expected_weight}"


def test_contact_force_equilibrium_gravcomp_on():
    """With gravcomp ON, total vertical force must match base mass * 9.81 within 0.05 N."""
    generate_mobile_models()
    model, data, _ = load_mobile_robot_model(base_type="wheeled")

    base_mass = sum(
        float(model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)])
        for name in BASE_BODIES
    )
    expected_base_weight = base_mass * 9.81

    m_tuck, d_tuck, _ = load_mobile_robot_model(base_type="wheeled")
    fz_tucked = _measure_vertical_force(m_tuck, d_tuck, arm_pose="tucked", arm_gravcomp=True)
    assert abs(fz_tucked - expected_base_weight) < 0.05, f"Base weight mismatch: {fz_tucked} vs {expected_base_weight}"
