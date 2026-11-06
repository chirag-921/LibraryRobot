"""Compile and smoke-test both mobile Panda MJCF variants using MuJoCo."""

import mujoco
import numpy as np
import yaml
from pathlib import Path

from library_robot.sim.mobile_model import load_mobile_robot_model


def named_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise AssertionError(f"Missing MuJoCo object: {name}")
    return object_id


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "robot.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as _stream:
    CONFIG = yaml.safe_load(_stream)


def test_variant(base_type):
    model, data, selected = load_mobile_robot_model(base_type=base_type)
    assert selected == base_type

    print(f"\n========== {base_type.upper()} MODEL ==========")
    print("nbody:", model.nbody)
    print("njnt:", model.njnt)
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)

    # Existing Panda arm/gripper joints and actuators must remain present.
    for name in [
        "joint1", "joint2", "joint3", "joint4",
        "joint5", "joint6", "joint7",
        "finger_joint1", "finger_joint2",
    ]:
        named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    for name in [f"actuator{i}" for i in range(1, 9)]:
        named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    if base_type == "wheeled":
        for name in [
            "base_freejoint",
            "left_wheel_hinge", "right_wheel_hinge",
            "front_caster_ball_joint", "rear_caster_ball_joint",
        ]:
            named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ["left_wheel_velocity", "right_wheel_velocity"]:
            aid = named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            speed_limit = float(CONFIG["base"]["wheeled"]["wheel_speed_limit"])
            assert np.allclose(model.actuator_ctrlrange[aid], [-speed_limit, speed_limit])
        assert model.nu == 10, f"Expected 10 actuators, found {model.nu}"
    else:
        for name in ["base_x", "base_y", "base_yaw"]:
            named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ["base_x_position", "base_y_position", "base_yaw_position"]:
            named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert model.nu == 11, f"Expected 11 actuators, found {model.nu}"

    # Panda home initialization should match the fixed Panda test setup.
    expected_home = dict(CONFIG["panda_home"]["arm_joints"])
    expected_home.update(dict(zip(
        ("finger_joint1", "finger_joint2"),
        CONFIG["panda_home"]["finger_positions"],
    )))
    for joint_name, expected in expected_home.items():
        jid = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qadr = model.jnt_qposadr[jid]
        assert abs(float(data.qpos[qadr]) - expected) < 1e-6, joint_name

    # Confirm finite state and that the model can advance without a runtime error.
    for _ in range(int(CONFIG["validation"]["simulation_steps"])):
        mujoco.mj_step(model, data)

    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))
    print(f"PASS: model compiled, Panda initialized, {int(CONFIG['validation']['simulation_steps'])} simulation steps completed.")


def main():
    test_variant("wheeled")
    test_variant("virtual")
    print("\nPASS: both mobile base variants passed the MuJoCo smoke test.")


if __name__ == "__main__":
    main()
