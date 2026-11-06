
import mujoco
import numpy as np


ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]

GRIPPER_JOINTS = [
    "finger_joint1",
    "finger_joint2",
]


def get_joint_positions(model, data, joint_names):
    """
    Read joint positions.

    Returns a dictionary containing joint angles
    in radians or slide positions in metres.
    """

    positions = {}

    for name in joint_names:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name
        )

        if joint_id == -1:
            raise ValueError(f"Joint not found: {name}")

        qpos_address = model.jnt_qposadr[joint_id]

        positions[name] = float(
            data.qpos[qpos_address]
        )

    return positions


def get_joint_velocities(model, data, joint_names):
    """Read joint velocities."""

    velocities = {}

    for name in joint_names:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name
        )

        if joint_id == -1:
            raise ValueError(f"Joint not found: {name}")

        dof_address = model.jnt_dofadr[joint_id]

        velocities[name] = float(
            data.qvel[dof_address]
        )

    return velocities


def get_body_pose(model, data, body_name="hand"):
    """
    Read a body's world position and orientation.

    Position: metres
    Rotation matrix: 3 x 3
    Quaternion: [w, x, y, z]
    """

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name
    )

    if body_id == -1:
        raise ValueError(f"Body not found: {body_name}")

    position = data.xpos[body_id].copy()

    rotation = data.xmat[body_id].reshape(
        3, 3
    ).copy()

    quaternion = data.xquat[body_id].copy()

    return {
        "position": position,
        "rotation": rotation,
        "quaternion": quaternion
    }


def get_robot_state(model, data):
    """Read the current Panda robot state."""

    arm_positions = get_joint_positions(
        model, data, ARM_JOINTS
    )

    arm_velocities = get_joint_velocities(
        model, data, ARM_JOINTS
    )

    gripper_positions = get_joint_positions(
        model, data, GRIPPER_JOINTS
    )

    hand_pose = get_body_pose(
        model, data, "hand"
    )

    return {
        "arm_positions": arm_positions,
        "arm_velocities": arm_velocities,
        "gripper_positions": gripper_positions,
        "hand_pose": hand_pose
    }
