
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


def get_body_jacobian(model, data, body_name="hand"):
    """
    Calculate translational and rotational Jacobians
    for a specified MuJoCo body.

    Returns:
        Jp: 3x7 translational Jacobian
        Jr: 3x7 rotational Jacobian
    """

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name
    )

    if body_id == -1:
        raise ValueError(
            f"Body not found: {body_name}"
        )

    # MuJoCo Jacobians initially contain all DoFs.
    # Our model has 9 DoFs, including two fingers.
    jacp_full = np.zeros((3, model.nv))
    jacr_full = np.zeros((3, model.nv))

    mujoco.mj_jacBody(
        model,
        data,
        jacp_full,
        jacr_full,
        body_id
    )

    # Extract only the seven Panda arm DoFs.
    arm_dof_addresses = []

    for joint_name in ARM_JOINTS:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name
        )

        if joint_id == -1:
            raise ValueError(
                f"Joint not found: {joint_name}"
            )

        arm_dof_addresses.append(
            model.jnt_dofadr[joint_id]
        )

    Jp = jacp_full[:, arm_dof_addresses]
    Jr = jacr_full[:, arm_dof_addresses]

    return Jp, Jr


def numerical_position_jacobian(
    model,
    data,
    body_name="hand",
    epsilon=1e-6
):
    """
    Calculate translational Jacobian using
    central finite differences.

    Used to verify MuJoCo's analytical Jacobian.
    """

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name
    )

    if body_id == -1:
        raise ValueError(
            f"Body not found: {body_name}"
        )

    numerical_J = np.zeros((3, len(ARM_JOINTS)))

    original_qpos = data.qpos.copy()

    try:

        for column, joint_name in enumerate(ARM_JOINTS):

            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name
            )

            qpos_address = model.jnt_qposadr[joint_id]

            # Position at q + epsilon
            data.qpos[:] = original_qpos
            data.qpos[qpos_address] += epsilon

            mujoco.mj_forward(model, data)

            position_plus = data.xpos[body_id].copy()

            # Position at q - epsilon
            data.qpos[:] = original_qpos
            data.qpos[qpos_address] -= epsilon

            mujoco.mj_forward(model, data)

            position_minus = data.xpos[body_id].copy()

            # Central finite difference
            numerical_J[:, column] = (
                position_plus - position_minus
            ) / (2 * epsilon)

    finally:

        # Restore original robot configuration
        data.qpos[:] = original_qpos

        mujoco.mj_forward(model, data)

    return numerical_J
