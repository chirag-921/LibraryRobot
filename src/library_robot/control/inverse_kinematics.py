
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


def solve_position_ik(
    model,
    data,
    target_position,
    body_name="hand",
    max_iterations=500,
    tolerance=1e-4,
    damping=0.05,
    step_size=0.5,
    max_joint_step=0.15
):
    """
    Position-only inverse kinematics using
    damped least squares.

    Parameters
    ----------
    model : mujoco.MjModel
    data : mujoco.MjData
    target_position : array-like
        Desired world position [x, y, z], in metres.

    body_name : str
        Body whose position is being controlled.

    Returns
    -------
    dict:
        success
        iterations
        position
        error
        joint_positions
    """

    target_position = np.asarray(
        target_position,
        dtype=float
    )

    if target_position.shape != (3,):
        raise ValueError(
            "Target position must contain [x, y, z]."
        )

    # Find hand body
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name
    )

    if body_id == -1:
        raise ValueError(
            f"Body not found: {body_name}"
        )

    # Get joint IDs and addresses
    joint_ids = []
    qpos_addresses = []
    dof_addresses = []

    lower_limits = []
    upper_limits = []

    for name in ARM_JOINTS:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name
        )

        if joint_id == -1:
            raise ValueError(
                f"Joint not found: {name}"
            )

        joint_ids.append(joint_id)

        qpos_addresses.append(
            model.jnt_qposadr[joint_id]
        )

        dof_addresses.append(
            model.jnt_dofadr[joint_id]
        )

        lower, upper = model.jnt_range[joint_id]

        lower_limits.append(lower + 0.01)
        upper_limits.append(upper - 0.01)

    qpos_addresses = np.asarray(
        qpos_addresses,
        dtype=int
    )

    dof_addresses = np.asarray(
        dof_addresses,
        dtype=int
    )

    lower_limits = np.asarray(lower_limits)
    upper_limits = np.asarray(upper_limits)

    # Initial forward kinematics
    mujoco.mj_forward(model, data)

    success = False
    final_error = None

    # Iterative IK solver
    for iteration in range(max_iterations):

        # Current hand position
        current_position = data.xpos[
            body_id
        ].copy()

        # Position error
        error = target_position - current_position

        error_norm = np.linalg.norm(error)

        if error_norm < tolerance:

            success = True
            final_error = error_norm

            break

        # Full MuJoCo Jacobian
        jacp_full = np.zeros(
            (3, model.nv)
        )

        jacr_full = np.zeros(
            (3, model.nv)
        )

        mujoco.mj_jacBody(
            model,
            data,
            jacp_full,
            jacr_full,
            body_id
        )

        # Select only the seven arm joints
        J = jacp_full[:, dof_addresses]

        # Damped least-squares solution
        A = (
            J @ J.T
            + damping**2 * np.eye(3)
        )

        dq = J.T @ np.linalg.solve(
            A,
            error
        )

        # Reduce update size for stability
        dq *= step_size

        # Limit each joint update
        dq = np.clip(
            dq,
            -max_joint_step,
            max_joint_step
        )

        # Update seven joint positions
        current_q = data.qpos[
            qpos_addresses
        ].copy()

        new_q = current_q + dq

        # Respect Panda joint limits
        new_q = np.clip(
            new_q,
            lower_limits,
            upper_limits
        )

        data.qpos[qpos_addresses] = new_q

        # Recalculate forward kinematics
        mujoco.mj_forward(model, data)

        final_error = error_norm

    # Final position verification
    mujoco.mj_forward(model, data)

    final_position = data.xpos[
        body_id
    ].copy()

    final_error_vector = (
        target_position - final_position
    )

    final_error = np.linalg.norm(
        final_error_vector
    )

    success = final_error < tolerance

    # Extract final joint positions
    final_joint_positions = {}

    for i, name in enumerate(ARM_JOINTS):

        final_joint_positions[name] = float(
            data.qpos[qpos_addresses[i]]
        )

    return {
        "success": success,
        "iterations": iteration + 1,
        "position": final_position,
        "target": target_position,
        "error": final_error,
        "joint_positions": final_joint_positions
    }
