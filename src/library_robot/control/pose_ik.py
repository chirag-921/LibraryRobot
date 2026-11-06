
import mujoco
import numpy as np

from scipy.spatial.transform import Rotation


ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]


def solve_pose_ik(
    model,
    data,
    target_position,
    target_rotation,
    body_name="hand",
    max_iterations=800,
    position_tolerance=1e-4,
    orientation_tolerance=1e-3,
    damping=0.05,
    step_size=0.5,
    max_joint_step=0.10,
    orientation_weight=0.1
):
    """
    Six-dimensional damped least-squares IK.

    Controls:
        1. Hand position (x, y, z)
        2. Hand orientation (rotation matrix)

    target_rotation must be a 3x3 rotation matrix.
    """

    target_position = np.asarray(
        target_position,
        dtype=float
    )

    target_rotation = np.asarray(
        target_rotation,
        dtype=float
    )

    if target_position.shape != (3,):
        raise ValueError(
            "Target position must have shape (3,)."
        )

    if target_rotation.shape != (3, 3):
        raise ValueError(
            "Target rotation must have shape (3, 3)."
        )

    # Validate rotation matrix
    if not np.allclose(
        target_rotation.T @ target_rotation,
        np.eye(3),
        atol=1e-5
    ):
        raise ValueError(
            "Target rotation must be orthogonal."
        )

    if not np.isclose(
        np.linalg.det(target_rotation),
        1.0,
        atol=1e-5
    ):
        raise ValueError(
            "Target rotation must have determinant +1."
        )

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name
    )

    if body_id == -1:
        raise ValueError(
            f"Body not found: {body_name}"
        )

    # Identify the seven arm DoFs
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

        qpos_addresses.append(
            model.jnt_qposadr[joint_id]
        )

        dof_addresses.append(
            model.jnt_dofadr[joint_id]
        )

        lower, upper = model.jnt_range[joint_id]

        lower_limits.append(lower + 0.01)
        upper_limits.append(upper - 0.01)

    qpos_addresses = np.asarray(qpos_addresses)
    dof_addresses = np.asarray(dof_addresses)

    lower_limits = np.asarray(lower_limits)
    upper_limits = np.asarray(upper_limits)

    success = False

    position_error_norm = float("inf")
    orientation_error_norm = float("inf")

    for iteration in range(max_iterations):

        mujoco.mj_forward(model, data)

        # Current position
        current_position = data.xpos[
            body_id
        ].copy()

        # Current orientation
        current_rotation = data.xmat[
            body_id
        ].reshape(3, 3).copy()

        # Position error
        position_error = (
            target_position - current_position
        )

        # Orientation error in world coordinates
        rotation_difference = (
            target_rotation @ current_rotation.T
        )

        orientation_error = (
            Rotation.from_matrix(
                rotation_difference
            ).as_rotvec()
        )

        position_error_norm = np.linalg.norm(
            position_error
        )

        orientation_error_norm = np.linalg.norm(
            orientation_error
        )

        # Check both position and orientation
        if (
            position_error_norm < position_tolerance
            and
            orientation_error_norm < orientation_tolerance
        ):

            success = True
            break

        # Calculate MuJoCo Jacobians
        jacp_full = np.zeros((3, model.nv))
        jacr_full = np.zeros((3, model.nv))

        mujoco.mj_jacBody(
            model,
            data,
            jacp_full,
            jacr_full,
            body_id
        )

        # Keep only the seven arm DoFs
        Jp = jacp_full[:, dof_addresses]
        Jr = jacr_full[:, dof_addresses]

        # Combined weighted 6D Jacobian
        J = np.vstack((
            Jp,
            orientation_weight * Jr
        ))

        error = np.concatenate((
            position_error,
            orientation_weight * orientation_error
        ))

        # Damped least-squares solution
        A = (
            J @ J.T
            + damping**2 * np.eye(6)
        )

        dq = J.T @ np.linalg.solve(
            A,
            error
        )

        dq *= step_size

        dq = np.clip(
            dq,
            -max_joint_step,
            max_joint_step
        )

        current_q = data.qpos[
            qpos_addresses
        ].copy()

        new_q = np.clip(
            current_q + dq,
            lower_limits,
            upper_limits
        )

        data.qpos[qpos_addresses] = new_q

    # Final verification
    mujoco.mj_forward(model, data)

    final_position = data.xpos[
        body_id
    ].copy()

    final_rotation = data.xmat[
        body_id
    ].reshape(3, 3).copy()

    final_position_error = np.linalg.norm(
        target_position - final_position
    )

    final_rotation_error = Rotation.from_matrix(
        target_rotation @ final_rotation.T
    ).magnitude()

    final_joints = {}

    for i, name in enumerate(ARM_JOINTS):
        final_joints[name] = float(
            data.qpos[qpos_addresses[i]]
        )

    success = (
        final_position_error < position_tolerance
        and
        final_rotation_error < orientation_tolerance
    )

    return {
        "success": success,
        "iterations": iteration + 1,
        "position": final_position,
        "rotation": final_rotation,
        "position_error": final_position_error,
        "orientation_error": final_rotation_error,
        "joint_positions": final_joints
    }
