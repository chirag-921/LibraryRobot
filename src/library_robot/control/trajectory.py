
import numpy as np

from library_robot.control.inverse_kinematics import (
    solve_position_ik
)


def generate_linear_cartesian_trajectory(
    model,
    ik_data,
    start_position,
    target_position,
    num_waypoints=20,
    body_name="hand",
    tolerance=1e-4
):
    """
    Generate a straight-line Cartesian path.

    Each Cartesian waypoint is converted into
    seven Panda joint targets using sequential IK.

    The previous waypoint's solution is used as
    the initial configuration for the next one.
    """

    start_position = np.asarray(
        start_position,
        dtype=float
    )

    target_position = np.asarray(
        target_position,
        dtype=float
    )

    if start_position.shape != (3,):
        raise ValueError(
            "Start position must be [x, y, z]."
        )

    if target_position.shape != (3,):
        raise ValueError(
            "Target position must be [x, y, z]."
        )

    if num_waypoints < 1:
        raise ValueError(
            "Number of waypoints must be at least 1."
        )

    # Generate evenly spaced Cartesian positions.
    all_points = np.linspace(
        start_position,
        target_position,
        num_waypoints + 1
    )

    # Exclude the starting point because the robot
    # is already at that position.
    path_points = all_points[1:]

    trajectory = []

    for index, point in enumerate(path_points):

        result = solve_position_ik(
            model,
            ik_data,
            point,
            body_name=body_name,
            max_iterations=500,
            tolerance=tolerance,
            damping=0.05,
            step_size=0.5,
            max_joint_step=0.15
        )

        if not result["success"]:
            raise RuntimeError(
                f"IK failed at waypoint {index + 1}. "
                f"Position error: {result['error']:.6f} m"
            )

        trajectory.append({
            "waypoint": index + 1,
            "position": point.copy(),
            "joint_positions": result["joint_positions"].copy(),
            "ik_error": result["error"]
        })

    return trajectory
