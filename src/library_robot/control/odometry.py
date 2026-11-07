"""Dead-reckoning wheel odometry and ground-truth error tracking for differential drive."""

import math
from typing import Any, Dict, Optional, Tuple


def wrap_angle(angle: float) -> float:
    """Normalize an angle to the interval [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class DiffDriveOdometry:
    """
    Integrates differential-drive wheel motion to maintain an estimated (dead-reckoned) pose.

    Can be updated using wheel joint angular velocities (rad/s) or wheel joint positions (rad).
    """

    def __init__(
        self,
        wheel_radius: float = 0.10,          # m; needs tuning
        wheel_separation: float = 0.62,      # m; needs tuning
        initial_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        if wheel_radius <= 0.0:
            raise ValueError(f"wheel_radius must be positive, got {wheel_radius}")
        if wheel_separation <= 0.0:
            raise ValueError(f"wheel_separation must be positive, got {wheel_separation}")

        self.wheel_radius = wheel_radius
        self.wheel_separation = wheel_separation

        self.x: float = float(initial_pose[0])
        self.y: float = float(initial_pose[1])
        self.yaw: float = wrap_angle(float(initial_pose[2]))

        self.total_distance_travelled: float = 0.0
        self._prev_left_pos: Optional[float] = None
        self._prev_right_pos: Optional[float] = None

    def reset(self, pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        """Reset odometry state to a specified pose."""
        self.x = float(pose[0])
        self.y = float(pose[1])
        self.yaw = wrap_angle(float(pose[2]))
        self.total_distance_travelled = 0.0
        self._prev_left_pos = None
        self._prev_right_pos = None

    def get_pose(self) -> Tuple[float, float, float]:
        """Return estimated (x, y, yaw) pose."""
        return self.x, self.y, self.yaw

    @property
    def distance_travelled(self) -> float:
        """Total accumulated arc distance travelled (m)."""
        return self.total_distance_travelled

    def update(
        self,
        wheel_left_vel: float,
        wheel_right_vel: float,
        dt: float,
    ) -> Tuple[float, float, float]:
        """
        Updates odometry estimate from wheel joint angular velocities (rad/s).

        Parameters
        ----------
        wheel_left_vel : float
            Left wheel angular velocity (rad/s).
        wheel_right_vel : float
            Right wheel angular velocity (rad/s).
        dt : float
            Simulation timestep (s).

        Returns
        -------
        Tuple[float, float, float]
            Updated (x, y, yaw) estimate.
        """
        if dt <= 0.0:
            return self.get_pose()

        delta_phi_left = wheel_left_vel * dt
        delta_phi_right = wheel_right_vel * dt

        delta_s_left = self.wheel_radius * delta_phi_left
        delta_s_right = self.wheel_radius * delta_phi_right

        delta_s = (delta_s_right + delta_s_left) / 2.0
        delta_theta = (delta_s_right - delta_s_left) / self.wheel_separation

        # Arc integration: midpoint heading
        midpoint_yaw = self.yaw + (delta_theta / 2.0)
        self.x += delta_s * math.cos(midpoint_yaw)
        self.y += delta_s * math.sin(midpoint_yaw)
        self.yaw = wrap_angle(self.yaw + delta_theta)

        self.total_distance_travelled += abs(delta_s)

        return self.get_pose()

    def update_from_positions(
        self,
        wheel_left_pos: float,
        wheel_right_pos: float,
    ) -> Tuple[float, float, float]:
        """
        Updates odometry directly from wheel encoder joint angles (rad).

        Avoids numerical integration drift from velocity discretization.
        """
        if self._prev_left_pos is None or self._prev_right_pos is None:
            self._prev_left_pos = wheel_left_pos
            self._prev_right_pos = wheel_right_pos
            return self.get_pose()

        delta_phi_left = wheel_left_pos - self._prev_left_pos
        delta_phi_right = wheel_right_pos - self._prev_right_pos

        self._prev_left_pos = wheel_left_pos
        self._prev_right_pos = wheel_right_pos

        delta_s_left = self.wheel_radius * delta_phi_left
        delta_s_right = self.wheel_radius * delta_phi_right

        delta_s = (delta_s_right + delta_s_left) / 2.0
        delta_theta = (delta_s_right - delta_s_left) / self.wheel_separation

        midpoint_yaw = self.yaw + (delta_theta / 2.0)
        self.x += delta_s * math.cos(midpoint_yaw)
        self.y += delta_s * math.sin(midpoint_yaw)
        self.yaw = wrap_angle(self.yaw + delta_theta)

        self.total_distance_travelled += abs(delta_s)

        return self.get_pose()


def compare_odometry_to_ground_truth(
    odometry_pose: Tuple[float, float, float],
    ground_truth_pose: Tuple[float, float, float],
    odom_distance: float,
    gt_distance: float,
) -> Dict[str, float]:
    """
    Compares dead-reckoned odometry against simulator ground truth.

    Parameters
    ----------
    odometry_pose : Tuple[float, float, float]
        Estimated (x, y, yaw).
    ground_truth_pose : Tuple[float, float, float]
        Simulator true (x, y, yaw).
    odom_distance : float
        Total distance integrated by wheel odometry (m).
    gt_distance : float
        Total distance travelled in simulator ground truth (m).

    Returns
    -------
    Dict[str, float]
        Error metrics including position error, heading error, distance, and slip.
        Slip is explicitly defined as odometry distance minus ground-truth distance.
    """
    x_odom, y_odom, yaw_odom = odometry_pose
    x_gt, y_gt, yaw_gt = ground_truth_pose

    pos_error = math.hypot(x_odom - x_gt, y_odom - y_gt)
    heading_error = abs(wrap_angle(yaw_odom - yaw_gt))

    # Definition from specification: slip = odometry distance minus ground-truth distance
    slip_distance = odom_distance - gt_distance
    slip_fraction = (slip_distance / odom_distance) if odom_distance > 1e-4 else 0.0

    return {
        "position_error_m": float(pos_error),
        "heading_error_rad": float(heading_error),
        "heading_error_deg": float(math.degrees(heading_error)),
        "odometry_distance_m": float(odom_distance),
        "ground_truth_distance_m": float(gt_distance),
        "slip_distance_m": float(slip_distance),
        "slip_fraction": float(slip_fraction),
    }


def get_base_ground_truth_pose(model: Any, data: Any) -> Tuple[float, float, float]:
    """
    Reads the ground-truth base (x, y, yaw) using sim.state_reader.

    Parameters
    ----------
    model : mujoco.MjModel
    data : mujoco.MjData

    Returns
    -------
    Tuple[float, float, float]
        (x, y, yaw) in world coordinates.
    """
    from library_robot.sim.state_reader import get_body_pose

    pose_dict = get_body_pose(model, data, "mobile_chassis")
    pos = pose_dict["position"]
    w, x, y, z = pose_dict["quaternion"]

    # Extract yaw from quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return float(pos[0]), float(pos[1]), float(yaw)
