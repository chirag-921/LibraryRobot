"""Differential-drive kinematics, acceleration limiting, and closed-loop base control.

Provides:
  - Forward & inverse unicycle kinematics: (v, omega) <-> (omega_left, omega_right)
  - DiffDriveKinematics: slew-rate acceleration and velocity limiter
  - ClosedLoopBaseController: unicycle waypoint reaching controller
  - Arm posture rule: set_arm_tucked_pose and check_arm_wheels_mutual_exclusion
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union


def wrap_angle(angle: float) -> float:
    """Normalize an angle to the interval [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def twist_to_wheel_speeds(
    v: float,
    omega: float,
    wheel_radius: float,
    wheel_separation: float,
) -> Tuple[float, float]:
    """
    Converts chassis linear and angular velocity to left and right wheel angular velocities.

    Parameters
    ----------
    v : float
        Forward linear velocity along chassis +X (m/s).
    omega : float
        Angular velocity around chassis +Z (rad/s, positive CCW).
    wheel_radius : float
        Radius of the drive wheels (m).
    wheel_separation : float
        Distance between left and right wheel contact centers (m).

    Returns
    -------
    Tuple[float, float]
        (omega_left, omega_right) in rad/s.
    """
    if wheel_radius <= 0.0:
        raise ValueError(f"wheel_radius must be positive, got {wheel_radius}")
    if wheel_separation <= 0.0:
        raise ValueError(f"wheel_separation must be positive, got {wheel_separation}")

    v_left = v - (omega * wheel_separation / 2.0)
    v_right = v + (omega * wheel_separation / 2.0)

    omega_left = v_left / wheel_radius
    omega_right = v_right / wheel_radius

    return omega_left, omega_right


def wheel_speeds_to_twist(
    omega_left: float,
    omega_right: float,
    wheel_radius: float,
    wheel_separation: float,
) -> Tuple[float, float]:
    """
    Converts left and right wheel angular velocities to chassis linear and angular velocity.

    Parameters
    ----------
    omega_left : float
        Left wheel angular velocity (rad/s).
    omega_right : float
        Right wheel angular velocity (rad/s).
    wheel_radius : float
        Radius of the drive wheels (m).
    wheel_separation : float
        Distance between left and right wheel contact centers (m).

    Returns
    -------
    Tuple[float, float]
        (v, omega) in (m/s, rad/s).
    """
    if wheel_radius <= 0.0:
        raise ValueError(f"wheel_radius must be positive, got {wheel_radius}")
    if wheel_separation <= 0.0:
        raise ValueError(f"wheel_separation must be positive, got {wheel_separation}")

    v = (wheel_radius / 2.0) * (omega_right + omega_left)
    omega = (wheel_radius / wheel_separation) * (omega_right - omega_left)

    return v, omega


class DiffDriveKinematics:
    """Enforces velocity and slew-rate acceleration limits for differential-drive robots."""

    def __init__(
        self,
        wheel_radius: float = 0.10,          # m; needs tuning
        wheel_separation: float = 0.62,      # m; needs tuning
        max_linear_velocity: float = 0.60,   # m/s; needs tuning
        max_angular_velocity: float = 1.20,  # rad/s; needs tuning
        max_linear_accel: float = 0.80,      # m/s^2; needs tuning
        max_angular_accel: float = 1.50,     # rad/s^2; needs tuning
        wheel_speed_limit: float = 8.0,      # rad/s; needs tuning
    ) -> None:
        self.wheel_radius = wheel_radius
        self.wheel_separation = wheel_separation
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.max_linear_accel = max_linear_accel
        self.max_angular_accel = max_angular_accel
        self.wheel_speed_limit = wheel_speed_limit

        self.current_v: float = 0.0
        self.current_omega: float = 0.0

    def reset(self, v: float = 0.0, omega: float = 0.0) -> None:
        """Reset internal rate states."""
        self.current_v = v
        self.current_omega = omega

    def limit_twist(
        self,
        target_v: float,
        target_omega: float,
        dt: float,
    ) -> Tuple[float, float]:
        """
        Ramps target (v, omega) subject to maximum linear and angular acceleration limits.

        Parameters
        ----------
        target_v : float
            Desired forward linear velocity (m/s).
        target_omega : float
            Desired angular velocity (rad/s).
        dt : float
            Timestep (s).

        Returns
        -------
        Tuple[float, float]
            Rate-limited (v, omega).
        """
        if dt <= 0.0:
            return self.current_v, self.current_omega

        # Clamp targets to absolute maximum limits
        clamped_target_v = max(-self.max_linear_velocity, min(self.max_linear_velocity, target_v))
        clamped_target_omega = max(-self.max_angular_velocity, min(self.max_angular_velocity, target_omega))

        # Enforce linear acceleration limit
        max_dv = self.max_linear_accel * dt
        delta_v = clamped_target_v - self.current_v
        delta_v_limited = max(-max_dv, min(max_dv, delta_v))
        self.current_v += delta_v_limited

        # Enforce angular acceleration limit
        max_domega = self.max_angular_accel * dt
        delta_omega = clamped_target_omega - self.current_omega
        delta_omega_limited = max(-max_domega, min(max_domega, delta_omega))
        self.current_omega += delta_omega_limited

        return self.current_v, self.current_omega

    def compute_wheel_commands(
        self,
        target_v: float,
        target_omega: float,
        dt: float,
    ) -> Tuple[float, float]:
        """
        Applies acceleration limits and converts to bounded left/right wheel speeds.

        Returns
        -------
        Tuple[float, float]
            (wheel_speed_left, wheel_speed_right) in rad/s, clipped to wheel_speed_limit.
        """
        lim_v, lim_w = self.limit_twist(target_v, target_omega, dt)
        w_left, w_right = twist_to_wheel_speeds(
            lim_v, lim_w, self.wheel_radius, self.wheel_separation
        )

        # Individual actuator speed limit
        w_left_clamped = max(-self.wheel_speed_limit, min(self.wheel_speed_limit, w_left))
        w_right_clamped = max(-self.wheel_speed_limit, min(self.wheel_speed_limit, w_right))

        return w_left_clamped, w_right_clamped


class ClosedLoopBaseController:
    """
    Phased closed-loop differential-drive controller to drive the robot to target (x, y, yaw).

    Implements a robust 4-phase finite state machine:
      1. TURN_TO_WAYPOINT: In-place turn to face the target position bearing.
      2. DRIVE_TO_WAYPOINT: Forward drive towards target position with heading correction.
      3. ALIGN_FINAL_HEADING: In-place turn to align with the final desired yaw.
      4. SETTLE_AND_HOLD: Holds zero velocity and requires the robot to stay within
         tolerances for settle_steps before reporting reached=True.
    """

    def __init__(
        self,
        kinematics: Optional[DiffDriveKinematics] = None,
        k_bearing: float = 2.0,            # gain for turning towards waypoint bearing; needs tuning
        k_drive: float = 0.8,              # gain for linear forward velocity; needs tuning
        k_heading: float = 1.8,            # gain for final heading alignment; needs tuning
        position_tolerance: float = 0.02,  # metres (2 cm); needs tuning
        heading_tolerance: float = 0.0349, # radians (~2.0 deg); needs tuning
        settle_steps: int = 50,            # steps (0.1 s at dt=0.002) to hold position; needs tuning
    ) -> None:
        self.kinematics = kinematics or DiffDriveKinematics()
        self.k_bearing = k_bearing
        self.k_drive = k_drive
        self.k_heading = k_heading
        self.position_tolerance = position_tolerance
        self.heading_tolerance = heading_tolerance
        self.settle_steps = settle_steps

        self.state: str = "TURN_TO_WAYPOINT"
        self.settle_count: int = 0

    def reset(self) -> None:
        """Reset internal controller state and rate limits."""
        self.kinematics.reset(0.0, 0.0)
        self.state = "TURN_TO_WAYPOINT"
        self.settle_count = 0

    def step(
        self,
        current_pose: Tuple[float, float, float],
        target_pose: Tuple[float, float, float],
        dt: float,
    ) -> Tuple[Tuple[float, float], bool, Dict[str, Any]]:
        """
        Calculates left and right wheel velocity commands for one control cycle.

        Parameters
        ----------
        current_pose : Tuple[float, float, float]
            Current (x, y, yaw).
        target_pose : Tuple[float, float, float]
            Target (x, y, yaw).
        dt : float
            Simulation/control timestep (s).

        Returns
        -------
        Tuple[Tuple[float, float], bool, Dict[str, Any]]
            ((cmd_left, cmd_right), reached_flag, debug_info)
        """
        x, y, yaw = current_pose
        x_target, y_target, yaw_target = target_pose

        dx = x_target - x
        dy = y_target - y
        dist = math.hypot(dx, dy)

        target_bearing = math.atan2(dy, dx)
        bearing_err = wrap_angle(target_bearing - yaw)
        final_heading_err = wrap_angle(yaw_target - yaw)

        # State Machine Transitions & Control Computation
        if self.state == "TURN_TO_WAYPOINT":
            if dist <= self.position_tolerance:
                self.state = "ALIGN_FINAL_HEADING"
                des_v = 0.0
                des_omega = 0.0
            elif abs(bearing_err) <= math.radians(6.0):  # Aligned to within ~6 deg
                self.state = "DRIVE_TO_WAYPOINT"
                des_v = 0.0
                des_omega = 0.0
            else:
                des_v = 0.0
                des_omega = max(-self.kinematics.max_angular_velocity, min(
                    self.kinematics.max_angular_velocity, self.k_bearing * bearing_err
                ))

        elif self.state == "DRIVE_TO_WAYPOINT":
            if dist <= self.position_tolerance:
                self.state = "ALIGN_FINAL_HEADING"
                des_v = 0.0
                des_omega = 0.0
            elif abs(bearing_err) > math.radians(35.0):
                # Heading deviated significantly; re-align
                self.state = "TURN_TO_WAYPOINT"
                des_v = 0.0
                des_omega = 0.0
            else:
                des_v = min(self.kinematics.max_linear_velocity, max(0.08, self.k_drive * dist))
                des_omega = max(-self.kinematics.max_angular_velocity, min(
                    self.kinematics.max_angular_velocity, self.k_bearing * bearing_err
                ))

        elif self.state == "ALIGN_FINAL_HEADING":
            if dist > self.position_tolerance * 1.5:
                # Drifted out of position circle during turn
                self.state = "TURN_TO_WAYPOINT"
                des_v = 0.0
                des_omega = 0.0
            elif abs(final_heading_err) <= self.heading_tolerance:
                self.state = "SETTLE_AND_HOLD"
                des_v = 0.0
                des_omega = 0.0
            else:
                des_v = 0.0
                des_omega = max(-self.kinematics.max_angular_velocity, min(
                    self.kinematics.max_angular_velocity, self.k_heading * final_heading_err
                ))

        elif self.state == "SETTLE_AND_HOLD":
            des_v = 0.0
            des_omega = 0.0
            if dist <= self.position_tolerance and abs(final_heading_err) <= self.heading_tolerance:
                self.settle_count += 1
            else:
                self.settle_count = 0
                self.state = "ALIGN_FINAL_HEADING"
        else:
            self.state = "TURN_TO_WAYPOINT"
            des_v = 0.0
            des_omega = 0.0

        reached = bool(self.state == "SETTLE_AND_HOLD" and self.settle_count >= self.settle_steps)

        cmd_left, cmd_right = self.kinematics.compute_wheel_commands(des_v, des_omega, dt)

        debug_info = {
            "state": self.state,
            "settle_count": self.settle_count,
            "dist_error_m": float(dist),
            "heading_error_rad": float(final_heading_err),
            "bearing_error_rad": float(bearing_err),
            "cmd_v_mps": float(self.kinematics.current_v),
            "cmd_omega_rads": float(self.kinematics.current_omega),
            "cmd_wheel_left": float(cmd_left),
            "cmd_wheel_right": float(cmd_right),
            "reached": reached,
        }

        return (cmd_left, cmd_right), reached, debug_info


# ==============================================================================
# ARM POSTURE RULE & MUTUAL EXCLUSION CHECK
# ==============================================================================

# Default reference arm tucked configuration
DEFAULT_TUCKED_ARM_JOINTS: Dict[str, float] = {
    "joint1": 0.0,
    "joint2": -0.785398,  # rad; tucked backward; needs tuning
    "joint3": 0.0,
    "joint4": -2.356194,  # rad; folded elbow; needs tuning
    "joint5": 0.0,
    "joint6": 1.570796,   # rad; wrist level; needs tuning
    "joint7": 0.785398,   # rad; needs tuning
}


def set_arm_tucked_pose(
    model: Any,
    data: Any,
    tucked_joints: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Sets the Franka Panda arm into the tucked driving pose to minimize footprint and tip risk.

    Parameters
    ----------
    model : mujoco.MjModel
    data : mujoco.MjData
    tucked_joints : dict, optional
        Mapping of joint name to target angle in radians.

    Returns
    -------
    Dict[str, float]
        Target joint angles applied.
    """
    import mujoco

    targets = tucked_joints or DEFAULT_TUCKED_ARM_JOINTS

    for name, pos in targets.items():
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id >= 0:
            data.qpos[model.jnt_qposadr[j_id]] = pos
            act_num = name.replace("joint", "")
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{act_num}")
            if act_id >= 0:
                data.ctrl[act_id] = pos

    mujoco.mj_forward(model, data)
    return targets


def check_arm_wheels_mutual_exclusion(
    wheel_commands: Tuple[float, float],
    arm_velocities: Union[Dict[str, float], List[float]],
    arm_vel_threshold: float = 0.05,     # rad/s; needs tuning
    wheel_cmd_threshold: float = 1e-3,   # rad/s; needs tuning
) -> bool:
    """
    Verifies the safety rule: wheel commands must be strictly zero whenever the arm is in motion,
    and the arm must be stationary while the mobile base drives.

    Parameters
    ----------
    wheel_commands : Tuple[float, float]
        (left_wheel_command, right_wheel_command)
    arm_velocities : dict or list
        Velocities of arm joints.
    arm_vel_threshold : float
        Tolerance for arm joint velocity (rad/s).
    wheel_cmd_threshold : float
        Tolerance for wheel command (rad/s).

    Returns
    -------
    bool
        True if mutual exclusion rule is respected, False if violated.
    """
    wheel_active = (
        abs(wheel_commands[0]) > wheel_cmd_threshold
        or abs(wheel_commands[1]) > wheel_cmd_threshold
    )

    if isinstance(arm_velocities, dict):
        vel_values = arm_velocities.values()
    else:
        vel_values = arm_velocities

    arm_active = any(abs(v) > arm_vel_threshold for v in vel_values)

    # Violation: wheels commanded to move while arm is still moving
    if wheel_active and arm_active:
        return False

    return True
