"""Control modules for Panda arm and differential-drive mobile base.

Kinematics and odometry tools can be imported and tested offline without
loading MuJoCo. Arm kinematics (IK/Jacobian) that require MuJoCo are loaded lazily.
"""

from .diff_drive import (
    wrap_angle,
    twist_to_wheel_speeds,
    wheel_speeds_to_twist,
    DiffDriveKinematics,
    ClosedLoopBaseController,
    set_arm_tucked_pose,
    check_arm_wheels_mutual_exclusion,
    DEFAULT_TUCKED_ARM_JOINTS,
)
from .odometry import (
    DiffDriveOdometry,
    compare_odometry_to_ground_truth,
    get_base_ground_truth_pose,
)

__all__ = [
    "wrap_angle",
    "twist_to_wheel_speeds",
    "wheel_speeds_to_twist",
    "DiffDriveKinematics",
    "ClosedLoopBaseController",
    "set_arm_tucked_pose",
    "check_arm_wheels_mutual_exclusion",
    "DEFAULT_TUCKED_ARM_JOINTS",
    "DiffDriveOdometry",
    "compare_odometry_to_ground_truth",
    "get_base_ground_truth_pose",
    "set_arm_targets",
    "set_gripper",
    "solve_position_ik",
    "solve_pose_ik",
    "get_body_jacobian",
    "numerical_position_jacobian",
    "generate_linear_cartesian_trajectory",
]

_LAZY_IMPORTS = {
    "set_arm_targets": (".joint_control", "set_arm_targets"),
    "set_gripper": (".joint_control", "set_gripper"),
    "solve_position_ik": (".inverse_kinematics", "solve_position_ik"),
    "solve_pose_ik": (".pose_ik", "solve_pose_ik"),
    "get_body_jacobian": (".jacobian", "get_body_jacobian"),
    "numerical_position_jacobian": (".jacobian", "numerical_position_jacobian"),
    "generate_linear_cartesian_trajectory": (".trajectory", "generate_linear_cartesian_trajectory"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        import importlib
        mod = importlib.import_module(module_name, package=__name__)
        return getattr(mod, attr_name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
