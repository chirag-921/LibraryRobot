
import mujoco


ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]


def get_id(model, object_type, name):
    """Get the ID of a named MuJoCo object."""

    object_id = mujoco.mj_name2id(
        model,
        object_type,
        name
    )

    if object_id == -1:
        raise ValueError(f"Object not found: {name}")

    return object_id


def set_arm_targets(model, data, targets):
    """
    Set the target positions of the seven Panda joints.

    targets: dictionary containing joint names and
             desired positions in radians.
    """

    for joint_name, target in targets.items():

        joint_id = get_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name
        )

        actuator_id = get_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "actuator" + joint_name[-1]
        )

        # Read joint limits
        if model.jnt_limited[joint_id]:

            lower, upper = model.jnt_range[joint_id]

            # Keep target within safe limits
            target = max(lower, min(upper, target))

        # Send target position to actuator
        data.ctrl[actuator_id] = target


def set_gripper(model, data, opening):
    """
    Control the Panda gripper.

    opening:
        0   = closed
        255 = fully open
    """

    actuator_id = get_id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "actuator8"
    )

    opening = max(0, min(255, opening))

    data.ctrl[actuator_id] = opening
