
import mujoco
import numpy as np

from library_robot.sim import load_model

from library_robot.sim.state_reader import (
    get_joint_positions,
    get_joint_velocities,
    get_body_pose,
    get_robot_state
)


def print_joint_positions(positions):

    print("\n========== ARM JOINT POSITIONS ==========")

    for name, value in positions.items():

        degrees = np.degrees(value)

        print(
            f"{name}: "
            f"{value:.5f} rad "
            f"({degrees:.2f} degrees)"
        )


def print_hand_pose(pose):

    print("\n========== HAND POSITION ==========")

    position = pose["position"]

    print(f"X: {position[0]:.5f} m")
    print(f"Y: {position[1]:.5f} m")
    print(f"Z: {position[2]:.5f} m")

    print("\n========== HAND ORIENTATION ==========")

    print(pose["rotation"])

    print("\nQuaternion [w, x, y, z]:")

    print(pose["quaternion"])


def main():

    # Load Panda model
    model, data = load_model()

    # Load home position
    key_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home"
    )

    if key_id == -1:
        raise RuntimeError("Home keyframe not found.")

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        key_id
    )

    # Calculate forward kinematics
    mujoco.mj_forward(model, data)

    print("\n" + "=" * 50)
    print("PANDA FORWARD KINEMATICS TEST")
    print("=" * 50)

    # Read the initial state
    initial_state = get_robot_state(model, data)

    print_joint_positions(
        initial_state["arm_positions"]
    )

    print("\n========== GRIPPER POSITIONS ==========")

    for name, value in initial_state[
        "gripper_positions"
    ].items():

        print(f"{name}: {value:.5f} m")

    initial_pose = initial_state["hand_pose"]

    print_hand_pose(initial_pose)

    # --------------------------------------
    # Test movement of joint1
    # --------------------------------------

    print("\n========== FK MOVEMENT TEST ==========")

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "joint1"
    )

    qpos_address = model.jnt_qposadr[joint_id]

    original_angle = data.qpos[qpos_address]

    # Change joint1 by 0.20 radians
    new_angle = original_angle + 0.20

    lower, upper = model.jnt_range[joint_id]

    new_angle = np.clip(
        new_angle,
        lower,
        upper
    )

    data.qpos[qpos_address] = new_angle

    # Recalculate the robot's Cartesian position
    mujoco.mj_forward(model, data)

    new_pose = get_body_pose(
        model,
        data,
        "hand"
    )

    # Calculate position change
    displacement = (
        new_pose["position"] -
        initial_pose["position"]
    )

    print("\nJoint1 original angle:")
    print(f"{original_angle:.5f} rad")

    print("\nJoint1 new angle:")
    print(f"{new_angle:.5f} rad")

    print("\nHand displacement:")

    print(f"X: {displacement[0]:.5f} m")
    print(f"Y: {displacement[1]:.5f} m")
    print(f"Z: {displacement[2]:.5f} m")

    print(
        "\nTotal displacement:",
        f"{np.linalg.norm(displacement):.5f} m"
    )

    # Restore the original state
    data.qpos[qpos_address] = original_angle

    mujoco.mj_forward(model, data)

    print("\nFK test completed successfully.")


if __name__ == "__main__":
    main()
