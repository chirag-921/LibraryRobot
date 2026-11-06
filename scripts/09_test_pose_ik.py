
import time
import mujoco
import mujoco.viewer
import numpy as np

from library_robot.sim import load_model

from library_robot.control.pose_ik import (
    solve_pose_ik
)

from library_robot.control.joint_control import (
    set_arm_targets,
    set_gripper
)


ARM_BODIES = [
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "hand",
    "left_finger",
    "right_finger"
]


def enable_gravity_compensation(model, data):

    for name in ARM_BODIES:

        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            name
        )

        if body_id == -1:
            raise ValueError(
                f"Body not found: {name}"
            )

        model.body_gravcomp[body_id] = 1.0

    mujoco.mj_setConst(model, data)


def main():

    # 1. Load Panda
    model, data = load_model()

    enable_gravity_compensation(
        model,
        data
    )

    # 2. Reset to home
    home_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home"
    )

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_id
    )

    mujoco.mj_forward(model, data)

    hand_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "hand"
    )

    # 3. Read starting pose
    start_position = data.xpos[
        hand_id
    ].copy()

    start_rotation = data.xmat[
        hand_id
    ].reshape(3, 3).copy()

    # Move the hand 1 cm along each axis
    target_position = start_position + np.array([
        0.01,
        0.01,
        0.01
    ])

    # Maintain the original orientation
    target_rotation = start_rotation.copy()

    print("\n========== POSE IK TEST ==========")

    print("\nStarting position:")
    print(start_position)

    print("\nTarget position:")
    print(target_position)

    print("\nOrientation will remain unchanged.")

    # 4. Separate data for IK calculation
    ik_data = mujoco.MjData(model)

    ik_data.qpos[:] = data.qpos
    ik_data.qvel[:] = data.qvel

    mujoco.mj_forward(model, ik_data)

    # 5. Solve pose IK
    result = solve_pose_ik(
        model,
        ik_data,
        target_position,
        target_rotation,
        body_name="hand"
    )

    print("\n========== IK RESULT ==========")

    print("Success:", result["success"])
    print("Iterations:", result["iterations"])

    print(
        "Position error:",
        f'{result["position_error"] * 1000:.4f} mm'
    )

    print(
        "Orientation error:",
        f'{np.degrees(result["orientation_error"]):.5f} degrees'
    )

    print("\nJoint targets:")

    for name, value in result[
        "joint_positions"
    ].items():

        print(
            f"{name}: {value:.5f} rad"
        )

    if not result["success"]:
        raise RuntimeError(
            "Pose IK did not converge. "
            "Robot movement has been stopped."
        )

    # 6. Send target to actuators
    set_arm_targets(
        model,
        data,
        result["joint_positions"]
    )

    set_gripper(model, data, 255)

    # 7. Simulate motion
    print("\nOpening viewer...")

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:

        timestep = model.opt.timestep

        total_steps = int(5.0 / timestep)

        steps_per_update = max(
            1,
            round(0.02 / timestep)
        )

        for start in range(
            0,
            total_steps,
            steps_per_update
        ):

            if not viewer.is_running():
                break

            batch_start = time.perf_counter()

            end = min(
                start + steps_per_update,
                total_steps
            )

            for _ in range(start, end):
                mujoco.mj_step(model, data)

            viewer.sync()

            elapsed = time.perf_counter() - batch_start

            desired_time = (
                (end - start) * timestep
            )

            remaining = desired_time - elapsed

            if remaining > 0:
                time.sleep(remaining)

        # 8. Verify actual final pose
        mujoco.mj_forward(model, data)

        actual_position = data.xpos[
            hand_id
        ].copy()

        actual_rotation = data.xmat[
            hand_id
        ].reshape(3, 3).copy()

        actual_position_error = np.linalg.norm(
            target_position - actual_position
        )

        from scipy.spatial.transform import Rotation

        actual_orientation_error = Rotation.from_matrix(
            target_rotation @ actual_rotation.T
        ).magnitude()

        print("\n========== FINAL POSE VERIFICATION ==========")

        print("Target position:")
        print(target_position)

        print("\nActual position:")
        print(actual_position)

        print(
            "\nFinal position error:",
            f"{actual_position_error * 1000:.4f} mm"
        )

        print(
            "Final orientation error:",
            f"{np.degrees(actual_orientation_error):.5f} degrees"
        )

        print("\nPose IK test completed.")

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
