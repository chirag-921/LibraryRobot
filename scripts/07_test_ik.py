
import time
import mujoco
import mujoco.viewer
import numpy as np

from library_robot.sim import load_model

from library_robot.control.inverse_kinematics import (
    solve_position_ik
)

from library_robot.control.joint_control import (
    set_arm_targets,
    set_gripper
)


def main():

    # =====================================
    # 1. Load Panda
    # =====================================

    model, data = load_model()
    
    # =====================================
    # Enable Panda gravity compensation
    # =====================================

    panda_bodies = [
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

    for body_name in panda_bodies:

        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name
        )

        if body_id == -1:
            raise ValueError(
                f"Body not found: {body_name}"
            )

        # Full gravity compensation
        model.body_gravcomp[body_id] = 1.0

    # Update MuJoCo model constants and flags
    mujoco.mj_setConst(model, data)

    print("\nGravity compensation enabled.")

    for body_name in panda_bodies:
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name
        )

        print(
            body_name,
            model.body_gravcomp[body_id]
        )


    home_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home"
    )

    if home_id == -1:
        raise RuntimeError("Home keyframe not found.")

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

    initial_position = data.xpos[hand_id].copy()

    # =====================================
    # 2. Calculate IK
    # =====================================

    target_position = initial_position + np.array([
        0.02,
        0.02,
        0.02
    ])

    print("\n========== IK TEST ==========")
    print("Initial position:", initial_position)
    print("Target position:", target_position)

    # Separate data for IK calculation
    ik_data = mujoco.MjData(model)

    ik_data.qpos[:] = data.qpos
    ik_data.qvel[:] = data.qvel

    mujoco.mj_forward(model, ik_data)

    result = solve_position_ik(
        model,
        ik_data,
        target_position,
        body_name="hand"
    )

    print("\n========== IK RESULT ==========")

    print("Success:", result["success"])
    print("Iterations:", result["iterations"])
    print("Position error:", result["error"])

    if not result["success"]:
        raise RuntimeError("IK solution did not converge.")

    targets = result["joint_positions"]

    print("\nCalculated joint targets:")

    for name, value in targets.items():
        print(f"{name}: {value:.5f} rad")

    # =====================================
    # 3. Send commands to actuators
    # =====================================

    set_arm_targets(
        model,
        data,
        targets
    )

    set_gripper(
        model,
        data,
        255
    )

    print("\n========== CONTROL DEBUG ==========")

    for i in range(7):

        print(
            f"Actuator {i+1}: "
            f"Target = {data.ctrl[i]:.5f}"
        )

    # =====================================
    # 4. Launch viewer
    # =====================================

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

        print("\n========== ROBOT MOVEMENT ==========")

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

            # Advance the physics
            for _ in range(start, end):
                mujoco.mj_step(model, data)

            # Refresh the 3D viewer
            viewer.sync()

            # Print joint readings every 0.5 seconds
            if start % max(1, int(0.5 / timestep)) < steps_per_update:

                print("\nActual versus target:")

                for i in range(7):

                    joint_name = f"joint{i+1}"

                    joint_id = mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        joint_name
                    )

                    address = model.jnt_qposadr[joint_id]

                    actual = data.qpos[address]
                    target = targets[joint_name]

                    print(
                        f"{joint_name}: "
                        f"actual={actual:.5f}, "
                        f"target={target:.5f}"
                    )

            elapsed = time.perf_counter() - batch_start

            desired_time = (
                (end - start) * timestep
            )

            remaining = desired_time - elapsed

            if remaining > 0:
                time.sleep(remaining)

        # =================================
        # 5. Final verification
        # =================================

        print("\n========== FINAL JOINT POSITIONS ==========")

        for i in range(7):

            joint_name = f"joint{i+1}"

            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name
            )

            address = model.jnt_qposadr[joint_id]

            actual = data.qpos[address]
            target = targets[joint_name]

            print(
                f"{joint_name}: "
                f"Actual={actual:.5f}, "
                f"Target={target:.5f}, "
                f"Error={abs(target-actual):.5f}"
            )
        # Final end-effector position verification
        mujoco.mj_forward(model, data)

        final_hand_position = data.xpos[
            hand_id
        ].copy()

        print("\n========== HAND POSITION VERIFICATION ==========")

        print("Target position:")
        print(target_position)

        print("\nActual final hand position:")
        print(final_hand_position)

        position_error = np.linalg.norm(
            target_position - final_hand_position
        )

        print("\nFinal hand position error:")
        print(f"{position_error:.6f} metres")
        print(f"{position_error * 1000:.3f} mm")
        print("\nFive-second movement test completed.")

        # Keep viewer open
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
