
import time
import mujoco
import mujoco.viewer
import numpy as np

from library_robot.sim import load_model

from library_robot.control.trajectory import (
    generate_linear_cartesian_trajectory
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

    for body_name in ARM_BODIES:

        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name
        )

        if body_id == -1:
            raise ValueError(
                f"Body not found: {body_name}"
            )

        model.body_gravcomp[body_id] = 1.0

    mujoco.mj_setConst(model, data)


def run_simulation_steps(
    model,
    data,
    viewer,
    number_of_steps
):
    """Advance simulation while keeping the viewer responsive."""

    timestep = model.opt.timestep

    steps_per_update = max(
        1,
        round(0.02 / timestep)
    )

    for start in range(
        0,
        number_of_steps,
        steps_per_update
    ):

        if not viewer.is_running():
            return False

        batch_start = time.perf_counter()

        end = min(
            start + steps_per_update,
            number_of_steps
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

    return True


def main():

    # =====================================
    # 1. Load model
    # =====================================

    model, data = load_model()

    enable_gravity_compensation(
        model,
        data
    )

    home_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home"
    )

    if home_id == -1:
        raise RuntimeError(
            "Home keyframe not found."
        )

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_id
    )

    mujoco.mj_forward(model, data)

    # =====================================
    # 2. Read initial hand position
    # =====================================

    hand_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "hand"
    )

    start_position = data.xpos[
        hand_id
    ].copy()

    # Move 2 cm along each Cartesian axis.
    target_position = start_position + np.array([
        0.02,
        0.02,
        0.02
    ])

    print("\n========== CARTESIAN TRAJECTORY ==========")

    print("Start position:")
    print(start_position)

    print("\nTarget position:")
    print(target_position)

    # =====================================
    # 3. Generate IK trajectory
    # =====================================

    ik_data = mujoco.MjData(model)

    ik_data.qpos[:] = data.qpos
    ik_data.qvel[:] = data.qvel

    mujoco.mj_forward(model, ik_data)

    number_of_waypoints = 20

    trajectory = generate_linear_cartesian_trajectory(
        model,
        ik_data,
        start_position,
        target_position,
        num_waypoints=number_of_waypoints,
        body_name="hand"
    )

    print(
        "\nNumber of Cartesian waypoints:",
        len(trajectory)
    )

    print("\nAll waypoints passed IK.")

    # =====================================
    # 4. Execute trajectory
    # =====================================

    set_gripper(
        model,
        data,
        255
    )

    # Each waypoint is held for 0.2 seconds.
    waypoint_duration = 0.20

    steps_per_waypoint = int(
        waypoint_duration / model.opt.timestep
    )

    print("\nOpening MuJoCo viewer...")

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:

        for item in trajectory:

            if not viewer.is_running():
                break

            # Send the current waypoint's joint targets.
            set_arm_targets(
                model,
                data,
                item["joint_positions"]
            )

            completed = run_simulation_steps(
                model,
                data,
                viewer,
                steps_per_waypoint
            )

            if not completed:
                break

            # Report every fifth waypoint.
            if item["waypoint"] % 5 == 0:

                actual_position = data.xpos[
                    hand_id
                ].copy()

                print(
                    f"\nWaypoint {item['waypoint']}/"
                    f"{number_of_waypoints}"
                )

                print("Desired:", item["position"])
                print("Actual: ", actual_position)

    # =====================================
    # 5. Final verification
    # =====================================

        if viewer.is_running():

            final_position = data.xpos[
                hand_id
            ].copy()

            final_error = np.linalg.norm(
                target_position - final_position
            )

            print("\n========== FINAL RESULT ==========")

            print("Target position:")
            print(target_position)

            print("\nActual position:")
            print(final_position)

            print("\nFinal position error:")
            print(f"{final_error:.6f} metres")

            print(
                f"{final_error * 1000:.3f} mm"
            )

            print("\nCartesian trajectory completed.")

            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)


if __name__ == "__main__":
    main()
