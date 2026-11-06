
import time
import mujoco
import mujoco.viewer

from library_robot.sim import load_model

from library_robot.control.joint_control import (
    set_arm_targets,
    set_gripper
)


def run_for(model, data, viewer, duration):
    """Run simulation for a specified duration."""

    timestep = model.opt.timestep

    total_steps = int(duration / timestep)

    # Synchronize viewer approximately every 20 ms
    steps_per_sync = max(
        1,
        round(0.02 / timestep)
    )

    for start in range(
        0,
        total_steps,
        steps_per_sync
    ):

        if not viewer.is_running():
            return False

        batch_start = time.perf_counter()

        batch_end = min(
            start + steps_per_sync,
            total_steps
        )

        # Advance simulation physics
        for _ in range(start, batch_end):
            mujoco.mj_step(model, data)

        # Refresh the 3D viewer
        viewer.sync()

        # Try to maintain approximately real-time simulation
        elapsed = time.perf_counter() - batch_start

        target_time = (
            (batch_end - start) * timestep
        )

        remaining = target_time - elapsed

        if remaining > 0:
            time.sleep(remaining)

    return True


def main():

    # Load Panda
    model, data = load_model()

    # Load predefined Panda home keyframe
    key_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home"
    )

    if key_id == -1:
        raise RuntimeError(
            "Panda home keyframe was not found."
        )

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        key_id
    )

    mujoco.mj_forward(model, data)

    # Set all seven arm targets to the home position
    home_targets = {}

    for joint_name in [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name
        )

        qpos_address = model.jnt_qposadr[joint_id]

        home_targets[joint_name] = (
            data.qpos[qpos_address]
        )

    set_arm_targets(
        model,
        data,
        home_targets
    )

    # Start with gripper open
    set_gripper(model, data, 255)

    print("\nPanda controller initialized.")
    print("Opening MuJoCo viewer...")

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:

        print("\n1. Holding home position.")

        if not run_for(model, data, viewer, 2.0):
            return

        # ----------------------------------
        # Move joint 1 slightly
        # ----------------------------------

        print("\n2. Moving joint1.")

        current_angle = home_targets["joint1"]

        target_angle = current_angle + 0.20

        set_arm_targets(
            model,
            data,
            {
                "joint1": target_angle
            }
        )

        if not run_for(model, data, viewer, 3.0):
            return

        # ----------------------------------
        # Return to home position
        # ----------------------------------

        print("\n3. Returning joint1 to home.")

        set_arm_targets(
            model,
            data,
            {
                "joint1": current_angle
            }
        )

        if not run_for(model, data, viewer, 3.0):
            return

        # ----------------------------------
        # Partially close gripper
        # ----------------------------------

        print("\n4. Closing gripper partially.")

        set_gripper(model, data, 120)

        if not run_for(model, data, viewer, 2.0):
            return

        # ----------------------------------
        # Open gripper
        # ----------------------------------

        print("\n5. Opening gripper.")

        set_gripper(model, data, 255)

        if not run_for(model, data, viewer, 2.0):
            return

        print("\nMotion test completed.")

        # Keep viewer available until manually closed
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
