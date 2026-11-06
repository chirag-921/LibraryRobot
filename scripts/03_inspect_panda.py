
import mujoco

from library_robot.sim import load_model


def main():

    model, data = load_model()

    print("\n" + "=" * 55)
    print("FRANKA PANDA - ROBOT MODEL INSPECTION")
    print("=" * 55)

    # ----------------------------------------
    # 1. Inspect joints
    # ----------------------------------------

    print("\n========== JOINTS ==========")

    for i in range(model.njnt):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            i
        )

        print(f"\nJoint ID: {i}")
        print(f"Name: {name}")
        print(f"Type: {model.jnt_type[i]}")
        print(f"Position address: {model.jnt_qposadr[i]}")
        print(f"Velocity address: {model.jnt_dofadr[i]}")

        if model.jnt_limited[i]:
            print(f"Joint limits: {model.jnt_range[i]}")

    # ----------------------------------------
    # 2. Inspect actuators
    # ----------------------------------------

    print("\n========== ACTUATORS ==========")

    for i in range(model.nu):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            i
        )

        print(f"\nActuator ID: {i}")
        print(f"Name: {name}")
        print(f"Control range: {model.actuator_ctrlrange[i]}")
        print(f"Control limited: {model.actuator_ctrllimited[i]}")
        print(f"Transmission type: {model.actuator_trntype[i]}")
        print(f"Transmission IDs: {model.actuator_trnid[i]}")

    # ----------------------------------------
    # 3. Inspect end-effector sites
    # ----------------------------------------

    print("\n========== SITES ==========")

    for i in range(model.nsite):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            i
        )

        print(f"Site ID: {i} | Name: {name}")

    # ----------------------------------------
    # 4. Inspect bodies
    # ----------------------------------------

    print("\n========== BODIES ==========")

    for i in range(model.nbody):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            i
        )

        print(f"Body ID: {i} | Name: {name}")

    print("\nInspection completed successfully.")


if __name__ == "__main__":
    main()
