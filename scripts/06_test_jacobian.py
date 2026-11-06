
import mujoco
import numpy as np

from library_robot.sim import load_model

from library_robot.control.jacobian import (
    get_body_jacobian,
    numerical_position_jacobian
)


def main():

    # Load Panda
    model, data = load_model()

    # Reset to home
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

    mujoco.mj_forward(model, data)

    print("\n" + "=" * 55)
    print("PANDA JACOBIAN VERIFICATION")
    print("=" * 55)

    # -----------------------------------
    # 1. Analytical Jacobian
    # -----------------------------------

    Jp, Jr = get_body_jacobian(
        model,
        data,
        "hand"
    )

    print("\nTRANSLATIONAL JACOBIAN (3 x 7)")
    print(Jp)

    print("\nROTATIONAL JACOBIAN (3 x 7)")
    print(Jr)

    # -----------------------------------
    # 2. Numerical Jacobian
    # -----------------------------------

    Jp_numerical = numerical_position_jacobian(
        model,
        data,
        "hand"
    )

    print("\nNUMERICAL TRANSLATIONAL JACOBIAN")
    print(Jp_numerical)

    # -----------------------------------
    # 3. Compare both Jacobians
    # -----------------------------------

    difference = Jp - Jp_numerical

    max_error = np.max(np.abs(difference))

    print("\nJACOBIAN DIFFERENCE")
    print(difference)

    print("\nMaximum absolute error:")
    print(f"{max_error:.10f}")

    # -----------------------------------
    # 4. Check joint1
    # -----------------------------------

    print("\nJOINT1 EFFECT ON HAND MOVEMENT")

    print("X velocity per rad/s:")
    print(f"{Jp[0, 0]:.6f}")

    print("Y velocity per rad/s:")
    print(f"{Jp[1, 0]:.6f}")

    print("Z velocity per rad/s:")
    print(f"{Jp[2, 0]:.6f}")

    # -----------------------------------
    # 5. Verification
    # -----------------------------------

    print("\n========== VERIFICATION ==========")

    if max_error < 1e-5:
        print("PASS: Analytical and numerical Jacobians agree.")
    else:
        print("CHECK: Jacobian difference is larger than expected.")

    print("\nJacobian test completed.")


if __name__ == "__main__":
    main()
