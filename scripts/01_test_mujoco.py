import mujoco
import numpy as np

MODEL_PATH = (
    "assets/mujoco_menagerie/"
    "franka_emika_panda/scene.xml"
)


def main():
    print("=" * 50)
    print("LIBRARY ROBOT - MUJOCO INSTALLATION TEST")
    print("=" * 50)

    # Print the MuJoCo version
    print("\nMuJoCo version:", mujoco.__version__)

    # Load the Panda model
    print("\nLoading Franka Panda...")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)

    # Create simulation data
    data = mujoco.MjData(model)

    # Calculate initial model state
    mujoco.mj_forward(model, data)

    # Display model information
    print("\nMODEL LOADED SUCCESSFULLY")

    print("Number of bodies:", model.nbody)
    print("Number of joints:", model.njnt)
    print("Number of actuators:", model.nu)
    print("Number of geoms:", model.ngeom)
    print("Number of sites:", model.nsite)

    print("Simulation timestep:", model.opt.timestep)

    # Check that the simulation state is finite
    print(
        "Simulation state valid:",
        np.all(np.isfinite(data.qpos))
    )

    print("\nInstallation test completed.")


if __name__ == "__main__":
    main()