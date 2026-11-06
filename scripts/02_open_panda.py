import mujoco
import mujoco.viewer


MODEL_PATH = (
    "assets/mujoco_menagerie/"
    "franka_emika_panda/scene.xml"
)


def main():

    # Load the MuJoCo model
    model = mujoco.MjModel.from_xml_path(
        MODEL_PATH
    )

    # Create simulation data
    data = mujoco.MjData(model)

    # Launch the interactive viewer
    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:

        while viewer.is_running():

            # Advance physics
            mujoco.mj_step(model, data)

            # Update the viewer
            viewer.sync()


if __name__ == "__main__":
    main()