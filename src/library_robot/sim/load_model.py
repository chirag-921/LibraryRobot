from pathlib import Path

import mujoco


def load_model(model_path=None):

    # Find the project root directory
    project_root = Path(__file__).resolve().parents[3]

    # Default Panda scene
    if model_path is None:
        model_path = (
            project_root
            / "assets"
            / "mujoco_menagerie"
            / "franka_emika_panda"
            / "scene.xml"
        )
    else:
        model_path = Path(model_path)

    # Check whether the XML exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"Robot model not found: {model_path}"
        )

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(
        str(model_path)
    )

    # Create simulation data
    data = mujoco.MjData(model)

    # Calculate initial state
    mujoco.mj_forward(model, data)

    print("MuJoCo model loaded successfully!")

    return model, data