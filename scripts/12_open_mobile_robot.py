"""Open the mobile Panda model selected by configs/robot.yaml."""

import time

import mujoco
import mujoco.viewer

from pathlib import Path
import yaml

from library_robot.sim.mobile_model import load_mobile_robot_model


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "robot.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as _stream:
    CONFIG = yaml.safe_load(_stream)


def main():
    model, data, base_type = load_mobile_robot_model(generate_if_missing=True)
    print(f"Loaded mobile Panda with base_type={base_type!r}")
    print("Close the MuJoCo window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()
            sync_period = float(CONFIG["validation"]["viewer_sync_period_s"])
            remaining = sync_period - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
