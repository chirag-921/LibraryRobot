LibraryRobot — Step 1A patch

This patch adds config-driven MJCF model generation for two selectable base variants:
- wheeled: freejoint chassis, two driven wheels, two ball-joint casters
- virtual: X/Y/yaw holonomic base with position actuators and low-friction supports

It does NOT yet add a differential-drive controller, odometry, navigation/path following,
parking skill, or the G3 gripper. Those are subsequent small steps after local model tests pass.

INSTALL
1. Back up your current C:\Robotics\LibraryRobot folder.
2. Extract this ZIP directly into C:\Robotics\LibraryRobot, preserving folders.
3. Open PowerShell:
   cd C:\Robotics\LibraryRobot
   .\.venv\Scripts\Activate.ps1
4. Generate models:
   python scripts\10_generate_mobile_models.py
5. Compile and smoke-test both models in your MuJoCo environment:
   python scripts\11_test_mobile_models.py
6. Open the selected model (configs\robot.yaml -> base_type):
   python scripts\12_open_mobile_robot.py

The original fixed Panda loader remains available via:
   from library_robot.sim import load_model

The new loader is:
   from library_robot.sim.mobile_model import load_mobile_robot_model
   model, data, base_type = load_mobile_robot_model()

All provisional physical/controller values are in configs/robot.yaml and marked
"needs tuning". Do not treat the model as validated for wheel direction, tip stability,
parking, collision-free travel, or end-to-end navigation until those tests are performed.
