# Step 1A — Configurable mobile Panda model variants

This increment adds model generation and loading only. It does **not** yet implement differential-drive navigation, odometry, parking, the `navigate(target)` skill, or the G3 gripper.

## New configuration

Edit `configs/robot.yaml`:

```yaml
base_type: wheeled  # wheeled | virtual
```

All proposed chassis, wheel, caster, actuator and virtual-controller values are starting values marked `needs tuning` in that file.

## Generate both models

From the project root, with the project's `.venv` active:

```powershell
python scripts\10_generate_mobile_models.py
```

Generated files:

```text
models/generated/wheeled_robot.xml
models/generated/virtual_robot.xml
```

The generator reads the original `assets/mujoco_menagerie/franka_emika_panda/panda.xml`; it does not modify the original Menagerie files.

## Load the selected variant from Python

```python
from library_robot.sim.mobile_model import load_mobile_robot_model

model, data, base_type = load_mobile_robot_model()
```

The loader reads `base_type` from `configs/robot.yaml`. A temporary override is also supported:

```python
model, data, base_type = load_mobile_robot_model(base_type="virtual")
```

The existing fixed-Panda loader API remains available:

```python
from library_robot.sim import load_model
```

## Tests and visual inspection

Generate both XML files, then compile and smoke-test both variants with the local MuJoCo installation:

```powershell
python scripts\10_generate_mobile_models.py
python scripts\11_test_mobile_models.py
```

Open the variant selected in `configs/robot.yaml`:

```powershell
python scripts\12_open_mobile_robot.py
```

The wheeled model contains a freejoint chassis, two wheel hinge joints with velocity actuators, and two ball-joint caster spheres. The virtual fallback contains X, Y and yaw joints with position actuators and four low-friction support spheres.

## Scope and limitations

- The wheel axis/orientation and forward direction still require viewer validation and a motor-direction test.
- No path follower or parking controller is included in this increment.
- The virtual base is a holonomic position-actuated fallback, not the old pre-existing implementation; the uploaded repository had no such implementation.
- No tipping, collision-free navigation, slip, wheel torque, or parking-success claims have been tested yet.
- All physical/controller values in `configs/robot.yaml` are provisional and need tuning.
