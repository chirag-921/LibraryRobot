"""Build and load mobile Panda models without modifying the Menagerie source.

The module generates two MJCF variants from the pristine Panda model:
- wheeled: freejoint chassis, two driven wheels, two ball-joint casters
- virtual: three-DoF holonomic chassis (x, y, yaw) with position actuators

All proposed physical/controller values are supplied by configs/robot.yaml.
"""

from __future__ import annotations

from pathlib import Path
import math
import xml.etree.ElementTree as ET
from typing import Any

_MUJOCO_IMPORT_ERROR: Exception | None = None
try:
    import mujoco
except (ImportError, OSError) as exc:
    # XML generation only edits MJCF and does not need the MuJoCo DLL.
    # On Windows, Application Control can raise OSError (WinError 4551)
    # while loading mujoco.dll; preserve that detail for runtime diagnostics.
    mujoco = None
    _MUJOCO_IMPORT_ERROR = exc

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "robot.yaml"
PANDA_XML_PATH = (
    PROJECT_ROOT
    / "assets"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "panda.xml"
)
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "generated"

def _fmt(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def _load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Robot configuration not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid robot configuration: {path}")
    base_type = config.get("base_type")
    if base_type not in {"wheeled", "virtual"}:
        raise ValueError("base_type must be either 'wheeled' or 'virtual'")
    return config


def _add_box(
    parent: ET.Element,
    *,
    name: str,
    half_size: list[float],
    position: list[float] | tuple[float, float, float] = (0.0, 0.0, 0.0),
    rgba: str = "0.22 0.25 0.29 1",
    mass: float | None = None,
) -> ET.Element:
    attrs = {
        "name": name,
        "type": "box",
        "pos": _fmt(position),
        "size": _fmt(half_size),
        "rgba": rgba,
    }
    if mass is not None:
        attrs["mass"] = f"{float(mass):.9g}"
    return ET.SubElement(parent, "geom", attrs)


def _add_wheeled_base(
    worldbody: ET.Element,
    actuator: ET.Element,
    base: dict[str, Any],
    panda_root: ET.Element,
) -> None:
    wheel = base["wheeled"]
    hx, hy, hz = map(float, base["chassis_half_size"])
    wheel_radius = float(wheel["wheel_radius"])
    wheel_separation = float(wheel["wheel_separation"])
    wheel_width = float(wheel["wheel_width"])
    base_center_z = wheel_radius + hz

    initial_x, initial_y, initial_yaw = map(float, base["initial_pose_xyyaw"])
    chassis = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "mobile_chassis",
            "pos": _fmt([initial_x, initial_y, base_center_z]),
        },
    )
    ET.SubElement(chassis, "freejoint", {"name": "base_freejoint"})
    _add_box(
        chassis,
        name="chassis_collision",
        half_size=[hx, hy, hz],
        rgba="0.20 0.23 0.27 1",
        mass=float(base["chassis_mass"]),
    )
    wheel_quat = "0.7071067812 0.7071067812 0 0"  # cylinder axis aligned with local Y
    wheel_z = -hz
    for side, y in (("left", wheel_separation / 2.0), ("right", -wheel_separation / 2.0)):
        wheel_body = ET.SubElement(
            chassis,
            "body",
            {"name": f"{side}_wheel", "pos": _fmt([0.0, y, wheel_z])},
        )
        ET.SubElement(
            wheel_body,
            "joint",
            {
                "name": f"{side}_wheel_hinge",
                "type": "hinge",
                "axis": "0 1 0",
                "damping": str(float(wheel["wheel_joint_damping"])),
                "armature": str(float(wheel["wheel_joint_armature"])),
            },
        )
        ET.SubElement(
            wheel_body,
            "geom",
            {
                "name": f"{side}_wheel_geom",
                "type": "cylinder",
                "size": _fmt([wheel_radius, wheel_width / 2.0]),
                "quat": wheel_quat,
                "mass": str(float(wheel["wheel_mass"])),
                "friction": _fmt(wheel["wheel_friction"]),
                "rgba": "0.055 0.06 0.07 1",
                "condim": "3",
            },
        )
        ET.SubElement(
            actuator,
            "velocity",
            {
                "name": f"{side}_wheel_velocity",
                "joint": f"{side}_wheel_hinge",
                "kv": str(float(wheel["wheel_velocity_gain"])),
                "ctrlrange": _fmt([
                    -float(wheel["wheel_speed_limit"]),
                    float(wheel["wheel_speed_limit"]),
                ]),
                "forcelimited": "true",
                "forcerange": _fmt([
                    -float(wheel["wheel_torque_limit"]),
                    float(wheel["wheel_torque_limit"]),
                ]),
            },
        )

    # Ball-joint caster spheres are free to rotate in any direction.
    # Their contact friction approximates low-friction omnidirectional casters.
    caster_radius = float(wheel["caster_radius"])
    caster_z = -(hz + caster_radius)
    for side, x in (("front", float(wheel["caster_x_offset"])), ("rear", -float(wheel["caster_x_offset"]))):
        caster_body = ET.SubElement(
            chassis,
            "body",
            {"name": f"{side}_ball_caster", "pos": _fmt([x, 0.0, caster_z])},
        )
        ET.SubElement(
            caster_body,
            "joint",
            {
                "name": f"{side}_caster_ball_joint",
                "type": "ball",
                "damping": str(float(wheel["caster_joint_damping"])),
            },
        )
        ET.SubElement(
            caster_body,
            "geom",
            {
                "name": f"{side}_caster_sphere",
                "type": "sphere",
                "size": str(caster_radius),
                "mass": str(float(wheel["caster_mass"])),
                "friction": _fmt(wheel["caster_friction"]),
                "rgba": "0.16 0.17 0.19 1",
                "condim": "3",
            },
        )

    panda_mount = list(map(float, base["panda_mount_position"]))
    panda_root.set("pos", _fmt(panda_mount))
    _apply_panda_gravcomp(panda_root, float(base["panda_gravity_compensation"]))
    chassis.append(panda_root)


def _add_virtual_base(
    worldbody: ET.Element,
    actuator: ET.Element,
    base: dict[str, Any],
    panda_root: ET.Element,
) -> None:
    virtual = base["virtual"]
    hx, hy, hz = map(float, base["chassis_half_size"])
    # Low-friction support spheres touch the floor while keeping the Panda
    # mount height consistent with the wheeled option for the starting config.
    support_radius = float(virtual["support_radius"])
    base_center_z = hz + 2.0 * support_radius

    chassis = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "mobile_chassis",
            "pos": _fmt([0.0, 0.0, base_center_z]),
        },
    )
    ET.SubElement(
        chassis,
        "joint",
        {
            "name": "base_x",
            "type": "slide",
            "axis": "1 0 0",
            "range": _fmt(virtual["x_range"]),
            "limited": "true",
        },
    )
    ET.SubElement(
        chassis,
        "joint",
        {
            "name": "base_y",
            "type": "slide",
            "axis": "0 1 0",
            "range": _fmt(virtual["y_range"]),
            "limited": "true",
        },
    )
    ET.SubElement(
        chassis,
        "joint",
        {
            "name": "base_yaw",
            "type": "hinge",
            "axis": "0 0 1",
            "range": _fmt(virtual["yaw_range"]),
            "limited": "true",
        },
    )
    _add_box(
        chassis,
        name="chassis_collision",
        half_size=[hx, hy, hz],
        rgba="0.20 0.23 0.27 1",
        mass=float(base["chassis_mass"]),
    )
    # Fixed low-friction spheres act as sliding supports for this idealized
    # holonomic fallback; unlike the wheeled option, the base is kinematic-like.
    support_z = -(hz + support_radius)
    for sx, sy, label in (
        (-1, -1, "rear_right"), (-1, 1, "rear_left"),
        (1, -1, "front_right"), (1, 1, "front_left"),
    ):
        ET.SubElement(
            chassis,
            "geom",
            {
                "name": f"virtual_support_{label}",
                "type": "sphere",
                "pos": _fmt([sx * float(virtual["support_x_offset"]), sy * float(virtual["support_y_offset"]), support_z]),
                "size": str(support_radius),
                "mass": str(float(virtual["support_mass"])),
                "friction": _fmt(virtual["support_friction"]),
                "rgba": "0.12 0.13 0.15 1",
                "condim": "3",
            },
        )
    panda_mount = list(map(float, base["panda_mount_position"]))
    panda_root.set("pos", _fmt(panda_mount))
    _apply_panda_gravcomp(panda_root, float(base["panda_gravity_compensation"]))
    chassis.append(panda_root)

    for name, joint, kp, kv, ctrlrange, forcerange in (
        ("base_x_position", "base_x", virtual["position_kp"], virtual["position_kv"], virtual["x_range"], [-virtual["position_force_limit"], virtual["position_force_limit"]]),
        ("base_y_position", "base_y", virtual["position_kp"], virtual["position_kv"], virtual["y_range"], [-virtual["position_force_limit"], virtual["position_force_limit"]]),
        ("base_yaw_position", "base_yaw", virtual["yaw_kp"], virtual["yaw_kv"], virtual["yaw_range"], [-virtual["yaw_torque_limit"], virtual["yaw_torque_limit"]]),
    ):
        ET.SubElement(
            actuator,
            "position",
            {
                "name": name,
                "joint": joint,
                "kp": str(float(kp)),
                "kv": str(float(kv)),
                "ctrlrange": _fmt(ctrlrange),
                "forcelimited": "true",
                "forcerange": _fmt(forcerange),
            },
        )


def _apply_panda_gravcomp(panda_root: ET.Element, gravcomp: float) -> None:
    for body in panda_root.iter("body"):
        body.set("gravcomp", str(gravcomp))


def _build_one_model(base_type: str, config: dict[str, Any]) -> Path:
    if not PANDA_XML_PATH.exists():
        raise FileNotFoundError(f"Panda XML not found: {PANDA_XML_PATH}")

    tree = ET.parse(PANDA_XML_PATH)
    root = tree.getroot()
    root.set("model", f"library_robot_{base_type}")

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", Path("../../assets/mujoco_menagerie/franka_emika_panda/assets").as_posix())

    simulation = config["simulation"]
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(float(simulation["timestep"])))
    option.set("gravity", _fmt(simulation["gravity"]))
    option.set("integrator", str(simulation.get("integrator", "implicitfast")))

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Panda MJCF does not contain <worldbody>")
    panda_root = worldbody.find("body[@name='link0']")
    if panda_root is None:
        raise ValueError("Panda MJCF does not contain the expected link0 root body")
    worldbody.remove(panda_root)

    # Replace the worldbody with a ground plane and a tracking light while
    # retaining the original Panda body tree, assets, tendons and actuators.
    for child in list(worldbody):
        worldbody.remove(child)
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "mobile_robot_key_light",
            "pos": _fmt(simulation["key_light_position"]),
            "dir": _fmt(simulation["key_light_direction"]),
            "directional": "true",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "ground",
            "type": "plane",
            "pos": "0 0 0",
            "size": "0 0 0.05",
            "rgba": "0.28 0.31 0.35 1",
            "friction": _fmt(simulation["ground_friction"]),
        },
    )

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")

    if base_type == "wheeled":
        _add_wheeled_base(worldbody, actuator, config["base"], panda_root)
    elif base_type == "virtual":
        _add_virtual_base(worldbody, actuator, config["base"], panda_root)
    else:
        raise ValueError(f"Unsupported base type: {base_type}")

    # The Menagerie keyframe describes the fixed Panda only. Remove it because
    # the new base introduces additional generalized coordinates; the loader
    # initializes named joints safely instead.
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)

    # Keep the original Panda self-collision exclusion.
    output_dir = MODEL_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{base_type}_robot.xml"

    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def generate_mobile_models(config_path: str | Path | None = None) -> dict[str, Path]:
    """Generate both selectable base variants from the same Panda source."""
    config = _load_config(config_path)
    return {
        "wheeled": _build_one_model("wheeled", config),
        "virtual": _build_one_model("virtual", config),
    }


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"Required joint is missing from mobile model: {name}")
    return joint_id


def _actuator_id(model: mujoco.MjModel, name: str) -> int:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator_id < 0:
        raise ValueError(f"Required actuator is missing from mobile model: {name}")
    return actuator_id


def initialize_mobile_robot(model: mujoco.MjModel, data: mujoco.MjData, base_type: str, config: dict[str, Any]) -> None:
    """Initialize chassis pose, Panda home joints and all actuator targets."""
    base = config["base"]
    x, y, yaw = map(float, base["initial_pose_xyyaw"])

    if base_type == "wheeled":
        free_id = _joint_id(model, "base_freejoint")
        adr = int(model.jnt_qposadr[free_id])
        base_z = float(base["wheeled"]["wheel_radius"]) + float(base["chassis_half_size"][2])
        data.qpos[adr:adr + 7] = [x, y, base_z, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        for caster_name in ("front_caster_ball_joint", "rear_caster_ball_joint"):
            caster_id = _joint_id(model, caster_name)
            caster_adr = int(model.jnt_qposadr[caster_id])
            data.qpos[caster_adr:caster_adr + 4] = [1.0, 0.0, 0.0, 0.0]
        for wheel_name in ("left_wheel_hinge", "right_wheel_hinge"):
            wheel_id = _joint_id(model, wheel_name)
            data.qpos[int(model.jnt_qposadr[wheel_id])] = 0.0
    elif base_type == "virtual":
        for name, value in (("base_x", x), ("base_y", y), ("base_yaw", yaw)):
            joint_id = _joint_id(model, name)
            data.qpos[int(model.jnt_qposadr[joint_id])] = value
    else:
        raise ValueError(f"Unsupported base type: {base_type}")

    for name, target in config["panda_home"]["arm_joints"].items():
        joint_id = _joint_id(model, name)
        qadr = int(model.jnt_qposadr[joint_id])
        data.qpos[qadr] = target
        actuator_id = _actuator_id(model, f"actuator{name[-1]}")
        data.ctrl[actuator_id] = target

    finger_positions = config["panda_home"]["finger_positions"]
    for name, target in zip(("finger_joint1", "finger_joint2"), finger_positions):
        joint_id = _joint_id(model, name)
        data.qpos[int(model.jnt_qposadr[joint_id])] = float(target)
    data.ctrl[_actuator_id(model, "actuator8")] = float(
        config["panda_home"]["gripper_actuator_command"]
    )

    if base_type == "wheeled":
        data.ctrl[_actuator_id(model, "left_wheel_velocity")] = 0.0
        data.ctrl[_actuator_id(model, "right_wheel_velocity")] = 0.0
    else:
        for name, value in (("base_x_position", x), ("base_y_position", y), ("base_yaw_position", yaw)):
            data.ctrl[_actuator_id(model, name)] = value

    mujoco.mj_forward(model, data)


def load_mobile_robot_model(
    config_path: str | Path | None = None,
    base_type: str | None = None,
    generate_if_missing: bool = False,
) -> tuple[mujoco.MjModel, mujoco.MjData, str]:
    """Load the configured mobile Panda model and initialize its state."""
    if mujoco is None:
        detail = (
            f" Original import error: {_MUJOCO_IMPORT_ERROR}"
            if _MUJOCO_IMPORT_ERROR is not None
            else ""
        )
        raise ImportError(
            "MuJoCo runtime is unavailable. MJCF generation can still run, "
            "but model compilation, physics and the viewer require the "
            "MuJoCo DLL to load successfully." + detail
        ) from _MUJOCO_IMPORT_ERROR
    config = _load_config(config_path)
    selected = base_type or config["base_type"]
    if selected not in {"wheeled", "virtual"}:
        raise ValueError("base_type must be 'wheeled' or 'virtual'")

    model_path = MODEL_OUTPUT_DIR / f"{selected}_robot.xml"
    if not model_path.exists():
        if not generate_if_missing:
            raise FileNotFoundError(
                f"Generated model not found: {model_path}. "
                "Run scripts/10_generate_mobile_models.py first."
            )
        generate_mobile_models(config_path)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    initialize_mobile_robot(model, data, selected, config)
    return model, data, selected
