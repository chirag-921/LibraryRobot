"""MuJoCo-independent structural tests for generated mobile robot XML."""

from pathlib import Path
import xml.etree.ElementTree as ET

from library_robot.sim.mobile_model import generate_mobile_models


def _find_named(parent, tag, name):
    return parent.find(f".//{tag}[@name='{name}']")


def test_generate_both_mobile_variants():
    paths = generate_mobile_models()
    assert set(paths) == {"wheeled", "virtual"}

    for base_type, path in paths.items():
        assert path.exists()
        root = ET.parse(path).getroot()
        assert root.tag == "mujoco"
        chassis = _find_named(root, "body", "mobile_chassis")
        assert chassis is not None
        assert _find_named(chassis, "body", "link0") is not None

        # Preserve all Panda joints and eight actuators.
        for i in range(1, 8):
            assert _find_named(root, "joint", f"joint{i}") is not None
            assert _find_named(root, "general", f"actuator{i}") is not None
        assert _find_named(root, "joint", "finger_joint1") is not None
        assert _find_named(root, "joint", "finger_joint2") is not None
        assert _find_named(root, "general", "actuator8") is not None

        if base_type == "wheeled":
            assert _find_named(root, "freejoint", "base_freejoint") is not None
            assert _find_named(root, "velocity", "left_wheel_velocity") is not None
            assert _find_named(root, "velocity", "right_wheel_velocity") is not None
            assert _find_named(root, "joint", "front_caster_ball_joint") is not None
            assert _find_named(root, "joint", "rear_caster_ball_joint") is not None
        else:
            for label in ("rear_right", "rear_left", "front_right", "front_left"):
                assert _find_named(root, "geom", f"virtual_support_{label}") is not None
            for name in ("base_x", "base_y", "base_yaw"):
                assert _find_named(root, "joint", name) is not None
            assert _find_named(root, "freejoint", "base_freejoint") is None
