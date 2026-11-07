"""Integration physics tests for the generated library scene and world.

Uses MuJoCo simulation to verify model loading, contacts, and physical stability.
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from library_robot.sim.scene_builder import (
    load_layout_config,
    build_library_scene_xml,
    build_library_world_xml,
    DEFAULT_LAYOUT_PATH,
    DEFAULT_ROBOT_CONFIG_PATH,
)
from library_robot.sim.mobile_model import _load_config as _load_robot_config


def test_library_scene_loads_in_mujoco(tmp_path):
    """Verify that generated library_scene.xml loads and compiles in MuJoCo."""
    mujoco = pytest.importorskip("mujoco")
    layout = load_layout_config(DEFAULT_LAYOUT_PATH)
    scene_xml = tmp_path / "test_scene.xml"
    build_library_scene_xml(layout, output_path=scene_xml)

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert model.nbody == 20  # worldbody + 4 walls(geoms in world) + 6 tables + 1 shelf + 12 books + window = 20 bodies
    assert model.ngeom == 38
    assert model.njnt == 12   # 12 freejoints for 12 books
    assert model.nu == 0      # no actuators in static scene


def test_library_world_loads_in_mujoco(tmp_path):
    """Verify that generated library_world.xml (scene + mobile robot) loads cleanly."""
    mujoco = pytest.importorskip("mujoco")
    layout = load_layout_config(DEFAULT_LAYOUT_PATH)
    world_xml = tmp_path / "test_world.xml"
    build_library_world_xml(layout, DEFAULT_ROBOT_CONFIG_PATH, output_path=world_xml)

    model = mujoco.MjModel.from_xml_path(str(world_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert model.nbody == 36
    assert model.ngeom == 124
    assert model.nu == 10     # 2 wheel velocity actuators + 7 arm joint pos + 1 gripper
    assert model.opt.timestep == 0.002

    # Check that robot chassis and arm exist
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mobile_chassis") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand") >= 0


def test_library_world_idle_stability_physics(tmp_path):
    """Verify 2-second physics stability of parked robot and standing books."""
    mujoco = pytest.importorskip("mujoco")
    layout = load_layout_config(DEFAULT_LAYOUT_PATH)
    robot_cfg = _load_robot_config(DEFAULT_ROBOT_CONFIG_PATH)

    world_xml = tmp_path / "test_world_stability.xml"
    build_library_world_xml(layout, DEFAULT_ROBOT_CONFIG_PATH, output_path=world_xml)

    model = mujoco.MjModel.from_xml_path(str(world_xml))
    data = mujoco.MjData(model)

    # Set tucked arm pose
    arm_targets = robot_cfg["panda_tucked"]["arm_joints"]
    for jname, target in arm_targets.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            data.qpos[int(model.jnt_qposadr[jid])] = float(target)
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{jname[-1]}")
            if act_id >= 0:
                data.ctrl[act_id] = float(target)

    mujoco.mj_forward(model, data)

    # Settle for 1.0 s (500 steps)
    for _ in range(500):
        mujoco.mj_step(model, data)

    book_bids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b["name"])
        for b in layout["books"]
    ]
    settled_pos = [data.xpos[bid].copy() for bid in book_bids]
    chassis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mobile_chassis")
    settled_chassis_pos = data.xpos[chassis_bid].copy()

    # Step another 1.0 s (500 steps)
    for _ in range(500):
        mujoco.mj_step(model, data)

    # Check book displacements <= 2 mm
    for bid, spos in zip(book_bids, settled_pos):
        disp = float(math.sqrt(sum((data.xpos[bid] - spos) ** 2)))
        assert disp <= 0.002, f"Book {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)} moved {disp*1000:.3f} mm > 2 mm"

    # Check base drift <= 2 mm
    base_drift = float(math.sqrt(sum((data.xpos[chassis_bid] - settled_chassis_pos) ** 2)))
    assert base_drift <= 0.002, f"Robot base crept by {base_drift*1000:.3f} mm > 2 mm"
