"""Offline unit tests for library layout and XML generation.

These tests run completely offline and DO NOT import or require MuJoCo.
"""

from __future__ import annotations

import copy
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest

from library_robot.sim.scene_builder import (
    load_layout_config,
    validate_layout,
    LayoutValidationError,
    build_library_scene_xml,
    build_library_world_xml,
    DEFAULT_LAYOUT_PATH,
    PROJECT_ROOT,
)


def test_load_and_validate_default_layout():
    """Verify that the default layout config is well-formed and passes validation."""
    layout = load_layout_config(DEFAULT_LAYOUT_PATH)
    validate_layout(layout)

    # 1. Room
    assert layout["room"]["size"] == [8.0, 8.0, 2.5]
    assert layout["room"]["floor_friction"] == [1.0, 0.01, 0.001]

    # 2. Tables (A-F)
    tables = layout["tables"]
    assert len(tables) == 6
    expected_tables = {f"Table {c}" for c in "ABCDEF"}
    assert {t["name"] for t in tables} == expected_tables

    # 3. Shelf (Shelf A with compartments A1-A6)
    shelf = layout["shelf"]
    assert shelf["name"] == "Shelf A"
    comps = shelf["compartments"]
    assert len(comps) == 6
    expected_comps = {f"A{i}" for i in range(1, 7)}
    assert {c["id"] for c in comps} == expected_comps

    # 4. Books
    books = layout["books"]
    assert 10 <= len(books) <= 15
    assert len(books) == 12
    for b in books:
        assert len(b["size"]) == 3
        assert b["size"] == [0.15, 0.03, 0.23]

    # 5. Content constraints
    ml_book = next((b for b in books if b.get("title") == "Machine Learning" or b["name"] == "book_machine_learning"), None)
    assert ml_book is not None
    assert ml_book["location"] == "Table B"

    a3_books = [b for b in books if b["location"] == "A3"]
    assert len(a3_books) == 0  # Compartment A3 must be empty

    # Window landmark
    window = layout["landmarks"]["window"]
    assert window["pos"] == [3.95, 3.65, 1.60]


def test_layout_validation_errors():
    """Verify that validate_layout rejects invalid configurations."""
    base_layout = load_layout_config(DEFAULT_LAYOUT_PATH)

    # Missing section
    bad_layout = copy.deepcopy(base_layout)
    del bad_layout["tables"]
    with pytest.raises(LayoutValidationError, match="Missing required layout sections"):
        validate_layout(bad_layout)

    # Duplicate table name
    bad_layout = copy.deepcopy(base_layout)
    bad_layout["tables"].append(copy.deepcopy(bad_layout["tables"][0]))
    with pytest.raises(LayoutValidationError, match="Duplicate table name"):
        validate_layout(bad_layout)

    # Duplicate compartment ID
    bad_layout = copy.deepcopy(base_layout)
    bad_layout["shelf"]["compartments"].append(copy.deepcopy(bad_layout["shelf"]["compartments"][0]))
    with pytest.raises(LayoutValidationError, match="Duplicate compartment ID"):
        validate_layout(bad_layout)

    # Table out of bounds
    bad_layout = copy.deepcopy(base_layout)
    bad_layout["tables"][0]["pos"] = [10.0, 0.0, 0.0]
    with pytest.raises(LayoutValidationError, match="outside room boundary"):
        validate_layout(bad_layout)

    # Table-table overlap
    bad_layout = copy.deepcopy(base_layout)
    bad_layout["tables"][1]["pos"] = bad_layout["tables"][0]["pos"]
    with pytest.raises(LayoutValidationError, match="Physical overlap detected"):
        validate_layout(bad_layout)

    # Unknown book location
    bad_layout = copy.deepcopy(base_layout)
    bad_layout["books"][0]["location"] = "InvalidLocation"
    with pytest.raises(LayoutValidationError, match="unknown location"):
        validate_layout(bad_layout)


def test_build_library_scene_xml_offline(tmp_path):
    """Verify that build_library_scene_xml generates valid XML without MuJoCo."""
    layout = load_layout_config(DEFAULT_LAYOUT_PATH)
    out_xml = tmp_path / "scene.xml"
    res_path = build_library_scene_xml(layout, output_path=out_xml)
    assert res_path.exists()

    tree = ET.parse(res_path)
    root = tree.getroot()
    assert root.tag == "mujoco"
    assert root.get("model") == "library_scene"

    # Compiler & Option
    compiler = root.find("compiler")
    assert compiler is not None
    assert compiler.get("angle") == "radian"

    option = root.find("option")
    assert option is not None
    assert option.get("timestep") == "0.002"
    assert option.get("gravity") == "0 0 -9.81"

    # Worldbody
    wb = root.find("worldbody")
    assert wb is not None

    # Single ground plane
    grounds = [g for g in wb.findall("geom") if g.get("name") == "ground"]
    assert len(grounds) == 1
    assert grounds[0].get("friction") == "1 0.01 0.001"

    # 4 walls
    walls = [g for g in wb.findall("geom") if g.get("name", "").startswith("wall_")]
    assert len(walls) == 4

    # 6 tables
    table_bodies = [b for b in wb.findall("body") if b.get("name", "").startswith("table_")]
    assert len(table_bodies) == 6

    # 1 shelf
    shelf_bodies = [b for b in wb.findall("body") if b.get("name") == "shelf_a"]
    assert len(shelf_bodies) == 1

    # 12 books
    book_bodies = [b for b in wb.findall("body") if b.get("name", "").startswith("book_")]
    assert len(book_bodies) == 12
    for b in book_bodies:
        fj = b.find("freejoint")
        assert fj is not None
        bg = b.find("geom")
        assert bg is not None
        assert bg.get("type") == "box"


def test_build_library_world_xml_composition_offline(tmp_path):
    """Verify clean composition of library_world.xml without duplicate grounds/lights."""
    layout = load_layout_config(DEFAULT_LAYOUT_PATH)
    out_xml = tmp_path / "world.xml"
    res_path = build_library_world_xml(layout, output_path=out_xml)
    assert res_path.exists()

    tree = ET.parse(res_path)
    root = tree.getroot()
    assert root.tag == "mujoco"
    assert root.get("model") == "library_world"

    wb = root.find("worldbody")
    assert wb is not None

    # Ensure exactly 1 ground geom
    grounds = [g for g in wb.findall("geom") if g.get("name") == "ground"]
    assert len(grounds) == 1

    # Robot chassis
    chassis = wb.find("body[@name='mobile_chassis']")
    assert chassis is not None
    assert chassis.find("freejoint[@name='base_freejoint']") is not None

    # Wheels and casters
    left_wheel = chassis.find("body[@name='left_wheel']")
    assert left_wheel is not None
    assert left_wheel.find("joint[@name='left_wheel_hinge']") is not None

    right_wheel = chassis.find("body[@name='right_wheel']")
    assert right_wheel is not None
    assert right_wheel.find("joint[@name='right_wheel_hinge']") is not None

    # Arm link0 attached to chassis
    link0 = chassis.find("body[@name='link0']")
    assert link0 is not None

    # Actuators
    act = root.find("actuator")
    assert act is not None
    act_names = {child.get("name") for child in act}
    assert "left_wheel_velocity" in act_names
    assert "right_wheel_velocity" in act_names
    assert "actuator1" in act_names  # Panda arm joint 1
    assert "actuator8" in act_names  # Panda gripper
