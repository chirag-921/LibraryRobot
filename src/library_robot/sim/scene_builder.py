"""Library scene and world XML generator for MuJoCo simulation.

Parses configs/library_layout.yaml and generates:
  - models/generated/library_scene.xml (static environment, tables, shelf, books)
  - models/generated/library_world.xml (scene + mobile robot at robot_start)

Offline capable: layout validation and XML generation run purely in Python
without importing the MuJoCo runtime DLL.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import yaml

from library_robot.sim.mobile_model import (
    PANDA_XML_PATH,
    _load_config as _load_robot_config,
    _add_wheeled_base,
    _add_virtual_base,
    _fmt,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LAYOUT_PATH = PROJECT_ROOT / "configs" / "library_layout.yaml"
DEFAULT_ROBOT_CONFIG_PATH = PROJECT_ROOT / "configs" / "robot.yaml"
OUTPUT_DIR = PROJECT_ROOT / "models" / "generated"


def load_layout_config(layout_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Loads and returns the YAML layout configuration."""
    path = Path(layout_path) if layout_path else DEFAULT_LAYOUT_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Library layout configuration not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid library layout format: {path}")
    return config


class LayoutValidationError(ValueError):
    """Raised when the library layout violates physical or structural constraints."""
    pass


def validate_layout(layout: Dict[str, Any]) -> None:
    """
    Validates library layout configuration for consistency and physical plausibility.
    Does not import MuJoCo.
    """
    # 1. Required top-level sections
    required_keys = {"room", "tables", "shelf", "books", "robot_start"}
    missing = required_keys - set(layout.keys())
    if missing:
        raise LayoutValidationError(f"Missing required layout sections: {missing}")

    # 2. Room bounds
    room_size = layout["room"]["size"]
    half_rx = room_size[0] / 2.0
    half_ry = room_size[1] / 2.0

    # 3. Tables validation & name uniqueness
    table_names = set()
    table_map = {}
    for table in layout["tables"]:
        name = table["name"]
        if name in table_names:
            raise LayoutValidationError(f"Duplicate table name: {name}")
        table_names.add(name)
        table_map[name] = table

        # Inside room boundary
        tx, ty, _ = table["pos"]
        length, width, _ = table["size"]
        if abs(tx) + length / 2.0 > half_rx or abs(ty) + width / 2.0 > half_ry:
            raise LayoutValidationError(f"Table '{name}' extends outside room boundary ({tx}, {ty})")

    # 4. Shelf & compartments validation
    shelf = layout["shelf"]
    sx, sy, _ = shelf["pos"]
    sw, sd, _ = shelf["size"]
    if abs(sx) + sw / 2.0 > half_rx or abs(sy) + sd / 2.0 > half_ry:
        raise LayoutValidationError(f"Shelf '{shelf['name']}' extends outside room boundary ({sx}, {sy})")

    compartment_ids = set()
    compartment_map = {}
    for comp in shelf["compartments"]:
        cid = comp["id"]
        if cid in compartment_ids:
            raise LayoutValidationError(f"Duplicate compartment ID: {cid}")
        compartment_ids.add(cid)
        compartment_map[cid] = comp

    # 5. Table-table and table-shelf overlap check
    all_obstacles = []
    for t in layout["tables"]:
        all_obstacles.append((t["name"], t["pos"][0], t["pos"][1], t["size"][0], t["size"][1]))
    all_obstacles.append((shelf["name"], shelf["pos"][0], shelf["pos"][1], shelf["size"][0], shelf["size"][1]))

    for i in range(len(all_obstacles)):
        for j in range(i + 1, len(all_obstacles)):
            name_a, ax, ay, alx, aly = all_obstacles[i]
            name_b, bx, by, blx, bly = all_obstacles[j]
            overlap_x = abs(ax - bx) < (alx + blx) / 2.0
            overlap_y = abs(ay - by) < (aly + bly) / 2.0
            if overlap_x and overlap_y:
                raise LayoutValidationError(f"Physical overlap detected between '{name_a}' and '{name_b}'")

    # 6. Robot start inside room
    rstart = layout["robot_start"]
    if abs(rstart["x"]) > half_rx - 0.5 or abs(rstart["y"]) > half_ry - 0.5:
        raise LayoutValidationError(f"Robot start pose ({rstart['x']}, {rstart['y']}) is outside safe room bounds")

    # 7. Books validation
    book_names = set()
    compartment_books: Dict[str, List[Dict[str, Any]]] = {cid: [] for cid in compartment_ids}
    clearance_m = float(layout.get("clearance_mm", 25.0)) / 1000.0

    for book in layout["books"]:
        bname = book["name"]
        if bname in book_names:
            raise LayoutValidationError(f"Duplicate book name: {bname}")
        book_names.add(bname)

        loc = book["location"]
        if loc not in table_names and loc not in compartment_ids:
            raise LayoutValidationError(
                f"Book '{bname}' specifies unknown location '{loc}'. Valid locations: {sorted(table_names | compartment_ids)}"
            )

        # Check compartment fit and headroom
        if loc in compartment_ids:
            compartment_books[loc].append(book)
            book_height = book["size"][2]
            comp_height = 0.32  # compartment inside height
            headroom = comp_height - book_height
            if headroom < 0.05:
                raise LayoutValidationError(
                    f"Book '{bname}' height {book_height} m leaves insufficient headroom ({headroom*100:.1f} cm < 5 cm) in {loc}"
                )

    # Check total width of books in each compartment with clearance
    comp_width = 0.38  # inside width
    for cid, b_list in compartment_books.items():
        if not b_list:
            continue
        total_thickness = sum(b["size"][1] for b in b_list)
        total_gaps = max(0, len(b_list) - 1) * clearance_m
        if total_thickness + total_gaps > comp_width:
            raise LayoutValidationError(
                f"Books in compartment {cid} exceed compartment width ({total_thickness+total_gaps:.3f} m > {comp_width:.3f} m)"
            )


def _add_box_geom(
    parent: ET.Element,
    *,
    name: str,
    half_size: List[float],
    pos: List[float] | Tuple[float, float, float] = (0.0, 0.0, 0.0),
    rgba: str = "0.8 0.8 0.8 1",
    friction: Optional[List[float]] = None,
    mass: Optional[float] = None,
    contype: int = 1,
    conaffinity: int = 1,
) -> ET.Element:
    attrs = {
        "name": name,
        "type": "box",
        "pos": _fmt(pos),
        "size": _fmt(half_size),
        "rgba": rgba,
        "contype": str(contype),
        "conaffinity": str(conaffinity),
    }
    if friction is not None:
        attrs["friction"] = _fmt(friction)
    if mass is not None:
        attrs["mass"] = f"{float(mass):.6g}"
    return ET.SubElement(parent, "geom", attrs)


def build_library_scene_xml(
    layout_config: Dict[str, Any],
    output_path: Optional[Path] = None,
    seed_override: Optional[int] = None,
) -> Path:
    """
    Generates library_scene.xml (room, tables, shelf, and standing books).
    Runs offline without requiring MuJoCo.
    """
    validate_layout(layout_config)
    seed = seed_override if seed_override is not None else int(layout_config.get("seed", 42))
    rng = random.Random(seed)

    root = ET.Element("mujoco", {"model": "library_scene"})
    compiler = ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})

    # Simulation options matching robot.yaml
    ET.SubElement(
        root,
        "option",
        {
            "integrator": "implicitfast",
            "timestep": "0.002",
            "gravity": "0 0 -9.81",
        },
    )

    # Offscreen buffer sizing for high-res camera renders
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1920", "offheight": "1080"})

    worldbody = ET.SubElement(root, "worldbody")

    # 1. Room Floor & Walls
    room = layout_config["room"]
    rx, ry, rz = room["size"]
    floor_friction = room.get("floor_friction", [1.0, 0.01, 0.001])
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "ground",
            "type": "plane",
            "pos": "0 0 0",
            "size": _fmt([rx / 2.0 + 1.0, ry / 2.0 + 1.0, 0.05]),
            "rgba": "0.85 0.87 0.88 1",
            "friction": _fmt(floor_friction),
        },
    )

    # 4 Walls
    wall_t = float(room.get("wall_thickness", 0.10))
    wall_h = rz
    # North wall (+Y), South wall (-Y), East wall (+X), West wall (-X)
    walls = [
        ("wall_north", [0.0, ry / 2.0 + wall_t / 2.0, wall_h / 2.0], [rx / 2.0 + wall_t, wall_t / 2.0, wall_h / 2.0]),
        ("wall_south", [0.0, -ry / 2.0 - wall_t / 2.0, wall_h / 2.0], [rx / 2.0 + wall_t, wall_t / 2.0, wall_h / 2.0]),
        ("wall_east", [rx / 2.0 + wall_t / 2.0, 0.0, wall_h / 2.0], [wall_t / 2.0, ry / 2.0, wall_h / 2.0]),
        ("wall_west", [-rx / 2.0 - wall_t / 2.0, 0.0, wall_h / 2.0], [wall_t / 2.0, ry / 2.0, wall_h / 2.0]),
    ]
    for wname, wpos, wsize in walls:
        _add_box_geom(worldbody, name=wname, half_size=wsize, pos=wpos, rgba="0.92 0.93 0.94 1")

    # Window landmark (visual box marker on wall)
    window_cfg = layout_config.get("landmarks", {}).get("window")
    if window_cfg:
        _add_box_geom(
            worldbody,
            name="landmark_window",
            half_size=[s / 2.0 for s in window_cfg["size"]],
            pos=window_cfg["pos"],
            rgba="0.5 0.8 0.95 0.7",
            contype=0,
            conaffinity=0,
        )

    # Lights (overhead illumination)
    ET.SubElement(worldbody, "light", {"name": "ceiling_light_center", "pos": "0 0 3.5", "dir": "0 0 -1", "diffuse": "0.8 0.8 0.8"})
    ET.SubElement(worldbody, "light", {"name": "ceiling_light_north", "pos": "0 2.5 3.5", "dir": "0 0 -1", "diffuse": "0.6 0.6 0.6"})
    ET.SubElement(worldbody, "light", {"name": "ceiling_light_south", "pos": "0 -2.5 3.5", "dir": "0 0 -1", "diffuse": "0.6 0.6 0.6"})

    # 2. Fixed Tables (Tables A through F)
    for table in layout_config["tables"]:
        tname = table["name"]
        t_id_str = tname.lower().replace(" ", "_")
        tx, ty, tz = table["pos"]
        length, depth, height = table["size"]  # e.g., 1.0, 0.40, 0.55

        t_body = ET.SubElement(
            worldbody,
            "body",
            {"name": t_id_str, "pos": _fmt([tx, ty, tz])},
        )
        # Table top box
        top_thickness = 0.04
        top_z = height - top_thickness / 2.0
        _add_box_geom(
            t_body,
            name=f"{t_id_str}_top",
            half_size=[length / 2.0, depth / 2.0, top_thickness / 2.0],
            pos=[0.0, 0.0, top_z],
            rgba="0.75 0.58 0.42 1",  # wood tone
            friction=[1.0, 0.01, 0.001],
        )
        # Central pedestal support
        _add_box_geom(
            t_body,
            name=f"{t_id_str}_pedestal",
            half_size=[length / 2.0 - 0.10, depth / 2.0 - 0.06, top_z / 2.0],
            pos=[0.0, 0.0, top_z / 2.0],
            rgba="0.30 0.32 0.35 1",  # dark support base
            friction=[1.0, 0.01, 0.001],
        )

    # 3. Fixed Bookshelf ("Shelf A")
    shelf = layout_config["shelf"]
    s_name = shelf["name"].lower().replace(" ", "_")
    sx, sy, sz = shelf["pos"]
    sw, sd, sh = shelf["size"]  # 1.30, 0.40, 0.85
    bt = float(shelf.get("board_thickness", 0.02))

    shelf_body = ET.SubElement(worldbody, "body", {"name": s_name, "pos": _fmt([sx, sy, sz])})
    shelf_wood_rgba = "0.55 0.38 0.25 1"

    # Outer shell: Back board, Left side, Right side, Top board, Bottom board
    # Back board (+Y side, since opening faces -Y south):
    _add_box_geom(shelf_body, name=f"{s_name}_back", half_size=[sw / 2.0, bt / 2.0, sh / 2.0], pos=[0.0, sd / 2.0 - bt / 2.0, sh / 2.0], rgba=shelf_wood_rgba)
    # Left side (-X):
    _add_box_geom(shelf_body, name=f"{s_name}_left", half_size=[bt / 2.0, sd / 2.0, sh / 2.0], pos=[-sw / 2.0 + bt / 2.0, 0.0, sh / 2.0], rgba=shelf_wood_rgba)
    # Right side (+X):
    _add_box_geom(shelf_body, name=f"{s_name}_right", half_size=[bt / 2.0, sd / 2.0, sh / 2.0], pos=[sw / 2.0 - bt / 2.0, 0.0, sh / 2.0], rgba=shelf_wood_rgba)
    # Bottom board (Z = bt/2):
    _add_box_geom(shelf_body, name=f"{s_name}_bottom", half_size=[sw / 2.0, sd / 2.0, bt / 2.0], pos=[0.0, 0.0, bt / 2.0], rgba=shelf_wood_rgba)
    # Top board (Z = sh - bt/2):
    _add_box_geom(shelf_body, name=f"{s_name}_top", half_size=[sw / 2.0, sd / 2.0, bt / 2.0], pos=[0.0, 0.0, sh - bt / 2.0], rgba=shelf_wood_rgba)
    # Middle shelf board at row split (Z ~ 0.48 m):
    mid_board_z = 0.48
    _add_box_geom(shelf_body, name=f"{s_name}_mid_shelf", half_size=[sw / 2.0 - bt, sd / 2.0 - bt, bt / 2.0], pos=[0.0, 0.0, mid_board_z], rgba=shelf_wood_rgba)
    # Vertical dividers between 3 columns (at X = -sw/6 and +sw/6):
    div_x1 = -sw / 6.0
    div_x2 = sw / 6.0
    _add_box_geom(shelf_body, name=f"{s_name}_div_l", half_size=[bt / 2.0, sd / 2.0 - bt, sh / 2.0 - bt], pos=[div_x1, 0.0, sh / 2.0], rgba=shelf_wood_rgba)
    _add_box_geom(shelf_body, name=f"{s_name}_div_r", half_size=[bt / 2.0, sd / 2.0 - bt, sh / 2.0 - bt], pos=[div_x2, 0.0, sh / 2.0], rgba=shelf_wood_rgba)

    # 4. Books (Manipulable rigid bodies with freejoints)
    table_dict = {t["name"]: t for t in layout_config["tables"]}
    comp_dict = {c["id"]: c for c in shelf["compartments"]}
    clearance_m = float(layout_config.get("clearance_mm", 25.0)) / 1000.0

    # Count books assigned per table to arrange them along table length
    table_books: Dict[str, List[Dict[str, Any]]] = {t["name"]: [] for t in layout_config["tables"]}
    for book in layout_config["books"]:
        if book["location"] in table_dict:
            table_books[book["location"]].append(book)

    for book in layout_config["books"]:
        b_name = book["name"]
        b_width, b_thick, b_height = book["size"]  # e.g., 0.15, 0.03, 0.23
        b_mass = float(book.get("mass", 0.40))
        b_rgba = _fmt(book["colour"])
        loc = book["location"]

        book_body = ET.SubElement(worldbody, "body", {"name": b_name})
        ET.SubElement(book_body, "freejoint", {"name": f"{b_name}_freejoint"})

        # Calculate initial spawn pose (spawning 1 mm above surface as requested in Amendment 6)
        spawn_clearance_z = 0.001  # 1 mm above surface

        if loc in table_dict:
            table = table_dict[loc]
            tx, ty, tz = table["pos"]
            _, tdepth, theight = table["size"]
            t_books = table_books[loc]
            slot_idx = int(book.get("slot_index", 0))

            # Distribute along table length (X)
            num_on_table = len(t_books)
            slot_span = (table["size"][0] - 0.30) / max(1, num_on_table)
            nominal_x = tx - (table["size"][0] - 0.30) / 2.0 + (slot_idx + 0.5) * slot_span
            nominal_y = ty  # near center of table width

            # Apply deterministic jitter from seed (+/- 2 cm, +/- 10 deg yaw)
            jitter_x = rng.uniform(-0.02, 0.02)
            jitter_y = rng.uniform(-0.02, 0.02)
            jitter_yaw = rng.uniform(-math.radians(10.0), math.radians(10.0))

            bx = nominal_x + jitter_x
            by = nominal_y + jitter_y
            bz = theight + b_height / 2.0 + spawn_clearance_z

            # Set position and yaw quaternion
            quat = [math.cos(jitter_yaw / 2.0), 0.0, 0.0, math.sin(jitter_yaw / 2.0)]
            book_body.set("pos", _fmt([bx, by, bz]))
            book_body.set("quat", _fmt(quat))

        elif loc in comp_dict:
            comp = comp_dict[loc]
            # World position of compartment
            cx = sx + comp["relative_center"][0]
            cy = sy + comp["relative_center"][1]
            cz = comp["surface_z"]

            # Multi-book offset along compartment width (X)
            slot_idx = int(book.get("slot_index", 0))
            offset_x = (slot_idx) * (b_thick + clearance_m)

            bx = cx + offset_x
            by = cy - 0.02  # slightly forward inside compartment
            bz = cz + b_height / 2.0 + spawn_clearance_z

            book_body.set("pos", _fmt([bx, by, bz]))
            # Orientation: spine aligned along Y, standing upright
            book_body.set("quat", "1.0 0.0 0.0 0.0")

        # Book collision box
        _add_box_geom(
            book_body,
            name=f"{b_name}_geom",
            half_size=[b_width / 2.0, b_thick / 2.0, b_height / 2.0],
            pos=[0.0, 0.0, 0.0],
            rgba=b_rgba,
            friction=[1.2, 0.01, 0.001],  # high friction for stability
            mass=b_mass,
            contype=1,
            conaffinity=1,
        )

    # Output paths
    out_dir = output_path.parent if output_path else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_path if output_path else out_dir / "library_scene.xml"

    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    tree.write(out_file, encoding="utf-8", xml_declaration=True)
    return out_file


def build_library_world_xml(
    layout_config: Dict[str, Any],
    robot_config_path: Optional[str | Path] = None,
    output_path: Optional[Path] = None,
    seed_override: Optional[int] = None,
) -> Path:
    """
    Composes library_world.xml (scene + mobile robot).

    Clean composition (Amendment 3):
    Extracts the mobile chassis and Panda tree from mobile_model.py,
    avoiding duplicate <geom name="ground">, conflicting lights, or broken mesh paths.
    """
    # 1. First build the scene
    scene_path = build_library_scene_xml(layout_config, seed_override=seed_override)
    scene_tree = ET.parse(scene_path)
    scene_root = scene_tree.getroot()
    scene_root.set("model", "library_world")

    scene_worldbody = scene_root.find("worldbody")
    if scene_worldbody is None:
        raise ValueError("Generated scene does not contain <worldbody>")

    # 2. Load pristine Panda XML to extract mesh/material assets and defaults
    if not PANDA_XML_PATH.exists():
        raise FileNotFoundError(f"Panda XML not found at {PANDA_XML_PATH}")
    panda_tree = ET.parse(PANDA_XML_PATH)
    panda_root_elem = panda_tree.getroot()

    out_dir = output_path.parent if output_path else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_path if output_path else out_dir / "library_world.xml"

    # Set compiler meshdir relative to output XML location
    compiler = scene_root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(scene_root, "compiler")
    panda_assets_dir = (PROJECT_ROOT / "assets" / "mujoco_menagerie" / "franka_emika_panda" / "assets").resolve()
    try:
        import os
        rel_meshdir = os.path.relpath(panda_assets_dir, out_file.parent)
        compiler.set("meshdir", rel_meshdir.replace("\\", "/"))
    except ValueError:
        compiler.set("meshdir", str(panda_assets_dir).replace("\\", "/"))

    # Copy defaults from Panda
    panda_default = panda_root_elem.find("default")
    if panda_default is not None:
        scene_root.insert(list(scene_root).index(compiler) + 1, copy.deepcopy(panda_default))

    # Copy assets (materials, meshes) from Panda
    panda_asset = panda_root_elem.find("asset")
    if panda_asset is not None:
        scene_root.insert(list(scene_root).index(compiler) + 2, copy.deepcopy(panda_asset))

    # Copy tendon from Panda
    panda_tendon = panda_root_elem.find("tendon")
    if panda_tendon is not None:
        scene_root.append(copy.deepcopy(panda_tendon))

    # Copy equality from Panda
    panda_equality = panda_root_elem.find("equality")
    if panda_equality is not None:
        scene_root.append(copy.deepcopy(panda_equality))

    # Copy contact from Panda
    panda_contact = panda_root_elem.find("contact")
    if panda_contact is not None:
        scene_root.append(copy.deepcopy(panda_contact))

    # 3. Add Actuator block
    scene_actuator = ET.SubElement(scene_root, "actuator")

    # 4. Compose Robot into Scene Worldbody at robot_start
    robot_cfg = _load_robot_config(robot_config_path or DEFAULT_ROBOT_CONFIG_PATH)
    base_type = robot_cfg.get("base_type", "wheeled")

    # Extract Panda arm root body (link0)
    panda_worldbody = panda_root_elem.find("worldbody")
    if panda_worldbody is None:
        raise ValueError("Panda MJCF missing <worldbody>")
    panda_arm_root = panda_worldbody.find("body[@name='link0']")
    if panda_arm_root is None:
        raise ValueError("Panda MJCF missing root link0 body")

    # Override robot initial pose with layout's robot_start
    rstart = layout_config["robot_start"]
    base_cfg = copy.deepcopy(robot_cfg["base"])
    base_cfg["initial_pose_xyyaw"] = [float(rstart["x"]), float(rstart["y"]), float(rstart["yaw"])]

    if base_type == "wheeled":
        _add_wheeled_base(scene_worldbody, scene_actuator, base_cfg, copy.deepcopy(panda_arm_root))
    elif base_type == "virtual":
        _add_virtual_base(scene_worldbody, scene_actuator, base_cfg, copy.deepcopy(panda_arm_root))
    else:
        raise ValueError(f"Unsupported base type: {base_type}")

    # Copy Panda general actuators from original Panda XML
    orig_actuator = panda_root_elem.find("actuator")
    if orig_actuator is not None:
        for child in orig_actuator:
            scene_actuator.append(copy.deepcopy(child))

    ET.indent(scene_root, space="  ")
    tree = ET.ElementTree(scene_root)
    tree.write(out_file, encoding="utf-8", xml_declaration=True)
    return out_file
