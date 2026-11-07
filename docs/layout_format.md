# Library Layout Configuration Specification

This document defines the schema and naming conventions of [`configs/library_layout.yaml`](file:///C:/Robotics/LibraryRobot/configs/library_layout.yaml).
Both the simulation scene generator and high-level task planners (rule-based baseline and LLM planner) reference these exact keys and names.

---

## 1. Top-Level Keys

| Key | Type | Description |
| :--- | :--- | :--- |
| `seed` | `int` | Deterministic random seed for book position and orientation jitter. |
| `clearance_mm` | `float` | Minimum allowable gap between book spines in shelf compartments (default: 25.0 mm). |
| `room` | `dict` | Enclosing room dimensions and boundary properties. |
| `landmarks` | `dict` | High-level visual references and designated stations. |
| `robot_start` | `dict` | Initial spawn pose and station name for the mobile robot. |
| `tables` | `list` | List of reading/study tables. |
| `shelf` | `dict` | Bookshelf configuration, structural boards, and compartment grid. |
| `books` | `list` | Manifest of all manipulable books in the environment. |

---

## 2. Room & Landmarks

### `room`
- `size`: `[length_x, width_y, height_z]` in metres (e.g., `[8.0, 8.0, 2.5]`).
- `wall_thickness`: `float` in metres (e.g., `0.10`).
- `floor_friction`: `[sliding, torsional, rolling]` friction coefficients (e.g., `[1.0, 0.01, 0.001]`).

### `landmarks`
- `window`: Position `[x, y, z]` and size `[dx, dy, dz]` on an exterior wall. Visual only.
  - Positioned on the East wall ($x = 3.95$) near Shelf A, making **Compartment A3** the unique answer to "the empty compartment nearest the window".
- `charging_dock`: Position `[x, y, z]` and associated proximity reference (`near_table`).

### `robot_start`
- `x`, `y`: `float` coordinates in world frame (metres).
- `yaw`: `float` heading angle in radians (e.g., $+1.570796$ facing North).
- `location_name`: `string` identifier (e.g., `"charging_dock"`).

---

## 3. Tables (`tables`)

Each entry in the `tables` list defines a fixed table:
- `name`: `string`, strictly named `"Table A"` through `"Table F"`.
- `pos`: `[x, y, z]` center position of the table base on the floor ($z=0.0$).
- `yaw`: `float` orientation in radians.
- `size`: `[length_x, depth_y, height_z]` in metres (e.g., `[1.0, 0.40, 0.55]`).
  - Table surface height is at $z = \text{height\_z} = 0.55\text{ m}$.
  - Depth of $0.40\text{ m}$ guarantees all books across the surface are reachable within the Franka Panda reach envelope ($R \le 0.805\text{ m}$ at wrist).

---

## 4. Bookshelf & Compartments (`shelf`)

- `name`: `string`, `"Shelf A"`.
- `pos`: `[x, y, z]` world position of the shelf frame base.
- `yaw`: `float` rotation angle.
- `size`: `[outer_width_x, outer_depth_y, outer_height_z]`.
- `rows`, `columns`: Grid dimensions (e.g., 2 rows $\times$ 3 columns = 6 compartments).
- `compartments`: List of compartment descriptors:
  - `id`: `string`, strictly `"A1"` to `"A6"`.
    - Row 1 (bottom shelf): `A1` (left/West), `A2` (center), `A3` (right/East).
    - Row 2 (top shelf): `A4` (left/West), `A5` (center), `A6` (right/East).
  - `row`, `col`: 1-based integer indices.
  - `relative_center`: `[dx, dy, dz]` offset from shelf frame origin.
  - `surface_z`: `float` elevation of the compartment floor above ground.

---

## 5. Books (`books`)

Each book is modeled as an independent rigid box with a 6-DoF `freejoint`:
- `name`: `string` body identifier in the MJCF model, formatted as `"book_<topic>"` (e.g., `"book_machine_learning"`).
- `title`: `string` human-readable book title (e.g., `"Machine Learning"`).
- `location`: Current table name (`"Table B"`) or compartment ID (`"A1"`).
- `slot_index`: Integer offset index along the surface width for multi-book placement.
- `size`: `[width_depth, thickness, height]` in metres (e.g., `[0.15, 0.03, 0.23]`).
- `mass`: `float` in kg (default: $0.40\text{ kg}$).
- `colour`: `[R, G, B, A]` RGBA list with values normalized between $0.0$ and $1.0$.
- `orientation`: `"upright"` (spine standing vertically).
