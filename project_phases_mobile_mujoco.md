# LLM-Guided Autonomous Mobile Librarian Robot: Updated 10-Phase Plan

**Simulator:** MuJoCo (Python 3.12). Not Isaac Sim.
**Robot:** Franka Panda (MuJoCo Menagerie) on an **autonomous wheeled mobile base** (differential drive with motors).
**Gripper:** G3 (3-finger: two holders + one extractor finger) is PRIMARY. G2H (Panda hand + fingertip hook) is backup 1. G2G (plain Panda hand, loosened shelf) is backup 2.
**Team:** two people. Phases 2, 3 and 4 can run in parallel. Phase 8 can start early against a mock executor.

> Status notes below come from my reading of an earlier snapshot of the repo (a zip, not the Git repo). Before starting, the agent must re-check the real repo and correct any status that is wrong. Every number in this document is a starting value that needs tuning.

---

## What "mobile (autonomous)" means in this project

1. **Mobile:** a physical wheeled base. Two powered drive wheels with velocity motors, plus passive casters. The base moves by wheel contact with the floor, not by teleporting x/y/yaw joints.
2. **Autonomous:** after an instruction is given, no human input is needed. The robot plans (planner), drives (path following and parking), manipulates (grasp/place), checks its own result (verify), and recovers (replan).
3. **What the robot senses:**
   - Its own pose from wheel odometry, compared with simulator ground truth to measure drift.
   - Object and shelf state from ground truth at first. Simulated range sensors or an RGB-D camera are optional stretch goals.
4. **Fallback:** a virtual holonomic base (x, y, yaw joints) stays selectable by config. Both base types expose the same `navigate(target)` skill, so the planner and benchmark never change.
5. **SLAM is not in scope** unless added as a stretch goal (for example occupancy mapping from simulated range sensors). Do not claim it in slides until it exists.

---

## Status Tracker

| Phase | Name | Status (verify in repo) |
|---|---|---|
| 1 | Setup and foundations | Mostly done; skill interface stubs missing |
| 2 | Library scene | Not started |
| 3 | Mobile robot (wheeled base + arm) | In progress: models generated, not validated |
| 4 | Arm control (FK, IK, Jacobian) | Mostly done; null-space and some tests to confirm |
| 5 | Gripper, grasp and place | Not started |
| 6 | Autonomous navigation | Not started |
| 7 | Integration and baseline planner | Not started |
| 8 | LLM planner | Not started (can start early) |
| 9 | Verify, failure injection, replanning | Not started |
| 10 | Benchmark, evaluation, report | Not started |

---

## Phase 1: Setup and foundations

**Goal:** working environment and a stable project skeleton.

**Already done (verify):** MuJoCo installed, Panda loaded from Menagerie, viewer demo, package layout `src/library_robot/{control,sim,navigation,planner,robot,skills,benchmark}`, configs, numbered scripts, pytest folder, Git repo.

**Remaining:**
- Define the skill interface as stubs: `navigate(target)`, `grasp(obj)`, `place(obj, shelf)`, `verify()`. Each returns a result object (success flag, failure code, message, duration). The planner depends only on this interface.
- Write a mock executor that prints actions, so the LLM planner can be developed without a working robot.
- Make sure both teammates can run the same setup (README with exact Windows PowerShell commands, pinned dependencies).
- Document the known `mujoco.dll` Application Control issue and its workaround. Keep XML-generation code and tests runnable without the DLL where possible.

**Output:** skill interface module, mock executor, README, pinned requirements.

**Done when:** a fresh clone runs the Panda viewer demo and the skill stubs return results through the mock executor.

---

## Phase 2: Library scene

**Goal:** a simulated library generated from code.

**What we do:**
- Layout file (YAML or JSON) naming everything: Tables A to F, shelf compartments (A1, A2, A3, ...), books (name, table, colour, size), window and wall positions, parking poses.
- `generate_scene.py` reads the layout and generates MJCF: floor, lights, 5 to 6 tables, one bookshelf with compartments, 10 to 15 books.
- Books have freejoints; tables and shelf are fixed. Use simple boxes for collision; Blender meshes can be added later for visuals only.
- Floor friction must suit the wheels (tune together with Phase 3).
- Two shelf modes selectable in the layout: **packed** (small or no gaps, for G3/G2H) and **loosened** (gaps, for G2G).
- Leave headroom above books so a hook finger can reach the top of a spine.
- Randomize book positions per trial using a seed.

**Output:** layout file, `generate_scene.py`, generated scene XML.

**Done when:** the scene loads, and no book falls or jitters during a 10-second simulation in both shelf modes.

---

## Phase 3: Mobile robot (wheeled base + arm)

**Goal:** a robot that physically drives and holds its arm steady.

**Already done (verify):** `configs/robot.yaml` selects `base_type` (wheeled or virtual); `sim/mobile_model.py` generates a wheeled model (freejoint chassis, two velocity-actuated wheels, two ball-joint casters) and a virtual model; structural XML test exists. Wheel direction, stability and slip are NOT yet validated.

**What we do:**
- **Validate the wheeled model:** motor direction test (positive command = forward), straight-line drive, in-place turn, tip test with the arm fully extended and carrying a book-mass payload.
- Fix mass, wheel track, wheel radius, friction, damping, motor `forcerange` and `ctrlrange` from test results (all current values are provisional).
- **Differential-drive controller:** convert (v, omega) to wheel speeds; velocity and acceleration limits.
- **Odometry** from wheel joint velocities, compared against ground truth to measure slip and drift.
- Arm posture rules: tucked while driving, extended only after parking; wheels held at zero velocity while the arm works.
- Mount the Panda in the base frame (single XML), decide mounting position, add the optional vertical lift only if reach to upper shelf rows needs it.
- State-reading helper: robot pose, joint angles, wheel speeds, named object poses (extend existing `sim/state_reader.py`).

**Output:** validated `robot.yaml`, base controller, odometry module, tip/drive test scripts.

**Done when:** the wheeled robot drives a straight line and a turn within tolerance, does not tip with the arm extended, and odometry error versus ground truth is measured and logged.

**Fallback:** if the wheeled base is not stable by the Gate date, run the benchmark on the virtual base and report wheeled results as partial.

---

## Phase 4: Arm control (FK, IK, Jacobian)

**Goal:** move the end-effector to any reachable pose, with the arm mounted on the base. This is the core course content.

**Already done (verify):** forward kinematics checks, Jacobian (MuJoCo and numerical), position IK, pose IK, joint and gripper control helpers, Cartesian linear trajectories, numbered demo scripts 01 to 09.

**Remaining / confirm:**
- Confirm the IK uses damped least squares and report its behavior near singularities.
- Null-space control of the redundant 7th joint to stay away from joint limits.
- IK expressed in the **base frame**, so targets come from world poses transformed by the robot's current pose.
- Straight-line Cartesian approach ("go down 10 cm above the book"), and gripper open/close.
- Accuracy test: 10 random reachable poses, about 1 to 2 cm position error; log success rate, iterations, singularity behavior.

**Output:** IK/Jacobian/arm controller modules and a repeatable accuracy test.

**Done when:** the gripper reaches 10 random reachable poses with about 1 to 2 cm accuracy, with the arm mounted on the base.

---

## Phase 5: Gripper, grasp and place

**Goal:** pick a book from a table and place it in a shelf compartment, with the gripper chosen by a feasibility gate.

**Gripper variants (config flag `gripper`):**

| ID | Variant | Role | Shelf |
|---|---|---|---|
| G3 | 3-finger: 2 holders + 1 extractor (hook) | Primary | Packed |
| G2H | Panda hand + fingertip hook | Backup 1 | Packed |
| G2G | Plain Panda hand | Backup 2 | Loosened |

**What we do:**
1. **G3 model first:** custom MJCF with three actuators, simple colliders, tuned friction. Gate G-A: model loads, opens and closes, no instability.
2. **Extraction primitive:** hook the top edge of the spine, pull, tilt the book 10 to 20 degrees; then reposition the holders, close, verify the grip by finger width.
3. **Feasibility test (gate G-B):** three packed book-sized boxes on the shelf; extract and grip the middle one; 30 trials per variant in **friction** mode, randomized offset. Proposed pass: at least 70% extraction plus grip success, neighbor disturbance below tolerance in at least 90% of trials, no unresolved jitter. If G3 fails, switch to G2H, then G2G, and log the reason.
4. **GRASP(book):** pre-grasp pose, approach, close, lift. Ground-truth book pose at first.
5. **Two grasp modes, reported separately:** weld/attach (stable, for pipeline runs) and friction (real contact, for gripper comparison).
6. **PLACE(book, shelf):** pre-place pose in front of the compartment, insert, release, retract.
7. **Collision-free motion:** RRT-Connect into the shelf; fall back to a hand-designed approach trajectory if RRT is too slow.
8. A success check after each step (is the book in the gripper or not).

**Output:** gripper model(s), `grasp.py`, `place.py`, `rrt.py`, feasibility test script and results.

**Done when:** with the robot parked and not navigating, it moves a book from a table to shelf A3 correctly at least 8 out of 10 times with the active gripper.

---

## Phase 6: Autonomous navigation

**Goal:** `navigate(location)` drives the wheeled robot to any table or shelf and parks accurately, with no human input.

**What we do:**
- Occupancy grid from the layout file (tables, shelf, walls); inflate obstacles by the robot radius.
- A* from start to goal; convert the path to smooth, drivable waypoints.
- **Path follower** (pure pursuit or turn-drive-turn) that respects the differential-drive constraint.
- **Parking controller:** slow down, align heading, stop within tolerance (start: +/- 2 cm, +/- 2 deg), then hold the wheels still.
- Fixed parking pose per table and shelf, chosen so the arm can reach the books (verify reach from each).
- **Pose estimate:** odometry between waypoints; correct drift using a simple reference (for example, known landmark poses from the layout). State which method is used.
- **Re-localize the target after parking:** recompute the book/shelf pose in the base frame before grasping, so parking error does not break IK.
- Obstacle response: if the path is blocked, stop and report a navigation failure so the planner can replan.
- Optional stretch: simulated range sensors and a local obstacle map.

**Output:** `occupancy_grid.py`, `astar.py`, path follower, parking controller, `navigate.py` (works for both base types).

**Done when:** the robot reaches every table and shelf from the start pose without collisions, parks within tolerance in at least 9 out of 10 runs, and the arm can reach from each parking pose.

---

## Phase 7: Skills integration and baseline planner

**Goal:** the full pipeline working end to end without an LLM.

**What we do:**
- Executor that runs an action list step by step through the skill interface (NAVIGATE, GRASP, PLACE, VERIFY).
- Rule-based baseline planner: a template/regex parser for fixed commands such as "Find the X book on Table B and place it in shelf A3". Give it a fair parser; its natural limits (new phrasing, "closest", "empty shelf") are what the project compares against.
- Log every run: step, duration, success or failure, failure code.
- Draft the benchmark task list (`tasks.json`) now, not at the end.

**Output:** `executor.py`, `baseline_planner.py`, logging, draft `tasks.json`.

**Done when:** 5 to 10 fixed commands run completely, from navigation to placement, on the wheeled base.

---

## Phase 8: LLM planner

**Goal:** the same system, with an LLM as the planner. Can start as soon as Phase 1's skill interface and mock executor exist.

**What we do:**
- Skill schema (JSON): each action's name and arguments.
- World state JSON from the layout and simulator: object positions, which shelves are empty, robot pose. Pre-compute spatial facts in code (for example distance to a table) rather than giving the LLM raw coordinates.
- Prompt: instruction, state, skill list, and 2 to 3 examples; low temperature (state the value).
- **Provider-agnostic LLM client:** one interface with swappable backends (a hosted API, a local model, or a mock that returns canned plans), so development and tests never depend on one provider's usage limits.
- Validator that rejects unknown actions, wrong objects, wrong arguments; retry with the error message.
- Plan once before execution, so the LLM is not in the control loop.
- Log latency, tokens or cost, and every generated plan.

**Output:** `llm_planner.py`, `prompts/`, `schema.json`, `validator.py`, LLM client with mock backend.

**Done when:** the same 5 to 10 commands work with the LLM, and some newly phrased commands (which the baseline cannot handle) also produce correct plans.

---

## Phase 9: Verify, failure injection and replanning

**Goal:** the robot detects its own mistakes and recovers autonomously.

**What we do:**
- `VERIFY()`: simulator ground truth checks the book is in the correct compartment.
- **Injected failures:**

| ID | Failure | Expected recovery |
|---|---|---|
| F1 | Target shelf already occupied | Replan to an empty compartment |
| F2 | Book dropped during carry | Re-locate and re-grasp |
| F3 | Path to table blocked | Replan route |
| F4 | Book not at the stated table | Search other tables or report failure |
| F5 | Extraction fails (hook slips) | Retry extraction, then fall back |

- Replanning loop: on failure, send the new world state and the error back to the LLM, with a retry limit (2 to 3).
- Run the same failures through the baseline.
- Optional: RGB-D book detection instead of ground truth; otherwise leave as future work.

**Output:** `verify.py`, `failure_injection.py`, replanning loop.

**Done when:** the LLM planner recovers from at least 2 to 3 different injected failures.

---

## Phase 10: Benchmark, evaluation and report

**Goal:** produce results and write them up.

**What we do:**
- Task set in one JSON file: T1 single-step (10), T2 multi-step (15), T3 rephrased commands (15), T4 spatial relations (15), T5 injected failures (15). Randomized book positions, several trials per task, fixed seeds.
- Planners: baseline, LLM, LLM + world model, LLM + verify/replan. Same navigation and manipulation stack for all.
- Run on the active gripper and the wheeled base (and the virtual base as a comparison if time allows).
- Metrics: task success, plan validity, hallucination rate, navigation success, grasp success (weld and friction separately), placement accuracy, collision rate, replanning success, planning latency, execution time. Base metrics: parking pose error, base-induced end-effector disturbance, wheel slip, tip margin, drive time, peak wheel torque.
- Failure codes: P planning, N navigation, X extraction, G grasp, L placement, V verification, S simulation.
- Statistics: mean, median, SD, IQR, 95% CI; McNemar for paired binary outcomes, Wilcoxon signed-rank for paired continuous metrics. Report per task category.
- Plots (at least 2 to 3), success and failure examples, before/after replanning cases, a clean `run_demo.py` that reproduces a full run live.
- Final report and slide deck update: MuJoCo (not Isaac Sim), wheeled autonomous base, gripper priority and gate results, the weld vs friction caveat, ground-truth perception.

**Output:** `tasks.json`, `run_eval.py`, plots, report, updated deck.

**Done when:** results for both planners appear in one table and at least 2 to 3 plots.

---

## Dependencies and Parallel Work

| Work | Depends on | Can run in parallel with |
|---|---|---|
| Phase 1 skill stubs and mock executor | none | everything |
| Phase 8 LLM planner | Phase 1 only | Phases 2 to 6 |
| Phase 2 scene | Phase 1 | Phases 3, 4 |
| Phase 3 wheeled robot | Phase 1 | Phases 2, 4 |
| Phase 4 arm control | Phase 1 | Phases 2, 3 |
| Phase 5 gripper/grasp/place | Phases 2, 3, 4 | Phase 6 start |
| Phase 6 navigation | Phases 2, 3 | Phase 5 |
| Phase 7 integration | Phases 5, 6 | Phase 8 finish |
| Phase 9 verify/replan | Phases 7, 8 | none |
| Phase 10 benchmark | Phases 7 to 9 | report writing |

**Suggested split:** Person 1 takes Phases 3, 6 (mobile base and navigation). Person 2 takes Phases 4, 5, then 8. Both do Phase 2, then 7, 9, 10 together.

---

## Decision Gates

Dates are placeholders. Replace with real deadlines. Keep about one week of buffer before the final deadline.

| Gate | When | Condition | If missed |
|---|---|---|---|
| G-W: wheeled base works | End of Phase 3 | Drives, turns, no tipping, odometry measured | Use virtual base, report wheeled as partial |
| G-A: G3 model works | Before Phase 5 grasping | Loads, opens/closes, stable | Start G2H immediately |
| G-B: gripper feasibility | End of gripper tests | Section 5 criteria met | Switch to G2H, then G2G |
| G-C: planner benchmark ready | Before Phase 10 | At least 8/10 correct placements on the active gripper | Reduce task count, keep weld mode |

---

## Instructions for the Coding Agent (Antigravity)

Paste this at the start of the session, after cloning the repo:

```
You are my robotics co-developer for "LLM-Guided Autonomous Mobile Librarian Robot" (MuJoCo, Python 3.12, Franka Panda on an autonomous wheeled mobile base). The full plan is in project_phases_mobile_mujoco.md in the repo root. Read it first.

Rules:
1. Inspect the real repo before changing anything. Compare it with the Status Tracker in the plan and tell me which statuses are wrong.
2. Work on ONE phase step at a time. Before editing, state which files change and how I verify it.
3. After each step give: files changed, PowerShell commands to run, expected output, what could fail.
4. Never claim something works unless I have shown test output. Mark every invented number "needs tuning".
5. Keep the skill interface (navigate, grasp, place, verify), the config keys, and the package layout stable. Ask before renaming.
6. Keep parameters in configs/, scripts in scripts/ (numbered), reusable code in src/library_robot/, tests in tests/ (pytest). Keep XML generation and tests runnable without loading the MuJoCo DLL where possible.
7. If a gate is missed, say so and propose the fallback.
8. Keep both base types (wheeled, virtual) and all gripper variants (G3, G2H, G2G) selectable by config.

First task: do a read-only audit and report (max 15 lines): what exists, what is wrong, risks, and the next 5 steps in order. Do not edit any file yet.
```

After the audit, suggested first working steps:

1. Phase 1 leftovers: skill interface stubs and mock executor.
2. Phase 3: wheeled base validation (motor direction, drive, turn, tip test).
3. Phase 3: differential-drive controller and odometry.
4. Phase 2: layout file and scene generator.
5. Phase 4: confirm the null-space IK and base-frame IK, then run the accuracy test.
