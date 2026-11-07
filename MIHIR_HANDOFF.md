# Handoff for Mihir: Planner and Benchmark Side

**Project:** LLM-Guided Autonomous Mobile Librarian Robot (MuJoCo, Python 3.12, Franka Panda on an autonomous wheeled base)
**Repo:** https://github.com/chirag-921/LibraryRobot
**Your role:** build the planners (LLM and rule-based baseline) and the benchmark task list. Chirag builds the robot, scene and navigation side.

> Anything marked "proposed" or "needs tuning" is a starting value or a draft, not a decision. Change it if it doesn't work, but tell Chirag, especially if it touches the shared contracts in Section 5.

---

## 1. Read These First

| File | What it gives you |
|---|---|
| `project_phases_mobile_mujoco.md` | The full 10-phase plan, status tracker, gates, and agent instructions |
| `benchmark_gripper_plan.md` | Task categories, metrics, failure codes, injected failures, statistics plan, logging schema |
| `README.md` | Setup steps |
| `src/library_robot/skills/` | The skill interface you plan against (after Chirag pushes it) |

If `benchmark_gripper_plan.md` is not in the repo yet, ask Chirag to commit it.

---

## 2. Setup (Windows, PowerShell)

```powershell
git clone https://github.com/chirag-921/LibraryRobot.git
cd LibraryRobot
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pytest tests
```

You do **not** need the MuJoCo viewer or the Menagerie model for your work. Your code must run offline with only Python and pytest. If you later want to run the robot scripts, the README shows how to download the Panda model.

---

## 3. Your Scope

You own these folders:

- `src/library_robot/planner/`
- `src/library_robot/benchmark/`
- `benchmark_tasks/`
- your own tests under `tests/`

You do **not** edit these (ask Chirag first):

- `src/library_robot/skills/` (the shared interface)
- `src/library_robot/sim/`, `src/library_robot/control/`
- `configs/robot.yaml`
- generated models in `models/`

---

## 4. What You Build

### Task 1: Skill schema and validator

- `schema.json` describing the four skills: `navigate(target)`, `grasp(obj)`, `place(obj, shelf)`, `verify()`.
- A validator that rejects unknown actions, unknown objects/tables/shelves, and wrong or missing arguments, and returns a clear error message.
- Tests for valid and invalid plans.

**Done when:** valid plans pass, and each kind of invalid plan produces a specific error message in tests.

### Task 2: World-state builder

- Builds a JSON world state from a layout: tables, shelf compartments, books (name, location, colour), which compartments are empty, robot location.
- Pre-compute spatial facts in code (nearest table to X, empty compartments, distance relations) instead of giving the LLM raw coordinates.
- Start with a hand-written fake layout (draft format in Section 5.2). Later it connects to Chirag's real layout file.

**Done when:** given the fake layout, the builder outputs valid JSON with the spatial facts, covered by tests.

### Task 3: LLM client and planner

- A provider-agnostic client interface with swappable backends:
  - **Mock backend** that returns canned plans (used by all tests).
  - **One real backend** you choose (a free-tier API or a local model). Pick something you can actually use; Chirag's own LLM usage limit is exhausted, so do not assume he can run it.
- API keys come from environment variables only. Never commit keys. Add `.env` to `.gitignore`.
- Planner prompt: instruction, world state, skill list, 2 to 3 examples. Low temperature (state the value you use).
- The LLM plans **once before execution**, not inside the control loop. Output is a JSON action list matching the schema.
- On an invalid plan, retry with the validator's error message (limit 2 to 3).
- Log every plan: instruction, world state, raw output, validity, latency, retries.

**Done when:** the mock backend drives the full planner and tests pass offline; the real backend produces valid plans for the same 5 to 10 commands.

### Task 4: Rule-based baseline planner

- Template/regex parser for fixed commands such as "Find the X book on Table B and place it in shelf A3".
- It must be a **fair** parser. Its natural limits (new phrasing, "closest", "empty shelf") are what the project compares against, so do not cripple it on purpose and do not make it handle everything.
- Returns the same JSON action format as the LLM planner.

**Done when:** it plans correctly for the fixed-template commands and fails cleanly (clear error, no crash) for the others.

### Task 5: Benchmark task list

- `benchmark_tasks/tasks.json` with about 70 tasks (proposed distribution):

| Category | Example | Count |
|---|---|---|
| T1 Single-step | "Go to Table B." | 10 |
| T2 Multi-step | "Find the Machine Learning book on Table B and place it in shelf A3." | 15 |
| T3 Rephrased | "Could you shelve the ML book from table B into A3?" | 15 |
| T4 Spatial relations | "Take the book closest to Table D and place it in the empty shelf nearest the window." | 15 |
| T5 Injected failures | Shelf occupied, book dropped, path blocked, book not at stated table | 15 |

- Each task has: id, category, instruction text, initial world state (or layout reference and seed), expected action sequence, and for T5 the injected failure ID (F1 to F5).
- Expected sequences are what a correct plan looks like. Where more than one plan is valid, list the acceptable alternatives or define a checking rule.

**Done when:** the file validates against a schema, and both planners can be run on it.

### Task 6: Dry-run evaluation with MockExecutor

- Run both planners on all tasks through `MockExecutor` and report plan validity and plan correctness per category.
- Output a CSV in the logging format (Section 6).

**Done when:** one command produces the per-category table for both planners.

### Later (after the above, if time allows)

- Phase 4 leftovers in `control/` (base-frame IK, null-space, the 10-pose accuracy test). Coordinate with Chirag first, since this touches his folders.

---

## 5. Shared Contracts (Do Not Change Alone)

### 5.1 Skill interface (from `skills/`)

Planners output action lists that `MockExecutor` (and later the real executor) runs. Read the actual files for the exact signatures. The intended shape is:

```json
{
  "plan": [
    {"action": "navigate", "args": {"target": "Table B"}},
    {"action": "grasp",    "args": {"obj": "Machine Learning"}},
    {"action": "navigate", "args": {"target": "Shelf A3"}},
    {"action": "place",    "args": {"obj": "Machine Learning", "shelf": "A3"}},
    {"action": "verify",   "args": {}}
  ]
}
```

Each skill returns a result with: success flag, failure code, message, duration. Failure codes: `P` planning, `N` navigation, `X` extraction, `G` grasp, `L` placement, `V` verification, `S` simulation.

If the real interface differs from this sketch, the repo wins. Tell Chirag about any mismatch.

### 5.2 Draft layout format (proposed)

Agree on this with Chirag early. The world-state builder and the scene generator both depend on it.

```yaml
tables: [A, B, C, D, E, F]
shelf:
  name: Shelf A
  compartments: [A1, A2, A3, A4, A5, A6]
books:
  - {name: "Machine Learning", table: B, colour: red}
  - {name: "Linear Algebra",   table: C, colour: blue}
landmarks:
  window: {near_table: D}
robot_start: {location: start}
```

### 5.3 Draft world-state JSON (proposed)

```json
{
  "robot": {"location": "start", "holding": null},
  "tables": {"A": [], "B": ["Machine Learning"], "C": ["Linear Algebra"]},
  "shelf": {"A1": null, "A2": "Robotics", "A3": null},
  "facts": {
    "empty_compartments": ["A1", "A3"],
    "nearest_table_to_window": "D"
  }
}
```

---

## 6. Logging Format (One Row Per Trial)

| Field | Description |
|---|---|
| trial_id | Unique ID |
| task_id, category | From the task set (T1 to T5) |
| planner | baseline, llm, llm_world_model, llm_verify_replan |
| seed | Random seed |
| plan_text | Generated action list |
| plan_valid | 1 or 0 |
| plan_correct | 1 or 0 against the expected sequence |
| hallucination | 1 if it names a nonexistent object or illegal action |
| latency_s | Planning time |
| retries | Validator retries used |
| failure_code | P, N, X, G, L, V, S, or empty |
| notes | Free text |

Robot-side fields (base type, gripper, parking error, collisions) are added later by the full benchmark runner.

---

## 7. Git Workflow

- Work on your own branch: `mihir/llm-planner`, `mihir/baseline`, `mihir/tasks`.
- Open a pull request into `main`; Chirag reviews. Chirag works on branches like `chirag/wheeled-base`.
- `git pull` before you start each session and push at least daily.
- Commit small and often, with clear messages.
- Never commit `.venv`, API keys, `.env`, or large model files.
- If `git status` shows thousands of files or anything strange, stop and ask.

```powershell
git checkout -b mihir/llm-planner
git add src tests
git commit -m "Add skill schema and validator"
git push -u origin mihir/llm-planner
```

---

## 8. Checkpoints

| When | What to show Chirag |
|---|---|
| After Task 1 | The schema, together: it is the contract between your planner and the executor |
| After Task 2 | The world-state format, to match it with the layout file |
| After Task 3 | Plan logs from the mock backend and the real backend on 5 to 10 commands |
| After Task 5 | The task list, to review that the expected sequences are right |
| After Task 6 | The per-category table for both planners |

Dates are placeholders: _fill in your real deadlines here_.

---

## 9. Coding Agent Prompt (Paste at Session Start)

```
You are helping build the planner side of "LLM-Guided Autonomous Mobile Librarian Robot" (MuJoCo, Python 3.12). Read project_phases_mobile_mujoco.md, benchmark_gripper_plan.md and MIHIR_HANDOFF.md in the repo root first.

My scope: Phase 8 (LLM planner), the baseline planner from Phase 7, and the benchmark task list. I work only in src/library_robot/planner/, src/library_robot/benchmark/, benchmark_tasks/ and tests/. Do not edit sim/, control/, configs/robot.yaml or the skills/ interface; if the interface needs a change, tell me and I will ask Chirag.

Rules:
1. First do a read-only audit: read skills/ (SkillResult, skill stubs, MockExecutor) and report what the planner must produce to drive it. Do not edit files yet.
2. One step at a time. Before editing, say which files change and how I verify it.
3. Never claim something works without pytest output. Mark invented values "needs tuning".
4. Planner code must not import MuJoCo. Everything must run offline.
5. The LLM client must be provider-agnostic: a mock backend (canned plans, for tests) plus one real backend I choose later. No API keys in the repo; read them from environment variables.
6. The LLM plans once before execution, not in the control loop. Output is a JSON action list matching the skill schema. A validator rejects unknown actions, unknown objects/shelves/tables, and wrong arguments, and retries with the error message (limit 2-3).
7. Log every plan: instruction, world state, raw output, validity, latency, retries.

Build order (see MIHIR_HANDOFF.md Section 4):
a. schema.json and validator with tests.
b. World-state builder from a fake layout, with pre-computed spatial facts.
c. LLM client interface, mock backend, and planner with prompt.
d. Rule-based baseline planner.
e. benchmark_tasks/tasks.json (about 70 tasks, T1 to T5).
f. Run both planners on the tasks through MockExecutor and report plan validity per category.

Start with the audit and the proposed plan for step a. Do not edit anything yet.
```

---

## 10. Quick Reference

- **Planner never touches joints.** It outputs skill calls only.
- **Plan once, then execute.** The LLM is not in the control loop.
- **Baseline must be fair.** Report where it wins as well as where it loses.
- **Ground truth is used for perception** in this project; the comparison isolates planning.
- **If something is ambiguous, ask Chirag** before changing a shared contract.
