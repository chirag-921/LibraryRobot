# Benchmark Plan: 3-Finger Gripper (Primary) with 2-Finger Backup

**Project:** LLM-Guided Autonomous Mobile Librarian Robot
**Simulator:** MuJoCo + Franka Panda (MuJoCo Menagerie)
**Purpose:** Define how the gripper choice is tested, when we switch to the backup, and how the benchmark stays comparable either way.

> All numeric thresholds in this document are proposed starting values, not measured results. Adjust them after the first feasibility trials.

---

## 1. Gripper Strategy

| | **Primary: 3-Finger Extractor Gripper** | **Backup: 2-Finger Parallel Gripper** |
|---|---|---|
| Concept | Two fingers hold the book (antipodal grasp). A third finger hooks the top edge of the spine and tilts the book out of a packed shelf. | Standard Panda hand with a custom fingertip hook (or a loosened shelf if the hook fails). |
| Why this role | Solves the packed-shelf problem directly and is the stronger manipulation contribution. | Validated, 1-DoF, simplest to control and simulate. |
| Build effort | High: custom MJCF model, three actuators, contact tuning. | Low: Menagerie hand plus a small fingertip change. |
| Main risk | Contact jitter, hook tuning, schedule overrun. | Cannot reach into a fully packed shelf without the hook or gaps. |
| Shelf condition | Packed (small or no gaps) | Gaps between books (fallback), or packed with fingertip hook |

**Principle:** the planner, skill interface, navigation, and benchmark tasks stay identical for both grippers. Only the `grasp` and `place` skill internals change, so the LLM-vs-baseline comparison is never affected by the gripper decision.

---

## 2. Gripper Variants Under Test

| ID | Variant | Role | Shelf |
|---|---|---|---|
| G3 | 3-finger: 2 holders + 1 extractor | **Primary** | Packed |
| G2H | 2-finger Panda hand + fingertip hook | **Backup 1** | Packed |
| G2G | 2-finger Panda hand, plain | **Backup 2** | Loosened (gaps) |

Switching order if the primary fails its gate: **G3, then G2H, then G2G.**

---

## 3. Grasp Modes

The proposal uses weld-constraint ("attach") grasps in the early phases because they are stable. That makes grasp success mostly a measure of arm positioning, not of the gripper. To make the gripper comparison meaningful, test two modes:

| Mode | Description | Used for |
|---|---|---|
| **W: Weld/attach** | Book is attached with a weld constraint once the gripper closes around it. | Full-pipeline runs, planner benchmark (Section 6) |
| **F: Friction** | Real contact-based grasp, no constraint. | Gripper feasibility and comparison (Section 4) |

Always report W and F results separately. Do not mix them in one success rate.

---

## 4. Gripper Feasibility Benchmark (Gate Test)

Run **before** committing to the full benchmark. Setup: three packed boxes with book dimensions on the shelf; target the middle one. Randomize target position by a small offset each trial.

### 4.1 Trial plan

| Variant | Mode | Trials | Shelf |
|---|---|---|---|
| G3 | F | 30 | Packed |
| G2H | F | 30 | Packed |
| G2G | F | 30 | Loosened |

### 4.2 Metrics

| Metric | Definition |
|---|---|
| Extraction success | Book tilted or pulled out at least a set angle or distance without moving neighbors beyond a small tolerance |
| Grip success | After closing, finger width matches book thickness (within tolerance) |
| Lift success | Book still held after lifting a fixed height and holding for a few seconds |
| Neighbor disturbance | Displacement of adjacent books (cm) |
| Book drop / ejection rate | Book slips or is ejected during close, lift, or carry |
| Cycle time | Seconds from approach start to lift complete |
| Jitter incidents | Count of unstable contact events or simulation pops |
| Tuning effort | Parameter changes needed (logged, qualitative) |

### 4.3 Gate criteria (proposed)

| Gate | Pass condition for G3 |
|---|---|
| Extraction + grip success (end-to-end, F mode) | At least 70% over 30 trials |
| Neighbor disturbance | Below tolerance in at least 90% of trials |
| Stability | No unresolved jitter after tuning |
| Schedule | Gripper model working by the gate date (Section 8) |

**Decision rule:**

- **G3 passes:** continue with G3 as primary. Keep G2H results as the comparison baseline.
- **G3 fails on performance or schedule:** switch to G2H. If G2H also fails its equivalent criteria, switch to G2G and loosen the shelf.
- Record the reason for any switch. A documented fallback decision is a legitimate result, not a failure.

---

## 5. Gripper Comparison Experiment (Optional Extension)

If time allows after the gate, run a controlled comparison on the same task set:

| Condition | Description |
|---|---|
| G3-F | 3-finger, friction grasp, packed shelf |
| G2H-F | 2-finger + hook, friction grasp, packed shelf |
| G2G-F | 2-finger plain, friction grasp, loosened shelf |

Report extraction, grip and lift success, neighbor disturbance, cycle time, and drop rate. Use McNemar's test for paired binary outcomes (same trial seeds across variants) and Wilcoxon signed-rank for paired continuous metrics (cycle time, disturbance).

---

## 6. Planner Benchmark (Gripper-Independent)

The main LLM-vs-baseline comparison uses the **active gripper** (G3 if it passed, otherwise the fallback) with the same navigation and manipulation stack for both planners.

### 6.1 Task categories

| Category | Example | Tasks (target) |
|---|---|---|
| T1: Single-step | "Go to Table B." | 10 |
| T2: Multi-step | "Find the Machine Learning book on Table B and place it in shelf A3." | 15 |
| T3: Rephrased commands | "Could you shelve the ML book from table B into A3?" | 15 |
| T4: Spatial relations | "Take the book closest to Table D and place it in the empty shelf nearest the window." | 15 |
| T5: Injected failures | Shelf occupied, book dropped, path blocked, book not at expected table | 15 |
| **Total** | | **70** |

Each task is run multiple times with randomized book positions and, for the LLM, a fixed low temperature (state the value used). Suggested starting point: 5 randomized trials per task. Adjust to fit the time available.

### 6.2 Planners compared

| Planner | Description |
|---|---|
| Baseline | Template/regex parser, fixed rules |
| LLM | Pretrained LLM, JSON action output, validator, retry |
| LLM + world model | Adds structured state (pre-computed spatial facts) |
| LLM + verify/replan | Adds VERIFY() and replanning with retry limit |

### 6.3 Metrics

| Metric | Definition |
|---|---|
| Task success rate | Instruction completed correctly end to end |
| Plan validity rate | Plans with only known actions, objects, arguments |
| Hallucination rate | Plans referencing nonexistent objects or illegal actions |
| Navigation success | Reached the parking pose without collision |
| Grasp success | Book acquired (report W and F separately) |
| Placement accuracy | Book in the correct compartment |
| Collision rate | Collisions during navigation or manipulation |
| Replanning success | Recovered after an injected failure within the retry limit |
| Planning latency | Seconds to produce a plan |
| Total execution time | End-to-end seconds |

---

## 7. Failure Taxonomy and Injected Failures

Every failed run is labeled with exactly one primary cause so planner errors can be separated from robot errors.

| Code | Category | Examples |
|---|---|---|
| P | Planning | Wrong action order, invalid action, hallucinated object |
| N | Navigation | Collision, did not reach the parking pose |
| X | Extraction | Hook missed the spine, book not tilted, neighbors disturbed (G3 and G2H only) |
| G | Grasp | Grip not confirmed, book slipped or ejected |
| L | Placement | Wrong compartment, insertion failed |
| V | Verification | Wrong verdict from VERIFY() |
| S | Simulation | Jitter, physics glitch |

**Separating X from G** is important: it shows whether failures come from pulling the book out or from holding it.

### Injected failure list

| ID | Failure | Expected recovery |
|---|---|---|
| F1 | Target shelf already occupied | Replan to an empty compartment |
| F2 | Book dropped during carry | Re-locate and re-grasp |
| F3 | Path to table blocked | Replan with a different route |
| F4 | Book not at the stated table | Search other tables or report failure |
| F5 | Extraction fails (hook slips) | Retry extraction, then fall back |

---

## 8. Milestones and Decision Gates

Dates are placeholders. Replace with real deadlines.

| Gate | Target | Condition | If missed |
|---|---|---|---|
| **G-A: G3 model works** | Before Phase 5 (grasp and place) | 3-finger model loads in MuJoCo, opens and closes, no instability | Start G2H immediately |
| **G-B: Feasibility test** | End of Phase 5 | Section 4 gate criteria met | Switch to G2H, loosen shelf if needed |
| **G-C: Planner benchmark ready** | Before Phase 10 | Active gripper reaches at least 8/10 correct placements (the Phase 5 "Done when") | Reduce task count, keep W mode |
| **Buffer** | About one week before the deadline | Reserved for overruns | Do not schedule new work here |

**Parallel hedge:** the person building the LLM planner (Phase 8) can work against a mock executor from the start, so the gripper decision never blocks the planner work.

---

## 9. Logging Schema

One row per trial, saved as CSV or JSON.

| Field | Description |
|---|---|
| trial_id | Unique ID |
| task_id, category | From the task set (T1 to T5) |
| planner | Baseline, LLM, LLM + world model, LLM + verify/replan |
| gripper | G3, G2H, or G2G |
| grasp_mode | W or F |
| seed | Random seed (book positions, LLM sampling) |
| plan_text | Generated action list |
| plan_valid | 1 or 0 |
| latency_s | Planning time |
| exec_time_s | Total execution time |
| extraction_ok, grip_ok, placed_ok | 1 or 0 |
| collisions | Count |
| failure_code | From Section 7, empty if success |
| injected_failure | F1 to F5 or none |
| replans | Number of replanning attempts |
| notes | Free text |

---

## 10. Statistical Analysis

- Report mean, median, standard deviation, IQR, and 95% confidence intervals.
- **Binary outcomes** (success, validity): McNemar's test on paired trials.
- **Paired continuous metrics** (latency, execution time, cycle time): Wilcoxon signed-rank test.
- Report per task category, not only overall, so it is clear where the LLM helps and where the baseline wins.
- State the number of trials, seeds, and the LLM temperature.

---

## 11. Expected Outputs

**Tables**

- Gripper feasibility results (G3 vs G2H vs G2G, F mode)
- Planner comparison by task category
- Failure breakdown by code

**Plots**

- Gripper success rates with confidence intervals
- Task success rate: baseline vs LLM variants
- Success versus task complexity
- Planning latency and execution time
- Failure breakdown by category

**Qualitative**

- Screenshots or video of the extraction and grasp sequence for each gripper variant
- Successful and failed extraction examples
- Before and after replanning scenarios

---

## 12. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| 3-finger model takes too long | Schedule overrun | Gate G-A, start G2H in parallel if the model is not working by the gate date |
| Contact jitter on thin books | Unreliable results | Box colliders, tuned friction, tuned solver settings, limited closing speed |
| Hook cannot reach the spine | Extraction fails | Add headroom above books in the shelf model, adjust hook geometry |
| Weld grasp hides gripper weakness | Misleading grasp success | Report W and F separately, run the Section 4 test in F mode |
| Results differ between grippers | Hard to compare planners | Run the planner benchmark on one fixed gripper |
| LLM nondeterminism | Noisy results | Low temperature, repeated trials, report the settings |

---

## 13. One-Line Summary for Slides

> Primary end-effector: 3-finger extractor gripper (two holders plus one hook finger) for packed shelves. Backup: Panda 2-finger hand with fingertip hook, then plain 2-finger with a loosened shelf. A feasibility gate decides which one the planner benchmark uses, so the LLM-vs-baseline comparison is never blocked by the gripper choice.
