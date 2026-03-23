# Reactive Sketch DSL vs Surface DSL: Design, Differences, and Experiment Plan

**Date:** 2026-03-21
**Files:** `reactive_sketch_dsl.py`, `reactive_sketch_interpreter.py`, `surface_dsl.py`, `surface_compiler.py`

---

## 1. What Is the Reactive Sketch DSL?

The reactive sketch is a **behavior-tree (BT) program representation** for the Doors environment.  A single fixed skeleton encodes a priority-ordered strategy:

```
While(Not(GoalReached),
  Fallback(
    B1: Sequence(Check(Pickable(key)),       Do(Pick(key)))
    B2: Sequence(Check(KnownLoc(key)),       Do(GoTo(LocOf(key))))
    B3: Sequence(Check(Reachable(GoalLoc)),  Do(GoTo(GoalLoc)))
    B4: Sequence(Check(ExistsFrontier),      Do(GoTo(Entrance(frontier))))
  )
)
```

where `key = KeyFor(NextLockedRoom)`.

**Synthesis scope:** only the **order** of branches B1–B4 within the Fallback.  That gives 4! = **24 candidate programs** regardless of D.

---

## 2. How Does This Differ from the Surface DSL?

The surface DSL is a **decision-list** representation.  A policy is an ordered sequence of guarded rules compiled into nested if-then-else AST nodes.

### 2.1 Side-by-Side Comparison

| Dimension | Surface DSL | Reactive Sketch |
|---|---|---|
| **Execution model** | Decision list (first-match if-then-else chain) | BT tick loop: While → Fallback → Sequence(Check, Do) |
| **Program structure** | Ordered rule list: `PickRule(0) >> MoveRule(0) >> GoalRule` | Fixed skeleton `While(Not(GoalReached), Fallback(B1..B4))` |
| **Predicates** | Grounded by key index: `PickReady(0)`, `NeedKey(0)` | Lifted with selectors: `Pickable(KeyFor(NextLockedRoom))` |
| **Compilation** | Compiles to raw AST (`Ite`/`Default`/`Flip`), evaluated by `interpreter.py` | Has its own tick interpreter (`reactive_sketch_interpreter.py`) |
| **State resolution** | Predicates are compiled at build time into AST conditions that read obs indices directly | Predicates are evaluated at tick time: `ReactiveContext` inspects current obs via relational runtime |
| **D-independence** | Each policy is D-specific (mentions key indices `0..K-1`) | Same `WhileNot` object works for any D — selectors resolve at tick time |
| **Default/fallback** | Explicit `GoalRule` at the end of the rule list | `GoToGoal` is Branch 3; if all branches fail, NOOP |

### 2.2 Search Space Size

| D | K (keys) | Surface (relaxed) | Reactive Sketch |
|---|---|---|---|
| 1 | 0 | 1 | 24 |
| 2 | 1 | 1 | 24 |
| 3 | 2 | 6 | 24 |
| 4 | 3 | 90 | 24 |
| 5 | 4 | 2,520 | 24 |
| 6 | 5 | 113,400 | 24 |

Surface grows as `(2K)!/2^K` (super-exponential in K).  Reactive sketch is always 24.

At small D, the surface space is tiny.  At D≥5, it becomes large enough that exhaustive evaluation is expensive.  The reactive sketch's constant-size space means exhaustive search is always trivial.

### 2.3 What Does "Lifted" Mean Here?

**Surface DSL programs mention concrete key indices:**

```
PickRule(0) >> MoveRule(0) >> PickRule(1) >> MoveRule(1) >> GoalRule
```

This policy is specific to D=3 (K=2).  It can only be compiled against a `DoorsGameConfig` with exactly 2 keys.

**Reactive sketch programs mention no indices at all:**

```
While(Not(GoalReached),
  Fallback(
    Sequence(Check(Pickable(KeyFor(NextLockedRoom))), Do(Pick(KeyFor(NextLockedRoom))))
    ...
  )
)
```

`NextLockedRoom` is resolved at tick time by inspecting the current observation: "scan lockable rooms in order, return the first one that is still locked."  The same policy object solves D=2, D=3, D=4, etc.

### 2.4 When Does Each Approach Fire an Action?

**Surface (decision list):**  At each step, the interpreter walks the Ite chain from the root.  The first condition that evaluates to `true` fires its action.  No further conditions are checked.

**Reactive (BT tick):**  At each step, the tick traverses the tree:
1. `WhileNot` checks `GoalReached` → if false, ticks the `Fallback`
2. `Fallback` ticks branches left-to-right
3. Each branch is `Sequence(Check, Do)`:
   - `Check` evaluates a predicate → SUCCESS or FAILURE
   - If SUCCESS, `Do` resolves the action → SUCCESS with an action index
   - `Sequence` returns SUCCESS only if both children succeed
4. `Fallback` returns the first branch that succeeds

The net effect is similar — priority-ordered condition→action matching — but the mechanism is BT status propagation rather than if-then-else chaining.

---

## 3. Worked Example: D=2 Solved by Both DSLs

**Layout:** `loc_room=[0,0,1,1]`, `key_loc=[1]`, `key_unlocks=[1]`, `start=0`, `goal=3`
**State vector:** `[at_loc0, at_loc1, at_loc2, at_loc3, room0_unlocked, room1_unlocked, key0_avail]`
**Initial:** `[1, 0, 0, 0, 1, 0, 1]` — agent at loc 0, room 0 open, room 1 locked, key 0 available.
**Optimal:** 3 steps = `2*(D-1) + 1`.

### 3.1 Surface DSL Trace

Policy: `PickRule(0) >> MoveRule(0) >> GoalRule`

Compiled AST:
```
Ite(And(Not(IsZero(1)), Not(IsZero(6))),   -- PickReady(0): at key loc AND key avail
    Flip(4),                                 -- Pick(0)
    Ite(IsZero(5),                           -- NeedKey(0): room 1 locked
        Flip(1),                             -- MoveToKey(0): go to loc 1
        Default(Flip(3))))                   -- GoalRule: go to loc 3
```

| Step | State | Rule 1 (PickReady) | Rule 2 (NeedKey) | Default | Action |
|------|-------|-------------------|-----------------|---------|--------|
| 1 | `[1,0,0,0,1,0,1]` | FAIL (at loc 0, not at key loc 1) | TRUE (room 1 locked) | — | `Flip(1)` = move to loc 1 |
| 2 | `[0,1,0,0,1,0,1]` | TRUE (at loc 1 AND key avail) | — | — | `Flip(4)` = pick key 0 |
| 3 | `[0,1,0,0,1,1,0]` | FAIL (key 0 gone) | FAIL (room 1 unlocked) | MoveToGoal | `Flip(3)` = move to loc 3 |

Result: **SOLVED in 3 steps.**

### 3.2 Reactive Sketch Trace

Policy: `While(Not(GoalReached), Fallback(B1, B2, B3, B4))` with order (1,2,3,4).

| Step | State | B1 Pickable? | B2 KnownLoc? | B3 Reachable? | B4 Frontier? | Action |
|------|-------|-------------|-------------|--------------|-------------|--------|
| 1 | `[1,0,0,0,1,0,1]` | FAIL (at loc 0 ≠ key loc 1) | SUCCESS (loc known) → GoTo(1) | — | — | move to loc 1 |
| 2 | `[0,1,0,0,1,0,1]` | SUCCESS (at loc 1, key avail) → Pick | — | — | — | pick key 0 |
| 3 | `[0,1,0,0,1,1,0]` | FAIL (NextLocked=None) | FAIL (NextLocked=None) | SUCCESS (goal room unlocked) → GoTo(3) | — | move to loc 3 |

Result: **SOLVED in 3 steps.**  Same actions as surface, different internal reasoning.

### 3.3 Key Difference in Step 1

In step 1, the **surface DSL** fires `MoveRule(0)` (NeedKey guard: "room 1 is locked → go get key 0").

The **reactive sketch** fires **B2** (KnownLoc guard: "I know where the key is → go there").

Both produce the same action (move to loc 1), but the *reason* is framed differently:
- Surface: "room 1 needs a key, so go get it"
- Reactive: "I know where the key is, so go pick it up"

This distinction becomes important for `known_map=False` scenarios (Section 6.5).

---

## 4. Key Finding: Branch Order Constraints

Running all 24 permutations on D=2 and D=3 reveals:

### 4.1 Results Table (D=2)

| Order | Solved | Steps |
|-------|--------|-------|
| (1, 2, 3, 4) | yes | 3 |
| (1, 2, 4, 3) | yes | 3 |
| (1, 3, 2, 4) | yes | 3 |
| (3, 1, 2, 4) | yes | 3 |
| All other 20 | no | — |

### 4.2 Why Only 4 of 24 Solve

**Critical constraint: B1 must appear before B2.**

In `known_map=True` mode, `KnownLocP` is trivially true whenever there is a locked room (key locations are always known from the config).  If B2 appears before B1, then B2 always fires — the agent keeps moving to the key location but never picks it up, because `Sequence(Check(KnownLoc), Do(GoTo))` succeeds before `Check(Pickable)` gets a chance to run.

The 4 solving orderings all have **B1 before B2**.  Among the remaining branches:
- B3 (GoToGoal) can precede B1 safely — it only fires when the goal is reachable (all rooms unlocked), which means there are no more keys to pick.
- B4 (Explore) can appear anywhere — in `known_map=True`, it is redundant with B2.

### 4.3 All 4 Solving Orderings Achieve Optimal Steps

Every solving ordering produces `2*(D-1) + 1` steps, matching the surface canonical policy.  There are no sub-optimal-but-still-solving orderings.  This holds for D=2, D=3, and D=4.

---

## 5. How to Run Experiments

### 5.1 Run Tests (29 tests)

```bash
python -m pytest tests/test_reactive_sketch.py -v
```

Covers: type construction, predicate evaluation, end-to-end solving for D=2/3/4, all 24 permutations, branch order sensitivity, determinism.

### 5.2 Run the Full Demo Script

```bash
python scripts/run_reactive_sketch_examples.py
```

Output includes:
1. The canonical reactive policy (pretty-printed BT)
2. Step-by-step D=2 and D=3 traces
3. All 24 permutations for D=2 and D=3 with solve/fail results
4. Summary of optimal orderings

### 5.3 Quick One-Liners

**See the reactive policy (no indices):**
```bash
python -c "
from alphazeropp.instances.doors.dsl.reactive_sketch_dsl import canonical_reactive_policy
print(canonical_reactive_policy().pretty())
"
```

**Run D=2 trace:**
```bash
python -c "
from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig, doors_initial_state
from alphazeropp.instances.doors.dsl.relational_runtime import DoorsRelationalRuntime
from alphazeropp.instances.doors.dsl.reactive_sketch_dsl import canonical_reactive_policy
from alphazeropp.instances.doors.dsl.reactive_sketch_interpreter import run_reactive_episode, format_reactive_trace

cfg = DoorsGameConfig(num_rooms=2, locs_per_room=2)
rt = DoorsRelationalRuntime(cfg)
policy = canonical_reactive_policy()
x0 = doors_initial_state(cfg)
env = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
result = run_reactive_episode(env, policy, rt, x0=x0, is_solved=cfg.is_solved)
print(format_reactive_trace(result, policy))
"
```

**Compare surface vs reactive for D=3:**
```bash
python -c "
from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig, doors_initial_state, compute_doors_derived_params
from alphazeropp.instances.doors.dsl.relational_runtime import DoorsRelationalRuntime
from alphazeropp.instances.doors.dsl.surface_grammar import canonical_policy
from alphazeropp.instances.doors.dsl.surface_compiler import compile_policy
from alphazeropp.instances.doors.dsl.reactive_sketch_dsl import canonical_reactive_policy
from alphazeropp.instances.doors.dsl.reactive_sketch_interpreter import run_reactive_episode
from alphazeropp.synthesis.interpreter import run_policy_episode

params = compute_doors_derived_params(3, 2)
cfg = DoorsGameConfig(num_rooms=3, locs_per_room=2, horizon=params['horizon'])
rt = DoorsRelationalRuntime(cfg)
x0 = doors_initial_state(cfg)

# Surface
surface_prog = compile_policy(canonical_policy(3), cfg)
env1 = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
r1 = run_policy_episode(env1, surface_prog, x0=x0, is_solved=cfg.is_solved)

# Reactive
reactive_pol = canonical_reactive_policy()
env2 = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
r2 = run_reactive_episode(env2, reactive_pol, rt, x0=x0, is_solved=cfg.is_solved)

print(f'Surface:  solved={r1.solved}, steps={r1.total_env_steps}, reward={r1.cumulative_reward:.4f}')
print(f'Reactive: solved={r2.solved}, steps={r2.total_env_steps}, reward={r2.cumulative_reward:.4f}')
"
```

### 5.4 Run the Surface DSL Enumeration

```bash
python scripts/enumerate_surface_dsl.py
```

Enumerates all relaxed surface policies for D=2..4, showing which solve and their step counts.

### 5.5 Run All Tests (No Regressions)

```bash
python -m pytest tests/ -v
```

---

## 6. Comprehensive Comparison Plan

### 6.1 Experiment 1: Exhaustive Solve Rate

**Goal:** For each D, what fraction of each DSL's search space actually solves the environment?

**Method:**
```
For D in [2, 3, 4, 5]:
    # Reactive: enumerate all 24 orderings
    For each permutation of (1,2,3,4):
        Build policy → run_reactive_episode → record (solved, steps, reward)

    # Surface: enumerate all relaxed policies
    For each policy in enumerate_relaxed_policies(D):
        Compile → run_policy_episode → record (solved, steps, reward)

    Report: solve_rate, optimal_count, step_distribution
```

**Expected outcome:** Both achieve the same optimal step count when they solve.  The fraction that solves differs — surface at D=2 has 1/1 (100%), reactive has 4/24 (17%).

**What to look for:**
- Do all solving reactive orderings produce optimal steps, or are some sub-optimal?
- How does the surface solve rate change as D grows?
- Are there surface policies that solve sub-optimally (more than `2(D-1)+1` steps)?

### 6.2 Experiment 2: Reward Comparison

**Goal:** Compare cumulative reward (including step penalties and unlock bonuses).

**Method:**
- Use `LeafEvaluator` from `synthesis/leaf_evaluator.py` with `metric="avg_reward"` or `metric="weighted"`
- Run on multiple frozen initial states (not just the default `doors_initial_state`)
- Compare:
  - Best reactive ordering vs canonical surface policy
  - Distribution across all solving orderings/policies

**What to look for:** Since both achieve the same step count, rewards should be identical on the same initial state.  If they differ, investigate which branch/rule fires differently.

### 6.3 Experiment 3: Search Space Scalability

**Goal:** Demonstrate the constant-size advantage of the reactive sketch.

**Method:** Compute and tabulate:

| D | Surface relaxed | Surface prefixes | Reactive | Stage (atoms, ≤2 stages) |
|---|---|---|---|---|
| 2 | `count_relaxed_policies(2)` | `len(enumerate_surface_prefixes(2))` | 24 | `count_stage_programs(cfg, cost_model)` |
| 3 | ... | ... | 24 | ... |
| ... | ... | ... | 24 | ... |

**What to look for:** The crossover point where surface space exceeds 24.  At D=3 (surface=6), reactive is larger.  At D=5 (surface=2520), reactive is much smaller.

### 6.4 Experiment 4: Non-Canonical Layouts

**Goal:** Test robustness when the game layout differs from the default.

**Method:**
```python
# Non-default layout: keys in different rooms, different goal
cfg_custom = DoorsGameConfig(
    num_rooms=3, locs_per_room=2,
    loc_room=[0, 0, 1, 1, 2, 2],
    key_loc=[3, 1],        # key 0 in room 1, key 1 in room 0 (swapped!)
    key_unlocks=[1, 2],
    start_loc=0, goal_loc=5,
    horizon=25,
)
```

- Run both the canonical surface policy and the canonical reactive ordering
- Does the surface policy still solve? (It should — it compiles against the config)
- Does the reactive sketch still solve? (It should — selectors resolve at tick time)

**What to look for:** Both should handle non-default layouts correctly since both ultimately resolve actions through the config.  The difference is readability: the reactive sketch's policy text doesn't change, while the surface policy's compiled AST will have different indices.

### 6.5 Experiment 5 (Future): Partial Observability (known_map=False)

**Goal:** Explore the scenario where key locations are unknown until the agent visits them.

**Current state:**
- The relational runtime supports `known_map=False` — `loc_of_key(k)` returns `None`
- The surface compiler's `PickReady(k)` returns `None` in partial mode (can't compile)
- The reactive sketch's `KnownLocP` would evaluate to `False`, making B2 non-trivially useful

**What would need to change:**
1. The reactive interpreter's `PickableP` evaluation needs a fallback when `loc_of_key` returns None (check all locations? use obs discovery?)
2. B4 (ExploresFrontier) becomes the primary exploration mechanism
3. The surface DSL would need a fundamentally different approach — it has no concept of "unknown location"

**Why this matters:** The reactive sketch's lifted predicates (`KnownLoc`, `Reachable`, `ExistsFrontier`) were designed with partial observability in mind.  The surface DSL's grounded atoms (`PickReady(k)`) assume full knowledge at compile time.

---

## 7. Code References

### Source Files

| File | Purpose |
|---|---|
| `src/alphazeropp/instances/doors/dsl/reactive_sketch_dsl.py` | BT node types, branch schemas, `canonical_reactive_policy()` |
| `src/alphazeropp/instances/doors/dsl/reactive_sketch_interpreter.py` | `ReactiveContext`, `tick()`, `run_reactive_episode()` |
| `src/alphazeropp/instances/doors/dsl/surface_dsl.py` | Surface DSL types: `PickRule`, `MoveRule`, `GoalRule`, `SurfacePolicy` |
| `src/alphazeropp/instances/doors/dsl/surface_compiler.py` | `compile_policy()` — surface → raw AST |
| `src/alphazeropp/instances/doors/dsl/surface_grammar.py` | `canonical_policy()`, `enumerate_relaxed_policies()`, `count_relaxed_policies()` |
| `src/alphazeropp/instances/doors/dsl/relational_runtime.py` | Typed query interface used by reactive interpreter |
| `src/alphazeropp/synthesis/interpreter.py` | Decision-list interpreter, `EpisodeResult`, `StepRecord` |
| `src/alphazeropp/synthesis/leaf_evaluator.py` | Program evaluation metrics (avg_reward, solve_rate, etc.) |

### Tests and Scripts

| File | Purpose |
|---|---|
| `tests/test_reactive_sketch.py` | 29 tests: types, context, solving, branch order |
| `scripts/run_reactive_sketch_examples.py` | Demo: traces + 24-permutation sweep |
| `scripts/run_reactive_sketch.py` | Production experiment runner with CLI, logging, plots |
| `scripts/enumerate_surface_dsl.py` | Surface policy enumeration and evaluation |

---

## 8. Running the Full Experiment Script

`scripts/run_reactive_sketch.py` provides a production-quality experiment runner modeled after `run_doors_derivation.py`.

### 8.1 Basic Usage

```bash
# Default: evaluate D=2,3,4 with weighted metric
python scripts/run_reactive_sketch.py

# Custom D values
python scripts/run_reactive_sketch.py --d-values 2 3 4 5

# Compare against all relaxed surface policies
python scripts/run_reactive_sketch.py --d-values 2 3 4 --compare-surface

# Skip plot generation
python scripts/run_reactive_sketch.py --d-values 2 3 --no-plot

# Use avg_reward metric
python scripts/run_reactive_sketch.py --metric avg_reward
```

### 8.2 CLI Arguments

| Flag | Default | Description |
|---|---|---|
| `--d-values` | `2 3 4` | Room counts to evaluate |
| `--locs-per-room` | `2` | Locations per room |
| `--metric` | `weighted` | Evaluation metric (`avg_reward`, `solve_rate`, `weighted`) |
| `--n-frozen` | `1` | Frozen initial states per evaluation |
| `--exp-dir` | auto | Override experiment directory |
| `--no-plot` | false | Skip plot generation |
| `--compare-surface` | false | Also evaluate relaxed surface policies |

### 8.3 Output Structure

Results are saved to `experiments/reactive_sketch/{timestamp}_D{min}-{max}_{metric}/`:

```
├── config.json              # CLI args and parameters
├── results.jsonl             # Per-(D, ordering/policy) results
├── summary.jsonl             # Per-D aggregate summary
└── permutation_sweep_D{...}_{metric}.png   # 2-panel plot
```

### 8.4 Terminal Output

```
============================================================
  Reactive Sketch — Exhaustive Permutation Sweep
============================================================

  D values:        [2, 3, 4]
  Metric:          weighted
  Frozen states:   1
  Compare surface: no

============================================================
  D=2 (K=1, 24 permutations)
============================================================

  [SWEEP D=2]   (1,2,3,4)  solved=yes  steps=  3  reward=+1.0700
  [SWEEP D=2]   (1,2,4,3)  solved=yes  steps=  3  reward=+1.0700
  ...

  [RESULT D=2] Solved: 4/24 (16.7%) | Optimal: 4/4 at 3 steps
               Best reward: +1.0700 | Worst solving: +1.0700

============================================================
  Summary
============================================================

    D      Reactive    Optimal   Best Reward
    2          4/24       4/4       +1.0700
    3          4/24       4/4       +1.1500
    4          4/24       4/4       +1.2300
```

### 8.5 Plot Description

Two-panel figure (`permutation_sweep_D{...}_{metric}.png`):

- **Left panel — Solve Rate by D:** Bar chart showing the fraction of 24 orderings that solve, with optional surface overlay.
- **Right panel — Reward Distribution by D:** Scatter plot of all 24 orderings' rewards per D value.  Green dots = solved, red dots = failed.  Surface policies shown as orange diamonds if `--compare-surface`.
