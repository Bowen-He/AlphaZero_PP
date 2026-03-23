# Report: Budget-Free Stage-Skeleton DSL — Changes and Execution Impact

**Date:** 2026-03-21
**Branch:** `feature/grammar-redesign`
**Scope:** Doors environment, `src/alphazeropp/instances/doors/dsl/`

---

## 1. Summary of Changes

A third program representation for the Doors environment was added alongside the two existing ones:

| Representation | Status | Search space (D=2) |
|---|---|---|
| Legacy budgeted AST grammar | Unchanged | ~52 billion |
| Surface sequence DSL (1a) | Unchanged | 1 |
| **Stage-skeleton DSL (new)** | **Added** | **6,175** |

The stage-skeleton DSL unbundles the surface DSL's opaque macro rules (e.g. `PickRule(k)`) into explicit `guard + action` pairs, allowing arbitrary guard composition via `And`/`Not` over typed domain atoms. Budget is completely absent from the syntax — search cost is an external model.

**Files changed:** 6 new source files, 1 modified source file, 1 new test file.

---

## 2. File-by-File Breakdown

All new files live under `src/alphazeropp/instances/doors/dsl/`.

### 2.1 `stage_dsl.py` — Type definitions (124 lines)

Defines the core data model. All types are frozen dataclasses (immutable, hashable).

| Type | Fields | Role |
|------|--------|------|
| `GuardNot` | `child: GuardExpr` | Negation combinator over guard expressions |
| `GuardAnd` | `left, right: GuardExpr` | Conjunction combinator over guard expressions |
| `GuardHole` | `max_depth: int` | Budget-free placeholder for unresolved guards |
| `ActionHole` | (none) | Budget-free placeholder for unresolved actions |
| `Stage` | `guard: GuardExpr\|GuardHole`, `action: SurfaceAction\|ActionHole` | Single guard→action pair |
| `StageProgram` | `stages: tuple[Stage,...]`, `default_action: SurfaceAction` | Complete ordered program |

**Key design decisions:**
- Guard/action atoms (`AtKeyLoc`, `KeyAvail`, `PickReady`, `NeedKey`, `RoomLocked`, `Pick`, `MoveToKey`, `MoveToGoal`) are imported directly from `surface_dsl.py` — zero duplication.
- `GuardNot`/`GuardAnd` are distinct from `ast_nodes.Not`/`ast_nodes.And`. They operate at the typed guard level; the AST versions operate at raw observation indices. The compiler bridges the two.
- `GuardHole(max_depth)` carries a search hint, but this is metadata, not syntax state. No `budget` integer appears anywhere.

**Type aliases:**
```python
GuardAtom = Union[AtKeyLoc, KeyAvail, RoomLocked, PickReady, NeedKey]
GuardExpr = Union[AtKeyLoc, KeyAvail, RoomLocked, PickReady, NeedKey, GuardNot, GuardAnd]
```

### 2.2 `stage_compiler.py` — Compilation to raw AST (63 lines)

Two public functions:

| Function | Signature | What it does |
|----------|-----------|-------------|
| `compile_guard` | `(guard: GuardExpr, cfg: DoorsGameConfig) -> Condition` | Recursively compiles guard expressions. Atoms delegate to `surface_compiler.compile_condition`. `GuardNot` → `ast_nodes.Not`, `GuardAnd` → `ast_nodes.And`. Raises on `GuardHole`. |
| `compile_stage_program` | `(prog: StageProgram, cfg: DoorsGameConfig) -> Program` | Right-folds stages into nested `Ite(guard, action, Ite(..., Default(default)))`. Raises if any holes remain. |

**Delegation pattern:** The compiler imports `surface_compiler.compile_condition` as `_compile_atom` and `surface_compiler.compile_action` directly. It only adds recursion for the two new combinator types. This means all observation-index resolution (key locations, room lock indices, etc.) remains in one place.

**Verified property:** The D=2 canonical stage program `[Stage(PickReady(0), Pick(0)), Stage(NeedKey(0), MoveToKey(0))]` compiles to the identical 12-node AST as the surface canonical policy `PickRule(0) >> MoveRule(0) >> GoalRule`. Test `test_canonical_d2_matches_surface` asserts `surface_ast == stage_ast`.

### 2.3 `stage_search_cost.py` — External cost model (80 lines)

Provides cost functions that are NOT part of the DSL syntax:

| Function / Class | Purpose |
|-----------------|---------|
| `guard_depth(guard) -> int` | Combinator nesting depth (atoms = 0, `Not(atom)` = 1, etc.) |
| `guard_node_count(guard) -> int` | Total nodes in guard expression (each atom or combinator = 1) |
| `stage_program_cost(prog) -> int` | Sum of guard node counts across all stages |
| `SearchCostModel(max_stages, max_guard_depth, max_guard_nodes)` | Frozen dataclass constraining the search space |

`SearchCostModel` has two admission methods:
- `admits_guard(guard)` — checks depth and optional node cap
- `admits_program(prog)` — checks stage count + all guards

### 2.4 `stage_grammar.py` — Enumeration and counting (170 lines)

| Function | Purpose | Complexity |
|----------|---------|-----------|
| `enumerate_guard_atoms(cfg)` | All typed atoms for a Doors config. D=2: 6, D=3: 11 | O(K + D) |
| `enumerate_guards(cfg, max_depth)` | All guards up to depth. Recursive: depth d includes depth d-1 + Not(d-1) + And(d-1, d-1) | O(G(d)^2) |
| `count_guards(cfg, max_depth)` | Closed-form: G(0)=A, G(d)=G(d-1)^2+2*G(d-1) | O(d) |
| `enumerate_actions(cfg)` | All typed actions. D=2: 3, D=3: 5 | O(K) |
| `enumerate_stage_programs(cfg, cost_model)` | Full Cartesian product enumeration. Raises if > 10M | O(total) |
| `count_stage_programs(cfg, cost_model)` | Closed-form: sum_{s=0}^{S} (n_guards * n_actions)^s | O(S) |

**Enumeration generates ordered stage sequences.** Two programs with the same stages in different order are distinct (order matters for guard priority).

### 2.5 `stage_diagnostics.py` — Three-way comparison (233 lines)

Compares the three representations on a common footing:

| Function | What it computes |
|----------|-----------------|
| `diagnose_budgeted(cfg)` | Calls `count_canonical_programs(n_sites, budget)` from legacy grammar. Count only (too large to enumerate). |
| `diagnose_surface(cfg)` | Enumerates all relaxed policies, compiles, evaluates on frozen states. Reports solving count and semantic distinctness. |
| `diagnose_stage(cfg, max_guard_depth, max_stages)` | Enumerates stage programs within cost model, compiles, evaluates. Auto-skips full enumeration if > 10M programs. |
| `format_report(stats_list)` | Formats results as markdown table. |
| `run_diagnostics(D, max_guard_depth)` | Convenience function running all three for a given D. |

**Semantic signature:** For each compiled program, `semantic_signature(prog, states)` runs `eval_program(prog, s)` on a frozen state suite and records the action index per state. Programs with identical signatures are semantically equivalent on those states. The state suite includes:
- Canonical initial state (agent at start, room 0 unlocked, all keys available)
- One variant per key with that key already picked (availability bit = 0)
- Agent relocated to key 0's location

### 2.6 `__init__.py` — Modified (lines 21–34 added)

Four new import blocks were appended after the existing surface DSL exports:

```python
from alphazeropp.instances.doors.dsl.stage_dsl import (
    GuardNot, GuardAnd, GuardHole, ActionHole, Stage, StageProgram,
)
from alphazeropp.instances.doors.dsl.stage_compiler import (
    compile_guard, compile_stage_program,
)
from alphazeropp.instances.doors.dsl.stage_search_cost import (
    SearchCostModel, guard_depth, guard_node_count, stage_program_cost,
)
from alphazeropp.instances.doors.dsl.stage_grammar import (
    enumerate_guard_atoms, enumerate_guards, enumerate_actions,
    enumerate_stage_programs, count_stage_programs,
)
```

All existing imports (lines 1–20) are unchanged.

### 2.7 `tests/test_stage_dsl.py` — 35 tests (296 lines)

| Test Class | Tests | What they cover |
|-----------|-------|----------------|
| `TestStageTypes` | 6 | Pretty-printing, `is_complete()`, hole detection, immutability |
| `TestStageCompiler` | 8 | Compilation correctness, node count, AST equality with surface, solve verification, combinator compilation, hole rejection, empty program |
| `TestSearchCost` | 8 | Depth/node-count arithmetic, cost model admission/rejection |
| `TestStageGrammar` | 8 | Atom/guard/action counts at D=2/D=3, enumeration-vs-count consistency, canonical presence, closed-form D=3 |
| `TestDiagnostics` | 1 | Smoke test: D=2 surface + stage diagnostics run and produce valid report |

---

## 3. Dependency Graph

```
surface_dsl.py  (existing, unchanged)
    │
    ├─── stage_dsl.py         imports guard/action atom types
    │        │
    │        ├─── stage_compiler.py    imports stage types + surface_compiler functions
    │        │
    │        ├─── stage_search_cost.py imports stage types
    │        │        │
    │        │        └─── stage_grammar.py  imports stage types + cost model + surface atoms
    │        │
    │        └─── stage_diagnostics.py imports stage_grammar + stage_compiler + stage_search_cost
    │                                  + surface_grammar + surface_compiler + budget_grammar
    │
surface_compiler.py  (existing, unchanged)
    │
    └─── stage_compiler.py    reuses compile_condition + compile_action

ast_nodes.py  (existing, unchanged)
    │
    └─── stage_compiler.py    imports Not, And, Ite, Default as AST node types

budget_grammar.py  (existing, unchanged)
    │
    └─── stage_diagnostics.py imports count_canonical_programs (for comparison only)
```

No new files import from or modify the legacy budgeted grammar path except `stage_diagnostics.py`, which reads counts for comparison.

---

## 4. Execution Flow Impact

### 4.1 Package import

When `from alphazeropp.instances.doors.dsl import *` is executed, the new `__init__.py` now also imports stage DSL symbols. This adds ~14 new names to the package namespace. Since all imports are lazy (no side effects in module-level code), the import cost is minimal — just loading the 5 new modules.

**Backward compatibility:** All existing imports resolve to the same objects. No symbols were removed or renamed. Code that only uses surface DSL types continues to work without changes.

### 4.2 Stage program construction and compilation

```python
# 1. Build a stage program (pure Python, no I/O)
prog = StageProgram(
    stages=(
        Stage(guard=PickReady(0), action=Pick(0)),
        Stage(guard=NeedKey(0), action=MoveToKey(0)),
    ),
    default_action=MoveToGoal(),
)

# 2. Compile to raw AST (pure, deterministic)
ast = compile_stage_program(prog, cfg)  # → Ite(And(...), Flip(...), Ite(IsZero(...), Flip(...), Default(Flip(...))))

# 3. Evaluate on environment (uses existing interpreter, unchanged)
result = run_policy_episode(env, ast, x0=x0, is_solved=cfg.is_solved)
```

The compilation pipeline is:
1. `compile_stage_program` iterates stages in reverse
2. For each stage, calls `compile_guard` (which recurses on `GuardNot`/`GuardAnd` and delegates atoms to `surface_compiler.compile_condition`)
3. Wraps each stage as `Ite(compiled_guard, compiled_action, rest)`
4. The innermost node is `Default(compile_action(default_action))`

The output is a standard `ast_nodes.Program` — the existing interpreter, leaf evaluator, and all downstream code work on it unchanged.

### 4.3 Enumeration pipeline

```python
# 1. Define search constraints
cost_model = SearchCostModel(max_stages=3, max_guard_depth=0)

# 2. Count without allocating
n = count_stage_programs(cfg, cost_model)  # D=2: 6175

# 3. Enumerate all programs
programs = enumerate_stage_programs(cfg, cost_model)  # list of 6175 StagePrograms

# 4. Compile and evaluate each
for prog in programs:
    ast = compile_stage_program(prog, cfg)
    action = eval_program(ast, some_state)
```

Enumeration uses `itertools.product` over the Cartesian product of (guard, action) slots. A safety check raises `ValueError` if the count exceeds `max_enumerate` (default 10M).

### 4.4 Diagnostics runner

```python
# Compares all three representations
report = run_diagnostics(D=2, max_guard_depth=0)
print(report)  # markdown table
```

For D=2, this enumerates surface (1 program) and stage (6175 programs), compiles all, runs each through `run_policy_episode`, and computes semantic signatures. The budgeted grammar is count-only (too large to enumerate).

---

## 5. What Was NOT Changed

| Component | Location | Status |
|-----------|----------|--------|
| Legacy budgeted grammar | `synthesis/budget_grammar.py` | Untouched |
| Budgeted derivation game | `synthesis/derivation_game.py` | Untouched |
| Factored derivation game | `synthesis/factored_derivation_game.py` | Untouched |
| Doors macros (budgeted) | `doors/dsl/doors_macros.py` | Untouched |
| Budgeted derivation configs | `doors/dsl/derivation_config.py` | Untouched |
| Surface DSL types | `doors/dsl/surface_dsl.py` | Untouched |
| Surface compiler | `doors/dsl/surface_compiler.py` | Untouched |
| Surface grammar | `doors/dsl/surface_grammar.py` | Untouched |
| Surface derivation game | `doors/dsl/surface_derivation_game.py` | Untouched |
| Surface derivation config | `doors/dsl/surface_derivation_config.py` | Untouched |
| AST node types | `synthesis/ast_nodes.py` | Untouched |
| Interpreter | `synthesis/interpreter.py` | Untouched |
| Doors game config | `doors/dsl/doors_config.py` | Untouched |
| All existing tests | `tests/test_surface_*.py` | Pass unchanged (64/64) |

---

## 6. Verified Search Space Numbers

All counts are verified by unit tests in `tests/test_stage_dsl.py`.

### Guard atoms

| D | K (keys) | Atoms per key | Room atoms | Total guard atoms |
|---|----------|--------------|------------|-------------------|
| 2 | 1 | 4 (AtKeyLoc, KeyAvail, PickReady, NeedKey) | 2 (RoomLocked(0,1)) | **6** |
| 3 | 2 | 8 (4 per key) | 3 (RoomLocked(0,1,2)) | **11** |

### Guards by depth

| D | Depth 0 | Depth 1 | Depth 2 |
|---|---------|---------|---------|
| 2 | 6 | 48 | 2,448 |
| 3 | 11 | 143 | 20,735 |

Formula: G(d) = G(d-1)^2 + 2*G(d-1) = G(d-1) * (G(d-1) + 2)

### Stage programs (depth 0, default=MoveToGoal)

| D | K | Slots (guards x actions) | Max stages | Total programs |
|---|---|--------------------------|------------|---------------|
| 2 | 1 | 6 x 3 = 18 | 3 | **6,175** |
| 3 | 2 | 11 x 5 = 55 | 3 | **169,456** |
| 3 | 2 | 11 x 5 = 55 | 5 | **512,604,456** |

Formula: sum_{s=0}^{S} (n_guards * n_actions)^s
