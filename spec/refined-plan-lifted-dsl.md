# Refined Plan: Lifted Decision-List DSL for Doors

**Date:** 2026-03-21
**Branch:** `feature/grammar-redesign`
**Prerequisite:** `relational_runtime.py` (already built, 52 tests passing)

---

## Goal

Implement a lifted DSL where programs contain **no grounded key/room indices**. A single lifted policy compiles to the correct AST for any game size D. The relational runtime is the sole interface to game config — no direct `cfg.key_loc[...]` access.

## Codebase Comparison

| Proposed Element | Classification | Existing Counterpart | Notes |
|---|---|---|---|
| `lifted_dsl.py` — type definitions | **New** | Pattern: `surface_dsl.py` frozen dataclasses with `pretty()` | Follow established dataclass + Union type pattern |
| `lifted_compiler.py` — compilation | **New** | Pattern: `surface_compiler.py` right-fold chaining | Prototype exists: `_build_lifted_policy` in `test_relational_runtime.py:226-248` |
| `compile_lifted_policy(policy, rt)` | **New** | Uses `DoorsRelationalRuntime` only | Verified by `TestLiftedCompilationSmoke` passing |
| `canonical_lifted_policy()` | **New** | Functionally equivalent to `canonical_policy(D)` | But D-independent: one expression for all D |
| `eval_lifted_decision_list.py` | **New** | Pattern: `scripts/enumerate_surface_dsl.py` | Same `section()` + `main()` pattern |
| `test_lifted_dsl.py` | **New** | Pattern: `tests/test_relational_runtime.py` | Same fixture + AST equivalence pattern |
| `__init__.py` update | **Modify** | Add lifted exports | Name collision: alias lifted `NeedKey` as `LiftedNeedKey` |

### Assumptions Verified Against Codebase

| Assumption | Status |
|---|---|
| `DoorsRelationalRuntime` exists with all needed methods | **Verified** — `relational_runtime.py`, 52 tests passing |
| `_build_lifted_policy` prototype produces correct AST | **Verified** — `test_lifted_matches_canonical_d2`, `test_lifted_matches_canonical_d3`, `test_lifted_d3_solves` all pass |
| `canonical_policy(D)` interleaves PickRule/MoveRule per room | **Verified** — `PickRule(0), MoveRule(0), PickRule(1), MoveRule(1), GoalRule` |
| AST nodes support `pretty()` and `node_count()` | **Verified** — `ast_nodes.py` |
| `run_policy_episode` available for solving tests | **Verified** — `interpreter.py` |

### Critical Design Issue Found

The canonical surface policy interleaves rules **per room**: `Pick(0), Move(0), Pick(1), Move(1), GoalRule`. Naively expanding two independent lifted rules (one Pickable, one NeedKey) would group them: `Pick(0), Pick(1), Move(0), Move(1)` — producing a **different AST** with different behavior.

**Solution:** Introduce `ForEachLockedRoom(rules)` construct with `CurrentRoom` placeholder. The compiler iterates rooms and emits ALL rules per room before moving to the next, matching the interleaved order.

### Deferred (Not Needed for Canonical Policy)

| Element | Reason for deferral |
|---|---|
| `NextUnlockedFrontierRoom` | Not used by canonical policy; can add later |
| `KnownLoc(KeySel)` | Always true in known_map mode; needed for partial_map (future) |
| `Reachable(LocSel)` | Implicit in canonical policy structure; not checked explicitly |
| `Entrance(RoomSel)` | Not used by canonical policy |

---

## Execution Plan

### Day 1: Type Definitions (`lifted_dsl.py`)

**Create:** `src/alphazeropp/instances/doors/dsl/lifted_dsl.py`

All frozen dataclasses with `pretty()` methods. No dependencies beyond `dataclasses` and `typing`.

**Selectors:**
```python
@dataclass(frozen=True)
class NextLockedRoom:         # pretty: "NextLockedRoom"
class GoalRoom:               # pretty: "GoalRoom"
class CurrentRoom:            # pretty: "CurrentRoom" — placeholder for ForEachLockedRoom
RoomSel = Union[NextLockedRoom, GoalRoom, CurrentRoom]

@dataclass(frozen=True)
class KeyFor:                 # pretty: "KeyFor(CurrentRoom)"
    room: RoomSel
KeySel = KeyFor

@dataclass(frozen=True)
class LocOf:                  # pretty: "LocOf(KeyFor(CurrentRoom))"
    key: KeySel
class GoalLoc:                # pretty: "GoalLoc"
LocSel = Union[LocOf, GoalLoc]
```

**Predicates:**
```python
@dataclass(frozen=True)
class Pickable:               # pretty: "Pickable(KeyFor(CurrentRoom))"
    key: KeySel
class NeedKey:                # pretty: "NeedKey(KeyFor(CurrentRoom))"
    key: KeySel
class GoalReached:            # pretty: "GoalReached"
LiftedPredicate = Union[Pickable, NeedKey, GoalReached]
```

**Actions:**
```python
@dataclass(frozen=True)
class LiftedPick:             # pretty: "Pick(KeyFor(CurrentRoom))"
    key: KeySel
class GoTo:                   # pretty: "GoTo(LocOf(KeyFor(CurrentRoom)))"
    loc: LocSel
class GoToGoal:               # pretty: "GoToGoal"
LiftedAction = Union[LiftedPick, GoTo, GoToGoal]
```

**Rules & Policy:**
```python
@dataclass(frozen=True)
class IfThen:                 # pretty: "if Pickable(...) then Pick(...)"
    predicate: LiftedPredicate
    action: LiftedAction

class LiftedDefault:          # pretty: "default GoToGoal"
    action: LiftedAction

class ForEachLockedRoom:      # pretty: "for each locked room: [rule1; rule2]"
    rules: tuple[IfThen, ...]

class LiftedPolicy:           # pretty: full policy string
    body: ForEachLockedRoom
    default: LiftedDefault
    # __post_init__ validates types
```

**Constructor:**
```python
def canonical_lifted_policy() -> LiftedPolicy:
    k = KeyFor(CurrentRoom())
    return LiftedPolicy(
        body=ForEachLockedRoom(rules=(
            IfThen(Pickable(k), LiftedPick(k)),
            IfThen(NeedKey(k), GoTo(LocOf(k))),
        )),
        default=LiftedDefault(GoToGoal()),
    )
```

**Justification:** Pure data types with no external dependencies. `canonical_lifted_policy()` is a static constructor — no compilation, no config needed.

### Day 1: Compiler (`lifted_compiler.py`)

**Create:** `src/alphazeropp/instances/doors/dsl/lifted_compiler.py`

Depends on `lifted_dsl.py` and `relational_runtime.py`. Imports AST nodes from `synthesis.ast_nodes`.

**Core function:**
```python
def compile_lifted_policy(policy: LiftedPolicy, rt: DoorsRelationalRuntime) -> Program:
    prog = _compile_default(policy.default, rt)
    for r in reversed(rt.lockable_rooms()):
        k = rt.key_for_room(r)
        assert k is not None
        for rule in reversed(policy.body.rules):
            cond = _compile_predicate(rule.predicate, rt, k)
            action = _compile_action(rule.action, rt, k)
            prog = Ite(cond, action, prog)
    return prog
```

**Double-reversed loop explained:**
- Outer: rooms in reverse `[room_2, room_1]` — so room 1's rules are outermost (checked first)
- Inner: rules in reverse `[NeedKey, Pickable]` — so Pickable wraps NeedKey (checked first)
- Result for D=3: `PickReady(0), NeedKey(0), PickReady(1), NeedKey(1), Default`
- This matches `_build_lifted_policy` and `canonical_policy(3)` exactly

**Helpers:**
- `_compile_predicate(pred, rt, k)`: `Pickable` → `rt.cond_pick_ready(k)`, `NeedKey` → `rt.cond_need_key(k)`, `GoalReached` → `rt.cond_at_loc(rt.goal_location())`
- `_compile_action(act, rt, k)`: `LiftedPick` → `rt.flip_pick(k)`, `GoTo(LocOf(...))` → `rt.flip_move_to_key(k)`, `GoTo(GoalLoc())` / `GoToGoal` → `rt.flip_move_to_goal()`
- `_compile_default(default, rt)`: `GoToGoal` → `Default(rt.flip_move_to_goal())`

**Justification:** Follows the proven `_build_lifted_policy` pattern from `test_relational_runtime.py:226-248`. Uses `DoorsRelationalRuntime` exclusively — no `cfg` access.

### Day 1: Tests (`test_lifted_dsl.py`)

**Create:** `tests/test_lifted_dsl.py`

**Fixtures:** `cfg_d2`, `cfg_d3`, `rt_d2`, `rt_d3` (same pattern as `test_relational_runtime.py`)

**TestLiftedTypes (~5 tests):**
- `canonical_lifted_policy()` pretty string contains `NextLockedRoom`, `Pickable`, `GoToGoal`
- Pretty string contains NO raw indices (`IsZero`, `Flip(`, `key_loc`, `cfg.`)
- Policy structure: 2 rules, Pickable first, NeedKey second, GoToGoal default
- `__post_init__` rejects invalid types

**TestLiftedCompilation (~4 tests):**
- `compile_lifted_policy(canonical_lifted_policy(), rt_d2).pretty() == compile_policy(canonical_policy(2), cfg_d2).pretty()`
- Same for D=3
- Node count matches for D=3 (22 nodes)
- Matches `_build_lifted_policy(rt)` output (import from test_relational_runtime)

**TestLiftedSolves (~3 tests):**
- D=2: `run_policy_episode` → solved, 3 steps
- D=3: → solved, 5 steps
- D=4: → solved, 7 steps (verifies D-generality with same policy)

**TestPrettyNeverEmitsIndices (~1 test):**
- Scan all `pretty()` outputs for forbidden patterns

**Justification:** Follows test patterns from `test_relational_runtime.py`. AST equivalence via `.pretty()` string comparison (established pattern). Episode solving via `run_policy_episode` (established pattern).

### Day 2: Script + Exports

**Create:** `scripts/eval_lifted_decision_list.py`

Follow `enumerate_surface_dsl.py` pattern: `sys.path.insert`, `section()` helper, `main()`.

Sections:
1. Print the canonical lifted policy (pretty — no indices)
2. For D=2..5: compile, print AST node count, verify equivalence with surface compiler
3. Behavioral traces for D=2 and D=3 via `format_trace(run_policy_episode(...))`
4. Summary table: D, steps, node count, equivalence status

**Modify:** `src/alphazeropp/instances/doors/dsl/__init__.py`

Add exports:
```python
from alphazeropp.instances.doors.dsl.lifted_dsl import (  # noqa: F401
    NextLockedRoom, GoalRoom, CurrentRoom, KeyFor, LocOf, GoalLoc,
    Pickable, NeedKey as LiftedNeedKey, GoalReached,
    LiftedPick, GoTo, GoToGoal as LiftedGoToGoal,
    IfThen, LiftedDefault, ForEachLockedRoom, LiftedPolicy,
    canonical_lifted_policy,
)
from alphazeropp.instances.doors.dsl.lifted_compiler import (  # noqa: F401
    compile_lifted_policy,
)
```

---

## Verification

1. `pytest tests/test_lifted_dsl.py -v` — all tests pass
2. `pytest tests/ -v` — no regressions (existing 52 + 73 + 35 tests)
3. `python scripts/eval_lifted_decision_list.py` — runs end-to-end, prints traces
4. AST equivalence verified for D=2, 3, 4:
   - `compile_lifted_policy(canonical_lifted_policy(), rt).pretty() == compile_policy(canonical_policy(D), cfg).pretty()`
5. Pretty-printer output contains zero raw indices, zero config field references
6. Same lifted policy expression works for D=2, D=3, D=4 without modification

## Critical Files

| File | Role |
|---|---|
| [relational_runtime.py](src/alphazeropp/instances/doors/dsl/relational_runtime.py) | Sole interface the lifted compiler uses |
| [surface_compiler.py](src/alphazeropp/instances/doors/dsl/surface_compiler.py) | Reference output for AST equivalence |
| [surface_grammar.py](src/alphazeropp/instances/doors/dsl/surface_grammar.py) | `canonical_policy(D)` for comparison |
| [test_relational_runtime.py:226-248](tests/test_relational_runtime.py) | `_build_lifted_policy` prototype |
| [surface_dsl.py](src/alphazeropp/instances/doors/dsl/surface_dsl.py) | Pattern for dataclass style |
| [enumerate_surface_dsl.py](scripts/enumerate_surface_dsl.py) | Pattern for script structure |
| [__init__.py](src/alphazeropp/instances/doors/dsl/__init__.py) | Update with lifted exports |
