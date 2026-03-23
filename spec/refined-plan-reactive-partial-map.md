# Refined Plan: Partial Observability & Explicit Memory for Reactive Sketch

**Date:** 2026-03-22
**Status:** Ready for execution
**Prerequisite:** Reactive sketch exact oracle (Stage 1) is complete — 38 tests passing.

---

## Context

The reactive sketch BT (4 branches, 24 permutations) operates under full observability (`known_map=True`). The `DoorsRelationalRuntime` already supports `known_map=False` (makes `loc_of_key()` return `None`), but there is no mechanism for runtime key discovery nor memory to track discoveries.

This plan adds:
- A `ReactiveMemory` dataclass for tracking discovered key locations and searched rooms
- A 5th branch ("explore unsearched room") enabling systematic exploration
- Memory-aware predicate evaluation (backward compatible)
- Exact enumeration of all 5! = 120 branch orderings
- A no-hidden-state-leak test proving the policy cannot access `cfg.key_loc`

---

## Codebase Comparison

| Original plan item | Classification | Rationale |
|---|---|---|
| `reactive_partial_map.py` (env wrapper) | **Redundant** | Obs vector already omits key locations. `known_map=False` in runtime suffices. |
| `reactive_memory.py` | **New** | No memory mechanism exists. |
| `reactive_runtime.py` | **Redundant** | `DoorsRelationalRuntime(known_map=False)` provides the right interface. Memory-aware resolution belongs in `ReactiveContext`, not a separate runtime. |
| `reactive_sketch_interpreter.py` | **Modify** | Already exists (319 lines). Add optional memory param + new episode runner. |
| `reactive_sketch_dsl.py` | **Modify** | Already exists (314 lines). Add B5 branch, 3 new types. |
| `run_reactive_partial_map_exact.py` | **New** | No existing partial-map enumeration script. |
| `test_reactive_partial_map.py` | **New** | Merges proposed `test_reactive_memory.py` to reduce file count. |
| `test_reactive_memory.py` | **Redundant** | Merged into `test_reactive_partial_map.py`. |
| `docs/reactive_stage2_partial_map.md` | **Redundant** | Script emits markdown to `results/`. Separate doc is redundant. |

**Existing utilities to reuse:**
- `DoorsRelationalRuntime(cfg, known_map=False)` — already returns `None` from `loc_of_key()` (`src/.../dsl/relational_runtime.py`)
- `make_frozen_state_suite(cfg)` — evaluation states (`src/.../dsl/stage_diagnostics.py:70`)
- `reactive_semantic_signature(policy, rt, states)` — behavioral fingerprints (`src/.../dsl/stage_diagnostics.py:43`)
- `run_reactive_episode()` — existing episode runner, model for new partial variant (`src/.../dsl/reactive_sketch_interpreter.py:217`)
- `EpisodeResult`, `StepRecord` — shared data types (`src/alphazeropp/synthesis/interpreter.py`)
- `doors_initial_state(cfg)` — initial state constructor (`src/.../dsl/doors_config.py`)
- `BRANCH_BUILDERS` dict — extensible by adding key 5 (`src/.../dsl/reactive_sketch_dsl.py:282`)

---

## Execution Plan

### Day 1, Step 1: Create `reactive_memory.py`

**File:** `src/alphazeropp/instances/doors/dsl/reactive_memory.py` (NEW, ~50 lines)
**Justification:** Pure dataclass with no interpreter dependencies. Can be tested immediately.

```python
@dataclass
class ReactiveMemory:
    known_key_locs: dict[int, int]   # KeyId → LocId
    searched_rooms: set[int]          # Visited RoomIds

    @staticmethod
    def empty() -> ReactiveMemory: ...

    def discover_keys_in_room(self, room: int, cfg: DoorsGameConfig) -> None:
        """Reveal keys located in `room` using cfg.key_loc (called by episode runner, never by policy)."""

    def mark_searched(self, room: int) -> None: ...
    def loc_of_key(self, k: int) -> int | None: ...
    def copy(self) -> ReactiveMemory: ...
```

**Design note:** `discover_keys_in_room(room, cfg)` takes `cfg` as argument — this simulates what a physical agent observes when entering a room. The BT policy never calls this; only the episode runner does. This is the key information-hiding boundary.

### Day 1, Step 2: Extend `reactive_sketch_dsl.py`

**File:** `src/alphazeropp/instances/doors/dsl/reactive_sketch_dsl.py` (MODIFY, +40 lines)
**Justification:** All existing types/functions untouched. Additions are pure frozen dataclasses and a new function.

Add:
1. `NextUnsearchedRoomSel` — frozen dataclass, first reachable room not in `memory.searched_rooms`
2. `FirstLocInRoomSel(room: BTRoomSel)` — frozen dataclass, first location INSIDE the room (distinct from `EntranceSel` which goes to room r-1)
3. `ExistsUnsearchedRoomP` — frozen dataclass predicate
4. Update union types: `BTRoomSel = Union[NextLockedRoomSel, NextUnsearchedRoomSel]`, extend `BTLocSel` and `BTPredicate`
5. `branch_explore_unsearched()` → `Sequence(Check(ExistsUnsearchedRoomP()), Do(GoToAction(FirstLocInRoomSel(NextUnsearchedRoomSel()))))`
6. `BRANCH_BUILDERS[5] = branch_explore_unsearched`
7. `canonical_partial_map_policy(branch_order=(1,2,5,3,4)) -> WhileNot` — validates permutation of (1,2,3,4,5)

**Backward compatibility:** `canonical_reactive_policy()` validates `sorted(order) == [1,2,3,4]` — unchanged. `BRANCH_BUILDERS` gains key 5; no existing code iterates all keys.

### Day 1, Step 3: Extend `reactive_sketch_interpreter.py`

**File:** `src/alphazeropp/instances/doors/dsl/reactive_sketch_interpreter.py` (MODIFY, +90 lines)
**Justification:** All changes are backward compatible (memory defaults to None). New episode runner is a separate function.

#### 3a. Add memory to ReactiveContext
- Change `__slots__` from `("rt", "obs")` to `("rt", "obs", "memory")`
- Add `memory=None` to `__init__`

#### 3b. Modify 3 existing methods (memory-aware routing)

**`resolve_loc` for `LocOfSel`** (line 93-97):
```python
# When memory present, use memory.loc_of_key(k) instead of rt.loc_of_key(k)
```

**`eval_predicate` for `PickableP`** (line 127):
```python
# When memory present, route key location through memory
```

**`eval_predicate` for `KnownLocP`** (line 134-138):
```python
# When memory present, check memory.loc_of_key(k) is not None
```

#### 3c. Add new predicate/selector resolution

- `ExistsUnsearchedRoomP`: iterate rooms 0..D-1, check `r not in memory.searched_rooms` AND `obs[obs_room_unlocked(r)] == 1.0`
- `NextUnsearchedRoomSel`: same iteration, return first match
- `FirstLocInRoomSel`: resolve room, then `rt.locations_in_room(r)[0]`

#### 3d. Add `run_reactive_partial_episode()`

```python
def run_reactive_partial_episode(
    env, policy, rt, cfg, x0=None, *, is_solved,
) -> EpisodeResult:
```

Differs from `run_reactive_episode()`:
1. Creates `ReactiveMemory.empty()` at start
2. Discovers keys in starting room (`cfg.loc_room[cfg.start_loc]`)
3. Passes `memory=memory` to `ReactiveContext` each tick
4. After each step, detects agent's room via `argmax(obs[:M])` → `cfg.loc_room[loc]`
5. If room not yet searched, calls `memory.discover_keys_in_room(room, cfg)`

### Day 1, Step 4: Modify `stage_diagnostics.py`

**File:** `src/alphazeropp/instances/doors/dsl/stage_diagnostics.py` (MODIFY, +5 lines)
**Justification:** `reactive_semantic_signature` needs optional memory for partial-map signatures.

Add `memory=None` parameter to `reactive_semantic_signature()`:
```python
def reactive_semantic_signature(policy, rt, states, memory=None):
    ...
    ctx = ReactiveContext(rt, s, memory=memory)
```

### Day 2, Step 5: Create tests

**File:** `tests/test_reactive_partial_map.py` (NEW, ~150 lines)
**Justification:** Tests each new component and the key acceptance criteria.

| Test class | Tests | What it verifies |
|---|---|---|
| `TestReactiveMemory` | 4 tests | Memory dataclass, discovery, isolation |
| `TestPartialMapPredicates` | 6 tests | KnownLocP, PickableP, ExistsUnsearchedRoomP with/without memory |
| `TestNoHiddenStateLeak` | 1 test | Two configs with different key_loc + identical obs+memory → same BT action |
| `TestPartialMapEpisodes` | 3 tests | Canonical solves D=2 in 3 steps; trace matches known_map; 120 permutations enumerate |
| `TestBackwardCompat` | 1 test | 4-branch `canonical_reactive_policy()` still works |

**No-hidden-state-leak test design:**
- Create `DoorsGameConfig(num_rooms=2, key_loc=[0])` and `DoorsGameConfig(num_rooms=2, key_loc=[1])`
- Build identical obs vectors (same at_loc, unlocked, key_avail) and identical empty memories
- Create `DoorsRelationalRuntime(cfg, known_map=False)` for each
- Tick `canonical_partial_map_policy()` on both → assert identical actions

### Day 2, Step 6: Create enumeration script

**File:** `scripts/run_reactive_partial_map_exact.py` (NEW, ~120 lines)
**Justification:** All building blocks exist. Script is pure composition.

CLI: `python scripts/run_reactive_partial_map_exact.py --D 2 [--output-dir results/reactive_partial_map]`

For each of 5! = 120 permutations:
1. Build `canonical_partial_map_policy(perm)`
2. Run `run_reactive_partial_episode()` with `DoorsRelationalRuntime(cfg, known_map=False)`
3. Compute behavioral signature
4. Group by signature → equivalence classes
5. Output `D{D}_policies.json`, `D{D}_equivalence.json`, `D{D}_summary.md`

---

## Verification

```bash
# 1. Backward compatibility — all 38 existing tests pass
pytest -q tests/test_reactive_sketch.py tests/test_reactive_sketch_exact.py -v

# 2. New partial-map tests pass
pytest -q tests/test_reactive_partial_map.py -v

# 3. 120-policy exact report
python scripts/run_reactive_partial_map_exact.py --D 2
# Verify: results/reactive_partial_map/D2_policies.json has 120 entries

# 4. Canonical partial-map policy solves D=2 in 3 steps
# Covered by test_canonical_partial_solves_d2

# 5. No-hidden-state-leak property
# Covered by TestNoHiddenStateLeak
```

---

## Files Summary

| File | Action | Est. lines |
|---|---|---|
| `src/alphazeropp/instances/doors/dsl/reactive_memory.py` | CREATE | ~50 |
| `src/alphazeropp/instances/doors/dsl/reactive_sketch_dsl.py` | MODIFY | +40 |
| `src/alphazeropp/instances/doors/dsl/reactive_sketch_interpreter.py` | MODIFY | +90 |
| `src/alphazeropp/instances/doors/dsl/stage_diagnostics.py` | MODIFY | +5 |
| `tests/test_reactive_partial_map.py` | CREATE | ~150 |
| `scripts/run_reactive_partial_map_exact.py` | CREATE | ~120 |
