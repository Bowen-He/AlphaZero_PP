# Refined Plan: Reactive Typed Grammar — Compositional Branch Search

**Date:** 2026-03-22
**Stage:** 3 (per roadmap in `specs/2026-03-22-report_reactive_sketch_derivation_game.md`)
**Prerequisite:** Stages 0-1 complete (fixed branches + partial observability, 53 tests passing)

---

## 1. Codebase Comparison

### Proposed → Existing Code Mapping

| Proposed file | Classification | Existing counterpart | Notes |
|---|---|---|---|
| `reactive_branch_catalog.py` | **NEW** | None | Predicate/action catalogs + legal compatibility matrix |
| `reactive_typed_grammar.py` | **NEW** | `surface_grammar.py`, `stage_grammar.py` | Follows count/enumerate pattern |
| `reactive_derivation_game.py` | **NEW** | `surface_derivation_game.py`, `stage_derivation_game.py` | Follows Game interface pattern |
| `reactive_signature.py` | **REDUNDANT** | `stage_diagnostics.py::reactive_semantic_signature()` | Already exists with memory support — skip |
| `reactive_sketch_dsl.py` | **MODIFY** | Existing, 378 lines | Add `TrueP` + `NoopAction` (2 dataclasses, 2 union updates) |
| `reactive_sketch_interpreter.py` | **MODIFY** | Existing, 457 lines | Handle new types in `eval_predicate()` + `resolve_action()` |
| `stage_diagnostics.py` | **MODIFY** | Existing, 297 lines | Add `diagnose_reactive_typed()` |
| `run_reactive_typed_search.py` | **NEW** | `run_reactive_exact.py`, `run_reactive_partial_map_exact.py` | Follows existing script patterns |
| `test_reactive_typed_grammar.py` | **NEW** | `test_reactive_sketch.py`, `test_surface_derivation_game.py` | Follows existing test patterns |
| `test_reactive_typed_search.py` | **REDUNDANT** | Merge into `test_reactive_typed_grammar.py` | Single test file |
| `docs/reactive_stage3_typed_grammar.md` | **SKIP** | Script emits reports to `results/` | No separate doc |

### Assumptions Validated Against Codebase

| Assumption | Status |
|---|---|
| BT node types (Check, Do, Sequence, Fallback, WhileNot) exist | **Confirmed** — `reactive_sketch_dsl.py` |
| All predicates/selectors/actions exist | **Confirmed** — 6 predicates, 2 room selectors, 4 loc selectors, 2 actions |
| `ReactiveContext` resolves predicates against obs | **Confirmed** — `reactive_sketch_interpreter.py` |
| `run_reactive_episode()` returns `EpisodeResult` | **Confirmed** — reuses `interpreter.py::EpisodeResult` |
| `reactive_semantic_signature()` exists | **Confirmed** — `stage_diagnostics.py` with `memory=None` param |
| `Game` base class with stash/unstash/clone | **Confirmed** — `core/game.py`, used by Surface and Stage games |
| `TrueP` predicate exists | **MISSING** — must add |
| `NoopAction` exists | **MISSING** — must add |
| `NeedKey` predicate needed | **NOT NEEDED** — equivalent to `ExistsUnlockedFrontierP()` in BT context |

### Existing Utilities to Reuse

| Utility | Location | How reused |
|---|---|---|
| `reactive_semantic_signature()` | `stage_diagnostics.py:43` | Behavioral dedup for typed policies |
| `make_frozen_state_suite()` | `stage_diagnostics.py:74` | Evaluation states for signatures |
| `run_reactive_episode()` | `reactive_sketch_interpreter.py:269` | Terminal evaluation in derivation game |
| `RepresentationStats` | `stage_diagnostics.py:101` | Report formatting |
| `format_report()` | `stage_diagnostics.py:263` | Markdown output |
| `DoorsRelationalRuntime` | `relational_runtime.py` | Config queries during evaluation |
| `doors_initial_state()` | `doors_config.py` | Starting state for episodes |
| `BRANCH_BUILDERS` dict | `reactive_sketch_dsl.py:321` | Reference for canonical branch definitions |

---

## 2. Implementation Steps

### Day 1: DSL Extensions + Catalog

#### Step 1: Add `TrueP` and `NoopAction` to `reactive_sketch_dsl.py`

**File:** `src/alphazeropp/instances/doors/dsl/reactive_sketch_dsl.py`

- Add `TrueP` frozen dataclass with `pretty() -> "True"`
- Add `NoopAction` frozen dataclass with `pretty() -> "Noop"`
- Update `BTPredicate` union to include `TrueP`
- Update `BTAction` union to include `NoopAction`

**Executable given:** All types are frozen dataclasses following the exact pattern of the 6 existing predicates and 2 existing actions. No structural changes.

#### Step 2: Handle new types in `reactive_sketch_interpreter.py`

**File:** `src/alphazeropp/instances/doors/dsl/reactive_sketch_interpreter.py`

- Add `TrueP` case to `eval_predicate()` (return True)
- Add `NoopAction` case to `resolve_action()` (return `rt.cfg.M + rt.cfg.K`)
- Update imports

**Executable given:** Both methods use isinstance chains. Adding cases is mechanical.

**Checkpoint:** `pytest tests/test_reactive_sketch.py tests/test_reactive_sketch_exact.py tests/test_reactive_partial_map.py` — all pass unchanged.

#### Step 3: Create `reactive_branch_catalog.py`

**File:** `src/alphazeropp/instances/doors/dsl/reactive_branch_catalog.py` (NEW, ~150 lines)

`ReactiveBranchCatalog` frozen dataclass with:
- 7 predicates, 5 actions (known_map mode)
- Legal matrix: (7, 5) bool array
- `build_branch()`, `build_policy()`, `legal_pairs()`, `n_legal_pairs()`
- Factory: `known_map_catalog(mode="typed"|"raw")`

**Legal matrix (typed mode, 8 pairs):**

```
              Pick  GoToKey  GoToGoal  GoToEnt  Noop
Pickable       Y      N        N        N       N
KnownLoc       N      Y        N        N       N
ExistsFr       N      N        N        Y       N
Reachable      N      N        Y        N       N
GoalReached    N      N        N        N       N   ← dead (WhileNot exits first)
True           Y      Y        Y        Y       N   ← True+Noop = infinite loop
ExistsUnsr     N      N        N        N       N   ← dead in known_map
```

**Legal matrix (raw mode, 24 pairs):** All pairs except GoalReached row, ExistsUnsearched row, and True+Noop cell.

**Executable given:** Pure data definitions. Uses existing BT node constructors from `reactive_sketch_dsl.py`.

### Day 2: Grammar Enumeration + Diagnostics

#### Step 4: Create `reactive_typed_grammar.py`

**File:** `src/alphazeropp/instances/doors/dsl/reactive_typed_grammar.py` (NEW, ~100 lines)

```python
count_reactive_typed_policies(n_branches, catalog) -> int
enumerate_reactive_typed_policies(n_branches, catalog, max_enumerate=10M) -> list[branch_specs]
evaluate_reactive_typed_policies(specs, catalog, cfg, rt, states) -> list[dict]
```

Uses `itertools.product(catalog.legal_pairs(), repeat=n_branches)`. Returns `branch_specs` tuples, not assembled WhileNot objects.

**Executable given:** Direct analog of `enumerate_stage_programs()` and `enumerate_relaxed_policies()`. The Cartesian product pattern is used in both.

#### Step 5: Extend `stage_diagnostics.py`

**File:** `src/alphazeropp/instances/doors/dsl/stage_diagnostics.py`

Add `diagnose_reactive_typed(cfg, n_branches=4, mode="typed") -> RepresentationStats`.

**Executable given:** Follows exact pattern of existing `diagnose_reactive()` at line 223. Uses same `reactive_semantic_signature()` and `make_frozen_state_suite()`.

### Day 3: Derivation Game

#### Step 6: Create `reactive_derivation_game.py`

**File:** `src/alphazeropp/instances/doors/dsl/reactive_derivation_game.py` (NEW, ~200 lines)

Follows `SurfaceDerivationGame` pattern:
- `ReactiveDerivationState` (frozen dataclass: branch_specs, pending_pred, decisions)
- `ReactiveDerivationGame(Game)` with:
  - `Discrete(max(n_pred, n_act))` action space
  - Fixed 2*N step episodes (pred/action alternation)
  - Legal mask: even steps → predicate mask, odd steps → `legal_matrix[pending_pred, :]`
  - Terminal: `catalog.build_policy()` → `run_reactive_episode()` → reward
  - Cache by `branch_specs` tuple
  - `stash_state()`, `unstash_state()`, `clone()`, `hashable_obs`

**Executable given:** Direct structural analog of `SurfaceDerivationGame` (248 lines) and `StageDerivationGame` (282 lines). Same base class, same interface pattern.

**No external dependencies required.**

### Day 4: CLI Script + Tests

#### Step 7: Create `scripts/run_reactive_typed_search.py`

**File:** `scripts/run_reactive_typed_search.py` (NEW, ~180 lines)

CLI: `python scripts/run_reactive_typed_search.py --D 2 3 --mode both`

Reports: total candidates, unique signatures, solve count/rate, first-solve index, comparison with fixed-permutation baseline.

**Executable given:** Follows pattern of `run_reactive_partial_map_exact.py`. Uses `enumerate_reactive_typed_policies()` + `evaluate_reactive_typed_policies()`.

#### Step 8: Create tests

**File:** `tests/test_reactive_typed_grammar.py` (NEW, ~200 lines)

- **TestCatalog** (8 tests): sizes, matrix shape, pair counts, masking, build_branch, build_policy
- **TestEnumeration** (5 tests): count consistency, D=2 count, canonical inclusion, solver existence, typed < raw
- **TestDerivationGame** (6 tests): episode length, mask alternation, compatibility, reward matching, stash/unstash, clone
- **TestAcceptanceCriteria** (3 tests): absurd pair masking, D=3 solver, typed fewer unique than raw

**Executable given:** Follows patterns of `test_reactive_sketch.py` (fixture-based, cfg_d2/cfg_d3) and `test_surface_derivation_game.py` (game lifecycle tests).

---

## 3. Files Summary

| File | Action | Lines | Depends on |
|---|---|---|---|
| `src/.../dsl/reactive_sketch_dsl.py` | MODIFY | +15 | — |
| `src/.../dsl/reactive_sketch_interpreter.py` | MODIFY | +10 | Step 1 |
| `src/.../dsl/reactive_branch_catalog.py` | CREATE | ~150 | Steps 1-2 |
| `src/.../dsl/reactive_typed_grammar.py` | CREATE | ~100 | Step 3 |
| `src/.../dsl/reactive_derivation_game.py` | CREATE | ~200 | Steps 3-4 |
| `src/.../dsl/stage_diagnostics.py` | MODIFY | +30 | Steps 3-4 |
| `scripts/run_reactive_typed_search.py` | CREATE | ~180 | Steps 4-5 |
| `tests/test_reactive_typed_grammar.py` | CREATE | ~200 | Steps 3-6 |

## 4. What We Skip (and Why)

| Skipped item | Reason |
|---|---|
| `reactive_signature.py` | `reactive_semantic_signature()` exists in `stage_diagnostics.py` with memory param |
| `reactive_leaf_evaluator.py` | Inline `run_reactive_episode()` in derivation game — simpler, no class hierarchy needed |
| `NeedKey` predicate | Equivalent to `ExistsUnlockedFrontierP()` in BT context — `NextLockedRoomSel` handles parameterization |
| `docs/reactive_stage3_typed_grammar.md` | Script emits markdown reports to `results/` |
| Beam/best-first search | 4K-330K policies are tractable for exact enumeration. Add later per `stage_search.py` pattern when N≥5 |
| `partial_map_catalog()` | Defer to follow-up — known_map mode proves the architecture; partial_map adds ExistsUnsearched + FirstLocIn actions |

## 5. Verification

```bash
# 1. Backward compatibility
pytest tests/test_reactive_sketch.py tests/test_reactive_sketch_exact.py tests/test_reactive_partial_map.py -v

# 2. New tests pass
pytest tests/test_reactive_typed_grammar.py -v

# 3. Typed search finds solver for D=3
python scripts/run_reactive_typed_search.py --D 3 --mode typed
# Expect: at least one policy solves

# 4. Typed masks absurd pairs (GoalReached→Pick is illegal)
# Verified by test_typed_masks_absurd_pairs

# 5. Typed < raw unique policies, both preserve solvers
python scripts/run_reactive_typed_search.py --D 2 --mode both
# Expect: typed distinct < raw distinct, both solve_count > 0
```
