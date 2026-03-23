"""Typed predicate/action catalogs for reactive BT branch composition.

Defines:
  ReactiveBranchCatalog — predicate + action vocabularies with legal matrix
  known_map_catalog()   — factory for known_map=True mode (7 predicates, 5 actions)

Two legal matrix modes:
  "typed" — strict semantic pairing (8 legal pairs per branch)
  "raw"   — cross-product with minimal masking (24 legal pairs per branch)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alphazeropp.instances.doors.dsl.reactive_sketch_dsl import (
    # Predicates
    PickableP, KnownLocP, ExistsUnlockedFrontierP, ReachableP,
    GoalReachedP, TrueP, ExistsUnsearchedRoomP,
    BTPredicate,
    # Actions
    PickAction, GoToAction, NoopAction,
    BTAction,
    # Selectors
    KeyForSel, NextLockedRoomSel, LocOfSel, GoalLocSel, EntranceSel,
    # Structural
    Check, Do, Sequence, Fallback, WhileNot, GoalReachedP as _GoalReachedP,
)


# ---------------------------------------------------------------------------
# Catalog dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReactiveBranchCatalog:
    """Typed catalog of predicates and actions with a legal compatibility matrix.

    The legal_matrix[p, a] is True iff predicate p may be paired with action a
    in a branch Sequence(Check(p), Do(a)).

    Attributes:
        predicates: Tuple of BT predicate instances.
        actions: Tuple of BT action instances.
        legal_matrix: Boolean array of shape (n_pred, n_act).
        predicate_names: Human-readable names for each predicate.
        action_names: Human-readable names for each action.
    """
    predicates: tuple[BTPredicate, ...]
    actions: tuple[BTAction, ...]
    legal_matrix: np.ndarray  # shape (n_pred, n_act), dtype=bool
    predicate_names: tuple[str, ...]
    action_names: tuple[str, ...]

    def __eq__(self, other):
        if not isinstance(other, ReactiveBranchCatalog):
            return NotImplemented
        return (
            self.predicates == other.predicates
            and self.actions == other.actions
            and np.array_equal(self.legal_matrix, other.legal_matrix)
            and self.predicate_names == other.predicate_names
            and self.action_names == other.action_names
        )

    def __hash__(self):
        return hash((self.predicates, self.actions, self.predicate_names, self.action_names))

    @property
    def n_predicates(self) -> int:
        return len(self.predicates)

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def n_legal_pairs(self) -> int:
        """Number of legal (predicate, action) pairs."""
        return int(self.legal_matrix.sum())

    def legal_pairs(self) -> list[tuple[int, int]]:
        """All (pred_idx, act_idx) where legal_matrix is True."""
        rows, cols = np.where(self.legal_matrix)
        return list(zip(rows.tolist(), cols.tolist()))

    def legal_actions_for_predicate(self, pred_idx: int) -> list[int]:
        """Action indices legal for a given predicate."""
        return list(np.where(self.legal_matrix[pred_idx])[0].tolist())

    def predicates_with_legal_actions(self) -> list[int]:
        """Predicate indices that have at least one legal action partner."""
        return list(np.where(self.legal_matrix.any(axis=1))[0].tolist())

    def build_branch(self, pred_idx: int, act_idx: int) -> Sequence:
        """Construct a BT branch: Sequence(Check(predicate), Do(action))."""
        return Sequence(children=(
            Check(self.predicates[pred_idx]),
            Do(self.actions[act_idx]),
        ))

    def build_policy(
        self, branch_specs: tuple[tuple[int, int], ...],
    ) -> WhileNot:
        """Assemble a full reactive BT from branch specifications.

        Args:
            branch_specs: Tuple of (pred_idx, act_idx) pairs, one per branch.

        Returns:
            WhileNot(GoalReachedP(), Fallback(branches...))
        """
        branches = tuple(
            self.build_branch(p, a) for p, a in branch_specs
        )
        return WhileNot(
            predicate=GoalReachedP(),
            child=Fallback(children=branches),
        )

    def spec_names(
        self, branch_specs: tuple[tuple[int, int], ...],
    ) -> list[str]:
        """Human-readable names for each branch in a spec."""
        return [
            f"{self.predicate_names[p]}→{self.action_names[a]}"
            for p, a in branch_specs
        ]


# ---------------------------------------------------------------------------
# Predicate / action indexes (known_map mode)
# ---------------------------------------------------------------------------

# Predicate indices
PRED_PICKABLE = 0
PRED_KNOWN_LOC = 1
PRED_EXISTS_FRONTIER = 2
PRED_REACHABLE = 3
PRED_GOAL_REACHED = 4
PRED_TRUE = 5
PRED_EXISTS_UNSEARCHED = 6

# Action indices
ACT_PICK = 0
ACT_GOTO_KEY = 1
ACT_GOTO_GOAL = 2
ACT_GOTO_ENTRANCE = 3
ACT_NOOP = 4


# ---------------------------------------------------------------------------
# Factory: known_map catalog
# ---------------------------------------------------------------------------

def _build_predicates() -> tuple[tuple[BTPredicate, ...], tuple[str, ...]]:
    """Build the 7-predicate catalog."""
    k = KeyForSel(NextLockedRoomSel())
    predicates = (
        PickableP(k),                    # 0: Pickable
        KnownLocP(k),                    # 1: KnownLoc
        ExistsUnlockedFrontierP(),       # 2: ExistsFrontier
        ReachableP(GoalLocSel()),        # 3: Reachable(GoalLoc)
        GoalReachedP(),                  # 4: GoalReached
        TrueP(),                         # 5: True
        ExistsUnsearchedRoomP(),         # 6: ExistsUnsearched
    )
    names = (
        "Pickable", "KnownLoc", "ExistsFrontier", "Reachable",
        "GoalReached", "True", "ExistsUnsearched",
    )
    return predicates, names


def _build_actions() -> tuple[tuple[BTAction, ...], tuple[str, ...]]:
    """Build the 5-action catalog."""
    k = KeyForSel(NextLockedRoomSel())
    actions = (
        PickAction(k),                                 # 0: Pick
        GoToAction(LocOfSel(k)),                       # 1: GoToKey
        GoToAction(GoalLocSel()),                      # 2: GoToGoal
        GoToAction(EntranceSel(NextLockedRoomSel())),  # 3: GoToEntrance
        NoopAction(),                                  # 4: Noop
    )
    names = ("Pick", "GoToKey", "GoToGoal", "GoToEntrance", "Noop")
    return actions, names


def _typed_legal_matrix() -> np.ndarray:
    """Strict semantic pairing: 8 legal pairs.

    Only the canonical pairings + unconditional (True) variants.
    GoalReached and ExistsUnsearched are fully masked in known_map mode.
    True+Noop is masked (infinite loop).
    """
    m = np.zeros((7, 5), dtype=bool)
    m[PRED_PICKABLE, ACT_PICK] = True          # Pickable → Pick
    m[PRED_KNOWN_LOC, ACT_GOTO_KEY] = True     # KnownLoc → GoToKey
    m[PRED_EXISTS_FRONTIER, ACT_GOTO_ENTRANCE] = True  # ExistsFrontier → GoToEntrance
    m[PRED_REACHABLE, ACT_GOTO_GOAL] = True    # Reachable → GoToGoal
    # GoalReached: fully masked (WhileNot exits before this fires)
    # True: all actions except Noop
    m[PRED_TRUE, ACT_PICK] = True
    m[PRED_TRUE, ACT_GOTO_KEY] = True
    m[PRED_TRUE, ACT_GOTO_GOAL] = True
    m[PRED_TRUE, ACT_GOTO_ENTRANCE] = True
    # ExistsUnsearched: masked in known_map mode
    return m


def _raw_legal_matrix() -> np.ndarray:
    """Cross-product with minimal masking: 24 legal pairs.

    All predicates × all actions, except:
    - GoalReached row (dead branch)
    - ExistsUnsearched row (dead in known_map)
    - True + Noop (infinite loop)
    """
    m = np.ones((7, 5), dtype=bool)
    # Mask GoalReached entirely
    m[PRED_GOAL_REACHED, :] = False
    # Mask ExistsUnsearched entirely (dead in known_map)
    m[PRED_EXISTS_UNSEARCHED, :] = False
    # Mask True + Noop (infinite loop)
    m[PRED_TRUE, ACT_NOOP] = False
    return m


def known_map_catalog(mode: str = "typed") -> ReactiveBranchCatalog:
    """Create a branch catalog for known_map=True mode.

    Args:
        mode: "typed" for strict semantic pairing (8 pairs),
              "raw" for cross-product with minimal masking (24 pairs).

    Returns:
        ReactiveBranchCatalog with 7 predicates and 5 actions.
    """
    predicates, pred_names = _build_predicates()
    actions, act_names = _build_actions()

    if mode == "typed":
        matrix = _typed_legal_matrix()
    elif mode == "raw":
        matrix = _raw_legal_matrix()
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'typed' or 'raw'.")

    return ReactiveBranchCatalog(
        predicates=predicates,
        actions=actions,
        legal_matrix=matrix,
        predicate_names=pred_names,
        action_names=act_names,
    )
