"""Prefix oracle for the reactive typed grammar.

For each prefix of a derivation episode (partial branch selection),
computes exact oracle labels by exhaustive completion:
  - V_max:  maximum reward achievable by any completion
  - P_solve: fraction of completions that solve
  - best_next_mask: which next actions can still reach V_max or a solving completion

The oracle is built by evaluating all complete policies once, then
aggregating bottom-up through a trie indexed by the decision sequence
(pred₁, act₁, pred₂, act₂, ...).
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from alphazeropp.instances.doors.dsl.reactive_branch_catalog import (
    ReactiveBranchCatalog,
)
from alphazeropp.instances.doors.dsl.reactive_leaf_evaluator import (
    ReactiveLeafEvaluator,
)


# ---------------------------------------------------------------------------
# Oracle entry
# ---------------------------------------------------------------------------

@dataclass
class PrefixOracleEntry:
    """Oracle labels for a single prefix node in the derivation trie."""
    # Prefix identification
    prefix_specs: tuple[tuple[int, int], ...]  # completed branches
    pending_pred: int | None  # predicate chosen for current branch, None if at branch boundary

    # Oracle labels
    v_max: float                # max reward over all completions
    p_solve: float              # fraction of completions that solve
    best_next_mask: np.ndarray  # shape (action_space_size,), bool
    n_completions: int          # total completions from this prefix
    n_solving: int              # solving completions from this prefix


# ---------------------------------------------------------------------------
# Trie node (internal)
# ---------------------------------------------------------------------------

@dataclass
class _TrieNode:
    """Internal trie node for aggregation."""
    children: dict[int, "_TrieNode"] = field(default_factory=dict)
    # Leaf data (only for complete policies)
    reward: float | None = None
    solved: bool | None = None


# ---------------------------------------------------------------------------
# Build prefix oracle
# ---------------------------------------------------------------------------

def build_prefix_oracle(
    n_branches: int,
    catalog: ReactiveBranchCatalog,
    evaluator: ReactiveLeafEvaluator,
    *,
    action_space_size: int | None = None,
) -> dict[tuple, PrefixOracleEntry]:
    """Exhaustively compute oracle labels for all prefixes.

    Algorithm:
    1. Enumerate all complete policies and evaluate each via the leaf evaluator.
    2. Build a trie indexed by the decision sequence (pred₀, act₀, pred₁, act₁, ...).
    3. Walk the trie bottom-up to compute V_max, P_solve, best_next_mask at each node.

    Args:
        n_branches: Number of branches in the Fallback.
        catalog: ReactiveBranchCatalog with legal matrix.
        evaluator: ReactiveLeafEvaluator for scoring complete policies.
        action_space_size: Size of the action mask (default: max(n_pred, n_act)).

    Returns:
        Dict mapping decision_prefix_tuple → PrefixOracleEntry.
        The empty tuple () is the root entry.
    """
    if action_space_size is None:
        action_space_size = max(catalog.n_predicates, catalog.n_actions)

    legal_pairs = catalog.legal_pairs()

    # Step 1: Evaluate all complete policies
    all_specs = list(itertools.product(legal_pairs, repeat=n_branches))
    rewards: dict[tuple, float] = {}
    solved: dict[tuple, bool] = {}
    for specs in all_specs:
        specs_tuple = tuple(specs)
        r = evaluator(specs_tuple)
        m = evaluator.get_all_metrics(specs_tuple)
        rewards[specs_tuple] = r
        solved[specs_tuple] = m["n_solved"] > 0

    # Step 2: Build trie
    # Decision sequence for a policy ((p0,a0), (p1,a1), ...) is
    # the flat sequence [p0, a0, p1, a1, ...]
    root = _TrieNode()
    for specs_tuple in all_specs:
        node = root
        for pred_idx, act_idx in specs_tuple:
            if pred_idx not in node.children:
                node.children[pred_idx] = _TrieNode()
            node = node.children[pred_idx]
            if act_idx not in node.children:
                node.children[act_idx] = _TrieNode()
            node = node.children[act_idx]
        node.reward = rewards[specs_tuple]
        node.solved = solved[specs_tuple]

    # Step 3: Walk trie and compute oracle labels
    oracle: dict[tuple, PrefixOracleEntry] = {}
    _walk_trie(root, (), (), None, n_branches, action_space_size, oracle)

    return oracle


def _walk_trie(
    node: _TrieNode,
    decision_prefix: tuple[int, ...],
    branch_specs: tuple[tuple[int, int], ...],
    pending_pred: int | None,
    n_branches: int,
    action_space_size: int,
    oracle: dict[tuple, PrefixOracleEntry],
) -> tuple[float, int, int]:
    """Recursive trie walk. Returns (v_max, n_completions, n_solving)."""

    # Leaf: complete policy
    if node.reward is not None:
        assert len(branch_specs) == n_branches
        entry = PrefixOracleEntry(
            prefix_specs=branch_specs,
            pending_pred=pending_pred,
            v_max=node.reward,
            p_solve=1.0 if node.solved else 0.0,
            best_next_mask=np.zeros(action_space_size, dtype=bool),  # terminal
            n_completions=1,
            n_solving=1 if node.solved else 0,
        )
        oracle[decision_prefix] = entry
        return node.reward, 1, (1 if node.solved else 0)

    # Internal node: aggregate children
    child_results: dict[int, tuple[float, int, int]] = {}
    for action_id, child_node in node.children.items():
        # Determine child's branch_specs and pending_pred
        if pending_pred is None:
            # We're at a branch boundary, choosing a predicate
            child_specs = branch_specs
            child_pending = action_id
        else:
            # We have a pending predicate, choosing an action
            child_specs = branch_specs + ((pending_pred, action_id),)
            child_pending = None

        child_v, child_n, child_s = _walk_trie(
            child_node,
            decision_prefix + (action_id,),
            child_specs,
            child_pending,
            n_branches,
            action_space_size,
            oracle,
        )
        child_results[action_id] = (child_v, child_n, child_s)

    # Aggregate
    v_max = max(v for v, _, _ in child_results.values())
    n_completions = sum(n for _, n, _ in child_results.values())
    n_solving = sum(s for _, _, s in child_results.values())

    # Best-next mask: which children can reach v_max or have a solving completion
    best_next_mask = np.zeros(action_space_size, dtype=bool)
    for action_id, (child_v, child_n, child_s) in child_results.items():
        if child_v >= v_max - 1e-9 or child_s > 0:
            best_next_mask[action_id] = True

    entry = PrefixOracleEntry(
        prefix_specs=branch_specs,
        pending_pred=pending_pred,
        v_max=v_max,
        p_solve=n_solving / n_completions if n_completions > 0 else 0.0,
        best_next_mask=best_next_mask,
        n_completions=n_completions,
        n_solving=n_solving,
    )
    oracle[decision_prefix] = entry

    return v_max, n_completions, n_solving
