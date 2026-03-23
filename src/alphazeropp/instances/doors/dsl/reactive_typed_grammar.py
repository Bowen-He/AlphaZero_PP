"""Enumeration and counting for the reactive typed grammar.

Provides:
  count_reactive_typed_policies   — closed-form count (n_legal_pairs ^ n_branches)
  enumerate_reactive_typed_policies — all branch-spec tuples via Cartesian product
  evaluate_reactive_typed_policies  — run episodes + compute signatures for all specs
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state,
)
from alphazeropp.instances.doors.dsl.reactive_branch_catalog import (
    ReactiveBranchCatalog,
)
from alphazeropp.instances.doors.dsl.relational_runtime import (
    DoorsRelationalRuntime,
)
from alphazeropp.instances.doors.dsl.stage_diagnostics import (
    reactive_semantic_signature,
)
from alphazeropp.instances.doors.dsl.reactive_sketch_interpreter import (
    run_reactive_episode,
)


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def count_reactive_typed_policies(
    n_branches: int,
    catalog: ReactiveBranchCatalog,
) -> int:
    """Total number of policies: n_legal_pairs ^ n_branches."""
    return catalog.n_legal_pairs() ** n_branches


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def enumerate_reactive_typed_policies(
    n_branches: int,
    catalog: ReactiveBranchCatalog,
    *,
    max_enumerate: int = 10_000_000,
) -> list[tuple[tuple[int, int], ...]]:
    """Enumerate all branch-spec tuples via Cartesian product of legal pairs.

    Each branch spec is a tuple of (pred_idx, act_idx) pairs — one per branch.
    Assembly into WhileNot BT nodes is deferred to catalog.build_policy().

    Args:
        n_branches: Number of branches in the Fallback node.
        catalog: ReactiveBranchCatalog with legal matrix.
        max_enumerate: Safety limit. Raises ValueError if exceeded.

    Returns:
        List of branch_specs tuples.
    """
    total = count_reactive_typed_policies(n_branches, catalog)
    if total > max_enumerate:
        raise ValueError(
            f"Too many policies to enumerate: {total:,} > {max_enumerate:,}. "
            f"Use max_enumerate= to increase the limit."
        )

    legal = catalog.legal_pairs()
    return [
        tuple(combo)
        for combo in itertools.product(legal, repeat=n_branches)
    ]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_reactive_typed_policies(
    branch_specs_list: list[tuple[tuple[int, int], ...]],
    catalog: ReactiveBranchCatalog,
    cfg: DoorsGameConfig,
    rt: DoorsRelationalRuntime,
    eval_states: list[np.ndarray],
) -> list[dict]:
    """Evaluate all policies: run episodes, compute behavioral signatures.

    For each branch_specs, builds the BT policy, runs one episode on the
    initial state, and computes a behavioral signature on eval_states.

    Returns:
        List of dicts with keys: branch_specs, branch_names, solved, reward,
        steps, action_trace, signature.
    """
    x0 = doors_initial_state(cfg)
    results = []

    for specs in branch_specs_list:
        policy = catalog.build_policy(specs)

        # Run episode
        env = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
        episode = run_reactive_episode(
            env, policy, rt, x0=x0, is_solved=cfg.is_solved,
        )

        # Behavioral signature
        sig = reactive_semantic_signature(policy, rt, eval_states)

        results.append({
            "branch_specs": specs,
            "branch_names": catalog.spec_names(specs),
            "solved": episode.solved,
            "reward": round(episode.cumulative_reward, 6),
            "steps": episode.total_env_steps,
            "action_trace": [step.action for step in episode.steps],
            "signature": sig,
        })

    return results


def assign_equivalence_classes(results: list[dict]) -> list[dict]:
    """Group results by behavioral signature into equivalence classes."""
    sig_to_class: dict[tuple, int] = {}
    class_members: dict[int, list] = defaultdict(list)
    next_id = 0

    for r in results:
        sig = tuple(r["signature"])
        if sig not in sig_to_class:
            sig_to_class[sig] = next_id
            next_id += 1
        cid = sig_to_class[sig]
        r["equiv_class_id"] = cid
        class_members[cid].append(r)

    classes = []
    for cid in range(next_id):
        members = class_members[cid]
        # Use any() for solved — single-tick signatures are a coarse proxy;
        # policies with the same signature can have different episode outcomes.
        n_solved = sum(1 for m in members if m["solved"])
        # Pick a solving member as the example if any exist
        example = next((m for m in members if m["solved"]), members[0])
        classes.append({
            "id": cid,
            "count": len(members),
            "solved": n_solved > 0,
            "n_solved": n_solved,
            "reward": example["reward"],
            "steps": example["steps"],
            "action_trace": example["action_trace"],
            "signature": example["signature"],
            "example_spec": example["branch_specs"],
            "example_names": example["branch_names"],
        })

    return classes
