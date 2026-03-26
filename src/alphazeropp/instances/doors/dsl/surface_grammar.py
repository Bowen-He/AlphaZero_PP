"""Canonical and relaxed search grammars for DoorsStageDSL(D).

Canonical grammar:  exactly one policy per D (the known-optimal ordering).
Relaxed grammar:    all monotonicity-constrained interleavings of
                    {PickRule(k), MoveRule(k)} for k in 0..K-1, ending
                    with GoalRule.
"""

from __future__ import annotations

import math
from itertools import permutations

from alphazeropp.instances.doors.dsl.surface_dsl import (
    PickRule, MoveRule, GoalRule, SurfacePolicy, SurfaceRule,
)


# ---------------------------------------------------------------------------
# Canonical policy
# ---------------------------------------------------------------------------

def canonical_policy(D: int) -> SurfacePolicy:
    """The unique canonical policy for *D* rooms.

    Structure: PickRule(0), MoveRule(0), ..., PickRule(K-1), MoveRule(K-1), GoalRule
    """
    if D < 1:
        raise ValueError(f"D must be >= 1, got {D}")
    K = D - 1
    rules: list[SurfaceRule] = []
    for k in range(K):
        rules.append(PickRule(k))
        rules.append(MoveRule(k))
    rules.append(GoalRule())
    return SurfacePolicy(tuple(rules))


# ---------------------------------------------------------------------------
# Relaxed grammar — counting
# ---------------------------------------------------------------------------

def count_relaxed_policies(D: int) -> int:
    """Closed-form count of relaxed policies: (2K)! / 2^K.

    Each of K keys contributes a PickRule/MoveRule pair.  All (2K)!
    interleavings are valid except those violating the monotonicity
    constraint PickRule(k) before MoveRule(k), which halves the count
    per key.

    D=1 → K=0 → 1 (just GoalRule).
    D=2 → K=1 → 1.
    D=3 → K=2 → 6.
    D=4 → K=3 → 90.
    """
    if D < 1:
        raise ValueError(f"D must be >= 1, got {D}")
    K = D - 1
    if K == 0:
        return 1
    return math.factorial(2 * K) // (2 ** K)


# ---------------------------------------------------------------------------
# Relaxed grammar — enumeration
# ---------------------------------------------------------------------------

def enumerate_relaxed_policies(
    D: int,
    *,
    max_enumerate: int = 100_000,
) -> list[SurfacePolicy]:
    """Enumerate all valid relaxed policies for *D* rooms.

    Generates all permutations of {PickRule(k), MoveRule(k) : k in 0..K-1}
    filtered by: PickRule(k) appears before MoveRule(k) for each k.
    GoalRule is appended at the end.

    Raises ValueError if the count exceeds *max_enumerate*.
    """
    if D < 1:
        raise ValueError(f"D must be >= 1, got {D}")

    K = D - 1
    if K == 0:
        return [SurfacePolicy((GoalRule(),))]

    count = count_relaxed_policies(D)
    if count > max_enumerate:
        raise ValueError(
            f"Relaxed policy count {count} for D={D} exceeds "
            f"max_enumerate={max_enumerate}"
        )

    # Build the rule pool: (rule_type, key_id) pairs
    # We use tagged tuples for permutation, then convert to rule objects.
    pool: list[tuple[str, int]] = []
    for k in range(K):
        pool.append(("pick", k))
        pool.append(("move", k))

    policies: list[SurfacePolicy] = []
    for perm in permutations(pool):
        if _satisfies_monotonicity(perm, K):
            rules = tuple(_to_rule(tag, kid) for tag, kid in perm) + (GoalRule(),)
            policies.append(SurfacePolicy(rules))

    assert len(policies) == count, f"Expected {count}, got {len(policies)}"
    return policies


def _satisfies_monotonicity(
    perm: tuple[tuple[str, int], ...],
    K: int,
) -> bool:
    """Check that PickRule(k) appears before MoveRule(k) for each k."""
    pick_seen: set[int] = set()
    for tag, kid in perm:
        if tag == "pick":
            pick_seen.add(kid)
        else:  # "move"
            if kid not in pick_seen:
                return False
    return True


def _to_rule(tag: str, kid: int) -> PickRule | MoveRule:
    if tag == "pick":
        return PickRule(kid)
    return MoveRule(kid)


# ---------------------------------------------------------------------------
# Prefix enumeration (for exact oracle values at small D)
# ---------------------------------------------------------------------------

def enumerate_surface_prefixes(
    D: int,
    *,
    max_enumerate: int = 1_000_000,
) -> list[tuple[SurfaceRule, ...]]:
    """All valid partial rule sequences for the strict surface game.

    Includes the empty prefix () and all complete policies.
    Uses BFS over the legal-mask state space.

    The strict game's legal mask:
      PickRule(k): legal iff k not picked and k not moved
      MoveRule(k): legal iff k picked and k not moved
      GoalRule:    legal iff all 2K non-goal rules placed
    """
    if D < 1:
        raise ValueError(f"D must be >= 1, got {D}")

    K = D - 1
    # BFS: each state is (rules_tuple, picked_mask, moved_mask)
    prefixes: list[tuple[SurfaceRule, ...]] = [()]
    frontier = [((), 0, 0)]  # (rules, picked_mask, moved_mask)

    while frontier:
        new_frontier = []
        for rules, picked, moved in frontier:
            n_placed = len(rules)
            # Generate legal successors
            for k in range(K):
                # PickRule(k)
                if not (picked & (1 << k)) and not (moved & (1 << k)):
                    new_rules = rules + (PickRule(k),)
                    new_picked = picked | (1 << k)
                    prefixes.append(new_rules)
                    new_frontier.append((new_rules, new_picked, moved))  # moved unchanged
                # MoveRule(k)
                if (picked & (1 << k)) and not (moved & (1 << k)):
                    new_rules = rules + (MoveRule(k),)
                    new_moved = moved | (1 << k)
                    prefixes.append(new_rules)
                    new_frontier.append((new_rules, picked, new_moved))
            # GoalRule: only if all 2K placed
            if n_placed == 2 * K:
                new_rules = rules + (GoalRule(),)
                prefixes.append(new_rules)
                # Terminal — don't add to frontier

            if len(prefixes) > max_enumerate:
                raise ValueError(
                    f"Prefix count exceeds max_enumerate={max_enumerate} for D={D}"
                )

        frontier = new_frontier

    return prefixes


def count_solving_policies(
    D: int,
    doors_cfg: "DoorsGameConfig",  # type: ignore[name-defined]
) -> int:
    """Count how many complete relaxed policies actually solve the environment.

    Requires: enumerate_relaxed_policies(D), compile_policy, run_policy_episode.
    """
    from alphazeropp.instances.doors.dsl.surface_compiler import compile_policy
    from alphazeropp.instances.doors.dsl.doors_config import doors_initial_state
    from alphazeropp.synthesis.interpreter import run_policy_episode

    policies = enumerate_relaxed_policies(D)
    x0 = doors_initial_state(doors_cfg)
    n_sites = doors_cfg.obs_size()
    solve_count = 0
    for policy in policies:
        prog = compile_policy(policy, doors_cfg)
        env = doors_cfg.make_env(n_sites, frozen_states=[x0])
        result = run_policy_episode(env, prog, x0=x0, is_solved=doors_cfg.is_solved)
        if result.solved:
            solve_count += 1
    return solve_count
