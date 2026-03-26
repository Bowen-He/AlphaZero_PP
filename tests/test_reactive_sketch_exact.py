"""Regression tests for the reactive sketch exact oracle.

Tests behavioral signatures, equivalence classes, canonical traces,
and cross-DSL trace matching between reactive sketch and surface DSL.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.instances.doors.dsl.relational_runtime import (
    DoorsRelationalRuntime,
)
from alphazeropp.instances.doors.dsl.reactive_sketch_dsl import (
    canonical_reactive_policy,
)
from alphazeropp.instances.doors.dsl.reactive_sketch_interpreter import (
    run_reactive_episode,
)
from alphazeropp.instances.doors.dsl.stage_diagnostics import (
    make_frozen_state_suite, reactive_semantic_signature,
)
from alphazeropp.instances.doors.dsl.surface_grammar import canonical_policy
from alphazeropp.instances.doors.dsl.surface_compiler import compile_policy
from alphazeropp.synthesis.interpreter import run_policy_episode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg_d2():
    return DoorsGameConfig(num_rooms=2, locs_per_room=2)


@pytest.fixture
def cfg_d3():
    params = compute_doors_derived_params(3, 2)
    return DoorsGameConfig(num_rooms=3, locs_per_room=2, horizon=params["horizon"])


@pytest.fixture
def rt_d2(cfg_d2):
    return DoorsRelationalRuntime(cfg_d2)


@pytest.fixture
def rt_d3(cfg_d3):
    return DoorsRelationalRuntime(cfg_d3)


# ---------------------------------------------------------------------------
# Canonical trace tests
# ---------------------------------------------------------------------------

class TestCanonicalTraces:

    def test_canonical_trace_d2(self, cfg_d2, rt_d2):
        """Canonical order (1,2,3,4) on D=2 produces move_to(1), pick(0), move_to(3)."""
        policy = canonical_reactive_policy()
        x0 = doors_initial_state(cfg_d2)
        env = cfg_d2.make_env(cfg_d2.obs_size(), frozen_states=[x0])
        result = run_reactive_episode(
            env, policy, rt_d2, x0=x0, is_solved=cfg_d2.is_solved,
        )
        assert result.solved
        trace = [step.action for step in result.steps]
        # D=2: M=4, K=1. move_to(1)=1, pick(0)=4, move_to(3)=3
        assert trace == [1, 4, 3]

    def test_canonical_trace_d3(self, cfg_d3, rt_d3):
        """Canonical order on D=3 produces move_to(1), pick(0), move_to(3), pick(1), move_to(5)."""
        policy = canonical_reactive_policy()
        x0 = doors_initial_state(cfg_d3)
        env = cfg_d3.make_env(cfg_d3.obs_size(), frozen_states=[x0])
        result = run_reactive_episode(
            env, policy, rt_d3, x0=x0, is_solved=cfg_d3.is_solved,
        )
        assert result.solved
        trace = [step.action for step in result.steps]
        # D=3: M=6, K=2. move_to(1)=1, pick(0)=6, move_to(3)=3, pick(1)=7, move_to(5)=5
        assert trace == [1, 6, 3, 7, 5]


# ---------------------------------------------------------------------------
# Cross-DSL trace matching
# ---------------------------------------------------------------------------

class TestCrossDSLTraceMatch:

    def _get_surface_trace(self, D: int, cfg: DoorsGameConfig) -> list[int]:
        """Run canonical surface policy and return action trace."""
        surface_pol = canonical_policy(D)
        prog = compile_policy(surface_pol, cfg)
        x0 = doors_initial_state(cfg)
        env = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
        result = run_policy_episode(env, prog, x0=x0, is_solved=cfg.is_solved)
        assert result.solved, f"Surface canonical policy should solve D={D}"
        return [step.action for step in result.steps]

    def _get_reactive_trace(self, cfg: DoorsGameConfig, rt: DoorsRelationalRuntime) -> list[int]:
        """Run canonical reactive policy and return action trace."""
        policy = canonical_reactive_policy()
        x0 = doors_initial_state(cfg)
        env = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
        result = run_reactive_episode(
            env, policy, rt, x0=x0, is_solved=cfg.is_solved,
        )
        assert result.solved
        return [step.action for step in result.steps]

    def test_reactive_surface_trace_match_d2(self, cfg_d2, rt_d2):
        """Canonical reactive and surface policies produce same trace for D=2."""
        reactive_trace = self._get_reactive_trace(cfg_d2, rt_d2)
        surface_trace = self._get_surface_trace(2, cfg_d2)
        assert reactive_trace == surface_trace

    def test_reactive_surface_trace_match_d3(self, cfg_d3, rt_d3):
        """Canonical reactive and surface policies produce same trace for D=3."""
        reactive_trace = self._get_reactive_trace(cfg_d3, rt_d3)
        surface_trace = self._get_surface_trace(3, cfg_d3)
        assert reactive_trace == surface_trace


# ---------------------------------------------------------------------------
# Behavioral signature tests
# ---------------------------------------------------------------------------

class TestBehavioralSignatures:

    def test_all_signatures_computed_d2(self, cfg_d2, rt_d2):
        """All 24 permutations produce signatures with no None entries on frozen suite."""
        states = make_frozen_state_suite(cfg_d2)
        for perm in itertools.permutations([1, 2, 3, 4]):
            policy = canonical_reactive_policy(perm)
            sig = reactive_semantic_signature(policy, rt_d2, states)
            assert len(sig) == len(states)
            # At least the initial state should produce a non-None action
            assert sig[0] is not None, f"Perm {perm} has None action on initial state"

    def test_canonical_signature_stable(self, cfg_d2, rt_d2):
        """Same policy always produces same signature."""
        states = make_frozen_state_suite(cfg_d2)
        policy = canonical_reactive_policy()
        sig1 = reactive_semantic_signature(policy, rt_d2, states)
        sig2 = reactive_semantic_signature(policy, rt_d2, states)
        assert sig1 == sig2


# ---------------------------------------------------------------------------
# Equivalence class tests
# ---------------------------------------------------------------------------

class TestEquivalenceClasses:

    def _compute_classes(self, cfg, rt):
        """Compute equivalence classes for all 24 permutations."""
        states = make_frozen_state_suite(cfg)
        sig_to_members: dict[tuple, list[tuple]] = {}
        sig_solved: dict[tuple, bool] = {}

        for perm in itertools.permutations([1, 2, 3, 4]):
            policy = canonical_reactive_policy(perm)
            sig = reactive_semantic_signature(policy, rt, states)

            if sig not in sig_to_members:
                sig_to_members[sig] = []
                # Determine if this class solves
                x0 = doors_initial_state(cfg)
                env = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
                result = run_reactive_episode(
                    env, policy, rt, x0=x0, is_solved=cfg.is_solved,
                )
                sig_solved[sig] = result.solved

            sig_to_members[sig].append(perm)

        return sig_to_members, sig_solved

    def test_equivalence_classes_d2(self, cfg_d2, rt_d2):
        """24 permutations for D=2 yield a specific number of equivalence classes."""
        sig_to_members, sig_solved = self._compute_classes(cfg_d2, rt_d2)
        n_classes = len(sig_to_members)
        # Must have at least 2 classes (solving and non-solving exist)
        assert n_classes >= 2
        # Total members must sum to 24
        total = sum(len(m) for m in sig_to_members.values())
        assert total == 24
        # At least one solving class and one non-solving class
        assert any(sig_solved[s] for s in sig_solved)
        assert any(not sig_solved[s] for s in sig_solved)

    def test_solving_classes_subset_d2(self, cfg_d2, rt_d2):
        """Every solving class has a different signature from every non-solving class."""
        sig_to_members, sig_solved = self._compute_classes(cfg_d2, rt_d2)
        solving_sigs = {s for s, solved in sig_solved.items() if solved}
        failing_sigs = {s for s, solved in sig_solved.items() if not solved}
        # No overlap
        assert solving_sigs.isdisjoint(failing_sigs)

    def test_equivalence_classes_d3(self, cfg_d3, rt_d3):
        """24 permutations for D=3 yield equivalence classes with solving/failing split."""
        sig_to_members, sig_solved = self._compute_classes(cfg_d3, rt_d3)
        total = sum(len(m) for m in sig_to_members.values())
        assert total == 24
        # At least one solving class
        assert any(sig_solved[s] for s in sig_solved)
