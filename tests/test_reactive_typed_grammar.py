"""Tests for the reactive typed grammar: catalog, enumeration, derivation game.

Covers: ReactiveBranchCatalog, legal matrix masking, enumeration consistency,
solver existence, derivation game lifecycle, and acceptance criteria.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.instances.doors.dsl.relational_runtime import (
    DoorsRelationalRuntime,
)
from alphazeropp.instances.doors.dsl.reactive_branch_catalog import (
    ReactiveBranchCatalog, known_map_catalog,
    PRED_PICKABLE, PRED_KNOWN_LOC, PRED_EXISTS_FRONTIER, PRED_REACHABLE,
    PRED_GOAL_REACHED, PRED_TRUE, PRED_EXISTS_UNSEARCHED,
    ACT_PICK, ACT_GOTO_KEY, ACT_GOTO_GOAL, ACT_GOTO_ENTRANCE, ACT_NOOP,
)
from alphazeropp.instances.doors.dsl.reactive_typed_grammar import (
    count_reactive_typed_policies,
    enumerate_reactive_typed_policies,
    evaluate_reactive_typed_policies,
    assign_equivalence_classes,
)
from alphazeropp.instances.doors.dsl.reactive_derivation_game import (
    ReactiveDerivationGame, ReactiveDerivationState,
)
from alphazeropp.instances.doors.dsl.reactive_sketch_interpreter import (
    run_reactive_episode,
)
from alphazeropp.instances.doors.dsl.reactive_sketch_dsl import (
    Check, Do, Sequence, Fallback, WhileNot, GoalReachedP, TrueP, NoopAction,
)
from alphazeropp.instances.doors.dsl.stage_diagnostics import (
    make_frozen_state_suite,
)


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
def typed_catalog():
    return known_map_catalog(mode="typed")


@pytest.fixture
def raw_catalog():
    return known_map_catalog(mode="raw")


# ---------------------------------------------------------------------------
# TestCatalog
# ---------------------------------------------------------------------------

class TestCatalog:

    def test_known_map_catalog_sizes(self, typed_catalog):
        """Catalog has 7 predicates and 5 actions."""
        assert typed_catalog.n_predicates == 7
        assert typed_catalog.n_actions == 5

    def test_legal_matrix_shape(self, typed_catalog):
        """Legal matrix is (7, 5) bool array."""
        assert typed_catalog.legal_matrix.shape == (7, 5)
        assert typed_catalog.legal_matrix.dtype == bool

    def test_typed_legal_pair_count(self, typed_catalog):
        """Typed mode has exactly 8 legal pairs."""
        assert typed_catalog.n_legal_pairs() == 8

    def test_raw_legal_pair_count(self, raw_catalog):
        """Raw mode has exactly 24 legal pairs."""
        assert raw_catalog.n_legal_pairs() == 24

    def test_goal_reached_fully_masked(self, typed_catalog, raw_catalog):
        """GoalReached row is fully masked in both modes."""
        assert not typed_catalog.legal_matrix[PRED_GOAL_REACHED].any()
        assert not raw_catalog.legal_matrix[PRED_GOAL_REACHED].any()

    def test_true_noop_masked(self, typed_catalog, raw_catalog):
        """True + Noop is illegal in both modes."""
        assert not typed_catalog.legal_matrix[PRED_TRUE, ACT_NOOP]
        assert not raw_catalog.legal_matrix[PRED_TRUE, ACT_NOOP]

    def test_exists_unsearched_masked_known_map(self, typed_catalog, raw_catalog):
        """ExistsUnsearched is fully masked in known_map mode."""
        assert not typed_catalog.legal_matrix[PRED_EXISTS_UNSEARCHED].any()
        assert not raw_catalog.legal_matrix[PRED_EXISTS_UNSEARCHED].any()

    def test_build_branch_structure(self, typed_catalog):
        """build_branch returns Sequence(Check, Do)."""
        branch = typed_catalog.build_branch(PRED_PICKABLE, ACT_PICK)
        assert isinstance(branch, Sequence)
        assert len(branch.children) == 2
        assert isinstance(branch.children[0], Check)
        assert isinstance(branch.children[1], Do)

    def test_build_policy_structure(self, typed_catalog):
        """build_policy returns WhileNot(GoalReachedP, Fallback(...))."""
        specs = (
            (PRED_PICKABLE, ACT_PICK),
            (PRED_KNOWN_LOC, ACT_GOTO_KEY),
            (PRED_REACHABLE, ACT_GOTO_GOAL),
            (PRED_EXISTS_FRONTIER, ACT_GOTO_ENTRANCE),
        )
        policy = typed_catalog.build_policy(specs)
        assert isinstance(policy, WhileNot)
        assert isinstance(policy.predicate, GoalReachedP)
        assert isinstance(policy.child, Fallback)
        assert len(policy.child.children) == 4

    def test_spec_names(self, typed_catalog):
        """spec_names produces human-readable branch descriptions."""
        specs = ((PRED_PICKABLE, ACT_PICK), (PRED_TRUE, ACT_GOTO_GOAL))
        names = typed_catalog.spec_names(specs)
        assert names == ["Pickable→Pick", "True→GoToGoal"]


# ---------------------------------------------------------------------------
# TestEnumeration
# ---------------------------------------------------------------------------

class TestEnumeration:

    def test_count_matches_enumerate_length(self, typed_catalog):
        """count == len(enumerate) for typed mode."""
        count = count_reactive_typed_policies(4, typed_catalog)
        specs = enumerate_reactive_typed_policies(4, typed_catalog)
        assert count == len(specs)

    def test_typed_d2_count(self, typed_catalog):
        """Typed mode with 4 branches: 8^4 = 4096."""
        assert count_reactive_typed_policies(4, typed_catalog) == 8 ** 4

    def test_raw_d2_count(self, raw_catalog):
        """Raw mode with 4 branches: 24^4 = 331776."""
        assert count_reactive_typed_policies(4, raw_catalog) == 24 ** 4

    def test_canonical_in_enumeration(self, typed_catalog):
        """The canonical 4-branch policy appears in typed enumeration."""
        canonical_spec = (
            (PRED_PICKABLE, ACT_PICK),
            (PRED_KNOWN_LOC, ACT_GOTO_KEY),
            (PRED_REACHABLE, ACT_GOTO_GOAL),
            (PRED_EXISTS_FRONTIER, ACT_GOTO_ENTRANCE),
        )
        specs = enumerate_reactive_typed_policies(4, typed_catalog)
        assert canonical_spec in specs

    def test_at_least_one_solver_d2(self, typed_catalog, cfg_d2):
        """At least one typed policy solves D=2."""
        rt = DoorsRelationalRuntime(cfg_d2)
        states = make_frozen_state_suite(cfg_d2)
        specs = enumerate_reactive_typed_policies(4, typed_catalog)
        results = evaluate_reactive_typed_policies(
            specs, typed_catalog, cfg_d2, rt, states,
        )
        solve_count = sum(1 for r in results if r["solved"])
        assert solve_count > 0

    def test_typed_fewer_than_raw(self, typed_catalog, raw_catalog):
        """Typed mode produces fewer policies than raw mode."""
        typed_count = count_reactive_typed_policies(4, typed_catalog)
        raw_count = count_reactive_typed_policies(4, raw_catalog)
        assert typed_count < raw_count

    def test_enumerate_raises_on_overflow(self, raw_catalog):
        """Enumeration raises ValueError when count exceeds max_enumerate."""
        with pytest.raises(ValueError, match="Too many"):
            enumerate_reactive_typed_policies(
                4, raw_catalog, max_enumerate=100,
            )


# ---------------------------------------------------------------------------
# TestDerivationGame
# ---------------------------------------------------------------------------

class TestDerivationGame:

    def test_episode_length(self, typed_catalog, cfg_d2):
        """Episode is exactly 2*N steps."""
        game = ReactiveDerivationGame(cfg_d2, typed_catalog, n_branches=4)
        obs, _ = game.reset()
        steps = 0
        terminated = False

        while not terminated:
            mask = game.get_action_mask()
            action = int(np.where(mask)[0][0])
            obs, reward, terminated, truncated, info = game.step(action)
            steps += 1

        assert steps == 8  # 2 * 4

    def test_legal_mask_alternation(self, typed_catalog, cfg_d2):
        """Even steps have predicate mask, odd steps have action mask."""
        game = ReactiveDerivationGame(cfg_d2, typed_catalog, n_branches=2)
        game.reset()

        # Step 0: predicate step
        mask0 = game.get_action_mask()
        # Predicates with legal actions should be enabled
        assert mask0[:typed_catalog.n_predicates].any()

        # Choose first legal predicate
        pred_idx = int(np.where(mask0)[0][0])
        game.step(pred_idx)

        # Step 1: action step
        mask1 = game.get_action_mask()
        # Should only have actions legal for the chosen predicate
        legal_actions = typed_catalog.legal_actions_for_predicate(pred_idx)
        for a in range(typed_catalog.n_actions):
            assert mask1[a] == (a in legal_actions)

    def test_action_mask_respects_compatibility(self, typed_catalog, cfg_d2):
        """After choosing Pickable, only Pick is legal in typed mode."""
        game = ReactiveDerivationGame(cfg_d2, typed_catalog, n_branches=2)
        game.reset()

        # Choose Pickable predicate
        game.step(PRED_PICKABLE)

        mask = game.get_action_mask()
        # Only Pick should be legal
        assert mask[ACT_PICK] == True
        assert mask[ACT_GOTO_KEY] == False
        assert mask[ACT_GOTO_GOAL] == False
        assert mask[ACT_GOTO_ENTRANCE] == False
        assert mask[ACT_NOOP] == False

    def test_terminal_reward_matches_direct(self, typed_catalog, cfg_d2):
        """Game reward equals direct run_reactive_episode reward."""
        game = ReactiveDerivationGame(cfg_d2, typed_catalog, n_branches=4)
        game.reset()

        # Play the canonical policy
        choices = [
            PRED_PICKABLE, ACT_PICK,
            PRED_KNOWN_LOC, ACT_GOTO_KEY,
            PRED_REACHABLE, ACT_GOTO_GOAL,
            PRED_EXISTS_FRONTIER, ACT_GOTO_ENTRANCE,
        ]
        for i, action in enumerate(choices):
            obs, reward, terminated, truncated, info = game.step(action)

        assert terminated
        game_reward = reward

        # Direct evaluation
        specs = (
            (PRED_PICKABLE, ACT_PICK),
            (PRED_KNOWN_LOC, ACT_GOTO_KEY),
            (PRED_REACHABLE, ACT_GOTO_GOAL),
            (PRED_EXISTS_FRONTIER, ACT_GOTO_ENTRANCE),
        )
        policy = typed_catalog.build_policy(specs)
        rt = DoorsRelationalRuntime(cfg_d2)
        x0 = doors_initial_state(cfg_d2)
        env = cfg_d2.make_env(cfg_d2.obs_size(), frozen_states=[x0])
        result = run_reactive_episode(
            env, policy, rt, x0=x0, is_solved=cfg_d2.is_solved,
        )

        assert game_reward == pytest.approx(result.cumulative_reward)

    def test_stash_unstash_roundtrip(self, typed_catalog, cfg_d2):
        """State can be stashed and unstashed without loss."""
        game = ReactiveDerivationGame(cfg_d2, typed_catalog, n_branches=2)
        game.reset()
        game.step(PRED_PICKABLE)

        stashed = game.stash_state()
        obs_before = game._encode_obs().copy()
        mask_before = game.get_action_mask().copy()

        # Modify game state
        game.step(ACT_PICK)

        # Restore
        game.unstash_state(stashed)
        obs_after = game._encode_obs()
        mask_after = game.get_action_mask()

        np.testing.assert_array_equal(obs_before, obs_after)
        np.testing.assert_array_equal(mask_before, mask_after)

    def test_clone_independence(self, typed_catalog, cfg_d2):
        """Mutations to clone don't affect original."""
        game = ReactiveDerivationGame(cfg_d2, typed_catalog, n_branches=2)
        game.reset()
        game.step(PRED_PICKABLE)

        clone = game.clone()
        obs_original = game._encode_obs().copy()

        # Advance clone
        clone.step(ACT_PICK)

        # Original unchanged
        obs_after = game._encode_obs()
        np.testing.assert_array_equal(obs_original, obs_after)

    def test_hashable_obs_changes_per_step(self, typed_catalog, cfg_d2):
        """hashable_obs is different after each step."""
        game = ReactiveDerivationGame(cfg_d2, typed_catalog, n_branches=2)
        game.reset()
        h0 = game.hashable_obs
        game.step(PRED_PICKABLE)
        h1 = game.hashable_obs
        game.step(ACT_PICK)
        h2 = game.hashable_obs

        assert h0 != h1
        assert h1 != h2
        assert h0 != h2


# ---------------------------------------------------------------------------
# TestAcceptanceCriteria
# ---------------------------------------------------------------------------

class TestAcceptanceCriteria:

    def test_typed_masks_absurd_pairs(self, typed_catalog):
        """GoalReached→Pick is illegal in typed mode."""
        assert not typed_catalog.legal_matrix[PRED_GOAL_REACHED, ACT_PICK]

    def test_typed_preserves_solver_d2(self, typed_catalog, cfg_d2):
        """Typed mode finds at least one solver for D=2."""
        rt = DoorsRelationalRuntime(cfg_d2)
        states = make_frozen_state_suite(cfg_d2)
        specs = enumerate_reactive_typed_policies(4, typed_catalog)
        results = evaluate_reactive_typed_policies(
            specs, typed_catalog, cfg_d2, rt, states,
        )
        assert any(r["solved"] for r in results)

    def test_typed_preserves_solver_d3(self, typed_catalog, cfg_d3):
        """Typed mode finds at least one solver for D=3."""
        rt = DoorsRelationalRuntime(cfg_d3)
        states = make_frozen_state_suite(cfg_d3)
        specs = enumerate_reactive_typed_policies(4, typed_catalog)
        results = evaluate_reactive_typed_policies(
            specs, typed_catalog, cfg_d3, rt, states,
        )
        assert any(r["solved"] for r in results)

    def test_typed_fewer_unique_than_raw(self, typed_catalog, raw_catalog, cfg_d2):
        """Typed has fewer semantically distinct policies than raw,
        while both preserve at least one solver."""
        rt = DoorsRelationalRuntime(cfg_d2)
        states = make_frozen_state_suite(cfg_d2)

        # Typed
        typed_specs = enumerate_reactive_typed_policies(4, typed_catalog)
        typed_results = evaluate_reactive_typed_policies(
            typed_specs, typed_catalog, cfg_d2, rt, states,
        )
        typed_sigs = {tuple(r["signature"]) for r in typed_results}
        typed_solves = any(r["solved"] for r in typed_results)

        # Raw
        raw_specs = enumerate_reactive_typed_policies(4, raw_catalog)
        raw_results = evaluate_reactive_typed_policies(
            raw_specs, raw_catalog, cfg_d2, rt, states,
        )
        raw_sigs = {tuple(r["signature"]) for r in raw_results}
        raw_solves = any(r["solved"] for r in raw_results)

        assert typed_solves, "Typed mode must find at least one solver"
        assert raw_solves, "Raw mode must find at least one solver"
        assert len(typed_sigs) < len(raw_sigs), (
            f"Typed distinct ({len(typed_sigs)}) should be < "
            f"raw distinct ({len(raw_sigs)})"
        )
