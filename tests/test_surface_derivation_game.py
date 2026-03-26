"""Tests for SurfaceDerivationGame: legal masks, episodes, state management, exact counts."""

from __future__ import annotations

import numpy as np
import pytest

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.instances.doors.dsl.surface_derivation_game import (
    SurfaceDerivationGame, SurfaceDerivationState, SURFACE_TOKEN_IDS,
)
from alphazeropp.instances.doors.dsl.surface_dsl import (
    PickRule, MoveRule, GoalRule, SurfacePolicy,
)
from alphazeropp.instances.doors.dsl.surface_grammar import (
    enumerate_relaxed_policies, enumerate_surface_prefixes,
    count_solving_policies, count_relaxed_policies,
)
from alphazeropp.instances.doors.dsl.surface_compiler import compile_policy
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_game(D: int) -> SurfaceDerivationGame:
    """Helper: build a SurfaceDerivationGame for D rooms."""
    params = compute_doors_derived_params(D, 2)
    cfg = DoorsGameConfig(num_rooms=D, locs_per_room=2, horizon=params["horizon"])
    n_sites = cfg.obs_size()
    leaf_eval = LeafEvaluator(
        n_sites, [doors_initial_state(cfg)], cfg,
        is_solved=cfg.is_solved, metric="solve_rate",
    )
    return SurfaceDerivationGame(num_rooms=D, leaf_evaluator=leaf_eval, doors_cfg=cfg)


@pytest.fixture
def game_d2():
    return _make_game(2)


@pytest.fixture
def game_d3():
    return _make_game(3)


# ---------------------------------------------------------------------------
# TestLegalMask
# ---------------------------------------------------------------------------

class TestLegalMask:
    """Legal mask correctness for the strict surface game."""

    def test_initial_mask_d2(self, game_d2):
        """D=2, K=1: initially only PickRule(0) is legal."""
        game_d2.reset_wrapper()
        mask = game_d2.get_action_mask()
        # Actions: [Pick(0), Move(0), GoalRule]
        assert mask.tolist() == [True, False, False]

    def test_after_pick_d2(self, game_d2):
        """After PickRule(0): MoveRule(0) legal, GoalRule not yet."""
        game_d2.reset_wrapper()
        game_d2.step_wrapper(0)  # PickRule(0)
        mask = game_d2.get_action_mask()
        assert mask.tolist() == [False, True, False]

    def test_after_all_d2(self, game_d2):
        """After Pick+Move: only GoalRule legal."""
        game_d2.reset_wrapper()
        game_d2.step_wrapper(0)  # PickRule(0)
        game_d2.step_wrapper(1)  # MoveRule(0)
        mask = game_d2.get_action_mask()
        assert mask.tolist() == [False, False, True]

    def test_move_before_pick_illegal_d3(self, game_d3):
        """D=3: MoveRule(k) illegal before PickRule(k)."""
        game_d3.reset_wrapper()
        mask = game_d3.get_action_mask()
        K = 2
        # Actions: [Pick(0), Pick(1), Move(0), Move(1), GoalRule]
        # Initially: only Pick(0) and Pick(1) legal
        assert mask[0] is np.True_  # Pick(0)
        assert mask[1] is np.True_  # Pick(1)
        assert mask[K + 0] is np.False_  # Move(0)
        assert mask[K + 1] is np.False_  # Move(1)
        assert mask[2 * K] is np.False_  # GoalRule

    def test_goal_illegal_before_complete(self, game_d3):
        """GoalRule illegal until all 2K rules placed."""
        game_d3.reset_wrapper()
        K = 2
        # Place Pick(0), Move(0), Pick(1) — still missing Move(1)
        game_d3.step_wrapper(0)  # Pick(0)
        game_d3.step_wrapper(K + 0)  # Move(0)
        game_d3.step_wrapper(1)  # Pick(1)
        mask = game_d3.get_action_mask()
        assert mask[2 * K] is np.False_  # GoalRule still illegal
        assert mask[K + 1] is np.True_  # Move(1) is legal

    def test_no_duplicate_pick(self, game_d3):
        """PickRule(k) illegal after already placed."""
        game_d3.reset_wrapper()
        game_d3.step_wrapper(0)  # Pick(0)
        mask = game_d3.get_action_mask()
        assert mask[0] is np.False_  # Pick(0) already placed

    def test_d3_full_canonical_mask_sequence(self, game_d3):
        """Walk through canonical D=3 sequence checking masks at each step."""
        game_d3.reset_wrapper()
        K = 2
        # Step 0: Pick(0)
        mask = game_d3.get_action_mask()
        assert mask[0] and mask[1]  # Pick(0), Pick(1) legal
        assert not mask[K] and not mask[K + 1]  # Moves illegal
        assert not mask[2 * K]  # Goal illegal

        game_d3.step_wrapper(0)  # Pick(0)
        mask = game_d3.get_action_mask()
        assert not mask[0]  # Pick(0) used
        assert mask[1]  # Pick(1) legal
        assert mask[K]  # Move(0) now legal
        assert not mask[K + 1]  # Move(1) still illegal

        game_d3.step_wrapper(K)  # Move(0)
        game_d3.step_wrapper(1)  # Pick(1)
        game_d3.step_wrapper(K + 1)  # Move(1)
        mask = game_d3.get_action_mask()
        assert mask.tolist() == [False, False, False, False, True]  # Only GoalRule


# ---------------------------------------------------------------------------
# TestEpisode
# ---------------------------------------------------------------------------

class TestEpisode:
    """Full episode tests."""

    def test_d2_canonical_episode(self, game_d2):
        """Canonical D=2 episode: solves with reward 1.0."""
        game_d2.reset_wrapper()
        game_d2.step_wrapper(0)  # Pick(0)
        game_d2.step_wrapper(1)  # Move(0)
        obs, reward, term, trunc, info = game_d2.step_wrapper(2)  # GoalRule
        assert term is True
        assert trunc is False
        assert reward == 1.0
        assert info["policy"].pretty() == "PickRule(0) >> MoveRule(0) >> GoalRule"

    def test_d3_canonical_episode(self, game_d3):
        """Canonical D=3 episode: solves with reward 1.0."""
        game_d3.reset_wrapper()
        K = 2
        game_d3.step_wrapper(0)      # Pick(0)
        game_d3.step_wrapper(K)      # Move(0)
        game_d3.step_wrapper(1)      # Pick(1)
        game_d3.step_wrapper(K + 1)  # Move(1)
        obs, reward, term, trunc, info = game_d3.step_wrapper(2 * K)  # GoalRule
        assert term is True
        assert reward == 1.0
        assert game_d3.step_count == 5

    def test_d2_obs_encoding(self, game_d2):
        """Verify observation shape and content."""
        game_d2.reset_wrapper()
        obs = game_d2.obs
        assert obs.shape == (6,)  # 2 * (2*1+1) = 6
        assert obs.dtype == np.float32
        # All PAD initially
        np.testing.assert_array_equal(obs, [0, 0, 0, 0, 0, 0])

        game_d2.step_wrapper(0)  # Pick(0)
        obs = game_d2.obs
        assert obs[0] == SURFACE_TOKEN_IDS["PICK"]
        assert obs[1] == 0.0  # key index 0

    def test_terminal_info_has_program(self, game_d2):
        """Terminal info dict contains compiled program and policy."""
        game_d2.reset_wrapper()
        game_d2.step_wrapper(0)
        game_d2.step_wrapper(1)
        _, _, _, _, info = game_d2.step_wrapper(2)
        assert "program" in info
        assert "policy" in info
        assert "leaf_value" in info

    def test_non_terminal_reward_zero(self, game_d3):
        """Non-terminal steps have reward 0."""
        game_d3.reset_wrapper()
        _, r, term, _, _ = game_d3.step_wrapper(0)  # Pick(0)
        assert r == 0.0
        assert term is False


# ---------------------------------------------------------------------------
# TestStateManagement
# ---------------------------------------------------------------------------

class TestStateManagement:
    """Stash/unstash and clone correctness."""

    def test_stash_unstash_roundtrip(self, game_d2):
        game_d2.reset_wrapper()
        game_d2.step_wrapper(0)  # Pick(0)
        stash = game_d2.stash_state()

        # Advance further
        game_d2.step_wrapper(1)  # Move(0)

        # Restore
        game_d2.unstash_state(stash)
        mask = game_d2.get_action_mask()
        assert mask.tolist() == [False, True, False]  # After Pick(0)

    def test_clone_independence(self, game_d2):
        game_d2.reset_wrapper()
        game_d2.step_wrapper(0)

        clone = game_d2.clone()
        clone.step_wrapper(1)  # Move(0) on clone

        # Original should still be at step 1 (after Pick(0))
        assert game_d2.step_count == 1
        assert clone.step_count == 2

    def test_hashable_obs_changes_per_step(self, game_d2):
        game_d2.reset_wrapper()
        h0 = game_d2.hashable_obs
        game_d2.step_wrapper(0)
        h1 = game_d2.hashable_obs
        assert h0 != h1

    def test_hashable_obs_is_hashable(self, game_d2):
        game_d2.reset_wrapper()
        h = game_d2.hashable_obs
        # Should be usable as dict key
        d = {h: 1}
        assert d[h] == 1


# ---------------------------------------------------------------------------
# TestExactCounts
# ---------------------------------------------------------------------------

class TestExactCounts:
    """Exact policy counts and solve counts."""

    def test_d2_exactly_1_policy(self):
        assert count_relaxed_policies(2) == 1

    def test_d3_exactly_6_policies(self):
        assert count_relaxed_policies(3) == 6

    def test_d5_exactly_2520_policies(self):
        assert count_relaxed_policies(5) == 2520

    def test_d2_solve_count(self):
        cfg = DoorsGameConfig(num_rooms=2, locs_per_room=2)
        assert count_solving_policies(2, cfg) == 1

    def test_d3_solve_count(self):
        params = compute_doors_derived_params(3, 2)
        cfg = DoorsGameConfig(num_rooms=3, locs_per_room=2, horizon=params["horizon"])
        assert count_solving_policies(3, cfg) == 3

    @pytest.mark.slow
    def test_d5_solve_count(self):
        params = compute_doors_derived_params(5, 2)
        cfg = DoorsGameConfig(num_rooms=5, locs_per_room=2, horizon=params["horizon"])
        assert count_solving_policies(5, cfg) == 105

    def test_d2_prefix_count(self):
        prefixes = enumerate_surface_prefixes(2)
        # D=2, K=1: () → Pick(0) → Move(0) → GoalRule = 4 prefixes
        assert len(prefixes) == 4

    def test_d3_prefix_count(self):
        prefixes = enumerate_surface_prefixes(3)
        assert len(prefixes) == 25

    def test_prefix_complete_matches_relaxed(self):
        """Complete prefixes must match relaxed policy count."""
        for D in [2, 3, 4]:
            prefixes = enumerate_surface_prefixes(D)
            complete = [p for p in prefixes if len(p) > 0 and isinstance(p[-1], GoalRule)]
            relaxed = enumerate_relaxed_policies(D)
            assert len(complete) == len(relaxed), f"D={D}: {len(complete)} != {len(relaxed)}"


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and D=1."""

    def test_d1_game(self):
        """D=1: only GoalRule, single-step episode."""
        game = _make_game(1)
        game.reset_wrapper()
        mask = game.get_action_mask()
        assert mask.tolist() == [True]  # Only GoalRule
        _, reward, term, _, info = game.step_wrapper(0)
        assert term is True
        assert "policy" in info
