"""Tests for StageDerivationGame — MCTS-compatible hole-filling game."""

from __future__ import annotations

import numpy as np
import pytest

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state,
)
from alphazeropp.instances.doors.dsl.surface_dsl import (
    PickReady, NeedKey, Pick, MoveToKey, MoveToGoal,
)
from alphazeropp.instances.doors.dsl.stage_grammar import (
    enumerate_guards, enumerate_actions,
)
from alphazeropp.instances.doors.dsl.stage_search_cost import SearchCostModel
from alphazeropp.instances.doors.dsl.stage_hole_selection import (
    LeftmostPolicy, StructureFirstPolicy,
)
from alphazeropp.instances.doors.dsl.stage_derivation_game import (
    StageDerivationGame, StageDerivationState, STAGE_TOKEN_IDS,
)
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_game(
    D: int = 2,
    max_stages: int | None = None,
    max_guard_depth: int = 0,
    policy: str = "leftmost",
) -> StageDerivationGame:
    cfg = DoorsGameConfig(num_rooms=D, locs_per_room=2)
    K = D - 1
    if max_stages is None:
        max_stages = 2 * K + 1
    cost_model = SearchCostModel(max_stages=max_stages, max_guard_depth=max_guard_depth)
    n_sites = cfg.obs_size()
    x0 = doors_initial_state(cfg)
    leaf_eval = LeafEvaluator(
        n_sites, [x0], cfg,
        metric="avg_reward", is_solved=cfg.is_solved,
    )
    hole_policy = LeftmostPolicy() if policy == "leftmost" else StructureFirstPolicy()
    return StageDerivationGame(cfg, cost_model, leaf_eval, hole_policy)


@pytest.fixture
def game_d2():
    return _make_game(D=2)


@pytest.fixture
def game_d3():
    return _make_game(D=3)


# ---------------------------------------------------------------------------
# TestActionVocabulary
# ---------------------------------------------------------------------------

class TestActionVocabulary:
    def test_action_space_size_d2(self):
        game = _make_game(D=2, max_guard_depth=0)
        # D=2: 6 guard atoms + 3 actions + 2 structure = 11
        assert game.action_space.n == 11

    def test_action_space_size_d3(self):
        game = _make_game(D=3, max_guard_depth=0)
        # D=3: 11 guard atoms + 5 actions + 2 structure = 18
        assert game.action_space.n == 18

    def test_guard_action_counts(self, game_d2):
        assert game_d2._n_guards == 6   # AtKeyLoc(0), KeyAvail(0), PickReady(0), NeedKey(0), RoomLocked(0), RoomLocked(1)
        assert game_d2._n_actions == 3   # Pick(0), MoveToKey(0), MoveToGoal

    def test_action_filling_roundtrip(self, game_d2):
        """Action→filling→action should be identity for all actions."""
        from alphazeropp.instances.doors.dsl.stage_partial import (
            StructureHole, GuardHoleRef, ActionHoleRef,
        )
        for action_idx in range(game_d2.action_space.n):
            cat, filling = game_d2._action_to_filling(action_idx)
            if cat == "structure":
                hole = StructureHole()
            elif cat == "guard":
                hole = GuardHoleRef(0)
            else:
                hole = ActionHoleRef(0)
            assert game_d2._filling_to_action(hole, filling) == action_idx


# ---------------------------------------------------------------------------
# TestLegalMask
# ---------------------------------------------------------------------------

class TestLegalMask:
    def test_initial_mask_structure_only(self, game_d2):
        game_d2.reset_wrapper()
        mask = game_d2.get_action_mask()
        # Only structure actions (AddStage=0, Finalize=1)
        assert mask[0] == True   # AddStage
        assert mask[1] == True   # Finalize
        assert mask[2:].sum() == 0  # no guard/action

    def test_after_addstage_leftmost_guards(self):
        game = _make_game(D=2, policy="leftmost")
        game.reset_wrapper()
        # Step 0: AddStage
        game.step_wrapper(0)
        mask = game.get_action_mask()
        # Leftmost: after AddStage, current hole is GuardHoleRef(0)
        assert mask[0] == False  # no structure
        assert mask[1] == False
        assert mask[2:2 + game._n_guards].all()  # all guards legal
        assert mask[2 + game._n_guards:].sum() == 0  # no actions yet

    def test_after_guard_fill_actions(self):
        game = _make_game(D=2, policy="leftmost")
        game.reset_wrapper()
        game.step_wrapper(0)  # AddStage
        game.step_wrapper(2)  # fill guard with first guard atom
        mask = game.get_action_mask()
        # Now ActionHoleRef(0) is current
        assert mask[:2].sum() == 0  # no structure
        assert mask[2:2 + game._n_guards].sum() == 0  # no guards
        assert mask[2 + game._n_guards:].all()  # all actions legal

    def test_max_stages_masks_addstage(self):
        game = _make_game(D=2, max_stages=1, policy="leftmost")
        game.reset_wrapper()
        # AddStage (now at 1 stage == max_stages)
        game.step_wrapper(0)
        # Fill guard + action to get back to structure
        game.step_wrapper(2)  # guard
        game.step_wrapper(2 + game._n_guards)  # action
        # Now at StructureHole, but max_stages reached
        mask = game.get_action_mask()
        assert mask[0] == False  # AddStage masked
        assert mask[1] == True   # Finalize still legal

    def test_structure_first_all_structure_initially(self):
        game = _make_game(D=2, policy="structure_first")
        game.reset_wrapper()
        mask = game.get_action_mask()
        assert mask[0] == True   # AddStage
        assert mask[1] == True   # Finalize
        assert mask[2:].sum() == 0

    def test_structure_first_stays_structure_after_addstage(self):
        game = _make_game(D=2, policy="structure_first")
        game.reset_wrapper()
        game.step_wrapper(0)  # AddStage
        mask = game.get_action_mask()
        # Structure-first: still in structure phase
        assert mask[0] == True  # AddStage (stages < max)
        assert mask[1] == True  # Finalize
        assert mask[2:].sum() == 0


# ---------------------------------------------------------------------------
# TestEpisode
# ---------------------------------------------------------------------------

class TestEpisode:
    def test_d2_zero_stage_episode(self, game_d2):
        """Finalize immediately → 1-step episode, Default(MoveToGoal)."""
        obs, info = game_d2.reset_wrapper()
        obs, reward, term, trunc, info = game_d2.step_wrapper(1)  # Finalize
        assert term
        assert not trunc
        assert "program" in info
        assert "stage_program" in info
        assert info["leaf_value"] is not None

    def test_d2_one_stage_episode(self):
        """AddStage → guard → action → Finalize."""
        game = _make_game(D=2, policy="leftmost")
        game.reset_wrapper()

        # Find action indices for PickReady(0), Pick(0)
        guard_idx = game._guard_to_idx[PickReady(0)]
        action_idx = game._action_to_idx[Pick(0)]

        game.step_wrapper(0)  # AddStage
        game.step_wrapper(2 + guard_idx)  # PickReady(0)
        game.step_wrapper(2 + game._n_guards + action_idx)  # Pick(0)

        # Now at StructureHole
        obs, reward, term, trunc, info = game.step_wrapper(1)  # Finalize
        assert term
        assert info["stage_program"].num_stages == 1

    def test_d2_canonical_solver(self):
        """Build canonical 2-stage solver: PickReady(0)→Pick(0), NeedKey(0)→MoveToKey(0)."""
        game = _make_game(D=2, max_stages=3, policy="leftmost")
        game.reset_wrapper()

        gi_pr = game._guard_to_idx[PickReady(0)]
        gi_nk = game._guard_to_idx[NeedKey(0)]
        ai_pk = game._action_to_idx[Pick(0)]
        ai_mk = game._action_to_idx[MoveToKey(0)]

        game.step_wrapper(0)  # AddStage 0
        game.step_wrapper(2 + gi_pr)  # PickReady(0)
        game.step_wrapper(2 + game._n_guards + ai_pk)  # Pick(0)

        game.step_wrapper(0)  # AddStage 1
        game.step_wrapper(2 + gi_nk)  # NeedKey(0)
        game.step_wrapper(2 + game._n_guards + ai_mk)  # MoveToKey(0)

        obs, reward, term, trunc, info = game.step_wrapper(1)  # Finalize
        assert term
        assert reward > 0.5  # should solve the environment
        assert info["stage_program"].num_stages == 2

    def test_non_terminal_reward_zero(self, game_d2):
        game_d2.reset_wrapper()
        obs, reward, term, trunc, info = game_d2.step_wrapper(0)  # AddStage
        assert not term
        assert reward == 0.0

    def test_structure_first_canonical(self):
        """Same canonical solver via structure-first order."""
        game = _make_game(D=2, max_stages=3, policy="structure_first")
        game.reset_wrapper()

        gi_pr = game._guard_to_idx[PickReady(0)]
        gi_nk = game._guard_to_idx[NeedKey(0)]
        ai_pk = game._action_to_idx[Pick(0)]
        ai_mk = game._action_to_idx[MoveToKey(0)]

        # Structure phase: add 2 stages, then finalize
        game.step_wrapper(0)  # AddStage 0
        game.step_wrapper(0)  # AddStage 1
        game.step_wrapper(1)  # Finalize

        # Fill phase: guard0, action0, guard1, action1
        game.step_wrapper(2 + gi_pr)
        game.step_wrapper(2 + game._n_guards + ai_pk)
        game.step_wrapper(2 + gi_nk)

        obs, reward, term, trunc, info = game.step_wrapper(
            2 + game._n_guards + ai_mk,
        )
        assert term
        assert reward > 0.5
        assert info["stage_program"].num_stages == 2

    def test_episode_length_counts(self):
        """Check step counts match expected."""
        game = _make_game(D=2, policy="leftmost")
        game.reset_wrapper()
        # 0-stage: 1 step
        game.step_wrapper(1)  # Finalize
        assert game.step_count == 1

        # 1-stage: 4 steps (AddStage, guard, action, Finalize)
        game.reset_wrapper()
        game.step_wrapper(0)
        game.step_wrapper(2)
        game.step_wrapper(2 + game._n_guards)
        game.step_wrapper(1)
        assert game.step_count == 4


# ---------------------------------------------------------------------------
# TestObservation
# ---------------------------------------------------------------------------

class TestObservation:
    def test_obs_shape(self, game_d2):
        obs, _ = game_d2.reset_wrapper()
        max_decisions = 3 * game_d2._max_stages + 1
        assert obs.shape == (2 * max_decisions,)

    def test_obs_all_pad_initially(self, game_d2):
        obs, _ = game_d2.reset_wrapper()
        assert (obs == 0.0).all()

    def test_obs_encodes_structure_decision(self, game_d2):
        game_d2.reset_wrapper()
        game_d2.step_wrapper(0)  # AddStage
        obs = game_d2.obs
        assert obs[0] == STAGE_TOKEN_IDS["STRUCTURE"]
        assert obs[1] == 0.0  # param=0 for AddStage

    def test_obs_encodes_finalize(self, game_d2):
        game_d2.reset_wrapper()
        game_d2.step_wrapper(1)  # Finalize
        obs = game_d2.obs
        assert obs[0] == STAGE_TOKEN_IDS["STRUCTURE"]
        assert obs[1] == 1.0  # param=1 for Finalize

    def test_obs_encodes_guard(self):
        game = _make_game(D=2, policy="leftmost")
        game.reset_wrapper()
        game.step_wrapper(0)  # AddStage
        game.step_wrapper(2 + 3)  # 4th guard atom
        obs = game.obs
        # Decision 0: structure(AddStage)
        assert obs[0] == STAGE_TOKEN_IDS["STRUCTURE"]
        # Decision 1: guard(idx=3)
        assert obs[2] == STAGE_TOKEN_IDS["GUARD"]
        assert obs[3] == 3.0

    def test_obs_dtype(self, game_d2):
        obs, _ = game_d2.reset_wrapper()
        assert obs.dtype == np.float32


# ---------------------------------------------------------------------------
# TestStateManagement
# ---------------------------------------------------------------------------

class TestStateManagement:
    def test_stash_unstash_roundtrip(self, game_d2):
        game_d2.reset_wrapper()
        game_d2.step_wrapper(0)  # AddStage
        stashed = game_d2.stash_state()

        # Advance further
        game_d2.step_wrapper(2)  # guard
        assert game_d2.step_count == 2

        # Unstash
        game_d2.unstash_state(stashed)
        assert game_d2.step_count == 1
        # Mask should show guard options again
        mask = game_d2.get_action_mask()
        assert mask[2:2 + game_d2._n_guards].all()

    def test_clone_independence(self, game_d2):
        game_d2.reset_wrapper()
        game_d2.step_wrapper(0)

        clone = game_d2.clone()
        clone.step_wrapper(2)  # advance clone

        # Original should still be at step 1
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
        d = {game_d2.hashable_obs: 1}
        game_d2.step_wrapper(0)
        d[game_d2.hashable_obs] = 2
        assert len(d) == 2


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_max_stages_1(self):
        game = _make_game(D=2, max_stages=1, policy="leftmost")
        game.reset_wrapper()
        game.step_wrapper(0)  # AddStage (now at max)
        game.step_wrapper(2)  # guard
        game.step_wrapper(2 + game._n_guards)  # action
        # StructureHole: only Finalize legal
        mask = game.get_action_mask()
        assert mask.sum() == 1
        assert mask[1] == True  # only Finalize
        obs, reward, term, _, info = game.step_wrapper(1)
        assert term

    def test_max_decisions_bound(self):
        """Episode should never exceed max_decisions steps."""
        game = _make_game(D=2, max_stages=3, policy="leftmost")
        game.reset_wrapper()
        steps = 0
        while not game.terminated:
            mask = game.get_action_mask()
            # Always pick the first legal action
            action = int(np.argmax(mask))
            game.step_wrapper(action)
            steps += 1
            if steps > 3 * game._max_stages + 1:
                pytest.fail("Exceeded max_decisions")
        assert steps <= 3 * game._max_stages + 1

    def test_no_dead_ends(self):
        """Every non-terminal state has at least one legal action."""
        game = _make_game(D=2, max_stages=2, policy="leftmost")
        game.reset_wrapper()
        while not game.terminated:
            mask = game.get_action_mask()
            assert mask.sum() >= 1, "Dead end detected!"
            action = int(np.argmax(mask))
            game.step_wrapper(action)
