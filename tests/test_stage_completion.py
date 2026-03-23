"""Tests for the stage-skeleton completion-order experiment."""

from __future__ import annotations

import pytest

from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig
from alphazeropp.instances.doors.dsl.surface_dsl import (
    PickReady, NeedKey, Pick, MoveToKey, MoveToGoal,
)
from alphazeropp.instances.doors.dsl.stage_dsl import (
    GuardHole, ActionHole, Stage, StageProgram,
)
from alphazeropp.instances.doors.dsl.stage_partial import (
    PartialStageProgram, StructureHole, GuardHoleRef, ActionHoleRef,
    AddStage, Finalize,
    list_holes, apply_filling, partial_depth,
)
from alphazeropp.instances.doors.dsl.stage_partial_eval import (
    partial_semantic_signature, UNRESOLVED,
)
from alphazeropp.instances.doors.dsl.stage_hole_selection import (
    LeftmostPolicy, StructureFirstPolicy,
)
from alphazeropp.instances.doors.dsl.stage_search import best_first_search
from alphazeropp.instances.doors.dsl.stage_search_cost import SearchCostModel
from alphazeropp.instances.doors.dsl.stage_diagnostics import make_frozen_state_suite
from alphazeropp.instances.doors.dsl.stage_partial_eval import make_rich_frozen_state_suite


@pytest.fixture
def cfg_d2():
    return DoorsGameConfig(num_rooms=2, locs_per_room=2)


@pytest.fixture
def cfg_d3():
    return DoorsGameConfig(num_rooms=3, locs_per_room=2)


# ---------------------------------------------------------------------------
# TestPartialStageProgram
# ---------------------------------------------------------------------------

class TestPartialStageProgram:
    def test_empty_partial(self):
        p = PartialStageProgram(stages=(), max_stages=3)
        assert not p.is_terminal()
        assert not p.structure_finalized

    def test_add_stage(self):
        p = PartialStageProgram(stages=(), max_stages=3)
        p2 = apply_filling(p, StructureHole(), AddStage())
        assert len(p2.stages) == 1
        assert isinstance(p2.stages[0].guard, GuardHole)
        assert isinstance(p2.stages[0].action, ActionHole)

    def test_finalize_empty(self):
        p = PartialStageProgram(stages=(), max_stages=3)
        p2 = apply_filling(p, StructureHole(), Finalize())
        assert p2.structure_finalized
        assert p2.is_terminal()  # 0 stages, all complete

    def test_fill_guard(self):
        p = PartialStageProgram(
            stages=(Stage(guard=GuardHole(), action=ActionHole()),),
            max_stages=3,
            structure_finalized=True,
        )
        p2 = apply_filling(p, GuardHoleRef(0), PickReady(0))
        assert p2.stages[0].guard == PickReady(0)
        assert isinstance(p2.stages[0].action, ActionHole)

    def test_fill_action(self):
        p = PartialStageProgram(
            stages=(Stage(guard=PickReady(0), action=ActionHole()),),
            max_stages=3,
            structure_finalized=True,
        )
        p2 = apply_filling(p, ActionHoleRef(0), Pick(0))
        assert p2.stages[0].action == Pick(0)
        assert p2.is_terminal()

    def test_to_stage_program(self):
        p = PartialStageProgram(
            stages=(Stage(guard=PickReady(0), action=Pick(0)),),
            max_stages=3,
            structure_finalized=True,
        )
        prog = p.to_stage_program()
        assert isinstance(prog, StageProgram)
        assert prog.stages == (Stage(guard=PickReady(0), action=Pick(0)),)

    def test_to_stage_program_raises_if_not_terminal(self):
        p = PartialStageProgram(stages=(), max_stages=3)
        with pytest.raises(ValueError):
            p.to_stage_program()

    def test_list_holes_empty(self):
        p = PartialStageProgram(stages=(), max_stages=3)
        holes = list_holes(p)
        assert holes == [StructureHole()]

    def test_list_holes_with_stage(self):
        p = PartialStageProgram(
            stages=(Stage(guard=GuardHole(), action=ActionHole()),),
            max_stages=3,
        )
        holes = list_holes(p)
        assert GuardHoleRef(0) in holes
        assert ActionHoleRef(0) in holes
        # StructureHole NOT present because stage 0 is incomplete
        assert StructureHole() not in holes

    def test_list_holes_complete_stage_unfinalised(self):
        p = PartialStageProgram(
            stages=(Stage(guard=PickReady(0), action=Pick(0)),),
            max_stages=3,
        )
        holes = list_holes(p)
        assert holes == [StructureHole()]

    def test_partial_depth(self):
        p = PartialStageProgram(stages=(), max_stages=3)
        assert partial_depth(p) == 0

        p2 = apply_filling(p, StructureHole(), AddStage())
        assert partial_depth(p2) == 1  # one structure decision

        p3 = apply_filling(p2, GuardHoleRef(0), PickReady(0))
        assert partial_depth(p3) == 2

        p4 = apply_filling(p3, ActionHoleRef(0), Pick(0))
        assert partial_depth(p4) == 3


# ---------------------------------------------------------------------------
# TestPartialSemanticSignature
# ---------------------------------------------------------------------------

class TestPartialSemanticSignature:
    def test_empty_unfinalised_all_unresolved(self, cfg_d2):
        p = PartialStageProgram(stages=(), max_stages=3)
        states = make_frozen_state_suite(cfg_d2)
        sig = partial_semantic_signature(p, states, cfg_d2)
        assert all(s == UNRESOLVED for s in sig)

    def test_empty_finalised_all_default(self, cfg_d2):
        p = PartialStageProgram(stages=(), max_stages=3, structure_finalized=True)
        states = make_frozen_state_suite(cfg_d2)
        sig = partial_semantic_signature(p, states, cfg_d2)
        # All states get default action (MoveToGoal = Flip(goal_loc))
        assert all(s == cfg_d2.goal_loc for s in sig)

    def test_complete_stage_resolves_some(self, cfg_d2):
        """PickReady(0) is true only when agent is at key 0 AND key available."""
        p = PartialStageProgram(
            stages=(Stage(guard=PickReady(0), action=Pick(0)),),
            max_stages=3,
            structure_finalized=True,
        )
        states = make_frozen_state_suite(cfg_d2)
        sig = partial_semantic_signature(p, states, cfg_d2)
        # Some states resolved to Pick(0) action, others to default
        assert cfg_d2.goal_loc in sig  # at least one state falls through to default

    def test_guard_hole_blocks_resolution(self, cfg_d2):
        p = PartialStageProgram(
            stages=(Stage(guard=GuardHole(), action=Pick(0)),),
            max_stages=3,
            structure_finalized=True,
        )
        states = make_frozen_state_suite(cfg_d2)
        sig = partial_semantic_signature(p, states, cfg_d2)
        assert all(s == UNRESOLVED for s in sig)


# ---------------------------------------------------------------------------
# TestHoleSelectionPolicies
# ---------------------------------------------------------------------------

class TestHoleSelectionPolicies:
    def test_leftmost_picks_guard_first(self):
        p = PartialStageProgram(
            stages=(Stage(guard=GuardHole(), action=ActionHole()),),
            max_stages=3,
            structure_finalized=True,
        )
        policy = LeftmostPolicy()
        assert policy.select_hole(p) == GuardHoleRef(0)

    def test_leftmost_picks_action_after_guard(self):
        p = PartialStageProgram(
            stages=(Stage(guard=PickReady(0), action=ActionHole()),),
            max_stages=3,
            structure_finalized=True,
        )
        policy = LeftmostPolicy()
        assert policy.select_hole(p) == ActionHoleRef(0)

    def test_leftmost_picks_structure_when_all_complete(self):
        p = PartialStageProgram(
            stages=(Stage(guard=PickReady(0), action=Pick(0)),),
            max_stages=3,
        )
        policy = LeftmostPolicy()
        assert policy.select_hole(p) == StructureHole()

    def test_structure_first_always_picks_structure(self):
        p = PartialStageProgram(
            stages=(Stage(guard=GuardHole(), action=ActionHole()),),
            max_stages=3,
        )
        policy = StructureFirstPolicy()
        assert policy.select_hole(p) == StructureHole()

    def test_structure_first_fills_after_finalize(self):
        p = PartialStageProgram(
            stages=(Stage(guard=GuardHole(), action=ActionHole()),),
            max_stages=3,
            structure_finalized=True,
        )
        policy = StructureFirstPolicy()
        assert policy.select_hole(p) == GuardHoleRef(0)


# ---------------------------------------------------------------------------
# TestBestFirstSearch — D=2 smoke
# ---------------------------------------------------------------------------

class TestBestFirstSearch:
    def test_d2_leftmost_finds_solver(self, cfg_d2):
        cost_model = SearchCostModel(max_stages=3, max_guard_depth=0)
        result = best_first_search(
            cfg_d2, cost_model, LeftmostPolicy(),
            max_expansions=100_000, dedup=False,
        )
        assert result.stats.solving_programs >= 1
        assert result.stats.time_to_first_solver is not None

    def test_d2_structure_first_finds_solver(self, cfg_d2):
        cost_model = SearchCostModel(max_stages=3, max_guard_depth=0)
        result = best_first_search(
            cfg_d2, cost_model, StructureFirstPolicy(),
            max_expansions=100_000, dedup=False,
        )
        assert result.stats.solving_programs >= 1

    def test_d2_both_find_same_solvers(self, cfg_d2):
        """Both policies find the same number of solvers (without dedup)."""
        cost_model = SearchCostModel(max_stages=3, max_guard_depth=0)

        r1 = best_first_search(
            cfg_d2, cost_model, LeftmostPolicy(),
            max_expansions=100_000, dedup=False,
        )
        r2 = best_first_search(
            cfg_d2, cost_model, StructureFirstPolicy(),
            max_expansions=100_000, dedup=False,
        )
        assert r1.stats.solving_programs == r2.stats.solving_programs

    def test_dedup_has_hits(self, cfg_d2):
        cost_model = SearchCostModel(max_stages=3, max_guard_depth=0)
        frozen = make_rich_frozen_state_suite(cfg_d2)
        result = best_first_search(
            cfg_d2, cost_model, LeftmostPolicy(), frozen,
            max_expansions=100_000, dedup=True,
        )
        assert result.stats.dedup_hits > 0

    def test_no_dedup_flag(self, cfg_d2):
        cost_model = SearchCostModel(max_stages=2, max_guard_depth=0)
        result = best_first_search(
            cfg_d2, cost_model, LeftmostPolicy(),
            max_expansions=100_000, dedup=False,
        )
        assert result.stats.dedup_hits == 0
        assert result.stats.semantically_distinct_partial == -1
