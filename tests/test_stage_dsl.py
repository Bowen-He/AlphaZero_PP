"""Tests for the budget-free stage-skeleton DSL."""

from __future__ import annotations

import pytest

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state,
)
from alphazeropp.instances.doors.dsl.surface_dsl import (
    AtKeyLoc, KeyAvail, RoomLocked, PickReady, NeedKey,
    Pick, MoveToKey, MoveToGoal,
)
from alphazeropp.instances.doors.dsl.stage_dsl import (
    GuardNot, GuardAnd, GuardHole, ActionHole,
    Stage, StageProgram,
)
from alphazeropp.instances.doors.dsl.stage_compiler import (
    compile_guard, compile_stage_program,
)
from alphazeropp.instances.doors.dsl.stage_search_cost import (
    SearchCostModel, guard_depth, guard_node_count, stage_program_cost,
)
from alphazeropp.instances.doors.dsl.stage_grammar import (
    enumerate_guard_atoms, enumerate_guards, enumerate_actions,
    enumerate_stage_programs, count_stage_programs, count_guards,
)
from alphazeropp.synthesis.ast_nodes import Ite, Default
from alphazeropp.synthesis.interpreter import run_policy_episode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg_d2():
    return DoorsGameConfig(num_rooms=2, locs_per_room=2)


@pytest.fixture
def cfg_d3():
    return DoorsGameConfig(num_rooms=3, locs_per_room=2)


def _canonical_stage_program_d2():
    """The D=2 canonical controller as a StageProgram."""
    return StageProgram(
        stages=(
            Stage(guard=PickReady(0), action=Pick(0)),
            Stage(guard=NeedKey(0), action=MoveToKey(0)),
        ),
        default_action=MoveToGoal(),
    )


# ---------------------------------------------------------------------------
# Type tests
# ---------------------------------------------------------------------------

class TestStageTypes:
    def test_stage_program_pretty(self):
        prog = _canonical_stage_program_d2()
        text = prog.pretty()
        assert "PickReady(0)" in text
        assert "NeedKey(0)" in text
        assert "MoveToGoal" in text

    def test_is_complete(self):
        prog = _canonical_stage_program_d2()
        assert prog.is_complete()

    def test_hole_not_complete(self):
        prog = StageProgram(
            stages=(Stage(guard=GuardHole(0), action=Pick(0)),),
        )
        assert not prog.is_complete()

    def test_action_hole_not_complete(self):
        prog = StageProgram(
            stages=(Stage(guard=PickReady(0), action=ActionHole()),),
        )
        assert not prog.is_complete()

    def test_guard_combinators(self):
        g = GuardAnd(PickReady(0), GuardNot(RoomLocked(0)))
        assert "And" in g.pretty()
        assert "Not" in g.pretty()

    def test_frozen(self):
        s = Stage(guard=PickReady(0), action=Pick(0))
        with pytest.raises(AttributeError):
            s.guard = NeedKey(0)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Compiler tests
# ---------------------------------------------------------------------------

class TestStageCompiler:
    def test_canonical_d2_compiles(self, cfg_d2):
        prog = _canonical_stage_program_d2()
        ast = compile_stage_program(prog, cfg_d2)
        assert isinstance(ast, Ite)

    def test_canonical_d2_node_count(self, cfg_d2):
        prog = _canonical_stage_program_d2()
        ast = compile_stage_program(prog, cfg_d2)
        # PickRule(0) guard = And(Not(IsZero), Not(IsZero)) = 5 nodes
        # + Flip(action) = 1 + Ite = 1 = 7 for first stage
        # NeedKey(0) guard = IsZero = 1 node + Flip = 1 + Ite = 1 = 3
        # Default(Flip) = 2
        # Total: 7 + 3 + 2 = 12
        assert ast.node_count() == 12

    def test_canonical_d2_matches_surface(self, cfg_d2):
        """Stage compiler produces same AST as surface compiler."""
        from alphazeropp.instances.doors.dsl.surface_grammar import canonical_policy
        from alphazeropp.instances.doors.dsl.surface_compiler import compile_policy

        surface_ast = compile_policy(canonical_policy(2), cfg_d2)
        stage_ast = compile_stage_program(_canonical_stage_program_d2(), cfg_d2)
        assert surface_ast == stage_ast

    def test_canonical_d2_solves(self, cfg_d2):
        prog = _canonical_stage_program_d2()
        ast = compile_stage_program(prog, cfg_d2)
        x0 = doors_initial_state(cfg_d2)
        env = cfg_d2.make_env(cfg_d2.obs_size(), frozen_states=[x0])
        result = run_policy_episode(env, ast, x0=x0, is_solved=cfg_d2.is_solved)
        assert result.solved
        assert result.total_env_steps == 3

    def test_guard_not_compiles(self, cfg_d2):
        prog = StageProgram(
            stages=(Stage(guard=GuardNot(RoomLocked(1)), action=MoveToGoal()),),
            default_action=MoveToGoal(),
        )
        ast = compile_stage_program(prog, cfg_d2)
        assert isinstance(ast, Ite)

    def test_guard_and_compiles(self, cfg_d2):
        prog = StageProgram(
            stages=(Stage(
                guard=GuardAnd(AtKeyLoc(0), KeyAvail(0)),
                action=Pick(0),
            ),),
            default_action=MoveToGoal(),
        )
        ast = compile_stage_program(prog, cfg_d2)
        assert isinstance(ast, Ite)

    def test_holes_raise(self, cfg_d2):
        prog = StageProgram(
            stages=(Stage(guard=GuardHole(0), action=Pick(0)),),
        )
        with pytest.raises(ValueError, match="holes"):
            compile_stage_program(prog, cfg_d2)

    def test_empty_stages(self, cfg_d2):
        """Zero stages → just Default(MoveToGoal)."""
        prog = StageProgram(stages=(), default_action=MoveToGoal())
        ast = compile_stage_program(prog, cfg_d2)
        assert isinstance(ast, Default)


# ---------------------------------------------------------------------------
# Cost model tests
# ---------------------------------------------------------------------------

class TestSearchCost:
    def test_atom_depth_zero(self):
        assert guard_depth(PickReady(0)) == 0

    def test_not_depth_one(self):
        assert guard_depth(GuardNot(PickReady(0))) == 1

    def test_and_depth_one(self):
        assert guard_depth(GuardAnd(PickReady(0), NeedKey(0))) == 1

    def test_nested_depth_two(self):
        g = GuardAnd(GuardNot(PickReady(0)), NeedKey(0))
        assert guard_depth(g) == 2

    def test_atom_node_count(self):
        assert guard_node_count(PickReady(0)) == 1

    def test_and_node_count(self):
        assert guard_node_count(GuardAnd(PickReady(0), NeedKey(0))) == 3

    def test_cost_model_admits(self):
        model = SearchCostModel(max_stages=3, max_guard_depth=0)
        assert model.admits_guard(PickReady(0))
        assert not model.admits_guard(GuardNot(PickReady(0)))

    def test_cost_model_admits_program(self):
        model = SearchCostModel(max_stages=2, max_guard_depth=0)
        prog = _canonical_stage_program_d2()
        assert model.admits_program(prog)

    def test_cost_model_rejects_too_many_stages(self):
        model = SearchCostModel(max_stages=1, max_guard_depth=0)
        prog = _canonical_stage_program_d2()
        assert not model.admits_program(prog)


# ---------------------------------------------------------------------------
# Grammar enumeration tests
# ---------------------------------------------------------------------------

class TestStageGrammar:
    def test_guard_atoms_d2(self, cfg_d2):
        atoms = enumerate_guard_atoms(cfg_d2)
        # K=1: AtKeyLoc(0), KeyAvail(0), PickReady(0), NeedKey(0), RoomLocked(0), RoomLocked(1)
        assert len(atoms) == 6

    def test_guard_atoms_d3(self, cfg_d3):
        atoms = enumerate_guard_atoms(cfg_d3)
        # K=2: 4 atoms per key * 2 keys + 3 rooms = 11
        assert len(atoms) == 11

    def test_guards_depth0_d2(self, cfg_d2):
        guards = enumerate_guards(cfg_d2, max_depth=0)
        assert len(guards) == 6

    def test_guards_depth1_d2(self, cfg_d2):
        guards = enumerate_guards(cfg_d2, max_depth=1)
        # G(0)=6, G(1) = 6 + 6 + 36 = 48
        assert len(guards) == 48

    def test_count_guards_matches_enumerate(self, cfg_d2):
        for d in range(3):
            counted = count_guards(cfg_d2, d)
            enumerated = len(enumerate_guards(cfg_d2, d))
            assert counted == enumerated, f"Mismatch at depth {d}: {counted} vs {enumerated}"

    def test_actions_d2(self, cfg_d2):
        actions = enumerate_actions(cfg_d2)
        # K=1: Pick(0), MoveToKey(0), MoveToGoal = 3
        assert len(actions) == 3

    def test_actions_d3(self, cfg_d3):
        actions = enumerate_actions(cfg_d3)
        # K=2: Pick(0,1), MoveToKey(0,1), MoveToGoal = 5
        assert len(actions) == 5

    def test_count_matches_enumerate_d2(self, cfg_d2):
        model = SearchCostModel(max_stages=3, max_guard_depth=0)
        counted = count_stage_programs(cfg_d2, model)
        enumerated = enumerate_stage_programs(cfg_d2, model)
        assert counted == len(enumerated)

    def test_count_d2_depth0(self, cfg_d2):
        model = SearchCostModel(max_stages=3, max_guard_depth=0)
        count = count_stage_programs(cfg_d2, model)
        # slots = 6 guards * 3 actions = 18
        # total = 1 + 18 + 324 + 5832 = 6175
        assert count == 6175

    def test_canonical_in_enumeration(self, cfg_d2):
        model = SearchCostModel(max_stages=3, max_guard_depth=0)
        programs = enumerate_stage_programs(cfg_d2, model)
        canonical = _canonical_stage_program_d2()
        assert canonical in programs

    def test_count_d3_depth0_s3(self, cfg_d3):
        """D=3 with max_stages=3, depth 0 should be tractable."""
        model = SearchCostModel(max_stages=3, max_guard_depth=0)
        count = count_stage_programs(cfg_d3, model)
        # slots = 11 * 5 = 55
        # total = 1 + 55 + 3025 + 166375 = 169456
        assert count == 169_456


# ---------------------------------------------------------------------------
# Diagnostic smoke test
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_diagnostics_d2_smoke(self, cfg_d2):
        from alphazeropp.instances.doors.dsl.stage_diagnostics import (
            diagnose_surface, diagnose_stage, format_report,
        )

        surface = diagnose_surface(cfg_d2)
        stage = diagnose_stage(cfg_d2, max_guard_depth=0, max_stages=2)

        assert surface.total_programs == 1
        assert surface.solving_programs == 1
        assert stage.total_programs > 1
        assert stage.solving_programs >= 1

        report = format_report([surface, stage])
        assert "Surface" in report
        assert "Stage" in report
