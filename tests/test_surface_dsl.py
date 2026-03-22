"""Tests for DoorsStageDSL(D): surface DSL, compiler, and search grammar."""

from __future__ import annotations

import numpy as np
import pytest

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.instances.doors.dsl.surface_dsl import (
    AtKeyLoc, KeyAvail, RoomLocked, PickReady, NeedKey,
    Pick, MoveToKey, MoveToGoal,
    PickRule, MoveRule, GoalRule,
    SurfacePolicy,
)
from alphazeropp.instances.doors.dsl.surface_compiler import (
    compile_condition, compile_action, compile_rule, compile_policy,
)
from alphazeropp.instances.doors.dsl.surface_grammar import (
    canonical_policy, count_relaxed_policies, enumerate_relaxed_policies,
)
from alphazeropp.synthesis.ast_nodes import (
    Flip, IsZero, Not, And, Ite, Default,
)
from alphazeropp.synthesis.interpreter import (
    eval_condition, eval_program, run_policy_episode,
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


# ---------------------------------------------------------------------------
# TestCompilerCorrectness — D=2
# ---------------------------------------------------------------------------

class TestCompilerCorrectness:
    """Compiler correctness on D=2."""

    def test_canonical_d2_compiles(self, cfg_d2):
        policy = canonical_policy(2)
        prog = compile_policy(policy, cfg_d2)
        assert isinstance(prog, Ite)

    def test_canonical_d2_node_count(self, cfg_d2):
        policy = canonical_policy(2)
        prog = compile_policy(policy, cfg_d2)
        # D=2, K=1: 1 PickRule(7) + 1 MoveRule(3) + Default(2) = 12
        assert prog.node_count() == 12

    def test_canonical_d2_solves(self, cfg_d2):
        policy = canonical_policy(2)
        prog = compile_policy(policy, cfg_d2)
        x0 = doors_initial_state(cfg_d2)
        n_sites = cfg_d2.obs_size()
        env = cfg_d2.make_env(n_sites, frozen_states=[x0])
        result = run_policy_episode(env, prog, x0=x0, is_solved=cfg_d2.is_solved)
        assert result.solved
        assert result.total_env_steps == 3

    def test_canonical_d2_action_trace(self, cfg_d2):
        """Verify the action trace: move to key, pick key, move to goal."""
        policy = canonical_policy(2)
        prog = compile_policy(policy, cfg_d2)
        x0 = doors_initial_state(cfg_d2)
        n_sites = cfg_d2.obs_size()
        env = cfg_d2.make_env(n_sites, frozen_states=[x0])
        result = run_policy_episode(env, prog, x0=x0, is_solved=cfg_d2.is_solved)
        actions = [s.action for s in result.steps]
        # D=2: M=4, key_loc[0]=1, pick=M+0=4, goal_loc=3
        assert actions == [1, 4, 3]  # MoveToKey(0), Pick(0), MoveToGoal

    def test_pick_rule_matches_macro(self, cfg_d2):
        """PickRule condition/action must match doors_macros.py output."""
        prog = compile_rule(PickRule(0), Default(Flip(0)), cfg_d2)
        assert isinstance(prog, Ite)
        # Condition: And(Not(IsZero(1)), Not(IsZero(6)))
        expected_cond = And(Not(IsZero(1)), Not(IsZero(6)))
        assert prog.cond.pretty() == expected_cond.pretty()
        # Action: Flip(4)  (M + k = 4 + 0)
        assert prog.action == Flip(4)

    def test_move_rule_matches_macro(self, cfg_d2):
        """MoveRule condition/action must match doors_macros.py output."""
        prog = compile_rule(MoveRule(0), Default(Flip(0)), cfg_d2)
        assert isinstance(prog, Ite)
        # Condition: IsZero(5)  (M + key_unlocks[0] = 4 + 1)
        assert prog.cond.pretty() == IsZero(5).pretty()
        # Action: Flip(1)  (key_loc[0] = 1)
        assert prog.action == Flip(1)

    def test_concrete_indices_d2(self, cfg_d2):
        """Verify all config-derived indices for D=2."""
        # M=4, D=2, K=1, key_loc=[1], key_unlocks=[1], goal_loc=3
        assert cfg_d2.M == 4
        assert cfg_d2.D == 2
        assert cfg_d2.K == 1
        assert cfg_d2.key_loc == [1]
        assert cfg_d2.key_unlocks == [1]
        assert cfg_d2.goal_loc == 3
        assert cfg_d2.obs_size() == 7


# ---------------------------------------------------------------------------
# TestParametricSmoke — D=3
# ---------------------------------------------------------------------------

class TestParametricSmoke:
    """Parametric compilation smoke tests for D=3."""

    def test_canonical_d3_compiles(self, cfg_d3):
        policy = canonical_policy(3)
        prog = compile_policy(policy, cfg_d3)
        assert isinstance(prog, Ite)

    def test_canonical_d3_node_count(self, cfg_d3):
        policy = canonical_policy(3)
        prog = compile_policy(policy, cfg_d3)
        # D=3, K=2: 2 PickRule(7) + 2 MoveRule(3) + Default(2) = 22
        assert prog.node_count() == 22

    def test_canonical_d3_solves(self, cfg_d3):
        policy = canonical_policy(3)
        prog = compile_policy(policy, cfg_d3)
        x0 = doors_initial_state(cfg_d3)
        n_sites = cfg_d3.obs_size()
        env = cfg_d3.make_env(n_sites, frozen_states=[x0])
        result = run_policy_episode(env, prog, x0=x0, is_solved=cfg_d3.is_solved)
        assert result.solved
        assert result.total_env_steps == 5

    def test_d3_config_indices(self, cfg_d3):
        """Spot-check config-derived indices for D=3."""
        # M=6, D=3, K=2, key_loc=[1,3], key_unlocks=[1,2], goal_loc=5
        assert cfg_d3.M == 6
        assert cfg_d3.K == 2
        assert cfg_d3.key_loc == [1, 3]
        assert cfg_d3.key_unlocks == [1, 2]
        assert cfg_d3.goal_loc == 5


# ---------------------------------------------------------------------------
# TestSurfaceSemantics
# ---------------------------------------------------------------------------

class TestSurfaceSemantics:
    """Individual surface atoms compile to the expected raw semantics."""

    def test_at_key_loc(self, cfg_d2):
        cond = compile_condition(AtKeyLoc(0), cfg_d2)
        # AtKeyLoc(0) -> Not(IsZero(key_loc[0])) = Not(IsZero(1))
        assert cond.pretty() == "Not(IsZero(1))"

        # State where agent IS at key location (obs[1] = 1.0)
        state_at = np.zeros(7, dtype=np.float32)
        state_at[1] = 1.0
        assert eval_condition(cond, state_at) is True

        # State where agent is NOT at key location (obs[1] = 0.0)
        state_away = np.zeros(7, dtype=np.float32)
        assert eval_condition(cond, state_away) is False

    def test_key_avail(self, cfg_d2):
        cond = compile_condition(KeyAvail(0), cfg_d2)
        # KeyAvail(0) -> Not(IsZero(M+D+0)) = Not(IsZero(6))
        assert cond.pretty() == "Not(IsZero(6))"

        state_avail = np.zeros(7, dtype=np.float32)
        state_avail[6] = 1.0
        assert eval_condition(cond, state_avail) is True

        state_used = np.zeros(7, dtype=np.float32)
        assert eval_condition(cond, state_used) is False

    def test_room_locked(self, cfg_d2):
        cond = compile_condition(RoomLocked(1), cfg_d2)
        # RoomLocked(1) -> IsZero(M+1) = IsZero(5)
        assert cond.pretty() == "IsZero(5)"

        # Room locked (obs[5] = 0)
        state_locked = np.zeros(7, dtype=np.float32)
        assert eval_condition(cond, state_locked) is True

        # Room unlocked (obs[5] = 1)
        state_open = np.zeros(7, dtype=np.float32)
        state_open[5] = 1.0
        assert eval_condition(cond, state_open) is False

    def test_pick_ready_conjunction(self, cfg_d2):
        cond = compile_condition(PickReady(0), cfg_d2)

        # Both true: at key loc AND key available
        state_ready = np.zeros(7, dtype=np.float32)
        state_ready[1] = 1.0  # at key location
        state_ready[6] = 1.0  # key available
        assert eval_condition(cond, state_ready) is True

        # Only at key loc, key not available
        state_no_key = np.zeros(7, dtype=np.float32)
        state_no_key[1] = 1.0
        assert eval_condition(cond, state_no_key) is False

        # Key available but not at key loc
        state_away = np.zeros(7, dtype=np.float32)
        state_away[6] = 1.0
        assert eval_condition(cond, state_away) is False

    def test_need_key(self, cfg_d2):
        cond = compile_condition(NeedKey(0), cfg_d2)
        # NeedKey(0) -> IsZero(M + key_unlocks[0]) = IsZero(5)
        assert cond.pretty() == "IsZero(5)"

    def test_goal_rule_compiles(self, cfg_d2):
        prog = compile_rule(GoalRule(), None, cfg_d2)
        assert isinstance(prog, Default)
        assert prog.action == Flip(3)  # goal_loc = 3

    def test_pick_action(self, cfg_d2):
        flip = compile_action(Pick(0), cfg_d2)
        assert flip == Flip(4)  # M + 0 = 4

    def test_move_to_key_action(self, cfg_d2):
        flip = compile_action(MoveToKey(0), cfg_d2)
        assert flip == Flip(1)  # key_loc[0] = 1

    def test_move_to_goal_action(self, cfg_d2):
        flip = compile_action(MoveToGoal(), cfg_d2)
        assert flip == Flip(3)  # goal_loc = 3


# ---------------------------------------------------------------------------
# TestSearchGrammar
# ---------------------------------------------------------------------------

class TestSearchGrammar:
    """Canonical and relaxed grammar tests."""

    def test_canonical_d2(self):
        p = canonical_policy(2)
        assert len(p.rules) == 3
        assert p.rules == (PickRule(0), MoveRule(0), GoalRule())

    def test_canonical_d3(self):
        p = canonical_policy(3)
        assert len(p.rules) == 5
        assert p.rules == (
            PickRule(0), MoveRule(0),
            PickRule(1), MoveRule(1),
            GoalRule(),
        )

    def test_count_relaxed_d2(self):
        assert count_relaxed_policies(2) == 1

    def test_count_relaxed_d3(self):
        assert count_relaxed_policies(3) == 6

    def test_count_relaxed_d4(self):
        assert count_relaxed_policies(4) == 90

    def test_count_relaxed_d5(self):
        assert count_relaxed_policies(5) == 2520

    def test_enumerate_relaxed_d3(self):
        policies = enumerate_relaxed_policies(3)
        assert len(policies) == 6
        for p in policies:
            assert isinstance(p.rules[-1], GoalRule)

    def test_relaxed_d3_solve_subset(self, cfg_d3):
        """Relaxed D=3 policies: canonical solves; not all interleavings do.

        Policies that place MoveRule(1) before MoveRule(0) may attempt to
        move to key 1's location (behind locked room 1) before room 1 is
        unlocked.  This is a real semantic distinction — the relaxed grammar
        is a *superset* that includes non-solving programs.
        """
        policies = enumerate_relaxed_policies(3)
        x0 = doors_initial_state(cfg_d3)
        n_sites = cfg_d3.obs_size()
        solve_count = 0
        for policy in policies:
            prog = compile_policy(policy, cfg_d3)
            env = cfg_d3.make_env(n_sites, frozen_states=[x0])
            result = run_policy_episode(
                env, prog, x0=x0, is_solved=cfg_d3.is_solved,
            )
            if result.solved:
                solve_count += 1
        # At least the canonical ordering must solve
        assert solve_count >= 1
        # Not all solve (some interleavings are semantically invalid)
        assert solve_count < len(policies)

    def test_canonical_in_relaxed(self):
        canonical = canonical_policy(3)
        relaxed = enumerate_relaxed_policies(3)
        assert canonical in relaxed

    def test_relaxed_d1(self):
        """D=1: just GoalRule."""
        assert count_relaxed_policies(1) == 1
        policies = enumerate_relaxed_policies(1)
        assert len(policies) == 1
        assert policies[0].rules == (GoalRule(),)


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Validation and edge cases."""

    def test_empty_policy_raises(self):
        with pytest.raises(ValueError, match="at least one rule"):
            SurfacePolicy(())

    def test_missing_goal_rule_raises(self):
        with pytest.raises(ValueError, match="GoalRule"):
            SurfacePolicy((PickRule(0),))

    def test_out_of_range_key_raises(self, cfg_d2):
        with pytest.raises(ValueError, match="Key index 5"):
            compile_condition(AtKeyLoc(5), cfg_d2)

    def test_out_of_range_room_raises(self, cfg_d2):
        with pytest.raises(ValueError, match="Room index 3"):
            compile_condition(RoomLocked(3), cfg_d2)

    def test_out_of_range_action_raises(self, cfg_d2):
        with pytest.raises(ValueError, match="Key index 2"):
            compile_action(Pick(2), cfg_d2)

    def test_d1_canonical(self):
        """D=1: canonical is just GoalRule."""
        p = canonical_policy(1)
        assert p.rules == (GoalRule(),)

    def test_d1_compiles(self):
        cfg = DoorsGameConfig(num_rooms=1, locs_per_room=2)
        p = canonical_policy(1)
        prog = compile_policy(p, cfg)
        assert isinstance(prog, Default)
        assert prog.action == Flip(cfg.goal_loc)

    def test_pretty_printing(self):
        p = canonical_policy(2)
        assert p.pretty() == "PickRule(0) >> MoveRule(0) >> GoalRule"
