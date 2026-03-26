"""Tests for DoorsRelationalRuntime — typed relational queries over DoorsGameConfig."""

from __future__ import annotations

import pytest

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.instances.doors.dsl.relational_runtime import (
    DoorsRelationalRuntime, RoomId, KeyId, LocId,
)
from alphazeropp.instances.doors.dsl.surface_dsl import (
    PickReady, NeedKey,
    Pick, MoveToKey, MoveToGoal,
    PickRule, MoveRule, GoalRule,
)
from alphazeropp.instances.doors.dsl.surface_compiler import (
    compile_condition, compile_action, compile_policy,
)
from alphazeropp.instances.doors.dsl.surface_grammar import canonical_policy
from alphazeropp.synthesis.ast_nodes import (
    Flip, IsZero, Not, And, Ite, Default,
)
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
# TestEntityQueries
# ---------------------------------------------------------------------------

class TestEntityQueries:
    """Entity enumeration and relational lookup queries."""

    def test_lockable_rooms_d2(self, rt_d2):
        rooms = rt_d2.lockable_rooms()
        assert rooms == [RoomId(1)]

    def test_lockable_rooms_d3(self, rt_d3):
        rooms = rt_d3.lockable_rooms()
        assert rooms == [RoomId(1), RoomId(2)]

    def test_key_for_room_0_returns_none(self, rt_d2):
        assert rt_d2.key_for_room(RoomId(0)) is None

    def test_key_for_room_1_d2(self, rt_d2):
        assert rt_d2.key_for_room(RoomId(1)) == KeyId(0)

    def test_key_for_room_d3(self, rt_d3):
        assert rt_d3.key_for_room(RoomId(1)) == KeyId(0)
        assert rt_d3.key_for_room(RoomId(2)) == KeyId(1)

    def test_key_for_room_roundtrip(self, rt_d3):
        """key_for_room(room_unlocked_by(k)) == k for all k."""
        for k in range(rt_d3.num_keys()):
            kid = KeyId(k)
            r = rt_d3.room_unlocked_by(kid)
            assert rt_d3.key_for_room(r) == kid

    def test_loc_of_key_d2(self, rt_d2):
        assert rt_d2.loc_of_key(KeyId(0)) == LocId(1)

    def test_loc_of_key_d3(self, rt_d3):
        assert rt_d3.loc_of_key(KeyId(0)) == LocId(1)
        assert rt_d3.loc_of_key(KeyId(1)) == LocId(3)

    def test_room_unlocked_by_d2(self, rt_d2):
        assert rt_d2.room_unlocked_by(KeyId(0)) == RoomId(1)

    def test_room_unlocked_by_d3(self, rt_d3):
        assert rt_d3.room_unlocked_by(KeyId(0)) == RoomId(1)
        assert rt_d3.room_unlocked_by(KeyId(1)) == RoomId(2)

    def test_goal_location_d2(self, rt_d2):
        assert rt_d2.goal_location() == LocId(3)

    def test_goal_location_d3(self, rt_d3):
        assert rt_d3.goal_location() == LocId(5)

    def test_room_of_location_d3(self, rt_d3):
        # D=3, locs_per_room=2: locs 0,1 → room 0; 2,3 → room 1; 4,5 → room 2
        assert rt_d3.room_of_location(LocId(0)) == RoomId(0)
        assert rt_d3.room_of_location(LocId(1)) == RoomId(0)
        assert rt_d3.room_of_location(LocId(2)) == RoomId(1)
        assert rt_d3.room_of_location(LocId(3)) == RoomId(1)
        assert rt_d3.room_of_location(LocId(4)) == RoomId(2)
        assert rt_d3.room_of_location(LocId(5)) == RoomId(2)

    def test_locations_in_room_d3(self, rt_d3):
        assert rt_d3.locations_in_room(RoomId(0)) == [LocId(0), LocId(1)]
        assert rt_d3.locations_in_room(RoomId(1)) == [LocId(2), LocId(3)]
        assert rt_d3.locations_in_room(RoomId(2)) == [LocId(4), LocId(5)]

    def test_all_locations_d2(self, rt_d2):
        assert rt_d2.all_locations() == [LocId(i) for i in range(4)]

    def test_num_keys_num_rooms(self, rt_d3):
        assert rt_d3.num_keys() == 2
        assert rt_d3.num_rooms() == 3


# ---------------------------------------------------------------------------
# TestObsIndices
# ---------------------------------------------------------------------------

class TestObsIndices:
    """Observation and action index resolution."""

    def test_obs_at_loc(self, rt_d2):
        for l in range(rt_d2.cfg.M):
            assert rt_d2.obs_at_loc(LocId(l)) == l

    def test_obs_room_unlocked_d2(self, rt_d2):
        # D=2, M=4: room 0 → obs[4], room 1 → obs[5]
        assert rt_d2.obs_room_unlocked(RoomId(0)) == 4
        assert rt_d2.obs_room_unlocked(RoomId(1)) == 5

    def test_obs_room_unlocked_d3(self, rt_d3):
        # D=3, M=6: room 0 → obs[6], room 1 → obs[7], room 2 → obs[8]
        assert rt_d3.obs_room_unlocked(RoomId(0)) == 6
        assert rt_d3.obs_room_unlocked(RoomId(1)) == 7
        assert rt_d3.obs_room_unlocked(RoomId(2)) == 8

    def test_obs_key_avail_d2(self, rt_d2):
        # D=2, M=4, D=2: key 0 → obs[4+2+0] = obs[6]
        assert rt_d2.obs_key_avail(KeyId(0)) == 6

    def test_obs_key_avail_d3(self, rt_d3):
        # D=3, M=6, D=3: key 0 → obs[6+3+0] = obs[9], key 1 → obs[10]
        assert rt_d3.obs_key_avail(KeyId(0)) == 9
        assert rt_d3.obs_key_avail(KeyId(1)) == 10

    def test_action_pick_d2(self, rt_d2):
        # Pick(0) = Flip(M + 0) = Flip(4)
        assert rt_d2.action_pick(KeyId(0)) == 4

    def test_action_pick_d3(self, rt_d3):
        # Pick(0) = Flip(6), Pick(1) = Flip(7)
        assert rt_d3.action_pick(KeyId(0)) == 6
        assert rt_d3.action_pick(KeyId(1)) == 7

    def test_action_move_to(self, rt_d2):
        for l in range(rt_d2.cfg.M):
            assert rt_d2.action_move_to(LocId(l)) == l


# ---------------------------------------------------------------------------
# TestASTEquivalence
# ---------------------------------------------------------------------------

class TestASTEquivalence:
    """AST fragments must match surface_compiler output exactly."""

    def test_cond_pick_ready_d2(self, rt_d2, cfg_d2):
        cond_rt = rt_d2.cond_pick_ready(KeyId(0))
        cond_sc = compile_condition(PickReady(0), cfg_d2)
        assert cond_rt.pretty() == cond_sc.pretty()

    def test_cond_pick_ready_d3(self, rt_d3, cfg_d3):
        for k in range(cfg_d3.K):
            cond_rt = rt_d3.cond_pick_ready(KeyId(k))
            cond_sc = compile_condition(PickReady(k), cfg_d3)
            assert cond_rt.pretty() == cond_sc.pretty(), f"Mismatch for k={k}"

    def test_cond_need_key_d2(self, rt_d2, cfg_d2):
        cond_rt = rt_d2.cond_need_key(KeyId(0))
        cond_sc = compile_condition(NeedKey(0), cfg_d2)
        assert cond_rt.pretty() == cond_sc.pretty()

    def test_cond_need_key_d3(self, rt_d3, cfg_d3):
        for k in range(cfg_d3.K):
            cond_rt = rt_d3.cond_need_key(KeyId(k))
            cond_sc = compile_condition(NeedKey(k), cfg_d3)
            assert cond_rt.pretty() == cond_sc.pretty(), f"Mismatch for k={k}"

    def test_flip_pick_d2(self, rt_d2, cfg_d2):
        assert rt_d2.flip_pick(KeyId(0)) == compile_action(Pick(0), cfg_d2)

    def test_flip_pick_d3(self, rt_d3, cfg_d3):
        for k in range(cfg_d3.K):
            assert rt_d3.flip_pick(KeyId(k)) == compile_action(Pick(k), cfg_d3)

    def test_flip_move_to_key_d2(self, rt_d2, cfg_d2):
        assert rt_d2.flip_move_to_key(KeyId(0)) == compile_action(MoveToKey(0), cfg_d2)

    def test_flip_move_to_key_d3(self, rt_d3, cfg_d3):
        for k in range(cfg_d3.K):
            assert rt_d3.flip_move_to_key(KeyId(k)) == compile_action(MoveToKey(k), cfg_d3)

    def test_flip_move_to_goal_d2(self, rt_d2, cfg_d2):
        assert rt_d2.flip_move_to_goal() == compile_action(MoveToGoal(), cfg_d2)

    def test_flip_move_to_goal_d3(self, rt_d3, cfg_d3):
        assert rt_d3.flip_move_to_goal() == compile_action(MoveToGoal(), cfg_d3)


# ---------------------------------------------------------------------------
# TestLiftedCompilationSmoke
# ---------------------------------------------------------------------------

def _build_lifted_policy(rt: DoorsRelationalRuntime):
    """Build the canonical policy by enumerating lockable_rooms().

    This simulates what the lifted compiler (plan <a>) will do:
    iterate rooms in order, generate PickRule + MoveRule per room,
    end with GoalRule.  Uses only relational runtime methods.
    """
    # Start with the default action (GoalRule)
    prog = Default(rt.flip_move_to_goal())

    # Build right-to-left: last room's rules wrap the default,
    # then each preceding room wraps that.
    for r in reversed(rt.lockable_rooms()):
        k = rt.key_for_room(r)
        assert k is not None

        # MoveRule(k): if need_key(k) then move_to_key(k)
        prog = Ite(rt.cond_need_key(k), rt.flip_move_to_key(k), prog)

        # PickRule(k): if pick_ready(k) then pick(k)
        prog = Ite(rt.cond_pick_ready(k), rt.flip_pick(k), prog)

    return prog


class TestLiftedCompilationSmoke:
    """Hand-built lifted compilation must produce identical ASTs."""

    def test_lifted_matches_canonical_d2(self, rt_d2, cfg_d2):
        lifted_prog = _build_lifted_policy(rt_d2)
        canonical_prog = compile_policy(canonical_policy(2), cfg_d2)
        assert lifted_prog.pretty() == canonical_prog.pretty()

    def test_lifted_matches_canonical_d3(self, rt_d3, cfg_d3):
        lifted_prog = _build_lifted_policy(rt_d3)
        canonical_prog = compile_policy(canonical_policy(3), cfg_d3)
        assert lifted_prog.pretty() == canonical_prog.pretty()

    def test_lifted_d2_solves(self, rt_d2, cfg_d2):
        prog = _build_lifted_policy(rt_d2)
        x0 = doors_initial_state(cfg_d2)
        n_sites = cfg_d2.obs_size()
        env = cfg_d2.make_env(n_sites, frozen_states=[x0])
        result = run_policy_episode(env, prog, x0=x0, is_solved=cfg_d2.is_solved)
        assert result.solved
        assert result.total_env_steps == 3

    def test_lifted_d3_solves(self, rt_d3, cfg_d3):
        prog = _build_lifted_policy(rt_d3)
        x0 = doors_initial_state(cfg_d3)
        n_sites = cfg_d3.obs_size()
        env = cfg_d3.make_env(n_sites, frozen_states=[x0])
        result = run_policy_episode(env, prog, x0=x0, is_solved=cfg_d3.is_solved)
        assert result.solved
        assert result.total_env_steps == 5

    def test_lifted_d3_node_count(self, rt_d3, cfg_d3):
        """Lifted and canonical must have the same node count."""
        lifted_prog = _build_lifted_policy(rt_d3)
        canonical_prog = compile_policy(canonical_policy(3), cfg_d3)
        assert lifted_prog.node_count() == canonical_prog.node_count()


# ---------------------------------------------------------------------------
# TestPartialMap
# ---------------------------------------------------------------------------

class TestPartialMap:
    """partial_map mode: location-dependent queries return None."""

    @pytest.fixture
    def rt_partial_d2(self, cfg_d2):
        return DoorsRelationalRuntime(cfg_d2, known_map=False)

    def test_loc_of_key_returns_none(self, rt_partial_d2):
        assert rt_partial_d2.loc_of_key(KeyId(0)) is None

    def test_cond_pick_ready_returns_none(self, rt_partial_d2):
        assert rt_partial_d2.cond_pick_ready(KeyId(0)) is None

    def test_flip_move_to_key_returns_none(self, rt_partial_d2):
        assert rt_partial_d2.flip_move_to_key(KeyId(0)) is None

    def test_static_queries_still_work(self, rt_partial_d2):
        """Config-static queries are unaffected by partial_map."""
        assert rt_partial_d2.key_for_room(RoomId(1)) == KeyId(0)
        assert rt_partial_d2.room_unlocked_by(KeyId(0)) == RoomId(1)
        assert rt_partial_d2.lockable_rooms() == [RoomId(1)]
        assert rt_partial_d2.goal_location() == LocId(3)

    def test_obs_queries_still_work(self, rt_partial_d2):
        """Obs index queries don't depend on key locations."""
        assert rt_partial_d2.obs_room_unlocked(RoomId(1)) == 5
        assert rt_partial_d2.obs_key_avail(KeyId(0)) == 6

    def test_flip_pick_still_works(self, rt_partial_d2):
        """Pick action doesn't depend on key location."""
        assert rt_partial_d2.flip_pick(KeyId(0)) == Flip(4)

    def test_flip_move_to_goal_still_works(self, rt_partial_d2):
        assert rt_partial_d2.flip_move_to_goal() == Flip(3)


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------

class TestValidation:
    """Config consistency checks."""

    def test_validate_default_d2(self, rt_d2):
        rt_d2.validate()  # should not raise

    def test_validate_default_d3(self, rt_d3):
        rt_d3.validate()  # should not raise

    def test_validate_partial_map(self, cfg_d2):
        rt = DoorsRelationalRuntime(cfg_d2, known_map=False)
        rt.validate()  # should not raise (skips key_loc checks)


# ---------------------------------------------------------------------------
# TestBoundsChecking
# ---------------------------------------------------------------------------

class TestBoundsChecking:
    """Out-of-range entity IDs raise ValueError."""

    def test_key_out_of_range(self, rt_d2):
        with pytest.raises(ValueError, match="KeyId 5"):
            rt_d2.room_unlocked_by(KeyId(5))

    def test_room_out_of_range(self, rt_d2):
        with pytest.raises(ValueError, match="RoomId 3"):
            rt_d2.obs_room_unlocked(RoomId(3))

    def test_loc_out_of_range(self, rt_d2):
        with pytest.raises(ValueError, match="LocId 10"):
            rt_d2.room_of_location(LocId(10))
