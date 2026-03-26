"""Tests for the reactive sketch DSL and tick-based interpreter."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.instances.doors.dsl.relational_runtime import (
    DoorsRelationalRuntime, RoomId, KeyId, LocId,
)
from alphazeropp.instances.doors.dsl.reactive_sketch_dsl import (
    TickStatus,
    NextLockedRoomSel, KeyForSel, LocOfSel, GoalLocSel, EntranceSel,
    GoalReachedP, PickableP, KnownLocP, ReachableP, ExistsUnlockedFrontierP,
    PickAction, GoToAction,
    Check, Do, Sequence, Fallback, WhileNot,
    branch_pick_if_ready, branch_goto_key, branch_goto_goal,
    branch_explore_frontier, canonical_reactive_policy,
)
from alphazeropp.instances.doors.dsl.reactive_sketch_interpreter import (
    ReactiveContext, TickResult, tick, run_reactive_episode,
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
def rt_d2(cfg_d2):
    return DoorsRelationalRuntime(cfg_d2)


@pytest.fixture
def rt_d3(cfg_d3):
    return DoorsRelationalRuntime(cfg_d3)


# ---------------------------------------------------------------------------
# TestReactiveTypes — construction and pretty-printing
# ---------------------------------------------------------------------------

class TestReactiveTypes:

    def test_canonical_policy_has_4_branches(self):
        policy = canonical_reactive_policy()
        assert isinstance(policy, WhileNot)
        assert isinstance(policy.child, Fallback)
        assert len(policy.child.children) == 4

    def test_branch_order_permutation(self):
        policy = canonical_reactive_policy((3, 1, 2, 4))
        branches = policy.child.children
        # B3 is first: its Check should be ReachableP
        assert isinstance(branches[0].children[0].predicate, ReachableP)
        # B1 is second: its Check should be PickableP
        assert isinstance(branches[1].children[0].predicate, PickableP)

    def test_invalid_branch_order_raises(self):
        with pytest.raises(ValueError, match="permutation"):
            canonical_reactive_policy((1, 2, 3))

    def test_pretty_no_raw_indices(self):
        policy = canonical_reactive_policy()
        text = policy.pretty()
        assert "IsZero" not in text
        assert "Flip(" not in text
        assert "cfg." not in text
        assert "NextLockedRoom" in text
        assert "GoalReached" in text

    def test_frozen_dataclasses(self):
        policy = canonical_reactive_policy()
        with pytest.raises(AttributeError):
            policy.predicate = GoalReachedP()

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError):
            Sequence(children=())

    def test_empty_fallback_raises(self):
        with pytest.raises(ValueError):
            Fallback(children=())


# ---------------------------------------------------------------------------
# TestReactiveContext — selector/predicate resolution
# ---------------------------------------------------------------------------

class TestReactiveContext:

    def test_next_locked_room_initial_d2(self, rt_d2, cfg_d2):
        obs = doors_initial_state(cfg_d2)
        ctx = ReactiveContext(rt_d2, obs)
        assert ctx.next_locked_room() == RoomId(1)

    def test_next_locked_room_all_unlocked(self, rt_d2, cfg_d2):
        obs = doors_initial_state(cfg_d2)
        # Unlock room 1: set obs[M + 1] = 1
        obs[cfg_d2.M + 1] = 1.0
        ctx = ReactiveContext(rt_d2, obs)
        assert ctx.next_locked_room() is None

    def test_resolve_key_for_next_locked(self, rt_d2, cfg_d2):
        obs = doors_initial_state(cfg_d2)
        ctx = ReactiveContext(rt_d2, obs)
        k = ctx.resolve_key(KeyForSel(NextLockedRoomSel()))
        assert k == KeyId(0)

    def test_resolve_key_none_when_all_unlocked(self, rt_d2, cfg_d2):
        obs = doors_initial_state(cfg_d2)
        obs[cfg_d2.M + 1] = 1.0
        ctx = ReactiveContext(rt_d2, obs)
        k = ctx.resolve_key(KeyForSel(NextLockedRoomSel()))
        assert k is None

    def test_eval_pickable_at_key_loc(self, rt_d2, cfg_d2):
        obs = doors_initial_state(cfg_d2)
        # Move agent to key 0's location (loc 1)
        obs[0] = 0.0  # leave start loc
        obs[1] = 1.0  # arrive at key loc
        ctx = ReactiveContext(rt_d2, obs)
        k_sel = KeyForSel(NextLockedRoomSel())
        assert ctx.eval_predicate(PickableP(k_sel)) is True

    def test_eval_pickable_wrong_loc(self, rt_d2, cfg_d2):
        obs = doors_initial_state(cfg_d2)
        ctx = ReactiveContext(rt_d2, obs)
        k_sel = KeyForSel(NextLockedRoomSel())
        # Agent at loc 0, key at loc 1 → not pickable
        assert ctx.eval_predicate(PickableP(k_sel)) is False

    def test_eval_reachable_goal_locked(self, rt_d2, cfg_d2):
        obs = doors_initial_state(cfg_d2)
        ctx = ReactiveContext(rt_d2, obs)
        # Goal is in room 1, which is locked
        assert ctx.eval_predicate(ReachableP(GoalLocSel())) is False

    def test_eval_reachable_goal_unlocked(self, rt_d2, cfg_d2):
        obs = doors_initial_state(cfg_d2)
        obs[cfg_d2.M + 1] = 1.0  # unlock room 1
        ctx = ReactiveContext(rt_d2, obs)
        assert ctx.eval_predicate(ReachableP(GoalLocSel())) is True

    def test_eval_exists_frontier_initial(self, rt_d2, cfg_d2):
        obs = doors_initial_state(cfg_d2)
        ctx = ReactiveContext(rt_d2, obs)
        assert ctx.eval_predicate(ExistsUnlockedFrontierP()) is True

    def test_eval_exists_frontier_all_unlocked(self, rt_d2, cfg_d2):
        obs = doors_initial_state(cfg_d2)
        obs[cfg_d2.M + 1] = 1.0
        ctx = ReactiveContext(rt_d2, obs)
        assert ctx.eval_predicate(ExistsUnlockedFrontierP()) is False

    def test_eval_goal_reached(self, rt_d2, cfg_d2):
        obs = doors_initial_state(cfg_d2)
        ctx = ReactiveContext(rt_d2, obs)
        assert ctx.eval_predicate(GoalReachedP()) is False
        # Place agent at goal
        obs[cfg_d2.goal_loc] = 1.0
        ctx2 = ReactiveContext(rt_d2, obs)
        assert ctx2.eval_predicate(GoalReachedP()) is True

    def test_eval_known_loc(self, rt_d2, cfg_d2):
        obs = doors_initial_state(cfg_d2)
        ctx = ReactiveContext(rt_d2, obs)
        k_sel = KeyForSel(NextLockedRoomSel())
        # known_map=True → always True
        assert ctx.eval_predicate(KnownLocP(k_sel)) is True

    def test_resolve_entrance(self, rt_d2, cfg_d2):
        obs = doors_initial_state(cfg_d2)
        ctx = ReactiveContext(rt_d2, obs)
        # NextLockedRoom=1, Entrance → first loc in room 0
        loc = ctx.resolve_loc(EntranceSel(NextLockedRoomSel()))
        assert loc == LocId(0)

    def test_resolve_entrance_d3(self, rt_d3, cfg_d3):
        obs = doors_initial_state(cfg_d3)
        ctx = ReactiveContext(rt_d3, obs)
        # NextLockedRoom=1, Entrance → first loc in room 0
        loc = ctx.resolve_loc(EntranceSel(NextLockedRoomSel()))
        assert loc == LocId(0)
        # Unlock room 1 → NextLockedRoom=2, Entrance → first loc in room 1
        obs[cfg_d3.M + 1] = 1.0
        ctx2 = ReactiveContext(rt_d3, obs)
        loc2 = ctx2.resolve_loc(EntranceSel(NextLockedRoomSel()))
        assert loc2 == LocId(2)


# ---------------------------------------------------------------------------
# TestReactiveSolves — end-to-end episode solving
# ---------------------------------------------------------------------------

class TestReactiveSolves:

    def test_canonical_order_solves_d2_in_3_steps(self, rt_d2, cfg_d2):
        policy = canonical_reactive_policy()
        x0 = doors_initial_state(cfg_d2)
        env = cfg_d2.make_env(cfg_d2.obs_size(), frozen_states=[x0])
        result = run_reactive_episode(
            env, policy, rt_d2, x0=x0, is_solved=cfg_d2.is_solved,
        )
        assert result.solved
        assert result.total_env_steps == 3

    def test_canonical_order_solves_d3_in_5_steps(self, rt_d3, cfg_d3):
        policy = canonical_reactive_policy()
        x0 = doors_initial_state(cfg_d3)
        env = cfg_d3.make_env(cfg_d3.obs_size(), frozen_states=[x0])
        result = run_reactive_episode(
            env, policy, rt_d3, x0=x0, is_solved=cfg_d3.is_solved,
        )
        assert result.solved
        assert result.total_env_steps == 5

    def test_canonical_order_solves_d4_in_7_steps(self):
        params = compute_doors_derived_params(4, 2)
        cfg = DoorsGameConfig(num_rooms=4, locs_per_room=2, horizon=params["horizon"])
        rt = DoorsRelationalRuntime(cfg)
        policy = canonical_reactive_policy()
        x0 = doors_initial_state(cfg)
        env = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
        result = run_reactive_episode(
            env, policy, rt, x0=x0, is_solved=cfg.is_solved,
        )
        assert result.solved
        assert result.total_env_steps == 7  # 2*(D-1) + 1

    def test_same_policy_works_for_all_d(self):
        """Same policy object solves D=2, 3, 4 with different runtimes."""
        policy = canonical_reactive_policy()
        for D in [2, 3, 4]:
            params = compute_doors_derived_params(D, 2)
            cfg = DoorsGameConfig(
                num_rooms=D, locs_per_room=2, horizon=params["horizon"],
            )
            rt = DoorsRelationalRuntime(cfg)
            x0 = doors_initial_state(cfg)
            env = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
            result = run_reactive_episode(
                env, policy, rt, x0=x0, is_solved=cfg.is_solved,
            )
            assert result.solved, f"Failed for D={D}"
            assert result.total_env_steps == 2 * (D - 1) + 1, f"Wrong steps for D={D}"

    def test_all_24_permutations_d2(self, rt_d2, cfg_d2):
        """At least one permutation solves D=2. Record which ones solve."""
        x0 = doors_initial_state(cfg_d2)
        solved_orders = []
        for perm in itertools.permutations([1, 2, 3, 4]):
            policy = canonical_reactive_policy(perm)
            env = cfg_d2.make_env(cfg_d2.obs_size(), frozen_states=[x0])
            result = run_reactive_episode(
                env, policy, rt_d2, x0=x0, is_solved=cfg_d2.is_solved,
            )
            if result.solved:
                solved_orders.append(perm)
        assert len(solved_orders) > 0, "No permutation solved D=2"
        # Canonical order must be among them
        assert (1, 2, 3, 4) in solved_orders


# ---------------------------------------------------------------------------
# TestBranchOrderMatters — branch ordering affects behavior
# ---------------------------------------------------------------------------

class TestBranchOrderMatters:

    def test_deterministic(self, rt_d2, cfg_d2):
        """Same order always produces same trace."""
        policy = canonical_reactive_policy((2, 1, 3, 4))
        x0 = doors_initial_state(cfg_d2)
        results = []
        for _ in range(3):
            env = cfg_d2.make_env(cfg_d2.obs_size(), frozen_states=[x0])
            r = run_reactive_episode(
                env, policy, rt_d2, x0=x0, is_solved=cfg_d2.is_solved,
            )
            results.append((r.solved, r.total_env_steps))
        assert all(r == results[0] for r in results)

    def test_order_2134_fails_d2(self, rt_d2, cfg_d2):
        """B2 before B1 fails: KnownLoc is trivially true (known_map),
        so B2 always fires and the agent never picks the key."""
        policy = canonical_reactive_policy((2, 1, 3, 4))
        x0 = doors_initial_state(cfg_d2)
        env = cfg_d2.make_env(cfg_d2.obs_size(), frozen_states=[x0])
        result = run_reactive_episode(
            env, policy, rt_d2, x0=x0, is_solved=cfg_d2.is_solved,
        )
        assert not result.solved  # B2 blocks B1 from ever picking

    def test_goal_first_order_still_solves_d2(self, rt_d2, cfg_d2):
        """Order (3,1,2,4): goal branch fires only when reachable."""
        policy = canonical_reactive_policy((3, 1, 2, 4))
        x0 = doors_initial_state(cfg_d2)
        env = cfg_d2.make_env(cfg_d2.obs_size(), frozen_states=[x0])
        result = run_reactive_episode(
            env, policy, rt_d2, x0=x0, is_solved=cfg_d2.is_solved,
        )
        # B3 fails initially (goal unreachable), falls through to B1/B2
        assert result.solved
