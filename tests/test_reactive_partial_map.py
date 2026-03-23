"""Tests for partial-map reactive sketch with explicit memory.

Covers: ReactiveMemory, memory-aware predicates, no-hidden-state-leak,
partial-map episode solving, and backward compatibility.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.instances.doors.dsl.reactive_memory import ReactiveMemory
from alphazeropp.instances.doors.dsl.relational_runtime import (
    DoorsRelationalRuntime, RoomId, KeyId,
)
from alphazeropp.instances.doors.dsl.reactive_sketch_dsl import (
    canonical_reactive_policy, canonical_partial_map_policy,
    NextLockedRoomSel, KeyForSel, KnownLocP, PickableP,
    ExistsUnsearchedRoomP,
)
from alphazeropp.instances.doors.dsl.reactive_sketch_interpreter import (
    ReactiveContext, tick, run_reactive_episode, run_reactive_partial_episode,
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
    return DoorsRelationalRuntime(cfg_d2, known_map=False)


@pytest.fixture
def rt_d3(cfg_d3):
    return DoorsRelationalRuntime(cfg_d3, known_map=False)


# ---------------------------------------------------------------------------
# TestReactiveMemory
# ---------------------------------------------------------------------------

class TestReactiveMemory:

    def test_empty_memory(self):
        """Fresh memory has no known keys and no searched rooms."""
        mem = ReactiveMemory.empty()
        assert len(mem.known_key_locs) == 0
        assert len(mem.searched_rooms) == 0
        assert mem.loc_of_key(0) is None

    def test_discover_keys_in_room(self, cfg_d2):
        """Discovering room 0 for D=2 reveals key 0 at loc 1."""
        mem = ReactiveMemory.empty()
        mem.discover_keys_in_room(0, cfg_d2)
        # Key 0 is at loc 1 (default layout: key k at loc k*locs_per_room+1)
        assert mem.loc_of_key(0) == 1

    def test_discover_no_cross_room(self, cfg_d3):
        """Discovering room 0 does not reveal keys in room 1."""
        mem = ReactiveMemory.empty()
        mem.discover_keys_in_room(0, cfg_d3)
        # Key 0 at loc 1 (room 0) should be discovered
        assert mem.loc_of_key(0) == 1
        # Key 1 at loc 3 (room 1) should NOT be discovered
        assert mem.loc_of_key(1) is None

    def test_copy_independence(self, cfg_d2):
        """Mutations to copy don't affect original."""
        mem = ReactiveMemory.empty()
        mem.discover_keys_in_room(0, cfg_d2)
        mem.mark_searched(0)

        copy = mem.copy()
        copy.known_key_locs[99] = 99
        copy.searched_rooms.add(99)

        assert 99 not in mem.known_key_locs
        assert 99 not in mem.searched_rooms


# ---------------------------------------------------------------------------
# TestPartialMapPredicates
# ---------------------------------------------------------------------------

class TestPartialMapPredicates:

    def test_known_loc_false_before_discovery(self, cfg_d2, rt_d2):
        """With known_map=False and empty memory, KnownLocP returns False."""
        obs = doors_initial_state(cfg_d2)
        mem = ReactiveMemory.empty()
        ctx = ReactiveContext(rt_d2, obs, memory=mem)
        pred = KnownLocP(KeyForSel(NextLockedRoomSel()))
        assert not ctx.eval_predicate(pred)

    def test_known_loc_true_after_discovery(self, cfg_d2, rt_d2):
        """After memory has key location, KnownLocP returns True."""
        obs = doors_initial_state(cfg_d2)
        mem = ReactiveMemory.empty()
        mem.discover_keys_in_room(0, cfg_d2)
        ctx = ReactiveContext(rt_d2, obs, memory=mem)
        pred = KnownLocP(KeyForSel(NextLockedRoomSel()))
        assert ctx.eval_predicate(pred)

    def test_pickable_false_unknown_loc(self, cfg_d2, rt_d2):
        """PickableP returns False when key location is undiscovered,
        even if agent is at the key's location and key is available."""
        obs = doors_initial_state(cfg_d2).copy()
        # Move agent to loc 1 (where key 0 is)
        obs[cfg_d2.start_loc] = 0.0
        obs[cfg_d2.key_loc[0]] = 1.0
        mem = ReactiveMemory.empty()
        ctx = ReactiveContext(rt_d2, obs, memory=mem)
        pred = PickableP(KeyForSel(NextLockedRoomSel()))
        assert not ctx.eval_predicate(pred)

    def test_pickable_true_after_discovery(self, cfg_d2, rt_d2):
        """PickableP works correctly after discovery."""
        obs = doors_initial_state(cfg_d2).copy()
        # Move agent to loc 1 (where key 0 is)
        obs[cfg_d2.start_loc] = 0.0
        obs[cfg_d2.key_loc[0]] = 1.0
        mem = ReactiveMemory.empty()
        mem.discover_keys_in_room(0, cfg_d2)
        ctx = ReactiveContext(rt_d2, obs, memory=mem)
        pred = PickableP(KeyForSel(NextLockedRoomSel()))
        assert ctx.eval_predicate(pred)

    def test_exists_unsearched_room(self, cfg_d2, rt_d2):
        """Returns True when reachable unsearched rooms exist."""
        obs = doors_initial_state(cfg_d2).copy()
        # Room 0 is unlocked. Mark room 0 as searched.
        mem = ReactiveMemory.empty()
        mem.mark_searched(0)
        ctx = ReactiveContext(rt_d2, obs, memory=mem)
        pred = ExistsUnsearchedRoomP()
        # Room 1 is locked (not reachable) so no unsearched reachable room
        assert not ctx.eval_predicate(pred)

        # Unlock room 1 → now unsearched AND reachable
        obs[cfg_d2.M + 1] = 1.0
        ctx2 = ReactiveContext(rt_d2, obs, memory=mem)
        assert ctx2.eval_predicate(pred)

    def test_exists_unsearched_all_searched(self, cfg_d2, rt_d2):
        """Returns False when all reachable rooms are searched."""
        obs = doors_initial_state(cfg_d2).copy()
        obs[cfg_d2.M + 1] = 1.0  # unlock room 1
        mem = ReactiveMemory.empty()
        mem.mark_searched(0)
        mem.mark_searched(1)
        ctx = ReactiveContext(rt_d2, obs, memory=mem)
        pred = ExistsUnsearchedRoomP()
        assert not ctx.eval_predicate(pred)


# ---------------------------------------------------------------------------
# TestNoHiddenStateLeak
# ---------------------------------------------------------------------------

class TestNoHiddenStateLeak:

    def test_no_leak_different_key_placement(self):
        """Two configs with different key_loc yield same BT action
        when obs + memory are identical (proves no cfg.key_loc leakage)."""
        # Config A: key 0 at loc 0
        cfg_a = DoorsGameConfig(
            num_rooms=2, locs_per_room=2,
            key_loc=[0], key_unlocks=[1],
        )
        # Config B: key 0 at loc 1
        cfg_b = DoorsGameConfig(
            num_rooms=2, locs_per_room=2,
            key_loc=[1], key_unlocks=[1],
        )

        rt_a = DoorsRelationalRuntime(cfg_a, known_map=False)
        rt_b = DoorsRelationalRuntime(cfg_b, known_map=False)

        # Identical obs and empty memory
        obs = doors_initial_state(cfg_a)  # same layout
        mem = ReactiveMemory.empty()

        policy = canonical_partial_map_policy()

        ctx_a = ReactiveContext(rt_a, obs, memory=mem)
        ctx_b = ReactiveContext(rt_b, obs, memory=mem)

        result_a = tick(policy, ctx_a)
        result_b = tick(policy, ctx_b)

        assert result_a.action == result_b.action
        assert result_a.status == result_b.status


# ---------------------------------------------------------------------------
# TestPartialMapEpisodes
# ---------------------------------------------------------------------------

class TestPartialMapEpisodes:

    def test_canonical_partial_solves_d2(self, cfg_d2, rt_d2):
        """Canonical partial-map policy (1,2,5,3,4) solves D=2 in 3 steps."""
        policy = canonical_partial_map_policy()
        x0 = doors_initial_state(cfg_d2)
        env = cfg_d2.make_env(cfg_d2.obs_size(), frozen_states=[x0])
        result = run_reactive_partial_episode(
            env, policy, rt_d2, cfg_d2, x0=x0, is_solved=cfg_d2.is_solved,
        )
        assert result.solved
        assert result.total_env_steps == 3
        trace = [step.action for step in result.steps]
        # move_to(1)=1, pick(0)=4, move_to(3)=3
        assert trace == [1, 4, 3]

    def test_canonical_partial_trace_matches_known_map_d2(self, cfg_d2):
        """For default layout, partial-map and known-map produce same trace."""
        # Known-map run
        rt_full = DoorsRelationalRuntime(cfg_d2, known_map=True)
        policy_full = canonical_reactive_policy()
        x0 = doors_initial_state(cfg_d2)
        env_full = cfg_d2.make_env(cfg_d2.obs_size(), frozen_states=[x0])
        result_full = run_reactive_episode(
            env_full, policy_full, rt_full, x0=x0, is_solved=cfg_d2.is_solved,
        )

        # Partial-map run
        rt_partial = DoorsRelationalRuntime(cfg_d2, known_map=False)
        policy_partial = canonical_partial_map_policy()
        env_partial = cfg_d2.make_env(cfg_d2.obs_size(), frozen_states=[x0])
        result_partial = run_reactive_partial_episode(
            env_partial, policy_partial, rt_partial, cfg_d2,
            x0=x0, is_solved=cfg_d2.is_solved,
        )

        trace_full = [step.action for step in result_full.steps]
        trace_partial = [step.action for step in result_partial.steps]
        assert trace_full == trace_partial

    def test_120_permutations_enumerated_d2(self, cfg_d2, rt_d2):
        """All 120 permutations run without error and at least one solves."""
        x0 = doors_initial_state(cfg_d2)
        solve_count = 0
        for perm in itertools.permutations([1, 2, 3, 4, 5]):
            policy = canonical_partial_map_policy(perm)
            env = cfg_d2.make_env(cfg_d2.obs_size(), frozen_states=[x0])
            result = run_reactive_partial_episode(
                env, policy, rt_d2, cfg_d2, x0=x0, is_solved=cfg_d2.is_solved,
            )
            if result.solved:
                solve_count += 1
        assert solve_count > 0, "At least one permutation should solve D=2"
        # Total must be 120
        total = sum(1 for _ in itertools.permutations([1, 2, 3, 4, 5]))
        assert total == 120


# ---------------------------------------------------------------------------
# TestBackwardCompatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    def test_existing_4branch_still_works(self, cfg_d2):
        """4-branch canonical_reactive_policy() still solves D=2."""
        rt = DoorsRelationalRuntime(cfg_d2, known_map=True)
        policy = canonical_reactive_policy()
        x0 = doors_initial_state(cfg_d2)
        env = cfg_d2.make_env(cfg_d2.obs_size(), frozen_states=[x0])
        result = run_reactive_episode(
            env, policy, rt, x0=x0, is_solved=cfg_d2.is_solved,
        )
        assert result.solved
        assert result.total_env_steps == 3
