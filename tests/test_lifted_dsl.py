"""Tests for the lifted decision-list DSL and compiler."""

from __future__ import annotations

import pytest

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.instances.doors.dsl.relational_runtime import (
    DoorsRelationalRuntime, KeyId,
)
from alphazeropp.instances.doors.dsl.lifted_dsl import (
    NextLockedRoom, GoalRoom, CurrentRoom,
    KeyFor, LocOf, GoalLoc,
    Pickable, NeedKey, GoalReached,
    LiftedPick, GoTo, GoToGoal,
    IfThen, LiftedDefault, ForEachLockedRoom, LiftedPolicy,
    canonical_lifted_policy,
)
from alphazeropp.instances.doors.dsl.lifted_compiler import (
    compile_lifted_policy,
)
from alphazeropp.instances.doors.dsl.surface_compiler import compile_policy
from alphazeropp.instances.doors.dsl.surface_grammar import canonical_policy
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
    params = compute_doors_derived_params(3, 2)
    return DoorsGameConfig(num_rooms=3, locs_per_room=2, horizon=params["horizon"])


@pytest.fixture
def rt_d2(cfg_d2):
    return DoorsRelationalRuntime(cfg_d2)


@pytest.fixture
def rt_d3(cfg_d3):
    return DoorsRelationalRuntime(cfg_d3)


# ---------------------------------------------------------------------------
# TestLiftedTypes
# ---------------------------------------------------------------------------

class TestLiftedTypes:
    """DSL type construction and pretty-printing."""

    def test_canonical_policy_pretty_contains_no_indices(self):
        policy = canonical_lifted_policy()
        text = policy.pretty()
        # Must contain lifted syntax
        assert "CurrentRoom" in text
        assert "Pickable" in text
        assert "GoToGoal" in text
        assert "for each locked room" in text
        # Must NOT contain raw indices or config references
        assert "IsZero" not in text
        assert "Flip(" not in text
        assert "key_loc" not in text
        assert "cfg." not in text

    def test_canonical_policy_structure(self):
        policy = canonical_lifted_policy()
        assert len(policy.body.rules) == 2
        assert isinstance(policy.body.rules[0].predicate, Pickable)
        assert isinstance(policy.body.rules[0].action, LiftedPick)
        assert isinstance(policy.body.rules[1].predicate, NeedKey)
        assert isinstance(policy.body.rules[1].action, GoTo)
        assert isinstance(policy.default.action, GoToGoal)

    def test_selector_nesting(self):
        k = KeyFor(CurrentRoom())
        loc = LocOf(k)
        assert loc.pretty() == "LocOf(KeyFor(CurrentRoom))"
        assert k.pretty() == "KeyFor(CurrentRoom)"

    def test_rule_pretty(self):
        k = KeyFor(CurrentRoom())
        rule = IfThen(Pickable(k), LiftedPick(k))
        expected = "if Pickable(KeyFor(CurrentRoom)) then Pick(KeyFor(CurrentRoom))"
        assert rule.pretty() == expected

    def test_default_pretty(self):
        d = LiftedDefault(GoToGoal())
        assert d.pretty() == "default GoToGoal"

    def test_empty_for_each_raises(self):
        with pytest.raises(ValueError, match="at least one rule"):
            ForEachLockedRoom(rules=())

    def test_invalid_body_type_raises(self):
        with pytest.raises(TypeError, match="ForEachLockedRoom"):
            LiftedPolicy(body="not a body", default=LiftedDefault(GoToGoal()))

    def test_invalid_default_type_raises(self):
        k = KeyFor(CurrentRoom())
        body = ForEachLockedRoom(rules=(
            IfThen(Pickable(k), LiftedPick(k)),
        ))
        with pytest.raises(TypeError, match="LiftedDefault"):
            LiftedPolicy(body=body, default="not a default")

    def test_frozen(self):
        policy = canonical_lifted_policy()
        with pytest.raises(AttributeError):
            policy.default = LiftedDefault(GoToGoal())


# ---------------------------------------------------------------------------
# TestLiftedCompilation
# ---------------------------------------------------------------------------

class TestLiftedCompilation:
    """Compiled lifted policies must match surface-compiled ASTs."""

    def test_matches_canonical_d2(self, rt_d2, cfg_d2):
        lifted_prog = compile_lifted_policy(canonical_lifted_policy(), rt_d2)
        surface_prog = compile_policy(canonical_policy(2), cfg_d2)
        assert lifted_prog.pretty() == surface_prog.pretty()

    def test_matches_canonical_d3(self, rt_d3, cfg_d3):
        lifted_prog = compile_lifted_policy(canonical_lifted_policy(), rt_d3)
        surface_prog = compile_policy(canonical_policy(3), cfg_d3)
        assert lifted_prog.pretty() == surface_prog.pretty()

    def test_node_count_d2(self, rt_d2, cfg_d2):
        lifted_prog = compile_lifted_policy(canonical_lifted_policy(), rt_d2)
        surface_prog = compile_policy(canonical_policy(2), cfg_d2)
        assert lifted_prog.node_count() == surface_prog.node_count()

    def test_node_count_d3(self, rt_d3, cfg_d3):
        lifted_prog = compile_lifted_policy(canonical_lifted_policy(), rt_d3)
        surface_prog = compile_policy(canonical_policy(3), cfg_d3)
        assert lifted_prog.node_count() == surface_prog.node_count()

    def test_d1_compiles_to_default(self):
        """D=1: no lockable rooms → just the default action."""
        cfg = DoorsGameConfig(num_rooms=1, locs_per_room=2)
        rt = DoorsRelationalRuntime(cfg)
        prog = compile_lifted_policy(canonical_lifted_policy(), rt)
        assert isinstance(prog, Default)


# ---------------------------------------------------------------------------
# TestLiftedSolves
# ---------------------------------------------------------------------------

class TestLiftedSolves:
    """Compiled lifted policies must solve environments correctly."""

    def test_solves_d2_in_3_steps(self, rt_d2, cfg_d2):
        prog = compile_lifted_policy(canonical_lifted_policy(), rt_d2)
        x0 = doors_initial_state(cfg_d2)
        env = cfg_d2.make_env(cfg_d2.obs_size(), frozen_states=[x0])
        result = run_policy_episode(env, prog, x0=x0, is_solved=cfg_d2.is_solved)
        assert result.solved
        assert result.total_env_steps == 3

    def test_solves_d3_in_5_steps(self, rt_d3, cfg_d3):
        prog = compile_lifted_policy(canonical_lifted_policy(), rt_d3)
        x0 = doors_initial_state(cfg_d3)
        env = cfg_d3.make_env(cfg_d3.obs_size(), frozen_states=[x0])
        result = run_policy_episode(env, prog, x0=x0, is_solved=cfg_d3.is_solved)
        assert result.solved
        assert result.total_env_steps == 5

    def test_solves_d4_in_7_steps(self):
        """Same lifted policy works for D=4 — verifies D-generality."""
        params = compute_doors_derived_params(4, 2)
        cfg = DoorsGameConfig(num_rooms=4, locs_per_room=2, horizon=params["horizon"])
        rt = DoorsRelationalRuntime(cfg)
        prog = compile_lifted_policy(canonical_lifted_policy(), rt)
        x0 = doors_initial_state(cfg)
        env = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
        result = run_policy_episode(env, prog, x0=x0, is_solved=cfg.is_solved)
        assert result.solved
        assert result.total_env_steps == 7  # 2*(D-1) + 1

    def test_same_policy_object_for_all_d(self):
        """The SAME policy object compiles correctly for D=2, 3, 4."""
        policy = canonical_lifted_policy()
        for D in [2, 3, 4]:
            params = compute_doors_derived_params(D, 2)
            cfg = DoorsGameConfig(
                num_rooms=D, locs_per_room=2, horizon=params["horizon"],
            )
            rt = DoorsRelationalRuntime(cfg)
            prog = compile_lifted_policy(policy, rt)
            x0 = doors_initial_state(cfg)
            env = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
            result = run_policy_episode(
                env, prog, x0=x0, is_solved=cfg.is_solved,
            )
            assert result.solved, f"Failed for D={D}"
            assert result.total_env_steps == 2 * (D - 1) + 1, f"Wrong steps for D={D}"


# ---------------------------------------------------------------------------
# TestPrettyNeverEmitsIndices
# ---------------------------------------------------------------------------

class TestPrettyNeverEmitsIndices:
    """Pretty-printer must never emit raw observation or config indices."""

    def test_policy_pretty_no_raw_indices(self):
        policy = canonical_lifted_policy()
        text = policy.pretty()
        forbidden = ["cfg.", "key_loc[", "key_unlocks[", "obs[", "IsZero(", "Flip("]
        for pattern in forbidden:
            assert pattern not in text, f"Found forbidden '{pattern}' in: {text}"

    def test_all_rules_pretty_no_raw_indices(self):
        policy = canonical_lifted_policy()
        for rule in policy.body.rules:
            text = rule.pretty()
            assert "(" not in text or "KeyFor(" in text or "Pickable(" in text or \
                "NeedKey(" in text or "GoTo(" in text or "LocOf(" in text or \
                "Pick(" in text, \
                f"Unexpected parenthesized content in: {text}"
