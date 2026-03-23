"""Tick-based interpreter for the reactive sketch DSL.

Provides:
  ReactiveContext  — resolves selectors and predicates against current obs
  tick(node, ctx)  — recursive BT tick returning TickResult
  run_reactive_episode(...)  — full episode loop reusing EpisodeResult
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from alphazeropp.instances.doors.dsl.relational_runtime import (
    DoorsRelationalRuntime, RoomId, KeyId, LocId,
)
from alphazeropp.instances.doors.dsl.reactive_sketch_dsl import (
    TickStatus,
    # Selectors
    NextLockedRoomSel, NextUnsearchedRoomSel,
    KeyForSel, LocOfSel, GoalLocSel, EntranceSel, FirstLocInRoomSel,
    BTRoomSel, BTKeySel, BTLocSel,
    # Predicates
    GoalReachedP, PickableP, KnownLocP, ReachableP,
    ExistsUnlockedFrontierP, ExistsUnsearchedRoomP, TrueP,
    BTPredicate,
    # Actions
    PickAction, GoToAction, NoopAction, BTAction,
    # Structural nodes
    Check, Do, Sequence, Fallback, WhileNot,
    BTNode,
)
from alphazeropp.synthesis.interpreter import StepRecord, EpisodeResult


# ---------------------------------------------------------------------------
# TickResult
# ---------------------------------------------------------------------------

@dataclass
class TickResult:
    """Result of ticking a BT node."""
    status: TickStatus
    action: int | None = None  # env action index if a Do node succeeded


# ---------------------------------------------------------------------------
# ReactiveContext — state-dependent selector/predicate resolution
# ---------------------------------------------------------------------------

class ReactiveContext:
    """Resolves BT selectors and predicates against the current observation.

    Holds a reference to the relational runtime (config queries) and the
    current observation vector (state queries).  When *memory* is provided,
    key-location queries are routed through memory instead of the runtime
    (partial-map mode).  All resolution is deterministic within a single tick.
    """

    __slots__ = ("rt", "obs", "memory")

    def __init__(self, rt: DoorsRelationalRuntime, obs: np.ndarray, memory=None):
        self.rt = rt
        self.obs = obs
        self.memory = memory  # Optional[ReactiveMemory]

    # --- Room resolution ---

    def next_locked_room(self) -> RoomId | None:
        """First lockable room that is still locked in current obs."""
        for r in self.rt.lockable_rooms():
            idx = self.rt.obs_room_unlocked(r)
            if self.obs[idx] == 0.0:
                return r
        return None

    def _has_undiscovered_key(self) -> bool:
        """True if any key's location is still unknown in memory."""
        if self.memory is None:
            return False
        for k in range(self.rt.num_keys()):
            if self.memory.loc_of_key(k) is None:
                return True
        return False

    def next_unsearched_room(self) -> RoomId | None:
        """First reachable unsearched room, only when keys remain undiscovered.

        Returns None if all key locations are already known (no reason
        to explore) or if no reachable unsearched room exists.
        """
        if self.memory is None:
            return None
        if not self._has_undiscovered_key():
            return None
        for r in range(self.rt.num_rooms()):
            if r not in self.memory.searched_rooms:
                idx = self.rt.obs_room_unlocked(RoomId(r))
                if self.obs[idx] == 1.0:
                    return RoomId(r)
        return None

    def resolve_room(self, sel: BTRoomSel) -> RoomId | None:
        if isinstance(sel, NextLockedRoomSel):
            return self.next_locked_room()
        if isinstance(sel, NextUnsearchedRoomSel):
            return self.next_unsearched_room()
        raise TypeError(f"Unknown room selector: {type(sel)}")

    # --- Key resolution ---

    def resolve_key(self, sel: BTKeySel) -> KeyId | None:
        if isinstance(sel, KeyForSel):
            r = self.resolve_room(sel.room)
            if r is None:
                return None
            return self.rt.key_for_room(r)
        raise TypeError(f"Unknown key selector: {type(sel)}")

    # --- Location resolution ---

    def resolve_loc(self, sel: BTLocSel) -> LocId | None:
        if isinstance(sel, LocOfSel):
            k = self.resolve_key(sel.key)
            if k is None:
                return None
            # In partial-map mode, route through memory
            if self.memory is not None:
                loc = self.memory.loc_of_key(int(k))
                return LocId(loc) if loc is not None else None
            return self.rt.loc_of_key(k)

        if isinstance(sel, GoalLocSel):
            return self.rt.goal_location()

        if isinstance(sel, EntranceSel):
            # Entrance of NextLockedRoom r → first location in room r-1
            # (the frontier of reachable space, since room r is locked)
            r = self.resolve_room(sel.room)
            if r is None:
                return None
            prev_room = RoomId(int(r) - 1)
            if int(prev_room) < 0:
                return None
            locs = self.rt.locations_in_room(prev_room)
            return locs[0] if locs else None

        if isinstance(sel, FirstLocInRoomSel):
            r = self.resolve_room(sel.room)
            if r is None:
                return None
            locs = self.rt.locations_in_room(r)
            return locs[0] if locs else None

        raise TypeError(f"Unknown loc selector: {type(sel)}")

    # --- Predicate evaluation ---

    def eval_predicate(self, pred: BTPredicate) -> bool:
        if isinstance(pred, GoalReachedP):
            goal = self.rt.goal_location()
            return bool(self.obs[self.rt.obs_at_loc(goal)] == 1.0)

        if isinstance(pred, PickableP):
            k = self.resolve_key(pred.key_sel)
            if k is None:
                return False
            if self.memory is not None:
                loc_val = self.memory.loc_of_key(int(k))
                loc = LocId(loc_val) if loc_val is not None else None
            else:
                loc = self.rt.loc_of_key(k)
            if loc is None:
                return False
            at_loc = bool(self.obs[self.rt.obs_at_loc(loc)] == 1.0)
            avail = bool(self.obs[self.rt.obs_key_avail(k)] == 1.0)
            return at_loc and avail

        if isinstance(pred, KnownLocP):
            k = self.resolve_key(pred.key_sel)
            if k is None:
                return False
            if self.memory is not None:
                return self.memory.loc_of_key(int(k)) is not None
            return self.rt.loc_of_key(k) is not None

        if isinstance(pred, ReachableP):
            loc = self.resolve_loc(pred.loc_sel)
            if loc is None:
                return False
            room = self.rt.room_of_location(loc)
            return bool(self.obs[self.rt.obs_room_unlocked(room)] == 1.0)

        if isinstance(pred, ExistsUnlockedFrontierP):
            return self.next_locked_room() is not None

        if isinstance(pred, ExistsUnsearchedRoomP):
            return self.next_unsearched_room() is not None

        if isinstance(pred, TrueP):
            return True

        raise TypeError(f"Unknown predicate: {type(pred)}")

    # --- Action resolution ---

    def resolve_action(self, action: BTAction) -> int | None:
        """Resolve a BT action to an env action index."""
        if isinstance(action, PickAction):
            k = self.resolve_key(action.key)
            if k is None:
                return None
            return self.rt.action_pick(k)

        if isinstance(action, GoToAction):
            loc = self.resolve_loc(action.loc)
            if loc is None:
                return None
            return self.rt.action_move_to(loc)

        if isinstance(action, NoopAction):
            return self.rt.cfg.M + self.rt.cfg.K  # noop action index

        raise TypeError(f"Unknown action: {type(action)}")


# ---------------------------------------------------------------------------
# BT tick
# ---------------------------------------------------------------------------

def tick(node: BTNode, ctx: ReactiveContext) -> TickResult:
    """Recursively tick a BT node, returning status and optional action."""
    if isinstance(node, Check):
        ok = ctx.eval_predicate(node.predicate)
        return TickResult(TickStatus.SUCCESS if ok else TickStatus.FAILURE)

    if isinstance(node, Do):
        action = ctx.resolve_action(node.action)
        if action is None:
            return TickResult(TickStatus.FAILURE)
        return TickResult(TickStatus.SUCCESS, action)

    if isinstance(node, Sequence):
        last_action = None
        for child in node.children:
            result = tick(child, ctx)
            if result.status == TickStatus.FAILURE:
                return TickResult(TickStatus.FAILURE)
            if result.action is not None:
                last_action = result.action
        return TickResult(TickStatus.SUCCESS, last_action)

    if isinstance(node, Fallback):
        for child in node.children:
            result = tick(child, ctx)
            if result.status == TickStatus.SUCCESS:
                return result
        return TickResult(TickStatus.FAILURE)

    if isinstance(node, WhileNot):
        if ctx.eval_predicate(node.predicate):
            return TickResult(TickStatus.SUCCESS)
        result = tick(node.child, ctx)
        return TickResult(TickStatus.RUNNING, result.action)

    raise TypeError(f"Unknown BT node: {type(node)}")


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_reactive_episode(
    env,
    policy: WhileNot,
    rt: DoorsRelationalRuntime,
    x0: Optional[np.ndarray] = None,
    *,
    is_solved: Callable[[np.ndarray], bool],
) -> EpisodeResult:
    """Run the reactive sketch policy on the environment.

    Each step: build ReactiveContext with current obs -> tick -> env.step.
    Reuses EpisodeResult and StepRecord from interpreter.py.

    Args:
        env: Gymnasium-compatible environment.
        policy: WhileNot root node of the reactive sketch.
        rt: DoorsRelationalRuntime for config queries.
        x0: Optional initial state override.
        is_solved: Callable to check if observation is solved.

    Returns:
        EpisodeResult with per-step records and summary statistics.
    """
    obs, _ = env.reset()
    if x0 is not None:
        env.state = x0.copy()
        obs = x0.copy()

    # NOOP action = last action index (M + K)
    noop_action = rt.cfg.M + rt.cfg.K

    steps: list[StepRecord] = []
    total_interp = 0
    cumulative_reward = 0.0
    done = False
    step_num = 0

    while not done:
        ctx = ReactiveContext(rt, obs)
        result = tick(policy, ctx)
        action = result.action if result.action is not None else noop_action

        prev_obs = obs.copy()
        obs, reward, terminated, truncated, info = env.step(action)
        step_num += 1
        total_interp += 1
        cumulative_reward += reward

        steps.append(StepRecord(
            step_num=step_num,
            state=prev_obs,
            action=action,
            reward=reward,
            interp_ops=1,
            rule_trace=[f"BT tick -> {result.status.value} -> action={action}"],
        ))

        done = terminated or truncated

    return EpisodeResult(
        steps=steps,
        total_env_steps=step_num,
        total_interp_ops=total_interp,
        final_state=obs.copy(),
        cumulative_reward=cumulative_reward,
        solved=is_solved(obs),
    )


# ---------------------------------------------------------------------------
# Partial-map episode runner
# ---------------------------------------------------------------------------

def run_reactive_partial_episode(
    env,
    policy: WhileNot,
    rt: DoorsRelationalRuntime,
    cfg,
    x0: Optional[np.ndarray] = None,
    *,
    is_solved: Callable[[np.ndarray], bool],
) -> EpisodeResult:
    """Run a partial-map reactive policy with memory updates.

    Like run_reactive_episode() but creates a ReactiveMemory and updates
    it after each step based on which room the agent enters.  The runtime
    should be created with known_map=False.

    Args:
        env: Gymnasium-compatible environment.
        policy: WhileNot root node (typically 5-branch).
        rt: DoorsRelationalRuntime (should have known_map=False).
        cfg: DoorsGameConfig — used only for discovery simulation.
        x0: Optional initial state override.
        is_solved: Callable to check if observation is solved.
    """
    from alphazeropp.instances.doors.dsl.reactive_memory import ReactiveMemory

    obs, _ = env.reset()
    if x0 is not None:
        env.state = x0.copy()
        obs = x0.copy()

    memory = ReactiveMemory.empty()
    # Discover keys in the starting room
    start_room = cfg.loc_room[cfg.start_loc]
    memory.discover_keys_in_room(start_room, cfg)
    memory.mark_searched(start_room)

    noop_action = rt.cfg.M + rt.cfg.K
    steps: list[StepRecord] = []
    total_interp = 0
    cumulative_reward = 0.0
    done = False
    step_num = 0

    while not done:
        ctx = ReactiveContext(rt, obs, memory=memory)
        result = tick(policy, ctx)
        action = result.action if result.action is not None else noop_action

        prev_obs = obs.copy()
        obs, reward, terminated, truncated, info = env.step(action)
        step_num += 1
        total_interp += 1
        cumulative_reward += reward

        # Discovery: determine agent's current room from obs
        agent_loc = int(np.argmax(obs[:cfg.M]))
        agent_room = cfg.loc_room[agent_loc]
        if agent_room not in memory.searched_rooms:
            memory.discover_keys_in_room(agent_room, cfg)
            memory.mark_searched(agent_room)

        steps.append(StepRecord(
            step_num=step_num,
            state=prev_obs,
            action=action,
            reward=reward,
            interp_ops=1,
            rule_trace=[f"BT tick -> {result.status.value} -> action={action}"],
        ))

        done = terminated or truncated

    return EpisodeResult(
        steps=steps,
        total_env_steps=step_num,
        total_interp_ops=total_interp,
        final_state=obs.copy(),
        cumulative_reward=cumulative_reward,
        solved=is_solved(obs),
    )


# ---------------------------------------------------------------------------
# Trace formatting
# ---------------------------------------------------------------------------

def format_reactive_trace(
    result: EpisodeResult,
    policy: Optional[WhileNot] = None,
) -> str:
    """Format a reactive episode trace as a human-readable string."""
    lines: list[str] = []

    if policy is not None:
        lines.append("=== Reactive Policy ===")
        lines.append(policy.pretty())
        lines.append("")

    for step in result.steps:
        state_str = "[" + ", ".join(str(int(b)) for b in step.state) + "]"
        lines.append(f"Step {step.step_num}: {state_str}")
        for rule_line in step.rule_trace:
            lines.append(f"  {rule_line}")
        lines.append(f"  reward={step.reward:+.4f}")
        lines.append("")

    final_str = "[" + ", ".join(str(int(b)) for b in result.final_state) + "]"
    status = "SOLVED" if result.solved else "NOT SOLVED"
    lines.append("=== Summary ===")
    lines.append(f"Final state: {final_str} -- {status}")
    lines.append(
        f"Env steps: {result.total_env_steps} | "
        f"Cumulative reward: {result.cumulative_reward:.4f}"
    )
    return "\n".join(lines)
