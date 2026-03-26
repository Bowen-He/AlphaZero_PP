# Doors Relational Semantics

## Purpose

This document describes the **relational runtime** — a typed, compile-time
query interface that mediates ALL access between the lifted DSL compiler and
the Doors game configuration (`DoorsGameConfig`).

The relational runtime exists because the lifted DSL uses **abstract
selectors** (like `NextLockedRoom`, `KeyFor(room)`) that contain no grounded
indices. The compiler must resolve these selectors into concrete observation
indices and action indices. The relational runtime provides the lookups that
make this resolution possible, in a typed and mode-aware way.

## State Vector Layout

The Doors observation vector has three regions:

```
state = [ agent_location | room_unlock_status | key_availability ]
          0..M-1           M..M+D-1             M+D..M+2D-2
```

| Region | Indices | Encoding | Example (D=3, M=6) |
|---|---|---|---|
| Agent location | `state[0:M]` | One-hot: exactly one bit is 1 | `state[0:6]` |
| Room unlock | `state[M:M+D]` | Binary: 1 = unlocked, 0 = locked | `state[6:9]` |
| Key availability | `state[M+D:M+2D-1]` | Binary: 1 = available, 0 = picked | `state[9:11]` |

## Observable vs Config-Static

| Field | Source | Type | Mutable at Runtime? | Notes |
|---|---|---|---|---|
| Agent location | `state[0:M]` | Observable | Yes | Changes every move |
| Room unlock status | `state[M:M+D]` | Observable | Yes (monotone) | Once unlocked, stays unlocked |
| Key availability | `state[M+D:M+2D-1]` | Observable | Yes (monotone) | Once picked, stays picked |
| Key locations | `cfg.key_loc` | Config-static | No | Fixed per instance |
| Key-room mapping | `cfg.key_unlocks` | Config-static | No | Fixed per instance |
| Goal location | `cfg.goal_loc` | Config-static | No | Fixed per instance |
| Room-location mapping | `cfg.loc_room` | Config-static | No | Fixed per instance |

**Observable** fields are part of the state vector and change during an
episode. The agent can read them directly.

**Config-static** fields are properties of the game instance, fixed before
the episode starts. In `known_map` mode, the compiler has access to these.
In `partial_map` mode, some of these (specifically key locations) are
unavailable.

## known_map vs partial_map

### known_map (default)

All config fields are available to the compiler. The relational runtime
returns concrete values for all queries:

```python
rt = DoorsRelationalRuntime(cfg, known_map=True)
rt.loc_of_key(KeyId(0))      # → LocId(1)
rt.cond_pick_ready(KeyId(0)) # → And(Not(IsZero(1)), Not(IsZero(6)))
rt.flip_move_to_key(KeyId(0))# → Flip(1)
```

The lifted compiler can fully expand all selectors at compile time, producing
a static AST that the existing interpreter evaluates.

### partial_map

Key locations (`cfg.key_loc`) are treated as unknown. Queries that depend on
key locations return `None` (⊥):

```python
rt = DoorsRelationalRuntime(cfg, known_map=False)
rt.loc_of_key(KeyId(0))      # → None  (⊥: "I don't know where key 0 is")
rt.cond_pick_ready(KeyId(0)) # → None  (can't check location if unknown)
rt.flip_move_to_key(KeyId(0))# → None  (can't move there if unknown)
```

Queries that DON'T depend on key locations still work:

```python
rt.key_for_room(RoomId(1))        # → KeyId(0)  (structural, not locational)
rt.room_unlocked_by(KeyId(0))     # → RoomId(1)
rt.obs_room_unlocked(RoomId(1))   # → ObsIdx(5)
rt.cond_need_key(KeyId(0))        # → IsZero(5)  (checks room lock, not key location)
```

The `None` return makes "unknown" an explicit case that the compiler must
handle — rather than silently using wrong data.

**Note:** `partial_map` mode defines the *interface* only. The Doors
environment itself is fully observable. A future environment variant would
be needed to actually hide key locations from the agent.

## Typed Entity IDs

The relational runtime uses `NewType` aliases to distinguish entity kinds:

| Type | Underlying | Range | Example |
|---|---|---|---|
| `RoomId` | `int` | 0..D-1 | `RoomId(1)` = room 1 |
| `KeyId` | `int` | 0..K-1 | `KeyId(0)` = key 0 |
| `LocId` | `int` | 0..M-1 | `LocId(3)` = location 3 |
| `ObsIdx` | `int` | 0..obs_size-1 | `ObsIdx(7)` = obs vector index 7 |

These are zero-cost at runtime (just `int` underneath) but make function
signatures self-documenting and catch category errors during development.

## How the Lifted Compiler Uses This

The lifted DSL has abstract selectors like `NextLockedRoom`. The compiler
expands these by enumerating entities from the relational runtime:

```python
# Lifted rule: IfThen(Pickable(KeyFor(NextLockedRoom)), Pick(KeyFor(NextLockedRoom)))
#
# Compiler expands NextLockedRoom by iterating lockable_rooms():

prog = default_prog
for r in reversed(rt.lockable_rooms()):        # [room_2, room_1]
    k = rt.key_for_room(r)                      # room_1 → key_0, room_2 → key_1
    cond = rt.cond_pick_ready(k)                # And(Not(IsZero(1)), Not(IsZero(9)))
    action = rt.flip_pick(k)                    # Flip(6)
    prog = Ite(cond, action, prog)

# Result: Ite(PickReady(0), Pick(0), Ite(PickReady(1), Pick(1), default))
# This is IDENTICAL to the grounded surface compiler output.
```

The relational runtime translates abstract intent into concrete indices,
so the lifted DSL never mentions raw numbers.
