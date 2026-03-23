# The Relational Runtime: What It Is, Why It Exists, and How It Works

**Date:** 2026-03-21
**Branch:** `feature/grammar-redesign`
**Files:** `relational_runtime.py`, `test_relational_runtime.py`, `docs/doors_relational_semantics.md`

---

## 1. What Changed

### Files created

| File | What it is |
|---|---|
| `src/alphazeropp/instances/doors/dsl/relational_runtime.py` | A Python class (`DoorsRelationalRuntime`) that wraps the game configuration and provides typed lookup functions |
| `tests/test_relational_runtime.py` | 52 tests verifying the new module produces identical results to the old compiler |
| `docs/doors_relational_semantics.md` | Documentation of the state vector layout and observable vs config-static fields |

### Files modified

| File | Change |
|---|---|
| `src/alphazeropp/instances/doors/dsl/__init__.py` | Added exports for the new module |

### What was NOT changed

The existing compiler (`surface_compiler.py`) was not modified. The existing DSL files were not modified. The existing tests all still pass. The relational runtime sits alongside the existing code — it adds a new way to do the same thing, without breaking the old way.

### Before vs after, side by side

Here is a concrete comparison. Suppose the compiler needs to generate the AST condition for "is key 0 pickable?" in a D=2 game.

**Before (surface_compiler.py, line 42-47):**
```python
# The compiler reads raw config fields directly:
def compile_condition(cond, cfg):
    if isinstance(cond, PickReady):
        return And(
            Not(IsZero(cfg.key_loc[cond.k])),     # cfg.key_loc[0] = 1
            Not(IsZero(cfg.M + cfg.D + cond.k)),   # cfg.M + cfg.D + 0 = 4+2+0 = 6
        )
# Result: And(Not(IsZero(1)), Not(IsZero(6)))
```

**After (relational_runtime.py, line 182-190):**
```python
# The relational runtime provides typed methods:
rt = DoorsRelationalRuntime(cfg)
cond = rt.cond_pick_ready(KeyId(0))
# Internally:
#   loc = rt.loc_of_key(KeyId(0))        → LocId(1)
#   rt.cond_at_loc(LocId(1))             → Not(IsZero(1))
#   rt.cond_key_avail(KeyId(0))          → Not(IsZero(6))
#   And(Not(IsZero(1)), Not(IsZero(6)))
# Result: And(Not(IsZero(1)), Not(IsZero(6)))
```

**The output is identical.** Both produce `And(Not(IsZero(1)), Not(IsZero(6)))`. The difference is in how the compiler gets there: direct field access vs typed method calls.

---

## 1A. Deep Dive: The Before/After Code, Line by Line

### The old compiler code, token by token

Here is the old code again. Below it, every single piece is explained.

```python
def compile_condition(cond, cfg):
    if isinstance(cond, PickReady):
        return And(
            Not(IsZero(cfg.key_loc[cond.k])),
            Not(IsZero(cfg.M + cfg.D + cond.k)),
        )
```

| Token | What it is | What it means in plain English |
|---|---|---|
| `def compile_condition(cond, cfg):` | Function definition | "Define a function called `compile_condition` that takes two inputs: `cond` (the condition to compile) and `cfg` (the game configuration)" |
| `cond` | A surface DSL object | The high-level condition we want to translate. In this case, `PickReady(0)` — "is key 0 ready to pick up?" |
| `cfg` | A `DoorsGameConfig` object | A data structure holding all the facts about this particular game layout: where keys are, which rooms they unlock, etc. |
| `if isinstance(cond, PickReady):` | Type check | "If the condition is of the PickReady type (i.e., checking whether a key is pickable)..." |
| `return And(...)` | Return an AST node | "Produce an `And` node — a boolean AND that requires both sub-conditions to be true" |
| `cfg.key_loc[cond.k]` | **Field access** | "Go into `cfg`, find its `key_loc` array, and look up the entry at index `cond.k`." If `cond.k` is 0, this reads `cfg.key_loc[0]`, which equals 1 (key 0 is at location 1). |
| `Not(IsZero(...))` | AST node construction | "Create an AST node that checks: is this observation index NOT zero?" `Not(IsZero(1))` means "obs[1] ≠ 0", which means "the agent IS at location 1." |
| `cfg.M + cfg.D + cond.k` | **Arithmetic on config fields** | "Take M (number of locations = 4), add D (number of rooms = 2), add the key index (0). Result: 6." Index 6 in the observation vector is the "key 0 available?" bit. |

**What is "field access"?**

A "field" is a named piece of data stored inside an object. `DoorsGameConfig` has several fields:

```python
cfg.key_loc      # a list: [1]         ← "key 0 is at location 1"
cfg.key_unlocks  # a list: [1]         ← "key 0 unlocks room 1"
cfg.M            # an integer: 4       ← "there are 4 locations"
cfg.D            # an integer: 2       ← "there are 2 rooms"
cfg.goal_loc     # an integer: 3       ← "the goal is at location 3"
```

"Field access" means reaching into the object and reading one of these values directly. When the compiler writes `cfg.key_loc[cond.k]`, it is:

1. Reaching into `cfg`
2. Finding the `key_loc` field (a list of key locations)
3. Indexing into that list with `cond.k` (the key number)
4. Getting back a raw integer (1)

This is like opening someone's filing cabinet, finding the drawer labeled "key_loc", and pulling out the file at position 0. You need to know the drawer's name and what the numbers inside it mean.

### The new relational runtime code, token by token

```python
rt = DoorsRelationalRuntime(cfg)
cond = rt.cond_pick_ready(KeyId(0))
```

| Token | What it is | What it means in plain English |
|---|---|---|
| `rt = DoorsRelationalRuntime(cfg)` | Create the runtime | "Create a relational runtime object that wraps the game configuration. From now on, we'll ask `rt` questions instead of digging into `cfg` directly." |
| `KeyId(0)` | A **typed** value | "The number 0, labeled as a key identifier." Not just `0` — explicitly `KeyId(0)` so it's clear we mean key number 0, not room 0 or location 0. |
| `rt.cond_pick_ready(KeyId(0))` | A **typed method call** | "Ask the runtime: give me the AST condition for 'is key 0 pickable?'" The runtime figures out the observation indices internally. |

**What is a "typed method call"?**

A "method" is a function that belongs to an object. `rt.cond_pick_ready(...)` is calling the method `cond_pick_ready` on the runtime object `rt`.

"Typed" means the input is labeled with what KIND of thing it is. `KeyId(0)` is not just the number 0 — it's "key number 0." If you accidentally passed `RoomId(0)` instead, the types wouldn't match, signaling a likely bug.

Here is what happens inside `rt.cond_pick_ready(KeyId(0))`, step by step:

```
Step 1: rt.loc_of_key(KeyId(0))
        "Where is key 0?"
        → Looks up cfg.key_loc[0] internally
        → Returns LocId(1)  — "location 1", with the Location type label

Step 2: rt.cond_at_loc(LocId(1))
        "Generate the AST condition for: is the agent at location 1?"
        → rt.obs_at_loc(LocId(1)) → ObsIdx(1)  — "observation index 1"
        → Not(IsZero(1))  — "obs[1] ≠ 0"

Step 3: rt.cond_key_avail(KeyId(0))
        "Generate the AST condition for: is key 0 available?"
        → rt.obs_key_avail(KeyId(0)) → ObsIdx(6)  — "observation index 6"
        → Not(IsZero(6))  — "obs[6] ≠ 0"

Step 4: And(step 2, step 3)
        → And(Not(IsZero(1)), Not(IsZero(6)))
```

### The filing cabinet vs the librarian

**Field access (old way)** is like going to a filing cabinet yourself:

```
You:      "I need to know where key 0 is."
You:      *opens cabinet*
You:      *finds drawer labeled 'key_loc'*
You:      *pulls out card at position 0*
You:      *reads: 1*
You:      "OK, it's at location 1."
You:      "Now I need the observation index... that's M + D + k = 4 + 2 + 0 = 6."
```

You need to know the cabinet's internal organization (which drawer, what the numbers mean, the formula M + D + k).

**Typed method calls (new way)** is like asking a librarian:

```
You:      "Where is key 0?"
Librarian: "Location 1."  (hands you a card labeled LocId(1))

You:      "What observation index tells me if key 0 is available?"
Librarian: "Index 6."  (hands you a card labeled ObsIdx(6))
```

You don't need to know the cabinet's internal organization. The librarian (relational runtime) handles that. And the cards are labeled (typed) so you can't accidentally confuse a location with a room number.

### What is wrong with reading fields from DoorsGameConfig directly?

There are three problems:

**Problem 1: The code doesn't explain itself.**

```python
# Old way:
Not(IsZero(cfg.key_loc[cond.k]))
```

To understand this line, you need to know:
- `cfg.key_loc` is an array of key locations
- `cond.k` is a key index
- `cfg.key_loc[0]` is 1 for D=2 (but different for other layouts)
- `IsZero(1)` checks observation bit 1
- `Not(IsZero(1))` means "agent IS at location 1"

None of this is visible in the code. You need a mental model of the config's internal structure.

```python
# New way:
rt.cond_at_loc(rt.loc_of_key(KeyId(0)))
```

This reads almost like English: "condition: at location of key 0." The code explains what it's doing.

**Problem 2: It's easy to confuse different kinds of integers.**

In the old code, `cfg.key_loc[0] = 1` and `cfg.key_unlocks[0] = 1` both return the number 1, but they mean completely different things:
- `cfg.key_loc[0] = 1` → "key 0 is at **location** 1"
- `cfg.key_unlocks[0] = 1` → "key 0 unlocks **room** 1"

If you accidentally swap them, you get a program that checks the wrong observation index. The code will run without errors but produce wrong behavior. This is a silent bug.

In the new code, `LocId(1)` and `RoomId(1)` are different types. If a function expects a `LocId` and you pass a `RoomId`, your IDE flags it. The types prevent the confusion.

**Problem 3: It makes the lifted DSL impossible.**

The future lifted DSL has abstract selectors like `NextLockedRoom` — "whichever room is locked next." To compile this, the compiler must iterate over all lockable rooms.

With direct field access, the lifted compiler would need to know the internal structure of `DoorsGameConfig`:

```python
# Lifted compiler with direct access (hypothetical — BAD):
for r in range(1, cfg.D):                    # "iterate rooms 1 to D-1"
    k = ???  # need inverse of cfg.key_unlocks... manually build it?
    loc = cfg.key_loc[k]                     # reach into cfg
    obs_idx = cfg.M + cfg.D + k              # do arithmetic
```

The lifted compiler would need to re-derive all the same formulas and inverse maps that the surface compiler already has. This duplicates logic and makes both compilers fragile.

With the relational runtime, the lifted compiler just asks questions:

```python
# Lifted compiler with relational runtime (GOOD):
for r in rt.lockable_rooms():                # "which rooms need keys?"
    k = rt.key_for_room(r)                    # "which key opens this room?"
    cond = rt.cond_pick_ready(k)              # "AST condition for pickability"
    action = rt.flip_pick(k)                  # "AST action for picking"
```

All the config-reading logic is in one place (the relational runtime). Both compilers — old and new — can use it.

---

## 1B. Lifted vs Non-Lifted: Explicit Comparison

### The same strategy, three ways

Consider the optimal strategy for D=2 (2 rooms, 1 key). In English:

```
1. If I'm at the key and it's available → pick it up
2. If the next room is locked → go to the key
3. Otherwise → go to the goal
```

Here is this SAME strategy expressed in three different DSLs, from most grounded to most lifted:

#### Non-lifted (current surface DSL)

```python
# SOURCE CODE:
PickRule(0)       # "if PickReady(key 0) then Pick(key 0)"
MoveRule(0)       # "if NeedKey(key 0) then MoveToKey(key 0)"
GoalRule          # "else MoveToGoal"

# WHAT THE COMPILER DOES:
# Reads cfg.key_loc[0] = 1, cfg.key_unlocks[0] = 1, cfg.M = 4, cfg.D = 2
# Produces AST directly using these numbers

# COMPILED AST:
Ite(And(Not(IsZero(1)), Not(IsZero(6))), Flip(4),
    Ite(IsZero(5), Flip(1),
        Default(Flip(3))))
```

The source code says `PickRule(0)` — naming key 0 explicitly. The program only works for game layouts where key 0 exists and is configured the way the compiler expects.

#### Lifted (future lifted DSL)

```python
# SOURCE CODE:
IfThen(Pickable(KeyFor(NextLockedRoom)),  Pick(KeyFor(NextLockedRoom)))
IfThen(KnownLoc(KeyFor(NextLockedRoom)),  GoTo(Loc(KeyFor(NextLockedRoom))))
Default(GoToGoal)

# WHAT THE COMPILER DOES:
# Asks rt.lockable_rooms() → [RoomId(1)]
# For room 1: asks rt.key_for_room(RoomId(1)) → KeyId(0)
# Then asks rt.cond_pick_ready(KeyId(0)) → And(Not(IsZero(1)), Not(IsZero(6)))
# And rt.flip_pick(KeyId(0)) → Flip(4)
# etc.

# COMPILED AST:
Ite(And(Not(IsZero(1)), Not(IsZero(6))), Flip(4),
    Ite(IsZero(5), Flip(1),
        Default(Flip(3))))
```

The source code says `KeyFor(NextLockedRoom)` — no key number anywhere. The SAME source code works for D=2, D=3, D=4, or D=100, because "the key for the next locked room" is a description that the compiler expands differently for each game size.

#### Side by side: how the same strategy scales from D=2 to D=3

**Non-lifted — you must rewrite the program:**

```
D=2 program:                          D=3 program:
  PickRule(0)                           PickRule(0)
  MoveRule(0)                           MoveRule(0)
  GoalRule                              PickRule(1)       ← NEW LINE
                                        MoveRule(1)       ← NEW LINE
                                        GoalRule

The D=2 program has 3 rules.
The D=3 program has 5 rules.
They are DIFFERENT programs.
```

**Lifted — the same program works for both:**

```
D=2 AND D=3 (same program):
  IfThen(Pickable(KeyFor(NextLockedRoom)),  Pick(KeyFor(NextLockedRoom)))
  IfThen(KnownLoc(KeyFor(NextLockedRoom)),  GoTo(Loc(KeyFor(NextLockedRoom))))
  Default(GoToGoal)

3 rules. Same program.
The compiler generates 3 Ite nodes for D=2, or 5 Ite nodes for D=3.
The programmer doesn't change anything.
```

### Advantages of lifting

| Property | Non-lifted (grounded) | Lifted |
|---|---|---|
| **Program size** | Grows with game size (2K+1 rules for K keys) | Fixed (3 rules regardless of game size) |
| **Transfer** | Cannot reuse programs across game sizes | Same program works for any game size |
| **Search space** | Different search space per game size | One search space for all sizes |
| **Readability** | Must know what key 0 means in this layout | Describes strategy in domain terms |
| **Semantic honesty** | Looks abstract but depends on hidden config | Truly abstract — no hidden data |

**The key advantage: search space reduction.** In program synthesis, the search algorithm explores programs to find one that solves the puzzle. With a grounded DSL, the search space for D=3 is different from D=2 — the algorithm starts from scratch for each game size. With a lifted DSL, the same search finds programs that work for ALL sizes.

---

## 1C. Why More Abstract = Harder to Compile

### The non-lifted compiler is simple

For `PickRule(0)`, the non-lifted compiler does one thing:

```
Input:  PickRule(0)
Look up: cfg.key_loc[0] = 1, obs index for key avail = 6
Output: Ite(And(Not(IsZero(1)), Not(IsZero(6))), Flip(4), else_prog)

That's it. One rule → one Ite node. Direct translation.
```

### The lifted compiler must do more work

For `IfThen(Pickable(KeyFor(NextLockedRoom)), Pick(KeyFor(NextLockedRoom)))`, the lifted compiler must:

**Step 1: Figure out what `NextLockedRoom` can be.**
```
Ask rt.lockable_rooms() → [RoomId(1), RoomId(2)]  (for D=3)
Answer: NextLockedRoom can be room 1 or room 2.
```

**Step 2: For each possible room, resolve the nested selectors.**
```
Room 1:
  KeyFor(Room 1) → rt.key_for_room(RoomId(1)) → KeyId(0)
  Pickable(KeyId(0)) → rt.cond_pick_ready(KeyId(0)) → And(Not(IsZero(1)), Not(IsZero(9)))
  Pick(KeyId(0)) → rt.flip_pick(KeyId(0)) → Flip(6)

Room 2:
  KeyFor(Room 2) → rt.key_for_room(RoomId(2)) → KeyId(1)
  Pickable(KeyId(1)) → rt.cond_pick_ready(KeyId(1)) → And(Not(IsZero(3)), Not(IsZero(10)))
  Pick(KeyId(1)) → rt.flip_pick(KeyId(1)) → Flip(7)
```

**Step 3: Chain the resolved cases into a multi-branch AST.**
```
Ite(And(Not(IsZero(1)), Not(IsZero(9))), Flip(6),     ← room 1's key
    Ite(And(Not(IsZero(3)), Not(IsZero(10))), Flip(7), ← room 2's key
        else_prog))
```

The non-lifted compiler translates `PickRule(0)` into 1 Ite node in one step.
The lifted compiler translates one lifted rule into 2 Ite nodes (for D=3) in three steps: enumerate, resolve, chain.

**Why is this harder?**
- The lifted compiler must **enumerate** — "how many rooms could NextLockedRoom be?" This requires the relational runtime.
- The lifted compiler must **resolve nested selectors** — `KeyFor(NextLockedRoom)` is a composition, not a direct lookup. It must resolve `NextLockedRoom` first, then look up `KeyFor` on the result.
- The lifted compiler must **handle the unknown case** — in `partial_map` mode, `rt.loc_of_key()` returns None. The compiler must decide what to do when a selector can't be resolved. The non-lifted compiler never faces this because it always has concrete indices.

The tradeoff is worth it: harder compilation, but the resulting programs are general and the search space is smaller.

---

## 1D. Multi-Stage Compilation: Exactly How It Works

### What "multi-stage" means

"Multi-stage" means the translation from source program to executed actions doesn't happen all at once. It happens in two separate stages, at two separate times, using two separate pieces of information:

```
STAGE 1: COMPILE TIME                    STAGE 2: RUN TIME
─────────────────────                    ────────────────────
When: Before the game starts             When: Every step during gameplay
Input: Game layout (cfg)                 Input: Current observation (state vector)
Who: The compiler + relational runtime   Who: The interpreter
Output: An AST (tree of Ite/IsZero/Flip) Output: An action (which button to press)
```

### Worked example: D=2, full trace through both stages

#### The game layout (known before the game starts)

```
Rooms: Room 0 (open), Room 1 (locked)
Locations: 0, 1, 2, 3
Key 0: at location 1, unlocks room 1
Goal: at location 3
Agent starts at: location 0

Observation vector (7 elements):
  [at_0, at_1, at_2, at_3, room0_open, room1_open, key0_avail]
    0     1     2     3       4           5            6
```

#### STAGE 1: Compile time (happens ONCE, before the game)

The compiler has the game layout. It produces an AST.

```
Source program:  PickRule(0), MoveRule(0), GoalRule

Compiler asks relational runtime:

  Q: "Which rooms need keys?"
  A: [RoomId(1)]

  Q: "Which key opens room 1?"
  A: KeyId(0)

  Q: "AST condition for: is key 0 pickable?"
  A: And(Not(IsZero(1)), Not(IsZero(6)))
     Translation: "obs[1] ≠ 0 AND obs[6] ≠ 0"
     Meaning: "agent at location 1 AND key 0 available"

  Q: "AST action for: pick key 0?"
  A: Flip(4)
     Meaning: "select action 4" (which the game engine interprets as "pick")

  Q: "AST condition for: does key 0's room need unlocking?"
  A: IsZero(5)
     Meaning: "obs[5] = 0" (room 1 is still locked)

  Q: "AST action for: move to key 0?"
  A: Flip(1)
     Meaning: "select action 1" (move to location 1, where key 0 is)

  Q: "AST action for: move to goal?"
  A: Flip(3)
     Meaning: "select action 3" (move to location 3, the goal)

Compiler chains these into an AST:

  Ite(And(Not(IsZero(1)), Not(IsZero(6))), Flip(4),
      Ite(IsZero(5), Flip(1),
          Default(Flip(3))))

In English:
  IF (agent at loc 1 AND key 0 available) THEN pick key 0
  ELIF (room 1 is locked) THEN move to loc 1
  ELSE move to goal

STAGE 1 IS DONE. The AST is frozen. It will not change.
The relational runtime is no longer needed.
```

#### STAGE 2: Run time (happens EVERY STEP during the game)

The interpreter has the frozen AST and the current game state. Each step, it walks the AST and picks an action.

```
═══════════════════════════════════════════════════════════
STEP 1: Agent at location 0, room 1 locked, key 0 available
═══════════════════════════════════════════════════════════

State vector: [1, 0, 0, 0, 1, 0, 1]
               ^              ^  ^  ^
               at loc 0       |  |  key 0 available
                       room 0 open  room 1 locked

Interpreter walks the AST:

  Node: Ite(And(Not(IsZero(1)), Not(IsZero(6))), Flip(4), ...)
  Check condition: And(Not(IsZero(1)), Not(IsZero(6)))
    IsZero(1): is state[1] = 0?  state[1] = 0.  YES, IsZero is TRUE.
    Not(TRUE) = FALSE.
    Short-circuit: And(FALSE, ...) = FALSE.  (Don't even check the second part.)
  Condition is FALSE → go to else branch.

  Node: Ite(IsZero(5), Flip(1), ...)
  Check condition: IsZero(5)
    Is state[5] = 0?  state[5] = 0.  YES.
  Condition is TRUE → return Flip(1).

  ACTION: 1 (move to location 1)

  Game engine moves agent to location 1.

═══════════════════════════════════════════════════════════
STEP 2: Agent at location 1, room 1 locked, key 0 available
═══════════════════════════════════════════════════════════

State vector: [0, 1, 0, 0, 1, 0, 1]
                  ^              ^
                  at loc 1       key 0 available

Interpreter walks the SAME AST:

  Node: Ite(And(Not(IsZero(1)), Not(IsZero(6))), Flip(4), ...)
  Check condition:
    IsZero(1): is state[1] = 0?  state[1] = 1.  NO, IsZero is FALSE.
    Not(FALSE) = TRUE.
    IsZero(6): is state[6] = 0?  state[6] = 1.  NO, IsZero is FALSE.
    Not(FALSE) = TRUE.
    And(TRUE, TRUE) = TRUE.
  Condition is TRUE → return Flip(4).

  ACTION: 4 (pick up key 0)

  Game engine: agent picks key 0. Room 1 unlocks.

═══════════════════════════════════════════════════════════
STEP 3: Agent at location 1, room 1 UNLOCKED, key 0 PICKED
═══════════════════════════════════════════════════════════

State vector: [0, 1, 0, 0, 1, 1, 0]
                  ^           ^  ^
                  at loc 1    |  room 1 now OPEN
                              key 0 no longer available

Interpreter walks the SAME AST:

  Node: Ite(And(Not(IsZero(1)), Not(IsZero(6))), Flip(4), ...)
  Check condition:
    IsZero(1): state[1] = 1.  FALSE.  Not(FALSE) = TRUE.
    IsZero(6): state[6] = 0.  TRUE.   Not(TRUE) = FALSE.
    And(TRUE, FALSE) = FALSE.
  Condition is FALSE → go to else.

  Node: Ite(IsZero(5), Flip(1), ...)
  Check condition: IsZero(5)
    Is state[5] = 0?  state[5] = 1.  NO. (Room 1 is now unlocked!)
  Condition is FALSE → go to else.

  Node: Default(Flip(3))
  No condition to check → return Flip(3).

  ACTION: 3 (move to goal)

  Game engine: agent moves to goal. PUZZLE SOLVED!

═══════════════════════════════════════════════════════════
SUMMARY
═══════════════════════════════════════════════════════════

Step 1: Move to key (action 1)     ← IsZero(5) was true (room locked)
Step 2: Pick key    (action 4)     ← PickReady was true (at key AND key available)
Step 3: Move to goal (action 3)    ← both conditions false, fell through to default

Total: 3 steps. Puzzle solved.
```

### What makes this "multi-stage"

The key insight is that **different information is available at different times**:

| Information | Available at compile time? | Available at run time? | Used by |
|---|---|---|---|
| Where is key 0? (location 1) | YES (from config) | Not needed | Compiler (to generate IsZero(1)) |
| Which key opens room 1? (key 0) | YES (from config) | Not needed | Compiler (to select which rules) |
| Is the agent at location 1 right now? | NO (game hasn't started) | YES (from state vector) | Interpreter (to evaluate IsZero(1)) |
| Is room 1 locked right now? | NO (game hasn't started) | YES (from state vector) | Interpreter (to evaluate IsZero(5)) |

The compiler resolves the **structural** questions (which keys, where they are, what obs indices to check). The interpreter resolves the **state** questions (what are the current obs values?).

This is why it's "multi-stage": structural facts are resolved at stage 1 (compile time), state facts are resolved at stage 2 (run time). Neither stage has all the information — they each contribute their part.

---

## 2. Compiler Basics

### What is a compiler?

A compiler is a program that translates code from one language to another. The most familiar example is a C compiler (like `gcc`), which translates C source code into machine instructions that a CPU can execute.

Every compiler has three things:

1. **Source language** — what the programmer writes
2. **Target language** — what the machine runs
3. **Translation rules** — how to convert source into target

### Our system's compilation pipeline

In our project, we synthesize small programs to control a game agent. The pipeline has three layers:

```
Layer 3 (source):     PickRule(0), MoveRule(0), GoalRule       ← human-readable
                                    ↓
                              [COMPILER]
                                    ↓
Layer 1 (target):     Ite(And(Not(IsZero(1)), Not(IsZero(6))), Flip(4), ...)   ← machine-executable
```

**Layer 3 (Surface DSL)** is the source language. Programs are written with named building blocks like `PickRule(0)` ("if ready to pick key 0, pick it") and `MoveRule(0)` ("if key 0's room is locked, go to key 0").

**Layer 1 (Raw AST)** is the target language. Programs are trees of generic nodes that reference observation vector indices by number: `IsZero(1)` means "is bit 1 of the observation zero?", `Flip(4)` means "select action number 4."

**The compiler** (`surface_compiler.py`) translates Layer 3 into Layer 1. It looks up the game configuration to turn names into numbers. For example, `MoveToKey(0)` becomes `Flip(1)` because the configuration says key 0 is at location 1.

### What is an AST?

AST stands for **Abstract Syntax Tree**. It is the data structure the compiler produces. In our system, an AST is a tree of nodes:

```
Ite                              ← "if ... then ... else ..."
├── And                          ← condition: "both of these are true"
│   ├── Not(IsZero(1))           ← "obs[1] ≠ 0" (agent at location 1)
│   └── Not(IsZero(6))           ← "obs[6] ≠ 0" (key 0 is available)
├── Flip(4)                      ← action: select action 4 (pick key 0)
└── Ite                          ← else: try next rule...
    ├── IsZero(5)                ← "obs[5] = 0" (room 1 is locked)
    ├── Flip(1)                  ← action: select action 1 (move to location 1)
    └── Default(Flip(3))         ← fallback: select action 3 (move to goal)
```

The interpreter walks this tree top-down. At each `Ite` node, it evaluates the condition on the current game state. If true, it returns that action. If false, it moves to the `else` branch. This is called **first-match semantics** — the first rule whose condition is true wins.

### What does the compiler need to know?

To translate `PickRule(0)` into the AST above, the compiler needs to answer questions like:

- "Where is key 0?" → location 1 (so it generates `IsZero(1)`)
- "What observation index tells me if key 0 is available?" → index 6 (so it generates `IsZero(6)`)
- "What action index picks up key 0?" → action 4 (so it generates `Flip(4)`)

Currently, the compiler answers these questions by directly reading fields from `DoorsGameConfig`:

```python
cfg.key_loc[0]      # → 1   (key 0 is at location 1)
cfg.M + cfg.D + 0   # → 6   (key availability is at obs index M+D+k)
cfg.M + 0           # → 4   (pick action is at index M+k)
```

The relational runtime provides a **different way to answer the same questions**, using typed, named functions instead of raw field access.

---

## 3. Historical Context: Where "Lifted" Comes From

### First-order logic and "lifting" (1960s-1980s)

The term "lifted" comes from mathematical logic and AI planning. In first-order logic, you can write statements in two ways:

**Grounded** (about specific objects):
```
loves(John, Mary)         "John loves Mary"
parent(Alice, Bob)        "Alice is Bob's parent"
```

**Lifted** (about variables — any object that fits):
```
∀x. parent(x, y) → older(x, y)     "for any x: if x is y's parent, then x is older than y"
```

The lifted version uses a **variable** `x` instead of naming a specific person. It describes a **general rule** that applies to all parents, not just Alice.

**Lifting** means going from specific instances to general patterns. **Grounding** means going the other direction — replacing variables with specific objects.

### STRIPS and PDDL planning (1971-2000s)

In classical AI planning (Fikes & Nilsson, 1971), the STRIPS language describes actions in terms of variables:

```
Action: pickup(?key, ?room)
Precondition: at(agent, ?room), in(?key, ?room), unlocked(?room)
Effect: holding(?key), ¬in(?key, ?room)
```

This is a **lifted action schema** — it works for any key in any room. To use it in a specific problem, you **ground** it:

```
pickup(key_0, room_0)     ← specific key in specific room
pickup(key_1, room_1)     ← different key, different room
```

PDDL (Planning Domain Definition Language, McDermott et al., 1998) formalized this distinction. A PDDL **domain** file describes lifted action schemas. A PDDL **problem** file provides the specific objects and initial state. The planner grounds the schemas against the objects to find a plan.

Our project follows the same pattern:
- The **lifted DSL** (future, plan `<a>`) describes strategies in terms of roles: "pick the key for the next locked room"
- The **relational runtime** grounds these descriptions against a specific game layout: "the next locked room is room 1, its key is key 0, key 0 is at location 1"

### Program synthesis and DSL design (2010s-present)

In program synthesis (Gulwani et al., 2017), a common pattern is to define a **domain-specific language** (DSL) at a high level of abstraction, then compile programs down to a lower-level representation for execution.

The tradeoff is always the same:
- **More abstract (lifted)** DSL → smaller search space, more general programs, but harder to compile
- **Less abstract (grounded)** DSL → larger search space, instance-specific programs, but simpler compilation

Our project has been using a grounded DSL. The relational runtime is the first step toward a lifted DSL.

### Partial evaluation and multi-stage compilation (Jones et al., 1993)

A **multi-stage compiler** doesn't translate everything at once. Instead, it resolves some information early (at "compile time") and leaves other information for later (at "run time").

In our system:
- **Compile time:** The compiler knows the game layout (how many rooms, where keys are). It resolves abstract descriptions into concrete AST nodes.
- **Run time:** The interpreter evaluates the AST against the current game state (where is the agent now? which rooms are unlocked?).

The relational runtime is a compile-time component. It answers questions about the game layout so the compiler can generate AST nodes. It does NOT answer questions about the current game state — that's the interpreter's job.

---

## 4. Grounded vs Lifted: The Core Distinction

### What "grounded" means

A **grounded** program refers to specific objects by their identity (usually an index number).

Our current surface DSL is grounded:
```python
PickRule(0)     # specifically key 0
MoveRule(0)     # specifically the door key 0 opens
PickRule(1)     # specifically key 1
MoveRule(1)     # specifically the door key 1 opens
GoalRule        # go to goal
```

The `0` and `1` are **ground terms** — they name particular keys. This program is written for a specific game instance where key 0 unlocks room 1 and key 1 unlocks room 2.

### What "lifted" means

A **lifted** program refers to objects by their **role** or **relationship**, not by index.

The future lifted DSL (plan `<a>`) is lifted:
```
IfThen(Pickable(KeyFor(NextLockedRoom)),  Pick(KeyFor(NextLockedRoom)))
IfThen(KnownLoc(KeyFor(NextLockedRoom)),  GoTo(Loc(KeyFor(NextLockedRoom))))
Default(GoToGoal)
```

There are no numbers. `NextLockedRoom` is a description: "whichever room is locked next." `KeyFor(NextLockedRoom)` is a composition: "the key that opens whichever room is locked next." These are **variables bound at compile time** by iterating over the game layout.

### Why the distinction matters

**Grounded programs are tied to a specific layout.** If you have a D=3 game (3 rooms, 2 keys), you write a program with `PickRule(0), MoveRule(0), PickRule(1), MoveRule(1)`. For a D=4 game (4 rooms, 3 keys), you need a completely different program with `PickRule(0), MoveRule(0), PickRule(1), MoveRule(1), PickRule(2), MoveRule(2)`. The search algorithm must start from scratch for each game size.

**Lifted programs describe a strategy that works for any size.** The lifted program above solves D=2, D=3, D=4, or D=100 — the compiler automatically generates the right number of `Ite` nodes by iterating `lockable_rooms()`.

**The grounded DSL is "semantically dishonest."** The surface syntax `PickRule(k)` looks like a high-level concept, but `k` is a raw index whose meaning depends on hidden configuration. The program text doesn't tell you what key 0 actually does — you have to look up `cfg.key_loc[0]` and `cfg.key_unlocks[0]` to understand it.

### Worked example: same strategy, grounded vs lifted

Consider the optimal strategy for D=2 (2 rooms, 1 key):

**In English:**
```
1. If I'm standing on the key and it's available → pick it up
2. If the door is locked → go to the key
3. Otherwise → go to the goal
```

**Grounded (current surface DSL):**
```python
PickRule(0)     # "if PickReady(0) then Pick(0)"  — key NUMBER 0
MoveRule(0)     # "if NeedKey(0) then MoveToKey(0)" — key NUMBER 0
GoalRule        # "else MoveToGoal"
```

**Lifted (future lifted DSL):**
```
IfThen(Pickable(KeyFor(NextLockedRoom)), Pick(KeyFor(NextLockedRoom)))
IfThen(KnownLoc(KeyFor(NextLockedRoom)), GoTo(Loc(KeyFor(NextLockedRoom))))
Default(GoToGoal)
```

Both compile to the exact same AST:
```
Ite(And(Not(IsZero(1)), Not(IsZero(6))), Flip(4),
    Ite(IsZero(5), Flip(1),
        Default(Flip(3))))
```

The difference is in how the source program expresses intent, not in what the machine executes.

---

## 5. What "Typed Compile-Time Query Interface" Means

This phrase describes the relational runtime. Let's break it down word by word.

### "Interface"

An interface is a set of functions that code can call without knowing the implementation details. The relational runtime provides an interface to `DoorsGameConfig`. Instead of the compiler reaching into `cfg.key_loc[0]` directly (knowing the internal field name and data layout), it calls `rt.loc_of_key(KeyId(0))`.

This is like a restaurant menu vs going into the kitchen yourself. The menu (interface) lists what you can order. You don't need to know how the kitchen (DoorsGameConfig) is organized internally.

**Before (direct access — no interface):**
```python
# The compiler knows the internal field names:
location = cfg.key_loc[0]           # "reach into the config and grab key_loc array"
room = cfg.key_unlocks[0]           # "reach in and grab key_unlocks array"
obs_idx = cfg.M + cfg.D + 0        # "do arithmetic on internal constants"
```

**After (interface):**
```python
# The compiler calls named functions:
location = rt.loc_of_key(KeyId(0))         # "where is key 0?"
room = rt.room_unlocked_by(KeyId(0))       # "what room does key 0 open?"
obs_idx = rt.obs_key_avail(KeyId(0))       # "what obs index tells me if key 0 is available?"
```

### "Query"

A query is a question you ask to get information. The relational runtime answers questions about the game:

```python
rt.lockable_rooms()                    # "which rooms need keys?"       → [RoomId(1)]
rt.key_for_room(RoomId(1))             # "which key opens room 1?"     → KeyId(0)
rt.loc_of_key(KeyId(0))                # "where is key 0?"             → LocId(1)
rt.obs_room_unlocked(RoomId(1))        # "what obs index is room 1's lock status?" → ObsIdx(5)
```

Each query takes typed inputs and returns typed outputs. The caller doesn't need to know the internal formula (like "M + D + k") — the query handles it.

### "Compile-Time"

"Compile-time" means these queries are answered **before the program runs**, during compilation. They answer questions about the game layout (which is fixed before each episode), not about the current game state (which changes every step).

The compiler asks the relational runtime questions ONCE, generates AST nodes based on the answers, and then the interpreter evaluates those AST nodes many times during gameplay. The relational runtime is never consulted during gameplay.

```
COMPILE TIME (happens once):
  Compiler: "Where is key 0?"
  Relational Runtime: "Location 1."
  Compiler: OK, I'll generate IsZero(1) to check if the agent is there.

  → Produces: And(Not(IsZero(1)), Not(IsZero(6)))

RUN TIME (happens every step):
  Interpreter: Is obs[1] zero? No (agent is at location 1).
  Interpreter: Is obs[6] zero? No (key 0 is available).
  Interpreter: Both true → return Flip(4) (pick key 0).
```

### "Typed"

"Typed" means every value has a label saying what **kind** of thing it is. In the relational runtime:

- `RoomId(1)` is a room identifier
- `KeyId(0)` is a key identifier
- `LocId(1)` is a location identifier
- `ObsIdx(5)` is an observation vector index

These are all just integers underneath. `RoomId(1)` and `LocId(1)` are both the number 1. But their **types** are different, because a room and a location are different concepts.

Why does this matter? Without types, it's easy to make mistakes:

```python
# Without types — which "1" is which?
room = 1
location = 1
key = 0
obs_index = 5
# Are any of these the same thing? It's impossible to tell.

# With types — the meaning is explicit:
room = RoomId(1)        # room number 1
location = LocId(1)     # location number 1
key = KeyId(0)          # key number 0
obs_index = ObsIdx(5)   # observation index 5
# Now each value carries its meaning.
```

The types are implemented using Python's `NewType`, which has zero runtime cost — they're just annotations for the programmer and IDE.

### Putting it together

A **typed compile-time query interface** is:
- A set of functions (**interface**)
- that answer questions (**query**) about the game layout
- before the program runs (**compile-time**)
- with inputs and outputs labeled by what they represent (**typed**)

---

## 6. What "Relational Runtime" Means

### Why "relational"?

"Relational" means "defined in terms of relationships between things." The relational runtime describes the game world as a set of relationships:

```
key_0  ──unlocks──→  room_1
key_0  ──located_at──→  location_1
room_1 ──contains──→  [location_2, location_3]
```

Compare this to the raw configuration, which is just arrays of numbers:

```python
cfg.key_unlocks = [1]       # key 0 unlocks... something numbered 1
cfg.key_loc = [1]           # key 0 is at... something numbered 1
cfg.loc_room = [0, 0, 1, 1] # locations map to rooms
```

In the raw config, `key_unlocks[0] = 1` and `key_loc[0] = 1` both produce the number 1, but they mean completely different things (a room vs a location). The relational interface makes these relationships explicit and typed.

### Why "runtime"?

This is slightly misleading. The name "relational runtime" does NOT mean it runs during gameplay. It means it is a **runtime system** in the compiler-design sense — a support library that the compiler depends on at the time it runs.

Think of it like a C program and the C runtime library (`libc`). The C runtime provides functions (`malloc`, `printf`) that compiled C programs use. Similarly, the relational runtime provides functions (`loc_of_key`, `key_for_room`) that the compiler uses during compilation.

A more precise name might be "relational compilation environment" or "relational query layer," but "relational runtime" is the standard term in compiler design for "the support library the compiler uses."

### The phone book analogy

Think of the relational runtime as a phone book for the game world:

```
PHONE BOOK (relational runtime):

  "Which rooms need keys?"
    → Room 1, Room 2

  "Who has the key to Room 1?"
    → Key 0

  "Where does Key 0 live?"
    → Location 1

  "What's the phone number (obs index) for Room 1's lock status?"
    → ObsIdx(7)
```

The old compiler didn't use a phone book. It memorized the raw address book directly:
```python
cfg.key_unlocks[0]     # "I know the answer is stored in slot 0 of this array"
cfg.M + cfg.D + 0      # "I know the formula to compute the obs index"
```

Both approaches get the same answer. But the phone book approach is:
1. **Self-documenting** — the question describes what you're asking
2. **Typed** — the answer tells you what kind of thing it is
3. **Extensible** — you can add `known_map`/`partial_map` modes without changing the callers

---

## 7. known_map vs partial_map: A Complete Walkthrough

### The setup

Consider a D=2 game (2 rooms, 1 key):

```
Room 0 (always open):  locations 0, 1
Room 1 (locked):        locations 2, 3

Key 0: at location 1, unlocks room 1
Goal: at location 3

Observation vector (7 elements):
  [at_loc0, at_loc1, at_loc2, at_loc3, room0_unlocked, room1_unlocked, key0_available]
     0        1        2        3          4               5               6
```

### known_map mode (default)

In known_map mode, the compiler knows everything about the game layout — including where keys are.

```python
cfg = DoorsGameConfig(num_rooms=2, locs_per_room=2)
rt = DoorsRelationalRuntime(cfg, known_map=True)

# Query: "Where is key 0?"
rt.loc_of_key(KeyId(0))
# Answer: LocId(1)
# The runtime looked up cfg.key_loc[0] and returned LocId(1).

# Query: "Generate the AST condition for: is key 0 pickable?"
rt.cond_pick_ready(KeyId(0))
# Answer: And(Not(IsZero(1)), Not(IsZero(6)))
#
# The runtime did this internally:
#   1. loc = rt.loc_of_key(KeyId(0))        → LocId(1)    (key 0 is at location 1)
#   2. rt.cond_at_loc(LocId(1))             → Not(IsZero(1))   (agent at location 1?)
#   3. rt.cond_key_avail(KeyId(0))          → Not(IsZero(6))   (key 0 available?)
#   4. And(step 2, step 3)                  → And(Not(IsZero(1)), Not(IsZero(6)))

# Query: "Generate the AST action for: move to key 0"
rt.flip_move_to_key(KeyId(0))
# Answer: Flip(1)
#
# The runtime did this:
#   1. loc = rt.loc_of_key(KeyId(0))        → LocId(1)
#   2. rt.action_move_to(LocId(1))          → 1
#   3. Flip(1)
```

Everything works. The compiler can generate complete AST nodes because it knows all the facts.

### partial_map mode

In partial_map mode, the compiler does NOT know where keys are. This simulates a harder version of the game where the agent must discover key locations by exploring.

```python
cfg = DoorsGameConfig(num_rooms=2, locs_per_room=2)
rt = DoorsRelationalRuntime(cfg, known_map=False)

# Query: "Where is key 0?"
rt.loc_of_key(KeyId(0))
# Answer: None
# The runtime REFUSES to answer because known_map=False.
# "I don't know where key 0 is."

# Query: "Generate the AST condition for: is key 0 pickable?"
rt.cond_pick_ready(KeyId(0))
# Answer: None
# The runtime can't generate this condition because PickReady requires
# knowing key 0's location (to check if the agent is standing on it).
# Without the location, the condition is undefined.

# Query: "Generate the AST action for: move to key 0"
rt.flip_move_to_key(KeyId(0))
# Answer: None
# Can't generate "move to key 0" if we don't know where key 0 is.

# BUT: Queries that DON'T need key locations still work:

rt.key_for_room(RoomId(1))           # → KeyId(0)  "key 0 opens room 1"
# This is structural knowledge (the key-room relationship), not locational.

rt.room_unlocked_by(KeyId(0))        # → RoomId(1)  "key 0 unlocks room 1"
# Same — structural, not locational.

rt.cond_need_key(KeyId(0))           # → IsZero(5)  "is room 1 locked?"
# This checks a room's lock status in the observation vector.
# The room's lock status is OBSERVABLE — the agent can see it.
# This does NOT depend on knowing where key 0 is physically located.

rt.obs_key_avail(KeyId(0))           # → ObsIdx(6)  "obs index for key 0's availability"
# Key availability is also observable — the agent can see it.

rt.flip_pick(KeyId(0))               # → Flip(4)  "pick key 0"
# Picking a key doesn't require knowing its location in advance.
# (If the agent is standing on it, the pick action works regardless.)
```

### Why does `None` matter?

In the old compiler, partial observability would be a silent bug:

```python
# Old compiler, partial_map scenario (hypothetical):
obs_index = cfg.key_loc[0]   # → 1, EVEN THOUGH THE AGENT DOESN'T KNOW THIS
# The compiler happily uses this hidden information.
# The resulting program "cheats" by knowing things the agent shouldn't.
```

With the relational runtime, `None` forces the compiler to handle the unknown case explicitly:

```python
loc = rt.loc_of_key(KeyId(0))
if loc is None:
    # The compiler MUST decide what to do when key location is unknown.
    # Options: generate a search subroutine, skip this rule, etc.
    # It CANNOT silently use hidden information.
```

This is what the original proposal means by "⊥ (bottom) makes 'unknown location' an explicit semantic case rather than an implicit compiler cheat."

### Current status

`partial_map` mode is **interface-only** right now. The Doors game environment is fully observable — the agent always knows everything. A future project would need to create a modified environment that actually hides key locations. But the interface is defined now so the lifted compiler can be designed to handle both cases from the start.

---

## 8. How This Connects to Plan `<a>` (The Lifted DSL)

### What plan `<a>` proposes

Plan `<a>` proposes a new DSL layer — the **lifted DSL** — that sits above the current surface DSL. Here is the full proposed syntax:

```
Types:
  RoomSel = NextLockedRoom | NextUnlockedFrontierRoom | GoalRoom
  KeySel  = KeyFor(RoomSel)
  LocSel  = Loc(KeySel) | Entrance(RoomSel) | GoalLoc

Predicates:
  GoalReached
  Pickable(KeySel)
  KnownLoc(KeySel)
  Reachable(LocSel)

Actions:
  Pick(KeySel)
  GoTo(LocSel)
  GoToGoal

Rules:
  IfThen(predicate, action)
  Default(action)
```

### What the lifted DSL needs from the relational runtime

The lifted DSL defines abstract **selectors** like `NextLockedRoom` and `KeyFor(room)`. The compiler must expand these into concrete AST nodes. Here is exactly what the compiler needs from the relational runtime:

| Lifted selector | Compiler needs | Relational runtime provides |
|---|---|---|
| `NextLockedRoom` | List of rooms that can be locked | `rt.lockable_rooms()` → [RoomId(1), RoomId(2)] |
| `KeyFor(room)` | Which key opens a given room | `rt.key_for_room(r)` → KeyId |
| `Loc(key)` | Where a key is located | `rt.loc_of_key(k)` → LocId or None |
| `Pickable(key)` | AST condition: at key AND key available | `rt.cond_pick_ready(k)` → AST Condition |
| `KnownLoc(key)` | Whether key location is known | `rt.loc_of_key(k) is not None` |
| `Pick(key)` | AST action: pick up key | `rt.flip_pick(k)` → Flip |
| `GoTo(loc)` | AST action: move to location | `rt.flip_move_to_key(k)` → Flip |
| `GoToGoal` | AST action: move to goal | `rt.flip_move_to_goal()` → Flip |

Without the relational runtime, the lifted compiler would need to read raw config fields directly — which defeats the purpose of having a lifted DSL.

### How the lifted compiler will work (preview)

The lifted compiler expands selectors by iterating over entities from the relational runtime. Here is how `IfThen(Pickable(KeyFor(NextLockedRoom)), Pick(KeyFor(NextLockedRoom)))` compiles for D=3:

```python
def compile_lifted_pick_rule(rt, else_prog):
    prog = else_prog

    # Step 1: Ask "which rooms can be locked?"
    rooms = rt.lockable_rooms()          # → [RoomId(1), RoomId(2)]

    # Step 2: For each room (in reverse, for right-fold nesting):
    for r in reversed(rooms):            # room 2, then room 1

        # Step 3: "Which key opens this room?"
        k = rt.key_for_room(r)           # room 2 → KeyId(1), room 1 → KeyId(0)

        # Step 4: "Generate the PickReady condition for this key"
        cond = rt.cond_pick_ready(k)     # KeyId(1) → And(Not(IsZero(3)), Not(IsZero(10)))
                                         # KeyId(0) → And(Not(IsZero(1)), Not(IsZero(9)))

        # Step 5: "Generate the Pick action for this key"
        action = rt.flip_pick(k)         # KeyId(1) → Flip(7), KeyId(0) → Flip(6)

        # Step 6: Wrap in an Ite node
        prog = Ite(cond, action, prog)

    return prog
```

The result is:
```
Ite(And(Not(IsZero(1)), Not(IsZero(9))),   ← PickReady(key 0)
    Flip(6),                                ← Pick(key 0)
    Ite(And(Not(IsZero(3)), Not(IsZero(10))), ← PickReady(key 1)
        Flip(7),                              ← Pick(key 1)
        else_prog))
```

This is **identical** to what the current grounded compiler produces for `PickRule(0), PickRule(1)`. The test `test_lifted_matches_canonical_d3` in `test_relational_runtime.py` verifies this.

### The dependency chain

```
Plan <a> (lifted DSL)
   │
   │ depends on
   ▼
Relational Runtime  ← THIS IS WHAT WE JUST BUILT
   │
   │ wraps
   ▼
DoorsGameConfig (raw game layout data)
```

The relational runtime is the foundation. Without it, the lifted DSL cannot be compiled.

---

## 9. Full Worked Example: D=3 Compilation, Step by Step

This section traces the compilation of the D=3 canonical policy through BOTH the old compiler and the new relational runtime, showing they produce identical output.

### The game layout (D=3)

```
Room 0 (always open):  locations 0, 1
Room 1 (locked):        locations 2, 3
Room 2 (locked):        locations 4, 5

Key 0: at location 1, unlocks room 1
Key 1: at location 3, unlocks room 2
Goal:  at location 5

Config values:
  M = 6 (locations), D = 3 (rooms), K = 2 (keys)
  key_loc = [1, 3]
  key_unlocks = [1, 2]
  goal_loc = 5

Observation vector (11 elements):
  [at_0, at_1, at_2, at_3, at_4, at_5, room0, room1, room2, key0, key1]
    0     1     2     3     4     5     6      7      8      9     10
```

### The grounded program

```python
policy = SurfacePolicy((
    PickRule(0),    # "if PickReady(0) then Pick(0)"
    MoveRule(0),    # "if NeedKey(0) then MoveToKey(0)"
    PickRule(1),    # "if PickReady(1) then Pick(1)"
    MoveRule(1),    # "if NeedKey(1) then MoveToKey(1)"
    GoalRule(),     # "else MoveToGoal"
))
```

### Old compiler trace (surface_compiler.py)

The compiler processes rules **right to left** (last rule first):

```
Step 1: GoalRule
  compile_action(MoveToGoal(), cfg)
    → Flip(cfg.goal_loc) = Flip(5)
  Result: Default(Flip(5))

Step 2: MoveRule(1)
  compile_condition(NeedKey(1), cfg)
    → IsZero(cfg.M + cfg.key_unlocks[1])
    → IsZero(6 + 2) = IsZero(8)
  compile_action(MoveToKey(1), cfg)
    → Flip(cfg.key_loc[1])
    → Flip(3)
  Result: Ite(IsZero(8), Flip(3), Default(Flip(5)))

Step 3: PickRule(1)
  compile_condition(PickReady(1), cfg)
    → And(
        compile_condition(AtKeyLoc(1), cfg),    → Not(IsZero(cfg.key_loc[1])) = Not(IsZero(3))
        compile_condition(KeyAvail(1), cfg),    → Not(IsZero(cfg.M + cfg.D + 1)) = Not(IsZero(10))
      )
    → And(Not(IsZero(3)), Not(IsZero(10)))
  compile_action(Pick(1), cfg)
    → Flip(cfg.M + 1) = Flip(7)
  Result: Ite(And(Not(IsZero(3)), Not(IsZero(10))), Flip(7),
              Ite(IsZero(8), Flip(3), Default(Flip(5))))

Step 4: MoveRule(0)
  compile_condition(NeedKey(0), cfg)
    → IsZero(cfg.M + cfg.key_unlocks[0])
    → IsZero(6 + 1) = IsZero(7)
  compile_action(MoveToKey(0), cfg)
    → Flip(cfg.key_loc[0]) = Flip(1)
  Result: Ite(IsZero(7), Flip(1),
              Ite(And(Not(IsZero(3)), Not(IsZero(10))), Flip(7),
                  Ite(IsZero(8), Flip(3), Default(Flip(5)))))

Step 5: PickRule(0)
  compile_condition(PickReady(0), cfg)
    → And(Not(IsZero(cfg.key_loc[0])), Not(IsZero(cfg.M + cfg.D + 0)))
    → And(Not(IsZero(1)), Not(IsZero(9)))
  compile_action(Pick(0), cfg)
    → Flip(cfg.M + 0) = Flip(6)
  Result (FINAL):
    Ite(And(Not(IsZero(1)), Not(IsZero(9))), Flip(6),
        Ite(IsZero(7), Flip(1),
            Ite(And(Not(IsZero(3)), Not(IsZero(10))), Flip(7),
                Ite(IsZero(8), Flip(3),
                    Default(Flip(5))))))
```

Note how the compiler directly reads `cfg.key_loc[0]`, `cfg.key_unlocks[1]`, `cfg.M`, `cfg.D`, etc.

### New relational runtime trace

The relational runtime builds the same AST, but through typed queries:

```
Setup:
  rt = DoorsRelationalRuntime(cfg)
  # Precomputes inverse map: {1: 0, 2: 1}  (room 1 → key 0, room 2 → key 1)

Step 1: Default action
  rt.flip_move_to_goal()
    → rt.goal_location()                → LocId(5)
    → rt.action_move_to(LocId(5))       → 5
    → Flip(5)
  Result: Default(Flip(5))

Step 2: Iterate lockable rooms (reversed)
  rt.lockable_rooms() → [RoomId(1), RoomId(2)]
  reversed → [RoomId(2), RoomId(1)]

Step 3: Room 2 — MoveRule
  k = rt.key_for_room(RoomId(2))        → KeyId(1)
  rt.cond_need_key(KeyId(1))
    → rt.room_unlocked_by(KeyId(1))     → RoomId(2)
    → rt.cond_room_locked(RoomId(2))
    → rt.obs_room_unlocked(RoomId(2))   → ObsIdx(6 + 2) = ObsIdx(8)
    → IsZero(8)
  rt.flip_move_to_key(KeyId(1))
    → rt.loc_of_key(KeyId(1))           → LocId(3)
    → rt.action_move_to(LocId(3))       → 3
    → Flip(3)
  Result: Ite(IsZero(8), Flip(3), Default(Flip(5)))

Step 4: Room 2 — PickRule
  rt.cond_pick_ready(KeyId(1))
    → rt.loc_of_key(KeyId(1))           → LocId(3)
    → rt.cond_at_loc(LocId(3))          → Not(IsZero(3))
    → rt.cond_key_avail(KeyId(1))
      → rt.obs_key_avail(KeyId(1))      → ObsIdx(6 + 3 + 1) = ObsIdx(10)
      → Not(IsZero(10))
    → And(Not(IsZero(3)), Not(IsZero(10)))
  rt.flip_pick(KeyId(1))
    → rt.action_pick(KeyId(1))          → 6 + 1 = 7
    → Flip(7)
  Result: Ite(And(Not(IsZero(3)), Not(IsZero(10))), Flip(7),
              Ite(IsZero(8), Flip(3), Default(Flip(5))))

Step 5: Room 1 — MoveRule
  k = rt.key_for_room(RoomId(1))        → KeyId(0)
  rt.cond_need_key(KeyId(0))
    → rt.room_unlocked_by(KeyId(0))     → RoomId(1)
    → rt.obs_room_unlocked(RoomId(1))   → ObsIdx(7)
    → IsZero(7)
  rt.flip_move_to_key(KeyId(0))
    → rt.loc_of_key(KeyId(0))           → LocId(1)
    → Flip(1)
  Result: Ite(IsZero(7), Flip(1),
              Ite(And(Not(IsZero(3)), Not(IsZero(10))), Flip(7),
                  Ite(IsZero(8), Flip(3), Default(Flip(5)))))

Step 6: Room 1 — PickRule
  rt.cond_pick_ready(KeyId(0))
    → rt.loc_of_key(KeyId(0))           → LocId(1)
    → rt.cond_at_loc(LocId(1))          → Not(IsZero(1))
    → rt.obs_key_avail(KeyId(0))        → ObsIdx(9)
    → rt.cond_key_avail(KeyId(0))       → Not(IsZero(9))
    → And(Not(IsZero(1)), Not(IsZero(9)))
  rt.flip_pick(KeyId(0))
    → rt.action_pick(KeyId(0))          → 6
    → Flip(6)
  Result (FINAL):
    Ite(And(Not(IsZero(1)), Not(IsZero(9))), Flip(6),
        Ite(IsZero(7), Flip(1),
            Ite(And(Not(IsZero(3)), Not(IsZero(10))), Flip(7),
                Ite(IsZero(8), Flip(3),
                    Default(Flip(5))))))
```

### Comparison

The final ASTs are identical:

```
OLD COMPILER:                              NEW RELATIONAL RUNTIME:
Ite(And(Not(IsZero(1)), Not(IsZero(9))),   Ite(And(Not(IsZero(1)), Not(IsZero(9))),
    Flip(6),                                   Flip(6),
    Ite(IsZero(7), Flip(1),                    Ite(IsZero(7), Flip(1),
        Ite(And(Not(IsZero(3)),                    Ite(And(Not(IsZero(3)),
                Not(IsZero(10))),                           Not(IsZero(10))),
            Flip(7),                                   Flip(7),
            Ite(IsZero(8), Flip(3),                    Ite(IsZero(8), Flip(3),
                Default(Flip(5))))))                       Default(Flip(5))))))

IDENTICAL ✓
```

The difference:
- The old compiler reads `cfg.key_loc[0]`, `cfg.key_unlocks[1]`, `cfg.M + cfg.D + 0` — raw field access with manual arithmetic.
- The relational runtime calls `rt.loc_of_key(KeyId(0))`, `rt.key_for_room(RoomId(1))`, `rt.obs_key_avail(KeyId(0))` — typed queries with named functions.

Both produce the same 22-node AST. Both solve D=3 in 5 steps. The relational runtime is a cleaner interface to the same data, designed so the future lifted DSL compiler never touches raw config fields.

---

## 10. Summary

| Question | Answer |
|---|---|
| What changed? | Added `relational_runtime.py` — a typed lookup service wrapping `DoorsGameConfig` |
| What is it? | A class (`DoorsRelationalRuntime`) with methods like `loc_of_key()`, `key_for_room()`, `cond_pick_ready()` |
| Why? | To enable the future lifted DSL (plan `<a>`), which needs typed queries to expand abstract selectors into concrete AST |
| Does it change how programs run? | No — the interpreter and AST are unchanged; only how the compiler gets its information |
| Does it replace the old compiler? | No — the old surface compiler is untouched; both coexist |
| What is known_map? | The compiler knows all key locations (current game); all queries return concrete values |
| What is partial_map? | The compiler does NOT know key locations (future game); location queries return None |
| How was it verified? | 52 tests: entity queries, obs indices, AST equivalence with old compiler, lifted compilation smoke test (solves D=3 in 5 steps) |
