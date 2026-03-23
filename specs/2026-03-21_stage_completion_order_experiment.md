# Stage-Skeleton DSL & Completion-Order Experiment

**Date:** 2026-03-21
**Branch:** `feature/grammar-redesign`
**Prerequisites:** 1a (surface DSL), 1b (stage-skeleton DSL)

---

## Background: Terminology, Literature, and Worked Examples

### What is a decision list?

A **decision list** (Rivest, 1987) is one of the simplest program structures in machine learning and program synthesis. It is an ordered sequence of if-then rules with a default:

```
if condition_1 then action_1
elif condition_2 then action_2
elif condition_3 then action_3
else default_action
```

Evaluation is **first-match**: scan top-to-bottom, fire the first rule whose condition is true. If no condition is true, execute the default. This is equivalent to a nested if-then-else tree, but written flat for readability.

Decision lists appear in classical AI as **production rule systems** (Newell & Simon, 1972), in machine learning as **rule lists** (Letham et al., 2015), and in programming language theory as **pattern matching** / **case expressions**. In this project, the programs we synthesize are decision lists over a binary observation vector.

### The three layers of representation

This project has three layers for writing decision-list programs. They express the same programs but at different levels of abstraction.

#### Layer 1: Raw AST (the machine layer)

The lowest layer. Programs are trees of generic nodes that reference observation indices directly by number. There is no concept of "key" or "room" — just integer indices into the observation vector.

**Vocabulary:**
- `IsZero(i)` — true if `obs[i] == 0`  (the only primitive **condition**)
- `Not(c)` — negation of condition `c`
- `And(c1, c2)` — conjunction of two conditions
- `Flip(i)` — select action at index `i`  (the only primitive **action**)
- `Ite(cond, action, else_prog)` — if cond then action, else continue
- `Default(action)` — always execute this action (terminal case)

**Example: D=2 optimal controller as raw AST.**

The Doors D=2 environment has observation vector of size 7:
```
obs = [at_loc0, at_loc1, at_loc2, at_loc3, room0_open, room1_open, key0_avail]
         0        1        2        3          4           5           6
```

The optimal policy is: "if you're at the key and the key is available, pick it up. Otherwise if the door is locked, go to the key. Otherwise go to the goal."

As a raw AST:
```
Ite(
  And(Not(IsZero(1)), Not(IsZero(6))),   ← "obs[1]≠0 AND obs[6]≠0"
  Flip(4),                                ← action index 4 = PICK(key 0)
  Ite(
    IsZero(5),                            ← "obs[5]==0" = room 1 is locked
    Flip(1),                              ← action index 1 = MOVE(to loc 1, where key 0 is)
    Default(Flip(3))                      ← action index 3 = MOVE(to loc 3, the goal)
  )
)
```

This is 12 AST nodes. It works, but it is opaque: you need to know that index 1 means "key 0's location" and index 5 means "room 1's lock status" to understand it.

**The term "condition"** in this codebase refers specifically to this raw AST layer: a tree of `IsZero`, `Not`, and `And` nodes that reads observation indices by number.

#### Layer 2: Stage-skeleton DSL (the typed layer)

The middle layer. Replaces numeric indices with **typed, named atoms** that carry domain meaning. A program is a sequence of **stages**, where each stage pairs a **guard** with an **action**.

**What is a guard?** A guard is a boolean expression composed of typed atoms from the Doors domain:

| Guard atom | Plain English | What it checks at the raw level |
|-----------|---------------|-------------------------------|
| `AtKeyLoc(k)` | "Am I standing at key k?" | `obs[key_k_location] ≠ 0` |
| `KeyAvail(k)` | "Is key k still on the ground?" | `obs[M + D + k] ≠ 0` |
| `PickReady(k)` | "Am I at key k AND it's available?" | `AtKeyLoc(k) AND KeyAvail(k)` |
| `NeedKey(k)` | "Is the room that key k unlocks still locked?" | `obs[M + key_k_unlocks] == 0` |
| `RoomLocked(r)` | "Is room r locked?" | `obs[M + r] == 0` |

Guards can also be composed: `GuardNot(g)` for negation, `GuardAnd(g1, g2)` for conjunction.

**Guard vs condition:** A **guard** is a typed, human-readable boolean expression (e.g., `PickReady(0)`). A **condition** is its compiled form as a raw AST tree (e.g., `And(Not(IsZero(1)), Not(IsZero(6)))`). The compiler translates guards to conditions by looking up observation indices from `DoorsGameConfig`. They express the same logic, but guards carry domain meaning while conditions are just index arithmetic.

**What is a stage?** A stage is a single `(guard, action)` pair: "if this guard is true, do this action." A **StageProgram** is an ordered list of stages plus a default action. The semantics are identical to a decision list: first true guard wins.

**The same D=2 controller as a StageProgram:**
```python
StageProgram(
    stages=(
        Stage(guard=PickReady(0), action=Pick(0)),     # "if ready to pick key 0, pick it"
        Stage(guard=NeedKey(0),   action=MoveToKey(0)), # "elif key 0's room is locked, go to key 0"
    ),
    default_action=MoveToGoal(),                        # "else go to the goal"
)
```

This compiles to exactly the same 12-node raw AST above. But it is readable: you can see the intent without knowing observation indices.

**Why "stage"?** Each stage represents one step in a priority-ordered plan: "first try this, then try that." The term comes from staged computation in PL theory, where a program is built in phases. Here, each stage is one priority level in the decision list.

#### Layer 3: Surface DSL (the macro layer)

The highest layer. Bundles specific guard+action pairs into **macros** — pre-made building blocks that cannot be decomposed.

**What is a macro?** A macro is a fixed, pre-defined stage:

| Macro | Expands to | Plain English |
|-------|-----------|---------------|
| `PickRule(k)` | `Stage(guard=PickReady(k), action=Pick(k))` | "If ready to pick key k, pick it" |
| `MoveRule(k)` | `Stage(guard=NeedKey(k), action=MoveToKey(k))` | "If key k's room is locked, go to key k" |
| `GoalRule` | default action = `MoveToGoal()` | "Otherwise, go to the goal" |

A **SurfacePolicy** is just an ordering of these macros. You cannot create a stage with a different guard-action pairing — only the pre-made macros exist.

**The same D=2 controller as a SurfacePolicy:**
```
PickRule(0) >> MoveRule(0) >> GoalRule
```

This is the most compact, but also the least flexible: you can only order the given macros, not compose new guards.

### How the three layers relate

```
Surface DSL (macros)        Stage DSL (typed)           Raw AST (indices)
─────────────────────       ──────────────────          ──────────────────
PickRule(0)            →    Stage(PickReady(0),     →   Ite(And(Not(IsZero(1)),
                                  Pick(0))                    Not(IsZero(6))),
                                                            Flip(4), ...)
MoveRule(0)            →    Stage(NeedKey(0),       →   Ite(IsZero(5),
                                  MoveToKey(0))             Flip(1), ...)
GoalRule               →    default=MoveToGoal()    →   Default(Flip(3))
```

Each layer compiles down to the next. The raw AST is what the interpreter actually executes. The higher layers are for human understanding and structured search.

### Why have multiple layers?

**Historical context.** In classical program synthesis (Gulwani et al., 2017), a common pattern is to define a **domain-specific language** (DSL) at a high level, then compile to a lower-level representation for execution. The high-level DSL constrains the search space (fewer programs to explore), while the low-level language is general and efficient.

Our system follows this pattern:
- The **raw AST** is general: it can express any decision list over any observation vector. But for D=2 Doors alone, there are ~52 billion valid AST programs (with budget ≤ 18). Almost all are useless.
- The **surface DSL** is maximally constrained: for D=2 there is exactly 1 valid policy. It finds the answer instantly but cannot explore alternatives.
- The **stage DSL** is the middle ground: for D=2 there are ~6,175 programs (with max 3 stages, depth-0 guards). This is small enough to search exhaustively, large enough to contain interesting alternatives.

### FAQ: The three confusions addressed

#### 1. "What's the difference between a guard and a condition?"

A **condition** is a node in the raw AST: `IsZero(5)`, `Not(IsZero(1))`, `And(Not(IsZero(1)), Not(IsZero(6)))`. It operates on observation vector indices (just numbers). It is what the interpreter evaluates at runtime.

A **guard** is the same boolean logic expressed with typed domain names: `NeedKey(0)`, `AtKeyLoc(0)`, `PickReady(0)`. It is what a human reads and what the search constructs.

The **compiler** translates one to the other:
```
PickReady(0)   ─compile→   And(Not(IsZero(1)), Not(IsZero(6)))
                            ↑                    ↑
                            AtKeyLoc(0)          KeyAvail(0)
                            "obs[1]≠0"           "obs[6]≠0"
```

Think of it like SQL vs machine code: `SELECT * FROM users WHERE age > 18` is the guard; the query plan with disk seeks and index lookups is the condition. Same result, different abstraction level.

#### 2. "What's the difference between stages and just writing everything sequentially?"

They are the same thing. A `StageProgram` **is** a sequential list — specifically, a decision list evaluated top-to-bottom with first-match semantics.

The word "stage" just names each entry in the list. Writing:

```python
StageProgram(stages=(
    Stage(guard=PickReady(0), action=Pick(0)),
    Stage(guard=NeedKey(0),   action=MoveToKey(0)),
), default_action=MoveToGoal())
```

is identical in meaning to the sequential pseudocode:

```
if PickReady(0):     return Pick(0)
elif NeedKey(0):     return MoveToKey(0)
else:                return MoveToGoal
```

The reason we wrap it in `Stage` objects rather than writing raw pseudocode is **for search**: each `Stage` can have a `GuardHole` (undecided guard) or `ActionHole` (undecided action), which the search algorithm fills in. The `Stage` dataclass is what lets us represent a partially-constructed program during synthesis.

So the distinction is not sequential vs non-sequential. It is: a **SurfacePolicy** is a sequential list of *fixed macros* (you can only reorder them), while a **StageProgram** is a sequential list of *arbitrary guard-action pairs* (you can mix and match any guard with any action).

#### 3. "What's the concrete difference between the surface DSL and the stage DSL?"

Let's use D=3 (3 rooms, 2 keys) and walk through a concrete example.

**Surface DSL — only fixed macros.** The 6 valid D=3 surface policies are all orderings of {PickRule(0), MoveRule(0), PickRule(1), MoveRule(1)} followed by GoalRule, where PickRule(k) comes before MoveRule(k). For example:

```
PickRule(0) >> MoveRule(0) >> PickRule(1) >> MoveRule(1) >> GoalRule
```

This means:
1. "If ready to pick key 0, pick it"
2. "Elif need key 0 (room 1 locked), go to key 0"
3. "Elif ready to pick key 1, pick it"
4. "Elif need key 1 (room 2 locked), go to key 1"
5. "Else go to goal"

Every surface policy uses exactly these guard-action pairings. You cannot say "if room 1 is locked, go to the goal" — that would pair `NeedKey(0)` with `MoveToGoal`, which is not one of the macros.

**Stage DSL — any guard with any action.** A stage program can mix freely:

```python
StageProgram(stages=(
    Stage(guard=RoomLocked(1), action=MoveToKey(0)),    # "if room 1 locked, go to key 0"
    Stage(guard=RoomLocked(2), action=MoveToKey(1)),    # "if room 2 locked, go to key 1"
    Stage(guard=KeyAvail(0),   action=Pick(0)),         # "if key 0 available, pick it"
), default_action=MoveToGoal())
```

This program:
- Uses `RoomLocked(1)` as a guard — a different atom than the surface DSL's `NeedKey(0)` (though they check the same obs index, they are different typed atoms with different semantics in general)
- Pairs `RoomLocked(1)` with `MoveToKey(0)` — a combination the surface DSL cannot express (surface DSL always pairs NeedKey(k) with MoveToKey(k), never with a different action)
- Uses `KeyAvail(0)` alone as a guard — surface DSL always bundles it with `AtKeyLoc(0)` inside `PickReady(0)`
- Has only 3 stages instead of 4 — surface DSL for D=3 always has exactly 4 non-goal rules

This program probably does not solve D=3 optimally (the guard priorities are wrong), but it is syntactically valid in the stage DSL. The stage DSL gives you more rope — you can build programs the surface DSL forbids, including many that don't work. The tradeoff is a larger search space.

**Summary of the difference:**

| | Surface DSL | Stage DSL |
|---|---|---|
| Building blocks | 3 fixed macros (PickRule, MoveRule, GoalRule) | Any guard + any action |
| Guard flexibility | None — fixed per macro | Full — any atom or composition |
| Action flexibility | None — fixed per macro | Full — any typed action |
| Search space (D=3) | 6 programs | 169,456 programs (depth 0, ≤3 stages) |
| Can express non-standard programs? | No | Yes |

---

## Part A: What the Stage-Skeleton DSL Is (1b recap with explicit examples)

### The problem with the surface DSL

The surface DSL (1a) bundles guard+action into opaque macros:

```
PickRule(0)   =   "if PickReady(0) then Pick(0)"       ← fixed, cannot decompose
MoveRule(0)   =   "if NeedKey(0)   then MoveToKey(0)"   ← fixed, cannot decompose
GoalRule      =   "else MoveToGoal"                      ← fixed
```

A surface policy is just an ordering of these macros. For D=3 (K=2), there are exactly `(2·2)! / 2² = 6` orderings. This is too coarse for finer experiments — you cannot explore different guard formulations or measure how guard complexity affects search.

### What the stage-skeleton DSL adds

The stage DSL unbundles each macro into an explicit `Stage(guard, action)` pair. Guards are composed from typed atoms; actions are typed actions. No budget appears anywhere.

**Concrete D=3 example — the canonical controller:**

```python
StageProgram(
    stages=(
        Stage(guard=PickReady(0), action=Pick(0)),       # stage 0
        Stage(guard=NeedKey(0),   action=MoveToKey(0)),   # stage 1
        Stage(guard=PickReady(1), action=Pick(1)),       # stage 2
        Stage(guard=NeedKey(1),   action=MoveToKey(1)),   # stage 3
    ),
    default_action=MoveToGoal(),
)
```

This compiles to exactly the same 22-node AST as the surface canonical policy:
```
Ite(And(Not(IsZero(1)), Not(IsZero(9))),    ← PickReady(0): at key0 AND key0 available
    Flip(6),                                 ← Pick(0): pick key 0
    Ite(IsZero(7),                           ← NeedKey(0): room 1 locked
        Flip(1),                             ← MoveToKey(0): move to key 0 location
        Ite(And(Not(IsZero(3)), Not(IsZero(10))),  ← PickReady(1)
            Flip(7),                         ← Pick(1)
            Ite(IsZero(8),                   ← NeedKey(1): room 2 locked
                Flip(3),                     ← MoveToKey(1)
                Default(Flip(5))))))         ← MoveToGoal
```

### The key difference: independent choice of guard and action

In the surface DSL, each macro is a **fixed pairing** — the guard and the action are welded together:

| Macro | Guard (fixed) | Action (fixed) | Can you change either? |
|-------|--------------|----------------|----------------------|
| `PickRule(k)` | `PickReady(k)` | `Pick(k)` | No |
| `MoveRule(k)` | `NeedKey(k)` | `MoveToKey(k)` | No |

You cannot pair `PickReady(0)` with `MoveToKey(1)`, or use `RoomLocked(1)` as a guard at all. The only degree of freedom is **ordering** the macros.

In the stage DSL, guard and action are **independently chosen**. Any of the 11 guard atoms can pair with any of the 5 actions = **55 possible stages**. Here is a side-by-side for D=3:

```
SURFACE DSL — guard and action are locked together:
  PickRule(0)  =  PickReady(0) → Pick(0)       ← cannot change this pairing
  MoveRule(0)  =  NeedKey(0)   → MoveToKey(0)   ← cannot change this pairing
  PickRule(1)  =  PickReady(1) → Pick(1)
  MoveRule(1)  =  NeedKey(1)   → MoveToKey(1)

STAGE DSL — guard and action are independent:
  Stage(guard=PickReady(0),  action=Pick(0))       ← same as PickRule(0)
  Stage(guard=NeedKey(0),    action=MoveToKey(0))   ← same as MoveRule(0)
  Stage(guard=RoomLocked(1), action=MoveToKey(0))   ← NEW: different guard, same action
  Stage(guard=KeyAvail(0),   action=MoveToGoal())   ← NEW: novel guard-action pair
  Stage(guard=PickReady(0),  action=MoveToKey(1))   ← NEW: familiar guard, different action
```

The last three stages are valid in the stage DSL but **impossible** in the surface DSL. This is the core difference: the surface DSL gives you 4 fixed building blocks to reorder; the stage DSL gives you 55 building blocks to choose from at each position.

### Why the surface DSL has exactly 6 policies for D=3

For D=3, K=2 keys, so the surface DSL has 4 non-goal macros: `{PickRule(0), MoveRule(0), PickRule(1), MoveRule(1)}`. A valid policy is any ordering of these 4, followed by `GoalRule`, subject to the constraint that `PickRule(k)` must appear before `MoveRule(k)` for each key k (you must pick the key before you can move through the door it unlocks).

- Total orderings of 4 items: `4! = 24`
- Each key imposes one constraint (PickRule before MoveRule), which cuts the count in half per key: `24 / 2² = 6`

The 6 policies are:
```
1. PickRule(0) >> MoveRule(0) >> PickRule(1) >> MoveRule(1) >> GoalRule
2. PickRule(0) >> PickRule(1) >> MoveRule(0) >> MoveRule(1) >> GoalRule
3. PickRule(0) >> PickRule(1) >> MoveRule(1) >> MoveRule(0) >> GoalRule
4. PickRule(1) >> PickRule(0) >> MoveRule(0) >> MoveRule(1) >> GoalRule
5. PickRule(1) >> PickRule(0) >> MoveRule(1) >> MoveRule(0) >> GoalRule
6. PickRule(1) >> MoveRule(1) >> PickRule(0) >> MoveRule(0) >> GoalRule
```

Of these 6, only 3 actually solve D=3 (the others attempt to move through locked rooms).

### Guard atom vocabulary for D=3

For D=3 (K=2 keys, D=3 rooms):

| Guard atom | Meaning | Obs index checked |
|-----------|---------|------------------|
| `AtKeyLoc(0)` | Agent at key 0's location (loc 1) | `obs[1] ≠ 0` |
| `AtKeyLoc(1)` | Agent at key 1's location (loc 3) | `obs[3] ≠ 0` |
| `KeyAvail(0)` | Key 0 still available | `obs[9] ≠ 0` |
| `KeyAvail(1)` | Key 1 still available | `obs[10] ≠ 0` |
| `PickReady(0)` | AtKeyLoc(0) AND KeyAvail(0) | compound |
| `PickReady(1)` | AtKeyLoc(1) AND KeyAvail(1) | compound |
| `NeedKey(0)` | Room 1 is locked | `obs[7] = 0` |
| `NeedKey(1)` | Room 2 is locked | `obs[8] = 0` |
| `RoomLocked(0)` | Room 0 locked (always false at start) | `obs[6] = 0` |
| `RoomLocked(1)` | Room 1 locked | `obs[7] = 0` |
| `RoomLocked(2)` | Room 2 locked | `obs[8] = 0` |

**Total: 11 atoms.** Actions: `Pick(0)`, `Pick(1)`, `MoveToKey(0)`, `MoveToKey(1)`, `MoveToGoal` = **5 actions.**

### Search space sizes

With `max_guard_depth=0` (atoms only, no Not/And), each stage slot has `11 × 5 = 55` possible guard-action pairs.

| D | K | Slots | max_stages=3 | max_stages=5 |
|---|---|-------|-------------|-------------|
| 2 | 1 | 18 | **6,175** | 1,960,183 |
| 3 | 2 | 55 | **169,456** | 512,604,456 |

Compare: surface DSL has 1 policy (D=2) or 6 policies (D=3). Budgeted AST grammar has ~52 billion (D=2).

### What you can run today (1b code)

```bash
# Run the 3-way comparison (budgeted vs surface vs stage) for D=2 and D=3
python -c "
from alphazeropp.instances.doors.dsl.stage_diagnostics import run_diagnostics
print(run_diagnostics(2))
print()
print(run_diagnostics(3))
"

# Run the stage DSL unit tests (35 tests)
python -m pytest tests/test_stage_dsl.py -v
```

### What 1b does NOT have

- No step-by-step construction (hole-filling). Programs are only enumerated as complete objects.
- No search algorithm over partial programs.
- No runnable training script (no MCTS / AlphaZero integration).

**This is what Part B adds.**

---

## Part B: Completion-Order Experiment (Task `<a>`)

### Goal

Compare two hole-selection orders for constructing stage-skeleton programs and measure how order affects search efficiency. Decide whether "leftmost hole" should remain the default.

### Why D=2 is not meaningful here

D=2 has K=1 key. The canonical controller needs 2 stages. With depth-0 guards and max_stages=3, the total search space is 6,175 programs. Every reasonable search strategy explores it exhaustively in milliseconds. There is no meaningful difference between completion orders at this scale.

**D=3 is the primary benchmark:** K=2 keys, 4 stages in the canonical controller, 169,456 programs at max_stages=3. Large enough that order matters; small enough for exact comparison.

### Search state: partial stage programs

A partial program is a `StageProgram` where some stages may have `GuardHole` or `ActionHole` placeholders, and the total number of stages may not yet be decided.

**Concrete D=3 example of a partial program:**

```
Stage 0: if PickReady(0) then Pick(0)        ← COMPLETE
Stage 1: if [GuardHole]  then [ActionHole]    ← both holes unfilled
(structure not finalized — might add more stages)
else MoveToGoal
```

This partial program has 3 unfilled "holes":
1. `GuardHoleRef(1)` — what guard should stage 1 test?
2. `ActionHoleRef(1)` — what action should stage 1 take?
3. `StructureHole` — should we add stage 2, or finalize here?

### The two completion orders — explicit step-by-step traces

Below is the same search finding the D=3 canonical controller, shown under each order. The canonical target is:

```
Stage 0: if PickReady(0) then Pick(0)
Stage 1: if NeedKey(0)   then MoveToKey(0)
Stage 2: if PickReady(1) then Pick(1)
Stage 3: if NeedKey(1)   then MoveToKey(1)
else MoveToGoal
```

#### Order A: Leftmost (interleaved)

Fill each stage completely before deciding whether to add another.

```
Step 0  [initial state]
        (no stages, structure not finalized)
        Hole selected: StructureHole
        Action: ADD STAGE → creates Stage(GuardHole, ActionHole)

Step 1  Stage 0: if [GuardHole] then [ActionHole]
        Hole selected: GuardHoleRef(0)        ← leftmost unfilled hole
        Action: fill with PickReady(0)

Step 2  Stage 0: if PickReady(0) then [ActionHole]
        Hole selected: ActionHoleRef(0)       ← next leftmost
        Action: fill with Pick(0)

Step 3  Stage 0: if PickReady(0) then Pick(0)   ← stage 0 complete
        Hole selected: StructureHole          ← all stages complete, decide structure
        Action: ADD STAGE → creates Stage(GuardHole, ActionHole)

Step 4  Stage 0: if PickReady(0) then Pick(0)
        Stage 1: if [GuardHole] then [ActionHole]
        Hole selected: GuardHoleRef(1)
        Action: fill with NeedKey(0)

Step 5  Stage 1: if NeedKey(0) then [ActionHole]
        Hole selected: ActionHoleRef(1)
        Action: fill with MoveToKey(0)

Step 6  StructureHole → ADD STAGE

Step 7  GuardHoleRef(2) → fill with PickReady(1)

Step 8  ActionHoleRef(2) → fill with Pick(1)

Step 9  StructureHole → ADD STAGE

Step 10 GuardHoleRef(3) → fill with NeedKey(1)

Step 11 ActionHoleRef(3) → fill with MoveToKey(1)

Step 12 StructureHole → FINALIZE

        TERMINAL — compile and evaluate.
        Total decisions: 13 (4 structure + 4 guards + 4 actions + 1 finalize)
```

**Branching at each step:**
- StructureHole: 2 choices (add stage / finalize)
- GuardHoleRef: 11 choices (one per guard atom, at depth 0)
- ActionHoleRef: 5 choices

#### Order B: Structure-first (scaffold then fill)

Decide all stage slots first, then fill left-to-right.

```
Step 0  (no stages, structure not finalized)
        Hole selected: StructureHole
        Action: ADD STAGE

Step 1  Stage 0: if [GuardHole] then [ActionHole]
        Hole selected: StructureHole          ← structure-first always picks this
        Action: ADD STAGE

Step 2  Stage 0: if [GuardHole] then [ActionHole]
        Stage 1: if [GuardHole] then [ActionHole]
        Hole selected: StructureHole → ADD STAGE

Step 3  StructureHole → ADD STAGE  (now 4 stages, all empty)

Step 4  StructureHole → FINALIZE   (committed to 4 stages)

        Now fill left-to-right:

Step 5  GuardHoleRef(0)  → PickReady(0)
Step 6  ActionHoleRef(0) → Pick(0)
Step 7  GuardHoleRef(1)  → NeedKey(0)
Step 8  ActionHoleRef(1) → MoveToKey(0)
Step 9  GuardHoleRef(2)  → PickReady(1)
Step 10 ActionHoleRef(2) → Pick(1)
Step 11 GuardHoleRef(3)  → NeedKey(1)
Step 12 ActionHoleRef(3) → MoveToKey(1)

        TERMINAL — compile and evaluate.
        Total decisions: 13 (5 structure + 4 guards + 4 actions)
```

**Key difference from A:** All structural decisions happen upfront. The search tree branches on structure first (how many stages?), then fills within each branch. Programs with different stage counts live in completely separate subtrees.

### What we measure

For each completion order, the best-first search explores the tree of partial programs. We record:

| Metric | What it tells us |
|--------|-----------------|
| Candidates expanded | Total nodes popped from the priority queue |
| Complete programs evaluated | Terminal nodes compiled and tested on the Doors env |
| Time to first solver | Wall-clock seconds until first program that solves the env |
| Semantically distinct partials | Unique partial signatures encountered (measures dedup power) |
| Branch factor by depth | How wide the tree is at each decision level |
| Dedup hit rate | Fraction of children pruned by semantic dedup |

### Partial semantic signatures (for dedup)

A partial program's signature records, for each frozen state, what action it would select — or `UNRESOLVED` if the answer depends on a hole.

**Example 1:** After step 6 above (stage 0 guard = PickReady(0), stage 0 action = hole, rest empty):

| State | Stage 0 guard | Outcome |
|-------|--------------|---------|
| S0 (initial) | PickReady(0) = false | falls through → GuardHole at stage 1 → **UNRESOLVED** |
| S1 (key0 picked) | PickReady(0) = false | → **UNRESOLVED** |
| S2 (key1 picked) | PickReady(0) = false | → **UNRESOLVED** |
| S3 (at key0 loc) | PickReady(0) = true, action = ActionHole → **UNRESOLVED** |

Signature: `(-1, -1, -1, -1)` where -1 = UNRESOLVED.

**Example 2:** After step 8 (stages 0 and 1 complete, stages 2 and 3 still holes):

| State | Evaluation trace | Action |
|-------|-----------------|--------|
| S0 | PickReady(0)=false → NeedKey(0)=true → MoveToKey(0) | index 1 |
| S1 | PickReady(0)=false → NeedKey(0)=true → MoveToKey(0) | index 1 |
| S2 | PickReady(0)=false → NeedKey(0)=true → MoveToKey(0) | index 1 |
| S3 | PickReady(0)=true → Pick(0) | index 6 |

Signature: `(1, 1, 1, 6)` — fully resolved even though stages 2 and 3 are still holes (no state reaches them).

**Dedup rule:** If two partial programs have the same signature and the new one has equal-or-higher cost, skip it.

### Search algorithm

Best-first search with a min-heap priority queue. Priority = `stage_program_cost()` (sum of guard node counts for filled guards; holes count as 0). Lower cost explored first.

The search is **exact** (not MCTS, not AlphaZero). It explores all reachable partial programs until the queue is empty or a max-expansion cap is hit. The **only variable** across the three runs is the hole-selection policy.

### Implementation plan

**New files** (all under `src/alphazeropp/instances/doors/dsl/`):

| File | Contents |
|------|----------|
| `stage_partial.py` | `PartialStageProgram`, `StructureHole`, `GuardHoleRef`, `ActionHoleRef`, `list_holes()`, `apply_filling()` |
| `stage_partial_eval.py` | `partial_semantic_signature()`, `partial_impact()`, guard compilation cache |
| `stage_hole_selection.py` | `HoleSelectionPolicy` ABC, `LeftmostPolicy`, `StructureFirstPolicy` |
| `stage_search.py` | `best_first_search()`, `SearchStats`, `SearchResult` |
| `stage_completion_experiment.py` | `run_comparison()`, `format_comparison_table()` |

**Scripts and tests:**

| File | Purpose |
|------|---------|
| `scripts/run_stage_completion_orders.py` | CLI: `--d 3 --max-stages 3` runs the 2-way comparison |
| `scripts/enumerate_stage_dsl.py` | CLI: prints stage DSL analysis (fills the 1b visibility gap) |
| `tests/test_stage_completion.py` | Unit tests for partials, signatures, policies, search |

**Report output:** `spec/report-doors-stage-completion-orders.md`

### Acceptance checks

1. D=2 smoke test runs in <10s, both policies find the canonical solver
2. D=3 exact comparison runs end-to-end with max_stages=3
3. Both policies find the same set of solving programs (different expansion order, same final set)
4. Semantic dedup can be toggled on/off (`--no-dedup`)
5. Report states whether leftmost-hole should remain the default
6. All tests pass (`pytest tests/test_stage_completion.py tests/test_stage_dsl.py -v`)

---

## Deliverables

### New source files (all under `src/alphazeropp/instances/doors/dsl/`)

| File | What it contains |
|------|-----------------|
| `stage_partial.py` | `PartialStageProgram` dataclass, `StructureHole`, `GuardHoleRef`, `ActionHoleRef` hole reference types, `list_holes()`, `apply_filling()`, `to_stage_program()` |
| `stage_partial_eval.py` | `partial_semantic_signature()` for computing action signatures on partial programs with holes, guard compilation cache |
| `stage_hole_selection.py` | `HoleSelectionPolicy` abstract base class, `LeftmostPolicy` (interleaved), `StructureFirstPolicy` (scaffold then fill) |
| `stage_search.py` | `best_first_search()` shared search algorithm, `SearchStats` and `SearchResult` dataclasses |
| `stage_completion_experiment.py` | `run_comparison()` experiment runner, `format_comparison_table()` report formatter |

### New test file

| File | What it tests |
|------|--------------|
| `tests/test_stage_completion.py` | Partial program construction, hole listing/filling, partial semantic signatures, each hole-selection policy on handcrafted examples, D=2 smoke for best-first search, dedup toggle |

### New scripts

| File | What it does |
|------|-------------|
| `scripts/run_stage_completion_orders.py` | CLI entry point: `--d 3 --max-stages 3` runs both orders, prints comparison table, optionally writes report |
| `scripts/enumerate_stage_dsl.py` | CLI entry point: prints stage DSL analysis (guard atoms, search space sizes, behavioral trace) — fills the 1b visibility gap |

### Report

| File | Contents |
|------|---------|
| `spec/report-doors-stage-completion-orders.md` | Definition of search state, definitions of both orders, comparison tables for all metrics at D=3, recommendation on default order, section on why D=2 is not meaningful |

### Modified files

| File | Change |
|------|--------|
| `src/alphazeropp/instances/doors/dsl/__init__.py` | Add exports for `stage_partial`, `stage_hole_selection`, `stage_search` |
