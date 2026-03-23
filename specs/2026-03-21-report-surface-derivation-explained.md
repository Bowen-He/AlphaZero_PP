# Surface Derivation: A Typed Grammar for Doors Program Synthesis

**Audience:** collaborators familiar with `scripts/run_doors_derivation.py` (the original
budget-grammar derivation game).

---

## 1  Motivation

The original derivation game searches over **all** syntactically valid AST programs up to a
budget.  For a 2-room Doors instance the optimal program has 12 AST nodes, and the grammar
at budgets 2..18 admits roughly **5 million** canonical programs.  Almost all of them are
semantically useless — they test irrelevant observation bits or execute actions with failed
preconditions.  The resulting reward landscape is nearly flat ("reward desert"), which
starves MCTS of gradient signal.

**Surface derivation** replaces the node-level search with a search over **rule orderings**.
Each rule is a domain-aware macro (e.g. "if ready to pick key k, pick it") that is
guaranteed to be semantically meaningful.  The compiled output is a standard AST executable
by the existing interpreter — no runtime changes are needed.

| Property | Original (budget grammar) | Surface |
|---|---|---|
| Search unit | single AST node | domain rule macro |
| Action space per step | ~200–300 productions | 2K+1 rules |
| Episode length | ~budget (12–100+) | 2K+1 (fixed) |
| Dead ends | yes | no |
| Programs that solve (D=3) | < 1% | 50% |

---

## 2  Background: The Doors Environment

### 2.1  Physical layout

There are **D** rooms, **M = 2D** locations (2 per room), and **K = D−1** keys.

- Room 0 starts **unlocked**; rooms 1..D−1 start **locked**.
- Key k sits at location `key_loc[k]` and unlocks room `key_unlocks[k]`.
- The agent starts at location 0.  The **goal** is to reach location M−1.

```
D = 3 example:

  Room 0 (unlocked)    Room 1 (locked)     Room 2 (locked)
  ┌──────────────┐     ┌──────────────┐    ┌──────────────┐
  │ loc 0 (start)│     │ loc 2        │    │ loc 4        │
  │ loc 1 [key0] │     │ loc 3 [key1] │    │ loc 5 (GOAL) │
  └──────────────┘     └──────────────┘    └──────────────┘
```

### 2.2  Observation vector  (size n_sites = M + 2D − 1)

| Indices | Semantic | Example D=3 |
|---|---|---|
| 0 .. M−1 | one-hot agent location | `at_loc[0..5]` |
| M .. M+D−1 | room unlock status (1=open) | `unlocked[0..2]` at indices 6,7,8 |
| M+D .. M+2D−2 | key availability (1=available) | `key_avail[0..1]` at indices 9,10 |

Initial state for D=3: `[1,0,0,0,0,0,  1,0,0,  1,1]`.

### 2.3  Actions  (Discrete, size M + K + 1)

| Range | Semantic | D=3 |
|---|---|---|
| 0 .. M−1 | MOVE_TO(location) | 0..5 |
| M .. M+K−1 | PICK(key) | 6,7 |
| M+K | NOOP | 8 |

`MOVE_TO(l)` succeeds only if `unlocked[loc_room[l]] = 1`.
`PICK(k)` succeeds only if the agent is at `key_loc[k]` **and** `key_avail[k] = 1`.
Failed preconditions silently become NOOP (step penalty still applies).

### 2.4  Rewards

| Event | Reward |
|---|---|
| Each step | −0.01 |
| Successful PICK (unlocks a room) | +0.10 |
| Reach goal location | +1.00 |

Optimal reward for D rooms: `1.0 + (D−1)×0.1 − (2(D−1)+1)×0.01`.

---

## 3  The Original Approach (Budget-Grammar Derivation)

### 3.1  AST node types

| Node | Meaning | Cost |
|---|---|---|
| `Flip(j)` | execute environment action j | 1 |
| `IsZero(j)` | test obs[j] == 0 | 1 |
| `Not(c)` | logical negation | 1 + child |
| `And(c1, c2)` | conjunction | 1 + children |
| `Ite(c, Flip(j), P)` | if c then action j else program P | 1 + c + 1 + P |
| `Default(Flip(j))` | always execute action j (base case) | 2 |

**Important:** because the observation uses 1 for "true" and 0 for "false",
`IsZero(j)` **negates** the semantic meaning.
- `Not(IsZero(1))` means "agent IS at location 1".
- `IsZero(7)` means "room 1 IS locked".

### 3.2  Budget-constrained CFG

The grammar uses two hole types with an integer budget:

```
ProgramHole(k):
  k = 2  →  Default(Flip(j))                    for j in [0, n_actions)
  k ≥ 5  →  Ite(ConditionHole(i), Flip(j),      for i in [1, k−4],
                 ProgramHole(k−2−i))              j in [0, n_actions)

ConditionHole(k):
  k = 1  →  IsZero(j)                           for j in [0, n_sites)
  k ≥ 2  →  Not(ConditionHole(k−1))
  k ≥ 3  →  And(ConditionHole(i), ConditionHole(k−1−i))
```

### 3.3  How the original derivation game works

1. Start with a single `ProgramHole(budget)` at the root.
2. At each step, the agent selects a **production** that expands the leftmost hole.
3. The game ends when no holes remain (complete program).
4. Terminal reward = `LeafEvaluator(program)` (runs the program on test states).
5. Intermediate reward = 0.

**The problem:**  At D=2 with budget 12 (optimal), the grammar generates ~5 million
distinct programs.  Over 99% score identically on the environment, producing a
near-constant reward landscape.  MCTS cannot distinguish good from bad partial derivations.

---

## 4  The Surface DSL

### 4.0  What are we building?

The goal is to synthesise a **decision list** — a program that, given an observation of the
Doors environment, decides which single action to execute.  A decision list is an ordered
chain of if-then-else rules tried **top to bottom**: the first rule whose condition matches
fires, and the rest are skipped.

In pseudocode, a decision list for D=2 looks like:

```
if   <condition 1>  then  <action 1>
elif <condition 2>  then  <action 2>
else                      <default action>
```

In the AST representation used by the interpreter, this becomes a **right-leaning binary
tree** of `Ite` (if-then-else) nodes, where each node's "else" branch points to the next
rule in the chain:

```
        Ite                          ← rule 1
       / | \
    cond1 act1  Ite                  ← rule 2
               / | \
            cond2 act2  Default      ← fallback
                         |
                        act3
```

Each `Ite` node has three children: a condition subtree, an action leaf, and the else-branch
(which is either another `Ite` or the terminal `Default`).

**The surface DSL defines the building blocks for this tree** — the conditions, actions, and
rules — using domain-aware names like `PickReady(k)` instead of raw observation indices.
The **compiler** then translates each surface rule into AST nodes and chains them into the
nested tree.

### 4.1  Surface conditions (what can be tested)

Each surface condition is a named, typed test of the environment state.  The "Compiled AST"
column shows what raw AST nodes it becomes after compilation.

| Surface condition | Meaning | Compiled AST |
|---|---|---|
| `AtKeyLoc(k)` | agent is at key k's location | `Not(IsZero(key_loc[k]))` |
| `KeyAvail(k)` | key k is available | `Not(IsZero(M + D + k))` |
| `RoomLocked(r)` | room r is locked | `IsZero(M + r)` |
| `PickReady(k)` | at key k AND key k available | `And(AtKeyLoc(k), KeyAvail(k))` |
| `NeedKey(k)` | room that key k unlocks is locked | `IsZero(M + key_unlocks[k])` |

**Concrete example (D=2):**  `M=4, D=2, key_loc=[1], key_unlocks=[1]`
- `PickReady(0)` = "at location 1 AND key 0 available"
  → `And(Not(IsZero(1)), Not(IsZero(6)))` — 5 AST nodes
- `NeedKey(0)` = "room 1 is locked"
  → `IsZero(5)` — 1 AST node

### 4.2  Surface actions (what can be done)

| Surface action | Meaning | Compiled AST |
|---|---|---|
| `Pick(k)` | pick up key k | `Flip(M + k)` |
| `MoveToKey(k)` | move to key k's location | `Flip(key_loc[k])` |
| `MoveToGoal` | move to goal location | `Flip(goal_loc)` |

**What is `goal_loc`?**  The goal location is always the *last* location in the layout:
`goal_loc = M − 1 = 2D − 1`.  It sits in the final room — the room the agent must
ultimately reach.

| D | M (locations) | goal_loc | Physical meaning |
|---|---|---|---|
| 2 | 4 | 3 | last location in room 1 |
| 3 | 6 | 5 | last location in room 2 |
| 4 | 8 | 7 | last location in room 3 |

`MoveToGoal` compiles to `Flip(goal_loc)`, which is the `MOVE_TO` environment action
targeting that location.  For D=2, `MoveToGoal` → `Flip(3)` = `MOVE_TO(location 3)`.
For D=3, `MoveToGoal` → `Flip(5)` = `MOVE_TO(location 5)`.  This value is computed in
`DoorsGameConfig.__post_init__()` as `self.goal_loc = len(self.loc_room) − 1`
(see `doors_config.py`).

**Concrete example (D=2):**
- `Pick(0)` → `Flip(4)` (action index 4 = PICK key 0)
- `MoveToKey(0)` → `Flip(1)` (action index 1 = MOVE to location 1)
- `MoveToGoal` → `Flip(3)` (action index 3 = MOVE to goal location)

### 4.3  Surface rules (the "production rules")

A **surface rule** is a self-contained if-then template that pairs one condition with one
action.  These are the "production rules" or "macros" — each one becomes a single `Ite`
node (or `Default` node) in the final AST:

| Rule | English | AST template | AST nodes |
|---|---|---|---|
| `PickRule(k)` | "If I can pick key k, pick it; else try next rule" | `Ite(PickReady(k), Pick(k), <else>)` | 7 |
| `MoveRule(k)` | "If I still need key k, go toward it; else try next rule" | `Ite(NeedKey(k), MoveToKey(k), <else>)` | 3 |
| `GoalRule` | "Move to goal (always)" | `Default(MoveToGoal)` | 2 |

The `<else>` placeholder is where the **next rule** in the chain gets plugged in.
`GoalRule` has no else — it is the fallback at the bottom of every decision list.

**Node counts explained (D=2):**
- `PickRule(0)` → `Ite(And(Not(IsZero(1)), Not(IsZero(6))), Flip(4), <else>)`
  Count: Ite(1) + And(1) + Not(1) + IsZero(1) + Not(1) + IsZero(1) + Flip(1) = **7**
- `MoveRule(0)` → `Ite(IsZero(5), Flip(1), <else>)`
  Count: Ite(1) + IsZero(1) + Flip(1) = **3**
- `GoalRule` → `Default(Flip(3))`
  Count: Default(1) + Flip(1) = **2**

### 4.4  Policy = an ordered sequence of rules (a decision list)

A **SurfacePolicy** is an ordered tuple of rules ending with `GoalRule`.  **Order matters**
because the program is a decision list — the first rule whose condition is true fires, and
all later rules are skipped.

For D=2, the only policy is:

```
PickRule(0)  >>  MoveRule(0)  >>  GoalRule
```

This reads as a priority list, tried top to bottom on each timestep:

```
Priority 1:  if PickReady(0)  then Pick(0)       ← can I pick key 0? do it
Priority 2:  if NeedKey(0)    then MoveToKey(0)   ← do I still need key 0? go get it
Priority 3:  (default)             MoveToGoal     ← otherwise head to goal
```

For D=3, the canonical policy is:

```
PickRule(0)  >>  MoveRule(0)  >>  PickRule(1)  >>  MoveRule(1)  >>  GoalRule
```

Which reads as:

```
Priority 1:  if PickReady(0)  then Pick(0)        ← can I pick key 0? do it
Priority 2:  if NeedKey(0)    then MoveToKey(0)    ← need key 0? go get it
Priority 3:  if PickReady(1)  then Pick(1)         ← can I pick key 1? do it
Priority 4:  if NeedKey(1)    then MoveToKey(1)    ← need key 1? go get it
Priority 5:  (default)             MoveToGoal      ← otherwise head to goal
```

### 4.5  Compilation: translating the rule sequence into a nested AST

**"Compilation"** is the mechanical process of turning a flat list of surface rules into the
nested AST tree that the interpreter can execute.  The key idea: each rule's `<else>`
placeholder must point to the rest of the chain.

#### Why right-to-left?

We build the tree from the **last** rule (innermost) to the **first** (outermost).  This is
because each rule needs to know its else-branch (the sub-tree for all subsequent rules)
before it can be constructed.  Starting from the end gives us that sub-tree at each step —
no backpatching needed.

#### Step-by-step compilation for D=2

**Input:** `PickRule(0)  >>  MoveRule(0)  >>  GoalRule`

**Step 1** — Compile the last rule (GoalRule):
```
prog = Default(Flip(3))

Tree so far:
    Default
      |
    Flip(3)
```

**Step 2** — Wrap with MoveRule(0), plugging `prog` into the else-branch:
```
prog = Ite(IsZero(5), Flip(1), prog)

Tree so far:
       Ite
      / | \
 IsZero Flip  Default
   (5)  (1)    |
              Flip(3)
```

**Step 3** — Wrap with PickRule(0), plugging `prog` into the else-branch:
```
prog = Ite(And(Not(IsZero(1)), Not(IsZero(6))), Flip(4), prog)

Final tree:
              Ite                                 ← PickRule(0)
            /  |  \
         And  Flip(4)  Ite                        ← MoveRule(0)
        / \           / | \
     Not   Not   IsZero Flip  Default             ← GoalRule
      |     |     (5)   (1)    |
  IsZero IsZero              Flip(3)
    (1)    (6)
```

Total AST nodes: 7 + 3 + 2 = **12** (matches the D=2 optimal program size).

The interpreter evaluates this tree top-down: check the root Ite's condition first.  If
true, execute Flip(4).  If false, descend into the else-branch (the next Ite), and so on.

#### Step-by-step compilation for D=3

**Input:** `PickRule(0)  >>  MoveRule(0)  >>  PickRule(1)  >>  MoveRule(1)  >>  GoalRule`

**Step 1** — GoalRule:
```
prog = Default(Flip(5))
```

**Step 2** — Wrap with MoveRule(1):
```
prog = Ite(IsZero(8), Flip(3), prog)
```

**Step 3** — Wrap with PickRule(1):
```
prog = Ite(And(Not(IsZero(3)), Not(IsZero(10))), Flip(7), prog)
```

**Step 4** — Wrap with MoveRule(0):
```
prog = Ite(IsZero(7), Flip(1), prog)
```

**Step 5** — Wrap with PickRule(0):
```
prog = Ite(And(Not(IsZero(1)), Not(IsZero(9))), Flip(6), prog)
```

**Final tree (D=3, 22 nodes):**
```
Ite ──────────────────────────────────────── PickRule(0)
├── And(Not(IsZero(1)), Not(IsZero(9)))
├── Flip(6)                                  [Pick key 0]
└── Ite ──────────────────────────────────── MoveRule(0)
    ├── IsZero(7)
    ├── Flip(1)                              [Move to key 0]
    └── Ite ──────────────────────────────── PickRule(1)
        ├── And(Not(IsZero(3)), Not(IsZero(10)))
        ├── Flip(7)                          [Pick key 1]
        └── Ite ──────────────────────────── MoveRule(1)
            ├── IsZero(8)
            ├── Flip(3)                      [Move to key 1]
            └── Default ─────────────────── GoalRule
                └── Flip(5)                  [Move to goal]
```

Total: 7 + 3 + 7 + 3 + 2 = **22** nodes.

The tree is right-leaning: the left child of each Ite is shallow (condition + action), while
the right child (else-branch) contains the entire rest of the chain.  This shape is
characteristic of decision lists / if-elif-else chains.

### 4.5a  Compiler internals: from surface atoms to raw AST

The step-by-step examples above show the *output* of compilation.  This section explains
the *code* that produces it — which functions are called, how they resolve domain-aware
names to raw observation indices, and how the project's three compiler layers relate.

#### The three-layer compiler architecture

The project implements three DSL layers.  Each adds progressively more abstraction on top
of the same raw AST target:

```
Layer 1 — Surface compiler (surface_compiler.py)
  Input:  SurfacePolicy (flat rule sequence)
  Output: Program (Ite / Default tree)
  Resolves: typed condition/action names → raw IsZero/Flip indices

Layer 2 — Stage compiler (stage_compiler.py)
  Input:  StageProgram (stages with guard combinators Not/And)
  Output: Program
  Resolves: guard combinators → nested AST condition trees
  Delegates surface atoms to Layer 1

Layer 3 — Lifted compiler (lifted_compiler.py)
  Input:  LiftedPolicy (abstract selectors like CurrentRoom, KeyFor)
  Output: Program
  Resolves: ForEachLockedRoom loop → concrete per-key stages
  Delegates everything else to the relational runtime (DoorsRelationalRuntime)
```

All three layers produce the **same `Program` type** (`Ite | Default` from `ast_nodes.py`),
so the interpreter, leaf evaluator, and all downstream tooling work identically regardless
of which compiler produced the tree.

#### Code walkthrough: the surface compiler

The surface compiler (`surface_compiler.py`) has three public functions:

```python
def compile_condition(cond: SurfaceCondition, cfg: DoorsGameConfig) -> Condition
def compile_action(act: SurfaceAction, cfg: DoorsGameConfig) -> Flip
def compile_policy(policy: SurfacePolicy, cfg: DoorsGameConfig) -> Program
```

**`compile_condition`** is a pattern match over the five condition types.  Each branch
translates a domain name into raw observation indices using `cfg`:

```
AtKeyLoc(k)   → Not(IsZero(cfg.key_loc[k]))         # obs[key_loc[k]] != 0
KeyAvail(k)   → Not(IsZero(cfg.M + cfg.D + k))       # obs[M+D+k] != 0
RoomLocked(r) → IsZero(cfg.M + r)                     # obs[M+r] == 0
PickReady(k)  → And(compile(AtKeyLoc(k)), compile(KeyAvail(k)))
NeedKey(k)    → IsZero(cfg.M + cfg.key_unlocks[k])    # obs[M+unlocks[k]] == 0
```

The key insight is that **every index comes from `DoorsGameConfig`**, never from a
hard-coded constant.  The same compiler works for any D as long as `cfg` is set up
correctly.

**`compile_action`** maps the three action types to environment action indices:

```
Pick(k)      → Flip(cfg.M + k)           # action M+k = PICK(key k)
MoveToKey(k) → Flip(cfg.key_loc[k])      # action key_loc[k] = MOVE_TO(key location)
MoveToGoal() → Flip(cfg.goal_loc)        # action goal_loc = MOVE_TO(goal)
```

**`compile_policy`** chains rules right-to-left (as described in Section 4.5):

```python
prog = Default(compile_action(GoalRule.action, cfg))   # innermost
for rule in reversed(policy.rules[:-1]):               # wrap outward
    cond = compile_condition(rule.condition, cfg)
    act  = compile_action(rule.action, cfg)
    prog = Ite(cond, act, prog)
return prog
```

#### The stage compiler: adding guard combinators

The stage compiler (`stage_compiler.py`) extends the surface vocabulary with boolean
combinators `GuardNot` and `GuardAnd`, allowing complex guard expressions like
`GuardAnd(PickReady(0), GuardNot(RoomLocked(1)))`.

Its key function `compile_guard` is recursive:

```python
def compile_guard(guard: GuardExpr, cfg: DoorsGameConfig) -> Condition:
    if isinstance(guard, GuardNot):
        return AstNot(compile_guard(guard.child, cfg))
    if isinstance(guard, GuardAnd):
        return AstAnd(compile_guard(guard.left, cfg), compile_guard(guard.right, cfg))
    # base case: a surface condition atom
    return compile_condition(guard, cfg)   # delegates to surface compiler
```

`compile_stage_program` assembles the full tree using the same right-fold pattern:

```python
def compile_stage_program(prog: StageProgram, cfg: DoorsGameConfig) -> Program:
    result = Default(compile_action(prog.default_action, cfg))
    for stage in reversed(prog.stages):
        cond = compile_guard(stage.guard, cfg)
        act  = compile_action(stage.action, cfg)
        result = Ite(cond, act, result)
    return result
```

The stage compiler **reuses** `compile_condition` and `compile_action` from the surface
compiler for all leaf-level atoms.  It only adds the recursive combinator handling on top.

#### The lifted compiler: D-independent programs

The lifted compiler (`lifted_compiler.py`) introduces **abstract selectors** — symbolic
references like `CurrentRoom` and `KeyFor(CurrentRoom)` that are not tied to any specific
key index.  A single lifted program works for *any* D.

The compilation uses `DoorsRelationalRuntime` (from `relational_runtime.py`) to expand
abstract queries at compile time:

```python
def compile_lifted_policy(policy: LiftedPolicy, rt: DoorsRelationalRuntime) -> Program:
    prog = Default(rt.flip_move_to_goal())
    for r in reversed(rt.lockable_rooms()):          # rooms 1..D-1
        k = rt.key_for_room(r)                        # resolve CurrentRoom → key index
        for rule in reversed(policy.body.rules):
            cond   = _compile_predicate(rule.predicate, rt, k)
            action = _compile_action(rule.action, rt, k)
            prog = Ite(cond, action, prog)
    return prog
```

The critical difference from the surface/stage compilers: the lifted compiler **never**
accesses `cfg.key_loc` or `cfg.key_unlocks` directly.  All index resolution goes through
`DoorsRelationalRuntime`, which supports both `known_map=True` (key locations available)
and `known_map=False` (locations unknown).

#### Summary: what each compiler adds

| Compiler | Extra capability over previous layer | Delegates to |
|---|---|---|
| Surface | Typed conditions/actions → raw AST indices | `DoorsGameConfig` |
| Stage | Guard combinators (`Not`, `And`) | Surface compiler for atoms |
| Lifted | Abstract selectors, D-independence | `DoorsRelationalRuntime` |

### 4.6  Key insight: sequential search, compiled tree

In the **original** budget-grammar derivation game, the search state *is* the partial AST
tree.  Each step fills one hole in the tree, and the tree grows incrementally during search.
The agent must reason about tree structure — which subtree to expand, how budget is
distributed between condition and else-branch, what node types fit where.  The search and
the program representation are the same object.

In the **surface** derivation game, there is no tree during search at all.  The search state
is a **flat list of rule names**:

```
Step 0:  ()
Step 1:  (PickRule(0),)
Step 2:  (PickRule(0), MoveRule(0))
Step 3:  (PickRule(0), MoveRule(0), GoalRule)   ← terminal
```

Each step just appends one rule to the list.  The nested `Ite` tree only materialises at the
very end, when `compile_policy()` is called on the completed sequence.

**This is a fundamental architectural shift: search operates on a sequence; the tree is a
compile-time artifact.**

The consequences are significant:

| Aspect | Original (tree-building) | Surface (sequence + compile) |
|---|---|---|
| Search state | Partial AST with typed holes | Flat tuple of rule names + bitmasks |
| Branching factor | Driven by grammar productions at each hole | Driven by which rules remain unplaced |
| State complexity | Tree topology, budget distribution, hole types | Just "which rules placed so far" |
| Tree structure | Decided incrementally during search | Decided all at once by the compiler |
| Dead ends | Yes (budget exhaustion, no legal productions) | No (at least one rule is always legal) |

The search problem is reduced from **tree construction** to **permutation selection** — choosing
an ordering of a fixed set of rules.  The structural decisions (how rules chain into
if-then-else, how conditions compile to AST nodes) are fully determined by the compiler and
require no search at all.

---

## 5  The Surface Derivation Game

The surface derivation game casts policy construction as a **fixed-length sequential
decision problem** compatible with the existing MCTS + neural-network infrastructure.

### 5.1  State

An immutable triple `(rules, picked_mask, moved_mask)`:
- `rules`: tuple of rules placed so far
- `picked_mask`: bitmask, bit k set iff `PickRule(k)` has been placed
- `moved_mask`: bitmask, bit k set iff `MoveRule(k)` has been placed

### 5.2  Action space  (Discrete, size 2K+1)

Recall that a Doors instance with **D** rooms has **K = D−1** keys and K locked doors.
The MCTS agent does **not** move around the Doors world directly. Instead, it
*constructs a program* one rule at a time. Each action appends a rule to a growing
sequence of instructions. Once the sequence is complete, the resulting program is
executed in the Doors environment and the reward it earns is returned to MCTS.

The action space has **2K+1** discrete actions, laid out as follows:

| Index | Rule | Plain-English meaning |
|---|---|---|
| 0 .. K−1 | `PickRule(k)` | "Append an instruction: *if key k is available and the agent is standing at key k's location, pick it up.*" This handles acquiring key k, which unlocks the door to room k+1. |
| K .. 2K−1 | `MoveRule(k−K)` | "Append an instruction: *if room k+1 is unlocked, move the agent from its current location into room k+1.*" This handles navigating through door k. |
| 2K | `GoalRule` | "Append a final instruction: *move the agent to the goal location.*" This is always the last rule in the sequence and signals that the program is complete. |

For example, with D=3 rooms (K=2 keys), the action space has 5 actions:
`{0: PickRule(0), 1: PickRule(1), 2: MoveRule(0), 3: MoveRule(1), 4: GoalRule}`.

### 5.3  Legal mask (monotonicity constraints)

| Rule | Legal iff |
|---|---|
| `PickRule(k)` | k not yet picked AND k not yet moved |
| `MoveRule(k)` | k already picked AND k not yet moved |
| `GoalRule` | all 2K non-goal rules have been placed |

These constraints enforce that you cannot "move through" a key before "picking it up" in
the rule ordering.  They guarantee:
- No dead ends (at least one action is always legal).
- Every complete sequence has exactly **2K+1** steps.

### 5.4  Termination and reward

The episode ends when `GoalRule` is placed (step 2K+1).  At that point:

1. Build `SurfacePolicy(rules)`.
2. Compile to a raw AST via `compile_policy`.
3. Evaluate with `LeafEvaluator` → scalar reward.
4. Intermediate steps: reward = 0.

### 5.5  Observation encoding

Fixed-length vector of shape `(2 × (2K+1),)` containing `(type_id, param)` pairs:

| Token | type_id | param |
|---|---|---|
| PAD (empty slot) | 0 | 0 |
| PickRule(k) | 1 | k |
| MoveRule(k) | 2 | k |
| GoalRule | 3 | 0 |

### 5.6  Observability properties

The derivation-game agent has **full observability** of its own construction history.
Several properties are worth highlighting:

1. **Complete construction visibility.**  The observation encodes every rule placed so
   far as `(type_id, param)` pairs.  The agent sees the exact sequence it has built —
   there is no hidden state in the derivation game itself.

2. **Flat, fixed-length representation.**  The observation is a flat vector of
   `2 × (2K+1)` integers, not a tree or graph.  This makes it directly consumable by
   standard neural network architectures (fully connected or transformer) without
   requiring tree-structured encoders.

3. **Implicit remaining-action inference.**  Because the agent knows which rules it has
   placed (from the observation) and what the full rule set is (from the game definition),
   it can infer which rules remain to be placed and which are currently legal.  The legal
   mask (Section 5.3) is a deterministic function of the observation.

4. **No Doors-environment observation.**  The derivation-game agent never sees the Doors
   environment's observation vector (agent location, room status, key availability).  It
   only observes its own construction choices.  The Doors environment is invisible during
   search — it appears only at terminal evaluation, when the completed policy is compiled
   and executed to compute the reward.

This is a fundamentally different observability regime from the Doors environment itself,
where the agent has partial information about locked rooms and key locations.  The
derivation game is a **fully observable** meta-game over **program construction**, even
though the object-level game may be partially observable.

---

## 6  The Relaxed Grammar

### 6.1  Canonical policy

For each D there is exactly **one** canonical policy — the known-optimal ordering:

```
PickRule(0) >> MoveRule(0) >> PickRule(1) >> MoveRule(1) >> ... >> GoalRule
```

### 6.2  Relaxed (search) grammar

Again let **K = D−1** be the number of keys. A relaxed policy is any ordering of
the 2K non-goal rules `{PickRule(0), MoveRule(0), PickRule(1), MoveRule(1), …}`
followed by `GoalRule`, subject to one constraint:

> **Monotonicity:** For each key k, `PickRule(k)` must appear **before** `MoveRule(k)`.

In plain language: you must add the "pick up key k" instruction before the
"move through door k" instruction. You *cannot* walk through a door before
your program knows how to acquire the key that opens it.

Beyond this per-key ordering, the rules for *different* keys can be freely
interleaved. For D=3 (K=2), some valid orderings are:

```
PickRule(0) >> MoveRule(0) >> PickRule(1) >> MoveRule(1) >> GoalRule   (canonical)
PickRule(0) >> PickRule(1) >> MoveRule(0) >> MoveRule(1) >> GoalRule   (pick both keys first)
PickRule(1) >> PickRule(0) >> MoveRule(0) >> MoveRule(1) >> GoalRule   (pick key 1 first)
```

**Counting the relaxed policies:**

1. There are **2K** non-goal rules to arrange (K pick rules + K move rules).
2. Without any constraint there would be **(2K)!** orderings.
3. For each key k, exactly half of all orderings place `PickRule(k)` before
   `MoveRule(k)` (by symmetry). This is one independent constraint per key.
4. With **K** independent constraints, we divide by **2^K**.

$$\text{Relaxed policies} = \frac{(2K)!}{2^K}$$

**Worked example for D=3 (K=2).**

The rule pool is `{P0, M0, P1, M1}` (using P for PickRule, M for MoveRule).

*Step 1:* All unconstrained orderings of 4 items: `(2K)! = 4! = 24`.

*Step 2:* Apply the monotonicity constraint `P(k)` before `M(k)` for each key k:
- For k=0: exactly half of the 24 orderings have P0 before M0 (by symmetry of swapping
  P0↔M0).  That keeps 12, eliminates 12.
- For k=1: of the remaining 12, exactly half have P1 before M1 (the two constraints are
  independent).  That keeps 6, eliminates 6.

Result: `24 / 2² = 24 / 4 = 6` valid relaxed policies.  These 6 are listed explicitly
in Section 8.3.

**How solving policies are computed.**

Each relaxed policy is tested against the Doors environment:

1. **Compile:** `compile_policy(policy, cfg)` translates the surface rule sequence into a
   raw AST program (a nested `Ite`/`Default` tree).
2. **Execute:** `run_policy_episode(env, prog, x0, is_solved)` runs the compiled program
   on the Doors environment from the canonical initial state, stepping until the horizon.
3. **Check:** A policy "solves" if the agent reaches `goal_loc` within the horizon.

This is implemented in `surface_grammar.py`'s `count_solving_policies()` function.

**There is not just one solving policy.**  For D=3, **3 out of 6** relaxed policies solve
the environment (policies #1, #2, and #4 in Section 8.3).  The common pattern: a policy
solves if and only if `MoveRule(0)` appears before `MoveRule(1)` — see Section 8.7 for
why.

**Not all relaxed policies solve the environment.** The ordering determines which "else"
branch fires first, and a badly ordered policy can attempt to move through a locked room
forever.

### 6.3  Counts by D

| D | K | Relaxed policies | Solving policies | Solve fraction |
|---|---|---|---|---|
| 1 | 0 | 1 | 1 | 100% |
| 2 | 1 | 1 | 1 | 100% |
| 3 | 2 | 6 | 3 | 50% |
| 4 | 3 | 90 | 15 | 17% |
| 5 | 4 | 2,520 | 105 | 4.2% |

### 6.3a  Comparison with the budget grammar

The budget grammar (`budget_grammar.py`) generates all syntactically valid AST programs
up to a node budget.  For comparison:

| D | K | n_sites | Budget (~1.5× optimal) | Budget-grammar programs (approx) | Surface relaxed policies | Reduction factor |
|---|---|---|---|---|---|---|
| 2 | 1 | 7 | ~18 | ~5,000,000 | 1 | ~5 × 10⁶ |
| 3 | 2 | 11 | ~34 | ~52,000,000,000 | 6 | ~10¹⁰ |

At D=2, the budget grammar produces approximately **5 million** canonical programs; the
surface grammar produces exactly **1** policy.  At D=3, the budget grammar produces
approximately **52 billion** canonical programs; the surface grammar produces exactly
**6** relaxed policies — a reduction of roughly 10 orders of magnitude.

The surface grammar achieves this reduction by encoding three kinds of domain knowledge
that the budget grammar lacks:

1. **Typed conditions:** Only semantically meaningful predicates (`PickReady`, `NeedKey`)
   are expressible, vs. arbitrary `IsZero`/`Not`/`And` combinations over all `n_sites`
   observation indices.
2. **Paired condition-action bundles:** Each rule pairs the correct condition with the
   correct action (e.g., `PickReady(k)` → `Pick(k)`), eliminating all semantically
   invalid pairings.
3. **Structural constraint:** The decision list always has exactly 2K+1 rules in a fixed
   if-elif-else shape, eliminating budget-distribution choices.

The cost of this reduction is **domain knowledge**: the surface DSL requires a human to
define the rule templates.  Section 14.2 discusses what this design does not capture, and
Section 12 analyses what domain knowledge is implicitly hardcoded ("cheats").

### 6.4  Complexity of the search problem

The surface derivation game is a **combinatorial construction problem** rather than
a typical reinforcement-learning task. In most RL settings the agent receives a reward
signal after every action (e.g. +1 for scoring, −1 for falling). Here the agent
receives **zero reward at every intermediate step** and only learns the quality of its
program at the very end, when the completed instruction sequence is executed in the
Doors environment. This makes the reward signal *sparse*: the agent must commit to all
2K+1 choices before discovering whether they were good.

**Branching factor.** At each step roughly K actions are legal (the exact number depends
on which rules have already been placed). This means the MCTS search tree has on the
order of K^(2K) leaf nodes — vastly more than the (2K)!/2^K valid policies, because
MCTS also explores partial sequences that lead to the same policy via different search
paths.

**Super-exponential growth.** The number of relaxed policies grows factorially with D:

| D (rooms) | K (keys) | Relaxed policies |
|---|---|---|
| 3 | 2 | 6 |
| 5 | 4 | 2,520 |
| 8 | 7 | 681 million |
| 10 | 9 | 12.1 billion |
| 12 | 11 | 5.5 × 10^17 |
| 15 | 14 | 1.9 × 10^25 |

For D ≤ 5 the entire policy space can be enumerated exhaustively. By D = 10 the space
contains billions of policies, yet MCTS still finds a solver while exploring less than
0.001% of the space — demonstrating that the neural network learns to generalize from a
tiny sample. At D = 12 the space explodes to 10^17 and the current training budget
(~138K unique programs in 15 iterations) is negligible; no solver is found. This sharp
cliff between D = 10 and D = 12 marks the frontier where the representation and search
budget are no longer sufficient (see Section 11 for experimental results).

---

## 7  Worked Example: D = 2 (K = 1)

### 7.1  Layout

```
D=2,  M=4,  K=1,  n_sites=7,  n_actions=6
key_loc = [1],  key_unlocks = [1],  goal_loc = 3

  Room 0 (unlocked)    Room 1 (locked)
  ┌──────────────┐     ┌──────────────┐
  │ loc 0 (start)│     │ loc 2        │
  │ loc 1 [key0] │     │ loc 3 (GOAL) │
  └──────────────┘     └──────────────┘

Observation indices:
  0: at_loc[0]    1: at_loc[1]    2: at_loc[2]    3: at_loc[3]
  4: unlocked[0]  5: unlocked[1]  6: key_avail[0]
```

### 7.2  Surface game

Action space (size 3):  `{0: PickRule(0),  1: MoveRule(0),  2: GoalRule}`

There is only **1 relaxed policy** (and it solves):

```
PickRule(0) >> MoveRule(0) >> GoalRule
```

Episode trace:

| Step | Legal mask | Action | Rule placed | State after |
|---|---|---|---|---|
| 0 (reset) | `[1, 0, 0]` | — | — | `rules = ()` |
| 1 | `[1, 0, 0]` | 0 | PickRule(0) | `rules = (P0,)` |
| 2 | `[0, 1, 0]` | 1 | MoveRule(0) | `rules = (P0, M0)` |
| 3 | `[0, 0, 1]` | 2 | GoalRule | `rules = (P0, M0, G)` — terminal |

### 7.3  Compiled AST (12 nodes)

```
if And(Not(IsZero(1)), Not(IsZero(6))):    # PickReady(0): at key loc AND key avail
    Flip(4)                                 # PICK(0)
elif IsZero(5):                             # NeedKey(0): room 1 locked
    Flip(1)                                 # MOVE_TO(1) → go to key
else:
    Flip(3)                                 # MOVE_TO(3) → go to goal
```

### 7.4  Behavioural trace

A **behavioural trace** (also called an **execution trace** or **program trace** in the
program synthesis literature) is a step-by-step execution log of the compiled program
running on the Doors environment.  For each timestep it shows:

1. Which rule conditions are evaluated (top to bottom in the decision list).
2. Which condition evaluates to true (the "firing" rule), or whether the default fires.
3. What environment action is taken as a result.
4. The resulting observation (state change).

In the codebase, the `_trace_rules()` function in `interpreter.py` generates these traces
programmatically.  The traces below are formatted by hand to match the same structure.

```
Initial obs: [1, 0, 0, 0,  1, 0,  1]
  → at loc 0,  room 0 open / room 1 locked,  key 0 available

Step 1: PickReady(0)?  at_loc[1]=0 → no.
        NeedKey(0)?    unlocked[1]=0 → yes → Flip(1) = MOVE_TO(1)
  obs:  [0, 1, 0, 0,  1, 0,  1]

Step 2: PickReady(0)?  at_loc[1]=1 AND key_avail[0]=1 → yes → Flip(4) = PICK(0)
  obs:  [0, 1, 0, 0,  1, 1,  0]     (room 1 unlocked, key 0 consumed)

Step 3: PickReady(0)?  key_avail[0]=0 → no.
        NeedKey(0)?    unlocked[1]=1 → no.
        Default → Flip(3) = MOVE_TO(3) → GOAL reached!
  obs:  [0, 0, 0, 1,  1, 1,  0]

Reward: 1.0 (goal) + 0.1 (key) − 3×0.01 (steps) = 1.07
```

---

## 8  Worked Example: D = 3 (K = 2)

### 8.1  Layout

```
D=3,  M=6,  K=2,  n_sites=11,  n_actions=9
key_loc = [1, 3],  key_unlocks = [1, 2],  goal_loc = 5

  Room 0 (unlocked)    Room 1 (locked)     Room 2 (locked)
  ┌──────────────┐     ┌──────────────┐    ┌──────────────┐
  │ loc 0 (start)│     │ loc 2        │    │ loc 4        │
  │ loc 1 [key0] │     │ loc 3 [key1] │    │ loc 5 (GOAL) │
  └──────────────┘     └──────────────┘    └──────────────┘

Observation indices:
  0-5:  at_loc[0..5]
  6:    unlocked[0]     7: unlocked[1]     8: unlocked[2]
  9:    key_avail[0]   10: key_avail[1]
```

### 8.2  Surface game

Action space (size 5):
```
{0: PickRule(0),  1: PickRule(1),  2: MoveRule(0),  3: MoveRule(1),  4: GoalRule}
```

### 8.3  All 6 relaxed policies and which solve

| # | Ordering | Solves? |
|---|---|---|
| 1 | P0, M0, P1, M1, G | **yes** |
| 2 | P0, P1, M0, M1, G | **yes** |
| 3 | P0, P1, M1, M0, G | no |
| 4 | P1, P0, M0, M1, G | **yes** |
| 5 | P1, P0, M1, M0, G | no |
| 6 | P1, M1, P0, M0, G | no |

**Pattern:** a policy solves iff `MoveRule(0)` appears before `MoveRule(1)`.
This is because `MoveRule(1)` directs the agent to location 3 (inside room 1),
which requires room 1 to already be unlocked.  If `MoveRule(0)` has not yet fired
to send the agent to pick key 0 and unlock room 1, the move fails as NOOP
and the agent is stuck.

### 8.4  Canonical policy episode trace

```
PickRule(0) >> MoveRule(0) >> PickRule(1) >> MoveRule(1) >> GoalRule
```

| Step | Legal mask | Action | Rule |
|---|---|---|---|
| 0 | `[1, 1, 0, 0, 0]` | — | — |
| 1 | `[1, 1, 0, 0, 0]` | 0 | PickRule(0) |
| 2 | `[0, 1, 1, 0, 0]` | 2 | MoveRule(0) |
| 3 | `[0, 1, 0, 0, 0]` | 1 | PickRule(1) |
| 4 | `[0, 0, 0, 1, 0]` | 3 | MoveRule(1) |
| 5 | `[0, 0, 0, 0, 1]` | 4 | GoalRule — terminal |

### 8.5  Compiled AST (22 nodes)

```
if And(Not(IsZero(1)), Not(IsZero(9))):     # PickReady(0)
    Flip(6)                                  # PICK(0)
elif IsZero(7):                              # NeedKey(0): room 1 locked
    Flip(1)                                  # MOVE_TO(1) → key 0 location
elif And(Not(IsZero(3)), Not(IsZero(10))):   # PickReady(1)
    Flip(7)                                  # PICK(1)
elif IsZero(8):                              # NeedKey(1): room 2 locked
    Flip(3)                                  # MOVE_TO(3) → key 1 location
else:
    Flip(5)                                  # MOVE_TO(5) → goal
```

### 8.6  Behavioural trace

(See Section 7.4 for definition of "behavioural trace".)

```
Initial: [1,0,0,0,0,0,  1,0,0,  1,1]

Step 1: PickReady(0)? at_loc[1]=0 → no.  NeedKey(0)? unlocked[1]=0 → yes → Flip(1)
  → MOVE_TO(1).   obs: [0,1,0,0,0,0,  1,0,0,  1,1]

Step 2: PickReady(0)? at_loc[1]=1, key_avail[0]=1 → yes → Flip(6)
  → PICK(0).      obs: [0,1,0,0,0,0,  1,1,0,  0,1]    room 1 unlocked

Step 3: PickReady(0)? key_avail[0]=0 → no.  NeedKey(0)? unlocked[1]=1 → no.
        PickReady(1)? at_loc[3]=0 → no.  NeedKey(1)? unlocked[2]=0 → yes → Flip(3)
  → MOVE_TO(3).   obs: [0,0,0,1,0,0,  1,1,0,  0,1]    (room 1 open, move succeeds)

Step 4: PickReady(1)? at_loc[3]=1, key_avail[1]=1 → yes → Flip(7)
  → PICK(1).      obs: [0,0,0,1,0,0,  1,1,1,  0,0]    room 2 unlocked

Step 5: All conditions false → Default Flip(5) = MOVE_TO(5) → GOAL!
  obs: [0,0,0,0,0,1,  1,1,1,  0,0]

Reward: 1.0 + 2×0.1 − 5×0.01 = 1.15  (optimal)
```

### 8.7  Why ordering #6 (P1, M1, P0, M0, G) fails

Compiled else-chain:
```
if PickReady(1):  Pick(1)        # check key 1 first
elif NeedKey(1):  MoveToKey(1)   # if room 2 locked → MOVE_TO(3)
elif PickReady(0): Pick(0)
elif NeedKey(0):  MoveToKey(0)
else: MoveToGoal
```

```
Initial: [1,0,0,0,0,0,  1,0,0,  1,1]

Step 1: PickReady(1)? at_loc[3]=0 → no.
        NeedKey(1)? unlocked[2]=0 → yes → Flip(3) = MOVE_TO(3)
        But loc 3 is in room 1 which is LOCKED → precondition fails → NOOP!
  Agent stays at loc 0.  Reward: −0.01.

Step 2: identical to step 1 → NOOP again.
...  (repeats until horizon, never progressing)
```

The agent is permanently stuck because the first triggered MoveRule targets a locked room
and blocks all subsequent rules from firing.

---

## 9  Search Space Comparison

| D | Budget-grammar programs (budget 2..1.5×optimal) | Surface relaxed policies | Reduction factor |
|---|---|---|---|
| 2 | ~5,000,000 | 1 | ~5 × 10⁶ |
| 3 | ~10⁹ (estimated) | 6 | ~10⁸ |
| 4 | — | 90 | — |
| 5 | — | 2,520 | — |
| 10 | — | ~12,100,000 | — |

Even at D=10 where the surface grammar has 12M policies, this is vastly smaller than
the budget grammar and every policy compiles to a valid if-then-else chain (no junk
programs).

**Reward desert elimination:**  At D=3, 50% of surface policies solve the environment
vs. < 1% of budget-grammar programs.  This gives MCTS meaningful reward signal from the
start.

---

## 10  Integration with AlphaZero

The surface derivation game implements the same `Game` interface as the original
derivation game and plugs directly into the existing training loop:

```
for iteration in 1..N:
    1. SELF-PLAY: play n_games surface-rule-sequencing games using MCTS
       - Network reads (type_id, param) observation → policy π, value v
       - MCTS uses π as prior, leaf evaluator for terminal backup
       - Collect training examples (partial_obs, π_MCTS, terminal_value)

    2. TRAIN: update network on replay buffer
       - Loss = MSE(value) + cross-entropy(policy)

    3. EVALUATE: pit new network vs. best, accept if win_rate ≥ 40%
```

**Key advantages over the original derivation game:**
- **Tiny action space:** 2K+1 actions (e.g. 5 for D=3) vs. hundreds of grammar productions
- **Short fixed-length episodes:** 2K+1 steps (e.g. 5 for D=3) vs. ~budget (22+ for D=3)
- **No dead ends:** the legal mask always has ≥1 legal action
- **Dense reward signal:** 50% of D=3 policies solve, vs. <1% of raw programs

---

## 11  Experimental Results: Scaling with D

### 11.1  Setup

All runs use the default surface derivation configuration:
- **Network:** DerivationPolicyValueNet, d_model=64, n_heads=4, n_layers=2
- **MCTS:** 80 simulations/move, backup_rule=max, rollout_n=4
- **Training:** 30 self-play games/iteration, 8 parallel workers
- **Evaluation metric:** weighted (combines solve_rate + avg_reward)
- **Single seed** (43), no multi-seed aggregation

### 11.2  Mini sweep results (2026-03-21)

| D | K | Action space | Relaxed policies | Solve% | Reward | 1st Solve | Unique explored | Wall/iter |
|---|---|---|---|---|---|---|---|---|
| 3 | 2 | 5 | 6 | **100%** | +1.15 | iter 1 | 6 (100%) | ~4s |
| 5 | 4 | 9 | 2,520 | **100%** | +1.31 | iter 1 | 2,459 (98%) | ~11s |
| 8 | 7 | 15 | 681M | **100%** | +1.55 | iter 2 | 91,845 (0.01%) | ~25s |
| 10 | 9 | 19 | 12.1B | **100%** | +1.71 | iter 7 | 107K (0.001%) | ~25s |
| 12 | 11 | 23 | 5.5 x 10^17 | **0%** | -0.35 | never | 138K | ~48s |
| 15 | 14 | 29 | 1.9 x 10^25 | **0%** | -0.55 | never | 159K | ~81s |

D=3 and D=10 used 30 iterations; D=5, 8, 12, 15 used 15 iterations.

### 11.3  Analysis

**Sharp difficulty cliff between D=10 and D=12.**  D=10 solves by iteration 7 while
exploring only 107K of 12.1 billion policies (0.001% of the search space).  D=12 fails
completely in 15 iterations despite exploring a similar absolute number of programs (138K).
The transition is abrupt, not gradual.

**At D <= 10, representation dominates over learning.**  The surface grammar's structural
constraints (legal mask, monotonicity) guide MCTS so effectively that even a tiny fraction
of the policy space is sufficient to find a solver.  At D=5, the system nearly exhausts
the entire grammar (2,459 of 2,520 policies = 98% coverage) within 15 iterations.

**At D >= 12, the current training budget is insufficient.**  The ~138K unique programs
explored is negligible relative to ~10^17 policies.  The reward stays negative throughout,
meaning no solving policy was ever encountered.  This is the regime where learning
(policy prior, value guidance) would need to contribute meaningfully, but the current
network capacity and training budget cannot bridge the gap.

**Wall clock scales approximately linearly with D** (~4s/iter at D=3, ~81s/iter at D=15),
driven by the episode length (2K+1 steps) and per-step MCTS cost.

**Reward signal before solving.**  At D=10, the best reward improves from -0.65 (iter 1) to
-0.35 (iter 4) before the first solver appears at iteration 7 (reward +1.71).  This
monotonic improvement suggests MCTS is progressively discovering better orderings even
before finding a complete solution.  At D=12, the reward plateaus at -0.35 and does not
improve further, suggesting the search is stuck in a local basin.

### 11.4  Comparison with earlier surface-vs-direct results

The earlier controlled comparison (see `report-doors-surface-vs-direct.md`) at D=2
and D=3 with 5 iterations showed the surface game dramatically outperforming the direct
game.  The mini sweep extends this to D=10, confirming that the surface representation
remains effective well beyond the regime tested in that comparison.

### 11.5  Implications for the full sweep

The mini sweep identifies **D = {8, 10, 12}** as the transition zone:
- D=8 solves trivially (iter 2) -- useful as a fast sanity check
- D=10 requires meaningful search (iter 7) -- good for studying learning dynamics
- D=12 does not solve -- good for testing whether more iterations, larger networks, or
  different MCTS parameters can break through

A full sweep should focus on this range with multiple seeds and the trained-vs-random
comparison to isolate learning's contribution.

---

## 12  Implicit Domain Knowledge ("Cheats") and the Case for Reactive Grammars

The surface DSL achieves its dramatic search-space reduction (Section 6.3a) by embedding
substantial domain knowledge into the rule templates.  This section makes that knowledge
explicit, explains why it limits the surface DSL's applicability, and motivates the
reactive sketch grammar as a step toward a more general formulation.

### 12.1  What is hardcoded in the surface DSL

Each surface rule bundles a **condition** and an **action** into an opaque macro.  The
agent's only degree of freedom is the ordering of these macros.  Specifically:

| Rule | Hardcoded condition | Hardcoded action | What the agent never discovers |
|---|---|---|---|
| `PickRule(k)` | `PickReady(k)` = at key location AND key available | `Pick(k)` | "I should pick keys" and "I need to be at the key's location first" |
| `MoveRule(k)` | `NeedKey(k)` = the room key k unlocks is locked | `MoveToKey(k)` | "I should move toward needed keys" and "locked rooms require keys" |
| `GoalRule` | (unconditional) | `MoveToGoal` | "After unlocking all rooms, go to the goal" |

**The search is reduced to ordering only — the WHAT (which conditions to test, which
actions to pair with them) is fully provided by domain experts.**

Additionally, the **compiler** resolves observation and action indices at compile time.
For example, `MoveToKey(0)` compiles to `Flip(cfg.key_loc[0])` — the literal index of
key 0's location.  This requires `cfg.key_loc` to be known, which is only true when
`known_map=True` (the agent has a complete map of the environment).

### 12.2  Why this matters: the `known_map=False` barrier

In a `known_map=False` (partial observability) setting, `cfg.key_loc[k]` is unknown at
compile time — the agent must discover key locations during execution.

**Concrete example (D=2, `known_map=False`):**

The surface DSL's `MoveToKey(0)` compiles to `Flip(key_loc[0])`.  But if the agent does
not know the map, `key_loc[0]` is not available at compile time.  The compiler cannot
produce a valid `Flip` instruction because it does not know which location to target.

```
Surface compilation (known_map=True):
  MoveToKey(0) → Flip(cfg.key_loc[0]) → Flip(1)     ✓ concrete action index

Surface compilation (known_map=False):
  MoveToKey(0) → Flip(???)                            ✗ key_loc[0] unknown
```

This is not a minor limitation — it means the entire surface DSL framework is restricted
to environments where the map is fully known before execution begins.

### 12.3  How the reactive sketch addresses this

The reactive sketch DSL (`reactive_sketch_dsl.py`) uses **lifted selectors** instead of
concrete indices:

| Surface DSL (grounded) | Reactive Sketch (lifted) |
|---|---|
| `MoveToKey(0)` → `Flip(key_loc[0])` | `GoTo(LocOf(KeyFor(NextLockedRoom)))` |
| `PickReady(0)` → `And(Not(IsZero(1)), Not(IsZero(6)))` | `Pickable(KeyFor(NextLockedRoom))` |
| `NeedKey(0)` → `IsZero(5)` | implicit in `NextLockedRoom` selector |

The key differences:

1. **D-independence:** The same reactive sketch program works for any D.  The selector
   `NextLockedRoom` resolves at tick time to whichever room is currently the first locked
   room.  No key indices appear in the program text.

2. **Tick-time resolution:** Predicates like `KnownLoc(key)` are evaluated against the
   *current* observation at each timestep via a `ReactiveContext`, not compiled to fixed
   AST nodes.  This enables handling `known_map=False` via a memory module that tracks
   discovered key locations.

3. **Fixed search space:** The reactive sketch has exactly 4! = 24 candidate branch
   orderings regardless of D, compared to the surface DSL's `(2K)!/2^K` which grows
   super-exponentially.

### 12.4  The reactive sketch's own limitations

The reactive sketch improves on the surface DSL but still embeds significant domain
knowledge:

- **Fixed branch schemas:** The 4 branches (pick-if-ready, goto-key, goto-goal,
  explore-frontier) are hardcoded.  The agent chooses only their priority ordering.
- **Fixed predicates and actions:** Each branch pairs a specific predicate with a specific
  action.  The agent never discovers *what* to check or *what* to do — only *when*
  (priority order).
- **4 branches only:** The structure assumes exactly 4 strategic behaviours are needed.
  An environment requiring a 5th behaviour cannot be expressed.

In this sense, the reactive sketch is "cheating" in the same way as the surface DSL —
just with different, more general cheats (lifted selectors instead of grounded indices).

### 12.5  The desideratum: compositional grammars

The ultimate goal is a grammar where the agent discovers **both** what to check and what
to do — not just the ordering.  This requires:

1. A **predicate catalog** enumerating all meaningful conditions (e.g., `AtKeyLoc(k)`,
   `KeyAvail(k)`, `RoomLocked(r)`, `GoalReached`, ...).
2. An **action catalog** enumerating all meaningful actions (e.g., `Pick(k)`,
   `MoveToKey(k)`, `MoveToGoal`, `Noop`, ...).
3. A **compositional rule grammar** where each branch is composed by selecting one
   predicate and one action from their respective catalogs.

For a D=3 Doors instance, the predicate catalog has ~50 entries and the action catalog
has ~30 entries, giving ~1,500 possible branches per rule slot.  With 2K+1 = 5 rule
slots, the search space is approximately 1,500⁵ ≈ 7.6 × 10¹⁵ programs — far larger than
the surface DSL's 6 but far smaller than the budget grammar's ~52 billion, and every
program is semantically well-typed.

This compositional grammar would:
- Preserve the surface DSL's semantic well-typedness (no junk programs).
- Allow the agent to discover condition-action pairings, not just orderings.
- Support `known_map=False` if the predicate catalog includes observability-aware
  predicates (e.g., `KnownLoc(k)`, `Explored(r)`).

### 12.6  Summary: the spectrum of domain knowledge

| Grammar | What the agent discovers | What is hardcoded | Search space (D=3) |
|---|---|---|---|
| Budget grammar | Everything (conditions, actions, tree structure) | AST node types only | ~52 × 10⁹ |
| **Surface DSL** | Ordering of pre-built rules | Conditions, actions, rule structure | 6 |
| Reactive sketch | Ordering of pre-built branches | Branch schemas, predicates, actions | 24 |
| Compositional (future) | Condition-action pairings + ordering | Predicate/action catalogs, rule structure | ~10¹⁵ |

Each step along this spectrum trades domain knowledge for generality.  The surface DSL
and reactive sketch demonstrate that aggressive domain encoding enables tractable search,
but they also reveal the cost: the agent cannot transfer to settings where the hardcoded
assumptions break (e.g., `known_map=False` for the surface DSL).

---

## 13  Connection to the Literature

The surface derivation design draws on several established ideas.  This section positions it
relative to the relevant literature.

### 13.1  Decision lists (Rivest, 1987)

The compiled output of the surface DSL is exactly a **decision list** in the sense of
Rivest (1987, "Learning decision lists"): an ordered sequence of (condition, action) pairs
tested top-to-bottom, with a default action at the end.  Decision lists are a well-studied
hypothesis class — they are strictly more expressive than conjunctions/disjunctions but less
expressive than arbitrary boolean functions.

In the standard decision-list learning setting, the learner sees labelled examples and finds
a list that classifies them correctly.  In our setting the "labels" are implicit — we
discover which list solves the Doors environment by executing each candidate on frozen
states.

The key difference is that classical decision-list learning searches over individual
conditions (often from a fixed attribute set), while our surface DSL **fixes the condition
templates** (PickReady, NeedKey) and searches only over their **ordering**.

### 13.2  Syntax-guided synthesis (SyGuS)

The original budget-grammar derivation game is a form of **syntax-guided synthesis**
(Alur et al., 2013, "Syntax-Guided Synthesis").  In SyGuS, a context-free grammar defines
the space of candidate programs, and a search algorithm (enumerative, CEGIS, or
stochastic) explores that space.  The budget grammar plays the role of the SyGuS grammar;
MCTS plays the role of the search algorithm.

The surface DSL goes a step further than standard SyGuS.  Rather than constraining the
grammar while keeping the search over individual AST productions, it **lifts the grammar
to macro-level constructs** and reduces the search to permutation selection.  This is
analogous to moving from token-level code generation to template-level program sketching
(Solar-Lezama, 2008, "Program Synthesis by Sketching"), where the programmer provides a
skeleton and the synthesiser fills in the holes.

In our case, the "sketch" is implicit: every policy has the same structure (2K+1 rules
ending with GoalRule); the only free variable is the ordering.

### 13.3  Macro-actions and options

`PickRule(k)` and `MoveRule(k)` are **macro-actions** that bundle a condition test and an
environment action into one reusable unit.  This is conceptually related to:

- **Options** in hierarchical RL (Sutton, Precup & Singh, 1999, "Between MDPs and
  semi-MDPs"): temporally extended actions with initiation sets and termination conditions.
  Our rules are simpler — each fires for exactly one timestep — but they serve the same
  purpose of lifting the action space.
- **Macro-operators** in automated planning (Botea, Müller & Schaeffer, 2005, "Macro-FF"):
  bundled sequences of planning operators that reduce search depth.  Our PickRule(k) is
  essentially a macro that replaces 7 grammar productions with one search decision.

The budget-grammar's `doors_macros.py` already introduced PickRule/MoveRule as macro
*productions* within the CFG.  The surface DSL takes this further by making macros the
*only* search actions, eliminating the raw grammar entirely during search.

### 13.4  Domain-specific languages for synthesis

The surface DSL is a **domain-specific language (DSL)** in the tradition of:

- **FlashFill** (Gulwani, 2011): a typed DSL for string transformations where every
  expressible program is semantically meaningful.
- **FlashMeta** (Polozov & Gulwani, 2015): a framework for building synthesis engines
  over DSLs by decomposing specifications.

The shared principle is **type-directed search-space restriction**: by building domain
knowledge into the language (conditions test meaningful predicates, actions correspond to
real environment operations), the fraction of semantically valid programs rises from < 1%
(generic grammar) to 50–100% (surface DSL).

### 13.5  AlphaZero for combinatorial optimisation

Using AlphaZero (Silver et al., 2018) for program synthesis is a single-player variant where
the "game" is program construction.  The surface derivation game turns this into a
**permutation-selection problem** — choosing an ordering of a fixed set of items — which is
closely related to scheduling and travelling-salesman formulations that have been studied
with MCTS (e.g., Laterre et al., 2019, "Ranked Reward" for combinatorial problems).

The surface game's fixed episode length (2K+1), absence of dead ends, and high solve-rate
density make it a much better fit for MCTS than the original tree-building game, which
violates most assumptions MCTS was designed for (Section 2 of the context document).

---

## 14  Design Choices and Future Directions

### 14.1  Why sequence-only rule placement (the current "1a" design)

The current surface derivation game is the **simplest possible** search over rule orderings:
each step appends one pre-built rule to a flat list.  This was a deliberate choice:

- **Minimal implementation surface:** the game class is ~250 lines and reuses the existing
  `Game` interface, `LeafEvaluator`, and `DerivationPolicyValueNet` without modification.
- **Exact analysis at small D:** at D=3 there are only 6 complete policies and 25 prefix
  states — small enough for exhaustive enumeration and oracle-value computation.
- **Clean comparison baseline:** by holding rule templates fixed and varying only the
  ordering, we isolate the effect of ordering from the effect of rule structure.

### 14.2  What the current design does NOT capture

Each surface rule (e.g., `PickRule(k)`) is an **opaque atom** during search.  It bundles
three logically separable decisions:

| Decision | Example for PickRule(0) |
|---|---|
| **Stage kind** | "this is a pick-type rule" (vs. move-type or goal) |
| **Guard** | `PickReady(0)` = And(AtKeyLoc(0), KeyAvail(0)) |
| **Action** | `Pick(0)` = Flip(M + 0) |

In the current design, choosing `PickRule(0)` commits to all three at once.  There is no
way to say "I want a pick-type rule here, but I haven't decided which key yet" or "I want
to test NeedKey(k), but I haven't decided what to do if it's true."

This matters because:

1. **Different sub-decisions have different information requirements.**  Choosing the stage
   kind (pick vs. move) can often be decided from the global structure alone, while choosing
   the specific key index requires local reasoning about which rooms are still locked.

2. **The number of sub-decisions grows differently.**  For D rooms and K keys, there are
   2K+1 atomic rules (current design), but the stage-skeleton decomposition has 2K+1
   stage-kind slots × guard choices × action choices — a finer-grained search space that
   admits different completion orders.

### 14.3  Future direction: stage-skeleton with hole-selection policies

The next step is to decompose each rule into **fillable slots** and study whether the order
in which slots are filled affects search efficiency.

**Stage-skeleton representation:**  A partially-specified policy where each stage has typed
holes:

```
Stage 0:  kind = PickRule,  key = ?,  guard = ?,  action = ?
Stage 1:  kind = ?,         key = ?,  guard = ?,  action = ?
...
Stage 2K: kind = GoalRule   (always fixed)
```

**Hole-selection policies** determine which hole to fill next:

| Policy | Strategy | Rationale |
|---|---|---|
| **Leftmost-hole** | Fill the first unfilled slot in linearised order | Generic default; matches the original budget-grammar behaviour |
| **Stage-first** | Choose a stage, fill all its slots before moving on | Groups related decisions; each stage is locally coherent |
| **Impact-first** | Fill the slot whose completion resolves the most frozen test states | Greedily maximises information gain; concentrates search on high-leverage decisions |

**Search method:**  A shared best-first search algorithm across all policies, with
identical cost metrics — only the hole-selection policy changes.  This enables a controlled
comparison of completion orders.

**Semantic deduplication:**  Two partial skeletons may produce identical behaviour across
all frozen test states despite differing syntactically.  A **behavioural signature** (mapping
each test state to its concrete action, UNRESOLVED, or INVALID) can deduplicate the search
frontier, potentially large savings at higher D.

**Primary benchmark:** D=3 is the smallest non-trivial instance (6 relaxed policies, 3
solving).  D=2 has only 1 policy and serves as a smoke test only — it cannot distinguish
between completion orders.

**What this will answer:**  Whether the generic leftmost-hole order (inherited from the
budget grammar) should remain the default for the typed stage-skeleton grammar, or whether a
domain-informed order (stage-first or impact-first) materially improves search efficiency.

---

## 15  Quick Start: Experimenting with the Surface DSL

### 15.1  Minimal Python snippet

The smallest self-contained experiment: build a stage program for D=2, compile it to an
AST, run it on the Doors environment, and check if it solves.

```python
from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state,
)
from alphazeropp.instances.doors.dsl.stage_dsl import Stage, StageProgram
from alphazeropp.instances.doors.dsl.surface_dsl import (
    PickReady, NeedKey, Pick, MoveToKey, MoveToGoal,
)
from alphazeropp.instances.doors.dsl.stage_compiler import compile_stage_program
from alphazeropp.synthesis.interpreter import run_policy_episode

# 1. Set up a D=2 Doors instance
cfg = DoorsGameConfig(num_rooms=2, locs_per_room=2)

# 2. Build a stage program (the canonical D=2 solver)
prog = StageProgram(
    stages=(
        Stage(guard=PickReady(0), action=Pick(0)),
        Stage(guard=NeedKey(0),   action=MoveToKey(0)),
    ),
    default_action=MoveToGoal(),
)

# 3. Compile to a raw AST
ast = compile_stage_program(prog, cfg)
print(ast.pretty())   # shows the nested Ite / Default tree

# 4. Run on the environment
x0  = doors_initial_state(cfg)
env = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
result = run_policy_episode(env, ast, x0=x0, is_solved=cfg.is_solved)

print(f"Solved: {result.solved}")
print(f"Steps:  {result.total_env_steps}")
print(f"Reward: {result.total_reward:.2f}")
# Expected: Solved: True, Steps: 3, Reward: 1.07
```

### 15.2  Running the AlphaZero training loop

The main training script uses MCTS + a neural network to *discover* solving orderings:

```bash
# Quick D=2 run (trivial — solves on iteration 1)
python scripts/run_doors_stage_derivation.py --d 2 --non-interactive

# D=3 with a specific seed
python scripts/run_doors_stage_derivation.py --d 3 --seeds 42 --non-interactive

# D=8 (the transition zone — still solves, takes a few iterations)
python scripts/run_doors_stage_derivation.py --d 8 --non-interactive
```

Output goes to `experiments/doors_stage_derivation/{timestamp}_stage_D{D}_K{K}_*/`.
Key files in the output directory:
- `config.json` — full configuration snapshot
- `training_metrics.png` — solve rate, reward, policy diversity over iterations
- `program_log.jsonl` — best program found at each iteration

### 15.3  Running tests as a sanity check

```bash
# Stage DSL: compiler, grammar, cost model
pytest tests/test_stage_dsl.py -v

# Stage derivation game: action space, legal masks, episodes
pytest tests/test_stage_derivation_game.py -v

# Reactive sketch: permutation sweep, equivalence classes
pytest tests/test_reactive_sketch.py -v
```

### 15.4  Key configuration knobs

The most important parameters in `stage_derivation_config.py`
(`DoorsStageDerivationConfig`):

| Parameter | Default | Effect |
|---|---|---|
| `num_rooms` | 3 | D — number of rooms.  K = D−1 keys, 2K+1 actions. |
| `max_stages` | `None` (= 2K+1) | Max stages in the decision list. |
| `max_guard_depth` | 0 | 0 = atom guards only; 1 = allows `Not`/`And` combinators. |
| `hole_policy` | `"leftmost"` | Hole-filling order: `"leftmost"` or `"structure_first"`. |
| `n_simulations` | 80 | MCTS simulations per move. Higher = stronger but slower. |
| `n_games_per_train` | 30 | Self-play games per training iteration. |
| `n_iterations` | 30 | Number of train→play→evaluate cycles. |
| `n_procs` | 8 | Parallel self-play workers. |

---

## 16  File Reference

| File | Role |
|---|---|
| `src/.../dsl/surface_dsl.py` | Typed DSL atoms (conditions, actions, rules, policies) |
| `src/.../dsl/surface_compiler.py` | Config-driven compilation: surface → raw AST |
| `src/.../dsl/surface_grammar.py` | Canonical/relaxed enumeration, counting, prefix BFS |
| `src/.../dsl/surface_derivation_game.py` | Game class for MCTS integration |
| `src/.../dsl/surface_derivation_config.py` | MetaConfig builder (wires game + network + training) |
| `scripts/run_doors_surface_derivation.py` | Standalone training script |
| `scripts/enumerate_surface_dsl.py` | Enumeration and search-space comparison report |
| `tests/test_surface_dsl.py` | Compiler correctness tests |
| `tests/test_surface_derivation_game.py` | Game logic, legal mask, episode tests |
| `tests/test_surface_vs_direct_smoke.py` | Agent integration smoke tests |
| `scripts/run_surface_mini_sweep.py` | Mini sweep across D values (Section 11) |
| `scripts/run_surface_vs_direct.py` | Surface vs direct comparison harness |
| `src/.../dsl/reactive_sketch_dsl.py` | Reactive sketch DSL: lifted BT representation (Section 12) |
| `src/.../dsl/reactive_sketch_interpreter.py` | Tick-based interpreter for reactive sketch |
