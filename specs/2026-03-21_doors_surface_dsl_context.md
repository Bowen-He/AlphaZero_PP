# Doors Program Synthesis: Full System Context (Updated with Surface DSL)

**Date:** 2026-03-21
**Status:** Reference document — self-contained, no external context needed.

**Contents:**
1. The Problem
2. AlphaZero Primer
3. The Doors Environment
4. The Grammar (DSL)
5. The Four Derivation Modes
6. The Training Pipeline
7. Current Hyperparameters
8. What Has Been Tried and What Failed
9. The Optimal Programs (D=2 and D=3)
10. Known Structural Issues
11. The New Surface DSL: DoorsStageDSL(D)
12. Codebase Map (updated)
13. Goals

---

## 1. The Problem

We want to **synthesize a program** (a reactive if-then-else policy) that solves a navigation task. The program reads observations and outputs actions. It must generalize: one program must solve all starting states.

The system uses **AlphaZero** (MCTS + neural network) to search the space of programs by treating program construction as a game: each "move" expands the program's AST by one grammar production. The terminal reward is the quality of the completed program when run on the actual environment.

**D=2 (2 rooms) is reliably solved.** The factored+macro grammar finds a solver in iteration 1 (~8K programs). Even flat grammar solves D=2 by iteration 7-8.
**D=3 (3 rooms) is unreliable.** With factored+macros, D=3 solves in some runs but fails in others despite 200K+ programs explored. Flat grammar never solves D=3.

---

## 2. AlphaZero Primer

AlphaZero combines a **neural network** (policy head + value head) with **Monte Carlo Tree Search (MCTS)**. In our system the "game" is program construction:

| Board Game Concept | Program Synthesis Equivalent |
|---|---|
| Game state | Partial AST (some nodes filled, some holes remaining) |
| Legal move | Grammar production to fill the leftmost hole |
| Game over | AST has no remaining holes — a complete program |
| Outcome/reward | Quality of the completed program on the target environment |
| Policy head learns | "Given this partial AST, which production should I expand next?" |
| Value head learns | "Given this partial AST, how good will the final program be?" |

This is a **single-player** variant — no opponent. AlphaZero was designed for domains with: (a) intermediate states that have intrinsic value, (b) opponent-driven curriculum, (c) high reward variance, (d) moderate branching. Program synthesis violates all four.

---

## 3. The Doors Environment

### 3.1 Physical Layout (D=3 example)

```
Room 0 (unlocked)     Room 1 (locked)       Room 2 (locked)
  loc 0 (start)         loc 2                  loc 4
  loc 1 [Key 0]         loc 3 [Key 1]          loc 5 (GOAL)
```

- D rooms, each with `locs_per_room` locations (default 2).
- K = D-1 keys. Key k sits at a fixed location and unlocks room k+1.
- Agent starts at loc 0. Goal: reach the last location.

### 3.2 Observation Vector

Size: `n_sites = M + 2D - 1` where `M = D * locs_per_room`.

For D=3: M=6, n_sites=11.

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Meaning | at_loc0 | at_loc1 | at_loc2 | at_loc3 | at_loc4 | at_loc5 | room0_open | room1_open | room2_open | key0_avail | key1_avail |

Initial state: `[1,0,0,0,0,0, 1,0,0, 1,1]` — at loc 0, room 0 open, both keys available.

### 3.3 Actions

`n_actions = M + K + 1` (D=3: 9 actions).

| Index | 0-5 | 6 | 7 | 8 |
|---|---|---|---|---|
| Meaning | MOVE(loc 0-5) | PICK(key 0) | PICK(key 1) | NOOP |

**Preconditions:**
- MOVE(loc): target room must be unlocked.
- PICK(key k): agent must be at key k's location AND key k must be available.
- Failed preconditions: action becomes NOOP (step penalty still applies).

### 3.4 Rewards

```
Every step:            -0.01  (step_penalty)
PICK that unlocks:     +0.10  (unlock_bonus)
Reach goal:            +1.00
Truncation at horizon: episode ends (horizon = max(15, 5 * optimal_steps))
```

For D=3: optimal_reward = 1.0 + 2(0.1) - 5(0.01) = **1.15**.

### 3.5 Reward Landscape (D=3)

| Keys picked | Env reward | Weighted metric (α=0.7) |
|---|---|---|
| 0 (stuck/NOOP) | -0.25 | -0.075 |
| 1 key | -0.15 | -0.045 |
| 2 keys | -0.05 | -0.015 |
| Solved (2 keys + goal) | +1.15 | +1.045 |

Non-solving range spans only **0.060** units. The cliff to solving is **1.060** (18x jump). Programs picking 0, 1, or 2 keys are nearly indistinguishable by reward.

---

## 4. The Grammar (DSL)

Programs are decision-list ASTs built from a context-free grammar with budget constraints.

### 4.1 AST Nodes

| Node | Semantics | Cost |
|---|---|---|
| `Flip(j)` | Execute action j | 1 |
| `IsZero(j)` | Test obs[j]==0 (returns bool) | 1 |
| `Not(cond)` | Logical negation | 1 + child |
| `And(cond1, cond2)` | Conjunction | 1 + children |
| `Ite(cond, Flip(j), else_prog)` | If cond then action j else continue | 1 + cond + 1 + else |
| `Default(Flip(j))` | Always action j (base case) | 2 |

### 4.2 Production Rules

```
Program(k):
  k == 2:  Default(Flip(j))                         for j in [0, n_actions)
  k >= 5:  Ite(Cond(i), Flip(j), Program(k-2-i))    for i in [1, k-4], j in [0, n_actions)

Condition(k):
  k == 1:  IsZero(j)                                 for j in [0, n_sites)
  k >= 2:  Not(Cond(k-1))                            (banned if parent is Not)
  k >= 3:  And(Cond(i), Cond(k-1-i))                 for i in [1, floor((k-1)/2)]
```

**IsZero semantics:** `IsZero(j)` tests `obs[j]==0`. Since obs uses 1 for "true":
- `IsZero(1)` = "NOT at loc 1"
- `Not(IsZero(1))` = "at loc 1"
- `Not(IsZero(9))` = "key 0 is available"
- `IsZero(7)` = "room 1 is locked"

### 4.3 Budget and Program Space

Budget `L` controls AST size (~1.5x optimal node count). For D=3: optimal=22 nodes, budget L=34.

**Pruning constraints:**
1. Action range: `Flip(j)` only for j in [0, n_actions).
2. Double-negation ban: `Not(Not(...))` suppressed.
3. One-hot groups: Contradictory literals in same group pruned.
4. Dead-end pruning (exact mode): Skip productions leading to impossible budgets.
5. Condition budget cap (mode 3): Caps condition size to ≤12 nodes.

---

## 5. The Four Derivation Modes

### 5.1 Mode 0: `doors` — Flat, And Enabled
Each production is a single atomic action. ~279 actions per step, ~15 steps per game.

### 5.2 Mode 1: `doors_no_and` — Flat, And Disabled
And removed. ~200 actions. Ablation baseline.

### 5.3 Mode 2: `doors_factored` — Factored, And Enabled
Productions split into structure phase + parameter phase. ~31 actions per step. Same program space as Mode 0.

### 5.4 Mode 3: `doors_d10_macro` — Factored + Macros + Condition Cap

Adds domain-specific macro productions:

**PickRule(k)** (7 AST nodes, 1 derivation step):
```
Ite(And(Not(IsZero(key_loc[k])), Not(IsZero(key_avail_idx[k]))),
    Flip(PICK_action[k]),
    ProgramHole(budget - 7))
```

**MoveRule(k)** (3 AST nodes, 1 derivation step):
```
Ite(IsZero(room_unlock_idx[k]),
    Flip(MOVE_to_key_loc[k]),
    ProgramHole(budget - 3))
```

~39 actions per step, ~10 steps for D=3. Condition cap = 12.

---

## 6. The Training Pipeline

### 6.1 AlphaZero Loop

```
for iteration in 1..n_iterations:
    1. SELF-PLAY: n_games derivation games using MCTS + network
       - Collect (partial_ast_obs, mcts_policy, terminal_reward)
    2. TRAIN: Update network on recent examples
       - Loss = MSE(value, reward) + 2.0 * CrossEntropy(policy, π)
    3. EVALUATE: Pit new vs old network
       - Accept if win_rate >= 0.40
```

### 6.2 Observation Encoding

Partial AST → fixed-size vector of `2 * budget` floats via preorder traversal:
```
obs = [type_id_0, param_0, type_id_1, param_1, ..., 0, 0, ...]
```
Node type IDs: PAD=0, Flip=1, IsZero=2, Not=3, And=4, Ite=5, Default=6, ProgramHole=7, ConditionHole=8.

### 6.3 Network Architecture

Transformer encoder: d_model=64, n_heads=4, n_layers=2. Input: (type_id, param) pairs → embed → CLS token → policy head + value head.

### 6.4 MCTS Details

UCB = Q_normalized + c_exploration × P(a) × √N_total / (1 + N(a)).

Q-normalization: (Q - Q_min)/(Q_max - Q_min). When uniform, defaults to 0.5 → pure exploration.

Defaults: n_sims=80, c_exploration=1.5, dirichlet_alpha=0.25, dirichlet_epsilon=0.40, rollout_n=4, rollout_mode="max", backup_rule="max".

### 6.5 Leaf Evaluation (Terminal Reward)

Complete program evaluated on frozen states:
```python
"weighted": alpha * solve_rate + (1-alpha) * avg_reward    # alpha=0.7
```
Programs cached by pretty-print string.

### 6.6 Reward Flow

Terminal step: `reward = leaf_eval(program)`. Non-terminal: 0.0. With discount=1.0, all steps get same value target.

---

## 7. Current Hyperparameters

| Component | Parameter | Value |
|---|---|---|
| **Problem** | D | 3 |
| | n_sites | 11 |
| | n_actions | 9 |
| | budget (L) | 34 |
| | horizon | 25 |
| **Grammar** | program_budget_mode | "max" |
| | allow_and / allow_not | True / True |
| | one_hot_groups | [[0,1,2,3,4,5]] |
| **Network** | d_model / n_heads / n_layers | 64 / 4 / 2 |
| | lr / batch_size / epochs | 3e-4 / 32 / 5 |
| **MCTS** | n_simulations | 80 |
| | c_exploration | 1.5 |
| | rollout_n / rollout_mode | 4 / "max" |
| | backup_rule | "max" |
| **Training** | n_games_per_train | 30 |
| | n_iterations | 30 |

---

## 8. What Has Been Tried and What Failed

### 8.1 D=2: Consistently Solved

| Mode | Solved at | Programs explored |
|---|---|---|
| Flat (20 games/iter) | iter 7 | 4,487 |
| Factored+macro | **iter 1** | 8,460 |

### 8.2 D=3: Unreliable — The Critical Gap

**Flat grammar: NEVER solves D=3** (0/13 runs, up to 125K programs explored, best reward -0.15).

**Factored+macros: Solves ~60-70% of runs.** Success is random, not learned — most solved runs find it at iteration 1. More compute doesn't guarantee success (688K programs failed; 85K succeeded). Training loop contributes nothing.

### 8.3 What Changes Between D=2 and D=3

| Dimension | D=2 | D=3 |
|---|---|---|
| Optimal program nodes | 12 | 22 |
| Budget (L) | 18 | 34 |
| Flat branching | ~90 | ~279 |
| Conjunctive conditions | 1 | 2 |
| Solver frequency (factored+macros) | ~1 in 5K | ~1 in 100K-200K |

### 8.4 The Core Failure: Reward Desert

91-99% of random programs score identically (-0.075). Consequences:
1. **Value head collapses** to constant predictor (~-0.075).
2. **MCTS degenerates** to pure exploration (Q_min == Q_max).
3. **Gate is useless** — new and old networks identical.
4. **Vicious cycle** of random search with MCTS overhead.

### 8.5 Direct Play Comparison

Direct RL (no synthesis layer) solves D=10 in 5 iterations. The synthesis layer collapses the rich per-step environment reward into a single sparse terminal signal.

---

## 9. The Optimal Programs

### D=2 (12 nodes)
```
if And(Not(IsZero(1)), Not(IsZero(6))):   # at loc 1 AND key 0 available
    Flip(4)                                #   → PICK key 0
elif IsZero(5):                            # room 1 locked
    Flip(1)                                #   → MOVE to loc 1
else:
    Flip(3)                                #   → MOVE to goal (loc 3)
```

### D=3 (22 nodes)
```
if And(Not(IsZero(1)), Not(IsZero(9))):      # at loc 1 AND key 0 available
    Flip(6)                                    #   → PICK key 0
elif IsZero(7):                                # room 1 locked
    Flip(1)                                    #   → MOVE to loc 1
elif And(Not(IsZero(3)), Not(IsZero(10))):    # at loc 3 AND key 1 available
    Flip(7)                                    #   → PICK key 1
elif IsZero(8):                                # room 2 locked
    Flip(3)                                    #   → MOVE to loc 3
else:
    Default(Flip(5))                           #   → MOVE to goal (loc 5)
```

**Structure:** Alternating PickRule/MoveRule pairs for each key, then default move-to-goal.

---

## 10. Known Structural Issues

1. **Synthesis layer destroys environment reward structure.** Direct RL on Doors D=10 solves in 5 iterations. The derivation game collapses rich per-step rewards into one terminal scalar.

2. **Value network has an impossible task.** It must predict terminal quality from partial AST alone. At early steps (e.g., `Ite(CondHole(5), Flip(6), ProgramHole(15))`), future reward depends entirely on unfilled holes.

3. **MCTS explores within one game, not across games.** Each tree search completes ~5 programs. With 30 games/iter, that's ~150 highly correlated programs per iteration.

4. **Policy network learns noise.** When Q-values are uniform, visit counts are driven by Dirichlet noise. Training targets become noise.

5. **No curriculum.** The problem difficulty is fixed from the start. No opponent provides graduated challenge.

---

## 11. The New Surface DSL: DoorsStageDSL(D)

### 11.1 Motivation

The generic grammar treats all observation indices and actions as opaque integers. For D=2 at optimal budget 12, this yields **5,088,405 canonical programs**. The overwhelming majority are semantically meaningless (e.g., testing "at loc 2 AND at loc 4" — impossible in the one-hot encoding).

The surface DSL encodes domain knowledge as typed semantic constructs. It does not modify the generic grammar — it is a separate, parallel path.

### 11.2 Surface Language Definition

For D rooms, K = D-1 keys:

**Condition atoms** (no raw observation indices):
| Construct | Meaning | Compiles to |
|---|---|---|
| `AtKeyLoc(k)` | Agent at key k's location | `Not(IsZero(cfg.key_loc[k]))` |
| `KeyAvail(k)` | Key k available | `Not(IsZero(cfg.M + cfg.D + k))` |
| `RoomLocked(r)` | Room r is locked | `IsZero(cfg.M + r)` |
| `PickReady(k)` | AtKeyLoc(k) ∧ KeyAvail(k) | `And(AtKeyLoc(k), KeyAvail(k))` |
| `NeedKey(k)` | Room key k unlocks is locked | `IsZero(cfg.M + cfg.key_unlocks[k])` |

**Action atoms** (no raw action indices):
| Construct | Meaning | Compiles to |
|---|---|---|
| `Pick(k)` | Pick up key k | `Flip(cfg.M + k)` |
| `MoveToKey(k)` | Move to key k's location | `Flip(cfg.key_loc[k])` |
| `MoveToGoal` | Move to goal | `Flip(cfg.goal_loc)` |

**Rule macros:**
| Construct | Meaning | AST nodes |
|---|---|---|
| `PickRule(k)` | if PickReady(k) then Pick(k) else continue | 7 |
| `MoveRule(k)` | if NeedKey(k) then MoveToKey(k) else continue | 3 |
| `GoalRule` | default MoveToGoal | 2 |

**Policy:** An ordered tuple of rules ending with GoalRule, e.g.:
```
PickRule(0) >> MoveRule(0) >> PickRule(1) >> MoveRule(1) >> GoalRule
```

### 11.3 Compiler

The compiler (`surface_compiler.py`) takes a `SurfacePolicy` + `DoorsGameConfig` and produces a raw `Program` AST using existing node types. It chains rules right-to-left: GoalRule becomes the innermost Default, each preceding rule wraps it as an Ite. **All index resolution is config-driven — no hardcoded constants.**

The compiled output is directly executable by the existing interpreter (eval_program, run_policy_episode).

### 11.4 Two Grammar Artifacts

**Canonical policy** (unique per D): The known-optimal ordering.
```python
canonical_policy(D) = PickRule(0) >> MoveRule(0) >> ... >> PickRule(K-1) >> MoveRule(K-1) >> GoalRule
```

**Relaxed search grammar:** All permutations of {PickRule(k), MoveRule(k) : k∈0..K-1} satisfying PickRule(k) before MoveRule(k) for each k, with GoalRule at the end.

Count: **(2K)! / 2^K** policies.

| D | K | Canonical | Relaxed | Generic canonical (optimal budget) |
|---|---|---|---|---|
| 2 | 1 | 1 | 1 | 5,088,405 |
| 3 | 2 | 1 | 6 | 6.3 × 10^18 |
| 4 | 3 | 1 | 90 | 1.1 × 10^26 |

**Not all relaxed policies solve.** Interleavings that place MoveRule for a later key before earlier keys are resolved may attempt moves through locked rooms.

### 11.5 D=2 Concrete Example

```python
# Surface DSL
policy = SurfacePolicy((PickRule(0), MoveRule(0), GoalRule()))
# pretty: "PickRule(0) >> MoveRule(0) >> GoalRule"

# Compile with D=2 config (M=4, D=2, K=1, key_loc=[1], key_unlocks=[1], goal_loc=3)
prog = compile_policy(policy, cfg)

# Compiled raw AST (12 nodes):
# if And(Not(IsZero(1)), Not(IsZero(6))):   ← PickReady(0)
#   Flip(4)                                  ← Pick(0)
# elif IsZero(5):                            ← NeedKey(0)
#   Flip(1)                                  ← MoveToKey(0)
# else:
#   Flip(3)                                  ← MoveToGoal

# Run with existing interpreter:
result = run_policy_episode(env, prog, x0=x0, is_solved=cfg.is_solved)
# solved=True, steps=3, actions=[1, 4, 3] (move to key, pick, move to goal)
```

### 11.6 What the Surface DSL Does NOT Do (Current Limitation)

The surface DSL is a **standalone compile-and-enumerate** layer. It does **not** integrate with the existing `DerivationGame` or MCTS. The architecture gap:

```
EXISTING PATH (unchanged):
  DerivationGame → budget grammar → ProgramHole expansion → raw AST → interpreter
  (MCTS chooses which grammar production to apply at each step)

NEW PATH (surface DSL):
  surface_grammar → enumerate SurfacePolicy instances → compile each → raw AST → interpreter
  (brute-force enumeration, no MCTS, no derivation game)
```

To use the surface DSL as a derivation game, a new game class is needed where:
- **State** = partial policy (growing list of rules, not yet ended with GoalRule)
- **Actions** = which rule to append: PickRule(k), MoveRule(k), or GoalRule
- **Legal mask** = enforce monotonicity (PickRule(k) before MoveRule(k))
- **Terminal** = when GoalRule is placed
- **Reward** = compile the completed SurfacePolicy, run on frozen states

This would be a much simpler game: action space 2K+1 (vs. hundreds of productions), episode length 2K+1 steps (vs. ~budget steps).

---

## 12. Codebase Map (Updated)

### 12.1 Key Directory Tree

```
AlphaZero_PP/
├── src/alphazeropp/
│   ├── core/                              # Generic AlphaZero (MCTS, Agent, Game interface)
│   │   ├── game.py                        # Abstract Game interface
│   │   ├── mcts.py                        # MCTS with UCB, backup rules, rollouts
│   │   └── agent.py                       # Agent: plays games using MCTS + network
│   │
│   ├── synthesis/                         # Domain-agnostic program synthesis
│   │   ├── ast_nodes.py                   # Flip, IsZero, Not, And, Ite, Default
│   │   ├── budget_grammar.py              # CFG with budget constraints; enumeration & counting
│   │   ├── derivation.py                  # DerivationState, Production, leftmost-hole expansion
│   │   ├── derivation_game.py             # DerivationGame: wraps derivation as core.Game (flat)
│   │   ├── factored_derivation_game.py    # FactoredDerivationGame: structure/parameter phases
│   │   ├── interpreter.py                 # eval_program, run_policy_episode, format_trace
│   │   └── leaf_evaluator.py              # Evaluate complete programs on environment; caching
│   │
│   ├── training/                          # Trainer, Evaluator, GatedTrainer
│   │
│   └── instances/doors/                   # Doors domain
│       ├── doors_pddl_lite.py             # Gymnasium environment: step(), reset(), rewards
│       ├── game.py                        # DoorsDirectGame (direct RL, no synthesis)
│       ├── oracle.py                      # Hand-coded optimal policy
│       └── dsl/                           # Doors DSL for program synthesis
│           ├── doors_config.py            # DoorsGameConfig, doors_initial_state()
│           ├── derivation_config.py       # Four derivation config classes (modes 0-3)
│           ├── doors_macros.py            # PickRule/MoveRule macro productions (budget grammar)
│           ├── surface_dsl.py             # ★ NEW: Typed surface DSL dataclasses
│           ├── surface_compiler.py        # ★ NEW: Compiler: surface DSL → raw AST
│           ├── surface_grammar.py         # ★ NEW: Canonical + relaxed enumeration
│           └── __init__.py                # Re-exports (updated)
│
├── scripts/
│   ├── run_doors_derivation.py            # PRIMARY: Doors synthesis training
│   ├── run_doors_direct.py                # Direct RL on Doors (no synthesis)
│   ├── enumerate_dsl.py                   # Enumerate generic grammar programs
│   └── enumerate_surface_dsl.py           # ★ NEW: Surface DSL enumeration + comparison
│
├── tests/
│   ├── test_dsl.py                        # AST/interpreter tests
│   ├── test_cfg_grammar.py                # Budget grammar counting tests
│   ├── test_doors_macros.py               # Macro production tests
│   ├── test_doors_pddl_lite.py            # Environment tests
│   ├── test_derivation_game.py            # Flat derivation game tests
│   ├── test_factored_derivation_game.py   # Factored game tests
│   └── test_surface_dsl.py               # ★ NEW: Surface DSL tests (38 tests, all passing)
│
├── specs/                                 # This document and prior context docs
└── spec/
    └── report-doors-stage-dsl.md          # ★ NEW: Surface DSL enumeration report
```

### 12.2 Architecture Layers

```
┌──────────────────────────────────────────────────┐
│ Scripts (entry points)                            │
└────────────┬─────────────────────────────────────┘
             │
┌────────────┴─────────────────────────────────────┐
│ Domain: instances/doors/                          │
│   Environment (doors_pddl_lite.py)                │
│   Direct play (game.py)                           │
│   DSL: config, macros, derivation configs         │
│   ★ Surface DSL: dsl, compiler, grammar           │
└────────────┬─────────────────────────────────────┘
             │
┌────────────┴─────────────────────────────────────┐
│ Synthesis engine (synthesis/)                      │
│   AST nodes, budget grammar, derivation           │
│   DerivationGame, FactoredDerivationGame          │
│   Interpreter, LeafEvaluator                      │
└────────────┬─────────────────────────────────────┘
             │
┌────────────┴─────────────────────────────────────┐
│ Core AlphaZero (core/)                             │
│   MCTS, Agent, Game interface                      │
└──────────────────────────────────────────────────┘
```

---

## 13. Goals

### Goal 1: Play the derivation game with the surface DSL grammar

Create a `SurfaceDerivationGame` that wraps the surface DSL as a Gymnasium-compatible game playable by MCTS/AlphaZero:

- **State:** A partial policy being constructed (a growing list of surface rules).
- **Actions:** Discrete choice of which rule to append next — `PickRule(0)`, ..., `PickRule(K-1)`, `MoveRule(0)`, ..., `MoveRule(K-1)`, `GoalRule`. Action space size: `2K + 1`.
- **Legal action mask:** Enforce that `PickRule(k)` appears before `MoveRule(k)` for each k. GoalRule is only legal when at least one key has been fully handled (or as a terminal action at any point).
- **Termination:** When `GoalRule` is placed.
- **Reward:** Compile the completed `SurfacePolicy` via `compile_policy`, evaluate the resulting raw AST on frozen states using the existing `LeafEvaluator` or `run_policy_episode`, return scalar reward.
- **Observation encoding:** Encode the partial rule sequence as a fixed-size vector compatible with the existing network architecture.
- **Episode length:** At most `2K + 1` steps (vs. ~budget steps in the generic grammar).
- **Expected properties for D=2:** Trivial — only 1 valid trajectory. Should solve immediately.
- **Expected properties for D=3:** 6 valid complete policies across 5 steps. MCTS should find the solver rapidly.
- **Expected properties for D≥5:** (2K)!/2^K trajectories. This is where MCTS search becomes valuable.

The game must implement the same `Game` interface as `DerivationGame` so it plugs into the existing AlphaZero training loop (`Trainer`, `Agent`, `MCTS`).

### Goal 2: Compare surface DSL derivation game vs. direct play on Doors

Run a controlled comparison:

| Experiment | Algorithm | What it searches |
|---|---|---|
| **A. Surface derivation game** | AlphaZero (MCTS + network) | Space of `SurfacePolicy` sequences |
| **B. Direct play on Doors** | AlphaZero (MCTS + network) | Raw action space of Doors environment |

For each, measure:
- Iterations to solve (D=2, D=3, D=5, D=10)
- Total programs / episodes explored
- Whether the learning loop contributes (or if success is random)
- Value head quality (does it learn a useful predictor?)

**Hypothesis:** The surface derivation game should solve D=3 reliably (only 6 candidate policies, 5-step episodes) and may scale better than both generic-grammar synthesis and direct play for moderate D, because:
- The search space is many orders of magnitude smaller.
- Episode length is O(K) instead of O(budget).
- Every terminal state is a semantically meaningful policy (no junk programs).
- The reward signal should be less compressed (fewer meaningless programs diluting the landscape).

**Counter-hypothesis:** For large D, the surface DSL's relaxed grammar may be too restrictive (not all solving policies are representable) or too loose (non-solving interleavings dominate). Direct play scales because it has rich per-step rewards. The comparison should reveal where each approach breaks.
