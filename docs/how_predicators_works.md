# How predicators works

A conceptual guide for someone trying to understand what this repo is
actually doing and what the surrounding research is about. Read this
before the papers — the papers assume you already have this picture.

---

## 1. The problem in one paragraph

A robot lives in a **continuous world**: object poses are floats, joints
have real-valued angles, gravity is 9.81 m/s². But the things we want
robots to do are naturally **discrete**: "stack block A on block B",
"make coffee", "press the button". Classical AI planning (PDDL, A*,
STRIPS) solves discrete problems beautifully but can't represent "the
gripper is 2 cm to the left." End-to-end reinforcement learning handles
continuous everything but can't plan over long horizons. predicators
sits in the middle and tries to **learn the bridge** between the two.

## 2. Bilevel planning — the GPS analogy

When Google Maps routes you from San Francisco to Los Angeles, it does
not simulate driving at 60 mph for six hours. It does two things in
parallel:

- **High-level**: search a graph of road segments (a discrete problem)
  for the shortest path.
- **Low-level**: a separate system handles steering, accelerating,
  braking, and lane changes (a continuous problem).

These two systems talk through a **shared vocabulary**: "take the I-5
ramp at exit 23". The high-level planner doesn't care about steering
angles; the low-level controller doesn't care about what city it's
heading to. That shared vocabulary is *exactly* what predicators is
about for robots.

In predicators terms:

- **High-level planner**: search over `STRIPSOperator`s ([Wikipedia:
  STRIPS](https://en.wikipedia.org/wiki/Stanford_Research_Institute_Problem_Solver))
  using [A*](https://en.wikipedia.org/wiki/A*_search_algorithm). Lives
  in `predicators/planning.py`.
- **Low-level controller**: continuous skills (option policies, IK,
  joint control). Lives in `predicators/ground_truth_models/<env>/options.py`
  and `predicators/pybullet_helpers/`.
- **Shared vocabulary**: **predicates** (`On`, `Holding`, `Clear`) and
  the **operators** that act on them (`Pick`, `Place`).

The research bet is: if you can *learn* that shared vocabulary instead
of hand-engineering it, bilevel planning becomes practical for new
domains without a roboticist writing PDDL by hand.

## 3. The vocabulary you actually need

Every concept lives in `predicators/structs.py` (a single 122 KB file).
Read this section before the code.

### `Type` and `Object`

A `Type` is a class of entity in the world — `block`, `robot`, `target`
— along with a fixed list of float features (`pose_x, pose_y, pose_z,
color_r, color_g, color_b, held`). An `Object` is a typed name like
`block0:block`. The state of one object is just its feature vector.

**Analogy**: types are like classes in OOP; objects are instances. The
features are the fields. The state is the heap snapshot.

### `State`

A dict from `Object` to a feature vector — the **full continuous world**.
This is what physics gives you. In a 5-block scene, the state might be
~50 floats.

### `Predicate`

A named, typed function that asks a yes/no question about a state:

- name: `"On"`
- types: `(block, block)`
- classifier: `(state, [b1, b2]) -> bool`

The classifier is just Python. For `On(?b1, ?b2)` it might check
`|b1.x - b2.x| < ε ∧ |b1.y - b2.y| < ε ∧ b1.z ≈ b2.z + height`.

A `GroundAtom` is a predicate *applied to specific objects*: `On(A, B)`.
In any given `State`, every ground atom is either true or false. The
**set of true ground atoms** is the state's **symbolic abstraction** —
the bridge between the continuous and discrete worlds.

**Analogy**: a predicate is a SQL query. A ground atom is the query
*applied* to specific row identifiers. The symbolic abstraction is the
full result set of "all queries that return true."

### `ParameterizedOption` (a "skill")

A closed-loop policy that runs for many timesteps and takes a
**continuous parameter**:

- `policy(state, params) -> action`
- `initiable(state, params) -> bool` — can we start from here?
- `terminal(state, params) -> bool` — are we done?
- `params_space` — e.g., a 2D placement target

Examples: `Pick(block, params)`, `Place(target, params)`.

**Analogy**: an option is a function with two argument levels — *what to
do* (the option name + objects) is symbolic; *exactly how to do it* (the
params) is continuous.

### `STRIPSOperator`

The classical-planning view of a skill — purely symbolic. Has lifted
parameters, **preconditions** (`LiftedAtom`s that must be true to apply
it), and **add/delete effects** (atoms it makes true/false):

```
Pick(?b: block, ?r: robot):
  pre:   {HandEmpty(?r), IsBlock(?b)}
  add:   {Holding(?r, ?b)}
  del:   {HandEmpty(?r)}
```

This is what the high-level planner sees. It doesn't know or care that
"Pick" involves a 7-DoF arm; it only knows about the symbolic transition.

### `NSRT` — the centerpiece

A **Neuro-Symbolic Relational Transition model** ties everything together:

1. A `STRIPSOperator` — the symbolic side (preconditions, effects).
2. A `ParameterizedOption` — the skill that realizes it.
3. A **sampler** — `(state, goal, objects) -> continuous parameters`.

So an NSRT says: *"to make `On(A, B)` true symbolically, run the `Place`
skill on `(A, B)`, and to pick good placement parameters, sample from
this learned distribution."*

**Analogy**: an NSRT is a recipe card.
- The STRIPS operator is the *prerequisites and effects* — "you need
  cold butter; you'll get rough dough."
- The option is the *technique* — "cream the butter and sugar."
- The sampler is the *intern's intuition* — "this oven runs hot, set
  it to 365 not 375."

The system as a whole is the cookbook (set of NSRTs) plus the chef
(planner) who decides the order of recipes for a given dinner (goal).

## 4. How planning actually runs — SeSamE

`predicators/planning.py` implements **SeSamE**: **Se**arch, **Sam**ple,
then **E**xecute.

1. **Abstract the current state.** Apply every predicate's classifier
   to the `State` to get the set of true ground atoms.
2. **Search** with A* over `STRIPSOperator`s from current atoms to atoms
   that satisfy the goal. Output: a **skeleton** — a sequence of ground
   NSRTs like `[Pick(A), Place(A, B)]`.
3. **Sample continuous parameters** for each step using the NSRTs'
   samplers. Forward-simulate via `env.simulate` to check feasibility.
   If a sample is infeasible (collision, unreachable), re-sample. If
   repeatedly infeasible, **backtrack** to step 2 and try a different
   skeleton.
4. **Execute** the resulting plan. `predicators/cogman.py` drives it.

**Why backtracking matters**: symbolic plans can be "downward
unrefinable" — looks fine on paper but no continuous parameters work
(e.g., a stacked tower goal that's physically impossible in the current
arrangement). Bilevel planning's correctness argument is that
backtracking eventually finds *a* refinable skeleton if any exists,
without trying every possible continuous parameter for every possible
skeleton.

For background, the term-of-art is **Task and Motion Planning (TAMP)**.
[Garrett et al., "Integrated Task and Motion Planning" (Annual Review
2021)](https://arxiv.org/abs/2010.01083) is the survey.

## 5. What's "learned" vs. what's given — the answer-key subtlety

This is the part that confuses everyone. The env files in
`predicators/envs/*` **define predicates explicitly**. So what is being
"learned"?

The predicates serve **two roles simultaneously**:

### Role 1: The answer key

Every env declares its predicates so the framework can:

- Define what counts as a **goal**. Goals are sets of `GroundAtom`s
  over the env's predicates (e.g., `Covers(block0, target0)`).
- **Label demonstrations** automatically. Each state in a demo
  trajectory has its symbolic atom set computed by running the env's
  predicate classifiers — that's the supervision signal for learning
  operators.
- Support **oracle baselines**. The `oracle` approach reads these
  directly.

### Role 2: Things to be (re)discovered

Look at `base_env.py:90`:

```python
@property
def target_predicates(self) -> Set[Predicate]:
    """Get the subset of self.predicates that we want to invent."""
    return self.predicates
```

This is the framework's formal acknowledgment that some approaches will
**pretend** the predicates are unknown and try to invent functional
equivalents from raw object features.

The mechanism is the `--excluded_predicates` CLI flag. Examples:

```bash
# Oracle: hands the agent everything. Solves trivially.
python predicators/main.py --env cover --approach oracle --seed 0

# Exclude one predicate: agent must operate without "Holding".
python predicators/main.py --env cover --approach grammar_search_invention \
    --seed 0 --excluded_predicates Holding

# Exclude everything except goal predicates: agent must invent the
# entire intermediate vocabulary.
python predicators/main.py --env cover --approach grammar_search_invention \
    --seed 0 --excluded_predicates all
```

**Analogy**: a teacher writes both the exam and the answer key. The exam
is the env's continuous states; the answer key is the env's predicates.
The student (agent) might be allowed to peek at the answer key (oracle)
or asked to figure it out from worked examples (predicate invention).
Either way, the answer key has to exist so we can grade them.

The *research contribution* of the predicate-invention papers
([AAAI 2023](https://arxiv.org/abs/2203.09634),
[VisualPredicator at ICLR 2025](https://arxiv.org/abs/2410.23156)) is
showing that the agent can **rediscover** something functionally
equivalent to the hand-coded predicates from data alone — and that the
resulting abstraction is good enough to plan with.

## 6. Map of approaches — who learns what

`predicators/approaches/` contains ~30 approach files. They differ by
**which parts of the NSRT they take as given vs. learn**.

| Approach | Predicates | Operators | Skills (options) | Samplers |
|---|---|---|---|---|
| `oracle` | given | given | given | given |
| `nsrt_learning` | given | **learn** | given | **learn** |
| `grammar_search_invention` | **invent** | **learn** | given | **learn** |
| `gnn_*` | given | replaced by a graph NN policy | n/a | n/a |
| `interactive_learning` | given | **learn online** with queries | given | **learn** |
| `active_sampler_learning` | given | given | given | **learn online** |
| `agent_*` (claude-agent-sdk) | varies | varies | varies | varies — driven by an LLM agent |

Three thematic clusters:

- **Symbol learning** (`nsrt_learning`, `grammar_search_invention`): the
  core "learn the bridge" line. Most of the canonical papers.
- **Policy learning** (`gnn_*`, `maple_q`): skip the planner entirely
  and train a neural policy — the "RL baseline" cluster.
- **Online / interactive** (`interactive_learning`,
  `active_sampler_learning`): the agent acts in the env, asks questions,
  and improves over multiple rounds.

The `agent_*` family is newer and uses LLM agents (via the
[claude-agent-sdk](https://docs.claude.com/en/api/agent-sdk/overview))
to drive learning — the LLM proposes predicates, operators, or samplers
and the env/planner provides feedback.

## 7. Reading the papers — suggested order

The README lists eight papers. Here's the order to read them in, with
what each one contributes. All links are arxiv.

1. **[Learning Symbolic Operators for TAMP](https://arxiv.org/abs/2103.00589)**
   (Silver*, Chitnis* et al., IROS 2021).
   Starting point. Defines the symbolic-operator-learning setup. Read
   for the *problem framing*, not for the specific algorithm — later
   papers refine it.

2. **[Learning Neuro-Symbolic Relational Transition Models for Bilevel
   Planning](https://arxiv.org/abs/2105.14074)** (Chitnis*, Silver* et
   al., IROS 2022). **The core NSRT paper.** Defines the central
   abstraction this codebase is named after. Read carefully.

3. **[Learning Neuro-Symbolic Skills for Bilevel
   Planning](https://arxiv.org/abs/2206.10680)** (Silver et al., CoRL
   2022). What if you don't have hand-coded options either? This paper
   learns them.

4. **[Predicate Invention for Bilevel
   Planning](https://arxiv.org/abs/2203.09634)** (Silver*, Chitnis* et
   al., AAAI 2023). **The predicate-invention paper.** Generates
   candidate predicates from a grammar and searches for the subset that
   yields a good abstraction. This is what
   `grammar_search_invention_approach.py` implements (73 KB; it's a lot).

5. **[Learning Efficient Abstract Planning Models that Choose What to
   Predict](https://arxiv.org/abs/2208.07737)** (Kumar*, McClinton* et
   al., CoRL 2023). Refines #4 — not every predicate is worth learning;
   pick the ones that matter for planning efficiency.

6. **[Embodied Active Learning of Relational State
   Abstractions](https://arxiv.org/abs/2303.04912)** (Li, Silver, CoLLAs
   2023). Online version: the robot acts in the env and queries a
   teacher to improve its abstraction over time.

7. **[VisualPredicator: Learning Abstract World Models with
   Neuro-Symbolic Predicates for Robot
   Planning](https://arxiv.org/abs/2410.23156)** (Liang et al., ICLR
   2025). What if state isn't feature vectors but images? Predicate
   classifiers become VLMs. This is the bridge to modern foundation-model
   work.

8. **[ExoPredicator: Learning Abstract Models of Dynamic
   Worlds](https://arxiv.org/abs/2509.26255)** (Liang et al., ICLR 2026).
   Extends #7 to dynamic worlds (other agents, exogenous events).

You can skip #1 if you just want the working algorithm; #2 is the
foundational paper. If your interest is predicate invention specifically,
read #2 → #4 → #5 → #7.

## 8. Pointers into the code

Once you've internalized §3–§4, this is the suggested code-reading order:

1. **`predicators/envs/cover.py`** — simplest concrete env. Read its
   `predicates` property, `simulate`, and option setup.
2. **`predicators/ground_truth_models/cover/nsrts.py`** — see how
   hand-coded NSRTs wire preconditions, effects, options, and samplers.
3. **`predicators/approaches/oracle_approach.py`** (or
   `bilevel_planning_approach.py`) — see how an approach assembles
   NSRTs into a working solver.
4. **`predicators/planning.py`** — the SeSamE loop. Skim first; come
   back when you want to understand backtracking.
5. **`predicators/nsrt_learning/nsrt_learning_main.py`** — the 5-stage
   learning pipeline.
6. **`predicators/approaches/grammar_search_invention_approach.py`** —
   the predicate-invention algorithm. Long file; come back to it after
   reading paper #4.

`predicators/structs.py` is the reference for every data type. Keep it
open while reading the rest.

## 9. Glossary

- **Abstraction** — In this codebase, the *symbolic* view of a continuous
  state: the set of true ground atoms.
- **Atom** — A predicate applied to specific objects. Lifted (uses
  variables) or ground (uses specific objects).
- **Bilevel planning** — Planning split into a discrete/symbolic high
  level and a continuous low level, with a refinement step that
  instantiates the low-level details for the high-level plan.
- **CFG** — The process-global mutable settings namespace
  (`predicators/settings.py`). Almost every module reads from it.
- **Demonstration** — A `(state, action)` trajectory, often labeled with
  options. Used as supervision for operator/sampler learning.
- **Downward refinable** — A symbolic plan that always has at least one
  valid continuous refinement. The Cover env is downward refinable;
  most realistic envs are not.
- **Goal** — A set of ground atoms (or a language description that gets
  parsed into one) the planner is trying to make true.
- **Grounded / lifted** — "Lifted" uses variables (`On(?b1, ?b2)`),
  "grounded" uses specific objects (`On(A, B)`). Operators are usually
  written lifted and then grounded against the current state's objects.
- **NSRT** — Neuro-Symbolic Relational Transition model. Triple of
  (STRIPSOperator, ParameterizedOption, sampler).
- **Option** — A skill with `policy / initiable / terminal / params_space`.
  From the [options framework](https://en.wikipedia.org/wiki/Hierarchical_reinforcement_learning)
  in hierarchical RL.
- **PDDL** — [Planning Domain Definition
  Language](https://en.wikipedia.org/wiki/Planning_Domain_Definition_Language).
  The classical-AI lingua franca for symbolic planning.
- **Predicate** — A typed, named yes/no function on states.
- **Sampler** — A learned distribution over continuous option
  parameters, conditioned on the symbolic state and goal.
- **SeSamE** — Search-and-Sample-then-Execute. The bilevel planning
  algorithm this codebase implements.
- **STRIPS** — [Stanford Research Institute Problem
  Solver](https://en.wikipedia.org/wiki/Stanford_Research_Institute_Problem_Solver).
  Origin of the precondition/add-effect/delete-effect operator format.
- **TAMP** — Task and Motion Planning. The broader research area.

## 10. Further reading

- [TAMP survey](https://arxiv.org/abs/2010.01083): Garrett, Chitnis et
  al., 2021. The "what is TAMP and why" reference.
- [Hierarchical RL options](https://www.sciencedirect.com/science/article/pii/S0004370299000521):
  Sutton, Precup, Singh, 1999. The original "options framework" paper.
- [PDDL primer](https://planning.wiki/): community-maintained reference
  for the planning-domain language and toolchain.
- [pybullet-planning](https://github.com/caelan/pybullet-planning):
  Caelan Garrett's library that
  `predicators/pybullet_helpers/` is structurally based on. Useful
  reference for the lower-level robotics machinery.
