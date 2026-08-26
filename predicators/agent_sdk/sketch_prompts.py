"""Prompt construction for bilevel plan-sketch solving/exploration.

Split out of ``bilevel_sketch`` (see that module's docstring for the
full layout); holds ``build_solve_prompt``, the solve/explore prompt
asking the agent for a plan sketch in the grammar that
``sketch_parsing`` reads back.
"""
from typing import Optional, Sequence, Set

from predicators import utils
from predicators.settings import CFG
from predicators.structs import ParameterizedOption, Predicate, Task


def build_solve_prompt(
    task: Task,
    *,
    all_predicates: Set[Predicate],
    all_options: Set[ParameterizedOption],
    trajectory_summary: str = "",
    tool_names: Optional[Sequence[str]] = None,
    experiment_guidance: str = "",
    scheduled_plans: Optional[Sequence[str]] = None,
    initial_image_section: str = "",
    propose_params: bool = False,
    require_tool_validation: bool = False,
    explore_mode: bool = False,
    ground_samplers: bool = False,
    journal: str = "",
    strategy: str = "",
    physics_margin: bool = False,
    policy_mode: bool = False,
) -> str:
    """Build the bilevel solve/explore prompt asking for a plan sketch.

    ``ground_samplers`` is the caller-threaded value of
    ``RefinementConfig.ground_samplers``; when False the prompt never
    mentions the ``~`` annotation channel.

    Mirrors ``AgentModelBasedApproach._build_solve_prompt`` but takes
    dependencies explicitly so explorers can reuse it.

    ``scheduled_plans`` lists sketch-line descriptions of exploration
    plans already generated this online-learning cycle (all of a cycle's
    requests are generated before any executes). When given, the prompt
    asks for a plan that still achieves the goal but differs meaningfully,
    so the cycle's interaction data is complementary instead of the same
    plan repeated per request.

    ``propose_params`` switches the prompt from "param-free sketch, search
    finds all continuous params" to "propose your best continuous params in
    ``[...]`` per step; the search refines them and samples on failure".

    ``require_tool_validation`` tells the agent it MUST submit a
    goal-reaching ``evaluate_option_plan`` run on the current task (the
    captured, validated plan is the only output) - used when the approach
    has no refinement fallback. When False, validation is merely
    encouraged.

    ``explore_mode`` marks the query as an exploration request whose
    sketch will run in the REAL environment as an experiment. It adds a
    belief-model disclosure (the simulator is base physics plus dynamics
    learned so far, so an unlearned mechanism shows zero effect) and an
    explicit delivery contract: a simulator-failing sketch is a valid
    deliverable when the goal depends on a mechanism the belief model
    lacks. Without it, agents have burned entire sessions exhaustively
    proving such a mechanism's absence instead of submitting the
    experiment that would let it be learned.

    ``journal`` is the run's solve-journal content (see
    ``predicators/agent_sdk/journal.py``): the curated record of earlier
    attempts' outcomes and lessons, injected so fresh-context sessions
    inherit what worked (and what was already swept) without inheriting
    failed attempts' conclusions.

    ``strategy`` is the learn-phase-maintained domain strategy document
    (``strategy.md``): the learn agent's best current natural-language
    account of how to solve tasks in this domain, rewritten freely
    across learning cycles. Injected as explicitly-advisory reference -
    the knowledge can be wrong, so the prompt tells the solver to
    re-verify rather than inherit.

    ``policy_mode`` (``CFG.agent_solve_policy_mode``): the deliverable
    becomes a closed-loop ./policy.py validated via ``evaluate_policy``
    instead of a fixed evaluate_option_plan capture; the submit and
    closing guidance swap to the policy contract.

    ``physics_margin`` is the caller-threaded value of
    ``CFG.agent_plan_validation_physics_margin``: when True (and
    ``require_tool_validation``), the submit guidance tells the agent
    the capture gate also sweeps the identified physical parameters'
    uncertainty range, and to pre-check designs with
    ``sim.run(..., physics_sweep=True)`` instead of discovering
    PARAM-SENSITIVE rejections one submission at a time.
    """
    assert not (explore_mode and require_tool_validation), (
        "explore_mode accepts an uncaptured experiment sketch, which "
        "contradicts the hard capture gate of require_tool_validation")
    assert not policy_mode or require_tool_validation, (
        "policy_mode is a hard capture gate (evaluate_policy), so it "
        "requires require_tool_validation")

    init_state = task.init
    objects = list(init_state)

    obj_strs = []
    for obj in sorted(objects, key=lambda o: o.name):
        obj_strs.append(f"  {obj.name}: {obj.type.name}")

    # Only expose goal atoms whose predicate is in the agent's current
    # predicate set. Approaches that strip env predicates (e.g.
    # agent_sim_predicate_invention) rely on goal_nl to communicate the
    # goal; leaking unfiltered task.goal atoms would expose predicates the
    # agent is supposed to invent for itself.
    goal_strs = [
        str(a) for a in sorted(task.goal, key=str)
        if a.predicate in all_predicates
    ]

    option_strs = []
    for opt in sorted(all_options, key=lambda o: o.name):
        type_sig = ", ".join(t.name for t in opt.types)
        params_dim = opt.params_space.shape[0]
        if params_dim > 0:
            low = opt.params_space.low.tolist()
            high = opt.params_space.high.tolist()
            label = "params" if propose_params else "auto-searched params"
            if opt.params_description:
                desc = ", ".join(opt.params_description)
                param_info = (f"  [{label}: {desc}, "
                              f"range {low} to {high}]")
            else:
                kind = f"{params_dim}d"
                param_info = (f"  [{label}: {kind}, "
                              f"range {low} to {high}]")
        else:
            param_info = ""
        option_strs.append(f"  {opt.name}({type_sig}){param_info}")

    atoms = utils.abstract(init_state, all_predicates)
    atom_strs = [str(a) for a in sorted(atoms, key=str)]

    state_str = init_state.dict_str(indent=2)

    tools_str = ""
    if tool_names:
        tool_list = "\n".join(f"  - {t}" for t in tool_names)
        tools_str = f"\n## Available Tools\n{tool_list}\n"

    experiment_section = ""
    if experiment_guidance:
        experiment_section = (f"\n## Experiment Guidance\n"
                              f"{experiment_guidance}\n")

    # Everything explore-specific lives in this ONE section: loop
    # context, belief-model semantics, and experiment-design principles.
    # The deliverable contract stays in closing_block; the system prompt
    # never repeats any of it.
    setting_section = ""
    if explore_mode:
        early_stop_note = ""
        if (CFG.online_learning_early_stopping
                and not CFG.online_learning_early_stopping_by_test_solve_rate):
            attempts_clause = (
                "every episode of a cycle"
                if CFG.online_learning_early_stopping_require_all_attempts else
                "each train task's first episode of a cycle")
            early_stop_note = (
                " The loop concludes early once the exploration plans "
                f"solve training: {attempts_clause} must reach the goal "
                "for real, AND the plan must have validated in the belief "
                "model - a lucky real success from a plan the model could "
                "not certify does not count. Once the belief model can "
                "validate a goal-reaching plan, submitting it (even "
                "unchanged) is how the loop concludes.")
        elif CFG.online_learning_early_stopping_by_test_solve_rate:
            n_perfect = (
                CFG.online_learning_early_stopping_consecutive_perfect_tests)
            phases = ("the next test phase solves" if n_perfect <= 1 else
                      f"{n_perfect} consecutive test phases each solve")
            early_stop_note = (
                f" The loop concludes early once {phases} every test "
                "task. Test attempts plan with the belief model, so what "
                "ends the loop is the belief model becoming reliably "
                "correct - your episodes count toward that only through "
                "the model corrections their data enables, not through "
                "reaching the goal themselves.")
        setting_section = (
            "\n## Exploration Setting\n"
            "You are the explorer in an online learning loop: the plan "
            "you submit runs in the REAL environment as an experiment, "
            "and its episode data is what the next learning phase uses "
            "to correct the belief model." + early_stop_note + "\n\n"
            "The simulator behind your tools is that belief model: known "
            "base physics plus whatever additional dynamics have been "
            "learned from real interaction data so far. A mechanism that "
            "has not been learned yet is simply ABSENT from it - the "
            "simulator shows zero effect no matter how you arrange the "
            "probe, and early in learning this can include the very "
            "mechanism the goal depends on. Treat a null effect after a "
            "few well-aimed probes as \"not in the belief model yet\", "
            "not as evidence about the real environment, and do not "
            "spend the session exhaustively confirming the absence.\n\n"
            "Design the experiment accordingly. A goal-reaching, "
            "simulator-validated plan is ideal when the model supports "
            "one; when the goal depends on a mechanism the model lacks, "
            "submit the plan most likely to achieve the goal in reality "
            "- reason from the goal description, the scene geometry, and "
            "physical common sense - and annotate the subgoals that "
            "SHOULD hold if the mechanism works: the disagreement "
            "between prediction and reality is exactly the signal "
            "exploration exists to collect, so a simulator-failing "
            "sketch is a valid deliverable, and grinding for a validated "
            "plan the model cannot produce is wasted budget. An "
            "experiment's information comes from the steps the belief "
            "model cannot predict. Before running, your sketch's "
            "continuous parameters are refined in the belief model, and "
            "when refinement cannot establish a step's annotated "
            "subgoals the plan is truncated just after that step - "
            "steps beyond the disagreement never execute. So reach the "
            "first such step as early and cheaply as the task allows, "
            "and follow it with a step whose outcome reveals whether "
            "the mechanism worked - a short plan that exercises the "
            "unknown beats a long one that spends the episode's steps "
            "on what the model already predicts.\n\n"
            "Experiment design - one episode, many measurements. Before "
            "sketching, list the mechanisms the goal depends on and "
            "mark each KNOWN (the belief model has predicted it "
            "correctly against real data) or OPEN (never observed, "
            "unverified, or flagged in the guidance below). Design the "
            "episode to settle as many OPEN items as its step budget "
            "allows, not one per episode: probes of independent "
            "mechanisms can share an episode when they touch disjoint "
            "objects and neither depends on the other's outcome. When "
            "an open item is a threshold or window (how close, how "
            "long, how aligned), stage a LADDER: several independent "
            "instances at staggered values bracketing the believed "
            "boundary - e.g. object pairs at several spacings - so one "
            "episode measures the boundary from both sides instead of "
            "contributing a single incidental sample; every threshold "
            "pinned this way is a cycle of drift-by-refit avoided "
            "later. Mind the truncation rule above when combining "
            "probes: annotate the subgoals of steps whose mechanism the "
            "belief model already CONTAINS (annotations there drive "
            "boundary-probing refinement and cost nothing), but for a "
            "mechanism the model entirely LACKS, either place that "
            "probe last or leave its subgoal annotation off and give "
            "explicit parameters - an annotation the model cannot "
            "establish truncates the plan there and silently discards "
            "every later probe, while the recorded trajectory captures "
            "the unannotated probe's outcome regardless. Spend no steps "
            "re-demonstrating what the model already predicts well "
            "beyond what later probes need as setup.\n\n"
            "When the belief model has learned no dynamics at all yet "
            "(the first cycle of a fresh run), coverage beats depth: "
            "design the episode to exercise every option and to create "
            "every object interaction the goal description names "
            "(contact, attachment, activation, stacking - whatever the "
            "domain's language suggests), so the first learning phase "
            "sees each mechanism at least once, instead of spending the "
            "episode polishing a single goal attempt whose failure "
            "reveals only its first missing mechanism.\n")

    scheduled_plans_section = ""
    if scheduled_plans:
        plan_blocks = "\n".join(f"Plan {i + 1}:\n{p}"
                                for i, p in enumerate(scheduled_plans))
        scheduled_plans_section = (
            "\n## Plans Already Scheduled This Cycle\n"
            "The plan(s) below are already queued to run on this same task "
            "before any learning happens, so their interaction data will be "
            "collected regardless of what you propose now.\n"
            f"{plan_blocks}\n"
            "\nPropose a plan whose DATA is complementary rather than "
            "redundant: cover open questions, mechanisms, or parameter "
            "regions the plan(s) above leave unmeasured. A goal-reaching "
            "plan is still preferred when it can carry that coverage; "
            "when it cannot, a designed experiment that settles what the "
            "scheduled plans will not is the better use of this episode. "
            "Only if the model is believed correct everywhere and no "
            "meaningfully different goal-reaching plan exists, repeat "
            "the best plan.\n")

    strategy_section = ""
    if strategy:
        strategy_section = (
            "\n## Domain Strategy (advisory, written during learning)\n"
            "The learning phase maintains this strategy document - its "
            "best current account of how to solve tasks in this domain "
            "(approach, mechanisms, parameter formulas, pitfalls). Use "
            "it as a reference and starting point, but you are NOT "
            "limited to it: it can be wrong or stale, so re-verify its "
            "load-bearing claims cheaply before building on them, and "
            "depart from it whenever your own measurements disagree.\n\n"
            f"{strategy}\n")

    journal_section = ""
    if journal:
        journal_section = (
            "\n## Solve Journal (record of earlier attempts)\n"
            "You start with fresh context. The journal below is this run's "
            "persistent record from earlier solve attempts and tasks: "
            "auto-recorded outcomes (captured plans, rewards, budgets) "
            "plus agent-recorded lessons. Use it - reproduce what worked, "
            "do not repeat parameter sweeps it already covers - but treat "
            "any recorded conclusion skeptically: re-verify cheap claims "
            "rather than inheriting them, especially from failed "
            "attempts.\n"
            "Journal protocol for this attempt:\n"
            "- A design the journal records as having reached the goal in "
            "the REAL environment is the INCUMBENT: reproduce it unless "
            "the journal also records it failing since, or a model update "
            "invalidates one of its steps. Every deviation from an "
            "execution-validated design - reordering steps, dropping a "
            "Wait, retargeting a parameter - is a NEW experiment carrying "
            "first-execution risk that belief validation does NOT retire "
            "(real option durations and placement scatter differ), so "
            "deviate only for a recorded reason and record that reason.\n"
            "- FIRST list the journal's untried leads, then execute or "
            "explicitly retire (with a measurement) each promising lead "
            "BEFORE re-opening a family an earlier attempt already marked "
            "exhausted or opening a brand-new one. Attempts have been "
            "wasted re-litigating condemned designs while a recorded, "
            "concrete, untried lead sat unexecuted.\n"
            "- A negative claim is only as broad as the family actually "
            "swept: before trusting 'X never works', check what was "
            "tested - a claim derived from one orientation, formula, or "
            "region says nothing about the rest.\n"
            "- If two entries conflict (one rules a mechanism out, another "
            "recommends it), BOTH demote to open questions: design the "
            "cheap experiment that decides between them instead of "
            "silently trusting either.\n"
            "Add your own lessons for future attempts with the "
            "record_journal tool (facts and measurements only).\n\n"
            f"{journal}\n")

    goal_nl_section = ""
    if task.goal_nl:
        goal_nl_section = f"\n## Goal Description\n{task.goal_nl}\n"

    # The env's public reward form (success condition + costs), when the
    # task ships an evaluator that states one. Without it, agents burn
    # turns reverse-engineering scores and induce false rules from them
    # (run_20260729_001752 hypothesized a "-0.10 binary route-rejected
    # flag" and a "-0.30 excess-blue penalty" from raw numbers the
    # formula decodes instantly). Public by design: reward FORM only,
    # never oracle quantities.
    scoring_section = ""
    evaluator = getattr(task, "evaluator", None)
    if evaluator is not None:
        objective = evaluator.objective_description()
        if objective:
            scoring_section = (
                "\n## Scoring (env ground-truth reward)\n"
                f"{objective}\n"
                "Decode every reward you observe with this scoring rule "
                "before hypothesizing any other mechanism - there are no "
                "hidden reward terms.\n")

    goal_atoms_section = ""
    if goal_strs:
        goal_atoms_section = (f"\n## Goal Atoms\n{chr(10).join(goal_strs)}\n")

    pred_strs = []
    for pred in sorted(all_predicates, key=lambda p: p.name):
        type_sig = ", ".join(t.name for t in pred.types)
        line = f"  {pred.name}({type_sig})"
        if pred.natural_language_assertion is not None:
            names = [t.name for t in pred.types]
            line += f" — {pred.natural_language_assertion(names)}"
        pred_strs.append(line)

    # Tool-availability-aware references: when explore_python replaces
    # the standalone refine tool (see
    # agent_planner_explore_python_keep_replaced_tools), guidance must
    # point at the probe equivalents instead of tools the session lacks.
    # ``tool_names=None`` keeps the legacy all-tools wording.
    tool_set = set(tool_names) if tool_names is not None else None

    def _has_tool(name: str) -> bool:
        return tool_set is None or name in tool_set

    probe_refine = (not _has_tool("refine_plan_sketch")
                    and _has_tool("explore_python"))
    refine_ref = ("`sim.refine` (in `explore_python`)"
                  if probe_refine else "`refine_plan_sketch`")
    if _has_tool("explore_python"):
        visualize_advice = (
            "- Use `explore_python` (`sim.reset(mods={...})`, then "
            "`sim.render(...)`) to move objects to candidate positions and "
            "orientations for free (no physics) and find the right region "
            "visually before testing.\n")
    else:
        # No visualization surface offered: no bullet, rather than
        # advice naming a capability the session lacks.
        visualize_advice = ""

    # Advice for a step the search reports stuck (SAMPLE_EXHAUSTED). When the
    # agent proposes params it tunes that step's values; otherwise it can only
    # change the skeleton.
    deep_tune_advice = (
        "deep-tune just that step (it needs precise values from you), then "
        "re-test it. When deep-tuning a step with `evaluate_option_plan`:\n"
        "- Inspect the rendered images in `./test_images/` to see what "
        "actually happened.\n"
        "- For a failure like an IK error or collision, use the image and "
        "object poses to reason about WHY and adjust params directionally — "
        "don't try random nearby values.\n" + visualize_advice +
        "- Vary ALL parameters, not just position — orientation and others "
        "affect both the outcome and whether the action succeeds.\n"
        "- Search coarse-to-fine: spread attempts across the full range; if "
        "several nearby values fail the same way, jump to a different region "
        "instead of continuing to tweak.\n"
        "- Before steering a search with a DERIVED formula or geometric "
        "prediction (e.g. which way an object moves, falls, or deflects), "
        "validate the formula on one clean controlled experiment first. A "
        "wrong formula makes correct designs look refuted, and that false "
        "negative then silently excludes the right design family from the "
        "rest of the search.")
    revise_sketch_advice = (
        "revise the sketch — try different objects, a different ordering, an "
        "added intermediate step, or a corrected subgoal annotation — then "
        "re-test.")

    if propose_params:
        ground_sampler_guidance = ""
        ground_sampler_format = ""
        if ground_samplers:
            ground_sampler_guidance = (
                " Confine its search near your estimate by appending a "
                "region `~ [w1, w2]` (per-parameter half-widths) after a "
                "step's `[params]`: the exact center is tried first, then "
                "every sample for that step stays inside "
                "`[center - w, center + w]` instead of the full range. For "
                "regions a fixed window cannot express (state-dependent or "
                "curved), write a function in `ground_samplers.py` "
                "(`GROUND_SAMPLERS = {\"my_sampler\": fn}`, "
                "`fn(state, subgoal_atoms, rng, objects) -> params`) and "
                "reference it as `~ my_sampler` instead; the file is "
                f"reloaded on every {refine_ref} call.")
            ground_sampler_format = (
                f"\n(For {refine_ref} only, a step may add a search "
                "region after its params: "
                "`OptionName(obj1:type1)[p1, p2] ~ [w1, w2] -> {...}`, or "
                "`... ~ my_sampler` naming a GROUND_SAMPLERS entry.)")
        sketch_kind_guidance = (
            "Generate a plan — the sequence of options with object arguments "
            "and continuous parameters in `[...]` per step (see each option's "
            "params and range above; use `[]` for options with no "
            "parameters).\n\n"
            "Explore designs before tuning parameters: when several "
            "qualitatively different designs could work (different objects, "
            "orientations, sides, orderings, or mechanisms), run a cheap "
            "test of each and compare failure modes BEFORE fine-tuning any "
            "one of them. Parameter tuning cannot rescue the wrong design - "
            "if a design keeps failing the same way as you tune it, switch "
            "designs rather than tightening values. And before building on "
            "a physics rule or constraint you inferred from a single "
            "observation, re-test it once with a clean experiment; a wrong "
            "rule adopted early can quietly rule out the correct designs.\n\n"
            "Spend effort on parameters in proportion to difficulty:\n"
            "- Where a WIDE range of values works, any reasonable value is "
            "fine — don't over-tune these.\n"
            "- Where good values are hard to hit — tight tolerances or exact "
            "relative placements (e.g. positioning one object at a precise "
            f"offset from another) - use {refine_ref} to search for a "
            "working value (it's slower) and read the value it found." +
            ground_sampler_guidance)
        format_block = (
            "Output the plan with one option per line in this format:\n"
            "  OptionName(obj1:type1, obj2:type2)[param1, param2] -> "
            "{Pred(obj1:type1), Pred2(obj1:type1, obj2:type2)}\n"
            "  Wait(robot:robot)[] -> "
            "{Pred3(obj1:type1), NOT Pred4(obj1:type1, obj2:type2)}" +
            ground_sampler_format)
    else:
        sketch_kind_guidance = (
            "Generate a plan sketch — the sequence of options with object "
            "arguments, WITHOUT continuous parameters; a backtracking search "
            "finds them for you.")
        format_block = (
            "Output the plan sketch with one option per line in this "
            "format:\n"
            "  OptionName(obj1:type1, obj2:type2) -> "
            "{Pred(obj1:type1), Pred2(obj1:type1, obj2:type2)}\n"
            "  Wait(robot:robot) -> "
            "{Pred3(obj1:type1), NOT Pred4(obj1:type1, obj2:type2)}")

    if require_tool_validation:
        stuck_advice = (deep_tune_advice
                        if propose_params else revise_sketch_advice)
        margin_guidance = ""
        if physics_margin:
            margin_guidance = (
                "Capture also requires the plan to succeed at a grid of "
                "perturbations spanning +-1 sigma of the identified "
                "physical parameters (the physics fit's own uncertainty); "
                "a plan that fails any point is reported PARAM-SENSITIVE "
                "instead of captured. Success can be NON-MONOTONIC in a "
                "physical parameter - a design can pass just above and "
                "just below a value and fail exactly at it - so tune "
                "designs that pass the WHOLE range: pre-check with "
                "`sim.run(plan_text, physics_sweep=True)` in explore_python "
                "(same points as the gate, one deterministic rollout each) "
                "instead of discovering rejections one submission at a "
                "time. ")
        if CFG.agent_plan_validation_rule_param_margin:
            margin_guidance += (
                "Capture additionally re-runs the plan under the "
                "calibrated posterior members of the LEARNED rule "
                "parameters (the fit's honest uncertainty about the "
                "thresholds and offsets it learned from data); failing "
                "under any member is rejected PARAM-SENSITIVE. So design "
                "MAX-MARGIN, not merely feasible: place every operating "
                "point at the CENTER of its learned feasibility window "
                "(aim an applicator at the target point itself, put a "
                "placement mid-window, leave slack on every timing) - a "
                "design that only works at the fitted point estimate of "
                "an uncertain constant will fail either this gate or the "
                "real environment, whose true constant sits somewhere in "
                "that posterior. Before submitting, name your plan's "
                "WEAKEST margin (the smallest distance from any step's "
                "operating point to a learned threshold) and widen it if "
                "it is smaller than the measured execution scatter. ")
        submit_guidance = (
            "SUBMIT via `evaluate_option_plan`: pass your full plan as text "
            "(one option per line, `Option(obj:type)[params] -> {subgoals}`, "
            "with EXACT params) and run it on the CURRENT task (omit "
            "task_idx). When it reaches the goal, that plan is captured as "
            "your answer, so do NOT finish until evaluate_option_plan "
            "CONFIRMS the capture. A goal-reaching plan is re-run several "
            "times before capture (simulation varies across runs; each "
            "rollout reports the motion-planner seed it ran at); if it is "
            "reported FLAKY, reproduce the failed rollout exactly (pass "
            "its reported seed as rollout_seed to evaluate_option_plan, or "
            "`sim.run(plan_text, seed=...)` in explore_python) to see WHY, "
            "then add margin to the fragile step and resubmit. For a plan "
            "you suspect is marginal, request a stricter gate up front "
            "with validation_rollouts=N (more repeats; never fewer than "
            "configured) or measure reliability first with "
            "`sim.run(plan_text, trials=N)`. " + margin_guidance +
            "CAPTURE FIRST, OPTIMIZE SECOND: when the reward charges for "
            "resources used (read the scoring section), a captured "
            "modest-reward solve outscores an uncaptured optimal attempt "
            "by the entire success bonus - so bank a ROBUST goal-reaching "
            "design early, even an over-built one (extra margin, extra "
            "resources), and only then spend remaining budget improving "
            "it. This is safe: a newly VALIDATED capture replaces the "
            "banked one, while a rejected submission (flaky, "
            "param-sensitive, evaluator-rejected, or short of the goal) "
            "never displaces it - but do not resubmit designs that are "
            "not strictly better, since a validated worse plan would "
            "replace the banked answer. Robust-but-wasteful designs live "
            "AWAY from the feasibility boundary that minimal designs sit "
            "on, so they are usually far easier to find and validate. "
            "It runs your EXACT parameters with no sampling. To find "
            f"working parameters you MAY use {refine_ref} (it searches "
            "but is slower); read the parameters it reports and submit them "
            "via evaluate_option_plan. If a step does not reach its subgoal, "
            + stuck_advice)
    elif propose_params:
        submit_guidance = (
            f"You may validate with {refine_ref} (it tries your "
            "parameters first, then samples) and deep-tune any step it "
            "reports stuck before finishing.")
    else:
        submit_guidance = (
            f"You may vet a sketch with {refine_ref} before finishing; "
            "the backtracking search will find continuous parameters.")
    # Explore-specific delivery guidance lives in the Exploration
    # Setting section above (single source); nothing to append here.

    if policy_mode:
        submit_guidance = (
            "## Deliverable: Closed-Loop Policy (./policy.py)\n"
            "Instead of a fixed plan, you deliver a PROGRAM that chooses "
            "the next option from the current state. Write it to "
            "./policy.py:\n\n"
            "    def get_option(state, memory):\n"
            "        ...\n\n"
            "- `state`: the current State object (read-only copy). Same "
            "API as explore_python: `state.get(obj, 'feature')`, iterate "
            "objects with `for obj in state`, `obj.name`, `obj.type`.\n"
            "- `memory`: a dict, initially empty, persisting across calls "
            "within ONE episode (phase flags, counters, cached "
            "measurements); reset between episodes. After a failed "
            "option, `memory['last_failure']` holds the failure text - "
            "branch on it to RECOVER (re-place a drifted block, adjust a "
            "target after a motion-planning refusal); it is None on clean "
            "steps.\n"
            "- Return ONE plan line as a string, in the exact sketch "
            "grammar `OptionName(obj:type, ...)[p1, p2]` with exact "
            "continuous parameters (`[]` for none; `->`/`~` annotations "
            "are ignored here). Return None to declare the episode "
            "finished.\n"
            "- Helpers available inside policy.py: `np` (numpy) and "
            "`atoms(state)` -> set of ground-atom strings.\n"
            "- Execution semantics (identical in the belief simulator and "
            "the real environment): get_option is called once per option "
            "boundary with the ACTUAL current state; option failures do "
            "NOT end the episode (they surface via memory['last_failure'] "
            "and you are asked again); exceptions in get_option or "
            "unparsable/ungroundable lines DO end it; a hard cap of "
            f"{CFG.agent_policy_max_options} options bounds every "
            "episode. After a failure, CHANGE something before retrying "
            "(parameters, target object, or action) - re-issuing the "
            "byte-identical line that just failed "
            f"{CFG.agent_policy_max_repeated_failures} times in a row "
            "ends the episode as a policy bug, because an unchanged "
            "command fails the same way. The same applies to a no-op "
            "livelock: re-issuing one identical line that keeps "
            "COMPLETING with no observable state change "
            f"{CFG.agent_policy_max_repeated_noops} times in a row also "
            "ends the episode - if your stage logic is not advancing, "
            "fix the stage test, do not re-send the same command.\n"
            "SUBMIT via `evaluate_policy` on the CURRENT task until it "
            "reaches the goal across all validation rollouts - the "
            "validated policy.py snapshot (taken at call time; later "
            "edits need a new call) is your ONLY accepted output. Test "
            "recovery behavior first: in explore_python, "
            "`sim.run_policy()` runs ./policy.py from the CURRENT probe "
            "state (including perturbed or mid-plan states), so check "
            "that the policy recovers from off-nominal states, not just "
            "the initial one. " + margin_guidance)
        closing_block = (
            "Your answer is ONLY accepted from a goal-reaching "
            "`evaluate_policy` run on the CURRENT task; final text alone "
            "is discarded, so never finish without that validated run. "
            "After the goal-reaching run, summarize the policy's strategy "
            "as your final text.")
    if require_tool_validation and not policy_mode:
        # Plain text is NOT a submission in this mode; saying "output the
        # plan lines" as the closing instruction has led agents (especially
        # right after an SDK context compaction, whose text-only summary
        # instruction bleeds into the task) to answer with an unvalidated
        # text sketch and finish, wasting the whole attempt.
        closing_block = (
            "Your answer is ONLY accepted from a goal-reaching "
            "`evaluate_option_plan` run on the CURRENT task; final text "
            "alone is discarded, so never finish without that validated "
            "run. Tool calls are permitted on every turn of this "
            "conversation. If an earlier context summary says a turn was "
            "text-only, that applied to writing the summary itself, not to "
            "this task; resume calling tools. After the goal-reaching run, "
            "repeat its plan lines as your final text.")
    elif explore_mode:
        closing_block = (
            "This is an EXPLORE query: your final plan-sketch text IS the "
            "deliverable (a simulator-validated capture is welcome but NOT "
            "required). Output ONLY the plan sketch lines at the end, "
            "after any analysis.")
    elif not policy_mode:
        # (In policy mode the closing block was set above.)
        closing_block = (
            "Output ONLY the plan sketch lines at the end, after any "
            "analysis.")

    atoms_block = chr(10).join(atom_strs) if atom_strs else (
        "  (none - no atom of the available predicates holds initially)")

    if explore_mode:
        opening = (
            "You are exploring a task environment to gather information. "
            "Generate a plan sketch to run in the real environment as an "
            "experiment. Achieving the goal is the most informative "
            "experiment available, so treat solving the task as part of "
            "information gathering.")
    else:
        opening = ("You are solving a task. Generate a plan sketch to "
                   "achieve the goal.")

    prompt = f"""{opening}
{goal_nl_section}{scoring_section}{goal_atoms_section}{setting_section}\
{experiment_section}
## Initial State Atoms
{atoms_block}

## Initial State Features
{state_str}
{initial_image_section}
## Objects
{chr(10).join(obj_strs)}

## Available Options
{chr(10).join(option_strs)}

## Available Predicates (for subgoal annotations)
{chr(10).join(pred_strs)}
{trajectory_summary}{tools_str}{strategy_section}{journal_section}\
{scheduled_plans_section}
## Instructions
Use your available tools to inspect the environment before producing the plan.

{sketch_kind_guidance}

{submit_guidance}

{format_block}

Follow the system prompt's Subgoal Annotations contract: annotate every \
step whose effect the available predicates can express (typed obj:type \
references in arguments and atoms alike), insert a Wait after any action \
whose subgoal depends on a delayed process, and annotate each Wait with \
the atoms that should end it. A delayed process needs its EXPLICIT Wait \
even when a belief rollout completes without one: belief and real option \
durations differ, so a plan that borrows other steps' incidental \
duration to cover a hidden timer validates in belief and loses the \
timing race in reality. A step without `-> {{atoms}}` is only \
checked for "executed" - search and execution monitoring are blind there.

{closing_block}"""

    return prompt
